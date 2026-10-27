        self.frame_dicts = self._process_dataframe()

        self._validate_columns()
        self._validate_encoding()
        self.prefix_uri = self._get_prefix_uri()
        self._handle_indexes()

    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        raise AbstractMethodError(self)

    @final
    def _validate_columns(self) -> None:
        """
        Validate elems_cols and attrs_cols.

        This method will check if columns is list-like.

        Raises
        ------
        ValueError
            * If value is not a list and less then length of nodes.
        """
        if self.attr_cols and not is_list_like(self.attr_cols):
            raise TypeError(
                f"{type(self.attr_cols).__name__} is not a valid type for attr_cols"
            )

        if self.elem_cols and not is_list_like(self.elem_cols):
            raise TypeError(
                f"{type(self.elem_cols).__name__} is not a valid type for elem_cols"
            )

    @final
    def _validate_encoding(self) -> None:
        """
        Validate encoding.

        This method will check if encoding is among listed under codecs.

        Raises
        ------
        LookupError
            * If encoding is not available in codecs.
        """

        codecs.lookup(self.encoding)

    @final
    def _process_dataframe(self) -> dict[int | str, dict[str, Any]]:
        """
        Adjust Data Frame to fit xml output.

        This method will adjust underlying data frame for xml output,
        including optionally replacing missing values and including indexes.
        """

        df = self.frame

        if self.index:
            df = df.reset_index()

        if self.na_rep is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Downcasting object dtype arrays",
                    category=FutureWarning,
                )
                df = df.fillna(self.na_rep)

        return df.to_dict(orient="index")

    @final
    def _handle_indexes(self) -> None:
        """
        Handle indexes.

        This method will add indexes into attr_cols or elem_cols.
        """

        if not self.index:
            return

        first_key = next(iter(self.frame_dicts))
        indexes: list[str] = [
            x for x in self.frame_dicts[first_key].keys() if x not in self.orig_cols
        ]

        if self.attr_cols:
            self.attr_cols = indexes + self.attr_cols

        if self.elem_cols:
            self.elem_cols = indexes + self.elem_cols

    def _get_prefix_uri(self) -> str:
        """
        Get uri of namespace prefix.

        This method retrieves corresponding URI to prefix in namespaces.

        Raises
        ------
        KeyError
            *If prefix is not included in namespace dict.
        """

        raise AbstractMethodError(self)

    @final
    def _other_namespaces(self) -> dict:
        """
        Define other namespaces.

        This method will build dictionary of namespaces attributes
        for root element, conditionally with optional namespaces and
        prefix.
        """

        nmsp_dict: dict[str, str] = {}
        if self.namespaces:
            nmsp_dict = {
                f"xmlns{p if p=='' else f':{p}'}": n
                for p, n in self.namespaces.items()
                if n != self.prefix_uri[1:-1]
            }

        return nmsp_dict

    @final
    def _build_attribs(self, d: dict[str, Any], elem_row: Any) -> Any:
        """
        Create attributes of row.

        This method adds attributes using attr_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """

        if not self.attr_cols:
            return elem_row

        for col in self.attr_cols:
            attr_name = self._get_flat_col_name(col)
            try:
                if not isna(d[col]):
                    elem_row.attrib[attr_name] = str(d[col])
            except KeyError:
                raise KeyError(f"no valid column, {col}")
        return elem_row

    @final
    def _get_flat_col_name(self, col: str | tuple) -> str:
        flat_col = col
        if isinstance(col, tuple):
            flat_col = (
                "".join([str(c) for c in col]).strip()
                if "" in col
                else "_".join([str(c) for c in col]).strip()
            )
        return f"{self.prefix_uri}{flat_col}"

    @cache_readonly
    def _sub_element_cls(self):
        raise AbstractMethodError(self)

    @final
    def _build_elems(self, d: dict[str, Any], elem_row: Any) -> None:
        """
        Create child elements of row.

        This method adds child elements using elem_cols to row element and
        works with tuples for multindex or hierarchical columns.
        """
        sub_element_cls = self._sub_element_cls

        if not self.elem_cols:
            return

        for col in self.elem_cols:
            elem_name = self._get_flat_col_name(col)
            try:
                val = None if isna(d[col]) or d[col] == "" else str(d[col])
                sub_element_cls(elem_row, elem_name).text = val
            except KeyError:
                raise KeyError(f"no valid column, {col}")

    @final
    def write_output(self) -> str | None:
        xml_doc = self._build_tree()

        if self.path_or_buffer is not None:
            with get_handle(
                self.path_or_buffer,
                "wb",
                compression=self.compression,
                storage_options=self.storage_options,
                is_text=False,
            ) as handles:
                handles.handle.write(xml_doc)
            return None

        else:
            return xml_doc.decode(self.encoding).rstrip()


class EtreeXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def _build_tree(self) -> bytes:
        from xml.etree.ElementTree import (
            Element,
            SubElement,
            tostring,
        )

        self.root = Element(
            f"{self.prefix_uri}{self.root_name}", attrib=self._other_namespaces()
        )

        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)

            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        self.out_xml = tostring(
            self.root,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        if self.pretty_print:
            self.out_xml = self._prettify_tree()

        if self.stylesheet is not None:
            raise ValueError(
                "To use stylesheet, you need lxml installed and selected as parser."
            )

        return self.out_xml

    def _get_prefix_uri(self) -> str:
        from xml.etree.ElementTree import register_namespace

        uri = ""
        if self.namespaces:
            for p, n in self.namespaces.items():
                if isinstance(p, str) and isinstance(n, str):
                    register_namespace(p, n)
            if self.prefix:
                try:
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError:
                    raise KeyError(f"{self.prefix} is not included in namespaces")
            elif "" in self.namespaces:
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ""

        return uri

    @cache_readonly
    def _sub_element_cls(self):
        from xml.etree.ElementTree import SubElement

        return SubElement

    def _prettify_tree(self) -> bytes:
        """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """

        from xml.dom.minidom import parseString

        dom = parseString(self.out_xml)

        return dom.toprettyxml(indent="  ", encoding=self.encoding)


class LxmlXMLFormatter(_BaseXMLFormatter):
    """
    Class for formatting data in xml using Python standard library
    modules: `xml.etree.ElementTree` and `xml.dom.minidom`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._convert_empty_str_key()

    def _build_tree(self) -> bytes:
        """
        Build tree from  data.

        This method initializes the root and builds attributes and elements
        with optional namespaces.
        """
        from lxml.etree import (
            Element,
            SubElement,
            tostring,
        )

        self.root = Element(f"{self.prefix_uri}{self.root_name}", nsmap=self.namespaces)

        for d in self.frame_dicts.values():
            elem_row = SubElement(self.root, f"{self.prefix_uri}{self.row_name}")

            if not self.attr_cols and not self.elem_cols:
                self.elem_cols = list(d.keys())
                self._build_elems(d, elem_row)

            else:
                elem_row = self._build_attribs(d, elem_row)
                self._build_elems(d, elem_row)

        self.out_xml = tostring(
            self.root,
            pretty_print=self.pretty_print,
            method="xml",
            encoding=self.encoding,
            xml_declaration=self.xml_declaration,
        )

        if self.stylesheet is not None:
            self.out_xml = self._transform_doc()

        return self.out_xml

    def _convert_empty_str_key(self) -> None:
        """
        Replace zero-length string in `namespaces`.

        This method will replace '' with None to align to `lxml`
        requirement that empty string prefixes are not allowed.
        """

        if self.namespaces and "" in self.namespaces.keys():
            self.namespaces[None] = self.namespaces.pop("", "default")

    def _get_prefix_uri(self) -> str:
        uri = ""
        if self.namespaces:
            if self.prefix:
                try:
                    uri = f"{{{self.namespaces[self.prefix]}}}"
                except KeyError:
                    raise KeyError(f"{self.prefix} is not included in namespaces")
            elif "" in self.namespaces:
                uri = f'{{{self.namespaces[""]}}}'
            else:
                uri = ""

        return uri

    @cache_readonly
    def _sub_element_cls(self):
        from lxml.etree import SubElement

        return SubElement

    def _transform_doc(self) -> bytes:
        """
        Parse stylesheet from file or buffer and run it.

        This method will parse stylesheet object into tree for parsing
        conditionally by its specific object type, then transforms
        original tree with XSLT script.
        """
        from lxml.etree import (
            XSLT,
            XMLParser,
            fromstring,
            parse,
        )

        style_doc = self.stylesheet
        assert style_doc is not None  # is ensured by caller

        handle_data = get_data_from_filepath(
            filepath_or_buffer=style_doc,
            encoding=self.encoding,
            compression=self.compression,
            storage_options=self.storage_options,
        )

        with preprocess_data(handle_data) as xml_data:
            curr_parser = XMLParser(encoding=self.encoding)

            if isinstance(xml_data, io.StringIO):
                xsl_doc = fromstring(
                    xml_data.getvalue().encode(self.encoding), parser=curr_parser
                )
            else:
                xsl_doc = parse(xml_data, parser=curr_parser)

        transformer = XSLT(xsl_doc)
        new_doc = transformer(self.root)

        return bytes(new_doc)
<|endoftext|>"""
Module for formatting output data in console (to string).
"""
from __future__ import annotations

from shutil import get_terminal_size
from typing import TYPE_CHECKING

import numpy as np

from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas.io.formats.format import DataFrameFormatter


class StringFormatter:
    """Formatter for string representation of a dataframe."""

    def __init__(self, fmt: DataFrameFormatter, line_width: int | None = None) -> None:
        self.fmt = fmt
        self.adj = fmt.adj
        self.frame = fmt.frame
        self.line_width = line_width

    def to_string(self) -> str:
        text = self._get_string_representation()
        if self.fmt.should_show_dimensions:
            text = f"{text}{self.fmt.dimensions_info}"
        return text

    def _get_strcols(self) -> list[list[str]]:
        strcols = self.fmt.get_strcols()
        if self.fmt.is_truncated:
            strcols = self._insert_dot_separators(strcols)
        return strcols

    def _get_string_representation(self) -> str:
        if self.fmt.frame.empty:
            return self._empty_info_line

        strcols = self._get_strcols()

        if self.line_width is None:
            # no need to wrap around just print the whole frame
            return self.adj.adjoin(1, *strcols)

        if self._need_to_wrap_around:
            return self._join_multiline(strcols)

        return self._fit_strcols_to_terminal_width(strcols)

    @property
    def _empty_info_line(self) -> str:
        return (
            f"Empty {type(self.frame).__name__}\n"
            f"Columns: {pprint_thing(self.frame.columns)}\n"
            f"Index: {pprint_thing(self.frame.index)}"
        )

    @property
    def _need_to_wrap_around(self) -> bool:
        return bool(self.fmt.max_cols is None or self.fmt.max_cols > 0)

    def _insert_dot_separators(self, strcols: list[list[str]]) -> list[list[str]]:
        str_index = self.fmt._get_formatted_index(self.fmt.tr_frame)
        index_length = len(str_index)

        if self.fmt.is_truncated_horizontally:
            strcols = self._insert_dot_separator_horizontal(strcols, index_length)

        if self.fmt.is_truncated_vertically:
            strcols = self._insert_dot_separator_vertical(strcols, index_length)

        return strcols

    @property
    def _adjusted_tr_col_num(self) -> int:
        return self.fmt.tr_col_num + 1 if self.fmt.index else self.fmt.tr_col_num

    def _insert_dot_separator_horizontal(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        strcols.insert(self._adjusted_tr_col_num, [" ..."] * index_length)
        return strcols

    def _insert_dot_separator_vertical(
        self, strcols: list[list[str]], index_length: int
    ) -> list[list[str]]:
        n_header_rows = index_length - len(self.fmt.tr_frame)
        row_num = self.fmt.tr_row_num
        for ix, col in enumerate(strcols):
            cwidth = self.adj.len(col[row_num])

            if self.fmt.is_truncated_horizontally:
                is_dot_col = ix == self._adjusted_tr_col_num
            else:
                is_dot_col = False

            if cwidth > 3 or is_dot_col:
                dots = "..."
            else:
                dots = ".."

            if ix == 0 and self.fmt.index:
                dot_mode = "left"
            elif is_dot_col:
                cwidth = 4
                dot_mode = "right"
            else:
                dot_mode = "right"

            dot_str = self.adj.justify([dots], cwidth, mode=dot_mode)[0]
            col.insert(row_num + n_header_rows, dot_str)
        return strcols

    def _join_multiline(self, strcols_input: Iterable[list[str]]) -> str:
        lwidth = self.line_width
        adjoin_width = 1
        strcols = list(strcols_input)

        if self.fmt.index:
            idx = strcols.pop(0)
            lwidth -= np.array([self.adj.len(x) for x in idx]).max() + adjoin_width

        col_widths = [
            np.array([self.adj.len(x) for x in col]).max() if len(col) > 0 else 0
            for col in strcols
        ]

        assert lwidth is not None
        col_bins = _binify(col_widths, lwidth)
        nbins = len(col_bins)

        str_lst = []
        start = 0
        for i, end in enumerate(col_bins):
            row = strcols[start:end]
            if self.fmt.index:
                row.insert(0, idx)
            if nbins > 1:
                nrows = len(row[-1])
                if end <= len(strcols) and i < nbins - 1:
                    row.append([" \\"] + ["  "] * (nrows - 1))
                else:
                    row.append([" "] * nrows)
            str_lst.append(self.adj.adjoin(adjoin_width, *row))
            start = end
        return "\n\n".join(str_lst)

    def _fit_strcols_to_terminal_width(self, strcols: list[list[str]]) -> str:
        from pandas import Series

        lines = self.adj.adjoin(1, *strcols).split("\n")
        max_len = Series(lines).str.len().max()
        # plus truncate dot col
        width, _ = get_terminal_size()
        dif = max_len - width
        # '+ 1' to avoid too wide repr (GH PR #17023)
        adj_dif = dif + 1
        col_lens = Series([Series(ele).str.len().max() for ele in strcols])
        n_cols = len(col_lens)
        counter = 0
        while adj_dif > 0 and n_cols > 1:
            counter += 1
            mid = round(n_cols / 2)
            mid_ix = col_lens.index[mid]
            col_len = col_lens[mid_ix]
            # adjoin adds one
            adj_dif -= col_len + 1
            col_lens = col_lens.drop(mid_ix)
            n_cols = len(col_lens)

        # subtract index column
        max_cols_fitted = n_cols - self.fmt.index
        # GH-21180. Ensure that we print at least two.
        max_cols_fitted = max(max_cols_fitted, 2)
        self.fmt.max_cols_fitted = max_cols_fitted

        # Call again _truncate to cut frame appropriately
        # and then generate string representation
        self.fmt.truncate()
        strcols = self._get_strcols()
        return self.adj.adjoin(1, *strcols)


def _binify(cols: list[int], line_width: int) -> list[int]:
    adjoin_width = 1
    bins = []
    curr_width = 0
    i_last_column = len(cols) - 1
    for i, w in enumerate(cols):
        w_adjoined = w + adjoin_width
        curr_width += w_adjoined
        if i_last_column == i:
            wrap = curr_width + 1 > line_width and i > 0
        else:
            wrap = curr_width + 2 > line_width and i > 0
        if wrap:
            bins.append(i)
            curr_width = w_adjoined

    bins.append(len(cols))
    return bins
<|endoftext|>"""
Internal module for console introspection
"""
from __future__ import annotations

from shutil import get_terminal_size


def get_console_size() -> tuple[int | None, int | None]:
    """
    Return console size as tuple = (width, height).

    Returns (None,None) in non-interactive session.
    """
    from pandas import get_option

    display_width = get_option("display.width")
    display_height = get_option("display.max_rows")

    # Consider
    # interactive shell terminal, can detect term size
    # interactive non-shell terminal (ipnb/ipqtconsole), cannot detect term
    # size non-interactive script, should disregard term size

    # in addition
    # width,height have default values, but setting to 'None' signals
    # should use Auto-Detection, But only in interactive shell-terminal.
    # Simple. yeah.

    if in_interactive_session():
        if in_ipython_frontend():
            # sane defaults for interactive non-shell terminal
            # match default for width,height in config_init
            from pandas._config.config import get_default_val

            terminal_width = get_default_val("display.width")
            terminal_height = get_default_val("display.max_rows")
        else:
            # pure terminal
            terminal_width, terminal_height = get_terminal_size()
    else:
        terminal_width, terminal_height = None, None

    # Note if the User sets width/Height to None (auto-detection)
    # and we're in a script (non-inter), this will return (None,None)
    # caller needs to deal.
    return display_width or terminal_width, display_height or terminal_height


# ----------------------------------------------------------------------
# Detect our environment


def in_interactive_session() -> bool:
    """
    Check if we're running in an interactive shell.

    Returns
    -------
    bool
        True if running under python/ipython interactive shell.
    """
    from pandas import get_option

    def check_main():
        try:
            import __main__ as main
        except ModuleNotFoundError:
            return get_option("mode.sim_interactive")
        return not hasattr(main, "__file__") or get_option("mode.sim_interactive")

    try:
        # error: Name '__IPYTHON__' is not defined
        return __IPYTHON__ or check_main()  # type: ignore[name-defined]
    except NameError:
        return check_main()


def in_ipython_frontend() -> bool:
    """
    Check if we're inside an IPython zmq frontend.

    Returns
    -------
    bool
    """
    try:
        # error: Name 'get_ipython' is not defined
        ip = get_ipython()  # type: ignore[name-defined]
        return "zmq" in str(type(ip)).lower()
    except NameError:
        pass

    return False
<|endoftext|>"""
Utilities for interpreting CSS from Stylers for formatting non-HTML outputs.
"""
from __future__ import annotations

import re
from typing import (
    TYPE_CHECKING,
    Callable,
)
import warnings

from pandas.errors import CSSWarning
from pandas.util._exceptions import find_stack_level

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterable,
        Iterator,
    )


def _side_expander(prop_fmt: str) -> Callable:
    """
    Wrapper to expand shorthand property into top, right, bottom, left properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """

    def expand(self, prop, value: str) -> Generator[tuple[str, str], None, None]:
        """
        Expand shorthand property into side-specific property (top, right, bottom, left)

        Parameters
        ----------
            prop (str): CSS property name
            value (str): String token for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        try:
            mapping = self.SIDE_SHORTHANDS[len(tokens)]
        except KeyError:
            warnings.warn(
                f'Could not expand "{prop}: {value}"',
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return
        for key, idx in zip(self.SIDES, mapping):
            yield prop_fmt.format(key), tokens[idx]

    return expand


def _border_expander(side: str = "") -> Callable:
    """
    Wrapper to expand 'border' property into border color, style, and width properties

    Parameters
    ----------
    side : str
        The border side to expand into properties

    Returns
    -------
        function: Return to call when a 'border(-{side}): {value}' string is encountered
    """
    if side != "":
        side = f"-{side}"

    def expand(self, prop, value: str) -> Generator[tuple[str, str], None, None]:
        """
        Expand border into color, style, and width tuples

        Parameters
        ----------
            prop : str
                CSS property name passed to styler
            value : str
                Value passed to styler for property

        Yields
        ------
            Tuple (str, str): Expanded property, value
        """
        tokens = value.split()
        if len(tokens) == 0 or len(tokens) > 3:
            warnings.warn(
                f'Too many tokens provided to "{prop}" (expected 1-3)',
                CSSWarning,
                stacklevel=find_stack_level(),
            )

        # TODO: Can we use current color as initial value to comply with CSS standards?
        border_declarations = {
            f"border{side}-color": "black",
            f"border{side}-style": "none",
            f"border{side}-width": "medium",
        }
        for token in tokens:
            if token.lower() in self.BORDER_STYLES:
                border_declarations[f"border{side}-style"] = token
            elif any(ratio in token.lower() for ratio in self.BORDER_WIDTH_RATIOS):
                border_declarations[f"border{side}-width"] = token
            else:
                border_declarations[f"border{side}-color"] = token
            # TODO: Warn user if item entered more than once (e.g. "border: red green")

        # Per CSS, "border" will reset previous "border-*" definitions
        yield from self.atomize(border_declarations.items())

    return expand


class CSSResolver:
    """
    A callable for parsing and resolving CSS to atomic properties.
    """

    UNIT_RATIOS = {
        "pt": ("pt", 1),
        "em": ("em", 1),
        "rem": ("pt", 12),
        "ex": ("em", 0.5),
        # 'ch':
        "px": ("pt", 0.75),
        "pc": ("pt", 12),
        "in": ("pt", 72),
        "cm": ("in", 1 / 2.54),
        "mm": ("in", 1 / 25.4),
        "q": ("mm", 0.25),
        "!!default": ("em", 0),
    }

    FONT_SIZE_RATIOS = UNIT_RATIOS.copy()
    FONT_SIZE_RATIOS.update(
        {
            "%": ("em", 0.01),
            "xx-small": ("rem", 0.5),
            "x-small": ("rem", 0.625),
            "small": ("rem", 0.8),
            "medium": ("rem", 1),
            "large": ("rem", 1.125),
            "x-large": ("rem", 1.5),
            "xx-large": ("rem", 2),
            "smaller": ("em", 1 / 1.2),
            "larger": ("em", 1.2),
            "!!default": ("em", 1),
        }
    )

    MARGIN_RATIOS = UNIT_RATIOS.copy()
    MARGIN_RATIOS.update({"none": ("pt", 0)})

    BORDER_WIDTH_RATIOS = UNIT_RATIOS.copy()
    BORDER_WIDTH_RATIOS.update(
        {
            "none": ("pt", 0),
            "thick": ("px", 4),
            "medium": ("px", 2),
            "thin": ("px", 1),
            # Default: medium only if solid
        }
    )

    BORDER_STYLES = [
        "none",
        "hidden",
        "dotted",
        "dashed",
        "solid",
        "double",
        "groove",
        "ridge",
        "inset",
        "outset",
        "mediumdashdot",
        "dashdotdot",
        "hair",
        "mediumdashdotdot",
        "dashdot",
        "slantdashdot",
        "mediumdashed",
    ]

    SIDE_SHORTHANDS = {
        1: [0, 0, 0, 0],
        2: [0, 1, 0, 1],
        3: [0, 1, 2, 1],
        4: [0, 1, 2, 3],
    }

    SIDES = ("top", "right", "bottom", "left")

    CSS_EXPANSIONS = {
        **{
            (f"border-{prop}" if prop else "border"): _border_expander(prop)
            for prop in ["", "top", "right", "bottom", "left"]
        },
        **{
            f"border-{prop}": _side_expander(f"border-{{:s}}-{prop}")
            for prop in ["color", "style", "width"]
        },
        "margin": _side_expander("margin-{:s}"),
        "padding": _side_expander("padding-{:s}"),
    }

    def __call__(
        self,
        declarations: str | Iterable[tuple[str, str]],
        inherited: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        The given declarations to atomic properties.

        Parameters
        ----------
        declarations_str : str | Iterable[tuple[str, str]]
            A CSS string or set of CSS declaration tuples
            e.g. "font-weight: bold; background: blue" or
            {("font-weight", "bold"), ("background", "blue")}
        inherited : dict, optional
            Atomic properties indicating the inherited style context in which
            declarations_str is to be resolved. ``inherited`` should already
            be resolved, i.e. valid output of this method.

        Returns
        -------
        dict
            Atomic CSS 2.2 properties.

        Examples
        --------
        >>> resolve = CSSResolver()
        >>> inherited = {'font-family': 'serif', 'font-weight': 'bold'}
        >>> out = resolve('''
        ...               border-color: BLUE RED;
        ...               font-size: 1em;
        ...               font-size: 2em;
        ...               font-weight: normal;
        ...               font-weight: inherit;
        ...               ''', inherited)
        >>> sorted(out.items())  # doctest: +NORMALIZE_WHITESPACE
        [('border-bottom-color', 'blue'),
         ('border-left-color', 'red'),
         ('border-right-color', 'red'),
         ('border-top-color', 'blue'),
         ('font-family', 'serif'),
         ('font-size', '24pt'),
         ('font-weight', 'bold')]
        """
        if isinstance(declarations, str):
            declarations = self.parse(declarations)
        props = dict(self.atomize(declarations))
        if inherited is None:
            inherited = {}

        props = self._update_initial(props, inherited)
        props = self._update_font_size(props, inherited)
        return self._update_other_units(props)

    def _update_initial(
        self,
        props: dict[str, str],
        inherited: dict[str, str],
    ) -> dict[str, str]:
        # 1. resolve inherited, initial
        for prop, val in inherited.items():
            if prop not in props:
                props[prop] = val

        new_props = props.copy()
        for prop, val in props.items():
            if val == "inherit":
                val = inherited.get(prop, "initial")

            if val in ("initial", None):
                # we do not define a complete initial stylesheet
                del new_props[prop]
            else:
                new_props[prop] = val
        return new_props

    def _update_font_size(
        self,
        props: dict[str, str],
        inherited: dict[str, str],
    ) -> dict[str, str]:
        # 2. resolve relative font size
        if props.get("font-size"):
            props["font-size"] = self.size_to_pt(
                props["font-size"],
                self._get_font_size(inherited),
                conversions=self.FONT_SIZE_RATIOS,
            )
        return props

    def _get_font_size(self, props: dict[str, str]) -> float | None:
        if props.get("font-size"):
            font_size_string = props["font-size"]
            return self._get_float_font_size_from_pt(font_size_string)
        return None

    def _get_float_font_size_from_pt(self, font_size_string: str) -> float:
        assert font_size_string.endswith("pt")
        return float(font_size_string.rstrip("pt"))

    def _update_other_units(self, props: dict[str, str]) -> dict[str, str]:
        font_size = self._get_font_size(props)
        # 3. TODO: resolve other font-relative units
        for side in self.SIDES:
            prop = f"border-{side}-width"
            if prop in props:
                props[prop] = self.size_to_pt(
                    props[prop],
                    em_pt=font_size,
                    conversions=self.BORDER_WIDTH_RATIOS,
                )

            for prop in [f"margin-{side}", f"padding-{side}"]:
                if prop in props:
                    # TODO: support %
                    props[prop] = self.size_to_pt(
                        props[prop],
                        em_pt=font_size,
                        conversions=self.MARGIN_RATIOS,
                    )
        return props

    def size_to_pt(self, in_val, em_pt=None, conversions=UNIT_RATIOS) -> str:
        def _error():
            warnings.warn(
                f"Unhandled size: {repr(in_val)}",
                CSSWarning,
                stacklevel=find_stack_level(),
            )
            return self.size_to_pt("1!!default", conversions=conversions)

        match = re.match(r"^(\S*?)([a-zA-Z%!].*)", in_val)
        if match is None:
            return _error()

        val, unit = match.groups()
        if val == "":
            # hack for 'large' etc.
            val = 1
        else:
            try:
                val = float(val)
            except ValueError:
                return _error()

        while unit != "pt":
            if unit == "em":
                if em_pt is None:
                    unit = "rem"
                else:
                    val *= em_pt
                    unit = "pt"
                continue

            try:
                unit, mul = conversions[unit]
            except KeyError:
                return _error()
            val *= mul

        val = round(val, 5)
        if int(val) == val:
            size_fmt = f"{int(val):d}pt"
        else:
            size_fmt = f"{val:f}pt"
        return size_fmt

    def atomize(self, declarations: Iterable) -> Generator[tuple[str, str], None, None]:
        for prop, value in declarations:
            prop = prop.lower()
            value = value.lower()
            if prop in self.CSS_EXPANSIONS:
                expand = self.CSS_EXPANSIONS[prop]
                yield from expand(self, prop, value)
            else:
                yield prop, value

    def parse(self, declarations_str: str) -> Iterator[tuple[str, str]]:
        """
        Generates (prop, value) pairs from declarations.

        In a future version may generate parsed tokens from tinycss/tinycss2

        Parameters
        ----------
        declarations_str : str
        """
        for decl in declarations_str.split(";"):
            if not decl.strip():
                continue
            prop, sep, val = decl.partition(":")
            prop = prop.strip().lower()
            # TODO: don't lowercase case sensitive parts of values (strings)
            val = val.strip().lower()
            if sep:
                yield prop, val
            else:
                warnings.warn(
                    f"Ill-formatted attribute: expected a colon in {repr(decl)}",
                    CSSWarning,
                    stacklevel=find_stack_level(),
                )
<|endoftext|>"""
Module for formatting output data into CSV files.
"""

from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
import csv as csvlib
import os
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.generic import (
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
)
from pandas.core.dtypes.missing import notna

from pandas.core.indexes.api import Index

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        FloatFormatType,
        IndexLabel,
        StorageOptions,
        WriteBuffer,
        npt,
    )

    from pandas.io.formats.format import DataFrameFormatter


_DEFAULT_CHUNKSIZE_CELLS = 100_000


class CSVFormatter:
    cols: npt.NDArray[np.object_]

    def __init__(
        self,
        formatter: DataFrameFormatter,
        path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] = "",
        sep: str = ",",
        cols: Sequence[Hashable] | None = None,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        encoding: str | None = None,
        errors: str = "strict",
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        lineterminator: str | None = "\n",
        chunksize: int | None = None,
        quotechar: str | None = '"',
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None:
        self.fmt = formatter

        self.obj = self.fmt.frame

        self.filepath_or_buffer = path_or_buf
        self.encoding = encoding
        self.compression: CompressionOptions = compression
        self.mode = mode
        self.storage_options = storage_options

        self.sep = sep
        self.index_label = self._initialize_index_label(index_label)
        self.errors = errors
        self.quoting = quoting or csvlib.QUOTE_MINIMAL
        self.quotechar = self._initialize_quotechar(quotechar)
        self.doublequote = doublequote
        self.escapechar = escapechar
        self.lineterminator = lineterminator or os.linesep
        self.date_format = date_format
        self.cols = self._initialize_columns(cols)
        self.chunksize = self._initialize_chunksize(chunksize)

    @property
    def na_rep(self) -> str:
        return self.fmt.na_rep

    @property
    def float_format(self) -> FloatFormatType | None:
        return self.fmt.float_format

    @property
    def decimal(self) -> str:
        return self.fmt.decimal

    @property
    def header(self) -> bool | SequenceNotStr[str]:
        return self.fmt.header

    @property
    def index(self) -> bool:
        return self.fmt.index

    def _initialize_index_label(self, index_label: IndexLabel | None) -> IndexLabel:
        if index_label is not False:
            if index_label is None:
                return self._get_index_label_from_obj()
            elif not isinstance(index_label, (list, tuple, np.ndarray, ABCIndex)):
                # given a string for a DF with Index
                return [index_label]
        return index_label

    def _get_index_label_from_obj(self) -> Sequence[Hashable]:
        if isinstance(self.obj.index, ABCMultiIndex):
            return self._get_index_label_multiindex()
        else:
            return self._get_index_label_flat()

    def _get_index_label_multiindex(self) -> Sequence[Hashable]:
        return [name or "" for name in self.obj.index.names]

    def _get_index_label_flat(self) -> Sequence[Hashable]:
        index_label = self.obj.index.name
        return [""] if index_label is None else [index_label]

    def _initialize_quotechar(self, quotechar: str | None) -> str | None:
        if self.quoting != csvlib.QUOTE_NONE:
            # prevents crash in _csv
            return quotechar
        return None

    @property
    def has_mi_columns(self) -> bool:
        return bool(isinstance(self.obj.columns, ABCMultiIndex))

    def _initialize_columns(
        self, cols: Iterable[Hashable] | None
    ) -> npt.NDArray[np.object_]:
        # validate mi options
        if self.has_mi_columns:
            if cols is not None:
                msg = "cannot specify cols with a MultiIndex on the columns"
                raise TypeError(msg)

        if cols is not None:
            if isinstance(cols, ABCIndex):
                cols = cols._get_values_for_csv(**self._number_format)
            else:
                cols = list(cols)
            self.obj = self.obj.loc[:, cols]

        # update columns to include possible multiplicity of dupes
        # and make sure cols is just a list of labels
        new_cols = self.obj.columns
        return new_cols._get_values_for_csv(**self._number_format)

    def _initialize_chunksize(self, chunksize: int | None) -> int:
        if chunksize is None:
            return (_DEFAULT_CHUNKSIZE_CELLS // (len(self.cols) or 1)) or 1
        return int(chunksize)

    @property
    def _number_format(self) -> dict[str, Any]:
        """Dictionary used for storing number formatting settings."""
        return {
            "na_rep": self.na_rep,
            "float_format": self.float_format,
            "date_format": self.date_format,
            "quoting": self.quoting,
            "decimal": self.decimal,
        }

    @cache_readonly
    def data_index(self) -> Index:
        data_index = self.obj.index
        if (
            isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex))
            and self.date_format is not None
        ):
            data_index = Index(
                [x.strftime(self.date_format) if notna(x) else "" for x in data_index]
            )
        elif isinstance(data_index, ABCMultiIndex):
            data_index = data_index.remove_unused_levels()
        return data_index

    @property
    def nlevels(self) -> int:
        if self.index:
            return getattr(self.data_index, "nlevels", 1)
        else:
            return 0

    @property
    def _has_aliases(self) -> bool:
        return isinstance(self.header, (tuple, list, np.ndarray, ABCIndex))

    @property
    def _need_to_save_header(self) -> bool:
        return bool(self._has_aliases or self.header)

    @property
    def write_cols(self) -> SequenceNotStr[Hashable]:
        if self._has_aliases:
            assert not isinstance(self.header, bool)
            if len(self.header) != len(self.cols):
                raise ValueError(
                    f"Writing {len(self.cols)} cols but got {len(self.header)} aliases"
                )
            return self.header
        else:
            # self.cols is an ndarray derived from Index._get_values_for_csv,
            #  so its entries are strings, i.e. hashable
            return cast(SequenceNotStr[Hashable], self.cols)

    @property
    def encoded_labels(self) -> list[Hashable]:
        encoded_labels: list[Hashable] = []

        if self.index and self.index_label:
            assert isinstance(self.index_label, Sequence)
            encoded_labels = list(self.index_label)

        if not self.has_mi_columns or self._has_aliases:
            encoded_labels += list(self.write_cols)

        return encoded_labels

    def save(self) -> None:
        """
        Create the writer & save.
        """
        # apply compression and byte/text conversion
        with get_handle(
            self.filepath_or_buffer,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            compression=self.compression,
            storage_options=self.storage_options,
        ) as handles:
            # Note: self.encoding is irrelevant here
            self.writer = csvlib.writer(
                handles.handle,
                lineterminator=self.lineterminator,
                delimiter=self.sep,
                quoting=self.quoting,
                doublequote=self.doublequote,
                escapechar=self.escapechar,
                quotechar=self.quotechar,
            )

            self._save()

    def _save(self) -> None:
        if self._need_to_save_header:
            self._save_header()
        self._save_body()

    def _save_header(self) -> None:
        if not self.has_mi_columns or self._has_aliases:
            self.writer.writerow(self.encoded_labels)
        else:
            for row in self._generate_multiindex_header_rows():
                self.writer.writerow(row)

    def _generate_multiindex_header_rows(self) -> Iterator[list[Hashable]]:
        columns = self.obj.columns
        for i in range(columns.nlevels):
            # we need at least 1 index column to write our col names
            col_line = []
            if self.index:
                # name is the first column
                col_line.append(columns.names[i])

                if isinstance(self.index_label, list) and len(self.index_label) > 1:
                    col_line.extend([""] * (len(self.index_label) - 1))

            col_line.extend(columns._get_level_values(i))
            yield col_line

        # Write out the index line if it's not empty.
        # Otherwise, we will print out an extraneous
        # blank line between the mi and the data rows.
        if self.encoded_labels and set(self.encoded_labels) != {""}:
            yield self.encoded_labels + [""] * len(columns)

    def _save_body(self) -> None:
        nrows = len(self.data_index)
        chunks = (nrows // self.chunksize) + 1
        for i in range(chunks):
            start_i = i * self.chunksize
            end_i = min(start_i + self.chunksize, nrows)
            if start_i >= end_i:
                break
            self._save_chunk(start_i, end_i)

    def _save_chunk(self, start_i: int, end_i: int) -> None:
        # create the data for a chunk
        slicer = slice(start_i, end_i)
        df = self.obj.iloc[slicer]

        res = df._get_values_for_csv(**self._number_format)
        data = list(res._iter_column_arrays())

        ix = self.data_index[slicer]._get_values_for_csv(**self._number_format)
        libwriters.write_csv_rows(
            data,
            ix,
            self.nlevels,
            self.cols,
            self.writer,
        )
<|endoftext|>"""
Module for formatting output data in HTML.
"""
from __future__ import annotations

from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    cast,
)

from pandas._config import get_option

from pandas._libs import lib

from pandas import (
    MultiIndex,
    option_context,
)

from pandas.io.common import is_url
from pandas.io.formats.format import (
    DataFrameFormatter,
    get_level_lengths,
)
from pandas.io.formats.printing import pprint_thing

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Mapping,
    )


class HTMLFormatter:
    """
    Internal class for formatting output data in html.
    This class is intended for shared functionality between
    DataFrame.to_html() and DataFrame._repr_html_().
    Any logic in common with other output formatting methods
    should ideally be inherited from classes in format.py
    and this class responsible for only producing html markup.
    """

    indent_delta: Final = 2

    def __init__(
        self,
        formatter: DataFrameFormatter,
        classes: str | list[str] | tuple[str, ...] | None = None,
        border: int | bool | None = None,
        table_id: str | None = None,
        render_links: bool = False,
    ) -> None:
        self.fmt = formatter
        self.classes = classes

        self.frame = self.fmt.frame
        self.columns = self.fmt.tr_frame.columns
        self.elements: list[str] = []
        self.bold_rows = self.fmt.bold_rows
        self.escape = self.fmt.escape
        self.show_dimensions = self.fmt.show_dimensions
        if border is None or border is True:
            border = cast(int, get_option("display.html.border"))
        elif not border:
            border = None

        self.border = border
        self.table_id = table_id
        self.render_links = render_links

        self.col_space = {}
        is_multi_index = isinstance(self.columns, MultiIndex)
        for column, value in self.fmt.col_space.items():
            col_space_value = f"{value}px" if isinstance(value, int) else value
            self.col_space[column] = col_space_value
            # GH 53885: Handling case where column is index
            # Flatten the data in the multi index and add in the map
            if is_multi_index and isinstance(column, tuple):
                for column_index in column:
                    self.col_space[str(column_index)] = col_space_value

    def to_string(self) -> str:
        lines = self.render()
        if any(isinstance(x, str) for x in lines):
            lines = [str(x) for x in lines]
        return "\n".join(lines)

    def render(self) -> list[str]:
        self._write_table()

        if self.should_show_dimensions:
            by = chr(215)  # ×  # noqa: RUF003
            self.write(
                f"<p>{len(self.frame)} rows {by} {len(self.frame.columns)} columns</p>"
            )

        return self.elements

    @property
    def should_show_dimensions(self) -> bool:
        return self.fmt.should_show_dimensions

    @property
    def show_row_idx_names(self) -> bool:
        return self.fmt.show_row_idx_names

    @property
    def show_col_idx_names(self) -> bool:
        return self.fmt.show_col_idx_names

    @property
    def row_levels(self) -> int:
        if self.fmt.index:
            # showing (row) index
            return self.frame.index.nlevels
        elif self.show_col_idx_names:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # If the row index is not displayed a column of
            # blank cells need to be included before the DataFrame values.
            return 1
        # not showing (row) index
        return 0

    def _get_columns_formatted_values(self) -> Iterable:
        return self.columns

    @property
    def is_truncated(self) -> bool:
        return self.fmt.is_truncated

    @property
    def ncols(self) -> int:
        return len(self.fmt.tr_frame.columns)

    def write(self, s: Any, indent: int = 0) -> None:
        rs = pprint_thing(s)
        self.elements.append(" " * indent + rs)

    def write_th(
        self, s: Any, header: bool = False, indent: int = 0, tags: str | None = None
    ) -> None:
        """
        Method for writing a formatted <th> cell.

        If col_space is set on the formatter then that is used for
        the value of min-width.

        Parameters
        ----------
        s : object
            The data to be written inside the cell.
        header : bool, default False
            Set to True if the <th> is for use inside <thead>.  This will
            cause min-width to be set if there is one.
        indent : int, default 0
            The indentation level of the cell.
        tags : str, default None
            Tags to include in the cell.

        Returns
        -------
        A written <th> cell.
        """
        col_space = self.col_space.get(s, None)

        if header and col_space is not None:
            tags = tags or ""
            tags += f'style="min-width: {col_space};"'

        self._write_cell(s, kind="th", indent=indent, tags=tags)

    def write_td(self, s: Any, indent: int = 0, tags: str | None = None) -> None:
        self._write_cell(s, kind="td", indent=indent, tags=tags)

    def _write_cell(
        self, s: Any, kind: str = "td", indent: int = 0, tags: str | None = None
    ) -> None:
        if tags is not None:
            start_tag = f"<{kind} {tags}>"
        else:
            start_tag = f"<{kind}>"

        if self.escape:
            # escape & first to prevent double escaping of &
            esc = {"&": r"&amp;", "<": r"&lt;", ">": r"&gt;"}
        else:
            esc = {}

        rs = pprint_thing(s, escape_chars=esc).strip()

        if self.render_links and is_url(rs):
            rs_unescaped = pprint_thing(s, escape_chars={}).strip()
            start_tag += f'<a href="{rs_unescaped}" target="_blank">'
            end_a = "</a>"
        else:
            end_a = ""

        self.write(f"{start_tag}{rs}{end_a}</{kind}>", indent)

    def write_tr(
        self,
        line: Iterable,
        indent: int = 0,
        indent_delta: int = 0,
        header: bool = False,
        align: str | None = None,
        tags: dict[int, str] | None = None,
        nindex_levels: int = 0,
    ) -> None:
        if tags is None:
            tags = {}

        if align is None:
            self.write("<tr>", indent)
        else:
            self.write(f'<tr style="text-align: {align};">', indent)
        indent += indent_delta

        for i, s in enumerate(line):
            val_tag = tags.get(i, None)
            if header or (self.bold_rows and i < nindex_levels):
                self.write_th(s, indent=indent, header=header, tags=val_tag)
            else:
                self.write_td(s, indent, tags=val_tag)

        indent -= indent_delta
        self.write("</tr>", indent)

    def _write_table(self, indent: int = 0) -> None:
        _classes = ["dataframe"]  # Default class.
        use_mathjax = get_option("display.html.use_mathjax")
        if not use_mathjax:
            _classes.append("tex2jax_ignore")
        if self.classes is not None:
            if isinstance(self.classes, str):
                self.classes = self.classes.split()
            if not isinstance(self.classes, (list, tuple)):
                raise TypeError(
                    "classes must be a string, list, "
                    f"or tuple, not {type(self.classes)}"
                )
            _classes.extend(self.classes)

        if self.table_id is None:
            id_section = ""
        else:
            id_section = f' id="{self.table_id}"'

        if self.border is None:
            border_attr = ""
        else:
            border_attr = f' border="{self.border}"'

        self.write(
            f'<table{border_attr} class="{" ".join(_classes)}"{id_section}>',
            indent,
        )

        if self.fmt.header or self.show_row_idx_names:
            self._write_header(indent + self.indent_delta)

        self._write_body(indent + self.indent_delta)

        self.write("</table>", indent)

    def _write_col_header(self, indent: int) -> None:
        row: list[Hashable]
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        if isinstance(self.columns, MultiIndex):
            template = 'colspan="{span:d}" halign="left"'

            sentinel: lib.NoDefault | bool
            if self.fmt.sparsify:
                # GH3547
                sentinel = lib.no_default
            else:
                sentinel = False
            levels = self.columns._format_multi(sparsify=sentinel, include_names=False)
            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            for lnum, (records, values) in enumerate(zip(level_lengths, levels)):
                if is_truncated_horizontally:
                    # modify the header lines
                    ins_col = self.fmt.tr_col_num
                    if self.fmt.sparsify:
                        recs_new = {}
                        # Increment tags after ... col.
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            elif tag + span > ins_col:
                                recs_new[tag] = span + 1
                                if lnum == inner_lvl:
                                    values = (
                                        values[:ins_col] + ("...",) + values[ins_col:]
                                    )
                                else:
                                    # sparse col headers do not receive a ...
                                    values = (
                                        values[:ins_col]
                                        + (values[ins_col - 1],)
                                        + values[ins_col:]
                                    )
                            else:
                                recs_new[tag] = span
                            # if ins_col lies between tags, all col headers
                            # get ...
                            if tag + span == ins_col:
                                recs_new[ins_col] = 1
                                values = values[:ins_col] + ("...",) + values[ins_col:]
                        records = recs_new
                        inner_lvl = len(level_lengths) - 1
                        if lnum == inner_lvl:
                            records[ins_col] = 1
                    else:
                        recs_new = {}
                        for tag, span in list(records.items()):
                            if tag >= ins_col:
                                recs_new[tag + 1] = span
                            else:
                                recs_new[tag] = span
                        recs_new[ins_col] = 1
                        records = recs_new
                        values = values[:ins_col] + ["..."] + values[ins_col:]

                # see gh-22579
                # Column Offset Bug with to_html(index=False) with
                # MultiIndex Columns and Index.
                # Initially fill row with blank cells before column names.
                # TODO: Refactor to remove code duplication with code
                # block below for standard columns index.
                row = [""] * (self.row_levels - 1)
                if self.fmt.index or self.show_col_idx_names:
                    # see gh-22747
                    # If to_html(index_names=False) do not show columns
                    # index names.
                    # TODO: Refactor to use _get_column_name_list from
                    # DataFrameFormatter class and create a
                    # _get_formatted_column_labels function for code
                    # parity with DataFrameFormatter class.
                    if self.fmt.show_index_names:
                        name = self.columns.names[lnum]
                        row.append(pprint_thing(name or ""))
                    else:
                        row.append("")

                tags = {}
                j = len(row)
                for i, v in enumerate(values):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template.format(span=records[i])
                    else:
                        continue
                    j += 1
                    row.append(v)
                self.write_tr(row, indent, self.indent_delta, tags=tags, header=True)
        else:
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Initially fill row with blank cells before column names.
            # TODO: Refactor to remove code duplication with code block
            # above for columns MultiIndex.
            row = [""] * (self.row_levels - 1)
            if self.fmt.index or self.show_col_idx_names:
                # see gh-22747
                # If to_html(index_names=False) do not show columns
                # index names.
                # TODO: Refactor to use _get_column_name_list from
                # DataFrameFormatter class.
                if self.fmt.show_index_names:
                    row.append(self.columns.name or "")
                else:
                    row.append("")
            row.extend(self._get_columns_formatted_values())
            align = self.fmt.justify

            if is_truncated_horizontally:
                ins_col = self.row_levels + self.fmt.tr_col_num
                row.insert(ins_col, "...")

            self.write_tr(row, indent, self.indent_delta, header=True, align=align)

    def _write_row_header(self, indent: int) -> None:
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        row = [x if x is not None else "" for x in self.frame.index.names] + [""] * (
            self.ncols + (1 if is_truncated_horizontally else 0)
        )
        self.write_tr(row, indent, self.indent_delta, header=True)

    def _write_header(self, indent: int) -> None:
        self.write("<thead>", indent)

        if self.fmt.header:
            self._write_col_header(indent + self.indent_delta)

        if self.show_row_idx_names:
            self._write_row_header(indent + self.indent_delta)

        self.write("</thead>", indent)

    def _get_formatted_values(self) -> dict[int, list[str]]:
        with option_context("display.max_colwidth", None):
            fmt_values = {i: self.fmt.format_col(i) for i in range(self.ncols)}
        return fmt_values

    def _write_body(self, indent: int) -> None:
        self.write("<tbody>", indent)
        fmt_values = self._get_formatted_values()

        # write values
        if self.fmt.index and isinstance(self.frame.index, MultiIndex):
            self._write_hierarchical_rows(fmt_values, indent + self.indent_delta)
        else:
            self._write_regular_rows(fmt_values, indent + self.indent_delta)

        self.write("</tbody>", indent)

    def _write_regular_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
    ) -> None:
        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically

        nrows = len(self.fmt.tr_frame)

        if self.fmt.index:
            fmt = self.fmt._get_formatter("__index__")
            if fmt is not None:
                index_values = self.fmt.tr_frame.index.map(fmt)
            else:
                # only reached with non-Multi index
                index_values = self.fmt.tr_frame.index._format_flat(include_name=False)

        row: list[str] = []
        for i in range(nrows):
            if is_truncated_vertically and i == (self.fmt.tr_row_num):
                str_sep_row = ["..."] * len(row)
                self.write_tr(
                    str_sep_row,
                    indent,
                    self.indent_delta,
                    tags=None,
                    nindex_levels=self.row_levels,
                )

            row = []
            if self.fmt.index:
                row.append(index_values[i])
            # see gh-22579
            # Column misalignment also occurs for
            # a standard index when the columns index is named.
            # Add blank cell before data cells.
            elif self.show_col_idx_names:
                row.append("")
            row.extend(fmt_values[j][i] for j in range(self.ncols))

            if is_truncated_horizontally:
                dot_col_ix = self.fmt.tr_col_num + self.row_levels
                row.insert(dot_col_ix, "...")
            self.write_tr(
                row, indent, self.indent_delta, tags=None, nindex_levels=self.row_levels
            )

    def _write_hierarchical_rows(
        self, fmt_values: Mapping[int, list[str]], indent: int
    ) -> None:
        template = 'rowspan="{span}" valign="top"'

        is_truncated_horizontally = self.fmt.is_truncated_horizontally
        is_truncated_vertically = self.fmt.is_truncated_vertically
        frame = self.fmt.tr_frame
        nrows = len(frame)

        assert isinstance(frame.index, MultiIndex)
        idx_values = frame.index._format_multi(sparsify=False, include_names=False)
        idx_values = list(zip(*idx_values))

        if self.fmt.sparsify:
            # GH3547
            sentinel = lib.no_default
            levels = frame.index._format_multi(sparsify=sentinel, include_names=False)

            level_lengths = get_level_lengths(levels, sentinel)
            inner_lvl = len(level_lengths) - 1
            if is_truncated_vertically:
                # Insert ... row and adjust idx_values and
                # level_lengths to take this into account.
                ins_row = self.fmt.tr_row_num
                inserted = False
                for lnum, records in enumerate(level_lengths):
                    rec_new = {}
                    for tag, span in list(records.items()):
                        if tag >= ins_row:
                            rec_new[tag + 1] = span
                        elif tag + span > ins_row:
                            rec_new[tag] = span + 1

                            # GH 14882 - Make sure insertion done once
                            if not inserted:
                                dot_row = list(idx_values[ins_row - 1])
                                dot_row[-1] = "..."
                                idx_values.insert(ins_row, tuple(dot_row))
                                inserted = True
                            else:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = "..."
                                idx_values[ins_row] = tuple(dot_row)
                        else:
                            rec_new[tag] = span
                        # If ins_row lies between tags, all cols idx cols
                        # receive ...
                        if tag + span == ins_row:
                            rec_new[ins_row] = 1
                            if lnum == 0:
                                idx_values.insert(
                                    ins_row, tuple(["..."] * len(level_lengths))
                                )

                            # GH 14882 - Place ... in correct level
                            elif inserted:
                                dot_row = list(idx_values[ins_row])
                                dot_row[inner_lvl - lnum] = "..."
                                idx_values[ins_row] = tuple(dot_row)
                    level_lengths[lnum] = rec_new

                level_lengths[inner_lvl][ins_row] = 1
                for ix_col in fmt_values:
                    fmt_values[ix_col].insert(ins_row, "...")
                nrows += 1

            for i in range(nrows):
                row = []
                tags = {}

                sparse_offset = 0
                j = 0
                for records, v in zip(level_lengths, idx_values[i]):
                    if i in records:
                        if records[i] > 1:
                            tags[j] = template.format(span=records[i])
                    else:
                        sparse_offset += 1
                        continue

                    j += 1
                    row.append(v)

                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if is_truncated_horizontally:
                    row.insert(
                        self.row_levels - sparse_offset + self.fmt.tr_col_num, "..."
                    )
                self.write_tr(
                    row,
                    indent,
                    self.indent_delta,
                    tags=tags,
                    nindex_levels=len(levels) - sparse_offset,
                )
        else:
            row = []
            for i in range(len(frame)):
                if is_truncated_vertically and i == (self.fmt.tr_row_num):
                    str_sep_row = ["..."] * len(row)
                    self.write_tr(
                        str_sep_row,
                        indent,
                        self.indent_delta,
                        tags=None,
                        nindex_levels=self.row_levels,
                    )

                idx_values = list(
                    zip(*frame.index._format_multi(sparsify=False, include_names=False))
                )
                row = []
                row.extend(idx_values[i])
                row.extend(fmt_values[j][i] for j in range(self.ncols))
                if is_truncated_horizontally:
                    row.insert(self.row_levels + self.fmt.tr_col_num, "...")
                self.write_tr(
                    row,
                    indent,
                    self.indent_delta,
                    tags=None,
                    nindex_levels=frame.index.nlevels,
                )


class NotebookFormatter(HTMLFormatter):
    """
    Internal class for formatting output data in html for display in Jupyter
    Notebooks. This class is intended for functionality specific to
    DataFrame._repr_html_() and DataFrame.to_html(notebook=True)
    """

    def _get_formatted_values(self) -> dict[int, list[str]]:
        return {i: self.fmt.format_col(i) for i in range(self.ncols)}

    def _get_columns_formatted_values(self) -> list[str]:
        # only reached with non-Multi Index
        return self.columns._format_flat(include_name=False)

    def write_style(self) -> None:
        # We use the "scoped" attribute here so that the desired
        # style properties for the data frame are not then applied
        # throughout the entire notebook.
        template_first = """\
            <style scoped>"""
        template_last = """\
            </style>"""
        template_select = """\
                .dataframe %s {
                    %s: %s;
                }"""
        element_props = [
            ("tbody tr th:only-of-type", "vertical-align", "middle"),
            ("tbody tr th", "vertical-align", "top"),
        ]
        if isinstance(self.columns, MultiIndex):
            element_props.append(("thead tr th", "text-align", "left"))
            if self.show_row_idx_names:
                element_props.append(
                    ("thead tr:last-of-type th", "text-align", "right")
                )
        else:
            element_props.append(("thead th", "text-align", "right"))
        template_mid = "\n\n".join(template_select % t for t in element_props)
        template = dedent(f"{template_first}\n{template_mid}\n{template_last}")
        self.write(template)

    def render(self) -> list[str]:
        self.write("<div>")
        self.write_style()
        super().render()
        self.write("</div>")
        return self.elements
<|endoftext|># ruff: noqa: TCH004
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # import modules that have public classes/functions
    from pandas.io.formats import style

    # and mark only those modules as public
    __all__ = ["style"]
<|endoftext|>"""
Internal module for formatting output data in csv, html, xml,
and latex files. This module also applies to display formatting.
"""
from __future__ import annotations

from collections.abc import (
    Generator,
    Hashable,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    cast,
)

import numpy as np

from pandas._config.config import (
    get_option,
    set_option,
)

from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
    NaT,
    Timedelta,
    Timestamp,
)
from pandas._libs.tslibs.nattype import NaTType

from pandas.core.dtypes.common import (
    is_complex_dtype,
    is_float,
    is_integer,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core.arrays import (
    Categorical,
    DatetimeArray,
    ExtensionArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    PeriodIndex,
    ensure_index,
)
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat

from pandas.io.common import (
    check_parent_directory,
    stringify_path,
)
from pandas.io.formats import printing

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Axes,
        ColspaceArgType,
        ColspaceType,
        CompressionOptions,
        FilePath,
        FloatFormatType,
        FormattersType,
        IndexLabel,
        SequenceNotStr,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import (
        DataFrame,
        Series,
    )


common_docstring: Final = """
        Parameters
        ----------
        buf : str, Path or StringIO-like, optional, default None
            Buffer to write to. If None, the output is returned as a string.
        columns : array-like, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : %(col_space_type)s, optional
            %(col_space)s.
        header : %(header_type)s, optional
            %(header)s.
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of ``NaN`` to use.
        formatters : list, tuple or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List/tuple must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. This function must return a unicode string and will be
            applied only to the non-``NaN`` elements, with ``NaN`` being
            handled by ``na_rep``.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
    """

VALID_JUSTIFY_PARAMETERS = (
    "left",
    "right",
    "center",
    "justify",
    "justify-all",
    "start",
    "end",
    "inherit",
    "match-parent",
    "initial",
    "unset",
)

return_docstring: Final = """
        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns
            None.
    """


class SeriesFormatter:
    """
    Implement the main logic of Series.to_string, which underlies
    Series.__repr__.
    """

    def __init__(
        self,
        series: Series,
        *,
        length: bool | str = True,
        header: bool = True,
        index: bool = True,
        na_rep: str = "NaN",
        name: bool = False,
        float_format: str | None = None,
        dtype: bool = True,
        max_rows: int | None = None,
        min_rows: int | None = None,
    ) -> None:
        self.series = series
        self.buf = StringIO()
        self.name = name
        self.na_rep = na_rep
        self.header = header
        self.length = length
        self.index = index
        self.max_rows = max_rows
        self.min_rows = min_rows

        if float_format is None:
            float_format = get_option("display.float_format")
        self.float_format = float_format
        self.dtype = dtype
        self.adj = printing.get_adjustment()

        self._chk_truncate()

    def _chk_truncate(self) -> None:
        self.tr_row_num: int | None

        min_rows = self.min_rows
        max_rows = self.max_rows
        # truncation determined by max_rows, actual truncated number of rows
        # used below by min_rows
        is_truncated_vertically = max_rows and (len(self.series) > max_rows)
        series = self.series
        if is_truncated_vertically:
            max_rows = cast(int, max_rows)
            if min_rows:
                # if min_rows is set (not None or 0), set max_rows to minimum
                # of both
                max_rows = min(min_rows, max_rows)
            if max_rows == 1:
                row_num = max_rows
                series = series.iloc[:max_rows]
            else:
                row_num = max_rows // 2
                series = concat((series.iloc[:row_num], series.iloc[-row_num:]))
            self.tr_row_num = row_num
        else:
            self.tr_row_num = None
        self.tr_series = series
        self.is_truncated_vertically = is_truncated_vertically

    def _get_footer(self) -> str:
        name = self.series.name
        footer = ""

        index = self.series.index
        if (
            isinstance(index, (DatetimeIndex, PeriodIndex, TimedeltaIndex))
            and index.freq is not None
        ):
            footer += f"Freq: {index.freqstr}"

        if self.name is not False and name is not None:
            if footer:
                footer += ", "

            series_name = printing.pprint_thing(name, escape_chars=("\t", "\r", "\n"))
            footer += f"Name: {series_name}"

        if self.length is True or (
            self.length == "truncate" and self.is_truncated_vertically
        ):
            if footer:
                footer += ", "
            footer += f"Length: {len(self.series)}"

        if self.dtype is not False and self.dtype is not None:
            dtype_name = getattr(self.tr_series.dtype, "name", None)
            if dtype_name:
                if footer:
                    footer += ", "
                footer += f"dtype: {printing.pprint_thing(dtype_name)}"

        # level infos are added to the end and in a new line, like it is done
        # for Categoricals
        if isinstance(self.tr_series.dtype, CategoricalDtype):
            level_info = self.tr_series._values._get_repr_footer()
            if footer:
                footer += "\n"
            footer += level_info

        return str(footer)

    def _get_formatted_values(self) -> list[str]:
        return format_array(
            self.tr_series._values,
            None,
            float_format=self.float_format,
            na_rep=self.na_rep,
            leading_space=self.index,
        )

    def to_string(self) -> str:
        series = self.tr_series
        footer = self._get_footer()

        if len(series) == 0:
            return f"{type(self.series).__name__}([], {footer})"

        index = series.index
        have_header = _has_names(index)
        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(include_names=True, sparsify=None)
            adj = printing.get_adjustment()
            fmt_index = adj.adjoin(2, *fmt_index).split("\n")
        else:
            fmt_index = index._format_flat(include_name=True)
        fmt_values = self._get_formatted_values()

        if self.is_truncated_vertically:
            n_header_rows = 0
            row_num = self.tr_row_num
            row_num = cast(int, row_num)
            width = self.adj.len(fmt_values[row_num - 1])
            if width > 3:
                dot_str = "..."
            else:
                dot_str = ".."
            # Series uses mode=center because it has single value columns
            # DataFrame uses mode=left
            dot_str = self.adj.justify([dot_str], width, mode="center")[0]
            fmt_values.insert(row_num + n_header_rows, dot_str)
            fmt_index.insert(row_num + 1, "")

        if self.index:
            result = self.adj.adjoin(3, *[fmt_index[1:], fmt_values])
        else:
            result = self.adj.adjoin(3, fmt_values)

        if self.header and have_header:
            result = fmt_index[0] + "\n" + result

        if footer:
            result += "\n" + footer

        return str("".join(result))


def get_dataframe_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(dataFrame) calls using DataFrame.to_string.

    Supplying these parameters to DataFrame.to_string is equivalent to calling
    ``repr(DataFrame)``. This is useful if you want to adjust the repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame([[1, 2], [3, 4]])
    >>> repr_params = pd.io.formats.format.get_dataframe_repr_params()
    >>> repr(df) == df.to_string(**repr_params)
    True
    """
    from pandas.io.formats import console

    if get_option("display.expand_frame_repr"):
        line_width, _ = console.get_console_size()
    else:
        line_width = None
    return {
        "max_rows": get_option("display.max_rows"),
        "min_rows": get_option("display.min_rows"),
        "max_cols": get_option("display.max_columns"),
        "max_colwidth": get_option("display.max_colwidth"),
        "show_dimensions": get_option("display.show_dimensions"),
        "line_width": line_width,
    }


def get_series_repr_params() -> dict[str, Any]:
    """Get the parameters used to repr(Series) calls using Series.to_string.

    Supplying these parameters to Series.to_string is equivalent to calling
    ``repr(series)``. This is useful if you want to adjust the series repr output.

    .. versionadded:: 1.4.0

    Example
    -------
    >>> import pandas as pd
    >>>
    >>> ser = pd.Series([1, 2, 3, 4])
    >>> repr_params = pd.io.formats.format.get_series_repr_params()
    >>> repr(ser) == ser.to_string(**repr_params)
    True
    """
    width, height = get_terminal_size()
    max_rows_opt = get_option("display.max_rows")
    max_rows = height if max_rows_opt == 0 else max_rows_opt
    min_rows = height if max_rows_opt == 0 else get_option("display.min_rows")

    return {
        "name": True,
        "dtype": True,
        "min_rows": min_rows,
        "max_rows": max_rows,
        "length": get_option("display.show_dimensions"),
    }


class DataFrameFormatter:
    """
    Class for processing dataframe formatting options and data.

    Used by DataFrame.to_string, which backs DataFrame.__repr__.
    """

    __doc__ = __doc__ if __doc__ else ""
    __doc__ += common_docstring + return_docstring

    def __init__(
        self,
        frame: DataFrame,
        columns: Axes | None = None,
        col_space: ColspaceArgType | None = None,
        header: bool | SequenceNotStr[str] = True,
        index: bool = True,
        na_rep: str = "NaN",
        formatters: FormattersType | None = None,
        justify: str | None = None,
        float_format: FloatFormatType | None = None,
        sparsify: bool | None = None,
        index_names: bool = True,
        max_rows: int | None = None,
        min_rows: int | None = None,
        max_cols: int | None = None,
        show_dimensions: bool | str = False,
        decimal: str = ".",
        bold_rows: bool = False,
        escape: bool = True,
    ) -> None:
        self.frame = frame
        self.columns = self._initialize_columns(columns)
        self.col_space = self._initialize_colspace(col_space)
        self.header = header
        self.index = index
        self.na_rep = na_rep
        self.formatters = self._initialize_formatters(formatters)
        self.justify = self._initialize_justify(justify)
        self.float_format = float_format
        self.sparsify = self._initialize_sparsify(sparsify)
        self.show_index_names = index_names
        self.decimal = decimal
        self.bold_rows = bold_rows
        self.escape = escape
        self.max_rows = max_rows
        self.min_rows = min_rows
        self.max_cols = max_cols
        self.show_dimensions = show_dimensions

        self.max_cols_fitted = self._calc_max_cols_fitted()
        self.max_rows_fitted = self._calc_max_rows_fitted()

        self.tr_frame = self.frame
        self.truncate()
        self.adj = printing.get_adjustment()

    def get_strcols(self) -> list[list[str]]:
        """
        Render a DataFrame to a list of columns (as lists of strings).
        """
        strcols = self._get_strcols_without_index()

        if self.index:
            str_index = self._get_formatted_index(self.tr_frame)
            strcols.insert(0, str_index)

        return strcols

    @property
    def should_show_dimensions(self) -> bool:
        return self.show_dimensions is True or (
            self.show_dimensions == "truncate" and self.is_truncated
        )

    @property
    def is_truncated(self) -> bool:
        return bool(self.is_truncated_horizontally or self.is_truncated_vertically)

    @property
    def is_truncated_horizontally(self) -> bool:
        return bool(self.max_cols_fitted and (len(self.columns) > self.max_cols_fitted))

    @property
    def is_truncated_vertically(self) -> bool:
        return bool(self.max_rows_fitted and (len(self.frame) > self.max_rows_fitted))

    @property
    def dimensions_info(self) -> str:
        return f"\n\n[{len(self.frame)} rows x {len(self.frame.columns)} columns]"

    @property
    def has_index_names(self) -> bool:
        return _has_names(self.frame.index)

    @property
    def has_column_names(self) -> bool:
        return _has_names(self.frame.columns)

    @property
    def show_row_idx_names(self) -> bool:
        return all((self.has_index_names, self.index, self.show_index_names))

    @property
    def show_col_idx_names(self) -> bool:
        return all((self.has_column_names, self.show_index_names, self.header))

    @property
    def max_rows_displayed(self) -> int:
        return min(self.max_rows or len(self.frame), len(self.frame))

    def _initialize_sparsify(self, sparsify: bool | None) -> bool:
        if sparsify is None:
            return get_option("display.multi_sparse")
        return sparsify

    def _initialize_formatters(
        self, formatters: FormattersType | None
    ) -> FormattersType:
        if formatters is None:
            return {}
        elif len(self.frame.columns) == len(formatters) or isinstance(formatters, dict):
            return formatters
        else:
            raise ValueError(
                f"Formatters length({len(formatters)}) should match "
                f"DataFrame number of columns({len(self.frame.columns)})"
            )

    def _initialize_justify(self, justify: str | None) -> str:
        if justify is None:
            return get_option("display.colheader_justify")
        else:
            return justify

    def _initialize_columns(self, columns: Axes | None) -> Index:
        if columns is not None:
            cols = ensure_index(columns)
            self.frame = self.frame[cols]
            return cols
        else:
            return self.frame.columns

    def _initialize_colspace(self, col_space: ColspaceArgType | None) -> ColspaceType:
        result: ColspaceType

        if col_space is None:
            result = {}
        elif isinstance(col_space, (int, str)):
            result = {"": col_space}
            result.update({column: col_space for column in self.frame.columns})
        elif isinstance(col_space, Mapping):
            for column in col_space.keys():
                if column not in self.frame.columns and column != "":
                    raise ValueError(
                        f"Col_space is defined for an unknown column: {column}"
                    )
            result = col_space
        else:
            if len(self.frame.columns) != len(col_space):
                raise ValueError(
                    f"Col_space length({len(col_space)}) should match "
                    f"DataFrame number of columns({len(self.frame.columns)})"
                )
            result = dict(zip(self.frame.columns, col_space))
        return result

    def _calc_max_cols_fitted(self) -> int | None:
        """Number of columns fitting the screen."""
        if not self._is_in_terminal():
            return self.max_cols

        width, _ = get_terminal_size()
        if self._is_screen_narrow(width):
            return width
        else:
            return self.max_cols

    def _calc_max_rows_fitted(self) -> int | None:
        """Number of rows with data fitting the screen."""
        max_rows: int | None

        if self._is_in_terminal():
            _, height = get_terminal_size()
            if self.max_rows == 0:
                # rows available to fill with actual data
                return height - self._get_number_of_auxiliary_rows()

            if self._is_screen_short(height):
                max_rows = height
            else:
                max_rows = self.max_rows
        else:
            max_rows = self.max_rows

        return self._adjust_max_rows(max_rows)

    def _adjust_max_rows(self, max_rows: int | None) -> int | None:
        """Adjust max_rows using display logic.

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options

        GH #37359
        """
        if max_rows:
            if (len(self.frame) > max_rows) and self.min_rows:
                # if truncated, set max_rows showed to min_rows
                max_rows = min(self.min_rows, max_rows)
        return max_rows

    def _is_in_terminal(self) -> bool:
        """Check if the output is to be shown in terminal."""
        return bool(self.max_cols == 0 or self.max_rows == 0)

    def _is_screen_narrow(self, max_width) -> bool:
        return bool(self.max_cols == 0 and len(self.frame.columns) > max_width)

    def _is_screen_short(self, max_height) -> bool:
        return bool(self.max_rows == 0 and len(self.frame) > max_height)

    def _get_number_of_auxiliary_rows(self) -> int:
        """Get number of rows occupied by prompt, dots and dimension info."""
        dot_row = 1
        prompt_row = 1
        num_rows = dot_row + prompt_row

        if self.show_dimensions:
            num_rows += len(self.dimensions_info.splitlines())

        if self.header:
            num_rows += 1

        return num_rows

    def truncate(self) -> None:
        """
        Check whether the frame should be truncated. If so, slice the frame up.
        """
        if self.is_truncated_horizontally:
            self._truncate_horizontally()

        if self.is_truncated_vertically:
            self._truncate_vertically()

    def _truncate_horizontally(self) -> None:
        """Remove columns, which are not to be displayed and adjust formatters.

        Attributes affected:
            - tr_frame
            - formatters
            - tr_col_num
        """
        assert self.max_cols_fitted is not None
        col_num = self.max_cols_fitted // 2
        if col_num >= 1:
            left = self.tr_frame.iloc[:, :col_num]
            right = self.tr_frame.iloc[:, -col_num:]
            self.tr_frame = concat((left, right), axis=1)

            # truncate formatter
            if isinstance(self.formatters, (list, tuple)):
                self.formatters = [
                    *self.formatters[:col_num],
                    *self.formatters[-col_num:],
                ]
        else:
            col_num = cast(int, self.max_cols)
            self.tr_frame = self.tr_frame.iloc[:, :col_num]
        self.tr_col_num = col_num

    def _truncate_vertically(self) -> None:
        """Remove rows, which are not to be displayed.

        Attributes affected:
            - tr_frame
            - tr_row_num
        """
        assert self.max_rows_fitted is not None
        row_num = self.max_rows_fitted // 2
        if row_num >= 1:
            _len = len(self.tr_frame)
            _slice = np.hstack([np.arange(row_num), np.arange(_len - row_num, _len)])
            self.tr_frame = self.tr_frame.iloc[_slice]
        else:
            row_num = cast(int, self.max_rows)
            self.tr_frame = self.tr_frame.iloc[:row_num, :]
        self.tr_row_num = row_num

    def _get_strcols_without_index(self) -> list[list[str]]:
        strcols: list[list[str]] = []

        if not is_list_like(self.header) and not self.header:
            for i, c in enumerate(self.tr_frame):
                fmt_values = self.format_col(i)
                fmt_values = _make_fixed_width(
                    strings=fmt_values,
                    justify=self.justify,
                    minimum=int(self.col_space.get(c, 0)),
                    adj=self.adj,
                )
                strcols.append(fmt_values)
            return strcols

        if is_list_like(self.header):
            # cast here since can't be bool if is_list_like
            self.header = cast(list[str], self.header)
            if len(self.header) != len(self.columns):
                raise ValueError(
                    f"Writing {len(self.columns)} cols "
                    f"but got {len(self.header)} aliases"
                )
            str_columns = [[label] for label in self.header]
        else:
            str_columns = self._get_formatted_column_labels(self.tr_frame)

        if self.show_row_idx_names:
            for x in str_columns:
                x.append("")

        for i, c in enumerate(self.tr_frame):
            cheader = str_columns[i]
            header_colwidth = max(
                int(self.col_space.get(c, 0)), *(self.adj.len(x) for x in cheader)
            )
            fmt_values = self.format_col(i)
            fmt_values = _make_fixed_width(
                fmt_values, self.justify, minimum=header_colwidth, adj=self.adj
            )

            max_len = max(*(self.adj.len(x) for x in fmt_values), header_colwidth)
            cheader = self.adj.justify(cheader, max_len, mode=self.justify)
            strcols.append(cheader + fmt_values)

        return strcols

    def format_col(self, i: int) -> list[str]:
        frame = self.tr_frame
        formatter = self._get_formatter(i)
        return format_array(
            frame.iloc[:, i]._values,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            space=self.col_space.get(frame.columns[i]),
            decimal=self.decimal,
            leading_space=self.index,
        )

    def _get_formatter(self, i: str | int) -> Callable | None:
        if isinstance(self.formatters, (list, tuple)):
            if is_integer(i):
                i = cast(int, i)
                return self.formatters[i]
            else:
                return None
        else:
            if is_integer(i) and i not in self.columns:
                i = self.columns[i]
            return self.formatters.get(i, None)

    def _get_formatted_column_labels(self, frame: DataFrame) -> list[list[str]]:
        from pandas.core.indexes.multi import sparsify_labels

        columns = frame.columns

        if isinstance(columns, MultiIndex):
            fmt_columns = columns._format_multi(sparsify=False, include_names=False)
            fmt_columns = list(zip(*fmt_columns))
            dtypes = self.frame.dtypes._values

            # if we have a Float level, they don't use leading space at all
            restrict_formatting = any(level.is_floating for level in columns.levels)
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))

            def space_format(x, y):
                if (
                    y not in self.formatters
                    and need_leadsp[x]
                    and not restrict_formatting
                ):
                    return " " + y
                return y

            str_columns_tuple = list(
                zip(*([space_format(x, y) for y in x] for x in fmt_columns))
            )
            if self.sparsify and len(str_columns_tuple):
                str_columns_tuple = sparsify_labels(str_columns_tuple)

            str_columns = [list(x) for x in zip(*str_columns_tuple)]
        else:
            fmt_columns = columns._format_flat(include_name=False)
            dtypes = self.frame.dtypes
            need_leadsp = dict(zip(fmt_columns, map(is_numeric_dtype, dtypes)))
            str_columns = [
                [" " + x if not self._get_formatter(i) and need_leadsp[x] else x]
                for i, x in enumerate(fmt_columns)
            ]
        # self.str_columns = str_columns
        return str_columns

    def _get_formatted_index(self, frame: DataFrame) -> list[str]:
        # Note: this is only used by to_string() and to_latex(), not by
        # to_html(). so safe to cast col_space here.
        col_space = {k: cast(int, v) for k, v in self.col_space.items()}
        index = frame.index
        columns = frame.columns
        fmt = self._get_formatter("__index__")

        if isinstance(index, MultiIndex):
            fmt_index = index._format_multi(
                sparsify=self.sparsify,
                include_names=self.show_row_idx_names,
                formatter=fmt,
            )
        else:
            fmt_index = [
                index._format_flat(include_name=self.show_row_idx_names, formatter=fmt)
            ]

        fmt_index = [
            tuple(
                _make_fixed_width(
                    list(x), justify="left", minimum=col_space.get("", 0), adj=self.adj
                )
            )
            for x in fmt_index
        ]

        adjoined = self.adj.adjoin(1, *fmt_index).split("\n")

        # empty space for columns
        if self.show_col_idx_names:
            col_header = [str(x) for x in self._get_column_name_list()]
        else:
            col_header = [""] * columns.nlevels

        if self.header:
            return col_header + adjoined
        else:
            return adjoined

    def _get_column_name_list(self) -> list[Hashable]:
        names: list[Hashable] = []
        columns = self.frame.columns
        if isinstance(columns, MultiIndex):
            names.extend("" if name is None else name for name in columns.names)
        else:
            names.append("" if columns.name is None else columns.name)
        return names


class DataFrameRenderer:
    """Class for creating dataframe output in multiple formats.

    Called in pandas.core.generic.NDFrame:
        - to_csv
        - to_latex

    Called in pandas.core.frame.DataFrame:
        - to_html
        - to_string

    Parameters
    ----------
    fmt : DataFrameFormatter
        Formatter with the formatting options.
    """

    def __init__(self, fmt: DataFrameFormatter) -> None:
        self.fmt = fmt

    def to_html(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        classes: str | list | tuple | None = None,
        notebook: bool = False,
        border: int | bool | None = None,
        table_id: str | None = None,
        render_links: bool = False,
    ) -> str | None:
        """
        Render a DataFrame to a html table.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding : str, default “utf-8”
            Set character encoding.
        classes : str or list-like
            classes to include in the `class` attribute of the opening
            ``<table>`` tag, in addition to the default "dataframe".
        notebook : {True, False}, optional, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            ``<table>`` tag. Default ``pd.options.display.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.
        """
        from pandas.io.formats.html import (
            HTMLFormatter,
            NotebookFormatter,
        )

        Klass = NotebookFormatter if notebook else HTMLFormatter

        html_formatter = Klass(
            self.fmt,
            classes=classes,
            border=border,
            table_id=table_id,
            render_links=render_links,
        )
        string = html_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_string(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        line_width: int | None = None,
    ) -> str | None:
        """
        Render a DataFrame to a console-friendly tabular output.

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        encoding: str, default “utf-8”
            Set character encoding.
        line_width : int, optional
            Width to wrap a line in characters.
        """
        from pandas.io.formats.string import StringFormatter

        string_formatter = StringFormatter(self.fmt, line_width=line_width)
        string = string_formatter.to_string()
        return save_to_buffer(string, buf=buf, encoding=encoding)

    def to_csv(
        self,
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        encoding: str | None = None,
        sep: str = ",",
        columns: Sequence[Hashable] | None = None,
        index_label: IndexLabel | None = None,
        mode: str = "w",
        compression: CompressionOptions = "infer",
        quoting: int | None = None,
        quotechar: str = '"',
        lineterminator: str | None = None,
        chunksize: int | None = None,
        date_format: str | None = None,
        doublequote: bool = True,
        escapechar: str | None = None,
        errors: str = "strict",
        storage_options: StorageOptions | None = None,
    ) -> str | None:
        """
        Render dataframe as comma-separated file.
        """
        from pandas.io.formats.csvs import CSVFormatter

        if path_or_buf is None:
            created_buffer = True
            path_or_buf = StringIO()
        else:
            created_buffer = False

        csv_formatter = CSVFormatter(
            path_or_buf=path_or_buf,
            lineterminator=lineterminator,
            sep=sep,
            encoding=encoding,
            errors=errors,
            compression=compression,
            quoting=quoting,
            cols=columns,
            index_label=index_label,
            mode=mode,
            chunksize=chunksize,
            quotechar=quotechar,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            storage_options=storage_options,
            formatter=self.fmt,
        )
        csv_formatter.save()

        if created_buffer:
            assert isinstance(path_or_buf, StringIO)
            content = path_or_buf.getvalue()
            path_or_buf.close()
            return content

        return None


def save_to_buffer(
    string: str,
    buf: FilePath | WriteBuffer[str] | None = None,
    encoding: str | None = None,
) -> str | None:
    """
    Perform serialization. Write to buf or return as string if buf is None.
    """
    with _get_buffer(buf, encoding=encoding) as fd:
        fd.write(string)
        if buf is None:
            # error: "WriteBuffer[str]" has no attribute "getvalue"
            return fd.getvalue()  # type: ignore[attr-defined]
        return None


@contextmanager
def _get_buffer(
    buf: FilePath | WriteBuffer[str] | None, encoding: str | None = None
) -> Generator[WriteBuffer[str], None, None] | Generator[StringIO, None, None]:
    """
    Context manager to open, yield and close buffer for filenames or Path-like
    objects, otherwise yield buf unchanged.
    """
    if buf is not None:
        buf = stringify_path(buf)
    else:
        buf = StringIO()

    if encoding is None:
        encoding = "utf-8"
    elif not isinstance(buf, str):
        raise ValueError("buf is not a file name and encoding is specified.")

    if hasattr(buf, "write"):
        # Incompatible types in "yield" (actual type "Union[str, WriteBuffer[str],
        # StringIO]", expected type "Union[WriteBuffer[str], StringIO]")
        yield buf  # type: ignore[misc]
    elif isinstance(buf, str):
        check_parent_directory(str(buf))
        with open(buf, "w", encoding=encoding, newline="") as f:
            # GH#30034 open instead of codecs.open prevents a file leak
            #  if we have an invalid encoding argument.
            # newline="" is needed to roundtrip correctly on
            #  windows test_to_latex_filename
            yield f
    else:
        raise TypeError("buf is not a file name and it has no write method")


# ----------------------------------------------------------------------
# Array formatters


def format_array(
    values: ArrayLike,
    formatter: Callable | None,
    float_format: FloatFormatType | None = None,
    na_rep: str = "NaN",
    digits: int | None = None,
    space: str | int | None = None,
    justify: str = "right",
    decimal: str = ".",
    leading_space: bool | None = True,
    quoting: int | None = None,
    fallback_formatter: Callable | None = None,
) -> list[str]:
    """
    Format an array for printing.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray
    formatter
    float_format
    na_rep
    digits
    space
    justify
    decimal
    leading_space : bool, optional, default True
        Whether the array should be formatted with a leading space.
        When an array as a column of a Series or DataFrame, we do want
        the leading space to pad between columns.

        When formatting an Index subclass
        (e.g. IntervalIndex._get_values_for_csv), we don't want the
        leading space since it should be left-aligned.
    fallback_formatter

    Returns
    -------
    List[str]
    """
    fmt_klass: type[_GenericArrayFormatter]
    if lib.is_np_dtype(values.dtype, "M"):
        fmt_klass = _Datetime64Formatter
        values = cast(DatetimeArray, values)
    elif isinstance(values.dtype, DatetimeTZDtype):
        fmt_klass = _Datetime64TZFormatter
        values = cast(DatetimeArray, values)
    elif lib.is_np_dtype(values.dtype, "m"):
        fmt_klass = _Timedelta64Formatter
        values = cast(TimedeltaArray, values)
    elif isinstance(values.dtype, ExtensionDtype):
        fmt_klass = _ExtensionArrayFormatter
    elif lib.is_np_dtype(values.dtype, "fc"):
        fmt_klass = FloatArrayFormatter
    elif lib.is_np_dtype(values.dtype, "iu"):
        fmt_klass = _IntArrayFormatter
    else:
        fmt_klass = _GenericArrayFormatter

    if space is None:
        space = 12

    if float_format is None:
        float_format = get_option("display.float_format")

    if digits is None:
        digits = get_option("display.precision")

    fmt_obj = fmt_klass(
        values,
        digits=digits,
        na_rep=na_rep,
        float_format=float_format,
        formatter=formatter,
        space=space,
        justify=justify,
        decimal=decimal,
        leading_space=leading_space,
        quoting=quoting,
        fallback_formatter=fallback_formatter,
    )

    return fmt_obj.get_result()


class _GenericArrayFormatter:
    def __init__(
        self,
        values: ArrayLike,
        digits: int = 7,
        formatter: Callable | None = None,
        na_rep: str = "NaN",
        space: str | int = 12,
        float_format: FloatFormatType | None = None,
        justify: str = "right",
        decimal: str = ".",
        quoting: int | None = None,
        fixed_width: bool = True,
        leading_space: bool | None = True,
        fallback_formatter: Callable | None = None,
    ) -> None:
        self.values = values
        self.digits = digits
        self.na_rep = na_rep
        self.space = space
        self.formatter = formatter
        self.float_format = float_format
        self.justify = justify
        self.decimal = decimal
        self.quoting = quoting
        self.fixed_width = fixed_width
        self.leading_space = leading_space
        self.fallback_formatter = fallback_formatter

    def get_result(self) -> list[str]:
        fmt_values = self._format_strings()
        return _make_fixed_width(fmt_values, self.justify)

    def _format_strings(self) -> list[str]:
        if self.float_format is None:
            float_format = get_option("display.float_format")
            if float_format is None:
                precision = get_option("display.precision")
                float_format = lambda x: _trim_zeros_single_float(
                    f"{x: .{precision:d}f}"
                )
        else:
            float_format = self.float_format

        if self.formatter is not None:
            formatter = self.formatter
        elif self.fallback_formatter is not None:
            formatter = self.fallback_formatter
        else:
            quote_strings = self.quoting is not None and self.quoting != QUOTE_NONE
            formatter = partial(
                printing.pprint_thing,
                escape_chars=("\t", "\r", "\n"),
                quote_strings=quote_strings,
            )

        def _format(x):
            if self.na_rep is not None and is_scalar(x) and isna(x):
                if x is None:
                    return "None"
                elif x is NA:
                    return str(NA)
                elif lib.is_float(x) and np.isinf(x):
                    # TODO(3.0): this will be unreachable when use_inf_as_na
                    #  deprecation is enforced
                    return str(x)
                elif x is NaT or isinstance(x, (np.datetime64, np.timedelta64)):
                    return "NaT"
                return self.na_rep
            elif isinstance(x, PandasObject):
                return str(x)
            elif isinstance(x, StringDtype) and x.na_value is NA:
                return repr(x)
            else:
                # object dtype
                return str(formatter(x))

        vals = self.values
        if not isinstance(vals, np.ndarray):
            raise TypeError(
                "ExtensionArray formatting should use _ExtensionArrayFormatter"
            )
        inferred = lib.map_infer(vals, is_float)
        is_float_type = (
            inferred
            # vals may have 2 or more dimensions
            & np.all(notna(vals), axis=tuple(range(1, len(vals.shape))))
        )
        leading_space = self.leading_space
        if leading_space is None:
            leading_space = is_float_type.any()

        fmt_values = []
        for i, v in enumerate(vals):
            if (not is_float_type[i] or self.formatter is not None) and leading_space:
                fmt_values.append(f" {_format(v)}")
            elif is_float_type[i]:
                fmt_values.append(float_format(v))
            else:
                if leading_space is False:
                    # False specifically, so that the default is
                    # to include a space if we get here.
                    tpl = "{v}"
                else:
                    tpl = " {v}"
                fmt_values.append(tpl.format(v=_format(v)))

        return fmt_values


class FloatArrayFormatter(_GenericArrayFormatter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # float_format is expected to be a string
        # formatter should be used to pass a function
        if self.float_format is not None and self.formatter is None:
            # GH21625, GH22270
            self.fixed_width = False
            if callable(self.float_format):
                self.formatter = self.float_format
                self.float_format = None

    def _value_formatter(
        self,
        float_format: FloatFormatType | None = None,
        threshold: float | None = None,
    ) -> Callable:
        """Returns a function to be applied on each value to format it"""
        # the float_format parameter supersedes self.float_format
        if float_format is None:
            float_format = self.float_format

        # we are going to compose different functions, to first convert to
        # a string, then replace the decimal symbol, and finally chop according
        # to the threshold

        # when there is no float_format, we use str instead of '%g'
        # because str(0.0) = '0.0' while '%g' % 0.0 = '0'
        if float_format:

            def base_formatter(v):
                assert float_format is not None  # for mypy
                # error: "str" not callable
                # error: Unexpected keyword argument "value" for "__call__" of
                # "EngFormatter"
                return (
                    float_format(value=v)  # type: ignore[operator,call-arg]
                    if notna(v)
                    else self.na_rep
                )

        else:

            def base_formatter(v):
                return str(v) if notna(v) else self.na_rep

        if self.decimal != ".":

            def decimal_formatter(v):
                return base_formatter(v).replace(".", self.decimal, 1)

        else:
            decimal_formatter = base_formatter

        if threshold is None:
            return decimal_formatter

        def formatter(value):
            if notna(value):
                if abs(value) > threshold:
                    return decimal_formatter(value)
                else:
                    return decimal_formatter(0.0)
            else:
                return self.na_rep

        return formatter

    def get_result_as_array(self) -> np.ndarray:
        """
        Returns the float values converted into strings using
        the parameters given at initialisation, as a numpy array
        """

        def format_with_na_rep(values: ArrayLike, formatter: Callable, na_rep: str):
            mask = isna(values)
            formatted = np.array(
                [
                    formatter(val) if not m else na_rep
                    for val, m in zip(values.ravel(), mask.ravel())
                ]
            ).reshape(values.shape)
            return formatted

        def format_complex_with_na_rep(
            values: ArrayLike, formatter: Callable, na_rep: str
        ):
            real_values = np.real(values).ravel()  # type: ignore[arg-type]
            imag_values = np.imag(values).ravel()  # type: ignore[arg-type]
            real_mask, imag_mask = isna(real_values), isna(imag_values)
            formatted_lst = []
            for val, real_val, imag_val, re_isna, im_isna in zip(
                values.ravel(),
                real_values,
                imag_values,
                real_mask,
                imag_mask,
            ):
                if not re_isna and not im_isna:
                    formatted_lst.append(formatter(val))
                elif not re_isna:  # xxx+nanj
                    formatted_lst.append(f"{formatter(real_val)}+{na_rep}j")
                elif not im_isna:  # nan[+/-]xxxj
                    # The imaginary part may either start with a "-" or a space
                    imag_formatted = formatter(imag_val).strip()
                    if imag_formatted.startswith("-"):
                        formatted_lst.append(f"{na_rep}{imag_formatted}j")
                    else:
                        formatted_lst.append(f"{na_rep}+{imag_formatted}j")
                else:  # nan+nanj
                    formatted_lst.append(f"{na_rep}+{na_rep}j")
            return np.array(formatted_lst).reshape(values.shape)

        if self.formatter is not None:
            return format_with_na_rep(self.values, self.formatter, self.na_rep)

        if self.fixed_width:
            threshold = get_option("display.chop_threshold")
        else:
            threshold = None

        # if we have a fixed_width, we'll need to try different float_format
        def format_values_with(float_format):
            formatter = self._value_formatter(float_format, threshold)

            # default formatter leaves a space to the left when formatting
            # floats, must be consistent for left-justifying NaNs (GH #25061)
            na_rep = " " + self.na_rep if self.justify == "left" else self.na_rep

            # different formatting strategies for complex and non-complex data
            # need to distinguish complex and float NaNs (GH #53762)
            values = self.values
            is_complex = is_complex_dtype(values)

            # separate the wheat from the chaff
            if is_complex:
                values = format_complex_with_na_rep(values, formatter, na_rep)
            else:
                values = format_with_na_rep(values, formatter, na_rep)

            if self.fixed_width:
                if is_complex:
                    result = _trim_zeros_complex(values, self.decimal)
                else:
                    result = _trim_zeros_float(values, self.decimal)
                return np.asarray(result, dtype="object")

            return values

        # There is a special default string when we are fixed-width
        # The default is otherwise to use str instead of a formatting string
        float_format: FloatFormatType | None
        if self.float_format is None:
            if self.fixed_width:
                if self.leading_space is True:
                    fmt_str = "{value: .{digits:d}f}"
                else:
                    fmt_str = "{value:.{digits:d}f}"
                float_format = partial(fmt_str.format, digits=self.digits)
            else:
                float_format = self.float_format
        else:
            float_format = lambda value: self.float_format % value

        formatted_values = format_values_with(float_format)

        if not self.fixed_width:
            return formatted_values

        # we need do convert to engineering format if some values are too small
        # and would appear as 0, or if some values are too big and take too
        # much space

        if len(formatted_values) > 0:
            maxlen = max(len(x) for x in formatted_values)
            too_long = maxlen > self.digits + 6
        else:
            too_long = False

        abs_vals = np.abs(self.values)
        # this is pretty arbitrary for now
        # large values: more that 8 characters including decimal symbol
        # and first digit, hence > 1e6
        has_large_values = (abs_vals > 1e6).any()
        has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()

        if has_small_values or (too_long and has_large_values):
            if self.leading_space is True:
                fmt_str = "{value: .{digits:d}e}"
            else:
                fmt_str = "{value:.{digits:d}e}"
            float_format = partial(fmt_str.format, digits=self.digits)
            formatted_values = format_values_with(float_format)

        return formatted_values

    def _format_strings(self) -> list[str]:
        return list(self.get_result_as_array())


class _IntArrayFormatter(_GenericArrayFormatter):
    def _format_strings(self) -> list[str]:
        if self.leading_space is False:
            formatter_str = lambda x: f"{x:d}".format(x=x)
        else:
            formatter_str = lambda x: f"{x: d}".format(x=x)
        formatter = self.formatter or formatter_str
        fmt_values = [formatter(x) for x in self.values]
        return fmt_values


class _Datetime64Formatter(_GenericArrayFormatter):
    values: DatetimeArray

    def __init__(
        self,
        values: DatetimeArray,
        nat_rep: str = "NaT",
        date_format: None = None,
        **kwargs,
    ) -> None:
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep
        self.date_format = date_format

    def _format_strings(self) -> list[str]:
        """we by definition have DO NOT have a TZ"""
        values = self.values

        if self.formatter is not None:
            return [self.formatter(x) for x in values]

        fmt_values = values._format_native_types(
            na_rep=self.nat_rep, date_format=self.date_format
        )
        return fmt_values.tolist()


class _ExtensionArrayFormatter(_GenericArrayFormatter):
    values: ExtensionArray

    def _format_strings(self) -> list[str]:
        values = self.values

        formatter = self.formatter
        fallback_formatter = None
        if formatter is None:
            fallback_formatter = values._formatter(boxed=True)

        if isinstance(values, Categorical):
            # Categorical is special for now, so that we can preserve tzinfo
            array = values._internal_get_values()
        else:
            array = np.asarray(values, dtype=object)

        fmt_values = format_array(
            array,
            formatter,
            float_format=self.float_format,
            na_rep=self.na_rep,
            digits=self.digits,
            space=self.space,
            justify=self.justify,
            decimal=self.decimal,
            leading_space=self.leading_space,
            quoting=self.quoting,
            fallback_formatter=fallback_formatter,
        )
        return fmt_values


def format_percentiles(
    percentiles: (np.ndarray | Sequence[float]),
) -> list[str]:
    """
    Outputs rounded and formatted percentiles.

    Parameters
    ----------
    percentiles : list-like, containing floats from interval [0,1]

    Returns
    -------
    formatted : list of strings

    Notes
    -----
    Rounding precision is chosen so that: (1) if any two elements of
    ``percentiles`` differ, they remain different after rounding
    (2) no entry is *rounded* to 0% or 100%.
    Any non-integer is always rounded to at least 1 decimal place.

    Examples
    --------
    Keeps all entries different after rounding:

    >>> format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
    ['1.999%', '2.001%', '50%', '66.667%', '99.99%']

    No element is rounded to 0% or 100% (unless already equal to it).
    Duplicates are allowed:

    >>> format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
    ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
    """
    percentiles = np.asarray(percentiles)

    # It checks for np.nan as well
    if (
        not is_numeric_dtype(percentiles)
        or not np.all(percentiles >= 0)
        or not np.all(percentiles <= 1)
    ):
        raise ValueError("percentiles should all be in the interval [0,1]")

    percentiles = 100 * percentiles
    prec = get_precision(percentiles)
    percentiles_round_type = percentiles.round(prec).astype(int)

    int_idx = np.isclose(percentiles_round_type, percentiles)

    if np.all(int_idx):
        out = percentiles_round_type.astype(str)
        return [i + "%" for i in out]

    unique_pcts = np.unique(percentiles)
    prec = get_precision(unique_pcts)
    out = np.empty_like(percentiles, dtype=object)
    out[int_idx] = percentiles[int_idx].round().astype(int).astype(str)

    out[~int_idx] = percentiles[~int_idx].round(prec).astype(str)
    return [i + "%" for i in out]


def get_precision(array: np.ndarray | Sequence[float]) -> int:
    to_begin = array[0] if array[0] > 0 else None
    to_end = 100 - array[-1] if array[-1] < 100 else None
    diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
    diff = abs(diff)
    prec = -np.floor(np.log10(np.min(diff))).astype(int)
    prec = max(1, prec)
    return prec


def _format_datetime64(x: NaTType | Timestamp, nat_rep: str = "NaT") -> str:
    if x is NaT:
        return nat_rep

    # Timestamp.__str__ falls back to datetime.datetime.__str__ = isoformat(sep=' ')
    # so it already uses string formatting rather than strftime (faster).
    return str(x)


def _format_datetime64_dateonly(
    x: NaTType | Timestamp,
    nat_rep: str = "NaT",
    date_format: str | None = None,
) -> str:
    if isinstance(x, NaTType):
        return nat_rep

    if date_format:
        return x.strftime(date_format)
    else:
        # Timestamp._date_repr relies on string formatting (faster than strftime)
        return x._date_repr


def get_format_datetime64(
    is_dates_only: bool, nat_rep: str = "NaT", date_format: str | None = None
) -> Callable:
    """Return a formatter callable taking a datetime64 as input and providing
    a string as output"""

    if is_dates_only:
        return lambda x: _format_datetime64_dateonly(
            x, nat_rep=nat_rep, date_format=date_format
        )
    else:
        return lambda x: _format_datetime64(x, nat_rep=nat_rep)


class _Datetime64TZFormatter(_Datetime64Formatter):
    values: DatetimeArray

    def _format_strings(self) -> list[str]:
        """we by definition have a TZ"""
        ido = self.values._is_dates_only
        values = self.values.astype(object)
        formatter = self.formatter or get_format_datetime64(
            ido, date_format=self.date_format
        )
        fmt_values = [formatter(x) for x in values]

        return fmt_values


class _Timedelta64Formatter(_GenericArrayFormatter):
    values: TimedeltaArray

    def __init__(
        self,
        values: TimedeltaArray,
        nat_rep: str = "NaT",
        **kwargs,
    ) -> None:
        # TODO: nat_rep is never passed, na_rep is.
        super().__init__(values, **kwargs)
        self.nat_rep = nat_rep

    def _format_strings(self) -> list[str]:
        formatter = self.formatter or get_format_timedelta64(
            self.values, nat_rep=self.nat_rep, box=False
        )
        return [formatter(x) for x in self.values]


def get_format_timedelta64(
    values: TimedeltaArray,
    nat_rep: str | float = "NaT",
    box: bool = False,
) -> Callable:
    """
    Return a formatter function for a range of timedeltas.
    These will all have the same format argument

    If box, then show the return in quotes
    """
    even_days = values._is_dates_only

    if even_days:
        format = None
    else:
        format = "long"

    def _formatter(x):
        if x is None or (is_scalar(x) and isna(x)):
            return nat_rep

        if not isinstance(x, Timedelta):
            x = Timedelta(x)

        # Timedelta._repr_base uses string formatting (faster than strftime)
        result = x._repr_base(format=format)
        if box:
            result = f"'{result}'"
        return result

    return _formatter


def _make_fixed_width(
    strings: list[str],
    justify: str = "right",
    minimum: int | None = None,
    adj: printing._TextAdjustment | None = None,
) -> list[str]:
    if len(strings) == 0 or justify == "all":
        return strings

    if adj is None:
        adjustment = printing.get_adjustment()
    else:
        adjustment = adj

    max_len = max(adjustment.len(x) for x in strings)

    if minimum is not None:
        max_len = max(minimum, max_len)

    conf_max = get_option("display.max_colwidth")
    if conf_max is not None and max_len > conf_max:
        max_len = conf_max

    def just(x: str) -> str:
        if conf_max is not None:
            if (conf_max > 3) & (adjustment.len(x) > max_len):
                x = x[: max_len - 3] + "..."
        return x

    strings = [just(x) for x in strings]
    result = adjustment.justify(strings, max_len, mode=justify)
    return result


def _trim_zeros_complex(str_complexes: ArrayLike, decimal: str = ".") -> list[str]:
    """
    Separates the real and imaginary parts from the complex number, and
    executes the _trim_zeros_float method on each of those.
    """
    real_part, imag_part = [], []
    for x in str_complexes:
        # Complex numbers are represented as "(-)xxx(+/-)xxxj"
        # The split will give [{"", "-"}, "xxx", "+/-", "xxx", "j", ""]
        # Therefore, the imaginary part is the 4th and 3rd last elements,
        # and the real part is everything before the imaginary part
        trimmed = re.split(r"([j+-])", x)
        real_part.append("".join(trimmed[:-4]))
        imag_part.append("".join(trimmed[-4:-2]))

    # We want to align the lengths of the real and imaginary parts of each complex
    # number, as well as the lengths the real (resp. complex) parts of all numbers
    # in the array
    n = len(str_complexes)
    padded_parts = _trim_zeros_float(real_part + imag_part, decimal)
    if len(padded_parts) == 0:
        return []
    padded_length = max(len(part) for part in padded_parts) - 1
    padded = [
        real_pt  # real part, possibly NaN
        + imag_pt[0]  # +/-
        + f"{imag_pt[1:]:>{padded_length}}"  # complex part (no sign), possibly nan
        + "j"
        for real_pt, imag_pt in zip(padded_parts[:n], padded_parts[n:])
    ]
    return padded


def _trim_zeros_single_float(str_float: str) -> str:
    """
    Trims trailing zeros after a decimal point,
    leaving just one if necessary.
    """
    str_float = str_float.rstrip("0")
    if str_float.endswith("."):
        str_float += "0"

    return str_float


def _trim_zeros_float(
    str_floats: ArrayLike | list[str], decimal: str = "."
) -> list[str]:
    """
    Trims the maximum number of trailing zeros equally from
    all numbers containing decimals, leaving just one if
    necessary.
    """
    trimmed = str_floats
    number_regex = re.compile(rf"^\s*[\+-]?[0-9]+\{decimal}[0-9]*$")

    def is_number_with_decimal(x) -> bool:
        return re.match(number_regex, x) is not None

    def should_trim(values: ArrayLike | list[str]) -> bool:
        """
        Determine if an array of strings should be trimmed.

        Returns True if all numbers containing decimals (defined by the
        above regular expression) within the array end in a zero, otherwise
        returns False.
        """
        numbers = [x for x in values if is_number_with_decimal(x)]
        return len(numbers) > 0 and all(x.endswith("0") for x in numbers)

    while should_trim(trimmed):
        trimmed = [x[:-1] if is_number_with_decimal(x) else x for x in trimmed]

    # leave one 0 after the decimal points if need be.
    result = [
        x + "0" if is_number_with_decimal(x) and x.endswith(decimal) else x
        for x in trimmed
    ]
    return result


def _has_names(index: Index) -> bool:
    if isinstance(index, MultiIndex):
        return com.any_not_none(*index.names)
    else:
        return index.name is not None


class EngFormatter:
    """
    Formats float values according to engineering format.

    Based on matplotlib.ticker.EngFormatter
    """

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "u",
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
    }

    def __init__(
        self, accuracy: int | None = None, use_eng_prefix: bool = False
    ) -> None:
        self.accuracy = accuracy
        self.use_eng_prefix = use_eng_prefix

    def __call__(self, num: float) -> str:
        """
        Formats a number in engineering notation, appending a letter
        representing the power of 1000 of the original number. Some examples:
        >>> format_eng = EngFormatter(accuracy=0, use_eng_prefix=True)
        >>> format_eng(0)
        ' 0'
        >>> format_eng = EngFormatter(accuracy=1, use_eng_prefix=True)
        >>> format_eng(1_000_000)
        ' 1.0M'
        >>> format_eng = EngFormatter(accuracy=2, use_eng_prefix=False)
        >>> format_eng("-1e-6")
        '-1.00E-06'

        @param num: the value to represent
        @type num: either a numeric value or a string that can be converted to
                   a numeric value (as per decimal.Decimal constructor)

        @return: engineering formatted string
        """
        dnum = Decimal(str(num))

        if Decimal.is_nan(dnum):
            return "NaN"

        if Decimal.is_infinite(dnum):
            return "inf"

        sign = 1

        if dnum < 0:  # pragma: no cover
            sign = -1
            dnum = -dnum

        if dnum != 0:
            pow10 = Decimal(int(math.floor(dnum.log10() / 3) * 3))
        else:
            pow10 = Decimal(0)

        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
        int_pow10 = int(pow10)

        if self.use_eng_prefix:
            prefix = self.ENG_PREFIXES[int_pow10]
        elif int_pow10 < 0:
            prefix = f"E-{-int_pow10:02d}"
        else:
            prefix = f"E+{int_pow10:02d}"

        mant = sign * dnum / (10**pow10)

        if self.accuracy is None:  # pragma: no cover
            format_str = "{mant: g}{prefix}"
        else:
            format_str = f"{{mant: .{self.accuracy:d}f}}{{prefix}}"

        formatted = format_str.format(mant=mant, prefix=prefix)

        return formatted


def set_eng_float_format(accuracy: int = 3, use_eng_prefix: bool = False) -> None:
    """
    Format float representation in DataFrame with SI notation.

    Parameters
    ----------
    accuracy : int, default 3
        Number of decimal digits after the floating point.
    use_eng_prefix : bool, default False
        Whether to represent a value with SI prefixes.

    Returns
    -------
    None

    Examples
    --------
    >>> df = pd.DataFrame([1e-9, 1e-3, 1, 1e3, 1e6])
    >>> df
                  0
    0  1.000000e-09
    1  1.000000e-03
    2  1.000000e+00
    3  1.000000e+03
    4  1.000000e+06

    >>> pd.set_eng_float_format(accuracy=1)
    >>> df
             0
    0  1.0E-09
    1  1.0E-03
    2  1.0E+00
    3  1.0E+03
    4  1.0E+06

    >>> pd.set_eng_float_format(use_eng_prefix=True)
    >>> df
            0
    0  1.000n
    1  1.000m
    2   1.000
    3  1.000k
    4  1.000M

    >>> pd.set_eng_float_format(accuracy=1, use_eng_prefix=True)
    >>> df
          0
    0  1.0n
    1  1.0m
    2   1.0
    3  1.0k
    4  1.0M

    >>> pd.set_option("display.float_format", None)  # unset option
    """
    set_option("display.float_format", EngFormatter(accuracy, use_eng_prefix))


def get_level_lengths(
    levels: Any, sentinel: bool | object | str = ""
) -> list[dict[int, int]]:
    """
    For each index in each level the function returns lengths of indexes.

    Parameters
    ----------
    levels : list of lists
        List of values on for level.
    sentinel : string, optional
        Value which states that no new index starts on there.

    Returns
    -------
    Returns list of maps. For each level returns map of indexes (key is index
    in row and value is length of index).
    """
    if len(levels) == 0:
        return []

    control = [True] * len(levels[0])

    result = []
    for level in levels:
        last_index = 0

        lengths = {}
        for i, key in enumerate(level):
            if control[i] and key == sentinel:
                pass
            else:
                control[i] = False
                lengths[last_index] = i - last_index
                last_index = i

        lengths[last_index] = len(level) - last_index

        result.append(lengths)

    return result


def buffer_put_lines(buf: WriteBuffer[str], lines: list[str]) -> None:
    """
    Appends lines to a buffer.

    Parameters
    ----------
    buf
        The buffer to write to
    lines
        The lines to append.
    """
    if any(isinstance(x, str) for x in lines):
        lines = [str(x) for x in lines]
    buf.write("\n".join(lines))
<|endoftext|>"""
Printing tools.
"""
from __future__ import annotations

from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
import sys
from typing import (
    Any,
    Callable,
    TypeVar,
    Union,
)
from unicodedata import east_asian_width

from pandas._config import get_option

from pandas.core.dtypes.inference import is_sequence

from pandas.io.formats.console import get_console_size

EscapeChars = Union[Mapping[str, str], Iterable[str]]
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def adjoin(space: int, *lists: list[str], **kwargs) -> str:
    """
    Glues together two sets of strings using the amount of space requested.
    The idea is to prettify.

    ----------
    space : int
        number of spaces for padding
    lists : str
        list of str which being joined
    strlen : callable
        function used to calculate the length of each str. Needed for unicode
        handling.
    justfunc : callable
        function used to justify str. Needed for unicode handling.
    """
    strlen = kwargs.pop("strlen", len)
    justfunc = kwargs.pop("justfunc", _adj_justify)

    newLists = []
    lengths = [max(map(strlen, x)) + space for x in lists[:-1]]
    # not the last one
    lengths.append(max(map(len, lists[-1])))
    maxLen = max(map(len, lists))
    for i, lst in enumerate(lists):
        nl = justfunc(lst, lengths[i], mode="left")
        nl = ([" " * lengths[i]] * (maxLen - len(lst))) + nl
        newLists.append(nl)
    toJoin = zip(*newLists)
    return "\n".join("".join(lines) for lines in toJoin)


def _adj_justify(texts: Iterable[str], max_len: int, mode: str = "right") -> list[str]:
    """
    Perform ljust, center, rjust against string or list-like
    """
    if mode == "left":
        return [x.ljust(max_len) for x in texts]
    elif mode == "center":
        return [x.center(max_len) for x in texts]
    else:
        return [x.rjust(max_len) for x in texts]


# Unicode consolidation
# ---------------------
#
# pprinting utility functions for generating Unicode text or
# bytes(3.x)/str(2.x) representations of objects.
# Try to use these as much as possible rather than rolling your own.
#
# When to use
# -----------
#
# 1) If you're writing code internal to pandas (no I/O directly involved),
#    use pprint_thing().
#
#    It will always return unicode text which can handled by other
#    parts of the package without breakage.
#
# 2) if you need to write something out to file, use
#    pprint_thing_encoded(encoding).
#
#    If no encoding is specified, it defaults to utf-8. Since encoding pure
#    ascii with utf-8 is a no-op you can safely use the default utf-8 if you're
#    working with straight ascii.


def _pprint_seq(
    seq: Sequence, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds
) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.

    bounds length of printed sequence, depending on options
    """
    if isinstance(seq, set):
        fmt = "{{{body}}}"
    else:
        fmt = "[{body}]" if hasattr(seq, "__setitem__") else "({body})"

    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or get_option("max_seq_items") or len(seq)

    s = iter(seq)
    # handle sets, no slicing
    r = [
        pprint_thing(next(s), _nest_lvl + 1, max_seq_items=max_seq_items, **kwds)
        for i in range(min(nitems, len(seq)))
    ]
    body = ", ".join(r)

    if nitems < len(seq):
        body += ", ..."
    elif isinstance(seq, tuple) and len(seq) == 1:
        body += ","

    return fmt.format(body=body)


def _pprint_dict(
    seq: Mapping, _nest_lvl: int = 0, max_seq_items: int | None = None, **kwds
) -> str:
    """
    internal. pprinter for iterables. you should probably use pprint_thing()
    rather than calling this directly.
    """
    fmt = "{{{things}}}"
    pairs = []

    pfmt = "{key}: {val}"

    if max_seq_items is False:
        nitems = len(seq)
    else:
        nitems = max_seq_items or get_option("max_seq_items") or len(seq)

    for k, v in list(seq.items())[:nitems]:
        pairs.append(
            pfmt.format(
                key=pprint_thing(k, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
                val=pprint_thing(v, _nest_lvl + 1, max_seq_items=max_seq_items, **kwds),
            )
        )

    if nitems < len(seq):
        return fmt.format(things=", ".join(pairs) + ", ...")
    else:
        return fmt.format(things=", ".join(pairs))


def pprint_thing(
    thing: Any,
    _nest_lvl: int = 0,
    escape_chars: EscapeChars | None = None,
    default_escapes: bool = False,
    quote_strings: bool = False,
    max_seq_items: int | None = None,
) -> str:
    """
    This function is the sanctioned way of converting objects
    to a string representation and properly handles nested sequences.

    Parameters
    ----------
    thing : anything to be formatted
    _nest_lvl : internal use only. pprint_thing() is mutually-recursive
        with pprint_sequence, this argument is used to keep track of the
        current nesting level, and limit it.
    escape_chars : list or dict, optional
        Characters to escape. If a dict is passed the values are the
        replacements
    default_escapes : bool, default False
        Whether the input escape characters replaces or adds to the defaults
    max_seq_items : int or None, default None
        Pass through to other pretty printers to limit sequence printing

    Returns
    -------
    str
    """

    def as_escaped_string(
        thing: Any, escape_chars: EscapeChars | None = escape_chars
    ) -> str:
        translate = {"\t": r"\t", "\n": r"\n", "\r": r"\r"}
        if isinstance(escape_chars, dict):
            if default_escapes:
                translate.update(escape_chars)
            else:
                translate = escape_chars
            escape_chars = list(escape_chars.keys())
        else:
            escape_chars = escape_chars or ()

        result = str(thing)
        for c in escape_chars:
            result = result.replace(c, translate[c])
        return result

    if hasattr(thing, "__next__"):
        return str(thing)
    elif isinstance(thing, dict) and _nest_lvl < get_option(
        "display.pprint_nest_depth"
    ):
        result = _pprint_dict(
            thing, _nest_lvl, quote_strings=True, max_seq_items=max_seq_items
        )
    elif is_sequence(thing) and _nest_lvl < get_option("display.pprint_nest_depth"):
        result = _pprint_seq(
            thing,
            _nest_lvl,
            escape_chars=escape_chars,
            quote_strings=quote_strings,
            max_seq_items=max_seq_items,
        )
    elif isinstance(thing, str) and quote_strings:
        result = f"'{as_escaped_string(thing)}'"
    else:
        result = as_escaped_string(thing)

    return result


def pprint_thing_encoded(
    object, encoding: str = "utf-8", errors: str = "replace"
) -> bytes:
    value = pprint_thing(object)  # get unicode representation of object
    return value.encode(encoding, errors)


def enable_data_resource_formatter(enable: bool) -> None:
    if "IPython" not in sys.modules:
        # definitely not in IPython
        return
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        # still not in IPython
        return

    formatters = ip.display_formatter.formatters
    mimetype = "application/vnd.dataresource+json"

    if enable:
        if mimetype not in formatters:
            # define tableschema formatter
            from IPython.core.formatters import BaseFormatter
            from traitlets import ObjectName

            class TableSchemaFormatter(BaseFormatter):
                print_method = ObjectName("_repr_data_resource_")
                _return_type = (dict,)

            # register it:
            formatters[mimetype] = TableSchemaFormatter()
        # enable it if it's been disabled:
        formatters[mimetype].enabled = True
    # unregister tableschema mime-type
    elif mimetype in formatters:
        formatters[mimetype].enabled = False


def default_pprint(thing: Any, max_seq_items: int | None = None) -> str:
    return pprint_thing(
        thing,
        escape_chars=("\t", "\r", "\n"),
        quote_strings=True,
        max_seq_items=max_seq_items,
    )


def format_object_summary(
    obj,
    formatter: Callable,
    is_justify: bool = True,
    name: str | None = None,
    indent_for_name: bool = True,
    line_break_each_value: bool = False,
) -> str:
    """
    Return the formatted obj as a unicode string

    Parameters
    ----------
    obj : object
        must be iterable and support __getitem__
    formatter : callable
        string formatter for an element
    is_justify : bool
        should justify the display
    name : name, optional
        defaults to the class name of the obj
    indent_for_name : bool, default True
        Whether subsequent lines should be indented to
        align with the name.
    line_break_each_value : bool, default False
        If True, inserts a line break for each value of ``obj``.
        If False, only break lines when the a line of values gets wider
        than the display width.

    Returns
    -------
    summary string
    """
    display_width, _ = get_console_size()
    if display_width is None:
        display_width = get_option("display.width") or 80
    if name is None:
        name = type(obj).__name__

    if indent_for_name:
        name_len = len(name)
        space1 = f'\n{(" " * (name_len + 1))}'
        space2 = f'\n{(" " * (name_len + 2))}'
    else:
        space1 = "\n"
        space2 = "\n "  # space for the opening '['

    n = len(obj)
    if line_break_each_value:
        # If we want to vertically align on each value of obj, we need to
        # separate values by a line break and indent the values
        sep = ",\n " + " " * len(name)
    else:
        sep = ","
    max_seq_items = get_option("display.max_seq_items") or n

    # are we a truncated display
    is_truncated = n > max_seq_items

    # adj can optionally handle unicode eastern asian width
    adj = get_adjustment()

    def _extend_line(
        s: str, line: str, value: str, display_width: int, next_line_prefix: str
    ) -> tuple[str, str]:
        if adj.len(line.rstrip()) + adj.len(value.rstrip()) >= display_width:
            s += line.rstrip()
            line = next_line_prefix
        line += value
        return s, line

    def best_len(values: list[str]) -> int:
        if values:
            return max(adj.len(x) for x in values)
        else:
            return 0

    close = ", "

    if n == 0:
        summary = f"[]{close}"
    elif n == 1 and not line_break_each_value:
        first = formatter(obj[0])
        summary = f"[{first}]{close}"
    elif n == 2 and not line_break_each_value:
        first = formatter(obj[0])
        last = formatter(obj[-1])
        summary = f"[{first}, {last}]{close}"
    else:
        if max_seq_items == 1:
            # If max_seq_items=1 show only last element
            head = []
            tail = [formatter(x) for x in obj[-1:]]
        elif n > max_seq_items:
            n = min(max_seq_items // 2, 10)
            head = [formatter(x) for x in obj[:n]]
            tail = [formatter(x) for x in obj[-n:]]
        else:
            head = []
            tail = [formatter(x) for x in obj]

        # adjust all values to max length if needed
        if is_justify:
            if line_break_each_value:
                # Justify each string in the values of head and tail, so the
                # strings will right align when head and tail are stacked
                # vertically.
                head, tail = _justify(head, tail)
            elif is_truncated or not (
                len(", ".join(head)) < display_width
                and len(", ".join(tail)) < display_width
            ):
                # Each string in head and tail should align with each other
                max_length = max(best_len(head), best_len(tail))
                head = [x.rjust(max_length) for x in head]
                tail = [x.rjust(max_length) for x in tail]
            # If we are not truncated and we are only a single
            # line, then don't justify

        if line_break_each_value:
            # Now head and tail are of type List[Tuple[str]]. Below we
            # convert them into List[str], so there will be one string per
            # value. Also truncate items horizontally if wider than
            # max_space
            max_space = display_width - len(space2)
            value = tail[0]
            max_items = 1
            for num_items in reversed(range(1, len(value) + 1)):
                pprinted_seq = _pprint_seq(value, max_seq_items=num_items)
                if len(pprinted_seq) < max_space:
                    max_items = num_items
                    break
            head = [_pprint_seq(x, max_seq_items=max_items) for x in head]
            tail = [_pprint_seq(x, max_seq_items=max_items) for x in tail]

        summary = ""
        line = space2

        for head_value in head:
            word = head_value + sep + " "
            summary, line = _extend_line(summary, line, word, display_width, space2)

        if is_truncated:
            # remove trailing space of last line
            summary += line.rstrip() + space2 + "..."
            line = space2

        for tail_item in tail[:-1]:
            word = tail_item + sep + " "
            summary, line = _extend_line(summary, line, word, display_width, space2)

        # last value: no sep added + 1 space of width used for trailing ','
        summary, line = _extend_line(summary, line, tail[-1], display_width - 2, space2)
        summary += line

        # right now close is either '' or ', '
        # Now we want to include the ']', but not the maybe space.
        close = "]" + close.rstrip(" ")
        summary += close

        if len(summary) > (display_width) or line_break_each_value:
            summary += space1
        else:  # one row
            summary += " "

        # remove initial space
        summary = "[" + summary[len(space2) :]

    return summary


def _justify(
    head: list[Sequence[str]], tail: list[Sequence[str]]
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """
    Justify items in head and tail, so they are right-aligned when stacked.

    Parameters
    ----------
    head : list-like of list-likes of strings
    tail : list-like of list-likes of strings

    Returns
    -------
    tuple of list of tuples of strings
        Same as head and tail, but items are right aligned when stacked
        vertically.

    Examples
    --------
    >>> _justify([['a', 'b']], [['abc', 'abcd']])
    ([('  a', '   b')], [('abc', 'abcd')])
    """
    combined = head + tail

    # For each position for the sequences in ``combined``,
    # find the length of the largest string.
    max_length = [0] * len(combined[0])
    for inner_seq in combined:
        length = [len(item) for item in inner_seq]
        max_length = [max(x, y) for x, y in zip(max_length, length)]

    # justify each item in each list-like in head and tail using max_length
    head_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in head
    ]
    tail_tuples = [
        tuple(x.rjust(max_len) for x, max_len in zip(seq, max_length)) for seq in tail
    ]
    return head_tuples, tail_tuples


class PrettyDict(dict[_KT, _VT]):
    """Dict extension to support abbreviated __repr__"""

    def __repr__(self) -> str:
        return pprint_thing(self)


class _TextAdjustment:
    def __init__(self) -> None:
        self.encoding = get_option("display.encoding")

    def len(self, text: str) -> int:
        return len(text)

    def justify(self, texts: Any, max_len: int, mode: str = "right") -> list[str]:
        """
        Perform ljust, center, rjust against string or list-like
        """
        if mode == "left":
            return [x.ljust(max_len) for x in texts]
        elif mode == "center":
            return [x.center(max_len) for x in texts]
        else:
            return [x.rjust(max_len) for x in texts]

    def adjoin(self, space: int, *lists, **kwargs) -> str:
        return adjoin(space, *lists, strlen=self.len, justfunc=self.justify, **kwargs)


class _EastAsianTextAdjustment(_TextAdjustment):
    def __init__(self) -> None:
        super().__init__()
        if get_option("display.unicode.ambiguous_as_wide"):
            self.ambiguous_width = 2
        else:
            self.ambiguous_width = 1

        # Definition of East Asian Width
        # https://unicode.org/reports/tr11/
        # Ambiguous width can be changed by option
        self._EAW_MAP = {"Na": 1, "N": 1, "W": 2, "F": 2, "H": 1}

    def len(self, text: str) -> int:
        """
        Calculate display width considering unicode East Asian Width
        """
        if not isinstance(text, str):
            return len(text)

        return sum(
            self._EAW_MAP.get(east_asian_width(c), self.ambiguous_width) for c in text
        )

    def justify(
        self, texts: Iterable[str], max_len: int, mode: str = "right"
    ) -> list[str]:
        # re-calculate padding space per str considering East Asian Width
        def _get_pad(t):
            return max_len - self.len(t) + len(t)

        if mode == "left":
            return [x.ljust(_get_pad(x)) for x in texts]
        elif mode == "center":
            return [x.center(_get_pad(x)) for x in texts]
        else:
            return [x.rjust(_get_pad(x)) for x in texts]


def get_adjustment() -> _TextAdjustment:
    use_east_asian_width = get_option("display.unicode.east_asian_width")
    if use_east_asian_width:
        return _EastAsianTextAdjustment()
    else:
        return _TextAdjustment()
<|endoftext|>"""
Module for applying conditional formatting to DataFrames and Series.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from functools import partial
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    overload,
)
import warnings

import numpy as np

from pandas._config import get_option

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import (
    Substitution,
    doc,
)
from pandas.util._exceptions import find_stack_level

import pandas as pd
from pandas import (
    IndexSlice,
    RangeIndex,
)
import pandas.core.common as com
from pandas.core.frame import (
    DataFrame,
    Series,
)
from pandas.core.generic import NDFrame
from pandas.core.shared_docs import _shared_docs

from pandas.io.formats.format import save_to_buffer

jinja2 = import_optional_dependency("jinja2", extra="DataFrame.style requires jinja2.")

from pandas.io.formats.style_render import (
    CSSProperties,
    CSSStyles,
    ExtFormatter,
    StylerRenderer,
    Subset,
    Tooltips,
    format_table_styles,
    maybe_convert_css_to_tuples,
    non_reducing_slice,
    refactor_levels,
)

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Hashable,
        Sequence,
    )

    from matplotlib.colors import Colormap

    from pandas._typing import (
        Axis,
        AxisInt,
        FilePath,
        IndexLabel,
        IntervalClosedType,
        Level,
        QuantileInterpolation,
        Scalar,
        StorageOptions,
        WriteBuffer,
        WriteExcelBuffer,
    )

    from pandas import ExcelWriter

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    has_mpl = True
except ImportError:
    has_mpl = False


@contextmanager
def _mpl(func: Callable) -> Generator[tuple[Any, Any], None, None]:
    if has_mpl:
        yield plt, mpl
    else:
        raise ImportError(f"{func.__name__} requires matplotlib.")


####
# Shared Doc Strings

subset_args = """subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function."""

properties_args = """props : str, default None
           CSS properties to use for highlighting. If ``props`` is given, ``color``
           is not used."""

coloring_args = """color : str, default '{default}'
           Background color to use for highlighting."""

buffering_args = """buf : str, path object, file-like object, optional
         String, path object (implementing ``os.PathLike[str]``), or file-like
         object implementing a string ``write()`` function. If ``None``, the result is
         returned as a string."""

encoding_args = """encoding : str, optional
              Character encoding setting for file output (and meta tags if available).
              Defaults to ``pandas.options.styler.render.encoding`` value of "utf-8"."""

#
###


class Styler(StylerRenderer):
    r"""
    Helps style a DataFrame or Series according to the data with HTML and CSS.

    Parameters
    ----------
    data : Series or DataFrame
        Data to be styled - either a Series or DataFrame.
    precision : int, optional
        Precision to round floats to. If not given defaults to
        ``pandas.options.styler.format.precision``.

        .. versionchanged:: 1.4.0
    table_styles : list-like, default None
        List of {selector: (attr, value)} dicts; see Notes.
    uuid : str, default None
        A unique identifier to avoid CSS collisions; generated automatically.
    caption : str, tuple, default None
        String caption to attach to the table. Tuple only used for LaTeX dual captions.
    table_attributes : str, default None
        Items that show up in the opening ``<table>`` tag
        in addition to automatic (by default) id.
    cell_ids : bool, default True
        If True, each cell will have an ``id`` attribute in their HTML tag.
        The ``id`` takes the form ``T_<uuid>_row<num_row>_col<num_col>``
        where ``<uuid>`` is the unique identifier, ``<num_row>`` is the row
        number and ``<num_col>`` is the column number.
    na_rep : str, optional
        Representation for missing values.
        If ``na_rep`` is None, no special formatting is applied, and falls back to
        ``pandas.options.styler.format.na_rep``.

    uuid_len : int, default 5
        If ``uuid`` is not specified, the length of the ``uuid`` to randomly generate
        expressed in hex characters, in range [0, 32].
    decimal : str, optional
        Character used as decimal separator for floats, complex and integers. If not
        given uses ``pandas.options.styler.format.decimal``.

        .. versionadded:: 1.3.0

    thousands : str, optional, default None
        Character used as thousands separator for floats, complex and integers. If not
        given uses ``pandas.options.styler.format.thousands``.

        .. versionadded:: 1.3.0

    escape : str, optional
        Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
        in cell display string with HTML-safe sequences.
        Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
        ``{``, ``}``, ``~``, ``^``, and ``\`` in the cell display string with
        LaTeX-safe sequences. Use 'latex-math' to replace the characters
        the same way as in 'latex' mode, except for math substrings,
        which either are surrounded by two characters ``$`` or start with
        the character ``\(`` and end with ``\)``.
        If not given uses ``pandas.options.styler.format.escape``.

        .. versionadded:: 1.3.0
    formatter : str, callable, dict, optional
        Object to define how values are displayed. See ``Styler.format``. If not given
        uses ``pandas.options.styler.format.formatter``.

        .. versionadded:: 1.4.0

    Attributes
    ----------
    env : Jinja2 jinja2.Environment
    template_html : Jinja2 Template
    template_html_table : Jinja2 Template
    template_html_style : Jinja2 Template
    template_latex : Jinja2 Template
    loader : Jinja2 Loader

    See Also
    --------
    DataFrame.style : Return a Styler object containing methods for building
        a styled HTML representation for the DataFrame.

    Notes
    -----
    Most styling will be done by passing style functions into
    ``Styler.apply`` or ``Styler.map``. Style functions should
    return values with strings containing CSS ``'attr: value'`` that will
    be applied to the indicated cells.

    If using in the Jupyter notebook, Styler has defined a ``_repr_html_``
    to automatically render itself. Otherwise call Styler.to_html to get
    the generated HTML.

    CSS classes are attached to the generated HTML

    * Index and Column names include ``index_name`` and ``level<k>``
      where `k` is its level in a MultiIndex
    * Index label cells include

      * ``row_heading``
      * ``row<n>`` where `n` is the numeric position of the row
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Column label cells include
      * ``col_heading``
      * ``col<n>`` where `n` is the numeric position of the column
      * ``level<k>`` where `k` is the level in a MultiIndex

    * Blank cells include ``blank``
    * Data cells include ``data``
    * Trimmed cells include ``col_trim`` or ``row_trim``.

    Any, or all, or these classes can be renamed by using the ``css_class_names``
    argument in ``Styler.set_table_classes``, giving a value such as
    *{"row": "MY_ROW_CLASS", "col_trim": "", "row_trim": ""}*.

    Examples
    --------
    >>> df = pd.DataFrame([[1.0, 2.0, 3.0], [4, 5, 6]], index=['a', 'b'],
    ...                   columns=['A', 'B', 'C'])
    >>> pd.io.formats.style.Styler(df, precision=2,
    ...                            caption="My table")  # doctest: +SKIP

    Please see:
    `Table Visualization <../../user_guide/style.ipynb>`_ for more examples.
    """

    def __init__(
        self,
        data: DataFrame | Series,
        precision: int | None = None,
        table_styles: CSSStyles | None = None,
        uuid: str | None = None,
        caption: str | tuple | list | None = None,
        table_attributes: str | None = None,
        cell_ids: bool = True,
        na_rep: str | None = None,
        uuid_len: int = 5,
        decimal: str | None = None,
        thousands: str | None = None,
        escape: str | None = None,
        formatter: ExtFormatter | None = None,
    ) -> None:
        super().__init__(
            data=data,
            uuid=uuid,
            uuid_len=uuid_len,
            table_styles=table_styles,
            table_attributes=table_attributes,
            caption=caption,
            cell_ids=cell_ids,
            precision=precision,
        )

        # validate ordered args
        thousands = thousands or get_option("styler.format.thousands")
        decimal = decimal or get_option("styler.format.decimal")
        na_rep = na_rep or get_option("styler.format.na_rep")
        escape = escape or get_option("styler.format.escape")
        formatter = formatter or get_option("styler.format.formatter")
        # precision is handled by superclass as default for performance

        self.format(
            formatter=formatter,
            precision=precision,
            na_rep=na_rep,
            escape=escape,
            decimal=decimal,
            thousands=thousands,
        )

    def concat(self, other: Styler) -> Styler:
        """
        Append another Styler to combine the output into a single table.

        .. versionadded:: 1.5.0

        Parameters
        ----------
        other : Styler
            The other Styler object which has already been styled and formatted. The
            data for this Styler must have the same columns as the original, and the
            number of index levels must also be the same to render correctly.

        Returns
        -------
        Styler

        Notes
        -----
        The purpose of this method is to extend existing styled dataframes with other
        metrics that may be useful but may not conform to the original's structure.
        For example adding a sub total row, or displaying metrics such as means,
        variance or counts.

        Styles that are applied using the ``apply``, ``map``, ``apply_index``
        and ``map_index``, and formatting applied with ``format`` and
        ``format_index`` will be preserved.

        .. warning::
            Only the output methods ``to_html``, ``to_string`` and ``to_latex``
            currently work with concatenated Stylers.

            Other output methods, including ``to_excel``, **do not** work with
            concatenated Stylers.

        The following should be noted:

          - ``table_styles``, ``table_attributes``, ``caption`` and ``uuid`` are all
            inherited from the original Styler and not ``other``.
          - hidden columns and hidden index levels will be inherited from the
            original Styler
          - ``css`` will be inherited from the original Styler, and the value of
            keys ``data``, ``row_heading`` and ``row`` will be prepended with
            ``foot0_``. If more concats are chained, their styles will be prepended
            with ``foot1_``, ''foot_2'', etc., and if a concatenated style have
            another concatanated style, the second style will be prepended with
            ``foot{parent}_foot{child}_``.

        A common use case is to concatenate user defined functions with
        ``DataFrame.agg`` or with described statistics via ``DataFrame.describe``.
        See examples.

        Examples
        --------
        A common use case is adding totals rows, or otherwise, via methods calculated
        in ``DataFrame.agg``.

        >>> df = pd.DataFrame([[4, 6], [1, 9], [3, 4], [5, 5], [9, 6]],
        ...                   columns=["Mike", "Jim"],
        ...                   index=["Mon", "Tue", "Wed", "Thurs", "Fri"])
        >>> styler = df.style.concat(df.agg(["sum"]).style)  # doctest: +SKIP

        .. figure:: ../../_static/style/footer_simple.png

        Since the concatenated object is a Styler the existing functionality can be
        used to conditionally format it as well as the original.

        >>> descriptors = df.agg(["sum", "mean", lambda s: s.dtype])
        >>> descriptors.index = ["Total", "Average", "dtype"]
        >>> other = (descriptors.style
        ...          .highlight_max(axis=1, subset=(["Total", "Average"], slice(None)))
        ...          .format(subset=("Average", slice(None)), precision=2, decimal=",")
        ...          .map(lambda v: "font-weight: bold;"))
        >>> styler = (df.style
        ...             .highlight_max(color="salmon")
        ...             .set_table_styles([{"selector": ".foot_row0",
        ...                                 "props": "border-top: 1px solid black;"}]))
        >>> styler.concat(other)  # doctest: +SKIP

        .. figure:: ../../_static/style/footer_extended.png

        When ``other`` has fewer index levels than the original Styler it is possible
        to extend the index in ``other``, with placeholder levels.

        >>> df = pd.DataFrame([[1], [2]],
        ...                   index=pd.MultiIndex.from_product([[0], [1, 2]]))
        >>> descriptors = df.agg(["sum"])
        >>> descriptors.index = pd.MultiIndex.from_product([[""], descriptors.index])
        >>> df.style.concat(descriptors.style)  # doctest: +SKIP
        """
        if not isinstance(other, Styler):
            raise TypeError("`other` must be of type `Styler`")
        if not self.data.columns.equals(other.data.columns):
            raise ValueError("`other.data` must have same columns as `Styler.data`")
        if not self.data.index.nlevels == other.data.index.nlevels:
            raise ValueError(
                "number of index levels must be same in `other` "
                "as in `Styler`. See documentation for suggestions."
            )
        self.concatenated.append(other)
        return self

    def _repr_html_(self) -> str | None:
        """
        Hooks into Jupyter notebook rich display system, which calls _repr_html_ by
        default if an object is returned at the end of a cell.
        """
        if get_option("styler.render.repr") == "html":
            return self.to_html()
        return None

    def _repr_latex_(self) -> str | None:
        if get_option("styler.render.repr") == "latex":
            return self.to_latex()
        return None

    def set_tooltips(
        self,
        ttips: DataFrame,
        props: CSSProperties | None = None,
        css_class: str | None = None,
    ) -> Styler:
        """
        Set the DataFrame of strings on ``Styler`` generating ``:hover`` tooltips.

        These string based tooltips are only applicable to ``<td>`` HTML elements,
        and cannot be used for column or index headers.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        ttips : DataFrame
            DataFrame containing strings that will be translated to tooltips, mapped
            by identical column and index values that must exist on the underlying
            Styler data. None, NaN values, and empty strings will be ignored and
            not affect the rendered HTML.
        props : list-like or str, optional
            List of (attr, value) tuples or a valid CSS string. If ``None`` adopts
            the internal default values described in notes.
        css_class : str, optional
            Name of the tooltip class used in CSS, should conform to HTML standards.
            Only useful if integrating tooltips with external CSS. If ``None`` uses the
            internal default value 'pd-t'.

        Returns
        -------
        Styler

        Notes
        -----
        Tooltips are created by adding `<span class="pd-t"></span>` to each data cell
        and then manipulating the table level CSS to attach pseudo hover and pseudo
        after selectors to produce the required the results.

        The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

        The property 'visibility: hidden;' is a key prerequisite to the hover
        functionality, and should always be included in any manual properties
        specification, using the ``props`` argument.

        Tooltips are not designed to be efficient, and can add large amounts of
        additional HTML for larger tables, since they also require that ``cell_ids``
        is forced to `True`.

        Examples
        --------
        Basic application

        >>> df = pd.DataFrame(data=[[0, 1], [2, 3]])
        >>> ttips = pd.DataFrame(
        ...    data=[["Min", ""], [np.nan, "Max"]], columns=df.columns, index=df.index
        ... )
        >>> s = df.style.set_tooltips(ttips).to_html()

        Optionally controlling the tooltip visual display

        >>> df.style.set_tooltips(ttips, css_class='tt-add', props=[
        ...     ('visibility', 'hidden'),
        ...     ('position', 'absolute'),
        ...     ('z-index', 1)])  # doctest: +SKIP
        >>> df.style.set_tooltips(ttips, css_class='tt-add',
        ...     props='visibility:hidden; position:absolute; z-index:1;')
        ... # doctest: +SKIP
        """
        if not self.cell_ids:
            # tooltips not optimised for individual cell check. requires reasonable
            # redesign and more extensive code for a feature that might be rarely used.
            raise NotImplementedError(
                "Tooltips can only render with 'cell_ids' is True."
            )
        if not ttips.index.is_unique or not ttips.columns.is_unique:
            raise KeyError(
                "Tooltips render only if `ttips` has unique index and columns."
            )
        if self.tooltips is None:  # create a default instance if necessary
            self.tooltips = Tooltips()
        self.tooltips.tt_data = ttips
        if props:
            self.tooltips.class_properties = props
        if css_class:
            self.tooltips.class_name = css_class

        return self

    @doc(
        NDFrame.to_excel,
        klass="Styler",
        storage_options=_shared_docs["storage_options"],
        storage_options_versionadded="1.5.0",
    )
    def to_excel(
        self,
        excel_writer: FilePath | WriteExcelBuffer | ExcelWriter,
        sheet_name: str = "Sheet1",
        na_rep: str = "",
        float_format: str | None = None,
        columns: Sequence[Hashable] | None = None,
        header: Sequence[Hashable] | bool = True,
        index: bool = True,
        index_label: IndexLabel | None = None,
        startrow: int = 0,
        startcol: int = 0,
        engine: str | None = None,
        merge_cells: bool = True,
        encoding: str | None = None,
        inf_rep: str = "inf",
        verbose: bool = True,
        freeze_panes: tuple[int, int] | None = None,
        storage_options: StorageOptions | None = None,
    ) -> None:
        from pandas.io.formats.excel import ExcelFormatter

        formatter = ExcelFormatter(
            self,
            na_rep=na_rep,
            cols=columns,
            header=header,
            float_format=float_format,
            index=index,
            index_label=index_label,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
        )
        formatter.write(
            excel_writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            freeze_panes=freeze_panes,
            engine=engine,
            storage_options=storage_options,
        )

    @overload
    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str],
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> None:
        ...

    @overload
    def to_latex(
        self,
        buf: None = ...,
        *,
        column_format: str | None = ...,
        position: str | None = ...,
        position_float: str | None = ...,
        hrules: bool | None = ...,
        clines: str | None = ...,
        label: str | None = ...,
        caption: str | tuple | None = ...,
        sparse_index: bool | None = ...,
        sparse_columns: bool | None = ...,
        multirow_align: str | None = ...,
        multicol_align: str | None = ...,
        siunitx: bool = ...,
        environment: str | None = ...,
        encoding: str | None = ...,
        convert_css: bool = ...,
    ) -> str:
        ...

    def to_latex(
        self,
        buf: FilePath | WriteBuffer[str] | None = None,
        *,
        column_format: str | None = None,
        position: str | None = None,
        position_float: str | None = None,
        hrules: bool | None = None,
        clines: str | None = None,
        label: str | None = None,
        caption: str | tuple | None = None,
        sparse_index: bool | None = None,
        sparse_columns: bool | None = None,
        multirow_align: str | None = None,
        multicol_align: str | None = None,
        siunitx: bool = False,
        environment: str | None = None,
        encoding: str | None = None,
        convert_css: bool = False,
    ) -> str | None:
        r"""
        Write Styler to a file, buffer or string in LaTeX format.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        buf : str, path object, file-like object, or None, default None
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a string ``write()`` function. If None, the result is
            returned as a string.
        column_format : str, optional
            The LaTeX column specification placed in location:

            \\begin{tabular}{<column_format>}

            Defaults to 'l' for index and
            non-numeric data columns, and, for numeric data columns,
            to 'r' by default, or 'S' if ``siunitx`` is ``True``.
        position : str, optional
            The LaTeX positional argument (e.g. 'h!') for tables, placed in location:

            ``\\begin{table}[<position>]``.
        position_float : {"centering", "raggedleft", "raggedright"}, optional
            The LaTeX float command placed in location:

            \\begin{table}[<position>]

            \\<position_float>

            Cannot be used if ``environment`` is "longtable".
        hrules : bool
            Set to `True` to add \\toprule, \\midrule and \\bottomrule from the
            {booktabs} LaTeX package.
            Defaults to ``pandas.options.styler.latex.hrules``, which is `False`.

            .. versionchanged:: 1.4.0
        clines : str, optional
            Use to control adding \\cline commands for the index labels separation.
            Possible values are:

              - `None`: no cline commands are added (default).
              - `"all;data"`: a cline is added for every index value extending the
                width of the table, including data entries.
              - `"all;index"`: as above with lines extending only the width of the
                index entries.
              - `"skip-last;data"`: a cline is added for each index value except the
                last level (which is never sparsified), extending the widtn of the
                table.
              - `"skip-last;index"`: as above with lines extending only the width of the
                index entries.

            .. versionadded:: 1.4.0
        label : str, optional
            The LaTeX label included as: \\label{<label>}.
            This is used with \\ref{<label>} in the main .tex file.
        caption : str, tuple, optional
            If string, the LaTeX table caption included as: \\caption{<caption>}.
            If tuple, i.e ("full caption", "short caption"), the caption included
            as: \\caption[<caption[1]>]{<caption[0]>}.
        sparse_index : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each row.
            Defaults to ``pandas.options.styler.sparse.index``, which is `True`.
        sparse_columns : bool, optional
            Whether to sparsify the display of a hierarchical index. Setting to False
            will display each explicit level element in a hierarchical key for each
            column. Defaults to ``pandas.options.styler.sparse.columns``, which
            is `True`.
        multirow_align : {"c", "t", "b", "naive"}, optional
            If sparsifying hierarchical MultiIndexes whether to align text centrally,
            at the top or bottom using the multirow package. If not given defaults to
            ``pandas.options.styler.latex.multirow_align``, which is `"c"`.
            If "naive" is given renders without multirow.

            .. versionchanged:: 1.4.0
        multicol_align : {"r", "c", "l", "naive-l", "naive-r"}, optional
            If sparsifying hierarchical MultiIndex columns whether to align text at
            the left, centrally, or at the right. If not given defaults to
            ``pandas.options.styler.latex.multicol_align``, which is "r".
            If a naive option is given renders without multicol.
            Pipe decorators can also be added to non-naive values to draw vertical
            rules, e.g. "\|r" will draw a rule on the left side of right aligned merged
            cells.

            .. versionchanged:: 1.4.0
        siunitx : bool, default False
            Set to ``True`` to structure LaTeX compatible with the {siunitx} package.
        environment : str, optional
            If given, the environment that will replace 'table' in ``\\begin{table}``.
            If 'longtable' is specified then a more suitable template is
            rendered. If not given defaults to
            ``pandas.options.styler.latex.environment``, which is `None`.

            .. versionadded:: 1.4.0
        encoding : str, optional
            Character encoding setting. Defaults
            to ``pandas.options.styler.render.encoding``, which is "utf-8".
        convert_css : bool, default False
            Convert simple cell-styles from CSS to LaTeX format. Any CSS not found in
            conversion table is dropped. A style can be forced by adding option
            `--latex`. See notes.

        Returns
        -------
        str or None
            If `buf` is None, returns the result as a string. Otherwise returns `None`.

        See Also
        --------
        Styler.format: Format the text display value of cells.

        Notes
        -----
        **Latex Packages**

        For the following features we recommend the following LaTeX inclusions:

        ===================== ==========================================================
        Feature               Inclusion
        ===================== ==========================================================
        sparse columns        none: included within default {tabular} environment
        sparse rows           \\usepackage{multirow}
        hrules                \\usepackage{booktabs}
        colors                \\usepackage[table]{xcolor}
        siunitx               \\usepackage{siunitx}
        bold (with siunitx)   | \\usepackage{etoolbox}
                              | \\robustify\\bfseries
                              | \\sisetup{detect-all = true}  *(within {document})*
        italic (with siunitx) | \\usepackage{etoolbox}
                              | \\robustify\\itshape
                              | \\sisetup{detect-all = true}  *(within {document})*
        environment           \\usepackage{longtable} if arg is "longtable"
                              | or any other relevant environment package
        hyperlinks            \\usepackage{hyperref}
        ===================== ==========================================================

        **Cell Styles**

        LaTeX styling can only be rendered if the accompanying styling functions have
        been constructed with appropriate LaTeX commands. All styling
        functionality is built around the concept of a CSS ``(<attribute>, <value>)``
        pair (see `Table Visualization <../../user_guide/style.ipynb>`_), and this
        should be replaced by a LaTeX
        ``(<command>, <options>)`` approach. Each cell will be styled individually
        using nested LaTeX commands with their accompanied options.

        For example the following code will highlight and bold a cell in HTML-CSS:

        >>> df = pd.DataFrame([[1,2], [3,4]])
        >>> s = df.style.highlight_max(axis=None,
        ...                            props='background-color:red; font-weight:bold;')
        >>> s.to_html()  # doctest: +SKIP

        The equivalent using LaTeX only commands is the following:

        >>> s = df.style.highlight_max(axis=None,
        ...                            props='cellcolor:{red}; bfseries: ;')
        >>> s.to_latex()  # doctest: +SKIP

        Internally these structured LaTeX ``(<command>, <options>)`` pairs
        are translated to the
        ``display_value`` with the default structure:
        ``\<command><options> <display_value>``.
        Where there are multiple commands the latter is nested recursively, so that
        the above example highlighted cell is rendered as
        ``\cellcolor{red} \bfseries 4``.

        Occasionally this format does not suit the applied command, or
        combination of LaTeX packages that is in use, so additional flags can be
        added to the ``<options>``, within the tuple, to result in different
        positions of required braces (the **default** being the same as ``--nowrap``):

        =================================== ============================================
        Tuple Format                           Output Structure
        =================================== ============================================
        (<command>,<options>)               \\<command><options> <display_value>
        (<command>,<options> ``--nowrap``)  \\<command><options> <display_value>
        (<command>,<options> ``--rwrap``)   \\<command><options>{<display_value>}
        (<command>,<options> ``--wrap``)    {\\<command><options> <display_value>}
        (<command>,<options> ``--lwrap``)   {\\<command><options>} <display_value>
        (<command>,<options> ``--dwrap``)   {\\<command><options>}{<display_value>}
        =================================== ============================================

        For example the `textbf` command for font-weight
        should always be used with `--rwrap` so ``('textbf', '--rwrap')`` will render a
        working cell, wrapped with braces, as ``\textbf{<display_value>}``.

        A more comprehensive example is as follows:

        >>> df = pd.DataFrame([[1, 2.2, "dogs"], [3, 4.4, "cats"], [2, 6.6, "cows"]],
        ...                   index=["ix1", "ix2", "ix3"],
        ...                   columns=["Integers", "Floats", "Strings"])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red};'
        ...           'textit:--rwrap; textbf:--rwrap;'
        ... )
        >>> s.to_latex()  # doctest: +SKIP

        .. figure:: ../../_static/style/latex_1.png

        **Table Styles**

        Internally Styler uses its ``table_styles`` object to parse the
        ``column_format``, ``position``, ``position_float``, and ``label``
        input arguments. These arguments are added to table styles in the format:

        .. code-block:: python

            set_table_styles([
                {"selector": "column_format", "props": f":{column_format};"},
                {"selector": "position", "props": f":{position};"},
                {"selector": "position_float", "props": f":{position_float};"},
                {"selector": "label", "props": f":{{{label.replace(':','§')}}};"}
            ], overwrite=False)

        Exception is made for the ``hrules`` argument which, in fact, controls all three
        commands: ``toprule``, ``bottomrule`` and ``midrule`` simultaneously. Instead of
        setting ``hrules`` to ``True``, it is also possible to set each
        individual rule definition, by manually setting the ``table_styles``,
        for example below we set a regular ``toprule``, set an ``hline`` for
        ``bottomrule`` and exclude the ``midrule``:

        .. code-block:: python

            set_table_styles([
                {'selector': 'toprule', 'props': ':toprule;'},
                {'selector': 'bottomrule', 'props': ':hline;'},
            ], overwrite=False)

        If other ``commands`` are added to table styles they will be detected, and
        positioned immediately above the '\\begin{tabular}' command. For example to
        add odd and even row coloring, from the {colortbl} package, in format
        ``\rowcolors{1}{pink}{red}``, use:

        .. code-block:: python

            set_table_styles([
                {'selector': 'rowcolors', 'props': ':{1}{pink}{red};'}
            ], overwrite=False)

        A more comprehensive example using these arguments is as follows:

        >>> df.columns = pd.MultiIndex.from_tuples([
        ...     ("Numeric", "Integers"),
        ...     ("Numeric", "Floats"),
        ...     ("Non-Numeric", "Strings")
        ... ])
        >>> df.index = pd.MultiIndex.from_tuples([
        ...     ("L0", "ix1"), ("L0", "ix2"), ("L1", "ix3")
        ... ])
        >>> s = df.style.highlight_max(
        ...     props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
        ... )
        >>> s.to_latex(
        ...     column_format="rrrrr", position="h", position_float="centering",
        ...     hrules=True, label="table:5", caption="Styled LaTeX Table",
        ...     multirow_align="t", multicol_align="r"
        ... )  # doctest: +SKIP

        .. figure:: ../../_static/style/latex_2.png

        **Formatting**

        To format values :meth:`Styler.format` should be used prior to calling
        `Styler.to_latex`, as well as other methods such as :meth:`Styler.hide`
        for example:

        >>> s.clear()
        >>> s.table_styles = []
        >>> s.caption = None
        >>> s.format({
        ...    ("Numeric", "Integers"): '\${}',
        ...    ("Numeric", "Floats"): '{:.3f}',
        ...    ("Non-Numeric", "Strings"): str.upper
        ... })  # doctest: +SKIP
                        Numeric      Non-Numeric
                  Integers   Floats    Strings
        L0    ix1       $1   2.200      DOGS
              ix2       $3   4.400      CATS
        L1    ix3       $2   6.600      COWS

        >>> s.to_latex()  # doctest: +SKIP
        \begin{tabular}{llrrl}
        {} & {} & \multicolumn{2}{r}{Numeric} & {Non-Numeric} \\
        {} & {} & {Integers} & {Floats} & {Strings} \\
        \multirow[c]{2}{*}{L0} & ix1 & \\$1 & 2.200 & DOGS \\
         & ix2 & \$3 & 4.400 & CATS \\
        L1 & ix3 & \$2 & 6.600 & COWS \\
        \end{tabular}

        **CSS Conversion**

        This method can convert a Styler constructured with HTML-CSS to LaTeX using
        the following limited conversions.

        ================== ==================== ============= ==========================
        CSS Attribute      CSS value            LaTeX Command LaTeX Options
        ================== ==================== ============= ==========================
        font-weight        | bold               | bfseries
                           | bolder             | bfseries
        font-style         | italic             | itshape
                           | oblique            | slshape
        background-color   | red                cellcolor     | {red}--lwrap
                           | #fe01ea                          | [HTML]{FE01EA}--lwrap
                           | #f0e                             | [HTML]{FF00EE}--lwrap
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}--lwrap
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}--lwrap
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}--lwrap
        color              | red                color         | {red}
                           | #fe01ea                          | [HTML]{FE01EA}
                           | #f0e                             | [HTML]{FF00EE}
                           | rgb(128,255,0)                   | [rgb]{0.5,1,0}
                           | rgba(128,0,0,0.5)                | [rgb]{0.5,0,0}
                           | rgb(25%,255,50%)                 | [rgb]{0.25,1,0.5}
        ================== ==================== ============= ==========================

        It is also possible to add user-defined LaTeX only styles to a HTML-CSS Styler
        using the ``--latex`` flag, and to add LaTeX parsing options that the
        converter will detect within a CSS-comment.

        >>> df = pd.DataFrame([[1]])
        >>> df.style.set_properties(
        ...     **{"font-weight": "bold /* --dwrap */", "Huge": "--latex--rwrap"}
        ... ).to_latex(convert_css=True)  # doctest: +SKIP
        \begin{tabular}{lr}
        {} & {0} \\
        0 & {\bfseries}{\Huge{1}} \\
        \end{tabular}

        Examples
        --------
        Below we give a complete step by step example adding some advanced features
        and noting some common gotchas.

        First we create the DataFrame and Styler as usual, including MultiIndex rows
        and columns, which allow for more advanced formatting options:

        >>> cidx = pd.MultiIndex.from_arrays([
        ...     ["Equity", "Equity", "Equity", "Equity",
        ...      "Stats", "Stats", "Stats", "Stats", "Rating"],
        ...     ["Energy", "Energy", "Consumer", "Consumer", "", "", "", "", ""],
        ...     ["BP", "Shell", "H&M", "Unilever",
        ...      "Std Dev", "Variance", "52w High", "52w Low", ""]
        ... ])
        >>> iidx = pd.MultiIndex.from_arrays([
        ...     ["Equity", "Equity", "Equity", "Equity"],
        ...     ["Energy", "Energy", "Consumer", "Consumer"],
        ...     ["BP", "Shell", "H&M", "Unilever"]
        ... ])
        >>> styler = pd.DataFrame([
        ...     [1, 0.8, 0.66, 0.72, 32.1678, 32.1678**2, 335.12, 240.89, "Buy"],
        ...     [0.8, 1.0, 0.69, 0.79, 1.876, 1.876**2, 14.12, 19.78, "Hold"],
        ...     [0.66, 0.69, 1.0, 0.86, 7, 7**2, 210.9, 140.6, "Buy"],
        ...     [0.72, 0.79, 0.86, 1.0, 213.76, 213.76**2, 2807, 3678, "Sell"],
        ... ], columns=cidx, index=iidx).style

        Second we will format the display and, since our table is quite wide, will
        hide the repeated level-0 of the index:

        >>> (styler.format(subset="Equity", precision=2)
        ...       .format(subset="Stats", precision=1, thousands=",")
        ...       .format(subset="Rating", formatter=str.upper)
        ...       .format_index(escape="latex", axis=1)
        ...       .format_index(escape="latex", axis=0)
        ...       .hide(level=0, axis=0))  # doctest: +SKIP

        Note that one of the string entries of the index and column headers is "H&M".
        Without applying the `escape="latex"` option to the `format_index` method the
        resultant LaTeX will fail to render, and the error returned is quite
        difficult to debug. Using the appropriate escape the "&" is converted to "\\&".

        Thirdly we will apply some (CSS-HTML) styles to our object. We will use a
        builtin method and also define our own method to highlight the stock
        recommendation:

        >>> def rating_color(v):
        ...     if v == "Buy": color = "#33ff85"
        ...     elif v == "Sell": color = "#ff5933"
        ...     else: color = "#ffdd33"
        ...     return f"color: {color}; font-weight: bold;"
        >>> (styler.background_gradient(cmap="inferno", subset="Equity", vmin=0, vmax=1)
        ...       .map(rating_color, subset="Rating"))  # doctest: +SKIP

        All the above styles will work with HTML (see below) and LaTeX upon conversion:

        .. figure:: ../../_static/style/latex_stocks_html.png

        However, we finally want to add one LaTeX only style
        (from the {graphicx} package), that is not easy to convert from CSS and
        pandas does not support it. Notice the `--latex` flag used here,
        as well as `--rwrap` to ensure this is formatted correctly and
        not ignored upon conversion.

        >>> styler.map_index(
        ...     lambda v: "rotatebox:{45}--rwrap--latex;", level=2, axis=1
        ... )  # doctest: +SKIP

        Finally we render our LaTeX adding in other options as required:

        >>> styler.to_latex(
        ...     caption="Selected stock correlation and simple statistics.",
        ...     clines="skip-last;data",
        ...     convert_css=True,
        ...     position_float="centering",
        ...     multicol_align="|c|",
        ...     hrules=True,
        ... )  # doctest: +SKIP
        \begin{table}
        \centering
        \caption{Selected stock correlation and simple statistics.}
        \begin{tabular}{llrrrrrrrrl}
        \toprule
         &  & \multicolumn{4}{|c|}{Equity} & \multicolumn{4}{|c|}{Stats} & Rating \\
         &  & \multicolumn{2}{|c|}{Energy} & \multicolumn{2}{|c|}{Consumer} &
        \multicolumn{4}{|c|}{} &  \\
         &  & \rotatebox{45}{BP} & \rotatebox{45}{Shell} & \rotatebox{45}{H\&M} &
        \rotatebox{45}{Unilever} & \rotatebox{45}{Std Dev} & \rotatebox{45}{Variance} &
        \rotatebox{45}{52w High} & \rotatebox{45}{52w Low} & \rotatebox{45}{} \\
        \midrule
        \multirow[c]{2}{*}{Energy} & BP & {\cellcolor[HTML]{FCFFA4}}
        \color[HTML]{000000} 1.00 & {\cellcolor[HTML]{FCA50A}} \color[HTML]{000000}
        0.80 & {\cellcolor[HTML]{EB6628}} \color[HTML]{F1F1F1} 0.66 &
        {\cellcolor[HTML]{F68013}} \color[HTML]{F1F1F1} 0.72 & 32.2 & 1,034.8 & 335.1
        & 240.9 & \color[HTML]{33FF85} \bfseries BUY \\
         & Shell & {\cellcolor[HTML]{FCA50A}} \color[HTML]{000000} 0.80 &
        {\cellcolor[HTML]{FCFFA4}} \color[HTML]{000000} 1.00 &
        {\cellcolor[HTML]{F1731D}} \color[HTML]{F1F1F1} 0.69 &
        {\cellcolor[HTML]{FCA108}} \color[HTML]{000000} 0.79 & 1.9 & 3.5 & 14.1 &
        19.8 & \color[HTML]{FFDD33} \bfseries HOLD \\
        \cline{1-11}
        \multirow[c]{2}{*}{Consumer} & H\&M & {\cellcolor[HTML]{EB6628}}
        \color[HTML]{F1F1F1} 0.66 & {\cellcolor[HTML]{F1731D}} \color[HTML]{F1F1F1}
        0.69 & {\cellcolor[HTML]{FCFFA4}} \color[HTML]{000000} 1.00 &
        {\cellcolor[HTML]{FAC42A}} \color[HTML]{000000} 0.86 & 7.0 & 49.0 & 210.9 &
        140.6 & \color[HTML]{33FF85} \bfseries BUY \\
         & Unilever & {\cellcolor[HTML]{F68013}} \color[HTML]{F1F1F1} 0.72 &
        {\cellcolor[HTML]{FCA108}} \color[HTML]{000000} 0.79 &
        {\cellcolor[HTML]{FAC42A}} \color[HTML]{000000} 0.86 &
        {\cellcolor[HTML]{FCFFA4}} \color[HTML]{000000} 1.00 & 213.8 & 45,693.3 &
        2,807.0 & 3,678.0 & \color[HTML]{FF5933} \bfseries SELL \\
        \cline{1-11}
        \bottomrule
        \end{tabular}
        \end{table}

        .. figure:: ../../_static/style/latex_stocks.png
        """
        obj = self._copy(deepcopy=True)  # manipulate table_styles on obj, not self

        table_selectors = (
            [style["selector"] for style in self.table_styles]
            if self.table_styles is not None
            else []
        )

        if column_format is not None:
            # add more recent setting to table_styles
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )
        elif "column_format" in table_selectors:
            pass  # adopt what has been previously set in table_styles
        else:
            # create a default: set float, complex, int cols to 'r' ('S'), index to 'l'
            _original_columns = self.data.columns
            self.data.columns = RangeIndex(stop=len(self.data.columns))
            numeric_cols = self.data._get_numeric_data().columns.to_list()
            self.data.columns = _original_columns
            column_format = ""
            for level in range(self.index.nlevels):
                column_format += "" if self.hide_index_[level] else "l"
            for ci, _ in enumerate(self.data.columns):
                if ci not in self.hidden_columns:
                    column_format += (
                        ("r" if not siunitx else "S") if ci in numeric_cols else "l"
                    )
            obj.set_table_styles(
                [{"selector": "column_format", "props": f":{column_format}"}],
                overwrite=False,
            )

        if position:
            obj.set_table_styles(
                [{"selector": "position", "props": f":{position}"}],
                overwrite=False,
            )

        if position_float:
            if environment == "longtable":
                raise ValueError(
                    "`position_float` cannot be used in 'longtable' `environment`"
                )
            if position_float not in ["raggedright", "raggedleft", "centering"]:
                raise ValueError(
                    f"`position_float` should be one of "
                    f"'raggedright', 'raggedleft', 'centering', "
                    f"got: '{position_float}'"
                )
