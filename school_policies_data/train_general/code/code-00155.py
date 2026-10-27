        Whether an empty array may be returned for a 'pattern' field.

    Returns
    -------
    data : ndarray
        Input data if no conversion necessary, or a converted version
    """

    if field is None:
        return data
    if field == "pattern":
        if no_pattern:
            return data
        else:
            return np.zeros(0)

    dtype = _field_to_dtype.get(field, None)
    if dtype is None:
        raise ValueError("Invalid field.")

    return np.asarray(data, dtype=dtype)


def _validate_symmetry(symmetry):
    """
    Check that the symmetry parameter is one that MatrixMarket allows..
    """
    if symmetry is None:
        return "general"

    symmetry = str(symmetry).lower()
    symmetries = ["general", "symmetric", "skew-symmetric", "hermitian"]
    if symmetry not in symmetries:
        raise ValueError("Invalid symmetry. Must be one of: " + ", ".join(symmetries))

    return symmetry


def mmread(source, *, spmatrix=True):
    """
    Reads the contents of a Matrix Market file-like 'source' into a matrix.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extensions .mtx, .mtz.gz)
        or open file-like object.
    spmatrix : bool, optional (default: True)
        If ``True``, return sparse ``coo_matrix``. Otherwise return ``coo_array``.

    Returns
    -------
    a : ndarray or coo_array
        Dense or sparse array depending on the matrix format in the
        Matrix Market file.

    Notes
    -----
    .. versionchanged:: 1.12.0
        C++ implementation.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mmread

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''

    ``mmread(source)`` returns the data as sparse array in COO format.

    >>> m = mmread(StringIO(text), spmatrix=False)
    >>> m
    <COOrdinate sparse array of dtype 'float64'
        with 7 stored elements and shape (5, 5)>
    >>> m.toarray()
    array([[0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 2., 3.],
           [4., 5., 6., 7., 0.],
           [0., 0., 0., 0., 0.]])

    This method is threaded.
    The default number of threads is equal to the number of CPUs in the system.
    Use `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ to override:

    >>> import threadpoolctl
    >>>
    >>> with threadpoolctl.threadpool_limits(limits=2):
    ...     m = mmread(StringIO(text), spmatrix=False)

    """
    cursor, stream_to_close = _get_read_cursor(source)

    if cursor.header.format == "array":
        mat = _read_body_array(cursor)
        if stream_to_close:
            stream_to_close.close()
        return mat
    else:
        triplet, shape = _read_body_coo(cursor, generalize_symmetry=True)
        if stream_to_close:
            stream_to_close.close()
        if spmatrix:
            return coo_matrix(triplet, shape=shape)
        return coo_array(triplet, shape=shape)


def mmwrite(target, a, comment=None, field=None, precision=None, symmetry="AUTO"):
    r"""
    Writes the sparse or dense array `a` to Matrix Market file-like `target`.

    Parameters
    ----------
    target : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object.
    a : array like
        Sparse or dense 2-D array.
    comment : str, optional
        Comments to be prepended to the Matrix Market file.
    field : None or str, optional
        Either 'real', 'complex', 'pattern', or 'integer'.
    precision : None or int, optional
        Number of digits to display for real or complex values.
    symmetry : None or str, optional
        Either 'AUTO', 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
        If symmetry is None the symmetry type of 'a' is determined by its
        values. If symmetry is 'AUTO' the symmetry type of 'a' is either
        determined or set to 'general', at mmwrite's discretion.

    Returns
    -------
    None

    Notes
    -----
    .. versionchanged:: 1.12.0
        C++ implementation.

    Examples
    --------
    >>> from io import BytesIO
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> from scipy.io import mmwrite

    Write a small NumPy array to a matrix market file.  The file will be
    written in the ``'array'`` format.

    >>> a = np.array([[1.0, 0, 0, 0], [0, 2.5, 0, 6.25]])
    >>> target = BytesIO()
    >>> mmwrite(target, a)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    2 4
    1
    0
    0
    2.5
    0
    0
    0
    6.25

    Add a comment to the output file, and set the precision to 3.

    >>> target = BytesIO()
    >>> mmwrite(target, a, comment='\n Some test data.\n', precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    % Some test data.
    %
    2 4
    1.00e+00
    0.00e+00
    0.00e+00
    2.50e+00
    0.00e+00
    0.00e+00
    0.00e+00
    6.25e+00

    Convert to a sparse matrix before calling ``mmwrite``.  This will
    result in the output format being ``'coordinate'`` rather than
    ``'array'``.

    >>> target = BytesIO()
    >>> mmwrite(target, coo_array(a), precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix coordinate real general
    %
    2 4 3
    1 1 1.00e+00
    2 2 2.50e+00
    2 4 6.25e+00

    Write a complex Hermitian array to a matrix market file.  Note that
    only six values are actually written to the file; the other values
    are implied by the symmetry.

    >>> z = np.array([[3, 1+2j, 4-3j], [1-2j, 1, -5j], [4+3j, 5j, 2.5]])
    >>> z
    array([[ 3. +0.j,  1. +2.j,  4. -3.j],
           [ 1. -2.j,  1. +0.j, -0. -5.j],
           [ 4. +3.j,  0. +5.j,  2.5+0.j]])

    >>> target = BytesIO()
    >>> mmwrite(target, z, precision=2)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array complex hermitian
    %
    3 3
    3.0e+00 0.0e+00
    1.0e+00 -2.0e+00
    4.0e+00 3.0e+00
    1.0e+00 0.0e+00
    0.0e+00 5.0e+00
    2.5e+00 0.0e+00

    This method is threaded.
    The default number of threads is equal to the number of CPUs in the system.
    Use `threadpoolctl <https://github.com/joblib/threadpoolctl>`_ to override:

    >>> import threadpoolctl
    >>>
    >>> target = BytesIO()
    >>> with threadpoolctl.threadpool_limits(limits=2):
    ...     mmwrite(target, a)

    """
    from . import _fmm_core

    if isinstance(a, list) or isinstance(a, tuple) or hasattr(a, "__array__"):
        a = np.asarray(a)

    if symmetry == "AUTO":
        if ALWAYS_FIND_SYMMETRY or (hasattr(a, "shape") and max(a.shape) < 100):
            symmetry = None
        else:
            symmetry = "general"

    if symmetry is None:
        symmetry = _mmio.MMFile()._get_symmetry(a)

    symmetry = _validate_symmetry(symmetry)
    cursor = _get_write_cursor(target, comment=comment,
                               precision=precision, symmetry=symmetry)

    if isinstance(a, np.ndarray):
        # Write dense numpy arrays
        a = _apply_field(a, field, no_pattern=True)
        _fmm_core.write_body_array(cursor, a)

    elif issparse(a):
        # Write sparse scipy matrices
        a = a.tocoo()

        if symmetry is not None and symmetry != "general":
            # A symmetric matrix only specifies the elements below the diagonal.
            # Ensure that the matrix satisfies this requirement.
            lower_triangle_mask = a.row >= a.col
            a = coo_array((a.data[lower_triangle_mask],
                              (a.row[lower_triangle_mask],
                               a.col[lower_triangle_mask])), shape=a.shape)

        data = _apply_field(a.data, field)
        _fmm_core.write_body_coo(cursor, a.shape, a.row, a.col, data)

    else:
        raise ValueError(f"unknown matrix type: {type(a)}")


def mminfo(source):
    """
    Return size and storage parameters from Matrix Market file-like 'source'.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object

    Returns
    -------
    rows : int
        Number of matrix rows.
    cols : int
        Number of matrix columns.
    entries : int
        Number of non-zero entries of a sparse matrix
        or rows*cols for a dense matrix.
    format : str
        Either 'coordinate' or 'array'.
    field : str
        Either 'real', 'complex', 'pattern', or 'integer'.
    symmetry : str
        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.

    Notes
    -----
    .. versionchanged:: 1.12.0
        C++ implementation.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mminfo

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''


    ``mminfo(source)`` returns the number of rows, number of columns,
    format, field type and symmetry attribute of the source file.

    >>> mminfo(StringIO(text))
    (5, 5, 7, 'coordinate', 'real', 'general')
    """
    cursor, stream_to_close = _get_read_cursor(source, 1)
    h = cursor.header
    cursor.close()
    if stream_to_close:
        stream_to_close.close()
    return h.nrows, h.ncols, h.nnz, h.format, h.field, h.symmetry
<|endoftext|># Last Change: Mon Aug 20 08:00 PM 2007 J
import re
import datetime

import numpy as np

import csv
import ctypes

"""A module to read arff files."""

__all__ = ['MetaData', 'loadarff', 'ArffError', 'ParseArffError']

# An Arff file is basically two parts:
#   - header
#   - data
#
# A header has each of its components starting by @META where META is one of
# the keyword (attribute of relation, for now).

# TODO:
#   - both integer and reals are treated as numeric -> the integer info
#    is lost!
#   - Replace ValueError by ParseError or something

# We know can handle the following:
#   - numeric and nominal attributes
#   - missing values for numeric attributes

r_meta = re.compile(r'^\s*@')
# Match a comment
r_comment = re.compile(r'^%')
# Match an empty line
r_empty = re.compile(r'^\s+$')
# Match a header line, that is a line which starts by @ + a word
r_headerline = re.compile(r'^\s*@\S*')
r_datameta = re.compile(r'^@[Dd][Aa][Tt][Aa]')
r_relation = re.compile(r'^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\s*(\S*)')
r_attribute = re.compile(r'^\s*@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\s*(..*$)')

r_nominal = re.compile(r'{(.+)}')
r_date = re.compile(r"[Dd][Aa][Tt][Ee]\s+[\"']?(.+?)[\"']?$")

# To get attributes name enclosed with ''
r_comattrval = re.compile(r"'(..+)'\s+(..+$)")
# To get normal attributes
r_wcomattrval = re.compile(r"(\S+)\s+(..+$)")

# ------------------------
# Module defined exception
# ------------------------


class ArffError(OSError):
    pass


class ParseArffError(ArffError):
    pass


# ----------
# Attributes
# ----------
class Attribute:

    type_name = None

    def __init__(self, name):
        self.name = name
        self.range = None
        self.dtype = np.object_

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.
        """
        return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        return None

    def __str__(self):
        """
        Parse a value of this type.
        """
        return self.name + ',' + self.type_name


class NominalAttribute(Attribute):

    type_name = 'nominal'

    def __init__(self, name, values):
        super().__init__(name)
        self.values = values
        self.range = values
        self.dtype = (np.bytes_, max(len(i) for i in values))

    @staticmethod
    def _get_nom_val(atrv):
        """Given a string containing a nominal type, returns a tuple of the
        possible values.

        A nominal type is defined as something framed between braces ({}).

        Parameters
        ----------
        atrv : str
           Nominal type definition

        Returns
        -------
        poss_vals : tuple
           possible values

        Examples
        --------
        >>> from scipy.io.arff._arffread import NominalAttribute
        >>> NominalAttribute._get_nom_val("{floup, bouga, fl, ratata}")
        ('floup', 'bouga', 'fl', 'ratata')
        """
        m = r_nominal.match(atrv)
        if m:
            attrs, _ = split_data_line(m.group(1))
            return tuple(attrs)
        else:
            raise ValueError("This does not look like a nominal string")

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For nominal attributes, the attribute string would be like '{<attr_1>,
         <attr2>, <attr_3>}'.
        """
        if attr_string[0] == '{':
            values = cls._get_nom_val(attr_string)
            return cls(name, values)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        if data_str in self.values:
            return data_str
        elif data_str == '?':
            return data_str
        else:
            raise ValueError(f"{str(data_str)} value not in {str(self.values)}")

    def __str__(self):
        msg = self.name + ",{"
        for i in range(len(self.values)-1):
            msg += self.values[i] + ","
        msg += self.values[-1]
        msg += "}"
        return msg


class NumericAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'numeric'
        self.dtype = np.float64

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For numeric attributes, the attribute string would be like
        'numeric' or 'int' or 'real'.
        """

        attr_string = attr_string.lower().strip()

        if (attr_string[:len('numeric')] == 'numeric' or
           attr_string[:len('int')] == 'int' or
           attr_string[:len('real')] == 'real'):
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.

        Parameters
        ----------
        data_str : str
           string to convert

        Returns
        -------
        f : float
           where float can be nan

        Examples
        --------
        >>> from scipy.io.arff._arffread import NumericAttribute
        >>> atr = NumericAttribute('atr')
        >>> atr.parse_data('1')
        1.0
        >>> atr.parse_data('1\\n')
        1.0
        >>> atr.parse_data('?\\n')
        nan
        """
        if '?' in data_str:
            return np.nan
        else:
            return float(data_str)

    def _basic_stats(self, data):
        nbfac = data.size * 1. / (data.size - 1)
        return (np.nanmin(data), np.nanmax(data),
                np.mean(data), np.std(data) * nbfac)


class StringAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'string'

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For string attributes, the attribute string would be like
        'string'.
        """

        attr_string = attr_string.lower().strip()

        if attr_string[:len('string')] == 'string':
            return cls(name)
        else:
            return None


class DateAttribute(Attribute):

    def __init__(self, name, date_format, datetime_unit):
        super().__init__(name)
        self.date_format = date_format
        self.datetime_unit = datetime_unit
        self.type_name = 'date'
        self.range = date_format
        self.dtype = np.datetime64(0, self.datetime_unit)

    @staticmethod
    def _get_date_format(atrv):
        m = r_date.match(atrv)
        if m:
            pattern = m.group(1).strip()
            # convert time pattern from Java's SimpleDateFormat to C's format
            datetime_unit = None
            if "yyyy" in pattern:
                pattern = pattern.replace("yyyy", "%Y")
                datetime_unit = "Y"
            elif "yy":
                pattern = pattern.replace("yy", "%y")
                datetime_unit = "Y"
            if "MM" in pattern:
                pattern = pattern.replace("MM", "%m")
                datetime_unit = "M"
            if "dd" in pattern:
                pattern = pattern.replace("dd", "%d")
                datetime_unit = "D"
            if "HH" in pattern:
                pattern = pattern.replace("HH", "%H")
                datetime_unit = "h"
            if "mm" in pattern:
                pattern = pattern.replace("mm", "%M")
                datetime_unit = "m"
            if "ss" in pattern:
                pattern = pattern.replace("ss", "%S")
                datetime_unit = "s"
            if "z" in pattern or "Z" in pattern:
                raise ValueError("Date type attributes with time zone not "
                                 "supported, yet")

            if datetime_unit is None:
                raise ValueError("Invalid or unsupported date format")

            return pattern, datetime_unit
        else:
            raise ValueError("Invalid or no date format")

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """

        attr_string_lower = attr_string.lower().strip()

        if attr_string_lower[:len('date')] == 'date':
            date_format, datetime_unit = cls._get_date_format(attr_string)
            return cls(name, date_format, datetime_unit)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        date_str = data_str.strip().strip("'").strip('"')
        if date_str == '?':
            return np.datetime64('NaT', self.datetime_unit)
        else:
            dt = datetime.datetime.strptime(date_str, self.date_format)
            return np.datetime64(dt).astype(
                f"datetime64[{self.datetime_unit}]")

    def __str__(self):
        return super().__str__() + ',' + self.date_format


class RelationalAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'relational'
        self.dtype = np.object_
        self.attributes = []
        self.dialect = None

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """

        attr_string_lower = attr_string.lower().strip()

        if attr_string_lower[:len('relational')] == 'relational':
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        # Copy-pasted
        elems = list(range(len(self.attributes)))

        escaped_string = data_str.encode().decode("unicode-escape")

        row_tuples = []

        for raw in escaped_string.split("\n"):
            row, self.dialect = split_data_line(raw, self.dialect)

            row_tuples.append(tuple(
                [self.attributes[i].parse_data(row[i]) for i in elems]))

        return np.array(row_tuples,
                        [(a.name, a.dtype) for a in self.attributes])

    def __str__(self):
        return (super().__str__() + '\n\t' +
                '\n\t'.join(str(a) for a in self.attributes))


# -----------------
# Various utilities
# -----------------
def to_attribute(name, attr_string):
    attr_classes = (NominalAttribute, NumericAttribute, DateAttribute,
                    StringAttribute, RelationalAttribute)

    for cls in attr_classes:
        attr = cls.parse_attribute(name, attr_string)
        if attr is not None:
            return attr

    raise ParseArffError(f"unknown attribute {attr_string}")


def csv_sniffer_has_bug_last_field():
    """
    Checks if the bug https://bugs.python.org/issue30157 is unpatched.
    """

    # We only compute this once.
    has_bug = getattr(csv_sniffer_has_bug_last_field, "has_bug", None)

    if has_bug is None:
        dialect = csv.Sniffer().sniff("3, 'a'")
        csv_sniffer_has_bug_last_field.has_bug = dialect.quotechar != "'"
        has_bug = csv_sniffer_has_bug_last_field.has_bug

    return has_bug


def workaround_csv_sniffer_bug_last_field(sniff_line, dialect, delimiters):
    """
    Workaround for the bug https://bugs.python.org/issue30157 if is unpatched.
    """
    if csv_sniffer_has_bug_last_field():
        # Reuses code from the csv module
        right_regex = r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?:$|\n)'  # noqa: E501

        for restr in (r'(?P<delim>[^\w\n"\'])(?P<space> ?)(?P<quote>["\']).*?(?P=quote)(?P=delim)',  # ,".*?",  # noqa: E501
                      r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?P<delim>[^\w\n"\'])(?P<space> ?)',  # .*?",  # noqa: E501
                      right_regex,  # ,".*?"
                      r'(?:^|\n)(?P<quote>["\']).*?(?P=quote)(?:$|\n)'):  # ".*?" (no delim, no space)  # noqa: E501
            regexp = re.compile(restr, re.DOTALL | re.MULTILINE)
            matches = regexp.findall(sniff_line)
            if matches:
                break

        # If it does not match the expression that was bugged,
        # then this bug does not apply
        if restr != right_regex:
            return

        groupindex = regexp.groupindex

        # There is only one end of the string
        assert len(matches) == 1
        m = matches[0]

        n = groupindex['quote'] - 1
        quote = m[n]

        n = groupindex['delim'] - 1
        delim = m[n]

        n = groupindex['space'] - 1
        space = bool(m[n])

        dq_regexp = re.compile(
            rf"(({re.escape(delim)})|^)\W*{quote}[^{re.escape(delim)}\n]*{quote}[^{re.escape(delim)}\n]*{quote}\W*(({re.escape(delim)})|$)", re.MULTILINE  # noqa: E501
        )

        doublequote = bool(dq_regexp.search(sniff_line))

        dialect.quotechar = quote
        if delim in delimiters:
            dialect.delimiter = delim
        dialect.doublequote = doublequote
        dialect.skipinitialspace = space


def split_data_line(line, dialect=None):
    delimiters = ",\t"

    # This can not be done in a per reader basis, and relational fields
    # can be HUGE
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

    # Remove the line end if any
    if line[-1] == '\n':
        line = line[:-1]
    
    # Remove potential trailing whitespace
    line = line.strip()
    
    sniff_line = line

    # Add a delimiter if none is present, so that the csv.Sniffer
    # does not complain for a single-field CSV.
    if not any(d in line for d in delimiters):
        sniff_line += ","

    if dialect is None:
        dialect = csv.Sniffer().sniff(sniff_line, delimiters=delimiters)
        workaround_csv_sniffer_bug_last_field(sniff_line=sniff_line,
                                              dialect=dialect,
                                              delimiters=delimiters)

    row = next(csv.reader([line], dialect))

    return row, dialect


# --------------
# Parsing header
# --------------
def tokenize_attribute(iterable, attribute):
    """Parse a raw string in header (e.g., starts by @attribute).

    Given a raw string attribute, try to get the name and type of the
    attribute. Constraints:

    * The first line must start with @attribute (case insensitive, and
      space like characters before @attribute are allowed)
    * Works also if the attribute is spread on multilines.
    * Works if empty lines or comments are in between

    Parameters
    ----------
    attribute : str
       the attribute string.

    Returns
    -------
    name : str
       name of the attribute
    value : str
       value of the attribute
    next : str
       next line to be parsed

    Examples
    --------
    If attribute is a string defined in python as r"floupi real", will
    return floupi as name, and real as value.

    >>> from scipy.io.arff._arffread import tokenize_attribute
    >>> iterable = iter([0] * 10) # dummy iterator
    >>> tokenize_attribute(iterable, r"@attribute floupi real")
    ('floupi', 'real', 0)

    If attribute is r"'floupi 2' real", will return 'floupi 2' as name,
    and real as value.

    >>> tokenize_attribute(iterable, r"  @attribute 'floupi 2' real   ")
    ('floupi 2', 'real', 0)

    """
    sattr = attribute.strip()
    mattr = r_attribute.match(sattr)
    if mattr:
        # atrv is everything after @attribute
        atrv = mattr.group(1)
        if r_comattrval.match(atrv):
            name, type = tokenize_single_comma(atrv)
            next_item = next(iterable)
        elif r_wcomattrval.match(atrv):
            name, type = tokenize_single_wcomma(atrv)
            next_item = next(iterable)
        else:
            # Not sure we should support this, as it does not seem supported by
            # weka.
            raise ValueError("multi line not supported yet")
    else:
        raise ValueError(f"First line unparsable: {sattr}")

    attribute = to_attribute(name, type)

    if type.lower() == 'relational':
        next_item = read_relational_attribute(iterable, attribute, next_item)
    #    raise ValueError("relational attributes not supported yet")

    return attribute, next_item


def tokenize_single_comma(val):
    # XXX we match twice the same string (here and at the caller level). It is
    # stupid, but it is easier for now...
    m = r_comattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ValueError("Error while tokenizing attribute") from e
    else:
        raise ValueError(f"Error while tokenizing single {val}")
    return name, type


def tokenize_single_wcomma(val):
    # XXX we match twice the same string (here and at the caller level). It is
    # stupid, but it is easier for now...
    m = r_wcomattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ValueError("Error while tokenizing attribute") from e
    else:
        raise ValueError(f"Error while tokenizing single {val}")
    return name, type


def read_relational_attribute(ofile, relational_attribute, i):
    """Read the nested attributes of a relational attribute"""

    r_end_relational = re.compile(r'^@[Ee][Nn][Dd]\s*' +
                                  relational_attribute.name + r'\s*$')

    while not r_end_relational.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                attr, i = tokenize_attribute(ofile, i)
                relational_attribute.attributes.append(attr)
            else:
                raise ValueError(f"Error parsing line {i}")
        else:
            i = next(ofile)

    i = next(ofile)
    return i


def read_header(ofile):
    """Read the header of the iterable ofile."""
    i = next(ofile)

    # Pass first comments
    while r_comment.match(i):
        i = next(ofile)

    # Header is everything up to DATA attribute ?
    relation = None
    attributes = []
    while not r_datameta.match(i):
        m = r_headerline.match(i)
        if m:
            isattr = r_attribute.match(i)
            if isattr:
                attr, i = tokenize_attribute(ofile, i)
                attributes.append(attr)
            else:
                isrel = r_relation.match(i)
                if isrel:
                    relation = isrel.group(1)
                else:
                    raise ValueError(f"Error parsing line {i}")
                i = next(ofile)
        else:
            i = next(ofile)

    return relation, attributes


class MetaData:
    """Small container to keep useful information on a ARFF dataset.

    Knows about attributes names and types.

    Examples
    --------
    ::

        data, meta = loadarff('iris.arff')
        # This will print the attributes names of the iris.arff dataset
        for i in meta:
            print(i)
        # This works too
        meta.names()
        # Getting attribute type
        types = meta.types()

    Methods
    -------
    names
    types

    Notes
    -----
    Also maintains the list of attributes in order, i.e., doing for i in
    meta, where meta is an instance of MetaData, will return the
    different attribute names in the order they were defined.
    """
    def __init__(self, rel, attr):
        self.name = rel
        self._attributes = {a.name: a for a in attr}

    def __repr__(self):
        msg = ""
        msg += f"Dataset: {self.name}\n"
        for i in self._attributes:
            msg += f"\t{i}'s type is {self._attributes[i].type_name}"
            if self._attributes[i].range:
                msg += f", range is {str(self._attributes[i].range)}"
            msg += '\n'
        return msg

    def __iter__(self):
        return iter(self._attributes)

    def __getitem__(self, key):
        attr = self._attributes[key]

        return (attr.type_name, attr.range)

    def names(self):
        """Return the list of attribute names.

        Returns
        -------
        attrnames : list of str
            The attribute names.
        """
        return list(self._attributes)

    def types(self):
        """Return the list of attribute types.

        Returns
        -------
        attr_types : list of str
            The attribute types.
        """
        attr_types = [self._attributes[name].type_name
                      for name in self._attributes]
        return attr_types


def loadarff(f):
    """
    Read an arff file.

    The data is returned as a record array, which can be accessed much like
    a dictionary of NumPy arrays. For example, if one of the attributes is
    called 'pressure', then its first 10 data points can be accessed from the
    ``data`` record array like so: ``data['pressure'][0:10]``


    Parameters
    ----------
    f : file-like or str
       File-like object to read from, or filename to open.

    Returns
    -------
    data : record array
       The data of the arff file, accessible by attribute names.
    meta : `MetaData`
       Contains information about the arff file such as name and
       type of attributes, the relation (name of the dataset), etc.

    Raises
    ------
    ParseArffError
        This is raised if the given file is not ARFF-formatted.
    NotImplementedError
        The ARFF file has an attribute which is not supported yet.

    Notes
    -----

    This function should be able to read most arff files. Not
    implemented functionality include:

    * date type attributes
    * string type attributes

    It can read files with numeric and nominal attributes. It cannot read
    files with sparse data ({} in the file). However, this function can
    read files with missing data (? in the file), representing the data
    points as NaNs.

    Examples
    --------
    >>> from scipy.io import arff
    >>> from io import StringIO
    >>> content = \"\"\"
    ... @relation foo
    ... @attribute width  numeric
    ... @attribute height numeric
    ... @attribute color  {red,green,blue,yellow,black}
    ... @data
    ... 5.0,3.25,blue
    ... 4.5,3.75,green
    ... 3.0,4.00,red
    ... \"\"\"
    >>> f = StringIO(content)
    >>> data, meta = arff.loadarff(f)
    >>> data
    array([(5.0, 3.25, 'blue'), (4.5, 3.75, 'green'), (3.0, 4.0, 'red')],
          dtype=[('width', '<f8'), ('height', '<f8'), ('color', '|S6')])
    >>> meta
    Dataset: foo
    \twidth's type is numeric
    \theight's type is numeric
    \tcolor's type is nominal, range is ('red', 'green', 'blue', 'yellow', 'black')

    """
    if hasattr(f, 'read'):
        ofile = f
    else:
        ofile = open(f)
    try:
        return _loadarff(ofile)
    finally:
        if ofile is not f:  # only close what we opened
            ofile.close()


def _loadarff(ofile):
    # Parse the header file
    try:
        rel, attr = read_header(ofile)
    except ValueError as e:
        msg = "Error while parsing header, error was: " + str(e)
        raise ParseArffError(msg) from e

    # Check whether we have a string attribute (not supported yet)
    hasstr = False
    for a in attr:
        if isinstance(a, StringAttribute):
            hasstr = True

    meta = MetaData(rel, attr)

    # XXX The following code is not great
    # Build the type descriptor descr and the list of converters to convert
    # each attribute to the suitable type (which should match the one in
    # descr).

    # This can be used once we want to support integer as integer values and
    # not as numeric anymore (using masked arrays ?).

    if hasstr:
        # How to support string efficiently ? Ideally, we should know the max
        # size of the string before allocating the numpy array.
        raise NotImplementedError("String attributes not supported yet, sorry")

    ni = len(attr)

    def generator(row_iter, delim=','):
        # TODO: this is where we are spending time (~80%). I think things
        # could be made more efficiently:
        #   - We could for example "compile" the function, because some values
        #   do not change here.
        #   - The function to convert a line to dtyped values could also be
        #   generated on the fly from a string and be executed instead of
        #   looping.
        #   - The regex are overkill: for comments, checking that a line starts
        #   by % should be enough and faster, and for empty lines, same thing
        #   --> this does not seem to change anything.

        # 'compiling' the range since it does not change
        # Note, I have already tried zipping the converters and
        # row elements and got slightly worse performance.
        elems = list(range(ni))

        dialect = None
        for raw in row_iter:
            # We do not abstract skipping comments and empty lines for
            # performance reasons.
            if r_comment.match(raw) or r_empty.match(raw):
                continue

            row, dialect = split_data_line(raw, dialect)

            yield tuple([attr[i].parse_data(row[i]) for i in elems])

    a = list(generator(ofile))
    # No error should happen here: it is a bug otherwise
    data = np.array(a, [(a.name, a.dtype) for a in attr])
    return data, meta

<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.io.arff` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'MetaData', 'loadarff', 'ArffError', 'ParseArffError',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="io.arff", module="arffread",
                                   private_modules=["_arffread"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Module to read ARFF files
=========================
ARFF is the standard data format for WEKA.
It is a text file format which support numerical, string and data values.
The format can also represent missing data and sparse data.

Notes
-----
The ARFF support in ``scipy.io`` provides file reading functionality only.
For more extensive ARFF functionality, see `liac-arff
<https://github.com/renatopp/liac-arff>`_.

See the `WEKA website <http://weka.wikispaces.com/ARFF>`_
for more details about the ARFF format and available datasets.

"""
from ._arffread import *
from . import _arffread

# Deprecated namespaces, to be removed in v2.0.0
from .import arffread

__all__ = _arffread.__all__ + ['arffread']

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""
Preliminary module to handle Fortran formats for IO. Does not use this outside
scipy.sparse io for now, until the API is deemed reasonable.

The *Format classes handle conversion between Fortran and Python format, and
FortranFormatParser can create *Format instances from raw Fortran format
strings (e.g. '(3I4)', '(10I3)', etc...)
"""
import re
import threading

import numpy as np


__all__ = ["BadFortranFormat", "FortranFormatParser", "IntFormat", "ExpFormat"]


TOKENS = {
    "LPAR": r"\(",
    "RPAR": r"\)",
    "INT_ID": r"I",
    "EXP_ID": r"E",
    "INT": r"\d+",
    "DOT": r"\.",
}


class BadFortranFormat(SyntaxError):
    pass


def number_digits(n):
    return int(np.floor(np.log10(np.abs(n))) + 1)


class IntFormat:
    @classmethod
    def from_number(cls, n, min=None):
        """Given an integer, returns a "reasonable" IntFormat instance to represent
        any number between 0 and n if n > 0, -n and n if n < 0

        Parameters
        ----------
        n : int
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : IntFormat
            IntFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        without losing precision. For example, IntFormat.from_number(1) will
        return an IntFormat instance of width 2, so that any 0 and 1 may be
        represented as 1-character strings without loss of information.
        """
        width = number_digits(n) + 1
        if n < 0:
            width += 1
        repeat = 80 // width
        return cls(width, min, repeat=repeat)

    def __init__(self, width, min=None, repeat=None):
        self.width = width
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = "IntFormat("
        if self.repeat:
            r += "%d" % self.repeat
        r += "I%d" % self.width
        if self.min:
            r += ".%d" % self.min
        return r + ")"

    @property
    def fortran_format(self):
        r = "("
        if self.repeat:
            r += "%d" % self.repeat
        r += "I%d" % self.width
        if self.min:
            r += ".%d" % self.min
        return r + ")"

    @property
    def python_format(self):
        return "%" + str(self.width) + "d"


class ExpFormat:
    @classmethod
    def from_number(cls, n, min=None):
        """Given a float number, returns a "reasonable" ExpFormat instance to
        represent any number between -n and n.

        Parameters
        ----------
        n : float
            max number one wants to be able to represent
        min : int
            minimum number of characters to use for the format

        Returns
        -------
        res : ExpFormat
            ExpFormat instance with reasonable (see Notes) computed width

        Notes
        -----
        Reasonable should be understood as the minimal string length necessary
        to avoid losing precision.
        """
        # len of one number in exp format: sign + 1|0 + "." +
        # number of digit for fractional part + 'E' + sign of exponent +
        # len of exponent
        finfo = np.finfo(n.dtype)
        # Number of digits for fractional part
        n_prec = finfo.precision + 1
        # Number of digits for exponential part
        n_exp = number_digits(np.max(np.abs([finfo.maxexp, finfo.minexp])))
        width = 1 + 1 + n_prec + 1 + n_exp + 1
        if n < 0:
            width += 1
        repeat = int(np.floor(80 / width))
        return cls(width, n_prec, min, repeat=repeat)

    def __init__(self, width, significand, min=None, repeat=None):
        """\
        Parameters
        ----------
        width : int
            number of characters taken by the string (includes space).
        """
        self.width = width
        self.significand = significand
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = "ExpFormat("
        if self.repeat:
            r += "%d" % self.repeat
        r += "E%d.%d" % (self.width, self.significand)
        if self.min:
            r += "E%d" % self.min
        return r + ")"

    @property
    def fortran_format(self):
        r = "("
        if self.repeat:
            r += "%d" % self.repeat
        r += "E%d.%d" % (self.width, self.significand)
        if self.min:
            r += "E%d" % self.min
        return r + ")"

    @property
    def python_format(self):
        return "%" + str(self.width-1) + "." + str(self.significand) + "E"


class Token:
    def __init__(self, type, value, pos):
        self.type = type
        self.value = value
        self.pos = pos

    def __str__(self):
        return f"""Token('{self.type}', "{self.value}")"""

    def __repr__(self):
        return self.__str__()


class Tokenizer:
    def __init__(self):
        self.tokens = list(TOKENS.keys())
        self.res = [re.compile(TOKENS[i]) for i in self.tokens]

    def input(self, s):
        self.data = s
        self.curpos = 0
        self.len = len(s)

    def next_token(self):
        curpos = self.curpos

        while curpos < self.len:
            for i, r in enumerate(self.res):
                m = r.match(self.data, curpos)
                if m is None:
                    continue
                else:
                    self.curpos = m.end()
                    return Token(self.tokens[i], m.group(), self.curpos)
            raise SyntaxError("Unknown character at position %d (%s)"
                              % (self.curpos, self.data[curpos]))


# Grammar for fortran format:
# format            : LPAR format_string RPAR
# format_string     : repeated | simple
# repeated          : repeat simple
# simple            : int_fmt | exp_fmt
# int_fmt           : INT_ID width
# exp_fmt           : simple_exp_fmt
# simple_exp_fmt    : EXP_ID width DOT significand
# extended_exp_fmt  : EXP_ID width DOT significand EXP_ID ndigits
# repeat            : INT
# width             : INT
# significand       : INT
# ndigits           : INT

# Naive fortran formatter - parser is hand-made
class FortranFormatParser:
    """Parser for Fortran format strings. The parse method returns a *Format
    instance.

    Notes
    -----
    Only ExpFormat (exponential format for floating values) and IntFormat
    (integer format) for now.
    """
    def __init__(self):
        self.tokenizer = threading.local()

    def parse(self, s):
        if not hasattr(self.tokenizer, 't'):
            self.tokenizer.t = Tokenizer()

        self.tokenizer.t.input(s)

        tokens = []

        try:
            while True:
                t = self.tokenizer.t.next_token()
                if t is None:
                    break
                else:
                    tokens.append(t)
            return self._parse_format(tokens)
        except SyntaxError as e:
            raise BadFortranFormat(str(e)) from e

    def _get_min(self, tokens):
        next = tokens.pop(0)
        if not next.type == "DOT":
            raise SyntaxError()
        next = tokens.pop(0)
        return next.value

    def _expect(self, token, tp):
        if not token.type == tp:
            raise SyntaxError()

    def _parse_format(self, tokens):
        if not tokens[0].type == "LPAR":
            raise SyntaxError("Expected left parenthesis at position "
                              "%d (got '%s')" % (0, tokens[0].value))
        elif not tokens[-1].type == "RPAR":
            raise SyntaxError("Expected right parenthesis at position "
                              f"{len(tokens)} (got '{tokens[-1].value}')")

        tokens = tokens[1:-1]
        types = [t.type for t in tokens]
        if types[0] == "INT":
            repeat = int(tokens.pop(0).value)
        else:
            repeat = None

        next = tokens.pop(0)
        if next.type == "INT_ID":
            next = self._next(tokens, "INT")
            width = int(next.value)
            if tokens:
                min = int(self._get_min(tokens))
            else:
                min = None
            return IntFormat(width, min, repeat)
        elif next.type == "EXP_ID":
            next = self._next(tokens, "INT")
            width = int(next.value)

            next = self._next(tokens, "DOT")

            next = self._next(tokens, "INT")
            significand = int(next.value)

            if tokens:
                next = self._next(tokens, "EXP_ID")

                next = self._next(tokens, "INT")
                min = int(next.value)
            else:
                min = None
            return ExpFormat(width, significand, min, repeat)
        else:
            raise SyntaxError(f"Invalid formatter type {next.value}")

    def _next(self, tokens, tp):
        if not len(tokens) > 0:
            raise SyntaxError()
        next = tokens.pop(0)
        self._expect(next, tp)
        return next
<|endoftext|>class spmatrix:
    """This class provides a base class for all sparse matrix classes.

    It cannot be instantiated.  Most of the work is provided by subclasses.
    """
    _allow_nd = (2,)

    @property
    def _bsr_container(self):
        from ._bsr import bsr_matrix
        return bsr_matrix

    @property
    def _coo_container(self):
        from ._coo import coo_matrix
        return coo_matrix

    @property
    def _csc_container(self):
        from ._csc import csc_matrix
        return csc_matrix

    @property
    def _csr_container(self):
        from ._csr import csr_matrix
        return csr_matrix

    @property
    def _dia_container(self):
        from ._dia import dia_matrix
        return dia_matrix

    @property
    def _dok_container(self):
        from ._dok import dok_matrix
        return dok_matrix

    @property
    def _lil_container(self):
        from ._lil import lil_matrix
        return lil_matrix

    # Restore matrix multiplication
    def __mul__(self, other):
        return self._matmul_dispatch(other)

    def __rmul__(self, other):
        return self._rmatmul_dispatch(other)

    # Restore matrix power
    def __pow__(self, power):
        from .linalg import matrix_power

        return matrix_power(self, power)

    ## Backward compatibility

    def set_shape(self, shape):
        """Set the shape of the matrix in-place"""
        # Make sure copy is False since this is in place
        # Make sure format is unchanged because we are doing a __dict__ swap
        new_self = self.reshape(shape, copy=False).asformat(self.format)
        self.__dict__ = new_self.__dict__

    def get_shape(self):
        """Get the shape of the matrix"""
        return self._shape

    shape = property(fget=get_shape, fset=set_shape,
                     doc="Shape of the matrix")

    def asfptype(self):
        """Upcast matrix to a floating point format (if necessary)"""
        return self._asfptype()

    def getmaxprint(self):
        """Maximum number of elements to display when printed."""
        return self._getmaxprint()

    def getformat(self):
        """Matrix storage format"""
        return self.format

    def getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole array, in
            each column, or in each row.
        """
        return self._getnnz(axis=axis)

    def getH(self):
        """Return the Hermitian transpose of this matrix.

        See Also
        --------
        numpy.matrix.getH : NumPy's implementation of `getH` for matrices
        """
        return self.conjugate().transpose()

    def getcol(self, j):
        """Returns a copy of column j of the matrix, as an (m x 1) sparse
        matrix (column vector).
        """
        return self._getcol(j)

    def getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n) sparse
        matrix (row vector).
        """
        return self._getrow(i)

    def todense(self, order=None, out=None):
        """
        Return a dense representation of this sparse matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array (or `numpy.matrix`) as the
            output buffer instead of allocating a new array to
            return. The provided array must have the same shape and
            dtype as the sparse matrix on which you are calling the
            method.

        Returns
        -------
        arr : numpy.matrix, 2-D
            A NumPy matrix object with the same shape and containing
            the same data represented by the sparse matrix, with the
            requested memory order. If `out` was passed and was an
            array (rather than a `numpy.matrix`), it will be filled
            with the appropriate values and returned wrapped in a
            `numpy.matrix` object that shares the same memory.
        """
        return super().todense(order, out)
<|endoftext|>import numpy as np
import scipy as sp

__all__ = ['save_npz', 'load_npz']


# Make loading safe vs. malicious input
PICKLE_KWARGS = dict(allow_pickle=False)


def save_npz(file, matrix, compressed=True):
    """ Save a sparse matrix or array to a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already
        there.
    matrix: spmatrix or sparray
        The sparse matrix or array to save.
        Supported formats: ``csc``, ``csr``, ``bsr``, ``dia`` or ``coo``.
    compressed : bool, optional
        Allow compressing the file. Default: True

    See Also
    --------
    scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.
    numpy.savez: Save several arrays into a ``.npz`` archive.
    numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import numpy as np
    >>> import scipy as sp
    >>> sparse_matrix = sp.sparse.csc_matrix([[0, 0, 3], [4, 0, 0]])
    >>> sparse_matrix
    <Compressed Sparse Column sparse matrix of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_matrix.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    >>> sp.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> sparse_matrix = sp.sparse.load_npz('/tmp/sparse_matrix.npz')

    >>> sparse_matrix
    <Compressed Sparse Column sparse matrix of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_matrix.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)
    """
    arrays_dict = {}
    if matrix.format in ('csc', 'csr', 'bsr'):
        arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    elif matrix.format == 'dia':
        arrays_dict.update(offsets=matrix.offsets)
    elif matrix.format == 'coo':
        arrays_dict.update(row=matrix.row, col=matrix.col)
    else:
        msg = f'Save is not implemented for sparse matrix of format {matrix.format}.'
        raise NotImplementedError(msg)
    arrays_dict.update(
        format=matrix.format.encode('ascii'),
        shape=matrix.shape,
        data=matrix.data
    )
    if isinstance(matrix, sp.sparse.sparray):
        arrays_dict.update(_is_array=True)
    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)


def load_npz(file):
    """ Load a sparse array/matrix from a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be loaded.

    Returns
    -------
    result : csc_array, csr_array, bsr_array, dia_array or coo_array
        A sparse array/matrix containing the loaded data.

    Raises
    ------
    OSError
        If the input file does not exist or cannot be read.

    See Also
    --------
    scipy.sparse.save_npz: Save a sparse array/matrix to a file using ``.npz`` format.
    numpy.load: Load several arrays from a ``.npz`` archive.

    Examples
    --------
    Store sparse array/matrix to disk, and load it again:

    >>> import numpy as np
    >>> import scipy as sp
    >>> sparse_array = sp.sparse.csc_array([[0, 0, 3], [4, 0, 0]])
    >>> sparse_array
    <Compressed Sparse Column sparse array of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_array.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    >>> sp.sparse.save_npz('/tmp/sparse_array.npz', sparse_array)
    >>> sparse_array = sp.sparse.load_npz('/tmp/sparse_array.npz')

    >>> sparse_array
    <Compressed Sparse Column sparse array of dtype 'int64'
        with 2 stored elements and shape (2, 3)>
    >>> sparse_array.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    In this example we force the result to be csr_array from csr_matrix
    >>> sparse_matrix = sp.sparse.csc_matrix([[0, 0, 3], [4, 0, 0]])
    >>> sp.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> tmp = sp.sparse.load_npz('/tmp/sparse_matrix.npz')
    >>> sparse_array = sp.sparse.csr_array(tmp)
    """
    with np.load(file, **PICKLE_KWARGS) as loaded:
        sparse_format = loaded.get('format')
        if sparse_format is None:
            raise ValueError(f'The file {file} does not contain '
                             f'a sparse array or matrix.')
        sparse_format = sparse_format.item()

        if not isinstance(sparse_format, str):
            # Play safe with Python 2 vs 3 backward compatibility;
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            sparse_format = sparse_format.decode('ascii')

        if loaded.get('_is_array'):
            sparse_type = sparse_format + '_array'
        else:
            sparse_type = sparse_format + '_matrix'

        try:
            cls = getattr(sp.sparse, f'{sparse_type}')
        except AttributeError as e:
            raise ValueError(f'Unknown format "{sparse_type}"') from e

        if sparse_format in ('csc', 'csr', 'bsr'):
            return cls((loaded['data'], loaded['indices'], loaded['indptr']),
                       shape=loaded['shape'])
        elif sparse_format == 'dia':
            return cls((loaded['data'], loaded['offsets']),
                       shape=loaded['shape'])
        elif sparse_format == 'coo':
            return cls((loaded['data'], (loaded['row'], loaded['col'])),
                       shape=loaded['shape'])
        else:
            raise NotImplementedError(f'Load is not implemented for '
                                      f'sparse matrix of format {sparse_format}.')
<|endoftext|>"""Compressed Sparse Row matrix format"""

__docformat__ = "restructuredtext en"

__all__ = ['csr_array', 'csr_matrix', 'isspmatrix_csr']

import numpy as np

from ._matrix import spmatrix
from ._base import _spbase, sparray
from ._sparsetools import (csr_tocsc, csr_tobsr, csr_count_blocks,
                           get_csr_submatrix, csr_sample_values)
from ._sputils import upcast

from ._compressed import _cs_matrix


class _csr_base(_cs_matrix):
    _format = 'csr'
    _allow_nd = (1, 2)

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        if self.ndim == 1:
            return self.copy() if copy else self
        M, N = self.shape
        return self._csc_container((self.data, self.indices,
                                    self.indptr), shape=(N, M), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    def tolil(self, copy=False):
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to lil format")
        lil = self._lil_container(self.shape, dtype=self.dtype)

        self.sum_duplicates()
        ptr,ind,dat = self.indptr,self.indices,self.data
        rows, data = lil.rows, lil.data

        for n in range(self.shape[0]):
            start = ptr[n]
            end = ptr[n+1]
            rows[n] = ind[start:end].tolist()
            data[n] = dat[start:end].tolist()

        return lil

    tolil.__doc__ = _spbase.tolil.__doc__

    def tocsr(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocsr.__doc__ = _spbase.tocsr.__doc__

    def tocsc(self, copy=False):
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to csc format")
        M, N = self.shape
        idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, M))
        indptr = np.empty(N + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        csr_tocsc(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        A = self._csc_container((data, indices, indptr), shape=self.shape)
        A.has_sorted_indices = True
        return A

    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tobsr(self, blocksize=None, copy=True):
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to bsr format")
        if blocksize is None:
            from ._spfuncs import estimate_blocksize
            return self.tobsr(blocksize=estimate_blocksize(self))

        elif blocksize == (1,1):
            arg1 = (self.data.reshape(-1,1,1),self.indices,self.indptr)
            return self._bsr_container(arg1, shape=self.shape, copy=copy)

        else:
            R,C = blocksize
            M,N = self.shape

            if R < 1 or C < 1 or M % R != 0 or N % C != 0:
                raise ValueError(f'invalid blocksize {blocksize}')

            blks = csr_count_blocks(M,N,R,C,self.indptr,self.indices)

            idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                        maxval=max(N//C, blks))
            indptr = np.empty(M//R+1, dtype=idx_dtype)
            indices = np.empty(blks, dtype=idx_dtype)
            data = np.zeros((blks,R,C), dtype=self.dtype)

            csr_tobsr(M, N, R, C,
                      self.indptr.astype(idx_dtype),
                      self.indices.astype(idx_dtype),
                      self.data,
                      indptr, indices, data.ravel())

            return self._bsr_container(
                (data, indices, indptr), shape=self.shape
            )

    tobsr.__doc__ = _spbase.tobsr.__doc__

    # these functions are used by the parent class (_cs_matrix)
    # to remove redundancy between csc_matrix and csr_array
    @staticmethod
    def _swap(x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x

    def __iter__(self):
        if self.ndim == 1:
            zero = self.dtype.type(0)
            u = 0
            for v, d in zip(self.indices, self.data):
                for _ in range(v - u):
                    yield zero
                yield d
                u = v + 1
            for _ in range(self.shape[0] - u):
                yield zero
            return

        indptr = np.zeros(2, dtype=self.indptr.dtype)
        # return 1d (sparray) or 2drow (spmatrix)
        shape = self.shape[1:] if isinstance(self, sparray) else (1, self.shape[1])
        i0 = 0
        for i1 in self.indptr[1:]:
            indptr[1] = i1 - i0
            indices = self.indices[i0:i1]
            data = self.data[i0:i1]
            yield self.__class__((data, indices, indptr), shape=shape, copy=True)
            i0 = i1

    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        if self.ndim == 1:
            if i not in (0, -1):
                raise IndexError(f'index ({i}) out of range')
            return self.reshape((1, self.shape[0]), copy=True)

        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i, i + 1, 0, N)
        return self.__class__((data, indices, indptr), shape=(1, N),
                              dtype=self.dtype, copy=False)

    def _getcol(self, i):
        """Returns a copy of column i. A (m x 1) sparse array (column vector).
        """
        if self.ndim == 1:
            raise ValueError("getcol not provided for 1d arrays. Use indexing A[j]")
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, 0, M, i, i + 1)
        return self.__class__((data, indices, indptr), shape=(M, 1),
                              dtype=self.dtype, copy=False)

    def _get_int(self, idx):
        spot = np.flatnonzero(self.indices == idx)
        if spot.size:
            return self.data[spot[0]]
        return self.data.dtype.type(0)

    def _get_slice(self, idx):
        if idx == slice(None):
            return self.copy()
        if idx.step in (1, None):
            ret = self._get_submatrix(0, idx, copy=True)
            return ret.reshape(ret.shape[-1])
        return self._minor_slice(idx)

    def _get_array(self, idx):
        idx_dtype = self._get_index_dtype(self.indices)
        idx = np.asarray(idx, dtype=idx_dtype)
        if idx.size == 0:
            return self.__class__([], dtype=self.dtype)

        M, N = 1, self.shape[0]
        row = np.zeros_like(idx, dtype=idx_dtype)
        col = np.asarray(idx, dtype=idx_dtype)
        val = np.empty(row.size, dtype=self.dtype)
        csr_sample_values(M, N, self.indptr, self.indices, self.data,
                          row.size, row, col, val)

        new_shape = col.shape if col.shape[0] > 1 else (col.shape[0],)
        return self.__class__(val.reshape(new_shape))

    def _get_intXarray(self, row, col):
        return self._getrow(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        # TODO: uncomment this once it's faster:
        # return self._getrow(row)._minor_slice(col)

        M, N = self.shape
        start, stop, stride = col.indices(N)

        ii, jj = self.indptr[row:row+2]
        row_indices = self.indices[ii:jj]
        row_data = self.data[ii:jj]

        if stride > 0:
            ind = (row_indices >= start) & (row_indices < stop)
        else:
            ind = (row_indices <= start) & (row_indices > stop)

        if abs(stride) > 1:
            ind &= (row_indices - start) % stride == 0

        row_indices = (row_indices[ind] - start) // stride
        row_data = row_data[ind]
        row_indptr = np.array([0, len(row_indices)])

        if stride < 0:
            row_data = row_data[::-1]
            row_indices = abs(row_indices[::-1])

        shape = (1, max(0, int(np.ceil(float(stop - start) / stride))))
        return self.__class__((row_data, row_indices, row_indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        return self._major_slice(row)._get_submatrix(minor=col)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        res = self._major_index_fancy(row)._get_submatrix(minor=col)
        if row.ndim > 1:
            return res.reshape(row.shape)
        return res

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            col = np.arange(*col.indices(self.shape[1]))
            return self._get_arrayXarray(row, col)
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    def _set_int(self, idx, x):
        self._set_many(0, idx, x)

    def _set_array(self, idx, x):
        x = np.broadcast_to(x, idx.shape)
        self._set_many(np.zeros_like(idx), idx, x)


def isspmatrix_csr(x):
    """Is `x` of csr_matrix type?

    Parameters
    ----------
    x
        object to check for being a csr matrix

    Returns
    -------
    bool
        True if `x` is a csr matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import csr_array, csr_matrix, coo_matrix, isspmatrix_csr
    >>> isspmatrix_csr(csr_matrix([[5]]))
    True
    >>> isspmatrix_csr(csr_array([[5]]))
    False
    >>> isspmatrix_csr(coo_matrix([[5]]))
    False
    """
    return isinstance(x, csr_matrix)


# This namespace class separates array from matrix with isinstance
class csr_array(_csr_base, sparray):
    """
    Compressed Sparse Row array.

    This can be instantiated in several ways:
        csr_array(D)
            where D is a 2-D ndarray

        csr_array(S)
            with another sparse array or matrix S (equivalent to S.tocsr())

        csr_array((M, N), [dtype])
            to construct an empty array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csr_array((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csr_array((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the array dimensions
            are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        CSR format data array of the array
    indices
        CSR format index array of the array
    indptr
        CSR format index pointer array of the array
    has_sorted_indices
    has_canonical_format
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products

    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical Format
        - Within each row, indices are sorted by column.
        - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> csr_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_array((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    Duplicate entries are summed together:

    >>> row = np.array([0, 1, 2, 0])
    >>> col = np.array([0, 1, 1, 0])
    >>> data = np.array([1, 2, 4, 8])
    >>> csr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[9, 0, 0],
           [0, 2, 0],
           [0, 4, 0]])

    As an example of how to construct a CSR array incrementally,
    the following snippet builds a term-document array from texts:

    >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
    >>> indptr = [0]
    >>> indices = []
    >>> data = []
    >>> vocabulary = {}
    >>> for d in docs:
    ...     for term in d:
    ...         index = vocabulary.setdefault(term, len(vocabulary))
    ...         indices.append(index)
    ...         data.append(1)
    ...     indptr.append(len(indices))
    ...
    >>> csr_array((data, indices, indptr), dtype=int).toarray()
    array([[2, 1, 0, 0],
           [0, 1, 1, 1]])

    """


class csr_matrix(spmatrix, _csr_base):
    """
    Compressed Sparse Row matrix.

    This can be instantiated in several ways:
        csr_matrix(D)
            where D is a 2-D ndarray

        csr_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocsr())

        csr_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the matrix dimensions
            are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        CSR format data array of the matrix
    indices
        CSR format index array of the matrix
    indptr
        CSR format index pointer array of the matrix
    has_sorted_indices
    has_canonical_format
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products

    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical Format
        - Within each row, indices are sorted by column.
        - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> csr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    Duplicate entries are summed together:

    >>> row = np.array([0, 1, 2, 0])
    >>> col = np.array([0, 1, 1, 0])
    >>> data = np.array([1, 2, 4, 8])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[9, 0, 0],
           [0, 2, 0],
           [0, 4, 0]])

    As an example of how to construct a CSR matrix incrementally,
    the following snippet builds a term-document matrix from texts:

    >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
    >>> indptr = [0]
    >>> indices = []
    >>> data = []
    >>> vocabulary = {}
    >>> for d in docs:
    ...     for term in d:
    ...         index = vocabulary.setdefault(term, len(vocabulary))
    ...         indices.append(index)
    ...         data.append(1)
    ...     indptr.append(len(indices))
    ...
    >>> csr_matrix((data, indices, indptr), dtype=int).toarray()
    array([[2, 1, 0, 0],
           [0, 1, 1, 1]])

    """

<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="sputils",
                                   private_modules=["_sputils"], all=__all__,
                                   attribute=name)
<|endoftext|>""" Utility functions for sparse matrix module
"""

import sys
from typing import Any, Literal, Union
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong


__all__ = ['upcast', 'getdtype', 'getdata', 'isscalarlike', 'isintlike',
           'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype',
           'broadcast_shapes']

supported_dtypes = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc,
                    np.uintc, np_long, np_ulong, np.longlong, np.ulonglong,
                    np.float32, np.float64, np.longdouble,
                    np.complex64, np.complex128, np.clongdouble]

_upcast_memo = {}


def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples
    --------
    >>> from scipy.sparse._sputils import upcast
    >>> upcast('int32')
    <class 'numpy.int32'>
    >>> upcast('bool')
    <class 'numpy.bool'>
    >>> upcast('int32','float32')
    <class 'numpy.float64'>
    >>> upcast('bool',complex,float)
    <class 'numpy.complex128'>

    """

    t = _upcast_memo.get(hash(args))
    if t is not None:
        return t

    upcast = np.result_type(*args)

    for t in supported_dtypes:
        if np.can_cast(upcast, t):
            _upcast_memo[hash(args)] = t
            return t

    raise TypeError(f'no supported conversion for types: {args!r}')


def upcast_char(*args):
    """Same as `upcast` but taking dtype.char as input (faster)."""
    t = _upcast_memo.get(args)
    if t is not None:
        return t
    t = upcast(*map(np.dtype, args))
    _upcast_memo[args] = t
    return t


def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    return (np.array([0], dtype=dtype) * scalar).dtype


def downcast_intp_index(arr):
    """
    Down-cast index array to np.intp dtype if it is of a larger dtype.

    Raise an error if the array contains a value that is too large for
    intp.
    """
    if arr.dtype.itemsize > np.dtype(np.intp).itemsize:
        if arr.size == 0:
            return arr.astype(np.intp)
        maxval = arr.max()
        minval = arr.min()
        if maxval > np.iinfo(np.intp).max or minval < np.iinfo(np.intp).min:
            raise ValueError("Cannot deal with arrays with indices larger "
                             "than the machine maximum address size "
                             "(e.g. 64-bit indices on 32-bit machine).")
        return arr.astype(np.intp)
    return arr


def to_native(A):
    """
    Ensure that the data type of the NumPy array `A` has native byte order.

    `A` must be a NumPy array.  If the data type of `A` does not have native
    byte order, a copy of `A` with a native byte order is returned. Otherwise
    `A` is returned.
    """
    dt = A.dtype
    if dt.isnative:
        # Don't call `asarray()` if A is already native, to avoid unnecessarily
        # creating a view of the input array.
        return A
    return np.asarray(A, dtype=dt.newbyteorder('native'))


def getdtype(dtype, a=None, default=None):
    """Form a supported numpy dtype based on input arguments.

    Returns a valid ``numpy.dtype`` from `dtype` if not None,
    or else ``a.dtype`` if possible, or else the given `default`
    if not None, or else raise a ``TypeError``.

    The resulting ``dtype`` must be in ``supported_dtypes``:
        bool_, int8, uint8, int16, uint16, int32, uint32,
        int64, uint64, longlong, ulonglong, float32, float64,
        longdouble, complex64, complex128, clongdouble
    """
    if dtype is None:
        try:
            newdtype = a.dtype
        except AttributeError as e:
            if default is not None:
                newdtype = np.dtype(default)
            else:
                raise TypeError("could not interpret data type") from e
    else:
        newdtype = np.dtype(dtype)

    if newdtype not in supported_dtypes:
        supported_dtypes_fmt = ", ".join(t.__name__ for t in supported_dtypes)
        raise ValueError(f"scipy.sparse does not support dtype {newdtype.name}. "
                         f"The only supported types are: {supported_dtypes_fmt}.")
    return newdtype


def getdata(obj, dtype=None, copy=False) -> np.ndarray:
    """
    This is a wrapper of `np.array(obj, dtype=dtype, copy=copy)`
    that will generate a warning if the result is an object array.
    """
    data = np.array(obj, dtype=dtype, copy=copy)
    # Defer to getdtype for checking that the dtype is OK.
    # This is called for the validation only; we don't need the return value.
    getdtype(data.dtype)
    return data


def safely_cast_index_arrays(A, idx_dtype=np.int32, msg=""):
    """Safely cast sparse array indices to `idx_dtype`.

    Check the shape of `A` to determine if it is safe to cast its index
    arrays to dtype `idx_dtype`. If any dimension in shape is larger than
    fits in the dtype, casting is unsafe so raise ``ValueError``.
    If safe, cast the index arrays to `idx_dtype` and return the result
    without changing the input `A`. The caller can assign results to `A`
    attributes if desired or use the recast index arrays directly.

    Unless downcasting is needed, the original index arrays are returned.
    You can test e.g. ``A.indptr is new_indptr`` to see if downcasting occurred.

    .. versionadded:: 1.15.0

    Parameters
    ----------
    A : sparse array or matrix
        The array for which index arrays should be downcast.
    idx_dtype : dtype
        Desired dtype. Should be an integer dtype (default: ``np.int32``).
        Most of scipy.sparse uses either int64 or int32.
    msg : string, optional
        A string to be added to the end of the ValueError message
        if the array shape is too big to fit in `idx_dtype`.
        The error message is ``f"<index> values too large for {msg}"``
        It should indicate why the downcasting is needed, e.g. "SuperLU",
        and defaults to f"dtype {idx_dtype}".

    Returns
    -------
    idx_arrays : ndarray or tuple of ndarrays
        Based on ``A.format``, index arrays are returned after casting to `idx_dtype`.
        For CSC/CSR, returns ``(indices, indptr)``.
        For COO, returns ``coords``.
        For DIA, returns ``offsets``.
        For BSR, returns ``(indices, indptr)``.

    Raises
    ------
    ValueError
        If the array has shape that would not fit in the new dtype, or if
        the sparse format does not use index arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> data = [3]
    >>> coords = (np.array([3]), np.array([1]))  # Note: int64 arrays
    >>> A = sparse.coo_array((data, coords))
    >>> A.coords[0].dtype
    dtype('int64')

    >>> # rescast after construction, raising exception if shape too big
    >>> coords = sparse.safely_cast_index_arrays(A, np.int32)
    >>> A.coords[0] is coords[0]  # False if casting is needed
    False
    >>> A.coords = coords  # set the index dtype of A
    >>> A.coords[0].dtype
    dtype('int32')
    """
    if not msg:
        msg = f"dtype {idx_dtype}"
    # check for safe downcasting
    max_value = np.iinfo(idx_dtype).max

    if A.format in ("csc", "csr"):
        # indptr[-1] is max b/c indptr always sorted
        if A.indptr[-1] > max_value:
            raise ValueError(f"indptr values too large for {msg}")

        # check shape vs dtype
        if max(*A.shape) > max_value:
            if (A.indices > max_value).any():
                raise ValueError(f"indices values too large for {msg}")

        indices = A.indices.astype(idx_dtype, copy=False)
        indptr = A.indptr.astype(idx_dtype, copy=False)
        return indices, indptr

    elif A.format == "coo":
        if max(*A.shape) > max_value:
            if any((co > max_value).any() for co in A.coords):
                raise ValueError(f"coords values too large for {msg}")
        return tuple(co.astype(idx_dtype, copy=False) for co in A.coords)

    elif A.format == "dia":
        if max(*A.shape) > max_value:
            if (A.offsets > max_value).any():
                raise ValueError(f"offsets values too large for {msg}")
        offsets = A.offsets.astype(idx_dtype, copy=False)
        return offsets

    elif A.format == 'bsr':
        R, C = A.blocksize
        if A.indptr[-1] * R > max_value:
            raise ValueError("indptr values too large for {msg}")
        if max(*A.shape) > max_value:
            if (A.indices * C > max_value).any():
                raise ValueError(f"indices values too large for {msg}")
        indices = A.indices.astype(idx_dtype, copy=False)
        indptr = A.indptr.astype(idx_dtype, copy=False)
        return indices, indptr

    else:
        raise TypeError(f'Format {A.format} is not associated with index arrays. '
                        'DOK and LIL have dict and list, not array.')


def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import sparse
    >>> # select index dtype based on shape
    >>> shape = (3, 3)
    >>> idx_dtype = sparse.get_index_dtype(maxval=max(shape))
    >>> data = [1.1, 3.0, 1.5]
    >>> indices = np.array([0, 1, 0], dtype=idx_dtype)
    >>> indptr = np.array([0, 2, 3, 3], dtype=idx_dtype)
    >>> A = sparse.csr_array((data, indices, indptr), shape=shape)
    >>> A.indptr.dtype
    dtype('int32')

    >>> # select based on larger of existing arrays and shape
    >>> shape = (3, 3)
    >>> idx_dtype = sparse.get_index_dtype(A.indptr, maxval=max(shape))
    >>> idx_dtype
    <class 'numpy.int32'>
    """
    # not using intc directly due to misinteractions with pythran
    if np.intc().itemsize != 4:
        return np.int64

    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)

    if maxval is not None:
        maxval = np.int64(maxval)
        if maxval > int32max:
            return np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = np.asarray(arr)
        if not np.can_cast(arr.dtype, np.int32):
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        # a bigger type not needed
                        continue
            return np.int64
    return np.int32


def get_sum_dtype(dtype: np.dtype) -> np.dtype:
    """Mimic numpy's casting for np.sum"""
    if dtype.kind == 'u' and np.can_cast(dtype, np.uint):
        return np.uint
    if np.can_cast(dtype, np.int_):
        return np.int_
    return dtype


def isscalarlike(x) -> bool:
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


def isintlike(x) -> bool:
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    # Fast-path check to eliminate non-scalar values. operator.index would
    # catch this case too, but the exception catching is slow.
    if np.ndim(x) != 0:
        return False
    try:
        operator.index(x)
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)
        except (TypeError, ValueError):
            return False
        if loose_int:
            msg = "Inexact indices into sparse matrices are not allowed"
            raise ValueError(msg)
        return loose_int
    return True


def isshape(x, nonneg=False, *, allow_nd=(2,)) -> bool:
    """Is x a valid tuple of dimensions?

    If nonneg, also checks that the dimensions are non-negative.
    Shapes of length in the tuple allow_nd are allowed.
    """
    ndim = len(x)
    if ndim not in allow_nd:
        return False

    for d in x:
        if not isintlike(d):
            return False
        if nonneg and d < 0:
            return False
    return True


def issequence(t) -> bool:
    return ((isinstance(t, list | tuple) and
            (len(t) == 0 or np.isscalar(t[0]))) or
            (isinstance(t, np.ndarray) and (t.ndim == 1)))


def ismatrix(t) -> bool:
    return ((isinstance(t, list | tuple) and
             len(t) > 0 and issequence(t[0])) or
            (isinstance(t, np.ndarray) and t.ndim == 2))


def isdense(x) -> bool:
    return isinstance(x, np.ndarray)


def validateaxis(axis) -> None:
    if axis is None:
        return
    axis_type = type(axis)

    # In NumPy, you can pass in tuples for 'axis', but they are
    # not very useful for sparse matrices given their limited
    # dimensions, so let's make it explicit that they are not
    # allowed to be passed in
    if isinstance(axis, tuple):
        raise TypeError("Tuples are not accepted for the 'axis' parameter. "
                        "Please pass in one of the following: "
                        "{-2, -1, 0, 1, None}.")

    # If not a tuple, check that the provided axis is actually
    # an integer and raise a TypeError similar to NumPy's
    if not np.issubdtype(np.dtype(axis_type), np.integer):
        raise TypeError(f"axis must be an integer, not {axis_type.__name__}")

    if not (-2 <= axis <= 1):
        raise ValueError("axis out of range")


def check_shape(args, current_shape=None, *, allow_nd=(2,)) -> tuple[int, ...]:
    """Imitate numpy.matrix handling of shape arguments

    Parameters
    ----------
    args : array_like
        Data structures providing information about the shape of the sparse array.
    current_shape : tuple, optional
        The current shape of the sparse array or matrix.
        If None (default), the current shape will be inferred from args.
    allow_nd : tuple of ints, optional default: (2,)
        If shape does not have a length in the tuple allow_nd an error is raised.

    Returns
    -------
    new_shape: tuple
        The new shape after validation.
    """
    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: 'shape'")
    if len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]), )
        else:
            new_shape = tuple(operator.index(arg) for arg in shape_iter)
    else:
        new_shape = tuple(operator.index(arg) for arg in args)

    if current_shape is None:
        if len(new_shape) not in allow_nd:
            raise ValueError(f'shape must have length in {allow_nd}. Got {new_shape=}')
        if any(d < 0 for d in new_shape):
            raise ValueError("'shape' elements cannot be negative")
    else:
        # Check the current size only if needed
        current_size = prod(current_shape)

        # Check for negatives
        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if not negative_indexes:
            new_size = prod(new_shape)
            if new_size != current_size:
                raise ValueError(f'cannot reshape array of size {current_size}'
                                 f' into shape {new_shape}')
        elif len(negative_indexes) == 1:
            skip = negative_indexes[0]
            specified = prod(new_shape[:skip] + new_shape[skip+1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple('newshape' if x < 0 else x for x in new_shape)
                raise ValueError(f'cannot reshape array of size {current_size}'
                                 f' into shape {err_shape}')
            new_shape = new_shape[:skip] + (unspecified,) + new_shape[skip+1:]
        else:
            raise ValueError('can only specify one unknown dimension')

    if len(new_shape) not in allow_nd:
        raise ValueError(f'shape must have length in {allow_nd}. Got {new_shape=}')

    return new_shape


def broadcast_shapes(*shapes):
    """Check if shapes can be broadcast and return resulting shape

    This is similar to the NumPy ``broadcast_shapes`` function but
    does not check memory consequences of the resulting dense matrix.

    Parameters
    ----------
    *shapes : tuple of shape tuples
        The tuple of shapes to be considered for broadcasting.
        Shapes should be tuples of non-negative integers.

    Returns
    -------
    new_shape : tuple of integers
        The shape that results from broadcasting th input shapes.
    """
    if not shapes:
        return ()
    shapes = [shp if isinstance(shp, (tuple, list)) else (shp,) for shp in shapes]
    big_shp = max(shapes, key=len)
    out = list(big_shp)
    for shp in shapes:
        if shp is big_shp:
            continue
        for i, x in enumerate(shp, start=-len(shp)):
            if x != 1 and x != out[i]:
                if out[i] != 1:
                    raise ValueError("shapes cannot be broadcast to a single shape.")
                out[i] = x
    return (*out,)


def check_reshape_kwargs(kwargs):
    """Unpack keyword arguments for reshape function.

    This is useful because keyword arguments after star arguments are not
    allowed in Python 2, but star keyword arguments are. This function unpacks
    'order' and 'copy' from the star keyword arguments (with defaults) and
    throws an error for any remaining.
    """

    order = kwargs.pop('order', 'C')
    copy = kwargs.pop('copy', False)
    if kwargs:  # Some unused kwargs remain
        raise TypeError("reshape() got unexpected keywords arguments: "
                        f"{', '.join(kwargs.keys())}")
    return order, copy


def is_pydata_spmatrix(m) -> bool:
    """
    Check whether object is pydata/sparse matrix, avoiding importing the module.
    """
    base_cls = getattr(sys.modules.get('sparse'), 'SparseArray', None)
    return base_cls is not None and isinstance(m, base_cls)


def convert_pydata_sparse_to_scipy(
    arg: Any,
    target_format: None | Literal["csc", "csr"] = None,
    accept_fv: Any = None,
) -> Union[Any, "sp.spmatrix"]:
    """
    Convert a pydata/sparse array to scipy sparse matrix,
    pass through anything else.
    """
    if is_pydata_spmatrix(arg):
        # The `accept_fv` keyword is new in PyData Sparse 0.15.4 (May 2024),
        # remove the `except` once the minimum supported version is >=0.15.4
        try:
            arg = arg.to_scipy_sparse(accept_fv=accept_fv)
        except TypeError:
            arg = arg.to_scipy_sparse()
        if target_format is not None:
            arg = arg.asformat(target_format)
        elif arg.format not in ("csc", "csr"):
            arg = arg.tocsc()
    return arg


###############################################################################
# Wrappers for NumPy types that are deprecated

# Numpy versions of these functions raise deprecation warnings, the
# ones below do not.

def matrix(*args, **kwargs):
    return np.array(*args, **kwargs).view(np.matrix)


def asmatrix(data, dtype=None):
    if isinstance(data, np.matrix) and (dtype is None or data.dtype == dtype):
        return data
    return np.asarray(data, dtype=dtype).view(np.matrix)

###############################################################################


def _todata(s) -> np.ndarray:
    """Access nonzero values, possibly after summing duplicates.

    Parameters
    ----------
    s : sparse array
        Input sparse array.

    Returns
    -------
    data: ndarray
      Nonzero values of the array, with shape (s.nnz,)

    """
    if isinstance(s, sp._data._data_matrix):
        return s._deduped_data()

    if isinstance(s, sp.dok_array):
        return np.fromiter(s.values(), dtype=s.dtype, count=s.nnz)

    if isinstance(s, sp.lil_array):
        data = np.empty(s.nnz, dtype=s.dtype)
        sp._csparsetools.lil_flatten_to_array(s.data, data)
        return data

    return s.tocoo()._deduped_data()
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'bsr_matmat',
    'bsr_matrix',
    'bsr_matvec',
    'bsr_matvecs',
    'bsr_sort_indices',
    'bsr_tocsr',
    'bsr_transpose',
    'check_shape',
    'csr_matmat_maxnnz',
    'getdata',
    'getdtype',
    'isshape',
    'isspmatrix_bsr',
    'spmatrix',
    'to_native',
    'upcast',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="bsr",
                                   private_modules=["_bsr"], all=__all__,
                                   attribute=name)
<|endoftext|>""" A sparse matrix in COOrdinate or 'triplet' format"""

__docformat__ = "restructuredtext en"

__all__ = ['coo_array', 'coo_matrix', 'isspmatrix_coo']

import math
from warnings import warn

import numpy as np

from .._lib._util import copy_if_needed
from ._matrix import spmatrix
from ._sparsetools import (coo_tocsr, coo_todense, coo_todense_nd,
                           coo_matvec, coo_matvec_nd, coo_matmat_dense,
                           coo_matmat_dense_nd)
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast_char, to_native, isshape, getdtype,
                       getdata, downcast_intp_index, get_index_dtype,
                       check_shape, check_reshape_kwargs, isscalarlike, isdense)

import operator


class _coo_base(_data_matrix, _minmax_mixin):
    _format = 'coo'
    _allow_nd = range(1, 65)

    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        _data_matrix.__init__(self, arg1, maxprint=maxprint)
        if not copy:
            copy = copy_if_needed

        if isinstance(arg1, tuple):
            if isshape(arg1, allow_nd=self._allow_nd):
                self._shape = check_shape(arg1, allow_nd=self._allow_nd)
                idx_dtype = self._get_index_dtype(maxval=max(self._shape))
                data_dtype = getdtype(dtype, default=float)
                self.coords = tuple(np.array([], dtype=idx_dtype)
                                     for _ in range(len(self._shape)))
                self.data = np.array([], dtype=data_dtype)
                self.has_canonical_format = True
            else:
                try:
                    obj, coords = arg1
                except (TypeError, ValueError) as e:
                    raise TypeError('invalid input format') from e

                if shape is None:
                    if any(len(idx) == 0 for idx in coords):
                        raise ValueError('cannot infer dimensions from zero '
                                         'sized index arrays')
                    shape = tuple(operator.index(np.max(idx)) + 1
                                  for idx in coords)
                self._shape = check_shape(shape, allow_nd=self._allow_nd)
                idx_dtype = self._get_index_dtype(coords,
                                                  maxval=max(self.shape),
                                                  check_contents=True)
                self.coords = tuple(np.array(idx, copy=copy, dtype=idx_dtype)
                                     for idx in coords)
                self.data = getdata(obj, copy=copy, dtype=dtype)
                self.has_canonical_format = False
        else:
            if issparse(arg1):
                if arg1.format == self.format and copy:
                    self.coords = tuple(idx.copy() for idx in arg1.coords)
                    self.data = arg1.data.astype(getdtype(dtype, arg1))  # copy=True
                    self._shape = check_shape(arg1.shape, allow_nd=self._allow_nd)
                    self.has_canonical_format = arg1.has_canonical_format
                else:
                    coo = arg1.tocoo()
                    self.coords = tuple(coo.coords)
                    self.data = coo.data.astype(getdtype(dtype, coo), copy=False)
                    self._shape = check_shape(coo.shape, allow_nd=self._allow_nd)
                    self.has_canonical_format = False
            else:
                # dense argument
                M = np.asarray(arg1)
                if not isinstance(self, sparray):
                    M = np.atleast_2d(M)
                    if M.ndim != 2:
                        raise TypeError(f'expected 2D array or matrix, not {M.ndim}D')

                self._shape = check_shape(M.shape, allow_nd=self._allow_nd)
                if shape is not None:
                    if check_shape(shape, allow_nd=self._allow_nd) != self._shape:
                        message = f'inconsistent shapes: {shape} != {self._shape}'
                        raise ValueError(message)

                index_dtype = self._get_index_dtype(maxval=max(self._shape))
                coords = M.nonzero()
                self.coords = tuple(idx.astype(index_dtype, copy=False)
                                     for idx in coords)
                self.data = getdata(M[coords], copy=copy, dtype=dtype)
                self.has_canonical_format = True

        if len(self._shape) > 2:
            self.coords = tuple(idx.astype(np.int64, copy=False) for idx in self.coords)

        self._check()

    @property
    def row(self):
        if self.ndim > 1:
            return self.coords[-2]
        result = np.zeros_like(self.col)
        result.setflags(write=False)
        return result


    @row.setter
    def row(self, new_row):
        if self.ndim < 2:
            raise ValueError('cannot set row attribute of a 1-dimensional sparse array')
        new_row = np.asarray(new_row, dtype=self.coords[-2].dtype)
        self.coords = self.coords[:-2] + (new_row,) + self.coords[-1:]

    @property
    def col(self):
        return self.coords[-1]

    @col.setter
    def col(self, new_col):
        new_col = np.asarray(new_col, dtype=self.coords[-1].dtype)
        self.coords = self.coords[:-1] + (new_col,)

    def reshape(self, *args, **kwargs):
        shape = check_shape(args, self.shape, allow_nd=self._allow_nd)
        order, copy = check_reshape_kwargs(kwargs)

        # Return early if reshape is not required
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        # When reducing the number of dimensions, we need to be careful about
        # index overflow. This is why we can't simply call
        # `np.ravel_multi_index()` followed by `np.unravel_index()` here.
        flat_coords = _ravel_coords(self.coords, self.shape, order=order)
        if len(shape) == 2:
            if order == 'C':
                new_coords = divmod(flat_coords, shape[1])
            else:
                new_coords = divmod(flat_coords, shape[0])[::-1]
        else:
            new_coords = np.unravel_index(flat_coords, shape, order=order)

        idx_dtype = self._get_index_dtype(self.coords, maxval=max(shape))
        new_coords = tuple(np.asarray(co, dtype=idx_dtype) for co in new_coords)

        # Handle copy here rather than passing on to the constructor so that no
        # copy will be made of `new_coords` regardless.
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data

        return self.__class__((new_data, new_coords), shape=shape, copy=False)

    reshape.__doc__ = _spbase.reshape.__doc__

    def _getnnz(self, axis=None):
        if axis is None or (axis == 0 and self.ndim == 1):
            nnz = len(self.data)
            if any(len(idx) != nnz for idx in self.coords):
                raise ValueError('all index and data arrays must have the '
                                 'same length')

            if self.data.ndim != 1 or any(idx.ndim != 1 for idx in self.coords):
                raise ValueError('coordinates and data arrays must be 1-D')

            return int(nnz)

        if axis < 0:
            axis += self.ndim
        if axis >= self.ndim:
            raise ValueError('axis out of bounds')

        return np.bincount(downcast_intp_index(self.coords[1 - axis]),
                           minlength=self.shape[1 - axis])

    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def count_nonzero(self, axis=None):
        self.sum_duplicates()
        if axis is None:
            return np.count_nonzero(self.data)

        if axis < 0:
            axis += self.ndim
        if axis < 0 or axis >= self.ndim:
            raise ValueError('axis out of bounds')
        mask = self.data != 0
        coord = self.coords[1 - axis][mask]
        return np.bincount(downcast_intp_index(coord), minlength=self.shape[1 - axis])

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def _check(self):
        """ Checks data structure for consistency """
        if self.ndim != len(self.coords):
            raise ValueError('mismatching number of index arrays for shape; '
                             f'got {len(self.coords)}, expected {self.ndim}')

        # index arrays should have integer data types
        for i, idx in enumerate(self.coords):
            if idx.dtype.kind != 'i':
                warn(f'index array {i} has non-integer dtype ({idx.dtype.name})',
                     stacklevel=3)

        idx_dtype = self._get_index_dtype(self.coords, maxval=max(self.shape))
        self.coords = tuple(np.asarray(idx, dtype=idx_dtype)
                             for idx in self.coords)
        self.data = to_native(self.data)

        if self.nnz > 0:
            for i, idx in enumerate(self.coords):
                if idx.max() >= self.shape[i]:
                    raise ValueError(f'axis {i} index {idx.max()} exceeds '
                                     f'matrix dimension {self.shape[i]}')
                if idx.min() < 0:
                    raise ValueError(f'negative axis {i} index: {idx.min()}')

    def transpose(self, axes=None, copy=False):
        if axes is None:
            axes = range(self.ndim)[::-1]
        elif isinstance(self, sparray):
            if not hasattr(axes, "__len__") or len(axes) != self.ndim:
                raise ValueError("axes don't match matrix dimensions")
            if len(set(axes)) != self.ndim:
                raise ValueError("repeated axis in transpose")
        elif axes != (1, 0):
            raise ValueError("Sparse matrices do not support an 'axes' "
                             "parameter because swapping dimensions is the "
                             "only logical permutation.")

        permuted_shape = tuple(self._shape[i] for i in axes)
        permuted_coords = tuple(self.coords[i] for i in axes)
        return self.__class__((self.data, permuted_coords),
                              shape=permuted_shape, copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    def resize(self, *shape) -> None:
        shape = check_shape(shape, allow_nd=self._allow_nd)
        if self.ndim > 2:
            raise ValueError("only 1-D or 2-D input accepted")
        if len(shape) > 2:
            raise ValueError("shape argument must be 1-D or 2-D")
        # Check for added dimensions.
        if len(shape) > self.ndim:
            flat_coords = _ravel_coords(self.coords, self.shape)
            max_size = math.prod(shape)
            self.coords = np.unravel_index(flat_coords[:max_size], shape)
            self.data = self.data[:max_size]
            self._shape = shape
            return

        # Check for removed dimensions.
        if len(shape) < self.ndim:
            tmp_shape = (
                self._shape[:len(shape) - 1]  # Original shape without last axis
                + (-1,)  # Last axis is used to flatten the array
                + (1,) * (self.ndim - len(shape))  # Pad with ones
            )
            tmp = self.reshape(tmp_shape)
            self.coords = tmp.coords[:len(shape)]
            self._shape = tmp.shape[:len(shape)]

        # Handle truncation of existing dimensions.
        is_truncating = any(old > new for old, new in zip(self.shape, shape))
        if is_truncating:
            mask = np.logical_and.reduce([
                idx < size for idx, size in zip(self.coords, shape)
            ])
            if not mask.all():
                self.coords = tuple(idx[mask] for idx in self.coords)
                self.data = self.data[mask]

        self._shape = shape

    resize.__doc__ = _spbase.resize.__doc__

    def toarray(self, order=None, out=None):
        B = self._process_toarray_args(order, out)
        fortran = int(B.flags.f_contiguous)
        if not fortran and not B.flags.c_contiguous:
            raise ValueError("Output array must be C or F contiguous")
        # This handles both 0D and 1D cases correctly regardless of the
        # original shape.
        if self.ndim == 1:
            coo_todense_nd(np.array([1]), self.nnz, self.ndim,
                           self.coords[0], self.data, B.ravel('A'), fortran)
        elif self.ndim == 2:
            M, N = self.shape
            coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                        B.ravel('A'), fortran)
        else:
            if fortran:
                strides = np.append(1, np.cumprod(self.shape[:-1]))
            else:
                strides = np.append(np.cumprod(self.shape[1:][::-1])[::-1], 1)
            coords = np.concatenate(self.coords)
            coo_todense_nd(strides, self.nnz, self.ndim,
                           coords, self.data, B.ravel('A'), fortran)
        # Note: reshape() doesn't copy here, but does return a new array (view).
        return B.reshape(self.shape)

    toarray.__doc__ = _spbase.toarray.__doc__

    def tocsc(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Column format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.ndim != 2:
            raise ValueError(f'Cannot convert. CSC format must be 2D. Got {self.ndim}D')
        if self.nnz == 0:
            return self._csc_container(self.shape, dtype=self.dtype)
        else:
            from ._csc import csc_array
            indptr, indices, data, shape = self._coo_to_compressed(csc_array._swap)

            x = self._csc_container((data, indices, indptr), shape=shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def tocsr(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Row format

        Duplicate entries will be summed together.

        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsr()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        if self.ndim > 2:
            raise ValueError(f'Cannot convert. CSR must be 1D or 2D. Got {self.ndim}D')
        if self.nnz == 0:
            return self._csr_container(self.shape, dtype=self.dtype)
        else:
            from ._csr import csr_array
            arrays = self._coo_to_compressed(csr_array._swap, copy=copy)
            indptr, indices, data, shape = arrays

            x = self._csr_container((data, indices, indptr), shape=self.shape)
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x

    def _coo_to_compressed(self, swap, copy=False):
        """convert (shape, coords, data) to (indptr, indices, data, shape)"""
        M, N = swap(self._shape_as_2d)
        # convert idx_dtype intc to int32 for pythran.
        # tested in scipy/optimize/tests/test__numdiff.py::test_group_columns
        idx_dtype = self._get_index_dtype(self.coords, maxval=max(self.nnz, N))

        if self.ndim == 1:
            indices = self.coords[0].copy() if copy else self.coords[0]
            nnz = len(indices)
            indptr = np.array([0, nnz], dtype=idx_dtype)
            data = self.data.copy() if copy else self.data
            return indptr, indices, data, self.shape

        # ndim == 2
        major, minor = swap(self.coords)
        nnz = len(major)
        major = major.astype(idx_dtype, copy=False)
        minor = minor.astype(idx_dtype, copy=False)

        indptr = np.empty(M + 1, dtype=idx_dtype)
        indices = np.empty_like(minor, dtype=idx_dtype)
        data = np.empty_like(self.data, dtype=self.dtype)

        coo_tocsr(M, N, nnz, major, minor, self.data, indptr, indices, data)
        return indptr, indices, data, self.shape

    def tocoo(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self

    tocoo.__doc__ = _spbase.tocoo.__doc__

    def todia(self, copy=False):
        if self.ndim != 2:
            raise ValueError(f'Cannot convert. DIA format must be 2D. Got {self.ndim}D')
        self.sum_duplicates()
        ks = self.col - self.row  # the diagonal for each nonzero
        diags, diag_idx = np.unique(ks, return_inverse=True)

        if len(diags) > 100:
            # probably undesired, should todia() have a maxdiags parameter?
            warn(f"Constructing a DIA matrix with {len(diags)} diagonals "
                 "is inefficient",
                 SparseEfficiencyWarning, stacklevel=2)

        #initialize and fill in data array
        if self.data.size == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
        else:
            data = np.zeros((len(diags), self.col.max()+1), dtype=self.dtype)
            data[diag_idx, self.col] = self.data

        return self._dia_container((data, diags), shape=self.shape)

    todia.__doc__ = _spbase.todia.__doc__

    def todok(self, copy=False):
        if self.ndim > 2:
            raise ValueError(f'Cannot convert. DOK must be 1D or 2D. Got {self.ndim}D')
        self.sum_duplicates()
        dok = self._dok_container(self.shape, dtype=self.dtype)
        # ensure that 1d coordinates are not tuples
        if self.ndim == 1:
            coords = self.coords[0]
        else:
            coords = zip(*self.coords)

        dok._dict = dict(zip(coords, self.data))
        return dok

    todok.__doc__ = _spbase.todok.__doc__

    def diagonal(self, k=0):
        if self.ndim != 2:
            raise ValueError("diagonal requires two dimensions")
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                        dtype=self.dtype)
        diag_mask = (self.row + k) == self.col

        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            inds = tuple(idx[diag_mask] for idx in self.coords)
            (row, _), data = self._sum_duplicates(inds, self.data[diag_mask])
        diag[row + min(k, 0)] = data

        return diag

    diagonal.__doc__ = _data_matrix.diagonal.__doc__

    def _setdiag(self, values, k):
        if self.ndim != 2:
            raise ValueError("setting a diagonal requires two dimensions")
        M, N = self.shape
        if values.ndim and not len(values):
            return
        idx_dtype = self.row.dtype

        # Determine which triples to keep and where to put the new ones.
        full_keep = self.col - self.row != k
        if k < 0:
            max_index = min(M+k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.col >= max_index)
            new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = np.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N-k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.row >= max_index)
            new_row = np.arange(max_index, dtype=idx_dtype)
            new_col = np.arange(k, k + max_index, dtype=idx_dtype)

        # Define the array of data consisting of the entries to be added.
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = np.empty(max_index, dtype=self.dtype)
            new_data[:] = values

        # Update the internal structure.
        self.coords = (np.concatenate((self.row[keep], new_row)),
                       np.concatenate((self.col[keep], new_col)))
        self.data = np.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    # needed by _data_matrix
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data. By default the index arrays are copied.
        """
        if copy:
            coords = tuple(idx.copy() for idx in self.coords)
        else:
            coords = self.coords
        return self.__class__((data, coords), shape=self.shape, dtype=data.dtype)

    def sum_duplicates(self) -> None:
        """Eliminate duplicate entries by adding them together

        This is an *in place* operation
        """
        if self.has_canonical_format:
            return
        summed = self._sum_duplicates(self.coords, self.data)
        self.coords, self.data = summed
        self.has_canonical_format = True

    def _sum_duplicates(self, coords, data):
        # Assumes coords not in canonical format.
        if len(data) == 0:
            return coords, data
        # Sort coords w.r.t. rows, then cols. This corresponds to C-order,
        # which we rely on for argmin/argmax to return the first index in the
        # same way that numpy does (in the case of ties).
        order = np.lexsort(coords[::-1])
        coords = tuple(idx[order] for idx in coords)
        data = data[order]
        unique_mask = np.logical_or.reduce([
            idx[1:] != idx[:-1] for idx in coords
        ])
        unique_mask = np.append(True, unique_mask)
        coords = tuple(idx[unique_mask] for idx in coords)
        unique_inds, = np.nonzero(unique_mask)
        data = np.add.reduceat(data, downcast_intp_index(unique_inds), dtype=self.dtype)
        return coords, data

    def eliminate_zeros(self):
        """Remove zero entries from the array/matrix

        This is an *in place* operation
        """
        mask = self.data != 0
        self.data = self.data[mask]
        self.coords = tuple(idx[mask] for idx in self.coords)

    #######################
    # Arithmetic handlers #
    #######################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        result = np.array(other, dtype=dtype, copy=True)
        fortran = int(result.flags.f_contiguous)
        if self.ndim == 1:
            coo_todense_nd(np.array([1]), self.nnz, self.ndim,
                           self.coords[0], self.data, result.ravel('A'), fortran)
        elif self.ndim == 2:
            M, N = self._shape_as_2d
            coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                        result.ravel('A'), fortran)
        else:
            if fortran:
                strides = np.append(1, np.cumprod(self.shape[:-1]))
            else:
                strides = np.append(np.cumprod(self.shape[1:][::-1])[::-1], 1)
            coords = np.concatenate(self.coords)
            coo_todense_nd(strides, self.nnz, self.ndim,
                           coords, self.data, result.ravel('A'), fortran)
        return self._container(result, copy=False)


    def _add_sparse(self, other):
        if self.ndim < 3:
            return self.tocsr()._add_sparse(other)

        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        other = self.__class__(other)
        new_data = np.concatenate((self.data, other.data))
        new_coords = tuple(np.concatenate((self.coords, other.coords), axis=1))
        A = self.__class__((new_data, new_coords), shape=self.shape)
        return A


    def _sub_sparse(self, other):
        if self.ndim < 3:
            return self.tocsr()._sub_sparse(other)

        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        other = self.__class__(other)
        new_data = np.concatenate((self.data, -other.data))
        new_coords = tuple(np.concatenate((self.coords, other.coords), axis=1))
        A = coo_array((new_data, new_coords), shape=self.shape)
        return A


    def _matmul_vector(self, other):
        if self.ndim > 2:
            result = np.zeros(math.prod(self.shape[:-1]),
                              dtype=upcast_char(self.dtype.char, other.dtype.char))
            shape = np.array(self.shape)
            strides = np.append(np.cumprod(shape[:-1][::-1])[::-1][1:], 1)
            coords = np.concatenate(self.coords)
            coo_matvec_nd(self.nnz, len(self.shape), strides, coords, self.data,
                          other, result)

            result = result.reshape(self.shape[:-1])
            return result

        # self.ndim <= 2
        result_shape = self.shape[0] if self.ndim > 1 else 1
        result = np.zeros(result_shape,
                          dtype=upcast_char(self.dtype.char, other.dtype.char))
        if self.ndim == 2:
            col = self.col
            row = self.row
        elif self.ndim == 1:
            col = self.coords[0]
            row = np.zeros_like(col)
        else:
            raise NotImplementedError(
                f"coo_matvec not implemented for ndim={self.ndim}")

        coo_matvec(self.nnz, row, col, self.data, other, result)
        # Array semantics return a scalar here, not a single-element array.
        if isinstance(self, sparray) and result_shape == 1:
            return result[0]
        return result


    def _rmatmul_dispatch(self, other):
        if isscalarlike(other):
            return self._mul_scalar(other)
        else:
            # Don't use asarray unless we have to
            try:
                o_ndim = other.ndim
            except AttributeError:
                other = np.asarray(other)
                o_ndim = other.ndim
            perm = tuple(range(o_ndim)[:-2]) + tuple(range(o_ndim)[-2:][::-1])
            tr = other.transpose(perm)

            s_ndim = self.ndim
            perm = tuple(range(s_ndim)[:-2]) + tuple(range(s_ndim)[-2:][::-1])
            ret = self.transpose(perm)._matmul_dispatch(tr)
            if ret is NotImplemented:
                return NotImplemented

            if s_ndim == 1 or o_ndim == 1:
                perm = range(ret.ndim)
            else:
                perm = tuple(range(ret.ndim)[:-2]) + tuple(range(ret.ndim)[-2:][::-1])
            return ret.transpose(perm)


    def _matmul_dispatch(self, other):
        if isscalarlike(other):
            return self.multiply(other)

        if not (issparse(other) or isdense(other)):
            # If it's a list or whatever, treat it like an array
            other_a = np.asanyarray(other)

            if other_a.ndim == 0 and other_a.dtype == np.object_:
                # Not interpretable as an array; return NotImplemented so that
                # other's __rmatmul__ can kick in if that's implemented.
                return NotImplemented

            try:
                other.shape
            except AttributeError:
                other = other_a

        if self.ndim < 3 and other.ndim < 3:
            return _spbase._matmul_dispatch(self, other)

        N = self.shape[-1]
        err_prefix = "matmul: dimension mismatch with signature"
        if other.__class__ is np.ndarray:
            if other.shape == (N,):
                return self._matmul_vector(other)
            if other.shape == (N, 1):
                result = self._matmul_vector(other.ravel())
                return result.reshape(*self.shape[:-1], 1)
            if other.ndim == 1:
                msg = f"{err_prefix} (n,k={N}),(k={other.shape[0]},)->(n,)"
                raise ValueError(msg)
            if other.shape[-2] == N:
                # check for batch dimensions compatibility
                batch_shape_A = self.shape[:-2]
                batch_shape_B = other.shape[:-2]
                if batch_shape_A != batch_shape_B:
                    try:
                        # This will raise an error if the shapes are not broadcastable
                        np.broadcast_shapes(batch_shape_A, batch_shape_B)
                    except ValueError:
                        raise ValueError("Batch dimensions are not broadcastable")

                return self._matmul_multivector(other)
            else:
                raise ValueError(
                    f"{err_prefix} (n,..,k={N}),(k={other.shape[-2]},..,m)->(n,..,m)"
                )


        if isscalarlike(other):
            # scalar value
            return self._mul_scalar(other)

        if issparse(other):
            self_is_1d = self.ndim == 1
            other_is_1d = other.ndim == 1

            # reshape to 2-D if self or other is 1-D
            if self_is_1d:
                self = self.reshape(self._shape_as_2d) # prepend 1 to shape

            if other_is_1d:
                other = other.reshape((other.shape[0], 1)) # append 1 to shape

            # Check if the inner dimensions match for matrix multiplication
            if N != other.shape[-2]:
                raise ValueError(
                    f"{err_prefix} (n,..,k={N}),(k={other.shape[-2]},..,m)->(n,..,m)"
                )

            # If A or B has more than 2 dimensions, check for
            # batch dimensions compatibility
            if self.ndim > 2 or other.ndim > 2:
                batch_shape_A = self.shape[:-2]
                batch_shape_B = other.shape[:-2]
                if batch_shape_A != batch_shape_B:
                    try:
                        # This will raise an error if the shapes are not broadcastable
                        np.broadcast_shapes(batch_shape_A, batch_shape_B)
                    except ValueError:
                        raise ValueError("Batch dimensions are not broadcastable")

            result = self._matmul_sparse(other)

            # reshape back if a or b were originally 1-D
            if self_is_1d:
                # if self was originally 1-D, reshape result accordingly
                result = result.reshape(tuple(result.shape[:-2]) +
                                        tuple(result.shape[-1:]))
            if other_is_1d:
                result = result.reshape(result.shape[:-1])
            return result


    def _matmul_multivector(self, other):
        result_dtype = upcast_char(self.dtype.char, other.dtype.char)
        if self.ndim >= 3 or other.ndim >= 3:
            # if self has shape (N,), reshape to (1,N)
            if self.ndim == 1:
                result = self.reshape(1, self.shape[0])._matmul_multivector(other)
                return result.reshape(tuple(other.shape[:-2]) + tuple(other.shape[-1:]))

            broadcast_shape = np.broadcast_shapes(self.shape[:-2], other.shape[:-2])
            self_shape = broadcast_shape + self.shape[-2:]
            other_shape = broadcast_shape + other.shape[-2:]

            self = self._broadcast_to(self_shape)
            other = np.broadcast_to(other, other_shape)
            result_shape = broadcast_shape + self.shape[-2:-1] + other.shape[-1:]
            result = np.zeros(result_shape, dtype=result_dtype)
            coo_matmat_dense_nd(self.nnz, len(self.shape), other.shape[-1],
                                np.array(other_shape), np.array(result_shape),
                                np.concatenate(self.coords),
                                self.data, other.ravel('C'), result)
            return result

        if self.ndim == 2:
            result_shape = (self.shape[0], other.shape[1])
            col = self.col
            row = self.row
        elif self.ndim == 1:
            result_shape = (other.shape[1],)
            col = self.coords[0]
            row = np.zeros_like(col)
        result = np.zeros(result_shape, dtype=result_dtype)
        coo_matmat_dense(self.nnz, other.shape[-1], row, col,
                         self.data, other.ravel('C'), result)
        return result.view(type=type(other))


    def dot(self, other):
        """Return the dot product of two arrays.

        Strictly speaking a dot product involves two vectors.
        But in the sense that an array with ndim >= 1 is a collection
        of vectors, the function computes the collection of dot products
        between each vector in the first array with each vector in the
        second array. The axis upon which the sum of products is performed
        is the last axis of the first array and the second to last axis of
        the second array. If the second array is 1-D, the last axis is used.

        Thus, if both arrays are 1-D, the inner product is returned.
        If both are 2-D, we have matrix multiplication. If `other` is 1-D,
        the sum product is taken along the last axis of each array. If
        `other` is N-D for N>=2, the sum product is over the last axis of
        the first array and the second-to-last axis of the second array.

        Parameters
        ----------
        other : array_like (dense or sparse)
            Second array

        Returns
        -------
        output : array (sparse or dense)
            The dot product of this array with `other`.
            It will be dense/sparse if `other` is dense/sparse.

        Examples
        --------

        >>> import numpy as np
        >>> from scipy.sparse import coo_array
        >>> A = coo_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> v = np.array([1, 0, -1])
        >>> A.dot(v)
        array([ 1, -3, -1], dtype=int64)

        For 2-D arrays it is the matrix product:

        >>> A = coo_array([[1, 0], [0, 1]])
        >>> B = coo_array([[4, 1], [2, 2]])
        >>> A.dot(B).toarray()
        array([[4, 1],
               [2, 2]])

        For 3-D arrays the shape extends unused axes by other unused axes.

        >>> A = coo_array(np.arange(3*4*5*6)).reshape((3,4,5,6))
        >>> B = coo_array(np.arange(3*4*5*6)).reshape((5,4,6,3))
        >>> A.dot(B).shape
        (3, 4, 5, 5, 4, 3)
        """
        if not (issparse(other) or isdense(other) or isscalarlike(other)):
            # If it's a list or whatever, treat it like an array
            o_array = np.asanyarray(other)

            if o_array.ndim == 0 and o_array.dtype == np.object_:
                raise TypeError(f"dot argument not supported type: '{type(other)}'")
            try:
                other.shape
            except AttributeError:
                other = o_array

        if self.ndim < 3 and (np.isscalar(other) or other.ndim<3):
            return _spbase.dot(self, other)
        # Handle scalar multiplication
        if np.isscalar(other):
            return self * other
        if isdense(other):
            return self._dense_dot(other)
        elif other.format != "coo":
            raise TypeError("input must be a COO matrix/array")
        elif self.ndim == 1 and other.ndim == 1:
            # Handle inner product of vectors (1-D arrays)
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"shapes {self.shape} and {other.shape}"
                                 " are not aligned for inner product")
            return self @ other
        elif self.ndim == 2 and other.ndim == 2:
            # Handle matrix multiplication (2-D arrays)
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"shapes {self.shape} and {other.shape}"
                                 " are not aligned for matmul")
            return self @ other
        else:
            return self._sparse_dot(other)


    def _sparse_dot(self, other):
        self_is_1d = self.ndim == 1
        other_is_1d = other.ndim == 1

        # reshape to 2-D if self or other is 1-D
        if self_is_1d:
            self = self.reshape(self._shape_as_2d)  # prepend 1 to shape
        if other_is_1d:
            other = other.reshape((other.shape[0], 1))  # append 1 to shape

        if self.shape[-1] != other.shape[-2]:
                raise ValueError(f"shapes {self.shape} and {other.shape}"
                                 " are not aligned for n-D dot")

        # Prepare the tensors for dot operation
        # Ravel non-reduced axes coordinates
        self_raveled_coords = _ravel_non_reduced_axes(self.coords,
                                                      self.shape, [self.ndim-1])
        other_raveled_coords = _ravel_non_reduced_axes(other.coords,
                                                       other.shape, [other.ndim-2])

        # Get the shape of the non-reduced axes
        self_nonreduced_shape = self.shape[:-1]
        other_nonreduced_shape = other.shape[:-2] + other.shape[-1:]

        # Create 2D coords arrays
        ravel_coords_shape_self = (math.prod(self_nonreduced_shape), self.shape[-1])
        ravel_coords_shape_other = (other.shape[-2], math.prod(other_nonreduced_shape))

        self_2d_coords = (self_raveled_coords, self.coords[-1])
        other_2d_coords = (other.coords[-2], other_raveled_coords)

        self_2d = coo_array((self.data, self_2d_coords), ravel_coords_shape_self)
        other_2d = coo_array((other.data, other_2d_coords), ravel_coords_shape_other)

        prod = (self_2d @ other_2d).tocoo() # routes via 2-D CSR

        # Combine the shapes of the non-reduced axes
        combined_shape = self_nonreduced_shape + other_nonreduced_shape

        # Unravel the 2D coordinates to get multi-dimensional coordinates
        shapes = (self_nonreduced_shape, other_nonreduced_shape)
        prod_coords = []
        for c, s in zip(prod.coords, shapes):
            prod_coords.extend(np.unravel_index(c, s))

        prod_arr = coo_array((prod.data, prod_coords), combined_shape)

        # reshape back if a or b were originally 1-D
        # TODO: Move this logic before computation of prod_coords for efficiency
        if self_is_1d:
            prod_arr = prod_arr.reshape(combined_shape[1:])
        if other_is_1d:
            prod_arr = prod_arr.reshape(combined_shape[:-1])

        return prod_arr

    def _dense_dot(self, other):
        self_is_1d = self.ndim == 1
        other_is_1d = other.ndim == 1

        # reshape to 2-D if self or other is 1-D
        if self_is_1d:
            self = self.reshape(self._shape_as_2d)  # prepend 1 to shape
        if other_is_1d:
            other = other.reshape((other.shape[0], 1))  # append 1 to shape

        if self.shape[-1] != other.shape[-2]:
                raise ValueError(f"shapes {self.shape} and {other.shape}"
                                 " are not aligned for n-D dot")

        new_shape_self = (
            self.shape[:-1] + (1,) * (len(other.shape) - 1) + self.shape[-1:]
        )
        new_shape_other = (1,) * (len(self.shape) - 1) + other.shape

        result_shape = self.shape[:-1] + other.shape[:-2] + other.shape[-1:]
        result = self.reshape(new_shape_self) @ other.reshape(new_shape_other)
        prod_arr = result.reshape(result_shape)

        # reshape back if a or b were originally 1-D
        if self_is_1d:
            prod_arr = prod_arr.reshape(result_shape[1:])
        if other_is_1d:
            prod_arr = prod_arr.reshape(result_shape[:-1])

        return prod_arr

    def tensordot(self, other, axes=2):
        """Return the tensordot product with another array along the given axes.

        The tensordot differs from dot and matmul in that any axis can be
        chosen for each of the first and second array and the sum of the
        products is computed just like for matrix multiplication, only not
        just for the rows of the first times the columns of the second. It
        takes the dot product of the collection of vectors along the specified
        axes.  Here we can even take the sum of the products along two or even
        more axes if desired. So, tensordot is a dot product computation
        applied to arrays of any dimension >= 1. It is like matmul but over
        arbitrary axes for each matrix.

        Given two tensors, `a` and `b`, and the desired axes specified as a
        2-tuple/list/array containing two sequences of axis numbers,
        ``(a_axes, b_axes)``, sum the products of `a`'s and `b`'s elements
        (components) over the axes specified by ``a_axes`` and ``b_axes``.
        The `axes` input can be a single non-negative integer, ``N``;
        if it is, then the last ``N`` dimensions of `a` and the first
        ``N`` dimensions of `b` are summed over.

        Parameters
        ----------
        a, b : array_like
            Tensors to "dot".

        axes : int or (2,) array_like
            * integer_like
              If an int N, sum over the last N axes of `a` and the first N axes
              of `b` in order. The sizes of the corresponding axes must match.
            * (2,) array_like
              A 2-tuple of sequences of axes to be summed over, the first applying
              to `a`, the second to `b`. The sequences must be the same length.
              The shape of the corresponding axes must match between `a` and `b`.

        Returns
        -------
        output : coo_array
            The tensor dot product of this array with `other`.
            It will be dense/sparse if `other` is dense/sparse.

        See Also
        --------
        dot

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.sparse
        >>> A = scipy.sparse.coo_array([[[2, 3], [0, 0]], [[0, 1], [0, 5]]])
        >>> A.shape
        (2, 2, 2)

        Integer axes N are shorthand for (range(-N, 0), range(0, N)):

        >>> A.tensordot(A, axes=1).toarray()
        array([[[[ 4,  9],
                 [ 0, 15]],
        <BLANKLINE>
                [[ 0,  0],
                 [ 0,  0]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[ 0,  1],
                 [ 0,  5]],
        <BLANKLINE>
                [[ 0,  5],
                 [ 0, 25]]]])
        >>> A.tensordot(A, axes=2).toarray()
        array([[ 4,  6],
               [ 0, 25]])
        >>> A.tensordot(A, axes=3)
        array(39)

        Using tuple for axes:

        >>> a = scipy.sparse.coo_array(np.arange(60).reshape(3,4,5))
        >>> b = np.arange(24).reshape(4,3,2)
        >>> c = a.tensordot(b, axes=([1,0],[0,1]))
        >>> c.shape
        (5, 2)
        >>> c
        array([[4400, 4730],
               [4532, 4874],
               [4664, 5018],
               [4796, 5162],
               [4928, 5306]])

        """
        if not isdense(other) and not issparse(other):
            # If it's a list or whatever, treat it like an array
            other_array = np.asanyarray(other)

            if other_array.ndim == 0 and other_array.dtype == np.object_:
                raise TypeError(f"tensordot arg not supported type: '{type(other)}'")
            try:
                other.shape
            except AttributeError:
                other = other_array

        axes_self, axes_other = _process_axes(self.ndim, other.ndim, axes)

        # Check for shape compatibility along specified axes
        if any(self.shape[ax] != other.shape[bx]
               for ax, bx in zip(axes_self, axes_other)):
            raise ValueError("sizes of the corresponding axes must match")

        if isdense(other):
            return self._dense_tensordot(other, axes_self, axes_other)
        else:
            return self._sparse_tensordot(other, axes_self, axes_other)


    def _sparse_tensordot(self, other, axes_self, axes_other):
        ndim_self = len(self.shape)
        ndim_other = len(other.shape)

        # Prepare the tensors for tensordot operation
        # Ravel non-reduced axes coordinates
        self_non_red_coords = _ravel_non_reduced_axes(self.coords, self.shape,
                                                      axes_self)
        self_reduced_coords = np.ravel_multi_index(
            [self.coords[ax] for ax in axes_self], [self.shape[ax] for ax in axes_self])
        other_non_red_coords = _ravel_non_reduced_axes(other.coords, other.shape,
                                                       axes_other)
        other_reduced_coords = np.ravel_multi_index(
            [other.coords[a] for a in axes_other], [other.shape[a] for a in axes_other]
        )
        # Get the shape of the non-reduced axes
        self_nonreduced_shape = tuple(self.shape[ax] for ax in range(ndim_self)
                              if ax not in axes_self)
        other_nonreduced_shape = tuple(other.shape[ax] for ax in range(ndim_other)
                               if ax not in axes_other)

        # Create 2D coords arrays
        ravel_coords_shape_self = (math.prod(self_nonreduced_shape),
                                math.prod([self.shape[ax] for ax in axes_self]))
        ravel_coords_shape_other = (math.prod([other.shape[ax] for ax in axes_other]),
                                    math.prod(other_nonreduced_shape))

        self_2d_coords = (self_non_red_coords, self_reduced_coords)
        other_2d_coords = (other_reduced_coords, other_non_red_coords)

        self_2d = coo_array((self.data, self_2d_coords), ravel_coords_shape_self)
        other_2d = coo_array((other.data, other_2d_coords), ravel_coords_shape_other)

        # Perform matrix multiplication (routed via 2-D CSR)
        prod = (self_2d @ other_2d).tocoo()

        # Combine the shapes of the non-contracted axes
        combined_shape = self_nonreduced_shape + other_nonreduced_shape

        # Unravel the 2D coordinates to get multi-dimensional coordinates
        coords = []
        for c, s in zip(prod.coords, (self_nonreduced_shape, other_nonreduced_shape)):
            if s:
                coords.extend(np.unravel_index(c, s))

        if coords == []:  # if result is scalar
            return sum(prod.data)

        # Construct the resulting COO array with combined coordinates and shape
        return coo_array((prod.data, coords), shape=combined_shape)


    def _dense_tensordot(self, other, axes_self, axes_other):
        ndim_self = len(self.shape)
        ndim_other = len(other.shape)

        non_reduced_axes_self = [ax for ax in range(ndim_self) if ax not in axes_self]
        reduced_shape_self = [self.shape[s] for s in axes_self]
        non_reduced_shape_self = [self.shape[s] for s in non_reduced_axes_self]

        non_reduced_axes_other = [ax for ax in range(ndim_other)
                                  if ax not in axes_other]
        reduced_shape_other = [other.shape[s] for s in axes_other]
        non_reduced_shape_other = [other.shape[s] for s in non_reduced_axes_other]

        permute_self = non_reduced_axes_self + axes_self
        permute_other = (
            non_reduced_axes_other[:-1] + axes_other + non_reduced_axes_other[-1:]
        )
        self = self.transpose(permute_self)
        other = np.transpose(other, permute_other)

        reshape_self = (*non_reduced_shape_self, math.prod(reduced_shape_self))
        reshape_other = (*non_reduced_shape_other[:-1], math.prod(reduced_shape_other),
                        *non_reduced_shape_other[-1:])

        return self.reshape(reshape_self).dot(other.reshape(reshape_other))


    def _matmul_sparse(self, other):
        """
        Perform sparse-sparse matrix multiplication for two n-D COO arrays.
        The method converts input n-D arrays to 2-D block array format,
        uses csr_matmat to multiply them, and then converts the
        result back to n-D COO array.

        Parameters:
        self (COO): The first n-D sparse array in COO format.
        other (COO): The second n-D sparse array in COO format.

        Returns:
        prod (COO): The resulting n-D sparse array after multiplication.
        """
        if self.ndim < 3 and other.ndim < 3:
            return _spbase._matmul_sparse(self, other)

        # Get the shapes of self and other
        self_shape = self.shape
        other_shape = other.shape

        # Determine the new shape to broadcast self and other
        broadcast_shape = np.broadcast_shapes(self_shape[:-2], other_shape[:-2])
        self_new_shape = tuple(broadcast_shape) + self_shape[-2:]
        other_new_shape = tuple(broadcast_shape) + other_shape[-2:]

        self_broadcasted = self._broadcast_to(self_new_shape)
        other_broadcasted = other._broadcast_to(other_new_shape)

        # Convert n-D COO arrays to 2-D block diagonal arrays
        self_block_diag = _block_diag(self_broadcasted)
        other_block_diag = _block_diag(other_broadcasted)

        # Use csr_matmat to perform sparse matrix multiplication
        prod_block_diag = (self_block_diag @ other_block_diag).tocoo()

        # Convert the 2-D block diagonal array back to n-D
        return _extract_block_diag(
            prod_block_diag,
            shape=(*broadcast_shape, self.shape[-2], other.shape[-1]),
        )


    def _broadcast_to(self, new_shape, copy=False):
        if self.shape == new_shape:
            return self.copy() if copy else self

        old_shape = self.shape

        # Check if the new shape is compatible for broadcasting
        if len(new_shape) < len(old_shape):
            raise ValueError("New shape must have at least as many dimensions"
                             " as the current shape")

        # Add leading ones to shape to ensure same length as `new_shape`
        shape = (1,) * (len(new_shape) - len(old_shape)) + tuple(old_shape)

        # Ensure the old shape can be broadcast to the new shape
        if any((o != 1 and o != n) for o, n in zip(shape, new_shape)):
            raise ValueError(f"current shape {old_shape} cannot be "
                             "broadcast to new shape {new_shape}")

        # Reshape the COO array to match the new dimensions
        self = self.reshape(shape)

        idx_dtype = get_index_dtype(self.coords, maxval=max(new_shape))
        coords = self.coords
        new_data = self.data
        new_coords = coords[-1:]  # Copy last coordinate to start
        cum_repeat = 1 # Cumulative repeat factor for broadcasting

        if shape[-1] != new_shape[-1]: # broadcasting the n-th (col) dimension
            repeat_count = new_shape[-1]
            cum_repeat *= repeat_count
            new_data = np.tile(new_data, repeat_count)
            new_dim = np.repeat(np.arange(0, repeat_count, dtype=idx_dtype), self.nnz)
            new_coords = (new_dim,)

        for i in range(-2, -(len(shape)+1), -1):
            if shape[i] != new_shape[i]:
                repeat_count = new_shape[i] # number of times to repeat data, coords
                cum_repeat *= repeat_count # update cumulative repeat factor
                nnz = len(new_data) # Number of non-zero elements so far

                # Tile data and coordinates to match the new repeat count
                new_data = np.tile(new_data, repeat_count)
                new_coords = tuple(np.tile(new_coords[i+1:], repeat_count))

                # Create new dimensions and stack them
                new_dim = np.repeat(np.arange(0, repeat_count, dtype=idx_dtype), nnz)
                new_coords = (new_dim,) + new_coords
            else:
                # If no broadcasting needed, tile the coordinates
                new_dim = np.tile(coords[i], cum_repeat)
                new_coords = (new_dim,) + new_coords

        return coo_array((new_data, new_coords), new_shape)


def _block_diag(self):
    """
    Converts an N-D COO array into a 2-D COO array in block diagonal form.

    Parameters:
    self (coo_array): An N-Dimensional COO sparse array.

    Returns:
    coo_array: A 2-Dimensional COO sparse array in block diagonal form.
    """
    if self.ndim<2:
        raise ValueError("array must have atleast dim=2")
    num_blocks = math.prod(self.shape[:-2])
    n_col = self.shape[-1]
    n_row = self.shape[-2]
    res_arr = self.reshape((num_blocks, n_row, n_col))
    new_coords = (
        res_arr.coords[1] + res_arr.coords[0] * res_arr.shape[1],
        res_arr.coords[2] + res_arr.coords[0] * res_arr.shape[2],
    )

    new_shape = (num_blocks * n_row, num_blocks * n_col)
    return coo_array((self.data, tuple(new_coords)), shape=new_shape)


def _extract_block_diag(self, shape):
    n_row, n_col = shape[-2], shape[-1]

    # Extract data and coordinates from the block diagonal COO array
    data = self.data
    row, col = self.row, self.col

    # Initialize new coordinates array
    new_coords = np.empty((len(shape), self.nnz), dtype=int)

    # Calculate within-block indices
    new_coords[-2] = row % n_row
    new_coords[-1] = col % n_col

    # Calculate coordinates for higher dimensions
    temp_block_idx = row // n_row
    for i in range(len(shape) - 3, -1, -1):
        size = shape[i]
        new_coords[i] = temp_block_idx % size
        temp_block_idx = temp_block_idx // size

    # Create the new COO array with the original n-D shape
    return coo_array((data, tuple(new_coords)), shape=shape)


def _process_axes(ndim_a, ndim_b, axes):
    if isinstance(axes, int):
        if axes < 1 or axes > min(ndim_a, ndim_b):
            raise ValueError("axes integer is out of bounds for input arrays")
        axes_a = list(range(ndim_a - axes, ndim_a))
        axes_b = list(range(axes))
    elif isinstance(axes, (tuple, list)):
        if len(axes) != 2:
            raise ValueError("axes must be a tuple/list of length 2")
        axes_a, axes_b = axes
        if len(axes_a) != len(axes_b):
            raise ValueError("axes lists/tuples must be of the same length")
        if any(ax >= ndim_a or ax < -ndim_a for ax in axes_a) or \
           any(bx >= ndim_b or bx < -ndim_b for bx in axes_b):
            raise ValueError("axes indices are out of bounds for input arrays")
    else:
        raise TypeError("axes must be an integer or a tuple/list of integers")

    axes_a = [axis + ndim_a if axis < 0 else axis for axis in axes_a]
    axes_b = [axis + ndim_b if axis < 0 else axis for axis in axes_b]
    return axes_a, axes_b


def _ravel_non_reduced_axes(coords, shape, axes):
    ndim = len(shape)
    non_reduced_axes = [ax for ax in range(ndim) if ax not in axes]

    if not non_reduced_axes:
        # Return an array with one row
        return np.zeros_like(coords[0])

    # Extract the shape of the non-reduced axes
    non_reduced_shape = [shape[ax] for ax in non_reduced_axes]

    # Extract the coordinates of the non-reduced axes
    non_reduced_coords = tuple(coords[idx] for idx in non_reduced_axes)

    # Ravel the coordinates into 1D
    return np.ravel_multi_index(non_reduced_coords, non_reduced_shape)


def _ravel_coords(coords, shape, order='C'):
    """Like np.ravel_multi_index, but avoids some overflow issues."""
    if len(coords) == 1:
        return coords[0]
    # Handle overflow as in https://github.com/scipy/scipy/pull/9132
    if len(coords) == 2:
        nrows, ncols = shape
        row, col = coords
        if order == 'C':
            maxval = (ncols * max(0, nrows - 1) + max(0, ncols - 1))
            idx_dtype = get_index_dtype(maxval=maxval)
            return np.multiply(ncols, row, dtype=idx_dtype) + col
        elif order == 'F':
            maxval = (nrows * max(0, ncols - 1) + max(0, nrows - 1))
            idx_dtype = get_index_dtype(maxval=maxval)
            return np.multiply(nrows, col, dtype=idx_dtype) + row
        else:
            raise ValueError("'order' must be 'C' or 'F'")
    return np.ravel_multi_index(coords, shape, order=order)


def isspmatrix_coo(x):
    """Is `x` of coo_matrix type?

    Parameters
    ----------
    x
        object to check for being a coo matrix

    Returns
    -------
    bool
        True if `x` is a coo matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import coo_array, coo_matrix, csr_matrix, isspmatrix_coo
    >>> isspmatrix_coo(coo_matrix([[5]]))
    True
    >>> isspmatrix_coo(coo_array([[5]]))
    False
    >>> isspmatrix_coo(csr_matrix([[5]]))
    False
    """
    return isinstance(x, coo_matrix)


# This namespace class separates array from matrix with isinstance
class coo_array(_coo_base, sparray):
    """
    A sparse array in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_array(D)
            where D is an ndarray

        coo_array(S)
            with another sparse array or matrix S (equivalent to S.tocoo())

        coo_array(shape, [dtype])
            to construct an empty sparse array with shape `shape`
            dtype is optional, defaulting to dtype='d'.

        coo_array((data, coords), [shape])
            to construct from existing data and index arrays:
                1. data[:]       the entries of the sparse array, in any order
                2. coords[i][:]  the axis-i coordinates of the data entries

            Where ``A[coords] = data``, and coords is a tuple of index arrays.
            When shape is not specified, it is inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the sparse array
    shape : tuple of integers
        Shape of the sparse array
    ndim : int
        Number of dimensions of the sparse array
    nnz
    size
    data
        COO format data array of the sparse array
    coords
        COO format tuple of index arrays
    has_canonical_format : bool
        Whether the matrix has sorted coordinates and no duplicates
    format
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse arrays
        - Once a COO array has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Canonical format
        - Entries and coordinates sorted by row, then column.
        - There are no duplicate entries (i.e. duplicate (i,j) locations)
        - Data arrays MAY have explicit zeros.

    Examples
    --------

    >>> # Constructing an empty sparse array
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> coo_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a sparse array using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_array((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a sparse array with duplicate coordinates
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_array((data, (row, col)), shape=(4, 4))
    >>> # Duplicate coordinates are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """


class coo_matrix(spmatrix, _coo_base):
    """
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            where D is a 2-D ndarray

        coo_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix
    has_canonical_format : bool
        Whether the matrix has sorted indices and no duplicates
    format
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a COO matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Canonical format
        - Entries and coordinates sorted by row, then column.
        - There are no duplicate entries (i.e. duplicate (i,j) locations)
        - Data arrays MAY have explicit zeros.

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate coordinates
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate coordinates are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """

    def __setstate__(self, state):
        if 'coords' not in state:
            # For retro-compatibility with the previous attributes
            # storing nnz coordinates for 2D COO matrix.
            state['coords'] = (state.pop('row'), state.pop('col'))
        self.__dict__.update(state)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'IndexMixin',
    'check_shape',
    'dok_matrix',
    'getdtype',
    'isdense',
    'isintlike',
    'isscalarlike',
    'isshape',
    'isspmatrix_dok',
    'itertools',
    'spmatrix',
    'upcast',
    'upcast_scalar',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="dok",
                                   private_modules=["_dok"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'IndexMixin',
    'SparseEfficiencyWarning',
    'check_shape',
    'csr_column_index1',
    'csr_column_index2',
    'csr_row_index',
    'csr_row_slice',
    'csr_sample_offsets',
    'csr_sample_values',
    'csr_todense',
    'downcast_intp_index',
    'get_csr_submatrix',
    'get_sum_dtype',
    'getdtype',
    'is_pydata_spmatrix',
    'isdense',
    'isintlike',
    'isscalarlike',
    'isshape',
    'operator',
    'to_native',
    'upcast',
    'upcast_char',
    'warn',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="compressed",
                                   private_modules=["_compressed"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Indexing mixin for sparse array/matrix classes.
"""
import numpy as np
from ._sputils import isintlike
from ._base import sparray, issparse

INT_TYPES = (int, np.integer)


def _broadcast_arrays(a, b):
    """
    Same as np.broadcast_arrays(a, b) but old writeability rules.

    NumPy >= 1.17.0 transitions broadcast_arrays to return
    read-only arrays. Set writeability explicitly to avoid warnings.
    Retain the old writeability rules, as our Cython code assumes
    the old behavior.
    """
    x, y = np.broadcast_arrays(a, b)
    x.flags.writeable = a.flags.writeable
    y.flags.writeable = b.flags.writeable
    return x, y


class IndexMixin:
    """
    This class provides common dispatching and validation logic for indexing.
    """
    def __getitem__(self, key):
        index, new_shape = self._validate_indices(key)

        # 1D array
        if len(index) == 1:
            idx = index[0]
            if isinstance(idx, np.ndarray):
                if idx.shape == ():
                    idx = idx.item()
            if isinstance(idx, INT_TYPES):
                res = self._get_int(idx)
            elif isinstance(idx, slice):
                res = self._get_slice(idx)
            else:  # assume array idx
                res = self._get_array(idx)

            # package the result and return
            if not isinstance(self, sparray):
                return res
            # handle np.newaxis in idx when result would otherwise be a scalar
            if res.shape == () and new_shape != ():
                if len(new_shape) == 1:
                    return self.__class__([res], shape=new_shape, dtype=self.dtype)
                if len(new_shape) == 2:
                    return self.__class__([[res]], shape=new_shape, dtype=self.dtype)
            return res.reshape(new_shape)

        # 2D array
        row, col = index

        # Dispatch to specialized methods.
        if isinstance(row, INT_TYPES):
            if isinstance(col, INT_TYPES):
                res = self._get_intXint(row, col)
            elif isinstance(col, slice):
                res = self._get_intXslice(row, col)
            elif col.ndim == 1:
                res = self._get_intXarray(row, col)
            elif col.ndim == 2:
                res = self._get_intXarray(row, col)
            else:
                raise IndexError('index results in >2 dimensions')
        elif isinstance(row, slice):
            if isinstance(col, INT_TYPES):
                res = self._get_sliceXint(row, col)
            elif isinstance(col, slice):
                if row == slice(None) and row == col:
                    res = self.copy()
                else:
                    res = self._get_sliceXslice(row, col)
            elif col.ndim == 1:
                res = self._get_sliceXarray(row, col)
            else:
                raise IndexError('index results in >2 dimensions')
        else:
            if isinstance(col, INT_TYPES):
                res = self._get_arrayXint(row, col)
            elif isinstance(col, slice):
                res = self._get_arrayXslice(row, col)
            # arrayXarray preprocess
            elif (row.ndim == 2 and row.shape[1] == 1
                and (col.ndim == 1 or col.shape[0] == 1)):
                # outer indexing
                res = self._get_columnXarray(row[:, 0], col.ravel())
            else:
                # inner indexing
                row, col = _broadcast_arrays(row, col)
                if row.shape != col.shape:
                    raise IndexError('number of row and column indices differ')
                if row.size == 0:
                    res = self.__class__(np.atleast_2d(row).shape, dtype=self.dtype)
                else:
                    res = self._get_arrayXarray(row, col)

        # handle spmatrix (must be 2d, dont let 1d new_shape start reshape)
        if not isinstance(self, sparray):
            if new_shape == () or (len(new_shape) == 1 and res.ndim != 0):
                # res handles cases not inflated by None
                return res
            if len(new_shape) == 1:
                # shape inflated to 1D by None in index. Make 2D
                new_shape = (1,) + new_shape
            # reshape if needed (when None changes shape, e.g. A[1,:,None])
            return res if new_shape == res.shape else res.reshape(new_shape)

        # package the result and return
        if res.shape != new_shape:
            # handle formats that support indexing but not 1D (lil for now)
            if self.format == "lil" and len(new_shape) != 2:
                if res.shape == ():
                    return self._coo_container([res], shape = new_shape)
                return res.tocoo().reshape(new_shape)
            return res.reshape(new_shape)
        return res

    def __setitem__(self, key, x):
        index, _ = self._validate_indices(key)

        # 1D array
        if len(index) == 1:
            idx = index[0]

            if issparse(x):
                x = x.toarray()
            else:
                x = np.asarray(x, dtype=self.dtype)

            if isinstance(idx, INT_TYPES):
                if x.size != 1:
                    raise ValueError('Trying to assign a sequence to an item')
                self._set_int(idx, x.flat[0])
                return

            if isinstance(idx, slice):
                # check for simple case of slice that gives 1 item
                # Note: Python `range` does not use lots of memory
                idx_range = range(*idx.indices(self.shape[0]))
                N = len(idx_range)
                if N == 1 and x.size == 1:
                    self._set_int(idx_range[0], x.flat[0])
                    return
                idx = np.arange(*idx.indices(self.shape[0]))
                idx_shape = idx.shape
            else:
                idx_shape = idx.squeeze().shape
            # broadcast scalar to full 1d
            if x.squeeze().shape != idx_shape:
                x = np.broadcast_to(x, idx.shape)
            if x.size != 0:
                self._set_array(idx, x)
            return

        # 2D array
        row, col = index

        if isinstance(row, INT_TYPES) and isinstance(col, INT_TYPES):
            x = np.asarray(x, dtype=self.dtype)
            if x.size != 1:
                raise ValueError('Trying to assign a sequence to an item')
            self._set_intXint(row, col, x.flat[0])
            return

        if isinstance(row, slice):
            row = np.arange(*row.indices(self.shape[0]))[:, None]
        else:
            row = np.atleast_1d(row)

        if isinstance(col, slice):
            col = np.arange(*col.indices(self.shape[1]))[None, :]
            if row.ndim == 1:
                row = row[:, None]
        else:
            col = np.atleast_1d(col)

        i, j = _broadcast_arrays(row, col)
        if i.shape != j.shape:
            raise IndexError('number of row and column indices differ')

        if issparse(x):
            if 0 in x.shape:
                return
            if i.ndim == 1:
                # Inner indexing, so treat them like row vectors.
                i = i[None]
                j = j[None]
            x = x.tocoo(copy=False).reshape(x._shape_as_2d, copy=True)
            broadcast_row = x.shape[0] == 1 and i.shape[0] != 1
            broadcast_col = x.shape[1] == 1 and i.shape[1] != 1
            if not ((broadcast_row or x.shape[0] == i.shape[0]) and
                    (broadcast_col or x.shape[1] == i.shape[1])):
                raise ValueError('shape mismatch in assignment')
            x.sum_duplicates()
            self._set_arrayXarray_sparse(i, j, x)
        else:
            # Make x and i into the same shape
            x = np.asarray(x, dtype=self.dtype)
            if x.squeeze().shape != i.squeeze().shape:
                x = np.broadcast_to(x, i.shape)
            if x.size == 0:
                return
            x = x.reshape(i.shape)
            self._set_arrayXarray(i, j, x)

    def _validate_indices(self, key):
        """Returns two tuples: (index tuple, requested shape tuple)"""
        # single ellipsis
        if key is Ellipsis:
            return (slice(None),) * self.ndim, self.shape

        if not isinstance(key, tuple):
            key = [key]

        ellps_pos = None
        index_1st = []
        prelim_ndim = 0
        for i, idx in enumerate(key):
            if idx is Ellipsis:
                if ellps_pos is not None:
                    raise IndexError('an index can only have a single ellipsis')
                ellps_pos = i
            elif idx is None:
                index_1st.append(idx)
            elif isinstance(idx, slice) or isintlike(idx):
                index_1st.append(idx)
                prelim_ndim += 1
            elif (ix := _compatible_boolean_index(idx, self.ndim)) is not None:
                index_1st.append(ix)
                prelim_ndim += ix.ndim
            elif issparse(idx):
                # TODO: make sparse matrix indexing work for sparray
                raise IndexError(
                    'Indexing with sparse matrices is not supported '
                    'except boolean indexing where matrix and index '
                    'are equal shapes.')
            else:  # dense array
                index_1st.append(np.asarray(idx))
                prelim_ndim += 1
        ellip_slices = (self.ndim - prelim_ndim) * [slice(None)]
        if ellip_slices:
            if ellps_pos is None:
                index_1st.extend(ellip_slices)
            else:
                index_1st = index_1st[:ellps_pos] + ellip_slices + index_1st[ellps_pos:]

        # second pass (have processed ellipsis and preprocessed arrays)
        idx_shape = []
        index_ndim = 0
        index = []
        array_indices = []
        for i, idx in enumerate(index_1st):
            if idx is None:
                idx_shape.append(1)
            elif isinstance(idx, slice):
                index.append(idx)
                Ms = self._shape[index_ndim]
                len_slice = len(range(*idx.indices(Ms)))
                idx_shape.append(len_slice)
                index_ndim += 1
            elif isintlike(idx):
                N = self._shape[index_ndim]
                if not (-N <= idx < N):
                    raise IndexError(f'index ({idx}) out of range')
                idx = int(idx + N if idx < 0 else idx)
                index.append(idx)
                index_ndim += 1
            # bool array (checked in first pass)
            elif idx.dtype.kind == 'b':
                ix = idx
                tmp_ndim = index_ndim + ix.ndim
                mid_shape = self._shape[index_ndim:tmp_ndim]
                if ix.shape != mid_shape:
                    raise IndexError(
                        f"bool index {i} has shape {mid_shape} instead of {ix.shape}"
                    )
                index.extend(ix.nonzero())
                array_indices.extend(range(index_ndim, tmp_ndim))
                index_ndim = tmp_ndim
            else:  # dense array
                N = self._shape[index_ndim]
                idx = self._asindices(idx, N)
                index.append(idx)
                array_indices.append(index_ndim)
                index_ndim += 1
        if index_ndim > self.ndim:
            raise IndexError(
                f'invalid index ndim. Array is {self.ndim}D. Index needs {index_ndim}D'
            )
        if len(array_indices) > 1:
            idx_arrays = _broadcast_arrays(*(index[i] for i in array_indices))
            if any(idx_arrays[0].shape != ix.shape for ix in idx_arrays[1:]):
                shapes = " ".join(str(ix.shape) for ix in idx_arrays)
                msg = (f'shape mismatch: indexing arrays could not be broadcast '
                       f'together with shapes {shapes}')
                raise IndexError(msg)
            # TODO: handle this for nD (adjacent arrays stay, separated move to start)
            idx_shape = list(idx_arrays[0].shape) + idx_shape
        elif len(array_indices) == 1:
            arr_index = array_indices[0]
            arr_shape = list(index[arr_index].shape)
            idx_shape = idx_shape[:arr_index] + arr_shape + idx_shape[arr_index:]
        if (ndim := len(idx_shape)) > 2:
            raise IndexError(f'Only 1D or 2D arrays allowed. Index makes {ndim}D')
        return tuple(index), tuple(idx_shape)

    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.

        Subclasses that need special validation can override this method.
        """
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            raise IndexError('invalid index') from e

        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be 1 or 2')

        if x.size == 0:
            return x

        # Check bounds
        max_indx = x.max()
        if max_indx >= length:
            raise IndexError('index (%d) out of range' % max_indx)

        min_indx = x.min()
        if min_indx < 0:
            if min_indx < -length:
                raise IndexError('index (%d) out of range' % min_indx)
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    def _getrow(self, i):
        """Return a copy of row i of the matrix, as a (1 x n) row vector.
        """
        M, N = self.shape
        i = int(i)
        if i < -M or i >= M:
            raise IndexError('index (%d) out of range' % i)
        if i < 0:
            i += M
        return self._get_intXslice(i, slice(None))

    def _getcol(self, i):
        """Return a copy of column i of the matrix, as a (m x 1) column vector.
        """
        M, N = self.shape
        i = int(i)
        if i < -N or i >= N:
            raise IndexError('index (%d) out of range' % i)
        if i < 0:
            i += N
        return self._get_sliceXint(slice(None), i)

    def _get_int(self, idx):
        raise NotImplementedError()

    def _get_slice(self, idx):
        raise NotImplementedError()

    def _get_array(self, idx):
        raise NotImplementedError()

    def _get_intXint(self, row, col):
        raise NotImplementedError()

    def _get_intXarray(self, row, col):
        raise NotImplementedError()

    def _get_intXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXint(self, row, col):
        raise NotImplementedError()

    def _get_sliceXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXint(self, row, col):
        raise NotImplementedError()

    def _get_arrayXslice(self, row, col):
        raise NotImplementedError()

    def _get_columnXarray(self, row, col):
        raise NotImplementedError()

    def _get_arrayXarray(self, row, col):
        raise NotImplementedError()

    def _set_int(self, idx, x):
        raise NotImplementedError()

    def _set_array(self, idx, x):
        raise NotImplementedError()

    def _set_intXint(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray(self, row, col, x):
        raise NotImplementedError()

    def _set_arrayXarray_sparse(self, row, col, x):
        # Fall back to densifying x
        x = np.asarray(x.toarray(), dtype=self.dtype)
        x, _ = _broadcast_arrays(x, row)
        self._set_arrayXarray(row, col, x)


def _compatible_boolean_index(idx, desired_ndim):
    """Check for boolean array or array-like. peek before asarray for array-like"""
    # use attribute ndim to indicate a compatible array and check dtype
    # if not, look at 1st element as quick rejection of bool, else slower asanyarray
    if not hasattr(idx, 'ndim'):
        # is first element boolean?
        try:
            ix = next(iter(idx), None)
            for _ in range(desired_ndim):
                if isinstance(ix, bool):
                    break
                ix = next(iter(ix), None)
            else:
                return None
        except TypeError:
            return None
        # since first is boolean, construct array and check all elements
        idx = np.asanyarray(idx)

    if idx.dtype.kind == 'b':
        return idx
    return None
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'csr_count_blocks',
    'csr_matrix',
    'csr_tobsr',
    'csr_tocsc',
    'get_csr_submatrix',
    'isspmatrix_csr',
    'spmatrix',
    'upcast',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="csr",
                                   private_modules=["_csr"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.sparse` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="sparse", module="spfuncs",
                                   private_modules=["_spfuncs"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Dictionary Of Keys based matrix"""

__docformat__ = "restructuredtext en"

__all__ = ['dok_array', 'dok_matrix', 'isspmatrix_dok']

import itertools
from warnings import warn
import numpy as np

from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin
from ._sputils import (isdense, getdtype, isshape, isintlike, isscalarlike,
                       upcast, upcast_scalar, check_shape)


class _dok_base(_spbase, IndexMixin, dict):
    _format = 'dok'
    _allow_nd = (1, 2)

    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        _spbase.__init__(self, arg1, maxprint=maxprint)

        if isinstance(arg1, tuple) and isshape(arg1, allow_nd=self._allow_nd):
            self._shape = check_shape(arg1, allow_nd=self._allow_nd)
            self._dict = {}
            self.dtype = getdtype(dtype, default=float)
        elif issparse(arg1):  # Sparse ctor
            if arg1.format == self.format:
                arg1 = arg1.copy() if copy else arg1
            else:
                arg1 = arg1.todok()

            if dtype is not None:
                arg1 = arg1.astype(dtype, copy=False)

            self._dict = arg1._dict
            self._shape = check_shape(arg1.shape, allow_nd=self._allow_nd)
            self.dtype = getdtype(arg1.dtype)
        else:  # Dense ctor
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise TypeError('Invalid input format.') from e

            if arg1.ndim > 2:
                raise ValueError(f"DOK arrays don't yet support {arg1.ndim}D input.")

            if arg1.ndim == 1:
                if dtype is not None:
                    arg1 = arg1.astype(dtype)
                self._dict = {i: v for i, v in enumerate(arg1) if v != 0}
                self.dtype = getdtype(arg1.dtype)
            else:
                d = self._coo_container(arg1, shape=shape, dtype=dtype).todok()
                self._dict = d._dict
                self.dtype = getdtype(d.dtype)
            self._shape = check_shape(arg1.shape, allow_nd=self._allow_nd)

    def update(self, val):
        # Prevent direct usage of update
        raise NotImplementedError("Direct update to DOK sparse format is not allowed.")

    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "_getnnz over an axis is not implemented for DOK format."
            )
        return len(self._dict)

    def count_nonzero(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "count_nonzero over an axis is not implemented for DOK format."
            )
        return sum(x != 0 for x in self.values())

    _getnnz.__doc__ = _spbase._getnnz.__doc__
    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def setdefault(self, key, default=None, /):
        return self._dict.setdefault(key, default)

    def __delitem__(self, key, /):
        del self._dict[key]

    def clear(self):
        return self._dict.clear()

    def pop(self, /, *args):
        return self._dict.pop(*args)

    def __reversed__(self):
        raise TypeError("reversed is not defined for dok_array type")

    def __or__(self, other):
        type_names = f"{type(self).__name__} and {type(other).__name__}"
        raise TypeError(f"unsupported operand type for |: {type_names}")

    def __ror__(self, other):
        type_names = f"{type(self).__name__} and {type(other).__name__}"
        raise TypeError(f"unsupported operand type for |: {type_names}")

    def __ior__(self, other):
        type_names = f"{type(self).__name__} and {type(other).__name__}"
        raise TypeError(f"unsupported operand type for |: {type_names}")

    def popitem(self):
        return self._dict.popitem()

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def get(self, key, default=0.0):
        """This provides dict.get method functionality with type checking"""
        if key in self._dict:
            return self._dict[key]
        if isintlike(key) and self.ndim == 1:
            key = (key,)
        if self.ndim != len(key):
            raise IndexError(f'Index {key} length needs to match self.shape')
        try:
            for i in key:
                assert isintlike(i)
        except (AssertionError, TypeError, ValueError) as e:
            raise IndexError('Index must be or consist of integers.') from e
        key = tuple(i + M if i < 0 else i for i, M in zip(key, self.shape))
        if any(i < 0 or i >= M for i, M in zip(key, self.shape)):
            raise IndexError('Index out of bounds.')
        if self.ndim == 1:
            key = key[0]
        return self._dict.get(key, default)

    # 1D get methods
    def _get_int(self, idx):
        return self._dict.get(idx, self.dtype.type(0))

    def _get_slice(self, idx):
        i_range = range(*idx.indices(self.shape[0]))
        return self._get_array(list(i_range))

    def _get_array(self, idx):
        idx = np.asarray(idx)
        if idx.ndim == 0:
            val = self._dict.get(int(idx), self.dtype.type(0))
            return np.array(val, stype=self.dtype)
        new_dok = self._dok_container(idx.shape, dtype=self.dtype)
        dok_vals = [self._dict.get(i, 0) for i in idx.ravel()]
        if dok_vals:
            if len(idx.shape) == 1:
                for i, v in enumerate(dok_vals):
                    if v:
                        new_dok._dict[i] = v
            else:
                new_idx = np.unravel_index(np.arange(len(dok_vals)), idx.shape)
                new_idx = new_idx[0] if len(new_idx) == 1 else zip(*new_idx)
                for i, v in zip(new_idx, dok_vals, strict=True):
                    if v:
                        new_dok._dict[i] = v
        return new_dok

    # 2D get methods
    def _get_intXint(self, row, col):
        return self._dict.get((row, col), self.dtype.type(0))

    def _get_intXslice(self, row, col):
        return self._get_sliceXslice(slice(row, row + 1), col)

    def _get_sliceXint(self, row, col):
        return self._get_sliceXslice(row, slice(col, col + 1))

    def _get_sliceXslice(self, row, col):
        row_start, row_stop, row_step = row.indices(self.shape[0])
        col_start, col_stop, col_step = col.indices(self.shape[1])
        row_range = range(row_start, row_stop, row_step)
        col_range = range(col_start, col_stop, col_step)
        shape = (len(row_range), len(col_range))
        # Switch paths only when advantageous
        # (count the iterations in the loops, adjust for complexity)
        if len(self) >= 2 * shape[0] * shape[1]:
            # O(nr*nc) path: loop over <row x col>
            return self._get_columnXarray(row_range, col_range)
        # O(nnz) path: loop over entries of self
        newdok = self._dok_container(shape, dtype=self.dtype)
        for key in self.keys():
            i, ri = divmod(int(key[0]) - row_start, row_step)
            if ri != 0 or i < 0 or i >= shape[0]:
                continue
            j, rj = divmod(int(key[1]) - col_start, col_step)
            if rj != 0 or j < 0 or j >= shape[1]:
                continue
            newdok._dict[i, j] = self._dict[key]
        return newdok

    def _get_intXarray(self, row, col):
        return self._get_columnXarray([row], col.ravel())

    def _get_arrayXint(self, row, col):
        res = self._get_columnXarray(row.ravel(), [col])
        if row.ndim > 1:
            return res.reshape(row.shape)
        return res

    def _get_sliceXarray(self, row, col):
        row = list(range(*row.indices(self.shape[0])))
        return self._get_columnXarray(row, col)

    def _get_arrayXslice(self, row, col):
        col = list(range(*col.indices(self.shape[1])))
        return self._get_columnXarray(row, col)

    def _get_columnXarray(self, row, col):
        # outer indexing
        newdok = self._dok_container((len(row), len(col)), dtype=self.dtype)

        for i, r in enumerate(row):
            for j, c in enumerate(col):
                v = self._dict.get((r, c), 0)
                if v:
                    newdok._dict[i, j] = v
        return newdok

    def _get_arrayXarray(self, row, col):
        # inner indexing
        i, j = map(np.atleast_2d, np.broadcast_arrays(row, col))
        newdok = self._dok_container(i.shape, dtype=self.dtype)

        for key in itertools.product(range(i.shape[0]), range(i.shape[1])):
            v = self._dict.get((i[key], j[key]), 0)
            if v:
                newdok._dict[key] = v
        return newdok

    # 1D set methods
    def _set_int(self, idx, x):
        if x:
            self._dict[idx] = x
        elif idx in self._dict:
            del self._dict[idx]

    def _set_array(self, idx, x):
        idx_set = idx.ravel()
        x_set = x.ravel()
        if len(idx_set) != len(x_set):
            if len(x_set) == 1:
                x_set = np.full(len(idx_set), x_set[0], dtype=self.dtype)
            else:
              raise ValueError("Need len(index)==len(data) or len(data)==1")
        for i, v in zip(idx_set, x_set):
            if v:
                self._dict[i] = v
            elif i in self._dict:
                del self._dict[i]

    # 2D set methods
    def _set_intXint(self, row, col, x):
        key = (row, col)
        if x:
            self._dict[key] = x
        elif key in self._dict:
            del self._dict[key]

    def _set_arrayXarray(self, row, col, x):
        row = list(map(int, row.ravel()))
        col = list(map(int, col.ravel()))
        x = x.ravel()
        self._dict.update(zip(zip(row, col), x))

        for i in np.nonzero(x == 0)[0]:
            key = (row[i], col[i])
            if self._dict[key] == 0:
                # may have been superseded by later update
                del self._dict[key]

    def __add__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = self._dok_container(self.shape, dtype=res_dtype)
            # Add this scalar to each element.
            for key in itertools.product(*[range(d) for d in self.shape]):
                aij = self._dict.get(key, 0) + other
                if aij:
                    new[key] = aij
        elif issparse(other):
            if other.shape != self.shape:
                raise ValueError("Matrix dimensions are not equal.")
            res_dtype = upcast(self.dtype, other.dtype)
            new = self._dok_container(self.shape, dtype=res_dtype)
            new._dict = self._dict.copy()
            if other.format == "dok":
                o_items = other.items()
            else:
                other = other.tocoo()
                if self.ndim == 1:
                    o_items = zip(other.coords[0], other.data)
                else:
                    o_items = zip(zip(*other.coords), other.data)
            with np.errstate(over='ignore'):
                new._dict.update((k, new[k] + v) for k, v in o_items)
        elif isdense(other):
            new = self.todense() + other
        else:
            return NotImplemented
        return new

    def __radd__(self, other):
        return self + other  # addition is commutative

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError(
                'Negating a sparse boolean matrix is not supported.'
            )
        new = self._dok_container(self.shape, dtype=self.dtype)
        new._dict.update((k, -v) for k, v in self.items())
        return new

    def _mul_scalar(self, other):
        res_dtype = upcast_scalar(self.dtype, other)
        # Multiply this scalar by every element.
        new = self._dok_container(self.shape, dtype=res_dtype)
        new._dict.update(((k, v * other) for k, v in self.items()))
        return new

    def _matmul_vector(self, other):
        res_dtype = upcast(self.dtype, other.dtype)

        # vector @ vector
        if self.ndim == 1:
            if issparse(other):
                if other.format == "dok":
                    keys = self.keys() & other.keys()
                else:
                    keys = self.keys() & other.tocoo().coords[0]
                return res_dtype(sum(self._dict[k] * other._dict[k] for k in keys))
            elif isdense(other):
                return res_dtype(sum(other[k] * v for k, v in self.items()))
            else:
                return NotImplemented

        # matrix @ vector
        result = np.zeros(self.shape[0], dtype=res_dtype)
        for (i, j), v in self.items():
            result[i] += v * other[j]
        return result

    def _matmul_multivector(self, other):
        result_dtype = upcast(self.dtype, other.dtype)
        # vector @ multivector
        if self.ndim == 1:
            # works for other 1d or 2d
            return sum(v * other[j] for j, v in self._dict.items())

        # matrix @ multivector
        M = self.shape[0]
        new_shape = (M,) if other.ndim == 1 else (M, other.shape[1])
        result = np.zeros(new_shape, dtype=result_dtype)
        for (i, j), v in self.items():
            result[i] += v * other[j]
        return result

    def __imul__(self, other):
        if isscalarlike(other):
            self._dict.update((k, v * other) for k, v in self.items())
            return self
        return NotImplemented

    def __truediv__(self, other):
        if isscalarlike(other):
            res_dtype = upcast_scalar(self.dtype, other)
            new = self._dok_container(self.shape, dtype=res_dtype)
            new._dict.update(((k, v / other) for k, v in self.items()))
            return new
        return self.tocsr() / other

    def __itruediv__(self, other):
        if isscalarlike(other):
            self._dict.update((k, v / other) for k, v in self.items())
            return self
        return NotImplemented

    def __reduce__(self):
        # this approach is necessary because __setstate__ is called after
        # __setitem__ upon unpickling and since __init__ is not called there
        # is no shape attribute hence it is not possible to unpickle it.
        return dict.__reduce__(self)

    def diagonal(self, k=0):
        if self.ndim == 2:
            return super().diagonal(k)
        raise ValueError("diagonal requires two dimensions")

    def transpose(self, axes=None, copy=False):
        if self.ndim == 1:
            return self.copy()

        if axes is not None and axes != (1, 0):
            raise ValueError(
                "Sparse arrays/matrices do not support "
                "an 'axes' parameter because swapping "
                "dimensions is the only logical permutation."
            )

        M, N = self.shape
        new = self._dok_container((N, M), dtype=self.dtype, copy=copy)
        new._dict.update((((right, left), val) for (left, right), val in self.items()))
        return new

    transpose.__doc__ = _spbase.transpose.__doc__

    def conjtransp(self):
        """DEPRECATED: Return the conjugate transpose.

        .. deprecated:: 1.14.0

            `conjtransp` is deprecated and will be removed in v1.16.0.
            Use ``.T.conj()`` instead.
        """
        msg = ("`conjtransp` is deprecated and will be removed in v1.16.0. "
                   "Use `.T.conj()` instead.")
        warn(msg, DeprecationWarning, stacklevel=2)

        if self.ndim == 1:
            new = self.tocoo()
            new.data = new.data.conjugate()
            return new

        M, N = self.shape
        new = self._dok_container((N, M), dtype=self.dtype)
        new._dict = {(right, left): np.conj(val) for (left, right), val in self.items()}
        return new

    def copy(self):
        new = self._dok_container(self.shape, dtype=self.dtype)
        new._dict.update(self._dict)
        return new

    copy.__doc__ = _spbase.copy.__doc__

    @classmethod
    def fromkeys(cls, iterable, value=1, /):
        tmp = dict.fromkeys(iterable, value)
        if isinstance(next(iter(tmp)), tuple):
            shape = tuple(max(idx) + 1 for idx in zip(*tmp))
        else:
            shape = (max(tmp) + 1,)
        result = cls(shape, dtype=type(value))
        result._dict = tmp
        return result

    def tocoo(self, copy=False):
        nnz = self.nnz
        if nnz == 0:
            return self._coo_container(self.shape, dtype=self.dtype)

        idx_dtype = self._get_index_dtype(maxval=max(self.shape))
        data = np.fromiter(self.values(), dtype=self.dtype, count=nnz)
        # handle 1d keys specially b/c not a tuple
        inds = zip(*self.keys()) if self.ndim > 1 else (self.keys(),)
        coords = tuple(np.fromiter(ix, dtype=idx_dtype, count=nnz) for ix in inds)
        A = self._coo_container((data, coords), shape=self.shape, dtype=self.dtype)
        A.has_canonical_format = True
        return A

    tocoo.__doc__ = _spbase.tocoo.__doc__

    def todok(self, copy=False):
        if copy:
            return self.copy()
        return self

    todok.__doc__ = _spbase.todok.__doc__

    def tocsc(self, copy=False):
        if self.ndim == 1:
            raise NotImplementedError("tocsr() not valid for 1d sparse array")
        return self.tocoo(copy=False).tocsc(copy=copy)

    tocsc.__doc__ = _spbase.tocsc.__doc__

    def resize(self, *shape):
        shape = check_shape(shape, allow_nd=self._allow_nd)
        if len(shape) != len(self.shape):
            # TODO implement resize across dimensions
            raise NotImplementedError

        if self.ndim == 1:
            newN = shape[-1]
            for i in list(self._dict):
                if i >= newN:
                    del self._dict[i]
            self._shape = shape
            return

        newM, newN = shape
        M, N = self.shape
        if newM < M or newN < N:
            # Remove all elements outside new dimensions
            for i, j in list(self.keys()):
                if i >= newM or j >= newN:
                    del self._dict[i, j]
        self._shape = shape

    resize.__doc__ = _spbase.resize.__doc__

    # Added for 1d to avoid `tocsr` from _base.py
    def astype(self, dtype, casting='unsafe', copy=True):
        dtype = np.dtype(dtype)
        if self.dtype != dtype:
            result = self._dok_container(self.shape, dtype=dtype)
            data = np.array(list(self._dict.values()), dtype=dtype)
            result._dict = dict(zip(self._dict, data))
            return result
        elif copy:
            return self.copy()
        return self


def isspmatrix_dok(x):
    """Is `x` of dok_array type?

    Parameters
    ----------
    x
        object to check for being a dok matrix

    Returns
    -------
    bool
        True if `x` is a dok matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dok_array, dok_matrix, coo_matrix, isspmatrix_dok
    >>> isspmatrix_dok(dok_matrix([[5]]))
    True
    >>> isspmatrix_dok(dok_array([[5]]))
    False
    >>> isspmatrix_dok(coo_matrix([[5]]))
    False
    """
    return isinstance(x, dok_matrix)


# This namespace class separates array from matrix with isinstance
class dok_array(_dok_base, sparray):
    """
    Dictionary Of Keys based sparse array.

    This is an efficient structure for constructing sparse
    arrays incrementally.

    This can be instantiated in several ways:
        dok_array(D)
            where D is a 2-D ndarray

        dok_array(S)
            with another sparse array or matrix S (equivalent to S.todok())

        dok_array((M,N), [dtype])
            create the array with initial shape (M,N)
