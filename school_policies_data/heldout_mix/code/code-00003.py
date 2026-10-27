        # XXX: We have to convert to int before converting to ZZ because
        # flint.fmpz cannot convert a mpmath mpf.
        return ZZ(int(a.context.nint(a)))

    def round_roots_to_integers_for_poly(self, T):
        """
        For a given polynomial *T*, round the roots of this resolvent to the
        nearest integers.

        Explanation
        ===========

        None of the integers returned by this method is guaranteed to be a
        root of the resolvent; however, if the resolvent has any integer roots
        (for the given polynomial *T*), then they must be among these.

        If the coefficients of the resolvent are also desired, then this method
        should not be used. Instead, use the ``eval_for_poly`` method. This
        method may be significantly faster than ``eval_for_poly``.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        Returns
        =======

        dict
            Keys are the indices of those permutations in ``self.s`` such that
            the corresponding root did round to a rational integer.

            Values are :ref:`ZZ`.


        """
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='roots')
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        return {
            i: self.round_mpf(r.real)
            for i, r in enumerate(approx_roots_of_self)
            if self.round_mpf(r.imag) == 0
        }

    def eval_for_poly(self, T, find_integer_root=False):
        r"""
        Compute the integer values of the coefficients of this resolvent, when
        plugging in the roots of a given polynomial.

        Parameters
        ==========

        T : :py:class:`~.Poly`

        find_integer_root : ``bool``, default ``False``
            If ``True``, then also determine whether the resolvent has an
            integer root, and return the first one found, along with its
            index, i.e. the index of the permutation ``self.s[i]`` it
            corresponds to.

        Returns
        =======

        Tuple ``(R, a, i)``

            ``R`` is this resolvent as a dense univariate polynomial over
            :ref:`ZZ`, i.e. a list of :ref:`ZZ`.

            If *find_integer_root* was ``True``, then ``a`` and ``i`` are the
            first integer root found, and its index, if one exists.
            Otherwise ``a`` and ``i`` are both ``None``.

        """
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='coeffs')
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        approx_coeffs_of_self = [c(*approx_roots_of_self) for c in self.esf_lambdas]

        R = []
        for c in approx_coeffs_of_self:
            if self.round_mpf(c.imag) != 0:
                # If precision was enough, this should never happen.
                raise ResolventException(f"Got non-integer coeff for resolvent: {c}")
            R.append(self.round_mpf(c.real))

        a0, i0 = None, None

        if find_integer_root:
            for i, r in enumerate(approx_roots_of_self):
                if self.round_mpf(r.imag) != 0:
                    continue
                if not dup_eval(R, (a := self.round_mpf(r.real)), ZZ):
                    a0, i0 = a, i
                    break

        return R, a0, i0


def wrap(text, width=80):
    """Line wrap a polynomial expression. """
    out = ''
    col = 0
    for c in text:
        if c == ' ' and col > width:
            c, col = '\n', 0
        else:
            col += 1
        out += c
    return out


def s_vars(n):
    """Form the symbols s1, s2, ..., sn to stand for elem. symm. polys. """
    return symbols([f's{i + 1}' for i in range(n)])


def sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=False):
    """
    Compute the coefficients of a resolvent as functions of the coefficients of
    the associated polynomial.

    F must be a sparse polynomial.
    """
    import time, sys
    # Roots of resolvent as multivariate forms over vars X:
    root_forms = [
        F.compose(list(zip(X, sigma(X))))
        for sigma in s
    ]

    # Coeffs of resolvent (besides lead coeff of 1) as symmetric forms over vars X:
    Y = [Dummy(f'Y{i}') for i in range(len(s))]
    coeff_forms = []
    for i in range(1, len(s) + 1):
        if verbose:
            print('----')
            print(f'Computing symmetric poly of degree {i}...')
            sys.stdout.flush()
        t0 = time.time()
        G = symmetric_poly(i, *Y)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            print('lambdifying...')
            sys.stdout.flush()
        t0 = time.time()
        C = lambdify(Y, (-1)**i*G)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()
        coeff_forms.append(C)

    coeffs = []
    for i, f in enumerate(coeff_forms):
        if verbose:
            print('----')
            print(f'Plugging root forms into elem symm poly {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        g = f(*root_forms)
        t1 = time.time()
        coeffs.append(g)
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    # Now symmetrize these coeffs. This means recasting them as polynomials in
    # the elementary symmetric polys over X.
    symmetrized = []
    symmetrization_times = []
    ss = s_vars(len(X))
    for i, A in list(enumerate(coeffs)):
        if verbose:
            print('-----')
            print(f'Coeff {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        B, rem, _ = A.symmetrize()
        t1 = time.time()
        if rem != 0:
            msg = f"Got nonzero remainder {rem} for resolvent (F, X, s) = ({F}, {X}, {s})"
            raise ResolventException(msg)
        B_str = str(B.as_expr(*ss))
        symmetrized.append(B_str)
        symmetrization_times.append(t1 - t0)
        if verbose:
            print(wrap(B_str))
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    return symmetrized, symmetrization_times


def define_resolvents():
    """Define all the resolvents for polys T of degree 4 through 6. """
    from sympy.combinatorics.galois import PGL2F5
    from sympy.combinatorics.permutations import Permutation

    R4, X4 = xring("X0,X1,X2,X3", ZZ, lex)
    X = X4

    # The one resolvent used in `_galois_group_degree_4_lookup()`:
    F40 = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    s40 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 2),
        Permutation(3)(0, 3),
        Permutation(3)(1, 2),
        Permutation(3)(2, 3),
    ]

    # First resolvent used in `_galois_group_degree_4_root_approx()`:
    F41 = X[0]*X[2] + X[1]*X[3]
    s41 = [
        Permutation(3),
        Permutation(3)(0, 1),
        Permutation(3)(0, 3)
    ]

    R5, X5 = xring("X0,X1,X2,X3,X4", ZZ, lex)
    X = X5

    # First resolvent used in `_galois_group_degree_5_hybrid()`,
    # and only one used in `_galois_group_degree_5_lookup_ext_factor()`:
    F51 = (  X[0]**2*(X[1]*X[4] + X[2]*X[3])
           + X[1]**2*(X[2]*X[0] + X[3]*X[4])
           + X[2]**2*(X[3]*X[1] + X[4]*X[0])
           + X[3]**2*(X[4]*X[2] + X[0]*X[1])
           + X[4]**2*(X[0]*X[3] + X[1]*X[2]))
    s51 = [
        Permutation(4),
        Permutation(4)(0, 1),
        Permutation(4)(0, 2),
        Permutation(4)(0, 3),
        Permutation(4)(0, 4),
        Permutation(4)(1, 4)
    ]

    R6, X6 = xring("X0,X1,X2,X3,X4,X5", ZZ, lex)
    X = X6

    # First resolvent used in `_galois_group_degree_6_lookup()`:
    H = PGL2F5()
    term0 = X[0]**2*X[5]**2*(X[1]*X[4] + X[2]*X[3])
    terms = {term0.compose(list(zip(X, s(X)))) for s in H.elements}
    F61 = sum(terms)
    s61 = [Permutation(5)] + [Permutation(5)(0, n) for n in range(1, 6)]

    # Second resolvent used in `_galois_group_degree_6_lookup()`:
    F62 = X[0]*X[1]*X[2] + X[3]*X[4]*X[5]
    s62 = [Permutation(5)] + [
        Permutation(5)(i, j + 3) for i in range(3) for j in range(3)
    ]

    return {
        (4, 0): (F40, X4, s40),
        (4, 1): (F41, X4, s41),
        (5, 1): (F51, X5, s51),
        (6, 1): (F61, X6, s61),
        (6, 2): (F62, X6, s62),
    }


def generate_lambda_lookup(verbose=False, trial_run=False):
    """
    Generate the whole lookup table of coeff lambdas, for all resolvents.
    """
    jobs = define_resolvents()
    lambda_lists = {}
    total_time = 0
    time_for_61 = 0
    time_for_61_last = 0
    for k, (F, X, s) in jobs.items():
        symmetrized, times = sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=verbose)

        total_time += sum(times)
        if k == (6, 1):
            time_for_61 = sum(times)
            time_for_61_last = times[-1]

        sv = s_vars(len(X))
        head = f'lambda {", ".join(str(v) for v in sv)}:'
        lambda_lists[k] = ',\n        '.join([
            f'{head} ({wrap(f)})'
            for f in symmetrized
        ])

        if trial_run:
            break

    table = (
         "# This table was generated by a call to\n"
         "# `sympy.polys.numberfields.galois_resolvents.generate_lambda_lookup()`.\n"
        f"# The entire job took {total_time:.2f}s.\n"
        f"# Of this, Case (6, 1) took {time_for_61:.2f}s.\n"
        f"# The final polynomial of Case (6, 1) alone took {time_for_61_last:.2f}s.\n"
         "resolvent_coeff_lambdas = {\n")

    for k, L in lambda_lists.items():
        table += f"    {k}: [\n"
        table +=  "        " + L + '\n'
        table +=  "    ],\n"
    table += "}\n"
    return table


def get_resolvent_by_lookup(T, number):
    """
    Use the lookup table, to return a resolvent (as dup) for a given
    polynomial *T*.

    Parameters
    ==========

    T : Poly
        The polynomial whose resolvent is needed

    number : int
        For some degrees, there are multiple resolvents.
        Use this to indicate which one you want.

    Returns
    =======

    dup

    """
    from sympy.polys.numberfields.resolvent_lookup import resolvent_coeff_lambdas
    degree = T.degree()
    L = resolvent_coeff_lambdas[(degree, number)]
    T_coeffs = T.rep.to_list()[1:]
    return [ZZ(1)] + [c(*T_coeffs) for c in L]


# Use
#   (.venv) $ python -m sympy.polys.numberfields.galois_resolvents
# to reproduce the table found in resolvent_lookup.py
if __name__ == "__main__":
    import sys
    verbose = '-v' in sys.argv[1:]
    trial_run = '-t' in sys.argv[1:]
    table = generate_lambda_lookup(verbose=verbose, trial_run=trial_run)
    print(table)
<|endoftext|>"""
Create a dist_info directory
As defined in the wheel specification
"""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, cast

from .. import _normalization
from .._shutil import rmdir as _rm
from .egg_info import egg_info as egg_info_cls

from distutils import log
from distutils.core import Command


class dist_info(Command):
    """
    This command is private and reserved for internal use of setuptools,
    users should rely on ``setuptools.build_meta`` APIs.
    """

    description = "DO NOT CALL DIRECTLY, INTERNAL ONLY: create .dist-info directory"

    user_options: ClassVar[
        list[tuple[str, str, str]] | list[tuple[str, str | None, str]]
    ] = [
        (
            'output-dir=',
            'o',
            (
                "directory inside of which the .dist-info will be"
                "created [default: top of the source tree]"
            ),
        ),
        ('tag-date', 'd', "Add date stamp (e.g. 20050528) to version number"),
        ('tag-build=', 'b', "Specify explicit tag to add to version number"),
        ('no-date', 'D', "Don't include date stamp [default]"),
        ('keep-egg-info', None, "*TRANSITIONAL* will be removed in the future"),
    ]

    boolean_options: ClassVar[list[str]] = ['tag-date', 'keep-egg-info']
    negative_opt: ClassVar[dict[str, str]] = {'no-date': 'tag-date'}

    def initialize_options(self):
        self.output_dir = None
        self.name = None
        self.dist_info_dir = None
        self.tag_date = None
        self.tag_build = None
        self.keep_egg_info = False

    def finalize_options(self) -> None:
        dist = self.distribution
        project_dir = dist.src_root or os.curdir
        self.output_dir = Path(self.output_dir or project_dir)

        egg_info = cast(egg_info_cls, self.reinitialize_command("egg_info"))
        egg_info.egg_base = str(self.output_dir)

        if self.tag_date:
            egg_info.tag_date = self.tag_date
        else:
            self.tag_date = egg_info.tag_date

        if self.tag_build:
            egg_info.tag_build = self.tag_build
        else:
            self.tag_build = egg_info.tag_build

        egg_info.finalize_options()
        self.egg_info = egg_info

        name = _normalization.safer_name(dist.get_name())
        version = _normalization.safer_best_effort_version(dist.get_version())
        self.name = f"{name}-{version}"
        self.dist_info_dir = os.path.join(self.output_dir, f"{self.name}.dist-info")

    @contextmanager
    def _maybe_bkp_dir(self, dir_path: str, requires_bkp: bool):
        if requires_bkp:
            bkp_name = f"{dir_path}.__bkp__"
            _rm(bkp_name, ignore_errors=True)
            shutil.copytree(dir_path, bkp_name, dirs_exist_ok=True, symlinks=True)
            try:
                yield
            finally:
                _rm(dir_path, ignore_errors=True)
                shutil.move(bkp_name, dir_path)
        else:
            yield

    def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.egg_info.run()
        egg_info_dir = self.egg_info.egg_info
        assert os.path.isdir(egg_info_dir), ".egg-info dir should have been created"

        log.info(f"creating '{os.path.abspath(self.dist_info_dir)}'")
        bdist_wheel = self.get_finalized_command('bdist_wheel')

        # TODO: if bdist_wheel if merged into setuptools, just add "keep_egg_info" there
        with self._maybe_bkp_dir(egg_info_dir, self.keep_egg_info):
            bdist_wheel.egg2dist(egg_info_dir, self.dist_info_dir)
<|endoftext|>r"""plistlib.py -- a tool to generate and parse MacOSX .plist files.

The property list (.plist) file format is a simple XML pickle supporting
basic object types, like dictionaries, lists, numbers and strings.
Usually the top level object is a dictionary.

To write out a plist file, use the dump(value, file)
function. 'value' is the top level object, 'file' is
a (writable) file object.

To parse a plist from a file, use the load(file) function,
with a (readable) file object as the only argument. It
returns the top level object (again, usually a dictionary).

To work with plist data in bytes objects, you can use loads()
and dumps().

Values can be strings, integers, floats, booleans, tuples, lists,
dictionaries (but only with string keys), Data, bytes, bytearray, or
datetime.datetime objects.

Generate Plist example:

    import datetime
    import plistlib

    pl = dict(
        aString = "Doodah",
        aList = ["A", "B", 12, 32.1, [1, 2, 3]],
        aFloat = 0.1,
        anInt = 728,
        aDict = dict(
            anotherString = "<hello & hi there!>",
            aThirdString = "M\xe4ssig, Ma\xdf",
            aTrueValue = True,
            aFalseValue = False,
        ),
        someData = b"<binary gunk>",
        someMoreData = b"<lots of binary gunk>" * 10,
        aDate = datetime.datetime.now()
    )
    print(plistlib.dumps(pl).decode())

Parse Plist example:

    import plistlib

    plist = b'''<plist version="1.0">
    <dict>
        <key>foo</key>
        <string>bar</string>
    </dict>
    </plist>'''
    pl = plistlib.loads(plist)
    print(pl["foo"])
"""
__all__ = [
    "InvalidFileException", "FMT_XML", "FMT_BINARY", "load", "dump", "loads", "dumps", "UID"
]

import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate


PlistFormat = enum.Enum('PlistFormat', 'FMT_XML FMT_BINARY', module=__name__)
globals().update(PlistFormat.__members__)

# Data larger than this will be read in chunks, to prevent extreme
# overallocation.
_MIN_READ_BUF_SIZE = 1 << 20

class UID:
    def __init__(self, data):
        if not isinstance(data, int):
            raise TypeError("data must be an int")
        if data >= 1 << 64:
            raise ValueError("UIDs cannot be >= 2**64")
        if data < 0:
            raise ValueError("UIDs must be positive")
        self.data = data

    def __index__(self):
        return self.data

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.data))

    def __reduce__(self):
        return self.__class__, (self.data,)

    def __eq__(self, other):
        if not isinstance(other, UID):
            return NotImplemented
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)

#
# XML support
#


# XML 'header'
PLISTHEADER = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
"""


# Regex to find any control chars, except for \t \n and \r
_controlCharPat = re.compile(
    r"[\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"
    r"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f]")

def _encode_base64(s, maxlinelength=76):
    # copied from base64.encodebytes(), with added maxlinelength argument
    maxbinsize = (maxlinelength//4)*3
    pieces = []
    for i in range(0, len(s), maxbinsize):
        chunk = s[i : i + maxbinsize]
        pieces.append(binascii.b2a_base64(chunk))
    return b''.join(pieces)

def _decode_base64(s):
    if isinstance(s, str):
        return binascii.a2b_base64(s.encode("utf-8"))

    else:
        return binascii.a2b_base64(s)

# Contents should conform to a subset of ISO 8601
# (in particular, YYYY '-' MM '-' DD 'T' HH ':' MM ':' SS 'Z'.  Smaller units
# may be omitted with #  a loss of precision)
_dateParser = re.compile(r"(?P<year>\d\d\d\d)(?:-(?P<month>\d\d)(?:-(?P<day>\d\d)(?:T(?P<hour>\d\d)(?::(?P<minute>\d\d)(?::(?P<second>\d\d))?)?)?)?)?Z", re.ASCII)


def _date_from_string(s):
    order = ('year', 'month', 'day', 'hour', 'minute', 'second')
    gd = _dateParser.match(s).groupdict()
    lst = []
    for key in order:
        val = gd[key]
        if val is None:
            break
        lst.append(int(val))
    return datetime.datetime(*lst)


def _date_to_string(d):
    return '%04d-%02d-%02dT%02d:%02d:%02dZ' % (
        d.year, d.month, d.day,
        d.hour, d.minute, d.second
    )

def _escape(text):
    m = _controlCharPat.search(text)
    if m is not None:
        raise ValueError("strings can't contain control characters; "
                         "use bytes instead")
    text = text.replace("\r\n", "\n")       # convert DOS line endings
    text = text.replace("\r", "\n")         # convert Mac line endings
    text = text.replace("&", "&amp;")       # escape '&'
    text = text.replace("<", "&lt;")        # escape '<'
    text = text.replace(">", "&gt;")        # escape '>'
    return text

class _PlistParser:
    def __init__(self, dict_type):
        self.stack = []
        self.current_key = None
        self.root = None
        self._dict_type = dict_type

    def parse(self, fileobj):
        self.parser = ParserCreate()
        self.parser.StartElementHandler = self.handle_begin_element
        self.parser.EndElementHandler = self.handle_end_element
        self.parser.CharacterDataHandler = self.handle_data
        self.parser.EntityDeclHandler = self.handle_entity_decl
        self.parser.ParseFile(fileobj)
        return self.root

    def handle_entity_decl(self, entity_name, is_parameter_entity, value, base, system_id, public_id, notation_name):
        # Reject plist files with entity declarations to avoid XML vulnerabilies in expat.
        # Regular plist files don't contain those declerations, and Apple's plutil tool does not
        # accept them either.
        raise InvalidFileException("XML entity declarations are not supported in plist files")

    def handle_begin_element(self, element, attrs):
        self.data = []
        handler = getattr(self, "begin_" + element, None)
        if handler is not None:
            handler(attrs)

    def handle_end_element(self, element):
        handler = getattr(self, "end_" + element, None)
        if handler is not None:
            handler()

    def handle_data(self, data):
        self.data.append(data)

    def add_object(self, value):
        if self.current_key is not None:
            if not isinstance(self.stack[-1], type({})):
                raise ValueError("unexpected element at line %d" %
                                 self.parser.CurrentLineNumber)
            self.stack[-1][self.current_key] = value
            self.current_key = None
        elif not self.stack:
            # this is the root object
            self.root = value
        else:
            if not isinstance(self.stack[-1], type([])):
                raise ValueError("unexpected element at line %d" %
                                 self.parser.CurrentLineNumber)
            self.stack[-1].append(value)

    def get_data(self):
        data = ''.join(self.data)
        self.data = []
        return data

    # element handlers

    def begin_dict(self, attrs):
        d = self._dict_type()
        self.add_object(d)
        self.stack.append(d)

    def end_dict(self):
        if self.current_key:
            raise ValueError("missing value for key '%s' at line %d" %
                             (self.current_key,self.parser.CurrentLineNumber))
        self.stack.pop()

    def end_key(self):
        if self.current_key or not isinstance(self.stack[-1], type({})):
            raise ValueError("unexpected key at line %d" %
                             self.parser.CurrentLineNumber)
        self.current_key = self.get_data()

    def begin_array(self, attrs):
        a = []
        self.add_object(a)
        self.stack.append(a)

    def end_array(self):
        self.stack.pop()

    def end_true(self):
        self.add_object(True)

    def end_false(self):
        self.add_object(False)

    def end_integer(self):
        raw = self.get_data()
        if raw.startswith('0x') or raw.startswith('0X'):
            self.add_object(int(raw, 16))
        else:
            self.add_object(int(raw))

    def end_real(self):
        self.add_object(float(self.get_data()))

    def end_string(self):
        self.add_object(self.get_data())

    def end_data(self):
        self.add_object(_decode_base64(self.get_data()))

    def end_date(self):
        self.add_object(_date_from_string(self.get_data()))


class _DumbXMLWriter:
    def __init__(self, file, indent_level=0, indent="\t"):
        self.file = file
        self.stack = []
        self._indent_level = indent_level
        self.indent = indent

    def begin_element(self, element):
        self.stack.append(element)
        self.writeln("<%s>" % element)
        self._indent_level += 1

    def end_element(self, element):
        assert self._indent_level > 0
        assert self.stack.pop() == element
        self._indent_level -= 1
        self.writeln("</%s>" % element)

    def simple_element(self, element, value=None):
        if value is not None:
            value = _escape(value)
            self.writeln("<%s>%s</%s>" % (element, value, element))

        else:
            self.writeln("<%s/>" % element)

    def writeln(self, line):
        if line:
            # plist has fixed encoding of utf-8

            # XXX: is this test needed?
            if isinstance(line, str):
                line = line.encode('utf-8')
            self.file.write(self._indent_level * self.indent)
            self.file.write(line)
        self.file.write(b'\n')


class _PlistWriter(_DumbXMLWriter):
    def __init__(
            self, file, indent_level=0, indent=b"\t", writeHeader=1,
            sort_keys=True, skipkeys=False):

        if writeHeader:
            file.write(PLISTHEADER)
        _DumbXMLWriter.__init__(self, file, indent_level, indent)
        self._sort_keys = sort_keys
        self._skipkeys = skipkeys

    def write(self, value):
        self.writeln("<plist version=\"1.0\">")
        self.write_value(value)
        self.writeln("</plist>")

    def write_value(self, value):
        if isinstance(value, str):
            self.simple_element("string", value)

        elif value is True:
            self.simple_element("true")

        elif value is False:
            self.simple_element("false")

        elif isinstance(value, int):
            if -1 << 63 <= value < 1 << 64:
                self.simple_element("integer", "%d" % value)
            else:
                raise OverflowError(value)

        elif isinstance(value, float):
            self.simple_element("real", repr(value))

        elif isinstance(value, dict):
            self.write_dict(value)

        elif isinstance(value, (bytes, bytearray)):
            self.write_bytes(value)

        elif isinstance(value, datetime.datetime):
            self.simple_element("date", _date_to_string(value))

        elif isinstance(value, (tuple, list)):
            self.write_array(value)

        else:
            raise TypeError("unsupported type: %s" % type(value))

    def write_bytes(self, data):
        self.begin_element("data")
        self._indent_level -= 1
        maxlinelength = max(
            16,
            76 - len(self.indent.replace(b"\t", b" " * 8) * self._indent_level))

        for line in _encode_base64(data, maxlinelength).split(b"\n"):
            if line:
                self.writeln(line)
        self._indent_level += 1
        self.end_element("data")

    def write_dict(self, d):
        if d:
            self.begin_element("dict")
            if self._sort_keys:
                items = sorted(d.items())
            else:
                items = d.items()

            for key, value in items:
                if not isinstance(key, str):
                    if self._skipkeys:
                        continue
                    raise TypeError("keys must be strings")
                self.simple_element("key", key)
                self.write_value(value)
            self.end_element("dict")

        else:
            self.simple_element("dict")

    def write_array(self, array):
        if array:
            self.begin_element("array")
            for value in array:
                self.write_value(value)
            self.end_element("array")

        else:
            self.simple_element("array")


def _is_fmt_xml(header):
    prefixes = (b'<?xml', b'<plist')

    for pfx in prefixes:
        if header.startswith(pfx):
            return True

    # Also check for alternative XML encodings, this is slightly
    # overkill because the Apple tools (and plistlib) will not
    # generate files with these encodings.
    for bom, encoding in (
                (codecs.BOM_UTF8, "utf-8"),
                (codecs.BOM_UTF16_BE, "utf-16-be"),
                (codecs.BOM_UTF16_LE, "utf-16-le"),
                # expat does not support utf-32
                #(codecs.BOM_UTF32_BE, "utf-32-be"),
                #(codecs.BOM_UTF32_LE, "utf-32-le"),
            ):
        if not header.startswith(bom):
            continue

        for start in prefixes:
            prefix = bom + start.decode('ascii').encode(encoding)
            if header[:len(prefix)] == prefix:
                return True

    return False

#
# Binary Plist
#


class InvalidFileException (ValueError):
    def __init__(self, message="Invalid file"):
        ValueError.__init__(self, message)

_BINARY_FORMAT = {1: 'B', 2: 'H', 4: 'L', 8: 'Q'}

_undefined = object()

class _BinaryPlistParser:
    """
    Read or write a binary plist file, following the description of the binary
    format.  Raise InvalidFileException in case of error, otherwise return the
    root object.

    see also: http://opensource.apple.com/source/CF/CF-744.18/CFBinaryPList.c
    """
    def __init__(self, dict_type):
        self._dict_type = dict_type

    def parse(self, fp):
        try:
            # The basic file format:
            # HEADER
            # object...
            # refid->offset...
            # TRAILER
            self._fp = fp
            self._fp.seek(-32, os.SEEK_END)
            trailer = self._fp.read(32)
            if len(trailer) != 32:
                raise InvalidFileException()
            (
                offset_size, self._ref_size, num_objects, top_object,
                offset_table_offset
            ) = struct.unpack('>6xBBQQQ', trailer)
            self._fp.seek(offset_table_offset)
            self._object_offsets = self._read_ints(num_objects, offset_size)
            self._objects = [_undefined] * num_objects
            return self._read_object(top_object)

        except (OSError, IndexError, struct.error, OverflowError,
                ValueError):
            raise InvalidFileException()

    def _get_size(self, tokenL):
        """ return the size of the next object."""
        if tokenL == 0xF:
            m = self._fp.read(1)[0] & 0x3
            s = 1 << m
            f = '>' + _BINARY_FORMAT[s]
            return struct.unpack(f, self._fp.read(s))[0]

        return tokenL

    def _read(self, size):
        cursize = min(size, _MIN_READ_BUF_SIZE)
        data = self._fp.read(cursize)
        while True:
            if len(data) != cursize:
                raise InvalidFileException
            if cursize == size:
                return data
            delta = min(cursize, size - cursize)
            data += self._fp.read(delta)
            cursize += delta

    def _read_ints(self, n, size):
        data = self._read(size * n)
        if size in _BINARY_FORMAT:
            return struct.unpack(f'>{n}{_BINARY_FORMAT[size]}', data)
        else:
            if not size:
                raise InvalidFileException()
            return tuple(int.from_bytes(data[i: i + size], 'big')
                         for i in range(0, size * n, size))

    def _read_refs(self, n):
        return self._read_ints(n, self._ref_size)

    def _read_object(self, ref):
        """
        read the object by reference.

        May recursively read sub-objects (content of an array/dict/set)
        """
        result = self._objects[ref]
        if result is not _undefined:
            return result

        offset = self._object_offsets[ref]
        self._fp.seek(offset)
        token = self._fp.read(1)[0]
        tokenH, tokenL = token & 0xF0, token & 0x0F

        if token == 0x00:
            result = None

        elif token == 0x08:
            result = False

        elif token == 0x09:
            result = True

        # The referenced source code also mentions URL (0x0c, 0x0d) and
        # UUID (0x0e), but neither can be generated using the Cocoa libraries.

        elif token == 0x0f:
            result = b''

        elif tokenH == 0x10:  # int
            result = int.from_bytes(self._fp.read(1 << tokenL),
                                    'big', signed=tokenL >= 3)

        elif token == 0x22: # real
            result = struct.unpack('>f', self._fp.read(4))[0]

        elif token == 0x23: # real
            result = struct.unpack('>d', self._fp.read(8))[0]

        elif token == 0x33:  # date
            f = struct.unpack('>d', self._fp.read(8))[0]
            # timestamp 0 of binary plists corresponds to 1/1/2001
            # (year of Mac OS X 10.0), instead of 1/1/1970.
            result = (datetime.datetime(2001, 1, 1) +
                      datetime.timedelta(seconds=f))

        elif tokenH == 0x40:  # data
            s = self._get_size(tokenL)
            result = self._read(s)

        elif tokenH == 0x50:  # ascii string
            s = self._get_size(tokenL)
            data = self._read(s)
            result = data.decode('ascii')

        elif tokenH == 0x60:  # unicode string
            s = self._get_size(tokenL) * 2
            data = self._read(s)
            result = data.decode('utf-16be')

        elif tokenH == 0x80:  # UID
            # used by Key-Archiver plist files
            result = UID(int.from_bytes(self._fp.read(1 + tokenL), 'big'))

        elif tokenH == 0xA0:  # array
            s = self._get_size(tokenL)
            obj_refs = self._read_refs(s)
            result = []
            self._objects[ref] = result
            result.extend(self._read_object(x) for x in obj_refs)

        # tokenH == 0xB0 is documented as 'ordset', but is not actually
        # implemented in the Apple reference code.

        # tokenH == 0xC0 is documented as 'set', but sets cannot be used in
        # plists.

        elif tokenH == 0xD0:  # dict
            s = self._get_size(tokenL)
            key_refs = self._read_refs(s)
            obj_refs = self._read_refs(s)
            result = self._dict_type()
            self._objects[ref] = result
            try:
                for k, o in zip(key_refs, obj_refs):
                    result[self._read_object(k)] = self._read_object(o)
            except TypeError:
                raise InvalidFileException()
        else:
            raise InvalidFileException()

        self._objects[ref] = result
        return result

def _count_to_size(count):
    if count < 1 << 8:
        return 1

    elif count < 1 << 16:
        return 2

    elif count < 1 << 32:
        return 4

    else:
        return 8

_scalars = (str, int, float, datetime.datetime, bytes)

class _BinaryPlistWriter (object):
    def __init__(self, fp, sort_keys, skipkeys):
        self._fp = fp
        self._sort_keys = sort_keys
        self._skipkeys = skipkeys

    def write(self, value):

        # Flattened object list:
        self._objlist = []

        # Mappings from object->objectid
        # First dict has (type(object), object) as the key,
        # second dict is used when object is not hashable and
        # has id(object) as the key.
        self._objtable = {}
        self._objidtable = {}

        # Create list of all objects in the plist
        self._flatten(value)

        # Size of object references in serialized containers
        # depends on the number of objects in the plist.
        num_objects = len(self._objlist)
        self._object_offsets = [0]*num_objects
        self._ref_size = _count_to_size(num_objects)

        self._ref_format = _BINARY_FORMAT[self._ref_size]

        # Write file header
        self._fp.write(b'bplist00')

        # Write object list
        for obj in self._objlist:
            self._write_object(obj)

        # Write refnum->object offset table
        top_object = self._getrefnum(value)
        offset_table_offset = self._fp.tell()
        offset_size = _count_to_size(offset_table_offset)
        offset_format = '>' + _BINARY_FORMAT[offset_size] * num_objects
        self._fp.write(struct.pack(offset_format, *self._object_offsets))

        # Write trailer
        sort_version = 0
        trailer = (
            sort_version, offset_size, self._ref_size, num_objects,
            top_object, offset_table_offset
        )
        self._fp.write(struct.pack('>5xBBBQQQ', *trailer))

    def _flatten(self, value):
        # First check if the object is in the object table, not used for
        # containers to ensure that two subcontainers with the same contents
        # will be serialized as distinct values.
        if isinstance(value, _scalars):
            if (type(value), value) in self._objtable:
                return

        elif id(value) in self._objidtable:
            return

        # Add to objectreference map
        refnum = len(self._objlist)
        self._objlist.append(value)
        if isinstance(value, _scalars):
            self._objtable[(type(value), value)] = refnum
        else:
            self._objidtable[id(value)] = refnum

        # And finally recurse into containers
        if isinstance(value, dict):
            keys = []
            values = []
            items = value.items()
            if self._sort_keys:
                items = sorted(items)

            for k, v in items:
                if not isinstance(k, str):
                    if self._skipkeys:
                        continue
                    raise TypeError("keys must be strings")
                keys.append(k)
                values.append(v)

            for o in itertools.chain(keys, values):
                self._flatten(o)

        elif isinstance(value, (list, tuple)):
            for o in value:
                self._flatten(o)

    def _getrefnum(self, value):
        if isinstance(value, _scalars):
            return self._objtable[(type(value), value)]
        else:
            return self._objidtable[id(value)]

    def _write_size(self, token, size):
        if size < 15:
            self._fp.write(struct.pack('>B', token | size))

        elif size < 1 << 8:
            self._fp.write(struct.pack('>BBB', token | 0xF, 0x10, size))

        elif size < 1 << 16:
            self._fp.write(struct.pack('>BBH', token | 0xF, 0x11, size))

        elif size < 1 << 32:
            self._fp.write(struct.pack('>BBL', token | 0xF, 0x12, size))

        else:
            self._fp.write(struct.pack('>BBQ', token | 0xF, 0x13, size))

    def _write_object(self, value):
        ref = self._getrefnum(value)
        self._object_offsets[ref] = self._fp.tell()
        if value is None:
            self._fp.write(b'\x00')

        elif value is False:
            self._fp.write(b'\x08')

        elif value is True:
            self._fp.write(b'\x09')

        elif isinstance(value, int):
            if value < 0:
                try:
                    self._fp.write(struct.pack('>Bq', 0x13, value))
                except struct.error:
                    raise OverflowError(value) from None
            elif value < 1 << 8:
                self._fp.write(struct.pack('>BB', 0x10, value))
            elif value < 1 << 16:
                self._fp.write(struct.pack('>BH', 0x11, value))
            elif value < 1 << 32:
                self._fp.write(struct.pack('>BL', 0x12, value))
            elif value < 1 << 63:
                self._fp.write(struct.pack('>BQ', 0x13, value))
            elif value < 1 << 64:
                self._fp.write(b'\x14' + value.to_bytes(16, 'big', signed=True))
            else:
                raise OverflowError(value)

        elif isinstance(value, float):
            self._fp.write(struct.pack('>Bd', 0x23, value))

        elif isinstance(value, datetime.datetime):
            f = (value - datetime.datetime(2001, 1, 1)).total_seconds()
            self._fp.write(struct.pack('>Bd', 0x33, f))

        elif isinstance(value, (bytes, bytearray)):
            self._write_size(0x40, len(value))
            self._fp.write(value)

        elif isinstance(value, str):
            try:
                t = value.encode('ascii')
                self._write_size(0x50, len(value))
            except UnicodeEncodeError:
                t = value.encode('utf-16be')
                self._write_size(0x60, len(t) // 2)

            self._fp.write(t)

        elif isinstance(value, UID):
            if value.data < 0:
                raise ValueError("UIDs must be positive")
            elif value.data < 1 << 8:
                self._fp.write(struct.pack('>BB', 0x80, value))
            elif value.data < 1 << 16:
                self._fp.write(struct.pack('>BH', 0x81, value))
            elif value.data < 1 << 32:
                self._fp.write(struct.pack('>BL', 0x83, value))
            elif value.data < 1 << 64:
                self._fp.write(struct.pack('>BQ', 0x87, value))
            else:
                raise OverflowError(value)

        elif isinstance(value, (list, tuple)):
            refs = [self._getrefnum(o) for o in value]
            s = len(refs)
            self._write_size(0xA0, s)
            self._fp.write(struct.pack('>' + self._ref_format * s, *refs))

        elif isinstance(value, dict):
            keyRefs, valRefs = [], []

            if self._sort_keys:
                rootItems = sorted(value.items())
            else:
                rootItems = value.items()

            for k, v in rootItems:
                if not isinstance(k, str):
                    if self._skipkeys:
                        continue
                    raise TypeError("keys must be strings")
                keyRefs.append(self._getrefnum(k))
                valRefs.append(self._getrefnum(v))

            s = len(keyRefs)
            self._write_size(0xD0, s)
            self._fp.write(struct.pack('>' + self._ref_format * s, *keyRefs))
            self._fp.write(struct.pack('>' + self._ref_format * s, *valRefs))

        else:
            raise TypeError(value)


def _is_fmt_binary(header):
    return header[:8] == b'bplist00'


#
# Generic bits
#

_FORMATS={
    FMT_XML: dict(
        detect=_is_fmt_xml,
        parser=_PlistParser,
        writer=_PlistWriter,
    ),
    FMT_BINARY: dict(
        detect=_is_fmt_binary,
        parser=_BinaryPlistParser,
        writer=_BinaryPlistWriter,
    )
}


def load(fp, *, fmt=None, dict_type=dict):
    """Read a .plist file. 'fp' should be a readable and binary file object.
    Return the unpacked root object (which usually is a dictionary).
    """
    if fmt is None:
        header = fp.read(32)
        fp.seek(0)
        for info in _FORMATS.values():
            if info['detect'](header):
                P = info['parser']
                break

        else:
            raise InvalidFileException()

    else:
        P = _FORMATS[fmt]['parser']

    p = P(dict_type=dict_type)
    return p.parse(fp)


def loads(value, *, fmt=None, dict_type=dict):
    """Read a .plist file from a bytes object.
    Return the unpacked root object (which usually is a dictionary).
    """
    fp = BytesIO(value)
    return load(fp, fmt=fmt, dict_type=dict_type)


def dump(value, fp, *, fmt=FMT_XML, sort_keys=True, skipkeys=False):
    """Write 'value' to a .plist file. 'fp' should be a writable,
    binary file object.
    """
    if fmt not in _FORMATS:
        raise ValueError("Unsupported format: %r"%(fmt,))

    writer = _FORMATS[fmt]["writer"](fp, sort_keys=sort_keys, skipkeys=skipkeys)
    writer.write(value)


def dumps(value, *, fmt=FMT_XML, skipkeys=False, sort_keys=True):
    """Return a bytes object with the contents for a .plist file.
    """
    fp = BytesIO()
    dump(value, fp, fmt=fmt, skipkeys=skipkeys, sort_keys=sort_keys)
    return fp.getvalue()
<|endoftext|>#! /usr/bin/env python3

"""Python interface for the 'lsprof' profiler.
   Compatible with the 'profile' module.
"""

__all__ = ["run", "runctx", "Profile"]

import _lsprof
import io
import profile as _pyprofile

# ____________________________________________________________
# Simple interface

def run(statement, filename=None, sort=-1):
    return _pyprofile._Utils(Profile).run(statement, filename, sort)

def runctx(statement, globals, locals, filename=None, sort=-1):
    return _pyprofile._Utils(Profile).runctx(statement, globals, locals,
                                             filename, sort)

run.__doc__ = _pyprofile.run.__doc__
runctx.__doc__ = _pyprofile.runctx.__doc__

# ____________________________________________________________

class Profile(_lsprof.Profiler):
    """Profile(timer=None, timeunit=None, subcalls=True, builtins=True)

    Builds a profiler object using the specified timer function.
    The default timer is a fast built-in one based on real time.
    For custom timer functions returning integers, timeunit can
    be a float specifying a scale (i.e. how long each integer unit
    is, in seconds).
    """

    # Most of the functionality is in the base class.
    # This subclass only adds convenient and backward-compatible methods.

    def print_stats(self, sort=-1):
        import pstats
        pstats.Stats(self).strip_dirs().sort_stats(sort).print_stats()

    def dump_stats(self, file):
        import marshal
        with open(file, 'wb') as f:
            self.create_stats()
            marshal.dump(self.stats, f)

    def create_stats(self):
        self.disable()
        self.snapshot_stats()

    def snapshot_stats(self):
        entries = self.getstats()
        self.stats = {}
        callersdicts = {}
        # call information
        for entry in entries:
            func = label(entry.code)
            nc = entry.callcount         # ncalls column of pstats (before '/')
            cc = nc - entry.reccallcount # ncalls column of pstats (after '/')
            tt = entry.inlinetime        # tottime column of pstats
            ct = entry.totaltime         # cumtime column of pstats
            callers = {}
            callersdicts[id(entry.code)] = callers
            self.stats[func] = cc, nc, tt, ct, callers
        # subcall information
        for entry in entries:
            if entry.calls:
                func = label(entry.code)
                for subentry in entry.calls:
                    try:
                        callers = callersdicts[id(subentry.code)]
                    except KeyError:
                        continue
                    nc = subentry.callcount
                    cc = nc - subentry.reccallcount
                    tt = subentry.inlinetime
                    ct = subentry.totaltime
                    if func in callers:
                        prev = callers[func]
                        nc += prev[0]
                        cc += prev[1]
                        tt += prev[2]
                        ct += prev[3]
                    callers[func] = nc, cc, tt, ct

    # The following two methods can be called by clients to use
    # a profiler to profile a statement, given as a string.

    def run(self, cmd):
        import __main__
        dict = __main__.__dict__
        return self.runctx(cmd, dict, dict)

    def runctx(self, cmd, globals, locals):
        self.enable()
        try:
            exec(cmd, globals, locals)
        finally:
            self.disable()
        return self

    # This method is more useful to profile a single function call.
    def runcall(self, func, /, *args, **kw):
        self.enable()
        try:
            return func(*args, **kw)
        finally:
            self.disable()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *exc_info):
        self.disable()

# ____________________________________________________________

def label(code):
    if isinstance(code, str):
        return ('~', 0, code)    # built-in functions ('~' sorts at the end)
    else:
        return (code.co_filename, code.co_firstlineno, code.co_name)

# ____________________________________________________________

def main():
    import os
    import sys
    import runpy
    import pstats
    from optparse import OptionParser
    usage = "cProfile.py [-o output_file_path] [-s sort] [-m module | scriptfile] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-o', '--outfile', dest="outfile",
        help="Save stats to <outfile>", default=None)
    parser.add_option('-s', '--sort', dest="sort",
        help="Sort order when printing to stdout, based on pstats.Stats class",
        default=-1,
        choices=sorted(pstats.Stats.sort_arg_dict_default))
    parser.add_option('-m', dest="module", action="store_true",
        help="Profile a library module", default=False)

    if not sys.argv[1:]:
        parser.print_usage()
        sys.exit(2)

    (options, args) = parser.parse_args()
    sys.argv[:] = args

    # The script that we're profiling may chdir, so capture the absolute path
    # to the output file at startup.
    if options.outfile is not None:
        options.outfile = os.path.abspath(options.outfile)

    if len(args) > 0:
        if options.module:
            code = "run_module(modname, run_name='__main__')"
            globs = {
                'run_module': runpy.run_module,
                'modname': args[0]
            }
        else:
            progname = args[0]
            sys.path.insert(0, os.path.dirname(progname))
            with io.open_code(progname) as fp:
                code = compile(fp.read(), progname, 'exec')
            globs = {
                '__file__': progname,
                '__name__': '__main__',
                '__package__': None,
                '__cached__': None,
            }
        try:
            runctx(code, globs, None, options.outfile, options.sort)
        except BrokenPipeError as exc:
            # Prevent "Exception ignored" during interpreter shutdown.
            sys.stdout = None
            sys.exit(exc.errno)
    else:
        parser.print_usage()
    return parser

# When invoked as main program, invoke the profiler on a script
if __name__ == '__main__':
    main()
<|endoftext|>"""Shared AIX support functions."""

import sys
import sysconfig

try:
    import subprocess
except ImportError:  # pragma: no cover
    # _aix_support is used in distutils by setup.py to build C extensions,
    # before subprocess dependencies like _posixsubprocess are available.
    import _bootsubprocess as subprocess


def _aix_tag(vrtl, bd):
    # type: (List[int], int) -> str
    # Infer the ABI bitwidth from maxsize (assuming 64 bit as the default)
    _sz = 32 if sys.maxsize == (2**31-1) else 64
    # vrtl[version, release, technology_level]
    return "aix-{:1x}{:1d}{:02d}-{:04d}-{}".format(vrtl[0], vrtl[1], vrtl[2], bd, _sz)


# extract version, release and technology level from a VRMF string
def _aix_vrtl(vrmf):
    # type: (str) -> List[int]
    v, r, tl = vrmf.split(".")[:3]
    return [int(v[-1]), int(r), int(tl)]


def _aix_bosmp64():
    # type: () -> Tuple[str, int]
    """
    Return a Tuple[str, int] e.g., ['7.1.4.34', 1806]
    The fileset bos.mp64 is the AIX kernel. It's VRMF and builddate
    reflect the current ABI levels of the runtime environment.
    """
    # We expect all AIX systems to have lslpp installed in this location
    out = subprocess.check_output(["/usr/bin/lslpp", "-Lqc", "bos.mp64"])
    out = out.decode("utf-8")
    out = out.strip().split(":")  # type: ignore
    # Use str() and int() to help mypy see types
    return (str(out[2]), int(out[-1]))


def aix_platform():
    # type: () -> str
    """
    AIX filesets are identified by four decimal values: V.R.M.F.
    V (version) and R (release) can be retreived using ``uname``
    Since 2007, starting with AIX 5.3 TL7, the M value has been
    included with the fileset bos.mp64 and represents the Technology
    Level (TL) of AIX. The F (Fix) value also increases, but is not
    relevant for comparing releases and binary compatibility.
    For binary compatibility the so-called builddate is needed.
    Again, the builddate of an AIX release is associated with bos.mp64.
    AIX ABI compatibility is described  as guaranteed at: https://www.ibm.com/\
    support/knowledgecenter/en/ssw_aix_72/install/binary_compatability.html

    For pep425 purposes the AIX platform tag becomes:
    "aix-{:1x}{:1d}{:02d}-{:04d}-{}".format(v, r, tl, builddate, bitsize)
    e.g., "aix-6107-1415-32" for AIX 6.1 TL7 bd 1415, 32-bit
    and, "aix-6107-1415-64" for AIX 6.1 TL7 bd 1415, 64-bit
    """
    vrmf, bd = _aix_bosmp64()
    return _aix_tag(_aix_vrtl(vrmf), bd)


# extract vrtl from the BUILD_GNU_TYPE as an int
def _aix_bgt():
    # type: () -> List[int]
    gnu_type = sysconfig.get_config_var("BUILD_GNU_TYPE")
    if not gnu_type:
        raise ValueError("BUILD_GNU_TYPE is not defined")
    return _aix_vrtl(vrmf=gnu_type)


def aix_buildtag():
    # type: () -> str
    """
    Return the platform_tag of the system Python was built on.
    """
    # AIX_BUILDDATE is defined by configure with:
    # lslpp -Lcq bos.mp64 | awk -F:  '{ print $NF }'
    build_date = sysconfig.get_config_var("AIX_BUILDDATE")
    try:
        build_date = int(build_date)
    except (ValueError, TypeError):
        raise ValueError(f"AIX_BUILDDATE is not defined or invalid: "
                         f"{build_date!r}")
    return _aix_tag(_aix_bgt(), build_date)
<|endoftext|>"""Exception classes raised by urllib.

The base exception class is URLError, which inherits from OSError.  It
doesn't define any behavior of its own, but is the base class for all
exceptions defined in this package.

HTTPError is an exception class that is also a valid HTTP response
instance.  It behaves this way because HTTP protocol errors are valid
responses, with a status code, headers, and a body.  In some contexts,
an application may want to handle an exception like a regular
response.
"""
import io
import urllib.response

__all__ = ['URLError', 'HTTPError', 'ContentTooShortError']


class URLError(OSError):
    # URLError is a sub-type of OSError, but it doesn't share any of
    # the implementation.  need to override __init__ and __str__.
    # It sets self.args for compatibility with other OSError
    # subclasses, but args doesn't have the typical format with errno in
    # slot 0 and strerror in slot 1.  This may be better than nothing.
    def __init__(self, reason, filename=None):
        self.args = reason,
        self.reason = reason
        if filename is not None:
            self.filename = filename

    def __str__(self):
        return '<urlopen error %s>' % self.reason


class HTTPError(URLError, urllib.response.addinfourl):
    """Raised when HTTP error occurs, but also acts like non-error return"""
    __super_init = urllib.response.addinfourl.__init__

    def __init__(self, url, code, msg, hdrs, fp):
        self.code = code
        self.msg = msg
        self.hdrs = hdrs
        self.fp = fp
        self.filename = url
        if fp is None:
            fp = io.BytesIO()
        self.__super_init(fp, hdrs, url, code)

    def __str__(self):
        return 'HTTP Error %s: %s' % (self.code, self.msg)

    def __repr__(self):
        return '<HTTPError %s: %r>' % (self.code, self.msg)

    # since URLError specifies a .reason attribute, HTTPError should also
    #  provide this attribute. See issue13211 for discussion.
    @property
    def reason(self):
        return self.msg

    @property
    def headers(self):
        return self.hdrs

    @headers.setter
    def headers(self, headers):
        self.hdrs = headers


class ContentTooShortError(URLError):
    """Exception raised when downloaded size does not match content-length."""
    def __init__(self, message, content):
        URLError.__init__(self, message)
        self.content = content
<|endoftext|>#
# big5.py: Python Unicode Codec for BIG5
#
# Written by Hye-Shik Chang <perky@FreeBSD.org>
#

import _codecs_tw, codecs
import _multibytecodec as mbc

codec = _codecs_tw.getcodec('big5')

class Codec(codecs.Codec):
    encode = codec.encode
    decode = codec.decode

class IncrementalEncoder(mbc.MultibyteIncrementalEncoder,
                         codecs.IncrementalEncoder):
    codec = codec

class IncrementalDecoder(mbc.MultibyteIncrementalDecoder,
                         codecs.IncrementalDecoder):
    codec = codec

class StreamReader(Codec, mbc.MultibyteStreamReader, codecs.StreamReader):
    codec = codec

class StreamWriter(Codec, mbc.MultibyteStreamWriter, codecs.StreamWriter):
    codec = codec

def getregentry():
    return codecs.CodecInfo(
        name='big5',
        encode=Codec().encode,
        decode=Codec().decode,
        incrementalencoder=IncrementalEncoder,
        incrementaldecoder=IncrementalDecoder,
        streamreader=StreamReader,
        streamwriter=StreamWriter,
    )
<|endoftext|>