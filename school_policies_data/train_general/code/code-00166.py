            compiled = re.compile(r"^([\w., ]+=)?\s*[\w\.]+\(.*\)$")
            if compiled.match(summary_str):
                self["Signature"] = summary_str
                if not self._is_at_section():
                    continue
            break

        if summary is not None:
            self["Summary"] = summary

        if not self._is_at_section():
            self["Extended Summary"] = self._read_to_next_section()

    def _parse(self):
        self._doc.reset()
        self._parse_summary()

        sections = list(self._read_sections())
        section_names = {section for section, content in sections}

        has_yields = "Yields" in section_names
        # We could do more tests, but we are not. Arbitrarily.
        if not has_yields and "Receives" in section_names:
            msg = "Docstring contains a Receives section but not Yields."
            raise ValueError(msg)

        for section, content in sections:
            if not section.startswith(".."):
                section = (s.capitalize() for s in section.split(" "))
                section = " ".join(section)
                if self.get(section):
                    self._error_location(
                        "The section %s appears twice in  %s"
                        % (section, "\n".join(self._doc._str))
                    )

            if section in ("Parameters", "Other Parameters", "Attributes", "Methods"):
                self[section] = self._parse_param_list(content)
            elif section in ("Returns", "Yields", "Raises", "Warns", "Receives"):
                self[section] = self._parse_param_list(
                    content, single_element_is_type=True
                )
            elif section.startswith(".. index::"):
                self["index"] = self._parse_index(section, content)
            elif section == "See Also":
                self["See Also"] = self._parse_see_also(content)
            else:
                self[section] = content

    @property
    def _obj(self):
        if hasattr(self, "_cls"):
            return self._cls
        elif hasattr(self, "_f"):
            return self._f
        return None

    def _error_location(self, msg, error=True):
        if self._obj is not None:
            # we know where the docs came from:
            try:
                filename = inspect.getsourcefile(self._obj)
            except TypeError:
                filename = None
            # Make UserWarning more descriptive via object introspection.
            # Skip if introspection fails
            name = getattr(self._obj, "__name__", None)
            if name is None:
                name = getattr(getattr(self._obj, "__class__", None), "__name__", None)
            if name is not None:
                msg += f" in the docstring of {name}"
            msg += f" in {filename}." if filename else ""
        if error:
            raise ValueError(msg)
        else:
            warn(msg, stacklevel=3)

    # string conversion routines

    def _str_header(self, name, symbol="-"):
        return [name, len(name) * symbol]

    def _str_indent(self, doc, indent=4):
        return [" " * indent + line for line in doc]

    def _str_signature(self):
        if self["Signature"]:
            return [self["Signature"].replace("*", r"\*")] + [""]
        return [""]

    def _str_summary(self):
        if self["Summary"]:
            return self["Summary"] + [""]
        return []

    def _str_extended_summary(self):
        if self["Extended Summary"]:
            return self["Extended Summary"] + [""]
        return []

    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            for param in self[name]:
                parts = []
                if param.name:
                    parts.append(param.name)
                if param.type:
                    parts.append(param.type)
                out += [" : ".join(parts)]
                if param.desc and "".join(param.desc).strip():
                    out += self._str_indent(param.desc)
            out += [""]
        return out

    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += [""]
        return out

    def _str_see_also(self, func_role):
        if not self["See Also"]:
            return []
        out = []
        out += self._str_header("See Also")
        out += [""]
        last_had_desc = True
        for funcs, desc in self["See Also"]:
            assert isinstance(funcs, list)
            links = []
            for func, role in funcs:
                if role:
                    link = f":{role}:`{func}`"
                elif func_role:
                    link = f":{func_role}:`{func}`"
                else:
                    link = f"`{func}`_"
                links.append(link)
            link = ", ".join(links)
            out += [link]
            if desc:
                out += self._str_indent([" ".join(desc)])
                last_had_desc = True
            else:
                last_had_desc = False
                out += self._str_indent([self.empty_description])

        if last_had_desc:
            out += [""]
        out += [""]
        return out

    def _str_index(self):
        idx = self["index"]
        out = []
        output_index = False
        default_index = idx.get("default", "")
        if default_index:
            output_index = True
        out += [f".. index:: {default_index}"]
        for section, references in idx.items():
            if section == "default":
                continue
            output_index = True
            out += [f"   :{section}: {', '.join(references)}"]
        if output_index:
            return out
        return ""

    def __str__(self, func_role=""):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        out += self._str_param_list("Parameters")
        for param_list in ("Attributes", "Methods"):
            out += self._str_param_list(param_list)
        for param_list in (
            "Returns",
            "Yields",
            "Receives",
            "Other Parameters",
            "Raises",
            "Warns",
        ):
            out += self._str_param_list(param_list)
        out += self._str_section("Warnings")
        out += self._str_see_also(func_role)
        for s in ("Notes", "References", "Examples"):
            out += self._str_section(s)
        out += self._str_index()
        return "\n".join(out)


def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    return textwrap.dedent("\n".join(lines)).split("\n")


class FunctionDoc(NumpyDocString):
    def __init__(self, func, role="func", doc=None, config=None):
        self._f = func
        self._role = role  # e.g. "func" or "meth"

        if doc is None:
            if func is None:
                raise ValueError("No function or docstring given")
            doc = inspect.getdoc(func) or ""
        if config is None:
            config = {}
        NumpyDocString.__init__(self, doc, config)

    def get_func(self):
        func_name = getattr(self._f, "__name__", self.__class__.__name__)
        if inspect.isclass(self._f):
            func = getattr(self._f, "__call__", self._f.__init__)
        else:
            func = self._f
        return func, func_name

    def __str__(self):
        out = ""

        func, func_name = self.get_func()

        roles = {"func": "function", "meth": "method"}

        if self._role:
            if self._role not in roles:
                print(f"Warning: invalid role {self._role}")
            out += f".. {roles.get(self._role, '')}:: {func_name}\n    \n\n"

        out += super().__str__(func_role=self._role)
        return out


class ObjDoc(NumpyDocString):
    def __init__(self, obj, doc=None, config=None):
        self._f = obj
        if config is None:
            config = {}
        NumpyDocString.__init__(self, doc, config=config)


class ClassDoc(NumpyDocString):
    extra_public_methods = ["__call__"]

    def __init__(self, cls, doc=None, modulename="", func_doc=FunctionDoc, config=None):
        if not inspect.isclass(cls) and cls is not None:
            raise ValueError(f"Expected a class or None, but got {cls!r}")
        self._cls = cls

        if "sphinx" in sys.modules:
            from sphinx.ext.autodoc import ALL
        else:
            ALL = object()

        if config is None:
            config = {}
        self.show_inherited_members = config.get("show_inherited_class_members", True)

        if modulename and not modulename.endswith("."):
            modulename += "."
        self._mod = modulename

        if doc is None:
            if cls is None:
                raise ValueError("No class or documentation string given")
            doc = pydoc.getdoc(cls)

        NumpyDocString.__init__(self, doc)

        _members = config.get("members", [])
        if _members is ALL:
            _members = None
        _exclude = config.get("exclude-members", [])

        if config.get("show_class_members", True) and _exclude is not ALL:

            def splitlines_x(s):
                if not s:
                    return []
                else:
                    return s.splitlines()

            for field, items in [
                ("Methods", self.methods),
                ("Attributes", self.properties),
            ]:
                if not self[field]:
                    doc_list = []
                    for name in sorted(items):
                        if name in _exclude or (_members and name not in _members):
                            continue
                        try:
                            doc_item = pydoc.getdoc(getattr(self._cls, name))
                            doc_list.append(Parameter(name, "", splitlines_x(doc_item)))
                        except AttributeError:
                            pass  # method doesn't exist
                    self[field] = doc_list

    @property
    def methods(self):
        if self._cls is None:
            return []
        return [
            name
            for name, func in inspect.getmembers(self._cls)
            if (
                (not name.startswith("_") or name in self.extra_public_methods)
                and isinstance(func, Callable)
                and self._is_show_member(name)
            )
        ]

    @property
    def properties(self):
        if self._cls is None:
            return []
        return [
            name
            for name, func in inspect.getmembers(self._cls)
            if (
                not name.startswith("_")
                and not self._should_skip_member(name, self._cls)
                and (
                    func is None
                    or isinstance(func, (property, cached_property))
                    or inspect.isdatadescriptor(func)
                )
                and self._is_show_member(name)
            )
        ]

    @staticmethod
    def _should_skip_member(name, klass):
        return (
            # Namedtuples should skip everything in their ._fields as the
            # docstrings for each of the members is: "Alias for field number X"
            issubclass(klass, tuple)
            and hasattr(klass, "_asdict")
            and hasattr(klass, "_fields")
            and name in klass._fields
        )

    def _is_show_member(self, name):
        return (
            # show all class members
            self.show_inherited_members
            # or class member is not inherited
            or name in self._cls.__dict__
        )


def get_doc_object(
    obj,
    what=None,
    doc=None,
    config=None,
    class_doc=ClassDoc,
    func_doc=FunctionDoc,
    obj_doc=ObjDoc,
):
    if what is None:
        if inspect.isclass(obj):
            what = "class"
        elif inspect.ismodule(obj):
            what = "module"
        elif isinstance(obj, Callable):
            what = "function"
        else:
            what = "object"
    if config is None:
        config = {}

    if what == "class":
        return class_doc(obj, func_doc=func_doc, doc=doc, config=config)
    elif what in ("function", "method"):
        return func_doc(obj, doc=doc, config=config)
    else:
        if doc is None:
            doc = pydoc.getdoc(obj)
        return obj_doc(obj, doc, config=config)
<|endoftext|>"""Utilities to allow inserting docstring fragments for common
parameters into function and method docstrings."""

from collections.abc import Callable, Iterable, Mapping
from typing import Protocol, TypeVar
import sys

__all__ = [
    "docformat",
    "inherit_docstring_from",
    "indentcount_lines",
    "filldoc",
    "unindent_dict",
    "unindent_string",
    "extend_notes_in_docstring",
    "replace_notes_in_docstring",
    "doc_replace",
]

_F = TypeVar("_F", bound=Callable[..., object])


class Decorator(Protocol):
    """A decorator of a function."""

    def __call__(self, func: _F, /) -> _F: ...


def docformat(docstring: str, docdict: Mapping[str, str] | None = None) -> str:
    """Fill a function docstring from variables in dictionary.

    Adapt the indent of the inserted docs

    Parameters
    ----------
    docstring : str
        A docstring from a function, possibly with dict formatting strings.
    docdict : dict[str, str], optional
        A dictionary with keys that match the dict formatting strings
        and values that are docstring fragments to be inserted. The
        indentation of the inserted docstrings is set to match the
        minimum indentation of the ``docstring`` by adding this
        indentation to all lines of the inserted string, except the
        first.

    Returns
    -------
    docstring : str
        string with requested ``docdict`` strings inserted.

    Examples
    --------
    >>> docformat(' Test string with %(value)s', {'value':'inserted value'})
    ' Test string with inserted value'
    >>> docstring = 'First line\\n    Second line\\n    %(value)s'
    >>> inserted_string = "indented\\nstring"
    >>> docdict = {'value': inserted_string}
    >>> docformat(docstring, docdict)
    'First line\\n    Second line\\n    indented\\n    string'
    """
    if not docstring:
        return docstring
    if docdict is None:
        docdict = {}
    if not docdict:
        return docstring
    lines = docstring.expandtabs().splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = indentcount_lines(lines[1:])
    indent = " " * icount
    # Insert this indent to dictionary docstrings
    indented = {}
    for name, dstr in docdict.items():
        lines = dstr.expandtabs().splitlines()
        try:
            newlines = [lines[0]]
            for line in lines[1:]:
                newlines.append(indent + line)
            indented[name] = "\n".join(newlines)
        except IndexError:
            indented[name] = dstr
    return docstring % indented


def inherit_docstring_from(cls: object) -> Decorator:
    """This decorator modifies the decorated function's docstring by
    replacing occurrences of '%(super)s' with the docstring of the
    method of the same name from the class `cls`.

    If the decorated method has no docstring, it is simply given the
    docstring of `cls`s method.

    Parameters
    ----------
    cls : type or object
        A class with a method with the same name as the decorated method.
        The docstring of the method in this class replaces '%(super)s' in the
        docstring of the decorated method.

    Returns
    -------
    decfunc : function
        The decorator function that modifies the __doc__ attribute
        of its argument.

    Examples
    --------
    In the following, the docstring for Bar.func created using the
    docstring of `Foo.func`.

    >>> class Foo:
    ...     def func(self):
    ...         '''Do something useful.'''
    ...         return
    ...
    >>> class Bar(Foo):
    ...     @inherit_docstring_from(Foo)
    ...     def func(self):
    ...         '''%(super)s
    ...         Do it fast.
    ...         '''
    ...         return
    ...
    >>> b = Bar()
    >>> b.func.__doc__
    'Do something useful.\n        Do it fast.\n        '
    """

    def _doc(func: _F) -> _F:
        cls_docstring = getattr(cls, func.__name__).__doc__
        func_docstring = func.__doc__
        if func_docstring is None:
            func.__doc__ = cls_docstring
        else:
            new_docstring = func_docstring % dict(super=cls_docstring)
            func.__doc__ = new_docstring
        return func

    return _doc


def extend_notes_in_docstring(cls: object, notes: str) -> Decorator:
    """This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It extends the 'Notes' section of that docstring to include
    the given `notes`.

    Parameters
    ----------
    cls : type or object
        A class with a method with the same name as the decorated method.
        The docstring of the method in this class replaces the docstring of the
        decorated method.
    notes : str
        Additional notes to append to the 'Notes' section of the docstring.

    Returns
    -------
    decfunc : function
        The decorator function that modifies the __doc__ attribute
        of its argument.
    """

    def _doc(func: _F) -> _F:
        cls_docstring = getattr(cls, func.__name__).__doc__
        # If python is called with -OO option,
        # there is no docstring
        if cls_docstring is None:
            return func
        end_of_notes = cls_docstring.find("        References\n")
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find("        Examples\n")
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        func.__doc__ = (
            cls_docstring[:end_of_notes] + notes + cls_docstring[end_of_notes:]
        )
        return func

    return _doc


def replace_notes_in_docstring(cls: object, notes: str) -> Decorator:
    """This decorator replaces the decorated function's docstring
    with the docstring from corresponding method in `cls`.
    It replaces the 'Notes' section of that docstring with
    the given `notes`.

    Parameters
    ----------
    cls : type or object
        A class with a method with the same name as the decorated method.
        The docstring of the method in this class replaces the docstring of the
        decorated method.
    notes : str
        The notes to replace the existing 'Notes' section with.

    Returns
    -------
    decfunc : function
        The decorator function that modifies the __doc__ attribute
        of its argument.
    """

    def _doc(func: _F) -> _F:
        cls_docstring = getattr(cls, func.__name__).__doc__
        notes_header = "        Notes\n        -----\n"
        # If python is called with -OO option,
        # there is no docstring
        if cls_docstring is None:
            return func
        start_of_notes = cls_docstring.find(notes_header)
        end_of_notes = cls_docstring.find("        References\n")
        if end_of_notes == -1:
            end_of_notes = cls_docstring.find("        Examples\n")
            if end_of_notes == -1:
                end_of_notes = len(cls_docstring)
        func.__doc__ = (
            cls_docstring[: start_of_notes + len(notes_header)]
            + notes
            + cls_docstring[end_of_notes:]
        )
        return func

    return _doc


def indentcount_lines(lines: Iterable[str]) -> int:
    """Minimum indent for all lines in line list

    Parameters
    ----------
    lines : Iterable[str]
        The lines to find the minimum indent of.

    Returns
    -------
    indent : int
        The minimum indent.


    Examples
    --------
    >>> lines = [' one', '  two', '   three']
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [' one']
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(['    '])
    0
    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def filldoc(docdict: Mapping[str, str], unindent_params: bool = True) -> Decorator:
    """Return docstring decorator using docdict variable dictionary.

    Parameters
    ----------
    docdict : dict[str, str]
        A dictionary containing name, docstring fragment pairs.
    unindent_params : bool, optional
        If True, strip common indentation from all parameters in docdict.
        Default is False.

    Returns
    -------
    decfunc : function
        The decorator function that applies dictionary to its
        argument's __doc__ attribute.
    """
    if unindent_params:
        docdict = unindent_dict(docdict)

    def decorate(func: _F) -> _F:
        # __doc__ may be None for optimized Python (-OO)
        doc = func.__doc__ or ""
        func.__doc__ = docformat(doc, docdict)
        return func

    return decorate


def unindent_dict(docdict: Mapping[str, str]) -> dict[str, str]:
    """Unindent all strings in a docdict.

    Parameters
    ----------
    docdict : dict[str, str]
        A dictionary with string values to unindent.

    Returns
    -------
    docdict : dict[str, str]
        The `docdict` dictionary but each of its string values are unindented.
    """
    can_dict: dict[str, str] = {}
    for name, dstr in docdict.items():
        can_dict[name] = unindent_string(dstr)
    return can_dict


def unindent_string(docstring: str) -> str:
    """Set docstring to minimum indent for all lines, including first.

    Parameters
    ----------
    docstring : str
        The input docstring to unindent.

    Returns
    -------
    docstring : str
        The unindented docstring.

    Examples
    --------
    >>> unindent_string(' two')
    'two'
    >>> unindent_string('  two\\n   three')
    'two\\n three'
    """
    lines = docstring.expandtabs().splitlines()
    icount = indentcount_lines(lines)
    if icount == 0:
        return docstring
    return "\n".join([line[icount:] for line in lines])


def doc_replace(obj: object, oldval: str, newval: str) -> Decorator:
    """Decorator to take the docstring from obj, with oldval replaced by newval

    Equivalent to ``func.__doc__ = obj.__doc__.replace(oldval, newval)``

    Parameters
    ----------
    obj : object
        A class or object whose docstring will be used as the basis for the
        replacement operation.
    oldval : str
        The string to search for in the docstring.
    newval : str
        The string to replace `oldval` with in the docstring.

    Returns
    -------
    decfunc : function
        A decorator function that replaces occurrences of `oldval` with `newval`
        in the docstring of the decorated function.
    """
    # __doc__ may be None for optimized Python (-OO)
    doc = (obj.__doc__ or "").replace(oldval, newval)

    def inner(func: _F) -> _F:
        func.__doc__ = doc
        return func

    return inner
<|endoftext|>"""
Disjoint set data structure
"""


class DisjointSet:
    """ Disjoint set data structure for incremental connectivity queries.

    .. versionadded:: 1.6.0

    Attributes
    ----------
    n_subsets : int
        The number of subsets.

    Methods
    -------
    add
    merge
    connected
    subset
    subset_size
    subsets
    __getitem__

    Notes
    -----
    This class implements the disjoint set [1]_, also known as the *union-find*
    or *merge-find* data structure. The *find* operation (implemented in
    `__getitem__`) implements the *path halving* variant. The *merge* method
    implements the *merge by size* variant.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    Examples
    --------
    >>> from scipy.cluster.hierarchy import DisjointSet

    Initialize a disjoint set:

    >>> disjoint_set = DisjointSet([1, 2, 3, 'a', 'b'])

    Merge some subsets:

    >>> disjoint_set.merge(1, 2)
    True
    >>> disjoint_set.merge(3, 'a')
    True
    >>> disjoint_set.merge('a', 'b')
    True
    >>> disjoint_set.merge('b', 'b')
    False

    Find root elements:

    >>> disjoint_set[2]
    1
    >>> disjoint_set['b']
    3

    Test connectivity:

    >>> disjoint_set.connected(1, 2)
    True
    >>> disjoint_set.connected(1, 'b')
    False

    List elements in disjoint set:

    >>> list(disjoint_set)
    [1, 2, 3, 'a', 'b']

    Get the subset containing 'a':

    >>> disjoint_set.subset('a')
    {'a', 3, 'b'}

    Get the size of the subset containing 'a' (without actually instantiating
    the subset):

    >>> disjoint_set.subset_size('a')
    3

    Get all subsets in the disjoint set:

    >>> disjoint_set.subsets()
    [{1, 2}, {'a', 3, 'b'}]
    """
    def __init__(self, elements=None):
        self.n_subsets = 0
        self._sizes = {}
        self._parents = {}
        # _nbrs is a circular linked list which links connected elements.
        self._nbrs = {}
        # _indices tracks the element insertion order in `__iter__`.
        self._indices = {}
        if elements is not None:
            for x in elements:
                self.add(x)

    def __iter__(self):
        """Returns an iterator of the elements in the disjoint set.

        Elements are ordered by insertion order.
        """
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

    def __contains__(self, x):
        return x in self._indices

    def __getitem__(self, x):
        """Find the root element of `x`.

        Parameters
        ----------
        x : hashable object
            Input element.

        Returns
        -------
        root : hashable object
            Root element of `x`.
        """
        if x not in self._indices:
            raise KeyError(x)

        # find by "path halving"
        parents = self._parents
        while self._indices[x] != self._indices[parents[x]]:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def add(self, x):
        """Add element `x` to disjoint set
        """
        if x in self._indices:
            return

        self._sizes[x] = 1
        self._parents[x] = x
        self._nbrs[x] = x
        self._indices[x] = len(self._indices)
        self.n_subsets += 1

    def merge(self, x, y):
        """Merge the subsets of `x` and `y`.

        The smaller subset (the child) is merged into the larger subset (the
        parent). If the subsets are of equal size, the root element which was
        first inserted into the disjoint set is selected as the parent.

        Parameters
        ----------
        x, y : hashable object
            Elements to merge.

        Returns
        -------
        merged : bool
            True if `x` and `y` were in disjoint sets, False otherwise.
        """
        xr = self[x]
        yr = self[y]
        if self._indices[xr] == self._indices[yr]:
            return False

        sizes = self._sizes
        if (sizes[xr], self._indices[yr]) < (sizes[yr], self._indices[xr]):
            xr, yr = yr, xr
        self._parents[yr] = xr
        self._sizes[xr] += self._sizes[yr]
        self._nbrs[xr], self._nbrs[yr] = self._nbrs[yr], self._nbrs[xr]
        self.n_subsets -= 1
        return True

    def connected(self, x, y):
        """Test whether `x` and `y` are in the same subset.

        Parameters
        ----------
        x, y : hashable object
            Elements to test.

        Returns
        -------
        result : bool
            True if `x` and `y` are in the same set, False otherwise.
        """
        return self._indices[self[x]] == self._indices[self[y]]

    def subset(self, x):
        """Get the subset containing `x`.

        Parameters
        ----------
        x : hashable object
            Input element.

        Returns
        -------
        result : set
            Subset containing `x`.
        """
        if x not in self._indices:
            raise KeyError(x)

        result = [x]
        nxt = self._nbrs[x]
        while self._indices[nxt] != self._indices[x]:
            result.append(nxt)
            nxt = self._nbrs[nxt]
        return set(result)

    def subset_size(self, x):
        """Get the size of the subset containing `x`.

        Note that this method is faster than ``len(self.subset(x))`` because
        the size is directly read off an internal field, without the need to
        instantiate the full subset.

        Parameters
        ----------
        x : hashable object
            Input element.

        Returns
        -------
        result : int
            Size of the subset containing `x`.
        """
        return self._sizes[self[x]]

    def subsets(self):
        """Get all the subsets in the disjoint set.

        Returns
        -------
        result : list
            Subsets in the disjoint set.
        """
        result = []
        visited = set()
        for x in self:
            if x not in visited:
                xset = self.subset(x)
                visited.update(xset)
                result.append(xset)
        return result
<|endoftext|>from numpy import arange, newaxis, hstack, prod, array


def _central_diff_weights(Np, ndiv=1):
    """
    Return weights for an Np-point central derivative.

    Assumes equally-spaced function points.

    If weights are in the vector w, then
    derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)

    Parameters
    ----------
    Np : int
        Number of points for the central derivative.
    ndiv : int, optional
        Number of divisions. Default is 1.

    Returns
    -------
    w : ndarray
        Weights for an Np-point central derivative. Its size is `Np`.

    Notes
    -----
    Can be inaccurate for a large number of points.

    Examples
    --------
    We can calculate a derivative value of a function.

    >>> def f(x):
    ...     return 2 * x**2 + 3
    >>> x = 3.0 # derivative point
    >>> h = 0.1 # differential step
    >>> Np = 3 # point number for central derivative
    >>> weights = _central_diff_weights(Np) # weights for first derivative
    >>> vals = [f(x + (i - Np/2) * h) for i in range(Np)]
    >>> sum(w * v for (w, v) in zip(weights, vals))/h
    11.79999999999998

    This value is close to the analytical solution:
    f'(x) = 4x, so f'(3) = 12

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Finite_difference

    """
    if Np < ndiv + 1:
        raise ValueError(
            "Number of points must be at least the derivative order + 1."
        )
    if Np % 2 == 0:
        raise ValueError("The number of points must be odd.")
    from scipy import linalg

    ho = Np >> 1
    x = arange(-ho, ho + 1.0)
    x = x[:, newaxis]
    X = x**0.0
    for k in range(1, Np):
        X = hstack([X, x**k])
    w = prod(arange(1, ndiv + 1), axis=0) * linalg.inv(X)[ndiv]
    return w


def _derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    """
    Find the nth derivative of a function at a point.

    Given a function, use a central difference formula with spacing `dx` to
    compute the nth derivative at `x0`.

    Parameters
    ----------
    func : function
        Input function.
    x0 : float
        The point at which the nth derivative is found.
    dx : float, optional
        Spacing.
    n : int, optional
        Order of the derivative. Default is 1.
    args : tuple, optional
        Arguments
    order : int, optional
        Number of points to use, must be odd.

    Notes
    -----
    Decreasing the step size too small can result in round-off error.

    Examples
    --------
    >>> def f(x):
    ...     return x**3 + x**2
    >>> _derivative(f, 1.0, dx=1e-6)
    4.9999999999217337

    """
    if order < n + 1:
        raise ValueError(
            "'order' (the number of points used to compute the derivative), "
            "must be at least the derivative order 'n' + 1."
        )
    if order % 2 == 0:
        raise ValueError(
            "'order' (the number of points used to compute the derivative) "
            "must be odd."
        )
    # pre-computed for n=1 and 2 and low-order for speed.
    if n == 1:
        if order == 3:
            weights = array([-1, 0, 1]) / 2.0
        elif order == 5:
            weights = array([1, -8, 0, 8, -1]) / 12.0
        elif order == 7:
            weights = array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
        elif order == 9:
            weights = array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
        else:
            weights = _central_diff_weights(order, 1)
    elif n == 2:
        if order == 3:
            weights = array([1, -2.0, 1])
        elif order == 5:
            weights = array([-1, 16, -30, 16, -1]) / 12.0
        elif order == 7:
            weights = array([2, -27, 270, -490, 270, -27, 2]) / 180.0
        elif order == 9:
            weights = (
                array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9])
                / 5040.0
            )
        else:
            weights = _central_diff_weights(order, 2)
    else:
        weights = _central_diff_weights(order, n)
    val = 0.0
    ho = order >> 1
    for k in range(order):
        val += weights[k] * func(x0 + (k - ho) * dx, *args)
    return val / prod((dx,) * n, axis=0)
<|endoftext|>import threading

import scipy._lib.decorator


__all__ = ['ReentrancyError', 'ReentrancyLock', 'non_reentrant']


class ReentrancyError(RuntimeError):
    pass


class ReentrancyLock:
    """
    Threading lock that raises an exception for reentrant calls.

    Calls from different threads are serialized, and nested calls from the
    same thread result to an error.

    The object can be used as a context manager or to decorate functions
    via the decorate() method.

    """

    def __init__(self, err_msg):
        self._rlock = threading.RLock()
        self._entered = False
        self._err_msg = err_msg

    def __enter__(self):
        self._rlock.acquire()
        if self._entered:
            self._rlock.release()
            raise ReentrancyError(self._err_msg)
        self._entered = True

    def __exit__(self, type, value, traceback):
        self._entered = False
        self._rlock.release()

    def decorate(self, func):
        def caller(func, *a, **kw):
            with self:
                return func(*a, **kw)
        return scipy._lib.decorator.decorate(func, caller)


def non_reentrant(err_msg=None):
    """
    Decorate a function with a threading lock and prevent reentrant calls.
    """
    def decorator(func):
        msg = err_msg
        if msg is None:
            msg = f"{func.__name__} is not re-entrant"
        lock = ReentrancyLock(msg)
        return lock.decorate(func)
    return decorator
<|endoftext|>"""
Generic test utilities.

"""

import inspect
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import threading
from importlib.util import module_from_spec, spec_from_file_location

import numpy as np
import scipy

try:
    # Need type: ignore[import-untyped] for mypy >= 1.6
    import cython  # type: ignore[import-untyped]
    from Cython.Compiler.Version import (  # type: ignore[import-untyped]
        version as cython_version,
    )
except ImportError:
    cython = None
else:
    from scipy._lib import _pep440
    required_version = '3.0.8'
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # too old or wrong cython, skip Cython API tests
        cython = None


__all__ = ['PytestTester', 'check_free_memory', '_TestPythranFunc', 'IS_MUSL']


IS_MUSL = False
# alternate way is
# from packaging.tags import sys_tags
#     _tags = list(sys_tags())
#     if 'musllinux' in _tags[0].platform:
_v = sysconfig.get_config_var('HOST_GNU_TYPE') or ''
if 'musl' in _v:
    IS_MUSL = True


IS_EDITABLE = 'editable' in scipy.__path__[0]


class FPUModeChangeWarning(RuntimeWarning):
    """Warning about FPU mode change"""
    pass


class PytestTester:
    """
    Run tests for this namespace

    ``scipy.test()`` runs tests for all of SciPy, with the default settings.
    When used from a submodule (e.g., ``scipy.cluster.test()``, only the tests
    for that namespace are run.

    Parameters
    ----------
    label : {'fast', 'full'}, optional
        Whether to run only the fast tests, or also those marked as slow.
        Default is 'fast'.
    verbose : int, optional
        Test output verbosity. Default is 1.
    extra_argv : list, optional
        Arguments to pass through to Pytest.
    doctests : bool, optional
        Whether to run doctests or not. Default is False.
    coverage : bool, optional
        Whether to run tests with code coverage measurements enabled.
        Default is False.
    tests : list of str, optional
        List of module names to run tests for. By default, uses the module
        from which the ``test`` function is called.
    parallel : int, optional
        Run tests in parallel with pytest-xdist, if number given is larger than
        1. Default is 1.

    """
    def __init__(self, module_name):
        self.module_name = module_name

    def __call__(self, label="fast", verbose=1, extra_argv=None, doctests=False,
                 coverage=False, tests=None, parallel=None):
        import pytest

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        pytest_args = ['--showlocals', '--tb=short']

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose and int(verbose) > 1:
            pytest_args += ["-" + "v"*(int(verbose)-1)]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        if tests is None:
            tests = [self.module_name]

        if parallel is not None and parallel > 1:
            if _pytest_has_xdist():
                pytest_args += ['-n', str(parallel)]
            else:
                import warnings
                warnings.warn('Could not run tests in parallel because '
                              'pytest-xdist plugin is not available.',
                              stacklevel=2)

        pytest_args += ['--pyargs'] + list(tests)

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return (code == 0)


class _TestPythranFunc:
    '''
    These are situations that can be tested in our pythran tests:
    - A function with multiple array arguments and then
      other positional and keyword arguments.
    - A function with array-like keywords (e.g. `def somefunc(x0, x1=None)`.
    Note: list/tuple input is not yet tested!

    `self.arguments`: A dictionary which key is the index of the argument,
                      value is tuple(array value, all supported dtypes)
    `self.partialfunc`: A function used to freeze some non-array argument
                        that of no interests in the original function
    '''
    ALL_INTEGER = [np.int8, np.int16, np.int32, np.int64, np.intc, np.intp]
    ALL_FLOAT = [np.float32, np.float64]
    ALL_COMPLEX = [np.complex64, np.complex128]

    def setup_method(self):
        self.arguments = {}
        self.partialfunc = None
        self.expected = None

    def get_optional_args(self, func):
        # get optional arguments with its default value,
        # used for testing keywords
        signature = inspect.signature(func)
        optional_args = {}
        for k, v in signature.parameters.items():
            if v.default is not inspect.Parameter.empty:
                optional_args[k] = v.default
        return optional_args

    def get_max_dtype_list_length(self):
        # get the max supported dtypes list length in all arguments
        max_len = 0
        for arg_idx in self.arguments:
            cur_len = len(self.arguments[arg_idx][1])
            if cur_len > max_len:
                max_len = cur_len
        return max_len

    def get_dtype(self, dtype_list, dtype_idx):
        # get the dtype from dtype_list via index
        # if the index is out of range, then return the last dtype
        if dtype_idx > len(dtype_list)-1:
            return dtype_list[-1]
        else:
            return dtype_list[dtype_idx]

    def test_all_dtypes(self):
        for type_idx in range(self.get_max_dtype_list_length()):
            args_array = []
            for arg_idx in self.arguments:
                new_dtype = self.get_dtype(self.arguments[arg_idx][1],
                                           type_idx)
                args_array.append(self.arguments[arg_idx][0].astype(new_dtype))
            self.pythranfunc(*args_array)

    def test_views(self):
        args_array = []
        for arg_idx in self.arguments:
            args_array.append(self.arguments[arg_idx][0][::-1][::-1])
        self.pythranfunc(*args_array)

    def test_strided(self):
        args_array = []
        for arg_idx in self.arguments:
            args_array.append(np.repeat(self.arguments[arg_idx][0],
                                        2, axis=0)[::2])
        self.pythranfunc(*args_array)


def _pytest_has_xdist():
    """
    Check if the pytest-xdist plugin is installed, providing parallel tests
    """
    # Check xdist exists without importing, otherwise pytests emits warnings
    from importlib.util import find_spec
    return find_spec('xdist') is not None


def check_free_memory(free_mb):
    """
    Check *free_mb* of memory is available, otherwise do pytest.skip
    """
    import pytest

    try:
        mem_free = _parse_size(os.environ['SCIPY_AVAILABLE_MEM'])
        msg = '{} MB memory required, but environment SCIPY_AVAILABLE_MEM={}'.format(
            free_mb, os.environ['SCIPY_AVAILABLE_MEM'])
    except KeyError:
        mem_free = _get_mem_available()
        if mem_free is None:
            pytest.skip("Could not determine available memory; set SCIPY_AVAILABLE_MEM "
                        "variable to free memory in MB to run the test.")
        msg = f'{free_mb} MB memory required, but {mem_free/1e6} MB available'

    if mem_free < free_mb * 1e6:
        pytest.skip(msg)


def _parse_size(size_str):
    suffixes = {'': 1e6,
                'b': 1.0,
                'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
                'kb': 1e3, 'Mb': 1e6, 'Gb': 1e9, 'Tb': 1e12,
                'kib': 1024.0, 'Mib': 1024.0**2, 'Gib': 1024.0**3, 'Tib': 1024.0**4}
    m = re.match(r'^\s*(\d+)\s*({})\s*$'.format('|'.join(suffixes.keys())),
                 size_str,
                 re.I)
    if not m or m.group(2) not in suffixes:
        raise ValueError("Invalid size string")

    return float(m.group(1)) * suffixes[m.group(2)]


def _get_mem_available():
    """
    Get information about memory available, not counting swap.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass

    if sys.platform.startswith('linux'):
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                p = line.split()
                info[p[0].strip(':').lower()] = float(p[1]) * 1e3

        if 'memavailable' in info:
            # Linux >= 3.14
            return info['memavailable']
        else:
            return info['memfree'] + info['cached']

    return None

def _test_cython_extension(tmp_path, srcdir):
    """
    Helper function to test building and importing Cython modules that
    make use of the Cython APIs for BLAS, LAPACK, optimize, and special.
    """
    import pytest
    try:
        subprocess.check_call(["meson", "--version"])
    except FileNotFoundError:
        pytest.skip("No usable 'meson' found")

    # Make safe for being called by multiple threads within one test
    tmp_path = tmp_path / str(threading.get_ident())

    # build the examples in a temporary directory
    mod_name = os.path.split(srcdir)[1]
    shutil.copytree(srcdir, tmp_path / mod_name)
    build_dir = tmp_path / mod_name / 'tests' / '_cython_examples'
    target_dir = build_dir / 'build'
    os.makedirs(target_dir, exist_ok=True)

    # Ensure we use the correct Python interpreter even when `meson` is
    # installed in a different Python environment (see numpy#24956)
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'")

    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--native-file", native_file,
                               "--vsenv", str(build_dir)],
                              cwd=target_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup",
                               "--native-file", native_file, str(build_dir)],
                              cwd=target_dir
                              )
    subprocess.check_call(["meson", "compile", "-vv"], cwd=target_dir)

    # import without adding the directory to sys.path
    suffix = sysconfig.get_config_var('EXT_SUFFIX')

    def load(modname):
        so = (target_dir / modname).with_suffix(suffix)
        spec = spec_from_file_location(modname, so)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # test that the module can be imported
    return load("extending"), load("extending_cpp")


def _run_concurrent_barrier(n_workers, fn, *args, **kwargs):
    """
    Run a given function concurrently across a given number of threads.

    This is equivalent to using a ThreadPoolExecutor, but using the threading
    primitives instead. This function ensures that the closure passed by
    parameter gets called concurrently by setting up a barrier before it gets
    called before any of the threads.

    Arguments
    ---------
    n_workers: int
        Number of concurrent threads to spawn.
    fn: callable
        Function closure to execute concurrently. Its first argument will
        be the thread id.
    *args: tuple
        Variable number of positional arguments to pass to the function.
    **kwargs: dict
        Keyword arguments to pass to the function.
    """
    barrier = threading.Barrier(n_workers)

    def closure(i, *args, **kwargs):
        barrier.wait()
        fn(i, *args, **kwargs)

    workers = []
    for i in range(0, n_workers):
        workers.append(threading.Thread(
            target=closure,
            args=(i,) + args, kwargs=kwargs))

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
<|endoftext|>"""Utility to compare pep440 compatible version strings.

The LooseVersion and StrictVersion classes that distutils provides don't
work; they don't recognize anything like alpha/beta/rc/dev versions.
"""

# Copyright (c) Donald Stufft and individual contributors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import collections
import itertools
import re


__all__ = [
    "parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN",
]


# BEGIN packaging/_structures.py


class Infinity:
    def __repr__(self):
        return "Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __neg__(self):
        return NegativeInfinity


Infinity = Infinity()


class NegativeInfinity:
    def __repr__(self):
        return "-Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __neg__(self):
        return Infinity


# BEGIN packaging/version.py


NegativeInfinity = NegativeInfinity()

_Version = collections.namedtuple(
    "_Version",
    ["epoch", "release", "dev", "pre", "post", "local"],
)


def parse(version):
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object depending on if the given version is
    a valid PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """


class _BaseVersion:

    def __hash__(self):
        return hash(self._key)

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)

    def _compare(self, other, method):
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return method(self._key, other._key)


class LegacyVersion(_BaseVersion):

    def __init__(self, version):
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

    def __str__(self):
        return self._version

    def __repr__(self):
        return f"<LegacyVersion({repr(str(self))})>"

    @property
    def public(self):
        return self._version

    @property
    def base_version(self):
        return self._version

    @property
    def local(self):
        return None

    @property
    def is_prerelease(self):
        return False

    @property
    def is_postrelease(self):
        return False


_legacy_version_component_re = re.compile(
    r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE,
)

_legacy_version_replacement_map = {
    "pre": "c", "preview": "c", "-": "final-", "rc": "c", "dev": "@",
}


def _parse_version_parts(s):
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)

        if not part or part == ".":
            continue

        if part[:1] in "0123456789":
            # pad for numeric comparison
            yield part.zfill(8)
        else:
            yield "*" + part

    # ensure that alpha/beta/candidate are before final
    yield "*final"


def _legacy_cmpkey(version):
    # We hardcode an epoch of -1 here. A PEP 440 version can only have an epoch
    # greater than or equal to 0. This will effectively put the LegacyVersion,
    # which uses the defacto standard originally implemented by setuptools,
    # as before all PEP 440 versions.
    epoch = -1

    # This scheme is taken from pkg_resources.parse_version setuptools prior to
    # its adoption of the packaging library.
    parts = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # remove "-" before a prerelease tag
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # remove trailing zeros from each series of numeric parts
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)
    parts = tuple(parts)

    return epoch, parts


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


class Version(_BaseVersion):

    _regex = re.compile(
        r"^\s*" + VERSION_PATTERN + r"\s*$",
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self, version):
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(
                match.group("pre_l"),
                match.group("pre_n"),
            ),
            post=_parse_letter_version(
                match.group("post_l"),
                match.group("post_n1") or match.group("post_n2"),
            ),
            dev=_parse_letter_version(
                match.group("dev_l"),
                match.group("dev_n"),
            ),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self):
        return f"<Version({repr(str(self))})>"

    def __str__(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append(f"{self._version.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))

        # Pre-release
        if self._version.pre is not None:
            parts.append("".join(str(x) for x in self._version.pre))

        # Post-release
        if self._version.post is not None:
            parts.append(f".post{self._version.post[1]}")

        # Development release
        if self._version.dev is not None:
            parts.append(f".dev{self._version.dev[1]}")

        # Local version segment
        if self._version.local is not None:
            parts.append(
                "+{}".format(".".join(str(x) for x in self._version.local))
            )

        return "".join(parts)

    @property
    def public(self):
        return str(self).split("+", 1)[0]

    @property
    def base_version(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append(f"{self._version.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))

        return "".join(parts)

    @property
    def local(self):
        version_string = str(self)
        if "+" in version_string:
            return version_string.split("+", 1)[1]

    @property
    def is_prerelease(self):
        return bool(self._version.dev or self._version.pre)

    @property
    def is_postrelease(self):
        return bool(self._version.post)


def _parse_letter_version(letter, number):
    if letter:
        # We assume there is an implicit 0 in a pre-release if there is
        # no numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower-case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume that if we are given a number but not given a letter,
        # then this is using the implicit post release syntax (e.g., 1.0-1)
        letter = "post"

        return letter, int(number)


_local_version_seperators = re.compile(r"[\._-]")


def _parse_local_version(local):
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_seperators.split(local)
        )


def _cmpkey(epoch, release, pre, post, dev, local):
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non-zero, then take the rest,
    # re-reverse it back into the correct order, and make it a tuple and use
    # that for our sorting key.
    release = tuple(
        reversed(list(
            itertools.dropwhile(
                lambda x: x == 0,
                reversed(release),
            )
        ))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre-segment, but we _only_ want to do this
    # if there is no pre- or a post-segment. If we have one of those, then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        pre = -Infinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        pre = Infinity

    # Versions without a post-segment should sort before those with one.
    if post is None:
        post = -Infinity

    # Versions without a development segment should sort after those with one.
    if dev is None:
        dev = Infinity

    if local is None:
        # Versions without a local segment should sort before those with one.
        local = -Infinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alphanumeric segments sort before numeric segments
        # - Alphanumeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        local = tuple(
            (i, "") if isinstance(i, int) else (-Infinity, i)
            for i in local
        )

    return epoch, release, pre, post, dev, local
<|endoftext|>"""
Module for testing automatic garbage collection of objects

.. autosummary::
   :toctree: generated/

   set_gc_state - enable or disable garbage collection
   gc_state - context manager for given state of garbage collector
   assert_deallocated - context manager to check for circular references on object

"""
import weakref
import gc

from contextlib import contextmanager
from platform import python_implementation

__all__ = ['set_gc_state', 'gc_state', 'assert_deallocated']


IS_PYPY = python_implementation() == 'PyPy'


class ReferenceError(AssertionError):
    pass


def set_gc_state(state):
    """ Set status of garbage collector """
    if gc.isenabled() == state:
        return
    if state:
        gc.enable()
    else:
        gc.disable()


@contextmanager
def gc_state(state):
    """ Context manager to set state of garbage collector to `state`

    Parameters
    ----------
    state : bool
        True for gc enabled, False for disabled

    Examples
    --------
    >>> with gc_state(False):
    ...     assert not gc.isenabled()
    >>> with gc_state(True):
    ...     assert gc.isenabled()
    """
    orig_state = gc.isenabled()
    set_gc_state(state)
    yield
    set_gc_state(orig_state)


@contextmanager
def assert_deallocated(func, *args, **kwargs):
    """Context manager to check that object is deallocated

    This is useful for checking that an object can be freed directly by
    reference counting, without requiring gc to break reference cycles.
    GC is disabled inside the context manager.

    This check is not available on PyPy.

    Parameters
    ----------
    func : callable
        Callable to create object to check
    \\*args : sequence
        positional arguments to `func` in order to create object to check
    \\*\\*kwargs : dict
        keyword arguments to `func` in order to create object to check

    Examples
    --------
    >>> class C: pass
    >>> with assert_deallocated(C) as c:
    ...     # do something
    ...     del c

    >>> class C:
    ...     def __init__(self):
    ...         self._circular = self # Make circular reference
    >>> with assert_deallocated(C) as c: #doctest: +IGNORE_EXCEPTION_DETAIL
    ...     # do something
    ...     del c
    Traceback (most recent call last):
        ...
    ReferenceError: Remaining reference(s) to object
    """
    if IS_PYPY:
        raise RuntimeError("assert_deallocated is unavailable on PyPy")

    with gc_state(False):
        obj = func(*args, **kwargs)
        ref = weakref.ref(obj)
        yield obj
        del obj
        if ref() is not None:
            raise ReferenceError("Remaining reference(s) to object")
<|endoftext|>"""`uarray` provides functions for generating multimethods that dispatch to
multiple different backends

This should be imported, rather than `_uarray` so that an installed version could
be used instead, if available. This means that users can call
`uarray.set_backend` directly instead of going through SciPy.

"""


# Prefer an installed version of uarray, if available
try:
    import uarray as _uarray
except ImportError:
    _has_uarray = False
else:
    from scipy._lib._pep440 import Version as _Version

    _has_uarray = _Version(_uarray.__version__) >= _Version("0.8")
    del _uarray
    del _Version


if _has_uarray:
    from uarray import *  # noqa: F403
    from uarray import _Function
else:
    from ._uarray import *  # noqa: F403
    from ._uarray import _Function  # noqa: F401

del _has_uarray
<|endoftext|># `_elementwise_iterative_method.py` includes tools for writing functions that
# - are vectorized to work elementwise on arrays,
# - implement non-trivial, iterative algorithms with a callback interface, and
# - return rich objects with iteration count, termination status, etc.
#
# Examples include:
# `scipy.optimize._chandrupatla._chandrupatla for scalar rootfinding,
# `scipy.optimize._chandrupatla._chandrupatla_minimize for scalar minimization,
# `scipy.optimize._differentiate._differentiate for numerical differentiation,
# `scipy.optimize._bracket._bracket_root for finding rootfinding brackets,
# `scipy.optimize._bracket._bracket_minimize for finding minimization brackets,
# `scipy.integrate._tanhsinh._tanhsinh` for numerical quadrature,
# `scipy.differentiate.derivative` for finite difference based differentiation.

import math
import numpy as np
from ._util import _RichResult, _call_callback_maybe_halt
from ._array_api import array_namespace, xp_size

_ESIGNERR = -1
_ECONVERR = -2
_EVALUEERR = -3
_ECALLBACK = -4
_EINPUTERR = -5
_ECONVERGED = 0
_EINPROGRESS = 1

def _initialize(func, xs, args, complex_ok=False, preserve_shape=None, xp=None):
    """Initialize abscissa, function, and args arrays for elementwise function

    Parameters
    ----------
    func : callable
        An elementwise function with signature

            func(x: ndarray, *args) -> ndarray

        where each element of ``x`` is a finite real and ``args`` is a tuple,
        which may contain an arbitrary number of arrays that are broadcastable
        with ``x``.
    xs : tuple of arrays
        Finite real abscissa arrays. Must be broadcastable.
    args : tuple, optional
        Additional positional arguments to be passed to `func`.
    preserve_shape : bool, default:False
        When ``preserve_shape=False`` (default), `func` may be passed
        arguments of any shape; `_scalar_optimization_loop` is permitted
        to reshape and compress arguments at will. When
        ``preserve_shape=False``, arguments passed to `func` must have shape
        `shape` or ``shape + (n,)``, where ``n`` is any integer.
    xp : namespace
        Namespace of array arguments in `xs`.

    Returns
    -------
    xs, fs, args : tuple of arrays
        Broadcasted, writeable, 1D abscissa and function value arrays (or
        NumPy floats, if appropriate). The dtypes of the `xs` and `fs` are
        `xfat`; the dtype of the `args` are unchanged.
    shape : tuple of ints
        Original shape of broadcasted arrays.
    xfat : NumPy dtype
        Result dtype of abscissae, function values, and args determined using
        `np.result_type`, except integer types are promoted to `np.float64`.

    Raises
    ------
    ValueError
        If the result dtype is not that of a real scalar

    Notes
    -----
    Useful for initializing the input of SciPy functions that accept
    an elementwise callable, abscissae, and arguments; e.g.
    `scipy.optimize._chandrupatla`.
    """
    nx = len(xs)
    xp = array_namespace(*xs) if xp is None else xp

    # Try to preserve `dtype`, but we need to ensure that the arguments are at
    # least floats before passing them into the function; integers can overflow
    # and cause failure.
    # There might be benefit to combining the `xs` into a single array and
    # calling `func` once on the combined array. For now, keep them separate.
    xas = xp.broadcast_arrays(*xs, *args)  # broadcast and rename
    xat = xp.result_type(*[xa.dtype for xa in xas])
    xat = xp.asarray(1.).dtype if xp.isdtype(xat, "integral") else xat
    xs, args = xas[:nx], xas[nx:]
    xs = [xp.asarray(x, dtype=xat) for x in xs]  # use copy=False when implemented
    fs = [xp.asarray(func(x, *args)) for x in xs]
    shape = xs[0].shape
    fshape = fs[0].shape

    if preserve_shape:
        # bind original shape/func now to avoid late-binding gotcha
        def func(x, *args, shape=shape, func=func,  **kwargs):
            i = (0,)*(len(fshape) - len(shape))
            return func(x[i], *args, **kwargs)
        shape = np.broadcast_shapes(fshape, shape)  # just shapes; use of NumPy OK
        xs = [xp.broadcast_to(x, shape) for x in xs]
        args = [xp.broadcast_to(arg, shape) for arg in args]

    message = ("The shape of the array returned by `func` must be the same as "
               "the broadcasted shape of `x` and all other `args`.")
    if preserve_shape is not None:  # only in tanhsinh for now
        message = f"When `preserve_shape=False`, {message.lower()}"
    shapes_equal = [f.shape == shape for f in fs]
    if not all(shapes_equal):  # use Python all to reduce overhead
        raise ValueError(message)

    # These algorithms tend to mix the dtypes of the abscissae and function
    # values, so figure out what the result will be and convert them all to
    # that type from the outset.
    xfat = xp.result_type(*([f.dtype for f in fs] + [xat]))
    if not complex_ok and not xp.isdtype(xfat, "real floating"):
        raise ValueError("Abscissae and function output must be real numbers.")
    xs = [xp.asarray(x, dtype=xfat, copy=True) for x in xs]
    fs = [xp.asarray(f, dtype=xfat, copy=True) for f in fs]

    # To ensure that we can do indexing, we'll work with at least 1d arrays,
    # but remember the appropriate shape of the output.
    xs = [xp.reshape(x, (-1,)) for x in xs]
    fs = [xp.reshape(f, (-1,)) for f in fs]
    args = [xp.reshape(xp.asarray(arg, copy=True), (-1,)) for arg in args]
    return func, xs, fs, args, shape, xfat, xp


def _loop(work, callback, shape, maxiter, func, args, dtype, pre_func_eval,
          post_func_eval, check_termination, post_termination_check,
          customize_result, res_work_pairs, xp, preserve_shape=False):
    """Main loop of a vectorized scalar optimization algorithm

    Parameters
    ----------
    work : _RichResult
        All variables that need to be retained between iterations. Must
        contain attributes `nit`, `nfev`, and `success`. All arrays are
        subject to being "compressed" if `preserve_shape is False`; nest
        arrays that should not be compressed inside another object (e.g.
        `dict` or `_RichResult`).
    callback : callable
        User-specified callback function
    shape : tuple of ints
        The shape of all output arrays
    maxiter :
        Maximum number of iterations of the algorithm
    func : callable
        The user-specified callable that is being optimized or solved
    args : tuple
        Additional positional arguments to be passed to `func`.
    dtype : NumPy dtype
        The common dtype of all abscissae and function values
    pre_func_eval : callable
        A function that accepts `work` and returns `x`, the active elements
        of `x` at which `func` will be evaluated. May modify attributes
        of `work` with any algorithmic steps that need to happen
         at the beginning of an iteration, before `func` is evaluated,
    post_func_eval : callable
        A function that accepts `x`, `func(x)`, and `work`. May modify
        attributes of `work` with any algorithmic steps that need to happen
         in the middle of an iteration, after `func` is evaluated but before
         the termination check.
    check_termination : callable
        A function that accepts `work` and returns `stop`, a boolean array
        indicating which of the active elements have met a termination
        condition.
    post_termination_check : callable
        A function that accepts `work`. May modify `work` with any algorithmic
        steps that need to happen after the termination check and before the
        end of the iteration.
    customize_result : callable
        A function that accepts `res` and `shape` and returns `shape`. May
        modify `res` (in-place) according to preferences (e.g. rearrange
        elements between attributes) and modify `shape` if needed.
    res_work_pairs : list of (str, str)
        Identifies correspondence between attributes of `res` and attributes
        of `work`; i.e., attributes of active elements of `work` will be
        copied to the appropriate indices of `res` when appropriate. The order
        determines the order in which _RichResult attributes will be
        pretty-printed.
    preserve_shape : bool, default: False
        Whether to compress the attributes of `work` (to avoid unnecessary
        computation on elements that have already converged).

    Returns
    -------
    res : _RichResult
        The final result object

    Notes
    -----
    Besides providing structure, this framework provides several important
    services for a vectorized optimization algorithm.

    - It handles common tasks involving iteration count, function evaluation
      count, a user-specified callback, and associated termination conditions.
    - It compresses the attributes of `work` to eliminate unnecessary
      computation on elements that have already converged.

    """
    if xp is None:
        raise NotImplementedError("Must provide xp.")

    cb_terminate = False

    # Initialize the result object and active element index array
    n_elements = math.prod(shape)
    active = xp.arange(n_elements)  # in-progress element indices
    res_dict = {i: xp.zeros(n_elements, dtype=dtype) for i, j in res_work_pairs}
    res_dict['success'] = xp.zeros(n_elements, dtype=xp.bool)
    res_dict['status'] = xp.full(n_elements, xp.asarray(_EINPROGRESS), dtype=xp.int32)
    res_dict['nit'] = xp.zeros(n_elements, dtype=xp.int32)
    res_dict['nfev'] = xp.zeros(n_elements, dtype=xp.int32)
    res = _RichResult(res_dict)
    work.args = args

    active = _check_termination(work, res, res_work_pairs, active,
                                check_termination, preserve_shape, xp)

    if callback is not None:
        temp = _prepare_result(work, res, res_work_pairs, active, shape,
                               customize_result, preserve_shape, xp)
        if _call_callback_maybe_halt(callback, temp):
            cb_terminate = True

    while work.nit < maxiter and xp_size(active) and not cb_terminate and n_elements:
        x = pre_func_eval(work)

        if work.args and work.args[0].ndim != x.ndim:
            # `x` always starts as 1D. If the SciPy function that uses
            # _loop added dimensions to `x`, we need to
            # add them to the elements of `args`.
            args = []
            for arg in work.args:
                n_new_dims = x.ndim - arg.ndim
                new_shape = arg.shape + (1,)*n_new_dims
                args.append(xp.reshape(arg, new_shape))
            work.args = args

        x_shape = x.shape
        if preserve_shape:
            x = xp.reshape(x, (shape + (-1,)))
        f = func(x, *work.args)
        f = xp.asarray(f, dtype=dtype)
        if preserve_shape:
            x = xp.reshape(x, x_shape)
            f = xp.reshape(f, x_shape)
        work.nfev += 1 if x.ndim == 1 else x.shape[-1]

        post_func_eval(x, f, work)

        work.nit += 1
        active = _check_termination(work, res, res_work_pairs, active,
                                    check_termination, preserve_shape, xp)

        if callback is not None:
            temp = _prepare_result(work, res, res_work_pairs, active, shape,
                                   customize_result, preserve_shape, xp)
            if _call_callback_maybe_halt(callback, temp):
                cb_terminate = True
                break
        if xp_size(active) == 0:
            break

        post_termination_check(work)

    work.status[:] = _ECALLBACK if cb_terminate else _ECONVERR
    return _prepare_result(work, res, res_work_pairs, active, shape,
                           customize_result, preserve_shape, xp)


def _check_termination(work, res, res_work_pairs, active, check_termination,
                       preserve_shape, xp):
    # Checks termination conditions, updates elements of `res` with
    # corresponding elements of `work`, and compresses `work`.

    stop = check_termination(work)

    if xp.any(stop):
        # update the active elements of the result object with the active
        # elements for which a termination condition has been met
        _update_active(work, res, res_work_pairs, active, stop, preserve_shape, xp)

        if preserve_shape:
            stop = stop[active]

        proceed = ~stop
        active = active[proceed]

        if not preserve_shape:
            # compress the arrays to avoid unnecessary computation
            for key, val in work.items():
                # Need to find a better way than these try/excepts
                # Somehow need to keep compressible numerical args separate
                if key == 'args':
                    continue
                try:
                    work[key] = val[proceed]
                except (IndexError, TypeError, KeyError):  # not a compressible array
                    work[key] = val
            work.args = [arg[proceed] for arg in work.args]

    return active


def _update_active(work, res, res_work_pairs, active, mask, preserve_shape, xp):
    # Update `active` indices of the arrays in result object `res` with the
    # contents of the scalars and arrays in `update_dict`. When provided,
    # `mask` is a boolean array applied both to the arrays in `update_dict`
    # that are to be used and to the arrays in `res` that are to be updated.
    update_dict = {key1: work[key2] for key1, key2 in res_work_pairs}
    update_dict['success'] = work.status == 0

    if mask is not None:
        if preserve_shape:
            active_mask = xp.zeros_like(mask)
            active_mask[active] = 1
            active_mask = active_mask & mask
            for key, val in update_dict.items():
                try:
                    res[key][active_mask] = val[active_mask]
                except (IndexError, TypeError, KeyError):
                    res[key][active_mask] = val
        else:
            active_mask = active[mask]
            for key, val in update_dict.items():
                try:
                    res[key][active_mask] = val[mask]
                except (IndexError, TypeError, KeyError):
                    res[key][active_mask] = val
    else:
        for key, val in update_dict.items():
            if preserve_shape:
                try:
                    val = val[active]
                except (IndexError, TypeError, KeyError):
                    pass
            res[key][active] = val


def _prepare_result(work, res, res_work_pairs, active, shape, customize_result,
                    preserve_shape, xp):
    # Prepare the result object `res` by creating a copy, copying the latest
    # data from work, running the provided result customization function,
    # and reshaping the data to the original shapes.
    res = res.copy()
    _update_active(work, res, res_work_pairs, active, None, preserve_shape, xp)

    shape = customize_result(res, shape)

    for key, val in res.items():
        # this looks like it won't work for xp != np if val is not numeric
        temp = xp.reshape(val, shape)
        res[key] = temp[()] if temp.ndim == 0 else temp

    res['_order_keys'] = ['success'] + [i for i, j in res_work_pairs]
    return _RichResult(**res)
<|endoftext|>import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import TypeAlias, TypeVar

import numpy as np
from scipy._lib._array_api import array_namespace, is_numpy, xp_size
from scipy._lib._docscrape import FunctionDoc, Parameter


AxisError: type[Exception]
ComplexWarning: type[Warning]
VisibleDeprecationWarning: type[Warning]

if np.lib.NumpyVersion(np.__version__) >= '1.25.0':
    from numpy.exceptions import (
        AxisError, ComplexWarning, VisibleDeprecationWarning,
        DTypePromotionError
    )
else:
    from numpy import (  # type: ignore[attr-defined, no-redef]
        AxisError, ComplexWarning, VisibleDeprecationWarning  # noqa: F401
    )
    DTypePromotionError = TypeError  # type: ignore

np_long: type
np_ulong: type

if np.lib.NumpyVersion(np.__version__) >= "2.0.0.dev0":
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                r".*In the future `np\.long` will be defined as.*",
                FutureWarning,
            )
            np_long = np.long  # type: ignore[attr-defined]
            np_ulong = np.ulong  # type: ignore[attr-defined]
    except AttributeError:
            np_long = np.int_
            np_ulong = np.uint
else:
    np_long = np.int_
    np_ulong = np.uint

IntNumber = int | np.integer
DecimalNumber = float | np.floating | np.integer

copy_if_needed: bool | None

if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    copy_if_needed = None
elif np.lib.NumpyVersion(np.__version__) < "1.28.0":
    copy_if_needed = False
else:
    # 2.0.0 dev versions, handle cases where copy may or may not exist
    try:
        np.array([1]).__array__(copy=None)  # type: ignore[call-overload]
        copy_if_needed = None
    except TypeError:
        copy_if_needed = False


_RNG: TypeAlias = np.random.Generator | np.random.RandomState
SeedType: TypeAlias = IntNumber | _RNG | None

GeneratorType = TypeVar("GeneratorType", bound=_RNG)

# Since Generator was introduced in numpy 1.17, the following condition is needed for
# backward compatibility
try:
    from numpy.random import Generator as Generator
except ImportError:
    class Generator:  # type: ignore[no-redef]
        pass


def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """Return elements chosen from two possibilities depending on a condition

    Equivalent to ``f(*arrays) if cond else fillvalue`` performed elementwise.

    Parameters
    ----------
    cond : array
        The condition (expressed as a boolean array).
    arrays : tuple of array
        Arguments to `f` (and `f2`). Must be broadcastable with `cond`.
    f : callable
        Where `cond` is True, output will be ``f(arr1[cond], arr2[cond], ...)``
    fillvalue : object
        If provided, value with which to fill output array where `cond` is
        not True.
    f2 : callable
        If provided, output will be ``f2(arr1[cond], arr2[cond], ...)`` where
        `cond` is not True.

    Returns
    -------
    out : array
        An array with elements from the output of `f` where `cond` is True
        and `fillvalue` (or elements from the output of `f2`) elsewhere. The
        returned array has data type determined by Type Promotion Rules
        with the output of `f` and `fillvalue` (or the output of `f2`).

    Notes
    -----
    ``xp.where(cond, x, fillvalue)`` requires explicitly forming `x` even where
    `cond` is False. This function evaluates ``f(arr1[cond], arr2[cond], ...)``
    onle where `cond` ``is True.

    Examples
    --------
    >>> import numpy as np
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    """
    xp = array_namespace(cond, *arrays)

    if (f2 is fillvalue is None) or (f2 is not None and fillvalue is not None):
        raise ValueError("Exactly one of `fillvalue` or `f2` must be given.")

    args = xp.broadcast_arrays(cond, *arrays)
    bool_dtype = xp.asarray([True]).dtype  # numpy 1.xx doesn't have `bool`
    cond, arrays = xp.astype(args[0], bool_dtype, copy=False), args[1:]

    temp1 = xp.asarray(f(*(arr[cond] for arr in arrays)))

    if f2 is None:
        # If `fillvalue` is a Python scalar and we convert to `xp.asarray`, it gets the
        # default `int` or `float` type of `xp`, so `result_type` could be wrong.
        # `result_type` should/will handle mixed array/Python scalars;
        # remove this special logic when it does.
        if type(fillvalue) in {bool, int, float, complex}:
            with np.errstate(invalid='ignore'):
                dtype = (temp1 * fillvalue).dtype
        else:
           dtype = xp.result_type(temp1.dtype, fillvalue)
        out = xp.full(cond.shape, dtype=dtype,
                      fill_value=xp.asarray(fillvalue, dtype=dtype))
    else:
        ncond = ~cond
        temp2 = xp.asarray(f2(*(arr[ncond] for arr in arrays)))
        dtype = xp.result_type(temp1, temp2)
        out = xp.empty(cond.shape, dtype=dtype)
        out[ncond] = temp2

    out[cond] = temp1

    return out


def _lazyselect(condlist, choicelist, arrays, default=0):
    """
    Mimic `np.select(condlist, choicelist)`.

    Notice, it assumes that all `arrays` are of the same shape or can be
    broadcasted together.

    All functions in `choicelist` must accept array arguments in the order
    given in `arrays` and must return an array of the same shape as broadcasted
    `arrays`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6)
    >>> np.select([x <3, x > 3], [x**2, x**3], default=0)
    array([  0,   1,   4,   0,  64, 125])

    >>> _lazyselect([x < 3, x > 3], [lambda x: x**2, lambda x: x**3], (x,))
    array([   0.,    1.,    4.,   0.,   64.,  125.])

    >>> a = -np.ones_like(x)
    >>> _lazyselect([x < 3, x > 3],
    ...             [lambda x, a: x**2, lambda x, a: a * x**3],
    ...             (x, a), default=np.nan)
    array([   0.,    1.,    4.,   nan,  -64., -125.])

    """
    arrays = np.broadcast_arrays(*arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=default, dtype=tcode)
    for func, cond in zip(choicelist, condlist):
        if np.all(cond is False):
            continue
        cond, _ = np.broadcast_arrays(cond, arrays[0])
        temp = tuple(np.extract(cond, arr) for arr in arrays)
        np.place(out, cond, func(*temp))
    return out


def _aligned_zeros(shape, dtype=float, order="C", align=None):
    """Allocate a new ndarray with aligned memory.

    Primary use case for this currently is working around a f2py issue
    in NumPy 1.9.1, where dtype.alignment is such that np.zeros() does
    not necessarily create arrays aligned up to it.

    """
    dtype = np.dtype(dtype)
    if align is None:
        align = dtype.alignment
    if not hasattr(shape, '__len__'):
        shape = (shape,)
    size = functools.reduce(operator.mul, shape) * dtype.itemsize
    buf = np.empty(size + align + 1, np.uint8)
    offset = buf.__array_interface__['data'][0] % align
    if offset != 0:
        offset = align - offset
    # Note: slices producing 0-size arrays do not necessarily change
    # data pointer --- so we use and allocate size+1
    buf = buf[offset:offset+size+1][:-1]
    data = np.ndarray(shape, dtype, buf, order=order)
    data.fill(0)
    return data


def _prune_array(array):
    """Return an array equivalent to the input array. If the input
    array is a view of a much larger array, copy its contents to a
    newly allocated array. Otherwise, return the input unchanged.
    """
    if array.base is not None and array.size < array.base.size // 2:
        return array.copy()
    return array


def float_factorial(n: int) -> float:
    """Compute the factorial and return as a float

    Returns infinity when result is too large for a double
    """
    return float(math.factorial(n)) if n < 171 else np.inf


_rng_desc = (
    r"""If `rng` is passed by keyword, types other than `numpy.random.Generator` are
    passed to `numpy.random.default_rng` to instantiate a ``Generator``.
    If `rng` is already a ``Generator`` instance, then the provided instance is
    used. Specify `rng` for repeatable function behavior.

    If this argument is passed by position or `{old_name}` is passed by keyword,
    legacy behavior for the argument `{old_name}` applies:

    - If `{old_name}` is None (or `numpy.random`), the `numpy.random.RandomState`
      singleton is used.
    - If `{old_name}` is an int, a new ``RandomState`` instance is used,
      seeded with `{old_name}`.
    - If `{old_name}` is already a ``Generator`` or ``RandomState`` instance then
      that instance is used.

    .. versionchanged:: 1.15.0
        As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
        transition from use of `numpy.random.RandomState` to
        `numpy.random.Generator`, this keyword was changed from `{old_name}` to `rng`.
        For an interim period, both keywords will continue to work, although only one
        may be specified at a time. After the interim period, function calls using the
        `{old_name}` keyword will emit warnings. The behavior of both `{old_name}` and
        `rng` are outlined above, but only the `rng` keyword should be used in new code.
        """
)


# SPEC 7
def _transition_to_rng(old_name, *, position_num=None, end_version=None,
                       replace_doc=True):
    """Example decorator to transition from old PRNG usage to new `rng` behavior

    Suppose the decorator is applied to a function that used to accept parameter
    `old_name='random_state'` either by keyword or as a positional argument at
    `position_num=1`. At the time of application, the name of the argument in the
    function signature is manually changed to the new name, `rng`. If positional
    use was allowed before, this is not changed.*

    - If the function is called with both `random_state` and `rng`, the decorator
      raises an error.
    - If `random_state` is provided as a keyword argument, the decorator passes
      `random_state` to the function's `rng` argument as a keyword. If `end_version`
      is specified, the decorator will emit a `DeprecationWarning` about the
      deprecation of keyword `random_state`.
    - If `random_state` is provided as a positional argument, the decorator passes
      `random_state` to the function's `rng` argument by position. If `end_version`
      is specified, the decorator will emit a `FutureWarning` about the changing
      interpretation of the argument.
    - If `rng` is provided as a keyword argument, the decorator validates `rng` using
      `numpy.random.default_rng` before passing it to the function.
    - If `end_version` is specified and neither `random_state` nor `rng` is provided
      by the user, the decorator checks whether `np.random.seed` has been used to set
      the global seed. If so, it emits a `FutureWarning`, noting that usage of
      `numpy.random.seed` will eventually have no effect. Either way, the decorator
      calls the function without explicitly passing the `rng` argument.

    If `end_version` is specified, a user must pass `rng` as a keyword to avoid
    warnings.

    After the deprecation period, the decorator can be removed, and the function
    can simply validate the `rng` argument by calling `np.random.default_rng(rng)`.

    * A `FutureWarning` is emitted when the PRNG argument is used by
      position. It indicates that the "Hinsen principle" (same
      code yielding different results in two versions of the software)
      will be violated, unless positional use is deprecated. Specifically:

      - If `None` is passed by position and `np.random.seed` has been used,
        the function will change from being seeded to being unseeded.
      - If an integer is passed by position, the random stream will change.
      - If `np.random` or an instance of `RandomState` is passed by position,
        an error will be raised.

      We suggest that projects consider deprecating positional use of
      `random_state`/`rng` (i.e., change their function signatures to
      ``def my_func(..., *, rng=None)``); that might not make sense
      for all projects, so this SPEC does not make that
      recommendation, neither does this decorator enforce it.

    Parameters
    ----------
    old_name : str
        The old name of the PRNG argument (e.g. `seed` or `random_state`).
    position_num : int, optional
        The (0-indexed) position of the old PRNG argument (if accepted by position).
        Maintainers are welcome to eliminate this argument and use, for example,
        `inspect`, if preferred.
    end_version : str, optional
        The full version number of the library when the behavior described in
        `DeprecationWarning`s and `FutureWarning`s will take effect. If left
        unspecified, no warnings will be emitted by the decorator.
    replace_doc : bool, default: True
        Whether the decorator should replace the documentation for parameter `rng` with
        `_rng_desc` (defined above), which documents both new `rng` keyword behavior
        and typical legacy `random_state`/`seed` behavior. If True, manually replace
        the first paragraph of the function's old `random_state`/`seed` documentation
        with the desired *final* `rng` documentation; this way, no changes to
        documentation are needed when the decorator is removed. Documentation of `rng`
        after the first blank line is preserved. Use False if the function's old
        `random_state`/`seed` behavior does not match that described by `_rng_desc`.

    """
    NEW_NAME = "rng"

    cmn_msg = (
        "To silence this warning and ensure consistent behavior in SciPy "
        f"{end_version}, control the RNG using argument `{NEW_NAME}`. Arguments passed "
        f"to keyword `{NEW_NAME}` will be validated by `np.random.default_rng`, so the "
        "behavior corresponding with a given value may change compared to use of "
        f"`{old_name}`. For example, "
        "1) `None` will result in unpredictable random numbers, "
        "2) an integer will result in a different stream of random numbers, (with the "
        "same distribution), and "
        "3) `np.random` or `RandomState` instances will result in an error. "
        "See the documentation of `default_rng` for more information."
    )

    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            # Determine how PRNG was passed
            as_old_kwarg = old_name in kwargs
            as_new_kwarg = NEW_NAME in kwargs
            as_pos_arg = position_num is not None and len(args) >= position_num + 1
            emit_warning = end_version is not None

            # Can only specify PRNG one of the three ways
            if int(as_old_kwarg) + int(as_new_kwarg) + int(as_pos_arg) > 1:
                message = (
                    f"{fun.__name__}() got multiple values for "
                    f"argument now known as `{NEW_NAME}`. Specify one of "
                    f"`{NEW_NAME}` or `{old_name}`."
                )
                raise TypeError(message)

            # Check whether global random state has been set
            global_seed_set = np.random.mtrand._rand._bit_generator._seed_seq is None

            if as_old_kwarg:  # warn about deprecated use of old kwarg
                kwargs[NEW_NAME] = kwargs.pop(old_name)
                if emit_warning:
                    message = (
                        f"Use of keyword argument `{old_name}` is "
                        f"deprecated and replaced by `{NEW_NAME}`.  "
                        f"Support for `{old_name}` will be removed "
                        f"in SciPy {end_version}. "
                    ) + cmn_msg
                    warnings.warn(message, DeprecationWarning, stacklevel=2)

            elif as_pos_arg:
                # Warn about changing meaning of positional arg

                # Note that this decorator does not deprecate positional use of the
                # argument; it only warns that the behavior will change in the future.
                # Simultaneously transitioning to keyword-only use is another option.

                arg = args[position_num]
                # If the argument is None and the global seed wasn't set, or if the
                # argument is one of a few new classes, the user will not notice change
                # in behavior.
                ok_classes = (
                    np.random.Generator,
                    np.random.SeedSequence,
                    np.random.BitGenerator,
                )
                if (arg is None and not global_seed_set) or isinstance(arg, ok_classes):
                    pass
                elif emit_warning:
                    message = (
                        f"Positional use of `{NEW_NAME}` (formerly known as "
                        f"`{old_name}`) is still allowed, but the behavior is "
                        "changing: the argument will be normalized using "
                        f"`np.random.default_rng` beginning in SciPy {end_version}, "
                        "and the resulting `Generator` will be used to generate "
                        "random numbers."
                    ) + cmn_msg
                    warnings.warn(message, FutureWarning, stacklevel=2)

            elif as_new_kwarg:  # no warnings; this is the preferred use
                # After the removal of the decorator, normalization with
                # np.random.default_rng will be done inside the decorated function
                kwargs[NEW_NAME] = np.random.default_rng(kwargs[NEW_NAME])

            elif global_seed_set and emit_warning:
                # Emit FutureWarning if `np.random.seed` was used and no PRNG was passed
                message = (
                    "The NumPy global RNG was seeded by calling "
                    f"`np.random.seed`. Beginning in {end_version}, this "
                    "function will no longer use the global RNG."
                ) + cmn_msg
                warnings.warn(message, FutureWarning, stacklevel=2)

            return fun(*args, **kwargs)

        if replace_doc:
            doc = FunctionDoc(wrapper)
            parameter_names = [param.name for param in doc['Parameters']]
            if 'rng' in parameter_names:
                _type = "{None, int, `numpy.random.Generator`}, optional"
                _desc = _rng_desc.replace("{old_name}", old_name)
                old_doc = doc['Parameters'][parameter_names.index('rng')].desc
                old_doc_keep = old_doc[old_doc.index("") + 1:] if "" in old_doc else []
                new_doc = [_desc] + old_doc_keep
                _rng_parameter_doc = Parameter('rng', _type, new_doc)
                doc['Parameters'][parameter_names.index('rng')] = _rng_parameter_doc
                doc = str(doc).split("\n", 1)[1]  # remove signature
                wrapper.__doc__ = str(doc)
        return wrapper

    return decorator


# copy-pasted from scikit-learn utils/validation.py
def check_random_state(seed):
    """Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral | np.integer):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState | np.random.Generator):
        return seed

    raise ValueError(f"'{seed}' cannot be used to seed a numpy.random.RandomState"
                     " instance")


def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for SciPy argument validation.

    Many SciPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.

    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    objects_ok : bool, optional
        True if arrays with dype('O') are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    as_inexact : bool, optional
        True to convert the input array to a np.inexact dtype.

    Returns
    -------
    ret : ndarray
        The converted validated array.

    """
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse arrays/matrices are not supported by this function. '
                   'Perhaps one of the `scipy.sparse.linalg` functions '
                   'would work instead.')
            raise ValueError(msg)
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok:
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    if as_inexact:
        if not np.issubdtype(a.dtype, np.inexact):
            a = toarray(a, dtype=np.float64)
    return a


def _validate_int(k, name, minimum=None):
    """
    Validate a scalar integer.

    This function can be used to validate an argument to a function
    that expects the value to be an integer.  It uses `operator.index`
    to validate the value (so, for example, k=2.0 results in a
    TypeError).

    Parameters
    ----------
    k : int
        The value to be validated.
    name : str
        The name of the parameter.
    minimum : int, optional
        An optional lower bound.
    """
    try:
        k = operator.index(k)
    except TypeError:
        raise TypeError(f'{name} must be an integer.') from None
    if minimum is not None and k < minimum:
        raise ValueError(f'{name} must be an integer not less '
                         f'than {minimum}') from None
    return k


# Add a replacement for inspect.getfullargspec()/
# The version below is borrowed from Django,
# https://github.com/django/django/pull/4846.

# Note an inconsistency between inspect.getfullargspec(func) and
# inspect.signature(func). If `func` is a bound method, the latter does *not*
# list `self` as a first argument, while the former *does*.
# Hence, cook up a common ground replacement: `getfullargspec_no_self` which
# mimics `inspect.getfullargspec` but does not list `self`.
#
# This way, the caller code does not need to know whether it uses a legacy
# .getfullargspec or a bright and shiny .signature.

FullArgSpec = namedtuple('FullArgSpec',
                         ['args', 'varargs', 'varkw', 'defaults',
                          'kwonlyargs', 'kwonlydefaults', 'annotations'])


def getfullargspec_no_self(func):
    """inspect.getfullargspec replacement using inspect.signature.

    If func is a bound method, do not list the 'self' parameter.

    Parameters
    ----------
    func : callable
        A callable to inspect

    Returns
    -------
    fullargspec : FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                              kwonlydefaults, annotations)

        NOTE: if the first argument of `func` is self, it is *not*, I repeat
        *not*, included in fullargspec.args.
        This is done for consistency between inspect.getargspec() under
        Python 2.x, and inspect.signature() under Python 3.x.

    """
    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY]
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = tuple(
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
            p.default is not p.empty)
    ) or None
    kwonlyargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    kwdefaults = {p.name: p.default for p in sig.parameters.values()
                  if p.kind == inspect.Parameter.KEYWORD_ONLY and
                  p.default is not p.empty}
    annotations = {p.name: p.annotation for p in sig.parameters.values()
                   if p.annotation is not p.empty}
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                       kwdefaults or None, annotations)


class _FunctionWrapper:
    """
    Object to wrap user's function, allowing picklability
    """
    def __init__(self, f, args):
        self.f = f
        self.args = [] if args is None else args

    def __call__(self, x):
        return self.f(x, *self.args)


class MapWrapper:
    """
    Parallelisation wrapper for working with map-like callables, such as
    `multiprocessing.Pool.map`.

    Parameters
    ----------
    pool : int or map-like callable
        If `pool` is an integer, then it specifies the number of threads to
        use for parallelization. If ``int(pool) == 1``, then no parallel
        processing is used and the map builtin is used.
        If ``pool == -1``, then the pool will utilize all available CPUs.
        If `pool` is a map-like callable that follows the same
        calling sequence as the built-in map function, then this callable is
        used for parallelization.
    """
    def __init__(self, pool=1):
        self.pool = None
        self._mapfunc = map
        self._own_pool = False

        if callable(pool):
            self.pool = pool
            self._mapfunc = self.pool
        else:
            from multiprocessing import Pool
            # user supplies a number
            if int(pool) == -1:
                # use as many processors as possible
                self.pool = Pool()
                self._mapfunc = self.pool.map
                self._own_pool = True
            elif int(pool) == 1:
                pass
            elif int(pool) > 1:
                # use the number of processors requested
                self.pool = Pool(processes=int(pool))
                self._mapfunc = self.pool.map
                self._own_pool = True
            else:
                raise RuntimeError("Number of workers specified must be -1,"
                                   " an int >= 1, or an object with a 'map' "
                                   "method")

    def __enter__(self):
        return self

    def terminate(self):
        if self._own_pool:
            self.pool.terminate()

    def join(self):
        if self._own_pool:
            self.pool.join()

    def close(self):
        if self._own_pool:
            self.pool.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._own_pool:
            self.pool.close()
            self.pool.terminate()

    def __call__(self, func, iterable):
        # only accept one iterable because that's all Pool.map accepts
        try:
            return self._mapfunc(func, iterable)
        except TypeError as e:
            # wrong number of arguments
            raise TypeError("The map-like callable must be of the"
                            " form f(func, iterable)") from e


def rng_integers(gen, low, high=None, size=None, dtype='int64',
                 endpoint=False):
    """
    Return random integers from low (inclusive) to high (exclusive), or if
    endpoint=True, low (inclusive) to high (inclusive). Replaces
    `RandomState.randint` (with endpoint=False) and
    `RandomState.random_integers` (with endpoint=True).

    Return random integers from the "discrete uniform" distribution of the
    specified dtype. If high is None (the default), then results are from
    0 to low.

    Parameters
    ----------
    gen : {None, np.random.RandomState, np.random.Generator}
        Random number generator. If None, then the np.random.RandomState
        singleton is used.
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        high=None, in which case this parameter is 0 and this value is used
        for high).
    high : int or array-like of ints
        If provided, one above the largest (signed) integer to be drawn from
        the distribution (see above for behavior if high=None). If array-like,
        must contain integer values.
    size : array-like of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. Default is None, in which case a single value is
        returned.
    dtype : {str, dtype}, optional
        Desired dtype of the result. All dtypes are determined by their name,
        i.e., 'int64', 'int', etc, so byteorder is not available and a specific
        precision may have different C types depending on the platform.
        The default value is 'int64'.
    endpoint : bool, optional
        If True, sample from the interval [low, high] instead of the default
        [low, high) Defaults to False.

    Returns
    -------
    out: int or ndarray of ints
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
    """
    if isinstance(gen, Generator):
        return gen.integers(low, high=high, size=size, dtype=dtype,
                            endpoint=endpoint)
    else:
        if gen is None:
            # default is RandomState singleton used by np.random.
            gen = np.random.mtrand._rand
        if endpoint:
            # inclusive of endpoint
            # remember that low and high can be arrays, so don't modify in
            # place
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)

        # exclusive
        return gen.randint(low, high=high, size=size, dtype=dtype)


@contextmanager
def _fixed_default_rng(seed=1638083107694713882823079058616272161):
    """Context with a fixed np.random.default_rng seed."""
    orig_fun = np.random.default_rng
    np.random.default_rng = lambda seed=seed: orig_fun(seed)
    try:
        yield
    finally:
        np.random.default_rng = orig_fun


def _rng_html_rewrite(func):
    """Rewrite the HTML rendering of ``np.random.default_rng``.

    This is intended to decorate
    ``numpydoc.docscrape_sphinx.SphinxDocString._str_examples``.

    Examples are only run by Sphinx when there are plot involved. Even so,
    it does not change the result values getting printed.
    """
    # hexadecimal or number seed, case-insensitive
    pattern = re.compile(r'np.random.default_rng\((0x[0-9A-F]+|\d+)\)', re.I)

    def _wrapped(*args, **kwargs):
        res = func(*args, **kwargs)
        lines = [
            re.sub(pattern, 'np.random.default_rng()', line)
            for line in res
        ]
        return lines

    return _wrapped


def _argmin(a, keepdims=False, axis=None):
    """
    argmin with a `keepdims` parameter.

    See https://github.com/numpy/numpy/issues/8710

    If axis is not None, a.shape[axis] must be greater than 0.
    """
    res = np.argmin(a, axis=axis)
    if keepdims and axis is not None:
        res = np.expand_dims(res, axis=axis)
    return res


def _first_nonnan(a, axis):
    """
    Return the first non-nan value along the given axis.

    If a slice is all nan, nan is returned for that slice.

    The shape of the return value corresponds to ``keepdims=True``.

    Examples
    --------
    >>> import numpy as np
    >>> nan = np.nan
    >>> a = np.array([[ 3.,  3., nan,  3.],
                      [ 1., nan,  2.,  4.],
                      [nan, nan,  9., -1.],
                      [nan,  5.,  4.,  3.],
                      [ 2.,  2.,  2.,  2.],
                      [nan, nan, nan, nan]])
    >>> _first_nonnan(a, axis=0)
    array([[3., 3., 2., 3.]])
    >>> _first_nonnan(a, axis=1)
    array([[ 3.],
           [ 1.],
           [ 9.],
           [ 5.],
           [ 2.],
           [nan]])
    """
    k = _argmin(np.isnan(a), axis=axis, keepdims=True)
    return np.take_along_axis(a, k, axis=axis)


def _nan_allsame(a, axis, keepdims=False):
    """
    Determine if the values along an axis are all the same.

    nan values are ignored.

    `a` must be a numpy array.

    `axis` is assumed to be normalized; that is, 0 <= axis < a.ndim.

    For an axis of length 0, the result is True.  That is, we adopt the
    convention that ``allsame([])`` is True. (There are no values in the
    input that are different.)

    `True` is returned for slices that are all nan--not because all the
    values are the same, but because this is equivalent to ``allsame([])``.

    Examples
    --------
    >>> from numpy import nan, array
    >>> a = array([[ 3.,  3., nan,  3.],
    ...            [ 1., nan,  2.,  4.],
    ...            [nan, nan,  9., -1.],
    ...            [nan,  5.,  4.,  3.],
    ...            [ 2.,  2.,  2.,  2.],
    ...            [nan, nan, nan, nan]])
    >>> _nan_allsame(a, axis=1, keepdims=True)
    array([[ True],
           [False],
           [False],
           [False],
           [ True],
           [ True]])
    """
    if axis is None:
        if a.size == 0:
            return True
        a = a.ravel()
        axis = 0
    else:
        shp = a.shape
        if shp[axis] == 0:
            shp = shp[:axis] + (1,)*keepdims + shp[axis + 1:]
            return np.full(shp, fill_value=True, dtype=bool)
    a0 = _first_nonnan(a, axis=axis)
    return ((a0 == a) | np.isnan(a)).all(axis=axis, keepdims=keepdims)


def _contains_nan(a, nan_policy='propagate', policies=None, *,
                  xp_omit_okay=False, xp=None):
    # Regarding `xp_omit_okay`: Temporarily, while `_axis_nan_policy` does not
    # handle non-NumPy arrays, most functions that call `_contains_nan` want
    # it to raise an error if `nan_policy='omit'` and `xp` is not `np`.
    # Some functions support `nan_policy='omit'` natively, so setting this to
    # `True` prevents the error from being raised.
    if xp is None:
        xp = array_namespace(a)
    not_numpy = not is_numpy(xp)

    if policies is None:
        policies = {'propagate', 'raise', 'omit'}
    if nan_policy not in policies:
        raise ValueError(f"nan_policy must be one of {set(policies)}.")

    if xp_size(a) == 0:
        contains_nan = False
    elif xp.isdtype(a.dtype, "real floating"):
        # Faster and less memory-intensive than xp.any(xp.isnan(a)), and unlike other
        # reductions, `max`/`min` won't return NaN unless there is a NaN in the data.
        contains_nan = xp.isnan(xp.max(a))
    elif xp.isdtype(a.dtype, "complex floating"):
        # Typically `real` and `imag` produce views; otherwise, `xp.any(xp.isnan(a))`
        # would be more efficient.
        contains_nan = xp.isnan(xp.max(xp.real(a))) | xp.isnan(xp.max(xp.imag(a)))
    elif is_numpy(xp) and np.issubdtype(a.dtype, object):
        contains_nan = False
        for el in a.ravel():
            # isnan doesn't work on non-numeric elements
            if np.issubdtype(type(el), np.number) and np.isnan(el):
                contains_nan = True
                break
    else:
        # Only `object` and `inexact` arrays can have NaNs
        contains_nan = False

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    if not xp_omit_okay and not_numpy and contains_nan and nan_policy=='omit':
        message = "`nan_policy='omit' is incompatible with non-NumPy arrays."
        raise ValueError(message)

    return contains_nan, nan_policy


def _rename_parameter(old_name, new_name, dep_version=None):
    """
    Generate decorator for backward-compatible keyword renaming.

    Apply the decorator generated by `_rename_parameter` to functions with a
    recently renamed parameter to maintain backward-compatibility.

    After decoration, the function behaves as follows:
    If only the new parameter is passed into the function, behave as usual.
    If only the old parameter is passed into the function (as a keyword), raise
    a DeprecationWarning if `dep_version` is provided, and behave as usual
    otherwise.
    If both old and new parameters are passed into the function, raise a
    DeprecationWarning if `dep_version` is provided, and raise the appropriate
    TypeError (function got multiple values for argument).

    Parameters
    ----------
    old_name : str
        Old name of parameter
    new_name : str
        New name of parameter
    dep_version : str, optional
        Version of SciPy in which old parameter was deprecated in the format
        'X.Y.Z'. If supplied, the deprecation message will indicate that
        support for the old parameter will be removed in version 'X.Y+2.Z'

    Notes
    -----
    Untested with functions that accept *args. Probably won't work as written.

    """
    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                if dep_version:
                    end_version = dep_version.split('.')
                    end_version[1] = str(int(end_version[1]) + 2)
                    end_version = '.'.join(end_version)
                    message = (f"Use of keyword argument `{old_name}` is "
                               f"deprecated and replaced by `{new_name}`.  "
                               f"Support for `{old_name}` will be removed "
                               f"in SciPy {end_version}.")
                    warnings.warn(message, DeprecationWarning, stacklevel=2)
                if new_name in kwargs:
                    message = (f"{fun.__name__}() got multiple values for "
                               f"argument now known as `{new_name}`")
                    raise TypeError(message)
                kwargs[new_name] = kwargs.pop(old_name)
            return fun(*args, **kwargs)
        return wrapper
    return decorator


def _rng_spawn(rng, n_children):
    # spawns independent RNGs from a parent RNG
    bg = rng._bit_generator
    ss = bg._seed_seq
    child_rngs = [np.random.Generator(type(bg)(child_ss))
                  for child_ss in ss.spawn(n_children)]
    return child_rngs


def _get_nan(*data, xp=None):
    xp = array_namespace(*data) if xp is None else xp
    # Get NaN of appropriate dtype for data
    data = [xp.asarray(item) for item in data]
    try:
        min_float = getattr(xp, 'float16', xp.float32)
        dtype = xp.result_type(*data, min_float)  # must be at least a float
    except DTypePromotionError:
        # fallback to float64
        dtype = xp.float64
    return xp.asarray(xp.nan, dtype=dtype)[()]


def normalize_axis_index(axis, ndim):
    # Check if `axis` is in the correct range and normalize it
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        raise AxisError(msg)

    if axis < 0:
        axis = axis + ndim
    return axis


def _call_callback_maybe_halt(callback, res):
    """Call wrapped callback; return True if algorithm should stop.

    Parameters
    ----------
    callback : callable or None
        A user-provided callback wrapped with `_wrap_callback`
    res : OptimizeResult
        Information about the current iterate

    Returns
    -------
    halt : bool
        True if minimization should stop

    """
    if callback is None:
        return False
    try:
        callback(res)
        return False
    except StopIteration:
        callback.stop_iteration = True
        return True


class _RichResult(dict):
    """ Container for multiple outputs with pretty-printing """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]

    def __repr__(self):
        order_keys = ['message', 'success', 'status', 'fun', 'funl', 'x', 'xl',
                      'col_ind', 'nit', 'lower', 'upper', 'eqlin', 'ineqlin',
                      'converged', 'flag', 'function_calls', 'iterations',
                      'root']
        order_keys = getattr(self, '_order_keys', order_keys)
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {'slack', 'con', 'crossover_nit', '_order_keys'}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def _indenter(s, n=0):
    """
    Ensures that lines after the first are indented by the specified amount
    """
    split = s.split("\n")
    indent = " "*n
    return ("\n" + indent).join(split)


def _float_formatter_10(x):
    """
    Returns a string representation of a float with exactly ten characters
    """
    if np.isposinf(x):
        return "       inf"
    elif np.isneginf(x):
        return "      -inf"
    elif np.isnan(x):
        return "       nan"
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)


def _dict_formatter(d, n=0, mplus=1, sorter=None):
    """
    Pretty printer for dictionaries

    `n` keeps track of the starting indentation;
    lines are indented by this much after a line break.
    `mplus` is additional left padding applied to keys
    """
    if isinstance(d, dict):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys
        s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                       _indenter(_dict_formatter(v, m+n+2, 0, sorter), m+2)
                       for k, v in sorter(d)])  # +2 for ': '
    else:
        # By default, NumPy arrays print with linewidth=76. `n` is
        # the indent at which a line begins printing, so it is subtracted
        # from the default to avoid exceeding 76 characters total.
        # `edgeitems` is the number of elements to include before and after
        # ellipses when arrays are not shown in full.
        # `threshold` is the maximum number of elements for which an
        # array is shown in full.
        # These values tend to work well for use with OptimizeResult.
        with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _float_formatter_10}):
            s = str(d)
    return s
<|endoftext|>"""
Extra testing functions that forbid 0d-input, see #21044

While the xp_assert_* functions generally aim to follow the conventions of the
underlying `xp` library, NumPy in particular is inconsistent in its handling
of scalars vs. 0d-arrays, see https://github.com/numpy/numpy/issues/24897.

For example, this means that the following operations (as of v2.0.1) currently
return scalars, even though a 0d-array would often be more appropriate:

    import numpy as np
    np.array(0) * 2     # scalar, not 0d array
    - np.array(0)       # scalar, not 0d-array
    np.sin(np.array(0)) # scalar, not 0d array
    np.mean([1, 2, 3])  # scalar, not 0d array

Libraries like CuPy tend to return a 0d-array in scenarios like those above,
and even `xp.asarray(0)[()]` remains a 0d-array there. To deal with the reality
of the inconsistencies present in NumPy, as well as 20+ years of code on top,
the `xp_assert_*` functions here enforce consistency in the only way that
doesn't go against the tide, i.e. by forbidding 0d-arrays as the return type.

However, when scalars are not generally the expected NumPy return type,
it remains preferable to use the assert functions from
the `scipy._lib._array_api` module, which have less surprising behaviour.
"""
from scipy._lib._array_api import array_namespace, is_numpy
from scipy._lib._array_api import (xp_assert_close as xp_assert_close_base,
                                   xp_assert_equal as xp_assert_equal_base,
                                   xp_assert_less as xp_assert_less_base)

__all__: list[str] = []


def _check_scalar(actual, desired, *, xp=None, **kwargs):
    __tracebackhide__ = True  # Hide traceback for py.test

    if xp is None:
        xp = array_namespace(actual)

    # necessary to handle non-numpy scalars, e.g. bare `0.0` has no shape
    desired = xp.asarray(desired)

    # Only NumPy distinguishes between scalars and arrays;
    # shape check in xp_assert_* is sufficient except for shape == ()
    if not (is_numpy(xp) and desired.shape == ()):
        return

    _msg = ("Result is a NumPy 0d-array. Many SciPy functions intend to follow "
            "the convention of many NumPy functions, returning a scalar when a "
            "0d-array would be correct. The specialized `xp_assert_*` functions "
            "in the `scipy._lib._array_api_no_0d` module err on the side of "
            "caution and do not accept 0d-arrays by default. If the correct "
            "result may legitimately be a 0d-array, pass `check_0d=True`, "
            "or use the `xp_assert_*` functions from `scipy._lib._array_api`.")
    assert xp.isscalar(actual), _msg


def xp_assert_equal(actual, desired, *, check_0d=False, **kwargs):
    # in contrast to xp_assert_equal_base, this defaults to check_0d=False,
    # but will do an extra check in that case, which forbids 0d-arrays for `actual`
    __tracebackhide__ = True  # Hide traceback for py.test

    # array-ness (check_0d == True) is taken care of by the *_base functions
    if not check_0d:
        _check_scalar(actual, desired, **kwargs)
    return xp_assert_equal_base(actual, desired, check_0d=check_0d, **kwargs)


def xp_assert_close(actual, desired, *, check_0d=False, **kwargs):
    # as for xp_assert_equal
    __tracebackhide__ = True

    if not check_0d:
        _check_scalar(actual, desired, **kwargs)
    return xp_assert_close_base(actual, desired, check_0d=check_0d, **kwargs)


def xp_assert_less(actual, desired, *, check_0d=False, **kwargs):
    # as for xp_assert_equal
    __tracebackhide__ = True

    if not check_0d:
        _check_scalar(actual, desired, **kwargs)
    return xp_assert_less_base(actual, desired, check_0d=check_0d, **kwargs)


def assert_array_almost_equal(actual, desired, decimal=6, *args, **kwds):
    """Backwards compatible replacement. In new code, use xp_assert_close instead.
    """
    rtol, atol = 0, 1.5*10**(-decimal)
    return xp_assert_close(actual, desired,
                           atol=atol, rtol=rtol, check_dtype=False, check_shape=False,
                           *args, **kwds)


def assert_almost_equal(actual, desired, decimal=7, *args, **kwds):
    """Backwards compatible replacement. In new code, use xp_assert_close instead.
    """
    rtol, atol = 0, 1.5*10**(-decimal)
    return xp_assert_close(actual, desired,
                           atol=atol, rtol=rtol, check_dtype=False, check_shape=False,
                           *args, **kwds)
<|endoftext|>"""Utility functions to use Python Array API compatible libraries.

For the context about the Array API see:
https://data-apis.org/array-api/latest/purpose_and_scope.html

The SciPy use case of the Array API is described on the following page:
https://data-apis.org/array-api/latest/use_cases.html#use-case-scipy
"""
import os

from types import ModuleType
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from scipy._lib import array_api_compat
from scipy._lib.array_api_compat import (
    is_array_api_obj,
    size as xp_size,
    numpy as np_compat,
    device as xp_device,
    is_numpy_namespace as is_numpy,
    is_cupy_namespace as is_cupy,
    is_torch_namespace as is_torch,
    is_jax_namespace as is_jax,
    is_array_api_strict_namespace as is_array_api_strict
)

__all__ = [
    '_asarray', 'array_namespace', 'assert_almost_equal', 'assert_array_almost_equal',
    'get_xp_devices',
    'is_array_api_strict', 'is_complex', 'is_cupy', 'is_jax', 'is_numpy', 'is_torch', 
    'SCIPY_ARRAY_API', 'SCIPY_DEVICE', 'scipy_namespace_for',
    'xp_assert_close', 'xp_assert_equal', 'xp_assert_less',
    'xp_copy', 'xp_copysign', 'xp_device',
    'xp_moveaxis_to_end', 'xp_ravel', 'xp_real', 'xp_sign', 'xp_size',
    'xp_take_along_axis', 'xp_unsupported_param_msg', 'xp_vector_norm',
]


# To enable array API and strict array-like input validation
SCIPY_ARRAY_API: str | bool = os.environ.get("SCIPY_ARRAY_API", False)
# To control the default device - for use in the test suite only
SCIPY_DEVICE = os.environ.get("SCIPY_DEVICE", "cpu")

_GLOBAL_CONFIG = {
    "SCIPY_ARRAY_API": SCIPY_ARRAY_API,
    "SCIPY_DEVICE": SCIPY_DEVICE,
}


Array: TypeAlias = Any  # To be changed to a Protocol later (see array-api#589)
ArrayLike: TypeAlias = Array | npt.ArrayLike


def _compliance_scipy(arrays):
    """Raise exceptions on known-bad subclasses.

    The following subclasses are not supported and raise and error:
    - `numpy.ma.MaskedArray`
    - `numpy.matrix`
    - NumPy arrays which do not have a boolean or numerical dtype
    - Any array-like which is neither array API compatible nor coercible by NumPy
    - Any array-like which is coerced by NumPy to an unsupported dtype
    """
    for i in range(len(arrays)):
        array = arrays[i]

        from scipy.sparse import issparse
        # this comes from `_util._asarray_validated`
        if issparse(array):
            msg = ('Sparse arrays/matrices are not supported by this function. '
                   'Perhaps one of the `scipy.sparse.linalg` functions '
                   'would work instead.')
            raise ValueError(msg)

        if isinstance(array, np.ma.MaskedArray):
            raise TypeError("Inputs of type `numpy.ma.MaskedArray` are not supported.")
        elif isinstance(array, np.matrix):
            raise TypeError("Inputs of type `numpy.matrix` are not supported.")
        if isinstance(array, np.ndarray | np.generic):
            dtype = array.dtype
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                raise TypeError(f"An argument has dtype `{dtype!r}`; "
                                f"only boolean and numerical dtypes are supported.")
        elif not is_array_api_obj(array):
            try:
                array = np.asanyarray(array)
            except TypeError:
                raise TypeError("An argument is neither array API compatible nor "
                                "coercible by NumPy.")
            dtype = array.dtype
            if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
                message = (
                    f"An argument was coerced to an unsupported dtype `{dtype!r}`; "
                    f"only boolean and numerical dtypes are supported."
                )
                raise TypeError(message)
            arrays[i] = array
    return arrays


def _check_finite(array: Array, xp: ModuleType) -> None:
    """Check for NaNs or Infs."""
    msg = "array must not contain infs or NaNs"
    try:
        if not xp.all(xp.isfinite(array)):
            raise ValueError(msg)
    except TypeError:
        raise ValueError(msg)


def array_namespace(*arrays: Array) -> ModuleType:
    """Get the array API compatible namespace for the arrays xs.

    Parameters
    ----------
    *arrays : sequence of array_like
        Arrays used to infer the common namespace.

    Returns
    -------
    namespace : module
        Common namespace.

    Notes
    -----
    Thin wrapper around `array_api_compat.array_namespace`.

    1. Check for the global switch: SCIPY_ARRAY_API. This can also be accessed
       dynamically through ``_GLOBAL_CONFIG['SCIPY_ARRAY_API']``.
    2. `_compliance_scipy` raise exceptions on known-bad subclasses. See
       its definition for more details.

    When the global switch is False, it defaults to the `numpy` namespace.
    In that case, there is no compliance check. This is a convenience to
    ease the adoption. Otherwise, arrays must comply with the new rules.
    """
    if not _GLOBAL_CONFIG["SCIPY_ARRAY_API"]:
        # here we could wrap the namespace if needed
        return np_compat

    _arrays = [array for array in arrays if array is not None]

    _arrays = _compliance_scipy(_arrays)

    return array_api_compat.array_namespace(*_arrays)


def _asarray(
        array: ArrayLike,
        dtype: Any = None,
        order: Literal['K', 'A', 'C', 'F'] | None = None,
        copy: bool | None = None,
        *,
        xp: ModuleType | None = None,
        check_finite: bool = False,
        subok: bool = False,
    ) -> Array:
    """SciPy-specific replacement for `np.asarray` with `order`, `check_finite`, and
    `subok`.

    Memory layout parameter `order` is not exposed in the Array API standard.
    `order` is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.

    `check_finite` is also not a keyword in the array API standard; included
    here for convenience rather than that having to be a separate function
    call inside SciPy functions.

    `subok` is included to allow this function to preserve the behaviour of
    `np.asanyarray` for NumPy based inputs.
    """
    if xp is None:
        xp = array_namespace(array)
    if is_numpy(xp):
        # Use NumPy API to support order
        if copy is True:
            array = np.array(array, order=order, dtype=dtype, subok=subok)
        elif subok:
            array = np.asanyarray(array, order=order, dtype=dtype)
        else:
            array = np.asarray(array, order=order, dtype=dtype)
    else:
        try:
            array = xp.asarray(array, dtype=dtype, copy=copy)
        except TypeError:
            coerced_xp = array_namespace(xp.asarray(3))
            array = coerced_xp.asarray(array, dtype=dtype, copy=copy)

    if check_finite:
        _check_finite(array, xp)

    return array


def xp_copy(x: Array, *, xp: ModuleType | None = None) -> Array:
    """
    Copies an array.

    Parameters
    ----------
    x : array

    xp : array_namespace

    Returns
    -------
    copy : array
        Copied array

    Notes
    -----
    This copy function does not offer all the semantics of `np.copy`, i.e. the
    `subok` and `order` keywords are not used.
    """
    # Note: for older NumPy versions, `np.asarray` did not support the `copy` kwarg,
    # so this uses our other helper `_asarray`.
    if xp is None:
        xp = array_namespace(x)

    return _asarray(x, copy=True, xp=xp)


def _strict_check(actual, desired, xp, *,
                  check_namespace=True, check_dtype=True, check_shape=True,
                  check_0d=True):
    __tracebackhide__ = True  # Hide traceback for py.test
    if check_namespace:
        _assert_matching_namespace(actual, desired)

    # only NumPy distinguishes between scalars and arrays; we do if check_0d=True.
    # do this first so we can then cast to array (and thus use the array API) below.
    if is_numpy(xp) and check_0d:
        _msg = ("Array-ness does not match:\n Actual: "
                f"{type(actual)}\n Desired: {type(desired)}")
        assert ((xp.isscalar(actual) and xp.isscalar(desired))
                or (not xp.isscalar(actual) and not xp.isscalar(desired))), _msg

    actual = xp.asarray(actual)
    desired = xp.asarray(desired)

    if check_dtype:
        _msg = f"dtypes do not match.\nActual: {actual.dtype}\nDesired: {desired.dtype}"
        assert actual.dtype == desired.dtype, _msg

    if check_shape:
        _msg = f"Shapes do not match.\nActual: {actual.shape}\nDesired: {desired.shape}"
        assert actual.shape == desired.shape, _msg

    desired = xp.broadcast_to(desired, actual.shape)
    return actual, desired


def _assert_matching_namespace(actual, desired):
    __tracebackhide__ = True  # Hide traceback for py.test
    actual = actual if isinstance(actual, tuple) else (actual,)
    desired_space = array_namespace(desired)
    for arr in actual:
        arr_space = array_namespace(arr)
        _msg = (f"Namespaces do not match.\n"
                f"Actual: {arr_space.__name__}\n"
                f"Desired: {desired_space.__name__}")
        assert arr_space == desired_space, _msg


def xp_assert_equal(actual, desired, *, check_namespace=True, check_dtype=True,
                    check_shape=True, check_0d=True, err_msg='', xp=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    if xp is None:
        xp = array_namespace(actual)

    actual, desired = _strict_check(
        actual, desired, xp, check_namespace=check_namespace,
        check_dtype=check_dtype, check_shape=check_shape,
        check_0d=check_0d
    )

    if is_cupy(xp):
        return xp.testing.assert_array_equal(actual, desired, err_msg=err_msg)
    elif is_torch(xp):
        # PyTorch recommends using `rtol=0, atol=0` like this
        # to test for exact equality
        err_msg = None if err_msg == '' else err_msg
        return xp.testing.assert_close(actual, desired, rtol=0, atol=0, equal_nan=True,
                                       check_dtype=False, msg=err_msg)
    # JAX uses `np.testing`
    return np.testing.assert_array_equal(actual, desired, err_msg=err_msg)


def xp_assert_close(actual, desired, *, rtol=None, atol=0, check_namespace=True,
                    check_dtype=True, check_shape=True, check_0d=True,
                    err_msg='', xp=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    if xp is None:
        xp = array_namespace(actual)

    actual, desired = _strict_check(
        actual, desired, xp,
        check_namespace=check_namespace, check_dtype=check_dtype,
        check_shape=check_shape, check_0d=check_0d
    )

    floating = xp.isdtype(actual.dtype, ('real floating', 'complex floating'))
    if rtol is None and floating:
        # multiplier of 4 is used as for `np.float64` this puts the default `rtol`
        # roughly half way between sqrt(eps) and the default for
        # `numpy.testing.assert_allclose`, 1e-7
        rtol = xp.finfo(actual.dtype).eps**0.5 * 4
    elif rtol is None:
        rtol = 1e-7

    if is_cupy(xp):
        return xp.testing.assert_allclose(actual, desired, rtol=rtol,
                                          atol=atol, err_msg=err_msg)
    elif is_torch(xp):
        err_msg = None if err_msg == '' else err_msg
        return xp.testing.assert_close(actual, desired, rtol=rtol, atol=atol,
                                       equal_nan=True, check_dtype=False, msg=err_msg)
    # JAX uses `np.testing`
    return np.testing.assert_allclose(actual, desired, rtol=rtol,
                                      atol=atol, err_msg=err_msg)


def xp_assert_less(actual, desired, *, check_namespace=True, check_dtype=True,
                   check_shape=True, check_0d=True, err_msg='', verbose=True, xp=None):
    __tracebackhide__ = True  # Hide traceback for py.test
    if xp is None:
        xp = array_namespace(actual)

    actual, desired = _strict_check(
        actual, desired, xp, check_namespace=check_namespace,
        check_dtype=check_dtype, check_shape=check_shape,
        check_0d=check_0d
    )

    if is_cupy(xp):
        return xp.testing.assert_array_less(actual, desired,
                                            err_msg=err_msg, verbose=verbose)
    elif is_torch(xp):
        if actual.device.type != 'cpu':
            actual = actual.cpu()
        if desired.device.type != 'cpu':
            desired = desired.cpu()
    # JAX uses `np.testing`
    return np.testing.assert_array_less(actual, desired,
                                        err_msg=err_msg, verbose=verbose)


def assert_array_almost_equal(actual, desired, decimal=6, *args, **kwds):
    """Backwards compatible replacement. In new code, use xp_assert_close instead.
    """
    rtol, atol = 0, 1.5*10**(-decimal)
    return xp_assert_close(actual, desired,
                           atol=atol, rtol=rtol, check_dtype=False, check_shape=False,
                           *args, **kwds)


def assert_almost_equal(actual, desired, decimal=7, *args, **kwds):
    """Backwards compatible replacement. In new code, use xp_assert_close instead.
    """
    rtol, atol = 0, 1.5*10**(-decimal)
    return xp_assert_close(actual, desired,
                           atol=atol, rtol=rtol, check_dtype=False, check_shape=False,
                           *args, **kwds)


def xp_unsupported_param_msg(param: Any) -> str:
    return f'Providing {param!r} is only supported for numpy arrays.'


def is_complex(x: Array, xp: ModuleType) -> bool:
    return xp.isdtype(x.dtype, 'complex floating')


def get_xp_devices(xp: ModuleType) -> list[str] | list[None]:
    """Returns a list of available devices for the given namespace."""
    devices: list[str] = []
    if is_torch(xp):
        devices += ['cpu']
        import torch # type: ignore[import]
        num_cuda = torch.cuda.device_count()
        for i in range(0, num_cuda):
            devices += [f'cuda:{i}']
        if torch.backends.mps.is_available():
            devices += ['mps']
        return devices
    elif is_cupy(xp):
        import cupy # type: ignore[import]
        num_cuda = cupy.cuda.runtime.getDeviceCount()
        for i in range(0, num_cuda):
            devices += [f'cuda:{i}']
        return devices
    elif is_jax(xp):
        import jax # type: ignore[import]
        num_cpu = jax.device_count(backend='cpu')
        for i in range(0, num_cpu):
            devices += [f'cpu:{i}']
        num_gpu = jax.device_count(backend='gpu')
        for i in range(0, num_gpu):
            devices += [f'gpu:{i}']
        num_tpu = jax.device_count(backend='tpu')
        for i in range(0, num_tpu):
            devices += [f'tpu:{i}']
        return devices

    # given namespace is not known to have a list of available devices;
    # return `[None]` so that one can use this in tests for `device=None`.
    return [None]


def scipy_namespace_for(xp: ModuleType) -> ModuleType | None:
    """Return the `scipy`-like namespace of a non-NumPy backend

    That is, return the namespace corresponding with backend `xp` that contains
    `scipy` sub-namespaces like `linalg` and `special`. If no such namespace
    exists, return ``None``. Useful for dispatching.
    """

    if is_cupy(xp):
        import cupyx  # type: ignore[import-not-found,import-untyped]
        return cupyx.scipy

    if is_jax(xp):
        import jax  # type: ignore[import-not-found]
        return jax.scipy

    if is_torch(xp):
        return xp

    return None


# temporary substitute for xp.moveaxis, which is not yet in all backends
# or covered by array_api_compat.
def xp_moveaxis_to_end(
        x: Array,
        source: int,
        /, *,
        xp: ModuleType | None = None) -> Array:
    xp = array_namespace(xp) if xp is None else xp
    axes = list(range(x.ndim))
    temp = axes.pop(source)
    axes = axes + [temp]
    return xp.permute_dims(x, axes)


# temporary substitute for xp.copysign, which is not yet in all backends
# or covered by array_api_compat.
def xp_copysign(x1: Array, x2: Array, /, *, xp: ModuleType | None = None) -> Array:
    # no attempt to account for special cases
    xp = array_namespace(x1, x2) if xp is None else xp
    abs_x1 = xp.abs(x1)
    return xp.where(x2 >= 0, abs_x1, -abs_x1)


# partial substitute for xp.sign, which does not cover the NaN special case
# that I need. (https://github.com/data-apis/array-api-compat/issues/136)
def xp_sign(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    xp = array_namespace(x) if xp is None else xp
    if is_numpy(xp):  # only NumPy implements the special cases correctly
        return xp.sign(x)
    sign = xp.zeros_like(x)
    one = xp.asarray(1, dtype=x.dtype)
    sign = xp.where(x > 0, one, sign)
    sign = xp.where(x < 0, -one, sign)
    sign = xp.where(xp.isnan(x), xp.nan*one, sign)
    return sign

# maybe use `scipy.linalg` if/when array API support is added
def xp_vector_norm(x: Array, /, *,
                   axis: int | tuple[int] | None = None,
                   keepdims: bool = False,
                   ord: int | float = 2,
                   xp: ModuleType | None = None) -> Array:
    xp = array_namespace(x) if xp is None else xp

    if SCIPY_ARRAY_API:
        # check for optional `linalg` extension
        if hasattr(xp, 'linalg'):
            return xp.linalg.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
        else:
            if ord != 2:
                raise ValueError(
                    "only the Euclidean norm (`ord=2`) is currently supported in "
                    "`xp_vector_norm` for backends not implementing the `linalg` "
                    "extension."
                )
            # return (x @ x)**0.5
            # or to get the right behavior with nd, complex arrays
            return xp.sum(xp.conj(x) * x, axis=axis, keepdims=keepdims)**0.5
    else:
        # to maintain backwards compatibility
        return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def xp_ravel(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    # Equivalent of np.ravel written in terms of array API
    # Even though it's one line, it comes up so often that it's worth having
    # this function for readability
    xp = array_namespace(x) if xp is None else xp
    return xp.reshape(x, (-1,))


def xp_real(x: Array, /, *, xp: ModuleType | None = None) -> Array:
    # Convenience wrapper of xp.real that allows non-complex input;
    # see data-apis/array-api#824
    xp = array_namespace(x) if xp is None else xp
    return xp.real(x) if xp.isdtype(x.dtype, 'complex floating') else x


def xp_take_along_axis(arr: Array,
                       indices: Array, /, *,
                       axis: int = -1,
                       xp: ModuleType | None = None) -> Array:
    # Dispatcher for np.take_along_axis for backends that support it;
    # see data-apis/array-api/pull#816
    xp = array_namespace(arr) if xp is None else xp
    if is_torch(xp):
        return xp.take_along_dim(arr, indices, dim=axis)
    elif is_array_api_strict(xp):
        raise NotImplementedError("Array API standard does not define take_along_axis")
    else:
        return xp.take_along_axis(arr, indices, axis)


# utility to broadcast arrays and promote to common dtype
def xp_broadcast_promote(*args, ensure_writeable=False, force_floating=False, xp=None):
    xp = array_namespace(*args) if xp is None else xp

    args = [(_asarray(arg, subok=True) if arg is not None else arg) for arg in args]
    args_not_none = [arg for arg in args if arg is not None]

    # determine minimum dtype
    default_float = xp.asarray(1.).dtype
    dtypes = [arg.dtype for arg in args_not_none]
    try:  # follow library's prefered mixed promotion rules
        dtype = xp.result_type(*dtypes)
        if force_floating and xp.isdtype(dtype, 'integral'):
            # If we were to add `default_float` before checking whether the result
            # type is otherwise integral, we risk promotion from lower float.
            dtype = xp.result_type(dtype, default_float)
    except TypeError:  # mixed type promotion isn't defined
        float_dtypes = [dtype for dtype in dtypes
                        if not xp.isdtype(dtype, 'integral')]
        if float_dtypes:
            dtype = xp.result_type(*float_dtypes, default_float)
        elif force_floating:
            dtype = default_float
        else:
            dtype = xp.result_type(*dtypes)

    # determine result shape
    shapes = {arg.shape for arg in args_not_none}
    try:
        shape = (np.broadcast_shapes(*shapes) if len(shapes) != 1
                 else args_not_none[0].shape)
    except ValueError as e:
        message = "Array shapes are incompatible for broadcasting."
        raise ValueError(message) from e

    out = []
    for arg in args:
        if arg is None:
            out.append(arg)
            continue

        # broadcast only if needed
        # Even if two arguments need broadcasting, this is faster than
        # `broadcast_arrays`, especially since we've already determined `shape`
        if arg.shape != shape:
            kwargs = {'subok': True} if is_numpy(xp) else {}
            arg = xp.broadcast_to(arg, shape, **kwargs)

        # convert dtype/copy only if needed
        if (arg.dtype != dtype) or ensure_writeable:
            arg = xp.astype(arg, dtype, copy=True)
        out.append(arg)

    return out


def xp_float_to_complex(arr: Array, xp: ModuleType | None = None) -> Array:
    xp = array_namespace(arr) if xp is None else xp
    arr_dtype = arr.dtype
    # The standard float dtypes are float32 and float64.
    # Convert float32 to complex64,
    # and float64 (and non-standard real dtypes) to complex128
    if xp.isdtype(arr_dtype, xp.float32):
        arr = xp.astype(arr, xp.complex64)
    elif xp.isdtype(arr_dtype, 'real floating'):
        arr = xp.astype(arr, xp.complex128)

    return arr


def xp_default_dtype(xp):
    """Query the namespace-dependent default floating-point dtype.
    """
    if is_torch(xp):
        # historically, we allow pytorch to keep its default of float32
        return xp.get_default_dtype()
    else:
        # we default to float64
        return xp.float64
<|endoftext|>import warnings

import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
)

from .framework import TrustRegion
from .problem import (
    ObjectiveFunction,
    BoundConstraints,
    LinearConstraints,
    NonlinearConstraints,
    Problem,
)
from .utils import (
    MaxEvalError,
    TargetSuccess,
    CallbackSuccess,
    FeasibleSuccess,
    exact_1d_array,
)
from .settings import (
    ExitStatus,
    Options,
    Constants,
    DEFAULT_OPTIONS,
    DEFAULT_CONSTANTS,
    PRINT_OPTIONS,
)


def minimize(
    fun,
    x0,
    args=(),
    bounds=None,
    constraints=(),
    callback=None,
    options=None,
    **kwargs,
):
    r"""
    Minimize a scalar function using the COBYQA method.

    The Constrained Optimization BY Quadratic Approximations (COBYQA) method is
    a derivative-free optimization method designed to solve general nonlinear
    optimization problems. A complete description of COBYQA is given in [3]_.

    Parameters
    ----------
    fun : {callable, None}
        Objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an array with shape (n,) and `args` is a tuple. If `fun`
        is ``None``, the objective function is assumed to be the zero function,
        resulting in a feasibility problem.
    x0 : array_like, shape (n,)
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function.
    bounds : {`scipy.optimize.Bounds`, array_like, shape (n, 2)}, optional
        Bound constraints of the problem. It can be one of the cases below.

        #. An instance of `scipy.optimize.Bounds`. For the time being, the
           argument ``keep_feasible`` is disregarded, and all the constraints
           are considered unrelaxable and will be enforced.
        #. An array with shape (n, 2). The bound constraints for ``x[i]`` are
           ``bounds[i][0] <= x[i] <= bounds[i][1]``. Set ``bounds[i][0]`` to
           :math:`-\infty` if there is no lower bound, and set ``bounds[i][1]``
           to :math:`\infty` if there is no upper bound.

        The COBYQA method always respect the bound constraints.
    constraints : {Constraint, list}, optional
        General constraints of the problem. It can be one of the cases below.

        #. An instance of `scipy.optimize.LinearConstraint`. The argument
           ``keep_feasible`` is disregarded.
        #. An instance of `scipy.optimize.NonlinearConstraint`. The arguments
           ``jac``, ``hess``, ``keep_feasible``, ``finite_diff_rel_step``, and
           ``finite_diff_jac_sparsity`` are disregarded.

        #. A list, each of whose elements are described in the cases above.

    callback : callable, optional
        A callback executed at each objective function evaluation. The method
        terminates if a ``StopIteration`` exception is raised by the callback
        function. Its signature can be one of the following:

            ``callback(intermediate_result)``

        where ``intermediate_result`` is a keyword parameter that contains an
        instance of `scipy.optimize.OptimizeResult`, with attributes ``x``
        and ``fun``, being the point at which the objective function is
        evaluated and the value of the objective function, respectively. The
        name of the parameter must be ``intermediate_result`` for the callback
        to be passed an instance of `scipy.optimize.OptimizeResult`.

        Alternatively, the callback function can have the signature:

            ``callback(xk)``

        where ``xk`` is the point at which the objective function is evaluated.
        Introspection is used to determine which of the signatures to invoke.
    options : dict, optional
        Options passed to the solver. Accepted keys are:

            disp : bool, optional
                Whether to print information about the optimization procedure.
                Default is ``False``.
            maxfev : int, optional
                Maximum number of function evaluations. Default is ``500 * n``.
            maxiter : int, optional
                Maximum number of iterations. Default is ``1000 * n``.
            target : float, optional
                Target on the objective function value. The optimization
                procedure is terminated when the objective function value of a
                feasible point is less than or equal to this target. Default is
                ``-numpy.inf``.
            feasibility_tol : float, optional
                Tolerance on the constraint violation. If the maximum
                constraint violation at a point is less than or equal to this
                tolerance, the point is considered feasible. Default is
                ``numpy.sqrt(numpy.finfo(float).eps)``.
            radius_init : float, optional
                Initial trust-region radius. Typically, this value should be in
                the order of one tenth of the greatest expected change to `x0`.
                Default is ``1.0``.
            radius_final : float, optional
                Final trust-region radius. It should indicate the accuracy
                required in the final values of the variables. Default is
                ``1e-6``.
            nb_points : int, optional
                Number of interpolation points used to build the quadratic
                models of the objective and constraint functions. Default is
                ``2 * n + 1``.
            scale : bool, optional
                Whether to scale the variables according to the bounds. Default
                is ``False``.
            filter_size : int, optional
                Maximum number of points in the filter. The filter is used to
                select the best point returned by the optimization procedure.
                Default is ``sys.maxsize``.
            store_history : bool, optional
                Whether to store the history of the function evaluations.
                Default is ``False``.
            history_size : int, optional
                Maximum number of function evaluations to store in the history.
                Default is ``sys.maxsize``.
            debug : bool, optional
                Whether to perform additional checks during the optimization
                procedure. This option should be used only for debugging
                purposes and is highly discouraged to general users. Default is
                ``False``.

        Other constants (from the keyword arguments) are described below. They
        are not intended to be changed by general users. They should only be
        changed by users with a deep understanding of the algorithm, who want
        to experiment with different settings.

    Returns
    -------
    `scipy.optimize.OptimizeResult`
        Result of the optimization procedure, with the following fields:

            message : str
                Description of the cause of the termination.
            success : bool
                Whether the optimization procedure terminated successfully.
            status : int
                Termination status of the optimization procedure.
            x : `numpy.ndarray`, shape (n,)
                Solution point.
            fun : float
                Objective function value at the solution point.
            maxcv : float
                Maximum constraint violation at the solution point.
            nfev : int
                Number of function evaluations.
            nit : int
                Number of iterations.

        If ``store_history`` is True, the result also has the following fields:

            fun_history : `numpy.ndarray`, shape (nfev,)
                History of the objective function values.
            maxcv_history : `numpy.ndarray`, shape (nfev,)
                History of the maximum constraint violations.

        A description of the termination statuses is given below.

        .. list-table::
            :widths: 25 75
            :header-rows: 1

            * - Exit status
              - Description
            * - 0
              - The lower bound for the trust-region radius has been reached.
            * - 1
              - The target objective function value has been reached.
            * - 2
              - All variables are fixed by the bound constraints.
            * - 3
              - The callback requested to stop the optimization procedure.
            * - 4
              - The feasibility problem received has been solved successfully.
            * - 5
              - The maximum number of function evaluations has been exceeded.
            * - 6
              - The maximum number of iterations has been exceeded.
            * - -1
              - The bound constraints are infeasible.
            * - -2
              - A linear algebra error occurred.

    Other Parameters
    ----------------
    decrease_radius_factor : float, optional
        Factor by which the trust-region radius is reduced when the reduction
        ratio is low or negative. Default is ``0.5``.
    increase_radius_factor : float, optional
        Factor by which the trust-region radius is increased when the reduction
        ratio is large. Default is ``numpy.sqrt(2.0)``.
    increase_radius_threshold : float, optional
        Threshold that controls the increase of the trust-region radius when
        the reduction ratio is large. Default is ``2.0``.
    decrease_radius_threshold : float, optional
        Threshold used to determine whether the trust-region radius should be
        reduced to the resolution. Default is ``1.4``.
    decrease_resolution_factor : float, optional
        Factor by which the resolution is reduced when the current value is far
        from its final value. Default is ``0.1``.
    large_resolution_threshold : float, optional
        Threshold used to determine whether the resolution is far from its
        final value. Default is ``250.0``.
    moderate_resolution_threshold : float, optional
        Threshold used to determine whether the resolution is close to its
        final value. Default is ``16.0``.
    low_ratio : float, optional
        Threshold used to determine whether the reduction ratio is low. Default
        is ``0.1``.
    high_ratio : float, optional
        Threshold used to determine whether the reduction ratio is high.
        Default is ``0.7``.
    very_low_ratio : float, optional
        Threshold used to determine whether the reduction ratio is very low.
        This is used to determine whether the models should be reset. Default
        is ``0.01``.
    penalty_increase_threshold : float, optional
        Threshold used to determine whether the penalty parameter should be
        increased. Default is ``1.5``.
    penalty_increase_factor : float, optional
        Factor by which the penalty parameter is increased. Default is ``2.0``.
    short_step_threshold : float, optional
        Factor used to determine whether the trial step is too short. Default
        is ``0.5``.
    low_radius_factor : float, optional
        Factor used to determine which interpolation point should be removed
        from the interpolation set at each iteration. Default is ``0.1``.
    byrd_omojokun_factor : float, optional
        Factor by which the trust-region radius is reduced for the computations
        of the normal step in the Byrd-Omojokun composite-step approach.
        Default is ``0.8``.
    threshold_ratio_constraints : float, optional
        Threshold used to determine which constraints should be taken into
        account when decreasing the penalty parameter. Default is ``2.0``.
    large_shift_factor : float, optional
        Factor used to determine whether the point around which the quadratic
        models are built should be updated. Default is ``10.0``.
    large_gradient_factor : float, optional
        Factor used to determine whether the models should be reset. Default is
        ``10.0``.
    resolution_factor : float, optional
        Factor by which the resolution is decreased. Default is ``2.0``.
    improve_tcg : bool, optional
        Whether to improve the steps computed by the truncated conjugate
        gradient method when the trust-region boundary is reached. Default is
        ``True``.

    References
    ----------
    .. [1] J. Nocedal and S. J. Wright. *Numerical Optimization*. Springer Ser.
       Oper. Res. Financ. Eng. Springer, New York, NY, USA, second edition,
       2006. `doi:10.1007/978-0-387-40065-5
       <https://doi.org/10.1007/978-0-387-40065-5>`_.
    .. [2] M. J. D. Powell. A direct search optimization method that models the
       objective and constraint functions by linear interpolation. In S. Gomez
       and J.-P. Hennart, editors, *Advances in Optimization and Numerical
       Analysis*, volume 275 of Math. Appl., pages 51--67. Springer, Dordrecht,
       Netherlands, 1994. `doi:10.1007/978-94-015-8330-5_4
       <https://doi.org/10.1007/978-94-015-8330-5_4>`_.
    .. [3] T. M. Ragonneau. *Model-Based Derivative-Free Optimization Methods
       and Software*. PhD thesis, Department of Applied Mathematics, The Hong
       Kong Polytechnic University, Hong Kong, China, 2022. URL:
       https://theses.lib.polyu.edu.hk/handle/200/12294.

    Examples
    --------
    To demonstrate how to use `minimize`, we first minimize the Rosenbrock
    function implemented in `scipy.optimize` in an unconstrained setting.

    .. testsetup::

        import numpy as np
        np.set_printoptions(precision=3, suppress=True)

    >>> from cobyqa import minimize
    >>> from scipy.optimize import rosen

    To solve the problem using COBYQA, run:

    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = minimize(rosen, x0)
    >>> res.x
    array([1., 1., 1., 1., 1.])

    To see how bound and constraints are handled using `minimize`, we solve
    Example 16.4 of [1]_, defined as

    .. math::

        \begin{aligned}
            \min_{x \in \mathbb{R}^2}   & \quad (x_1 - 1)^2 + (x_2 - 2.5)^2\\
            \text{s.t.}                 & \quad -x_1 + 2x_2 \le 2,\\
                                        & \quad x_1 + 2x_2 \le 6,\\
                                        & \quad x_1 - 2x_2 \le 2,\\
                                        & \quad x_1 \ge 0,\\
                                        & \quad x_2 \ge 0.
        \end{aligned}

    >>> import numpy as np
    >>> from scipy.optimize import Bounds, LinearConstraint

    Its objective function can be implemented as:

    >>> def fun(x):
    ...     return (x[0] - 1.0)**2 + (x[1] - 2.5)**2

    This problem can be solved using `minimize` as:

    >>> x0 = [2.0, 0.0]
    >>> bounds = Bounds([0.0, 0.0], np.inf)
    >>> constraints = LinearConstraint([
    ...     [-1.0, 2.0],
    ...     [1.0, 2.0],
    ...     [1.0, -2.0],
    ... ], -np.inf, [2.0, 6.0, 2.0])
    >>> res = minimize(fun, x0, bounds=bounds, constraints=constraints)
    >>> res.x
    array([1.4, 1.7])

    To see how nonlinear constraints are handled, we solve Problem (F) of [2]_,
    defined as

    .. math::

        \begin{aligned}
            \min_{x \in \mathbb{R}^2}   & \quad -x_1 - x_2\\
            \text{s.t.}                 & \quad x_1^2 - x_2 \le 0,\\
                                        & \quad x_1^2 + x_2^2 \le 1.
        \end{aligned}

    >>> from scipy.optimize import NonlinearConstraint

    Its objective and constraint functions can be implemented as:

    >>> def fun(x):
    ...     return -x[0] - x[1]
    >>>
    >>> def cub(x):
    ...     return [x[0]**2 - x[1], x[0]**2 + x[1]**2]

    This problem can be solved using `minimize` as:

    >>> x0 = [1.0, 1.0]
    >>> constraints = NonlinearConstraint(cub, -np.inf, [0.0, 1.0])
    >>> res = minimize(fun, x0, constraints=constraints)
    >>> res.x
    array([0.707, 0.707])

    Finally, to see how to supply linear and nonlinear constraints
    simultaneously, we solve Problem (G) of [2]_, defined as

    .. math::

        \begin{aligned}
            \min_{x \in \mathbb{R}^3}   & \quad x_3\\
            \text{s.t.}                 & \quad 5x_1 - x_2 + x_3 \ge 0,\\
                                        & \quad -5x_1 - x_2 + x_3 \ge 0,\\
                                        & \quad x_1^2 + x_2^2 + 4x_2 \le x_3.
        \end{aligned}

    Its objective and nonlinear constraint functions can be implemented as:

    >>> def fun(x):
    ...     return x[2]
    >>>
    >>> def cub(x):
    ...     return x[0]**2 + x[1]**2 + 4.0*x[1] - x[2]

    This problem can be solved using `minimize` as:

    >>> x0 = [1.0, 1.0, 1.0]
    >>> constraints = [
    ...     LinearConstraint(
    ...         [[5.0, -1.0, 1.0], [-5.0, -1.0, 1.0]],
    ...         [0.0, 0.0],
    ...         np.inf,
    ...     ),
    ...     NonlinearConstraint(cub, -np.inf, 0.0),
    ... ]
    >>> res = minimize(fun, x0, constraints=constraints)
    >>> res.x
    array([ 0., -3., -3.])
    """
    # Get basic options that are needed for the initialization.
    if options is None:
        options = {}
    else:
        options = dict(options)
    verbose = options.get(Options.VERBOSE, DEFAULT_OPTIONS[Options.VERBOSE])
    verbose = bool(verbose)
    feasibility_tol = options.get(
        Options.FEASIBILITY_TOL,
        DEFAULT_OPTIONS[Options.FEASIBILITY_TOL],
    )
    feasibility_tol = float(feasibility_tol)
    scale = options.get(Options.SCALE, DEFAULT_OPTIONS[Options.SCALE])
    scale = bool(scale)
    store_history = options.get(
        Options.STORE_HISTORY,
        DEFAULT_OPTIONS[Options.STORE_HISTORY],
    )
    store_history = bool(store_history)
    if Options.HISTORY_SIZE in options and options[Options.HISTORY_SIZE] <= 0:
        raise ValueError("The size of the history must be positive.")
    history_size = options.get(
        Options.HISTORY_SIZE,
        DEFAULT_OPTIONS[Options.HISTORY_SIZE],
    )
    history_size = int(history_size)
    if Options.FILTER_SIZE in options and options[Options.FILTER_SIZE] <= 0:
        raise ValueError("The size of the filter must be positive.")
    filter_size = options.get(
        Options.FILTER_SIZE,
        DEFAULT_OPTIONS[Options.FILTER_SIZE],
    )
    filter_size = int(filter_size)
    debug = options.get(Options.DEBUG, DEFAULT_OPTIONS[Options.DEBUG])
    debug = bool(debug)

    # Initialize the objective function.
    if not isinstance(args, tuple):
        args = (args,)
    obj = ObjectiveFunction(fun, verbose, debug, *args)

    # Initialize the bound constraints.
    if not hasattr(x0, "__len__"):
        x0 = [x0]
    n_orig = len(x0)
    bounds = BoundConstraints(_get_bounds(bounds, n_orig))

    # Initialize the constraints.
    linear_constraints, nonlinear_constraints = _get_constraints(constraints)
    linear = LinearConstraints(linear_constraints, n_orig, debug)
    nonlinear = NonlinearConstraints(nonlinear_constraints, verbose, debug)

    # Initialize the problem (and remove the fixed variables).
    pb = Problem(
        obj,
        x0,
        bounds,
        linear,
        nonlinear,
        callback,
        feasibility_tol,
        scale,
        store_history,
        history_size,
        filter_size,
        debug,
    )

    # Set the default options.
    _set_default_options(options, pb.n)
    constants = _set_default_constants(**kwargs)

    # Initialize the models and skip the computations whenever possible.
    if not pb.bounds.is_feasible:
        # The bound constraints are infeasible.
        return _build_result(
            pb,
            0.0,
            False,
            ExitStatus.INFEASIBLE_ERROR,
            0,
            options,
        )
    elif pb.n == 0:
        # All variables are fixed by the bound constraints.
        return _build_result(
            pb,
            0.0,
            True,
            ExitStatus.FIXED_SUCCESS,
            0,
            options,
        )
    if verbose:
        print("Starting the optimization procedure.")
        print(f"Initial trust-region radius: {options[Options.RHOBEG]}.")
        print(f"Final trust-region radius: {options[Options.RHOEND]}.")
        print(
            f"Maximum number of function evaluations: "
            f"{options[Options.MAX_EVAL]}."
        )
        print(f"Maximum number of iterations: {options[Options.MAX_ITER]}.")
        print()
    try:
        framework = TrustRegion(pb, options, constants)
    except TargetSuccess:
        # The target on the objective function value has been reached
        return _build_result(
            pb,
            0.0,
            True,
            ExitStatus.TARGET_SUCCESS,
            0,
            options,
        )
    except CallbackSuccess:
        # The callback raised a StopIteration exception.
        return _build_result(
            pb,
            0.0,
            True,
            ExitStatus.CALLBACK_SUCCESS,
            0,
            options,
        )
    except FeasibleSuccess:
        # The feasibility problem has been solved successfully.
        return _build_result(
            pb,
            0.0,
            True,
            ExitStatus.FEASIBLE_SUCCESS,
            0,
            options,
        )
    except MaxEvalError:
        # The maximum number of function evaluations has been exceeded.
        return _build_result(
            pb,
            0.0,
            False,
            ExitStatus.MAX_ITER_WARNING,
            0,
            options,
        )
    except np.linalg.LinAlgError:
        # The construction of the initial interpolation set failed.
        return _build_result(
            pb,
            0.0,
            False,
            ExitStatus.LINALG_ERROR,
            0,
            options,
        )

    # Start the optimization procedure.
    success = False
    n_iter = 0
    k_new = None
    n_short_steps = 0
    n_very_short_steps = 0
    n_alt_models = 0
    while True:
        # Stop the optimization procedure if the maximum number of iterations
        # has been exceeded. We do not write the main loop as a for loop
        # because we want to access the number of iterations outside the loop.
        if n_iter >= options[Options.MAX_ITER]:
            status = ExitStatus.MAX_ITER_WARNING
            break
        n_iter += 1

        # Update the point around which the quadratic models are built.
        if (
            np.linalg.norm(
                framework.x_best - framework.models.interpolation.x_base
            )
            >= constants[Constants.LARGE_SHIFT_FACTOR] * framework.radius
        ):
            framework.shift_x_base(options)

        # Evaluate the trial step.
        radius_save = framework.radius
        normal_step, tangential_step = framework.get_trust_region_step(options)
        step = normal_step + tangential_step
        s_norm = np.linalg.norm(step)

        # If the trial step is too short, we do not attempt to evaluate the
        # objective and constraint functions. Instead, we reduce the
        # trust-region radius and check whether the resolution should be
        # enhanced and whether the geometry of the interpolation set should be
        # improved. Otherwise, we entertain a classical iteration. The
        # criterion for performing an exceptional jump is taken from NEWUOA.
        if (
            s_norm
            <= constants[Constants.SHORT_STEP_THRESHOLD] * framework.resolution
        ):
            framework.radius *= constants[Constants.DECREASE_RESOLUTION_FACTOR]
            if radius_save > framework.resolution:
                n_short_steps = 0
                n_very_short_steps = 0
            else:
                n_short_steps += 1
                n_very_short_steps += 1
                if s_norm > 0.1 * framework.resolution:
                    n_very_short_steps = 0
            enhance_resolution = n_short_steps >= 5 or n_very_short_steps >= 3
            if enhance_resolution:
                n_short_steps = 0
                n_very_short_steps = 0
                improve_geometry = False
            else:
                try:
                    k_new, dist_new = framework.get_index_to_remove()
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break
                improve_geometry = dist_new > max(
                    framework.radius,
                    constants[Constants.RESOLUTION_FACTOR]
                    * framework.resolution,
                )
        else:
            # Increase the penalty parameter if necessary.
            same_best_point = framework.increase_penalty(step)
            if same_best_point:
                # Evaluate the objective and constraint functions.
                try:
                    fun_val, cub_val, ceq_val = _eval(
                        pb,
                        framework,
                        step,
                        options,
                    )
                except TargetSuccess:
                    status = ExitStatus.TARGET_SUCCESS
                    success = True
                    break
                except FeasibleSuccess:
                    status = ExitStatus.FEASIBLE_SUCCESS
                    success = True
                    break
                except CallbackSuccess:
                    status = ExitStatus.CALLBACK_SUCCESS
                    success = True
                    break
                except MaxEvalError:
                    status = ExitStatus.MAX_EVAL_WARNING
                    break

                # Perform a second-order correction step if necessary.
                merit_old = framework.merit(
                    framework.x_best,
                    framework.fun_best,
                    framework.cub_best,
                    framework.ceq_best,
                )
                merit_new = framework.merit(
                    framework.x_best + step, fun_val, cub_val, ceq_val
                )
                if (
                    pb.type == "nonlinearly constrained"
                    and merit_new > merit_old
                    and np.linalg.norm(normal_step)
                    > constants[Constants.BYRD_OMOJOKUN_FACTOR] ** 2.0
                    * framework.radius
                ):
                    soc_step = framework.get_second_order_correction_step(
                        step, options
                    )
                    if np.linalg.norm(soc_step) > 0.0:
                        step += soc_step

                        # Evaluate the objective and constraint functions.
                        try:
                            fun_val, cub_val, ceq_val = _eval(
                                pb,
                                framework,
                                step,
                                options,
                            )
                        except TargetSuccess:
                            status = ExitStatus.TARGET_SUCCESS
                            success = True
                            break
                        except FeasibleSuccess:
                            status = ExitStatus.FEASIBLE_SUCCESS
                            success = True
                            break
                        except CallbackSuccess:
                            status = ExitStatus.CALLBACK_SUCCESS
                            success = True
                            break
                        except MaxEvalError:
                            status = ExitStatus.MAX_EVAL_WARNING
                            break

                # Calculate the reduction ratio.
                ratio = framework.get_reduction_ratio(
                    step,
                    fun_val,
                    cub_val,
                    ceq_val,
                )

                # Choose an interpolation point to remove.
                try:
                    k_new = framework.get_index_to_remove(
                        framework.x_best + step
                    )[0]
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break

                # Update the interpolation set.
                try:
                    ill_conditioned = framework.models.update_interpolation(
                        k_new, framework.x_best + step, fun_val, cub_val,
                        ceq_val
                    )
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break
                framework.set_best_index()

                # Update the trust-region radius.
                framework.update_radius(step, ratio)

                # Attempt to replace the models by the alternative ones.
                if framework.radius <= framework.resolution:
                    if ratio >= constants[Constants.VERY_LOW_RATIO]:
                        n_alt_models = 0
                    else:
                        n_alt_models += 1
                        grad = framework.models.fun_grad(framework.x_best)
                        try:
                            grad_alt = framework.models.fun_alt_grad(
                                framework.x_best
                            )
                        except np.linalg.LinAlgError:
                            status = ExitStatus.LINALG_ERROR
                            break
                        if np.linalg.norm(grad) < constants[
                            Constants.LARGE_GRADIENT_FACTOR
                        ] * np.linalg.norm(grad_alt):
                            n_alt_models = 0
                        if n_alt_models >= 3:
                            try:
                                framework.models.reset_models()
                            except np.linalg.LinAlgError:
                                status = ExitStatus.LINALG_ERROR
                                break
                            n_alt_models = 0

                # Update the Lagrange multipliers.
                framework.set_multipliers(framework.x_best + step)

                # Check whether the resolution should be enhanced.
                try:
                    k_new, dist_new = framework.get_index_to_remove()
                except np.linalg.LinAlgError:
                    status = ExitStatus.LINALG_ERROR
                    break
                improve_geometry = (
                    ill_conditioned
                    or ratio <= constants[Constants.LOW_RATIO]
                    and dist_new
                    > max(
                        framework.radius,
                        constants[Constants.RESOLUTION_FACTOR]
                        * framework.resolution,
                    )
                )
                enhance_resolution = (
                    radius_save <= framework.resolution
                    and ratio <= constants[Constants.LOW_RATIO]
                    and not improve_geometry
                )
            else:
                # When increasing the penalty parameter, the best point so far
                # may change. In this case, we restart the iteration.
                enhance_resolution = False
                improve_geometry = False

        # Reduce the resolution if necessary.
        if enhance_resolution:
            if framework.resolution <= options[Options.RHOEND]:
                success = True
                status = ExitStatus.RADIUS_SUCCESS
                break
            framework.enhance_resolution(options)
            framework.decrease_penalty()

            if verbose:
                maxcv_val = pb.maxcv(
                    framework.x_best, framework.cub_best, framework.ceq_best
                )
                _print_step(
                    f"New trust-region radius: {framework.resolution}",
                    pb,
                    pb.build_x(framework.x_best),
                    framework.fun_best,
                    maxcv_val,
                    pb.n_eval,
                    n_iter,
                )
                print()

        # Improve the geometry of the interpolation set if necessary.
        if improve_geometry:
            try:
                step = framework.get_geometry_step(k_new, options)
            except np.linalg.LinAlgError:
                status = ExitStatus.LINALG_ERROR
                break

            # Evaluate the objective and constraint functions.
            try:
                fun_val, cub_val, ceq_val = _eval(pb, framework, step, options)
            except TargetSuccess:
                status = ExitStatus.TARGET_SUCCESS
                success = True
                break
            except FeasibleSuccess:
                status = ExitStatus.FEASIBLE_SUCCESS
                success = True
                break
            except CallbackSuccess:
                status = ExitStatus.CALLBACK_SUCCESS
                success = True
                break
            except MaxEvalError:
                status = ExitStatus.MAX_EVAL_WARNING
                break

            # Update the interpolation set.
            try:
                framework.models.update_interpolation(
                    k_new,
                    framework.x_best + step,
                    fun_val,
                    cub_val,
                    ceq_val,
                )
            except np.linalg.LinAlgError:
                status = ExitStatus.LINALG_ERROR
                break
            framework.set_best_index()

    return _build_result(
        pb,
        framework.penalty,
        success,
        status,
        n_iter,
        options,
    )


def _get_bounds(bounds, n):
    """
    Uniformize the bounds.
    """
    if bounds is None:
        return Bounds(np.full(n, -np.inf), np.full(n, np.inf))
    elif isinstance(bounds, Bounds):
        if bounds.lb.shape != (n,) or bounds.ub.shape != (n,):
            raise ValueError(f"The bounds must have {n} elements.")
        return Bounds(bounds.lb, bounds.ub)
    elif hasattr(bounds, "__len__"):
        bounds = np.asarray(bounds)
        if bounds.shape != (n, 2):
            raise ValueError(
                "The shape of the bounds is not compatible with "
                "the number of variables."
            )
        return Bounds(bounds[:, 0], bounds[:, 1])
    else:
        raise TypeError(
            "The bounds must be an instance of "
            "scipy.optimize.Bounds or an array-like object."
        )


def _get_constraints(constraints):
    """
    Extract the linear and nonlinear constraints.
    """
    if isinstance(constraints, dict) or not hasattr(constraints, "__len__"):
        constraints = (constraints,)

    # Extract the linear and nonlinear constraints.
    linear_constraints = []
    nonlinear_constraints = []
    for constraint in constraints:
        if isinstance(constraint, LinearConstraint):
            lb = exact_1d_array(
                constraint.lb,
                "The lower bound of the linear constraints must be a vector.",
            )
            ub = exact_1d_array(
                constraint.ub,
                "The upper bound of the linear constraints must be a vector.",
            )
            linear_constraints.append(
                LinearConstraint(
                    constraint.A,
                    *np.broadcast_arrays(lb, ub),
                )
            )
        elif isinstance(constraint, NonlinearConstraint):
            lb = exact_1d_array(
                constraint.lb,
                "The lower bound of the "
                "nonlinear constraints must be a "
                "vector.",
            )
            ub = exact_1d_array(
                constraint.ub,
                "The upper bound of the "
                "nonlinear constraints must be a "
                "vector.",
            )
            nonlinear_constraints.append(
                NonlinearConstraint(
                    constraint.fun,
                    *np.broadcast_arrays(lb, ub),
                )
            )
        elif isinstance(constraint, dict):
            if "type" not in constraint or constraint["type"] not in (
                "eq",
                "ineq",
            ):
                raise ValueError('The constraint type must be "eq" or "ineq".')
            if "fun" not in constraint or not callable(constraint["fun"]):
                raise ValueError("The constraint function must be callable.")
            nonlinear_constraints.append(
                {
                    "fun": constraint["fun"],
                    "type": constraint["type"],
                    "args": constraint.get("args", ()),
                }
            )
        else:
            raise TypeError(
                "The constraints must be instances of "
                "scipy.optimize.LinearConstraint, "
                "scipy.optimize.NonlinearConstraint, or dict."
            )
    return linear_constraints, nonlinear_constraints


def _set_default_options(options, n):
    """
    Set the default options.
    """
    if Options.RHOBEG in options and options[Options.RHOBEG] <= 0.0:
        raise ValueError("The initial trust-region radius must be positive.")
    if Options.RHOEND in options and options[Options.RHOEND] < 0.0:
        raise ValueError("The final trust-region radius must be nonnegative.")
    if Options.RHOBEG in options and Options.RHOEND in options:
        if options[Options.RHOBEG] < options[Options.RHOEND]:
            raise ValueError(
                "The initial trust-region radius must be greater "
                "than or equal to the final trust-region radius."
            )
    elif Options.RHOBEG in options:
        options[Options.RHOEND.value] = np.min(
            [
                DEFAULT_OPTIONS[Options.RHOEND],
                options[Options.RHOBEG],
            ]
        )
    elif Options.RHOEND in options:
        options[Options.RHOBEG.value] = np.max(
            [
                DEFAULT_OPTIONS[Options.RHOBEG],
                options[Options.RHOEND],
            ]
        )
    else:
        options[Options.RHOBEG.value] = DEFAULT_OPTIONS[Options.RHOBEG]
        options[Options.RHOEND.value] = DEFAULT_OPTIONS[Options.RHOEND]
    options[Options.RHOBEG.value] = float(options[Options.RHOBEG])
    options[Options.RHOEND.value] = float(options[Options.RHOEND])
    if Options.NPT in options and options[Options.NPT] <= 0:
        raise ValueError("The number of interpolation points must be "
                         "positive.")
    if (
        Options.NPT in options
        and options[Options.NPT] > ((n + 1) * (n + 2)) // 2
    ):
        raise ValueError(
            f"The number of interpolation points must be at most "
            f"{((n + 1) * (n + 2)) // 2}."
        )
    options.setdefault(Options.NPT.value, DEFAULT_OPTIONS[Options.NPT](n))
    options[Options.NPT.value] = int(options[Options.NPT])
    if Options.MAX_EVAL in options and options[Options.MAX_EVAL] <= 0:
        raise ValueError(
            "The maximum number of function evaluations must be positive."
        )
    options.setdefault(
        Options.MAX_EVAL.value,
        np.max(
            [
                DEFAULT_OPTIONS[Options.MAX_EVAL](n),
                options[Options.NPT] + 1,
            ]
        ),
    )
    options[Options.MAX_EVAL.value] = int(options[Options.MAX_EVAL])
    if Options.MAX_ITER in options and options[Options.MAX_ITER] <= 0:
        raise ValueError("The maximum number of iterations must be positive.")
    options.setdefault(
        Options.MAX_ITER.value,
        DEFAULT_OPTIONS[Options.MAX_ITER](n),
    )
    options[Options.MAX_ITER.value] = int(options[Options.MAX_ITER])
    options.setdefault(Options.TARGET.value, DEFAULT_OPTIONS[Options.TARGET])
    options[Options.TARGET.value] = float(options[Options.TARGET])
    options.setdefault(
        Options.FEASIBILITY_TOL.value,
        DEFAULT_OPTIONS[Options.FEASIBILITY_TOL],
    )
    options[Options.FEASIBILITY_TOL.value] = float(
        options[Options.FEASIBILITY_TOL]
    )
    options.setdefault(Options.VERBOSE.value, DEFAULT_OPTIONS[Options.VERBOSE])
    options[Options.VERBOSE.value] = bool(options[Options.VERBOSE])
    options.setdefault(Options.SCALE.value, DEFAULT_OPTIONS[Options.SCALE])
    options[Options.SCALE.value] = bool(options[Options.SCALE])
    options.setdefault(
        Options.FILTER_SIZE.value,
        DEFAULT_OPTIONS[Options.FILTER_SIZE],
    )
    options[Options.FILTER_SIZE.value] = int(options[Options.FILTER_SIZE])
    options.setdefault(
        Options.STORE_HISTORY.value,
        DEFAULT_OPTIONS[Options.STORE_HISTORY],
    )
    options[Options.STORE_HISTORY.value] = bool(options[Options.STORE_HISTORY])
    options.setdefault(
        Options.HISTORY_SIZE.value,
        DEFAULT_OPTIONS[Options.HISTORY_SIZE],
    )
    options[Options.HISTORY_SIZE.value] = int(options[Options.HISTORY_SIZE])
    options.setdefault(Options.DEBUG.value, DEFAULT_OPTIONS[Options.DEBUG])
    options[Options.DEBUG.value] = bool(options[Options.DEBUG])

    # Check whether they are any unknown options.
    for key in options:
        if key not in Options.__members__.values():
            warnings.warn(f"Unknown option: {key}.", RuntimeWarning, 3)


def _set_default_constants(**kwargs):
    """
    Set the default constants.
    """
    constants = dict(kwargs)
    constants.setdefault(
        Constants.DECREASE_RADIUS_FACTOR.value,
        DEFAULT_CONSTANTS[Constants.DECREASE_RADIUS_FACTOR],
    )
    constants[Constants.DECREASE_RADIUS_FACTOR.value] = float(
        constants[Constants.DECREASE_RADIUS_FACTOR]
    )
    if (
        constants[Constants.DECREASE_RADIUS_FACTOR] <= 0.0
        or constants[Constants.DECREASE_RADIUS_FACTOR] >= 1.0
    ):
        raise ValueError(
            "The constant decrease_radius_factor must be in the interval "
            "(0, 1)."
        )
    constants.setdefault(
        Constants.INCREASE_RADIUS_THRESHOLD.value,
        DEFAULT_CONSTANTS[Constants.INCREASE_RADIUS_THRESHOLD],
    )
    constants[Constants.INCREASE_RADIUS_THRESHOLD.value] = float(
        constants[Constants.INCREASE_RADIUS_THRESHOLD]
    )
    if constants[Constants.INCREASE_RADIUS_THRESHOLD] <= 1.0:
        raise ValueError(
            "The constant increase_radius_threshold must be greater than 1."
        )
    if (
        Constants.INCREASE_RADIUS_FACTOR in constants
        and constants[Constants.INCREASE_RADIUS_FACTOR] <= 1.0
    ):
        raise ValueError(
            "The constant increase_radius_factor must be greater than 1."
        )
    if (
        Constants.DECREASE_RADIUS_THRESHOLD in constants
        and constants[Constants.DECREASE_RADIUS_THRESHOLD] <= 1.0
    ):
        raise ValueError(
            "The constant decrease_radius_threshold must be greater than 1."
        )
    if (
        Constants.INCREASE_RADIUS_FACTOR in constants
        and Constants.DECREASE_RADIUS_THRESHOLD in constants
    ):
        if (
            constants[Constants.DECREASE_RADIUS_THRESHOLD]
            >= constants[Constants.INCREASE_RADIUS_FACTOR]
        ):
            raise ValueError(
                "The constant decrease_radius_threshold must be "
                "less than increase_radius_factor."
            )
    elif Constants.INCREASE_RADIUS_FACTOR in constants:
        constants[Constants.DECREASE_RADIUS_THRESHOLD.value] = np.min(
            [
                DEFAULT_CONSTANTS[Constants.DECREASE_RADIUS_THRESHOLD],
                0.5 * (1.0 + constants[Constants.INCREASE_RADIUS_FACTOR]),
            ]
        )
    elif Constants.DECREASE_RADIUS_THRESHOLD in constants:
        constants[Constants.INCREASE_RADIUS_FACTOR.value] = np.max(
            [
                DEFAULT_CONSTANTS[Constants.INCREASE_RADIUS_FACTOR],
                2.0 * constants[Constants.DECREASE_RADIUS_THRESHOLD],
            ]
        )
    else:
        constants[Constants.INCREASE_RADIUS_FACTOR.value] = DEFAULT_CONSTANTS[
            Constants.INCREASE_RADIUS_FACTOR
        ]
        constants[Constants.DECREASE_RADIUS_THRESHOLD.value] = (
            DEFAULT_CONSTANTS[Constants.DECREASE_RADIUS_THRESHOLD])
    constants.setdefault(
        Constants.DECREASE_RESOLUTION_FACTOR.value,
        DEFAULT_CONSTANTS[Constants.DECREASE_RESOLUTION_FACTOR],
    )
    constants[Constants.DECREASE_RESOLUTION_FACTOR.value] = float(
        constants[Constants.DECREASE_RESOLUTION_FACTOR]
    )
    if (
        constants[Constants.DECREASE_RESOLUTION_FACTOR] <= 0.0
        or constants[Constants.DECREASE_RESOLUTION_FACTOR] >= 1.0
    ):
        raise ValueError(
            "The constant decrease_resolution_factor must be in the interval "
            "(0, 1)."
        )
    if (
        Constants.LARGE_RESOLUTION_THRESHOLD in constants
        and constants[Constants.LARGE_RESOLUTION_THRESHOLD] <= 1.0
    ):
        raise ValueError(
            "The constant large_resolution_threshold must be greater than 1."
        )
    if (
        Constants.MODERATE_RESOLUTION_THRESHOLD in constants
        and constants[Constants.MODERATE_RESOLUTION_THRESHOLD] <= 1.0
    ):
        raise ValueError(
            "The constant moderate_resolution_threshold must be greater than "
            "1."
        )
    if (
        Constants.LARGE_RESOLUTION_THRESHOLD in constants
        and Constants.MODERATE_RESOLUTION_THRESHOLD in constants
    ):
        if (
            constants[Constants.MODERATE_RESOLUTION_THRESHOLD]
            > constants[Constants.LARGE_RESOLUTION_THRESHOLD]
        ):
            raise ValueError(
                "The constant moderate_resolution_threshold "
                "must be at most large_resolution_threshold."
            )
    elif Constants.LARGE_RESOLUTION_THRESHOLD in constants:
        constants[Constants.MODERATE_RESOLUTION_THRESHOLD.value] = np.min(
            [
                DEFAULT_CONSTANTS[Constants.MODERATE_RESOLUTION_THRESHOLD],
                constants[Constants.LARGE_RESOLUTION_THRESHOLD],
            ]
        )
    elif Constants.MODERATE_RESOLUTION_THRESHOLD in constants:
        constants[Constants.LARGE_RESOLUTION_THRESHOLD.value] = np.max(
            [
                DEFAULT_CONSTANTS[Constants.LARGE_RESOLUTION_THRESHOLD],
                constants[Constants.MODERATE_RESOLUTION_THRESHOLD],
            ]
        )
    else:
        constants[Constants.LARGE_RESOLUTION_THRESHOLD.value] = (
            DEFAULT_CONSTANTS[Constants.LARGE_RESOLUTION_THRESHOLD]
        )
        constants[Constants.MODERATE_RESOLUTION_THRESHOLD.value] = (
            DEFAULT_CONSTANTS[Constants.MODERATE_RESOLUTION_THRESHOLD]
        )
    if Constants.LOW_RATIO in constants and (
        constants[Constants.LOW_RATIO] <= 0.0
        or constants[Constants.LOW_RATIO] >= 1.0
    ):
        raise ValueError(
            "The constant low_ratio must be in the interval (0, 1)."
        )
    if Constants.HIGH_RATIO in constants and (
        constants[Constants.HIGH_RATIO] <= 0.0
        or constants[Constants.HIGH_RATIO] >= 1.0
    ):
        raise ValueError(
            "The constant high_ratio must be in the interval (0, 1)."
        )
    if Constants.LOW_RATIO in constants and Constants.HIGH_RATIO in constants:
        if constants[Constants.LOW_RATIO] > constants[Constants.HIGH_RATIO]:
            raise ValueError(
                "The constant low_ratio must be at most high_ratio."
            )
    elif Constants.LOW_RATIO in constants:
        constants[Constants.HIGH_RATIO.value] = np.max(
            [
                DEFAULT_CONSTANTS[Constants.HIGH_RATIO],
                constants[Constants.LOW_RATIO],
            ]
        )
    elif Constants.HIGH_RATIO in constants:
        constants[Constants.LOW_RATIO.value] = np.min(
            [
                DEFAULT_CONSTANTS[Constants.LOW_RATIO],
                constants[Constants.HIGH_RATIO],
            ]
        )
    else:
        constants[Constants.LOW_RATIO.value] = DEFAULT_CONSTANTS[
            Constants.LOW_RATIO
        ]
        constants[Constants.HIGH_RATIO.value] = DEFAULT_CONSTANTS[
            Constants.HIGH_RATIO
        ]
    constants.setdefault(
        Constants.VERY_LOW_RATIO.value,
        DEFAULT_CONSTANTS[Constants.VERY_LOW_RATIO],
    )
