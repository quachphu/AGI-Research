    _PyCapsule_IsValid.restype = ctypes.c_bool
    _PyCapsule_SetContext = ctypes.pythonapi.PyCapsule_SetContext
    _PyCapsule_SetContext.argtypes = (ctypes.py_object, ctypes.c_void_p)
    _PyCapsule_SetDestructor = ctypes.pythonapi.PyCapsule_SetDestructor
    _PyCapsule_SetDestructor.argtypes = (ctypes.py_object, ctypes.c_void_p)
    _PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
    _PyCapsule_SetName.argtypes = (ctypes.py_object, ctypes.c_char_p)
    _PyCapsule_SetPointer = ctypes.pythonapi.PyCapsule_SetPointer
    _PyCapsule_SetPointer.argtypes = (ctypes.py_object, ctypes.c_void_p)
    #from _socket import CAPI as _testcapsule
    _testcapsule_name = b'dill._dill._testcapsule'
    _testcapsule = _PyCapsule_New(
        ctypes.cast(_PyCapsule_New, ctypes.c_void_p),
        ctypes.c_char_p(_testcapsule_name),
        None
    )
    PyCapsuleType = type(_testcapsule)
    @register(PyCapsuleType)
    def save_capsule(pickler, obj):
        logger.trace(pickler, "Cap: %s", obj)
        name = _PyCapsule_GetName(obj)
        #warnings.warn('Pickling a PyCapsule (%s) does not pickle any C data structures and could cause segmentation faults or other memory errors when unpickling.' % (name,), PicklingWarning)
        pointer = _PyCapsule_GetPointer(obj, name)
        context = _PyCapsule_GetContext(obj)
        destructor = _PyCapsule_GetDestructor(obj)
        pickler.save_reduce(_create_capsule, (pointer, name, context, destructor), obj=obj)
        logger.trace(pickler, "# Cap")
    _incedental_reverse_typemap['PyCapsuleType'] = PyCapsuleType
    _reverse_typemap['PyCapsuleType'] = PyCapsuleType
    _incedental_types.add(PyCapsuleType)
else:
    _testcapsule = None

@register(ContextType)
def save_context(pickler, obj):
    logger.trace(pickler, "Cx: %s", obj)
    pickler.save_reduce(ContextType, tuple(obj.items()), obj=obj)
    logger.trace(pickler, "# Cx")


#############################
# A quick fix for issue #500
# This should be removed when a better solution is found.

if hasattr(dataclasses, "_HAS_DEFAULT_FACTORY_CLASS"):
    @register(dataclasses._HAS_DEFAULT_FACTORY_CLASS)
    def save_dataclasses_HAS_DEFAULT_FACTORY_CLASS(pickler, obj):
        logger.trace(pickler, "DcHDF: %s", obj)
        pickler.write(GLOBAL + b"dataclasses\n_HAS_DEFAULT_FACTORY\n")
        logger.trace(pickler, "# DcHDF")

if hasattr(dataclasses, "MISSING"):
    @register(type(dataclasses.MISSING))
    def save_dataclasses_MISSING_TYPE(pickler, obj):
        logger.trace(pickler, "DcM: %s", obj)
        pickler.write(GLOBAL + b"dataclasses\nMISSING\n")
        logger.trace(pickler, "# DcM")

if hasattr(dataclasses, "KW_ONLY"):
    @register(type(dataclasses.KW_ONLY))
    def save_dataclasses_KW_ONLY_TYPE(pickler, obj):
        logger.trace(pickler, "DcKWO: %s", obj)
        pickler.write(GLOBAL + b"dataclasses\nKW_ONLY\n")
        logger.trace(pickler, "# DcKWO")

if hasattr(dataclasses, "_FIELD_BASE"):
    @register(dataclasses._FIELD_BASE)
    def save_dataclasses_FIELD_BASE(pickler, obj):
        logger.trace(pickler, "DcFB: %s", obj)
        pickler.write(GLOBAL + b"dataclasses\n" + obj.name.encode() + b"\n")
        logger.trace(pickler, "# DcFB")

#############################

# quick sanity checking
def pickles(obj,exact=False,safe=False,**kwds):
    """
    Quick check if object pickles with dill.

    If *exact=True* then an equality test is done to check if the reconstructed
    object matches the original object.

    If *safe=True* then any exception will raised in copy signal that the
    object is not picklable, otherwise only pickling errors will be trapped.

    Additional keyword arguments are as :func:`dumps` and :func:`loads`.
    """
    if safe: exceptions = (Exception,) # RuntimeError, ValueError
    else:
        exceptions = (TypeError, AssertionError, NotImplementedError, PicklingError, UnpicklingError)
    try:
        pik = copy(obj, **kwds)
        #FIXME: should check types match first, then check content if "exact"
        try:
            #FIXME: should be "(pik == obj).all()" for numpy comparison, though that'll fail if shapes differ
            result = bool(pik.all() == obj.all())
        except (AttributeError, TypeError):
            warnings.filterwarnings('ignore') #FIXME: be specific
            result = pik == obj
            if warnings.filters: del warnings.filters[0]
        if hasattr(result, 'toarray'): # for unusual types like sparse matrix
            result = result.toarray().all()
        if result: return True
        if not exact:
            result = type(pik) == type(obj)
            if result: return result
            # class instances might have been dumped with byref=False
            return repr(type(pik)) == repr(type(obj)) #XXX: InstanceType?
        return False
    except exceptions:
        return False

def check(obj, *args, **kwds):
    """
    Check pickling of an object across another process.

    *python* is the path to the python interpreter (defaults to sys.executable)

    Set *verbose=True* to print the unpickled object in the other process.

    Additional keyword arguments are as :func:`dumps` and :func:`loads`.
    """
   # == undocumented ==
   # python -- the string path or executable name of the selected python
   # verbose -- if True, be verbose about printing warning messages
   # all other args and kwds are passed to dill.dumps #FIXME: ignore on load
    verbose = kwds.pop('verbose', False)
    python = kwds.pop('python', None)
    if python is None:
        import sys
        python = sys.executable
    # type check
    isinstance(python, str)
    import subprocess
    fail = True
    try:
        _obj = dumps(obj, *args, **kwds)
        fail = False
    finally:
        if fail and verbose:
            print("DUMP FAILED")
    #FIXME: fails if python interpreter path contains spaces
    # Use the following instead (which also processes the 'ignore' keyword):
    #    ignore = kwds.pop('ignore', None)
    #    unpickle = "dill.loads(%s, ignore=%s)"%(repr(_obj), repr(ignore))
    #    cmd = [python, "-c", "import dill; print(%s)"%unpickle]
    #    msg = "SUCCESS" if not subprocess.call(cmd) else "LOAD FAILED"
    msg = "%s -c import dill; print(dill.loads(%s))" % (python, repr(_obj))
    msg = "SUCCESS" if not subprocess.call(msg.split(None,2)) else "LOAD FAILED"
    if verbose:
        print(msg)
    return

# use to protect against missing attributes
def is_dill(pickler, child=None):
    "check the dill-ness of your pickler"
    if child is False or not hasattr(pickler.__class__, 'mro'):
        return 'dill' in pickler.__module__
    return Pickler in pickler.__class__.mro()

def _extend():
    """extend pickle with all of dill's registered types"""
    # need to have pickle not choke on _main_module?  use is_dill(pickler)
    for t,func in Pickler.dispatch.items():
        try:
            StockPickler.dispatch[t] = func
        except Exception: #TypeError, PicklingError, UnpicklingError
            logger.trace(pickler, "skip: %s", t)
    return

del diff, _use_diff, use_diff

# EOF
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Anirudh Vegesana (avegesan@cs.stanford.edu)
# Copyright (c) 2021-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
Provides shims for compatibility between versions of dill and Python.

Compatibility shims should be provided in this file. Here are two simple example
use cases.

Deprecation of constructor function:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Assume that we were transitioning _import_module in _dill.py to
the builtin function importlib.import_module when present.

@move_to(_dill)
def _import_module(import_name):
    ... # code already in _dill.py

_import_module = Getattr(importlib, 'import_module', Getattr(_dill, '_import_module', None))

The code will attempt to find import_module in the importlib module. If not
present, it will use the _import_module function in _dill.

Emulate new Python behavior in older Python versions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CellType.cell_contents behaves differently in Python 3.6 and 3.7. It is
read-only in Python 3.6 and writable and deletable in 3.7.

if _dill.OLD37 and _dill.HAS_CTYPES and ...:
    @move_to(_dill)
    def _setattr(object, name, value):
        if type(object) is _dill.CellType and name == 'cell_contents':
            _PyCell_Set.argtypes = (ctypes.py_object, ctypes.py_object)
            _PyCell_Set(object, value)
        else:
            setattr(object, name, value)
... # more cases below

_setattr = Getattr(_dill, '_setattr', setattr)

_dill._setattr will be used when present to emulate Python 3.7 functionality in
older versions of Python while defaulting to the standard setattr in 3.7+.

See this PR for the discussion that lead to this system:
https://github.com/uqfoundation/dill/pull/443
"""

import inspect
import sys

_dill = sys.modules['dill._dill']


class Reduce(object):
    """
    Reduce objects are wrappers used for compatibility enforcement during
    unpickle-time. They should only be used in calls to pickler.save and
    other Reduce objects. They are only evaluated within unpickler.load.

    Pickling a Reduce object makes the two implementations equivalent:

    pickler.save(Reduce(*reduction))

    pickler.save_reduce(*reduction, obj=reduction)
    """
    __slots__ = ['reduction']
    def __new__(cls, *reduction, **kwargs):
        """
        Args:
            *reduction: a tuple that matches the format given here:
              https://docs.python.org/3/library/pickle.html#object.__reduce__
            is_callable: a bool to indicate that the object created by
              unpickling `reduction` is callable. If true, the current Reduce
              is allowed to be used as the function in further save_reduce calls
              or Reduce objects.
        """
        is_callable = kwargs.get('is_callable', False) # Pleases Py2. Can be removed later
        if is_callable:
            self = object.__new__(_CallableReduce)
        else:
            self = object.__new__(Reduce)
        self.reduction = reduction
        return self
    def __repr__(self):
        return 'Reduce%s' % (self.reduction,)
    def __copy__(self):
        return self # pragma: no cover
    def __deepcopy__(self, memo):
        return self # pragma: no cover
    def __reduce__(self):
        return self.reduction
    def __reduce_ex__(self, protocol):
        return self.__reduce__()

class _CallableReduce(Reduce):
    # A version of Reduce for functions. Used to trick pickler.save_reduce into
    # thinking that Reduce objects of functions are themselves meaningful functions.
    def __call__(self, *args, **kwargs):
        reduction = self.__reduce__()
        func = reduction[0]
        f_args = reduction[1]
        obj = func(*f_args)
        return obj(*args, **kwargs)

__NO_DEFAULT = _dill.Sentinel('Getattr.NO_DEFAULT')

def Getattr(object, name, default=__NO_DEFAULT):
    """
    A Reduce object that represents the getattr operation. When unpickled, the
    Getattr will access an attribute 'name' of 'object' and return the value
    stored there. If the attribute doesn't exist, the default value will be
    returned if present.

    The following statements are equivalent:

    Getattr(collections, 'OrderedDict')
    Getattr(collections, 'spam', None)
    Getattr(*args)

    Reduce(getattr, (collections, 'OrderedDict'))
    Reduce(getattr, (collections, 'spam', None))
    Reduce(getattr, args)

    During unpickling, the first two will result in collections.OrderedDict and
    None respectively because the first attribute exists and the second one does
    not, forcing it to use the default value given in the third argument.
    """

    if default is Getattr.NO_DEFAULT:
        reduction = (getattr, (object, name))
    else:
        reduction = (getattr, (object, name, default))

    return Reduce(*reduction, is_callable=callable(default))

Getattr.NO_DEFAULT = __NO_DEFAULT
del __NO_DEFAULT

def move_to(module, name=None):
    def decorator(func):
        if name is None:
            fname = func.__name__
        else:
            fname = name
        module.__dict__[fname] = func
        func.__module__ = module.__name__
        return func
    return decorator

def register_shim(name, default):
    """
    A easier to understand and more compact way of "softly" defining a function.
    These two pieces of code are equivalent:

    if _dill.OLD3X:
        def _create_class():
            ...
    _create_class = register_shim('_create_class', types.new_class)

    if _dill.OLD3X:
        @move_to(_dill)
        def _create_class():
            ...
    _create_class = Getattr(_dill, '_create_class', types.new_class)

    Intuitively, it creates a function or object in the versions of dill/python
    that require special reimplementations, and use a core library or default
    implementation if that function or object does not exist.
    """
    func = globals().get(name)
    if func is not None:
        _dill.__dict__[name] = func
        func.__module__ = _dill.__name__

    if default is Getattr.NO_DEFAULT:
        reduction = (getattr, (_dill, name))
    else:
        reduction = (getattr, (_dill, name, default))

    return Reduce(*reduction, is_callable=callable(default))

######################
## Compatibility Shims are defined below
######################

_CELL_EMPTY = register_shim('_CELL_EMPTY', None)

_setattr = register_shim('_setattr', setattr)
_delattr = register_shim('_delattr', delattr)
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE

# author, version, license, and long description
try: # the package is installed
    from .__info__ import __version__, __author__, __doc__, __license__
except: # pragma: no cover
    import os
    import sys
    parent = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(parent)
    # get distribution meta info
    from version import (__version__, __author__,
                         get_license_text, get_readme_as_rst)
    __license__ = get_license_text(os.path.join(parent, 'LICENSE'))
    __license__ = "\n%s" % __license__
    __doc__ = get_readme_as_rst(os.path.join(parent, 'README.md'))
    del os, sys, parent, get_license_text, get_readme_as_rst


from ._dill import (
    dump, dumps, load, loads, copy,
    Pickler, Unpickler, register, pickle, pickles, check,
    DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, HANDLE_FMODE, CONTENTS_FMODE, FILE_FMODE,
    PickleError, PickleWarning, PicklingError, PicklingWarning, UnpicklingError,
    UnpicklingWarning,
)
from .session import (
    dump_module, load_module, load_module_asdict,
    dump_session, load_session # backward compatibility
)
from . import detect, logger, session, source, temp

# get global settings
from .settings import settings

# make sure "trace" is turned off
logger.trace(False)

objects = {}
# local import of dill._objects
#from . import _objects
#objects.update(_objects.succeeds)
#del _objects

# local import of dill.objtypes
from . import objtypes as types

def load_types(pickleable=True, unpickleable=True):
    """load pickleable and/or unpickleable types to ``dill.types``

    ``dill.types`` is meant to mimic the ``types`` module, providing a
    registry of object types.  By default, the module is empty (for import
    speed purposes). Use the ``load_types`` function to load selected object
    types to the ``dill.types`` module.

    Args:
        pickleable (bool, default=True): if True, load pickleable types.
        unpickleable (bool, default=True): if True, load unpickleable types.

    Returns:
        None
    """
    from importlib import reload
    # local import of dill.objects
    from . import _objects
    if pickleable:
        objects.update(_objects.succeeds)
    else:
        [objects.pop(obj,None) for obj in _objects.succeeds]
    if unpickleable:
        objects.update(_objects.failures)
    else:
        [objects.pop(obj,None) for obj in _objects.failures]
    objects.update(_objects.registered)
    del _objects
    # reset contents of types to 'empty'
    [types.__dict__.pop(obj) for obj in list(types.__dict__.keys()) \
                             if obj.find('Type') != -1]
    # add corresponding types from objects to types
    reload(types)

def extend(use_dill=True):
    '''add (or remove) dill types to/from the pickle registry

    by default, ``dill`` populates its types to ``pickle.Pickler.dispatch``.
    Thus, all ``dill`` types are available upon calling ``'import pickle'``.
    To drop all ``dill`` types from the ``pickle`` dispatch, *use_dill=False*.

    Args:
        use_dill (bool, default=True): if True, extend the dispatch table.

    Returns:
        None
    '''
    from ._dill import _revert_extension, _extend
    if use_dill: _extend()
    else: _revert_extension()
    return

extend()


def license():
    """print license"""
    print (__license__)
    return

def citation():
    """print citation"""
    print (__doc__[-491:-118])
    return

# end of file
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Author: Leonardo Gama (@leogama)
# Copyright (c) 2008-2015 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
Pickle and restore the intepreter session.
"""

__all__ = [
    'dump_module', 'load_module', 'load_module_asdict',
    'dump_session', 'load_session' # backward compatibility
]

import re
import os
import sys
import warnings
import pathlib
import tempfile

TEMPDIR = pathlib.PurePath(tempfile.gettempdir())

# Type hints.
from typing import Optional, Union

from dill import _dill, Pickler, Unpickler
from ._dill import (
    BuiltinMethodType, FunctionType, MethodType, ModuleType, TypeType,
    _import_module, _is_builtin_module, _is_imported_module, _main_module,
    _reverse_typemap, __builtin__, UnpicklingError,
)

def _module_map():
    """get map of imported modules"""
    from collections import defaultdict
    from types import SimpleNamespace
    modmap = SimpleNamespace(
        by_name=defaultdict(list),
        by_id=defaultdict(list),
        top_level={},
    )
    for modname, module in sys.modules.items():
        if modname in ('__main__', '__mp_main__') or not isinstance(module, ModuleType):
            continue
        if '.' not in modname:
            modmap.top_level[id(module)] = modname
        for objname, modobj in module.__dict__.items():
            modmap.by_name[objname].append((modobj, modname))
            modmap.by_id[id(modobj)].append((modobj, objname, modname))
    return modmap

IMPORTED_AS_TYPES = (ModuleType, TypeType, FunctionType, MethodType, BuiltinMethodType)
if 'PyCapsuleType' in _reverse_typemap:
    IMPORTED_AS_TYPES += (_reverse_typemap['PyCapsuleType'],)
IMPORTED_AS_MODULES = ('ctypes', 'typing', 'subprocess', 'threading',
                               r'concurrent\.futures(\.\w+)?', r'multiprocessing(\.\w+)?')
IMPORTED_AS_MODULES = tuple(re.compile(x) for x in IMPORTED_AS_MODULES)

def _lookup_module(modmap, name, obj, main_module):
    """lookup name or id of obj if module is imported"""
    for modobj, modname in modmap.by_name[name]:
        if modobj is obj and sys.modules[modname] is not main_module:
            return modname, name
    __module__ = getattr(obj, '__module__', None)
    if isinstance(obj, IMPORTED_AS_TYPES) or (__module__ is not None
            and any(regex.fullmatch(__module__) for regex in IMPORTED_AS_MODULES)):
        for modobj, objname, modname in modmap.by_id[id(obj)]:
            if sys.modules[modname] is not main_module:
                return modname, objname
    return None, None

def _stash_modules(main_module):
    modmap = _module_map()
    newmod = ModuleType(main_module.__name__)

    imported = []
    imported_as = []
    imported_top_level = []  # keep separated for backward compatibility
    original = {}
    for name, obj in main_module.__dict__.items():
        if obj is main_module:
            original[name] = newmod  # self-reference
        elif obj is main_module.__dict__:
            original[name] = newmod.__dict__
        # Avoid incorrectly matching a singleton value in another package (ex.: __doc__).
        elif any(obj is singleton for singleton in (None, False, True)) \
                or isinstance(obj, ModuleType) and _is_builtin_module(obj):  # always saved by ref
            original[name] = obj
        else:
            source_module, objname = _lookup_module(modmap, name, obj, main_module)
            if source_module is not None:
                if objname == name:
                    imported.append((source_module, name))
                else:
                    imported_as.append((source_module, objname, name))
            else:
                try:
                    imported_top_level.append((modmap.top_level[id(obj)], name))
                except KeyError:
                    original[name] = obj

    if len(original) < len(main_module.__dict__):
        newmod.__dict__.update(original)
        newmod.__dill_imported = imported
        newmod.__dill_imported_as = imported_as
        newmod.__dill_imported_top_level = imported_top_level
        if getattr(newmod, '__loader__', None) is None and _is_imported_module(main_module):
            # Trick _is_imported_module() to force saving as an imported module.
            newmod.__loader__ = True  # will be discarded by save_module()
        return newmod
    else:
        return main_module

def _restore_modules(unpickler, main_module):
    try:
        for modname, name in main_module.__dict__.pop('__dill_imported'):
            main_module.__dict__[name] = unpickler.find_class(modname, name)
        for modname, objname, name in main_module.__dict__.pop('__dill_imported_as'):
            main_module.__dict__[name] = unpickler.find_class(modname, objname)
        for modname, name in main_module.__dict__.pop('__dill_imported_top_level'):
            main_module.__dict__[name] = __import__(modname)
    except KeyError:
        pass

#NOTE: 06/03/15 renamed main_module to main
def dump_module(
    filename: Union[str, os.PathLike] = None,
    module: Optional[Union[ModuleType, str]] = None,
    refimported: bool = False,
    **kwds
) -> None:
    """Pickle the current state of :py:mod:`__main__` or another module to a file.

    Save the contents of :py:mod:`__main__` (e.g. from an interactive
    interpreter session), an imported module, or a module-type object (e.g.
    built with :py:class:`~types.ModuleType`), to a file. The pickled
    module can then be restored with the function :py:func:`load_module`.

    Args:
        filename: a path-like object or a writable stream. If `None`
            (the default), write to a named file in a temporary directory.
        module: a module object or the name of an importable module. If `None`
            (the default), :py:mod:`__main__` is saved.
        refimported: if `True`, all objects identified as having been imported
            into the module's namespace are saved by reference. *Note:* this is
            similar but independent from ``dill.settings[`byref`]``, as
            ``refimported`` refers to virtually all imported objects, while
            ``byref`` only affects select objects.
        **kwds: extra keyword arguments passed to :py:class:`Pickler()`.

    Raises:
       :py:exc:`PicklingError`: if pickling fails.

    Examples:

        - Save current interpreter session state:

          >>> import dill
          >>> squared = lambda x: x*x
          >>> dill.dump_module() # save state of __main__ to /tmp/session.pkl

        - Save the state of an imported/importable module:

          >>> import dill
          >>> import pox
          >>> pox.plus_one = lambda x: x+1
          >>> dill.dump_module('pox_session.pkl', module=pox)

        - Save the state of a non-importable, module-type object:

          >>> import dill
          >>> from types import ModuleType
          >>> foo = ModuleType('foo')
          >>> foo.values = [1,2,3]
          >>> import math
          >>> foo.sin = math.sin
          >>> dill.dump_module('foo_session.pkl', module=foo, refimported=True)

        - Restore the state of the saved modules:

          >>> import dill
          >>> dill.load_module()
          >>> squared(2)
          4
          >>> pox = dill.load_module('pox_session.pkl')
          >>> pox.plus_one(1)
          2
          >>> foo = dill.load_module('foo_session.pkl')
          >>> [foo.sin(x) for x in foo.values]
          [0.8414709848078965, 0.9092974268256817, 0.1411200080598672]

        - Use `refimported` to save imported objects by reference:

          >>> import dill
          >>> from html.entities import html5
          >>> type(html5), len(html5)
          (dict, 2231)
          >>> import io
          >>> buf = io.BytesIO()
          >>> dill.dump_module(buf) # saves __main__, with html5 saved by value
          >>> len(buf.getvalue()) # pickle size in bytes
          71665
          >>> buf = io.BytesIO()
          >>> dill.dump_module(buf, refimported=True) # html5 saved by reference
          >>> len(buf.getvalue())
          438

    *Changed in version 0.3.6:* Function ``dump_session()`` was renamed to
    ``dump_module()``.  Parameters ``main`` and ``byref`` were renamed to
    ``module`` and ``refimported``, respectively.

    Note:
        Currently, ``dill.settings['byref']`` and ``dill.settings['recurse']``
        don't apply to this function.
    """
    for old_par, par in [('main', 'module'), ('byref', 'refimported')]:
        if old_par in kwds:
            message = "The argument %r has been renamed %r" % (old_par, par)
            if old_par == 'byref':
                message += " to distinguish it from dill.settings['byref']"
            warnings.warn(message + ".", PendingDeprecationWarning)
            if locals()[par]:  # the defaults are None and False
                raise TypeError("both %r and %r arguments were used" % (par, old_par))
    refimported = kwds.pop('byref', refimported)
    module = kwds.pop('main', module)

    from .settings import settings
    protocol = settings['protocol']
    main = module
    if main is None:
        main = _main_module
    elif isinstance(main, str):
        main = _import_module(main)
    if not isinstance(main, ModuleType):
        raise TypeError("%r is not a module" % main)
    if hasattr(filename, 'write'):
        file = filename
    else:
        if filename is None:
            filename = str(TEMPDIR/'session.pkl')
        file = open(filename, 'wb')
    try:
        pickler = Pickler(file, protocol, **kwds)
        pickler._original_main = main
        if refimported:
            main = _stash_modules(main)
        pickler._main = main     #FIXME: dill.settings are disabled
        pickler._byref = False   # disable pickling by name reference
        pickler._recurse = False # disable pickling recursion for globals
        pickler._session = True  # is best indicator of when pickling a session
        pickler._first_pass = True
        pickler._main_modified = main is not pickler._original_main
        pickler.dump(main)
    finally:
        if file is not filename:  # if newly opened file
            file.close()
    return

# Backward compatibility.
def dump_session(filename=None, main=None, byref=False, **kwds):
    warnings.warn("dump_session() has been renamed dump_module()", PendingDeprecationWarning)
    dump_module(filename, module=main, refimported=byref, **kwds)
dump_session.__doc__ = dump_module.__doc__

class _PeekableReader:
    """lightweight stream wrapper that implements peek()"""
    def __init__(self, stream):
        self.stream = stream
    def read(self, n):
        return self.stream.read(n)
    def readline(self):
        return self.stream.readline()
    def tell(self):
        return self.stream.tell()
    def close(self):
        return self.stream.close()
    def peek(self, n):
        stream = self.stream
        try:
            if hasattr(stream, 'flush'): stream.flush()
            position = stream.tell()
            stream.seek(position)  # assert seek() works before reading
            chunk = stream.read(n)
            stream.seek(position)
            return chunk
        except (AttributeError, OSError):
            raise NotImplementedError("stream is not peekable: %r", stream) from None

def _make_peekable(stream):
    """return stream as an object with a peek() method"""
    import io
    if hasattr(stream, 'peek'):
        return stream
    if not (hasattr(stream, 'tell') and hasattr(stream, 'seek')):
        try:
            return io.BufferedReader(stream)
        except Exception:
            pass
    return _PeekableReader(stream)

def _identify_module(file, main=None):
    """identify the name of the module stored in the given file-type object"""
    from pickletools import genops
    UNICODE = {'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE'}
    found_import = False
    try:
        for opcode, arg, pos in genops(file.peek(256)):
            if not found_import:
                if opcode.name in ('GLOBAL', 'SHORT_BINUNICODE') and \
                        arg.endswith('_import_module'):
                    found_import = True
            else:
                if opcode.name in UNICODE:
                    return arg
        else:
            raise UnpicklingError("reached STOP without finding main module")
    except (NotImplementedError, ValueError) as error:
        # ValueError occours when the end of the chunk is reached (without a STOP).
        if isinstance(error, NotImplementedError) and main is not None:
            # file is not peekable, but we have main.
            return None
        raise UnpicklingError("unable to identify main module") from error

def load_module(
    filename: Union[str, os.PathLike] = None,
    module: Optional[Union[ModuleType, str]] = None,
    **kwds
) -> Optional[ModuleType]:
    """Update the selected module (default is :py:mod:`__main__`) with
    the state saved at ``filename``.

    Restore a module to the state saved with :py:func:`dump_module`. The
    saved module can be :py:mod:`__main__` (e.g. an interpreter session),
    an imported module, or a module-type object (e.g. created with
    :py:class:`~types.ModuleType`).

    When restoring the state of a non-importable module-type object, the
    current instance of this module may be passed as the argument ``main``.
    Otherwise, a new instance is created with :py:class:`~types.ModuleType`
    and returned.

    Args:
        filename: a path-like object or a readable stream. If `None`
            (the default), read from a named file in a temporary directory.
        module: a module object or the name of an importable module;
            the module name and kind (i.e. imported or non-imported) must
            match the name and kind of the module stored at ``filename``.
        **kwds: extra keyword arguments passed to :py:class:`Unpickler()`.

    Raises:
        :py:exc:`UnpicklingError`: if unpickling fails.
        :py:exc:`ValueError`: if the argument ``main`` and module saved
            at ``filename`` are incompatible.

    Returns:
        A module object, if the saved module is not :py:mod:`__main__` or
        a module instance wasn't provided with the argument ``main``.

    Examples:

        - Save the state of some modules:

          >>> import dill
          >>> squared = lambda x: x*x
          >>> dill.dump_module() # save state of __main__ to /tmp/session.pkl
          >>>
          >>> import pox # an imported module
          >>> pox.plus_one = lambda x: x+1
          >>> dill.dump_module('pox_session.pkl', module=pox)
          >>>
          >>> from types import ModuleType
          >>> foo = ModuleType('foo') # a module-type object
          >>> foo.values = [1,2,3]
          >>> import math
          >>> foo.sin = math.sin
          >>> dill.dump_module('foo_session.pkl', module=foo, refimported=True)

        - Restore the state of the interpreter:

          >>> import dill
          >>> dill.load_module() # updates __main__ from /tmp/session.pkl
          >>> squared(2)
          4

        - Load the saved state of an importable module:

          >>> import dill
          >>> pox = dill.load_module('pox_session.pkl')
          >>> pox.plus_one(1)
          2
          >>> import sys
          >>> pox in sys.modules.values()
          True

        - Load the saved state of a non-importable module-type object:

          >>> import dill
          >>> foo = dill.load_module('foo_session.pkl')
          >>> [foo.sin(x) for x in foo.values]
          [0.8414709848078965, 0.9092974268256817, 0.1411200080598672]
          >>> import math
          >>> foo.sin is math.sin # foo.sin was saved by reference
          True
          >>> import sys
          >>> foo in sys.modules.values()
          False

        - Update the state of a non-importable module-type object:

          >>> import dill
          >>> from types import ModuleType
          >>> foo = ModuleType('foo')
          >>> foo.values = ['a','b']
          >>> foo.sin = lambda x: x*x
          >>> dill.load_module('foo_session.pkl', module=foo)
          >>> [foo.sin(x) for x in foo.values]
          [0.8414709848078965, 0.9092974268256817, 0.1411200080598672]

    *Changed in version 0.3.6:* Function ``load_session()`` was renamed to
    ``load_module()``. Parameter ``main`` was renamed to ``module``.

    See also:
        :py:func:`load_module_asdict` to load the contents of module saved
        with :py:func:`dump_module` into a dictionary.
    """
    if 'main' in kwds:
        warnings.warn(
            "The argument 'main' has been renamed 'module'.",
            PendingDeprecationWarning
        )
        if module is not None:
            raise TypeError("both 'module' and 'main' arguments were used")
        module = kwds.pop('main')
    main = module
    if hasattr(filename, 'read'):
        file = filename
    else:
        if filename is None:
            filename = str(TEMPDIR/'session.pkl')
        file = open(filename, 'rb')
    try:
        file = _make_peekable(file)
        #FIXME: dill.settings are disabled
        unpickler = Unpickler(file, **kwds)
        unpickler._session = True

        # Resolve unpickler._main
        pickle_main = _identify_module(file, main)
        if main is None and pickle_main is not None:
            main = pickle_main
        if isinstance(main, str):
            if main.startswith('__runtime__.'):
                # Create runtime module to load the session into.
                main = ModuleType(main.partition('.')[-1])
            else:
                main = _import_module(main)
        if main is not None:
            if not isinstance(main, ModuleType):
                raise TypeError("%r is not a module" % main)
            unpickler._main = main
        else:
            main = unpickler._main

        # Check against the pickle's main.
        is_main_imported = _is_imported_module(main)
        if pickle_main is not None:
            is_runtime_mod = pickle_main.startswith('__runtime__.')
            if is_runtime_mod:
                pickle_main = pickle_main.partition('.')[-1]
            error_msg = "can't update{} module{} %r with the saved state of{} module{} %r"
            if is_runtime_mod and is_main_imported:
                raise ValueError(
                    error_msg.format(" imported", "", "", "-type object")
                    % (main.__name__, pickle_main)
                )
            if not is_runtime_mod and not is_main_imported:
                raise ValueError(
                    error_msg.format("", "-type object", " imported", "")
                    % (pickle_main, main.__name__)
                )
            if main.__name__ != pickle_main:
                raise ValueError(error_msg.format("", "", "", "") % (main.__name__, pickle_main))

        # This is for find_class() to be able to locate it.
        if not is_main_imported:
            runtime_main = '__runtime__.%s' % main.__name__
            sys.modules[runtime_main] = main

        loaded = unpickler.load()
    finally:
        if not hasattr(filename, 'read'):  # if newly opened file
            file.close()
        try:
            del sys.modules[runtime_main]
        except (KeyError, NameError):
            pass
    assert loaded is main
    _restore_modules(unpickler, main)
    if main is _main_module or main is module:
        return None
    else:
        return main

# Backward compatibility.
def load_session(filename=None, main=None, **kwds):
    warnings.warn("load_session() has been renamed load_module().", PendingDeprecationWarning)
    load_module(filename, module=main, **kwds)
load_session.__doc__ = load_module.__doc__

def load_module_asdict(
    filename: Union[str, os.PathLike] = None,
    update: bool = False,
    **kwds
) -> dict:
    """
    Load the contents of a saved module into a dictionary.

    ``load_module_asdict()`` is the near-equivalent of::

        lambda filename: vars(dill.load_module(filename)).copy()

    however, does not alter the original module. Also, the path of
    the loaded module is stored in the ``__session__`` attribute.

    Args:
        filename: a path-like object or a readable stream. If `None`
            (the default), read from a named file in a temporary directory.
        update: if `True`, initialize the dictionary with the current state
            of the module prior to loading the state stored at filename.
        **kwds: extra keyword arguments passed to :py:class:`Unpickler()`

    Raises:
        :py:exc:`UnpicklingError`: if unpickling fails

    Returns:
        A copy of the restored module's dictionary.

    Note:
        If ``update`` is True, the corresponding module may first be imported
        into the current namespace before the saved state is loaded from
        filename to the dictionary. Note that any module that is imported into
        the current namespace as a side-effect of using ``update`` will not be
        modified by loading the saved module in filename to a dictionary.

    Example:
        >>> import dill
        >>> alist = [1, 2, 3]
        >>> anum = 42
        >>> dill.dump_module()
        >>> anum = 0
        >>> new_var = 'spam'
        >>> main = dill.load_module_asdict()
        >>> main['__name__'], main['__session__']
        ('__main__', '/tmp/session.pkl')
        >>> main is globals() # loaded objects don't reference globals
        False
        >>> main['alist'] == alist
        True
        >>> main['alist'] is alist # was saved by value
        False
        >>> main['anum'] == anum # changed after the session was saved
        False
        >>> new_var in main # would be True if the option 'update' was set
        False
    """
    if 'module' in kwds:
        raise TypeError("'module' is an invalid keyword argument for load_module_asdict()")
    if hasattr(filename, 'read'):
        file = filename
    else:
        if filename is None:
            filename = str(TEMPDIR/'session.pkl')
        file = open(filename, 'rb')
    try:
        file = _make_peekable(file)
        main_name = _identify_module(file)
        old_main = sys.modules.get(main_name)
        main = ModuleType(main_name)
        if update:
            if old_main is None:
                old_main = _import_module(main_name)
            main.__dict__.update(old_main.__dict__)
        else:
            main.__builtins__ = __builtin__
        sys.modules[main_name] = main
        load_module(file, **kwds)
    finally:
        if not hasattr(filename, 'read'):  # if newly opened file
            file.close()
        try:
            if old_main is None:
                del sys.modules[main_name]
            else:
                sys.modules[main_name] = old_main
        except NameError:  # failed before setting old_main
            pass
    main.__session__ = str(filename)
    return main.__dict__


# Internal exports for backward compatibility with dill v0.3.5.1
# Can't be placed in dill._dill because of circular import problems.
for name in (
    '_lookup_module', '_module_map', '_restore_modules', '_stash_modules',
    'dump_session', 'load_session' # backward compatibility functions
):
    setattr(_dill, name, globals()[name])
del name
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
all Python Standard Library object types (currently: CH 1-15 @ 2.7)
and some other common object types (i.e. numpy.ndarray)

to load more objects and types, use dill.load_types()
"""

# non-local import of dill.objects
from dill import objects
for _type in objects.keys():
    exec("%s = type(objects['%s'])" % (_type,_type))
    
del objects
try:
    del _type
except NameError:
    pass
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
Methods for serialized objects (or source code) stored in temporary files
and file-like objects.
"""
#XXX: better instead to have functions write to any given file-like object ?
#XXX: currently, all file-like objects are created by the function...

__all__ = ['dump_source', 'dump', 'dumpIO_source', 'dumpIO',\
           'load_source', 'load', 'loadIO_source', 'loadIO',\
           'capture']

import contextlib


@contextlib.contextmanager
def capture(stream='stdout'):
    """builds a context that temporarily replaces the given stream name

    >>> with capture('stdout') as out:
    ...   print ("foo!")
    ... 
    >>> print (out.getvalue())
    foo!

    """
    import sys
    from io import StringIO
    orig = getattr(sys, stream)
    setattr(sys, stream, StringIO())
    try:
        yield getattr(sys, stream)
    finally:
        setattr(sys, stream, orig)


def b(x): # deal with b'foo' versus 'foo'
    import codecs
    return codecs.latin_1_encode(x)[0]

def load_source(file, **kwds):
    """load an object that was stored with dill.temp.dump_source

    file: filehandle
    alias: string name of stored object
    mode: mode to open the file, one of: {'r', 'rb'}

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, alias='_f')
    >>> _f = dill.temp.load_source(pyfile)
    >>> _f(4)
    16
    """
    alias = kwds.pop('alias', None)
    mode = kwds.pop('mode', 'r')
    fname = getattr(file, 'name', file) # fname=file.name or fname=file (if str)
    source = open(fname, mode=mode, **kwds).read()
    if not alias:
        tag = source.strip().splitlines()[-1].split()
        if tag[0] != '#NAME:':
            stub = source.splitlines()[0]
            raise IOError("unknown name for code: %s" % stub)
        alias = tag[-1]
    local = {}
    exec(source, local)
    _ = eval("%s" % alias, local)
    return _

def dump_source(object, **kwds):
    """write object source to a NamedTemporaryFile (instead of dill.dump)
Loads with "import" or "dill.temp.load_source".  Returns the filehandle.

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, alias='_f')
    >>> _f = dill.temp.load_source(pyfile)
    >>> _f(4)
    16

    >>> f = lambda x: x**2
    >>> pyfile = dill.temp.dump_source(f, dir='.')
    >>> modulename = os.path.basename(pyfile.name).split('.py')[0]
    >>> exec('from %s import f as _f' % modulename)
    >>> _f(4)
    16

Optional kwds:
    If 'alias' is specified, the object will be renamed to the given string.

    If 'prefix' is specified, the file name will begin with that prefix,
    otherwise a default prefix is used.
    
    If 'dir' is specified, the file will be created in that directory,
    otherwise a default directory is used.
    
    If 'text' is specified and true, the file is opened in text
    mode.  Else (the default) the file is opened in binary mode.  On
    some operating systems, this makes no difference.

NOTE: Keep the return value for as long as you want your file to exist !
    """ #XXX: write a "load_source"?
    from .source import importable, getname
    import tempfile
    kwds.setdefault('delete', True)
    kwds.pop('suffix', '') # this is *always* '.py'
    alias = kwds.pop('alias', '') #XXX: include an alias so a name is known
    name = str(alias) or getname(object)
    name = "\n#NAME: %s\n" % name
    #XXX: assumes kwds['dir'] is writable and on $PYTHONPATH
    file = tempfile.NamedTemporaryFile(suffix='.py', **kwds)
    file.write(b(''.join([importable(object, alias=alias),name])))
    file.flush()
    return file

def load(file, **kwds):
    """load an object that was stored with dill.temp.dump

    file: filehandle
    mode: mode to open the file, one of: {'r', 'rb'}

    >>> dumpfile = dill.temp.dump([1, 2, 3, 4, 5])
    >>> dill.temp.load(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    mode = kwds.pop('mode', 'rb')
    name = getattr(file, 'name', file) # name=file.name or name=file (if str)
    return pickle.load(open(name, mode=mode, **kwds))

def dump(object, **kwds):
    """dill.dump of object to a NamedTemporaryFile.
Loads with "dill.temp.load".  Returns the filehandle.

    >>> dumpfile = dill.temp.dump([1, 2, 3, 4, 5])
    >>> dill.temp.load(dumpfile)
    [1, 2, 3, 4, 5]

Optional kwds:
    If 'suffix' is specified, the file name will end with that suffix,
    otherwise there will be no suffix.
    
    If 'prefix' is specified, the file name will begin with that prefix,
    otherwise a default prefix is used.
    
    If 'dir' is specified, the file will be created in that directory,
    otherwise a default directory is used.
    
    If 'text' is specified and true, the file is opened in text
    mode.  Else (the default) the file is opened in binary mode.  On
    some operating systems, this makes no difference.

NOTE: Keep the return value for as long as you want your file to exist !
    """
    import dill as pickle
    import tempfile
    kwds.setdefault('delete', True)
    file = tempfile.NamedTemporaryFile(**kwds)
    pickle.dump(object, file)
    file.flush()
    return file

def loadIO(buffer, **kwds):
    """load an object that was stored with dill.temp.dumpIO

    buffer: buffer object

    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])
    >>> dill.temp.loadIO(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    from io import BytesIO as StringIO
    value = getattr(buffer, 'getvalue', buffer) # value or buffer.getvalue
    if value != buffer: value = value() # buffer.getvalue()
    return pickle.load(StringIO(value))

def dumpIO(object, **kwds):
    """dill.dump of object to a buffer.
Loads with "dill.temp.loadIO".  Returns the buffer object.

    >>> dumpfile = dill.temp.dumpIO([1, 2, 3, 4, 5])
    >>> dill.temp.loadIO(dumpfile)
    [1, 2, 3, 4, 5]
    """
    import dill as pickle
    from io import BytesIO as StringIO
    file = StringIO()
    pickle.dump(object, file)
    file.flush()
    return file

def loadIO_source(buffer, **kwds):
    """load an object that was stored with dill.temp.dumpIO_source

    buffer: buffer object
    alias: string name of stored object

    >>> f = lambda x:x**2
    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')
    >>> _f = dill.temp.loadIO_source(pyfile)
    >>> _f(4)
    16
    """
    alias = kwds.pop('alias', None)
    source = getattr(buffer, 'getvalue', buffer) # source or buffer.getvalue
    if source != buffer: source = source() # buffer.getvalue()
    source = source.decode() # buffer to string
    if not alias:
        tag = source.strip().splitlines()[-1].split()
        if tag[0] != '#NAME:':
            stub = source.splitlines()[0]
            raise IOError("unknown name for code: %s" % stub)
        alias = tag[-1]
    local = {}
    exec(source, local)
    _ = eval("%s" % alias, local)
    return _

def dumpIO_source(object, **kwds):
    """write object source to a buffer (instead of dill.dump)
Loads by with dill.temp.loadIO_source.  Returns the buffer object.

    >>> f = lambda x:x**2
    >>> pyfile = dill.temp.dumpIO_source(f, alias='_f')
    >>> _f = dill.temp.loadIO_source(pyfile)
    >>> _f(4)
    16

Optional kwds:
    If 'alias' is specified, the object will be renamed to the given string.
    """
    from .source import importable, getname
    from io import BytesIO as StringIO
    alias = kwds.pop('alias', '') #XXX: include an alias so a name is known
    name = str(alias) or getname(object)
    name = "\n#NAME: %s\n" % name
    #XXX: assumes kwds['dir'] is writable and on $PYTHONPATH
    file = StringIO()
    file.write(b(''.join([importable(object, alias=alias),name])))
    file.flush()
    return file


del contextlib


# EOF
<|endoftext|>#!/usr/bin/env python
#
# Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
# Copyright (c) 2008-2016 California Institute of Technology.
# Copyright (c) 2016-2026 The Uncertainty Quantification Foundation.
# License: 3-clause BSD.  The full license text is available at:
#  - https://github.com/uqfoundation/dill/blob/master/LICENSE
"""
Methods for detecting objects leading to pickling failures.
"""

import dis
from inspect import ismethod, isfunction, istraceback, isframe, iscode

from .pointers import parent, reference, at, parents, children
from .logger import trace

__all__ = ['baditems','badobjects','badtypes','code','errors','freevars',
           'getmodule','globalvars','nestedcode','nestedglobals','outermost',
           'referredglobals','referrednested','trace','varnames']

def getmodule(object, _filename=None, force=False):
    """get the module of the object"""
    from inspect import getmodule as getmod
    module = getmod(object, _filename)
    if module or not force: return module
    import builtins
    from .source import getname
    name = getname(object, force=True)
    return builtins if name in vars(builtins).keys() else None

def outermost(func): # is analogous to getsource(func,enclosing=True)
    """get outermost enclosing object (i.e. the outer function in a closure)

    NOTE: this is the object-equivalent of getsource(func, enclosing=True)
    """
    if ismethod(func):
        _globals = func.__func__.__globals__ or {}
    elif isfunction(func):
        _globals = func.__globals__ or {}
    else:
        return #XXX: or raise? no matches
    _globals = _globals.items()
    # get the enclosing source
    from .source import getsourcelines
    try: lines,lnum = getsourcelines(func, enclosing=True)
    except Exception: #TypeError, IOError
        lines,lnum = [],None
    code = ''.join(lines)
    # get all possible names,objects that are named in the enclosing source
    _locals = ((name,obj) for (name,obj) in _globals if name in code)
    # now only save the objects that generate the enclosing block
    for name,obj in _locals: #XXX: don't really need 'name'
        try:
            if getsourcelines(obj) == (lines,lnum): return obj
        except Exception: #TypeError, IOError
            pass
    return #XXX: or raise? no matches

def nestedcode(func, recurse=True): #XXX: or return dict of {co_name: co} ?
    """get the code objects for any nested functions (e.g. in a closure)"""
    func = code(func)
    if not iscode(func): return [] #XXX: or raise? no matches
    nested = set()
    for co in func.co_consts:
        if co is None: continue
        co = code(co)
        if co:
            nested.add(co)
            if recurse: nested |= set(nestedcode(co, recurse=True))
    return list(nested)

def code(func):
    """get the code object for the given function or method

    NOTE: use dill.source.getsource(CODEOBJ) to get the source code
    """
    if ismethod(func): func = func.__func__
    if isfunction(func): func = func.__code__
    if istraceback(func): func = func.tb_frame
    if isframe(func): func = func.f_code
    if iscode(func): return func
    return

#XXX: ugly: parse dis.dis for name after "<code object" in line and in globals?
def referrednested(func, recurse=True): #XXX: return dict of {__name__: obj} ?
    """get functions defined inside of func (e.g. inner functions in a closure)

    NOTE: results may differ if the function has been executed or not.
    If len(nestedcode(func)) > len(referrednested(func)), try calling func().
    If possible, python builds code objects, but delays building functions
    until func() is called.
    """
    import gc
    funcs = set()
    # get the code objects, and try to track down by referrence
    for co in nestedcode(func, recurse):
        # look for function objects that refer to the code object
        for obj in gc.get_referrers(co):
            # get methods
            _ = getattr(obj, '__func__', None) # ismethod
            if getattr(_, '__code__', None) is co: funcs.add(obj)
            # get functions
            elif getattr(obj, '__code__', None) is co: funcs.add(obj)
            # get frame objects
            elif getattr(obj, 'f_code', None) is co: funcs.add(obj)
            # get code objects
            elif hasattr(obj, 'co_code') and obj is co: funcs.add(obj)
#     frameobjs => func.__code__.co_varnames not in func.__code__.co_cellvars
#     funcobjs => func.__code__.co_cellvars not in func.__code__.co_varnames
#     frameobjs are not found, however funcobjs are...
#     (see: test_mixins.quad ... and test_mixins.wtf)
#     after execution, code objects get compiled, and then may be found by gc
    return list(funcs)


def freevars(func):
    """get objects defined in enclosing code that are referred to by func

    returns a dict of {name:object}"""
    if ismethod(func): func = func.__func__
    if isfunction(func):
        closures = func.__closure__ or ()
        func = func.__code__.co_freevars # get freevars
    else:
        return {}

    def get_cell_contents():
        for name, c in zip(func, closures):
            try:
                cell_contents = c.cell_contents
            except ValueError: # cell is empty
                continue
            yield name, c.cell_contents

    return dict(get_cell_contents())

# thanks to Davies Liu for recursion of globals
def nestedglobals(func, recurse=True):
    """get the names of any globals found within func"""
    func = code(func)
    if func is None: return list()
    import sys
    from .temp import capture
    CAN_NULL = sys.hexversion >= 0x30b00a7 # NULL may be prepended >= 3.11a7
    names = set()
    with capture('stdout') as out:
        try:
            dis.dis(func) #XXX: dis.dis(None) disassembles last traceback
        except IndexError:
            pass #FIXME: HACK for IS_PYPY (3.11)
    for line in out.getvalue().splitlines():
        if '_GLOBAL' in line:
            name = line.split('(')[-1].split(')')[0]
            if CAN_NULL:
                names.add(name.replace('NULL + ', '').replace(' + NULL', ''))
            else:
                names.add(name)
    for co in getattr(func, 'co_consts', tuple()):
        if co and recurse and iscode(co):
            names.update(nestedglobals(co, recurse=True))
    return list(names)

def referredglobals(func, recurse=True, builtin=False):
    """get the names of objects in the global scope referred to by func"""
    return globalvars(func, recurse, builtin).keys()

def globalvars(func, recurse=True, builtin=False):
    """get objects defined in global scope that are referred to by func

    return a dict of {name:object}"""
    if ismethod(func): func = func.__func__
    if isfunction(func):
        globs = vars(getmodule(sum)).copy() if builtin else {}
        # get references from within closure
        orig_func, func = func, set()
        for obj in orig_func.__closure__ or {}:
            try:
                cell_contents = obj.cell_contents
            except ValueError: # cell is empty
                pass
            else:
                _vars = globalvars(cell_contents, recurse, builtin) or {}
                func.update(_vars) #XXX: (above) be wary of infinte recursion?
                globs.update(_vars)
        # get globals
        globs.update(orig_func.__globals__ or {})
        # get names of references
        if not recurse:
            func.update(orig_func.__code__.co_names)
        else:
            func.update(nestedglobals(orig_func.__code__))
            # find globals for all entries of func
            for key in func.copy(): #XXX: unnecessary...?
                nested_func = globs.get(key)
                if nested_func is orig_func:
                   #func.remove(key) if key in func else None
                    continue  #XXX: globalvars(func, False)?
                func.update(globalvars(nested_func, True, builtin))
    elif iscode(func):
        globs = vars(getmodule(sum)).copy() if builtin else {}
       #globs.update(globals())
        if not recurse:
            func = func.co_names # get names
        else:
            orig_func = func.co_name # to stop infinite recursion
            func = set(nestedglobals(func))
            # find globals for all entries of func
            for key in func.copy(): #XXX: unnecessary...?
                if key is orig_func:
                   #func.remove(key) if key in func else None
                    continue  #XXX: globalvars(func, False)?
                nested_func = globs.get(key)
                func.update(globalvars(nested_func, True, builtin))
    else:
        return {}
    #NOTE: if name not in __globals__, then we skip it...
    return dict((name,globs[name]) for name in func if name in globs)


def varnames(func):
    """get names of variables defined by func

    returns a tuple (local vars, local vars referrenced by nested functions)"""
    func = code(func)
    if not iscode(func):
        return () #XXX: better ((),())? or None?
    return func.co_varnames, func.co_cellvars


def baditems(obj, exact=False, safe=False): #XXX: obj=globals() ?
    """get items in object that fail to pickle"""
    if not hasattr(obj,'__iter__'): # is not iterable
        return [j for j in (badobjects(obj,0,exact,safe),) if j is not None]
    obj = obj.values() if getattr(obj,'values',None) else obj
    _obj = [] # can't use a set, as items may be unhashable
    [_obj.append(badobjects(i,0,exact,safe)) for i in obj if i not in _obj]
    return [j for j in _obj if j is not None]


def badobjects(obj, depth=0, exact=False, safe=False):
    """get objects that fail to pickle"""
    from dill import pickles
    if not depth:
        if pickles(obj,exact,safe): return None
        return obj
    return dict(((attr, badobjects(getattr(obj,attr),depth-1,exact,safe)) \
           for attr in dir(obj) if not pickles(getattr(obj,attr),exact,safe)))

def badtypes(obj, depth=0, exact=False, safe=False):
    """get types for objects that fail to pickle"""
    from dill import pickles
    if not depth:
        if pickles(obj,exact,safe): return None
        return type(obj)
    return dict(((attr, badtypes(getattr(obj,attr),depth-1,exact,safe)) \
           for attr in dir(obj) if not pickles(getattr(obj,attr),exact,safe)))

def errors(obj, depth=0, exact=False, safe=False):
    """get errors for objects that fail to pickle"""
    from dill import pickles, copy
    if not depth:
        try:
            pik = copy(obj)
            if exact:
                assert pik == obj, \
                    "Unpickling produces %s instead of %s" % (pik,obj)
            assert type(pik) == type(obj), \
                "Unpickling produces %s instead of %s" % (type(pik),type(obj))
            return None
        except Exception:
            import sys
            return sys.exc_info()[1]
    _dict = {}
    for attr in dir(obj):
        try:
            _attr = getattr(obj,attr)
        except Exception:
            import sys
            _dict[attr] = sys.exc_info()[1]
            continue
        if not pickles(_attr,exact,safe):
            _dict[attr] = errors(_attr,depth-1,exact,safe)
    return _dict


# EOF
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import dataclasses
import math
import textwrap
from typing import Any

import torch
from torch import inf


@dataclasses.dataclass
class __PrinterOptions:
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: bool | None = None


PRINT_OPTS = __PrinterOptions()


# We could use **kwargs, but this will give better docs
def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    r"""Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by
            `torch._tensor_str._Formatter`. This value is automatically chosen
            by the framework.

    Example::

        >>> # Limit the precision of elements
        >>> torch.set_printoptions(precision=2)
        >>> torch.tensor([1.12345])
        tensor([1.12])
        >>> # Limit the number of elements shown
        >>> torch.set_printoptions(threshold=5)
        >>> torch.arange(10)
        tensor([0, 1, 2, ..., 7, 8, 9])
        >>> # Restore defaults
        >>> torch.set_printoptions(profile='default')
        >>> torch.tensor([1.12345])
        tensor([1.1235])
        >>> torch.arange(10)
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode


def get_printoptions() -> dict[str, Any]:
    r"""Gets the current options for printing, as a dictionary that
    can be passed as ``**kwargs`` to set_printoptions().
    """
    return dataclasses.asdict(PRINT_OPTS)


@contextlib.contextmanager
def printoptions(**kwargs):
    r"""Context manager that temporarily changes the print options.  Accepted
    arguments are same as :func:`set_printoptions`."""
    old_kwargs = get_printoptions()
    set_printoptions(**kwargs)
    try:
        yield
    finally:
        set_printoptions(**old_kwargs)


def tensor_totype(t):
    dtype = (
        torch.float
        if (
            t.is_mps
            or (t.is_xpu and not torch.xpu.get_device_properties(t.device).has_fp64)
            or t.is_maia
        )
        else torch.double
    )
    return t.to(dtype=dtype)


class _Formatter:
    def __init__(self, tensor):
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1

        with torch.no_grad():
            tensor_view = tensor.reshape(-1)

        if not self.floating_dtype:
            for value in tensor_view:
                value_str = f"{value}"
                self.max_width = max(self.max_width, len(value_str))

        else:
            if tensor.dtype == torch.float4_e2m1fn_x2:  # type: ignore[attr-defined]
                # torch.float4_e2m1fn_x2 is special and does not support the casts necessary
                # to print it, we choose to display the uint8 representation here for
                # convenience of being able to print a tensor.
                # TODO(#146647): extend this to other dtypes without casts defined, such
                # as the bits, uint1..7 and int1..7 dtypes.
                tensor_view = tensor_view.view(torch.uint8)

            nonzero_finite_vals = torch.masked_select(
                tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0)
            )

            if nonzero_finite_vals.numel() == 0:
                # no valid number, do nothing
                return

            if tensor.dtype == torch.float8_e8m0fnu:  # type: ignore[attr-defined]
                # float8_e8m0fnu is special and does not define arithmetic ops,
                # and printing code further in this file assumes the existence
                # of various arithmetic ops to figure out what to print. We hack
                # and convert to float here to make printing work correctly.
                # TODO(#113663): also add the other float8 dtypes here after arithmetic
                # support for them is removed
                nonzero_finite_vals = nonzero_finite_vals.float()

            # Convert to double (or float) for easy calculation. HalfTensor overflows with 1e8, and there's no div() on CPU.
            nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs())
            nonzero_finite_min = tensor_totype(nonzero_finite_abs.min())
            nonzero_finite_max = tensor_totype(nonzero_finite_abs.max())

            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    self.int_mode = False
                    break

            self.sci_mode = (
                nonzero_finite_max / nonzero_finite_min > 1000.0
                or nonzero_finite_max > 1.0e8
                or nonzero_finite_min < 1.0e-4
                if PRINT_OPTS.sci_mode is None
                else PRINT_OPTS.sci_mode
            )

            if self.int_mode:
                # in int_mode for floats, all numbers are integers, and we append a decimal to nonfinites
                # to indicate that the tensor is of floating type. add 1 to the len to account for this.
                if self.sci_mode:
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = f"{value:.0f}"
                        self.max_width = max(self.max_width, len(value_str) + 1)
            else:
                # Check if scientific representation should be used.
                if self.sci_mode:
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
                        self.max_width = max(self.max_width, len(value_str))

    def width(self):
        return self.max_width

    def format(self, value):
        if self.floating_dtype:
            if self.sci_mode:
                ret = f"{{:{self.max_width}.{PRINT_OPTS.precision}e}}".format(value)
            elif self.int_mode:
                ret = f"{value:.0f}"
                if not (math.isinf(value) or math.isnan(value)):
                    ret += "."
            else:
                ret = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
        else:
            ret = f"{value}"
        return (self.max_width - len(ret)) * " " + ret


def _scalar_str(self, formatter1, formatter2=None):
    if formatter2 is not None:
        real_str = _scalar_str(self.real, formatter1)
        imag_str = (_scalar_str(self.imag, formatter2) + "j").lstrip()
        # handles negative numbers, +0.0, -0.0
        if imag_str[0] == "+" or imag_str[0] == "-":
            return real_str + imag_str
        else:
            return real_str + "+" + imag_str
    else:
        return formatter1.format(self.item())


def _vector_str(self, indent, summarize, formatter1, formatter2=None):
    # length includes spaces and comma between elements
    element_length = formatter1.width() + 2
    if formatter2 is not None:
        # width for imag_formatter + an extra j for complex
        element_length += formatter2.width() + 1

    elements_per_line = max(
        1, math.floor((PRINT_OPTS.linewidth - indent) / (element_length))
    )

    def _val_formatter(val, formatter1=formatter1, formatter2=formatter2):
        if formatter2 is not None:
            real_str = formatter1.format(val.real)
            imag_str = (formatter2.format(val.imag) + "j").lstrip()
            # handles negative numbers, +0.0, -0.0
            if imag_str[0] == "+" or imag_str[0] == "-":
                return real_str + imag_str
            else:
                return real_str + "+" + imag_str
        else:
            return formatter1.format(val)

    if self.dtype == torch.float4_e2m1fn_x2:  # type: ignore[attr-defined]
        # torch.float4_e2m1fn_x2 is special and does not support the casts necessary
        # to print it, we choose to display the uint8 representation here for
        # convenience of being able to print a tensor.
        # TODO(#146647): extend this to other dtypes without casts defined, such
        # as the bits, uint1..7 and int1..7 dtypes.
        self = self.view(torch.uint8)

    if summarize and not PRINT_OPTS.edgeitems:
        # Deal with edge case that negative zero is zero
        data = ["..."]
    elif summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        data = (
            [_val_formatter(val) for val in self[: PRINT_OPTS.edgeitems].tolist()]
            + [" ..."]
            + [_val_formatter(val) for val in self[-PRINT_OPTS.edgeitems :].tolist()]
        )
    else:
        data = [_val_formatter(val) for val in self.tolist()]

    data_lines = [
        data[i : i + elements_per_line] for i in range(0, len(data), elements_per_line)
    ]
    lines = [", ".join(line) for line in data_lines]
    return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"


# formatter2 is only used for printing complex tensors.
# For complex tensors, formatter1 and formatter2 are the formatters for tensor.real
# and tensor.imag respectively
def _tensor_str_with_formatter(self, indent, summarize, formatter1, formatter2=None):
    dim = self.dim()

    if dim == 0:
        return _scalar_str(self, formatter1, formatter2)

    if dim == 1:
        return _vector_str(self, indent, summarize, formatter1, formatter2)

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = (
            [
                _tensor_str_with_formatter(
                    self[i], indent + 1, summarize, formatter1, formatter2
                )
                for i in range(PRINT_OPTS.edgeitems)
            ]
            + ["..."]
            + [
                _tensor_str_with_formatter(
                    self[i], indent + 1, summarize, formatter1, formatter2
                )
                for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))
            ]
        )
    else:
        slices = [
            _tensor_str_with_formatter(
                self[i], indent + 1, summarize, formatter1, formatter2
            )
            for i in range(self.size(0))
        ]

    tensor_str = ("," + "\n" * (dim - 1) + " " * (indent + 1)).join(slices)
    return "[" + tensor_str + "]"


def _tensor_str(self, indent):
    if self.numel() == 0:
        return "[]"

    summarize = self.numel() > PRINT_OPTS.threshold

    if self._is_zerotensor():
        self = self.clone()

    # handle the negative bit
    if self.is_neg():
        self = self.resolve_neg()

    # TODO: Remove me when `masked_select` is implemented for FP8
    if self.dtype in [
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ]:
        self = self.half()

    if self.dtype.is_complex:
        # handle the conjugate bit
        self = self.resolve_conj()
        real_formatter = _Formatter(
            get_summarized_data(self.real) if summarize else self.real
        )
        imag_formatter = _Formatter(
            get_summarized_data(self.imag) if summarize else self.imag
        )
        return _tensor_str_with_formatter(
            self, indent, summarize, real_formatter, imag_formatter
        )
    else:
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        return _tensor_str_with_formatter(self, indent, summarize, formatter)


def _add_suffixes(tensor_str, suffixes, indent, force_newline):
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind("\n") + 1
    for suffix in suffixes:
        suffix_len = len(suffix)
        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(",\n" + " " * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            tensor_strs.append(", " + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append(")")
    return "".join(tensor_strs)


def get_summarized_data(self):
    dim = self.dim()
    if dim == 0:
        return self
    if dim == 1:
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return torch.cat(
                (self[: PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems :])
            )
        else:
            return self
    if not PRINT_OPTS.edgeitems:
        return self.new_empty([0] * self.dim())
    elif self.size(0) > 2 * PRINT_OPTS.edgeitems:
        start = [self[i] for i in range(PRINT_OPTS.edgeitems)]
        end = [self[i] for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))]
        return torch.stack([get_summarized_data(x) for x in (start + end)])
    else:
        return torch.stack([get_summarized_data(x) for x in self])


def _str_intern(inp, *, tensor_contents=None):
    if torch._C._functorch.is_functorch_wrapped_tensor(inp):
        return _functorch_wrapper_str_intern(inp, tensor_contents=tensor_contents)
    is_plain_tensor = type(inp) is torch.Tensor or type(inp) is torch.nn.Parameter
    if inp.is_nested:
        prefix = "nested_tensor("
    elif is_plain_tensor:
        prefix = "tensor("
    else:
        prefix = f"{type(inp).__name__}("
    indent = len(prefix)
    suffixes = []
    custom_contents_provided = tensor_contents is not None
    if custom_contents_provided:
        tensor_str = tensor_contents

    # This is used to extract the primal value and thus disable the forward AD
    # within this function.
    # TODO(albanD) This needs to be updated when more than one level is supported
    self, tangent = torch.autograd.forward_ad.unpack_dual(inp)

    # Note [Print tensor device]:
    # A general logic here is we only print device when it doesn't match
    # the device specified in default tensor type.
    # Currently torch.set_default_tensor_type() only supports CPU/CUDA, thus
    # torch._C._get_default_device() only returns either cpu or cuda.
    # In other cases, we don't have a way to set them as default yet,
    # and we should always print out device for them.
    if (
        self.device.type != torch._C._get_default_device()
        or (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        )
        or (self.device.type == "mps")
    ):
        suffixes.append("device='" + str(self.device) + "'")

    # Tensor printing performs tensor operations like slice, indexing, etc to make it in a
    # representable format. These operations on ipu/xla/lazy/mtia tensor results in compilations. Hence,
    # to avoid compilations, copying the tensor to cpu before printing.
    if self.device.type in ["xla", "lazy", "ipu", "mtia"]:
        self = self.to("cpu")

    # TODO: add an API to map real -> complex dtypes
    _default_complex_dtype = (
        torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
    )
    has_default_dtype = self.dtype in (
        torch.get_default_dtype(),
        _default_complex_dtype,
        torch.int64,
        torch.bool,
    )
    if self.is_sparse:
        suffixes.append("size=" + str(tuple(self.shape)))
        from torch._subclasses.fake_tensor import is_fake_tensor

        is_meta = self.is_meta or is_fake_tensor(self)
        if not is_meta:
            suffixes.append("nnz=" + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        if not custom_contents_provided:
            indices_prefix = "indices=tensor("
            indices = self._indices().detach()
            if is_meta:
                indices_str = "..."
            else:
                indices_str = _tensor_str(indices, indent + len(indices_prefix))
            if is_meta or indices.numel() == 0:
                indices_str += ", size=" + str(tuple(indices.shape))
            values_prefix = "values=tensor("
            values = self._values().detach()
            if is_meta:
                values_str = "..."
            else:
                values_str = _tensor_str(values, indent + len(values_prefix))
            if is_meta or values.numel() == 0:
                values_str += ", size=" + str(tuple(values.shape))
            tensor_str = (
                indices_prefix
                + indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    elif self.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        from torch._subclasses.fake_tensor import is_fake_tensor

        suffixes.append("size=" + str(tuple(self.shape)))
        is_meta = self.is_meta or is_fake_tensor(self)
        if not is_meta:
            suffixes.append("nnz=" + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        if not custom_contents_provided:
            compressed_indices_method, plain_indices_method = {
                torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
                torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
            }[self.layout]
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                cdimname, pdimname = "row", "column"
            else:
                cdimname, pdimname = "column", "row"
            compressed_indices_prefix = f"c{cdimname[:3]}_indices=tensor("
            compressed_indices = compressed_indices_method(self).detach()
            if is_meta:
                compressed_indices_str = "..."
            else:
                compressed_indices_str = _tensor_str(
                    compressed_indices, indent + len(compressed_indices_prefix)
                )
            if compressed_indices.numel() == 0 or is_meta:
                compressed_indices_str += ", size=" + str(
                    tuple(compressed_indices.shape)
                )
            plain_indices_prefix = f"{pdimname[:3]}_indices=tensor("
            plain_indices = plain_indices_method(self).detach()
            if is_meta:
                plain_indices_str = "..."
            else:
                plain_indices_str = _tensor_str(
                    plain_indices, indent + len(plain_indices_prefix)
                )
            if plain_indices.numel() == 0 or is_meta:
                plain_indices_str += ", size=" + str(tuple(plain_indices.shape))
            values_prefix = "values=tensor("
            values = self.values().detach()
            if is_meta:
                values_str = "..."
            else:
                values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0 or is_meta:
                values_str += ", size=" + str(tuple(values.shape))
            tensor_str = (
                compressed_indices_prefix
                + compressed_indices_str
                + "),\n"
                + " " * indent
                + plain_indices_prefix
                + plain_indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    elif self.is_quantized:
        suffixes.append("size=" + str(tuple(self.shape)))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        suffixes.append("quantization_scheme=" + str(self.qscheme()))
        if (
            self.qscheme() == torch.per_tensor_affine
            or self.qscheme() == torch.per_tensor_symmetric
        ):
            suffixes.append("scale=" + str(self.q_scale()))
            suffixes.append("zero_point=" + str(self.q_zero_point()))
        elif (
            self.qscheme() == torch.per_channel_affine
            or self.qscheme() == torch.per_channel_symmetric
            or self.qscheme() == torch.per_channel_affine_float_qparams
        ):
            suffixes.append("scale=" + str(self.q_per_channel_scales()))
            suffixes.append("zero_point=" + str(self.q_per_channel_zero_points()))
            suffixes.append("axis=" + str(self.q_per_channel_axis()))
        if not custom_contents_provided:
            tensor_str = _tensor_str(self.dequantize(), indent)
    elif self.is_nested:
        if not custom_contents_provided:

            def indented_str(s, indent):
                return "\n".join(f"  {line}" for line in s.split("\n"))

            strs = ",\n".join(
                indented_str(str(t), indent + 1)
                for t in torch.ops.aten.unbind.int(self, 0)
            )
            tensor_str = f"[\n{strs}\n]"
    elif torch._is_functional_tensor(self):
        prefix = "_to_functional_tensor("
        tensor_str = repr(torch._from_functional_tensor(self))
    else:
        # Circular import problem, so we import it here
        from torch._subclasses.fake_tensor import is_fake_tensor

        if self.is_meta or is_fake_tensor(self):
            suffixes.append("size=" + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append("dtype=" + str(self.dtype))
            # TODO: This implies that ellipses is valid syntax for allocating
            # a meta tensor or FakeTensor, which it could be, but it isn't right now
            if not custom_contents_provided:
                tensor_str = "..."
        else:
            if self.numel() == 0 and not self.is_sparse:
                # Explicitly print the shape if it is not (0,), to match NumPy behavior
                if self.dim() != 1:
                    suffixes.append("size=" + str(tuple(self.shape)))

                # In an empty tensor, there are no elements to infer if the dtype
                # should be int64, so it must be shown explicitly.
                if self.dtype != torch.get_default_dtype():
                    suffixes.append("dtype=" + str(self.dtype))
                if not custom_contents_provided:
                    tensor_str = "[]"
            else:
                if not PRINT_OPTS.edgeitems:
                    suffixes.append("size=" + str(tuple(self.shape)))

                if not has_default_dtype:
                    suffixes.append("dtype=" + str(self.dtype))

                if not custom_contents_provided:
                    if self.layout != torch.strided:
                        tensor_str = _tensor_str(self.to_dense(), indent)
                    else:
                        tensor_str = _tensor_str(self, indent)

    if self.layout != torch.strided:
        suffixes.append("layout=" + str(self.layout))

    # Use inp here to get the original grad_fn and not the one generated by the forward grad
    # unpacking.
    grad_fn_name = None
    try:
        grad_fn = inp.grad_fn
    except RuntimeError:
        # Accessing the grad_fn calls rebasing logic which would cause an error
        # if that tensor is a view created in no-grad mode modified in-place in
        # no-grad mode. See: https://github.com/pytorch/pytorch/issues/99968
        grad_fn_name = "Invalid"

    if grad_fn_name is None and grad_fn is not None:  # type: ignore[possibly-undefined]
        # pyrefly: ignore [unbound-name]
        grad_fn_name = type(grad_fn).__name__
        if grad_fn_name == "CppFunction":
            # pyrefly: ignore [unbound-name]
            grad_fn_name = grad_fn.name().rsplit("::", 1)[-1]

    if grad_fn_name is not None:
        suffixes.append(f"grad_fn=<{grad_fn_name}>")
    elif inp.requires_grad:
        suffixes.append("requires_grad=True")

    if tangent is not None:
        suffixes.append(f"tangent={tangent}")

    string_repr = _add_suffixes(
        prefix + tensor_str,  # type: ignore[possibly-undefined]
        suffixes,
        indent,
        force_newline=self.is_sparse,
    )

    # Check if this instance is flagged as a parameter and change the repr accordingly.
    # Unfortunately, this function has to be aware of this detail.
    # NB: This is currently skipped for plain tensor parameters to maintain BC. In the future,
    # this should be done for those as well to produce a valid repr.
    if isinstance(self, torch.nn.Parameter) and not is_plain_tensor:
        string_repr = f"Parameter({string_repr})"

    return string_repr


def _functorch_wrapper_str_intern(tensor, *, tensor_contents=None):
    level = torch._C._functorch.maybe_get_level(tensor)
    if level == -1:
        raise AssertionError("expected functorch level to be >= 0, got -1")

    if torch._C._functorch.is_functionaltensor(tensor):
        # Since we're unwrapping the FunctionalTensorWrapper, we need to make sure
        # that it's up to date first
        torch._sync(tensor)

    value = torch._C._functorch.get_unwrapped(tensor)
    value_repr = repr(value)

    indented_value_repr = textwrap.indent(value_repr, " " * 4)
    if torch._C._functorch.is_batchedtensor(tensor):
        bdim = torch._C._functorch.maybe_get_bdim(tensor)
        if bdim == -1:
            raise AssertionError("expected batch dimension to be >= 0, got -1")
        return (
            f"BatchedTensor(lvl={level}, bdim={bdim}, value=\n{indented_value_repr}\n)"
        )
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        return f"GradTrackingTensor(lvl={level}, value=\n{indented_value_repr}\n)"
    if torch._C._functorch.is_functionaltensor(tensor):
        return f"FunctionalTensor(lvl={level}, value=\\\n{value_repr})"

    raise ValueError("We don't know how to print this, please file us an issue")


def _str(self, *, tensor_contents=None):
    with torch.no_grad(), torch.utils._python_dispatch._disable_current_modes():
        guard = torch._C._DisableFuncTorch()  # noqa: F841
        return _str_intern(self, tensor_contents=tensor_contents)
<|endoftext|># mypy: allow-untyped-defs
"""Locally Optimal Block Preconditioned Conjugate Gradient methods."""
# Author: Pearu Peterson
# Created: February 2020

import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function


__all__ = ["lobpcg"]


def _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U):
    # compute F, such that F_ij = (d_j - d_i)^{-1} for i != j, F_ii = 0
    F = D.unsqueeze(-2) - D.unsqueeze(-1)
    F.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
    F.pow_(-1)

    # A.grad = U (D.grad + (U^T U.grad * F)) U^T
    Ut = U.mT.contiguous()
    res = torch.matmul(
        U, torch.matmul(torch.diag_embed(D_grad) + torch.matmul(Ut, U_grad) * F, Ut)
    )

    return res


def _polynomial_coefficients_given_roots(roots):
    """
    Given the `roots` of a polynomial, find the polynomial's coefficients.

    If roots = (r_1, ..., r_n), then the method returns
    coefficients (a_0, a_1, ..., a_n (== 1)) so that
    p(x) = (x - r_1) * ... * (x - r_n)
         = x^n + a_{n-1} * x^{n-1} + ... a_1 * x_1 + a_0

    Note: for better performance requires writing a low-level kernel
    """
    poly_order = roots.shape[-1]
    poly_coeffs_shape = list(roots.shape)
    # we assume p(x) = x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0,
    # so poly_coeffs = {a_0, ..., a_n, a_{n+1}(== 1)},
    # but we insert one extra coefficient to enable better vectorization below
    poly_coeffs_shape[-1] += 2
    poly_coeffs = roots.new_zeros(poly_coeffs_shape)
    poly_coeffs[..., 0] = 1
    poly_coeffs[..., -1] = 1

    # perform the Horner's rule
    for i in range(1, poly_order + 1):
        # note that it is computationally hard to compute backward for this method,
        # because then given the coefficients it would require finding the roots and/or
        # calculating the sensitivity based on the Vieta's theorem.
        # So the code below tries to circumvent the explicit root finding by series
        # of operations on memory copies imitating the Horner's method.
        # The memory copies are required to construct nodes in the computational graph
        # by exploiting the explicit (not in-place, separate node for each step)
        # recursion of the Horner's method.
        # Needs more memory, O(... * k^2), but with only O(... * k^2) complexity.
        poly_coeffs_new = poly_coeffs.clone() if roots.requires_grad else poly_coeffs
        out = poly_coeffs_new.narrow(-1, poly_order - i, i + 1)
        out -= roots.narrow(-1, i - 1, 1) * poly_coeffs.narrow(
            -1, poly_order - i + 1, i + 1
        )
        poly_coeffs = poly_coeffs_new

    return poly_coeffs.narrow(-1, 1, poly_order + 1)


def _polynomial_value(poly, x, zero_power, transition):
    """
    A generic method for computing poly(x) using the Horner's rule.

    Args:
      poly (Tensor): the (possibly batched) 1D Tensor representing
                     polynomial coefficients such that
                     poly[..., i] = (a_{i_0}, ..., a{i_n} (==1)), and
                     poly(x) = poly[..., 0] * zero_power + ... + poly[..., n] * x^n

      x (Tensor): the value (possible batched) to evaluate the polynomial `poly` at.

      zero_power (Tensor): the representation of `x^0`. It is application-specific.

      transition (Callable): the function that accepts some intermediate result `int_val`,
                             the `x` and a specific polynomial coefficient
                             `poly[..., k]` for some iteration `k`.
                             It basically performs one iteration of the Horner's rule
                             defined as `x * int_val + poly[..., k] * zero_power`.
                             Note that `zero_power` is not a parameter,
                             because the step `+ poly[..., k] * zero_power` depends on `x`,
                             whether it is a vector, a matrix, or something else, so this
                             functionality is delegated to the user.
    """

    res = zero_power.clone()
    for k in range(poly.size(-1) - 2, -1, -1):
        res = transition(res, x, poly[..., k])
    return res


def _matrix_polynomial_value(poly, x, zero_power=None):
    """
    Evaluates `poly(x)` for the (batched) matrix input `x`.
    Check out `_polynomial_value` function for more details.
    """

    # matrix-aware Horner's rule iteration
    def transition(curr_poly_val, x, poly_coeff):
        res = x.matmul(curr_poly_val)
        res.diagonal(dim1=-2, dim2=-1).add_(poly_coeff.unsqueeze(-1))
        return res

    if zero_power is None:
        zero_power = torch.eye(
            x.size(-1), x.size(-1), dtype=x.dtype, device=x.device
        ).view(*([1] * len(list(x.shape[:-2]))), x.size(-1), x.size(-1))

    return _polynomial_value(poly, x, zero_power, transition)


def _vector_polynomial_value(poly, x, zero_power=None):
    """
    Evaluates `poly(x)` for the (batched) vector input `x`.
    Check out `_polynomial_value` function for more details.
    """

    # vector-aware Horner's rule iteration
    def transition(curr_poly_val, x, poly_coeff):
        res = torch.addcmul(poly_coeff.unsqueeze(-1), x, curr_poly_val)
        return res

    if zero_power is None:
        zero_power = x.new_ones(1).expand(x.shape)

    return _polynomial_value(poly, x, zero_power, transition)


def _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest):
    # compute a projection operator onto an orthogonal subspace spanned by the
    # columns of U defined as (I - UU^T)
    Ut = U.mT.contiguous()
    proj_U_ortho = -U.matmul(Ut)
    proj_U_ortho.diagonal(dim1=-2, dim2=-1).add_(1)

    # compute U_ortho, a basis for the orthogonal complement to the span(U),
    # by projecting a random [..., m, m - k] matrix onto the subspace spanned
    # by the columns of U.
    #
    # fix generator for determinism
    gen = torch.Generator(A.device)

    # orthogonal complement to the span(U)
    U_ortho = proj_U_ortho.matmul(
        torch.randn(
            (*A.shape[:-1], A.size(-1) - D.size(-1)),
            dtype=A.dtype,
            device=A.device,
            generator=gen,
        )
    )
    U_ortho_t = U_ortho.mT.contiguous()

    # compute the coefficients of the characteristic polynomial of the tensor D.
    # Note that D is diagonal, so the diagonal elements are exactly the roots
    # of the characteristic polynomial.
    chr_poly_D = _polynomial_coefficients_given_roots(D)

    # the code below finds the explicit solution to the Sylvester equation
    # U_ortho^T A U_ortho dX - dX D = -U_ortho^T A U
    # and incorporates it into the whole gradient stored in the `res` variable.
    #
    # Equivalent to the following naive implementation:
    # res = A.new_zeros(A.shape)
    # p_res = A.new_zeros(*A.shape[:-1], D.size(-1))
    # for k in range(1, chr_poly_D.size(-1)):
    #     p_res.zero_()
    #     for i in range(0, k):
    #         p_res += (A.matrix_power(k - 1 - i) @ U_grad) * D.pow(i).unsqueeze(-2)
    #     res -= chr_poly_D[k] * (U_ortho @ poly_D_at_A.inverse() @ U_ortho_t @  p_res @ U.t())
    #
    # Note that dX is a differential, so the gradient contribution comes from the backward sensitivity
    # Tr(f(U_grad, D_grad, A, U, D)^T dX) = Tr(g(U_grad, A, U, D)^T dA) for some functions f and g,
    # and we need to compute g(U_grad, A, U, D)
    #
    # The naive implementation is based on the paper
    # Hu, Qingxi, and Daizhan Cheng.
    # "The polynomial solution to the Sylvester matrix equation."
    # Applied mathematics letters 19.9 (2006): 859-864.
    #
    # We can modify the computation of `p_res` from above in a more efficient way
    # p_res =   U_grad * (chr_poly_D[1] * D.pow(0) + ... + chr_poly_D[k] * D.pow(k)).unsqueeze(-2)
    #       + A U_grad * (chr_poly_D[2] * D.pow(0) + ... + chr_poly_D[k] * D.pow(k - 1)).unsqueeze(-2)
    #       + ...
    #       + A.matrix_power(k - 1) U_grad * chr_poly_D[k]
    # Note that this saves us from redundant matrix products with A (elimination of matrix_power)
    U_grad_projected = U_grad
    series_acc = U_grad_projected.new_zeros(U_grad_projected.shape)
    for k in range(1, chr_poly_D.size(-1)):
        poly_D = _vector_polynomial_value(chr_poly_D[..., k:], D)
        series_acc += U_grad_projected * poly_D.unsqueeze(-2)
        U_grad_projected = A.matmul(U_grad_projected)

    # compute chr_poly_D(A) which essentially is:
    #
    # chr_poly_D_at_A = A.new_zeros(A.shape)
    # for k in range(chr_poly_D.size(-1)):
    #     chr_poly_D_at_A += chr_poly_D[k] * A.matrix_power(k)
    #
    # Note, however, for better performance we use the Horner's rule
    chr_poly_D_at_A = _matrix_polynomial_value(chr_poly_D, A)

    # compute the action of `chr_poly_D_at_A` restricted to U_ortho_t
    chr_poly_D_at_A_to_U_ortho = torch.matmul(
        U_ortho_t, torch.matmul(chr_poly_D_at_A, U_ortho)
    )
    # we need to invert 'chr_poly_D_at_A_to_U_ortho`, for that we compute its
    # Cholesky decomposition and then use `torch.cholesky_solve` for better stability.
    # Cholesky decomposition requires the input to be positive-definite.
    # Note that `chr_poly_D_at_A_to_U_ortho` is positive-definite if
    # 1. `largest` == False, or
    # 2. `largest` == True and `k` is even
    # under the assumption that `A` has distinct eigenvalues.
    #
    # check if `chr_poly_D_at_A_to_U_ortho` is positive-definite or negative-definite
    chr_poly_D_at_A_to_U_ortho_sign = -1 if (largest and (k % 2 == 1)) else +1
    chr_poly_D_at_A_to_U_ortho_L = torch.linalg.cholesky(
        chr_poly_D_at_A_to_U_ortho_sign * chr_poly_D_at_A_to_U_ortho
    )

    # compute the gradient part in span(U)
    res = _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)

    # incorporate the Sylvester equation solution into the full gradient
    # it resides in span(U_ortho)
    res -= U_ortho.matmul(
        chr_poly_D_at_A_to_U_ortho_sign
        * torch.cholesky_solve(
            U_ortho_t.matmul(series_acc), chr_poly_D_at_A_to_U_ortho_L
        )
    ).matmul(Ut)

    return res


def _symeig_backward(D_grad, U_grad, A, D, U, largest):
    # if `U` is square, then the columns of `U` is a complete eigenspace
    if U.size(-1) == U.size(-2):
        return _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)
    else:
        return _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest)


class LOBPCGAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        A: Tensor,
        k: int | None = None,
        B: Tensor | None = None,
        X: Tensor | None = None,
        n: int | None = None,
        iK: Tensor | None = None,
        niter: int | None = None,
        tol: float | None = None,
        largest: bool | None = None,
        method: str | None = None,
        tracker: None = None,
        ortho_iparams: dict[str, int] | None = None,
        ortho_fparams: dict[str, float] | None = None,
        ortho_bparams: dict[str, bool] | None = None,
    ) -> tuple[Tensor, Tensor]:
        # makes sure that input is contiguous for efficiency.
        # Note: autograd does not support dense gradients for sparse input yet.
        A = A.contiguous() if (not A.is_sparse) else A
        if B is not None:
            B = B.contiguous() if (not B.is_sparse) else B

        D, U = _lobpcg(
            A,
            k,
            B,
            X,
            n,
            iK,
            niter,
            tol,
            largest,
            method,
            tracker,
            ortho_iparams,
            ortho_fparams,
            ortho_bparams,
        )

        ctx.save_for_backward(A, B, D, U)
        ctx.largest = largest

        return D, U

    @staticmethod
    def backward(ctx, D_grad, U_grad):  # pyrefly: ignore  # bad-override
        A_grad = B_grad = None
        grads = [None] * 14

        A, B, D, U = ctx.saved_tensors
        largest = ctx.largest

        # lobpcg.backward has some limitations. Checks for unsupported input
        if A.is_sparse or (B is not None and B.is_sparse and ctx.needs_input_grad[2]):
            raise ValueError(
                "lobpcg.backward does not support sparse input yet. "
                "Note that lobpcg.forward does though."
            )
        if (
            A.dtype in (torch.complex64, torch.complex128)
            or B is not None
            and B.dtype in (torch.complex64, torch.complex128)
        ):
            raise ValueError(
                "lobpcg.backward does not support complex input yet. "
                "Note that lobpcg.forward does though."
            )
        if B is not None:
            raise ValueError(
                "lobpcg.backward does not support backward with B != I yet."
            )

        if largest is None:
            largest = True

        # symeig backward
        if B is None:
            A_grad = _symeig_backward(D_grad, U_grad, A, D, U, largest)

        # A has index 0
        grads[0] = A_grad
        # B has index 2
        grads[2] = B_grad
        return tuple(grads)


def lobpcg(
    A: Tensor,
    k: int | None = None,
    B: Tensor | None = None,
    X: Tensor | None = None,
    n: int | None = None,
    iK: Tensor | None = None,
    niter: int | None = None,
    tol: float | None = None,
    largest: bool | None = None,
    method: str | None = None,
    tracker: None = None,
    ortho_iparams: dict[str, int] | None = None,
    ortho_fparams: dict[str, float] | None = None,
    ortho_bparams: dict[str, bool] | None = None,
) -> tuple[Tensor, Tensor]:
    """Find the k largest (or smallest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive definite generalized
    eigenvalue problem using matrix-free LOBPCG methods.

    This function is a front-end to the following LOBPCG algorithms
    selectable via `method` argument:

      `method="basic"` - the LOBPCG method introduced by Andrew
      Knyazev, see [Knyazev2001]. A less robust method, may fail when
      Cholesky is applied to singular input.

      `method="ortho"` - the LOBPCG method with orthogonal basis
      selection [StathopoulosEtal2002]. A robust method.

    Supported inputs are dense, sparse, and batches of dense matrices.

    .. note:: In general, the basic method spends least time per
      iteration. However, the robust methods converge much faster and
      are more stable. So, the usage of the basic method is generally
      not recommended but there exist cases where the usage of the
      basic method may be preferred.

    .. warning:: The backward method does not support sparse and complex inputs.
      It works only when `B` is not provided (i.e. `B == None`).
      We are actively working on extensions, and the details of
      the algorithms are going to be published promptly.

    .. warning:: While it is assumed that `A` is symmetric, `A.grad` is not.
      To make sure that `A.grad` is symmetric, so that `A - t * A.grad` is symmetric
      in first-order optimization routines, prior to running `lobpcg`
      we do the following symmetrization map: `A -> (A + A.t()) / 2`.
      The map is performed only when the `A` requires gradients.

    .. warning:: LOBPCG algorithm is not applicable when the number of `A`'s rows
      is smaller than 3x the number of requested eigenpairs `n`.

    Args:

      A (Tensor): the input tensor of size :math:`(*, m, m)`

      k (integer, optional): the number of requested
                  eigenpairs. Default is the number of :math:`X`
                  columns (when specified) or `1`.

      B (Tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When not specified, `B` is interpreted as
                  identity matrix.

      X (tensor, optional): the input tensor of size :math:`(*, m, n)`
                  where `k <= n <= m`. When specified, it is used as
                  initial approximation of eigenvectors. X must be a
                  dense tensor.

      n (integer, optional): if :math:`X` is not specified then `n`
                  specifies the size of the generated random
                  approximation of eigenvectors. Default value for `n`
                  is `k`. If :math:`X` is specified, any provided value of `n` is
                  ignored and `n` is automatically set to the number of
                  columns in :math:`X`.

      iK (tensor, optional): the input tensor of size :math:`(*, m,
                  m)`. When specified, it will be used as preconditioner.

      niter (int, optional): maximum number of iterations. When
                 reached, the iteration process is hard-stopped and
                 the current approximation of eigenpairs is returned.
                 For infinite iteration but until convergence criteria
                 is met, use `-1`.

      tol (float, optional): residual tolerance for stopping
                 criterion. Default is `feps ** 0.5` where `feps` is
                 smallest non-zero floating-point number of the given
                 input tensor `A` data type.

      largest (bool, optional): when True, solve the eigenproblem for
                 the largest eigenvalues. Otherwise, solve the
                 eigenproblem for smallest eigenvalues. Default is
                 `True`.

      method (str, optional): select LOBPCG method. See the
                 description of the function above. Default is
                 "ortho".

      tracker (callable, optional) : a function for tracing the
                 iteration process. When specified, it is called at
                 each iteration step with LOBPCG instance as an
                 argument. The LOBPCG instance holds the full state of
                 the iteration process in the following attributes:

                   `iparams`, `fparams`, `bparams` - dictionaries of
                   integer, float, and boolean valued input
                   parameters, respectively

                   `ivars`, `fvars`, `bvars`, `tvars` - dictionaries
                   of integer, float, boolean, and Tensor valued
                   iteration variables, respectively.

                   `A`, `B`, `iK` - input Tensor arguments.

                   `E`, `X`, `S`, `R` - iteration Tensor variables.

                 For instance:

                   `ivars["istep"]` - the current iteration step
                   `X` - the current approximation of eigenvectors
                   `E` - the current approximation of eigenvalues
                   `R` - the current residual
                   `ivars["converged_count"]` - the current number of converged eigenpairs
                   `tvars["rerr"]` - the current state of convergence criteria

                 Note that when `tracker` stores Tensor objects from
                 the LOBPCG instance, it must make copies of these.

                 If `tracker` sets `bvars["force_stop"] = True`, the
                 iteration process will be hard-stopped.

      ortho_iparams, ortho_fparams, ortho_bparams (dict, optional):
                 various parameters to LOBPCG algorithm when using
                 `method="ortho"`.

    Returns:

      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`

      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`

    References:

      [Knyazev2001] Andrew V. Knyazev. (2001) Toward the Optimal
      Preconditioned Eigensolver: Locally Optimal Block Preconditioned
      Conjugate Gradient Method. SIAM J. Sci. Comput., 23(2),
      517-541. (25 pages)
      https://epubs.siam.org/doi/abs/10.1137/S1064827500366124

      [StathopoulosEtal2002] Andreas Stathopoulos and Kesheng
      Wu. (2002) A Block Orthogonalization Procedure with Constant
      Synchronization Requirements. SIAM J. Sci. Comput., 23(6),
      2165-2182. (18 pages)
      https://epubs.siam.org/doi/10.1137/S1064827500370883

      [DuerschEtal2018] Jed A. Duersch, Meiyue Shao, Chao Yang, Ming
      Gu. (2018) A Robust and Efficient Implementation of LOBPCG.
      SIAM J. Sci. Comput., 40(5), C655-C676. (22 pages)
      https://arxiv.org/abs/1704.07458

    """

    if not torch.jit.is_scripting():
        tensor_ops = (A, B, X, iK)
        if not set(map(type, tensor_ops)).issubset(
            (torch.Tensor, type(None))
        ) and has_torch_function(tensor_ops):
            return handle_torch_function(
                lobpcg,
                tensor_ops,
                A,
                k=k,
                B=B,
                X=X,
                n=n,
                iK=iK,
                niter=niter,
                tol=tol,
                largest=largest,
                method=method,
                tracker=tracker,
                ortho_iparams=ortho_iparams,
                ortho_fparams=ortho_fparams,
                ortho_bparams=ortho_bparams,
            )

    if not torch._jit_internal.is_scripting():
        if A.requires_grad or (B is not None and B.requires_grad):
            # While it is expected that `A` is symmetric,
            # the `A_grad` might be not. Therefore we perform the trick below,
            # so that `A_grad` becomes symmetric.
            # The symmetrization is important for first-order optimization methods,
            # so that (A - alpha * A_grad) is still a symmetric matrix.
            # Same holds for `B`.
            A_sym = (A + A.mT) / 2
            B_sym = (B + B.mT) / 2 if (B is not None) else None

            return LOBPCGAutogradFunction.apply(
                A_sym,
                k,
                B_sym,
                X,
                n,
                iK,
                niter,
                tol,
                largest,
                method,
                tracker,
                ortho_iparams,
                ortho_fparams,
                ortho_bparams,
            )
    else:
        if A.requires_grad or (B is not None and B.requires_grad):
            raise RuntimeError(
                "Script and require grads is not supported atm. "
                "If you just want to do the forward, use .detach() "
                "on A and B before calling into lobpcg"
            )

    return _lobpcg(
        A,
        k,
        B,
        X,
        n,
        iK,
        niter,
        tol,
        largest,
        method,
        tracker,
        ortho_iparams,
        ortho_fparams,
        ortho_bparams,
    )


def _lobpcg(
    A: Tensor,
    k: int | None = None,
    B: Tensor | None = None,
    X: Tensor | None = None,
    n: int | None = None,
    iK: Tensor | None = None,
    niter: int | None = None,
    tol: float | None = None,
    largest: bool | None = None,
    method: str | None = None,
    tracker: None = None,
    ortho_iparams: dict[str, int] | None = None,
    ortho_fparams: dict[str, float] | None = None,
    ortho_bparams: dict[str, bool] | None = None,
) -> tuple[Tensor, Tensor]:
    # A must be square:
    if A.shape[-2] != A.shape[-1]:
        raise AssertionError(f"A must be square, got shape {A.shape}")
    if B is not None:
        # A and B must have the same shapes:
        if A.shape != B.shape:
            raise AssertionError(
                f"A and B must have same shape, got {A.shape} vs {B.shape}"
            )

    dtype = _utils.get_floating_dtype(A)
    device = A.device
    if tol is None:
        feps = {torch.float32: 1.2e-07, torch.float64: 2.23e-16}[dtype]
        tol = feps**0.5

    m = A.shape[-1]
    k = (1 if X is None else X.shape[-1]) if k is None else k
    n = (k if n is None else n) if X is None else X.shape[-1]

    if m < 3 * n:
        raise ValueError(
            f"LOBPCG algorithm is not applicable when the number of A rows (={m})"
            f" is smaller than 3 x the number of requested eigenpairs (={n})"
        )

    method = "ortho" if method is None else method

    iparams = {
        "m": m,
        "n": n,
        "k": k,
        "niter": 1000 if niter is None else niter,
    }

    fparams = {
        "tol": tol,
    }

    bparams = {"largest": True if largest is None else largest}

    if method == "ortho":
        if ortho_iparams is not None:
            iparams.update(ortho_iparams)
        if ortho_fparams is not None:
            fparams.update(ortho_fparams)
        if ortho_bparams is not None:
            bparams.update(ortho_bparams)
        iparams["ortho_i_max"] = iparams.get("ortho_i_max", 3)
        iparams["ortho_j_max"] = iparams.get("ortho_j_max", 3)
        fparams["ortho_tol"] = fparams.get("ortho_tol", tol)
        fparams["ortho_tol_drop"] = fparams.get("ortho_tol_drop", tol)
        fparams["ortho_tol_replace"] = fparams.get("ortho_tol_replace", tol)
        bparams["ortho_use_drop"] = bparams.get("ortho_use_drop", False)

    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker  # type: ignore[method-assign]

    if len(A.shape) > 2:
        N = int(torch.prod(torch.tensor(A.shape[:-2])))
        bA = A.reshape((N,) + A.shape[-2:])
        bB = B.reshape((N,) + A.shape[-2:]) if B is not None else None
        bX = X.reshape((N,) + X.shape[-2:]) if X is not None else None
        bE = torch.empty((N, k), dtype=dtype, device=device)
        bXret = torch.empty((N, m, k), dtype=dtype, device=device)

        for i in range(N):
            A_ = bA[i]
            B_ = bB[i] if bB is not None else None
            X_ = (
                torch.randn((m, n), dtype=dtype, device=device) if bX is None else bX[i]
            )
            if not (len(X_.shape) == 2 and X_.shape == (m, n)):
                raise AssertionError(
                    f"X_ shape mismatch: got {X_.shape}, expected {(m, n)}"
                )
            iparams["batch_index"] = i
            worker = LOBPCG(A_, B_, X_, iK, iparams, fparams, bparams, method, tracker)
            worker.run()
            bE[i] = worker.E[:k]
            bXret[i] = worker.X[:, :k]

        if not torch.jit.is_scripting():
            LOBPCG.call_tracker = LOBPCG_call_tracker_orig  # type: ignore[method-assign]

        return bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k))

    X = torch.randn((m, n), dtype=dtype, device=device) if X is None else X
    if not (len(X.shape) == 2 and X.shape == (m, n)):
        raise AssertionError(f"X shape mismatch: got {X.shape}, expected {(m, n)}")

    worker = LOBPCG(A, B, X, iK, iparams, fparams, bparams, method, tracker)

    worker.run()

    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker_orig  # type: ignore[method-assign]

    return worker.E[:k], worker.X[:, :k]


class LOBPCG:
    """Worker class of LOBPCG methods."""

    def __init__(
        self,
        A: Tensor | None,
        B: Tensor | None,
        X: Tensor,
        iK: Tensor | None,
        iparams: dict[str, int],
        fparams: dict[str, float],
        bparams: dict[str, bool],
        method: str,
        tracker: None,
    ) -> None:
        # constant parameters
        self.A = A
        self.B = B
        self.iK = iK
        self.iparams = iparams
        self.fparams = fparams
        self.bparams = bparams
        self.method = method
        self.tracker = tracker
        m = iparams["m"]
        n = iparams["n"]

        # variable parameters
        self.X = X
        self.E = torch.zeros((n,), dtype=X.dtype, device=X.device)
        self.R = torch.zeros((m, n), dtype=X.dtype, device=X.device)
        self.S = torch.zeros((m, 3 * n), dtype=X.dtype, device=X.device)
        self.tvars: dict[str, Tensor] = {}
        self.ivars: dict[str, int] = {"istep": 0}
        self.fvars: dict[str, float] = {"_": 0.0}
        self.bvars: dict[str, bool] = {"_": False}

    def __str__(self):
        lines = ["LOBPCG:"]
        lines += [f"  iparams={self.iparams}"]
        lines += [f"  fparams={self.fparams}"]
        lines += [f"  bparams={self.bparams}"]
        lines += [f"  ivars={self.ivars}"]
        lines += [f"  fvars={self.fvars}"]
        lines += [f"  bvars={self.bvars}"]
        lines += [f"  tvars={self.tvars}"]
        lines += [f"  A={self.A}"]
        lines += [f"  B={self.B}"]
        lines += [f"  iK={self.iK}"]
        lines += [f"  X={self.X}"]
        lines += [f"  E={self.E}"]
        r = ""
        for line in lines:
            r += line + "\n"
        return r

    def update(self):
        """Set and update iteration variables."""
        if self.ivars["istep"] == 0:
            X_norm = float(torch.linalg.vector_norm(self.X))
            iX_norm = X_norm**-1
            A_norm = (
                float(torch.linalg.vector_norm(_utils.matmul(self.A, self.X))) * iX_norm
            )
            B_norm = (
                float(torch.linalg.vector_norm(_utils.matmul(self.B, self.X))) * iX_norm
            )
            self.fvars["X_norm"] = X_norm
            self.fvars["A_norm"] = A_norm
            self.fvars["B_norm"] = B_norm
            self.ivars["iterations_left"] = self.iparams["niter"]
            self.ivars["converged_count"] = 0
            self.ivars["converged_end"] = 0

        if self.method == "ortho":
            self._update_ortho()
        else:
            self._update_basic()

        self.ivars["iterations_left"] = self.ivars["iterations_left"] - 1
        self.ivars["istep"] = self.ivars["istep"] + 1

    def update_residual(self):
        """Update residual R from A, B, X, E."""
        mm = _utils.matmul
        self.R = mm(self.A, self.X) - mm(self.B, self.X) * self.E

    def update_converged_count(self):
        """Determine the number of converged eigenpairs using backward stable
        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].

        Users may redefine this method for custom convergence criteria.
        """
        # (...) -> int
        prev_count = self.ivars["converged_count"]
        tol = self.fparams["tol"]
        A_norm = self.fvars["A_norm"]
        B_norm = self.fvars["B_norm"]
        E, X, R = self.E, self.X, self.R
        rerr = torch.linalg.vector_norm(R, ord=2, dim=(0,)) / (
            torch.linalg.vector_norm(X, ord=2, dim=(0,))
            * (A_norm + torch.abs(E[: X.shape[-1]]) * B_norm)
        )
        converged = rerr < tol
        count = 0
        for b in converged:
            if not b:
                # ignore convergence of following pairs to ensure
                # strict ordering of eigenpairs
                break
            count += 1
        if count < prev_count:
            raise AssertionError(
                f"the number of converged eigenpairs (was {prev_count}, got {count}) cannot decrease"
            )
        self.ivars["converged_count"] = count
        self.tvars["rerr"] = rerr
        return count

    def stop_iteration(self):
        """Return True to stop iterations.

        Note that tracker (if defined) can force-stop iterations by
        setting ``worker.bvars['force_stop'] = True``.
        """
        return (
            self.bvars.get("force_stop", False)
            or self.ivars["iterations_left"] == 0
            or self.ivars["converged_count"] >= self.iparams["k"]
        )

    def run(self):
        """Run LOBPCG iterations.

        Use this method as a template for implementing LOBPCG
        iteration scheme with custom tracker that is compatible with
        TorchScript.
        """
        self.update()

        if not torch.jit.is_scripting() and self.tracker is not None:
            self.call_tracker()

        while not self.stop_iteration():
            self.update()

            if not torch.jit.is_scripting() and self.tracker is not None:
                self.call_tracker()

    @torch.jit.unused
    def call_tracker(self):
        """Interface for tracking iteration process in Python mode.

        Tracking the iteration process is disabled in TorchScript
        mode. In fact, one should specify tracker=None when JIT
        compiling functions using lobpcg.
        """
        # do nothing when in TorchScript mode

    # Internal methods

    def _update_basic(self):
        """
        Update or initialize iteration variables when `method == "basic"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            E, Z = _utils.symeig(M, largest)
            self.X[:] = mm(self.X, mm(Ri, Z))
            self.E[:] = E
            np = 0
            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X

            W = _utils.matmul(self.iK, self.R)
            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W
        else:
            S_ = self.S[:, nc:ns]
            Ri = self._get_rayleigh_ritz_transform(S_)
            M = _utils.qform(_utils.qform(self.A, S_), Ri)
            E_, Z = _utils.symeig(M, largest)
            self.X[:, nc:] = mm(S_, mm(Ri, Z[:, : n - nc]))
            self.E[nc:] = E_[: n - nc]
            P = mm(S_, mm(Ri, Z[:, n : 2 * n - nc]))
            np = P.shape[-1]

            self.update_residual()
            nc = self.update_converged_count()
            self.S[..., :n] = self.X
            self.S[:, n : n + np] = P
            W = _utils.matmul(self.iK, self.R[:, nc:])

            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

    def _update_ortho(self):
        """
        Update or initialize iteration variables when `method == "ortho"`.
        """
        mm = torch.matmul
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        if self.ivars["istep"] == 0:
            Ri = self._get_rayleigh_ritz_transform(self.X)
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            _E, Z = _utils.symeig(M, largest)
            self.X = mm(self.X, mm(Ri, Z))
            self.update_residual()
            np = 0
            nc = self.update_converged_count()
            self.S[:, :n] = self.X
            W = self._get_ortho(self.R, self.X)
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

        else:
            S_ = self.S[:, nc:ns]
            # Rayleigh-Ritz procedure
            E_, Z = _utils.symeig(_utils.qform(self.A, S_), largest)

            # Update E, X, P
            self.X[:, nc:] = mm(S_, Z[:, : n - nc])
            self.E[nc:] = E_[: n - nc]
            P = mm(S_, mm(Z[:, n - nc :], _utils.basis(Z[: n - nc, n - nc :].mT)))
            np = P.shape[-1]

            # check convergence
            self.update_residual()
            nc = self.update_converged_count()

            # update S
            self.S[:, :n] = self.X
            self.S[:, n : n + np] = P
            W = self._get_ortho(self.R[:, nc:], self.S[:, : n + np])
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            self.S[:, n + np : ns] = W

    def _get_rayleigh_ritz_transform(self, S):
        """Return a transformation matrix that is used in Rayleigh-Ritz
        procedure for reducing a general eigenvalue problem :math:`(S^TAS)
        C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T
        S^TAS Ri) Z = Z E` where `C = Ri Z`.

        .. note:: In the original Rayleigh-Ritz procedure in
          [DuerschEtal2018], the problem is formulated as follows::

            SAS = S^T A S
            SBS = S^T B S
            D = (<diagonal matrix of SBS>) ** -1/2
            R^T R = Cholesky(D SBS D)
            Ri = D R^-1
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          To reduce the number of matrix products (denoted by empty
          space between matrices), here we introduce element-wise
          products (denoted by symbol `*`) so that the Rayleigh-Ritz
          procedure becomes::

            SAS = S^T A S
            SBS = S^T B S
            d = (<diagonal of SBS>) ** -1/2    # this is 1-d column vector
            dd = d d^T                         # this is 2-d matrix
            R^T R = Cholesky(dd * SBS)
            Ri = R^-1 * d                      # broadcasting
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          where `dd` is 2-d matrix that replaces matrix products `D M
          D` with one element-wise product `M * dd`; and `d` replaces
          matrix product `D M` with element-wise product `M *
          d`. Also, creating the diagonal matrix `D` is avoided.

        Args:
        S (Tensor): the matrix basis for the search subspace, size is
                    :math:`(m, n)`.

        Returns:
        Ri (tensor): upper-triangular transformation matrix of size
                     :math:`(n, n)`.

        """
        B = self.B
        SBS = _utils.qform(B, S)
        d_row = SBS.diagonal(0, -2, -1) ** -0.5
        d_col = d_row.reshape(d_row.shape[0], 1)
        # TODO use torch.linalg.cholesky_solve once it is implemented
        R = torch.linalg.cholesky((SBS * d_row) * d_col, upper=True)
        return torch.linalg.solve_triangular(
            R, d_row.diag_embed(), upper=True, left=False
        )

    def _get_svqb(self, U: Tensor, drop: bool, tau: float) -> Tensor:
        """Return B-orthonormal U.

        .. note:: When `drop` is `False` then `svqb` is based on the
                  Algorithm 4 from [DuerschPhD2015] that is a slight
                  modification of the corresponding algorithm
                  introduced in [StathopoulosWu2002].

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          drop (bool) : when True, drop columns that
                     contribution to the `span([U])` is small.
          tau (float) : positive tolerance

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`), size
                       is (m, n1), where `n1 = n` if `drop` is `False,
                       otherwise `n1 <= n`.

        """
        if torch.numel(U) == 0:
            return U
        UBU = _utils.qform(self.B, U)
        d = UBU.diagonal(0, -2, -1)

        # Detect and drop exact zero columns from U. While the test
        # `abs(d) == 0` is unlikely to be True for random data, it is
        # possible to construct input data to lobpcg where it will be
        # True leading to a failure (notice the `d ** -0.5` operation
        # in the original algorithm). To prevent the failure, we drop
        # the exact zero columns here and then continue with the
        # original algorithm below.
        nz = torch.where(abs(d) != 0.0)
        if len(nz) != 1:
            raise AssertionError(f"expected nz to have length 1, got {nz}")
        if len(nz[0]) < len(d):
            U = U[:, nz[0]]
            if torch.numel(U) == 0:
                return U
            UBU = _utils.qform(self.B, U)
            d = UBU.diagonal(0, -2, -1)
            nz = torch.where(abs(d) != 0.0)
            if len(nz[0]) != len(d):
                raise AssertionError(
                    f"expected nz[0] length {len(d)}, got {len(nz[0])}"
                )

        # The original algorithm 4 from [DuerschPhD2015].
        d_col = (d**-0.5).reshape(d.shape[0], 1)
        DUBUD = (UBU * d_col) * d_col.mT
        E, Z = _utils.symeig(DUBUD)
        t = tau * abs(E).max()
        if drop:
            keep = torch.where(E > t)
            if len(keep) != 1:
                raise AssertionError(f"expected keep to have length 1, got {keep}")
            E = E[keep[0]]
            Z = Z[:, keep[0]]
            d_col = d_col[keep[0]]
        else:
            E[(torch.where(E < t))[0]] = t

        return torch.matmul(
            U * d_col.mT,
            Z * E**-0.5,
        )

    def _get_ortho(self, U, V):
        """Return B-orthonormal U with columns are B-orthogonal to V.

        .. note:: When `bparams["ortho_use_drop"] == False` then
                  `_get_ortho` is based on the Algorithm 3 from
                  [DuerschPhD2015] that is a slight modification of
                  the corresponding algorithm introduced in
                  [StathopoulosWu2002]. Otherwise, the method
                  implements Algorithm 6 from [DuerschPhD2015]

        .. note:: If all U columns are B-collinear to V then the
                  returned tensor U will be empty.

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          V (Tensor) : B-orthogonal external basis, size is (m, k)

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`)
                       such that :math:`V^T B U=0`, size is (m, n1),
                       where `n1 = n` if `drop` is `False, otherwise
                       `n1 <= n`.
        """
        mm = torch.matmul
        mm_B = _utils.matmul
        m = self.iparams["m"]
        tau_ortho = self.fparams["ortho_tol"]
        tau_drop = self.fparams["ortho_tol_drop"]
        tau_replace = self.fparams["ortho_tol_replace"]
        i_max = self.iparams["ortho_i_max"]
        j_max = self.iparams["ortho_j_max"]
        # when use_drop==True, enable dropping U columns that have
        # small contribution to the `span([U, V])`.
        use_drop = self.bparams["ortho_use_drop"]

        # clean up variables from the previous call
        for vkey in list(self.fvars.keys()):
            if vkey.startswith("ortho_") and vkey.endswith("_rerr"):
                self.fvars.pop(vkey)
        self.ivars.pop("ortho_i", 0)
        self.ivars.pop("ortho_j", 0)

        BV_norm = torch.linalg.vector_norm(mm_B(self.B, V))
        BU = mm_B(self.B, U)
        VBU = mm(V.mT, BU)
        i = j = 0
        for i in range(i_max):
            U = U - mm(V, VBU)
            drop = False
            tau_svqb = tau_drop
            for j in range(j_max):
                if use_drop:
                    U = self._get_svqb(U, drop, tau_svqb)
                    drop = True
                    tau_svqb = tau_replace
                else:
                    U = self._get_svqb(U, False, tau_replace)
                if torch.numel(U) == 0:
                    # all initial U columns are B-collinear to V
                    self.ivars["ortho_i"] = i
                    self.ivars["ortho_j"] = j
                    return U
                BU = mm_B(self.B, U)
                UBU = mm(U.mT, BU)
                U_norm = torch.linalg.vector_norm(U)
                BU_norm = torch.linalg.vector_norm(BU)
                R = UBU - torch.eye(UBU.shape[-1], device=UBU.device, dtype=UBU.dtype)
                R_norm = torch.linalg.vector_norm(R)
                # https://github.com/pytorch/pytorch/issues/33810 workaround:
                rerr = float(R_norm) * float(BU_norm * U_norm) ** -1
                vkey = f"ortho_UBUmI_rerr[{i}, {j}]"
                self.fvars[vkey] = rerr
                if rerr < tau_ortho:
                    break
            VBU = mm(V.mT, BU)
            VBU_norm = torch.linalg.vector_norm(VBU)
            U_norm = torch.linalg.vector_norm(U)
            rerr = float(VBU_norm) * float(BV_norm * U_norm) ** -1
            vkey = f"ortho_VBU_rerr[{i}]"
            self.fvars[vkey] = rerr
            if rerr < tau_ortho:
                break
            if m < U.shape[-1] + V.shape[-1]:
                # TorchScript needs the class var to be assigned to a local to
                # do optional type refinement
                B = self.B
                if B is None:
                    raise AssertionError("B must not be None")
                raise ValueError(
                    "Overdetermined shape of U:"
                    f" #B-cols(={B.shape[-1]}) >= #U-cols(={U.shape[-1]}) + #V-cols(={V.shape[-1]}) must hold"
                )
        self.ivars["ortho_i"] = i
        self.ivars["ortho_j"] = j
        return U


# Calling tracker is separated from LOBPCG definitions because
# TorchScript does not support user-defined callback arguments:
LOBPCG_call_tracker_orig = LOBPCG.call_tracker


def LOBPCG_call_tracker(self):
    self.tracker(self)
<|endoftext|>from typing_extensions import deprecated

import torch


# Preserved only for BC reasons
@deprecated(
    "`torch._streambase._StreamBase` is deprecated. Please use `torch.Stream` instead.",
    category=FutureWarning,
)
class _StreamBase(torch.Stream):
    pass


@deprecated(
    "`torch._streambase._EventBase` is deprecated. Please use `torch.Event` instead.",
    category=FutureWarning,
)
class _EventBase(torch.Event):
    pass
<|endoftext|># mypy: allow-untyped-defs
import inspect

from torch._custom_op.impl import (
    _custom_op_with_schema,
    _find_custom_op,
    infer_schema,
    parse_qualname,
    validate_namespace,
)
from torch.library import get_ctx


__all__ = [
    "custom_op",
    "impl",
    "impl_abstract",
    "get_ctx",
    "impl_save_for_backward",
    "impl_backward",
]


def custom_op(qualname, func_or_schema=None):
    r"""Register a new custom operator

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
      various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs.

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Arguments:
        qualname (str): Should be a string that looks like
            "namespace::operator_name". Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        func_or_schema (Union[Callable, str]): Each PyTorch operator needs a
            schema that tells PyTorch the types of the inputs/outputs.
            If this is a Callable, we will automatically infer the schema from
            the type annotations on the function (see examples). Otherwise,
            if you don't want to use type annotations, you may provide us the
            schema string.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Step 1: define the custom op.
        >>> # We need to provide the API a "prototype function"
        >>> # (a function that returns NotImplementedError), from which
        >>> # we will infer the types of the inputs and outputs.
        >>> @torch._custom_ops.custom_op("mylibrary::numpy_sin")
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     raise NotImplementedError
        >>>
        >>> # The custom op is now accessible via the torch.ops module:
        >>> torch.ops.mylibrary.numpy_sin
        >>>
        >>> # Step 2: Register an implementation for various PyTorch subsystems
        >>>
        >>> # Register an implementation for CPU tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cpu")
        >>> def numpy_sin_impl_cpu(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Register an implementation for CUDA tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cuda")
        >>> def numpy_sin_impl_cuda(x):
        >>>     return torch.from_numpy(np.sin(x.cpu().numpy())).to(x.device)
        >>>
        >>> x = torch.randn(3)
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cpu
        >>>
        >>> x_cuda = x.cuda()
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cuda

    """
    ns, name = parse_qualname(qualname)
    validate_namespace(ns)

    def inner(func):
        if not inspect.isfunction(func):
            raise ValueError(
                f"custom_op(...)(func): Expected `func` to be a Python "
                f"function, got: {type(func)}"
            )

        if func.__name__ != name:
            raise ValueError(
                f"custom_op(qualname='{qualname}', ...)(func): expected `func` "
                f"to have name '{name}' but got '{func.__name__}'. "
                f"Please either change the name of `func` or the qualname that "
                f"is passed to `custom_op`"
            )

        schema = infer_schema(func, mutates_args=())
        _custom_op_with_schema(qualname, schema)
        return func

    if func_or_schema is None:
        return inner
    if isinstance(func_or_schema, str):
        _custom_op_with_schema(qualname, func_or_schema)
    else:
        return inner(func_or_schema)


def impl(qualname, *, device_types=("cpu", "cuda"), func=None):
    r"""Register an implementation for a device type for this custom op.

    If the op is passed multiple Tensor inputs with different device
    types, it will dispatch to the registered implementation for the highest
    priority device type among those present.
    The supported device types, in order of priority, are {'cuda', 'cpu'}.

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Arguments:
        device_types (str or Iterable[str]): the device type(s) to register the function for.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Step 1: define the custom op.
        >>> # We need to provide the API a "prototype function"
        >>> # (a function that returns NotImplementedError), from which
        >>> # we will infer the types of the inputs and outputs.
        >>> @torch._custom_ops.custom_op("mylibrary::numpy_cos")
        >>> def numpy_cos(x: Tensor) -> Tensor:
        >>>     raise NotImplementedError
        >>>
        >>> # The custom op is now accessible via the torch.ops module:
        >>> torch.ops.mylibrary.numpy_cos
        >>>
        >>> # Step 2: Register an implementation for various PyTorch subsystems
        >>>
        >>> # Register an implementation for CPU tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_cos", device_types="cpu")
        >>> def numpy_cos_impl_cpu(x):
        >>>     return torch.from_numpy(np.cos(x.numpy()))
        >>>
        >>> # Register an implementation for CUDA tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_cos", device_types="cuda")
        >>> def numpy_cos_impl_cuda(x):
        >>>     return torch.from_numpy(np.cos(x.cpu().numpy())).to(x.device)
        >>>
        >>> x = torch.randn(3)
        >>> torch.ops.mylibrary.numpy_cos(x)  # calls numpy_cos_impl_cpu
        >>>
        >>> x_cuda = x.cuda()
        >>> torch.ops.mylibrary.numpy_cos(x)  # calls numpy_cos_impl_cuda

    """

    def inner(func):
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        custom_op.impl(device_types, _stacklevel=3)(func)
        return func

    if func is None:
        return inner
    return inner(func)


def impl_abstract(qualname, *, func=None):
    r"""Register an abstract implementation for this operator.

    An "abstract implementation" specifies the behavior of this operator on
    Tensors that carry no data. Given some input Tensors with certain properties
    (sizes/strides/storage_offset/device), it specifies what the properties of
    the output Tensors are.

    The abstract implementation has the same signature as the operator.
    It is run for both FakeTensors and meta tensors. To write an abstract
    implementation, assume that all Tensor inputs to the operator are
    regular CPU/CUDA/Meta tensors, but they do not have storage, and
    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
    The abstract implementation must consist of only PyTorch operations
    (and may not directly access the storage or data of any input or
    intermediate Tensors).

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Examples::
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> @torch._custom_ops.custom_op("mylibrary::custom_linear")
        >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        >>>     raise NotImplementedError
        >>>
        >>> @torch._custom_ops.impl_abstract("mylibrary::custom_linear")
        >>> def custom_linear_abstract(x, weight):
        >>>     assert x.dim() == 2
        >>>     assert weight.dim() == 2
        >>>     assert bias.dim() == 1
        >>>     assert x.shape[1] == weight.shape[1]
        >>>     assert weight.shape[0] == bias.shape[0]
        >>>     assert x.device == weight.device
        >>>
        >>>     return (x @ weight.t()) + bias
        >>>
        >>> # Example 2: an operator with data-dependent output shape
        >>> @torch._custom_ops.custom_op('mylibrary::custom_nonzero')
        >>> def custom_nonzero(x: Tensor) -> Tensor:
        >>>     ...
        >>>
        >>> @torch._custom_ops.impl_abstract("mylibrary::custom_nonzero")
        >>> def custom_nonzero_abstract(x):
        >>>     # Number of nonzero-elements is data-dependent.
        >>>     # Since we cannot peek at the data in an abstract impl,
        >>>     # we use the ctx object to construct a new symint that
        >>>     # represents the data-dependent size.
        >>>     ctx = torch._custom_ops.get_ctx()
        >>>     nnz = ctx.create_unbacked_symint()
        >>>     shape = [x.dim(), nnz]
        >>>     result = x.new_empty(shape, dtype=torch.long)
        >>>     return result
        >>>
        >>> @torch._custom_ops.impl("mylibrary::custom_nonzero")
        >>> def custom_nonzero_impl(x):
        >>>     x_np = to_numpy(x)
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     # unbacked symbolic ints in PyTorch must be >= 2, so we
        >>>     # constrain the range to at least 2
        >>>     if res.shape[0] <= 1:
        >>>         raise RuntimeError("not supported")
        >>>     return torch.tensor(res, device=x.device)

    """
    import torch.library

    return torch.library.register_fake(qualname, func, _stacklevel=2)


def impl_save_for_backward(qualname, *, func=None):
    r"""Register a function that tells us what to save for backward.

    Please see :func:`impl_backward` for more details.
    """

    def inner(func):
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        custom_op.impl_save_for_backward(_stacklevel=3)(func)
        return func

    if func is None:
        return inner
    return inner(func)


def impl_backward(qualname, output_differentiability=None, *, func=None):
    r"""Registers a backward formula for an operator.

    In order for an operator to work with autograd, you need to register
    a backward formula. There are two pieces to this:
    1. You must give us a function to specify what to save for backward.
       Call this the "save for backward" function.
    2. You must give us a function that computes gradients. Call this the
       "backward" function.

    Use `impl_save_for_backward` to define a "save for backward" function
    that specifies what gets saved for backward. The function should accept
    two arguments ``(inputs, output)`` and return the quantities to be saved
    for backward.

    During runtime, when you call the operator in a forwards pass, PyTorch
    will invoke the "save for backward" function with the inputs and output
    of the operator.

    Use `impl_backward` to define the "backward" function. The backward
    function must accept ``(ctx, saved, *grads)``:
    - ``ctx`` is a context object where we may provide information
    - ``saved`` is exactly what gets returned from the "save for backward"
      function
    - ``grads`` is one or more gradients. The number of gradients matches
      the number of outputs of the operator.

    The backward function must return a dict that maps the name of
    an input to the operator to its corresponding gradient. All inputs that
    were declared to be Tensors in the operator definition must be accounted
    for in the dict. The gradient may be a Tensor or None.

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    """

    def inner(func):
        custom_op = _find_custom_op(qualname, also_check_torch_library=True)
        custom_op.impl_backward(output_differentiability, _stacklevel=3)(func)
        return func

    if func is None:
        return inner
    return inner(func)


def _destroy(qualname):
    """De-registers a custom op. For testing purposes only"""
    custom_op = _find_custom_op(qualname)
    custom_op._destroy()
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch


__all__ = [
    "set_rng_state",
    "get_rng_state",
    "manual_seed",
    "seed",
    "initial_seed",
    "fork_rng",
    "thread_safe_generator",
]


if TYPE_CHECKING:
    from torch.utils.data._utils.worker import WorkerInfo

from torch._C import default_generator


def set_rng_state(new_state: torch.Tensor) -> None:
    r"""Sets the random number generator state.

    .. note:: This function only works for CPU. For CUDA, please use
        :func:`torch.manual_seed`, which works for both CPU and CUDA.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    default_generator.set_state(new_state)


def get_rng_state() -> torch.Tensor:
    r"""Returns the random number generator state as a `torch.ByteTensor`.

    .. note:: The returned state is for the default generator on CPU only.

    See also: :func:`torch.random.fork_rng`.
    """
    return default_generator.get_state()


def manual_seed(seed) -> torch._C.Generator:
    r"""Sets the seed for generating random numbers on all devices. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    """
    return _manual_seed_impl(seed)


def _manual_seed_impl(seed) -> torch._C.Generator:
    seed = int(seed)
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    import torch.mps

    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    import torch.xpu

    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    import torch.mtia

    if not torch.mtia._is_in_bad_fork():
        torch.mtia.manual_seed_all(seed)

    _seed_custom_device(seed)

    return default_generator.manual_seed(seed)


def seed() -> int:
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number on all devices. Returns a 64 bit number used to seed the RNG.
    """
    seed = default_generator.seed()
    import torch.cuda

    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    import torch.mps

    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    import torch.xpu

    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    import torch.mtia

    if not torch.mtia._is_in_bad_fork():
        torch.mtia.manual_seed_all(seed)

    _seed_custom_device(seed)

    return seed


def _seed_custom_device(seed) -> None:
    r"""Sets the seed to generate random numbers for custom device.

    Args:
        seed (int): The desired seed.

    See [Note: support the custom device with privateuse1]
    """
    seed = int(seed)
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    if hasattr(torch, custom_backend_name):
        custom_device_mod = getattr(torch, custom_backend_name)
        _bad_fork_name = "_is_in_bad_fork"
        _seed_all_name = "manual_seed_all"
        if hasattr(custom_device_mod, _bad_fork_name) and hasattr(
            custom_device_mod, _seed_all_name
        ):
            if not getattr(custom_device_mod, _bad_fork_name)():
                getattr(custom_device_mod, _seed_all_name)(seed)
        else:
            message = f"Set seed for `{custom_backend_name}` device does not take effect, please add API's "
            message += f"`{_bad_fork_name}` and `{_seed_all_name}` to `{custom_backend_name}` device module."
            warnings.warn(message, UserWarning, stacklevel=3)


def initial_seed() -> int:
    r"""Returns the initial seed for generating random numbers as a
    Python `long`.

    .. note:: The returned seed is for the default generator on CPU only.
    """
    return default_generator.initial_seed()


_fork_rng_warned_already = False


@contextlib.contextmanager
def fork_rng(
    devices=None,
    enabled=True,
    _caller="fork_rng",
    _devices_kw="devices",
    device_type=None,
) -> Generator:
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str): device type str, default is ``None``, in which case the type
            is taken from :func:`torch.accelerator.current_accelerator`, falling back
            to ``"cuda"`` when the type cannot be determined. As for supported devices,
            see details in :ref:`accelerator<accelerators>`
    """

    if device_type == "meta":
        yield
        return

    acc = torch.accelerator.current_accelerator()
    # Default to cuda instead of CPU since CPU rng is always forked
    device_type = device_type or (acc.type if acc is not None else "cuda")

    device_mod = torch.get_device_module(device_type)
    if device_mod is None:
        raise RuntimeError(
            f"torch has no module of `{device_type}`, you should register "
            + "a module by `torch._register_device_module`."
        )
    global _fork_rng_warned_already

    # Internal arguments:
    #   _caller: the function which called fork_rng, which the user used
    #   _devices_kw: the devices keyword of _caller

    if not enabled:
        yield
        return

    if devices is None:
        num_devices = device_mod.device_count()
        if num_devices > 1 and not _fork_rng_warned_already:
            message = (
                f"{device_type.upper()} reports that you have {num_devices} available devices, and "
                f"you have used {_caller} without explicitly specifying which devices are being used. "
                f"For safety, we initialize *every* {device_type.upper()} device by default, which can "
                f"be quite slow if you have a lot of {device_type.upper()}s. If you know that you are only"
                f" making use of a few {device_type.upper()} devices, set the environment variable "
                f"{device_type.upper()}_VISIBLE_DEVICES or the '{_devices_kw}' keyword argument of {_caller} "
                "with the set of devices you are actually using. For example, if you are using CPU only, "
                "set device.upper()_VISIBLE_DEVICES= or devices=[]; if you are using device 0 only, "
                f"set {device_type.upper()}_VISIBLE_DEVICES=0 or devices=[0].  To initialize all devices "
                f"and suppress this warning, set the '{_devices_kw}' keyword argument to "
                f"`range(torch.{device_type}.device_count())`."
            )
            warnings.warn(message, stacklevel=2)
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = torch.get_rng_state()
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        for device, device_rng_state in zip(devices, device_rng_states):
            device_mod.set_rng_state(device_rng_state, device)


def thread_safe_generator() -> torch.Generator | None:
    """Returns a thread-safe random number generator for use in DataLoader workers.
    This function provides a convenient way for transforms and user code to use
    thread-safe random number generation without manually checking worker context.
    When called in a DataLoader thread worker, returns the worker's thread-local
    :class:`torch.Generator`. When called in the main process or process workers,
    returns ``None`` (which causes PyTorch functions to use the default global RNG).
    Returns:
        Optional[torch.Generator]: Thread-local generator in thread workers, None otherwise.
    Example::
        >>> from torch.random import thread_safe_generator
        >>> generator = thread_safe_generator()
        >>> torch.randint(0, 10, (5,), generator=generator)
    Example with transforms::
        >>> from torch.random import thread_safe_generator
        >>> class MyRandomTransform:
        ...     def __call__(self, img):
        ...         generator = thread_safe_generator()
        ...         offset = torch.randint(0, 10, (2,), generator=generator)
        ...         return img[..., offset[0]:, offset[1]:]
    """
    # Lazy import to avoid circular dependency during torch module initialization
    # torch.__init__ loads torch.random early, but torch.utils.data triggers
    # torch.distributed which needs torch to be fully initialized
    from torch.utils.data import get_worker_info

    worker_info: WorkerInfo | None = get_worker_info()
    if (
        worker_info is not None
        and worker_info.worker_method == "thread"
        and worker_info.rng is not None
    ):
        return worker_info.rng.torch_generator
    return None
<|endoftext|>import inspect

import torch
from torch.utils._pytree import register_pytree_node, SequenceKey


__all__ = ["pytree_register_structseq", "all_return_types"]

all_return_types = []

# error: Module has no attribute "_return_types"
return_types = torch._C._return_types  # type: ignore[attr-defined]


def pytree_register_structseq(cls):
    def structseq_flatten(structseq):
        return list(structseq), None

    def structseq_flatten_with_keys(structseq):
        values, context = structseq_flatten(structseq)
        return [(SequenceKey(i), v) for i, v in enumerate(values)], context

    def structseq_unflatten(values, context):
        return cls(values)

    register_pytree_node(
        cls,
        structseq_flatten,
        structseq_unflatten,
        flatten_with_keys_fn=structseq_flatten_with_keys,
    )


for name in dir(return_types):
    if name.startswith("__"):
        continue

    _attr = getattr(return_types, name)
    globals()[name] = _attr

    if not name.startswith("_"):
        # pyrefly: ignore [unresolvable-dunder-all]
        __all__.append(name)
        all_return_types.append(_attr)

    # Today everything in torch.return_types is a structseq, aka a "namedtuple"-like
    # thing defined by the Python C-API. We're going to need to modify this when that
    # is no longer the case.
    # NB: I don't know how to check that something is a "structseq" so we do a fuzzy
    # check for tuple
    if inspect.isclass(_attr) and issubclass(_attr, tuple):
        pytree_register_structseq(_attr)
<|endoftext|>import types
from typing import Any

import torch._C


class _ClassNamespace(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__("torch.classes" + name)
        self.name = name

    def __getattr__(self, attr: str) -> Any:
        proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)
        if proxy is None:
            raise RuntimeError(f"Class {self.name}.{attr} not registered!")
        return proxy


class _Classes(types.ModuleType):
    __file__ = "_classes.py"

    def __init__(self) -> None:
        super().__init__("torch.classes")

    def __getattr__(self, name: str) -> _ClassNamespace:
        namespace = _ClassNamespace(name)
        setattr(self, name, namespace)
        return namespace

    @property
    def loaded_libraries(self) -> Any:
        return torch.ops.loaded_libraries

    def load_library(self, path: str) -> None:
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom classes with the PyTorch JIT runtime. This allows dynamically
        loading custom classes. For this, you should compile your class
        and the static registration code into a shared library object, and then
        call ``torch.classes.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.classes.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        torch.ops.load_library(path)


# The classes "namespace"
classes = _Classes()
<|endoftext|>"""Adds docstrings to torch.Size functions"""

import torch._C
from torch._C import _add_docstr as add_docstr


def add_docstr_all(method: str, docstr: str) -> None:
    add_docstr(getattr(torch._C.Size, method), docstr)


add_docstr_all(
    "numel",
    """
numel() -> int

Returns the number of elements a :class:`torch.Tensor` with the given size would contain.

More formally, for a tensor ``x = tensor.ones(10, 10)`` with size ``s = torch.Size([10, 10])``,
``x.numel() == x.size().numel() == s.numel() == 100`` holds true.

Example::

    >>> x=torch.ones(10, 10)
    >>> s=x.size()
    >>> s
    torch.Size([10, 10])
    >>> s.numel()
    100
    >>> x.numel() == s.numel()
    True


.. warning::

    This function does not return the number of dimensions described by :class:`torch.Size`, but instead the number
    of elements a :class:`torch.Tensor` with that size would contain.

""",
)
<|endoftext|># mypy: allow-untyped-defs
import copyreg
import enum
import functools
import itertools
import warnings
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from numbers import Number
from typing import Any, cast, Concatenate, TypeVar, Union
from typing_extensions import ParamSpec

import torch
import torch._C as _C
from torch.overrides import (
    get_default_nowrap_functions,
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)


_P = ParamSpec("_P")
_TensorLike = TypeVar("_TensorLike", bound=_C.TensorBase)


def _handle_torch_function_and_wrap_type_error_to_not_implemented(
    f: Callable[Concatenate[_TensorLike, _P], "Tensor"],
) -> Callable[Concatenate[_TensorLike, _P], "Tensor"]:
    @functools.wraps(f)
    def wrapped(self: _TensorLike, *args: _P.args, **kwargs: _P.kwargs) -> "Tensor":
        try:
            # See https://github.com/pytorch/pytorch/issues/75462
            sargs = self, *args
            if has_torch_function(sargs):
                return handle_torch_function(wrapped, sargs, self, *args, **kwargs)
            return f(self, *args, **kwargs)
        except TypeError:
            return NotImplemented

    return wrapped


# Should not be used, this is kept only for BC of loading old serialized Tensor subclasses
def _rebuild_from_type(func, type, args, dict):
    if type is Tensor:
        return func(*args)

    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret


def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)
    # Tensor does define __setstate__ even though it doesn't define
    # __getstate__. So only use __setstate__ if it is NOT the one defined
    # on Tensor
    if (
        getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
        is not Tensor.__setstate__
    ):
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret


def _dtype_to_typestr(dtype):
    # CUDA devices are little-endian and tensors are stored in native byte
    # order. 1-byte entries are endian-agnostic.
    return {
        torch.complex64: "<c8",
        torch.complex128: "<c16",
        torch.bfloat16: "<V2",  # Same as ml_dtypes.bfloat16.dtype.str.
        torch.float16: "<f2",
        torch.float32: "<f4",
        torch.float64: "<f8",
        torch.uint8: "|u1",
        torch.int8: "|i1",
        torch.uint16: "<u2",
        torch.int16: "<i2",
        torch.uint32: "<u4",
        torch.int32: "<i4",
        torch.uint64: "<u8",
        torch.int64: "<i8",
        torch.bool: "|b1",
    }[dtype]


# NB: If you subclass Tensor, and want to share the subclassed class
# across processes, you must also update torch/multiprocessing/reductions.py
# to define a ForkingPickler serialization mode for the class.
#
# NB: If you add a new method to Tensor, you must update
# torch/_C/__init__.pyi.in to add a type annotation for your method;
# otherwise, it will not show up in autocomplete.
class Tensor(torch._C.TensorBase):
    _is_param: bool
    # pyrefly: ignore [missing-attribute]
    __dlpack_c_exchange_api__: object = torch._C._dlpack_exchange_api()

    def _clear_non_serializable_cached_data(self):
        r"""Clears any data cached in the tensor's ``__dict__`` that would prevent the tensor
        from being serialized.

        For example, subclasses with custom dispatched sizes / strides cache this info in
        non-serializable PyCapsules within the ``__dict__``, and this must be cleared out for
        serialization to function.

        Any subclass that overrides this MUST call ``super()._clear_non_serializable_cached_data().``
        Additional data cleared within the override must be able to be re-cached transparently
        to avoid breaking subclass functionality.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor._clear_non_serializable_cached_data, (self,), self
            )
        # NB: Wrapper subclasses that implement custom-dispatched sizes / strides cache
        # this info via non-serializable PyCapsules.
        CACHED_SIZES_STRIDES_KEYS = [
            "_sym_sizes_capsule",
            "_sym_sizes_capsule_len",
            "_sym_strides_capsule",
            "_sym_strides_capsule_len",
        ]
        for key in CACHED_SIZES_STRIDES_KEYS:
            self.__dict__.pop(key, None)

    def __deepcopy__(self, memo):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__deepcopy__, (self,), self, memo)
        if not self.is_leaf:
            raise RuntimeError(
                "Only Tensors created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment.  "
                "If you were attempting to deepcopy a module, this may be because "
                "of a torch.nn.utils.weight_norm usage, "
                "see https://github.com/pytorch/pytorch/pull/103001"
            )
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            # TODO: skipping storage copy is wrong for meta, as meta
            # does accurate alias tracking; however, the code below
            # doesn't work because of
            # https://github.com/pytorch/pytorch/issues/47442
            # Update the test in test_serialization if you remove 'meta' from here
            if (
                self.is_sparse
                or self.device.type
                in ["lazy", "xla", "mtia", "mps", "maia", "meta", "ipu"]
                or (
                    not torch._C._has_storage(self)
                    and self.device.type == torch._C._get_privateuse1_backend_name()
                )
                or (type(self) is not Tensor and self.data_ptr() == 0)
            ):
                new_tensor = self.clone()
                if type(new_tensor) is not type(self):
                    raise RuntimeError(
                        "The default implementation of __deepcopy__() for wrapper subclasses "
                        "only works for subclass types that implement clone() and for which "
                        "cloning returns another instance of the same subclass. You should either "
                        "properly implement clone() for your subclass or override __deepcopy__() "
                        "if it is intended behavior for clone() to return an instance of a "
                        "different type."
                    )
            else:
                new_storage = self._typed_storage()._deepcopy(memo)
                if self.is_quantized:
                    # quantizer_params can be different type based on torch attribute
                    quantizer_params: (
                        tuple[torch.qscheme, float, int]
                        | tuple[torch.qscheme, Tensor, Tensor, int]
                    )
                    if self.qscheme() == torch.per_tensor_affine:
                        quantizer_params = (
                            self.qscheme(),
                            self.q_scale(),
                            self.q_zero_point(),
                        )
                    elif self.qscheme() in (
                        torch.per_channel_affine,
                        torch.per_channel_affine_float_qparams,
                    ):
                        quantizer_params = (
                            self.qscheme(),
                            self.q_per_channel_scales(),
                            self.q_per_channel_zero_points(),
                            self.q_per_channel_axis(),
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported qscheme {self.qscheme()} in deepcopy"
                        )
                    # TODO: Once we decide to break serialization FC, no longer
                    # need to wrap with TypedStorage
                    new_tensor = torch._utils._rebuild_qtensor(
                        torch.storage.TypedStorage(
                            wrap_storage=new_storage._untyped_storage,
                            dtype=self.dtype,
                            _internal=True,
                        ),
                        self.storage_offset(),
                        self.size(),
                        self.stride(),
                        quantizer_params,
                        self.requires_grad,
                        self._backward_hooks,
                    )
                    if type(new_tensor) is not type(self):
                        raise RuntimeError(
                            "The default implementation of __deepcopy__() for quantized tensors "
                            "expects the tensor returned by torch._utils._rebuild_qtensor() to "
                            "match the type of the instance being copied. If you encounter this, "
                            "please open an issue on PyTorch's GitHub."
                        )
                else:
                    new_tensor = self.new_empty([])
                    if type(new_tensor) is not type(self):
                        raise RuntimeError(
                            "The default implementation of __deepcopy__() for non-wrapper subclasses "
                            "only works for subclass types that implement new_empty() and for which "
                            "that function returns another instance of the same subclass. You should "
                            "either properly implement new_empty() for your subclass or override "
                            "__deepcopy__() if it is intended behavior for new_empty() to return "
                            "an instance of a different type."
                        )
                    new_tensor.set_(
                        new_storage, self.storage_offset(), self.size(), self.stride()
                    )
                    if self.is_conj():
                        new_tensor = new_tensor.conj_physical()
                    if self.is_neg():
                        new_tensor = new_tensor.neg()
            if self.requires_grad:
                new_tensor.requires_grad_()
            if self.grad is not None:
                new_tensor.grad = self.grad.__deepcopy__(memo)

            if type(self) is not Tensor:
                if type(new_tensor) is not type(self):
                    raise RuntimeError(
                        "Type of deepcopy result does not match the type of the source tensor. "
                        "If you encounter this, please open an issue on PyTorch's GitHub."
                    )

                # Plain Tensors don't have slots
                slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
                for slot in slots_to_save:
                    if hasattr(self, slot):
                        setattr(new_tensor, slot, deepcopy(getattr(self, slot), memo))

            # don't try to deepcopy non-serializable cached data
            self._clear_non_serializable_cached_data()
            new_tensor.__dict__ = deepcopy(self.__dict__, memo)

            memo[id(self)] = new_tensor
            return new_tensor

    def __reduce_ex__(self, proto):
        materialize_fake_tensors = (
            torch.serialization._serialization_tls.materialize_fake_tensors
        )
        state = torch._utils._get_obj_state(self)
        # Ignore all state when using FakeTensor with skip_data(materialize_fake_tensors) because FakeTensor has
        # some state that cannot be pickled
        if (
            # TODO: remove hasattr, it's a hack to support versions of torch that
            # don't have _subclasses
            hasattr(torch, "_subclasses")
            and type(self) is torch._subclasses.fake_tensor.FakeTensor
            and materialize_fake_tensors
        ) or (type(self) is Tensor and not state):
            # Fast path for regular tensor without Python state.
            return self._reduce_ex_internal(proto)
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reduce_ex__, (self,), self, proto)
        func, args = self._reduce_ex_internal(proto)
        # sizes / strides cache needs to be cleared here because it'll just be re-cached
        # if cleared earlier. Note that state references the -actual- tensor dict.
        self._clear_non_serializable_cached_data()
        return (_rebuild_from_type_v2, (func, type(self), args, state))

    def storage(self):
        r"""
        storage() -> torch.TypedStorage

        Returns the underlying :class:`TypedStorage`.

        .. warning::

            :class:`TypedStorage` is deprecated. It will be removed in the future, and
            :class:`UntypedStorage` will be the only storage class. To access the
            :class:`UntypedStorage` directly, use :attr:`Tensor.untyped_storage()`.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage, (self,), self)

        torch.storage._warn_typed_storage_removal(stacklevel=2)
        return self._typed_storage()

    # For internal use only, to avoid raising deprecation warning
    def _typed_storage(self):
        untyped_storage = self.untyped_storage()
        return torch.TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

    def _reduce_ex_internal(self, proto):
        from torch.utils.hooks import warn_if_has_hooks

        # See Note [Don't serialize hooks]
        warn_if_has_hooks(self)
        backward_hooks: dict[Any, Any] = OrderedDict()

        skip_data = torch.serialization._serialization_tls.skip_data
        materialize_fake_tensors = (
            torch.serialization._serialization_tls.materialize_fake_tensors
        )

        if self.device.type in ["xla", "maia", "mtia"] or (
            not torch._C._has_storage(self)
            and self.device.type == torch._C._get_privateuse1_backend_name()
        ):
            if skip_data:
                raise RuntimeError(
                    "Cannot serialize tensors on backends with no storage under skip_data context manager"
                )
            cpu_tensor = self.cpu()
            return (
                torch._utils._rebuild_device_tensor_from_cpu_tensor,
                (cpu_tensor, self.dtype, str(self.device), self.requires_grad),
            )
        if self.device.type == "meta":
            # NB: This implementation BREAKS storage sharing.  Current
            # hypothesis is that no one cares for meta tensors.
            if skip_data:
                warnings.warn(
                    "Serializing tensors on the meta device under skip_data context manager is a no-op",
                    stacklevel=2,
                )
            arg_meta = (
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
            )
            return (torch._utils._rebuild_meta_tensor_no_storage, arg_meta)
        if self.is_quantized:
            if skip_data:
                raise RuntimeError(
                    "Cannot serialize qtensor under skip_data context manager, file an issue if you need this feature"
                )
            # quantizer_params can be different type based on torch attribute
            quantizer_params: (
                tuple[torch.qscheme, float, int] | tuple[Any, Tensor, Tensor, int]
            )
            if self.qscheme() == torch.per_tensor_affine:
                quantizer_params = (
                    torch.per_tensor_affine,
                    self.q_scale(),
                    self.q_zero_point(),
                )
            elif self.qscheme() in (
                torch.per_channel_affine,
                torch.per_channel_affine_float_qparams,
            ):
                # convert scales and zero points to tuple to avoid recursive calls
                # when/if we get multi-axis quantized tensors in the future, the shape
                # is recoverable from the main tensor shape
                quantizer_params = (
                    torch.per_channel_affine,
                    self.q_per_channel_scales(),
                    self.q_per_channel_zero_points(),
                    self.q_per_channel_axis(),
                )
            else:
                raise RuntimeError(
                    f"Serialization is not supported for tensors of type {self.qscheme()}"
                )
            # TODO: Once we decide to break serialization FC, no longer
            # need to wrap with TypedStorage
            args_qtensor = (
                torch.storage.TypedStorage(
                    wrap_storage=self._typed_storage()._untyped_storage,
                    dtype=self.dtype,
                    _internal=True,
                ),
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                quantizer_params,
                self.requires_grad,
                backward_hooks,
            )
            return (torch._utils._rebuild_qtensor, args_qtensor)
        elif self.is_sparse:
            if self.layout == torch.sparse_coo:
                args_sparse = (
                    self.layout,
                    (self._indices(), self._values(), self.size(), self.is_coalesced()),
                )
            else:
                raise NotImplementedError(
                    f"sparse tensor __reduce_ex__ for layout `{self.layout}`"
                )
            return (torch._utils._rebuild_sparse_tensor, args_sparse)
        elif self.layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices, plain_indices = (
                    self.crow_indices(),
                    self.col_indices(),
                )
            else:
                compressed_indices, plain_indices = (
                    self.ccol_indices(),
                    self.row_indices(),
                )
            args_sparse_compressed = (
                self.layout,
                (
                    compressed_indices,
                    plain_indices,
                    self.values(),
                    self.size(),
                ),
            )
            return (torch._utils._rebuild_sparse_tensor, args_sparse_compressed)
        elif self.is_nested:
            if skip_data:
                raise RuntimeError(
                    "Cannot serialize nested tensor under skip_data context manager, file an issue if you need this feature"
                )
            args_nested = (
                # NB: values() currently returns the storage as a buffer in an unsafe way.
                # Ideally, we'd use a private API for this instead. TODO: Switch to this if
                # we ever get around to adding it.
                self.values(),
                self._nested_tensor_size(),
                self._nested_tensor_strides(),
                self._nested_tensor_storage_offsets(),
            )
            return (torch._utils._rebuild_nested_tensor, args_nested)
        elif (
            type(self) is not torch.Tensor
            and type(self).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__
            and (
                isinstance(self, torch._subclasses.functional_tensor.FunctionalTensor)
                or (
                    not torch._subclasses.fake_tensor.is_fake_tensor(self)
                    and self.data_ptr() == 0
                )
            )
        ):
            arg_wrapper_subclass = (
                type(self),
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.storage_offset(),
                self.layout,
                self.device,
                self.requires_grad,
            )
            return (torch._utils._rebuild_wrapper_subclass, arg_wrapper_subclass)
        elif (
            type(self) is not torch.Tensor
            and type(self).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__
            and (
                torch._subclasses.fake_tensor.is_fake_tensor(self)
                and not (skip_data and materialize_fake_tensors)
            )
        ):
            arg_wrapper_subclass = (
                type(self),
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.storage_offset(),
                self.layout,
                self.device,
                self.requires_grad,
            )
            return (torch._utils._rebuild_wrapper_subclass, arg_wrapper_subclass)
        else:
            v3_dtypes = torch.storage._new_dtypes()
            if self.dtype in v3_dtypes:
                rebuild_func = torch._utils._rebuild_tensor_v3
                storage = self.untyped_storage()
            else:
                # TODO: Once we decide to break serialization FC, no longer
                # need to wrap with TypedStorage
                rebuild_func = torch._utils._rebuild_tensor_v2  # type: ignore[assignment]
                storage = torch.storage.TypedStorage(
                    wrap_storage=self._typed_storage()._untyped_storage,
                    dtype=self.dtype,
                    _internal=True,
                )  # type: ignore[assignment]

            # TODO: remove hasattr, it's a hack to support versions of torch that
            # don't have _subclasses
            if (
                hasattr(torch, "_subclasses")
                and torch._subclasses.fake_tensor.is_fake_tensor(self)
                and skip_data
            ):
                storage._fake_device = self.device

            args = (
                storage,
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
                backward_hooks,
            )  # previously was self._backward_hooks

            if isinstance(storage, torch.storage.UntypedStorage):
                args = args + (self.dtype,)  # type: ignore[assignment]

            metadata = torch._utils.get_tensor_metadata(self)
            if metadata:
                args = args + (metadata,)  # type: ignore[assignment]

            return (rebuild_func, args)

    def __setstate__(self, state):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__setstate__, (self,), self, state)
        # Warning: this method is NOT called when you torch.load() a tensor;
        # that is managed by _rebuild_tensor_v2
        if not self.is_leaf:
            raise RuntimeError("__setstate__ can be only called on leaf Tensors")
        if len(state) == 4:
            # legacy serialization of Tensor

            self.set_(*state)
            return
        elif len(state) == 5:
            # legacy serialization of Variable
            self.data = state[0]
            state = (state[3], state[4], state[2])
        # The setting of _backward_hooks is expected to be a no-op.
        # See Note [Don't serialize hooks]
        self.requires_grad, _, self._backward_hooks = state

    def __repr__(self, *, tensor_contents=None):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.__repr__, (self,), self, tensor_contents=tensor_contents
            )
        # All strings are unicode in Python 3.
        return torch._tensor_str._str(self, tensor_contents=tensor_contents)

    def backward(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ):
        r"""Computes the gradient of current tensor wrt graph leaves.

        The graph is differentiated using the chain rule. If the tensor is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying a ``gradient``.
        It should be a tensor of matching type and shape, that represents
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to zero
        ``.grad`` attributes or set them to ``None`` before calling it.
        See :ref:`Default gradient layouts<default-grad-layouts>`
        for details on the memory layout of accumulated gradients.

        .. note::

            If you run any forward ops, create ``gradient``, and/or call ``backward``
            in a user-specified CUDA stream context, see
            :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

        .. note::

            When ``inputs`` are provided and a given input is not a leaf,
            the current implementation will call its grad_fn (though it is not strictly needed to get this gradients).
            It is an implementation detail on which the user should not rely.
            See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

        Args:
            gradient (Tensor, optional): The gradient of the function
                being differentiated w.r.t. ``self``.
                This argument can be omitted if ``self`` is a scalar. Defaults to ``None``.
            retain_graph (bool, optional): If ``False``, the graph used to compute the grads will be freed;
                If ``True``, it will be retained. The default is ``None``, in which case the value is inferred from ``create_graph``
                (i.e., the graph is retained only when higher-order derivative tracking is requested). Note that in nearly all cases
                setting this option to True is not needed and often can be worked around in a much more efficient way.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``.
            inputs (Sequence[Tensor] or dict[str, Tensor], optional): Inputs w.r.t. which
                the gradient will be accumulated into ``.grad``. All other tensors will be
                ignored. If not provided, the gradient is accumulated into all the leaf
                Tensors that were used to compute the :attr:`tensors`. A dict of tensors
                (e.g. ``dict(model.named_parameters())``) is also accepted.
                Defaults to ``None``.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.backward,
                (self,),
                self,
                gradient=gradient,
                retain_graph=retain_graph,
                create_graph=create_graph,
                inputs=inputs,
            )
        torch.autograd.backward(
            self, gradient, retain_graph, create_graph, inputs=inputs
        )

    def index(self, positions, dims):
        """
        Index a regular tensor by binding specified positions to dims.

        This converts a regular tensor to a first-class tensor by binding
        the specified positional dimensions to Dim objects.

        Args:
            positions: Tuple of dimension positions to bind
            dims: Dim objects or tuple of Dim objects to bind to

        Returns:
            First-class tensor with specified dimensions bound
        """
        # TODO: make it possible to dispatch on positions/dims
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.index,
                (self,),
                self,
                positions,
                dims,
            )

        from functorch.dim import index

        return index(self, positions, dims)

    def register_hook(self, hook):
        r"""Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::

            hook(grad) -> Tensor or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v.grad

             2
             4
             6
            [torch.FloatTensor of size (3,)]

            >>> h.remove()  # removes the hook
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.register_hook, (self,), self, hook)
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)

        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def register_post_accumulate_grad_hook(self, hook):
        r"""Registers a backward hook that runs after grad accumulation.

        The hook will be called after all gradients for a tensor have been accumulated,
        meaning that the .grad field has been updated on that tensor. The post
        accumulate grad hook is ONLY applicable for leaf tensors (tensors without a
        .grad_fn field). Registering this hook on a non-leaf tensor will error!

        The hook should have the following signature::

            hook(param: Tensor) -> None

        Note that, unlike other autograd hooks, this hook operates on the tensor
        that requires grad and not the grad itself. The hook can in-place modify
        and access its Tensor argument, including its .grad field.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks. Since
            this hook runs during the backward pass, it will run in no_grad mode (unless
            create_graph is True). You can use torch.enable_grad() to re-enable autograd
            within the hook if you need it.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> lr = 0.01
            >>> # simulate a simple SGD update
            >>> h = v.register_post_accumulate_grad_hook(lambda p: p.add_(p.grad, alpha=-lr))
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v
            tensor([-0.0100, -0.0200, -0.0300], requires_grad=True)

            >>> h.remove()  # removes the hook
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.register_post_accumulate_grad_hook, (self,), self, hook
            )
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )
        if self.grad_fn is not None:
            raise RuntimeError(
                "post accumulate grad hooks cannot be registered on non-leaf tensors"
            )
        if self._post_accumulate_grad_hooks is None:
            self._post_accumulate_grad_hooks: dict[Any, Any] = (
                # pyrefly: ignore [bad-assignment]
                OrderedDict()
            )

        from torch.utils.hooks import RemovableHandle

        handle = RemovableHandle(self._post_accumulate_grad_hooks)
        self._post_accumulate_grad_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        def trim(str):
            return "\n".join([line.strip() for line in str.split("\n")])

        raise RuntimeError(
            trim(
                r"""reinforce() was removed.
            Use torch.distributions instead.
            See https://pytorch.org/docs/main/distributions.html

            Instead of:

            probs = policy_network(state)
            action = probs.multinomial()
            next_state, reward = env.step(action)
            action.reinforce(reward)
            action.backward()

            Use:

            probs = policy_network(state)
            # NOTE: categorical is equivalent to what used to be called multinomial
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            next_state, reward = env.step(action)
            loss = -m.log_prob(action) * reward
            loss.backward()
        """
            )
        )

    detach = _C._add_docstr(
        _C.TensorBase.detach,
        r"""
    Returns a new Tensor, detached from the current graph.

    The result will never require gradient.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.

    .. note::

      Returned Tensor shares the same storage with the original one.
      In-place modifications on either of them will be seen, and may trigger
      errors in correctness checks.
    """,
    )

    detach_ = _C._add_docstr(
        _C.TensorBase.detach_,
        r"""
    Detaches the Tensor from the graph that created it, making it a leaf.
    Views cannot be detached in-place.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.
    """,
    )

    def is_shared(self):
        r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.is_shared, (self,), self)
        return self._typed_storage()._is_shared()

    def share_memory_(self):
        r"""Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.

        See :meth:`torch.UntypedStorage.share_memory_` for more details.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.share_memory_, (self,), self)
        self._typed_storage()._share_memory_()
        return self

    def module_load(self, other, assign=False):
        r"""Defines how to transform ``other`` when loading it into ``self`` in :meth:`~nn.Module.load_state_dict`.

        Used when :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        It is expected that ``self`` is a parameter or buffer in an ``nn.Module`` and ``other`` is the
        value in the state dictionary with the corresponding key, this method defines
        how ``other`` is remapped before being swapped with ``self`` via
        :func:`~torch.utils.swap_tensors` in :meth:`~nn.Module.load_state_dict`.

        .. note::
            This method should always return a new object that is not ``self`` or ``other``.
            For example, the default implementation returns ``self.copy_(other).detach()``
            if ``assign`` is ``False`` or ``other.detach()`` if ``assign`` is ``True``.

        Args:
            other (Tensor): value in state dict with key corresponding to ``self``
            assign (bool): the assign argument passed to :meth:`nn.Module.load_state_dict`

        """
        if has_torch_function_variadic(self, other):
            return handle_torch_function(
                Tensor.module_load, (self, other), self, other, assign=assign
            )

        if assign:
            return other.detach()
        else:
            return self.copy_(other).detach()

    def __reversed__(self):
        r"""Reverses the tensor along dimension 0."""
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reversed__, (self,), self)
        if self.dim() == 0:
            return self
        else:
            return self.flip(0)

    def norm(
        self,
        p: float | str | None = "fro",
        dim=None,
        keepdim=False,
        dtype=None,
    ):
        r"""See :func:`torch.linalg.norm`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.norm, (self,), self, p=p, dim=dim, keepdim=keepdim, dtype=dtype
            )
        return torch.norm(self, p, dim, keepdim, dtype=dtype)

    def cholesky(self, upper=False):
        from ._linalg_utils import cholesky

        return cholesky(self, upper=upper)

    def solve(self, other):
        from torch._linalg_utils import solve

        return solve(self, other)

    def lstsq(self, other):
        from torch._linalg_utils import lstsq

        return lstsq(self, other)

    def eig(self, eigenvectors=False):
        from torch._linalg_utils import eig

        return eig(self, eigenvectors=eigenvectors)

    def qr(self, some=True):
        from torch._linalg_utils import qr

        return qr(self, some=some)

    def symeig(self, eigenvectors=False):
        from torch._linalg_utils import _symeig

        return _symeig(self, eigenvectors=eigenvectors)

    def lu(self, pivot=True, get_infos=False):
        r"""See :func:`torch.lu`"""
        # If get_infos is True, then we don't need to check for errors and vice versa
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.lu, (self,), self, pivot=pivot, get_infos=get_infos
            )

        LU, pivots, infos = torch._lu_with_info(
            self, pivot=pivot, check_errors=(not get_infos)
        )
        if get_infos:
            return LU, pivots, infos
        else:
            return LU, pivots

    def stft(
        self,
        n_fft: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        window: "Tensor | None" = None,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
