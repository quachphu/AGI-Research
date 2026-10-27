        return len(self._data)
<|endoftext|>"""Record of phased-in incompatible language changes.

Each line is of the form:

    FeatureName = "_Feature(" OptionalRelease "," MandatoryRelease ","
                              CompilerFlag ")"

where, normally, OptionalRelease < MandatoryRelease, and both are 5-tuples
of the same form as sys.version_info:

    (PY_MAJOR_VERSION, # the 2 in 2.1.0a3; an int
     PY_MINOR_VERSION, # the 1; an int
     PY_MICRO_VERSION, # the 0; an int
     PY_RELEASE_LEVEL, # "alpha", "beta", "candidate" or "final"; string
     PY_RELEASE_SERIAL # the 3; an int
    )

OptionalRelease records the first release in which

    from __future__ import FeatureName

was accepted.

In the case of MandatoryReleases that have not yet occurred,
MandatoryRelease predicts the release in which the feature will become part
of the language.

Else MandatoryRelease records when the feature became part of the language;
in releases at or after that, modules no longer need

    from __future__ import FeatureName

to use the feature in question, but may continue to use such imports.

MandatoryRelease may also be None, meaning that a planned feature got
dropped.

Instances of class _Feature have two corresponding methods,
.getOptionalRelease() and .getMandatoryRelease().

CompilerFlag is the (bitfield) flag that should be passed in the fourth
argument to the builtin function compile() to enable the feature in
dynamically compiled code.  This flag is stored in the .compiler_flag
attribute on _Future instances.  These values must match the appropriate
#defines of CO_xxx flags in Include/cpython/compile.h.

No feature line is ever to be deleted from this file.
"""

all_feature_names = [
    "nested_scopes",
    "generators",
    "division",
    "absolute_import",
    "with_statement",
    "print_function",
    "unicode_literals",
    "barry_as_FLUFL",
    "generator_stop",
    "annotations",
]

__all__ = ["all_feature_names"] + all_feature_names

# The CO_xxx symbols are defined here under the same names defined in
# code.h and used by compile.h, so that an editor search will find them here.
# However, they're not exported in __all__, because they don't really belong to
# this module.
CO_NESTED = 0x0010                      # nested_scopes
CO_GENERATOR_ALLOWED = 0                # generators (obsolete, was 0x1000)
CO_FUTURE_DIVISION = 0x20000            # division
CO_FUTURE_ABSOLUTE_IMPORT = 0x40000     # perform absolute imports by default
CO_FUTURE_WITH_STATEMENT = 0x80000      # with statement
CO_FUTURE_PRINT_FUNCTION = 0x100000     # print function
CO_FUTURE_UNICODE_LITERALS = 0x200000   # unicode string literals
CO_FUTURE_BARRY_AS_BDFL = 0x400000
CO_FUTURE_GENERATOR_STOP = 0x800000     # StopIteration becomes RuntimeError in generators
CO_FUTURE_ANNOTATIONS = 0x1000000       # annotations become strings at runtime


class _Feature:

    def __init__(self, optionalRelease, mandatoryRelease, compiler_flag):
        self.optional = optionalRelease
        self.mandatory = mandatoryRelease
        self.compiler_flag = compiler_flag

    def getOptionalRelease(self):
        """Return first release in which this feature was recognized.

        This is a 5-tuple, of the same form as sys.version_info.
        """
        return self.optional

    def getMandatoryRelease(self):
        """Return release in which this feature will become mandatory.

        This is a 5-tuple, of the same form as sys.version_info, or, if
        the feature was dropped, is None.
        """
        return self.mandatory

    def __repr__(self):
        return "_Feature" + repr((self.optional,
                                  self.mandatory,
                                  self.compiler_flag))


nested_scopes = _Feature((2, 1, 0, "beta",  1),
                         (2, 2, 0, "alpha", 0),
                         CO_NESTED)

generators = _Feature((2, 2, 0, "alpha", 1),
                      (2, 3, 0, "final", 0),
                      CO_GENERATOR_ALLOWED)

division = _Feature((2, 2, 0, "alpha", 2),
                    (3, 0, 0, "alpha", 0),
                    CO_FUTURE_DIVISION)

absolute_import = _Feature((2, 5, 0, "alpha", 1),
                           (3, 0, 0, "alpha", 0),
                           CO_FUTURE_ABSOLUTE_IMPORT)

with_statement = _Feature((2, 5, 0, "alpha", 1),
                          (2, 6, 0, "alpha", 0),
                          CO_FUTURE_WITH_STATEMENT)

print_function = _Feature((2, 6, 0, "alpha", 2),
                          (3, 0, 0, "alpha", 0),
                          CO_FUTURE_PRINT_FUNCTION)

unicode_literals = _Feature((2, 6, 0, "alpha", 2),
                            (3, 0, 0, "alpha", 0),
                            CO_FUTURE_UNICODE_LITERALS)

barry_as_FLUFL = _Feature((3, 1, 0, "alpha", 2),
                          (4, 0, 0, "alpha", 0),
                          CO_FUTURE_BARRY_AS_BDFL)

generator_stop = _Feature((3, 5, 0, "beta", 1),
                          (3, 7, 0, "alpha", 0),
                          CO_FUTURE_GENERATOR_STOP)

annotations = _Feature((3, 7, 0, "beta", 1),
                       (3, 11, 0, "alpha", 0),
                       CO_FUTURE_ANNOTATIONS)
<|endoftext|># Copyright 2007 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Abstract Base Classes (ABCs) according to PEP 3119."""


def abstractmethod(funcobj):
    """A decorator indicating abstract methods.

    Requires that the metaclass is ABCMeta or derived from it.  A
    class that has a metaclass derived from ABCMeta cannot be
    instantiated unless all of its abstract methods are overridden.
    The abstract methods can be called using any of the normal
    'super' call mechanisms.  abstractmethod() may be used to declare
    abstract methods for properties and descriptors.

    Usage:

        class C(metaclass=ABCMeta):
            @abstractmethod
            def my_abstract_method(self, ...):
                ...
    """
    funcobj.__isabstractmethod__ = True
    return funcobj


class abstractclassmethod(classmethod):
    """A decorator indicating abstract classmethods.

    Deprecated, use 'classmethod' with 'abstractmethod' instead:

        class C(ABC):
            @classmethod
            @abstractmethod
            def my_abstract_classmethod(cls, ...):
                ...

    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractstaticmethod(staticmethod):
    """A decorator indicating abstract staticmethods.

    Deprecated, use 'staticmethod' with 'abstractmethod' instead:

        class C(ABC):
            @staticmethod
            @abstractmethod
            def my_abstract_staticmethod(...):
                ...

    """

    __isabstractmethod__ = True

    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super().__init__(callable)


class abstractproperty(property):
    """A decorator indicating abstract properties.

    Deprecated, use 'property' with 'abstractmethod' instead:

        class C(ABC):
            @property
            @abstractmethod
            def my_abstract_property(self):
                ...

    """

    __isabstractmethod__ = True


try:
    from _abc import (get_cache_token, _abc_init, _abc_register,
                      _abc_instancecheck, _abc_subclasscheck, _get_dump,
                      _reset_registry, _reset_caches)
except ImportError:
    from _py_abc import ABCMeta, get_cache_token
    ABCMeta.__module__ = 'abc'
else:
    class ABCMeta(type):
        """Metaclass for defining Abstract Base Classes (ABCs).

        Use this metaclass to create an ABC.  An ABC can be subclassed
        directly, and then acts as a mix-in class.  You can also register
        unrelated concrete classes (even built-in classes) and unrelated
        ABCs as 'virtual subclasses' -- these and their descendants will
        be considered subclasses of the registering ABC by the built-in
        issubclass() function, but the registering ABC won't show up in
        their MRO (Method Resolution Order) nor will method
        implementations defined by the registering ABC be callable (not
        even via super()).
        """
        def __new__(mcls, name, bases, namespace, **kwargs):
            cls = super().__new__(mcls, name, bases, namespace, **kwargs)
            _abc_init(cls)
            return cls

        def register(cls, subclass):
            """Register a virtual subclass of an ABC.

            Returns the subclass, to allow usage as a class decorator.
            """
            return _abc_register(cls, subclass)

        def __instancecheck__(cls, instance):
            """Override for isinstance(instance, cls)."""
            return _abc_instancecheck(cls, instance)

        def __subclasscheck__(cls, subclass):
            """Override for issubclass(subclass, cls)."""
            return _abc_subclasscheck(cls, subclass)

        def _dump_registry(cls, file=None):
            """Debug helper to print the ABC registry."""
            print(f"Class: {cls.__module__}.{cls.__qualname__}", file=file)
            print(f"Inv. counter: {get_cache_token()}", file=file)
            (_abc_registry, _abc_cache, _abc_negative_cache,
             _abc_negative_cache_version) = _get_dump(cls)
            print(f"_abc_registry: {_abc_registry!r}", file=file)
            print(f"_abc_cache: {_abc_cache!r}", file=file)
            print(f"_abc_negative_cache: {_abc_negative_cache!r}", file=file)
            print(f"_abc_negative_cache_version: {_abc_negative_cache_version!r}",
                  file=file)

        def _abc_registry_clear(cls):
            """Clear the registry (for debugging or testing)."""
            _reset_registry(cls)

        def _abc_caches_clear(cls):
            """Clear the caches (for debugging or testing)."""
            _reset_caches(cls)


def update_abstractmethods(cls):
    """Recalculate the set of abstract methods of an abstract class.

    If a class has had one of its abstract methods implemented after the
    class was created, the method will not be considered implemented until
    this function is called. Alternatively, if a new abstract method has been
    added to the class, it will only be considered an abstract method of the
    class after this function is called.

    This function should be called before any use is made of the class,
    usually in class decorators that add methods to the subject class.

    Returns cls, to allow usage as a class decorator.

    If cls is not an instance of ABCMeta, does nothing.
    """
    if not hasattr(cls, '__abstractmethods__'):
        # We check for __abstractmethods__ here because cls might by a C
        # implementation or a python implementation (especially during
        # testing), and we want to handle both cases.
        return cls

    abstracts = set()
    # Check the existing abstract methods of the parents, keep only the ones
    # that are not implemented.
    for scls in cls.__bases__:
        for name in getattr(scls, '__abstractmethods__', ()):
            value = getattr(cls, name, None)
            if getattr(value, "__isabstractmethod__", False):
                abstracts.add(name)
    # Also add any other newly added abstract methods.
    for name, value in cls.__dict__.items():
        if getattr(value, "__isabstractmethod__", False):
            abstracts.add(name)
    cls.__abstractmethods__ = frozenset(abstracts)
    return cls


class ABC(metaclass=ABCMeta):
    """Helper class that provides a standard way to create an ABC using
    inheritance.
    """
    __slots__ = ()
<|endoftext|>__all__ = ["TopologicalSorter", "CycleError"]

_NODE_OUT = -1
_NODE_DONE = -2


class _NodeInfo:
    __slots__ = "node", "npredecessors", "successors"

    def __init__(self, node):
        # The node this class is augmenting.
        self.node = node

        # Number of predecessors, generally >= 0. When this value falls to 0,
        # and is returned by get_ready(), this is set to _NODE_OUT and when the
        # node is marked done by a call to done(), set to _NODE_DONE.
        self.npredecessors = 0

        # List of successor nodes. The list can contain duplicated elements as
        # long as they're all reflected in the successor's npredecessors attribute.
        self.successors = []


class CycleError(ValueError):
    """Subclass of ValueError raised by TopologicalSorter.prepare if cycles
    exist in the working graph.

    If multiple cycles exist, only one undefined choice among them will be reported
    and included in the exception. The detected cycle can be accessed via the second
    element in the *args* attribute of the exception instance and consists in a list
    of nodes, such that each node is, in the graph, an immediate predecessor of the
    next node in the list. In the reported list, the first and the last node will be
    the same, to make it clear that it is cyclic.
    """

    pass


class TopologicalSorter:
    """Provides functionality to topologically sort a graph of hashable nodes"""

    def __init__(self, graph=None):
        self._node2info = {}
        self._ready_nodes = None
        self._npassedout = 0
        self._nfinished = 0

        if graph is not None:
            for node, predecessors in graph.items():
                self.add(node, *predecessors)

    def _get_nodeinfo(self, node):
        if (result := self._node2info.get(node)) is None:
            self._node2info[node] = result = _NodeInfo(node)
        return result

    def add(self, node, *predecessors):
        """Add a new node and its predecessors to the graph.

        Both the *node* and all elements in *predecessors* must be hashable.

        If called multiple times with the same node argument, the set of dependencies
        will be the union of all dependencies passed in.

        It is possible to add a node with no dependencies (*predecessors* is not provided)
        as well as provide a dependency twice. If a node that has not been provided before
        is included among *predecessors* it will be automatically added to the graph with
        no predecessors of its own.

        Raises ValueError if called after "prepare".
        """
        if self._ready_nodes is not None:
            raise ValueError("Nodes cannot be added after a call to prepare()")

        # Create the node -> predecessor edges
        nodeinfo = self._get_nodeinfo(node)
        nodeinfo.npredecessors += len(predecessors)

        # Create the predecessor -> node edges
        for pred in predecessors:
            pred_info = self._get_nodeinfo(pred)
            pred_info.successors.append(node)

    def prepare(self):
        """Mark the graph as finished and check for cycles in the graph.

        If any cycle is detected, "CycleError" will be raised, but "get_ready" can
        still be used to obtain as many nodes as possible until cycles block more
        progress. After a call to this function, the graph cannot be modified and
        therefore no more nodes can be added using "add".
        """
        if self._ready_nodes is not None:
            raise ValueError("cannot prepare() more than once")

        self._ready_nodes = [
            i.node for i in self._node2info.values() if i.npredecessors == 0
        ]
        # ready_nodes is set before we look for cycles on purpose:
        # if the user wants to catch the CycleError, that's fine,
        # they can continue using the instance to grab as many
        # nodes as possible before cycles block more progress
        cycle = self._find_cycle()
        if cycle:
            raise CycleError(f"nodes are in a cycle", cycle)

    def get_ready(self):
        """Return a tuple of all the nodes that are ready.

        Initially it returns all nodes with no predecessors; once those are marked
        as processed by calling "done", further calls will return all new nodes that
        have all their predecessors already processed. Once no more progress can be made,
        empty tuples are returned.

        Raises ValueError if called without calling "prepare" previously.
        """
        if self._ready_nodes is None:
            raise ValueError("prepare() must be called first")

        # Get the nodes that are ready and mark them
        result = tuple(self._ready_nodes)
        n2i = self._node2info
        for node in result:
            n2i[node].npredecessors = _NODE_OUT

        # Clean the list of nodes that are ready and update
        # the counter of nodes that we have returned.
        self._ready_nodes.clear()
        self._npassedout += len(result)

        return result

    def is_active(self):
        """Return ``True`` if more progress can be made and ``False`` otherwise.

        Progress can be made if cycles do not block the resolution and either there
        are still nodes ready that haven't yet been returned by "get_ready" or the
        number of nodes marked "done" is less than the number that have been returned
        by "get_ready".

        Raises ValueError if called without calling "prepare" previously.
        """
        if self._ready_nodes is None:
            raise ValueError("prepare() must be called first")
        return self._nfinished < self._npassedout or bool(self._ready_nodes)

    def __bool__(self):
        return self.is_active()

    def done(self, *nodes):
        """Marks a set of nodes returned by "get_ready" as processed.

        This method unblocks any successor of each node in *nodes* for being returned
        in the future by a call to "get_ready".

        Raises :exec:`ValueError` if any node in *nodes* has already been marked as
        processed by a previous call to this method, if a node was not added to the
        graph by using "add" or if called without calling "prepare" previously or if
        node has not yet been returned by "get_ready".
        """

        if self._ready_nodes is None:
            raise ValueError("prepare() must be called first")

        n2i = self._node2info

        for node in nodes:

            # Check if we know about this node (it was added previously using add()
            if (nodeinfo := n2i.get(node)) is None:
                raise ValueError(f"node {node!r} was not added using add()")

            # If the node has not being returned (marked as ready) previously, inform the user.
            stat = nodeinfo.npredecessors
            if stat != _NODE_OUT:
                if stat >= 0:
                    raise ValueError(
                        f"node {node!r} was not passed out (still not ready)"
                    )
                elif stat == _NODE_DONE:
                    raise ValueError(f"node {node!r} was already marked done")
                else:
                    assert False, f"node {node!r}: unknown status {stat}"

            # Mark the node as processed
            nodeinfo.npredecessors = _NODE_DONE

            # Go to all the successors and reduce the number of predecessors, collecting all the ones
            # that are ready to be returned in the next get_ready() call.
            for successor in nodeinfo.successors:
                successor_info = n2i[successor]
                successor_info.npredecessors -= 1
                if successor_info.npredecessors == 0:
                    self._ready_nodes.append(successor)
            self._nfinished += 1

    def _find_cycle(self):
        n2i = self._node2info
        stack = []
        itstack = []
        seen = set()
        node2stacki = {}

        for node in n2i:
            if node in seen:
                continue

            while True:
                if node in seen:
                    # If we have seen already the node and is in the
                    # current stack we have found a cycle.
                    if node in node2stacki:
                        return stack[node2stacki[node] :] + [node]
                    # else go on to get next successor
                else:
                    seen.add(node)
                    itstack.append(iter(n2i[node].successors).__next__)
                    node2stacki[node] = len(stack)
                    stack.append(node)

                # Backtrack to the topmost stack entry with
                # at least another successor.
                while stack:
                    try:
                        node = itstack[-1]()
                        break
                    except StopIteration:
                        del node2stacki[stack.pop()]
                        itstack.pop()
                else:
                    break
        return None

    def static_order(self):
        """Returns an iterable of nodes in a topological order.

        The particular order that is returned may depend on the specific
        order in which the items were inserted in the graph.

        Using this method does not require to call "prepare" or "done". If any
        cycle is detected, :exc:`CycleError` will be raised.
        """
        self.prepare()
        while self.is_active():
            node_group = self.get_ready()
            yield from node_group
            self.done(*node_group)
<|endoftext|>#! /usr/bin/env python3
#
# Class for profiling python code. rev 1.0  6/2/94
#
# Written by James Roskind
# Based on prior profile module by Sjoerd Mullender...
#   which was hacked somewhat by: Guido van Rossum

"""Class for profiling Python code."""

# Copyright Disney Enterprises, Inc.  All Rights Reserved.
# Licensed to PSF under a Contributor Agreement
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.  See the License for the specific language
# governing permissions and limitations under the License.


import io
import sys
import time
import marshal

__all__ = ["run", "runctx", "Profile"]

# Sample timer for use with
#i_count = 0
#def integer_timer():
#       global i_count
#       i_count = i_count + 1
#       return i_count
#itimes = integer_timer # replace with C coded timer returning integers

class _Utils:
    """Support class for utility functions which are shared by
    profile.py and cProfile.py modules.
    Not supposed to be used directly.
    """

    def __init__(self, profiler):
        self.profiler = profiler

    def run(self, statement, filename, sort):
        prof = self.profiler()
        try:
            prof.run(statement)
        except SystemExit:
            pass
        finally:
            self._show(prof, filename, sort)

    def runctx(self, statement, globals, locals, filename, sort):
        prof = self.profiler()
        try:
            prof.runctx(statement, globals, locals)
        except SystemExit:
            pass
        finally:
            self._show(prof, filename, sort)

    def _show(self, prof, filename, sort):
        if filename is not None:
            prof.dump_stats(filename)
        else:
            prof.print_stats(sort)


#**************************************************************************
# The following are the static member functions for the profiler class
# Note that an instance of Profile() is *not* needed to call them.
#**************************************************************************

def run(statement, filename=None, sort=-1):
    """Run statement under profiler optionally saving results in filename

    This function takes a single argument that can be passed to the
    "exec" statement, and an optional file name.  In all cases this
    routine attempts to "exec" its first argument and gather profiling
    statistics from the execution. If no file name is present, then this
    function automatically prints a simple profiling report, sorted by the
    standard name string (file/line/function-name) that is presented in
    each line.
    """
    return _Utils(Profile).run(statement, filename, sort)

def runctx(statement, globals, locals, filename=None, sort=-1):
    """Run statement under profiler, supplying your own globals and locals,
    optionally saving results in filename.

    statement and filename have the same semantics as profile.run
    """
    return _Utils(Profile).runctx(statement, globals, locals, filename, sort)


class Profile:
    """Profiler class.

    self.cur is always a tuple.  Each such tuple corresponds to a stack
    frame that is currently active (self.cur[-2]).  The following are the
    definitions of its members.  We use this external "parallel stack" to
    avoid contaminating the program that we are profiling. (old profiler
    used to write into the frames local dictionary!!) Derived classes
    can change the definition of some entries, as long as they leave
    [-2:] intact (frame and previous tuple).  In case an internal error is
    detected, the -3 element is used as the function name.

    [ 0] = Time that needs to be charged to the parent frame's function.
           It is used so that a function call will not have to access the
           timing data for the parent frame.
    [ 1] = Total time spent in this frame's function, excluding time in
           subfunctions (this latter is tallied in cur[2]).
    [ 2] = Total time spent in subfunctions, excluding time executing the
           frame's function (this latter is tallied in cur[1]).
    [-3] = Name of the function that corresponds to this frame.
    [-2] = Actual frame that we correspond to (used to sync exception handling).
    [-1] = Our parent 6-tuple (corresponds to frame.f_back).

    Timing data for each function is stored as a 5-tuple in the dictionary
    self.timings[].  The index is always the name stored in self.cur[-3].
    The following are the definitions of the members:

    [0] = The number of times this function was called, not counting direct
          or indirect recursion,
    [1] = Number of times this function appears on the stack, minus one
    [2] = Total time spent internal to this function
    [3] = Cumulative time that this function was present on the stack.  In
          non-recursive functions, this is the total execution time from start
          to finish of each invocation of a function, including time spent in
          all subfunctions.
    [4] = A dictionary indicating for each function name, the number of times
          it was called by us.
    """

    bias = 0  # calibration constant

    def __init__(self, timer=None, bias=None):
        self.timings = {}
        self.cur = None
        self.cmd = ""
        self.c_func_name = ""

        if bias is None:
            bias = self.bias
        self.bias = bias     # Materialize in local dict for lookup speed.

        if not timer:
            self.timer = self.get_time = time.process_time
            self.dispatcher = self.trace_dispatch_i
        else:
            self.timer = timer
            t = self.timer() # test out timer function
            try:
                length = len(t)
            except TypeError:
                self.get_time = timer
                self.dispatcher = self.trace_dispatch_i
            else:
                if length == 2:
                    self.dispatcher = self.trace_dispatch
                else:
                    self.dispatcher = self.trace_dispatch_l
                # This get_time() implementation needs to be defined
                # here to capture the passed-in timer in the parameter
                # list (for performance).  Note that we can't assume
                # the timer() result contains two values in all
                # cases.
                def get_time_timer(timer=timer, sum=sum):
                    return sum(timer())
                self.get_time = get_time_timer
        self.t = self.get_time()
        self.simulate_call('profiler')

    # Heavily optimized dispatch routine for time.process_time() timer

    def trace_dispatch(self, frame, event, arg):
        timer = self.timer
        t = timer()
        t = t[0] + t[1] - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame,t):
            t = timer()
            self.t = t[0] + t[1]
        else:
            r = timer()
            self.t = r[0] + r[1] - t # put back unrecorded delta

    # Dispatch routine for best timer program (return = scalar, fastest if
    # an integer but float works too -- and time.process_time() relies on that).

    def trace_dispatch_i(self, frame, event, arg):
        timer = self.timer
        t = timer() - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame, t):
            self.t = timer()
        else:
            self.t = timer() - t  # put back unrecorded delta

    # Dispatch routine for macintosh (timer returns time in ticks of
    # 1/60th second)

    def trace_dispatch_mac(self, frame, event, arg):
        timer = self.timer
        t = timer()/60.0 - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame, t):
            self.t = timer()/60.0
        else:
            self.t = timer()/60.0 - t  # put back unrecorded delta

    # SLOW generic dispatch routine for timer returning lists of numbers

    def trace_dispatch_l(self, frame, event, arg):
        get_time = self.get_time
        t = get_time() - self.t - self.bias

        if event == "c_call":
            self.c_func_name = arg.__name__

        if self.dispatch[event](self, frame, t):
            self.t = get_time()
        else:
            self.t = get_time() - t # put back unrecorded delta

    # In the event handlers, the first 3 elements of self.cur are unpacked
    # into vrbls w/ 3-letter names.  The last two characters are meant to be
    # mnemonic:
    #     _pt  self.cur[0] "parent time"   time to be charged to parent frame
    #     _it  self.cur[1] "internal time" time spent directly in the function
    #     _et  self.cur[2] "external time" time spent in subfunctions

    def trace_dispatch_exception(self, frame, t):
        rpt, rit, ret, rfn, rframe, rcur = self.cur
        if (rframe is not frame) and rcur:
            return self.trace_dispatch_return(rframe, t)
        self.cur = rpt, rit+t, ret, rfn, rframe, rcur
        return 1


    def trace_dispatch_call(self, frame, t):
        if self.cur and frame.f_back is not self.cur[-2]:
            rpt, rit, ret, rfn, rframe, rcur = self.cur
            if not isinstance(rframe, Profile.fake_frame):
                assert rframe.f_back is frame.f_back, ("Bad call", rfn,
                                                       rframe, rframe.f_back,
                                                       frame, frame.f_back)
                self.trace_dispatch_return(rframe, 0)
                assert (self.cur is None or \
                        frame.f_back is self.cur[-2]), ("Bad call",
                                                        self.cur[-3])
        fcode = frame.f_code
        fn = (fcode.co_filename, fcode.co_firstlineno, fcode.co_name)
        self.cur = (t, 0, 0, fn, frame, self.cur)
        timings = self.timings
        if fn in timings:
            cc, ns, tt, ct, callers = timings[fn]
            timings[fn] = cc, ns + 1, tt, ct, callers
        else:
            timings[fn] = 0, 0, 0, 0, {}
        return 1

    def trace_dispatch_c_call (self, frame, t):
        fn = ("", 0, self.c_func_name)
        self.cur = (t, 0, 0, fn, frame, self.cur)
        timings = self.timings
        if fn in timings:
            cc, ns, tt, ct, callers = timings[fn]
            timings[fn] = cc, ns+1, tt, ct, callers
        else:
            timings[fn] = 0, 0, 0, 0, {}
        return 1

    def trace_dispatch_return(self, frame, t):
        if frame is not self.cur[-2]:
            assert frame is self.cur[-2].f_back, ("Bad return", self.cur[-3])
            self.trace_dispatch_return(self.cur[-2], 0)

        # Prefix "r" means part of the Returning or exiting frame.
        # Prefix "p" means part of the Previous or Parent or older frame.

        rpt, rit, ret, rfn, frame, rcur = self.cur
        rit = rit + t
        frame_total = rit + ret

        ppt, pit, pet, pfn, pframe, pcur = rcur
        self.cur = ppt, pit + rpt, pet + frame_total, pfn, pframe, pcur

        timings = self.timings
        cc, ns, tt, ct, callers = timings[rfn]
        if not ns:
            # This is the only occurrence of the function on the stack.
            # Else this is a (directly or indirectly) recursive call, and
            # its cumulative time will get updated when the topmost call to
            # it returns.
            ct = ct + frame_total
            cc = cc + 1

        if pfn in callers:
            callers[pfn] = callers[pfn] + 1  # hack: gather more
            # stats such as the amount of time added to ct courtesy
            # of this specific call, and the contribution to cc
            # courtesy of this call.
        else:
            callers[pfn] = 1

        timings[rfn] = cc, ns - 1, tt + rit, ct, callers

        return 1


    dispatch = {
        "call": trace_dispatch_call,
        "exception": trace_dispatch_exception,
        "return": trace_dispatch_return,
        "c_call": trace_dispatch_c_call,
        "c_exception": trace_dispatch_return,  # the C function returned
        "c_return": trace_dispatch_return,
        }


    # The next few functions play with self.cmd. By carefully preloading
    # our parallel stack, we can force the profiled result to include
    # an arbitrary string as the name of the calling function.
    # We use self.cmd as that string, and the resulting stats look
    # very nice :-).

    def set_cmd(self, cmd):
        if self.cur[-1]: return   # already set
        self.cmd = cmd
        self.simulate_call(cmd)

    class fake_code:
        def __init__(self, filename, line, name):
            self.co_filename = filename
            self.co_line = line
            self.co_name = name
            self.co_firstlineno = 0

        def __repr__(self):
            return repr((self.co_filename, self.co_line, self.co_name))

    class fake_frame:
        def __init__(self, code, prior):
            self.f_code = code
            self.f_back = prior

    def simulate_call(self, name):
        code = self.fake_code('profile', 0, name)
        if self.cur:
            pframe = self.cur[-2]
        else:
            pframe = None
        frame = self.fake_frame(code, pframe)
        self.dispatch['call'](self, frame, 0)

    # collect stats from pending stack, including getting final
    # timings for self.cmd frame.

    def simulate_cmd_complete(self):
        get_time = self.get_time
        t = get_time() - self.t
        while self.cur[-1]:
            # We *can* cause assertion errors here if
            # dispatch_trace_return checks for a frame match!
            self.dispatch['return'](self, self.cur[-2], t)
            t = 0
        self.t = get_time() - t


    def print_stats(self, sort=-1):
        import pstats
        pstats.Stats(self).strip_dirs().sort_stats(sort). \
                  print_stats()

    def dump_stats(self, file):
        with open(file, 'wb') as f:
            self.create_stats()
            marshal.dump(self.stats, f)

    def create_stats(self):
        self.simulate_cmd_complete()
        self.snapshot_stats()

    def snapshot_stats(self):
        self.stats = {}
        for func, (cc, ns, tt, ct, callers) in self.timings.items():
            callers = callers.copy()
            nc = 0
            for callcnt in callers.values():
                nc += callcnt
            self.stats[func] = cc, nc, tt, ct, callers


    # The following two methods can be called by clients to use
    # a profiler to profile a statement, given as a string.

    def run(self, cmd):
        import __main__
        dict = __main__.__dict__
        return self.runctx(cmd, dict, dict)

    def runctx(self, cmd, globals, locals):
        self.set_cmd(cmd)
        sys.setprofile(self.dispatcher)
        try:
            exec(cmd, globals, locals)
        finally:
            sys.setprofile(None)
        return self

    # This method is more useful to profile a single function call.
    def runcall(self, func, /, *args, **kw):
        self.set_cmd(repr(func))
        sys.setprofile(self.dispatcher)
        try:
            return func(*args, **kw)
        finally:
            sys.setprofile(None)


    #******************************************************************
    # The following calculates the overhead for using a profiler.  The
    # problem is that it takes a fair amount of time for the profiler
    # to stop the stopwatch (from the time it receives an event).
    # Similarly, there is a delay from the time that the profiler
    # re-starts the stopwatch before the user's code really gets to
    # continue.  The following code tries to measure the difference on
    # a per-event basis.
    #
    # Note that this difference is only significant if there are a lot of
    # events, and relatively little user code per event.  For example,
    # code with small functions will typically benefit from having the
    # profiler calibrated for the current platform.  This *could* be
    # done on the fly during init() time, but it is not worth the
    # effort.  Also note that if too large a value specified, then
    # execution time on some functions will actually appear as a
    # negative number.  It is *normal* for some functions (with very
    # low call counts) to have such negative stats, even if the
    # calibration figure is "correct."
    #
    # One alternative to profile-time calibration adjustments (i.e.,
    # adding in the magic little delta during each event) is to track
    # more carefully the number of events (and cumulatively, the number
    # of events during sub functions) that are seen.  If this were
    # done, then the arithmetic could be done after the fact (i.e., at
    # display time).  Currently, we track only call/return events.
    # These values can be deduced by examining the callees and callers
    # vectors for each functions.  Hence we *can* almost correct the
    # internal time figure at print time (note that we currently don't
    # track exception event processing counts).  Unfortunately, there
    # is currently no similar information for cumulative sub-function
    # time.  It would not be hard to "get all this info" at profiler
    # time.  Specifically, we would have to extend the tuples to keep
    # counts of this in each frame, and then extend the defs of timing
    # tuples to include the significant two figures. I'm a bit fearful
    # that this additional feature will slow the heavily optimized
    # event/time ratio (i.e., the profiler would run slower, fur a very
    # low "value added" feature.)
    #**************************************************************

    def calibrate(self, m, verbose=0):
        if self.__class__ is not Profile:
            raise TypeError("Subclasses must override .calibrate().")

        saved_bias = self.bias
        self.bias = 0
        try:
            return self._calibrate_inner(m, verbose)
        finally:
            self.bias = saved_bias

    def _calibrate_inner(self, m, verbose):
        get_time = self.get_time

        # Set up a test case to be run with and without profiling.  Include
        # lots of calls, because we're trying to quantify stopwatch overhead.
        # Do not raise any exceptions, though, because we want to know
        # exactly how many profile events are generated (one call event, +
        # one return event, per Python-level call).

        def f1(n):
            for i in range(n):
                x = 1

        def f(m, f1=f1):
            for i in range(m):
                f1(100)

        f(m)    # warm up the cache

        # elapsed_noprofile <- time f(m) takes without profiling.
        t0 = get_time()
        f(m)
        t1 = get_time()
        elapsed_noprofile = t1 - t0
        if verbose:
            print("elapsed time without profiling =", elapsed_noprofile)

        # elapsed_profile <- time f(m) takes with profiling.  The difference
        # is profiling overhead, only some of which the profiler subtracts
        # out on its own.
        p = Profile()
        t0 = get_time()
        p.runctx('f(m)', globals(), locals())
        t1 = get_time()
        elapsed_profile = t1 - t0
        if verbose:
            print("elapsed time with profiling =", elapsed_profile)

        # reported_time <- "CPU seconds" the profiler charged to f and f1.
        total_calls = 0.0
        reported_time = 0.0
        for (filename, line, funcname), (cc, ns, tt, ct, callers) in \
                p.timings.items():
            if funcname in ("f", "f1"):
                total_calls += cc
                reported_time += tt

        if verbose:
            print("'CPU seconds' profiler reported =", reported_time)
            print("total # calls =", total_calls)
        if total_calls != m + 1:
            raise ValueError("internal error: total calls = %d" % total_calls)

        # reported_time - elapsed_noprofile = overhead the profiler wasn't
        # able to measure.  Divide by twice the number of calls (since there
        # are two profiler events per call in this test) to get the hidden
        # overhead per event.
        mean = (reported_time - elapsed_noprofile) / 2.0 / total_calls
        if verbose:
            print("mean stopwatch overhead per profile event =", mean)
        return mean

#****************************************************************************

def main():
    import os
    from optparse import OptionParser

    usage = "profile.py [-o output_file_path] [-s sort] [-m module | scriptfile] [arg] ..."
    parser = OptionParser(usage=usage)
    parser.allow_interspersed_args = False
    parser.add_option('-o', '--outfile', dest="outfile",
        help="Save stats to <outfile>", default=None)
    parser.add_option('-m', dest="module", action="store_true",
        help="Profile a library module.", default=False)
    parser.add_option('-s', '--sort', dest="sort",
        help="Sort order when printing to stdout, based on pstats.Stats class",
        default=-1)

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
            import runpy
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
<|endoftext|>#
# Secret Labs' Regular Expression Engine
#
# various symbols used by the regular expression engine.
# run this script to update the _sre include files!
#
# Copyright (c) 1998-2001 by Secret Labs AB.  All rights reserved.
#
# See the sre.py file for information on usage and redistribution.
#

"""Internal support module for sre"""

# update when constants are added or removed

MAGIC = 20171005

from _sre import MAXREPEAT, MAXGROUPS

# SRE standard exception (access as sre.error)
# should this really be here?

class error(Exception):
    """Exception raised for invalid regular expressions.

    Attributes:

        msg: The unformatted error message
        pattern: The regular expression pattern
        pos: The index in the pattern where compilation failed (may be None)
        lineno: The line corresponding to pos (may be None)
        colno: The column corresponding to pos (may be None)
    """

    __module__ = 're'

    def __init__(self, msg, pattern=None, pos=None):
        self.msg = msg
        self.pattern = pattern
        self.pos = pos
        if pattern is not None and pos is not None:
            msg = '%s at position %d' % (msg, pos)
            if isinstance(pattern, str):
                newline = '\n'
            else:
                newline = b'\n'
            self.lineno = pattern.count(newline, 0, pos) + 1
            self.colno = pos - pattern.rfind(newline, 0, pos)
            if newline in pattern:
                msg = '%s (line %d, column %d)' % (msg, self.lineno, self.colno)
        else:
            self.lineno = self.colno = None
        super().__init__(msg)


class _NamedIntConstant(int):
    def __new__(cls, value, name):
        self = super(_NamedIntConstant, cls).__new__(cls, value)
        self.name = name
        return self

    def __repr__(self):
        return self.name

    __reduce__ = None

MAXREPEAT = _NamedIntConstant(MAXREPEAT, 'MAXREPEAT')

def _makecodes(names):
    names = names.strip().split()
    items = [_NamedIntConstant(i, name) for i, name in enumerate(names)]
    globals().update({item.name: item for item in items})
    return items

# operators
# failure=0 success=1 (just because it looks better that way :-)
OPCODES = _makecodes("""
    FAILURE SUCCESS

    ANY ANY_ALL
    ASSERT ASSERT_NOT
    AT
    BRANCH
    CALL
    CATEGORY
    CHARSET BIGCHARSET
    GROUPREF GROUPREF_EXISTS
    IN
    INFO
    JUMP
    LITERAL
    MARK
    MAX_UNTIL
    MIN_UNTIL
    NOT_LITERAL
    NEGATE
    RANGE
    REPEAT
    REPEAT_ONE
    SUBPATTERN
    MIN_REPEAT_ONE

    GROUPREF_IGNORE
    IN_IGNORE
    LITERAL_IGNORE
    NOT_LITERAL_IGNORE

    GROUPREF_LOC_IGNORE
    IN_LOC_IGNORE
    LITERAL_LOC_IGNORE
    NOT_LITERAL_LOC_IGNORE

    GROUPREF_UNI_IGNORE
    IN_UNI_IGNORE
    LITERAL_UNI_IGNORE
    NOT_LITERAL_UNI_IGNORE
    RANGE_UNI_IGNORE

    MIN_REPEAT MAX_REPEAT
""")
del OPCODES[-2:] # remove MIN_REPEAT and MAX_REPEAT

# positions
ATCODES = _makecodes("""
    AT_BEGINNING AT_BEGINNING_LINE AT_BEGINNING_STRING
    AT_BOUNDARY AT_NON_BOUNDARY
    AT_END AT_END_LINE AT_END_STRING

    AT_LOC_BOUNDARY AT_LOC_NON_BOUNDARY

    AT_UNI_BOUNDARY AT_UNI_NON_BOUNDARY
""")

# categories
CHCODES = _makecodes("""
    CATEGORY_DIGIT CATEGORY_NOT_DIGIT
    CATEGORY_SPACE CATEGORY_NOT_SPACE
    CATEGORY_WORD CATEGORY_NOT_WORD
    CATEGORY_LINEBREAK CATEGORY_NOT_LINEBREAK

    CATEGORY_LOC_WORD CATEGORY_LOC_NOT_WORD

    CATEGORY_UNI_DIGIT CATEGORY_UNI_NOT_DIGIT
    CATEGORY_UNI_SPACE CATEGORY_UNI_NOT_SPACE
    CATEGORY_UNI_WORD CATEGORY_UNI_NOT_WORD
    CATEGORY_UNI_LINEBREAK CATEGORY_UNI_NOT_LINEBREAK
""")


# replacement operations for "ignore case" mode
OP_IGNORE = {
    LITERAL: LITERAL_IGNORE,
    NOT_LITERAL: NOT_LITERAL_IGNORE,
}

OP_LOCALE_IGNORE = {
    LITERAL: LITERAL_LOC_IGNORE,
    NOT_LITERAL: NOT_LITERAL_LOC_IGNORE,
}

OP_UNICODE_IGNORE = {
    LITERAL: LITERAL_UNI_IGNORE,
    NOT_LITERAL: NOT_LITERAL_UNI_IGNORE,
}

AT_MULTILINE = {
    AT_BEGINNING: AT_BEGINNING_LINE,
    AT_END: AT_END_LINE
}

AT_LOCALE = {
    AT_BOUNDARY: AT_LOC_BOUNDARY,
    AT_NON_BOUNDARY: AT_LOC_NON_BOUNDARY
}

AT_UNICODE = {
    AT_BOUNDARY: AT_UNI_BOUNDARY,
    AT_NON_BOUNDARY: AT_UNI_NON_BOUNDARY
}

CH_LOCALE = {
    CATEGORY_DIGIT: CATEGORY_DIGIT,
    CATEGORY_NOT_DIGIT: CATEGORY_NOT_DIGIT,
    CATEGORY_SPACE: CATEGORY_SPACE,
    CATEGORY_NOT_SPACE: CATEGORY_NOT_SPACE,
    CATEGORY_WORD: CATEGORY_LOC_WORD,
    CATEGORY_NOT_WORD: CATEGORY_LOC_NOT_WORD,
    CATEGORY_LINEBREAK: CATEGORY_LINEBREAK,
    CATEGORY_NOT_LINEBREAK: CATEGORY_NOT_LINEBREAK
}

CH_UNICODE = {
    CATEGORY_DIGIT: CATEGORY_UNI_DIGIT,
    CATEGORY_NOT_DIGIT: CATEGORY_UNI_NOT_DIGIT,
    CATEGORY_SPACE: CATEGORY_UNI_SPACE,
    CATEGORY_NOT_SPACE: CATEGORY_UNI_NOT_SPACE,
    CATEGORY_WORD: CATEGORY_UNI_WORD,
    CATEGORY_NOT_WORD: CATEGORY_UNI_NOT_WORD,
    CATEGORY_LINEBREAK: CATEGORY_UNI_LINEBREAK,
    CATEGORY_NOT_LINEBREAK: CATEGORY_UNI_NOT_LINEBREAK
}

# flags
SRE_FLAG_TEMPLATE = 1 # template mode (disable backtracking)
SRE_FLAG_IGNORECASE = 2 # case insensitive
SRE_FLAG_LOCALE = 4 # honour system locale
SRE_FLAG_MULTILINE = 8 # treat target as multiline string
SRE_FLAG_DOTALL = 16 # treat target as a single string
SRE_FLAG_UNICODE = 32 # use unicode "locale"
SRE_FLAG_VERBOSE = 64 # ignore whitespace and comments
SRE_FLAG_DEBUG = 128 # debugging
SRE_FLAG_ASCII = 256 # use ascii "locale"

# flags for INFO primitive
SRE_INFO_PREFIX = 1 # has prefix
SRE_INFO_LITERAL = 2 # entire pattern is literal (given by prefix)
SRE_INFO_CHARSET = 4 # pattern starts with character from given set

if __name__ == "__main__":
    def dump(f, d, prefix):
        items = sorted(d)
        for item in items:
            f.write("#define %s_%s %d\n" % (prefix, item, item))
    with open("sre_constants.h", "w") as f:
        f.write("""\
/*
 * Secret Labs' Regular Expression Engine
 *
 * regular expression matching engine
 *
 * NOTE: This file is generated by sre_constants.py.  If you need
 * to change anything in here, edit sre_constants.py and run it.
 *
 * Copyright (c) 1997-2001 by Secret Labs AB.  All rights reserved.
 *
 * See the _sre.c file for information on usage and redistribution.
 */

""")

        f.write("#define SRE_MAGIC %d\n" % MAGIC)

        dump(f, OPCODES, "SRE_OP")
        dump(f, ATCODES, "SRE")
        dump(f, CHCODES, "SRE")

        f.write("#define SRE_FLAG_TEMPLATE %d\n" % SRE_FLAG_TEMPLATE)
        f.write("#define SRE_FLAG_IGNORECASE %d\n" % SRE_FLAG_IGNORECASE)
        f.write("#define SRE_FLAG_LOCALE %d\n" % SRE_FLAG_LOCALE)
        f.write("#define SRE_FLAG_MULTILINE %d\n" % SRE_FLAG_MULTILINE)
        f.write("#define SRE_FLAG_DOTALL %d\n" % SRE_FLAG_DOTALL)
        f.write("#define SRE_FLAG_UNICODE %d\n" % SRE_FLAG_UNICODE)
        f.write("#define SRE_FLAG_VERBOSE %d\n" % SRE_FLAG_VERBOSE)
        f.write("#define SRE_FLAG_DEBUG %d\n" % SRE_FLAG_DEBUG)
        f.write("#define SRE_FLAG_ASCII %d\n" % SRE_FLAG_ASCII)

        f.write("#define SRE_INFO_PREFIX %d\n" % SRE_INFO_PREFIX)
        f.write("#define SRE_INFO_LITERAL %d\n" % SRE_INFO_LITERAL)
        f.write("#define SRE_INFO_CHARSET %d\n" % SRE_INFO_CHARSET)

    print("done")
<|endoftext|>"""
Define names for built-in types that aren't directly accessible as a builtin.
"""
import sys

# Iterators in Python aren't a matter of type but of protocol.  A large
# and changing number of builtin types implement *some* flavor of
# iterator.  Don't check the type!  Use hasattr to check for both
# "__iter__" and "__next__" attributes instead.

def _f(): pass
FunctionType = type(_f)
LambdaType = type(lambda: None)         # Same as FunctionType
CodeType = type(_f.__code__)
MappingProxyType = type(type.__dict__)
SimpleNamespace = type(sys.implementation)

def _cell_factory():
    a = 1
    def f():
        nonlocal a
    return f.__closure__[0]
CellType = type(_cell_factory())

def _g():
    yield 1
GeneratorType = type(_g())

async def _c(): pass
_c = _c()
CoroutineType = type(_c)
_c.close()  # Prevent ResourceWarning

async def _ag():
    yield
_ag = _ag()
AsyncGeneratorType = type(_ag)

class _C:
    def _m(self): pass
MethodType = type(_C()._m)

BuiltinFunctionType = type(len)
BuiltinMethodType = type([].append)     # Same as BuiltinFunctionType

WrapperDescriptorType = type(object.__init__)
MethodWrapperType = type(object().__str__)
MethodDescriptorType = type(str.join)
ClassMethodDescriptorType = type(dict.__dict__['fromkeys'])

ModuleType = type(sys)

try:
    raise TypeError
except TypeError:
    tb = sys.exc_info()[2]
    TracebackType = type(tb)
    FrameType = type(tb.tb_frame)
    tb = None; del tb

# For Jython, the following two types are identical
GetSetDescriptorType = type(FunctionType.__code__)
MemberDescriptorType = type(FunctionType.__globals__)

del sys, _f, _g, _C, _c, _ag  # Not for export


# Provide a PEP 3115 compliant mechanism for class creation
def new_class(name, bases=(), kwds=None, exec_body=None):
    """Create a class object dynamically using the appropriate metaclass."""
    resolved_bases = resolve_bases(bases)
    meta, ns, kwds = prepare_class(name, resolved_bases, kwds)
    if exec_body is not None:
        exec_body(ns)
    if resolved_bases is not bases:
        ns['__orig_bases__'] = bases
    return meta(name, resolved_bases, ns, **kwds)

def resolve_bases(bases):
    """Resolve MRO entries dynamically as specified by PEP 560."""
    new_bases = list(bases)
    updated = False
    shift = 0
    for i, base in enumerate(bases):
        if isinstance(base, type) and not isinstance(base, GenericAlias):
            continue
        if not hasattr(base, "__mro_entries__"):
            continue
        new_base = base.__mro_entries__(bases)
        updated = True
        if not isinstance(new_base, tuple):
            raise TypeError("__mro_entries__ must return a tuple")
        else:
            new_bases[i+shift:i+shift+1] = new_base
            shift += len(new_base) - 1
    if not updated:
        return bases
    return tuple(new_bases)

def prepare_class(name, bases=(), kwds=None):
    """Call the __prepare__ method of the appropriate metaclass.

    Returns (metaclass, namespace, kwds) as a 3-tuple

    *metaclass* is the appropriate metaclass
    *namespace* is the prepared class namespace
    *kwds* is an updated copy of the passed in kwds argument with any
    'metaclass' entry removed. If no kwds argument is passed in, this will
    be an empty dict.
    """
    if kwds is None:
        kwds = {}
    else:
        kwds = dict(kwds) # Don't alter the provided mapping
    if 'metaclass' in kwds:
        meta = kwds.pop('metaclass')
    else:
        if bases:
            meta = type(bases[0])
        else:
            meta = type
    if isinstance(meta, type):
        # when meta is a type, we first determine the most-derived metaclass
        # instead of invoking the initial candidate directly
        meta = _calculate_meta(meta, bases)
    if hasattr(meta, '__prepare__'):
        ns = meta.__prepare__(name, bases, **kwds)
    else:
        ns = {}
    return meta, ns, kwds

def _calculate_meta(meta, bases):
    """Calculate the most derived metaclass."""
    winner = meta
    for base in bases:
        base_meta = type(base)
        if issubclass(winner, base_meta):
            continue
        if issubclass(base_meta, winner):
            winner = base_meta
            continue
        # else:
        raise TypeError("metaclass conflict: "
                        "the metaclass of a derived class "
                        "must be a (non-strict) subclass "
                        "of the metaclasses of all its bases")
    return winner

class DynamicClassAttribute:
    """Route attribute access on a class to __getattr__.

    This is a descriptor, used to define attributes that act differently when
    accessed through an instance and through a class.  Instance access remains
    normal, but access to an attribute through a class will be routed to the
    class's __getattr__ method; this is done by raising AttributeError.

    This allows one to have properties active on an instance, and have virtual
    attributes on the class with the same name.  (Enum used this between Python
    versions 3.4 - 3.9 .)

    Subclass from this to use a different method of accessing virtual atributes
    and still be treated properly by the inspect module. (Enum uses this since
    Python 3.10 .)

    """
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        # next two lines make DynamicClassAttribute act the same as property
        self.__doc__ = doc or fget.__doc__
        self.overwrite_doc = doc is None
        # support for abstract methods
        self.__isabstractmethod__ = bool(getattr(fget, '__isabstractmethod__', False))

    def __get__(self, instance, ownerclass=None):
        if instance is None:
            if self.__isabstractmethod__:
                return self
            raise AttributeError()
        elif self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(instance)

    def __set__(self, instance, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(instance, value)

    def __delete__(self, instance):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(instance)

    def getter(self, fget):
        fdoc = fget.__doc__ if self.overwrite_doc else None
        result = type(self)(fget, self.fset, self.fdel, fdoc or self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result

    def setter(self, fset):
        result = type(self)(self.fget, fset, self.fdel, self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result

    def deleter(self, fdel):
        result = type(self)(self.fget, self.fset, fdel, self.__doc__)
        result.overwrite_doc = self.overwrite_doc
        return result


class _GeneratorWrapper:
    # TODO: Implement this in C.
    def __init__(self, gen):
        self.__wrapped = gen
        self.__isgen = gen.__class__ is GeneratorType
        self.__name__ = getattr(gen, '__name__', None)
        self.__qualname__ = getattr(gen, '__qualname__', None)
    def send(self, val):
        return self.__wrapped.send(val)
    def throw(self, tp, *rest):
        return self.__wrapped.throw(tp, *rest)
    def close(self):
        return self.__wrapped.close()
    @property
    def gi_code(self):
        return self.__wrapped.gi_code
    @property
    def gi_frame(self):
        return self.__wrapped.gi_frame
    @property
    def gi_running(self):
        return self.__wrapped.gi_running
    @property
    def gi_yieldfrom(self):
        return self.__wrapped.gi_yieldfrom
    cr_code = gi_code
    cr_frame = gi_frame
    cr_running = gi_running
    cr_await = gi_yieldfrom
    def __next__(self):
        return next(self.__wrapped)
    def __iter__(self):
        if self.__isgen:
            return self.__wrapped
        return self
    __await__ = __iter__

def coroutine(func):
    """Convert regular generator function to a coroutine."""

    if not callable(func):
        raise TypeError('types.coroutine() expects a callable')

    if (func.__class__ is FunctionType and
        getattr(func, '__code__', None).__class__ is CodeType):

        co_flags = func.__code__.co_flags

        # Check if 'func' is a coroutine function.
        # (0x180 == CO_COROUTINE | CO_ITERABLE_COROUTINE)
        if co_flags & 0x180:
            return func

        # Check if 'func' is a generator function.
        # (0x20 == CO_GENERATOR)
        if co_flags & 0x20:
            # TODO: Implement this in C.
            co = func.__code__
            # 0x100 == CO_ITERABLE_COROUTINE
            func.__code__ = co.replace(co_flags=co.co_flags | 0x100)
            return func

    # The following code is primarily to support functions that
    # return generator-like objects (for instance generators
    # compiled with Cython).

    # Delay functools and _collections_abc import for speeding up types import.
    import functools
    import _collections_abc
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        coro = func(*args, **kwargs)
        if (coro.__class__ is CoroutineType or
            coro.__class__ is GeneratorType and coro.gi_code.co_flags & 0x100):
            # 'coro' is a native coroutine object or an iterable coroutine
            return coro
        if (isinstance(coro, _collections_abc.Generator) and
            not isinstance(coro, _collections_abc.Coroutine)):
            # 'coro' is either a pure Python generator iterator, or it
            # implements collections.abc.Generator (and does not implement
            # collections.abc.Coroutine).
            return _GeneratorWrapper(coro)
        # 'coro' is either an instance of collections.abc.Coroutine or
        # some other object -- pass it through.
        return coro

    return wrapped

GenericAlias = type(list[int])
UnionType = type(int | str)

EllipsisType = type(Ellipsis)
NoneType = type(None)
NotImplementedType = type(NotImplemented)

__all__ = [n for n in globals() if n[:1] != '_']
<|endoftext|># Wrapper module for _ssl, providing some additional facilities
# implemented in Python.  Written by Bill Janssen.

"""This module provides some more Pythonic support for SSL.

Object types:

  SSLSocket -- subtype of socket.socket which does SSL over the socket

Exceptions:

  SSLError -- exception raised for I/O errors

Functions:

  cert_time_to_seconds -- convert time string used for certificate
                          notBefore and notAfter functions to integer
                          seconds past the Epoch (the time values
                          returned from time.time())

  get_server_certificate (addr, ssl_version, ca_certs, timeout) -- Retrieve the
                          certificate from the server at the specified
                          address and return it as a PEM-encoded string


Integer constants:

SSL_ERROR_ZERO_RETURN
SSL_ERROR_WANT_READ
SSL_ERROR_WANT_WRITE
SSL_ERROR_WANT_X509_LOOKUP
SSL_ERROR_SYSCALL
SSL_ERROR_SSL
SSL_ERROR_WANT_CONNECT

SSL_ERROR_EOF
SSL_ERROR_INVALID_ERROR_CODE

The following group define certificate requirements that one side is
allowing/requiring from the other side:

CERT_NONE - no certificates from the other side are required (or will
            be looked at if provided)
CERT_OPTIONAL - certificates are not required, but if provided will be
                validated, and if validation fails, the connection will
                also fail
CERT_REQUIRED - certificates are required, and will be validated, and
                if validation fails, the connection will also fail

The following constants identify various SSL protocol variants:

PROTOCOL_SSLv2
PROTOCOL_SSLv3
PROTOCOL_SSLv23
PROTOCOL_TLS
PROTOCOL_TLS_CLIENT
PROTOCOL_TLS_SERVER
PROTOCOL_TLSv1
PROTOCOL_TLSv1_1
PROTOCOL_TLSv1_2

The following constants identify various SSL alert message descriptions as per
http://www.iana.org/assignments/tls-parameters/tls-parameters.xml#tls-parameters-6

ALERT_DESCRIPTION_CLOSE_NOTIFY
ALERT_DESCRIPTION_UNEXPECTED_MESSAGE
ALERT_DESCRIPTION_BAD_RECORD_MAC
ALERT_DESCRIPTION_RECORD_OVERFLOW
ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
ALERT_DESCRIPTION_HANDSHAKE_FAILURE
ALERT_DESCRIPTION_BAD_CERTIFICATE
ALERT_DESCRIPTION_UNSUPPORTED_CERTIFICATE
ALERT_DESCRIPTION_CERTIFICATE_REVOKED
ALERT_DESCRIPTION_CERTIFICATE_EXPIRED
ALERT_DESCRIPTION_CERTIFICATE_UNKNOWN
ALERT_DESCRIPTION_ILLEGAL_PARAMETER
ALERT_DESCRIPTION_UNKNOWN_CA
ALERT_DESCRIPTION_ACCESS_DENIED
ALERT_DESCRIPTION_DECODE_ERROR
ALERT_DESCRIPTION_DECRYPT_ERROR
ALERT_DESCRIPTION_PROTOCOL_VERSION
ALERT_DESCRIPTION_INSUFFICIENT_SECURITY
ALERT_DESCRIPTION_INTERNAL_ERROR
ALERT_DESCRIPTION_USER_CANCELLED
ALERT_DESCRIPTION_NO_RENEGOTIATION
ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION
ALERT_DESCRIPTION_CERTIFICATE_UNOBTAINABLE
ALERT_DESCRIPTION_UNRECOGNIZED_NAME
ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY
"""

import sys
import os
from collections import namedtuple
from enum import Enum as _Enum, IntEnum as _IntEnum, IntFlag as _IntFlag

import _ssl             # if we can't import it, let the error propagate

from _ssl import OPENSSL_VERSION_NUMBER, OPENSSL_VERSION_INFO, OPENSSL_VERSION
from _ssl import _SSLContext, MemoryBIO, SSLSession
from _ssl import (
    SSLError, SSLZeroReturnError, SSLWantReadError, SSLWantWriteError,
    SSLSyscallError, SSLEOFError, SSLCertVerificationError
    )
from _ssl import txt2obj as _txt2obj, nid2obj as _nid2obj
from _ssl import RAND_status, RAND_add, RAND_bytes, RAND_pseudo_bytes
try:
    from _ssl import RAND_egd
except ImportError:
    # LibreSSL does not provide RAND_egd
    pass


from _ssl import (
    HAS_SNI, HAS_ECDH, HAS_NPN, HAS_ALPN, HAS_SSLv2, HAS_SSLv3, HAS_TLSv1,
    HAS_TLSv1_1, HAS_TLSv1_2, HAS_TLSv1_3
)
from _ssl import _DEFAULT_CIPHERS, _OPENSSL_API_VERSION


_IntEnum._convert_(
    '_SSLMethod', __name__,
    lambda name: name.startswith('PROTOCOL_') and name != 'PROTOCOL_SSLv23',
    source=_ssl)

_IntFlag._convert_(
    'Options', __name__,
    lambda name: name.startswith('OP_'),
    source=_ssl)

_IntEnum._convert_(
    'AlertDescription', __name__,
    lambda name: name.startswith('ALERT_DESCRIPTION_'),
    source=_ssl)

_IntEnum._convert_(
    'SSLErrorNumber', __name__,
    lambda name: name.startswith('SSL_ERROR_'),
    source=_ssl)

_IntFlag._convert_(
    'VerifyFlags', __name__,
    lambda name: name.startswith('VERIFY_'),
    source=_ssl)

_IntEnum._convert_(
    'VerifyMode', __name__,
    lambda name: name.startswith('CERT_'),
    source=_ssl)

PROTOCOL_SSLv23 = _SSLMethod.PROTOCOL_SSLv23 = _SSLMethod.PROTOCOL_TLS
_PROTOCOL_NAMES = {value: name for name, value in _SSLMethod.__members__.items()}

_SSLv2_IF_EXISTS = getattr(_SSLMethod, 'PROTOCOL_SSLv2', None)


class TLSVersion(_IntEnum):
    MINIMUM_SUPPORTED = _ssl.PROTO_MINIMUM_SUPPORTED
    SSLv3 = _ssl.PROTO_SSLv3
    TLSv1 = _ssl.PROTO_TLSv1
    TLSv1_1 = _ssl.PROTO_TLSv1_1
    TLSv1_2 = _ssl.PROTO_TLSv1_2
    TLSv1_3 = _ssl.PROTO_TLSv1_3
    MAXIMUM_SUPPORTED = _ssl.PROTO_MAXIMUM_SUPPORTED


class _TLSContentType(_IntEnum):
    """Content types (record layer)

    See RFC 8446, section B.1
    """
    CHANGE_CIPHER_SPEC = 20
    ALERT = 21
    HANDSHAKE = 22
    APPLICATION_DATA = 23
    # pseudo content types
    HEADER = 0x100
    INNER_CONTENT_TYPE = 0x101


class _TLSAlertType(_IntEnum):
    """Alert types for TLSContentType.ALERT messages

    See RFC 8466, section B.2
    """
    CLOSE_NOTIFY = 0
    UNEXPECTED_MESSAGE = 10
    BAD_RECORD_MAC = 20
    DECRYPTION_FAILED = 21
    RECORD_OVERFLOW = 22
    DECOMPRESSION_FAILURE = 30
    HANDSHAKE_FAILURE = 40
    NO_CERTIFICATE = 41
    BAD_CERTIFICATE = 42
    UNSUPPORTED_CERTIFICATE = 43
    CERTIFICATE_REVOKED = 44
    CERTIFICATE_EXPIRED = 45
    CERTIFICATE_UNKNOWN = 46
    ILLEGAL_PARAMETER = 47
    UNKNOWN_CA = 48
    ACCESS_DENIED = 49
    DECODE_ERROR = 50
    DECRYPT_ERROR = 51
    EXPORT_RESTRICTION = 60
    PROTOCOL_VERSION = 70
    INSUFFICIENT_SECURITY = 71
    INTERNAL_ERROR = 80
    INAPPROPRIATE_FALLBACK = 86
    USER_CANCELED = 90
    NO_RENEGOTIATION = 100
    MISSING_EXTENSION = 109
    UNSUPPORTED_EXTENSION = 110
    CERTIFICATE_UNOBTAINABLE = 111
    UNRECOGNIZED_NAME = 112
    BAD_CERTIFICATE_STATUS_RESPONSE = 113
    BAD_CERTIFICATE_HASH_VALUE = 114
    UNKNOWN_PSK_IDENTITY = 115
    CERTIFICATE_REQUIRED = 116
    NO_APPLICATION_PROTOCOL = 120


class _TLSMessageType(_IntEnum):
    """Message types (handshake protocol)

    See RFC 8446, section B.3
    """
    HELLO_REQUEST = 0
    CLIENT_HELLO = 1
    SERVER_HELLO = 2
    HELLO_VERIFY_REQUEST = 3
    NEWSESSION_TICKET = 4
    END_OF_EARLY_DATA = 5
    HELLO_RETRY_REQUEST = 6
    ENCRYPTED_EXTENSIONS = 8
    CERTIFICATE = 11
    SERVER_KEY_EXCHANGE = 12
    CERTIFICATE_REQUEST = 13
    SERVER_DONE = 14
    CERTIFICATE_VERIFY = 15
    CLIENT_KEY_EXCHANGE = 16
    FINISHED = 20
    CERTIFICATE_URL = 21
    CERTIFICATE_STATUS = 22
    SUPPLEMENTAL_DATA = 23
    KEY_UPDATE = 24
    NEXT_PROTO = 67
    MESSAGE_HASH = 254
    CHANGE_CIPHER_SPEC = 0x0101


if sys.platform == "win32":
    from _ssl import enum_certificates, enum_crls

from socket import socket, SOCK_STREAM, create_connection
from socket import SOL_SOCKET, SO_TYPE, _GLOBAL_DEFAULT_TIMEOUT
import socket as _socket
import base64        # for DER-to-PEM translation
import errno
import warnings


socket_error = OSError  # keep that public name in module namespace

CHANNEL_BINDING_TYPES = ['tls-unique']

HAS_NEVER_CHECK_COMMON_NAME = hasattr(_ssl, 'HOSTFLAG_NEVER_CHECK_SUBJECT')


_RESTRICTED_SERVER_CIPHERS = _DEFAULT_CIPHERS

CertificateError = SSLCertVerificationError


def _dnsname_match(dn, hostname):
    """Matching according to RFC 6125, section 6.4.3

    - Hostnames are compared lower case.
    - For IDNA, both dn and hostname must be encoded as IDN A-label (ACE).
    - Partial wildcards like 'www*.example.org', multiple wildcards, sole
      wildcard or wildcards in labels other then the left-most label are not
      supported and a CertificateError is raised.
    - A wildcard must match at least one character.
    """
    if not dn:
        return False

    wildcards = dn.count('*')
    # speed up common case w/o wildcards
    if not wildcards:
        return dn.lower() == hostname.lower()

    if wildcards > 1:
        raise CertificateError(
            "too many wildcards in certificate DNS name: {!r}.".format(dn))

    dn_leftmost, sep, dn_remainder = dn.partition('.')

    if '*' in dn_remainder:
        # Only match wildcard in leftmost segment.
        raise CertificateError(
            "wildcard can only be present in the leftmost label: "
            "{!r}.".format(dn))

    if not sep:
        # no right side
        raise CertificateError(
            "sole wildcard without additional labels are not support: "
            "{!r}.".format(dn))

    if dn_leftmost != '*':
        # no partial wildcard matching
        raise CertificateError(
            "partial wildcards in leftmost label are not supported: "
            "{!r}.".format(dn))

    hostname_leftmost, sep, hostname_remainder = hostname.partition('.')
    if not hostname_leftmost or not sep:
        # wildcard must match at least one char
        return False
    return dn_remainder.lower() == hostname_remainder.lower()


def _inet_paton(ipname):
    """Try to convert an IP address to packed binary form

    Supports IPv4 addresses on all platforms and IPv6 on platforms with IPv6
    support.
    """
    # inet_aton() also accepts strings like '1', '127.1', some also trailing
    # data like '127.0.0.1 whatever'.
    try:
        addr = _socket.inet_aton(ipname)
    except OSError:
        # not an IPv4 address
        pass
    else:
        if _socket.inet_ntoa(addr) == ipname:
            # only accept injective ipnames
            return addr
        else:
            # refuse for short IPv4 notation and additional trailing data
            raise ValueError(
                "{!r} is not a quad-dotted IPv4 address.".format(ipname)
            )

    try:
        return _socket.inet_pton(_socket.AF_INET6, ipname)
    except OSError:
        raise ValueError("{!r} is neither an IPv4 nor an IP6 "
                         "address.".format(ipname))
    except AttributeError:
        # AF_INET6 not available
        pass

    raise ValueError("{!r} is not an IPv4 address.".format(ipname))


def _ipaddress_match(cert_ipaddress, host_ip):
    """Exact matching of IP addresses.

    RFC 6125 explicitly doesn't define an algorithm for this
    (section 1.7.2 - "Out of Scope").
    """
    # OpenSSL may add a trailing newline to a subjectAltName's IP address,
    # commonly woth IPv6 addresses. Strip off trailing \n.
    ip = _inet_paton(cert_ipaddress.rstrip())
    return ip == host_ip


def match_hostname(cert, hostname):
    """Verify that *cert* (in decoded format as returned by
    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125
    rules are followed.

    The function matches IP addresses rather than dNSNames if hostname is a
    valid ipaddress string. IPv4 addresses are supported on all platforms.
    IPv6 addresses are supported on platforms with IPv6 support (AF_INET6
    and inet_pton).

    CertificateError is raised on failure. On success, the function
    returns nothing.
    """
    warnings.warn(
        "ssl.match_hostname() is deprecated",
        category=DeprecationWarning,
        stacklevel=2
    )
    if not cert:
        raise ValueError("empty or no certificate, match_hostname needs a "
                         "SSL socket or SSL context with either "
                         "CERT_OPTIONAL or CERT_REQUIRED")
    try:
        host_ip = _inet_paton(hostname)
    except ValueError:
        # Not an IP address (common case)
        host_ip = None
    dnsnames = []
    san = cert.get('subjectAltName', ())
    for key, value in san:
        if key == 'DNS':
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == 'IP Address':
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)
    if not dnsnames:
        # The subject is only checked when there is no dNSName entry
        # in subjectAltName
        for sub in cert.get('subject', ()):
            for key, value in sub:
                # XXX according to RFC 2818, the most specific Common Name
                # must be used.
                if key == 'commonName':
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)
    if len(dnsnames) > 1:
        raise CertificateError("hostname %r "
            "doesn't match either of %s"
            % (hostname, ', '.join(map(repr, dnsnames))))
    elif len(dnsnames) == 1:
        raise CertificateError("hostname %r "
            "doesn't match %r"
            % (hostname, dnsnames[0]))
    else:
        raise CertificateError("no appropriate commonName or "
            "subjectAltName fields were found")


DefaultVerifyPaths = namedtuple("DefaultVerifyPaths",
    "cafile capath openssl_cafile_env openssl_cafile openssl_capath_env "
    "openssl_capath")

def get_default_verify_paths():
    """Return paths to default cafile and capath.
    """
    parts = _ssl.get_default_verify_paths()

    # environment vars shadow paths
    cafile = os.environ.get(parts[0], parts[1])
    capath = os.environ.get(parts[2], parts[3])

    return DefaultVerifyPaths(cafile if os.path.isfile(cafile) else None,
                              capath if os.path.isdir(capath) else None,
                              *parts)


class _ASN1Object(namedtuple("_ASN1Object", "nid shortname longname oid")):
    """ASN.1 object identifier lookup
    """
    __slots__ = ()

    def __new__(cls, oid):
        return super().__new__(cls, *_txt2obj(oid, name=False))

    @classmethod
    def fromnid(cls, nid):
        """Create _ASN1Object from OpenSSL numeric ID
        """
        return super().__new__(cls, *_nid2obj(nid))

    @classmethod
    def fromname(cls, name):
        """Create _ASN1Object from short name, long name or OID
        """
        return super().__new__(cls, *_txt2obj(name, name=True))


class Purpose(_ASN1Object, _Enum):
    """SSLContext purpose flags with X509v3 Extended Key Usage objects
    """
    SERVER_AUTH = '1.3.6.1.5.5.7.3.1'
    CLIENT_AUTH = '1.3.6.1.5.5.7.3.2'


class SSLContext(_SSLContext):
    """An SSLContext holds various SSL-related configuration options and
    data, such as certificates and possibly a private key."""
    _windows_cert_stores = ("CA", "ROOT")

    sslsocket_class = None  # SSLSocket is assigned later.
    sslobject_class = None  # SSLObject is assigned later.

    def __new__(cls, protocol=None, *args, **kwargs):
        if protocol is None:
            warnings.warn(
                "ssl.SSLContext() without protocol argument is deprecated.",
                category=DeprecationWarning,
                stacklevel=2
            )
            protocol = PROTOCOL_TLS
        self = _SSLContext.__new__(cls, protocol)
        return self

    def _encode_hostname(self, hostname):
        if hostname is None:
            return None
        elif isinstance(hostname, str):
            return hostname.encode('idna').decode('ascii')
        else:
            return hostname.decode('ascii')

    def wrap_socket(self, sock, server_side=False,
                    do_handshake_on_connect=True,
                    suppress_ragged_eofs=True,
                    server_hostname=None, session=None):
        # SSLSocket class handles server_hostname encoding before it calls
        # ctx._wrap_socket()
        return self.sslsocket_class._create(
            sock=sock,
            server_side=server_side,
            do_handshake_on_connect=do_handshake_on_connect,
            suppress_ragged_eofs=suppress_ragged_eofs,
            server_hostname=server_hostname,
            context=self,
            session=session
        )

    def wrap_bio(self, incoming, outgoing, server_side=False,
                 server_hostname=None, session=None):
        # Need to encode server_hostname here because _wrap_bio() can only
        # handle ASCII str.
        return self.sslobject_class._create(
            incoming, outgoing, server_side=server_side,
            server_hostname=self._encode_hostname(server_hostname),
            session=session, context=self,
        )

    def set_npn_protocols(self, npn_protocols):
        warnings.warn(
            "ssl NPN is deprecated, use ALPN instead",
            DeprecationWarning,
            stacklevel=2
        )
        protos = bytearray()
        for protocol in npn_protocols:
            b = bytes(protocol, 'ascii')
            if len(b) == 0 or len(b) > 255:
                raise SSLError('NPN protocols must be 1 to 255 in length')
            protos.append(len(b))
            protos.extend(b)

        self._set_npn_protocols(protos)

    def set_servername_callback(self, server_name_callback):
        if server_name_callback is None:
            self.sni_callback = None
        else:
            if not callable(server_name_callback):
                raise TypeError("not a callable object")

            def shim_cb(sslobj, servername, sslctx):
                servername = self._encode_hostname(servername)
                return server_name_callback(sslobj, servername, sslctx)

            self.sni_callback = shim_cb

    def set_alpn_protocols(self, alpn_protocols):
        protos = bytearray()
        for protocol in alpn_protocols:
            b = bytes(protocol, 'ascii')
            if len(b) == 0 or len(b) > 255:
                raise SSLError('ALPN protocols must be 1 to 255 in length')
            protos.append(len(b))
            protos.extend(b)

        self._set_alpn_protocols(protos)

    def _load_windows_store_certs(self, storename, purpose):
        certs = bytearray()
        try:
            for cert, encoding, trust in enum_certificates(storename):
                # CA certs are never PKCS#7 encoded
                if encoding == "x509_asn":
                    if trust is True or purpose.oid in trust:
                        certs.extend(cert)
        except PermissionError:
            warnings.warn("unable to enumerate Windows certificate store")
        if certs:
            self.load_verify_locations(cadata=certs)
        return certs

    def load_default_certs(self, purpose=Purpose.SERVER_AUTH):
        if not isinstance(purpose, _ASN1Object):
            raise TypeError(purpose)
        if sys.platform == "win32":
            for storename in self._windows_cert_stores:
                self._load_windows_store_certs(storename, purpose)
        self.set_default_verify_paths()

    if hasattr(_SSLContext, 'minimum_version'):
        @property
        def minimum_version(self):
            return TLSVersion(super().minimum_version)

        @minimum_version.setter
        def minimum_version(self, value):
            if value == TLSVersion.SSLv3:
                self.options &= ~Options.OP_NO_SSLv3
            super(SSLContext, SSLContext).minimum_version.__set__(self, value)

        @property
        def maximum_version(self):
            return TLSVersion(super().maximum_version)

        @maximum_version.setter
        def maximum_version(self, value):
            super(SSLContext, SSLContext).maximum_version.__set__(self, value)

    @property
    def options(self):
        return Options(super().options)

    @options.setter
    def options(self, value):
        super(SSLContext, SSLContext).options.__set__(self, value)

    if hasattr(_ssl, 'HOSTFLAG_NEVER_CHECK_SUBJECT'):
        @property
        def hostname_checks_common_name(self):
            ncs = self._host_flags & _ssl.HOSTFLAG_NEVER_CHECK_SUBJECT
            return ncs != _ssl.HOSTFLAG_NEVER_CHECK_SUBJECT

        @hostname_checks_common_name.setter
        def hostname_checks_common_name(self, value):
            if value:
                self._host_flags &= ~_ssl.HOSTFLAG_NEVER_CHECK_SUBJECT
            else:
                self._host_flags |= _ssl.HOSTFLAG_NEVER_CHECK_SUBJECT
    else:
        @property
        def hostname_checks_common_name(self):
            return True

    @property
    def _msg_callback(self):
        """TLS message callback

        The message callback provides a debugging hook to analyze TLS
        connections. The callback is called for any TLS protocol message
        (header, handshake, alert, and more), but not for application data.
        Due to technical  limitations, the callback can't be used to filter
        traffic or to abort a connection. Any exception raised in the
        callback is delayed until the handshake, read, or write operation
        has been performed.

        def msg_cb(conn, direction, version, content_type, msg_type, data):
            pass

        conn
            :class:`SSLSocket` or :class:`SSLObject` instance
        direction
            ``read`` or ``write``
        version
            :class:`TLSVersion` enum member or int for unknown version. For a
            frame header, it's the header version.
        content_type
            :class:`_TLSContentType` enum member or int for unsupported
            content type.
        msg_type
            Either a :class:`_TLSContentType` enum number for a header
            message, a :class:`_TLSAlertType` enum member for an alert
            message, a :class:`_TLSMessageType` enum member for other
            messages, or int for unsupported message types.
        data
            Raw, decrypted message content as bytes
        """
        inner = super()._msg_callback
        if inner is not None:
            return inner.user_function
        else:
            return None

    @_msg_callback.setter
    def _msg_callback(self, callback):
        if callback is None:
            super(SSLContext, SSLContext)._msg_callback.__set__(self, None)
            return

        if not hasattr(callback, '__call__'):
            raise TypeError(f"{callback} is not callable.")

        def inner(conn, direction, version, content_type, msg_type, data):
            try:
                version = TLSVersion(version)
            except ValueError:
                pass

            try:
                content_type = _TLSContentType(content_type)
            except ValueError:
                pass

            if content_type == _TLSContentType.HEADER:
                msg_enum = _TLSContentType
            elif content_type == _TLSContentType.ALERT:
                msg_enum = _TLSAlertType
            else:
                msg_enum = _TLSMessageType
            try:
                msg_type = msg_enum(msg_type)
            except ValueError:
                pass

            return callback(conn, direction, version,
                            content_type, msg_type, data)

        inner.user_function = callback

        super(SSLContext, SSLContext)._msg_callback.__set__(self, inner)

    @property
    def protocol(self):
        return _SSLMethod(super().protocol)

    @property
    def verify_flags(self):
        return VerifyFlags(super().verify_flags)

    @verify_flags.setter
    def verify_flags(self, value):
        super(SSLContext, SSLContext).verify_flags.__set__(self, value)

    @property
    def verify_mode(self):
        value = super().verify_mode
        try:
            return VerifyMode(value)
        except ValueError:
            return value

    @verify_mode.setter
    def verify_mode(self, value):
        super(SSLContext, SSLContext).verify_mode.__set__(self, value)


def create_default_context(purpose=Purpose.SERVER_AUTH, *, cafile=None,
                           capath=None, cadata=None):
    """Create a SSLContext object with default settings.

    NOTE: The protocol and settings may change anytime without prior
          deprecation. The values represent a fair balance between maximum
          compatibility and security.
    """
    if not isinstance(purpose, _ASN1Object):
        raise TypeError(purpose)

    # SSLContext sets OP_NO_SSLv2, OP_NO_SSLv3, OP_NO_COMPRESSION,
    # OP_CIPHER_SERVER_PREFERENCE, OP_SINGLE_DH_USE and OP_SINGLE_ECDH_USE
    # by default.
    if purpose == Purpose.SERVER_AUTH:
        # verify certs and host name in client mode
        context = SSLContext(PROTOCOL_TLS_CLIENT)
        context.verify_mode = CERT_REQUIRED
        context.check_hostname = True
    elif purpose == Purpose.CLIENT_AUTH:
        context = SSLContext(PROTOCOL_TLS_SERVER)
    else:
        raise ValueError(purpose)

    if cafile or capath or cadata:
        context.load_verify_locations(cafile, capath, cadata)
    elif context.verify_mode != CERT_NONE:
        # no explicit cafile, capath or cadata but the verify mode is
        # CERT_OPTIONAL or CERT_REQUIRED. Let's try to load default system
        # root CA certificates for the given purpose. This may fail silently.
        context.load_default_certs(purpose)
    # OpenSSL 1.1.1 keylog file
    if hasattr(context, 'keylog_filename'):
        keylogfile = os.environ.get('SSLKEYLOGFILE')
        if keylogfile and not sys.flags.ignore_environment:
            context.keylog_filename = keylogfile
    return context

def _create_unverified_context(protocol=None, *, cert_reqs=CERT_NONE,
                           check_hostname=False, purpose=Purpose.SERVER_AUTH,
                           certfile=None, keyfile=None,
                           cafile=None, capath=None, cadata=None):
    """Create a SSLContext object for Python stdlib modules

    All Python stdlib modules shall use this function to create SSLContext
    objects in order to keep common settings in one place. The configuration
    is less restrict than create_default_context()'s to increase backward
    compatibility.
    """
    if not isinstance(purpose, _ASN1Object):
        raise TypeError(purpose)

    # SSLContext sets OP_NO_SSLv2, OP_NO_SSLv3, OP_NO_COMPRESSION,
    # OP_CIPHER_SERVER_PREFERENCE, OP_SINGLE_DH_USE and OP_SINGLE_ECDH_USE
    # by default.
    if purpose == Purpose.SERVER_AUTH:
        # verify certs and host name in client mode
        if protocol is None:
            protocol = PROTOCOL_TLS_CLIENT
    elif purpose == Purpose.CLIENT_AUTH:
        if protocol is None:
            protocol = PROTOCOL_TLS_SERVER
    else:
        raise ValueError(purpose)

    context = SSLContext(protocol)
    context.check_hostname = check_hostname
    if cert_reqs is not None:
        context.verify_mode = cert_reqs
    if check_hostname:
        context.check_hostname = True

    if keyfile and not certfile:
        raise ValueError("certfile must be specified")
    if certfile or keyfile:
        context.load_cert_chain(certfile, keyfile)

    # load CA root certs
    if cafile or capath or cadata:
        context.load_verify_locations(cafile, capath, cadata)
    elif context.verify_mode != CERT_NONE:
        # no explicit cafile, capath or cadata but the verify mode is
        # CERT_OPTIONAL or CERT_REQUIRED. Let's try to load default system
        # root CA certificates for the given purpose. This may fail silently.
        context.load_default_certs(purpose)
    # OpenSSL 1.1.1 keylog file
    if hasattr(context, 'keylog_filename'):
        keylogfile = os.environ.get('SSLKEYLOGFILE')
        if keylogfile and not sys.flags.ignore_environment:
            context.keylog_filename = keylogfile
    return context

# Used by http.client if no context is explicitly passed.
_create_default_https_context = create_default_context


# Backwards compatibility alias, even though it's not a public name.
_create_stdlib_context = _create_unverified_context


class SSLObject:
    """This class implements an interface on top of a low-level SSL object as
    implemented by OpenSSL. This object captures the state of an SSL connection
    but does not provide any network IO itself. IO needs to be performed
    through separate "BIO" objects which are OpenSSL's IO abstraction layer.

    This class does not have a public constructor. Instances are returned by
    ``SSLContext.wrap_bio``. This class is typically used by framework authors
    that want to implement asynchronous IO for SSL through memory buffers.

    When compared to ``SSLSocket``, this object lacks the following features:

     * Any form of network IO, including methods such as ``recv`` and ``send``.
     * The ``do_handshake_on_connect`` and ``suppress_ragged_eofs`` machinery.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError(
            f"{self.__class__.__name__} does not have a public "
            f"constructor. Instances are returned by SSLContext.wrap_bio()."
        )

    @classmethod
    def _create(cls, incoming, outgoing, server_side=False,
                 server_hostname=None, session=None, context=None):
        self = cls.__new__(cls)
        sslobj = context._wrap_bio(
            incoming, outgoing, server_side=server_side,
            server_hostname=server_hostname,
            owner=self, session=session
        )
        self._sslobj = sslobj
        return self

    @property
    def context(self):
        """The SSLContext that is currently in use."""
        return self._sslobj.context

    @context.setter
    def context(self, ctx):
        self._sslobj.context = ctx

    @property
    def session(self):
        """The SSLSession for client socket."""
        return self._sslobj.session

    @session.setter
    def session(self, session):
        self._sslobj.session = session

    @property
    def session_reused(self):
        """Was the client session reused during handshake"""
        return self._sslobj.session_reused

    @property
    def server_side(self):
        """Whether this is a server-side socket."""
        return self._sslobj.server_side

    @property
    def server_hostname(self):
        """The currently set server hostname (for SNI), or ``None`` if no
        server hostname is set."""
        return self._sslobj.server_hostname

    def read(self, len=1024, buffer=None):
        """Read up to 'len' bytes from the SSL object and return them.

        If 'buffer' is provided, read into this buffer and return the number of
        bytes read.
        """
        if buffer is not None:
            v = self._sslobj.read(len, buffer)
        else:
            v = self._sslobj.read(len)
        return v

    def write(self, data):
        """Write 'data' to the SSL object and return the number of bytes
        written.

        The 'data' argument must support the buffer interface.
        """
        return self._sslobj.write(data)

    def getpeercert(self, binary_form=False):
        """Returns a formatted version of the data in the certificate provided
        by the other end of the SSL channel.

        Return None if no certificate was provided, {} if a certificate was
        provided, but not validated.
        """
        return self._sslobj.getpeercert(binary_form)

    def selected_npn_protocol(self):
        """Return the currently selected NPN protocol as a string, or ``None``
        if a next protocol was not negotiated or if NPN is not supported by one
        of the peers."""
        warnings.warn(
            "ssl NPN is deprecated, use ALPN instead",
            DeprecationWarning,
            stacklevel=2
        )

    def selected_alpn_protocol(self):
        """Return the currently selected ALPN protocol as a string, or ``None``
        if a next protocol was not negotiated or if ALPN is not supported by one
        of the peers."""
        return self._sslobj.selected_alpn_protocol()

    def cipher(self):
        """Return the currently selected cipher as a 3-tuple ``(name,
        ssl_version, secret_bits)``."""
        return self._sslobj.cipher()

    def shared_ciphers(self):
        """Return a list of ciphers shared by the client during the handshake or
        None if this is not a valid server connection.
        """
        return self._sslobj.shared_ciphers()

    def compression(self):
        """Return the current compression algorithm in use, or ``None`` if
        compression was not negotiated or not supported by one of the peers."""
        return self._sslobj.compression()

    def pending(self):
        """Return the number of bytes that can be read immediately."""
        return self._sslobj.pending()

    def do_handshake(self):
        """Start the SSL/TLS handshake."""
        self._sslobj.do_handshake()

    def unwrap(self):
        """Start the SSL shutdown handshake."""
        return self._sslobj.shutdown()

    def get_channel_binding(self, cb_type="tls-unique"):
        """Get channel binding data for current connection.  Raise ValueError
        if the requested `cb_type` is not supported.  Return bytes of the data
        or None if the data is not available (e.g. before the handshake)."""
        return self._sslobj.get_channel_binding(cb_type)

    def version(self):
        """Return a string identifying the protocol version used by the
        current SSL channel. """
        return self._sslobj.version()

    def verify_client_post_handshake(self):
        return self._sslobj.verify_client_post_handshake()


def _sslcopydoc(func):
    """Copy docstring from SSLObject to SSLSocket"""
    func.__doc__ = getattr(SSLObject, func.__name__).__doc__
    return func


class SSLSocket(socket):
    """This class implements a subtype of socket.socket that wraps
    the underlying OS socket in an SSL context when necessary, and
    provides read and write methods over that channel. """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            f"{self.__class__.__name__} does not have a public "
            f"constructor. Instances are returned by "
            f"SSLContext.wrap_socket()."
        )

    @classmethod
    def _create(cls, sock, server_side=False, do_handshake_on_connect=True,
                suppress_ragged_eofs=True, server_hostname=None,
                context=None, session=None):
        if sock.getsockopt(SOL_SOCKET, SO_TYPE) != SOCK_STREAM:
            raise NotImplementedError("only stream sockets are supported")
        if server_side:
            if server_hostname:
                raise ValueError("server_hostname can only be specified "
                                 "in client mode")
            if session is not None:
                raise ValueError("session can only be specified in "
                                 "client mode")
        if context.check_hostname and not server_hostname:
            raise ValueError("check_hostname requires server_hostname")

        kwargs = dict(
            family=sock.family, type=sock.type, proto=sock.proto,
            fileno=sock.fileno()
        )
        self = cls.__new__(cls, **kwargs)
        super(SSLSocket, self).__init__(**kwargs)
        sock_timeout = sock.gettimeout()
        sock.detach()

        self._context = context
        self._session = session
        self._closed = False
        self._sslobj = None
        self.server_side = server_side
        self.server_hostname = context._encode_hostname(server_hostname)
        self.do_handshake_on_connect = do_handshake_on_connect
        self.suppress_ragged_eofs = suppress_ragged_eofs

        # See if we are connected
        try:
            self.getpeername()
        except OSError as e:
            if e.errno != errno.ENOTCONN:
                raise
            connected = False
            blocking = self.getblocking()
            self.setblocking(False)
            try:
                # We are not connected so this is not supposed to block, but
                # testing revealed otherwise on macOS and Windows so we do
                # the non-blocking dance regardless. Our raise when any data
                # is found means consuming the data is harmless.
                notconn_pre_handshake_data = self.recv(1)
            except OSError as e:
                # EINVAL occurs for recv(1) on non-connected on unix sockets.
                if e.errno not in (errno.ENOTCONN, errno.EINVAL):
                    raise
                notconn_pre_handshake_data = b''
            self.setblocking(blocking)
            if notconn_pre_handshake_data:
                # This prevents pending data sent to the socket before it was
                # closed from escaping to the caller who could otherwise
                # presume it came through a successful TLS connection.
                reason = "Closed before TLS handshake with data in recv buffer."
                notconn_pre_handshake_data_error = SSLError(e.errno, reason)
                # Add the SSLError attributes that _ssl.c always adds.
                notconn_pre_handshake_data_error.reason = reason
                notconn_pre_handshake_data_error.library = None
                try:
                    self.close()
                except OSError:
                    pass
                raise notconn_pre_handshake_data_error
        else:
            connected = True

        self.settimeout(sock_timeout)  # Must come after setblocking() calls.
        self._connected = connected
        if connected:
            # create the SSL object
            try:
                self._sslobj = self._context._wrap_socket(
                    self, server_side, self.server_hostname,
                    owner=self, session=self._session,
                )
                if do_handshake_on_connect:
                    timeout = self.gettimeout()
                    if timeout == 0.0:
                        # non-blocking
                        raise ValueError("do_handshake_on_connect should not be specified for non-blocking sockets")
                    self.do_handshake()
            except (OSError, ValueError):
                self.close()
                raise
        return self

    @property
    @_sslcopydoc
    def context(self):
        return self._context

    @context.setter
    def context(self, ctx):
        self._context = ctx
        self._sslobj.context = ctx

    @property
    @_sslcopydoc
    def session(self):
        if self._sslobj is not None:
            return self._sslobj.session

    @session.setter
    def session(self, session):
        self._session = session
        if self._sslobj is not None:
            self._sslobj.session = session

    @property
    @_sslcopydoc
    def session_reused(self):
        if self._sslobj is not None:
            return self._sslobj.session_reused

    def dup(self):
        raise NotImplementedError("Can't dup() %s instances" %
                                  self.__class__.__name__)

    def _checkClosed(self, msg=None):
        # raise an exception here if you wish to check for spurious closes
        pass

    def _check_connected(self):
        if not self._connected:
            # getpeername() will raise ENOTCONN if the socket is really
            # not connected; note that we can be connected even without
            # _connected being set, e.g. if connect() first returned
            # EAGAIN.
            self.getpeername()

    def read(self, len=1024, buffer=None):
        """Read up to LEN bytes and return them.
        Return zero-length string on EOF."""

        self._checkClosed()
        if self._sslobj is None:
            raise ValueError("Read on closed or unwrapped SSL socket.")
        try:
            if buffer is not None:
                return self._sslobj.read(len, buffer)
            else:
                return self._sslobj.read(len)
        except SSLError as x:
            if x.args[0] == SSL_ERROR_EOF and self.suppress_ragged_eofs:
                if buffer is not None:
                    return 0
                else:
                    return b''
            else:
                raise

    def write(self, data):
        """Write DATA to the underlying SSL channel.  Returns
        number of bytes of DATA actually transmitted."""

        self._checkClosed()
        if self._sslobj is None:
            raise ValueError("Write on closed or unwrapped SSL socket.")
        return self._sslobj.write(data)

    @_sslcopydoc
    def getpeercert(self, binary_form=False):
        self._checkClosed()
        self._check_connected()
        return self._sslobj.getpeercert(binary_form)

    @_sslcopydoc
    def selected_npn_protocol(self):
        self._checkClosed()
        warnings.warn(
            "ssl NPN is deprecated, use ALPN instead",
            DeprecationWarning,
            stacklevel=2
        )
        return None

    @_sslcopydoc
    def selected_alpn_protocol(self):
        self._checkClosed()
        if self._sslobj is None or not _ssl.HAS_ALPN:
            return None
        else:
            return self._sslobj.selected_alpn_protocol()

    @_sslcopydoc
    def cipher(self):
        self._checkClosed()
        if self._sslobj is None:
            return None
        else:
            return self._sslobj.cipher()

    @_sslcopydoc
    def shared_ciphers(self):
        self._checkClosed()
        if self._sslobj is None:
            return None
        else:
            return self._sslobj.shared_ciphers()

    @_sslcopydoc
    def compression(self):
        self._checkClosed()
        if self._sslobj is None:
            return None
        else:
            return self._sslobj.compression()

    def send(self, data, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError(
                    "non-zero flags not allowed in calls to send() on %s" %
                    self.__class__)
            return self._sslobj.write(data)
        else:
            return super().send(data, flags)

    def sendto(self, data, flags_or_addr, addr=None):
        self._checkClosed()
        if self._sslobj is not None:
            raise ValueError("sendto not allowed on instances of %s" %
                             self.__class__)
        elif addr is None:
            return super().sendto(data, flags_or_addr)
        else:
            return super().sendto(data, flags_or_addr, addr)

    def sendmsg(self, *args, **kwargs):
        # Ensure programs don't send data unencrypted if they try to
        # use this method.
        raise NotImplementedError("sendmsg not allowed on instances of %s" %
                                  self.__class__)

    def sendall(self, data, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError(
                    "non-zero flags not allowed in calls to sendall() on %s" %
                    self.__class__)
            count = 0
            with memoryview(data) as view, view.cast("B") as byte_view:
                amount = len(byte_view)
                while count < amount:
                    v = self.send(byte_view[count:])
                    count += v
        else:
            return super().sendall(data, flags)

    def sendfile(self, file, offset=0, count=None):
        """Send a file, possibly by using os.sendfile() if this is a
        clear-text socket.  Return the total number of bytes sent.
        """
        if self._sslobj is not None:
            return self._sendfile_use_send(file, offset, count)
        else:
            # os.sendfile() works with plain sockets only
            return super().sendfile(file, offset, count)

    def recv(self, buflen=1024, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError(
                    "non-zero flags not allowed in calls to recv() on %s" %
                    self.__class__)
            return self.read(buflen)
        else:
            return super().recv(buflen, flags)

    def recv_into(self, buffer, nbytes=None, flags=0):
        self._checkClosed()
        if buffer and (nbytes is None):
            nbytes = len(buffer)
        elif nbytes is None:
            nbytes = 1024
        if self._sslobj is not None:
            if flags != 0:
                raise ValueError(
                  "non-zero flags not allowed in calls to recv_into() on %s" %
                  self.__class__)
            return self.read(nbytes, buffer)
        else:
            return super().recv_into(buffer, nbytes, flags)

    def recvfrom(self, buflen=1024, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            raise ValueError("recvfrom not allowed on instances of %s" %
                             self.__class__)
        else:
            return super().recvfrom(buflen, flags)

    def recvfrom_into(self, buffer, nbytes=None, flags=0):
        self._checkClosed()
        if self._sslobj is not None:
            raise ValueError("recvfrom_into not allowed on instances of %s" %
                             self.__class__)
        else:
            return super().recvfrom_into(buffer, nbytes, flags)

    def recvmsg(self, *args, **kwargs):
        raise NotImplementedError("recvmsg not allowed on instances of %s" %
                                  self.__class__)

    def recvmsg_into(self, *args, **kwargs):
        raise NotImplementedError("recvmsg_into not allowed on instances of "
                                  "%s" % self.__class__)

    @_sslcopydoc
    def pending(self):
        self._checkClosed()
        if self._sslobj is not None:
            return self._sslobj.pending()
        else:
            return 0

    def shutdown(self, how):
        self._checkClosed()
        self._sslobj = None
        super().shutdown(how)

    @_sslcopydoc
    def unwrap(self):
        if self._sslobj:
            s = self._sslobj.shutdown()
            self._sslobj = None
            return s
        else:
            raise ValueError("No SSL wrapper around " + str(self))

    @_sslcopydoc
    def verify_client_post_handshake(self):
        if self._sslobj:
            return self._sslobj.verify_client_post_handshake()
        else:
            raise ValueError("No SSL wrapper around " + str(self))

    def _real_close(self):
        self._sslobj = None
        super()._real_close()

    @_sslcopydoc
    def do_handshake(self, block=False):
        self._check_connected()
        timeout = self.gettimeout()
        try:
            if timeout == 0.0 and block:
                self.settimeout(None)
            self._sslobj.do_handshake()
        finally:
            self.settimeout(timeout)

    def _real_connect(self, addr, connect_ex):
        if self.server_side:
            raise ValueError("can't connect in server-side mode")
        # Here we assume that the socket is client-side, and not
        # connected at the time of the call.  We connect it, then wrap it.
        if self._connected or self._sslobj is not None:
            raise ValueError("attempt to connect already-connected SSLSocket!")
        self._sslobj = self.context._wrap_socket(
            self, False, self.server_hostname,
            owner=self, session=self._session
        )
        try:
            if connect_ex:
                rc = super().connect_ex(addr)
            else:
                rc = None
                super().connect(addr)
            if not rc:
                self._connected = True
                if self.do_handshake_on_connect:
                    self.do_handshake()
            return rc
        except (OSError, ValueError):
            self._sslobj = None
            raise

    def connect(self, addr):
        """Connects to remote ADDR, and then wraps the connection in
        an SSL channel."""
        self._real_connect(addr, False)

    def connect_ex(self, addr):
        """Connects to remote ADDR, and then wraps the connection in
        an SSL channel."""
        return self._real_connect(addr, True)

    def accept(self):
        """Accepts a new connection from a remote client, and returns
        a tuple containing that new connection wrapped with a server-side
        SSL channel, and the address of the remote client."""

        newsock, addr = super().accept()
        newsock = self.context.wrap_socket(newsock,
                    do_handshake_on_connect=self.do_handshake_on_connect,
                    suppress_ragged_eofs=self.suppress_ragged_eofs,
                    server_side=True)
        return newsock, addr

    @_sslcopydoc
    def get_channel_binding(self, cb_type="tls-unique"):
        if self._sslobj is not None:
            return self._sslobj.get_channel_binding(cb_type)
        else:
            if cb_type not in CHANNEL_BINDING_TYPES:
                raise ValueError(
                    "{0} channel binding type not implemented".format(cb_type)
                )
            return None

    @_sslcopydoc
    def version(self):
        if self._sslobj is not None:
            return self._sslobj.version()
        else:
            return None


# Python does not support forward declaration of types.
SSLContext.sslsocket_class = SSLSocket
SSLContext.sslobject_class = SSLObject


def wrap_socket(sock, keyfile=None, certfile=None,
                server_side=False, cert_reqs=CERT_NONE,
                ssl_version=PROTOCOL_TLS, ca_certs=None,
                do_handshake_on_connect=True,
                suppress_ragged_eofs=True,
                ciphers=None):
    warnings.warn(
        "ssl.wrap_socket() is deprecated, use SSLContext.wrap_socket()",
        category=DeprecationWarning,
        stacklevel=2
    )
    if server_side and not certfile:
        raise ValueError("certfile must be specified for server-side "
                         "operations")
    if keyfile and not certfile:
        raise ValueError("certfile must be specified")
    context = SSLContext(ssl_version)
    context.verify_mode = cert_reqs
    if ca_certs:
        context.load_verify_locations(ca_certs)
    if certfile:
        context.load_cert_chain(certfile, keyfile)
    if ciphers:
        context.set_ciphers(ciphers)
    return context.wrap_socket(
        sock=sock, server_side=server_side,
        do_handshake_on_connect=do_handshake_on_connect,
        suppress_ragged_eofs=suppress_ragged_eofs
    )

# some utility functions

def cert_time_to_seconds(cert_time):
    """Return the time in seconds since the Epoch, given the timestring
    representing the "notBefore" or "notAfter" date from a certificate
    in ``"%b %d %H:%M:%S %Y %Z"`` strptime format (C locale).

    "notBefore" or "notAfter" dates must use UTC (RFC 5280).

    Month is one of: Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
    UTC should be specified as GMT (see ASN1_TIME_print())
    """
    from time import strptime
    from calendar import timegm

    months = (
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    )
    time_format = ' %d %H:%M:%S %Y GMT' # NOTE: no month, fixed GMT
    try:
        month_number = months.index(cert_time[:3].title()) + 1
    except ValueError:
        raise ValueError('time data %r does not match '
                         'format "%%b%s"' % (cert_time, time_format))
    else:
        # found valid month
        tt = strptime(cert_time[3:], time_format)
        # return an integer, the previous mktime()-based implementation
        # returned a float (fractional seconds are always zero here).
        return timegm((tt[0], month_number) + tt[2:6])

PEM_HEADER = "-----BEGIN CERTIFICATE-----"
PEM_FOOTER = "-----END CERTIFICATE-----"

def DER_cert_to_PEM_cert(der_cert_bytes):
    """Takes a certificate in binary DER format and returns the
    PEM version of it as a string."""

    f = str(base64.standard_b64encode(der_cert_bytes), 'ASCII', 'strict')
    ss = [PEM_HEADER]
    ss += [f[i:i+64] for i in range(0, len(f), 64)]
    ss.append(PEM_FOOTER + '\n')
    return '\n'.join(ss)

def PEM_cert_to_DER_cert(pem_cert_string):
    """Takes a certificate in ASCII PEM format and returns the
    DER-encoded version of it as a byte sequence"""

    if not pem_cert_string.startswith(PEM_HEADER):
        raise ValueError("Invalid PEM encoding; must start with %s"
                         % PEM_HEADER)
    if not pem_cert_string.strip().endswith(PEM_FOOTER):
        raise ValueError("Invalid PEM encoding; must end with %s"
                         % PEM_FOOTER)
    d = pem_cert_string.strip()[len(PEM_HEADER):-len(PEM_FOOTER)]
    return base64.decodebytes(d.encode('ASCII', 'strict'))

def get_server_certificate(addr, ssl_version=PROTOCOL_TLS_CLIENT,
                           ca_certs=None, timeout=_GLOBAL_DEFAULT_TIMEOUT):
    """Retrieve the certificate from the server at the specified address,
    and return it as a PEM-encoded string.
    If 'ca_certs' is specified, validate the server cert against it.
    If 'ssl_version' is specified, use it in the connection attempt.
    If 'timeout' is specified, use it in the connection attempt.
    """

    host, port = addr
    if ca_certs is not None:
        cert_reqs = CERT_REQUIRED
    else:
        cert_reqs = CERT_NONE
    context = _create_stdlib_context(ssl_version,
                                     cert_reqs=cert_reqs,
                                     cafile=ca_certs)
    with create_connection(addr, timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=host) as sslsock:
            dercert = sslsock.getpeercert(True)
    return DER_cert_to_PEM_cert(dercert)

def get_protocol_name(protocol_code):
    return _PROTOCOL_NAMES.get(protocol_code, '<unknown>')
<|endoftext|>from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle

# Import types and functions implemented in C
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces


def _format_size(size, sign):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(size) < 100 and unit != 'B':
            # 3 digits (xx.x UNIT)
            if sign:
                return "%+.1f %s" % (size, unit)
            else:
                return "%.1f %s" % (size, unit)
        if abs(size) < 10 * 1024 or unit == 'TiB':
            # 4 or 5 digits (xxxx UNIT)
            if sign:
                return "%+.0f %s" % (size, unit)
            else:
                return "%.0f %s" % (size, unit)
        size /= 1024


class Statistic:
    """
    Statistic difference on memory allocations between two Snapshot instance.
    """

    __slots__ = ('traceback', 'size', 'count')

    def __init__(self, traceback, size, count):
        self.traceback = traceback
        self.size = size
        self.count = count

    def __hash__(self):
        return hash((self.traceback, self.size, self.count))

    def __eq__(self, other):
        if not isinstance(other, Statistic):
            return NotImplemented
        return (self.traceback == other.traceback
                and self.size == other.size
                and self.count == other.count)

    def __str__(self):
        text = ("%s: size=%s, count=%i"
                 % (self.traceback,
                    _format_size(self.size, False),
                    self.count))
        if self.count:
            average = self.size / self.count
            text += ", average=%s" % _format_size(average, False)
        return text

    def __repr__(self):
        return ('<Statistic traceback=%r size=%i count=%i>'
                % (self.traceback, self.size, self.count))

    def _sort_key(self):
        return (self.size, self.count, self.traceback)


class StatisticDiff:
    """
    Statistic difference on memory allocations between an old and a new
    Snapshot instance.
    """
    __slots__ = ('traceback', 'size', 'size_diff', 'count', 'count_diff')

    def __init__(self, traceback, size, size_diff, count, count_diff):
        self.traceback = traceback
        self.size = size
        self.size_diff = size_diff
        self.count = count
        self.count_diff = count_diff

    def __hash__(self):
        return hash((self.traceback, self.size, self.size_diff,
                     self.count, self.count_diff))

    def __eq__(self, other):
        if not isinstance(other, StatisticDiff):
            return NotImplemented
        return (self.traceback == other.traceback
                and self.size == other.size
                and self.size_diff == other.size_diff
                and self.count == other.count
                and self.count_diff == other.count_diff)

    def __str__(self):
        text = ("%s: size=%s (%s), count=%i (%+i)"
                % (self.traceback,
                   _format_size(self.size, False),
                   _format_size(self.size_diff, True),
                   self.count,
                   self.count_diff))
        if self.count:
            average = self.size / self.count
            text += ", average=%s" % _format_size(average, False)
        return text

    def __repr__(self):
        return ('<StatisticDiff traceback=%r size=%i (%+i) count=%i (%+i)>'
                % (self.traceback, self.size, self.size_diff,
                   self.count, self.count_diff))

    def _sort_key(self):
        return (abs(self.size_diff), self.size,
                abs(self.count_diff), self.count,
                self.traceback)


def _compare_grouped_stats(old_group, new_group):
    statistics = []
    for traceback, stat in new_group.items():
        previous = old_group.pop(traceback, None)
        if previous is not None:
            stat = StatisticDiff(traceback,
                                 stat.size, stat.size - previous.size,
                                 stat.count, stat.count - previous.count)
        else:
            stat = StatisticDiff(traceback,
                                 stat.size, stat.size,
                                 stat.count, stat.count)
        statistics.append(stat)

    for traceback, stat in old_group.items():
        stat = StatisticDiff(traceback, 0, -stat.size, 0, -stat.count)
        statistics.append(stat)
    return statistics


@total_ordering
class Frame:
    """
    Frame of a traceback.
    """
    __slots__ = ("_frame",)

    def __init__(self, frame):
        # frame is a tuple: (filename: str, lineno: int)
        self._frame = frame

    @property
    def filename(self):
        return self._frame[0]

    @property
    def lineno(self):
        return self._frame[1]

    def __eq__(self, other):
        if not isinstance(other, Frame):
            return NotImplemented
        return (self._frame == other._frame)

    def __lt__(self, other):
        if not isinstance(other, Frame):
            return NotImplemented
        return (self._frame < other._frame)

    def __hash__(self):
        return hash(self._frame)

    def __str__(self):
        return "%s:%s" % (self.filename, self.lineno)

    def __repr__(self):
        return "<Frame filename=%r lineno=%r>" % (self.filename, self.lineno)


@total_ordering
class Traceback(Sequence):
    """
    Sequence of Frame instances sorted from the oldest frame
    to the most recent frame.
    """
    __slots__ = ("_frames", '_total_nframe')

    def __init__(self, frames, total_nframe=None):
        Sequence.__init__(self)
        # frames is a tuple of frame tuples: see Frame constructor for the
        # format of a frame tuple; it is reversed, because _tracemalloc
        # returns frames sorted from most recent to oldest, but the
        # Python API expects oldest to most recent
        self._frames = tuple(reversed(frames))
        self._total_nframe = total_nframe

    @property
    def total_nframe(self):
        return self._total_nframe

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(Frame(trace) for trace in self._frames[index])
        else:
            return Frame(self._frames[index])

    def __contains__(self, frame):
        return frame._frame in self._frames

    def __hash__(self):
        return hash(self._frames)

    def __eq__(self, other):
        if not isinstance(other, Traceback):
            return NotImplemented
        return (self._frames == other._frames)

    def __lt__(self, other):
        if not isinstance(other, Traceback):
            return NotImplemented
        return (self._frames < other._frames)

    def __str__(self):
        return str(self[0])

    def __repr__(self):
        s = f"<Traceback {tuple(self)}"
        if self._total_nframe is None:
            s += ">"
        else:
            s += f" total_nframe={self.total_nframe}>"
        return s

    def format(self, limit=None, most_recent_first=False):
        lines = []
        if limit is not None:
            if limit > 0:
                frame_slice = self[-limit:]
            else:
                frame_slice = self[:limit]
        else:
            frame_slice = self

        if most_recent_first:
            frame_slice = reversed(frame_slice)
        for frame in frame_slice:
            lines.append('  File "%s", line %s'
                         % (frame.filename, frame.lineno))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                lines.append('    %s' % line)
        return lines


def get_object_traceback(obj):
    """
    Get the traceback where the Python object *obj* was allocated.
    Return a Traceback instance.

    Return None if the tracemalloc module is not tracing memory allocations or
    did not trace the allocation of the object.
    """
    frames = _get_object_traceback(obj)
    if frames is not None:
        return Traceback(frames)
    else:
        return None


class Trace:
    """
    Trace of a memory block.
    """
    __slots__ = ("_trace",)

    def __init__(self, trace):
        # trace is a tuple: (domain: int, size: int, traceback: tuple).
        # See Traceback constructor for the format of the traceback tuple.
        self._trace = trace

    @property
    def domain(self):
        return self._trace[0]

    @property
    def size(self):
        return self._trace[1]

    @property
    def traceback(self):
        return Traceback(*self._trace[2:])

    def __eq__(self, other):
        if not isinstance(other, Trace):
            return NotImplemented
        return (self._trace == other._trace)

    def __hash__(self):
        return hash(self._trace)

    def __str__(self):
        return "%s: %s" % (self.traceback, _format_size(self.size, False))

    def __repr__(self):
        return ("<Trace domain=%s size=%s, traceback=%r>"
                % (self.domain, _format_size(self.size, False), self.traceback))


class _Traces(Sequence):
    def __init__(self, traces):
        Sequence.__init__(self)
        # traces is a tuple of trace tuples: see Trace constructor
        self._traces = traces

    def __len__(self):
        return len(self._traces)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(Trace(trace) for trace in self._traces[index])
        else:
            return Trace(self._traces[index])

    def __contains__(self, trace):
        return trace._trace in self._traces

    def __eq__(self, other):
        if not isinstance(other, _Traces):
            return NotImplemented
        return (self._traces == other._traces)

    def __repr__(self):
        return "<Traces len=%s>" % len(self)


def _normalize_filename(filename):
    filename = os.path.normcase(filename)
    if filename.endswith('.pyc'):
        filename = filename[:-1]
    return filename


class BaseFilter:
    def __init__(self, inclusive):
        self.inclusive = inclusive

    def _match(self, trace):
        raise NotImplementedError


class Filter(BaseFilter):
    def __init__(self, inclusive, filename_pattern,
                 lineno=None, all_frames=False, domain=None):
        super().__init__(inclusive)
        self.inclusive = inclusive
        self._filename_pattern = _normalize_filename(filename_pattern)
        self.lineno = lineno
        self.all_frames = all_frames
        self.domain = domain

    @property
    def filename_pattern(self):
        return self._filename_pattern

    def _match_frame_impl(self, filename, lineno):
        filename = _normalize_filename(filename)
        if not fnmatch.fnmatch(filename, self._filename_pattern):
            return False
        if self.lineno is None:
            return True
        else:
            return (lineno == self.lineno)

    def _match_frame(self, filename, lineno):
        return self._match_frame_impl(filename, lineno) ^ (not self.inclusive)

    def _match_traceback(self, traceback):
        if self.all_frames:
            if any(self._match_frame_impl(filename, lineno)
                   for filename, lineno in traceback):
                return self.inclusive
            else:
                return (not self.inclusive)
        else:
            filename, lineno = traceback[0]
            return self._match_frame(filename, lineno)

    def _match(self, trace):
        domain, size, traceback, total_nframe = trace
        res = self._match_traceback(traceback)
        if self.domain is not None:
            if self.inclusive:
                return res and (domain == self.domain)
            else:
                return res or (domain != self.domain)
        return res


class DomainFilter(BaseFilter):
    def __init__(self, inclusive, domain):
        super().__init__(inclusive)
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    def _match(self, trace):
        domain, size, traceback, total_nframe = trace
        return (domain == self.domain) ^ (not self.inclusive)


class Snapshot:
    """
    Snapshot of traces of memory blocks allocated by Python.
    """

    def __init__(self, traces, traceback_limit):
        # traces is a tuple of trace tuples: see _Traces constructor for
        # the exact format
        self.traces = _Traces(traces)
        self.traceback_limit = traceback_limit

    def dump(self, filename):
        """
        Write the snapshot into a file.
        """
        with open(filename, "wb") as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """
        Load a snapshot from a file.
        """
        with open(filename, "rb") as fp:
            return pickle.load(fp)

    def _filter_trace(self, include_filters, exclude_filters, trace):
        if include_filters:
            if not any(trace_filter._match(trace)
                       for trace_filter in include_filters):
                return False
        if exclude_filters:
            if any(not trace_filter._match(trace)
                   for trace_filter in exclude_filters):
                return False
        return True

    def filter_traces(self, filters):
        """
        Create a new Snapshot instance with a filtered traces sequence, filters
        is a list of Filter or DomainFilter instances.  If filters is an empty
        list, return a new Snapshot instance with a copy of the traces.
        """
        if not isinstance(filters, Iterable):
            raise TypeError("filters must be a list of filters, not %s"
                            % type(filters).__name__)
        if filters:
            include_filters = []
            exclude_filters = []
            for trace_filter in filters:
                if trace_filter.inclusive:
                    include_filters.append(trace_filter)
                else:
                    exclude_filters.append(trace_filter)
            new_traces = [trace for trace in self.traces._traces
                          if self._filter_trace(include_filters,
                                                exclude_filters,
                                                trace)]
        else:
            new_traces = self.traces._traces.copy()
        return Snapshot(new_traces, self.traceback_limit)

    def _group_by(self, key_type, cumulative):
        if key_type not in ('traceback', 'filename', 'lineno'):
            raise ValueError("unknown key_type: %r" % (key_type,))
        if cumulative and key_type not in ('lineno', 'filename'):
            raise ValueError("cumulative mode cannot by used "
                             "with key type %r" % key_type)

        stats = {}
        tracebacks = {}
        if not cumulative:
            for trace in self.traces._traces:
                domain, size, trace_traceback, total_nframe = trace
                try:
                    traceback = tracebacks[trace_traceback]
                except KeyError:
                    if key_type == 'traceback':
                        frames = trace_traceback
                    elif key_type == 'lineno':
                        frames = trace_traceback[:1]
                    else: # key_type == 'filename':
                        frames = ((trace_traceback[0][0], 0),)
                    traceback = Traceback(frames)
                    tracebacks[trace_traceback] = traceback
                try:
                    stat = stats[traceback]
                    stat.size += size
                    stat.count += 1
                except KeyError:
                    stats[traceback] = Statistic(traceback, size, 1)
        else:
            # cumulative statistics
            for trace in self.traces._traces:
                domain, size, trace_traceback, total_nframe = trace
                for frame in trace_traceback:
                    try:
                        traceback = tracebacks[frame]
                    except KeyError:
                        if key_type == 'lineno':
                            frames = (frame,)
                        else: # key_type == 'filename':
                            frames = ((frame[0], 0),)
                        traceback = Traceback(frames)
                        tracebacks[frame] = traceback
                    try:
                        stat = stats[traceback]
                        stat.size += size
                        stat.count += 1
                    except KeyError:
                        stats[traceback] = Statistic(traceback, size, 1)
        return stats

    def statistics(self, key_type, cumulative=False):
        """
        Group statistics by key_type. Return a sorted list of Statistic
        instances.
        """
        grouped = self._group_by(key_type, cumulative)
        statistics = list(grouped.values())
        statistics.sort(reverse=True, key=Statistic._sort_key)
        return statistics

    def compare_to(self, old_snapshot, key_type, cumulative=False):
        """
        Compute the differences with an old snapshot old_snapshot. Get
        statistics as a sorted list of StatisticDiff instances, grouped by
        group_by.
        """
        new_group = self._group_by(key_type, cumulative)
        old_group = old_snapshot._group_by(key_type, cumulative)
        statistics = _compare_grouped_stats(old_group, new_group)
        statistics.sort(reverse=True, key=StatisticDiff._sort_key)
        return statistics


def take_snapshot():
    """
    Take a snapshot of traces of memory blocks allocated by Python.
    """
    if not is_tracing():
        raise RuntimeError("the tracemalloc module must be tracing memory "
                           "allocations to take a snapshot")
    traces = _get_traces()
    traceback_limit = get_traceback_limit()
    return Snapshot(traces, traceback_limit)
<|endoftext|>"""Thread-local objects.

(Note that this module provides a Python version of the threading.local
 class.  Depending on the version of Python you're using, there may be a
 faster one available.  You should always import the `local` class from
 `threading`.)

Thread-local objects support the management of thread-local data.
If you have data that you want to be local to a thread, simply create
a thread-local object and use its attributes:

  >>> mydata = local()
  >>> mydata.number = 42
  >>> mydata.number
  42

You can also access the local-object's dictionary:

  >>> mydata.__dict__
  {'number': 42}
  >>> mydata.__dict__.setdefault('widgets', [])
  []
  >>> mydata.widgets
  []

What's important about thread-local objects is that their data are
local to a thread. If we access the data in a different thread:

  >>> log = []
  >>> def f():
  ...     items = sorted(mydata.__dict__.items())
  ...     log.append(items)
  ...     mydata.number = 11
  ...     log.append(mydata.number)

  >>> import threading
  >>> thread = threading.Thread(target=f)
  >>> thread.start()
  >>> thread.join()
  >>> log
  [[], 11]

we get different data.  Furthermore, changes made in the other thread
don't affect data seen in this thread:

  >>> mydata.number
  42

Of course, values you get from a local object, including a __dict__
attribute, are for whatever thread was current at the time the
attribute was read.  For that reason, you generally don't want to save
these values across threads, as they apply only to the thread they
came from.

You can create custom local objects by subclassing the local class:

  >>> class MyLocal(local):
  ...     number = 2
  ...     def __init__(self, /, **kw):
  ...         self.__dict__.update(kw)
  ...     def squared(self):
  ...         return self.number ** 2

This can be useful to support default values, methods and
initialization.  Note that if you define an __init__ method, it will be
called each time the local object is used in a separate thread.  This
is necessary to initialize each thread's dictionary.

Now if we create a local object:

  >>> mydata = MyLocal(color='red')

Now we have a default number:

  >>> mydata.number
  2

an initial color:

  >>> mydata.color
  'red'
  >>> del mydata.color

And a method that operates on the data:

  >>> mydata.squared()
  4

As before, we can access the data in a separate thread:

  >>> log = []
  >>> thread = threading.Thread(target=f)
  >>> thread.start()
  >>> thread.join()
  >>> log
  [[('color', 'red')], 11]

without affecting this thread's data:

  >>> mydata.number
  2
  >>> mydata.color
  Traceback (most recent call last):
  ...
  AttributeError: 'MyLocal' object has no attribute 'color'

Note that subclasses can define slots, but they are not thread
local. They are shared across threads:

  >>> class MyLocal(local):
  ...     __slots__ = 'number'

  >>> mydata = MyLocal()
  >>> mydata.number = 42
  >>> mydata.color = 'red'

So, the separate thread:

  >>> thread = threading.Thread(target=f)
  >>> thread.start()
  >>> thread.join()

affects what we see:

  >>> mydata.number
  11

>>> del mydata
"""

from weakref import ref
from contextlib import contextmanager

__all__ = ["local"]

# We need to use objects from the threading module, but the threading
# module may also want to use our `local` class, if support for locals
# isn't compiled in to the `thread` module.  This creates potential problems
# with circular imports.  For that reason, we don't import `threading`
# until the bottom of this file (a hack sufficient to worm around the
# potential problems).  Note that all platforms on CPython do have support
# for locals in the `thread` module, and there is no circular import problem
# then, so problems introduced by fiddling the order of imports here won't
# manifest.

class _localimpl:
    """A class managing thread-local dicts"""
    __slots__ = 'key', 'dicts', 'localargs', 'locallock', '__weakref__'

    def __init__(self):
        # The key used in the Thread objects' attribute dicts.
        # We keep it a string for speed but make it unlikely to clash with
        # a "real" attribute.
        self.key = '_threading_local._localimpl.' + str(id(self))
        # { id(Thread) -> (ref(Thread), thread-local dict) }
        self.dicts = {}

    def get_dict(self):
        """Return the dict for the current thread. Raises KeyError if none
        defined."""
        thread = current_thread()
        return self.dicts[id(thread)][1]

    def create_dict(self):
        """Create a new dict for the current thread, and return it."""
        localdict = {}
        key = self.key
        thread = current_thread()
        idt = id(thread)
        def local_deleted(_, key=key):
            # When the localimpl is deleted, remove the thread attribute.
            thread = wrthread()
            if thread is not None:
                del thread.__dict__[key]
        def thread_deleted(_, idt=idt):
            # When the thread is deleted, remove the local dict.
            # Note that this is suboptimal if the thread object gets
            # caught in a reference loop. We would like to be called
            # as soon as the OS-level thread ends instead.
            local = wrlocal()
            if local is not None:
                dct = local.dicts.pop(idt)
        wrlocal = ref(self, local_deleted)
        wrthread = ref(thread, thread_deleted)
        thread.__dict__[key] = wrlocal
        self.dicts[idt] = wrthread, localdict
        return localdict


@contextmanager
def _patch(self):
    impl = object.__getattribute__(self, '_local__impl')
    try:
        dct = impl.get_dict()
    except KeyError:
        dct = impl.create_dict()
        args, kw = impl.localargs
        self.__init__(*args, **kw)
    with impl.locallock:
        object.__setattr__(self, '__dict__', dct)
        yield


class local:
    __slots__ = '_local__impl', '__dict__'

    def __new__(cls, /, *args, **kw):
        if (args or kw) and (cls.__init__ is object.__init__):
            raise TypeError("Initialization arguments are not supported")
        self = object.__new__(cls)
        impl = _localimpl()
        impl.localargs = (args, kw)
        impl.locallock = RLock()
        object.__setattr__(self, '_local__impl', impl)
        # We need to create the thread dict in anticipation of
        # __init__ being called, to make sure we don't call it
        # again ourselves.
        impl.create_dict()
        return self

    def __getattribute__(self, name):
        with _patch(self):
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == '__dict__':
            raise AttributeError(
                "%r object attribute '__dict__' is read-only"
                % self.__class__.__name__)
        with _patch(self):
            return object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name == '__dict__':
            raise AttributeError(
                "%r object attribute '__dict__' is read-only"
                % self.__class__.__name__)
        with _patch(self):
            return object.__delattr__(self, name)


from threading import current_thread, RLock
<|endoftext|>"""Helper to provide extensibility for pickle.

This is only useful to add pickle support for extension types defined in
C, not for instances of user-defined classes.
"""

__all__ = ["pickle", "constructor",
           "add_extension", "remove_extension", "clear_extension_cache"]

dispatch_table = {}

def pickle(ob_type, pickle_function, constructor_ob=None):
    if not callable(pickle_function):
        raise TypeError("reduction functions must be callable")
    dispatch_table[ob_type] = pickle_function

    # The constructor_ob function is a vestige of safe for unpickling.
    # There is no reason for the caller to pass it anymore.
    if constructor_ob is not None:
        constructor(constructor_ob)

def constructor(object):
    if not callable(object):
        raise TypeError("constructors must be callable")

# Example: provide pickling support for complex numbers.

try:
    complex
except NameError:
    pass
else:

    def pickle_complex(c):
        return complex, (c.real, c.imag)

    pickle(complex, pickle_complex, complex)

def pickle_union(obj):
    import functools, operator
    return functools.reduce, (operator.or_, obj.__args__)

pickle(type(int | str), pickle_union)

# Support for pickling new-style objects

def _reconstructor(cls, base, state):
    if base is object:
        obj = object.__new__(cls)
    else:
        obj = base.__new__(cls, state)
        if base.__init__ != object.__init__:
            base.__init__(obj, state)
    return obj

_HEAPTYPE = 1<<9
_new_type = type(int.__new__)

# Python code for object.__reduce_ex__ for protocols 0 and 1

def _reduce_ex(self, proto):
    assert proto < 2
    cls = self.__class__
    for base in cls.__mro__:
        if hasattr(base, '__flags__') and not base.__flags__ & _HEAPTYPE:
            break
        new = base.__new__
        if isinstance(new, _new_type) and new.__self__ is base:
            break
    else:
        base = object # not really reachable
    if base is object:
        state = None
    else:
        if base is cls:
            raise TypeError(f"cannot pickle {cls.__name__!r} object")
        state = base(self)
    args = (cls, base, state)
    try:
        getstate = self.__getstate__
    except AttributeError:
        if getattr(self, "__slots__", None):
            raise TypeError(f"cannot pickle {cls.__name__!r} object: "
                            f"a class that defines __slots__ without "
                            f"defining __getstate__ cannot be pickled "
                            f"with protocol {proto}") from None
        try:
            dict = self.__dict__
        except AttributeError:
            dict = None
    else:
        dict = getstate()
    if dict:
        return _reconstructor, args, dict
    else:
        return _reconstructor, args

# Helper for __reduce_ex__ protocol 2

def __newobj__(cls, *args):
    return cls.__new__(cls, *args)

def __newobj_ex__(cls, args, kwargs):
    """Used by pickle protocol 4, instead of __newobj__ to allow classes with
    keyword-only arguments to be pickled correctly.
    """
    return cls.__new__(cls, *args, **kwargs)

def _slotnames(cls):
    """Return a list of slot names for a given class.

    This needs to find slots defined by the class and its bases, so we
    can't simply return the __slots__ attribute.  We must walk down
    the Method Resolution Order and concatenate the __slots__ of each
    class found there.  (This assumes classes don't modify their
    __slots__ attribute to misrepresent their slots after the class is
    defined.)
    """

    # Get the value from a cache in the class if possible
    names = cls.__dict__.get("__slotnames__")
    if names is not None:
        return names

    # Not cached -- calculate the value
    names = []
    if not hasattr(cls, "__slots__"):
        # This class has no slots
        pass
    else:
        # Slots found -- gather slot names from all base classes
        for c in cls.__mro__:
            if "__slots__" in c.__dict__:
                slots = c.__dict__['__slots__']
                # if class has a single slot, it can be given as a string
                if isinstance(slots, str):
                    slots = (slots,)
                for name in slots:
                    # special descriptors
                    if name in ("__dict__", "__weakref__"):
                        continue
                    # mangled names
                    elif name.startswith('__') and not name.endswith('__'):
                        stripped = c.__name__.lstrip('_')
                        if stripped:
                            names.append('_%s%s' % (stripped, name))
                        else:
                            names.append(name)
                    else:
                        names.append(name)

    # Cache the outcome in the class if at all possible
    try:
        cls.__slotnames__ = names
    except:
        pass # But don't die if we can't

    return names

# A registry of extension codes.  This is an ad-hoc compression
# mechanism.  Whenever a global reference to <module>, <name> is about
# to be pickled, the (<module>, <name>) tuple is looked up here to see
# if it is a registered extension code for it.  Extension codes are
# universal, so that the meaning of a pickle does not depend on
# context.  (There are also some codes reserved for local use that
# don't have this restriction.)  Codes are positive ints; 0 is
# reserved.

_extension_registry = {}                # key -> code
_inverted_registry = {}                 # code -> key
_extension_cache = {}                   # code -> object
# Don't ever rebind those names:  pickling grabs a reference to them when
# it's initialized, and won't see a rebinding.

def add_extension(module, name, code):
    """Register an extension code."""
    code = int(code)
    if not 1 <= code <= 0x7fffffff:
        raise ValueError("code out of range")
    key = (module, name)
    if (_extension_registry.get(key) == code and
        _inverted_registry.get(code) == key):
        return # Redundant registrations are benign
    if key in _extension_registry:
        raise ValueError("key %s is already registered with code %s" %
                         (key, _extension_registry[key]))
    if code in _inverted_registry:
        raise ValueError("code %s is already in use for key %s" %
                         (code, _inverted_registry[code]))
    _extension_registry[key] = code
    _inverted_registry[code] = key

def remove_extension(module, name, code):
    """Unregister an extension code.  For testing only."""
    key = (module, name)
    if (_extension_registry.get(key) != code or
        _inverted_registry.get(code) != key):
        raise ValueError("key %s is not registered with code %s" %
                         (key, code))
    del _extension_registry[key]
    del _inverted_registry[code]
    if code in _extension_cache:
        del _extension_cache[code]

def clear_extension_cache():
    _extension_cache.clear()

# Standard extension code assignments

# Reserved ranges

# First  Last Count  Purpose
#     1   127   127  Reserved for Python standard library
#   128   191    64  Reserved for Zope
#   192   239    48  Reserved for 3rd parties
#   240   255    16  Reserved for private use (will never be assigned)
#   256   Inf   Inf  Reserved for future assignment

# Extension codes are assigned by the Python Software Foundation.
<|endoftext|>#
# Secret Labs' Regular Expression Engine
#
# convert template to internal format
#
# Copyright (c) 1997-2001 by Secret Labs AB.  All rights reserved.
#
# See the sre.py file for information on usage and redistribution.
#

"""Internal support module for sre"""

import _sre
import sre_parse
from sre_constants import *

assert _sre.MAGIC == MAGIC, "SRE module mismatch"

_LITERAL_CODES = {LITERAL, NOT_LITERAL}
_REPEATING_CODES = {REPEAT, MIN_REPEAT, MAX_REPEAT}
_SUCCESS_CODES = {SUCCESS, FAILURE}
_ASSERT_CODES = {ASSERT, ASSERT_NOT}
_UNIT_CODES = _LITERAL_CODES | {ANY, IN}

# Sets of lowercase characters which have the same uppercase.
_equivalences = (
    # LATIN SMALL LETTER I, LATIN SMALL LETTER DOTLESS I
    (0x69, 0x131), # iı
    # LATIN SMALL LETTER S, LATIN SMALL LETTER LONG S
    (0x73, 0x17f), # sſ
    # MICRO SIGN, GREEK SMALL LETTER MU
    (0xb5, 0x3bc), # µμ
    # COMBINING GREEK YPOGEGRAMMENI, GREEK SMALL LETTER IOTA, GREEK PROSGEGRAMMENI
    (0x345, 0x3b9, 0x1fbe), # \u0345ιι
    # GREEK SMALL LETTER IOTA WITH DIALYTIKA AND TONOS, GREEK SMALL LETTER IOTA WITH DIALYTIKA AND OXIA
    (0x390, 0x1fd3), # ΐΐ
    # GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND TONOS, GREEK SMALL LETTER UPSILON WITH DIALYTIKA AND OXIA
    (0x3b0, 0x1fe3), # ΰΰ
    # GREEK SMALL LETTER BETA, GREEK BETA SYMBOL
    (0x3b2, 0x3d0), # βϐ
    # GREEK SMALL LETTER EPSILON, GREEK LUNATE EPSILON SYMBOL
    (0x3b5, 0x3f5), # εϵ
    # GREEK SMALL LETTER THETA, GREEK THETA SYMBOL
    (0x3b8, 0x3d1), # θϑ
    # GREEK SMALL LETTER KAPPA, GREEK KAPPA SYMBOL
    (0x3ba, 0x3f0), # κϰ
    # GREEK SMALL LETTER PI, GREEK PI SYMBOL
    (0x3c0, 0x3d6), # πϖ
    # GREEK SMALL LETTER RHO, GREEK RHO SYMBOL
    (0x3c1, 0x3f1), # ρϱ
    # GREEK SMALL LETTER FINAL SIGMA, GREEK SMALL LETTER SIGMA
    (0x3c2, 0x3c3), # ςσ
    # GREEK SMALL LETTER PHI, GREEK PHI SYMBOL
    (0x3c6, 0x3d5), # φϕ
    # CYRILLIC SMALL LETTER VE, CYRILLIC SMALL LETTER ROUNDED VE
    (0x432, 0x1c80), # вᲀ
    # CYRILLIC SMALL LETTER DE, CYRILLIC SMALL LETTER LONG-LEGGED DE
    (0x434, 0x1c81), # дᲁ
    # CYRILLIC SMALL LETTER O, CYRILLIC SMALL LETTER NARROW O
    (0x43e, 0x1c82), # оᲂ
    # CYRILLIC SMALL LETTER ES, CYRILLIC SMALL LETTER WIDE ES
    (0x441, 0x1c83), # сᲃ
    # CYRILLIC SMALL LETTER TE, CYRILLIC SMALL LETTER TALL TE, CYRILLIC SMALL LETTER THREE-LEGGED TE
    (0x442, 0x1c84, 0x1c85), # тᲄᲅ
    # CYRILLIC SMALL LETTER HARD SIGN, CYRILLIC SMALL LETTER TALL HARD SIGN
    (0x44a, 0x1c86), # ъᲆ
    # CYRILLIC SMALL LETTER YAT, CYRILLIC SMALL LETTER TALL YAT
    (0x463, 0x1c87), # ѣᲇ
    # CYRILLIC SMALL LETTER UNBLENDED UK, CYRILLIC SMALL LETTER MONOGRAPH UK
    (0x1c88, 0xa64b), # ᲈꙋ
    # LATIN SMALL LETTER S WITH DOT ABOVE, LATIN SMALL LETTER LONG S WITH DOT ABOVE
    (0x1e61, 0x1e9b), # ṡẛ
    # LATIN SMALL LIGATURE LONG S T, LATIN SMALL LIGATURE ST
    (0xfb05, 0xfb06), # ﬅﬆ
)

# Maps the lowercase code to lowercase codes which have the same uppercase.
_ignorecase_fixes = {i: tuple(j for j in t if i != j)
                     for t in _equivalences for i in t}

def _combine_flags(flags, add_flags, del_flags,
                   TYPE_FLAGS=sre_parse.TYPE_FLAGS):
    if add_flags & TYPE_FLAGS:
        flags &= ~TYPE_FLAGS
    return (flags | add_flags) & ~del_flags

def _compile(code, pattern, flags):
    # internal: compile a (sub)pattern
    emit = code.append
    _len = len
    LITERAL_CODES = _LITERAL_CODES
    REPEATING_CODES = _REPEATING_CODES
    SUCCESS_CODES = _SUCCESS_CODES
    ASSERT_CODES = _ASSERT_CODES
    iscased = None
    tolower = None
    fixes = None
    if flags & SRE_FLAG_IGNORECASE and not flags & SRE_FLAG_LOCALE:
        if flags & SRE_FLAG_UNICODE:
            iscased = _sre.unicode_iscased
            tolower = _sre.unicode_tolower
            fixes = _ignorecase_fixes
        else:
            iscased = _sre.ascii_iscased
            tolower = _sre.ascii_tolower
    for op, av in pattern:
        if op in LITERAL_CODES:
            if not flags & SRE_FLAG_IGNORECASE:
                emit(op)
                emit(av)
            elif flags & SRE_FLAG_LOCALE:
                emit(OP_LOCALE_IGNORE[op])
                emit(av)
            elif not iscased(av):
                emit(op)
                emit(av)
            else:
                lo = tolower(av)
                if not fixes:  # ascii
                    emit(OP_IGNORE[op])
                    emit(lo)
                elif lo not in fixes:
                    emit(OP_UNICODE_IGNORE[op])
                    emit(lo)
                else:
                    emit(IN_UNI_IGNORE)
                    skip = _len(code); emit(0)
                    if op is NOT_LITERAL:
                        emit(NEGATE)
                    for k in (lo,) + fixes[lo]:
                        emit(LITERAL)
                        emit(k)
                    emit(FAILURE)
                    code[skip] = _len(code) - skip
        elif op is IN:
            charset, hascased = _optimize_charset(av, iscased, tolower, fixes)
            if flags & SRE_FLAG_IGNORECASE and flags & SRE_FLAG_LOCALE:
                emit(IN_LOC_IGNORE)
            elif not hascased:
                emit(IN)
            elif not fixes:  # ascii
                emit(IN_IGNORE)
            else:
                emit(IN_UNI_IGNORE)
            skip = _len(code); emit(0)
            _compile_charset(charset, flags, code)
            code[skip] = _len(code) - skip
        elif op is ANY:
            if flags & SRE_FLAG_DOTALL:
                emit(ANY_ALL)
            else:
                emit(ANY)
        elif op in REPEATING_CODES:
            if flags & SRE_FLAG_TEMPLATE:
                raise error("internal: unsupported template operator %r" % (op,))
            if _simple(av[2]):
                if op is MAX_REPEAT:
                    emit(REPEAT_ONE)
                else:
                    emit(MIN_REPEAT_ONE)
                skip = _len(code); emit(0)
                emit(av[0])
                emit(av[1])
                _compile(code, av[2], flags)
                emit(SUCCESS)
                code[skip] = _len(code) - skip
            else:
                emit(REPEAT)
                skip = _len(code); emit(0)
                emit(av[0])
                emit(av[1])
                _compile(code, av[2], flags)
                code[skip] = _len(code) - skip
                if op is MAX_REPEAT:
                    emit(MAX_UNTIL)
                else:
                    emit(MIN_UNTIL)
        elif op is SUBPATTERN:
            group, add_flags, del_flags, p = av
            if group:
                emit(MARK)
                emit((group-1)*2)
            # _compile_info(code, p, _combine_flags(flags, add_flags, del_flags))
            _compile(code, p, _combine_flags(flags, add_flags, del_flags))
            if group:
                emit(MARK)
                emit((group-1)*2+1)
        elif op in SUCCESS_CODES:
            emit(op)
        elif op in ASSERT_CODES:
            emit(op)
            skip = _len(code); emit(0)
            if av[0] >= 0:
                emit(0) # look ahead
            else:
                lo, hi = av[1].getwidth()
                if lo != hi:
                    raise error("look-behind requires fixed-width pattern")
                emit(lo) # look behind
            _compile(code, av[1], flags)
            emit(SUCCESS)
            code[skip] = _len(code) - skip
        elif op is CALL:
            emit(op)
            skip = _len(code); emit(0)
            _compile(code, av, flags)
            emit(SUCCESS)
            code[skip] = _len(code) - skip
        elif op is AT:
            emit(op)
            if flags & SRE_FLAG_MULTILINE:
                av = AT_MULTILINE.get(av, av)
            if flags & SRE_FLAG_LOCALE:
                av = AT_LOCALE.get(av, av)
            elif flags & SRE_FLAG_UNICODE:
                av = AT_UNICODE.get(av, av)
            emit(av)
        elif op is BRANCH:
            emit(op)
            tail = []
            tailappend = tail.append
            for av in av[1]:
                skip = _len(code); emit(0)
                # _compile_info(code, av, flags)
                _compile(code, av, flags)
                emit(JUMP)
                tailappend(_len(code)); emit(0)
                code[skip] = _len(code) - skip
            emit(FAILURE) # end of branch
            for tail in tail:
                code[tail] = _len(code) - tail
        elif op is CATEGORY:
            emit(op)
            if flags & SRE_FLAG_LOCALE:
                av = CH_LOCALE[av]
            elif flags & SRE_FLAG_UNICODE:
                av = CH_UNICODE[av]
            emit(av)
        elif op is GROUPREF:
            if not flags & SRE_FLAG_IGNORECASE:
                emit(op)
            elif flags & SRE_FLAG_LOCALE:
                emit(GROUPREF_LOC_IGNORE)
            elif not fixes:  # ascii
                emit(GROUPREF_IGNORE)
            else:
                emit(GROUPREF_UNI_IGNORE)
            emit(av-1)
        elif op is GROUPREF_EXISTS:
            emit(op)
            emit(av[0]-1)
            skipyes = _len(code); emit(0)
            _compile(code, av[1], flags)
            if av[2]:
                emit(JUMP)
                skipno = _len(code); emit(0)
                code[skipyes] = _len(code) - skipyes + 1
                _compile(code, av[2], flags)
                code[skipno] = _len(code) - skipno
            else:
                code[skipyes] = _len(code) - skipyes + 1
        else:
            raise error("internal: unsupported operand type %r" % (op,))

def _compile_charset(charset, flags, code):
    # compile charset subprogram
    emit = code.append
    for op, av in charset:
        emit(op)
        if op is NEGATE:
            pass
        elif op is LITERAL:
            emit(av)
        elif op is RANGE or op is RANGE_UNI_IGNORE:
            emit(av[0])
            emit(av[1])
        elif op is CHARSET:
            code.extend(av)
        elif op is BIGCHARSET:
            code.extend(av)
        elif op is CATEGORY:
            if flags & SRE_FLAG_LOCALE:
                emit(CH_LOCALE[av])
            elif flags & SRE_FLAG_UNICODE:
                emit(CH_UNICODE[av])
            else:
                emit(av)
        else:
            raise error("internal: unsupported set operator %r" % (op,))
    emit(FAILURE)

def _optimize_charset(charset, iscased=None, fixup=None, fixes=None):
    # internal: optimize character set
    out = []
    tail = []
    charmap = bytearray(256)
    hascased = False
    for op, av in charset:
        while True:
            try:
                if op is LITERAL:
                    if fixup:
                        lo = fixup(av)
                        charmap[lo] = 1
                        if fixes and lo in fixes:
                            for k in fixes[lo]:
                                charmap[k] = 1
                        if not hascased and iscased(av):
                            hascased = True
                    else:
                        charmap[av] = 1
                elif op is RANGE:
                    r = range(av[0], av[1]+1)
                    if fixup:
                        if fixes:
                            for i in map(fixup, r):
                                charmap[i] = 1
                                if i in fixes:
                                    for k in fixes[i]:
                                        charmap[k] = 1
                        else:
                            for i in map(fixup, r):
                                charmap[i] = 1
                        if not hascased:
                            hascased = any(map(iscased, r))
                    else:
                        for i in r:
                            charmap[i] = 1
                elif op is NEGATE:
                    out.append((op, av))
                else:
                    tail.append((op, av))
            except IndexError:
                if len(charmap) == 256:
                    # character set contains non-UCS1 character codes
                    charmap += b'\0' * 0xff00
                    continue
                # Character set contains non-BMP character codes.
                # For range, all BMP characters in the range are already
                # proceeded.
                if fixup:
                    hascased = True
                    # For now, IN_UNI_IGNORE+LITERAL and
                    # IN_UNI_IGNORE+RANGE_UNI_IGNORE work for all non-BMP
                    # characters, because two characters (at least one of
                    # which is not in the BMP) match case-insensitively
                    # if and only if:
                    # 1) c1.lower() == c2.lower()
                    # 2) c1.lower() == c2 or c1.lower().upper() == c2
                    # Also, both c.lower() and c.lower().upper() are single
                    # characters for every non-BMP character.
                    if op is RANGE:
                        op = RANGE_UNI_IGNORE
                tail.append((op, av))
            break

    # compress character map
    runs = []
    q = 0
    while True:
        p = charmap.find(1, q)
        if p < 0:
            break
        if len(runs) >= 2:
            runs = None
            break
        q = charmap.find(0, p)
        if q < 0:
            runs.append((p, len(charmap)))
            break
        runs.append((p, q))
    if runs is not None:
        # use literal/range
        for p, q in runs:
            if q - p == 1:
                out.append((LITERAL, p))
            else:
                out.append((RANGE, (p, q - 1)))
        out += tail
        # if the case was changed or new representation is more compact
        if hascased or len(out) < len(charset):
            return out, hascased
        # else original character set is good enough
        return charset, hascased

    # use bitmap
    if len(charmap) == 256:
        data = _mk_bitmap(charmap)
        out.append((CHARSET, data))
        out += tail
        return out, hascased

    # To represent a big charset, first a bitmap of all characters in the
    # set is constructed. Then, this bitmap is sliced into chunks of 256
    # characters, duplicate chunks are eliminated, and each chunk is
    # given a number. In the compiled expression, the charset is
    # represented by a 32-bit word sequence, consisting of one word for
    # the number of different chunks, a sequence of 256 bytes (64 words)
    # of chunk numbers indexed by their original chunk position, and a
    # sequence of 256-bit chunks (8 words each).

    # Compression is normally good: in a typical charset, large ranges of
    # Unicode will be either completely excluded (e.g. if only cyrillic
    # letters are to be matched), or completely included (e.g. if large
    # subranges of Kanji match). These ranges will be represented by
    # chunks of all one-bits or all zero-bits.

    # Matching can be also done efficiently: the more significant byte of
    # the Unicode character is an index into the chunk number, and the
    # less significant byte is a bit index in the chunk (just like the
    # CHARSET matching).

    charmap = bytes(charmap) # should be hashable
    comps = {}
    mapping = bytearray(256)
    block = 0
    data = bytearray()
    for i in range(0, 65536, 256):
        chunk = charmap[i: i + 256]
        if chunk in comps:
            mapping[i // 256] = comps[chunk]
        else:
            mapping[i // 256] = comps[chunk] = block
            block += 1
            data += chunk
    data = _mk_bitmap(data)
    data[0:0] = [block] + _bytes_to_codes(mapping)
    out.append((BIGCHARSET, data))
    out += tail
    return out, hascased

_CODEBITS = _sre.CODESIZE * 8
MAXCODE = (1 << _CODEBITS) - 1
_BITS_TRANS = b'0' + b'1' * 255
def _mk_bitmap(bits, _CODEBITS=_CODEBITS, _int=int):
    s = bits.translate(_BITS_TRANS)[::-1]
    return [_int(s[i - _CODEBITS: i], 2)
            for i in range(len(s), 0, -_CODEBITS)]

def _bytes_to_codes(b):
    # Convert block indices to word array
    a = memoryview(b).cast('I')
    assert a.itemsize == _sre.CODESIZE
    assert len(a) * a.itemsize == len(b)
    return a.tolist()

def _simple(p):
    # check if this subpattern is a "simple" operator
    if len(p) != 1:
        return False
    op, av = p[0]
    if op is SUBPATTERN:
        return av[0] is None and _simple(av[-1])
    return op in _UNIT_CODES

def _generate_overlap_table(prefix):
    """
    Generate an overlap table for the following prefix.
    An overlap table is a table of the same size as the prefix which
    informs about the potential self-overlap for each index in the prefix:
    - if overlap[i] == 0, prefix[i:] can't overlap prefix[0:...]
    - if overlap[i] == k with 0 < k <= i, prefix[i-k+1:i+1] overlaps with
      prefix[0:k]
    """
    table = [0] * len(prefix)
    for i in range(1, len(prefix)):
        idx = table[i - 1]
        while prefix[i] != prefix[idx]:
            if idx == 0:
                table[i] = 0
                break
            idx = table[idx - 1]
        else:
            table[i] = idx + 1
    return table

def _get_iscased(flags):
    if not flags & SRE_FLAG_IGNORECASE:
        return None
    elif flags & SRE_FLAG_UNICODE:
        return _sre.unicode_iscased
    else:
        return _sre.ascii_iscased

def _get_literal_prefix(pattern, flags):
    # look for literal prefix
    prefix = []
    prefixappend = prefix.append
    prefix_skip = None
    iscased = _get_iscased(flags)
    for op, av in pattern.data:
        if op is LITERAL:
            if iscased and iscased(av):
                break
            prefixappend(av)
        elif op is SUBPATTERN:
            group, add_flags, del_flags, p = av
            flags1 = _combine_flags(flags, add_flags, del_flags)
            if flags1 & SRE_FLAG_IGNORECASE and flags1 & SRE_FLAG_LOCALE:
                break
            prefix1, prefix_skip1, got_all = _get_literal_prefix(p, flags1)
            if prefix_skip is None:
                if group is not None:
                    prefix_skip = len(prefix)
                elif prefix_skip1 is not None:
                    prefix_skip = len(prefix) + prefix_skip1
            prefix.extend(prefix1)
            if not got_all:
                break
        else:
            break
    else:
        return prefix, prefix_skip, True
    return prefix, prefix_skip, False

def _get_charset_prefix(pattern, flags):
    while True:
        if not pattern.data:
            return None
        op, av = pattern.data[0]
        if op is not SUBPATTERN:
            break
        group, add_flags, del_flags, pattern = av
        flags = _combine_flags(flags, add_flags, del_flags)
        if flags & SRE_FLAG_IGNORECASE and flags & SRE_FLAG_LOCALE:
            return None

    iscased = _get_iscased(flags)
    if op is LITERAL:
        if iscased and iscased(av):
            return None
        return [(op, av)]
    elif op is BRANCH:
        charset = []
        charsetappend = charset.append
        for p in av[1]:
            if not p:
                return None
            op, av = p[0]
            if op is LITERAL and not (iscased and iscased(av)):
                charsetappend((op, av))
            else:
                return None
        return charset
    elif op is IN:
        charset = av
        if iscased:
            for op, av in charset:
                if op is LITERAL:
                    if iscased(av):
                        return None
                elif op is RANGE:
                    if av[1] > 0xffff:
                        return None
                    if any(map(iscased, range(av[0], av[1]+1))):
                        return None
        return charset
    return None

def _compile_info(code, pattern, flags):
    # internal: compile an info block.  in the current version,
    # this contains min/max pattern width, and an optional literal
    # prefix or a character map
    lo, hi = pattern.getwidth()
    if hi > MAXCODE:
        hi = MAXCODE
    if lo == 0:
        code.extend([INFO, 4, 0, lo, hi])
        return
    # look for a literal prefix
    prefix = []
    prefix_skip = 0
    charset = [] # not used
    if not (flags & SRE_FLAG_IGNORECASE and flags & SRE_FLAG_LOCALE):
        # look for literal prefix
        prefix, prefix_skip, got_all = _get_literal_prefix(pattern, flags)
        # if no prefix, look for charset prefix
        if not prefix:
            charset = _get_charset_prefix(pattern, flags)
##     if prefix:
##         print("*** PREFIX", prefix, prefix_skip)
##     if charset:
##         print("*** CHARSET", charset)
    # add an info block
    emit = code.append
    emit(INFO)
    skip = len(code); emit(0)
    # literal flag
    mask = 0
    if prefix:
        mask = SRE_INFO_PREFIX
        if prefix_skip is None and got_all:
            mask = mask | SRE_INFO_LITERAL
    elif charset:
        mask = mask | SRE_INFO_CHARSET
    emit(mask)
    # pattern length
    if lo < MAXCODE:
        emit(lo)
    else:
        emit(MAXCODE)
        prefix = prefix[:MAXCODE]
    emit(min(hi, MAXCODE))
    # add literal prefix
    if prefix:
        emit(len(prefix)) # length
        if prefix_skip is None:
            prefix_skip =  len(prefix)
        emit(prefix_skip) # skip
        code.extend(prefix)
        # generate overlap table
        code.extend(_generate_overlap_table(prefix))
    elif charset:
        charset, hascased = _optimize_charset(charset)
        assert not hascased
        _compile_charset(charset, flags, code)
    code[skip] = len(code) - skip

def isstring(obj):
    return isinstance(obj, (str, bytes))

def _code(p, flags):

    flags = p.state.flags | flags
    code = []

    # compile info block
    _compile_info(code, p, flags)

    # compile the pattern
    _compile(code, p.data, flags)

    code.append(SUCCESS)

    return code

def _hex_code(code):
    return '[%s]' % ', '.join('%#0*x' % (_sre.CODESIZE*2+2, x) for x in code)

def dis(code):
    import sys

    labels = set()
    level = 0
    offset_width = len(str(len(code) - 1))

    def dis_(start, end):
        def print_(*args, to=None):
            if to is not None:
                labels.add(to)
                args += ('(to %d)' % (to,),)
            print('%*d%s ' % (offset_width, start, ':' if start in labels else '.'),
                  end='  '*(level-1))
            print(*args)

        def print_2(*args):
            print(end=' '*(offset_width + 2*level))
            print(*args)

        nonlocal level
        level += 1
        i = start
        while i < end:
            start = i
            op = code[i]
            i += 1
            op = OPCODES[op]
            if op in (SUCCESS, FAILURE, ANY, ANY_ALL,
                      MAX_UNTIL, MIN_UNTIL, NEGATE):
                print_(op)
            elif op in (LITERAL, NOT_LITERAL,
                        LITERAL_IGNORE, NOT_LITERAL_IGNORE,
                        LITERAL_UNI_IGNORE, NOT_LITERAL_UNI_IGNORE,
                        LITERAL_LOC_IGNORE, NOT_LITERAL_LOC_IGNORE):
                arg = code[i]
                i += 1
                print_(op, '%#02x (%r)' % (arg, chr(arg)))
            elif op is AT:
                arg = code[i]
                i += 1
                arg = str(ATCODES[arg])
                assert arg[:3] == 'AT_'
                print_(op, arg[3:])
            elif op is CATEGORY:
                arg = code[i]
                i += 1
                arg = str(CHCODES[arg])
                assert arg[:9] == 'CATEGORY_'
                print_(op, arg[9:])
            elif op in (IN, IN_IGNORE, IN_UNI_IGNORE, IN_LOC_IGNORE):
                skip = code[i]
                print_(op, skip, to=i+skip)
                dis_(i+1, i+skip)
                i += skip
            elif op in (RANGE, RANGE_UNI_IGNORE):
                lo, hi = code[i: i+2]
                i += 2
                print_(op, '%#02x %#02x (%r-%r)' % (lo, hi, chr(lo), chr(hi)))
            elif op is CHARSET:
                print_(op, _hex_code(code[i: i + 256//_CODEBITS]))
                i += 256//_CODEBITS
            elif op is BIGCHARSET:
                arg = code[i]
                i += 1
                mapping = list(b''.join(x.to_bytes(_sre.CODESIZE, sys.byteorder)
                                        for x in code[i: i + 256//_sre.CODESIZE]))
                print_(op, arg, mapping)
                i += 256//_sre.CODESIZE
                level += 1
                for j in range(arg):
                    print_2(_hex_code(code[i: i + 256//_CODEBITS]))
                    i += 256//_CODEBITS
                level -= 1
            elif op in (MARK, GROUPREF, GROUPREF_IGNORE, GROUPREF_UNI_IGNORE,
                        GROUPREF_LOC_IGNORE):
                arg = code[i]
                i += 1
                print_(op, arg)
            elif op is JUMP:
                skip = code[i]
                print_(op, skip, to=i+skip)
                i += 1
            elif op is BRANCH:
                skip = code[i]
                print_(op, skip, to=i+skip)
                while skip:
                    dis_(i+1, i+skip)
                    i += skip
                    start = i
                    skip = code[i]
                    if skip:
                        print_('branch', skip, to=i+skip)
                    else:
                        print_(FAILURE)
                i += 1
            elif op in (REPEAT, REPEAT_ONE, MIN_REPEAT_ONE):
                skip, min, max = code[i: i+3]
                if max == MAXREPEAT:
                    max = 'MAXREPEAT'
                print_(op, skip, min, max, to=i+skip)
                dis_(i+3, i+skip)
                i += skip
            elif op is GROUPREF_EXISTS:
                arg, skip = code[i: i+2]
                print_(op, arg, skip, to=i+skip)
                i += 2
            elif op in (ASSERT, ASSERT_NOT):
                skip, arg = code[i: i+2]
                print_(op, skip, arg, to=i+skip)
                dis_(i+2, i+skip)
                i += skip
            elif op is INFO:
                skip, flags, min, max = code[i: i+4]
                if max == MAXREPEAT:
                    max = 'MAXREPEAT'
                print_(op, skip, bin(flags), min, max, to=i+skip)
                start = i+4
                if flags & SRE_INFO_PREFIX:
                    prefix_len, prefix_skip = code[i+4: i+6]
                    print_2('  prefix_skip', prefix_skip)
                    start = i + 6
                    prefix = code[start: start+prefix_len]
                    print_2('  prefix',
                            '[%s]' % ', '.join('%#02x' % x for x in prefix),
                            '(%r)' % ''.join(map(chr, prefix)))
                    start += prefix_len
                    print_2('  overlap', code[start: start+prefix_len])
                    start += prefix_len
                if flags & SRE_INFO_CHARSET:
                    level += 1
                    print_2('in')
                    dis_(start, i+skip)
                    level -= 1
                i += skip
            else:
                raise ValueError(op)

        level -= 1

    dis_(0, len(code))


def compile(p, flags=0):
    # internal: convert pattern list to internal format

    if isstring(p):
        pattern = p
        p = sre_parse.parse(p, flags)
    else:
        pattern = None

    code = _code(p, flags)

    if flags & SRE_FLAG_DEBUG:
        print()
        dis(code)

    # map in either direction
    groupindex = p.state.groupdict
    indexgroup = [None] * p.state.groups
    for k, i in groupindex.items():
        indexgroup[i] = k

    return _sre.compile(
        pattern, flags | p.state.flags, code,
        p.state.groups-1,
        groupindex, tuple(indexgroup)
        )
<|endoftext|>"""Interface to the liblzma compression library.

This module provides a class for reading and writing compressed files,
classes for incremental (de)compression, and convenience functions for
one-shot (de)compression.

These classes and functions support both the XZ and legacy LZMA
container formats, as well as raw compressed data streams.
"""

__all__ = [
    "CHECK_NONE", "CHECK_CRC32", "CHECK_CRC64", "CHECK_SHA256",
    "CHECK_ID_MAX", "CHECK_UNKNOWN",
    "FILTER_LZMA1", "FILTER_LZMA2", "FILTER_DELTA", "FILTER_X86", "FILTER_IA64",
    "FILTER_ARM", "FILTER_ARMTHUMB", "FILTER_POWERPC", "FILTER_SPARC",
    "FORMAT_AUTO", "FORMAT_XZ", "FORMAT_ALONE", "FORMAT_RAW",
    "MF_HC3", "MF_HC4", "MF_BT2", "MF_BT3", "MF_BT4",
    "MODE_FAST", "MODE_NORMAL", "PRESET_DEFAULT", "PRESET_EXTREME",

    "LZMACompressor", "LZMADecompressor", "LZMAFile", "LZMAError",
    "open", "compress", "decompress", "is_check_supported",
]

import builtins
import io
import os
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
import _compression


_MODE_CLOSED   = 0
_MODE_READ     = 1
# Value 2 no longer used
_MODE_WRITE    = 3


class LZMAFile(_compression.BaseStream):

    """A file object providing transparent LZMA (de)compression.

    An LZMAFile can act as a wrapper for an existing file object, or
    refer directly to a named file on disk.

    Note that LZMAFile provides a *binary* file interface - data read
    is returned as bytes, and data to be written must be given as bytes.
    """

    def __init__(self, filename=None, mode="r", *,
                 format=None, check=-1, preset=None, filters=None):
        """Open an LZMA-compressed file in binary mode.

        filename can be either an actual file name (given as a str,
        bytes, or PathLike object), in which case the named file is
        opened, or it can be an existing file object to read from or
        write to.

        mode can be "r" for reading (default), "w" for (over)writing,
        "x" for creating exclusively, or "a" for appending. These can
        equivalently be given as "rb", "wb", "xb" and "ab" respectively.

        format specifies the container format to use for the file.
        If mode is "r", this defaults to FORMAT_AUTO. Otherwise, the
        default is FORMAT_XZ.

        check specifies the integrity check to use. This argument can
        only be used when opening a file for writing. For FORMAT_XZ,
        the default is CHECK_CRC64. FORMAT_ALONE and FORMAT_RAW do not
        support integrity checks - for these formats, check must be
        omitted, or be CHECK_NONE.

        When opening a file for reading, the *preset* argument is not
        meaningful, and should be omitted. The *filters* argument should
        also be omitted, except when format is FORMAT_RAW (in which case
        it is required).

        When opening a file for writing, the settings used by the
        compressor can be specified either as a preset compression
        level (with the *preset* argument), or in detail as a custom
        filter chain (with the *filters* argument). For FORMAT_XZ and
        FORMAT_ALONE, the default is to use the PRESET_DEFAULT preset
        level. For FORMAT_RAW, the caller must always specify a filter
        chain; the raw compressor does not support preset compression
        levels.

        preset (if provided) should be an integer in the range 0-9,
        optionally OR-ed with the constant PRESET_EXTREME.

        filters (if provided) should be a sequence of dicts. Each dict
        should have an entry for "id" indicating ID of the filter, plus
        additional entries for options to the filter.
        """
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED

        if mode in ("r", "rb"):
            if check != -1:
                raise ValueError("Cannot specify an integrity check "
                                 "when opening a file for reading")
            if preset is not None:
                raise ValueError("Cannot specify a preset compression "
                                 "level when opening a file for reading")
            if format is None:
                format = FORMAT_AUTO
            mode_code = _MODE_READ
        elif mode in ("w", "wb", "a", "ab", "x", "xb"):
            if format is None:
                format = FORMAT_XZ
            mode_code = _MODE_WRITE
            self._compressor = LZMACompressor(format=format, check=check,
                                              preset=preset, filters=filters)
            self._pos = 0
        else:
            raise ValueError("Invalid mode: {!r}".format(mode))

        if isinstance(filename, (str, bytes, os.PathLike)):
            if "b" not in mode:
                mode += "b"
            self._fp = builtins.open(filename, mode)
            self._closefp = True
            self._mode = mode_code
        elif hasattr(filename, "read") or hasattr(filename, "write"):
            self._fp = filename
            self._mode = mode_code
        else:
            raise TypeError("filename must be a str, bytes, file or PathLike object")

        if self._mode == _MODE_READ:
            raw = _compression.DecompressReader(self._fp, LZMADecompressor,
                trailing_error=LZMAError, format=format, filters=filters)
            self._buffer = io.BufferedReader(raw)

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        if self._mode == _MODE_CLOSED:
            return
        try:
            if self._mode == _MODE_READ:
                self._buffer.close()
                self._buffer = None
            elif self._mode == _MODE_WRITE:
                self._fp.write(self._compressor.flush())
                self._compressor = None
        finally:
            try:
                if self._closefp:
                    self._fp.close()
            finally:
                self._fp = None
                self._closefp = False
                self._mode = _MODE_CLOSED

    @property
    def closed(self):
        """True if this file is closed."""
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking."""
        return self.readable() and self._buffer.seekable()

    def readable(self):
        """Return whether the file was opened for reading."""
        self._check_not_closed()
        return self._mode == _MODE_READ

    def writable(self):
        """Return whether the file was opened for writing."""
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    def peek(self, size=-1):
        """Return buffered data without advancing the file position.

        Always returns at least one byte of data, unless at EOF.
        The exact number of bytes returned is unspecified.
        """
        self._check_can_read()
        # Relies on the undocumented fact that BufferedReader.peek() always
        # returns at least one byte (except at EOF)
        return self._buffer.peek(size)

    def read(self, size=-1):
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b"" if the file is already at EOF.
        """
        self._check_can_read()
        return self._buffer.read(size)

    def read1(self, size=-1):
        """Read up to size uncompressed bytes, while trying to avoid
        making multiple reads from the underlying stream. Reads up to a
        buffer's worth of data if size is negative.

        Returns b"" if the file is at EOF.
        """
        self._check_can_read()
        if size < 0:
            size = io.DEFAULT_BUFFER_SIZE
        return self._buffer.read1(size)

    def readline(self, size=-1):
        """Read a line of uncompressed bytes from the file.

        The terminating newline (if present) is retained. If size is
        non-negative, no more than size bytes will be read (in which
        case the line may be incomplete). Returns b'' if already at EOF.
        """
        self._check_can_read()
        return self._buffer.readline(size)

    def write(self, data):
        """Write a bytes object to the file.

        Returns the number of uncompressed bytes written, which is
        always the length of data in bytes. Note that due to buffering,
        the file on disk may not reflect the data written until close()
        is called.
        """
        self._check_can_write()
        if isinstance(data, (bytes, bytearray)):
            length = len(data)
        else:
            # accept any data that supports the buffer protocol
            data = memoryview(data)
            length = data.nbytes

        compressed = self._compressor.compress(data)
        self._fp.write(compressed)
        self._pos += length
        return length

    def seek(self, offset, whence=io.SEEK_SET):
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Possible values for whence are:

            0: start of stream (default): offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        self._check_can_seek()
        return self._buffer.seek(offset, whence)

    def tell(self):
        """Return the current file position."""
        self._check_not_closed()
        if self._mode == _MODE_READ:
            return self._buffer.tell()
        return self._pos


def open(filename, mode="rb", *,
         format=None, check=-1, preset=None, filters=None,
         encoding=None, errors=None, newline=None):
    """Open an LZMA-compressed file in binary or text mode.

    filename can be either an actual file name (given as a str, bytes,
    or PathLike object), in which case the named file is opened, or it
    can be an existing file object to read from or write to.

    The mode argument can be "r", "rb" (default), "w", "wb", "x", "xb",
    "a", or "ab" for binary mode, or "rt", "wt", "xt", or "at" for text
    mode.

    The format, check, preset and filters arguments specify the
    compression settings, as for LZMACompressor, LZMADecompressor and
    LZMAFile.

    For binary mode, this function is equivalent to the LZMAFile
    constructor: LZMAFile(filename, mode, ...). In this case, the
    encoding, errors and newline arguments must not be provided.

    For text mode, an LZMAFile object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error
    handling behavior, and line ending(s).

    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    lz_mode = mode.replace("t", "")
    binary_file = LZMAFile(filename, lz_mode, format=format, check=check,
                           preset=preset, filters=filters)

    if "t" in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


def compress(data, format=FORMAT_XZ, check=-1, preset=None, filters=None):
    """Compress a block of data.

    Refer to LZMACompressor's docstring for a description of the
    optional arguments *format*, *check*, *preset* and *filters*.

    For incremental compression, use an LZMACompressor instead.
    """
    comp = LZMACompressor(format, check, preset, filters)
    return comp.compress(data) + comp.flush()


def decompress(data, format=FORMAT_AUTO, memlimit=None, filters=None):
    """Decompress a block of data.

    Refer to LZMADecompressor's docstring for a description of the
    optional arguments *format*, *check* and *filters*.

    For incremental decompression, use an LZMADecompressor instead.
    """
    results = []
    while True:
        decomp = LZMADecompressor(format, memlimit, filters)
        try:
            res = decomp.decompress(data)
        except LZMAError:
            if results:
                break  # Leftover data is not a valid LZMA/XZ stream; ignore it.
            else:
                raise  # Error on the first iteration; bail out.
        results.append(res)
        if not decomp.eof:
            raise LZMAError("Compressed data ended before the "
                            "end-of-stream marker was reached")
        data = decomp.unused_data
        if not data:
            break
    return b"".join(results)
<|endoftext|>"""Guess the MIME type of a file.

This module defines two useful functions:

guess_type(url, strict=True) -- guess the MIME type and encoding of a URL.

guess_extension(type, strict=True) -- guess the extension for a given MIME type.

It also contains the following, for tuning the behavior:

Data:

knownfiles -- list of files to parse
inited -- flag set when init() has been called
suffix_map -- dictionary mapping suffixes to suffixes
encodings_map -- dictionary mapping suffixes to encodings
types_map -- dictionary mapping suffixes to types

Functions:

init([files]) -- parse a list of files, default knownfiles (on Windows, the
  default values are taken from the registry)
read_mime_types(file) -- parse one file, return a dictionary or None
"""

import os
import sys
import posixpath
import urllib.parse

try:
    from _winapi import _mimetypes_read_windows_registry
except ImportError:
    _mimetypes_read_windows_registry = None

try:
    import winreg as _winreg
except ImportError:
    _winreg = None

__all__ = [
    "knownfiles", "inited", "MimeTypes",
    "guess_type", "guess_all_extensions", "guess_extension",
    "add_type", "init", "read_mime_types",
    "suffix_map", "encodings_map", "types_map", "common_types"
]

knownfiles = [
    "/etc/mime.types",
    "/etc/httpd/mime.types",                    # Mac OS X
    "/etc/httpd/conf/mime.types",               # Apache
    "/etc/apache/mime.types",                   # Apache 1
    "/etc/apache2/mime.types",                  # Apache 2
    "/usr/local/etc/httpd/conf/mime.types",
    "/usr/local/lib/netscape/mime.types",
    "/usr/local/etc/httpd/conf/mime.types",     # Apache 1.2
    "/usr/local/etc/mime.types",                # Apache 1.3
    ]

inited = False
_db = None


class MimeTypes:
    """MIME-types datastore.

    This datastore can handle information from mime.types-style files
    and supports basic determination of MIME type from a filename or
    URL, and can guess a reasonable extension given a MIME type.
    """

    def __init__(self, filenames=(), strict=True):
        if not inited:
            init()
        self.encodings_map = _encodings_map_default.copy()
        self.suffix_map = _suffix_map_default.copy()
        self.types_map = ({}, {}) # dict for (non-strict, strict)
        self.types_map_inv = ({}, {})
        for (ext, type) in _types_map_default.items():
            self.add_type(type, ext, True)
        for (ext, type) in _common_types_default.items():
            self.add_type(type, ext, False)
        for name in filenames:
            self.read(name, strict)

    def add_type(self, type, ext, strict=True):
        """Add a mapping between a type and an extension.

        When the extension is already known, the new
        type will replace the old one. When the type
        is already known the extension will be added
        to the list of known extensions.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
        self.types_map[strict][ext] = type
        exts = self.types_map_inv[strict].setdefault(type, [])
        if ext not in exts:
            exts.append(ext)

    def guess_type(self, url, strict=True):
        """Guess the type of a file which is either a URL or a path-like object.

        Return value is a tuple (type, encoding) where type is None if
        the type can't be guessed (no or unknown suffix) or a string
        of the form type/subtype, usable for a MIME Content-type
        header; and encoding is None for no encoding or the name of
        the program used to encode (e.g. compress or gzip).  The
        mappings are table driven.  Encoding suffixes are case
        sensitive; type suffixes are first tried case sensitive, then
        case insensitive.

        The suffixes .tgz, .taz and .tz (case sensitive!) are all
        mapped to '.tar.gz'.  (This is table-driven too, using the
        dictionary suffix_map.)

        Optional `strict' argument when False adds a bunch of commonly found,
        but non-standard types.
        """
        url = os.fspath(url)
        scheme, url = urllib.parse._splittype(url)
        if scheme == 'data':
            # syntax of data URLs:
            # dataurl   := "data:" [ mediatype ] [ ";base64" ] "," data
            # mediatype := [ type "/" subtype ] *( ";" parameter )
            # data      := *urlchar
            # parameter := attribute "=" value
            # type/subtype defaults to "text/plain"
            comma = url.find(',')
            if comma < 0:
                # bad data URL
                return None, None
            semi = url.find(';', 0, comma)
            if semi >= 0:
                type = url[:semi]
            else:
                type = url[:comma]
            if '=' in type or '/' not in type:
                type = 'text/plain'
            return type, None           # never compressed, so encoding is None
        base, ext = posixpath.splitext(url)
        while (ext_lower := ext.lower()) in self.suffix_map:
            base, ext = posixpath.splitext(base + self.suffix_map[ext_lower])
        # encodings_map is case sensitive
        if ext in self.encodings_map:
            encoding = self.encodings_map[ext]
            base, ext = posixpath.splitext(base)
        else:
            encoding = None
        ext = ext.lower()
        types_map = self.types_map[True]
        if ext in types_map:
            return types_map[ext], encoding
        elif strict:
            return None, encoding
        types_map = self.types_map[False]
        if ext in types_map:
            return types_map[ext], encoding
        else:
            return None, encoding

    def guess_all_extensions(self, type, strict=True):
        """Guess the extensions for a file based on its MIME type.

        Return value is a list of strings giving the possible filename
        extensions, including the leading dot ('.').  The extension is not
        guaranteed to have been associated with any particular data stream,
        but would be mapped to the MIME type `type' by guess_type().

        Optional `strict' argument when false adds a bunch of commonly found,
        but non-standard types.
        """
        type = type.lower()
        extensions = list(self.types_map_inv[True].get(type, []))
        if not strict:
            for ext in self.types_map_inv[False].get(type, []):
                if ext not in extensions:
                    extensions.append(ext)
        return extensions

    def guess_extension(self, type, strict=True):
        """Guess the extension for a file based on its MIME type.

        Return value is a string giving a filename extension,
        including the leading dot ('.').  The extension is not
        guaranteed to have been associated with any particular data
        stream, but would be mapped to the MIME type `type' by
        guess_type().  If no extension can be guessed for `type', None
        is returned.

        Optional `strict' argument when false adds a bunch of commonly found,
        but non-standard types.
        """
        extensions = self.guess_all_extensions(type, strict)
        if not extensions:
            return None
        return extensions[0]

    def read(self, filename, strict=True):
        """
        Read a single mime.types-format file, specified by pathname.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
        with open(filename, encoding='utf-8') as fp:
            self.readfp(fp, strict)

    def readfp(self, fp, strict=True):
        """
        Read a single mime.types-format file.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
        while 1:
            line = fp.readline()
            if not line:
                break
            words = line.split()
            for i in range(len(words)):
                if words[i][0] == '#':
                    del words[i:]
                    break
            if not words:
                continue
            type, suffixes = words[0], words[1:]
            for suff in suffixes:
                self.add_type(type, '.' + suff, strict)

    def read_windows_registry(self, strict=True):
        """
        Load the MIME types database from Windows registry.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """

        if not _mimetypes_read_windows_registry and not _winreg:
            return

        add_type = self.add_type
        if strict:
            add_type = lambda type, ext: self.add_type(type, ext, True)

        # Accelerated function if it is available
        if _mimetypes_read_windows_registry:
            _mimetypes_read_windows_registry(add_type)
        elif _winreg:
            self._read_windows_registry(add_type)

    @classmethod
    def _read_windows_registry(cls, add_type):
        def enum_types(mimedb):
            i = 0
            while True:
                try:
                    ctype = _winreg.EnumKey(mimedb, i)
                except OSError:
                    break
                else:
                    if '\0' not in ctype:
                        yield ctype
                i += 1

        with _winreg.OpenKey(_winreg.HKEY_CLASSES_ROOT, '') as hkcr:
            for subkeyname in enum_types(hkcr):
                try:
                    with _winreg.OpenKey(hkcr, subkeyname) as subkey:
                        # Only check file extensions
                        if not subkeyname.startswith("."):
                            continue
                        # raises OSError if no 'Content Type' value
                        mimetype, datatype = _winreg.QueryValueEx(
                            subkey, 'Content Type')
                        if datatype != _winreg.REG_SZ:
                            continue
                        add_type(mimetype, subkeyname)
                except OSError:
                    continue

def guess_type(url, strict=True):
    """Guess the type of a file based on its URL.

    Return value is a tuple (type, encoding) where type is None if the
    type can't be guessed (no or unknown suffix) or a string of the
    form type/subtype, usable for a MIME Content-type header; and
    encoding is None for no encoding or the name of the program used
    to encode (e.g. compress or gzip).  The mappings are table
