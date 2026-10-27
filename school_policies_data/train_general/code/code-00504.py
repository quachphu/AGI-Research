   12: h    BINGET     1
   14: .    STOP
highest protocol among opcodes = 2

Try protocol 3 with annotations:

>>> dis(pickle.dumps(T, 3), annotate=1)
    0: \x80 PROTO      3 Protocol version indicator.
    2: ]    EMPTY_LIST   Push an empty list.
    3: q    BINPUT     0 Store the stack top into the memo.  The stack is not popped.
    5: h    BINGET     0 Read an object from the memo and push it on the stack.
    7: \x85 TUPLE1       Build a one-tuple out of the topmost item on the stack.
    8: q    BINPUT     1 Store the stack top into the memo.  The stack is not popped.
   10: a    APPEND       Append an object to a list.
   11: 0    POP          Discard the top stack item, shrinking the stack by one item.
   12: h    BINGET     1 Read an object from the memo and push it on the stack.
   14: .    STOP         Stop the unpickling machine.
highest protocol among opcodes = 2

"""

_memo_test = r"""
>>> import pickle
>>> import io
>>> f = io.BytesIO()
>>> p = pickle.Pickler(f, 2)
>>> x = [1, 2, 3]
>>> p.dump(x)
>>> p.dump(x)
>>> f.seek(0)
0
>>> memo = {}
>>> dis(f, memo=memo)
    0: \x80 PROTO      2
    2: ]    EMPTY_LIST
    3: q    BINPUT     0
    5: (    MARK
    6: K        BININT1    1
    8: K        BININT1    2
   10: K        BININT1    3
   12: e        APPENDS    (MARK at 5)
   13: .    STOP
highest protocol among opcodes = 2
>>> dis(f, memo=memo)
   14: \x80 PROTO      2
   16: h    BINGET     0
   18: .    STOP
highest protocol among opcodes = 2
"""

__test__ = {'disassembler_test': _dis_test,
            'disassembler_memo_test': _memo_test,
           }

def _test():
    import doctest
    return doctest.testmod()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='disassemble one or more pickle files')
    parser.add_argument(
        'pickle_file', type=argparse.FileType('br'),
        nargs='*', help='the pickle file')
    parser.add_argument(
        '-o', '--output', default=sys.stdout, type=argparse.FileType('w'),
        help='the file where the output should be written')
    parser.add_argument(
        '-m', '--memo', action='store_true',
        help='preserve memo between disassemblies')
    parser.add_argument(
        '-l', '--indentlevel', default=4, type=int,
        help='the number of blanks by which to indent a new MARK level')
    parser.add_argument(
        '-a', '--annotate',  action='store_true',
        help='annotate each line with a short opcode description')
    parser.add_argument(
        '-p', '--preamble', default="==> {name} <==",
        help='if more than one pickle file is specified, print this before'
        ' each disassembly')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='run self-test suite')
    parser.add_argument(
        '-v', action='store_true',
        help='run verbosely; only affects self-test run')
    args = parser.parse_args()
    if args.test:
        _test()
    else:
        annotate = 30 if args.annotate else 0
        if not args.pickle_file:
            parser.print_help()
        elif len(args.pickle_file) == 1:
            dis(args.pickle_file[0], args.output, None,
                args.indentlevel, annotate)
        else:
            memo = {} if args.memo else None
            for f in args.pickle_file:
                preamble = args.preamble.format(name=f.name)
                args.output.write(preamble + '\n')
                dis(f, args.output, memo, args.indentlevel, annotate)
<|endoftext|>#!/usr/bin/env python3

# portions copyright 2001, Autonomous Zones Industries, Inc., all rights...
# err...  reserved and offered to the public under the terms of the
# Python 2.2 license.
# Author: Zooko O'Whielacronx
# http://zooko.com/
# mailto:zooko@zooko.com
#
# Copyright 2000, Mojam Media, Inc., all rights reserved.
# Author: Skip Montanaro
#
# Copyright 1999, Bioreason, Inc., all rights reserved.
# Author: Andrew Dalke
#
# Copyright 1995-1997, Automatrix, Inc., all rights reserved.
# Author: Skip Montanaro
#
# Copyright 1991-1995, Stichting Mathematisch Centrum, all rights reserved.
#
#
# Permission to use, copy, modify, and distribute this Python software and
# its associated documentation for any purpose without fee is hereby
# granted, provided that the above copyright notice appears in all copies,
# and that both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of neither Automatrix,
# Bioreason or Mojam Media be used in advertising or publicity pertaining to
# distribution of the software without specific, written prior permission.
#
"""program/module to trace Python program or function execution

Sample use, command line:
  trace.py -c -f counts --ignore-dir '$prefix' spam.py eggs
  trace.py -t --ignore-dir '$prefix' spam.py eggs
  trace.py --trackcalls spam.py eggs

Sample use, programmatically
  import sys

  # create a Trace object, telling it what to ignore, and whether to
  # do tracing or line-counting or both.
  tracer = trace.Trace(ignoredirs=[sys.base_prefix, sys.base_exec_prefix,],
                       trace=0, count=1)
  # run the new command using the given tracer
  tracer.run('main()')
  # make a report, placing output in /tmp
  r = tracer.results()
  r.write_results(show_missing=True, coverdir="/tmp")
"""
__all__ = ['Trace', 'CoverageResults']

import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time

import threading

PRAGMA_NOCOVER = "#pragma NO COVER"

class _Ignore:
    def __init__(self, modules=None, dirs=None):
        self._mods = set() if not modules else set(modules)
        self._dirs = [] if not dirs else [os.path.normpath(d)
                                          for d in dirs]
        self._ignore = { '<string>': 1 }

    def names(self, filename, modulename):
        if modulename in self._ignore:
            return self._ignore[modulename]

        # haven't seen this one before, so see if the module name is
        # on the ignore list.
        if modulename in self._mods:  # Identical names, so ignore
            self._ignore[modulename] = 1
            return 1

        # check if the module is a proper submodule of something on
        # the ignore list
        for mod in self._mods:
            # Need to take some care since ignoring
            # "cmp" mustn't mean ignoring "cmpcache" but ignoring
            # "Spam" must also mean ignoring "Spam.Eggs".
            if modulename.startswith(mod + '.'):
                self._ignore[modulename] = 1
                return 1

        # Now check that filename isn't in one of the directories
        if filename is None:
            # must be a built-in, so we must ignore
            self._ignore[modulename] = 1
            return 1

        # Ignore a file when it contains one of the ignorable paths
        for d in self._dirs:
            # The '+ os.sep' is to ensure that d is a parent directory,
            # as compared to cases like:
            #  d = "/usr/local"
            #  filename = "/usr/local.py"
            # or
            #  d = "/usr/local.py"
            #  filename = "/usr/local.py"
            if filename.startswith(d + os.sep):
                self._ignore[modulename] = 1
                return 1

        # Tried the different ways, so we don't ignore this module
        self._ignore[modulename] = 0
        return 0

def _modname(path):
    """Return a plausible module name for the path."""

    base = os.path.basename(path)
    filename, ext = os.path.splitext(base)
    return filename

def _fullmodname(path):
    """Return a plausible module name for the path."""

    # If the file 'path' is part of a package, then the filename isn't
    # enough to uniquely identify it.  Try to do the right thing by
    # looking in sys.path for the longest matching prefix.  We'll
    # assume that the rest is the package name.

    comparepath = os.path.normcase(path)
    longest = ""
    for dir in sys.path:
        dir = os.path.normcase(dir)
        if comparepath.startswith(dir) and comparepath[len(dir)] == os.sep:
            if len(dir) > len(longest):
                longest = dir

    if longest:
        base = path[len(longest) + 1:]
    else:
        base = path
    # the drive letter is never part of the module name
    drive, base = os.path.splitdrive(base)
    base = base.replace(os.sep, ".")
    if os.altsep:
        base = base.replace(os.altsep, ".")
    filename, ext = os.path.splitext(base)
    return filename.lstrip(".")

class CoverageResults:
    def __init__(self, counts=None, calledfuncs=None, infile=None,
                 callers=None, outfile=None):
        self.counts = counts
        if self.counts is None:
            self.counts = {}
        self.counter = self.counts.copy() # map (filename, lineno) to count
        self.calledfuncs = calledfuncs
        if self.calledfuncs is None:
            self.calledfuncs = {}
        self.calledfuncs = self.calledfuncs.copy()
        self.callers = callers
        if self.callers is None:
            self.callers = {}
        self.callers = self.callers.copy()
        self.infile = infile
        self.outfile = outfile
        if self.infile:
            # Try to merge existing counts file.
            try:
                with open(self.infile, 'rb') as f:
                    counts, calledfuncs, callers = pickle.load(f)
                self.update(self.__class__(counts, calledfuncs, callers=callers))
            except (OSError, EOFError, ValueError) as err:
                print(("Skipping counts file %r: %s"
                                      % (self.infile, err)), file=sys.stderr)

    def is_ignored_filename(self, filename):
        """Return True if the filename does not refer to a file
        we want to have reported.
        """
        return filename.startswith('<') and filename.endswith('>')

    def update(self, other):
        """Merge in the data from another CoverageResults"""
        counts = self.counts
        calledfuncs = self.calledfuncs
        callers = self.callers
        other_counts = other.counts
        other_calledfuncs = other.calledfuncs
        other_callers = other.callers

        for key in other_counts:
            counts[key] = counts.get(key, 0) + other_counts[key]

        for key in other_calledfuncs:
            calledfuncs[key] = 1

        for key in other_callers:
            callers[key] = 1

    def write_results(self, show_missing=True, summary=False, coverdir=None):
        """
        Write the coverage results.

        :param show_missing: Show lines that had no hits.
        :param summary: Include coverage summary per module.
        :param coverdir: If None, the results of each module are placed in its
                         directory, otherwise it is included in the directory
                         specified.
        """
        if self.calledfuncs:
            print()
            print("functions called:")
            calls = self.calledfuncs
            for filename, modulename, funcname in sorted(calls):
                print(("filename: %s, modulename: %s, funcname: %s"
                       % (filename, modulename, funcname)))

        if self.callers:
            print()
            print("calling relationships:")
            lastfile = lastcfile = ""
            for ((pfile, pmod, pfunc), (cfile, cmod, cfunc)) \
                    in sorted(self.callers):
                if pfile != lastfile:
                    print()
                    print("***", pfile, "***")
                    lastfile = pfile
                    lastcfile = ""
                if cfile != pfile and lastcfile != cfile:
                    print("  -->", cfile)
                    lastcfile = cfile
                print("    %s.%s -> %s.%s" % (pmod, pfunc, cmod, cfunc))

        # turn the counts data ("(filename, lineno) = count") into something
        # accessible on a per-file basis
        per_file = {}
        for filename, lineno in self.counts:
            lines_hit = per_file[filename] = per_file.get(filename, {})
            lines_hit[lineno] = self.counts[(filename, lineno)]

        # accumulate summary info, if needed
        sums = {}

        for filename, count in per_file.items():
            if self.is_ignored_filename(filename):
                continue

            if filename.endswith(".pyc"):
                filename = filename[:-1]

            if coverdir is None:
                dir = os.path.dirname(os.path.abspath(filename))
                modulename = _modname(filename)
            else:
                dir = coverdir
                if not os.path.exists(dir):
                    os.makedirs(dir)
                modulename = _fullmodname(filename)

            # If desired, get a list of the line numbers which represent
            # executable content (returned as a dict for better lookup speed)
            if show_missing:
                lnotab = _find_executable_linenos(filename)
            else:
                lnotab = {}
            source = linecache.getlines(filename)
            coverpath = os.path.join(dir, modulename + ".cover")
            with open(filename, 'rb') as fp:
                encoding, _ = tokenize.detect_encoding(fp.readline)
            n_hits, n_lines = self.write_results_file(coverpath, source,
                                                      lnotab, count, encoding)
            if summary and n_lines:
                percent = int(100 * n_hits / n_lines)
                sums[modulename] = n_lines, percent, modulename, filename


        if summary and sums:
            print("lines   cov%   module   (path)")
            for m in sorted(sums):
                n_lines, percent, modulename, filename = sums[m]
                print("%5d   %3d%%   %s   (%s)" % sums[m])

        if self.outfile:
            # try and store counts and module info into self.outfile
            try:
                with open(self.outfile, 'wb') as f:
                    pickle.dump((self.counts, self.calledfuncs, self.callers),
                                f, 1)
            except OSError as err:
                print("Can't save counts files because %s" % err, file=sys.stderr)

    def write_results_file(self, path, lines, lnotab, lines_hit, encoding=None):
        """Return a coverage results file in path."""
        # ``lnotab`` is a dict of executable lines, or a line number "table"

        try:
            outfile = open(path, "w", encoding=encoding)
        except OSError as err:
            print(("trace: Could not open %r for writing: %s "
                                  "- skipping" % (path, err)), file=sys.stderr)
            return 0, 0

        n_lines = 0
        n_hits = 0
        with outfile:
            for lineno, line in enumerate(lines, 1):
                # do the blank/comment match to try to mark more lines
                # (help the reader find stuff that hasn't been covered)
                if lineno in lines_hit:
                    outfile.write("%5d: " % lines_hit[lineno])
                    n_hits += 1
                    n_lines += 1
                elif lineno in lnotab and not PRAGMA_NOCOVER in line:
                    # Highlight never-executed lines, unless the line contains
                    # #pragma: NO COVER
                    outfile.write(">>>>>> ")
                    n_lines += 1
                else:
                    outfile.write("       ")
                outfile.write(line.expandtabs(8))

        return n_hits, n_lines

def _find_lines_from_code(code, strs):
    """Return dict where keys are lines in the line number table."""
    linenos = {}

    for _, lineno in dis.findlinestarts(code):
        if lineno not in strs:
            linenos[lineno] = 1

    return linenos

def _find_lines(code, strs):
    """Return lineno dict for all code objects reachable from code."""
    # get all of the lineno information from the code of this scope level
    linenos = _find_lines_from_code(code, strs)

    # and check the constants for references to other code objects
    for c in code.co_consts:
        if inspect.iscode(c):
            # find another code object, so recurse into it
            linenos.update(_find_lines(c, strs))
    return linenos

def _find_strings(filename, encoding=None):
    """Return a dict of possible docstring positions.

    The dict maps line numbers to strings.  There is an entry for
    line that contains only a string or a part of a triple-quoted
    string.
    """
    d = {}
    # If the first token is a string, then it's the module docstring.
    # Add this special case so that the test in the loop passes.
    prev_ttype = token.INDENT
    with open(filename, encoding=encoding) as f:
        tok = tokenize.generate_tokens(f.readline)
        for ttype, tstr, start, end, line in tok:
            if ttype == token.STRING:
                if prev_ttype == token.INDENT:
                    sline, scol = start
                    eline, ecol = end
                    for i in range(sline, eline + 1):
                        d[i] = 1
            prev_ttype = ttype
    return d

def _find_executable_linenos(filename):
    """Return dict where keys are line numbers in the line number table."""
    try:
        with tokenize.open(filename) as f:
            prog = f.read()
            encoding = f.encoding
    except OSError as err:
        print(("Not printing coverage data for %r: %s"
                              % (filename, err)), file=sys.stderr)
        return {}
    code = compile(prog, filename, "exec")
    strs = _find_strings(filename, encoding)
    return _find_lines(code, strs)

class Trace:
    def __init__(self, count=1, trace=1, countfuncs=0, countcallers=0,
                 ignoremods=(), ignoredirs=(), infile=None, outfile=None,
                 timing=False):
        """
        @param count true iff it should count number of times each
                     line is executed
        @param trace true iff it should print out each line that is
                     being counted
        @param countfuncs true iff it should just output a list of
                     (filename, modulename, funcname,) for functions
                     that were called at least once;  This overrides
                     `count' and `trace'
        @param ignoremods a list of the names of modules to ignore
        @param ignoredirs a list of the names of directories to ignore
                     all of the (recursive) contents of
        @param infile file from which to read stored counts to be
                     added into the results
        @param outfile file in which to write the results
        @param timing true iff timing information be displayed
        """
        self.infile = infile
        self.outfile = outfile
        self.ignore = _Ignore(ignoremods, ignoredirs)
        self.counts = {}   # keys are (filename, linenumber)
        self.pathtobasename = {} # for memoizing os.path.basename
        self.donothing = 0
        self.trace = trace
        self._calledfuncs = {}
        self._callers = {}
        self._caller_cache = {}
        self.start_time = None
        if timing:
            self.start_time = _time()
        if countcallers:
            self.globaltrace = self.globaltrace_trackcallers
        elif countfuncs:
            self.globaltrace = self.globaltrace_countfuncs
        elif trace and count:
            self.globaltrace = self.globaltrace_lt
            self.localtrace = self.localtrace_trace_and_count
        elif trace:
            self.globaltrace = self.globaltrace_lt
            self.localtrace = self.localtrace_trace
        elif count:
            self.globaltrace = self.globaltrace_lt
            self.localtrace = self.localtrace_count
        else:
            # Ahem -- do nothing?  Okay.
            self.donothing = 1

    def run(self, cmd):
        import __main__
        dict = __main__.__dict__
        self.runctx(cmd, dict, dict)

    def runctx(self, cmd, globals=None, locals=None):
        if globals is None: globals = {}
        if locals is None: locals = {}
        if not self.donothing:
            threading.settrace(self.globaltrace)
            sys.settrace(self.globaltrace)
        try:
            exec(cmd, globals, locals)
        finally:
            if not self.donothing:
                sys.settrace(None)
                threading.settrace(None)

    def runfunc(self, func, /, *args, **kw):
        result = None
        if not self.donothing:
            sys.settrace(self.globaltrace)
        try:
            result = func(*args, **kw)
        finally:
            if not self.donothing:
                sys.settrace(None)
        return result

    def file_module_function_of(self, frame):
        code = frame.f_code
        filename = code.co_filename
        if filename:
            modulename = _modname(filename)
        else:
            modulename = None

        funcname = code.co_name
        clsname = None
        if code in self._caller_cache:
            if self._caller_cache[code] is not None:
                clsname = self._caller_cache[code]
        else:
            self._caller_cache[code] = None
            ## use of gc.get_referrers() was suggested by Michael Hudson
            # all functions which refer to this code object
            funcs = [f for f in gc.get_referrers(code)
                         if inspect.isfunction(f)]
            # require len(func) == 1 to avoid ambiguity caused by calls to
            # new.function(): "In the face of ambiguity, refuse the
            # temptation to guess."
            if len(funcs) == 1:
                dicts = [d for d in gc.get_referrers(funcs[0])
                             if isinstance(d, dict)]
                if len(dicts) == 1:
                    classes = [c for c in gc.get_referrers(dicts[0])
                                   if hasattr(c, "__bases__")]
                    if len(classes) == 1:
                        # ditto for new.classobj()
                        clsname = classes[0].__name__
                        # cache the result - assumption is that new.* is
                        # not called later to disturb this relationship
                        # _caller_cache could be flushed if functions in
                        # the new module get called.
                        self._caller_cache[code] = clsname
        if clsname is not None:
            funcname = "%s.%s" % (clsname, funcname)

        return filename, modulename, funcname

    def globaltrace_trackcallers(self, frame, why, arg):
        """Handler for call events.

        Adds information about who called who to the self._callers dict.
        """
        if why == 'call':
            # XXX Should do a better job of identifying methods
            this_func = self.file_module_function_of(frame)
            parent_func = self.file_module_function_of(frame.f_back)
            self._callers[(parent_func, this_func)] = 1

    def globaltrace_countfuncs(self, frame, why, arg):
        """Handler for call events.

        Adds (filename, modulename, funcname) to the self._calledfuncs dict.
        """
        if why == 'call':
            this_func = self.file_module_function_of(frame)
            self._calledfuncs[this_func] = 1

    def globaltrace_lt(self, frame, why, arg):
        """Handler for call events.

        If the code block being entered is to be ignored, returns `None',
        else returns self.localtrace.
        """
        if why == 'call':
            code = frame.f_code
            filename = frame.f_globals.get('__file__', None)
            if filename:
                # XXX _modname() doesn't work right for packages, so
                # the ignore support won't work right for packages
                modulename = _modname(filename)
                if modulename is not None:
                    ignore_it = self.ignore.names(filename, modulename)
                    if not ignore_it:
                        if self.trace:
                            print((" --- modulename: %s, funcname: %s"
                                   % (modulename, code.co_name)))
                        return self.localtrace
            else:
                return None

    def localtrace_trace_and_count(self, frame, why, arg):
        if why == "line":
            # record the file name and line number of every trace
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            key = filename, lineno
            self.counts[key] = self.counts.get(key, 0) + 1

            if self.start_time:
                print('%.2f' % (_time() - self.start_time), end=' ')
            bname = os.path.basename(filename)
            print("%s(%d): %s" % (bname, lineno,
                                  linecache.getline(filename, lineno)), end='')
        return self.localtrace

    def localtrace_trace(self, frame, why, arg):
        if why == "line":
            # record the file name and line number of every trace
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

            if self.start_time:
                print('%.2f' % (_time() - self.start_time), end=' ')
            bname = os.path.basename(filename)
            print("%s(%d): %s" % (bname, lineno,
                                  linecache.getline(filename, lineno)), end='')
        return self.localtrace

    def localtrace_count(self, frame, why, arg):
        if why == "line":
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            key = filename, lineno
            self.counts[key] = self.counts.get(key, 0) + 1
        return self.localtrace

    def results(self):
        return CoverageResults(self.counts, infile=self.infile,
                               outfile=self.outfile,
                               calledfuncs=self._calledfuncs,
                               callers=self._callers)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='version', version='trace 2.0')

    grp = parser.add_argument_group('Main options',
            'One of these (or --report) must be given')

    grp.add_argument('-c', '--count', action='store_true',
            help='Count the number of times each line is executed and write '
                 'the counts to <module>.cover for each module executed, in '
                 'the module\'s directory. See also --coverdir, --file, '
                 '--no-report below.')
    grp.add_argument('-t', '--trace', action='store_true',
            help='Print each line to sys.stdout before it is executed')
    grp.add_argument('-l', '--listfuncs', action='store_true',
            help='Keep track of which functions are executed at least once '
                 'and write the results to sys.stdout after the program exits. '
                 'Cannot be specified alongside --trace or --count.')
    grp.add_argument('-T', '--trackcalls', action='store_true',
            help='Keep track of caller/called pairs and write the results to '
                 'sys.stdout after the program exits.')

    grp = parser.add_argument_group('Modifiers')

    _grp = grp.add_mutually_exclusive_group()
    _grp.add_argument('-r', '--report', action='store_true',
            help='Generate a report from a counts file; does not execute any '
                 'code. --file must specify the results file to read, which '
                 'must have been created in a previous run with --count '
                 '--file=FILE')
    _grp.add_argument('-R', '--no-report', action='store_true',
            help='Do not generate the coverage report files. '
                 'Useful if you want to accumulate over several runs.')

    grp.add_argument('-f', '--file',
            help='File to accumulate counts over several runs')
    grp.add_argument('-C', '--coverdir',
            help='Directory where the report files go. The coverage report '
                 'for <package>.<module> will be written to file '
                 '<dir>/<package>/<module>.cover')
    grp.add_argument('-m', '--missing', action='store_true',
            help='Annotate executable lines that were not executed with '
                 '">>>>>> "')
    grp.add_argument('-s', '--summary', action='store_true',
            help='Write a brief summary for each file to sys.stdout. '
                 'Can only be used with --count or --report')
    grp.add_argument('-g', '--timing', action='store_true',
            help='Prefix each line with the time since the program started. '
                 'Only used while tracing')

    grp = parser.add_argument_group('Filters',
            'Can be specified multiple times')
    grp.add_argument('--ignore-module', action='append', default=[],
            help='Ignore the given module(s) and its submodules '
                 '(if it is a package). Accepts comma separated list of '
                 'module names.')
    grp.add_argument('--ignore-dir', action='append', default=[],
            help='Ignore files in the given directory '
                 '(multiple directories can be joined by os.pathsep).')

    parser.add_argument('--module', action='store_true', default=False,
                        help='Trace a module. ')
    parser.add_argument('progname', nargs='?',
            help='file to run as main program')
    parser.add_argument('arguments', nargs=argparse.REMAINDER,
            help='arguments to the program')

    opts = parser.parse_args()

    if opts.ignore_dir:
        _prefix = sysconfig.get_path("stdlib")
        _exec_prefix = sysconfig.get_path("platstdlib")

    def parse_ignore_dir(s):
        s = os.path.expanduser(os.path.expandvars(s))
        s = s.replace('$prefix', _prefix).replace('$exec_prefix', _exec_prefix)
        return os.path.normpath(s)

    opts.ignore_module = [mod.strip()
                          for i in opts.ignore_module for mod in i.split(',')]
    opts.ignore_dir = [parse_ignore_dir(s)
                       for i in opts.ignore_dir for s in i.split(os.pathsep)]

    if opts.report:
        if not opts.file:
            parser.error('-r/--report requires -f/--file')
        results = CoverageResults(infile=opts.file, outfile=opts.file)
        return results.write_results(opts.missing, opts.summary, opts.coverdir)

    if not any([opts.trace, opts.count, opts.listfuncs, opts.trackcalls]):
        parser.error('must specify one of --trace, --count, --report, '
                     '--listfuncs, or --trackcalls')

    if opts.listfuncs and (opts.count or opts.trace):
        parser.error('cannot specify both --listfuncs and (--trace or --count)')

    if opts.summary and not opts.count:
        parser.error('--summary can only be used with --count or --report')

    if opts.progname is None:
        parser.error('progname is missing: required with the main options')

    t = Trace(opts.count, opts.trace, countfuncs=opts.listfuncs,
              countcallers=opts.trackcalls, ignoremods=opts.ignore_module,
              ignoredirs=opts.ignore_dir, infile=opts.file,
              outfile=opts.file, timing=opts.timing)
    try:
        if opts.module:
            import runpy
            module_name = opts.progname
            mod_name, mod_spec, code = runpy._get_module_details(module_name)
            sys.argv = [code.co_filename, *opts.arguments]
            globs = {
                '__name__': '__main__',
                '__file__': code.co_filename,
                '__package__': mod_spec.parent,
                '__loader__': mod_spec.loader,
                '__spec__': mod_spec,
                '__cached__': None,
            }
        else:
            sys.argv = [opts.progname, *opts.arguments]
            sys.path[0] = os.path.dirname(opts.progname)

            with io.open_code(opts.progname) as fp:
                code = compile(fp.read(), opts.progname, 'exec')
            # try to emulate __main__ namespace as much as possible
            globs = {
                '__file__': opts.progname,
                '__name__': '__main__',
                '__package__': None,
                '__cached__': None,
            }
        t.runctx(code, globs, globs)
    except OSError as err:
        sys.exit("Cannot run file %r because: %s" % (sys.argv[0], err))
    except SystemExit:
        pass

    results = t.results()

    if not opts.no_report:
        results.write_results(opts.missing, opts.summary, opts.coverdir)

if __name__=='__main__':
    main()
<|endoftext|>"""Filename globbing utility."""

import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys

__all__ = ["glob", "iglob", "escape"]

def glob(pathname, *, root_dir=None, dir_fd=None, recursive=False):
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    return list(iglob(pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive))

def iglob(pathname, *, root_dir=None, dir_fd=None, recursive=False):
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    sys.audit("glob.glob", pathname, recursive)
    sys.audit("glob.glob/2", pathname, recursive, root_dir, dir_fd)
    if root_dir is not None:
        root_dir = os.fspath(root_dir)
    else:
        root_dir = pathname[:0]
    it = _iglob(pathname, root_dir, dir_fd, recursive, False)
    if not pathname or recursive and _isrecursive(pathname[:2]):
        try:
            s = next(it)  # skip empty string
            if s:
                it = itertools.chain((s,), it)
        except StopIteration:
            pass
    return it

def _iglob(pathname, root_dir, dir_fd, recursive, dironly):
    dirname, basename = os.path.split(pathname)
    if not has_magic(pathname):
        assert not dironly
        if basename:
            if _lexists(_join(root_dir, pathname), dir_fd):
                yield pathname
        else:
            # Patterns ending with a slash should match only directories
            if _isdir(_join(root_dir, dirname), dir_fd):
                yield pathname
        return
    if not dirname:
        if recursive and _isrecursive(basename):
            yield from _glob2(root_dir, basename, dir_fd, dironly)
        else:
            yield from _glob1(root_dir, basename, dir_fd, dironly)
        return
    # `os.path.split()` returns the argument itself as a dirname if it is a
    # drive or UNC path.  Prevent an infinite recursion if a drive or UNC path
    # contains magic characters (i.e. r'\\?\C:').
    if dirname != pathname and has_magic(dirname):
        dirs = _iglob(dirname, root_dir, dir_fd, recursive, True)
    else:
        dirs = [dirname]
    if has_magic(basename):
        if recursive and _isrecursive(basename):
            glob_in_dir = _glob2
        else:
            glob_in_dir = _glob1
    else:
        glob_in_dir = _glob0
    for dirname in dirs:
        for name in glob_in_dir(_join(root_dir, dirname), basename, dir_fd, dironly):
            yield os.path.join(dirname, name)

# These 2 helper functions non-recursively glob inside a literal directory.
# They return a list of basenames.  _glob1 accepts a pattern while _glob0
# takes a literal basename (so it only has to check for its existence).

def _glob1(dirname, pattern, dir_fd, dironly):
    names = _listdir(dirname, dir_fd, dironly)
    if not _ishidden(pattern):
        names = (x for x in names if not _ishidden(x))
    return fnmatch.filter(names, pattern)

def _glob0(dirname, basename, dir_fd, dironly):
    if basename:
        if _lexists(_join(dirname, basename), dir_fd):
            return [basename]
    else:
        # `os.path.split()` returns an empty basename for paths ending with a
        # directory separator.  'q*x/' should match only directories.
        if _isdir(dirname, dir_fd):
            return [basename]
    return []

# Following functions are not public but can be used by third-party code.

def glob0(dirname, pattern):
    return _glob0(dirname, pattern, None, False)

def glob1(dirname, pattern):
    return _glob1(dirname, pattern, None, False)

# This helper function recursively yields relative pathnames inside a literal
# directory.

def _glob2(dirname, pattern, dir_fd, dironly):
    assert _isrecursive(pattern)
    yield pattern[:0]
    yield from _rlistdir(dirname, dir_fd, dironly)

# If dironly is false, yields all file names inside a directory.
# If dironly is true, yields only directory names.
def _iterdir(dirname, dir_fd, dironly):
    try:
        fd = None
        fsencode = None
        if dir_fd is not None:
            if dirname:
                fd = arg = os.open(dirname, _dir_open_flags, dir_fd=dir_fd)
            else:
                arg = dir_fd
            if isinstance(dirname, bytes):
                fsencode = os.fsencode
        elif dirname:
            arg = dirname
        elif isinstance(dirname, bytes):
            arg = bytes(os.curdir, 'ASCII')
        else:
            arg = os.curdir
        try:
            with os.scandir(arg) as it:
                for entry in it:
                    try:
                        if not dironly or entry.is_dir():
                            if fsencode is not None:
                                yield fsencode(entry.name)
                            else:
                                yield entry.name
                    except OSError:
                        pass
        finally:
            if fd is not None:
                os.close(fd)
    except OSError:
        return

def _listdir(dirname, dir_fd, dironly):
    with contextlib.closing(_iterdir(dirname, dir_fd, dironly)) as it:
        return list(it)

# Recursively yields relative pathnames inside a literal directory.
def _rlistdir(dirname, dir_fd, dironly):
    names = _listdir(dirname, dir_fd, dironly)
    for x in names:
        if not _ishidden(x):
            yield x
            path = _join(dirname, x) if dirname else x
            for y in _rlistdir(path, dir_fd, dironly):
                yield _join(x, y)


def _lexists(pathname, dir_fd):
    # Same as os.path.lexists(), but with dir_fd
    if dir_fd is None:
        return os.path.lexists(pathname)
    try:
        os.lstat(pathname, dir_fd=dir_fd)
    except (OSError, ValueError):
        return False
    else:
        return True

def _isdir(pathname, dir_fd):
    # Same as os.path.isdir(), but with dir_fd
    if dir_fd is None:
        return os.path.isdir(pathname)
    try:
        st = os.stat(pathname, dir_fd=dir_fd)
    except (OSError, ValueError):
        return False
    else:
        return stat.S_ISDIR(st.st_mode)

def _join(dirname, basename):
    # It is common if dirname or basename is empty
    if not dirname or not basename:
        return dirname or basename
    return os.path.join(dirname, basename)

magic_check = re.compile('([*?[])')
magic_check_bytes = re.compile(b'([*?[])')

def has_magic(s):
    if isinstance(s, bytes):
        match = magic_check_bytes.search(s)
    else:
        match = magic_check.search(s)
    return match is not None

def _ishidden(path):
    return path[0] in ('.', b'.'[0])

def _isrecursive(pattern):
    if isinstance(pattern, bytes):
        return pattern == b'**'
    else:
        return pattern == '**'

def escape(pathname):
    """Escape all special characters.
    """
    # Escaping is done by wrapping any of "*?[" between square brackets.
    # Metacharacters do not work in the drive part and shouldn't be escaped.
    drive, pathname = os.path.splitdrive(pathname)
    if isinstance(pathname, bytes):
        pathname = magic_check_bytes.sub(br'[\1]', pathname)
    else:
        pathname = magic_check.sub(r'[\1]', pathname)
    return drive + pathname


_dir_open_flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
<|endoftext|>"""Stuff to parse Sun and NeXT audio files.

An audio file consists of a header followed by the data.  The structure
of the header is as follows.

        +---------------+
        | magic word    |
        +---------------+
        | header size   |
        +---------------+
        | data size     |
        +---------------+
        | encoding      |
        +---------------+
        | sample rate   |
        +---------------+
        | # of channels |
        +---------------+
        | info          |
        |               |
        +---------------+

The magic word consists of the 4 characters '.snd'.  Apart from the
info field, all header fields are 4 bytes in size.  They are all
32-bit unsigned integers encoded in big-endian byte order.

The header size really gives the start of the data.
The data size is the physical size of the data.  From the other
parameters the number of frames can be calculated.
The encoding gives the way in which audio samples are encoded.
Possible values are listed below.
The info field currently consists of an ASCII string giving a
human-readable description of the audio file.  The info field is
padded with NUL bytes to the header size.

Usage.

Reading audio files:
        f = sunau.open(file, 'r')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods read(), seek(), and close().
When the setpos() and rewind() methods are not used, the seek()
method is not  necessary.

This returns an instance of a class with the following public methods:
        getnchannels()  -- returns number of audio channels (1 for
                           mono, 2 for stereo)
        getsampwidth()  -- returns sample width in bytes
        getframerate()  -- returns sampling frequency
        getnframes()    -- returns number of audio frames
        getcomptype()   -- returns compression type ('NONE' or 'ULAW')
        getcompname()   -- returns human-readable version of
                           compression type ('not compressed' matches 'NONE')
        getparams()     -- returns a namedtuple consisting of all of the
                           above in the above order
        getmarkers()    -- returns None (for compatibility with the
                           aifc module)
        getmark(id)     -- raises an error since the mark does not
                           exist (for compatibility with the aifc module)
        readframes(n)   -- returns at most n frames of audio
        rewind()        -- rewind to the beginning of the audio stream
        setpos(pos)     -- seek to the specified position
        tell()          -- return the current position
        close()         -- close the instance (make it unusable)
The position returned by tell() and the position given to setpos()
are compatible and have nothing to do with the actual position in the
file.
The close() method is called automatically when the class instance
is destroyed.

Writing audio files:
        f = sunau.open(file, 'w')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods write(), tell(), seek(), and
close().

This returns an instance of a class with the following public methods:
        setnchannels(n) -- set the number of channels
        setsampwidth(n) -- set the sample width
        setframerate(n) -- set the frame rate
        setnframes(n)   -- set the number of frames
        setcomptype(type, name)
                        -- set the compression type and the
                           human-readable compression type
        setparams(tuple)-- set all parameters at once
        tell()          -- return current position in output file
        writeframesraw(data)
                        -- write audio frames without pathing up the
                           file header
        writeframes(data)
                        -- write audio frames and patch up the file header
        close()         -- patch up the file header and close the
                           output file
You should set the parameters before the first writeframesraw or
writeframes.  The total number of frames does not need to be set,
but when it is set to the correct value, the header does not have to
be patched up.
It is best to first set all parameters, perhaps possibly the
compression type, and then write audio frames using writeframesraw.
When all frames have been written, either call writeframes(b'') or
close() to patch up the sizes in the header.
The close() method is called automatically when the class instance
is destroyed.
"""

from collections import namedtuple


_sunau_params = namedtuple('_sunau_params',
                           'nchannels sampwidth framerate nframes comptype compname')

# from <multimedia/audio_filehdr.h>
AUDIO_FILE_MAGIC = 0x2e736e64
AUDIO_FILE_ENCODING_MULAW_8 = 1
AUDIO_FILE_ENCODING_LINEAR_8 = 2
AUDIO_FILE_ENCODING_LINEAR_16 = 3
AUDIO_FILE_ENCODING_LINEAR_24 = 4
AUDIO_FILE_ENCODING_LINEAR_32 = 5
AUDIO_FILE_ENCODING_FLOAT = 6
AUDIO_FILE_ENCODING_DOUBLE = 7
AUDIO_FILE_ENCODING_ADPCM_G721 = 23
AUDIO_FILE_ENCODING_ADPCM_G722 = 24
AUDIO_FILE_ENCODING_ADPCM_G723_3 = 25
AUDIO_FILE_ENCODING_ADPCM_G723_5 = 26
AUDIO_FILE_ENCODING_ALAW_8 = 27

# from <multimedia/audio_hdr.h>
AUDIO_UNKNOWN_SIZE = 0xFFFFFFFF        # ((unsigned)(~0))

_simple_encodings = [AUDIO_FILE_ENCODING_MULAW_8,
                     AUDIO_FILE_ENCODING_LINEAR_8,
                     AUDIO_FILE_ENCODING_LINEAR_16,
                     AUDIO_FILE_ENCODING_LINEAR_24,
                     AUDIO_FILE_ENCODING_LINEAR_32,
                     AUDIO_FILE_ENCODING_ALAW_8]

class Error(Exception):
    pass

def _read_u32(file):
    x = 0
    for i in range(4):
        byte = file.read(1)
        if not byte:
            raise EOFError
        x = x*256 + ord(byte)
    return x

def _write_u32(file, x):
    data = []
    for i in range(4):
        d, m = divmod(x, 256)
        data.insert(0, int(m))
        x = d
    file.write(bytes(data))

class Au_read:

    def __init__(self, f):
        if type(f) == type(''):
            import builtins
            f = builtins.open(f, 'rb')
            self._opened = True
        else:
            self._opened = False
        self.initfp(f)

    def __del__(self):
        if self._file:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def initfp(self, file):
        self._file = file
        self._soundpos = 0
        magic = int(_read_u32(file))
        if magic != AUDIO_FILE_MAGIC:
            raise Error('bad magic number')
        self._hdr_size = int(_read_u32(file))
        if self._hdr_size < 24:
            raise Error('header size too small')
        if self._hdr_size > 100:
            raise Error('header size ridiculously large')
        self._data_size = _read_u32(file)
        if self._data_size != AUDIO_UNKNOWN_SIZE:
            self._data_size = int(self._data_size)
        self._encoding = int(_read_u32(file))
        if self._encoding not in _simple_encodings:
            raise Error('encoding not (yet) supported')
        if self._encoding in (AUDIO_FILE_ENCODING_MULAW_8,
                  AUDIO_FILE_ENCODING_ALAW_8):
            self._sampwidth = 2
            self._framesize = 1
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_8:
            self._framesize = self._sampwidth = 1
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_16:
            self._framesize = self._sampwidth = 2
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_24:
            self._framesize = self._sampwidth = 3
        elif self._encoding == AUDIO_FILE_ENCODING_LINEAR_32:
            self._framesize = self._sampwidth = 4
        else:
            raise Error('unknown encoding')
        self._framerate = int(_read_u32(file))
        self._nchannels = int(_read_u32(file))
        if not self._nchannels:
            raise Error('bad # of channels')
        self._framesize = self._framesize * self._nchannels
        if self._hdr_size > 24:
            self._info = file.read(self._hdr_size - 24)
            self._info, _, _ = self._info.partition(b'\0')
        else:
            self._info = b''
        try:
            self._data_pos = file.tell()
        except (AttributeError, OSError):
            self._data_pos = None

    def getfp(self):
        return self._file

    def getnchannels(self):
        return self._nchannels

    def getsampwidth(self):
        return self._sampwidth

    def getframerate(self):
        return self._framerate

    def getnframes(self):
        if self._data_size == AUDIO_UNKNOWN_SIZE:
            return AUDIO_UNKNOWN_SIZE
        if self._encoding in _simple_encodings:
            return self._data_size // self._framesize
        return 0                # XXX--must do some arithmetic here

    def getcomptype(self):
        if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
            return 'ULAW'
        elif self._encoding == AUDIO_FILE_ENCODING_ALAW_8:
            return 'ALAW'
        else:
            return 'NONE'

    def getcompname(self):
        if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
            return 'CCITT G.711 u-law'
        elif self._encoding == AUDIO_FILE_ENCODING_ALAW_8:
            return 'CCITT G.711 A-law'
        else:
            return 'not compressed'

    def getparams(self):
        return _sunau_params(self.getnchannels(), self.getsampwidth(),
                  self.getframerate(), self.getnframes(),
                  self.getcomptype(), self.getcompname())

    def getmarkers(self):
        return None

    def getmark(self, id):
        raise Error('no marks')

    def readframes(self, nframes):
        if self._encoding in _simple_encodings:
            if nframes == AUDIO_UNKNOWN_SIZE:
                data = self._file.read()
            else:
                data = self._file.read(nframes * self._framesize)
            self._soundpos += len(data) // self._framesize
            if self._encoding == AUDIO_FILE_ENCODING_MULAW_8:
                import audioop
                data = audioop.ulaw2lin(data, self._sampwidth)
            return data
        return None             # XXX--not implemented yet

    def rewind(self):
        if self._data_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._data_pos)
        self._soundpos = 0

    def tell(self):
        return self._soundpos

    def setpos(self, pos):
        if pos < 0 or pos > self.getnframes():
            raise Error('position not in range')
        if self._data_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._data_pos + pos * self._framesize)
        self._soundpos = pos

    def close(self):
        file = self._file
        if file:
            self._file = None
            if self._opened:
                file.close()

class Au_write:

    def __init__(self, f):
        if type(f) == type(''):
            import builtins
            f = builtins.open(f, 'wb')
            self._opened = True
        else:
            self._opened = False
        self.initfp(f)

    def __del__(self):
        if self._file:
            self.close()
        self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def initfp(self, file):
        self._file = file
        self._framerate = 0
        self._nchannels = 0
        self._sampwidth = 0
        self._framesize = 0
        self._nframes = AUDIO_UNKNOWN_SIZE
        self._nframeswritten = 0
        self._datawritten = 0
        self._datalength = 0
        self._info = b''
        self._comptype = 'ULAW' # default is U-law

    def setnchannels(self, nchannels):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nchannels not in (1, 2, 4):
            raise Error('only 1, 2, or 4 channels supported')
        self._nchannels = nchannels

    def getnchannels(self):
        if not self._nchannels:
            raise Error('number of channels not set')
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if sampwidth not in (1, 2, 3, 4):
            raise Error('bad sample width')
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if not self._framerate:
            raise Error('sample width not specified')
        return self._sampwidth

    def setframerate(self, framerate):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._framerate = framerate

    def getframerate(self):
        if not self._framerate:
            raise Error('frame rate not set')
        return self._framerate

    def setnframes(self, nframes):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nframes < 0:
            raise Error('# of frames cannot be negative')
        self._nframes = nframes

    def getnframes(self):
        return self._nframeswritten

    def setcomptype(self, type, name):
        if type in ('NONE', 'ULAW'):
            self._comptype = type
        else:
            raise Error('unknown compression type')

    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        if self._comptype == 'ULAW':
            return 'CCITT G.711 u-law'
        elif self._comptype == 'ALAW':
            return 'CCITT G.711 A-law'
        else:
            return 'not compressed'

    def setparams(self, params):
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        self.setnchannels(nchannels)
        self.setsampwidth(sampwidth)
        self.setframerate(framerate)
        self.setnframes(nframes)
        self.setcomptype(comptype, compname)

    def getparams(self):
        return _sunau_params(self.getnchannels(), self.getsampwidth(),
                  self.getframerate(), self.getnframes(),
                  self.getcomptype(), self.getcompname())

    def tell(self):
        return self._nframeswritten

    def writeframesraw(self, data):
        if not isinstance(data, (bytes, bytearray)):
            data = memoryview(data).cast('B')
        self._ensure_header_written()
        if self._comptype == 'ULAW':
            import audioop
            data = audioop.lin2ulaw(data, self._sampwidth)
        nframes = len(data) // self._framesize
        self._file.write(data)
        self._nframeswritten = self._nframeswritten + nframes
        self._datawritten = self._datawritten + len(data)

    def writeframes(self, data):
        self.writeframesraw(data)
        if self._nframeswritten != self._nframes or \
                  self._datalength != self._datawritten:
            self._patchheader()

    def close(self):
        if self._file:
            try:
                self._ensure_header_written()
                if self._nframeswritten != self._nframes or \
                        self._datalength != self._datawritten:
                    self._patchheader()
                self._file.flush()
            finally:
                file = self._file
                self._file = None
                if self._opened:
                    file.close()

    #
    # private methods
    #

    def _ensure_header_written(self):
        if not self._nframeswritten:
            if not self._nchannels:
                raise Error('# of channels not specified')
            if not self._sampwidth:
                raise Error('sample width not specified')
            if not self._framerate:
                raise Error('frame rate not specified')
            self._write_header()

    def _write_header(self):
        if self._comptype == 'NONE':
            if self._sampwidth == 1:
                encoding = AUDIO_FILE_ENCODING_LINEAR_8
                self._framesize = 1
            elif self._sampwidth == 2:
                encoding = AUDIO_FILE_ENCODING_LINEAR_16
                self._framesize = 2
            elif self._sampwidth == 3:
                encoding = AUDIO_FILE_ENCODING_LINEAR_24
                self._framesize = 3
            elif self._sampwidth == 4:
                encoding = AUDIO_FILE_ENCODING_LINEAR_32
                self._framesize = 4
            else:
                raise Error('internal error')
        elif self._comptype == 'ULAW':
            encoding = AUDIO_FILE_ENCODING_MULAW_8
            self._framesize = 1
        else:
            raise Error('internal error')
        self._framesize = self._framesize * self._nchannels
        _write_u32(self._file, AUDIO_FILE_MAGIC)
        header_size = 25 + len(self._info)
        header_size = (header_size + 7) & ~7
        _write_u32(self._file, header_size)
        if self._nframes == AUDIO_UNKNOWN_SIZE:
            length = AUDIO_UNKNOWN_SIZE
        else:
            length = self._nframes * self._framesize
        try:
            self._form_length_pos = self._file.tell()
        except (AttributeError, OSError):
            self._form_length_pos = None
        _write_u32(self._file, length)
        self._datalength = length
        _write_u32(self._file, encoding)
        _write_u32(self._file, self._framerate)
        _write_u32(self._file, self._nchannels)
        self._file.write(self._info)
        self._file.write(b'\0'*(header_size - len(self._info) - 24))

    def _patchheader(self):
        if self._form_length_pos is None:
            raise OSError('cannot seek')
        self._file.seek(self._form_length_pos)
        _write_u32(self._file, self._datawritten)
        self._datalength = self._datawritten
        self._file.seek(0, 2)

def open(f, mode=None):
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return Au_read(f)
    elif mode in ('w', 'wb'):
        return Au_write(f)
    else:
        raise Error("mode must be 'r', 'rb', 'w', or 'wb'")
<|endoftext|>"""Utilities for with-statement contexts.  See PEP 343."""
import abc
import sys
import _collections_abc
from collections import deque
from functools import wraps
from types import MethodType, GenericAlias

__all__ = ["asynccontextmanager", "contextmanager", "closing", "nullcontext",
           "AbstractContextManager", "AbstractAsyncContextManager",
           "AsyncExitStack", "ContextDecorator", "ExitStack",
           "redirect_stdout", "redirect_stderr", "suppress", "aclosing"]


class AbstractContextManager(abc.ABC):

    """An abstract base class for context managers."""

    __class_getitem__ = classmethod(GenericAlias)

    def __enter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractContextManager:
            return _collections_abc._check_methods(C, "__enter__", "__exit__")
        return NotImplemented


class AbstractAsyncContextManager(abc.ABC):

    """An abstract base class for asynchronous context managers."""

    __class_getitem__ = classmethod(GenericAlias)

    async def __aenter__(self):
        """Return `self` upon entering the runtime context."""
        return self

    @abc.abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback):
        """Raise any exception triggered within the runtime context."""
        return None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractAsyncContextManager:
            return _collections_abc._check_methods(C, "__aenter__",
                                                   "__aexit__")
        return NotImplemented


class ContextDecorator(object):
    "A base class or mixin that enables context managers to work as decorators."

    def _recreate_cm(self):
        """Return a recreated instance of self.

        Allows an otherwise one-shot context manager like
        _GeneratorContextManager to support use as
        a decorator via implicit recreation.

        This is a private interface just for _GeneratorContextManager.
        See issue #11647 for details.
        """
        return self

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            with self._recreate_cm():
                return func(*args, **kwds)
        return inner


class AsyncContextDecorator(object):
    "A base class or mixin that enables async context managers to work as decorators."

    def _recreate_cm(self):
        """Return a recreated instance of self.
        """
        return self

    def __call__(self, func):
        @wraps(func)
        async def inner(*args, **kwds):
            async with self._recreate_cm():
                return await func(*args, **kwds)
        return inner


class _GeneratorContextManagerBase:
    """Shared functionality for @contextmanager and @asynccontextmanager."""

    def __init__(self, func, args, kwds):
        self.gen = func(*args, **kwds)
        self.func, self.args, self.kwds = func, args, kwds
        # Issue 19330: ensure context manager instances have good docstrings
        doc = getattr(func, "__doc__", None)
        if doc is None:
            doc = type(self).__doc__
        self.__doc__ = doc
        # Unfortunately, this still doesn't provide good help output when
        # inspecting the created context manager instances, since pydoc
        # currently bypasses the instance docstring and shows the docstring
        # for the class instead.
        # See http://bugs.python.org/issue19404 for more details.

    def _recreate_cm(self):
        # _GCMB instances are one-shot context managers, so the
        # CM must be recreated each time a decorated function is
        # called
        return self.__class__(self.func, self.args, self.kwds)


class _GeneratorContextManager(
    _GeneratorContextManagerBase,
    AbstractContextManager,
    ContextDecorator,
):
    """Helper for @contextmanager decorator."""

    def __enter__(self):
        # do not keep args and kwds alive unnecessarily
        # they are only needed for recreation, which is not possible anymore
        del self.args, self.kwds, self.func
        try:
            return next(self.gen)
        except StopIteration:
            raise RuntimeError("generator didn't yield") from None

    def __exit__(self, typ, value, traceback):
        if typ is None:
            try:
                next(self.gen)
            except StopIteration:
                return False
            else:
                raise RuntimeError("generator didn't stop")
        else:
            if value is None:
                # Need to force instantiation so we can reliably
                # tell if we get the same exception back
                value = typ()
            try:
                self.gen.throw(typ, value, traceback)
            except StopIteration as exc:
                # Suppress StopIteration *unless* it's the same exception that
                # was passed to throw().  This prevents a StopIteration
                # raised inside the "with" statement from being suppressed.
                return exc is not value
            except RuntimeError as exc:
                # Don't re-raise the passed in exception. (issue27122)
                if exc is value:
                    return False
                # Avoid suppressing if a StopIteration exception
                # was passed to throw() and later wrapped into a RuntimeError
                # (see PEP 479 for sync generators; async generators also
                # have this behavior). But do this only if the exception wrapped
                # by the RuntimeError is actually Stop(Async)Iteration (see
                # issue29692).
                if (
                    isinstance(value, StopIteration)
                    and exc.__cause__ is value
                ):
                    return False
                raise
            except BaseException as exc:
                # only re-raise if it's *not* the exception that was
                # passed to throw(), because __exit__() must not raise
                # an exception unless __exit__() itself failed.  But throw()
                # has to raise the exception to signal propagation, so this
                # fixes the impedance mismatch between the throw() protocol
                # and the __exit__() protocol.
                if exc is not value:
                    raise
                return False
            raise RuntimeError("generator didn't stop after throw()")

class _AsyncGeneratorContextManager(
    _GeneratorContextManagerBase,
    AbstractAsyncContextManager,
    AsyncContextDecorator,
):
    """Helper for @asynccontextmanager decorator."""

    async def __aenter__(self):
        # do not keep args and kwds alive unnecessarily
        # they are only needed for recreation, which is not possible anymore
        del self.args, self.kwds, self.func
        try:
            return await anext(self.gen)
        except StopAsyncIteration:
            raise RuntimeError("generator didn't yield") from None

    async def __aexit__(self, typ, value, traceback):
        if typ is None:
            try:
                await anext(self.gen)
            except StopAsyncIteration:
                return False
            else:
                raise RuntimeError("generator didn't stop")
        else:
            if value is None:
                # Need to force instantiation so we can reliably
                # tell if we get the same exception back
                value = typ()
            try:
                await self.gen.athrow(typ, value, traceback)
            except StopAsyncIteration as exc:
                # Suppress StopIteration *unless* it's the same exception that
                # was passed to throw().  This prevents a StopIteration
                # raised inside the "with" statement from being suppressed.
                return exc is not value
            except RuntimeError as exc:
                # Don't re-raise the passed in exception. (issue27122)
                if exc is value:
                    return False
                # Avoid suppressing if a Stop(Async)Iteration exception
                # was passed to athrow() and later wrapped into a RuntimeError
                # (see PEP 479 for sync generators; async generators also
                # have this behavior). But do this only if the exception wrapped
                # by the RuntimeError is actully Stop(Async)Iteration (see
                # issue29692).
                if (
                    isinstance(value, (StopIteration, StopAsyncIteration))
                    and exc.__cause__ is value
                ):
                    return False
                raise
            except BaseException as exc:
                # only re-raise if it's *not* the exception that was
                # passed to throw(), because __exit__() must not raise
                # an exception unless __exit__() itself failed.  But throw()
                # has to raise the exception to signal propagation, so this
                # fixes the impedance mismatch between the throw() protocol
                # and the __exit__() protocol.
                if exc is not value:
                    raise
                return False
            raise RuntimeError("generator didn't stop after athrow()")


def contextmanager(func):
    """@contextmanager decorator.

    Typical usage:

        @contextmanager
        def some_generator(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    This makes this:

        with some_generator(<arguments>) as <variable>:
            <body>

    equivalent to this:

        <setup>
        try:
            <variable> = <value>
            <body>
        finally:
            <cleanup>
    """
    @wraps(func)
    def helper(*args, **kwds):
        return _GeneratorContextManager(func, args, kwds)
    return helper


def asynccontextmanager(func):
    """@asynccontextmanager decorator.

    Typical usage:

        @asynccontextmanager
        async def some_async_generator(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    This makes this:

        async with some_async_generator(<arguments>) as <variable>:
            <body>

    equivalent to this:

        <setup>
        try:
            <variable> = <value>
            <body>
        finally:
            <cleanup>
    """
    @wraps(func)
    def helper(*args, **kwds):
        return _AsyncGeneratorContextManager(func, args, kwds)
    return helper


class closing(AbstractContextManager):
    """Context to automatically close something at the end of a block.

    Code like this:

        with closing(<module>.open(<arguments>)) as f:
            <block>

    is equivalent to this:

        f = <module>.open(<arguments>)
        try:
            <block>
        finally:
            f.close()

    """
    def __init__(self, thing):
        self.thing = thing
    def __enter__(self):
        return self.thing
    def __exit__(self, *exc_info):
        self.thing.close()


class aclosing(AbstractAsyncContextManager):
    """Async context manager for safely finalizing an asynchronously cleaned-up
    resource such as an async generator, calling its ``aclose()`` method.

    Code like this:

        async with aclosing(<module>.fetch(<arguments>)) as agen:
            <block>

    is equivalent to this:

        agen = <module>.fetch(<arguments>)
        try:
            <block>
        finally:
            await agen.aclose()

    """
    def __init__(self, thing):
        self.thing = thing
    async def __aenter__(self):
        return self.thing
    async def __aexit__(self, *exc_info):
        await self.thing.aclose()


class _RedirectStream(AbstractContextManager):

    _stream = None

    def __init__(self, new_target):
        self._new_target = new_target
        # We use a list of old targets to make this CM re-entrant
        self._old_targets = []

    def __enter__(self):
        self._old_targets.append(getattr(sys, self._stream))
        setattr(sys, self._stream, self._new_target)
        return self._new_target

    def __exit__(self, exctype, excinst, exctb):
        setattr(sys, self._stream, self._old_targets.pop())


class redirect_stdout(_RedirectStream):
    """Context manager for temporarily redirecting stdout to another file.

        # How to send help() to stderr
        with redirect_stdout(sys.stderr):
            help(dir)

        # How to write help() to a file
        with open('help.txt', 'w') as f:
            with redirect_stdout(f):
                help(pow)
    """

    _stream = "stdout"


class redirect_stderr(_RedirectStream):
    """Context manager for temporarily redirecting stderr to another file."""

    _stream = "stderr"


class suppress(AbstractContextManager):
    """Context manager to suppress specified exceptions

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         with suppress(FileNotFoundError):
             os.remove(somefile)
         # Execution still resumes here if the file was already removed
    """

    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        # Unlike isinstance and issubclass, CPython exception handling
        # currently only looks at the concrete type hierarchy (ignoring
        # the instance and subclass checking hooks). While Guido considers
        # that a bug rather than a feature, it's a fairly hard one to fix
        # due to various internal implementation details. suppress provides
        # the simpler issubclass based semantics, rather than trying to
        # exactly reproduce the limitations of the CPython interpreter.
        #
        # See http://bugs.python.org/issue12029 for more details
        return exctype is not None and issubclass(exctype, self._exceptions)


class _BaseExitStack:
    """A base class for ExitStack and AsyncExitStack."""

    @staticmethod
    def _create_exit_wrapper(cm, cm_exit):
        return MethodType(cm_exit, cm)

    @staticmethod
    def _create_cb_wrapper(callback, /, *args, **kwds):
        def _exit_wrapper(exc_type, exc, tb):
            callback(*args, **kwds)
        return _exit_wrapper

    def __init__(self):
        self._exit_callbacks = deque()

    def pop_all(self):
        """Preserve the context stack by transferring it to a new instance."""
        new_stack = type(self)()
        new_stack._exit_callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        return new_stack

    def push(self, exit):
        """Registers a callback with the standard __exit__ method signature.

        Can suppress exceptions the same way __exit__ method can.
        Also accepts any object with an __exit__ method (registering a call
        to the method instead of the object itself).
        """
        # We use an unbound method rather than a bound method to follow
        # the standard lookup behaviour for special methods.
        _cb_type = type(exit)

        try:
            exit_method = _cb_type.__exit__
        except AttributeError:
            # Not a context manager, so assume it's a callable.
            self._push_exit_callback(exit)
        else:
            self._push_cm_exit(exit, exit_method)
        return exit  # Allow use as a decorator.

    def enter_context(self, cm):
        """Enters the supplied context manager.

        If successful, also pushes its __exit__ method as a callback and
        returns the result of the __enter__ method.
        """
        # We look up the special methods on the type to match the with
        # statement.
        _cm_type = type(cm)
        _exit = _cm_type.__exit__
        result = _cm_type.__enter__(cm)
        self._push_cm_exit(cm, _exit)
        return result

    def callback(self, callback, /, *args, **kwds):
        """Registers an arbitrary callback and arguments.

        Cannot suppress exceptions.
        """
        _exit_wrapper = self._create_cb_wrapper(callback, *args, **kwds)

        # We changed the signature, so using @wraps is not appropriate, but
        # setting __wrapped__ may still help with introspection.
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper)
        return callback  # Allow use as a decorator

    def _push_cm_exit(self, cm, cm_exit):
        """Helper to correctly register callbacks to __exit__ methods."""
        _exit_wrapper = self._create_exit_wrapper(cm, cm_exit)
        self._push_exit_callback(_exit_wrapper, True)

    def _push_exit_callback(self, callback, is_sync=True):
        self._exit_callbacks.append((is_sync, callback))


# Inspired by discussions on http://bugs.python.org/issue13585
class ExitStack(_BaseExitStack, AbstractContextManager):
    """Context manager for dynamic management of a stack of exit callbacks.

    For example:
        with ExitStack() as stack:
            files = [stack.enter_context(open(fname)) for fname in filenames]
            # All opened files will automatically be closed at the end of
            # the with statement, even if attempts to open files later
            # in the list raise an exception.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        received_exc = exc_details[0] is not None

        # We manipulate the exception state so it behaves as though
        # we were actually nesting multiple with statements
        frame_exc = sys.exc_info()[1]
        def _fix_exception_context(new_exc, old_exc):
            # Context may not be correct, so find the end of the chain
            while 1:
                exc_context = new_exc.__context__
                if exc_context is None or exc_context is old_exc:
                    # Context is already set correctly (see issue 20317)
                    return
                if exc_context is frame_exc:
                    break
                new_exc = exc_context
            # Change the end of the chain to point to the exception
            # we expect it to reference
            new_exc.__context__ = old_exc

        # Callbacks are invoked in LIFO order to match the behaviour of
        # nested context managers
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            is_sync, cb = self._exit_callbacks.pop()
            assert is_sync
            try:
                if cb(*exc_details):
                    suppressed_exc = True
                    pending_raise = False
                    exc_details = (None, None, None)
            except:
                new_exc_details = sys.exc_info()
                # simulate the stack of exceptions by setting the context
                _fix_exception_context(new_exc_details[1], exc_details[1])
                pending_raise = True
                exc_details = new_exc_details
        if pending_raise:
            try:
                # bare "raise exc_details[1]" replaces our carefully
                # set-up context
                fixed_ctx = exc_details[1].__context__
                raise exc_details[1]
            except BaseException:
                exc_details[1].__context__ = fixed_ctx
                raise
        return received_exc and suppressed_exc

    def close(self):
        """Immediately unwind the context stack."""
        self.__exit__(None, None, None)


# Inspired by discussions on https://bugs.python.org/issue29302
class AsyncExitStack(_BaseExitStack, AbstractAsyncContextManager):
    """Async context manager for dynamic management of a stack of exit
    callbacks.

    For example:
        async with AsyncExitStack() as stack:
            connections = [await stack.enter_async_context(get_connection())
                for i in range(5)]
            # All opened connections will automatically be released at the
            # end of the async with statement, even if attempts to open a
            # connection later in the list raise an exception.
    """

    @staticmethod
    def _create_async_exit_wrapper(cm, cm_exit):
        return MethodType(cm_exit, cm)

    @staticmethod
    def _create_async_cb_wrapper(callback, /, *args, **kwds):
        async def _exit_wrapper(exc_type, exc, tb):
            await callback(*args, **kwds)
        return _exit_wrapper

    async def enter_async_context(self, cm):
        """Enters the supplied async context manager.

        If successful, also pushes its __aexit__ method as a callback and
        returns the result of the __aenter__ method.
        """
        _cm_type = type(cm)
        _exit = _cm_type.__aexit__
        result = await _cm_type.__aenter__(cm)
        self._push_async_cm_exit(cm, _exit)
        return result

    def push_async_exit(self, exit):
        """Registers a coroutine function with the standard __aexit__ method
        signature.

        Can suppress exceptions the same way __aexit__ method can.
        Also accepts any object with an __aexit__ method (registering a call
        to the method instead of the object itself).
        """
        _cb_type = type(exit)
        try:
            exit_method = _cb_type.__aexit__
        except AttributeError:
            # Not an async context manager, so assume it's a coroutine function
            self._push_exit_callback(exit, False)
        else:
            self._push_async_cm_exit(exit, exit_method)
        return exit  # Allow use as a decorator

    def push_async_callback(self, callback, /, *args, **kwds):
        """Registers an arbitrary coroutine function and arguments.

        Cannot suppress exceptions.
        """
        _exit_wrapper = self._create_async_cb_wrapper(callback, *args, **kwds)

        # We changed the signature, so using @wraps is not appropriate, but
        # setting __wrapped__ may still help with introspection.
        _exit_wrapper.__wrapped__ = callback
        self._push_exit_callback(_exit_wrapper, False)
        return callback  # Allow use as a decorator

    async def aclose(self):
        """Immediately unwind the context stack."""
        await self.__aexit__(None, None, None)

    def _push_async_cm_exit(self, cm, cm_exit):
        """Helper to correctly register coroutine function to __aexit__
        method."""
        _exit_wrapper = self._create_async_exit_wrapper(cm, cm_exit)
        self._push_exit_callback(_exit_wrapper, False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_details):
        received_exc = exc_details[0] is not None

        # We manipulate the exception state so it behaves as though
        # we were actually nesting multiple with statements
        frame_exc = sys.exc_info()[1]
        def _fix_exception_context(new_exc, old_exc):
            # Context may not be correct, so find the end of the chain
            while 1:
                exc_context = new_exc.__context__
                if exc_context is None or exc_context is old_exc:
                    # Context is already set correctly (see issue 20317)
                    return
                if exc_context is frame_exc:
                    break
                new_exc = exc_context
            # Change the end of the chain to point to the exception
            # we expect it to reference
            new_exc.__context__ = old_exc

        # Callbacks are invoked in LIFO order to match the behaviour of
        # nested context managers
        suppressed_exc = False
        pending_raise = False
        while self._exit_callbacks:
            is_sync, cb = self._exit_callbacks.pop()
            try:
                if is_sync:
                    cb_suppress = cb(*exc_details)
                else:
                    cb_suppress = await cb(*exc_details)

                if cb_suppress:
                    suppressed_exc = True
                    pending_raise = False
                    exc_details = (None, None, None)
            except:
                new_exc_details = sys.exc_info()
                # simulate the stack of exceptions by setting the context
                _fix_exception_context(new_exc_details[1], exc_details[1])
                pending_raise = True
                exc_details = new_exc_details
        if pending_raise:
            try:
                # bare "raise exc_details[1]" replaces our carefully
                # set-up context
                fixed_ctx = exc_details[1].__context__
                raise exc_details[1]
            except BaseException:
                exc_details[1].__context__ = fixed_ctx
                raise
        return received_exc and suppressed_exc


class nullcontext(AbstractContextManager, AbstractAsyncContextManager):
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass

    async def __aenter__(self):
        return self.enter_result

    async def __aexit__(self, *excinfo):
        pass
<|endoftext|>"""Utilities to get a password and/or the current user name.

getpass(prompt[, stream]) - Prompt for a password, with echo turned off.
getuser() - Get the user name from the environment or password database.

GetPassWarning - This UserWarning is issued when getpass() cannot prevent
                 echoing of the password contents while reading.

On Windows, the msvcrt module will be used.

"""

# Authors: Piers Lauder (original)
#          Guido van Rossum (Windows support and cleanup)
#          Gregory P. Smith (tty support & GetPassWarning)

import contextlib
import io
import os
import sys
import warnings

__all__ = ["getpass","getuser","GetPassWarning"]


class GetPassWarning(UserWarning): pass


def unix_getpass(prompt='Password: ', stream=None):
    """Prompt for a password, with echo turned off.

    Args:
      prompt: Written on stream to ask for the input.  Default: 'Password: '
      stream: A writable file object to display the prompt.  Defaults to
              the tty.  If no tty is available defaults to sys.stderr.
    Returns:
      The seKr3t input.
    Raises:
      EOFError: If our input tty or stdin was closed.
      GetPassWarning: When we were unable to turn echo off on the input.

    Always restores terminal settings before returning.
    """
    passwd = None
    with contextlib.ExitStack() as stack:
        try:
            # Always try reading and writing directly on the tty first.
            fd = os.open('/dev/tty', os.O_RDWR|os.O_NOCTTY)
            tty = io.FileIO(fd, 'w+')
            stack.enter_context(tty)
            input = io.TextIOWrapper(tty)
            stack.enter_context(input)
            if not stream:
                stream = input
        except OSError:
            # If that fails, see if stdin can be controlled.
            stack.close()
            try:
                fd = sys.stdin.fileno()
            except (AttributeError, ValueError):
                fd = None
                passwd = fallback_getpass(prompt, stream)
            input = sys.stdin
            if not stream:
                stream = sys.stderr

        if fd is not None:
            try:
                old = termios.tcgetattr(fd)     # a copy to save
                new = old[:]
                new[3] &= ~termios.ECHO  # 3 == 'lflags'
                tcsetattr_flags = termios.TCSAFLUSH
                if hasattr(termios, 'TCSASOFT'):
                    tcsetattr_flags |= termios.TCSASOFT
                try:
                    termios.tcsetattr(fd, tcsetattr_flags, new)
                    passwd = _raw_input(prompt, stream, input=input)
                finally:
                    termios.tcsetattr(fd, tcsetattr_flags, old)
                    stream.flush()  # issue7208
            except termios.error:
                if passwd is not None:
                    # _raw_input succeeded.  The final tcsetattr failed.  Reraise
                    # instead of leaving the terminal in an unknown state.
                    raise
                # We can't control the tty or stdin.  Give up and use normal IO.
                # fallback_getpass() raises an appropriate warning.
                if stream is not input:
                    # clean up unused file objects before blocking
                    stack.close()
                passwd = fallback_getpass(prompt, stream)

        stream.write('\n')
        return passwd


def win_getpass(prompt='Password: ', stream=None):
    """Prompt for password with echo off, using Windows getwch()."""
    if sys.stdin is not sys.__stdin__:
        return fallback_getpass(prompt, stream)

    for c in prompt:
        msvcrt.putwch(c)
    pw = ""
    while 1:
        c = msvcrt.getwch()
        if c == '\r' or c == '\n':
            break
        if c == '\003':
            raise KeyboardInterrupt
        if c == '\b':
            pw = pw[:-1]
        else:
            pw = pw + c
    msvcrt.putwch('\r')
    msvcrt.putwch('\n')
    return pw


def fallback_getpass(prompt='Password: ', stream=None):
    warnings.warn("Can not control echo on the terminal.", GetPassWarning,
                  stacklevel=2)
    if not stream:
        stream = sys.stderr
    print("Warning: Password input may be echoed.", file=stream)
    return _raw_input(prompt, stream)


def _raw_input(prompt="", stream=None, input=None):
    # This doesn't save the string in the GNU readline history.
    if not stream:
        stream = sys.stderr
    if not input:
        input = sys.stdin
    prompt = str(prompt)
    if prompt:
        try:
            stream.write(prompt)
        except UnicodeEncodeError:
            # Use replace error handler to get as much as possible printed.
            prompt = prompt.encode(stream.encoding, 'replace')
            prompt = prompt.decode(stream.encoding)
            stream.write(prompt)
        stream.flush()
    # NOTE: The Python C API calls flockfile() (and unlock) during readline.
    line = input.readline()
    if not line:
        raise EOFError
    if line[-1] == '\n':
        line = line[:-1]
    return line


def getuser():
    """Get the username from the environment or password database.

    First try various environment variables, then the password
    database.  This works on Windows as long as USERNAME is set.

    """

    for name in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        user = os.environ.get(name)
        if user:
            return user

    # If this fails, the exception will "explain" why
    import pwd
    return pwd.getpwuid(os.getuid())[0]

# Bind the name getpass to the appropriate function
try:
    import termios
    # it's possible there is an incompatible termios from the
    # McMillan Installer, make sure we have a UNIX-compatible termios
    termios.tcgetattr, termios.tcsetattr
except (ImportError, AttributeError):
    try:
        import msvcrt
    except ImportError:
        getpass = fallback_getpass
    else:
        getpass = win_getpass
else:
    getpass = unix_getpass
<|endoftext|>"""Access to Python's configuration information."""

import os
import sys
from os.path import pardir, realpath

__all__ = [
    'get_config_h_filename',
    'get_config_var',
    'get_config_vars',
    'get_makefile_filename',
    'get_path',
    'get_path_names',
    'get_paths',
    'get_platform',
    'get_python_version',
    'get_scheme_names',
    'parse_config_h',
]

# Keys for get_config_var() that are never converted to Python integers.
_ALWAYS_STR = {
    'MACOSX_DEPLOYMENT_TARGET',
}

_INSTALL_SCHEMES = {
    'posix_prefix': {
        'stdlib': '{installed_base}/{platlibdir}/python{py_version_short}',
        'platstdlib': '{platbase}/{platlibdir}/python{py_version_short}',
        'purelib': '{base}/lib/python{py_version_short}/site-packages',
        'platlib': '{platbase}/{platlibdir}/python{py_version_short}/site-packages',
        'include':
            '{installed_base}/include/python{py_version_short}{abiflags}',
        'platinclude':
            '{installed_platbase}/include/python{py_version_short}{abiflags}',
        'scripts': '{base}/bin',
        'data': '{base}',
        },
    'deb_system': {
        'stdlib': '{installed_base}/lib/python{py_version_short}',
        'platstdlib': '{platbase}/lib/python{py_version_short}',
        'purelib': '{base}/lib/python3/dist-packages',
        'platlib': '{platbase}/lib/python3/dist-packages',
        'include':
            '{installed_base}/include/python{py_version_short}{abiflags}',
        'platinclude':
            '{installed_platbase}/include/python{py_version_short}{abiflags}',
        'scripts': '{base}/bin',
        'data': '{base}',
        },
    'posix_local': {
        'stdlib': '{installed_base}/lib/python{py_version_short}',
        'platstdlib': '{platbase}/lib/python{py_version_short}',
        'purelib': '{base}/local/lib/python{py_version_short}/dist-packages',
        'platlib': '{platbase}/local/lib/python{py_version_short}/dist-packages',
        'include':
            '{installed_base}/include/python{py_version_short}{abiflags}',
        'platinclude':
            '{installed_platbase}/include/python{py_version_short}{abiflags}',
        'scripts': '{base}/local/bin',
        'data': '{base}/local',
        },
    'posix_home': {
        'stdlib': '{installed_base}/lib/python',
        'platstdlib': '{base}/lib/python',
        'purelib': '{base}/lib/python',
        'platlib': '{base}/lib/python',
        'include': '{installed_base}/include/python',
        'platinclude': '{installed_base}/include/python',
        'scripts': '{base}/bin',
        'data': '{base}',
        },
    'nt': {
        'stdlib': '{installed_base}/Lib',
        'platstdlib': '{base}/Lib',
        'purelib': '{base}/Lib/site-packages',
        'platlib': '{base}/Lib/site-packages',
        'include': '{installed_base}/Include',
        'platinclude': '{installed_base}/Include',
        'scripts': '{base}/Scripts',
        'data': '{base}',
        },
    }


# NOTE: site.py has copy of this function.
# Sync it when modify this function.
def _getuserbase():
    env_base = os.environ.get("PYTHONUSERBASE", None)
    if env_base:
        return env_base

    # VxWorks has no home directories
    if sys.platform == "vxworks":
        return None

    def joinuser(*args):
        return os.path.expanduser(os.path.join(*args))

    if os.name == "nt":
        base = os.environ.get("APPDATA") or "~"
        return joinuser(base, "Python")

    if sys.platform == "darwin" and sys._framework:
        return joinuser("~", "Library", sys._framework,
                        f"{sys.version_info[0]}.{sys.version_info[1]}")

    return joinuser("~", ".local")

_HAS_USER_BASE = (_getuserbase() is not None)

if _HAS_USER_BASE:
    _INSTALL_SCHEMES |= {
        # NOTE: When modifying "purelib" scheme, update site._get_path() too.
        'nt_user': {
            'stdlib': '{userbase}/Python{py_version_nodot_plat}',
            'platstdlib': '{userbase}/Python{py_version_nodot_plat}',
            'purelib': '{userbase}/Python{py_version_nodot_plat}/site-packages',
            'platlib': '{userbase}/Python{py_version_nodot_plat}/site-packages',
            'include': '{userbase}/Python{py_version_nodot_plat}/Include',
            'scripts': '{userbase}/Python{py_version_nodot_plat}/Scripts',
            'data': '{userbase}',
            },
        'posix_user': {
            'stdlib': '{userbase}/{platlibdir}/python{py_version_short}',
            'platstdlib': '{userbase}/{platlibdir}/python{py_version_short}',
            'purelib': '{userbase}/lib/python{py_version_short}/site-packages',
            'platlib': '{userbase}/lib/python{py_version_short}/site-packages',
            'include': '{userbase}/include/python{py_version_short}',
            'scripts': '{userbase}/bin',
            'data': '{userbase}',
            },
        'osx_framework_user': {
            'stdlib': '{userbase}/lib/python',
            'platstdlib': '{userbase}/lib/python',
            'purelib': '{userbase}/lib/python/site-packages',
            'platlib': '{userbase}/lib/python/site-packages',
            'include': '{userbase}/include/python{py_version_short}',
            'scripts': '{userbase}/bin',
            'data': '{userbase}',
            },
    }

_SCHEME_KEYS = ('stdlib', 'platstdlib', 'purelib', 'platlib', 'include',
                'scripts', 'data')

_PY_VERSION = sys.version.split()[0]
_PY_VERSION_SHORT = f'{sys.version_info[0]}.{sys.version_info[1]}'
_PY_VERSION_SHORT_NO_DOT = f'{sys.version_info[0]}{sys.version_info[1]}'
_PREFIX = os.path.normpath(sys.prefix)
_BASE_PREFIX = os.path.normpath(sys.base_prefix)
_EXEC_PREFIX = os.path.normpath(sys.exec_prefix)
_BASE_EXEC_PREFIX = os.path.normpath(sys.base_exec_prefix)
_CONFIG_VARS = None
_USER_BASE = None

# Regexes needed for parsing Makefile (and similar syntaxes,
# like old-style Setup files).
_variable_rx = r"([a-zA-Z][a-zA-Z0-9_]+)\s*=\s*(.*)"
_findvar1_rx = r"\$\(([A-Za-z][A-Za-z0-9_]*)\)"
_findvar2_rx = r"\${([A-Za-z][A-Za-z0-9_]*)}"


def _safe_realpath(path):
    try:
        return realpath(path)
    except OSError:
        return path

if sys.executable:
    _PROJECT_BASE = os.path.dirname(_safe_realpath(sys.executable))
else:
    # sys.executable can be empty if argv[0] has been changed and Python is
    # unable to retrieve the real program name
    _PROJECT_BASE = _safe_realpath(os.getcwd())

if (os.name == 'nt' and
    _PROJECT_BASE.lower().endswith(('\\pcbuild\\win32', '\\pcbuild\\amd64'))):
    _PROJECT_BASE = _safe_realpath(os.path.join(_PROJECT_BASE, pardir, pardir))

# set for cross builds
if "_PYTHON_PROJECT_BASE" in os.environ:
    _PROJECT_BASE = _safe_realpath(os.environ["_PYTHON_PROJECT_BASE"])

def _is_python_source_dir(d):
    for fn in ("Setup", "Setup.local"):
        if os.path.isfile(os.path.join(d, "Modules", fn)):
            return True
    return False

_sys_home = getattr(sys, '_home', None)

if os.name == 'nt':
    def _fix_pcbuild(d):
        if d and os.path.normcase(d).startswith(
                os.path.normcase(os.path.join(_PREFIX, "PCbuild"))):
            return _PREFIX
        return d
    _PROJECT_BASE = _fix_pcbuild(_PROJECT_BASE)
    _sys_home = _fix_pcbuild(_sys_home)

def is_python_build(check_home=False):
    if check_home and _sys_home:
        return _is_python_source_dir(_sys_home)
    return _is_python_source_dir(_PROJECT_BASE)

_PYTHON_BUILD = is_python_build(True)

if _PYTHON_BUILD:
    for scheme in ('posix_prefix', 'posix_home', 'posix_local', 'deb_system'):
        # On POSIX-y platforms, Python will:
        # - Build from .h files in 'headers' (which is only added to the
        #   scheme when building CPython)
        # - Install .h files to 'include'
        scheme = _INSTALL_SCHEMES[scheme]
        scheme['headers'] = scheme['include']
        scheme['include'] = '{srcdir}/Include'
        scheme['platinclude'] = '{projectbase}/.'


def _subst_vars(s, local_vars):
    try:
        return s.format(**local_vars)
    except KeyError as var:
        try:
            return s.format(**os.environ)
        except KeyError:
            raise AttributeError(f'{var}') from None

def _extend_dict(target_dict, other_dict):
    target_keys = target_dict.keys()
    for key, value in other_dict.items():
        if key in target_keys:
            continue
        target_dict[key] = value


def _expand_vars(scheme, vars):
    res = {}
    if vars is None:
        vars = {}
    _extend_dict(vars, get_config_vars())

    for key, value in _INSTALL_SCHEMES[scheme].items():
        if os.name in ('posix', 'nt'):
            value = os.path.expanduser(value)
        res[key] = os.path.normpath(_subst_vars(value, vars))
    return res


def _get_preferred_schemes():
    if os.name == 'nt':
        return {
            'prefix': 'nt',
            'home': 'posix_home',
            'user': 'nt_user',
        }
    if sys.platform == 'darwin' and sys._framework:
        return {
            'prefix': 'posix_prefix',
            'home': 'posix_home',
            'user': 'osx_framework_user',
        }

    if sys.base_prefix != sys.prefix or hasattr(sys, "real_prefix"):
        # virtual environments
        prefix_scheme = 'posix_prefix'
    else:
        # default to /usr for package builds, /usr/local otherwise
        deb_build = os.environ.get('DEB_PYTHON_INSTALL_LAYOUT', 'posix_local')
        if deb_build in ('deb', 'deb_system'):
            prefix_scheme = 'deb_system'
        else:
            prefix_scheme = 'posix_local'

    return {
        'prefix': prefix_scheme,
        'home': 'posix_home',
        'user': 'posix_user',
    }


def get_preferred_scheme(key):
    scheme = _get_preferred_schemes()[key]
    if scheme not in _INSTALL_SCHEMES:
        raise ValueError(
            f"{key!r} returned {scheme!r}, which is not a valid scheme "
            f"on this platform"
        )
    return scheme


def get_default_scheme():
    return get_preferred_scheme('prefix')


def _parse_makefile(filename, vars=None, keep_unresolved=True):
    """Parse a Makefile-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    import re

    if vars is None:
        vars = {}
    done = {}
    notdone = {}

    with open(filename, encoding=sys.getfilesystemencoding(),
              errors="surrogateescape") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        m = re.match(_variable_rx, line)
        if m:
            n, v = m.group(1, 2)
            v = v.strip()
            # `$$' is a literal `$' in make
            tmpv = v.replace('$$', '')

            if "$" in tmpv:
                notdone[n] = v
            else:
                try:
                    if n in _ALWAYS_STR:
                        raise ValueError

                    v = int(v)
                except ValueError:
                    # insert literal `$'
                    done[n] = v.replace('$$', '$')
                else:
                    done[n] = v

    # do variable interpolation here
    variables = list(notdone.keys())

    # Variables with a 'PY_' prefix in the makefile. These need to
    # be made available without that prefix through sysconfig.
    # Special care is needed to ensure that variable expansion works, even
    # if the expansion uses the name without a prefix.
    renamed_variables = ('CFLAGS', 'LDFLAGS', 'CPPFLAGS')

    while len(variables) > 0:
        for name in tuple(variables):
            value = notdone[name]
            m1 = re.search(_findvar1_rx, value)
            m2 = re.search(_findvar2_rx, value)
            if m1 and m2:
                m = m1 if m1.start() < m2.start() else m2
            else:
                m = m1 if m1 else m2
            if m is not None:
                n = m.group(1)
                found = True
                if n in done:
                    item = str(done[n])
                elif n in notdone:
                    # get it on a subsequent round
                    found = False
                elif n in os.environ:
                    # do it like make: fall back to environment
                    item = os.environ[n]

                elif n in renamed_variables:
                    if (name.startswith('PY_') and
                        name[3:] in renamed_variables):
                        item = ""

                    elif 'PY_' + n in notdone:
                        found = False

                    else:
                        item = str(done['PY_' + n])

                else:
                    done[n] = item = ""

                if found:
                    after = value[m.end():]
                    value = value[:m.start()] + item + after
                    if "$" in after:
                        notdone[name] = value
                    else:
                        try:
                            if name in _ALWAYS_STR:
                                raise ValueError
                            value = int(value)
                        except ValueError:
                            done[name] = value.strip()
                        else:
                            done[name] = value
                        variables.remove(name)

                        if name.startswith('PY_') \
                        and name[3:] in renamed_variables:

                            name = name[3:]
                            if name not in done:
                                done[name] = value

            else:
                # Adds unresolved variables to the done dict.
                # This is disabled when called from distutils.sysconfig
                if keep_unresolved:
                    done[name] = value
                # bogus variable reference (e.g. "prefix=$/opt/python");
                # just drop it since we can't deal
                variables.remove(name)

    # strip spurious spaces
    for k, v in done.items():
        if isinstance(v, str):
            done[k] = v.strip()

    # save the results in the global dictionary
    vars.update(done)
    return vars


def get_makefile_filename():
    """Return the path of the Makefile."""
    if _PYTHON_BUILD:
        return os.path.join(_sys_home or _PROJECT_BASE, "Makefile")
    if hasattr(sys, 'abiflags'):
        config_dir_name = f'config-{_PY_VERSION_SHORT}{sys.abiflags}'
    else:
        config_dir_name = 'config'
    if hasattr(sys.implementation, '_multiarch'):
        config_dir_name += f'-{sys.implementation._multiarch}'
    return os.path.join(get_path('stdlib'), config_dir_name, 'Makefile')


def _get_sysconfigdata_name():
    multiarch = getattr(sys.implementation, '_multiarch', '')
    return os.environ.get(
        '_PYTHON_SYSCONFIGDATA_NAME',
        f'_sysconfigdata_{sys.abiflags}_{multiarch}',
    )


def _generate_posix_vars():
    """Generate the Python module containing build-time variables."""
    import pprint
    vars = {}
    # load the installed Makefile:
    makefile = get_makefile_filename()
    try:
        _parse_makefile(makefile, vars)
    except OSError as e:
        msg = f"invalid Python installation: unable to open {makefile}"
        if hasattr(e, "strerror"):
            msg = f"{msg} ({e.strerror})"
        raise OSError(msg)
    # load the installed pyconfig.h:
    config_h = get_config_h_filename()
    try:
        with open(config_h, encoding="utf-8") as f:
            parse_config_h(f, vars)
    except OSError as e:
        msg = f"invalid Python installation: unable to open {config_h}"
        if hasattr(e, "strerror"):
            msg = f"{msg} ({e.strerror})"
        raise OSError(msg)
    # On AIX, there are wrong paths to the linker scripts in the Makefile
    # -- these paths are relative to the Python source, but when installed
    # the scripts are in another directory.
    if _PYTHON_BUILD:
        vars['BLDSHARED'] = vars['LDSHARED']

    # There's a chicken-and-egg situation on OS X with regards to the
    # _sysconfigdata module after the changes introduced by #15298:
    # get_config_vars() is called by get_platform() as part of the
    # `make pybuilddir.txt` target -- which is a precursor to the
    # _sysconfigdata.py module being constructed.  Unfortunately,
    # get_config_vars() eventually calls _init_posix(), which attempts
    # to import _sysconfigdata, which we won't have built yet.  In order
    # for _init_posix() to work, if we're on Darwin, just mock up the
    # _sysconfigdata module manually and populate it with the build vars.
    # This is more than sufficient for ensuring the subsequent call to
    # get_platform() succeeds.
    name = _get_sysconfigdata_name()
    if 'darwin' in sys.platform:
        import types
        module = types.ModuleType(name)
        module.build_time_vars = vars
        sys.modules[name] = module

    pybuilddir = f'build/lib.{get_platform()}-{_PY_VERSION_SHORT}'
    if hasattr(sys, "gettotalrefcount"):
        pybuilddir += '-pydebug'
    os.makedirs(pybuilddir, exist_ok=True)
    destfile = os.path.join(pybuilddir, name + '.py')

    with open(destfile, 'w', encoding='utf8') as f:
        f.write('# system configuration generated and used by'
                ' the sysconfig module\n')
        f.write('build_time_vars = ')
        pprint.pprint(vars, stream=f)

    # Create file used for sys.path fixup -- see Modules/getpath.c
    with open('pybuilddir.txt', 'w', encoding='utf8') as f:
        f.write(pybuilddir)

def _init_posix(vars):
    """Initialize the module as appropriate for POSIX systems."""
    # _sysconfigdata is generated at build time, see _generate_posix_vars()
    name = _get_sysconfigdata_name()
    _temp = __import__(name, globals(), locals(), ['build_time_vars'], 0)
    build_time_vars = _temp.build_time_vars
    vars.update(build_time_vars)

def _init_non_posix(vars):
    """Initialize the module as appropriate for NT"""
    # set basic install directories
    import _imp
    vars['LIBDEST'] = get_path('stdlib')
    vars['BINLIBDEST'] = get_path('platstdlib')
    vars['INCLUDEPY'] = get_path('include')
    vars['EXT_SUFFIX'] = _imp.extension_suffixes()[0]
    vars['EXE'] = '.exe'
    vars['VERSION'] = _PY_VERSION_SHORT_NO_DOT
    vars['BINDIR'] = os.path.dirname(_safe_realpath(sys.executable))
    vars['TZPATH'] = ''

#
# public APIs
#


def parse_config_h(fp, vars=None):
    """Parse a config.h-style file.

    A dictionary containing name/value pairs is returned.  If an
    optional dictionary is passed in as the second argument, it is
    used instead of a new dictionary.
    """
    if vars is None:
        vars = {}
    import re
    define_rx = re.compile("#define ([A-Z][A-Za-z0-9_]+) (.*)\n")
    undef_rx = re.compile("/[*] #undef ([A-Z][A-Za-z0-9_]+) [*]/\n")

    while True:
        line = fp.readline()
        if not line:
            break
        m = define_rx.match(line)
        if m:
            n, v = m.group(1, 2)
            try:
                if n in _ALWAYS_STR:
                    raise ValueError
                v = int(v)
            except ValueError:
                pass
            vars[n] = v
        else:
            m = undef_rx.match(line)
            if m:
                vars[m.group(1)] = 0
    return vars


def get_config_h_filename():
    """Return the path of pyconfig.h."""
    if _PYTHON_BUILD:
        if os.name == "nt":
            inc_dir = os.path.join(_sys_home or _PROJECT_BASE, "PC")
        else:
            inc_dir = _sys_home or _PROJECT_BASE
    else:
        inc_dir = get_path('platinclude', 'posix_prefix')
    return os.path.join(inc_dir, 'pyconfig.h')


def get_scheme_names():
    """Return a tuple containing the schemes names."""
    return tuple(sorted(_INSTALL_SCHEMES))


def get_path_names():
    """Return a tuple containing the paths names."""
    return _SCHEME_KEYS


def get_paths(scheme=get_default_scheme(), vars=None, expand=True):
    """Return a mapping containing an install scheme.

    ``scheme`` is the install scheme name. If not provided, it will
    return the default scheme for the current platform.
    """
    if expand:
        return _expand_vars(scheme, vars)
    else:
        return _INSTALL_SCHEMES[scheme]


def get_path(name, scheme=get_default_scheme(), vars=None, expand=True):
    """Return a path corresponding to the scheme.

    ``scheme`` is the install scheme name.
    """
    return get_paths(scheme, vars, expand)[name]


def get_config_vars(*args):
    """With no arguments, return a dictionary of all configuration
    variables relevant for the current platform.

    On Unix, this means every variable defined in Python's installed Makefile;
    On Windows it's a much smaller set.

    With arguments, return a list of values that result from looking up
    each argument in the configuration variable dictionary.
    """
    global _CONFIG_VARS
    if _CONFIG_VARS is None:
        _CONFIG_VARS = {}
        # Normalized versions of prefix and exec_prefix are handy to have;
        # in fact, these are the standard versions used most places in the
        # Distutils.
        _CONFIG_VARS['prefix'] = _PREFIX
        _CONFIG_VARS['exec_prefix'] = _EXEC_PREFIX
        _CONFIG_VARS['py_version'] = _PY_VERSION
        _CONFIG_VARS['py_version_short'] = _PY_VERSION_SHORT
        _CONFIG_VARS['py_version_nodot'] = _PY_VERSION_SHORT_NO_DOT
        _CONFIG_VARS['installed_base'] = _BASE_PREFIX
        _CONFIG_VARS['base'] = _PREFIX
        _CONFIG_VARS['installed_platbase'] = _BASE_EXEC_PREFIX
        _CONFIG_VARS['platbase'] = _EXEC_PREFIX
        _CONFIG_VARS['projectbase'] = _PROJECT_BASE
        _CONFIG_VARS['platlibdir'] = sys.platlibdir
        try:
            _CONFIG_VARS['abiflags'] = sys.abiflags
        except AttributeError:
            # sys.abiflags may not be defined on all platforms.
            _CONFIG_VARS['abiflags'] = ''
        try:
            _CONFIG_VARS['py_version_nodot_plat'] = sys.winver.replace('.', '')
        except AttributeError:
            _CONFIG_VARS['py_version_nodot_plat'] = ''

        if os.name == 'nt':
            _init_non_posix(_CONFIG_VARS)
        if os.name == 'posix':
            _init_posix(_CONFIG_VARS)
        # For backward compatibility, see issue19555
        SO = _CONFIG_VARS.get('EXT_SUFFIX')
        if SO is not None:
            _CONFIG_VARS['SO'] = SO
        if _HAS_USER_BASE:
            # Setting 'userbase' is done below the call to the
            # init function to enable using 'get_config_var' in
            # the init-function.
            _CONFIG_VARS['userbase'] = _getuserbase()

        multiarch = get_config_var('MULTIARCH')
        if multiarch:
            _CONFIG_VARS['multiarchsubdir'] = '/' + multiarch
        else:
            _CONFIG_VARS['multiarchsubdir'] = ''

        # Always convert srcdir to an absolute path
        srcdir = _CONFIG_VARS.get('srcdir', _PROJECT_BASE)
        if os.name == 'posix':
            if _PYTHON_BUILD:
                # If srcdir is a relative path (typically '.' or '..')
                # then it should be interpreted relative to the directory
                # containing Makefile.
                base = os.path.dirname(get_makefile_filename())
                srcdir = os.path.join(base, srcdir)
            else:
                # srcdir is not meaningful since the installation is
                # spread about the filesystem.  We choose the
                # directory containing the Makefile since we know it
                # exists.
                srcdir = os.path.dirname(get_makefile_filename())
        _CONFIG_VARS['srcdir'] = _safe_realpath(srcdir)

        # OS X platforms require special customization to handle
        # multi-architecture, multi-os-version installers
        if sys.platform == 'darwin':
            import _osx_support
            _osx_support.customize_config_vars(_CONFIG_VARS)

    if args:
        vals = []
        for name in args:
            vals.append(_CONFIG_VARS.get(name))
        return vals
    else:
        return _CONFIG_VARS


def get_config_var(name):
    """Return the value of a single variable using the dictionary returned by
    'get_config_vars()'.

    Equivalent to get_config_vars().get(name)
    """
    if name == 'SO':
        import warnings
        warnings.warn('SO is deprecated, use EXT_SUFFIX', DeprecationWarning, 2)
    return get_config_vars().get(name)


def get_platform():
    """Return a string that identifies the current platform.

    This is used mainly to distinguish platform-specific build directories and
    platform-specific built distributions.  Typically includes the OS name and
    version and the architecture (as supplied by 'os.uname()'), although the
    exact information included depends on the OS; on Linux, the kernel version
    isn't particularly important.

    Examples of returned values:
       linux-i586
       linux-alpha (?)
       solaris-2.6-sun4u

    Windows will return one of:
       win-amd64 (64bit Windows on AMD64 (aka x86_64, Intel64, EM64T, etc)
       win32 (all others - specifically, sys.platform is returned)

    For other non-POSIX platforms, currently just returns 'sys.platform'.

    """
    if os.name == 'nt':
        if 'amd64' in sys.version.lower():
            return 'win-amd64'
        if '(arm)' in sys.version.lower():
            return 'win-arm32'
        if '(arm64)' in sys.version.lower():
            return 'win-arm64'
        return sys.platform

    if os.name != "posix" or not hasattr(os, 'uname'):
        # XXX what about the architecture? NT is Intel or Alpha
        return sys.platform

    # Set for cross builds explicitly
    if "_PYTHON_HOST_PLATFORM" in os.environ:
        return os.environ["_PYTHON_HOST_PLATFORM"]

    # Try to distinguish various flavours of Unix
    osname, host, release, version, machine = os.uname()

    # Convert the OS name to lowercase, remove '/' characters, and translate
    # spaces (for "Power Macintosh")
    osname = osname.lower().replace('/', '')
    machine = machine.replace(' ', '_')
    machine = machine.replace('/', '-')

    if osname[:5] == "linux":
        # At least on Linux/Intel, 'machine' is the processor --
        # i386, etc.
        # XXX what about Alpha, SPARC, etc?
        return  f"{osname}-{machine}"
    elif osname[:5] == "sunos":
        if release[0] >= "5":           # SunOS 5 == Solaris 2
            osname = "solaris"
            release = f"{int(release[0]) - 3}.{release[2:]}"
            # We can't use "platform.architecture()[0]" because a
            # bootstrap problem. We use a dict to get an error
            # if some suspicious happens.
            bitness = {2147483647:"32bit", 9223372036854775807:"64bit"}
            machine += f".{bitness[sys.maxsize]}"
        # fall through to standard osname-release-machine representation
    elif osname[:3] == "aix":
        from _aix_support import aix_platform
        return aix_platform()
    elif osname[:6] == "cygwin":
        osname = "cygwin"
        import re
        rel_re = re.compile(r'[\d.]+')
        m = rel_re.match(release)
        if m:
            release = m.group()
    elif osname[:6] == "darwin":
        import _osx_support
        osname, release, machine = _osx_support.get_platform_osx(
                                            get_config_vars(),
                                            osname, release, machine)

    return f"{osname}-{release}-{machine}"


def get_python_version():
    return _PY_VERSION_SHORT


def expand_makefile_vars(s, vars):
    """Expand Makefile-style variables -- "${foo}" or "$(foo)" -- in
    'string' according to 'vars' (a dictionary mapping variable names to
    values).  Variables not present in 'vars' are silently expanded to the
    empty string.  The variable values in 'vars' should not contain further
    variable expansions; if 'vars' is the output of 'parse_makefile()',
    you're fine.  Returns a variable-expanded version of 's'.
    """
    import re

    # This algorithm does multiple expansion, so if vars['foo'] contains
    # "${bar}", it will expand ${foo} to ${bar}, and then expand
    # ${bar}... and so forth.  This is fine as long as 'vars' comes from
    # 'parse_makefile()', which takes care of such expansions eagerly,
    # according to make's variable expansion semantics.

    while True:
        m = re.search(_findvar1_rx, s) or re.search(_findvar2_rx, s)
        if m:
            (beg, end) = m.span()
            s = s[0:beg] + vars.get(m.group(1)) + s[end:]
        else:
            break
    return s


def _print_dict(title, data):
    for index, (key, value) in enumerate(sorted(data.items())):
        if index == 0:
            print(f'{title}: ')
        print(f'\t{key} = "{value}"')


def _main():
    """Display all information sysconfig detains."""
    if '--generate-posix-vars' in sys.argv:
        _generate_posix_vars()
        return
    print(f'Platform: "{get_platform()}"')
    print(f'Python version: "{get_python_version()}"')
    print(f'Current installation scheme: "{get_default_scheme()}"')
    print()
    _print_dict('Paths', get_paths())
    print()
    _print_dict('Variables', get_config_vars())


if __name__ == '__main__':
    _main()
<|endoftext|>"""Mailcap file handling.  See RFC 1524."""

import os
import warnings
import re

__all__ = ["getcaps","findmatch"]


def lineno_sort_key(entry):
    # Sort in ascending order, with unspecified entries at the end
    if 'lineno' in entry:
        return 0, entry['lineno']
    else:
        return 1, 0

_find_unsafe = re.compile(r'[^\xa1-\U0010FFFF\w@+=:,./-]').search

class UnsafeMailcapInput(Warning):
    """Warning raised when refusing unsafe input"""


# Part 1: top-level interface.

def getcaps():
    """Return a dictionary containing the mailcap database.

    The dictionary maps a MIME type (in all lowercase, e.g. 'text/plain')
    to a list of dictionaries corresponding to mailcap entries.  The list
    collects all the entries for that MIME type from all available mailcap
    files.  Each dictionary contains key-value pairs for that MIME type,
    where the viewing command is stored with the key "view".

    """
    caps = {}
    lineno = 0
    for mailcap in listmailcapfiles():
        try:
            fp = open(mailcap, 'r')
        except OSError:
            continue
        with fp:
            morecaps, lineno = _readmailcapfile(fp, lineno)
        for key, value in morecaps.items():
            if not key in caps:
                caps[key] = value
            else:
                caps[key] = caps[key] + value
    return caps

def listmailcapfiles():
    """Return a list of all mailcap files found on the system."""
    # This is mostly a Unix thing, but we use the OS path separator anyway
    if 'MAILCAPS' in os.environ:
        pathstr = os.environ['MAILCAPS']
        mailcaps = pathstr.split(os.pathsep)
    else:
        if 'HOME' in os.environ:
            home = os.environ['HOME']
        else:
            # Don't bother with getpwuid()
            home = '.' # Last resort
        mailcaps = [home + '/.mailcap', '/etc/mailcap',
                '/usr/etc/mailcap', '/usr/local/etc/mailcap']
    return mailcaps


# Part 2: the parser.
def readmailcapfile(fp):
    """Read a mailcap file and return a dictionary keyed by MIME type."""
    warnings.warn('readmailcapfile is deprecated, use getcaps instead',
                  DeprecationWarning, 2)
    caps, _ = _readmailcapfile(fp, None)
    return caps


def _readmailcapfile(fp, lineno):
    """Read a mailcap file and return a dictionary keyed by MIME type.

    Each MIME type is mapped to an entry consisting of a list of
    dictionaries; the list will contain more than one such dictionary
    if a given MIME type appears more than once in the mailcap file.
    Each dictionary contains key-value pairs for that MIME type, where
    the viewing command is stored with the key "view".
    """
    caps = {}
    while 1:
        line = fp.readline()
        if not line: break
        # Ignore comments and blank lines
        if line[0] == '#' or line.strip() == '':
            continue
        nextline = line
        # Join continuation lines
        while nextline[-2:] == '\\\n':
            nextline = fp.readline()
            if not nextline: nextline = '\n'
            line = line[:-2] + nextline
        # Parse the line
        key, fields = parseline(line)
        if not (key and fields):
            continue
        if lineno is not None:
            fields['lineno'] = lineno
            lineno += 1
        # Normalize the key
        types = key.split('/')
        for j in range(len(types)):
            types[j] = types[j].strip()
        key = '/'.join(types).lower()
        # Update the database
        if key in caps:
            caps[key].append(fields)
        else:
            caps[key] = [fields]
    return caps, lineno

def parseline(line):
    """Parse one entry in a mailcap file and return a dictionary.

    The viewing command is stored as the value with the key "view",
    and the rest of the fields produce key-value pairs in the dict.
    """
    fields = []
    i, n = 0, len(line)
    while i < n:
        field, i = parsefield(line, i, n)
        fields.append(field)
        i = i+1 # Skip semicolon
    if len(fields) < 2:
        return None, None
    key, view, rest = fields[0], fields[1], fields[2:]
    fields = {'view': view}
    for field in rest:
        i = field.find('=')
        if i < 0:
            fkey = field
            fvalue = ""
        else:
            fkey = field[:i].strip()
            fvalue = field[i+1:].strip()
        if fkey in fields:
            # Ignore it
            pass
        else:
            fields[fkey] = fvalue
    return key, fields

def parsefield(line, i, n):
    """Separate one key-value pair in a mailcap entry."""
    start = i
    while i < n:
        c = line[i]
        if c == ';':
            break
        elif c == '\\':
            i = i+2
        else:
            i = i+1
    return line[start:i].strip(), i


# Part 3: using the database.

def findmatch(caps, MIMEtype, key='view', filename="/dev/null", plist=[]):
    """Find a match for a mailcap entry.

    Return a tuple containing the command line, and the mailcap entry
    used; (None, None) if no match is found.  This may invoke the
    'test' command of several matching entries before deciding which
    entry to use.

    """
    if _find_unsafe(filename):
        msg = "Refusing to use mailcap with filename %r. Use a safe temporary filename." % (filename,)
        warnings.warn(msg, UnsafeMailcapInput)
        return None, None
    entries = lookup(caps, MIMEtype, key)
    # XXX This code should somehow check for the needsterminal flag.
    for e in entries:
        if 'test' in e:
            test = subst(e['test'], filename, plist)
            if test is None:
                continue
            if test and os.system(test) != 0:
                continue
        command = subst(e[key], MIMEtype, filename, plist)
        if command is not None:
            return command, e
    return None, None

def lookup(caps, MIMEtype, key=None):
    entries = []
    if MIMEtype in caps:
        entries = entries + caps[MIMEtype]
    MIMEtypes = MIMEtype.split('/')
    MIMEtype = MIMEtypes[0] + '/*'
    if MIMEtype in caps:
        entries = entries + caps[MIMEtype]
    if key is not None:
        entries = [e for e in entries if key in e]
    entries = sorted(entries, key=lineno_sort_key)
    return entries

def subst(field, MIMEtype, filename, plist=[]):
    # XXX Actually, this is Unix-specific
    res = ''
    i, n = 0, len(field)
    while i < n:
        c = field[i]; i = i+1
        if c != '%':
            if c == '\\':
                c = field[i:i+1]; i = i+1
            res = res + c
        else:
            c = field[i]; i = i+1
            if c == '%':
                res = res + c
            elif c == 's':
                res = res + filename
            elif c == 't':
                if _find_unsafe(MIMEtype):
                    msg = "Refusing to substitute MIME type %r into a shell command." % (MIMEtype,)
                    warnings.warn(msg, UnsafeMailcapInput)
                    return None
                res = res + MIMEtype
            elif c == '{':
                start = i
                while i < n and field[i] != '}':
                    i = i+1
                name = field[start:i]
                i = i+1
                param = findparam(name, plist)
                if _find_unsafe(param):
                    msg = "Refusing to substitute parameter %r (%s) into a shell command" % (param, name)
                    warnings.warn(msg, UnsafeMailcapInput)
                    return None
                res = res + param
            # XXX To do:
            # %n == number of parts if type is multipart/*
            # %F == list of alternating type and filename for parts
            else:
                res = res + '%' + c
    return res

def findparam(name, plist):
    name = name.lower() + '='
    n = len(name)
    for p in plist:
        if p[:n].lower() == name:
            return p[n:]
    return ''


# Part 4: test program.

def test():
    import sys
    caps = getcaps()
    if not sys.argv[1:]:
        show(caps)
        return
    for i in range(1, len(sys.argv), 2):
        args = sys.argv[i:i+2]
        if len(args) < 2:
            print("usage: mailcap [MIMEtype file] ...")
            return
        MIMEtype = args[0]
        file = args[1]
        command, e = findmatch(caps, MIMEtype, 'view', file)
        if not command:
            print("No viewer found for", type)
        else:
            print("Executing:", command)
            sts = os.system(command)
            sts = os.waitstatus_to_exitcode(sts)
            if sts:
                print("Exit status:", sts)

def show(caps):
    print("Mailcap files:")
    for fn in listmailcapfiles(): print("\t" + fn)
    print()
    if not caps: caps = getcaps()
    print("Mailcap entries:")
    print()
    ckeys = sorted(caps)
    for type in ckeys:
        print(type)
        entries = caps[type]
        for e in entries:
            keys = sorted(e)
            for k in keys:
                print("  %-15s" % k, e[k])
            print()

if __name__ == '__main__':
    test()
<|endoftext|>"""Pseudo terminal utilities."""

# Bugs: No signal handling.  Doesn't set slave termios and window size.
#       Only tested on Linux, FreeBSD, and macOS.
# See:  W. Richard Stevens. 1992.  Advanced Programming in the
#       UNIX Environment.  Chapter 19.
# Author: Steen Lumholt -- with additions by Guido.

from select import select
import os
import sys
import tty

# names imported directly for test mocking purposes
from os import close, waitpid
from tty import setraw, tcgetattr, tcsetattr

__all__ = ["openpty", "fork", "spawn"]

STDIN_FILENO = 0
STDOUT_FILENO = 1
STDERR_FILENO = 2

CHILD = 0

def openpty():
    """openpty() -> (master_fd, slave_fd)
    Open a pty master/slave pair, using os.openpty() if possible."""

    try:
        return os.openpty()
    except (AttributeError, OSError):
        pass
    master_fd, slave_name = _open_terminal()
    slave_fd = slave_open(slave_name)
    return master_fd, slave_fd

def master_open():
    """master_open() -> (master_fd, slave_name)
    Open a pty master and return the fd, and the filename of the slave end.
    Deprecated, use openpty() instead."""

    try:
        master_fd, slave_fd = os.openpty()
    except (AttributeError, OSError):
        pass
    else:
        slave_name = os.ttyname(slave_fd)
        os.close(slave_fd)
        return master_fd, slave_name

    return _open_terminal()

def _open_terminal():
    """Open pty master and return (master_fd, tty_name)."""
    for x in 'pqrstuvwxyzPQRST':
        for y in '0123456789abcdef':
            pty_name = '/dev/pty' + x + y
            try:
                fd = os.open(pty_name, os.O_RDWR)
            except OSError:
                continue
            return (fd, '/dev/tty' + x + y)
    raise OSError('out of pty devices')

def slave_open(tty_name):
    """slave_open(tty_name) -> slave_fd
    Open the pty slave and acquire the controlling terminal, returning
    opened filedescriptor.
    Deprecated, use openpty() instead."""

    result = os.open(tty_name, os.O_RDWR)
    try:
        from fcntl import ioctl, I_PUSH
    except ImportError:
        return result
    try:
        ioctl(result, I_PUSH, "ptem")
        ioctl(result, I_PUSH, "ldterm")
    except OSError:
        pass
    return result

def fork():
    """fork() -> (pid, master_fd)
    Fork and make the child a session leader with a controlling terminal."""

    try:
        pid, fd = os.forkpty()
    except (AttributeError, OSError):
        pass
    else:
        if pid == CHILD:
            try:
                os.setsid()
            except OSError:
                # os.forkpty() already set us session leader
                pass
        return pid, fd

    master_fd, slave_fd = openpty()
    pid = os.fork()
    if pid == CHILD:
        # Establish a new session.
        os.setsid()
        os.close(master_fd)

        # Slave becomes stdin/stdout/stderr of child.
        os.dup2(slave_fd, STDIN_FILENO)
        os.dup2(slave_fd, STDOUT_FILENO)
        os.dup2(slave_fd, STDERR_FILENO)
        if slave_fd > STDERR_FILENO:
            os.close(slave_fd)

        # Explicitly open the tty to make it become a controlling tty.
        tmp_fd = os.open(os.ttyname(STDOUT_FILENO), os.O_RDWR)
        os.close(tmp_fd)
    else:
        os.close(slave_fd)

    # Parent and child process.
    return pid, master_fd

def _writen(fd, data):
    """Write all the data to a descriptor."""
    while data:
        n = os.write(fd, data)
        data = data[n:]

def _read(fd):
    """Default read function."""
    return os.read(fd, 1024)

def _copy(master_fd, master_read=_read, stdin_read=_read):
    """Parent copy loop.
    Copies
            pty master -> standard output   (master_read)
            standard input -> pty master    (stdin_read)"""
    fds = [master_fd, STDIN_FILENO]
    while fds:
        rfds, _wfds, _xfds = select(fds, [], [])

        if master_fd in rfds:
            # Some OSes signal EOF by returning an empty byte string,
            # some throw OSErrors.
            try:
                data = master_read(master_fd)
            except OSError:
                data = b""
            if not data:  # Reached EOF.
                return    # Assume the child process has exited and is
                          # unreachable, so we clean up.
            else:
                os.write(STDOUT_FILENO, data)

        if STDIN_FILENO in rfds:
            data = stdin_read(STDIN_FILENO)
            if not data:
                fds.remove(STDIN_FILENO)
            else:
                _writen(master_fd, data)

def spawn(argv, master_read=_read, stdin_read=_read):
    """Create a spawned process."""
    if type(argv) == type(''):
        argv = (argv,)
    sys.audit('pty.spawn', argv)

    pid, master_fd = fork()
    if pid == CHILD:
        os.execlp(argv[0], *argv)

    try:
        mode = tcgetattr(STDIN_FILENO)
        setraw(STDIN_FILENO)
        restore = True
    except tty.error:    # This is the same as termios.error
        restore = False

    try:
        _copy(master_fd, master_read, stdin_read)
    finally:
        if restore:
            tcsetattr(STDIN_FILENO, tty.TCSAFLUSH, mode)

    close(master_fd)
    return waitpid(pid, 0)[1]
<|endoftext|>import contextlib
import os
import pathlib
import shutil
import stat
import sys
import zipfile

__all__ = ['ZipAppError', 'create_archive', 'get_interpreter']


# The __main__.py used if the users specifies "-m module:fn".
# Note that this will always be written as UTF-8 (module and
# function names can be non-ASCII in Python 3).
# We add a coding cookie even though UTF-8 is the default in Python 3
# because the resulting archive may be intended to be run under Python 2.
MAIN_TEMPLATE = """\
# -*- coding: utf-8 -*-
import {module}
{module}.{fn}()
"""


# The Windows launcher defaults to UTF-8 when parsing shebang lines if the
# file has no BOM. So use UTF-8 on Windows.
# On Unix, use the filesystem encoding.
if sys.platform.startswith('win'):
    shebang_encoding = 'utf-8'
else:
    shebang_encoding = sys.getfilesystemencoding()


class ZipAppError(ValueError):
    pass


@contextlib.contextmanager
def _maybe_open(archive, mode):
    if isinstance(archive, (str, os.PathLike)):
        with open(archive, mode) as f:
            yield f
    else:
        yield archive


def _write_file_prefix(f, interpreter):
    """Write a shebang line."""
    if interpreter:
        shebang = b'#!' + interpreter.encode(shebang_encoding) + b'\n'
        f.write(shebang)


def _copy_archive(archive, new_archive, interpreter=None):
    """Copy an application archive, modifying the shebang line."""
    with _maybe_open(archive, 'rb') as src:
        # Skip the shebang line from the source.
        # Read 2 bytes of the source and check if they are #!.
        first_2 = src.read(2)
        if first_2 == b'#!':
            # Discard the initial 2 bytes and the rest of the shebang line.
            first_2 = b''
            src.readline()

        with _maybe_open(new_archive, 'wb') as dst:
            _write_file_prefix(dst, interpreter)
            # If there was no shebang, "first_2" contains the first 2 bytes
            # of the source file, so write them before copying the rest
            # of the file.
            dst.write(first_2)
            shutil.copyfileobj(src, dst)

    if interpreter and isinstance(new_archive, str):
        os.chmod(new_archive, os.stat(new_archive).st_mode | stat.S_IEXEC)


def create_archive(source, target=None, interpreter=None, main=None,
                   filter=None, compressed=False):
    """Create an application archive from SOURCE.

    The SOURCE can be the name of a directory, or a filename or a file-like
    object referring to an existing archive.

    The content of SOURCE is packed into an application archive in TARGET,
    which can be a filename or a file-like object.  If SOURCE is a directory,
    TARGET can be omitted and will default to the name of SOURCE with .pyz
    appended.

    The created application archive will have a shebang line specifying
    that it should run with INTERPRETER (there will be no shebang line if
    INTERPRETER is None), and a __main__.py which runs MAIN (if MAIN is
    not specified, an existing __main__.py will be used).  It is an error
    to specify MAIN for anything other than a directory source with no
    __main__.py, and it is an error to omit MAIN if the directory has no
    __main__.py.
    """
    # Are we copying an existing archive?
    source_is_file = False
    if hasattr(source, 'read') and hasattr(source, 'readline'):
        source_is_file = True
    else:
        source = pathlib.Path(source)
        if source.is_file():
            source_is_file = True

    if source_is_file:
        _copy_archive(source, target, interpreter)
        return

    # We are creating a new archive from a directory.
    if not source.exists():
        raise ZipAppError("Source does not exist")
    has_main = (source / '__main__.py').is_file()
    if main and has_main:
        raise ZipAppError(
            "Cannot specify entry point if the source has __main__.py")
    if not (main or has_main):
        raise ZipAppError("Archive has no entry point")

    main_py = None
    if main:
        # Check that main has the right format.
        mod, sep, fn = main.partition(':')
        mod_ok = all(part.isidentifier() for part in mod.split('.'))
        fn_ok = all(part.isidentifier() for part in fn.split('.'))
        if not (sep == ':' and mod_ok and fn_ok):
            raise ZipAppError("Invalid entry point: " + main)
        main_py = MAIN_TEMPLATE.format(module=mod, fn=fn)

    if target is None:
        target = source.with_suffix('.pyz')
    elif not hasattr(target, 'write'):
        target = pathlib.Path(target)

    with _maybe_open(target, 'wb') as fd:
        _write_file_prefix(fd, interpreter)
        compression = (zipfile.ZIP_DEFLATED if compressed else
                       zipfile.ZIP_STORED)
        with zipfile.ZipFile(fd, 'w', compression=compression) as z:
            for child in source.rglob('*'):
                arcname = child.relative_to(source)
                if filter is None or filter(arcname):
                    z.write(child, arcname.as_posix())
            if main_py:
                z.writestr('__main__.py', main_py.encode('utf-8'))

    if interpreter and not hasattr(target, 'write'):
        target.chmod(target.stat().st_mode | stat.S_IEXEC)


def get_interpreter(archive):
    with _maybe_open(archive, 'rb') as f:
        if f.read(2) == b'#!':
            return f.readline().strip().decode(shebang_encoding)


def main(args=None):
    """Run the zipapp command line interface.

    The ARGS parameter lets you specify the argument list directly.
    Omitting ARGS (or setting it to None) works as for argparse, using
    sys.argv[1:] as the argument list.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default=None,
            help="The name of the output archive. "
                 "Required if SOURCE is an archive.")
    parser.add_argument('--python', '-p', default=None,
            help="The name of the Python interpreter to use "
                 "(default: no shebang line).")
    parser.add_argument('--main', '-m', default=None,
            help="The main function of the application "
                 "(default: use an existing __main__.py).")
    parser.add_argument('--compress', '-c', action='store_true',
            help="Compress files with the deflate method. "
                 "Files are stored uncompressed by default.")
    parser.add_argument('--info', default=False, action='store_true',
            help="Display the interpreter from the archive.")
    parser.add_argument('source',
            help="Source directory (or existing archive).")

    args = parser.parse_args(args)

    # Handle `python -m zipapp archive.pyz --info`.
    if args.info:
        if not os.path.isfile(args.source):
            raise SystemExit("Can only get info for an archive file")
        interpreter = get_interpreter(args.source)
        print("Interpreter: {}".format(interpreter or "<none>"))
        sys.exit(0)

    if os.path.isfile(args.source):
        if args.output is None or (os.path.exists(args.output) and
                                   os.path.samefile(args.source, args.output)):
            raise SystemExit("In-place editing of archives is not supported")
        if args.main:
            raise SystemExit("Cannot change the main function when copying")

    create_archive(args.source, args.output,
                   interpreter=args.python, main=args.main,
                   compressed=args.compress)


if __name__ == '__main__':
    main()
<|endoftext|># Author: Steven J. Bethard <steven.bethard@gmail.com>.
# New maintainer as of 29 August 2019:  Raymond Hettinger <raymond.hettinger@gmail.com>

"""Command-line parsing library

This module is an optparse-inspired command-line parsing library that:

    - handles both optional and positional arguments
    - produces highly informative usage messages
    - supports parsers that dispatch to sub-parsers

The following is a simple usage example that sums integers from the
command-line and writes the result to a file::

    parser = argparse.ArgumentParser(
        description='sum the integers at the command line')
    parser.add_argument(
        'integers', metavar='int', nargs='+', type=int,
        help='an integer to be summed')
    parser.add_argument(
        '--log', default=sys.stdout, type=argparse.FileType('w'),
        help='the file where the sum should be written')
    args = parser.parse_args()
    args.log.write('%s' % sum(args.integers))
    args.log.close()

The module contains the following public classes:

    - ArgumentParser -- The main entry point for command-line parsing. As the
        example above shows, the add_argument() method is used to populate
        the parser with actions for optional and positional arguments. Then
        the parse_args() method is invoked to convert the args at the
        command-line into an object with attributes.

    - ArgumentError -- The exception raised by ArgumentParser objects when
        there are errors with the parser's actions. Errors raised while
        parsing the command-line are caught by ArgumentParser and emitted
        as command-line messages.

    - FileType -- A factory for defining types of files to be created. As the
        example above shows, instances of FileType are typically passed as
        the type= argument of add_argument() calls.

    - Action -- The base class for parser actions. Typically actions are
        selected by passing strings like 'store_true' or 'append_const' to
        the action= argument of add_argument(). However, for greater
        customization of ArgumentParser actions, subclasses of Action may
        be defined and passed as the action= argument.

    - HelpFormatter, RawDescriptionHelpFormatter, RawTextHelpFormatter,
        ArgumentDefaultsHelpFormatter -- Formatter classes which
        may be passed as the formatter_class= argument to the
        ArgumentParser constructor. HelpFormatter is the default,
        RawDescriptionHelpFormatter and RawTextHelpFormatter tell the parser
        not to change the formatting for help text, and
        ArgumentDefaultsHelpFormatter adds information about argument defaults
        to the help.

All other classes in this module are considered implementation details.
(Also note that HelpFormatter and RawDescriptionHelpFormatter are only
considered public as object names -- the API of the formatter objects is
still considered an implementation detail.)
"""

__version__ = '1.1'
__all__ = [
    'ArgumentParser',
    'ArgumentError',
    'ArgumentTypeError',
    'BooleanOptionalAction',
    'FileType',
    'HelpFormatter',
    'ArgumentDefaultsHelpFormatter',
    'RawDescriptionHelpFormatter',
    'RawTextHelpFormatter',
    'MetavarTypeHelpFormatter',
    'Namespace',
    'Action',
    'ONE_OR_MORE',
    'OPTIONAL',
    'PARSER',
    'REMAINDER',
    'SUPPRESS',
    'ZERO_OR_MORE',
]


import os as _os
import re as _re
import sys as _sys

try:
    from gettext import gettext as _, ngettext
except ImportError:
    def _(message):
        return message
    def ngettext(singular,plural,n):
        if n == 1:
            return singular
        else:
            return plural

SUPPRESS = '==SUPPRESS=='

OPTIONAL = '?'
ZERO_OR_MORE = '*'
ONE_OR_MORE = '+'
PARSER = 'A...'
REMAINDER = '...'
_UNRECOGNIZED_ARGS_ATTR = '_unrecognized_args'

# =============================
# Utility functions and classes
# =============================

class _AttributeHolder(object):
    """Abstract base class that provides __repr__.

    The __repr__ method returns a string in the format::
        ClassName(attr=name, attr=name, ...)
    The attributes are determined either by a class-level attribute,
    '_kwarg_names', or by inspecting the instance __dict__.
    """

    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))

    def _get_kwargs(self):
        return list(self.__dict__.items())

    def _get_args(self):
        return []


def _copy_items(items):
    if items is None:
        return []
    # The copy module is used only in the 'append' and 'append_const'
    # actions, and it is needed only when the default value isn't a list.
    # Delay its import for speeding up the common case.
    if type(items) is list:
        return items[:]
    import copy
    return copy.copy(items)


# ===============
# Formatting Help
# ===============

class HelpFormatter(object):
    """Formatter for generating usage messages and argument help strings.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def __init__(self,
                 prog,
                 indent_increment=2,
                 max_help_position=24,
                 width=None):

        # default setting for width
        if width is None:
            try:
                import shutil as _shutil
                width = _shutil.get_terminal_size().columns
                width -= 2
            except ImportError:
                width = 70

        self._prog = prog
        self._indent_increment = indent_increment
        self._max_help_position = min(max_help_position,
                                      max(width - 20, indent_increment * 2))
        self._width = width

        self._current_indent = 0
        self._level = 0
        self._action_max_length = 0

        self._root_section = self._Section(self, None)
        self._current_section = self._root_section

        self._whitespace_matcher = _re.compile(r'\s+', _re.ASCII)
        self._long_break_matcher = _re.compile(r'\n\n\n+')

    # ===============================
    # Section and indentation methods
    # ===============================
    def _indent(self):
        self._current_indent += self._indent_increment
        self._level += 1

    def _dedent(self):
        self._current_indent -= self._indent_increment
        assert self._current_indent >= 0, 'Indent decreased below 0.'
        self._level -= 1

    class _Section(object):

        def __init__(self, formatter, parent, heading=None):
            self.formatter = formatter
            self.parent = parent
            self.heading = heading
            self.items = []

        def format_help(self):
            # format the indented section
            if self.parent is not None:
                self.formatter._indent()
            join = self.formatter._join_parts
            item_help = join([func(*args) for func, args in self.items])
            if self.parent is not None:
                self.formatter._dedent()

            # return nothing if the section was empty
            if not item_help:
                return ''

            # add the heading if the section was non-empty
            if self.heading is not SUPPRESS and self.heading is not None:
                current_indent = self.formatter._current_indent
                heading = '%*s%s:\n' % (current_indent, '', self.heading)
            else:
                heading = ''

            # join the section-initial newline, the heading and the help
            return join(['\n', heading, item_help, '\n'])

    def _add_item(self, func, args):
        self._current_section.items.append((func, args))

    # ========================
    # Message building methods
    # ========================
    def start_section(self, heading):
        self._indent()
        section = self._Section(self, self._current_section, heading)
        self._add_item(section.format_help, [])
        self._current_section = section

    def end_section(self):
        self._current_section = self._current_section.parent
        self._dedent()

    def add_text(self, text):
        if text is not SUPPRESS and text is not None:
            self._add_item(self._format_text, [text])

    def add_usage(self, usage, actions, groups, prefix=None):
        if usage is not SUPPRESS:
            args = usage, actions, groups, prefix
            self._add_item(self._format_usage, args)

    def add_argument(self, action):
        if action.help is not SUPPRESS:

            # find all invocations
            get_invocation = self._format_action_invocation
            invocations = [get_invocation(action)]
            for subaction in self._iter_indented_subactions(action):
                invocations.append(get_invocation(subaction))

            # update the maximum item length
            invocation_length = max(map(len, invocations))
            action_length = invocation_length + self._current_indent
            self._action_max_length = max(self._action_max_length,
                                          action_length)

            # add the item to the list
            self._add_item(self._format_action, [action])

    def add_arguments(self, actions):
        for action in actions:
            self.add_argument(action)

    # =======================
    # Help-formatting methods
    # =======================
    def format_help(self):
        help = self._root_section.format_help()
        if help:
            help = self._long_break_matcher.sub('\n\n', help)
            help = help.strip('\n') + '\n'
        return help

    def _join_parts(self, part_strings):
        return ''.join([part
                        for part in part_strings
                        if part and part is not SUPPRESS])

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = _('usage: ')

        # if usage is specified, use that
        if usage is not None:
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = '%(prog)s' % dict(prog=self._prog)

        # if optionals and positionals are available, calculate usage
        elif usage is None:
            prog = '%(prog)s' % dict(prog=self._prog)

            # split optionals from positionals
            optionals = []
            positionals = []
            for action in actions:
                if action.option_strings:
                    optionals.append(action)
                else:
                    positionals.append(action)

            # build full usage string
            format = self._format_actions_usage
            action_usage = format(optionals + positionals, groups)
            usage = ' '.join([s for s in [prog, action_usage] if s])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(usage) > text_width:

                # break usage into wrappable parts
                part_regexp = (
                    r'\(.*?\)+(?=\s|$)|'
                    r'\[.*?\]+(?=\s|$)|'
                    r'\S+'
                )
                opt_usage = format(optionals, groups)
                pos_usage = format(positionals, groups)
                opt_parts = _re.findall(part_regexp, opt_usage)
                pos_parts = _re.findall(part_regexp, pos_usage)
                assert ' '.join(opt_parts) == opt_usage
                assert ' '.join(pos_parts) == pos_usage

                # helper for wrapping lines
                def get_lines(parts, indent, prefix=None):
                    lines = []
                    line = []
                    if prefix is not None:
                        line_len = len(prefix) - 1
                    else:
                        line_len = len(indent) - 1
                    for part in parts:
                        if line_len + 1 + len(part) > text_width and line:
                            lines.append(indent + ' '.join(line))
                            line = []
                            line_len = len(indent) - 1
                        line.append(part)
                        line_len += len(part) + 1
                    if line:
                        lines.append(indent + ' '.join(line))
                    if prefix is not None:
                        lines[0] = lines[0][len(indent):]
                    return lines

                # if prog is short, follow it with optionals or positionals
                if len(prefix) + len(prog) <= 0.75 * text_width:
                    indent = ' ' * (len(prefix) + len(prog) + 1)
                    if opt_parts:
                        lines = get_lines([prog] + opt_parts, indent, prefix)
                        lines.extend(get_lines(pos_parts, indent))
                    elif pos_parts:
                        lines = get_lines([prog] + pos_parts, indent, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix)
                    parts = opt_parts + pos_parts
                    lines = get_lines(parts, indent)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(get_lines(opt_parts, indent))
                        lines.extend(get_lines(pos_parts, indent))
                    lines = [prog] + lines

                # join lines into usage
                usage = '\n'.join(lines)

        # prefix with 'usage:'
        return '%s%s\n\n' % (prefix, usage)

    def _format_actions_usage(self, actions, groups):
        # find group indices and identify actions in groups
        group_actions = set()
        inserts = {}
        for group in groups:
            if not group._group_actions:
                raise ValueError(f'empty group {group}')

            try:
                start = actions.index(group._group_actions[0])
            except ValueError:
                continue
            else:
                group_action_count = len(group._group_actions)
                end = start + group_action_count
                if actions[start:end] == group._group_actions:

                    suppressed_actions_count = 0
                    for action in group._group_actions:
                        group_actions.add(action)
                        if action.help is SUPPRESS:
                            suppressed_actions_count += 1

                    exposed_actions_count = group_action_count - suppressed_actions_count

                    if not group.required:
                        if start in inserts:
                            inserts[start] += ' ['
                        else:
                            inserts[start] = '['
                        if end in inserts:
                            inserts[end] += ']'
                        else:
                            inserts[end] = ']'
                    elif exposed_actions_count > 1:
                        if start in inserts:
                            inserts[start] += ' ('
                        else:
                            inserts[start] = '('
                        if end in inserts:
                            inserts[end] += ')'
                        else:
                            inserts[end] = ')'
                    for i in range(start + 1, end):
                        inserts[i] = '|'

        # collect all actions format strings
        parts = []
        for i, action in enumerate(actions):

            # suppressed arguments are marked with None
            # remove | separators for suppressed arguments
            if action.help is SUPPRESS:
                parts.append(None)
                if inserts.get(i) == '|':
                    inserts.pop(i)
                elif inserts.get(i + 1) == '|':
                    inserts.pop(i + 1)

            # produce all arg strings
            elif not action.option_strings:
                default = self._get_default_metavar_for_positional(action)
                part = self._format_args(action, default)

                # if it's in a group, strip the outer []
                if action in group_actions:
                    if part[0] == '[' and part[-1] == ']':
                        part = part[1:-1]

                # add the action string to the list
                parts.append(part)

            # produce the first way to invoke the option in brackets
            else:
                option_string = action.option_strings[0]

                # if the Optional doesn't take a value, format is:
                #    -s or --long
                if action.nargs == 0:
                    part = action.format_usage()

                # if the Optional takes a value, format is:
                #    -s ARGS or --long ARGS
                else:
                    default = self._get_default_metavar_for_optional(action)
                    args_string = self._format_args(action, default)
                    part = '%s %s' % (option_string, args_string)

                # make it look optional if it's not required or in a group
                if not action.required and action not in group_actions:
                    part = '[%s]' % part

                # add the action string to the list
                parts.append(part)

        # insert things at the necessary indices
        for i in sorted(inserts, reverse=True):
            parts[i:i] = [inserts[i]]

        # join all the action items with spaces
        text = ' '.join([item for item in parts if item is not None])

        # clean up separators for mutually exclusive groups
        open = r'[\[(]'
        close = r'[\])]'
        text = _re.sub(r'(%s) ' % open, r'\1', text)
        text = _re.sub(r' (%s)' % close, r'\1', text)
        text = _re.sub(r'%s *%s' % (open, close), r'', text)
        text = text.strip()

        # return the text
        return text

    def _format_text(self, text):
        if '%(prog)' in text:
            text = text % dict(prog=self._prog)
        text_width = max(self._width - self._current_indent, 11)
        indent = ' ' * self._current_indent
        return self._fill_text(text, text_width, indent) + '\n\n'

    def _format_action(self, action):
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        # no help; start on same line and add a final newline
        if not action.help:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            tup = self._current_indent, '', action_width, action_header
            action_header = '%*s%-*s  ' % tup
            indent_first = 0

        # long action name; start on the next line
        else:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        if action.help and action.help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
                for line in help_lines[1:]:
                    parts.append('%*s%s\n' % (help_position, '', line))

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # return a single string
        return self._join_parts(parts)

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar

        else:
            parts = []

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append('%s %s' % (option_string, args_string))

            return ', '.join(parts)

    def _metavar_formatter(self, action, default_metavar):
        if action.metavar is not None:
            result = action.metavar
        elif action.choices is not None:
            choice_strs = [str(choice) for choice in action.choices]
            result = '{%s}' % ','.join(choice_strs)
        else:
            result = default_metavar

        def format(tuple_size):
            if isinstance(result, tuple):
                return result
            else:
                return (result, ) * tuple_size
        return format

    def _format_args(self, action, default_metavar):
        get_metavar = self._metavar_formatter(action, default_metavar)
        if action.nargs is None:
            result = '%s' % get_metavar(1)
        elif action.nargs == OPTIONAL:
            result = '[%s]' % get_metavar(1)
        elif action.nargs == ZERO_OR_MORE:
            metavar = get_metavar(1)
            if len(metavar) == 2:
                result = '[%s [%s ...]]' % metavar
            else:
                result = '[%s ...]' % metavar
        elif action.nargs == ONE_OR_MORE:
            result = '%s [%s ...]' % get_metavar(2)
        elif action.nargs == REMAINDER:
            result = '...'
        elif action.nargs == PARSER:
            result = '%s ...' % get_metavar(1)
        elif action.nargs == SUPPRESS:
            result = ''
        else:
            try:
                formats = ['%s' for _ in range(action.nargs)]
            except TypeError:
                raise ValueError("invalid nargs value") from None
            result = ' '.join(formats) % get_metavar(action.nargs)
        return result

    def _expand_help(self, action):
        params = dict(vars(action), prog=self._prog)
        for name in list(params):
            if params[name] is SUPPRESS:
                del params[name]
        for name in list(params):
            if hasattr(params[name], '__name__'):
                params[name] = params[name].__name__
        if params.get('choices') is not None:
            choices_str = ', '.join([str(c) for c in params['choices']])
            params['choices'] = choices_str
        return self._get_help_string(action) % params

    def _iter_indented_subactions(self, action):
        try:
            get_subactions = action._get_subactions
        except AttributeError:
            pass
        else:
            self._indent()
            yield from get_subactions()
            self._dedent()

    def _split_lines(self, text, width):
        text = self._whitespace_matcher.sub(' ', text).strip()
        # The textwrap module is used only for formatting help.
        # Delay its import for speeding up the common usage of argparse.
        import textwrap
        return textwrap.wrap(text, width)

    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        import textwrap
        return textwrap.fill(text, width,
                             initial_indent=indent,
                             subsequent_indent=indent)

    def _get_help_string(self, action):
        return action.help

    def _get_default_metavar_for_optional(self, action):
        return action.dest.upper()

    def _get_default_metavar_for_positional(self, action):
        return action.dest


class RawDescriptionHelpFormatter(HelpFormatter):
    """Help message formatter which retains any formatting in descriptions.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))


class RawTextHelpFormatter(RawDescriptionHelpFormatter):
    """Help message formatter which retains formatting of all help text.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _split_lines(self, text, width):
        return text.splitlines()


class ArgumentDefaultsHelpFormatter(HelpFormatter):
    """Help message formatter which adds default values to argument help.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not SUPPRESS:
                defaulting_nargs = [OPTIONAL, ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help


class MetavarTypeHelpFormatter(HelpFormatter):
    """Help message formatter which uses the argument 'type' as the default
    metavar value (instead of the argument 'dest')

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_default_metavar_for_optional(self, action):
        return action.type.__name__

    def _get_default_metavar_for_positional(self, action):
        return action.type.__name__



# =====================
# Options and Arguments
# =====================

def _get_action_name(argument):
    if argument is None:
        return None
    elif argument.option_strings:
        return '/'.join(argument.option_strings)
    elif argument.metavar not in (None, SUPPRESS):
        return argument.metavar
    elif argument.dest not in (None, SUPPRESS):
        return argument.dest
    elif argument.choices:
        return '{' + ','.join(argument.choices) + '}'
    else:
        return None


class ArgumentError(Exception):
    """An error from creating or using an argument (optional or positional).

    The string value of this exception is the message, augmented with
    information about the argument that caused it.
    """

    def __init__(self, argument, message):
        self.argument_name = _get_action_name(argument)
        self.message = message

    def __str__(self):
        if self.argument_name is None:
            format = '%(message)s'
        else:
            format = 'argument %(argument_name)s: %(message)s'
        return format % dict(message=self.message,
                             argument_name=self.argument_name)


class ArgumentTypeError(Exception):
    """An error from trying to convert a command line string to a type."""
    pass


# ==============
# Action classes
# ==============

class Action(_AttributeHolder):
    """Information about how to convert command line strings to Python objects.

    Action objects are used by an ArgumentParser to represent the information
    needed to parse a single argument from one or more strings from the
    command line. The keyword arguments to the Action constructor are also
    all attributes of Action instances.

    Keyword Arguments:

        - option_strings -- A list of command-line option strings which
            should be associated with this action.

        - dest -- The name of the attribute to hold the created object(s)

        - nargs -- The number of command-line arguments that should be
            consumed. By default, one argument will be consumed and a single
            value will be produced.  Other values include:
                - N (an integer) consumes N arguments (and produces a list)
                - '?' consumes zero or one arguments
                - '*' consumes zero or more arguments (and produces a list)
                - '+' consumes one or more arguments (and produces a list)
            Note that the difference between the default and nargs=1 is that
            with the default, a single value will be produced, while with
            nargs=1, a list containing a single value will be produced.

        - const -- The value to be produced if the option is specified and the
            option uses an action that takes no values.

        - default -- The value to be produced if the option is not specified.

        - type -- A callable that accepts a single string argument, and
            returns the converted value.  The standard Python types str, int,
            float, and complex are useful examples of such callables.  If None,
            str is used.

        - choices -- A container of values that should be allowed. If not None,
            after a command-line argument has been converted to the appropriate
            type, an exception will be raised if it is not a member of this
            collection.

        - required -- True if the action must always be specified at the
            command line. This is only meaningful for optional command-line
            arguments.

        - help -- The help string describing the argument.

        - metavar -- The name to be used for the option's argument with the
            help string. If None, the 'dest' value will be used as the name.
    """

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        self.option_strings = option_strings
        self.dest = dest
        self.nargs = nargs
        self.const = const
        self.default = default
        self.type = type
        self.choices = choices
        self.required = required
        self.help = help
        self.metavar = metavar

    def _get_kwargs(self):
        names = [
            'option_strings',
            'dest',
            'nargs',
            'const',
            'default',
            'type',
            'choices',
            'required',
            'help',
            'metavar',
        ]
        return [(name, getattr(self, name)) for name in names]

    def format_usage(self):
        return self.option_strings[0]

    def __call__(self, parser, namespace, values, option_string=None):
        raise NotImplementedError(_('.__call__() not defined'))

class BooleanOptionalAction(Action):
    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):

        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        if help is not None and default is not None and default is not SUPPRESS:
            help += " (default: %(default)s)"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            setattr(namespace, self.dest, not option_string.startswith('--no-'))

    def format_usage(self):
        return ' | '.join(self.option_strings)


class _StoreAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if nargs == 0:
            raise ValueError('nargs for store actions must be != 0; if you '
                             'have nothing to store, actions such as store '
                             'true or store const may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        super(_StoreAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class _StoreConstAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 const,
                 default=None,
                 required=False,
                 help=None,
                 metavar=None):
        super(_StoreConstAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)


class _StoreTrueAction(_StoreConstAction):

    def __init__(self,
                 option_strings,
                 dest,
                 default=False,
                 required=False,
                 help=None):
        super(_StoreTrueAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            const=True,
            default=default,
            required=required,
            help=help)


class _StoreFalseAction(_StoreConstAction):

    def __init__(self,
                 option_strings,
                 dest,
                 default=True,
                 required=False,
                 help=None):
        super(_StoreFalseAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            const=False,
            default=default,
            required=required,
            help=help)


class _AppendAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if nargs == 0:
            raise ValueError('nargs for append actions must be != 0; if arg '
                             'strings are not supplying the value to append, '
                             'the append const action may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        super(_AppendAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.append(values)
        setattr(namespace, self.dest, items)


class _AppendConstAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 const,
                 default=None,
                 required=False,
                 help=None,
                 metavar=None):
        super(_AppendConstAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.append(self.const)
        setattr(namespace, self.dest, items)


class _CountAction(Action):

    def __init__(self,
                 option_strings,
                 dest,
                 default=None,
                 required=False,
                 help=None):
        super(_CountAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        count = getattr(namespace, self.dest, None)
        if count is None:
            count = 0
        setattr(namespace, self.dest, count + 1)


class _HelpAction(Action):

    def __init__(self,
                 option_strings,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help=None):
        super(_HelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        parser.exit()


class _VersionAction(Action):

    def __init__(self,
                 option_strings,
                 version=None,
                 dest=SUPPRESS,
                 default=SUPPRESS,
                 help="show program's version number and exit"):
        super(_VersionAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)
        self.version = version

    def __call__(self, parser, namespace, values, option_string=None):
        version = self.version
        if version is None:
            version = parser.version
        formatter = parser._get_formatter()
        formatter.add_text(version)
        parser._print_message(formatter.format_help(), _sys.stdout)
        parser.exit()


class _SubParsersAction(Action):

    class _ChoicesPseudoAction(Action):

        def __init__(self, name, aliases, help):
            metavar = dest = name
            if aliases:
                metavar += ' (%s)' % ', '.join(aliases)
            sup = super(_SubParsersAction._ChoicesPseudoAction, self)
            sup.__init__(option_strings=[], dest=dest, help=help,
                         metavar=metavar)

    def __init__(self,
                 option_strings,
                 prog,
                 parser_class,
                 dest=SUPPRESS,
                 required=False,
                 help=None,
                 metavar=None):

        self._prog_prefix = prog
        self._parser_class = parser_class
        self._name_parser_map = {}
        self._choices_actions = []

        super(_SubParsersAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=PARSER,
            choices=self._name_parser_map,
            required=required,
            help=help,
            metavar=metavar)

    def add_parser(self, name, **kwargs):
        # set prog from the existing prefix
        if kwargs.get('prog') is None:
            kwargs['prog'] = '%s %s' % (self._prog_prefix, name)

        aliases = kwargs.pop('aliases', ())

        # create a pseudo-action to hold the choice help
        if 'help' in kwargs:
            help = kwargs.pop('help')
            choice_action = self._ChoicesPseudoAction(name, aliases, help)
            self._choices_actions.append(choice_action)

        # create the parser and add it to the map
        parser = self._parser_class(**kwargs)
        self._name_parser_map[name] = parser

        # make parser available under aliases also
        for alias in aliases:
            self._name_parser_map[alias] = parser

        return parser

    def _get_subactions(self):
        return self._choices_actions

    def __call__(self, parser, namespace, values, option_string=None):
        parser_name = values[0]
        arg_strings = values[1:]

        # set the parser name if requested
        if self.dest is not SUPPRESS:
            setattr(namespace, self.dest, parser_name)

        # select the parser
        try:
            parser = self._name_parser_map[parser_name]
        except KeyError:
            args = {'parser_name': parser_name,
                    'choices': ', '.join(self._name_parser_map)}
            msg = _('unknown parser %(parser_name)r (choices: %(choices)s)') % args
            raise ArgumentError(self, msg)

        # parse all the remaining options into the namespace
        # store any unrecognized options on the object, so that the top
        # level parser can decide what to do with them

        # In case this subparser defines new defaults, we parse them
        # in a new namespace object and then update the original
        # namespace for the relevant parts.
        subnamespace, arg_strings = parser.parse_known_args(arg_strings, None)
        for key, value in vars(subnamespace).items():
            setattr(namespace, key, value)

        if arg_strings:
            vars(namespace).setdefault(_UNRECOGNIZED_ARGS_ATTR, [])
            getattr(namespace, _UNRECOGNIZED_ARGS_ATTR).extend(arg_strings)

class _ExtendAction(_AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.extend(values)
        setattr(namespace, self.dest, items)

# ==============
# Type classes
# ==============

class FileType(object):
    """Factory for creating file object types

    Instances of FileType are typically passed as type= arguments to the
    ArgumentParser add_argument() method.

    Keyword Arguments:
        - mode -- A string indicating how the file is to be opened. Accepts the
            same values as the builtin open() function.
        - bufsize -- The file's desired buffer size. Accepts the same values as
            the builtin open() function.
        - encoding -- The file's encoding. Accepts the same values as the
            builtin open() function.
        - errors -- A string indicating how encoding and decoding errors are to
            be handled. Accepts the same value as the builtin open() function.
    """

    def __init__(self, mode='r', bufsize=-1, encoding=None, errors=None):
        self._mode = mode
        self._bufsize = bufsize
        self._encoding = encoding
        self._errors = errors

    def __call__(self, string):
        # the special argument "-" means sys.std{in,out}
        if string == '-':
            if 'r' in self._mode:
                return _sys.stdin.buffer if 'b' in self._mode else _sys.stdin
            elif any(c in self._mode for c in 'wax'):
                return _sys.stdout.buffer if 'b' in self._mode else _sys.stdout
            else:
                msg = _('argument "-" with mode %r') % self._mode
                raise ValueError(msg)

        # all other arguments are used as file names
        try:
            return open(string, self._mode, self._bufsize, self._encoding,
                        self._errors)
        except OSError as e:
            args = {'filename': string, 'error': e}
            message = _("can't open '%(filename)s': %(error)s")
            raise ArgumentTypeError(message % args)

    def __repr__(self):
        args = self._mode, self._bufsize
        kwargs = [('encoding', self._encoding), ('errors', self._errors)]
        args_str = ', '.join([repr(arg) for arg in args if arg != -1] +
                             ['%s=%r' % (kw, arg) for kw, arg in kwargs
                              if arg is not None])
        return '%s(%s)' % (type(self).__name__, args_str)

# ===========================
# Optional and Positional Parsing
# ===========================

class Namespace(_AttributeHolder):
    """Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.
    """

    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __eq__(self, other):
        if not isinstance(other, Namespace):
            return NotImplemented
        return vars(self) == vars(other)

    def __contains__(self, key):
        return key in self.__dict__


class _ActionsContainer(object):

    def __init__(self,
                 description,
                 prefix_chars,
                 argument_default,
                 conflict_handler):
        super(_ActionsContainer, self).__init__()

        self.description = description
        self.argument_default = argument_default
        self.prefix_chars = prefix_chars
        self.conflict_handler = conflict_handler

        # set up registries
        self._registries = {}

        # register actions
        self.register('action', None, _StoreAction)
        self.register('action', 'store', _StoreAction)
        self.register('action', 'store_const', _StoreConstAction)
        self.register('action', 'store_true', _StoreTrueAction)
        self.register('action', 'store_false', _StoreFalseAction)
        self.register('action', 'append', _AppendAction)
        self.register('action', 'append_const', _AppendConstAction)
        self.register('action', 'count', _CountAction)
        self.register('action', 'help', _HelpAction)
        self.register('action', 'version', _VersionAction)
        self.register('action', 'parsers', _SubParsersAction)
        self.register('action', 'extend', _ExtendAction)

        # raise an exception if the conflict handler is invalid
        self._get_handler()

        # action storage
        self._actions = []
        self._option_string_actions = {}

        # groups
        self._action_groups = []
        self._mutually_exclusive_groups = []

        # defaults storage
        self._defaults = {}

        # determines whether an "option" looks like a negative number
        self._negative_number_matcher = _re.compile(r'^-\d+$|^-\d*\.\d+$')

        # whether or not there are any optionals that look like negative
        # numbers -- uses a list so it can be shared and edited
        self._has_negative_number_optionals = []

    # ====================
    # Registration methods
    # ====================
    def register(self, registry_name, value, object):
        registry = self._registries.setdefault(registry_name, {})
        registry[value] = object

    def _registry_get(self, registry_name, value, default=None):
        return self._registries[registry_name].get(value, default)

    # ==================================
    # Namespace default accessor methods
    # ==================================
    def set_defaults(self, **kwargs):
        self._defaults.update(kwargs)

        # if these defaults match any existing arguments, replace
        # the previous default on the object with the new one
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]

    def get_default(self, dest):
        for action in self._actions:
            if action.dest == dest and action.default is not None:
                return action.default
        return self._defaults.get(dest, None)


    # =======================
    # Adding argument actions
    # =======================
    def add_argument(self, *args, **kwargs):
        """
        add_argument(dest, ..., name=value, ...)
        add_argument(option_string, option_string, ..., name=value, ...)
        """

        # if no positional args are supplied or only one is supplied and
        # it doesn't look like an option string, parse a positional
        # argument
        chars = self.prefix_chars
        if not args or len(args) == 1 and args[0][0] not in chars:
            if args and 'dest' in kwargs:
                raise ValueError('dest supplied twice for positional argument')
            kwargs = self._get_positional_kwargs(*args, **kwargs)

        # otherwise, we're adding an optional argument
        else:
            kwargs = self._get_optional_kwargs(*args, **kwargs)

        # if no default was supplied, use the parser-level default
        if 'default' not in kwargs:
            dest = kwargs['dest']
            if dest in self._defaults:
                kwargs['default'] = self._defaults[dest]
            elif self.argument_default is not None:
                kwargs['default'] = self.argument_default

        # create the action object, and add it to the parser
        action_class = self._pop_action_class(kwargs)
        if not callable(action_class):
            raise ValueError('unknown action "%s"' % (action_class,))
        action = action_class(**kwargs)

        # raise an error if the action type is not callable
        type_func = self._registry_get('type', action.type, action.type)
        if not callable(type_func):
            raise ValueError('%r is not callable' % (type_func,))

        if type_func is FileType:
            raise ValueError('%r is a FileType class object, instance of it'
                             ' must be passed' % (type_func,))

        # raise an error if the metavar does not match the type
        if hasattr(self, "_get_formatter"):
            try:
                self._get_formatter()._format_args(action, None)
            except TypeError:
                raise ValueError("length of metavar tuple does not match nargs")

        return self._add_action(action)

    def add_argument_group(self, *args, **kwargs):
        group = _ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        group = _MutuallyExclusiveGroup(self, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group

    def _add_action(self, action):
        # resolve any conflicts
        self._check_conflict(action)

        # add to actions list
        self._actions.append(action)
        action.container = self

        # index the action by any option strings it has
        for option_string in action.option_strings:
            self._option_string_actions[option_string] = action

        # set the flag if any option strings look like negative numbers
        for option_string in action.option_strings:
            if self._negative_number_matcher.match(option_string):
                if not self._has_negative_number_optionals:
                    self._has_negative_number_optionals.append(True)

        # return the created action
        return action

    def _remove_action(self, action):
        self._actions.remove(action)

    def _add_container_actions(self, container):
        # collect groups by titles
        title_group_map = {}
        for group in self._action_groups:
            if group.title in title_group_map:
                msg = _('cannot merge actions - two groups are named %r')
                raise ValueError(msg % (group.title))
            title_group_map[group.title] = group

        # map each action to its group
        group_map = {}
        for group in container._action_groups:

            # if a group with the title exists, use that, otherwise
            # create a new group matching the container's group
            if group.title not in title_group_map:
                title_group_map[group.title] = self.add_argument_group(
                    title=group.title,
                    description=group.description,
                    conflict_handler=group.conflict_handler)

            # map the actions to their new group
            for action in group._group_actions:
                group_map[action] = title_group_map[group.title]

        # add container's mutually exclusive groups
        # NOTE: if add_mutually_exclusive_group ever gains title= and
        # description= then this code will need to be expanded as above
        for group in container._mutually_exclusive_groups:
            mutex_group = self.add_mutually_exclusive_group(
                required=group.required)

            # map the actions to their new mutex group
            for action in group._group_actions:
                group_map[action] = mutex_group

        # add all actions to this container or their group
        for action in container._actions:
            group_map.get(action, self)._add_action(action)

    def _get_positional_kwargs(self, dest, **kwargs):
        # make sure required is not specified
        if 'required' in kwargs:
            msg = _("'required' is an invalid argument for positionals")
            raise TypeError(msg)

        # mark positional arguments as required if at least one is
        # always required
        if kwargs.get('nargs') not in [OPTIONAL, ZERO_OR_MORE]:
            kwargs['required'] = True
        if kwargs.get('nargs') == ZERO_OR_MORE and 'default' not in kwargs:
            kwargs['required'] = True

        # return the keyword arguments with no option strings
        return dict(kwargs, dest=dest, option_strings=[])

    def _get_optional_kwargs(self, *args, **kwargs):
        # determine short and long option strings
        option_strings = []
        long_option_strings = []
        for option_string in args:
            # error on strings that don't start with an appropriate prefix
            if not option_string[0] in self.prefix_chars:
                args = {'option': option_string,
                        'prefix_chars': self.prefix_chars}
                msg = _('invalid option string %(option)r: '
                        'must start with a character %(prefix_chars)r')
                raise ValueError(msg % args)

            # strings starting with two prefix characters are long options
            option_strings.append(option_string)
            if len(option_string) > 1 and option_string[1] in self.prefix_chars:
                long_option_strings.append(option_string)

        # infer destination, '--foo-bar' -> 'foo_bar' and '-x' -> 'x'
        dest = kwargs.pop('dest', None)
        if dest is None:
            if long_option_strings:
                dest_option_string = long_option_strings[0]
            else:
                dest_option_string = option_strings[0]
            dest = dest_option_string.lstrip(self.prefix_chars)
            if not dest:
                msg = _('dest= is required for options like %r')
                raise ValueError(msg % option_string)
            dest = dest.replace('-', '_')

        # return the updated keyword arguments
        return dict(kwargs, dest=dest, option_strings=option_strings)

    def _pop_action_class(self, kwargs, default=None):
        action = kwargs.pop('action', default)
        return self._registry_get('action', action, action)

    def _get_handler(self):
        # determine function from conflict handler string
        handler_func_name = '_handle_conflict_%s' % self.conflict_handler
        try:
            return getattr(self, handler_func_name)
        except AttributeError:
            msg = _('invalid conflict_resolution value: %r')
            raise ValueError(msg % self.conflict_handler)

    def _check_conflict(self, action):

        # find all options that conflict with this option
        confl_optionals = []
        for option_string in action.option_strings:
            if option_string in self._option_string_actions:
                confl_optional = self._option_string_actions[option_string]
                confl_optionals.append((option_string, confl_optional))

        # resolve any conflicts
        if confl_optionals:
            conflict_handler = self._get_handler()
            conflict_handler(action, confl_optionals)

    def _handle_conflict_error(self, action, conflicting_actions):
        message = ngettext('conflicting option string: %s',
