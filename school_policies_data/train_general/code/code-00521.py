                                                 warnings=self.warnings)
            except TypeError:
                # didn't accept the verbosity, buffer or failfast arguments
                testRunner = self.testRunner()
        else:
            # it is assumed to be a TestRunner instance
            testRunner = self.testRunner
        self.result = testRunner.run(self.test)
        if self.exit:
            sys.exit(not self.result.wasSuccessful())

main = TestProgram
<|endoftext|>import signal
import weakref

from functools import wraps

__unittest = True


class _InterruptHandler(object):
    def __init__(self, default_handler):
        self.called = False
        self.original_handler = default_handler
        if isinstance(default_handler, int):
            if default_handler == signal.SIG_DFL:
                # Pretend it's signal.default_int_handler instead.
                default_handler = signal.default_int_handler
            elif default_handler == signal.SIG_IGN:
                # Not quite the same thing as SIG_IGN, but the closest we
                # can make it: do nothing.
                def default_handler(unused_signum, unused_frame):
                    pass
            else:
                raise TypeError("expected SIGINT signal handler to be "
                                "signal.SIG_IGN, signal.SIG_DFL, or a "
                                "callable object")
        self.default_handler = default_handler

    def __call__(self, signum, frame):
        installed_handler = signal.getsignal(signal.SIGINT)
        if installed_handler is not self:
            # if we aren't the installed handler, then delegate immediately
            # to the default handler
            self.default_handler(signum, frame)

        if self.called:
            self.default_handler(signum, frame)
        self.called = True
        for result in _results.keys():
            result.stop()

_results = weakref.WeakKeyDictionary()
def registerResult(result):
    _results[result] = 1

def removeResult(result):
    return bool(_results.pop(result, None))

_interrupt_handler = None
def installHandler():
    global _interrupt_handler
    if _interrupt_handler is None:
        default_handler = signal.getsignal(signal.SIGINT)
        _interrupt_handler = _InterruptHandler(default_handler)
        signal.signal(signal.SIGINT, _interrupt_handler)


def removeHandler(method=None):
    if method is not None:
        @wraps(method)
        def inner(*args, **kwargs):
            initial = signal.getsignal(signal.SIGINT)
            removeHandler()
            try:
                return method(*args, **kwargs)
            finally:
                signal.signal(signal.SIGINT, initial)
        return inner

    global _interrupt_handler
    if _interrupt_handler is not None:
        signal.signal(signal.SIGINT, _interrupt_handler.original_handler)
<|endoftext|>"""Various utility functions."""

from collections import namedtuple, Counter
from os.path import commonprefix

__unittest = True

_MAX_LENGTH = 80
_PLACEHOLDER_LEN = 12
_MIN_BEGIN_LEN = 5
_MIN_END_LEN = 5
_MIN_COMMON_LEN = 5
_MIN_DIFF_LEN = _MAX_LENGTH - \
               (_MIN_BEGIN_LEN + _PLACEHOLDER_LEN + _MIN_COMMON_LEN +
                _PLACEHOLDER_LEN + _MIN_END_LEN)
assert _MIN_DIFF_LEN >= 0

def _shorten(s, prefixlen, suffixlen):
    skip = len(s) - prefixlen - suffixlen
    if skip > _PLACEHOLDER_LEN:
        s = '%s[%d chars]%s' % (s[:prefixlen], skip, s[len(s) - suffixlen:])
    return s

def _common_shorten_repr(*args):
    args = tuple(map(safe_repr, args))
    maxlen = max(map(len, args))
    if maxlen <= _MAX_LENGTH:
        return args

    prefix = commonprefix(args)
    prefixlen = len(prefix)

    common_len = _MAX_LENGTH - \
                 (maxlen - prefixlen + _MIN_BEGIN_LEN + _PLACEHOLDER_LEN)
    if common_len > _MIN_COMMON_LEN:
        assert _MIN_BEGIN_LEN + _PLACEHOLDER_LEN + _MIN_COMMON_LEN + \
               (maxlen - prefixlen) < _MAX_LENGTH
        prefix = _shorten(prefix, _MIN_BEGIN_LEN, common_len)
        return tuple(prefix + s[prefixlen:] for s in args)

    prefix = _shorten(prefix, _MIN_BEGIN_LEN, _MIN_COMMON_LEN)
    return tuple(prefix + _shorten(s[prefixlen:], _MIN_DIFF_LEN, _MIN_END_LEN)
                 for s in args)

def safe_repr(obj, short=False):
    try:
        result = repr(obj)
    except Exception:
        result = object.__repr__(obj)
    if not short or len(result) < _MAX_LENGTH:
        return result
    return result[:_MAX_LENGTH] + ' [truncated]...'

def strclass(cls):
    return "%s.%s" % (cls.__module__, cls.__qualname__)

def sorted_list_difference(expected, actual):
    """Finds elements in only one or the other of two, sorted input lists.

    Returns a two-element tuple of lists.    The first list contains those
    elements in the "expected" list but not in the "actual" list, and the
    second contains those elements in the "actual" list but not in the
    "expected" list.    Duplicate elements in either input list are ignored.
    """
    i = j = 0
    missing = []
    unexpected = []
    while True:
        try:
            e = expected[i]
            a = actual[j]
            if e < a:
                missing.append(e)
                i += 1
                while expected[i] == e:
                    i += 1
            elif e > a:
                unexpected.append(a)
                j += 1
                while actual[j] == a:
                    j += 1
            else:
                i += 1
                try:
                    while expected[i] == e:
                        i += 1
                finally:
                    j += 1
                    while actual[j] == a:
                        j += 1
        except IndexError:
            missing.extend(expected[i:])
            unexpected.extend(actual[j:])
            break
    return missing, unexpected


def unorderable_list_difference(expected, actual):
    """Same behavior as sorted_list_difference but
    for lists of unorderable items (like dicts).

    As it does a linear search per item (remove) it
    has O(n*n) performance."""
    missing = []
    while expected:
        item = expected.pop()
        try:
            actual.remove(item)
        except ValueError:
            missing.append(item)

    # anything left in actual is unexpected
    return missing, actual

def three_way_cmp(x, y):
    """Return -1 if x < y, 0 if x == y and 1 if x > y"""
    return (x > y) - (x < y)

_Mismatch = namedtuple('Mismatch', 'actual expected value')

def _count_diff_all_purpose(actual, expected):
    'Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ'
    # elements need not be hashable
    s, t = list(actual), list(expected)
    m, n = len(s), len(t)
    NULL = object()
    result = []
    for i, elem in enumerate(s):
        if elem is NULL:
            continue
        cnt_s = cnt_t = 0
        for j in range(i, m):
            if s[j] == elem:
                cnt_s += 1
                s[j] = NULL
        for j, other_elem in enumerate(t):
            if other_elem == elem:
                cnt_t += 1
                t[j] = NULL
        if cnt_s != cnt_t:
            diff = _Mismatch(cnt_s, cnt_t, elem)
            result.append(diff)

    for i, elem in enumerate(t):
        if elem is NULL:
            continue
        cnt_t = 0
        for j in range(i, n):
            if t[j] == elem:
                cnt_t += 1
                t[j] = NULL
        diff = _Mismatch(0, cnt_t, elem)
        result.append(diff)
    return result

def _count_diff_hashable(actual, expected):
    'Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ'
    # elements must be hashable
    s, t = Counter(actual), Counter(expected)
    result = []
    for elem, cnt_s in s.items():
        cnt_t = t.get(elem, 0)
        if cnt_s != cnt_t:
            diff = _Mismatch(cnt_s, cnt_t, elem)
            result.append(diff)
    for elem, cnt_t in t.items():
        if elem not in s:
            diff = _Mismatch(0, cnt_t, elem)
            result.append(diff)
    return result
<|endoftext|>"""Loading unittests."""

import os
import re
import sys
import traceback
import types
import functools
import warnings

from fnmatch import fnmatch, fnmatchcase

from . import case, suite, util

__unittest = True

# what about .pyc (etc)
# we would need to avoid loading the same tests multiple times
# from '.py', *and* '.pyc'
VALID_MODULE_NAME = re.compile(r'[_a-z]\w*\.py$', re.IGNORECASE)


class _FailedTest(case.TestCase):
    _testMethodName = None

    def __init__(self, method_name, exception):
        self._exception = exception
        super(_FailedTest, self).__init__(method_name)

    def __getattr__(self, name):
        if name != self._testMethodName:
            return super(_FailedTest, self).__getattr__(name)
        def testFailure():
            raise self._exception
        return testFailure


def _make_failed_import_test(name, suiteClass):
    message = 'Failed to import test module: %s\n%s' % (
        name, traceback.format_exc())
    return _make_failed_test(name, ImportError(message), suiteClass, message)

def _make_failed_load_tests(name, exception, suiteClass):
    message = 'Failed to call load_tests:\n%s' % (traceback.format_exc(),)
    return _make_failed_test(
        name, exception, suiteClass, message)

def _make_failed_test(methodname, exception, suiteClass, message):
    test = _FailedTest(methodname, exception)
    return suiteClass((test,)), message

def _make_skipped_test(methodname, exception, suiteClass):
    @case.skip(str(exception))
    def testSkipped(self):
        pass
    attrs = {methodname: testSkipped}
    TestClass = type("ModuleSkipped", (case.TestCase,), attrs)
    return suiteClass((TestClass(methodname),))

def _jython_aware_splitext(path):
    if path.lower().endswith('$py.class'):
        return path[:-9]
    return os.path.splitext(path)[0]


class TestLoader(object):
    """
    This class is responsible for loading tests according to various criteria
    and returning them wrapped in a TestSuite
    """
    testMethodPrefix = 'test'
    sortTestMethodsUsing = staticmethod(util.three_way_cmp)
    testNamePatterns = None
    suiteClass = suite.TestSuite
    _top_level_dir = None

    def __init__(self):
        super(TestLoader, self).__init__()
        self.errors = []
        # Tracks packages which we have called into via load_tests, to
        # avoid infinite re-entrancy.
        self._loading_packages = set()

    def loadTestsFromTestCase(self, testCaseClass):
        """Return a suite of all test cases contained in testCaseClass"""
        if issubclass(testCaseClass, suite.TestSuite):
            raise TypeError("Test cases should not be derived from "
                            "TestSuite. Maybe you meant to derive from "
                            "TestCase?")
        testCaseNames = self.getTestCaseNames(testCaseClass)
        if not testCaseNames and hasattr(testCaseClass, 'runTest'):
            testCaseNames = ['runTest']
        loaded_suite = self.suiteClass(map(testCaseClass, testCaseNames))
        return loaded_suite

    # XXX After Python 3.5, remove backward compatibility hacks for
    # use_load_tests deprecation via *args and **kws.  See issue 16662.
    def loadTestsFromModule(self, module, *args, pattern=None, **kws):
        """Return a suite of all test cases contained in the given module"""
        # This method used to take an undocumented and unofficial
        # use_load_tests argument.  For backward compatibility, we still
        # accept the argument (which can also be the first position) but we
        # ignore it and issue a deprecation warning if it's present.
        if len(args) > 0 or 'use_load_tests' in kws:
            warnings.warn('use_load_tests is deprecated and ignored',
                          DeprecationWarning)
            kws.pop('use_load_tests', None)
        if len(args) > 1:
            # Complain about the number of arguments, but don't forget the
            # required `module` argument.
            complaint = len(args) + 1
            raise TypeError('loadTestsFromModule() takes 1 positional argument but {} were given'.format(complaint))
        if len(kws) != 0:
            # Since the keyword arguments are unsorted (see PEP 468), just
            # pick the alphabetically sorted first argument to complain about,
            # if multiple were given.  At least the error message will be
            # predictable.
            complaint = sorted(kws)[0]
            raise TypeError("loadTestsFromModule() got an unexpected keyword argument '{}'".format(complaint))
        tests = []
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, case.TestCase):
                tests.append(self.loadTestsFromTestCase(obj))

        load_tests = getattr(module, 'load_tests', None)
        tests = self.suiteClass(tests)
        if load_tests is not None:
            try:
                return load_tests(self, tests, pattern)
            except Exception as e:
                error_case, error_message = _make_failed_load_tests(
                    module.__name__, e, self.suiteClass)
                self.errors.append(error_message)
                return error_case
        return tests

    def loadTestsFromName(self, name, module=None):
        """Return a suite of all test cases given a string specifier.

        The name may resolve either to a module, a test case class, a
        test method within a test case class, or a callable object which
        returns a TestCase or TestSuite instance.

        The method optionally resolves the names relative to a given module.
        """
        parts = name.split('.')
        error_case, error_message = None, None
        if module is None:
            parts_copy = parts[:]
            while parts_copy:
                try:
                    module_name = '.'.join(parts_copy)
                    module = __import__(module_name)
                    break
                except ImportError:
                    next_attribute = parts_copy.pop()
                    # Last error so we can give it to the user if needed.
                    error_case, error_message = _make_failed_import_test(
                        next_attribute, self.suiteClass)
                    if not parts_copy:
                        # Even the top level import failed: report that error.
                        self.errors.append(error_message)
                        return error_case
            parts = parts[1:]
        obj = module
        for part in parts:
            try:
                parent, obj = obj, getattr(obj, part)
            except AttributeError as e:
                # We can't traverse some part of the name.
                if (getattr(obj, '__path__', None) is not None
                    and error_case is not None):
                    # This is a package (no __path__ per importlib docs), and we
                    # encountered an error importing something. We cannot tell
                    # the difference between package.WrongNameTestClass and
                    # package.wrong_module_name so we just report the
                    # ImportError - it is more informative.
                    self.errors.append(error_message)
                    return error_case
                else:
                    # Otherwise, we signal that an AttributeError has occurred.
                    error_case, error_message = _make_failed_test(
                        part, e, self.suiteClass,
                        'Failed to access attribute:\n%s' % (
                            traceback.format_exc(),))
                    self.errors.append(error_message)
                    return error_case

        if isinstance(obj, types.ModuleType):
            return self.loadTestsFromModule(obj)
        elif isinstance(obj, type) and issubclass(obj, case.TestCase):
            return self.loadTestsFromTestCase(obj)
        elif (isinstance(obj, types.FunctionType) and
              isinstance(parent, type) and
              issubclass(parent, case.TestCase)):
            name = parts[-1]
            inst = parent(name)
            # static methods follow a different path
            if not isinstance(getattr(inst, name), types.FunctionType):
                return self.suiteClass([inst])
        elif isinstance(obj, suite.TestSuite):
            return obj
        if callable(obj):
            test = obj()
            if isinstance(test, suite.TestSuite):
                return test
            elif isinstance(test, case.TestCase):
                return self.suiteClass([test])
            else:
                raise TypeError("calling %s returned %s, not a test" %
                                (obj, test))
        else:
            raise TypeError("don't know how to make test from: %s" % obj)

    def loadTestsFromNames(self, names, module=None):
        """Return a suite of all test cases found using the given sequence
        of string specifiers. See 'loadTestsFromName()'.
        """
        suites = [self.loadTestsFromName(name, module) for name in names]
        return self.suiteClass(suites)

    def getTestCaseNames(self, testCaseClass):
        """Return a sorted sequence of method names found within testCaseClass
        """
        def shouldIncludeMethod(attrname):
            if not attrname.startswith(self.testMethodPrefix):
                return False
            testFunc = getattr(testCaseClass, attrname)
            if not callable(testFunc):
                return False
            fullName = f'%s.%s.%s' % (
                testCaseClass.__module__, testCaseClass.__qualname__, attrname
            )
            return self.testNamePatterns is None or \
                any(fnmatchcase(fullName, pattern) for pattern in self.testNamePatterns)
        testFnNames = list(filter(shouldIncludeMethod, dir(testCaseClass)))
        if self.sortTestMethodsUsing:
            testFnNames.sort(key=functools.cmp_to_key(self.sortTestMethodsUsing))
        return testFnNames

    def discover(self, start_dir, pattern='test*.py', top_level_dir=None):
        """Find and return all test modules from the specified start
        directory, recursing into subdirectories to find them and return all
        tests found within them. Only test files that match the pattern will
        be loaded. (Using shell style pattern matching.)

        All test modules must be importable from the top level of the project.
        If the start directory is not the top level directory then the top
        level directory must be specified separately.

        If a test package name (directory with '__init__.py') matches the
        pattern then the package will be checked for a 'load_tests' function. If
        this exists then it will be called with (loader, tests, pattern) unless
        the package has already had load_tests called from the same discovery
        invocation, in which case the package module object is not scanned for
        tests - this ensures that when a package uses discover to further
        discover child tests that infinite recursion does not happen.

        If load_tests exists then discovery does *not* recurse into the package,
        load_tests is responsible for loading all tests in the package.

        The pattern is deliberately not stored as a loader attribute so that
        packages can continue discovery themselves. top_level_dir is stored so
        load_tests does not need to pass this argument in to loader.discover().

        Paths are sorted before being imported to ensure reproducible execution
        order even on filesystems with non-alphabetical ordering like ext3/4.
        """
        set_implicit_top = False
        if top_level_dir is None and self._top_level_dir is not None:
            # make top_level_dir optional if called from load_tests in a package
            top_level_dir = self._top_level_dir
        elif top_level_dir is None:
            set_implicit_top = True
            top_level_dir = start_dir

        top_level_dir = os.path.abspath(top_level_dir)

        if not top_level_dir in sys.path:
            # all test modules must be importable from the top level directory
            # should we *unconditionally* put the start directory in first
            # in sys.path to minimise likelihood of conflicts between installed
            # modules and development versions?
            sys.path.insert(0, top_level_dir)
        self._top_level_dir = top_level_dir

        is_not_importable = False
        is_namespace = False
        tests = []
        if os.path.isdir(os.path.abspath(start_dir)):
            start_dir = os.path.abspath(start_dir)
            if start_dir != top_level_dir:
                is_not_importable = not os.path.isfile(os.path.join(start_dir, '__init__.py'))
        else:
            # support for discovery from dotted module names
            try:
                __import__(start_dir)
            except ImportError:
                is_not_importable = True
            else:
                the_module = sys.modules[start_dir]
                top_part = start_dir.split('.')[0]
                try:
                    start_dir = os.path.abspath(
                       os.path.dirname((the_module.__file__)))
                except AttributeError:
                    # look for namespace packages
                    try:
                        spec = the_module.__spec__
                    except AttributeError:
                        spec = None

                    if spec and spec.loader is None:
                        if spec.submodule_search_locations is not None:
                            is_namespace = True

                            for path in the_module.__path__:
                                if (not set_implicit_top and
                                    not path.startswith(top_level_dir)):
                                    continue
                                self._top_level_dir = \
                                    (path.split(the_module.__name__
                                         .replace(".", os.path.sep))[0])
                                tests.extend(self._find_tests(path,
                                                              pattern,
                                                              namespace=True))
                    elif the_module.__name__ in sys.builtin_module_names:
                        # builtin module
                        raise TypeError('Can not use builtin modules '
                                        'as dotted module names') from None
                    else:
                        raise TypeError(
                            'don\'t know how to discover from {!r}'
                            .format(the_module)) from None

                if set_implicit_top:
                    if not is_namespace:
                        self._top_level_dir = \
                           self._get_directory_containing_module(top_part)
                        sys.path.remove(top_level_dir)
                    else:
                        sys.path.remove(top_level_dir)

        if is_not_importable:
            raise ImportError('Start directory is not importable: %r' % start_dir)

        if not is_namespace:
            tests = list(self._find_tests(start_dir, pattern))
        return self.suiteClass(tests)

    def _get_directory_containing_module(self, module_name):
        module = sys.modules[module_name]
        full_path = os.path.abspath(module.__file__)

        if os.path.basename(full_path).lower().startswith('__init__.py'):
            return os.path.dirname(os.path.dirname(full_path))
        else:
            # here we have been given a module rather than a package - so
            # all we can do is search the *same* directory the module is in
            # should an exception be raised instead
            return os.path.dirname(full_path)

    def _get_name_from_path(self, path):
        if path == self._top_level_dir:
            return '.'
        path = _jython_aware_splitext(os.path.normpath(path))

        _relpath = os.path.relpath(path, self._top_level_dir)
        assert not os.path.isabs(_relpath), "Path must be within the project"
        assert not _relpath.startswith('..'), "Path must be within the project"

        name = _relpath.replace(os.path.sep, '.')
        return name

    def _get_module_from_name(self, name):
        __import__(name)
        return sys.modules[name]

    def _match_path(self, path, full_path, pattern):
        # override this method to use alternative matching strategy
        return fnmatch(path, pattern)

    def _find_tests(self, start_dir, pattern, namespace=False):
        """Used by discovery. Yields test suites it loads."""
        # Handle the __init__ in this package
        name = self._get_name_from_path(start_dir)
        # name is '.' when start_dir == top_level_dir (and top_level_dir is by
        # definition not a package).
        if name != '.' and name not in self._loading_packages:
            # name is in self._loading_packages while we have called into
            # loadTestsFromModule with name.
            tests, should_recurse = self._find_test_path(
                start_dir, pattern, namespace)
            if tests is not None:
                yield tests
            if not should_recurse:
                # Either an error occurred, or load_tests was used by the
                # package.
                return
        # Handle the contents.
        paths = sorted(os.listdir(start_dir))
        for path in paths:
            full_path = os.path.join(start_dir, path)
            tests, should_recurse = self._find_test_path(
                full_path, pattern, namespace)
            if tests is not None:
                yield tests
            if should_recurse:
                # we found a package that didn't use load_tests.
                name = self._get_name_from_path(full_path)
                self._loading_packages.add(name)
                try:
                    yield from self._find_tests(full_path, pattern, namespace)
                finally:
                    self._loading_packages.discard(name)

    def _find_test_path(self, full_path, pattern, namespace=False):
        """Used by discovery.

        Loads tests from a single file, or a directories' __init__.py when
        passed the directory.

        Returns a tuple (None_or_tests_from_file, should_recurse).
        """
        basename = os.path.basename(full_path)
        if os.path.isfile(full_path):
            if not VALID_MODULE_NAME.match(basename):
                # valid Python identifiers only
                return None, False
            if not self._match_path(basename, full_path, pattern):
                return None, False
            # if the test file matches, load it
            name = self._get_name_from_path(full_path)
            try:
                module = self._get_module_from_name(name)
            except case.SkipTest as e:
                return _make_skipped_test(name, e, self.suiteClass), False
            except:
                error_case, error_message = \
                    _make_failed_import_test(name, self.suiteClass)
                self.errors.append(error_message)
                return error_case, False
            else:
                mod_file = os.path.abspath(
                    getattr(module, '__file__', full_path))
                realpath = _jython_aware_splitext(
                    os.path.realpath(mod_file))
                fullpath_noext = _jython_aware_splitext(
                    os.path.realpath(full_path))
                if realpath.lower() != fullpath_noext.lower():
                    module_dir = os.path.dirname(realpath)
                    mod_name = _jython_aware_splitext(
                        os.path.basename(full_path))
                    expected_dir = os.path.dirname(full_path)
                    msg = ("%r module incorrectly imported from %r. Expected "
                           "%r. Is this module globally installed?")
                    raise ImportError(
                        msg % (mod_name, module_dir, expected_dir))
                return self.loadTestsFromModule(module, pattern=pattern), False
        elif os.path.isdir(full_path):
            if (not namespace and
                not os.path.isfile(os.path.join(full_path, '__init__.py'))):
                return None, False

            load_tests = None
            tests = None
            name = self._get_name_from_path(full_path)
            try:
                package = self._get_module_from_name(name)
            except case.SkipTest as e:
                return _make_skipped_test(name, e, self.suiteClass), False
            except:
                error_case, error_message = \
                    _make_failed_import_test(name, self.suiteClass)
                self.errors.append(error_message)
                return error_case, False
            else:
                load_tests = getattr(package, 'load_tests', None)
                # Mark this package as being in load_tests (possibly ;))
                self._loading_packages.add(name)
                try:
                    tests = self.loadTestsFromModule(package, pattern=pattern)
                    if load_tests is not None:
                        # loadTestsFromModule(package) has loaded tests for us.
                        return tests, False
                    return tests, True
                finally:
                    self._loading_packages.discard(name)
        else:
            return None, False


defaultTestLoader = TestLoader()


def _makeLoader(prefix, sortUsing, suiteClass=None, testNamePatterns=None):
    loader = TestLoader()
    loader.sortTestMethodsUsing = sortUsing
    loader.testMethodPrefix = prefix
    loader.testNamePatterns = testNamePatterns
    if suiteClass:
        loader.suiteClass = suiteClass
    return loader

def getTestCaseNames(testCaseClass, prefix, sortUsing=util.three_way_cmp, testNamePatterns=None):
    return _makeLoader(prefix, sortUsing, testNamePatterns=testNamePatterns).getTestCaseNames(testCaseClass)

def makeSuite(testCaseClass, prefix='test', sortUsing=util.three_way_cmp,
              suiteClass=suite.TestSuite):
    return _makeLoader(prefix, sortUsing, suiteClass).loadTestsFromTestCase(
        testCaseClass)

def findTestCases(module, prefix='test', sortUsing=util.three_way_cmp,
                  suiteClass=suite.TestSuite):
    return _makeLoader(prefix, sortUsing, suiteClass).loadTestsFromModule(\
        module)
<|endoftext|>"""Test case implementation"""

import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types

from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
                   _count_diff_hashable, _common_shorten_repr)

__unittest = True

_subtest_msg_sentinel = object()

DIFF_OMITTED = ('\nDiff is %s characters long. '
                 'Set self.maxDiff to None to see it.')

class SkipTest(Exception):
    """
    Raise this exception in a test to skip it.

    Usually you can use TestCase.skipTest() or one of the skipping decorators
    instead of raising this directly.
    """

class _ShouldStop(Exception):
    """
    The test should stop.
    """

class _UnexpectedSuccess(Exception):
    """
    The test was supposed to fail, but it didn't!
    """


class _Outcome(object):
    def __init__(self, result=None):
        self.expecting_failure = False
        self.result = result
        self.result_supports_subtests = hasattr(result, "addSubTest")
        self.success = True
        self.skipped = []
        self.expectedFailure = None
        self.errors = []

    @contextlib.contextmanager
    def testPartExecutor(self, test_case, isTest=False):
        old_success = self.success
        self.success = True
        try:
            yield
        except KeyboardInterrupt:
            raise
        except SkipTest as e:
            self.success = False
            self.skipped.append((test_case, str(e)))
        except _ShouldStop:
            pass
        except:
            exc_info = sys.exc_info()
            if self.expecting_failure:
                self.expectedFailure = exc_info
            else:
                self.success = False
                self.errors.append((test_case, exc_info))
            # explicitly break a reference cycle:
            # exc_info -> frame -> exc_info
            exc_info = None
        else:
            if self.result_supports_subtests and self.success:
                self.errors.append((test_case, None))
        finally:
            self.success = self.success and old_success


def _id(obj):
    return obj


_module_cleanups = []
def addModuleCleanup(function, /, *args, **kwargs):
    """Same as addCleanup, except the cleanup items are called even if
    setUpModule fails (unlike tearDownModule)."""
    _module_cleanups.append((function, args, kwargs))


def doModuleCleanups():
    """Execute all module cleanup functions. Normally called for you after
    tearDownModule."""
    exceptions = []
    while _module_cleanups:
        function, args, kwargs = _module_cleanups.pop()
        try:
            function(*args, **kwargs)
        except Exception as exc:
            exceptions.append(exc)
    if exceptions:
        # Swallows all but first exception. If a multi-exception handler
        # gets written we should use that here instead.
        raise exceptions[0]


def skip(reason):
    """
    Unconditionally skip a test.
    """
    def decorator(test_item):
        if not isinstance(test_item, type):
            @functools.wraps(test_item)
            def skip_wrapper(*args, **kwargs):
                raise SkipTest(reason)
            test_item = skip_wrapper

        test_item.__unittest_skip__ = True
        test_item.__unittest_skip_why__ = reason
        return test_item
    if isinstance(reason, types.FunctionType):
        test_item = reason
        reason = ''
        return decorator(test_item)
    return decorator

def skipIf(condition, reason):
    """
    Skip a test if the condition is true.
    """
    if condition:
        return skip(reason)
    return _id

def skipUnless(condition, reason):
    """
    Skip a test unless the condition is true.
    """
    if not condition:
        return skip(reason)
    return _id

def expectedFailure(test_item):
    test_item.__unittest_expecting_failure__ = True
    return test_item

def _is_subtype(expected, basetype):
    if isinstance(expected, tuple):
        return all(_is_subtype(e, basetype) for e in expected)
    return isinstance(expected, type) and issubclass(expected, basetype)

class _BaseTestCaseContext:

    def __init__(self, test_case):
        self.test_case = test_case

    def _raiseFailure(self, standardMsg):
        msg = self.test_case._formatMessage(self.msg, standardMsg)
        raise self.test_case.failureException(msg)

class _AssertRaisesBaseContext(_BaseTestCaseContext):

    def __init__(self, expected, test_case, expected_regex=None):
        _BaseTestCaseContext.__init__(self, test_case)
        self.expected = expected
        self.test_case = test_case
        if expected_regex is not None:
            expected_regex = re.compile(expected_regex)
        self.expected_regex = expected_regex
        self.obj_name = None
        self.msg = None

    def handle(self, name, args, kwargs):
        """
        If args is empty, assertRaises/Warns is being used as a
        context manager, so check for a 'msg' kwarg and return self.
        If args is not empty, call a callable passing positional and keyword
        arguments.
        """
        try:
            if not _is_subtype(self.expected, self._base_type):
                raise TypeError('%s() arg 1 must be %s' %
                                (name, self._base_type_str))
            if not args:
                self.msg = kwargs.pop('msg', None)
                if kwargs:
                    raise TypeError('%r is an invalid keyword argument for '
                                    'this function' % (next(iter(kwargs)),))
                return self

            callable_obj, *args = args
            try:
                self.obj_name = callable_obj.__name__
            except AttributeError:
                self.obj_name = str(callable_obj)
            with self:
                callable_obj(*args, **kwargs)
        finally:
            # bpo-23890: manually break a reference cycle
            self = None


class _AssertRaisesContext(_AssertRaisesBaseContext):
    """A context manager used to implement TestCase.assertRaises* methods."""

    _base_type = BaseException
    _base_type_str = 'an exception type or tuple of exception types'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            try:
                exc_name = self.expected.__name__
            except AttributeError:
                exc_name = str(self.expected)
            if self.obj_name:
                self._raiseFailure("{} not raised by {}".format(exc_name,
                                                                self.obj_name))
            else:
                self._raiseFailure("{} not raised".format(exc_name))
        else:
            traceback.clear_frames(tb)
        if not issubclass(exc_type, self.expected):
            # let unexpected exceptions pass through
            return False
        # store exception, without traceback, for later retrieval
        self.exception = exc_value.with_traceback(None)
        if self.expected_regex is None:
            return True

        expected_regex = self.expected_regex
        if not expected_regex.search(str(exc_value)):
            self._raiseFailure('"{}" does not match "{}"'.format(
                     expected_regex.pattern, str(exc_value)))
        return True

    __class_getitem__ = classmethod(types.GenericAlias)


class _AssertWarnsContext(_AssertRaisesBaseContext):
    """A context manager used to implement TestCase.assertWarns* methods."""

    _base_type = Warning
    _base_type_str = 'a warning type or tuple of warning types'

    def __enter__(self):
        # The __warningregistry__'s need to be in a pristine state for tests
        # to work properly.
        for v in list(sys.modules.values()):
            if getattr(v, '__warningregistry__', None):
                v.__warningregistry__ = {}
        self.warnings_manager = warnings.catch_warnings(record=True)
        self.warnings = self.warnings_manager.__enter__()
        warnings.simplefilter("always", self.expected)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.warnings_manager.__exit__(exc_type, exc_value, tb)
        if exc_type is not None:
            # let unexpected exceptions pass through
            return
        try:
            exc_name = self.expected.__name__
        except AttributeError:
            exc_name = str(self.expected)
        first_matching = None
        for m in self.warnings:
            w = m.message
            if not isinstance(w, self.expected):
                continue
            if first_matching is None:
                first_matching = w
            if (self.expected_regex is not None and
                not self.expected_regex.search(str(w))):
                continue
            # store warning for later retrieval
            self.warning = w
            self.filename = m.filename
            self.lineno = m.lineno
            return
        # Now we simply try to choose a helpful failure message
        if first_matching is not None:
            self._raiseFailure('"{}" does not match "{}"'.format(
                     self.expected_regex.pattern, str(first_matching)))
        if self.obj_name:
            self._raiseFailure("{} not triggered by {}".format(exc_name,
                                                               self.obj_name))
        else:
            self._raiseFailure("{} not triggered".format(exc_name))


class _OrderedChainMap(collections.ChainMap):
    def __iter__(self):
        seen = set()
        for mapping in self.maps:
            for k in mapping:
                if k not in seen:
                    seen.add(k)
                    yield k


class TestCase(object):
    """A class whose instances are single test cases.

    By default, the test code itself should be placed in a method named
    'runTest'.

    If the fixture may be used for many test cases, create as
    many test methods as are needed. When instantiating such a TestCase
    subclass, specify in the constructor arguments the name of the test method
    that the instance is to execute.

    Test authors should subclass TestCase for their own tests. Construction
    and deconstruction of the test's environment ('fixture') can be
    implemented by overriding the 'setUp' and 'tearDown' methods respectively.

    If it is necessary to override the __init__ method, the base class
    __init__ method must always be called. It is important that subclasses
    should not change the signature of their __init__ method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.

    When subclassing TestCase, you can set these attributes:
    * failureException: determines which exception will be raised when
        the instance's assertion methods fail; test methods raising this
        exception will be deemed to have 'failed' rather than 'errored'.
    * longMessage: determines whether long messages (including repr of
        objects used in assert methods) will be printed on failure in *addition*
        to any explicit message passed.
    * maxDiff: sets the maximum length of a diff in failure messages
        by assert methods using difflib. It is looked up as an instance
        attribute so can be configured by individual tests if required.
    """

    failureException = AssertionError

    longMessage = True

    maxDiff = 80*8

    # If a string is longer than _diffThreshold, use normal comparison instead
    # of difflib.  See #11763.
    _diffThreshold = 2**16

    def __init_subclass__(cls, *args, **kwargs):
        # Attribute used by TestSuite for classSetUp
        cls._classSetupFailed = False
        cls._class_cleanups = []
        super().__init_subclass__(*args, **kwargs)

    def __init__(self, methodName='runTest'):
        """Create an instance of the class that will use the named test
           method when executed. Raises a ValueError if the instance does
           not have a method with the specified name.
        """
        self._testMethodName = methodName
        self._outcome = None
        self._testMethodDoc = 'No test'
        try:
            testMethod = getattr(self, methodName)
        except AttributeError:
            if methodName != 'runTest':
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError("no such test method in %s: %s" %
                      (self.__class__, methodName))
        else:
            self._testMethodDoc = testMethod.__doc__
        self._cleanups = []
        self._subtest = None

        # Map types to custom assertEqual functions that will compare
        # instances of said type in more detail to generate a more useful
        # error message.
        self._type_equality_funcs = {}
        self.addTypeEqualityFunc(dict, 'assertDictEqual')
        self.addTypeEqualityFunc(list, 'assertListEqual')
        self.addTypeEqualityFunc(tuple, 'assertTupleEqual')
        self.addTypeEqualityFunc(set, 'assertSetEqual')
        self.addTypeEqualityFunc(frozenset, 'assertSetEqual')
        self.addTypeEqualityFunc(str, 'assertMultiLineEqual')

    def addTypeEqualityFunc(self, typeobj, function):
        """Add a type specific assertEqual style function to compare a type.

        This method is for use by TestCase subclasses that need to register
        their own type equality functions to provide nicer error messages.

        Args:
            typeobj: The data type to call this function on when both values
                    are of the same type in assertEqual().
            function: The callable taking two arguments and an optional
                    msg= argument that raises self.failureException with a
                    useful error message when the two arguments are not equal.
        """
        self._type_equality_funcs[typeobj] = function

    def addCleanup(self, function, /, *args, **kwargs):
        """Add a function, with arguments, to be called when the test is
        completed. Functions added are called on a LIFO basis and are
        called after tearDown on test failure or success.

        Cleanup items are called even if setUp fails (unlike tearDown)."""
        self._cleanups.append((function, args, kwargs))

    @classmethod
    def addClassCleanup(cls, function, /, *args, **kwargs):
        """Same as addCleanup, except the cleanup items are called even if
        setUpClass fails (unlike tearDownClass)."""
        cls._class_cleanups.append((function, args, kwargs))

    def setUp(self):
        "Hook method for setting up the test fixture before exercising it."
        pass

    def tearDown(self):
        "Hook method for deconstructing the test fixture after testing it."
        pass

    @classmethod
    def setUpClass(cls):
        "Hook method for setting up class fixture before running tests in the class."

    @classmethod
    def tearDownClass(cls):
        "Hook method for deconstructing the class fixture after running all tests in the class."

    def countTestCases(self):
        return 1

    def defaultTestResult(self):
        return result.TestResult()

    def shortDescription(self):
        """Returns a one-line description of the test, or None if no
        description has been provided.

        The default implementation of this method returns the first line of
        the specified test method's docstring.
        """
        doc = self._testMethodDoc
        return doc.strip().split("\n")[0].strip() if doc else None


    def id(self):
        return "%s.%s" % (strclass(self.__class__), self._testMethodName)

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented

        return self._testMethodName == other._testMethodName

    def __hash__(self):
        return hash((type(self), self._testMethodName))

    def __str__(self):
        return "%s (%s)" % (self._testMethodName, strclass(self.__class__))

    def __repr__(self):
        return "<%s testMethod=%s>" % \
               (strclass(self.__class__), self._testMethodName)

    def _addSkip(self, result, test_case, reason):
        addSkip = getattr(result, 'addSkip', None)
        if addSkip is not None:
            addSkip(test_case, reason)
        else:
            warnings.warn("TestResult has no addSkip method, skips not reported",
                          RuntimeWarning, 2)
            result.addSuccess(test_case)

    @contextlib.contextmanager
    def subTest(self, msg=_subtest_msg_sentinel, **params):
        """Return a context manager that will return the enclosed block
        of code in a subtest identified by the optional message and
        keyword parameters.  A failure in the subtest marks the test
        case as failed but resumes execution at the end of the enclosed
        block, allowing further test code to be executed.
        """
        if self._outcome is None or not self._outcome.result_supports_subtests:
            yield
            return
        parent = self._subtest
        if parent is None:
            params_map = _OrderedChainMap(params)
        else:
            params_map = parent.params.new_child(params)
        self._subtest = _SubTest(self, msg, params_map)
        try:
            with self._outcome.testPartExecutor(self._subtest, isTest=True):
                yield
            if not self._outcome.success:
                result = self._outcome.result
                if result is not None and result.failfast:
                    raise _ShouldStop
            elif self._outcome.expectedFailure:
                # If the test is expecting a failure, we really want to
                # stop now and register the expected failure.
                raise _ShouldStop
        finally:
            self._subtest = parent

    def _feedErrorsToResult(self, result, errors):
        for test, exc_info in errors:
            if isinstance(test, _SubTest):
                result.addSubTest(test.test_case, test, exc_info)
            elif exc_info is not None:
                if issubclass(exc_info[0], self.failureException):
                    result.addFailure(test, exc_info)
                else:
                    result.addError(test, exc_info)

    def _addExpectedFailure(self, result, exc_info):
        try:
            addExpectedFailure = result.addExpectedFailure
        except AttributeError:
            warnings.warn("TestResult has no addExpectedFailure method, reporting as passes",
                          RuntimeWarning)
            result.addSuccess(self)
        else:
            addExpectedFailure(self, exc_info)

    def _addUnexpectedSuccess(self, result):
        try:
            addUnexpectedSuccess = result.addUnexpectedSuccess
        except AttributeError:
            warnings.warn("TestResult has no addUnexpectedSuccess method, reporting as failure",
                          RuntimeWarning)
            # We need to pass an actual exception and traceback to addFailure,
            # otherwise the legacy result can choke.
            try:
                raise _UnexpectedSuccess from None
            except _UnexpectedSuccess:
                result.addFailure(self, sys.exc_info())
        else:
            addUnexpectedSuccess(self)

    def _callSetUp(self):
        self.setUp()

    def _callTestMethod(self, method):
        method()

    def _callTearDown(self):
        self.tearDown()

    def _callCleanup(self, function, /, *args, **kwargs):
        function(*args, **kwargs)

    def run(self, result=None):
        if result is None:
            result = self.defaultTestResult()
            startTestRun = getattr(result, 'startTestRun', None)
            stopTestRun = getattr(result, 'stopTestRun', None)
            if startTestRun is not None:
                startTestRun()
        else:
            stopTestRun = None

        result.startTest(self)
        try:
            testMethod = getattr(self, self._testMethodName)
            if (getattr(self.__class__, "__unittest_skip__", False) or
                getattr(testMethod, "__unittest_skip__", False)):
                # If the class or method was skipped.
                skip_why = (getattr(self.__class__, '__unittest_skip_why__', '')
                            or getattr(testMethod, '__unittest_skip_why__', ''))
                self._addSkip(result, self, skip_why)
                return result

            expecting_failure = (
                getattr(self, "__unittest_expecting_failure__", False) or
                getattr(testMethod, "__unittest_expecting_failure__", False)
            )
            outcome = _Outcome(result)
            try:
                self._outcome = outcome

                with outcome.testPartExecutor(self):
                    self._callSetUp()
                if outcome.success:
                    outcome.expecting_failure = expecting_failure
                    with outcome.testPartExecutor(self, isTest=True):
                        self._callTestMethod(testMethod)
                    outcome.expecting_failure = False
                    with outcome.testPartExecutor(self):
                        self._callTearDown()

                self.doCleanups()
                for test, reason in outcome.skipped:
                    self._addSkip(result, test, reason)
                self._feedErrorsToResult(result, outcome.errors)
                if outcome.success:
                    if expecting_failure:
                        if outcome.expectedFailure:
                            self._addExpectedFailure(result, outcome.expectedFailure)
                        else:
                            self._addUnexpectedSuccess(result)
                    else:
                        result.addSuccess(self)
                return result
            finally:
                # explicitly break reference cycles:
                # outcome.errors -> frame -> outcome -> outcome.errors
                # outcome.expectedFailure -> frame -> outcome -> outcome.expectedFailure
                outcome.errors.clear()
                outcome.expectedFailure = None

                # clear the outcome, no more needed
                self._outcome = None

        finally:
            result.stopTest(self)
            if stopTestRun is not None:
                stopTestRun()

    def doCleanups(self):
        """Execute all cleanup functions. Normally called for you after
        tearDown."""
        outcome = self._outcome or _Outcome()
        while self._cleanups:
            function, args, kwargs = self._cleanups.pop()
            with outcome.testPartExecutor(self):
                self._callCleanup(function, *args, **kwargs)

        # return this for backwards compatibility
        # even though we no longer use it internally
        return outcome.success

    @classmethod
    def doClassCleanups(cls):
        """Execute all class cleanup functions. Normally called for you after
        tearDownClass."""
        cls.tearDown_exceptions = []
        while cls._class_cleanups:
            function, args, kwargs = cls._class_cleanups.pop()
            try:
                function(*args, **kwargs)
            except Exception:
                cls.tearDown_exceptions.append(sys.exc_info())

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def debug(self):
        """Run the test without collecting errors in a TestResult"""
        testMethod = getattr(self, self._testMethodName)
        if (getattr(self.__class__, "__unittest_skip__", False) or
            getattr(testMethod, "__unittest_skip__", False)):
            # If the class or method was skipped.
            skip_why = (getattr(self.__class__, '__unittest_skip_why__', '')
                        or getattr(testMethod, '__unittest_skip_why__', ''))
            raise SkipTest(skip_why)

        self._callSetUp()
        self._callTestMethod(testMethod)
        self._callTearDown()
        while self._cleanups:
            function, args, kwargs = self._cleanups.pop()
            self._callCleanup(function, *args, **kwargs)

    def skipTest(self, reason):
        """Skip this test."""
        raise SkipTest(reason)

    def fail(self, msg=None):
        """Fail immediately, with the given message."""
        raise self.failureException(msg)

    def assertFalse(self, expr, msg=None):
        """Check that the expression is false."""
        if expr:
            msg = self._formatMessage(msg, "%s is not false" % safe_repr(expr))
            raise self.failureException(msg)

    def assertTrue(self, expr, msg=None):
        """Check that the expression is true."""
        if not expr:
            msg = self._formatMessage(msg, "%s is not true" % safe_repr(expr))
            raise self.failureException(msg)

    def _formatMessage(self, msg, standardMsg):
        """Honour the longMessage attribute when generating failure messages.
        If longMessage is False this means:
        * Use only an explicit message if it is provided
        * Otherwise use the standard message for the assert

        If longMessage is True:
        * Use the standard message
        * If an explicit message is provided, plus ' : ' and the explicit message
        """
        if not self.longMessage:
            return msg or standardMsg
        if msg is None:
            return standardMsg
        try:
            # don't switch to '{}' formatting in Python 2.X
            # it changes the way unicode input is handled
            return '%s : %s' % (standardMsg, msg)
        except UnicodeDecodeError:
            return  '%s : %s' % (safe_repr(standardMsg), safe_repr(msg))

    def assertRaises(self, expected_exception, *args, **kwargs):
        """Fail unless an exception of class expected_exception is raised
           by the callable when invoked with specified positional and
           keyword arguments. If a different type of exception is
           raised, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertRaises(SomeException):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertRaises
           is used as a context object.

           The context manager keeps a reference to the exception as
           the 'exception' attribute. This allows you to inspect the
           exception after the assertion::

               with self.assertRaises(SomeException) as cm:
                   do_something()
               the_exception = cm.exception
               self.assertEqual(the_exception.error_code, 3)
        """
        context = _AssertRaisesContext(expected_exception, self)
        try:
            return context.handle('assertRaises', args, kwargs)
        finally:
            # bpo-23890: manually break a reference cycle
            context = None

    def assertWarns(self, expected_warning, *args, **kwargs):
        """Fail unless a warning of class warnClass is triggered
           by the callable when invoked with specified positional and
           keyword arguments.  If a different type of warning is
           triggered, it will not be handled: depending on the other
           warning filtering rules in effect, it might be silenced, printed
           out, or raised as an exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertWarns(SomeWarning):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertWarns
           is used as a context object.

           The context manager keeps a reference to the first matching
           warning as the 'warning' attribute; similarly, the 'filename'
           and 'lineno' attributes give you information about the line
           of Python code from which the warning was triggered.
           This allows you to inspect the warning after the assertion::

               with self.assertWarns(SomeWarning) as cm:
                   do_something()
               the_warning = cm.warning
               self.assertEqual(the_warning.some_attribute, 147)
        """
        context = _AssertWarnsContext(expected_warning, self)
        return context.handle('assertWarns', args, kwargs)

    def assertLogs(self, logger=None, level=None):
        """Fail unless a log message of level *level* or higher is emitted
        on *logger_name* or its children.  If omitted, *level* defaults to
        INFO and *logger* defaults to the root logger.

        This method must be used as a context manager, and will yield
        a recording object with two attributes: `output` and `records`.
        At the end of the context manager, the `output` attribute will
        be a list of the matching formatted log messages and the
        `records` attribute will be a list of the corresponding LogRecord
        objects.

        Example::

            with self.assertLogs('foo', level='INFO') as cm:
                logging.getLogger('foo').info('first message')
                logging.getLogger('foo.bar').error('second message')
            self.assertEqual(cm.output, ['INFO:foo:first message',
                                         'ERROR:foo.bar:second message'])
        """
        # Lazy import to avoid importing logging if it is not needed.
        from ._log import _AssertLogsContext
        return _AssertLogsContext(self, logger, level, no_logs=False)

    def assertNoLogs(self, logger=None, level=None):
        """ Fail unless no log messages of level *level* or higher are emitted
        on *logger_name* or its children.

        This method must be used as a context manager.
        """
        from ._log import _AssertLogsContext
        return _AssertLogsContext(self, logger, level, no_logs=True)

    def _getAssertEqualityFunc(self, first, second):
        """Get a detailed comparison function for the types of the two args.

        Returns: A callable accepting (first, second, msg=None) that will
        raise a failure exception if first != second with a useful human
        readable error message for those types.
        """
        #
        # NOTE(gregory.p.smith): I considered isinstance(first, type(second))
        # and vice versa.  I opted for the conservative approach in case
        # subclasses are not intended to be compared in detail to their super
        # class instances using a type equality func.  This means testing
        # subtypes won't automagically use the detailed comparison.  Callers
        # should use their type specific assertSpamEqual method to compare
        # subclasses if the detailed comparison is desired and appropriate.
        # See the discussion in http://bugs.python.org/issue2578.
        #
        if type(first) is type(second):
            asserter = self._type_equality_funcs.get(type(first))
            if asserter is not None:
                if isinstance(asserter, str):
                    asserter = getattr(self, asserter)
                return asserter

        return self._baseAssertEqual

    def _baseAssertEqual(self, first, second, msg=None):
        """The default assertEqual implementation, not type specific."""
        if not first == second:
            standardMsg = '%s != %s' % _common_shorten_repr(first, second)
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def assertEqual(self, first, second, msg=None):
        """Fail if the two objects are unequal as determined by the '=='
           operator.
        """
        assertion_func = self._getAssertEqualityFunc(first, second)
        assertion_func(first, second, msg=msg)

    def assertNotEqual(self, first, second, msg=None):
        """Fail if the two objects are equal as determined by the '!='
           operator.
        """
        if not first != second:
            msg = self._formatMessage(msg, '%s == %s' % (safe_repr(first),
                                                          safe_repr(second)))
            raise self.failureException(msg)

    def assertAlmostEqual(self, first, second, places=None, msg=None,
                          delta=None):
        """Fail if the two objects are unequal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is more than the given
           delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           If the two objects compare equal then they will automatically
           compare almost equal.
        """
        if first == second:
            # shortcut
            return
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        diff = abs(first - second)
        if delta is not None:
            if diff <= delta:
                return

            standardMsg = '%s != %s within %s delta (%s difference)' % (
                safe_repr(first),
                safe_repr(second),
                safe_repr(delta),
                safe_repr(diff))
        else:
            if places is None:
                places = 7

            if round(diff, places) == 0:
                return

            standardMsg = '%s != %s within %r places (%s difference)' % (
                safe_repr(first),
                safe_repr(second),
                places,
                safe_repr(diff))
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertNotAlmostEqual(self, first, second, places=None, msg=None,
                             delta=None):
        """Fail if the two objects are equal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           difference between the two objects is less than the given delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most significant digit).

           Objects that are equal automatically fail.
        """
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")
        diff = abs(first - second)
        if delta is not None:
            if not (first == second) and diff > delta:
                return
            standardMsg = '%s == %s within %s delta (%s difference)' % (
                safe_repr(first),
                safe_repr(second),
                safe_repr(delta),
                safe_repr(diff))
        else:
            if places is None:
                places = 7
            if not (first == second) and round(diff, places) != 0:
                return
            standardMsg = '%s == %s within %r places' % (safe_repr(first),
                                                         safe_repr(second),
                                                         places)

        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        """An equality assertion for ordered sequences (like lists and tuples).

        For the purposes of this function, a valid ordered sequence type is one
        which can be indexed, has a length, and has an equality operator.

        Args:
            seq1: The first sequence to compare.
            seq2: The second sequence to compare.
            seq_type: The expected datatype of the sequences, or None if no
                    datatype should be enforced.
            msg: Optional message to use on failure instead of a list of
                    differences.
        """
        if seq_type is not None:
            seq_type_name = seq_type.__name__
            if not isinstance(seq1, seq_type):
                raise self.failureException('First sequence is not a %s: %s'
                                        % (seq_type_name, safe_repr(seq1)))
            if not isinstance(seq2, seq_type):
                raise self.failureException('Second sequence is not a %s: %s'
                                        % (seq_type_name, safe_repr(seq2)))
        else:
            seq_type_name = "sequence"

        differing = None
        try:
            len1 = len(seq1)
        except (TypeError, NotImplementedError):
            differing = 'First %s has no length.    Non-sequence?' % (
                    seq_type_name)

        if differing is None:
            try:
                len2 = len(seq2)
            except (TypeError, NotImplementedError):
                differing = 'Second %s has no length.    Non-sequence?' % (
                        seq_type_name)

        if differing is None:
            if seq1 == seq2:
                return

            differing = '%ss differ: %s != %s\n' % (
                    (seq_type_name.capitalize(),) +
                    _common_shorten_repr(seq1, seq2))

            for i in range(min(len1, len2)):
                try:
                    item1 = seq1[i]
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('\nUnable to index element %d of first %s\n' %
                                 (i, seq_type_name))
                    break

                try:
                    item2 = seq2[i]
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('\nUnable to index element %d of second %s\n' %
                                 (i, seq_type_name))
                    break

                if item1 != item2:
                    differing += ('\nFirst differing element %d:\n%s\n%s\n' %
                                 ((i,) + _common_shorten_repr(item1, item2)))
                    break
            else:
                if (len1 == len2 and seq_type is None and
                    type(seq1) != type(seq2)):
                    # The sequences are the same, but have differing types.
                    return

            if len1 > len2:
                differing += ('\nFirst %s contains %d additional '
                             'elements.\n' % (seq_type_name, len1 - len2))
                try:
                    differing += ('First extra element %d:\n%s\n' %
                                  (len2, safe_repr(seq1[len2])))
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('Unable to index element %d '
                                  'of first %s\n' % (len2, seq_type_name))
            elif len1 < len2:
                differing += ('\nSecond %s contains %d additional '
                             'elements.\n' % (seq_type_name, len2 - len1))
                try:
                    differing += ('First extra element %d:\n%s\n' %
                                  (len1, safe_repr(seq2[len1])))
                except (TypeError, IndexError, NotImplementedError):
                    differing += ('Unable to index element %d '
                                  'of second %s\n' % (len1, seq_type_name))
        standardMsg = differing
        diffMsg = '\n' + '\n'.join(
            difflib.ndiff(pprint.pformat(seq1).splitlines(),
                          pprint.pformat(seq2).splitlines()))

        standardMsg = self._truncateMessage(standardMsg, diffMsg)
        msg = self._formatMessage(msg, standardMsg)
        self.fail(msg)

    def _truncateMessage(self, message, diff):
        max_diff = self.maxDiff
        if max_diff is None or len(diff) <= max_diff:
            return message + diff
        return message + (DIFF_OMITTED % len(diff))

    def assertListEqual(self, list1, list2, msg=None):
        """A list-specific equality assertion.

        Args:
            list1: The first list to compare.
            list2: The second list to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        """
        self.assertSequenceEqual(list1, list2, msg, seq_type=list)

    def assertTupleEqual(self, tuple1, tuple2, msg=None):
        """A tuple-specific equality assertion.

        Args:
            tuple1: The first tuple to compare.
            tuple2: The second tuple to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.
        """
        self.assertSequenceEqual(tuple1, tuple2, msg, seq_type=tuple)

    def assertSetEqual(self, set1, set2, msg=None):
        """A set-specific equality assertion.

        Args:
            set1: The first set to compare.
            set2: The second set to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        assertSetEqual uses ducktyping to support different types of sets, and
        is optimized for sets specifically (parameters must support a
        difference method).
        """
        try:
            difference1 = set1.difference(set2)
        except TypeError as e:
            self.fail('invalid type when attempting set difference: %s' % e)
        except AttributeError as e:
            self.fail('first argument does not support set difference: %s' % e)

        try:
            difference2 = set2.difference(set1)
        except TypeError as e:
            self.fail('invalid type when attempting set difference: %s' % e)
        except AttributeError as e:
            self.fail('second argument does not support set difference: %s' % e)

        if not (difference1 or difference2):
            return

        lines = []
        if difference1:
            lines.append('Items in the first set but not the second:')
            for item in difference1:
                lines.append(repr(item))
        if difference2:
            lines.append('Items in the second set but not the first:')
            for item in difference2:
                lines.append(repr(item))

        standardMsg = '\n'.join(lines)
        self.fail(self._formatMessage(msg, standardMsg))

    def assertIn(self, member, container, msg=None):
        """Just like self.assertTrue(a in b), but with a nicer default message."""
        if member not in container:
            standardMsg = '%s not found in %s' % (safe_repr(member),
                                                  safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertNotIn(self, member, container, msg=None):
        """Just like self.assertTrue(a not in b), but with a nicer default message."""
        if member in container:
            standardMsg = '%s unexpectedly found in %s' % (safe_repr(member),
                                                        safe_repr(container))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIs(self, expr1, expr2, msg=None):
        """Just like self.assertTrue(a is b), but with a nicer default message."""
        if expr1 is not expr2:
            standardMsg = '%s is not %s' % (safe_repr(expr1),
                                             safe_repr(expr2))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNot(self, expr1, expr2, msg=None):
        """Just like self.assertTrue(a is not b), but with a nicer default message."""
        if expr1 is expr2:
            standardMsg = 'unexpectedly identical: %s' % (safe_repr(expr1),)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertDictEqual(self, d1, d2, msg=None):
        self.assertIsInstance(d1, dict, 'First argument is not a dictionary')
        self.assertIsInstance(d2, dict, 'Second argument is not a dictionary')

        if d1 != d2:
            standardMsg = '%s != %s' % _common_shorten_repr(d1, d2)
            diff = ('\n' + '\n'.join(difflib.ndiff(
                           pprint.pformat(d1).splitlines(),
                           pprint.pformat(d2).splitlines())))
            standardMsg = self._truncateMessage(standardMsg, diff)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertDictContainsSubset(self, subset, dictionary, msg=None):
        """Checks whether dictionary is a superset of subset."""
        warnings.warn('assertDictContainsSubset is deprecated',
                      DeprecationWarning,
                      stacklevel=2)
        missing = []
        mismatched = []
        for key, value in subset.items():
            if key not in dictionary:
                missing.append(key)
            elif value != dictionary[key]:
                mismatched.append('%s, expected: %s, actual: %s' %
                                  (safe_repr(key), safe_repr(value),
                                   safe_repr(dictionary[key])))

        if not (missing or mismatched):
            return

        standardMsg = ''
        if missing:
            standardMsg = 'Missing: %s' % ','.join(safe_repr(m) for m in
                                                    missing)
        if mismatched:
            if standardMsg:
                standardMsg += '; '
            standardMsg += 'Mismatched values: %s' % ','.join(mismatched)

        self.fail(self._formatMessage(msg, standardMsg))


    def assertCountEqual(self, first, second, msg=None):
        """Asserts that two iterables have the same elements, the same number of
        times, without regard to order.

            self.assertEqual(Counter(list(first)),
                             Counter(list(second)))

         Example:
            - [0, 1, 1] and [1, 0, 1] compare equal.
            - [0, 0, 1] and [0, 1] compare unequal.

        """
        first_seq, second_seq = list(first), list(second)
        try:
            first = collections.Counter(first_seq)
            second = collections.Counter(second_seq)
        except TypeError:
            # Handle case with unhashable elements
            differences = _count_diff_all_purpose(first_seq, second_seq)
        else:
            if first == second:
                return
            differences = _count_diff_hashable(first_seq, second_seq)

        if differences:
            standardMsg = 'Element counts were not equal:\n'
            lines = ['First has %d, Second has %d:  %r' % diff for diff in differences]
            diffMsg = '\n'.join(lines)
            standardMsg = self._truncateMessage(standardMsg, diffMsg)
            msg = self._formatMessage(msg, standardMsg)
            self.fail(msg)

    def assertMultiLineEqual(self, first, second, msg=None):
        """Assert that two multi-line strings are equal."""
        self.assertIsInstance(first, str, 'First argument is not a string')
        self.assertIsInstance(second, str, 'Second argument is not a string')

        if first != second:
            # don't use difflib if the strings are too long
            if (len(first) > self._diffThreshold or
                len(second) > self._diffThreshold):
                self._baseAssertEqual(first, second, msg)
            firstlines = first.splitlines(keepends=True)
            secondlines = second.splitlines(keepends=True)
            if len(firstlines) == 1 and first.strip('\r\n') == first:
                firstlines = [first + '\n']
                secondlines = [second + '\n']
            standardMsg = '%s != %s' % _common_shorten_repr(first, second)
            diff = '\n' + ''.join(difflib.ndiff(firstlines, secondlines))
            standardMsg = self._truncateMessage(standardMsg, diff)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertLess(self, a, b, msg=None):
        """Just like self.assertTrue(a < b), but with a nicer default message."""
        if not a < b:
            standardMsg = '%s not less than %s' % (safe_repr(a), safe_repr(b))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertLessEqual(self, a, b, msg=None):
        """Just like self.assertTrue(a <= b), but with a nicer default message."""
        if not a <= b:
            standardMsg = '%s not less than or equal to %s' % (safe_repr(a), safe_repr(b))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertGreater(self, a, b, msg=None):
        """Just like self.assertTrue(a > b), but with a nicer default message."""
        if not a > b:
            standardMsg = '%s not greater than %s' % (safe_repr(a), safe_repr(b))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertGreaterEqual(self, a, b, msg=None):
        """Just like self.assertTrue(a >= b), but with a nicer default message."""
        if not a >= b:
            standardMsg = '%s not greater than or equal to %s' % (safe_repr(a), safe_repr(b))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNone(self, obj, msg=None):
        """Same as self.assertTrue(obj is None), with a nicer default message."""
        if obj is not None:
            standardMsg = '%s is not None' % (safe_repr(obj),)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsNotNone(self, obj, msg=None):
        """Included for symmetry with assertIsNone."""
        if obj is None:
            standardMsg = 'unexpectedly None'
            self.fail(self._formatMessage(msg, standardMsg))

    def assertIsInstance(self, obj, cls, msg=None):
        """Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."""
        if not isinstance(obj, cls):
            standardMsg = '%s is not an instance of %r' % (safe_repr(obj), cls)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertNotIsInstance(self, obj, cls, msg=None):
        """Included for symmetry with assertIsInstance."""
        if isinstance(obj, cls):
            standardMsg = '%s is an instance of %r' % (safe_repr(obj), cls)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertRaisesRegex(self, expected_exception, expected_regex,
                          *args, **kwargs):
        """Asserts that the message in a raised exception matches a regex.

        Args:
            expected_exception: Exception class expected to be raised.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertRaisesRegex is used as a context manager.
        """
        context = _AssertRaisesContext(expected_exception, self, expected_regex)
        return context.handle('assertRaisesRegex', args, kwargs)

    def assertWarnsRegex(self, expected_warning, expected_regex,
                         *args, **kwargs):
        """Asserts that the message in a triggered warning matches a regexp.
        Basic functioning is similar to assertWarns() with the addition
        that only warnings whose messages also match the regular expression
        are considered successful matches.

        Args:
            expected_warning: Warning class expected to be triggered.
            expected_regex: Regex (re.Pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertWarnsRegex is used as a context manager.
        """
        context = _AssertWarnsContext(expected_warning, self, expected_regex)
        return context.handle('assertWarnsRegex', args, kwargs)

    def assertRegex(self, text, expected_regex, msg=None):
        """Fail the test unless the text matches the regular expression."""
        if isinstance(expected_regex, (str, bytes)):
            assert expected_regex, "expected_regex must not be empty."
            expected_regex = re.compile(expected_regex)
        if not expected_regex.search(text):
            standardMsg = "Regex didn't match: %r not found in %r" % (
                expected_regex.pattern, text)
            # _formatMessage ensures the longMessage option is respected
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)

    def assertNotRegex(self, text, unexpected_regex, msg=None):
        """Fail the test if the text matches the regular expression."""
        if isinstance(unexpected_regex, (str, bytes)):
            unexpected_regex = re.compile(unexpected_regex)
        match = unexpected_regex.search(text)
        if match:
            standardMsg = 'Regex matched: %r matches %r in %r' % (
                text[match.start() : match.end()],
                unexpected_regex.pattern,
                text)
            # _formatMessage ensures the longMessage option is respected
            msg = self._formatMessage(msg, standardMsg)
            raise self.failureException(msg)


    def _deprecate(original_func):
        def deprecated_func(*args, **kwargs):
            warnings.warn(
                'Please use {0} instead.'.format(original_func.__name__),
                DeprecationWarning, 2)
            return original_func(*args, **kwargs)
        return deprecated_func

    # see #9424
    failUnlessEqual = assertEquals = _deprecate(assertEqual)
    failIfEqual = assertNotEquals = _deprecate(assertNotEqual)
    failUnlessAlmostEqual = assertAlmostEquals = _deprecate(assertAlmostEqual)
    failIfAlmostEqual = assertNotAlmostEquals = _deprecate(assertNotAlmostEqual)
    failUnless = assert_ = _deprecate(assertTrue)
    failUnlessRaises = _deprecate(assertRaises)
    failIf = _deprecate(assertFalse)
    assertRaisesRegexp = _deprecate(assertRaisesRegex)
    assertRegexpMatches = _deprecate(assertRegex)
    assertNotRegexpMatches = _deprecate(assertNotRegex)



class FunctionTestCase(TestCase):
    """A test case that wraps a test function.

    This is useful for slipping pre-existing test functions into the
    unittest framework. Optionally, set-up and tidy-up functions can be
    supplied. As with TestCase, the tidy-up ('tearDown') function will
    always be called if the set-up ('setUp') function ran successfully.
    """

    def __init__(self, testFunc, setUp=None, tearDown=None, description=None):
        super(FunctionTestCase, self).__init__()
        self._setUpFunc = setUp
        self._tearDownFunc = tearDown
        self._testFunc = testFunc
        self._description = description

    def setUp(self):
        if self._setUpFunc is not None:
            self._setUpFunc()

    def tearDown(self):
        if self._tearDownFunc is not None:
            self._tearDownFunc()

    def runTest(self):
        self._testFunc()

    def id(self):
        return self._testFunc.__name__

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        return self._setUpFunc == other._setUpFunc and \
               self._tearDownFunc == other._tearDownFunc and \
               self._testFunc == other._testFunc and \
               self._description == other._description

    def __hash__(self):
        return hash((type(self), self._setUpFunc, self._tearDownFunc,
                     self._testFunc, self._description))

    def __str__(self):
        return "%s (%s)" % (strclass(self.__class__),
                            self._testFunc.__name__)

    def __repr__(self):
        return "<%s tec=%s>" % (strclass(self.__class__),
                                     self._testFunc)

    def shortDescription(self):
        if self._description is not None:
            return self._description
        doc = self._testFunc.__doc__
        return doc and doc.split("\n")[0].strip() or None


class _SubTest(TestCase):

    def __init__(self, test_case, message, params):
        super().__init__()
        self._message = message
        self.test_case = test_case
        self.params = params
        self.failureException = test_case.failureException

    def runTest(self):
        raise NotImplementedError("subtests cannot be run directly")

    def _subDescription(self):
        parts = []
        if self._message is not _subtest_msg_sentinel:
            parts.append("[{}]".format(self._message))
        if self.params:
            params_desc = ', '.join(
                "{}={!r}".format(k, v)
                for (k, v) in self.params.items())
            parts.append("({})".format(params_desc))
        return " ".join(parts) or '(<subtest>)'

    def id(self):
        return "{} {}".format(self.test_case.id(), self._subDescription())

    def shortDescription(self):
        """Returns a one-line description of the subtest, or None if no
        description has been provided.
        """
        return self.test_case.shortDescription()

    def __str__(self):
        return "{} {}".format(self.test_case, self._subDescription())
<|endoftext|>import logging
import collections

from .case import _BaseTestCaseContext


_LoggingWatcher = collections.namedtuple("_LoggingWatcher",
                                         ["records", "output"])

class _CapturingHandler(logging.Handler):
    """
    A logging handler capturing all (raw and formatted) logging output.
    """

    def __init__(self):
        logging.Handler.__init__(self)
        self.watcher = _LoggingWatcher([], [])

    def flush(self):
        pass

    def emit(self, record):
        self.watcher.records.append(record)
        msg = self.format(record)
        self.watcher.output.append(msg)


class _AssertLogsContext(_BaseTestCaseContext):
    """A context manager for assertLogs() and assertNoLogs() """

    LOGGING_FORMAT = "%(levelname)s:%(name)s:%(message)s"

    def __init__(self, test_case, logger_name, level, no_logs):
        _BaseTestCaseContext.__init__(self, test_case)
        self.logger_name = logger_name
        if level:
            self.level = logging._nameToLevel.get(level, level)
        else:
            self.level = logging.INFO
        self.msg = None
        self.no_logs = no_logs

    def __enter__(self):
        if isinstance(self.logger_name, logging.Logger):
            logger = self.logger = self.logger_name
        else:
            logger = self.logger = logging.getLogger(self.logger_name)
        formatter = logging.Formatter(self.LOGGING_FORMAT)
        handler = _CapturingHandler()
        handler.setLevel(self.level)
        handler.setFormatter(formatter)
        self.watcher = handler.watcher
        self.old_handlers = logger.handlers[:]
        self.old_level = logger.level
        self.old_propagate = logger.propagate
        logger.handlers = [handler]
        logger.setLevel(self.level)
        logger.propagate = False
        if self.no_logs:
            return
        return handler.watcher

    def __exit__(self, exc_type, exc_value, tb):
        self.logger.handlers = self.old_handlers
        self.logger.propagate = self.old_propagate
        self.logger.setLevel(self.old_level)

        if exc_type is not None:
            # let unexpected exceptions pass through
            return False

        if self.no_logs:
            # assertNoLogs
            if len(self.watcher.records) > 0:
                self._raiseFailure(
                    "Unexpected logs found: {!r}".format(
                        self.watcher.output
                    )
                )

        else:
            # assertLogs
            if len(self.watcher.records) == 0:
                self._raiseFailure(
                    "no logs of level {} or higher triggered on {}"
                    .format(logging.getLevelName(self.level), self.logger.name))
<|endoftext|>"""Running tests"""

import sys
import time
import warnings

from . import result
from .signals import registerResult

__unittest = True


class _WritelnDecorator(object):
    """Used to decorate file-like objects with a handy 'writeln' method"""
    def __init__(self,stream):
        self.stream = stream

    def __getattr__(self, attr):
        if attr in ('stream', '__getstate__'):
            raise AttributeError(attr)
        return getattr(self.stream,attr)

    def writeln(self, arg=None):
        if arg:
            self.write(arg)
        self.write('\n') # text-mode streams translate to \r\n if needed


class TextTestResult(result.TestResult):
    """A test result class that can print formatted text results to a stream.

    Used by TextTestRunner.
    """
    separator1 = '=' * 70
    separator2 = '-' * 70

    def __init__(self, stream, descriptions, verbosity):
        super(TextTestResult, self).__init__(stream, descriptions, verbosity)
        self.stream = stream
        self.showAll = verbosity > 1
        self.dots = verbosity == 1
        self.descriptions = descriptions

    def getDescription(self, test):
        doc_first_line = test.shortDescription()
        if self.descriptions and doc_first_line:
            return '\n'.join((str(test), doc_first_line))
        else:
            return str(test)

    def startTest(self, test):
        super(TextTestResult, self).startTest(test)
        if self.showAll:
            self.stream.write(self.getDescription(test))
            self.stream.write(" ... ")
            self.stream.flush()

    def addSuccess(self, test):
        super(TextTestResult, self).addSuccess(test)
        if self.showAll:
            self.stream.writeln("ok")
            self.stream.flush()
        elif self.dots:
            self.stream.write('.')
            self.stream.flush()

    def addError(self, test, err):
        super(TextTestResult, self).addError(test, err)
        if self.showAll:
            self.stream.writeln("ERROR")
            self.stream.flush()
        elif self.dots:
            self.stream.write('E')
            self.stream.flush()

    def addFailure(self, test, err):
        super(TextTestResult, self).addFailure(test, err)
        if self.showAll:
            self.stream.writeln("FAIL")
            self.stream.flush()
        elif self.dots:
            self.stream.write('F')
            self.stream.flush()

    def addSkip(self, test, reason):
        super(TextTestResult, self).addSkip(test, reason)
        if self.showAll:
            self.stream.writeln("skipped {0!r}".format(reason))
            self.stream.flush()
        elif self.dots:
            self.stream.write("s")
            self.stream.flush()

    def addExpectedFailure(self, test, err):
        super(TextTestResult, self).addExpectedFailure(test, err)
        if self.showAll:
            self.stream.writeln("expected failure")
            self.stream.flush()
        elif self.dots:
            self.stream.write("x")
            self.stream.flush()

    def addUnexpectedSuccess(self, test):
        super(TextTestResult, self).addUnexpectedSuccess(test)
        if self.showAll:
            self.stream.writeln("unexpected success")
            self.stream.flush()
        elif self.dots:
            self.stream.write("u")
            self.stream.flush()

    def printErrors(self):
        if self.dots or self.showAll:
            self.stream.writeln()
            self.stream.flush()
        self.printErrorList('ERROR', self.errors)
        self.printErrorList('FAIL', self.failures)

    def printErrorList(self, flavour, errors):
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln("%s: %s" % (flavour,self.getDescription(test)))
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)
            self.stream.flush()


class TextTestRunner(object):
    """A test runner class that displays results in textual form.

    It prints out the names of tests as they are run, errors as they
    occur, and a summary of the results at the end of the test run.
    """
    resultclass = TextTestResult

    def __init__(self, stream=None, descriptions=True, verbosity=1,
                 failfast=False, buffer=False, resultclass=None, warnings=None,
                 *, tb_locals=False):
        """Construct a TextTestRunner.

        Subclasses should accept **kwargs to ensure compatibility as the
        interface changes.
        """
        if stream is None:
            stream = sys.stderr
        self.stream = _WritelnDecorator(stream)
        self.descriptions = descriptions
        self.verbosity = verbosity
        self.failfast = failfast
        self.buffer = buffer
        self.tb_locals = tb_locals
        self.warnings = warnings
        if resultclass is not None:
            self.resultclass = resultclass

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        "Run the given test case or test suite."
        result = self._makeResult()
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals
        with warnings.catch_warnings():
            if self.warnings:
                # if self.warnings is set, use it to filter all the warnings
                warnings.simplefilter(self.warnings)
                # if the filter is 'default' or 'always', special-case the
                # warnings from the deprecated unittest methods to show them
                # no more than once per module, because they can be fairly
                # noisy.  The -Wd and -Wa flags can be used to bypass this
                # only when self.warnings is None.
                if self.warnings in ['default', 'always']:
                    warnings.filterwarnings('module',
                            category=DeprecationWarning,
                            message=r'Please use assert\w+ instead.')
            startTime = time.perf_counter()
            startTestRun = getattr(result, 'startTestRun', None)
            if startTestRun is not None:
                startTestRun()
            try:
                test(result)
            finally:
                stopTestRun = getattr(result, 'stopTestRun', None)
                if stopTestRun is not None:
                    stopTestRun()
            stopTime = time.perf_counter()
        timeTaken = stopTime - startTime
        result.printErrors()
        if hasattr(result, 'separator2'):
            self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln("Ran %d test%s in %.3fs" %
                            (run, run != 1 and "s" or "", timeTaken))
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(len, (result.expectedFailures,
                                result.unexpectedSuccesses,
                                result.skipped))
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped = results

        infos = []
        if not result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = len(result.failures), len(result.errors)
            if failed:
                infos.append("failures=%d" % failed)
            if errored:
                infos.append("errors=%d" % errored)
        else:
            self.stream.write("OK")
        if skipped:
            infos.append("skipped=%d" % skipped)
        if expectedFails:
            infos.append("expected failures=%d" % expectedFails)
        if unexpectedSuccesses:
            infos.append("unexpected successes=%d" % unexpectedSuccesses)
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        self.stream.flush()
        return result
<|endoftext|>"""TestSuite"""

import sys

from . import case
from . import util

__unittest = True


def _call_if_exists(parent, attr):
    func = getattr(parent, attr, lambda: None)
    func()


class BaseTestSuite(object):
    """A simple test suite that doesn't provide class or module shared fixtures.
    """
    _cleanup = True

    def __init__(self, tests=()):
        self._tests = []
        self._removed_tests = 0
        self.addTests(tests)

    def __repr__(self):
        return "<%s tests=%s>" % (util.strclass(self.__class__), list(self))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return list(self) == list(other)

    def __iter__(self):
        return iter(self._tests)

    def countTestCases(self):
        cases = self._removed_tests
        for test in self:
            if test:
                cases += test.countTestCases()
        return cases

    def addTest(self, test):
        # sanity checks
        if not callable(test):
            raise TypeError("{} is not callable".format(repr(test)))
        if isinstance(test, type) and issubclass(test,
                                                 (case.TestCase, TestSuite)):
            raise TypeError("TestCases and TestSuites must be instantiated "
                            "before passing them to addTest()")
        self._tests.append(test)

    def addTests(self, tests):
        if isinstance(tests, str):
            raise TypeError("tests must be an iterable of tests, not a string")
        for test in tests:
            self.addTest(test)

    def run(self, result):
        for index, test in enumerate(self):
            if result.shouldStop:
                break
            test(result)
            if self._cleanup:
                self._removeTestAtIndex(index)
        return result

    def _removeTestAtIndex(self, index):
        """Stop holding a reference to the TestCase at index."""
        try:
            test = self._tests[index]
        except TypeError:
            # support for suite implementations that have overridden self._tests
            pass
        else:
            # Some unittest tests add non TestCase/TestSuite objects to
            # the suite.
            if hasattr(test, 'countTestCases'):
                self._removed_tests += test.countTestCases()
            self._tests[index] = None

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def debug(self):
        """Run the tests without collecting errors in a TestResult"""
        for test in self:
            test.debug()


class TestSuite(BaseTestSuite):
    """A test suite is a composite test consisting of a number of TestCases.

    For use, create an instance of TestSuite, then add test case instances.
    When all tests have been added, the suite can be passed to a test
    runner, such as TextTestRunner. It will run the individual test cases
    in the order in which they were added, aggregating the results. When
    subclassing, do not forget to call the base class constructor.
    """

    def run(self, result, debug=False):
        topLevel = False
        if getattr(result, '_testRunEntered', False) is False:
            result._testRunEntered = topLevel = True

        for index, test in enumerate(self):
            if result.shouldStop:
                break

            if _isnotsuite(test):
                self._tearDownPreviousClass(test, result)
                self._handleModuleFixture(test, result)
                self._handleClassSetUp(test, result)
                result._previousTestClass = test.__class__

                if (getattr(test.__class__, '_classSetupFailed', False) or
                    getattr(result, '_moduleSetUpFailed', False)):
                    continue

            if not debug:
                test(result)
            else:
                test.debug()

            if self._cleanup:
                self._removeTestAtIndex(index)

        if topLevel:
            self._tearDownPreviousClass(None, result)
            self._handleModuleTearDown(result)
            result._testRunEntered = False
        return result

    def debug(self):
        """Run the tests without collecting errors in a TestResult"""
        debug = _DebugResult()
        self.run(debug, True)

    ################################

    def _handleClassSetUp(self, test, result):
        previousClass = getattr(result, '_previousTestClass', None)
        currentClass = test.__class__
        if currentClass == previousClass:
            return
        if result._moduleSetUpFailed:
            return
        if getattr(currentClass, "__unittest_skip__", False):
            return

        failed = False
        try:
            currentClass._classSetupFailed = False
        except TypeError:
            # test may actually be a function
            # so its class will be a builtin-type
            pass

        setUpClass = getattr(currentClass, 'setUpClass', None)
        doClassCleanups = getattr(currentClass, 'doClassCleanups', None)
        if setUpClass is not None:
            _call_if_exists(result, '_setupStdout')
            try:
                try:
                    setUpClass()
                except Exception as e:
                    if isinstance(result, _DebugResult):
                        raise
                    failed = True
                    try:
                        currentClass._classSetupFailed = True
                    except TypeError:
                        pass
                    className = util.strclass(currentClass)
                    self._createClassOrModuleLevelException(result, e,
                                                            'setUpClass',
                                                            className)
                if failed and doClassCleanups is not None:
                    doClassCleanups()
                    for exc_info in currentClass.tearDown_exceptions:
                        self._createClassOrModuleLevelException(
                                result, exc_info[1], 'setUpClass', className,
                                info=exc_info)
            finally:
                _call_if_exists(result, '_restoreStdout')

    def _get_previous_module(self, result):
        previousModule = None
        previousClass = getattr(result, '_previousTestClass', None)
        if previousClass is not None:
            previousModule = previousClass.__module__
        return previousModule


    def _handleModuleFixture(self, test, result):
        previousModule = self._get_previous_module(result)
        currentModule = test.__class__.__module__
        if currentModule == previousModule:
            return

        self._handleModuleTearDown(result)


        result._moduleSetUpFailed = False
        try:
            module = sys.modules[currentModule]
        except KeyError:
            return
        setUpModule = getattr(module, 'setUpModule', None)
        if setUpModule is not None:
            _call_if_exists(result, '_setupStdout')
            try:
                try:
                    setUpModule()
                except Exception as e:
                    if isinstance(result, _DebugResult):
                        raise
                    result._moduleSetUpFailed = True
                    self._createClassOrModuleLevelException(result, e,
                                                            'setUpModule',
                                                            currentModule)
                if result._moduleSetUpFailed:
                    try:
                        case.doModuleCleanups()
                    except Exception as e:
                        self._createClassOrModuleLevelException(result, e,
                                                                'setUpModule',
                                                                currentModule)
            finally:
                _call_if_exists(result, '_restoreStdout')

    def _createClassOrModuleLevelException(self, result, exc, method_name,
                                           parent, info=None):
        errorName = f'{method_name} ({parent})'
        self._addClassOrModuleLevelException(result, exc, errorName, info)

    def _addClassOrModuleLevelException(self, result, exception, errorName,
                                        info=None):
        error = _ErrorHolder(errorName)
        addSkip = getattr(result, 'addSkip', None)
        if addSkip is not None and isinstance(exception, case.SkipTest):
            addSkip(error, str(exception))
        else:
            if not info:
                result.addError(error, sys.exc_info())
            else:
                result.addError(error, info)

    def _handleModuleTearDown(self, result):
        previousModule = self._get_previous_module(result)
        if previousModule is None:
            return
        if result._moduleSetUpFailed:
            return

        try:
            module = sys.modules[previousModule]
        except KeyError:
            return

        _call_if_exists(result, '_setupStdout')
        try:
            tearDownModule = getattr(module, 'tearDownModule', None)
            if tearDownModule is not None:
                try:
                    tearDownModule()
                except Exception as e:
                    if isinstance(result, _DebugResult):
                        raise
                    self._createClassOrModuleLevelException(result, e,
                                                            'tearDownModule',
                                                            previousModule)
            try:
                case.doModuleCleanups()
            except Exception as e:
                if isinstance(result, _DebugResult):
                    raise
                self._createClassOrModuleLevelException(result, e,
                                                        'tearDownModule',
                                                        previousModule)
        finally:
            _call_if_exists(result, '_restoreStdout')

    def _tearDownPreviousClass(self, test, result):
        previousClass = getattr(result, '_previousTestClass', None)
        currentClass = test.__class__
        if currentClass == previousClass or previousClass is None:
            return
        if getattr(previousClass, '_classSetupFailed', False):
            return
        if getattr(result, '_moduleSetUpFailed', False):
            return
        if getattr(previousClass, "__unittest_skip__", False):
            return

        tearDownClass = getattr(previousClass, 'tearDownClass', None)
        doClassCleanups = getattr(previousClass, 'doClassCleanups', None)
        if tearDownClass is None and doClassCleanups is None:
            return

        _call_if_exists(result, '_setupStdout')
        try:
            if tearDownClass is not None:
                try:
                    tearDownClass()
                except Exception as e:
                    if isinstance(result, _DebugResult):
                        raise
                    className = util.strclass(previousClass)
                    self._createClassOrModuleLevelException(result, e,
                                                            'tearDownClass',
                                                            className)
            if doClassCleanups is not None:
                doClassCleanups()
                for exc_info in previousClass.tearDown_exceptions:
                    if isinstance(result, _DebugResult):
                        raise exc_info[1]
                    className = util.strclass(previousClass)
                    self._createClassOrModuleLevelException(result, exc_info[1],
                                                            'tearDownClass',
                                                            className,
                                                            info=exc_info)
        finally:
            _call_if_exists(result, '_restoreStdout')


class _ErrorHolder(object):
    """
    Placeholder for a TestCase inside a result. As far as a TestResult
    is concerned, this looks exactly like a unit test. Used to insert
    arbitrary errors into a test suite run.
    """
    # Inspired by the ErrorHolder from Twisted:
    # http://twistedmatrix.com/trac/browser/trunk/twisted/trial/runner.py

    # attribute used by TestResult._exc_info_to_string
    failureException = None

    def __init__(self, description):
        self.description = description

    def id(self):
        return self.description

    def shortDescription(self):
        return None

    def __repr__(self):
        return "<ErrorHolder description=%r>" % (self.description,)

    def __str__(self):
        return self.id()

    def run(self, result):
        # could call result.addError(...) - but this test-like object
        # shouldn't be run anyway
        pass

    def __call__(self, result):
        return self.run(result)

    def countTestCases(self):
        return 0

def _isnotsuite(test):
    "A crude way to tell apart testcases and suites with duck-typing"
    try:
        iter(test)
    except TypeError:
        return True
    return False


class _DebugResult(object):
    "Used by the TestSuite to hold previous class when running in debug."
    _previousTestClass = None
    _moduleSetUpFailed = False
    shouldStop = False
<|endoftext|>"""
Python unit testing framework, based on Erich Gamma's JUnit and Kent Beck's
Smalltalk testing framework (used with permission).

This module contains the core framework classes that form the basis of
specific test cases and suites (TestCase, TestSuite etc.), and also a
text-based utility class for running the tests and reporting the results
 (TextTestRunner).

Simple usage:

    import unittest

    class IntegerArithmeticTestCase(unittest.TestCase):
        def testAdd(self):  # test method names begin with 'test'
            self.assertEqual((1 + 2), 3)
            self.assertEqual(0 + 1, 1)
        def testMultiply(self):
            self.assertEqual((0 * 10), 0)
            self.assertEqual((5 * 8), 40)

    if __name__ == '__main__':
        unittest.main()

Further information is available in the bundled documentation, and from

  http://docs.python.org/library/unittest.html

Copyright (c) 1999-2003 Steve Purcell
Copyright (c) 2003-2010 Python Software Foundation
This module is free software, and you may redistribute it and/or modify
it under the same terms as Python itself, so long as this copyright message
and disclaimer are retained in their original form.

IN NO EVENT SHALL THE AUTHOR BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF
THIS CODE, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

THE AUTHOR SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE.  THE CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THERE IS NO OBLIGATION WHATSOEVER TO PROVIDE MAINTENANCE,
SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

__all__ = ['TestResult', 'TestCase', 'IsolatedAsyncioTestCase', 'TestSuite',
           'TextTestRunner', 'TestLoader', 'FunctionTestCase', 'main',
           'defaultTestLoader', 'SkipTest', 'skip', 'skipIf', 'skipUnless',
           'expectedFailure', 'TextTestResult', 'installHandler',
           'registerResult', 'removeResult', 'removeHandler',
           'addModuleCleanup']

# Expose obsolete functions for backwards compatibility
__all__.extend(['getTestCaseNames', 'makeSuite', 'findTestCases'])

__unittest = True

from .result import TestResult
from .case import (addModuleCleanup, TestCase, FunctionTestCase, SkipTest, skip,
                   skipIf, skipUnless, expectedFailure)
from .suite import BaseTestSuite, TestSuite
from .loader import (TestLoader, defaultTestLoader, makeSuite, getTestCaseNames,
                     findTestCases)
from .main import TestProgram, main
from .runner import TextTestRunner, TextTestResult
from .signals import installHandler, registerResult, removeResult, removeHandler
# IsolatedAsyncioTestCase will be imported lazily.

# deprecated
_TextTestResult = TextTestResult

# There are no tests here, so don't try to run anything discovered from
# introspecting the symbols (e.g. FunctionTestCase). Instead, all our
# tests come from within unittest.test.
def load_tests(loader, tests, pattern):
    import os.path
    # top level directory cached on loader instance
    this_dir = os.path.dirname(__file__)
    return loader.discover(start_dir=this_dir, pattern=pattern)


# Lazy import of IsolatedAsyncioTestCase from .async_case
# It imports asyncio, which is relatively heavy, but most tests
# do not need it.

def __dir__():
    return globals().keys() | {'IsolatedAsyncioTestCase'}

def __getattr__(name):
    if name == 'IsolatedAsyncioTestCase':
        global IsolatedAsyncioTestCase
        from .async_case import IsolatedAsyncioTestCase
        return IsolatedAsyncioTestCase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
<|endoftext|># mock.py
# Test tools for mocking and patching.
# Maintained by Michael Foord
# Backport for other versions of Python available from
# https://pypi.org/project/mock

__all__ = (
    'Mock',
    'MagicMock',
    'patch',
    'sentinel',
    'DEFAULT',
    'ANY',
    'call',
    'create_autospec',
    'AsyncMock',
    'FILTER_DIR',
    'NonCallableMock',
    'NonCallableMagicMock',
    'mock_open',
    'PropertyMock',
    'seal',
)


import asyncio
import contextlib
import io
import inspect
import pprint
import sys
import builtins
from asyncio import iscoroutinefunction
from types import CodeType, ModuleType, MethodType
from unittest.util import safe_repr
from functools import wraps, partial
from threading import RLock


class InvalidSpecError(Exception):
    """Indicates that an invalid value was used as a mock spec."""


_builtins = {name for name in dir(builtins) if not name.startswith('_')}

FILTER_DIR = True

# Workaround for issue #12370
# Without this, the __class__ properties wouldn't be set correctly
_safe_super = super

def _is_async_obj(obj):
    if _is_instance_mock(obj) and not isinstance(obj, AsyncMock):
        return False
    if hasattr(obj, '__func__'):
        obj = getattr(obj, '__func__')
    return iscoroutinefunction(obj) or inspect.isawaitable(obj)


def _is_async_func(func):
    if getattr(func, '__code__', None):
        return iscoroutinefunction(func)
    else:
        return False


def _is_instance_mock(obj):
    # can't use isinstance on Mock objects because they override __class__
    # The base class for all mocks is NonCallableMock
    return issubclass(type(obj), NonCallableMock)


def _is_exception(obj):
    return (
        isinstance(obj, BaseException) or
        isinstance(obj, type) and issubclass(obj, BaseException)
    )


def _extract_mock(obj):
    # Autospecced functions will return a FunctionType with "mock" attribute
    # which is the actual mock object that needs to be used.
    if isinstance(obj, FunctionTypes) and hasattr(obj, 'mock'):
        return obj.mock
    else:
        return obj


def _get_signature_object(func, as_instance, eat_self):
    """
    Given an arbitrary, possibly callable object, try to create a suitable
    signature object.
    Return a (reduced func, signature) tuple, or None.
    """
    if isinstance(func, type) and not as_instance:
        # If it's a type and should be modelled as a type, use __init__.
        func = func.__init__
        # Skip the `self` argument in __init__
        eat_self = True
    elif not isinstance(func, FunctionTypes):
        # If we really want to model an instance of the passed type,
        # __call__ should be looked up, not __init__.
        try:
            func = func.__call__
        except AttributeError:
            return None
    if eat_self:
        sig_func = partial(func, None)
    else:
        sig_func = func
    try:
        return func, inspect.signature(sig_func)
    except ValueError:
        # Certain callable types are not supported by inspect.signature()
        return None


def _check_signature(func, mock, skipfirst, instance=False):
    sig = _get_signature_object(func, instance, skipfirst)
    if sig is None:
        return
    func, sig = sig
    def checksig(self, /, *args, **kwargs):
        sig.bind(*args, **kwargs)
    _copy_func_details(func, checksig)
    type(mock)._mock_check_sig = checksig
    type(mock).__signature__ = sig


def _copy_func_details(func, funcopy):
    # we explicitly don't copy func.__dict__ into this copy as it would
    # expose original attributes that should be mocked
    for attribute in (
        '__name__', '__doc__', '__text_signature__',
        '__module__', '__defaults__', '__kwdefaults__',
    ):
        try:
            setattr(funcopy, attribute, getattr(func, attribute))
        except AttributeError:
            pass


def _callable(obj):
    if isinstance(obj, type):
        return True
    if isinstance(obj, (staticmethod, classmethod, MethodType)):
        return _callable(obj.__func__)
    if getattr(obj, '__call__', None) is not None:
        return True
    return False


def _is_list(obj):
    # checks for list or tuples
    # XXXX badly named!
    return type(obj) in (list, tuple)


def _instance_callable(obj):
    """Given an object, return True if the object is callable.
    For classes, return True if instances would be callable."""
    if not isinstance(obj, type):
        # already an instance
        return getattr(obj, '__call__', None) is not None

    # *could* be broken by a class overriding __mro__ or __dict__ via
    # a metaclass
    for base in (obj,) + obj.__mro__:
        if base.__dict__.get('__call__') is not None:
            return True
    return False


def _set_signature(mock, original, instance=False):
    # creates a function with signature (*args, **kwargs) that delegates to a
    # mock. It still does signature checking by calling a lambda with the same
    # signature as the original.

    skipfirst = isinstance(original, type)
    result = _get_signature_object(original, instance, skipfirst)
    if result is None:
        return mock
    func, sig = result
    def checksig(*args, **kwargs):
        sig.bind(*args, **kwargs)
    _copy_func_details(func, checksig)

    name = original.__name__
    if not name.isidentifier():
        name = 'funcopy'
    context = {'_checksig_': checksig, 'mock': mock}
    src = """def %s(*args, **kwargs):
    _checksig_(*args, **kwargs)
    return mock(*args, **kwargs)""" % name
    exec (src, context)
    funcopy = context[name]
    _setup_func(funcopy, mock, sig)
    return funcopy


def _setup_func(funcopy, mock, sig):
    funcopy.mock = mock

    def assert_called_with(*args, **kwargs):
        return mock.assert_called_with(*args, **kwargs)
    def assert_called(*args, **kwargs):
        return mock.assert_called(*args, **kwargs)
    def assert_not_called(*args, **kwargs):
        return mock.assert_not_called(*args, **kwargs)
    def assert_called_once(*args, **kwargs):
        return mock.assert_called_once(*args, **kwargs)
    def assert_called_once_with(*args, **kwargs):
        return mock.assert_called_once_with(*args, **kwargs)
    def assert_has_calls(*args, **kwargs):
        return mock.assert_has_calls(*args, **kwargs)
    def assert_any_call(*args, **kwargs):
        return mock.assert_any_call(*args, **kwargs)
    def reset_mock():
        funcopy.method_calls = _CallList()
        funcopy.mock_calls = _CallList()
        mock.reset_mock()
        ret = funcopy.return_value
        if _is_instance_mock(ret) and not ret is mock:
            ret.reset_mock()

    funcopy.called = False
    funcopy.call_count = 0
    funcopy.call_args = None
    funcopy.call_args_list = _CallList()
    funcopy.method_calls = _CallList()
    funcopy.mock_calls = _CallList()

    funcopy.return_value = mock.return_value
    funcopy.side_effect = mock.side_effect
    funcopy._mock_children = mock._mock_children

    funcopy.assert_called_with = assert_called_with
    funcopy.assert_called_once_with = assert_called_once_with
    funcopy.assert_has_calls = assert_has_calls
    funcopy.assert_any_call = assert_any_call
    funcopy.reset_mock = reset_mock
    funcopy.assert_called = assert_called
    funcopy.assert_not_called = assert_not_called
    funcopy.assert_called_once = assert_called_once
    funcopy.__signature__ = sig

    mock._mock_delegate = funcopy


def _setup_async_mock(mock):
    mock._is_coroutine = asyncio.coroutines._is_coroutine
    mock.await_count = 0
    mock.await_args = None
    mock.await_args_list = _CallList()

    # Mock is not configured yet so the attributes are set
    # to a function and then the corresponding mock helper function
    # is called when the helper is accessed similar to _setup_func.
    def wrapper(attr, /, *args, **kwargs):
        return getattr(mock.mock, attr)(*args, **kwargs)

    for attribute in ('assert_awaited',
                      'assert_awaited_once',
                      'assert_awaited_with',
                      'assert_awaited_once_with',
                      'assert_any_await',
                      'assert_has_awaits',
                      'assert_not_awaited'):

        # setattr(mock, attribute, wrapper) causes late binding
        # hence attribute will always be the last value in the loop
        # Use partial(wrapper, attribute) to ensure the attribute is bound
        # correctly.
        setattr(mock, attribute, partial(wrapper, attribute))


def _is_magic(name):
    return '__%s__' % name[2:-2] == name


class _SentinelObject(object):
    "A unique, named, sentinel object."
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'sentinel.%s' % self.name

    def __reduce__(self):
        return 'sentinel.%s' % self.name


class _Sentinel(object):
    """Access attributes to return a named object, usable as a sentinel."""
    def __init__(self):
        self._sentinels = {}

    def __getattr__(self, name):
        if name == '__bases__':
            # Without this help(unittest.mock) raises an exception
            raise AttributeError
        return self._sentinels.setdefault(name, _SentinelObject(name))

    def __reduce__(self):
        return 'sentinel'


sentinel = _Sentinel()

DEFAULT = sentinel.DEFAULT
_missing = sentinel.MISSING
_deleted = sentinel.DELETED


_allowed_names = {
    'return_value', '_mock_return_value', 'side_effect',
    '_mock_side_effect', '_mock_parent', '_mock_new_parent',
    '_mock_name', '_mock_new_name'
}


def _delegating_property(name):
    _allowed_names.add(name)
    _the_name = '_mock_' + name
    def _get(self, name=name, _the_name=_the_name):
        sig = self._mock_delegate
        if sig is None:
            return getattr(self, _the_name)
        return getattr(sig, name)
    def _set(self, value, name=name, _the_name=_the_name):
        sig = self._mock_delegate
        if sig is None:
            self.__dict__[_the_name] = value
        else:
            setattr(sig, name, value)

    return property(_get, _set)



class _CallList(list):

    def __contains__(self, value):
        if not isinstance(value, list):
            return list.__contains__(self, value)
        len_value = len(value)
        len_self = len(self)
        if len_value > len_self:
            return False

        for i in range(0, len_self - len_value + 1):
            sub_list = self[i:i+len_value]
            if sub_list == value:
                return True
        return False

    def __repr__(self):
        return pprint.pformat(list(self))


def _check_and_set_parent(parent, value, name, new_name):
    value = _extract_mock(value)

    if not _is_instance_mock(value):
        return False
    if ((value._mock_name or value._mock_new_name) or
        (value._mock_parent is not None) or
        (value._mock_new_parent is not None)):
        return False

    _parent = parent
    while _parent is not None:
        # setting a mock (value) as a child or return value of itself
        # should not modify the mock
        if _parent is value:
            return False
        _parent = _parent._mock_new_parent

    if new_name:
        value._mock_new_parent = parent
        value._mock_new_name = new_name
    if name:
        value._mock_parent = parent
        value._mock_name = name
    return True

# Internal class to identify if we wrapped an iterator object or not.
class _MockIter(object):
    def __init__(self, obj):
        self.obj = iter(obj)
    def __next__(self):
        return next(self.obj)

class Base(object):
    _mock_return_value = DEFAULT
    _mock_side_effect = None
    def __init__(self, /, *args, **kwargs):
        pass



class NonCallableMock(Base):
    """A non-callable version of `Mock`"""

    # Store a mutex as a class attribute in order to protect concurrent access
    # to mock attributes. Using a class attribute allows all NonCallableMock
    # instances to share the mutex for simplicity.
    #
    # See https://github.com/python/cpython/issues/98624 for why this is
    # necessary.
    _lock = RLock()

    def __new__(cls, /, *args, **kw):
        # every instance has its own class
        # so we can create magic methods on the
        # class without stomping on other mocks
        bases = (cls,)
        if not issubclass(cls, AsyncMockMixin):
            # Check if spec is an async object or function
            bound_args = _MOCK_SIG.bind_partial(cls, *args, **kw).arguments
            spec_arg = bound_args.get('spec_set', bound_args.get('spec'))
            if spec_arg is not None and _is_async_obj(spec_arg):
                bases = (AsyncMockMixin, cls)
        new = type(cls.__name__, bases, {'__doc__': cls.__doc__})
        instance = _safe_super(NonCallableMock, cls).__new__(new)
        return instance


    def __init__(
            self, spec=None, wraps=None, name=None, spec_set=None,
            parent=None, _spec_state=None, _new_name='', _new_parent=None,
            _spec_as_instance=False, _eat_self=None, unsafe=False, **kwargs
        ):
        if _new_parent is None:
            _new_parent = parent

        __dict__ = self.__dict__
        __dict__['_mock_parent'] = parent
        __dict__['_mock_name'] = name
        __dict__['_mock_new_name'] = _new_name
        __dict__['_mock_new_parent'] = _new_parent
        __dict__['_mock_sealed'] = False

        if spec_set is not None:
            spec = spec_set
            spec_set = True
        if _eat_self is None:
            _eat_self = parent is not None

        self._mock_add_spec(spec, spec_set, _spec_as_instance, _eat_self)

        __dict__['_mock_children'] = {}
        __dict__['_mock_wraps'] = wraps
        __dict__['_mock_delegate'] = None

        __dict__['_mock_called'] = False
        __dict__['_mock_call_args'] = None
        __dict__['_mock_call_count'] = 0
        __dict__['_mock_call_args_list'] = _CallList()
        __dict__['_mock_mock_calls'] = _CallList()

        __dict__['method_calls'] = _CallList()
        __dict__['_mock_unsafe'] = unsafe

        if kwargs:
            self.configure_mock(**kwargs)

        _safe_super(NonCallableMock, self).__init__(
            spec, wraps, name, spec_set, parent,
            _spec_state
        )


    def attach_mock(self, mock, attribute):
        """
        Attach a mock as an attribute of this one, replacing its name and
        parent. Calls to the attached mock will be recorded in the
        `method_calls` and `mock_calls` attributes of this one."""
        inner_mock = _extract_mock(mock)

        inner_mock._mock_parent = None
        inner_mock._mock_new_parent = None
        inner_mock._mock_name = ''
        inner_mock._mock_new_name = None

        setattr(self, attribute, mock)


    def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)


    def _mock_add_spec(self, spec, spec_set, _spec_as_instance=False,
                       _eat_self=False):
        _spec_class = None
        _spec_signature = None
        _spec_asyncs = []

        for attr in dir(spec):
            if iscoroutinefunction(getattr(spec, attr, None)):
                _spec_asyncs.append(attr)

        if spec is not None and not _is_list(spec):
            if isinstance(spec, type):
                _spec_class = spec
            else:
                _spec_class = type(spec)
            res = _get_signature_object(spec,
                                        _spec_as_instance, _eat_self)
            _spec_signature = res and res[1]

            spec = dir(spec)

        __dict__ = self.__dict__
        __dict__['_spec_class'] = _spec_class
        __dict__['_spec_set'] = spec_set
        __dict__['_spec_signature'] = _spec_signature
        __dict__['_mock_methods'] = spec
        __dict__['_spec_asyncs'] = _spec_asyncs

    def __get_return_value(self):
        ret = self._mock_return_value
        if self._mock_delegate is not None:
            ret = self._mock_delegate.return_value

        if ret is DEFAULT:
            ret = self._get_child_mock(
                _new_parent=self, _new_name='()'
            )
            self.return_value = ret
        return ret


    def __set_return_value(self, value):
        if self._mock_delegate is not None:
            self._mock_delegate.return_value = value
        else:
            self._mock_return_value = value
            _check_and_set_parent(self, value, None, '()')

    __return_value_doc = "The value to be returned when the mock is called."
    return_value = property(__get_return_value, __set_return_value,
                            __return_value_doc)


    @property
    def __class__(self):
        if self._spec_class is None:
            return type(self)
        return self._spec_class

    called = _delegating_property('called')
    call_count = _delegating_property('call_count')
    call_args = _delegating_property('call_args')
    call_args_list = _delegating_property('call_args_list')
    mock_calls = _delegating_property('mock_calls')


    def __get_side_effect(self):
        delegated = self._mock_delegate
        if delegated is None:
            return self._mock_side_effect
        sf = delegated.side_effect
        if (sf is not None and not callable(sf)
                and not isinstance(sf, _MockIter) and not _is_exception(sf)):
            sf = _MockIter(sf)
            delegated.side_effect = sf
        return sf

    def __set_side_effect(self, value):
        value = _try_iter(value)
        delegated = self._mock_delegate
        if delegated is None:
            self._mock_side_effect = value
        else:
            delegated.side_effect = value

    side_effect = property(__get_side_effect, __set_side_effect)


    def reset_mock(self,  visited=None,*, return_value=False, side_effect=False):
        "Restore the mock object to its initial state."
        if visited is None:
            visited = []
        if id(self) in visited:
            return
        visited.append(id(self))

        self.called = False
        self.call_args = None
        self.call_count = 0
        self.mock_calls = _CallList()
        self.call_args_list = _CallList()
        self.method_calls = _CallList()

        if return_value:
            self._mock_return_value = DEFAULT
        if side_effect:
            self._mock_side_effect = None

        for child in self._mock_children.values():
            if isinstance(child, _SpecState) or child is _deleted:
                continue
            child.reset_mock(visited, return_value=return_value, side_effect=side_effect)

        ret = self._mock_return_value
        if _is_instance_mock(ret) and ret is not self:
            ret.reset_mock(visited)


    def configure_mock(self, /, **kwargs):
        """Set attributes on the mock through keyword arguments.

        Attributes plus return values and side effects can be set on child
        mocks using standard dot notation and unpacking a dictionary in the
        method call:

        >>> attrs = {'method.return_value': 3, 'other.side_effect': KeyError}
        >>> mock.configure_mock(**attrs)"""
        for arg, val in sorted(kwargs.items(),
                               # we sort on the number of dots so that
                               # attributes are set before we set attributes on
                               # attributes
                               key=lambda entry: entry[0].count('.')):
            args = arg.split('.')
            final = args.pop()
            obj = self
            for entry in args:
                obj = getattr(obj, entry)
            setattr(obj, final, val)


    def __getattr__(self, name):
        if name in {'_mock_methods', '_mock_unsafe'}:
            raise AttributeError(name)
        elif self._mock_methods is not None:
            if name not in self._mock_methods or name in _all_magics:
                raise AttributeError("Mock object has no attribute %r" % name)
        elif _is_magic(name):
            raise AttributeError(name)
        if not self._mock_unsafe and (not self._mock_methods or name not in self._mock_methods):
            if name.startswith(('assert', 'assret', 'asert', 'aseert', 'assrt')):
                raise AttributeError(
                    f"{name!r} is not a valid assertion. Use a spec "
                    f"for the mock if {name!r} is meant to be an attribute.")

        with NonCallableMock._lock:
            result = self._mock_children.get(name)
            if result is _deleted:
                raise AttributeError(name)
            elif result is None:
                wraps = None
                if self._mock_wraps is not None:
                    # XXXX should we get the attribute without triggering code
                    # execution?
                    wraps = getattr(self._mock_wraps, name)

                result = self._get_child_mock(
                    parent=self, name=name, wraps=wraps, _new_name=name,
                    _new_parent=self
                )
                self._mock_children[name]  = result

            elif isinstance(result, _SpecState):
                try:
                    result = create_autospec(
                        result.spec, result.spec_set, result.instance,
                        result.parent, result.name
                    )
                except InvalidSpecError:
                    target_name = self.__dict__['_mock_name'] or self
                    raise InvalidSpecError(
                        f'Cannot autospec attr {name!r} from target '
                        f'{target_name!r} as it has already been mocked out. '
                        f'[target={self!r}, attr={result.spec!r}]')
                self._mock_children[name]  = result

        return result


    def _extract_mock_name(self):
        _name_list = [self._mock_new_name]
        _parent = self._mock_new_parent
        last = self

        dot = '.'
        if _name_list == ['()']:
            dot = ''

        while _parent is not None:
            last = _parent

            _name_list.append(_parent._mock_new_name + dot)
            dot = '.'
            if _parent._mock_new_name == '()':
                dot = ''

            _parent = _parent._mock_new_parent

        _name_list = list(reversed(_name_list))
        _first = last._mock_name or 'mock'
        if len(_name_list) > 1:
            if _name_list[1] not in ('()', '().'):
                _first += '.'
        _name_list[0] = _first
        return ''.join(_name_list)

    def __repr__(self):
        name = self._extract_mock_name()

        name_string = ''
        if name not in ('mock', 'mock.'):
            name_string = ' name=%r' % name

        spec_string = ''
        if self._spec_class is not None:
            spec_string = ' spec=%r'
            if self._spec_set:
                spec_string = ' spec_set=%r'
            spec_string = spec_string % self._spec_class.__name__
        return "<%s%s%s id='%s'>" % (
            type(self).__name__,
            name_string,
            spec_string,
            id(self)
        )


    def __dir__(self):
        """Filter the output of `dir(mock)` to only useful members."""
        if not FILTER_DIR:
            return object.__dir__(self)

        extras = self._mock_methods or []
        from_type = dir(type(self))
        from_dict = list(self.__dict__)
        from_child_mocks = [
            m_name for m_name, m_value in self._mock_children.items()
            if m_value is not _deleted]

        from_type = [e for e in from_type if not e.startswith('_')]
        from_dict = [e for e in from_dict if not e.startswith('_') or
                     _is_magic(e)]
        return sorted(set(extras + from_type + from_dict + from_child_mocks))


    def __setattr__(self, name, value):
        if name in _allowed_names:
            # property setters go through here
            return object.__setattr__(self, name, value)
        elif (self._spec_set and self._mock_methods is not None and
            name not in self._mock_methods and
            name not in self.__dict__):
            raise AttributeError("Mock object has no attribute '%s'" % name)
        elif name in _unsupported_magics:
            msg = 'Attempting to set unsupported magic method %r.' % name
            raise AttributeError(msg)
        elif name in _all_magics:
            if self._mock_methods is not None and name not in self._mock_methods:
                raise AttributeError("Mock object has no attribute '%s'" % name)

            if not _is_instance_mock(value):
                setattr(type(self), name, _get_method(name, value))
                original = value
                value = lambda *args, **kw: original(self, *args, **kw)
            else:
                # only set _new_name and not name so that mock_calls is tracked
                # but not method calls
                _check_and_set_parent(self, value, None, name)
                setattr(type(self), name, value)
                self._mock_children[name] = value
        elif name == '__class__':
            self._spec_class = value
            return
        else:
            if _check_and_set_parent(self, value, name, name):
                self._mock_children[name] = value

        if self._mock_sealed and not hasattr(self, name):
            mock_name = f'{self._extract_mock_name()}.{name}'
            raise AttributeError(f'Cannot set {mock_name}')

        return object.__setattr__(self, name, value)


    def __delattr__(self, name):
        if name in _all_magics and name in type(self).__dict__:
            delattr(type(self), name)
            if name not in self.__dict__:
                # for magic methods that are still MagicProxy objects and
                # not set on the instance itself
                return

        obj = self._mock_children.get(name, _missing)
        if name in self.__dict__:
            _safe_super(NonCallableMock, self).__delattr__(name)
        elif obj is _deleted:
            raise AttributeError(name)
        if obj is not _missing:
            del self._mock_children[name]
        self._mock_children[name] = _deleted


    def _format_mock_call_signature(self, args, kwargs):
        name = self._mock_name or 'mock'
        return _format_call_signature(name, args, kwargs)


    def _format_mock_failure_message(self, args, kwargs, action='call'):
        message = 'expected %s not found.\nExpected: %s\nActual: %s'
        expected_string = self._format_mock_call_signature(args, kwargs)
        call_args = self.call_args
        actual_string = self._format_mock_call_signature(*call_args)
        return message % (action, expected_string, actual_string)


    def _get_call_signature_from_name(self, name):
        """
        * If call objects are asserted against a method/function like obj.meth1
        then there could be no name for the call object to lookup. Hence just
        return the spec_signature of the method/function being asserted against.
        * If the name is not empty then remove () and split by '.' to get
        list of names to iterate through the children until a potential
        match is found. A child mock is created only during attribute access
        so if we get a _SpecState then no attributes of the spec were accessed
        and can be safely exited.
        """
        if not name:
            return self._spec_signature

        sig = None
        names = name.replace('()', '').split('.')
        children = self._mock_children

        for name in names:
            child = children.get(name)
            if child is None or isinstance(child, _SpecState):
                break
            else:
                # If an autospecced object is attached using attach_mock the
                # child would be a function with mock object as attribute from
                # which signature has to be derived.
                child = _extract_mock(child)
                children = child._mock_children
                sig = child._spec_signature

        return sig


    def _call_matcher(self, _call):
        """
        Given a call (or simply an (args, kwargs) tuple), return a
        comparison key suitable for matching with other calls.
        This is a best effort method which relies on the spec's signature,
        if available, or falls back on the arguments themselves.
        """

        if isinstance(_call, tuple) and len(_call) > 2:
            sig = self._get_call_signature_from_name(_call[0])
        else:
            sig = self._spec_signature

        if sig is not None:
            if len(_call) == 2:
                name = ''
                args, kwargs = _call
            else:
                name, args, kwargs = _call
            try:
                bound_call = sig.bind(*args, **kwargs)
                return call(name, bound_call.args, bound_call.kwargs)
            except TypeError as e:
                return e.with_traceback(None)
        else:
            return _call

    def assert_not_called(self):
        """assert that the mock was never called.
        """
        if self.call_count != 0:
            msg = ("Expected '%s' to not have been called. Called %s times.%s"
                   % (self._mock_name or 'mock',
                      self.call_count,
                      self._calls_repr()))
            raise AssertionError(msg)

    def assert_called(self):
        """assert that the mock was called at least once
        """
        if self.call_count == 0:
            msg = ("Expected '%s' to have been called." %
                   (self._mock_name or 'mock'))
            raise AssertionError(msg)

    def assert_called_once(self):
        """assert that the mock was called only once.
        """
        if not self.call_count == 1:
            msg = ("Expected '%s' to have been called once. Called %s times.%s"
                   % (self._mock_name or 'mock',
                      self.call_count,
                      self._calls_repr()))
            raise AssertionError(msg)

    def assert_called_with(self, /, *args, **kwargs):
        """assert that the last call was made with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            actual = 'not called.'
            error_message = ('expected call not found.\nExpected: %s\nActual: %s'
                    % (expected, actual))
            raise AssertionError(error_message)

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs)
            return msg
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.call_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
            raise AssertionError(_error_message()) from cause


    def assert_called_once_with(self, /, *args, **kwargs):
        """assert that the mock was called exactly once and that that call was
        with the specified arguments."""
        if not self.call_count == 1:
            msg = ("Expected '%s' to be called once. Called %s times.%s"
                   % (self._mock_name or 'mock',
                      self.call_count,
                      self._calls_repr()))
            raise AssertionError(msg)
        return self.assert_called_with(*args, **kwargs)


    def assert_has_calls(self, calls, any_order=False):
        """assert the mock has been called with the specified calls.
        The `mock_calls` list is checked for the calls.

        If `any_order` is False (the default) then the calls must be
        sequential. There can be extra calls before or after the
        specified calls.

        If `any_order` is True then the calls can be in any order, but
        they must all appear in `mock_calls`."""
        expected = [self._call_matcher(c) for c in calls]
        cause = next((e for e in expected if isinstance(e, Exception)), None)
        all_calls = _CallList(self._call_matcher(c) for c in self.mock_calls)
        if not any_order:
            if expected not in all_calls:
                if cause is None:
                    problem = 'Calls not found.'
                else:
                    problem = ('Error processing expected calls.\n'
                               'Errors: {}').format(
                                   [e if isinstance(e, Exception) else None
                                    for e in expected])
                raise AssertionError(
                    f'{problem}\n'
                    f'Expected: {_CallList(calls)}'
                    f'{self._calls_repr(prefix="Actual").rstrip(".")}'
                ) from cause
            return

        all_calls = list(all_calls)

        not_found = []
        for kall in expected:
            try:
                all_calls.remove(kall)
            except ValueError:
                not_found.append(kall)
        if not_found:
            raise AssertionError(
                '%r does not contain all of %r in its call list, '
                'found %r instead' % (self._mock_name or 'mock',
                                      tuple(not_found), all_calls)
            ) from cause


    def assert_any_call(self, /, *args, **kwargs):
        """assert the mock has been called with the specified arguments.

        The assert passes if the mock has *ever* been called, unlike
        `assert_called_with` and `assert_called_once_with` that only pass if
        the call is the most recent one."""
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        cause = expected if isinstance(expected, Exception) else None
        actual = [self._call_matcher(c) for c in self.call_args_list]
        if cause or expected not in _AnyComparer(actual):
            expected_string = self._format_mock_call_signature(args, kwargs)
            raise AssertionError(
                '%s call not found' % expected_string
            ) from cause


    def _get_child_mock(self, /, **kw):
        """Create the child mocks for attributes and return value.
        By default child mocks will be the same type as the parent.
        Subclasses of Mock may want to override this to customize the way
        child mocks are made.

        For non-callable mocks the callable variant will be used (rather than
        any custom subclass)."""
        if self._mock_sealed:
            attribute = f".{kw['name']}" if "name" in kw else "()"
            mock_name = self._extract_mock_name() + attribute
            raise AttributeError(mock_name)

        _new_name = kw.get("_new_name")
        if _new_name in self.__dict__['_spec_asyncs']:
            return AsyncMock(**kw)

        _type = type(self)
        if issubclass(_type, MagicMock) and _new_name in _async_method_magics:
            # Any asynchronous magic becomes an AsyncMock
            klass = AsyncMock
        elif issubclass(_type, AsyncMockMixin):
            if (_new_name in _all_sync_magics or
                    self._mock_methods and _new_name in self._mock_methods):
                # Any synchronous method on AsyncMock becomes a MagicMock
                klass = MagicMock
            else:
                klass = AsyncMock
        elif not issubclass(_type, CallableMixin):
            if issubclass(_type, NonCallableMagicMock):
                klass = MagicMock
            elif issubclass(_type, NonCallableMock):
                klass = Mock
        else:
            klass = _type.__mro__[1]
        return klass(**kw)


    def _calls_repr(self, prefix="Calls"):
        """Renders self.mock_calls as a string.

        Example: "\nCalls: [call(1), call(2)]."

        If self.mock_calls is empty, an empty string is returned. The
        output will be truncated if very long.
        """
        if not self.mock_calls:
            return ""
        return f"\n{prefix}: {safe_repr(self.mock_calls)}."


_MOCK_SIG = inspect.signature(NonCallableMock.__init__)


class _AnyComparer(list):
    """A list which checks if it contains a call which may have an
    argument of ANY, flipping the components of item and self from
    their traditional locations so that ANY is guaranteed to be on
    the left."""
    def __contains__(self, item):
        for _call in self:
            assert len(item) == len(_call)
            if all([
                expected == actual
                for expected, actual in zip(item, _call)
            ]):
                return True
        return False


def _try_iter(obj):
    if obj is None:
        return obj
    if _is_exception(obj):
        return obj
    if _callable(obj):
        return obj
    try:
        return iter(obj)
    except TypeError:
        # XXXX backwards compatibility
        # but this will blow up on first call - so maybe we should fail early?
        return obj


class CallableMixin(Base):

    def __init__(self, spec=None, side_effect=None, return_value=DEFAULT,
                 wraps=None, name=None, spec_set=None, parent=None,
                 _spec_state=None, _new_name='', _new_parent=None, **kwargs):
        self.__dict__['_mock_return_value'] = return_value
        _safe_super(CallableMixin, self).__init__(
            spec, wraps, name, spec_set, parent,
            _spec_state, _new_name, _new_parent, **kwargs
        )

        self.side_effect = side_effect


    def _mock_check_sig(self, /, *args, **kwargs):
        # stub method that can be replaced with one with a specific signature
        pass


    def __call__(self, /, *args, **kwargs):
        # can't use self in-case a function / method we are mocking uses self
        # in the signature
        self._mock_check_sig(*args, **kwargs)
        self._increment_mock_call(*args, **kwargs)
        return self._mock_call(*args, **kwargs)


    def _mock_call(self, /, *args, **kwargs):
        return self._execute_mock_call(*args, **kwargs)

    def _increment_mock_call(self, /, *args, **kwargs):
        self.called = True
        self.call_count += 1

        # handle call_args
        # needs to be set here so assertions on call arguments pass before
        # execution in the case of awaited calls
        _call = _Call((args, kwargs), two=True)
        self.call_args = _call
        self.call_args_list.append(_call)

        # initial stuff for method_calls:
        do_method_calls = self._mock_parent is not None
        method_call_name = self._mock_name

        # initial stuff for mock_calls:
        mock_call_name = self._mock_new_name
        is_a_call = mock_call_name == '()'
        self.mock_calls.append(_Call(('', args, kwargs)))

        # follow up the chain of mocks:
        _new_parent = self._mock_new_parent
        while _new_parent is not None:

            # handle method_calls:
            if do_method_calls:
                _new_parent.method_calls.append(_Call((method_call_name, args, kwargs)))
                do_method_calls = _new_parent._mock_parent is not None
                if do_method_calls:
                    method_call_name = _new_parent._mock_name + '.' + method_call_name

            # handle mock_calls:
            this_mock_call = _Call((mock_call_name, args, kwargs))
            _new_parent.mock_calls.append(this_mock_call)

            if _new_parent._mock_new_name:
                if is_a_call:
                    dot = ''
                else:
                    dot = '.'
                is_a_call = _new_parent._mock_new_name == '()'
                mock_call_name = _new_parent._mock_new_name + dot + mock_call_name

            # follow the parental chain:
            _new_parent = _new_parent._mock_new_parent

    def _execute_mock_call(self, /, *args, **kwargs):
        # separate from _increment_mock_call so that awaited functions are
        # executed separately from their call, also AsyncMock overrides this method

        effect = self.side_effect
        if effect is not None:
            if _is_exception(effect):
                raise effect
            elif not _callable(effect):
                result = next(effect)
                if _is_exception(result):
                    raise result
            else:
                result = effect(*args, **kwargs)

            if result is not DEFAULT:
                return result

        if self._mock_return_value is not DEFAULT:
            return self.return_value

        if self._mock_wraps is not None:
            return self._mock_wraps(*args, **kwargs)

        return self.return_value



class Mock(CallableMixin, NonCallableMock):
    """
    Create a new `Mock` object. `Mock` takes several optional arguments
    that specify the behaviour of the Mock object:

    * `spec`: This can be either a list of strings or an existing object (a
      class or instance) that acts as the specification for the mock object. If
      you pass in an object then a list of strings is formed by calling dir on
      the object (excluding unsupported magic attributes and methods). Accessing
      any attribute not in this list will raise an `AttributeError`.

      If `spec` is an object (rather than a list of strings) then
      `mock.__class__` returns the class of the spec object. This allows mocks
      to pass `isinstance` tests.

    * `spec_set`: A stricter variant of `spec`. If used, attempting to *set*
      or get an attribute on the mock that isn't on the object passed as
      `spec_set` will raise an `AttributeError`.

    * `side_effect`: A function to be called whenever the Mock is called. See
      the `side_effect` attribute. Useful for raising exceptions or
      dynamically changing return values. The function is called with the same
      arguments as the mock, and unless it returns `DEFAULT`, the return
      value of this function is used as the return value.

      If `side_effect` is an iterable then each call to the mock will return
      the next value from the iterable. If any of the members of the iterable
      are exceptions they will be raised instead of returned.

    * `return_value`: The value returned when the mock is called. By default
      this is a new Mock (created on first access). See the
      `return_value` attribute.

    * `wraps`: Item for the mock object to wrap. If `wraps` is not None then
      calling the Mock will pass the call through to the wrapped object
      (returning the real result). Attribute access on the mock will return a
      Mock object that wraps the corresponding attribute of the wrapped object
      (so attempting to access an attribute that doesn't exist will raise an
      `AttributeError`).

      If the mock has an explicit `return_value` set then calls are not passed
      to the wrapped object and the `return_value` is returned instead.

    * `name`: If the mock has a name then it will be used in the repr of the
      mock. This can be useful for debugging. The name is propagated to child
      mocks.

    Mocks can also be called with arbitrary keyword arguments. These will be
    used to set attributes on the mock after it is created.
    """


def _dot_lookup(thing, comp, import_path):
    try:
        return getattr(thing, comp)
    except AttributeError:
        __import__(import_path)
        return getattr(thing, comp)


def _importer(target):
    components = target.split('.')
    import_path = components.pop(0)
    thing = __import__(import_path)

    for comp in components:
        import_path += ".%s" % comp
        thing = _dot_lookup(thing, comp, import_path)
    return thing


# _check_spec_arg_typos takes kwargs from commands like patch and checks that
# they don't contain common misspellings of arguments related to autospeccing.
def _check_spec_arg_typos(kwargs_to_check):
    typos = ("autospect", "auto_spec", "set_spec")
    for typo in typos:
        if typo in kwargs_to_check:
            raise RuntimeError(
                f"{typo!r} might be a typo; use unsafe=True if this is intended"
            )


class _patch(object):

    attribute_name = None
    _active_patches = []

    def __init__(
            self, getter, attribute, new, spec, create,
            spec_set, autospec, new_callable, kwargs, *, unsafe=False
        ):
        if new_callable is not None:
            if new is not DEFAULT:
                raise ValueError(
                    "Cannot use 'new' and 'new_callable' together"
                )
            if autospec is not None:
                raise ValueError(
                    "Cannot use 'autospec' and 'new_callable' together"
                )
        if not unsafe:
            _check_spec_arg_typos(kwargs)
        if _is_instance_mock(spec):
            raise InvalidSpecError(
                f'Cannot spec attr {attribute!r} as the spec '
                f'has already been mocked out. [spec={spec!r}]')
        if _is_instance_mock(spec_set):
            raise InvalidSpecError(
                f'Cannot spec attr {attribute!r} as the spec_set '
                f'target has already been mocked out. [spec_set={spec_set!r}]')

        self.getter = getter
        self.attribute = attribute
        self.new = new
        self.new_callable = new_callable
        self.spec = spec
        self.create = create
        self.has_local = False
        self.spec_set = spec_set
        self.autospec = autospec
        self.kwargs = kwargs
        self.additional_patchers = []


    def copy(self):
        patcher = _patch(
            self.getter, self.attribute, self.new, self.spec,
            self.create, self.spec_set,
            self.autospec, self.new_callable, self.kwargs
        )
        patcher.attribute_name = self.attribute_name
        patcher.additional_patchers = [
            p.copy() for p in self.additional_patchers
        ]
        return patcher


    def __call__(self, func):
        if isinstance(func, type):
            return self.decorate_class(func)
        if inspect.iscoroutinefunction(func):
            return self.decorate_async_callable(func)
        return self.decorate_callable(func)


    def decorate_class(self, klass):
        for attr in dir(klass):
            if not attr.startswith(patch.TEST_PREFIX):
                continue

            attr_value = getattr(klass, attr)
            if not hasattr(attr_value, "__call__"):
                continue

            patcher = self.copy()
            setattr(klass, attr, patcher(attr_value))
        return klass


    @contextlib.contextmanager
    def decoration_helper(self, patched, args, keywargs):
        extra_args = []
        with contextlib.ExitStack() as exit_stack:
            for patching in patched.patchings:
                arg = exit_stack.enter_context(patching)
                if patching.attribute_name is not None:
                    keywargs.update(arg)
                elif patching.new is DEFAULT:
                    extra_args.append(arg)

            args += tuple(extra_args)
            yield (args, keywargs)


    def decorate_callable(self, func):
        # NB. Keep the method in sync with decorate_async_callable()
        if hasattr(func, 'patchings'):
            func.patchings.append(self)
            return func

        @wraps(func)
        def patched(*args, **keywargs):
            with self.decoration_helper(patched,
                                        args,
                                        keywargs) as (newargs, newkeywargs):
                return func(*newargs, **newkeywargs)

        patched.patchings = [self]
        return patched


    def decorate_async_callable(self, func):
        # NB. Keep the method in sync with decorate_callable()
        if hasattr(func, 'patchings'):
            func.patchings.append(self)
            return func

        @wraps(func)
        async def patched(*args, **keywargs):
            with self.decoration_helper(patched,
                                        args,
                                        keywargs) as (newargs, newkeywargs):
                return await func(*newargs, **newkeywargs)

        patched.patchings = [self]
        return patched


    def get_original(self):
        target = self.getter()
        name = self.attribute

        original = DEFAULT
        local = False

        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True

        if name in _builtins and isinstance(target, ModuleType):
            self.create = True

        if not self.create and original is DEFAULT:
            raise AttributeError(
                "%s does not have the attribute %r" % (target, name)
            )
        return original, local


    def __enter__(self):
        """Perform the patch."""
        new, spec, spec_set = self.new, self.spec, self.spec_set
        autospec, kwargs = self.autospec, self.kwargs
        new_callable = self.new_callable
        self.target = self.getter()

        # normalise False to None
        if spec is False:
            spec = None
        if spec_set is False:
            spec_set = None
        if autospec is False:
            autospec = None

        if spec is not None and autospec is not None:
            raise TypeError("Can't specify spec and autospec")
        if ((spec is not None or autospec is not None) and
            spec_set not in (True, None)):
            raise TypeError("Can't provide explicit spec_set *and* spec or autospec")

        original, local = self.get_original()

        if new is DEFAULT and autospec is None:
            inherit = False
            if spec is True:
                # set spec to the object we are replacing
                spec = original
                if spec_set is True:
                    spec_set = original
                    spec = None
            elif spec is not None:
                if spec_set is True:
                    spec_set = spec
                    spec = None
            elif spec_set is True:
                spec_set = original

            if spec is not None or spec_set is not None:
                if original is DEFAULT:
                    raise TypeError("Can't use 'spec' with create=True")
                if isinstance(original, type):
                    # If we're patching out a class and there is a spec
                    inherit = True
            if spec is None and _is_async_obj(original):
                Klass = AsyncMock
            else:
                Klass = MagicMock
            _kwargs = {}
            if new_callable is not None:
                Klass = new_callable
            elif spec is not None or spec_set is not None:
                this_spec = spec
                if spec_set is not None:
                    this_spec = spec_set
                if _is_list(this_spec):
                    not_callable = '__call__' not in this_spec
                else:
                    not_callable = not callable(this_spec)
                if _is_async_obj(this_spec):
                    Klass = AsyncMock
                elif not_callable:
                    Klass = NonCallableMagicMock

            if spec is not None:
                _kwargs['spec'] = spec
            if spec_set is not None:
                _kwargs['spec_set'] = spec_set

            # add a name to mocks
            if (isinstance(Klass, type) and
                issubclass(Klass, NonCallableMock) and self.attribute):
                _kwargs['name'] = self.attribute

            _kwargs.update(kwargs)
            new = Klass(**_kwargs)

            if inherit and _is_instance_mock(new):
                # we can only tell if the instance should be callable if the
                # spec is not a list
                this_spec = spec
                if spec_set is not None:
                    this_spec = spec_set
                if (not _is_list(this_spec) and not
                    _instance_callable(this_spec)):
                    Klass = NonCallableMagicMock

                _kwargs.pop('name')
                new.return_value = Klass(_new_parent=new, _new_name='()',
                                         **_kwargs)
        elif autospec is not None:
            # spec is ignored, new *must* be default, spec_set is treated
            # as a boolean. Should we check spec is not None and that spec_set
            # is a bool?
            if new is not DEFAULT:
                raise TypeError(
                    "autospec creates the mock for you. Can't specify "
                    "autospec and new."
                )
            if original is DEFAULT:
                raise TypeError("Can't use 'autospec' with create=True")
            spec_set = bool(spec_set)
            if autospec is True:
                autospec = original

            if _is_instance_mock(self.target):
                raise InvalidSpecError(
                    f'Cannot autospec attr {self.attribute!r} as the patch '
                    f'target has already been mocked out. '
                    f'[target={self.target!r}, attr={autospec!r}]')
            if _is_instance_mock(autospec):
                target_name = getattr(self.target, '__name__', self.target)
                raise InvalidSpecError(
                    f'Cannot autospec attr {self.attribute!r} from target '
                    f'{target_name!r} as it has already been mocked out. '
                    f'[target={self.target!r}, attr={autospec!r}]')

            new = create_autospec(autospec, spec_set=spec_set,
                                  _name=self.attribute, **kwargs)
        elif kwargs:
            # can't set keyword args when we aren't creating the mock
            # XXXX If new is a Mock we could call new.configure_mock(**kwargs)
            raise TypeError("Can't pass kwargs to a mock we aren't creating")

        new_attr = new

        self.temp_original = original
        self.is_local = local
        self._exit_stack = contextlib.ExitStack()
        try:
            setattr(self.target, self.attribute, new_attr)
            if self.attribute_name is not None:
                extra_args = {}
                if self.new is DEFAULT:
                    extra_args[self.attribute_name] =  new
                for patching in self.additional_patchers:
                    arg = self._exit_stack.enter_context(patching)
                    if patching.new is DEFAULT:
                        extra_args.update(arg)
                return extra_args

            return new
        except:
            if not self.__exit__(*sys.exc_info()):
                raise

    def __exit__(self, *exc_info):
        """Undo the patch."""
        if self.is_local and self.temp_original is not DEFAULT:
            setattr(self.target, self.attribute, self.temp_original)
        else:
            delattr(self.target, self.attribute)
            if not self.create and (not hasattr(self.target, self.attribute) or
                        self.attribute in ('__doc__', '__module__',
                                           '__defaults__', '__annotations__',
                                           '__kwdefaults__')):
                # needed for proxy objects like django settings
                setattr(self.target, self.attribute, self.temp_original)

        del self.temp_original
        del self.is_local
        del self.target
        exit_stack = self._exit_stack
        del self._exit_stack
        return exit_stack.__exit__(*exc_info)


    def start(self):
        """Activate a patch, returning any created mock."""
        result = self.__enter__()
        self._active_patches.append(self)
        return result


    def stop(self):
        """Stop an active patch."""
        try:
            self._active_patches.remove(self)
        except ValueError:
            # If the patch hasn't been started this will fail
            return None

        return self.__exit__(None, None, None)



def _get_target(target):
    try:
        target, attribute = target.rsplit('.', 1)
    except (TypeError, ValueError, AttributeError):
        raise TypeError(
            f"Need a valid target to patch. You supplied: {target!r}")
    getter = lambda: _importer(target)
    return getter, attribute


def _patch_object(
        target, attribute, new=DEFAULT, spec=None,
        create=False, spec_set=None, autospec=None,
        new_callable=None, *, unsafe=False, **kwargs
    ):
    """
    patch the named member (`attribute`) on an object (`target`) with a mock
    object.

    `patch.object` can be used as a decorator, class decorator or a context
    manager. Arguments `new`, `spec`, `create`, `spec_set`,
    `autospec` and `new_callable` have the same meaning as for `patch`. Like
    `patch`, `patch.object` takes arbitrary keyword arguments for configuring
    the mock object it creates.

    When used as a class decorator `patch.object` honours `patch.TEST_PREFIX`
    for choosing which methods to wrap.
    """
    if type(target) is str:
        raise TypeError(
            f"{target!r} must be the actual object to be patched, not a str"
        )
    getter = lambda: target
    return _patch(
        getter, attribute, new, spec, create,
        spec_set, autospec, new_callable, kwargs, unsafe=unsafe
    )


def _patch_multiple(target, spec=None, create=False, spec_set=None,
                    autospec=None, new_callable=None, **kwargs):
    """Perform multiple patches in a single call. It takes the object to be
    patched (either as an object or a string to fetch the object by importing)
    and keyword arguments for the patches::

        with patch.multiple(settings, FIRST_PATCH='one', SECOND_PATCH='two'):
            ...

    Use `DEFAULT` as the value if you want `patch.multiple` to create
    mocks for you. In this case the created mocks are passed into a decorated
    function by keyword, and a dictionary is returned when `patch.multiple` is
    used as a context manager.

    `patch.multiple` can be used as a decorator, class decorator or a context
    manager. The arguments `spec`, `spec_set`, `create`,
    `autospec` and `new_callable` have the same meaning as for `patch`. These
    arguments will be applied to *all* patches done by `patch.multiple`.

    When used as a class decorator `patch.multiple` honours `patch.TEST_PREFIX`
    for choosing which methods to wrap.
    """
    if type(target) is str:
        getter = lambda: _importer(target)
    else:
        getter = lambda: target

    if not kwargs:
        raise ValueError(
            'Must supply at least one keyword argument with patch.multiple'
        )
    # need to wrap in a list for python 3, where items is a view
    items = list(kwargs.items())
    attribute, new = items[0]
    patcher = _patch(
        getter, attribute, new, spec, create, spec_set,
        autospec, new_callable, {}
    )
    patcher.attribute_name = attribute
    for attribute, new in items[1:]:
        this_patcher = _patch(
            getter, attribute, new, spec, create, spec_set,
            autospec, new_callable, {}
        )
        this_patcher.attribute_name = attribute
        patcher.additional_patchers.append(this_patcher)
    return patcher


def patch(
        target, new=DEFAULT, spec=None, create=False,
        spec_set=None, autospec=None, new_callable=None, *, unsafe=False, **kwargs
    ):
    """
    `patch` acts as a function decorator, class decorator or a context
    manager. Inside the body of the function or with statement, the `target`
    is patched with a `new` object. When the function/with statement exits
    the patch is undone.

    If `new` is omitted, then the target is replaced with an
    `AsyncMock if the patched object is an async function or a
    `MagicMock` otherwise. If `patch` is used as a decorator and `new` is
    omitted, the created mock is passed in as an extra argument to the
    decorated function. If `patch` is used as a context manager the created
    mock is returned by the context manager.

    `target` should be a string in the form `'package.module.ClassName'`. The
    `target` is imported and the specified object replaced with the `new`
    object, so the `target` must be importable from the environment you are
    calling `patch` from. The target is imported when the decorated function
    is executed, not at decoration time.

    The `spec` and `spec_set` keyword arguments are passed to the `MagicMock`
    if patch is creating one for you.

    In addition you can pass `spec=True` or `spec_set=True`, which causes
    patch to pass in the object being mocked as the spec/spec_set object.

    `new_callable` allows you to specify a different class, or callable object,
    that will be called to create the `new` object. By default `AsyncMock` is
    used for async functions and `MagicMock` for the rest.

    A more powerful form of `spec` is `autospec`. If you set `autospec=True`
    then the mock will be created with a spec from the object being replaced.
    All attributes of the mock will also have the spec of the corresponding
    attribute of the object being replaced. Methods and functions being
    mocked will have their arguments checked and will raise a `TypeError` if
    they are called with the wrong signature. For mocks replacing a class,
    their return value (the 'instance') will have the same spec as the class.

    Instead of `autospec=True` you can pass `autospec=some_object` to use an
    arbitrary object as the spec instead of the one being replaced.

    By default `patch` will fail to replace attributes that don't exist. If
    you pass in `create=True`, and the attribute doesn't exist, patch will
    create the attribute for you when the patched function is called, and
    delete it again afterwards. This is useful for writing tests against
    attributes that your production code creates at runtime. It is off by
    default because it can be dangerous. With it switched on you can write
    passing tests against APIs that don't actually exist!

    Patch can be used as a `TestCase` class decorator. It works by
    decorating each test method in the class. This reduces the boilerplate
    code when your test methods share a common patchings set. `patch` finds
    tests by looking for method names that start with `patch.TEST_PREFIX`.
    By default this is `test`, which matches the way `unittest` finds tests.
    You can specify an alternative prefix by setting `patch.TEST_PREFIX`.

    Patch can be used as a context manager, with the with statement. Here the
    patching applies to the indented block after the with statement. If you
    use "as" then the patched object will be bound to the name after the
    "as"; very useful if `patch` is creating a mock object for you.

    Patch will raise a `RuntimeError` if passed some common misspellings of
    the arguments autospec and spec_set. Pass the argument `unsafe` with the
    value True to disable that check.

    `patch` takes arbitrary keyword arguments. These will be passed to
    `AsyncMock` if the patched object is asynchronous, to `MagicMock`
    otherwise or to `new_callable` if specified.

    `patch.dict(...)`, `patch.multiple(...)` and `patch.object(...)` are
    available for alternate use-cases.
    """
    getter, attribute = _get_target(target)
    return _patch(
        getter, attribute, new, spec, create,
        spec_set, autospec, new_callable, kwargs, unsafe=unsafe
    )


class _patch_dict(object):
    """
    Patch a dictionary, or dictionary like object, and restore the dictionary
    to its original state after the test.

    `in_dict` can be a dictionary or a mapping like container. If it is a
    mapping then it must at least support getting, setting and deleting items
    plus iterating over keys.

    `in_dict` can also be a string specifying the name of the dictionary, which
    will then be fetched by importing it.

    `values` can be a dictionary of values to set in the dictionary. `values`
    can also be an iterable of `(key, value)` pairs.

    If `clear` is True then the dictionary will be cleared before the new
    values are set.

    `patch.dict` can also be called with arbitrary keyword arguments to set
    values in the dictionary::

        with patch.dict('sys.modules', mymodule=Mock(), other_module=Mock()):
            ...

    `patch.dict` can be used as a context manager, decorator or class
    decorator. When used as a class decorator `patch.dict` honours
    `patch.TEST_PREFIX` for choosing which methods to wrap.
    """

    def __init__(self, in_dict, values=(), clear=False, **kwargs):
        self.in_dict = in_dict
        # support any argument supported by dict(...) constructor
        self.values = dict(values)
        self.values.update(kwargs)
        self.clear = clear
        self._original = None


    def __call__(self, f):
        if isinstance(f, type):
            return self.decorate_class(f)
        if inspect.iscoroutinefunction(f):
            return self.decorate_async_callable(f)
        return self.decorate_callable(f)


    def decorate_callable(self, f):
        @wraps(f)
        def _inner(*args, **kw):
            self._patch_dict()
            try:
                return f(*args, **kw)
            finally:
                self._unpatch_dict()

        return _inner


    def decorate_async_callable(self, f):
        @wraps(f)
        async def _inner(*args, **kw):
            self._patch_dict()
            try:
                return await f(*args, **kw)
            finally:
                self._unpatch_dict()

        return _inner


    def decorate_class(self, klass):
        for attr in dir(klass):
            attr_value = getattr(klass, attr)
            if (attr.startswith(patch.TEST_PREFIX) and
                 hasattr(attr_value, "__call__")):
                decorator = _patch_dict(self.in_dict, self.values, self.clear)
                decorated = decorator(attr_value)
                setattr(klass, attr, decorated)
        return klass


    def __enter__(self):
        """Patch the dict."""
        self._patch_dict()
        return self.in_dict


    def _patch_dict(self):
        values = self.values
        if isinstance(self.in_dict, str):
            self.in_dict = _importer(self.in_dict)
        in_dict = self.in_dict
        clear = self.clear

        try:
            original = in_dict.copy()
        except AttributeError:
            # dict like object with no copy method
            # must support iteration over keys
            original = {}
            for key in in_dict:
                original[key] = in_dict[key]
        self._original = original

        if clear:
            _clear_dict(in_dict)

        try:
            in_dict.update(values)
        except AttributeError:
            # dict like object with no update method
            for key in values:
                in_dict[key] = values[key]


    def _unpatch_dict(self):
        in_dict = self.in_dict
        original = self._original

        _clear_dict(in_dict)

        try:
            in_dict.update(original)
        except AttributeError:
            for key in original:
                in_dict[key] = original[key]


    def __exit__(self, *args):
        """Unpatch the dict."""
        if self._original is not None:
            self._unpatch_dict()
        return False


    def start(self):
        """Activate a patch, returning any created mock."""
        result = self.__enter__()
        _patch._active_patches.append(self)
        return result


    def stop(self):
        """Stop an active patch."""
        try:
            _patch._active_patches.remove(self)
        except ValueError:
            # If the patch hasn't been started this will fail
            return None

        return self.__exit__(None, None, None)


def _clear_dict(in_dict):
    try:
        in_dict.clear()
    except AttributeError:
        keys = list(in_dict)
        for key in keys:
            del in_dict[key]


def _patch_stopall():
    """Stop all active patches. LIFO to unroll nested patches."""
    for patch in reversed(_patch._active_patches):
        patch.stop()


patch.object = _patch_object
patch.dict = _patch_dict
patch.multiple = _patch_multiple
patch.stopall = _patch_stopall
patch.TEST_PREFIX = 'test'

magic_methods = (
    "lt le gt ge eq ne "
    "getitem setitem delitem "
    "len contains iter "
    "hash str sizeof "
    "enter exit "
    # we added divmod and rdivmod here instead of numerics
    # because there is no idivmod
    "divmod rdivmod neg pos abs invert "
    "complex int float index "
    "round trunc floor ceil "
    "bool next "
    "fspath "
    "aiter "
)

numerics = (
    "add sub mul matmul div floordiv mod lshift rshift and xor or pow truediv"
)
inplace = ' '.join('i%s' % n for n in numerics.split())
right = ' '.join('r%s' % n for n in numerics.split())

# not including __prepare__, __instancecheck__, __subclasscheck__
# (as they are metaclass methods)
# __del__ is not supported at all as it causes problems if it exists

_non_defaults = {
    '__get__', '__set__', '__delete__', '__reversed__', '__missing__',
    '__reduce__', '__reduce_ex__', '__getinitargs__', '__getnewargs__',
    '__getstate__', '__setstate__', '__getformat__', '__setformat__',
    '__repr__', '__dir__', '__subclasses__', '__format__',
    '__getnewargs_ex__',
}


def _get_method(name, func):
    "Turns a callable object (like a mock) into a real function"
    def method(self, /, *args, **kw):
        return func(self, *args, **kw)
    method.__name__ = name
    return method


_magics = {
    '__%s__' % method for method in
    ' '.join([magic_methods, numerics, inplace, right]).split()
}

# Magic methods used for async `with` statements
_async_method_magics = {"__aenter__", "__aexit__", "__anext__"}
# Magic methods that are only used with async calls but are synchronous functions themselves
_sync_async_magics = {"__aiter__"}
_async_magics = _async_method_magics | _sync_async_magics

_all_sync_magics = _magics | _non_defaults
_all_magics = _all_sync_magics | _async_magics

_unsupported_magics = {
    '__getattr__', '__setattr__',
    '__init__', '__new__', '__prepare__',
    '__instancecheck__', '__subclasscheck__',
    '__del__'
}

_calculate_return_value = {
    '__hash__': lambda self: object.__hash__(self),
    '__str__': lambda self: object.__str__(self),
    '__sizeof__': lambda self: object.__sizeof__(self),
    '__fspath__': lambda self: f"{type(self).__name__}/{self._extract_mock_name()}/{id(self)}",
}

_return_values = {
    '__lt__': NotImplemented,
    '__gt__': NotImplemented,
    '__le__': NotImplemented,
    '__ge__': NotImplemented,
    '__int__': 1,
    '__contains__': False,
    '__len__': 0,
    '__exit__': False,
    '__complex__': 1j,
    '__float__': 1.0,
    '__bool__': True,
    '__index__': 1,
    '__aexit__': False,
}


def _get_eq(self):
    def __eq__(other):
        ret_val = self.__eq__._mock_return_value
        if ret_val is not DEFAULT:
            return ret_val
        if self is other:
            return True
        return NotImplemented
    return __eq__

def _get_ne(self):
    def __ne__(other):
        if self.__ne__._mock_return_value is not DEFAULT:
            return DEFAULT
        if self is other:
            return False
        return NotImplemented
    return __ne__

def _get_iter(self):
    def __iter__():
        ret_val = self.__iter__._mock_return_value
        if ret_val is DEFAULT:
            return iter([])
        # if ret_val was already an iterator, then calling iter on it should
        # return the iterator unchanged
        return iter(ret_val)
    return __iter__

def _get_async_iter(self):
    def __aiter__():
        ret_val = self.__aiter__._mock_return_value
        if ret_val is DEFAULT:
            return _AsyncIterator(iter([]))
        return _AsyncIterator(iter(ret_val))
    return __aiter__

_side_effect_methods = {
    '__eq__': _get_eq,
    '__ne__': _get_ne,
    '__iter__': _get_iter,
    '__aiter__': _get_async_iter
}



def _set_return_value(mock, method, name):
    fixed = _return_values.get(name, DEFAULT)
    if fixed is not DEFAULT:
        method.return_value = fixed
        return

    return_calculator = _calculate_return_value.get(name)
    if return_calculator is not None:
        return_value = return_calculator(mock)
        method.return_value = return_value
        return

    side_effector = _side_effect_methods.get(name)
    if side_effector is not None:
        method.side_effect = side_effector(mock)



class MagicMixin(Base):
    def __init__(self, /, *args, **kw):
        self._mock_set_magics()  # make magic work for kwargs in init
        _safe_super(MagicMixin, self).__init__(*args, **kw)
        self._mock_set_magics()  # fix magic broken by upper level init


    def _mock_set_magics(self):
        orig_magics = _magics | _async_method_magics
        these_magics = orig_magics

        if getattr(self, "_mock_methods", None) is not None:
            these_magics = orig_magics.intersection(self._mock_methods)

            remove_magics = set()
            remove_magics = orig_magics - these_magics

            for entry in remove_magics:
                if entry in type(self).__dict__:
                    # remove unneeded magic methods
                    delattr(self, entry)

        # don't overwrite existing attributes if called a second time
        these_magics = these_magics - set(type(self).__dict__)

        _type = type(self)
        for entry in these_magics:
            setattr(_type, entry, MagicProxy(entry, self))



class NonCallableMagicMock(MagicMixin, NonCallableMock):
    """A version of `MagicMock` that isn't callable."""
    def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()


class AsyncMagicMixin(MagicMixin):
    def __init__(self, /, *args, **kw):
        self._mock_set_magics()  # make magic work for kwargs in init
        _safe_super(AsyncMagicMixin, self).__init__(*args, **kw)
        self._mock_set_magics()  # fix magic broken by upper level init

class MagicMock(MagicMixin, Mock):
    """
    MagicMock is a subclass of Mock with default implementations
    of most of the magic methods. You can use MagicMock without having to
    configure the magic methods yourself.

    If you use the `spec` or `spec_set` arguments then *only* magic
    methods that exist in the spec will be created.

    Attributes and the return value of a `MagicMock` will also be `MagicMocks`.
    """
    def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)
        self._mock_set_magics()



class MagicProxy(Base):
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def create_mock(self):
        entry = self.name
        parent = self.parent
        m = parent._get_child_mock(name=entry, _new_name=entry,
                                   _new_parent=parent)
        setattr(parent, entry, m)
        _set_return_value(parent, m, entry)
        return m

    def __get__(self, obj, _type=None):
        return self.create_mock()


class AsyncMockMixin(Base):
    await_count = _delegating_property('await_count')
    await_args = _delegating_property('await_args')
    await_args_list = _delegating_property('await_args_list')

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # iscoroutinefunction() checks _is_coroutine property to say if an
        # object is a coroutine. Without this check it looks to see if it is a
        # function/method, which in this case it is not (since it is an
        # AsyncMock).
        # It is set through __dict__ because when spec_set is True, this
        # attribute is likely undefined.
        self.__dict__['_is_coroutine'] = asyncio.coroutines._is_coroutine
        self.__dict__['_mock_await_count'] = 0
        self.__dict__['_mock_await_args'] = None
        self.__dict__['_mock_await_args_list'] = _CallList()
        code_mock = NonCallableMock(spec_set=CodeType)
        code_mock.co_flags = inspect.CO_COROUTINE
        self.__dict__['__code__'] = code_mock
        self.__dict__['__name__'] = 'AsyncMock'
        self.__dict__['__defaults__'] = tuple()
        self.__dict__['__kwdefaults__'] = {}
        self.__dict__['__annotations__'] = None

    async def _execute_mock_call(self, /, *args, **kwargs):
        # This is nearly just like super(), except for special handling
        # of coroutines

        _call = _Call((args, kwargs), two=True)
        self.await_count += 1
        self.await_args = _call
        self.await_args_list.append(_call)

        effect = self.side_effect
        if effect is not None:
            if _is_exception(effect):
                raise effect
            elif not _callable(effect):
                try:
                    result = next(effect)
                except StopIteration:
                    # It is impossible to propogate a StopIteration
                    # through coroutines because of PEP 479
                    raise StopAsyncIteration
                if _is_exception(result):
                    raise result
            elif iscoroutinefunction(effect):
                result = await effect(*args, **kwargs)
            else:
                result = effect(*args, **kwargs)

            if result is not DEFAULT:
                return result

        if self._mock_return_value is not DEFAULT:
            return self.return_value

        if self._mock_wraps is not None:
            if iscoroutinefunction(self._mock_wraps):
                return await self._mock_wraps(*args, **kwargs)
            return self._mock_wraps(*args, **kwargs)

        return self.return_value

    def assert_awaited(self):
        """
        Assert that the mock was awaited at least once.
        """
        if self.await_count == 0:
            msg = f"Expected {self._mock_name or 'mock'} to have been awaited."
            raise AssertionError(msg)

    def assert_awaited_once(self):
        """
        Assert that the mock was awaited exactly once.
        """
        if not self.await_count == 1:
            msg = (f"Expected {self._mock_name or 'mock'} to have been awaited once."
                   f" Awaited {self.await_count} times.")
            raise AssertionError(msg)

    def assert_awaited_with(self, /, *args, **kwargs):
        """
        Assert that the last await was with the specified arguments.
        """
        if self.await_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            raise AssertionError(f'Expected await: {expected}\nNot awaited')

        def _error_message():
            msg = self._format_mock_failure_message(args, kwargs, action='await')
            return msg

        expected = self._call_matcher(_Call((args, kwargs), two=True))
        actual = self._call_matcher(self.await_args)
        if actual != expected:
            cause = expected if isinstance(expected, Exception) else None
            raise AssertionError(_error_message()) from cause

    def assert_awaited_once_with(self, /, *args, **kwargs):
        """
        Assert that the mock was awaited exactly once and with the specified
        arguments.
        """
        if not self.await_count == 1:
            msg = (f"Expected {self._mock_name or 'mock'} to have been awaited once."
                   f" Awaited {self.await_count} times.")
            raise AssertionError(msg)
        return self.assert_awaited_with(*args, **kwargs)

    def assert_any_await(self, /, *args, **kwargs):
        """
        Assert the mock has ever been awaited with the specified arguments.
        """
        expected = self._call_matcher(_Call((args, kwargs), two=True))
        cause = expected if isinstance(expected, Exception) else None
        actual = [self._call_matcher(c) for c in self.await_args_list]
        if cause or expected not in _AnyComparer(actual):
            expected_string = self._format_mock_call_signature(args, kwargs)
            raise AssertionError(
                '%s await not found' % expected_string
            ) from cause

    def assert_has_awaits(self, calls, any_order=False):
        """
        Assert the mock has been awaited with the specified calls.
        The :attr:`await_args_list` list is checked for the awaits.

        If `any_order` is False (the default) then the awaits must be
        sequential. There can be extra calls before or after the
        specified awaits.

        If `any_order` is True then the awaits can be in any order, but
        they must all appear in :attr:`await_args_list`.
        """
        expected = [self._call_matcher(c) for c in calls]
        cause = next((e for e in expected if isinstance(e, Exception)), None)
        all_awaits = _CallList(self._call_matcher(c) for c in self.await_args_list)
        if not any_order:
            if expected not in all_awaits:
                if cause is None:
                    problem = 'Awaits not found.'
                else:
                    problem = ('Error processing expected awaits.\n'
                               'Errors: {}').format(
                                   [e if isinstance(e, Exception) else None
                                    for e in expected])
                raise AssertionError(
                    f'{problem}\n'
                    f'Expected: {_CallList(calls)}\n'
                    f'Actual: {self.await_args_list}'
                ) from cause
            return

        all_awaits = list(all_awaits)

        not_found = []
        for kall in expected:
            try:
                all_awaits.remove(kall)
            except ValueError:
                not_found.append(kall)
        if not_found:
            raise AssertionError(
                '%r not all found in await list' % (tuple(not_found),)
            ) from cause

    def assert_not_awaited(self):
        """
        Assert that the mock was never awaited.
        """
        if self.await_count != 0:
            msg = (f"Expected {self._mock_name or 'mock'} to not have been awaited."
