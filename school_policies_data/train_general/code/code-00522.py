                   f" Awaited {self.await_count} times.")
            raise AssertionError(msg)

    def reset_mock(self, /, *args, **kwargs):
        """
        See :func:`.Mock.reset_mock()`
        """
        super().reset_mock(*args, **kwargs)
        self.await_count = 0
        self.await_args = None
        self.await_args_list = _CallList()


class AsyncMock(AsyncMockMixin, AsyncMagicMixin, Mock):
    """
    Enhance :class:`Mock` with features allowing to mock
    an async function.

    The :class:`AsyncMock` object will behave so the object is
    recognized as an async function, and the result of a call is an awaitable:

    >>> mock = AsyncMock()
    >>> iscoroutinefunction(mock)
    True
    >>> inspect.isawaitable(mock())
    True


    The result of ``mock()`` is an async function which will have the outcome
    of ``side_effect`` or ``return_value``:

    - if ``side_effect`` is a function, the async function will return the
      result of that function,
    - if ``side_effect`` is an exception, the async function will raise the
      exception,
    - if ``side_effect`` is an iterable, the async function will return the
      next value of the iterable, however, if the sequence of result is
      exhausted, ``StopIteration`` is raised immediately,
    - if ``side_effect`` is not defined, the async function will return the
      value defined by ``return_value``, hence, by default, the async function
      returns a new :class:`AsyncMock` object.

    If the outcome of ``side_effect`` or ``return_value`` is an async function,
    the mock async function obtained when the mock object is called will be this
    async function itself (and not an async function returning an async
    function).

    The test author can also specify a wrapped object with ``wraps``. In this
    case, the :class:`Mock` object behavior is the same as with an
    :class:`.Mock` object: the wrapped object may have methods
    defined as async function functions.

    Based on Martin Richard's asynctest project.
    """


class _ANY(object):
    "A helper object that compares equal to everything."

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def __repr__(self):
        return '<ANY>'

ANY = _ANY()



def _format_call_signature(name, args, kwargs):
    message = '%s(%%s)' % name
    formatted_args = ''
    args_string = ', '.join([repr(arg) for arg in args])
    kwargs_string = ', '.join([
        '%s=%r' % (key, value) for key, value in kwargs.items()
    ])
    if args_string:
        formatted_args = args_string
    if kwargs_string:
        if formatted_args:
            formatted_args += ', '
        formatted_args += kwargs_string

    return message % formatted_args



class _Call(tuple):
    """
    A tuple for holding the results of a call to a mock, either in the form
    `(args, kwargs)` or `(name, args, kwargs)`.

    If args or kwargs are empty then a call tuple will compare equal to
    a tuple without those values. This makes comparisons less verbose::

        _Call(('name', (), {})) == ('name',)
        _Call(('name', (1,), {})) == ('name', (1,))
        _Call(((), {'a': 'b'})) == ({'a': 'b'},)

    The `_Call` object provides a useful shortcut for comparing with call::

        _Call(((1, 2), {'a': 3})) == call(1, 2, a=3)
        _Call(('foo', (1, 2), {'a': 3})) == call.foo(1, 2, a=3)

    If the _Call has no name then it will match any name.
    """
    def __new__(cls, value=(), name='', parent=None, two=False,
                from_kall=True):
        args = ()
        kwargs = {}
        _len = len(value)
        if _len == 3:
            name, args, kwargs = value
        elif _len == 2:
            first, second = value
            if isinstance(first, str):
                name = first
                if isinstance(second, tuple):
                    args = second
                else:
                    kwargs = second
            else:
                args, kwargs = first, second
        elif _len == 1:
            value, = value
            if isinstance(value, str):
                name = value
            elif isinstance(value, tuple):
                args = value
            else:
                kwargs = value

        if two:
            return tuple.__new__(cls, (args, kwargs))

        return tuple.__new__(cls, (name, args, kwargs))


    def __init__(self, value=(), name=None, parent=None, two=False,
                 from_kall=True):
        self._mock_name = name
        self._mock_parent = parent
        self._mock_from_kall = from_kall


    def __eq__(self, other):
        try:
            len_other = len(other)
        except TypeError:
            return NotImplemented

        self_name = ''
        if len(self) == 2:
            self_args, self_kwargs = self
        else:
            self_name, self_args, self_kwargs = self

        if (getattr(self, '_mock_parent', None) and getattr(other, '_mock_parent', None)
                and self._mock_parent != other._mock_parent):
            return False

        other_name = ''
        if len_other == 0:
            other_args, other_kwargs = (), {}
        elif len_other == 3:
            other_name, other_args, other_kwargs = other
        elif len_other == 1:
            value, = other
            if isinstance(value, tuple):
                other_args = value
                other_kwargs = {}
            elif isinstance(value, str):
                other_name = value
                other_args, other_kwargs = (), {}
            else:
                other_args = ()
                other_kwargs = value
        elif len_other == 2:
            # could be (name, args) or (name, kwargs) or (args, kwargs)
            first, second = other
            if isinstance(first, str):
                other_name = first
                if isinstance(second, tuple):
                    other_args, other_kwargs = second, {}
                else:
                    other_args, other_kwargs = (), second
            else:
                other_args, other_kwargs = first, second
        else:
            return False

        if self_name and other_name != self_name:
            return False

        # this order is important for ANY to work!
        return (other_args, other_kwargs) == (self_args, self_kwargs)


    __ne__ = object.__ne__


    def __call__(self, /, *args, **kwargs):
        if self._mock_name is None:
            return _Call(('', args, kwargs), name='()')

        name = self._mock_name + '()'
        return _Call((self._mock_name, args, kwargs), name=name, parent=self)


    def __getattr__(self, attr):
        if self._mock_name is None:
            return _Call(name=attr, from_kall=False)
        name = '%s.%s' % (self._mock_name, attr)
        return _Call(name=name, parent=self, from_kall=False)


    def __getattribute__(self, attr):
        if attr in tuple.__dict__:
            raise AttributeError
        return tuple.__getattribute__(self, attr)


    def _get_call_arguments(self):
        if len(self) == 2:
            args, kwargs = self
        else:
            name, args, kwargs = self

        return args, kwargs

    @property
    def args(self):
        return self._get_call_arguments()[0]

    @property
    def kwargs(self):
        return self._get_call_arguments()[1]

    def __repr__(self):
        if not self._mock_from_kall:
            name = self._mock_name or 'call'
            if name.startswith('()'):
                name = 'call%s' % name
            return name

        if len(self) == 2:
            name = 'call'
            args, kwargs = self
        else:
            name, args, kwargs = self
            if not name:
                name = 'call'
            elif not name.startswith('()'):
                name = 'call.%s' % name
            else:
                name = 'call%s' % name
        return _format_call_signature(name, args, kwargs)


    def call_list(self):
        """For a call object that represents multiple calls, `call_list`
        returns a list of all the intermediate calls as well as the
        final call."""
        vals = []
        thing = self
        while thing is not None:
            if thing._mock_from_kall:
                vals.append(thing)
            thing = thing._mock_parent
        return _CallList(reversed(vals))


call = _Call(from_kall=False)


def create_autospec(spec, spec_set=False, instance=False, _parent=None,
                    _name=None, *, unsafe=False, **kwargs):
    """Create a mock object using another object as a spec. Attributes on the
    mock will use the corresponding attribute on the `spec` object as their
    spec.

    Functions or methods being mocked will have their arguments checked
    to check that they are called with the correct signature.

    If `spec_set` is True then attempting to set attributes that don't exist
    on the spec object will raise an `AttributeError`.

    If a class is used as a spec then the return value of the mock (the
    instance of the class) will have the same spec. You can use a class as the
    spec for an instance object by passing `instance=True`. The returned mock
    will only be callable if instances of the mock are callable.

    `create_autospec` will raise a `RuntimeError` if passed some common
    misspellings of the arguments autospec and spec_set. Pass the argument
    `unsafe` with the value True to disable that check.

    `create_autospec` also takes arbitrary keyword arguments that are passed to
    the constructor of the created mock."""
    if _is_list(spec):
        # can't pass a list instance to the mock constructor as it will be
        # interpreted as a list of strings
        spec = type(spec)

    is_type = isinstance(spec, type)
    if _is_instance_mock(spec):
        raise InvalidSpecError(f'Cannot autospec a Mock object. '
                               f'[object={spec!r}]')
    is_async_func = _is_async_func(spec)
    _kwargs = {'spec': spec}
    if spec_set:
        _kwargs = {'spec_set': spec}
    elif spec is None:
        # None we mock with a normal mock without a spec
        _kwargs = {}
    if _kwargs and instance:
        _kwargs['_spec_as_instance'] = True
    if not unsafe:
        _check_spec_arg_typos(kwargs)

    _kwargs.update(kwargs)

    Klass = MagicMock
    if inspect.isdatadescriptor(spec):
        # descriptors don't have a spec
        # because we don't know what type they return
        _kwargs = {}
    elif is_async_func:
        if instance:
            raise RuntimeError("Instance can not be True when create_autospec "
                               "is mocking an async function")
        Klass = AsyncMock
    elif not _callable(spec):
        Klass = NonCallableMagicMock
    elif is_type and instance and not _instance_callable(spec):
        Klass = NonCallableMagicMock

    _name = _kwargs.pop('name', _name)

    _new_name = _name
    if _parent is None:
        # for a top level object no _new_name should be set
        _new_name = ''

    mock = Klass(parent=_parent, _new_parent=_parent, _new_name=_new_name,
                 name=_name, **_kwargs)

    if isinstance(spec, FunctionTypes):
        # should only happen at the top level because we don't
        # recurse for functions
        mock = _set_signature(mock, spec)
        if is_async_func:
            _setup_async_mock(mock)
    else:
        _check_signature(spec, mock, is_type, instance)

    if _parent is not None and not instance:
        _parent._mock_children[_name] = mock

    if is_type and not instance and 'return_value' not in kwargs:
        mock.return_value = create_autospec(spec, spec_set, instance=True,
                                            _name='()', _parent=mock)

    for entry in dir(spec):
        if _is_magic(entry):
            # MagicMock already does the useful magic methods for us
            continue

        # XXXX do we need a better way of getting attributes without
        # triggering code execution (?) Probably not - we need the actual
        # object to mock it so we would rather trigger a property than mock
        # the property descriptor. Likewise we want to mock out dynamically
        # provided attributes.
        # XXXX what about attributes that raise exceptions other than
        # AttributeError on being fetched?
        # we could be resilient against it, or catch and propagate the
        # exception when the attribute is fetched from the mock
        try:
            original = getattr(spec, entry)
        except AttributeError:
            continue

        kwargs = {'spec': original}
        if spec_set:
            kwargs = {'spec_set': original}

        if not isinstance(original, FunctionTypes):
            new = _SpecState(original, spec_set, mock, entry, instance)
            mock._mock_children[entry] = new
        else:
            parent = mock
            if isinstance(spec, FunctionTypes):
                parent = mock.mock

            skipfirst = _must_skip(spec, entry, is_type)
            kwargs['_eat_self'] = skipfirst
            if iscoroutinefunction(original):
                child_klass = AsyncMock
            else:
                child_klass = MagicMock
            new = child_klass(parent=parent, name=entry, _new_name=entry,
                              _new_parent=parent,
                              **kwargs)
            mock._mock_children[entry] = new
            _check_signature(original, new, skipfirst=skipfirst)

        # so functions created with _set_signature become instance attributes,
        # *plus* their underlying mock exists in _mock_children of the parent
        # mock. Adding to _mock_children may be unnecessary where we are also
        # setting as an instance attribute?
        if isinstance(new, FunctionTypes):
            setattr(mock, entry, new)

    return mock


def _must_skip(spec, entry, is_type):
    """
    Return whether we should skip the first argument on spec's `entry`
    attribute.
    """
    if not isinstance(spec, type):
        if entry in getattr(spec, '__dict__', {}):
            # instance attribute - shouldn't skip
            return False
        spec = spec.__class__

    for klass in spec.__mro__:
        result = klass.__dict__.get(entry, DEFAULT)
        if result is DEFAULT:
            continue
        if isinstance(result, (staticmethod, classmethod)):
            return False
        elif isinstance(result, FunctionTypes):
            # Normal method => skip if looked up on type
            # (if looked up on instance, self is already skipped)
            return is_type
        else:
            return False

    # function is a dynamically provided attribute
    return is_type


class _SpecState(object):

    def __init__(self, spec, spec_set=False, parent=None,
                 name=None, ids=None, instance=False):
        self.spec = spec
        self.ids = ids
        self.spec_set = spec_set
        self.parent = parent
        self.instance = instance
        self.name = name


FunctionTypes = (
    # python function
    type(create_autospec),
    # instance method
    type(ANY.__eq__),
)


file_spec = None


def _to_stream(read_data):
    if isinstance(read_data, bytes):
        return io.BytesIO(read_data)
    else:
        return io.StringIO(read_data)


def mock_open(mock=None, read_data=''):
    """
    A helper function to create a mock to replace the use of `open`. It works
    for `open` called directly or used as a context manager.

    The `mock` argument is the mock object to configure. If `None` (the
    default) then a `MagicMock` will be created for you, with the API limited
    to methods or attributes available on standard file handles.

    `read_data` is a string for the `read`, `readline` and `readlines` of the
    file handle to return.  This is an empty string by default.
    """
    _read_data = _to_stream(read_data)
    _state = [_read_data, None]

    def _readlines_side_effect(*args, **kwargs):
        if handle.readlines.return_value is not None:
            return handle.readlines.return_value
        return _state[0].readlines(*args, **kwargs)

    def _read_side_effect(*args, **kwargs):
        if handle.read.return_value is not None:
            return handle.read.return_value
        return _state[0].read(*args, **kwargs)

    def _readline_side_effect(*args, **kwargs):
        yield from _iter_side_effect()
        while True:
            yield _state[0].readline(*args, **kwargs)

    def _iter_side_effect():
        if handle.readline.return_value is not None:
            while True:
                yield handle.readline.return_value
        for line in _state[0]:
            yield line

    def _next_side_effect():
        if handle.readline.return_value is not None:
            return handle.readline.return_value
        return next(_state[0])

    global file_spec
    if file_spec is None:
        import _io
        file_spec = list(set(dir(_io.TextIOWrapper)).union(set(dir(_io.BytesIO))))

    if mock is None:
        mock = MagicMock(name='open', spec=open)

    handle = MagicMock(spec=file_spec)
    handle.__enter__.return_value = handle

    handle.write.return_value = None
    handle.read.return_value = None
    handle.readline.return_value = None
    handle.readlines.return_value = None

    handle.read.side_effect = _read_side_effect
    _state[1] = _readline_side_effect()
    handle.readline.side_effect = _state[1]
    handle.readlines.side_effect = _readlines_side_effect
    handle.__iter__.side_effect = _iter_side_effect
    handle.__next__.side_effect = _next_side_effect

    def reset_data(*args, **kwargs):
        _state[0] = _to_stream(read_data)
        if handle.readline.side_effect == _state[1]:
            # Only reset the side effect if the user hasn't overridden it.
            _state[1] = _readline_side_effect()
            handle.readline.side_effect = _state[1]
        return DEFAULT

    mock.side_effect = reset_data
    mock.return_value = handle
    return mock


class PropertyMock(Mock):
    """
    A mock intended to be used as a property, or other descriptor, on a class.
    `PropertyMock` provides `__get__` and `__set__` methods so you can specify
    a return value when it is fetched.

    Fetching a `PropertyMock` instance from an object calls the mock, with
    no args. Setting it calls the mock with the value being set.
    """
    def _get_child_mock(self, /, **kwargs):
        return MagicMock(**kwargs)

    def __get__(self, obj, obj_type=None):
        return self()
    def __set__(self, obj, val):
        self(val)


def seal(mock):
    """Disable the automatic generation of child mocks.

    Given an input Mock, seals it to ensure no further mocks will be generated
    when accessing an attribute that was not already defined.

    The operation recursively seals the mock passed in, meaning that
    the mock itself, any mocks generated by accessing one of its attributes,
    and all assigned mocks without a name or spec will be sealed.
    """
    mock._mock_sealed = True
    for attr in dir(mock):
        try:
            m = getattr(mock, attr)
        except AttributeError:
            continue
        if not isinstance(m, NonCallableMock):
            continue
        if isinstance(m._mock_children.get(attr), _SpecState):
            continue
        if m._mock_new_parent is mock:
            seal(m)


class _AsyncIterator:
    """
    Wraps an iterator in an asynchronous iterator.
    """
    def __init__(self, iterator):
        self.iterator = iterator
        code_mock = NonCallableMock(spec_set=CodeType)
        code_mock.co_flags = inspect.CO_ITERABLE_COROUTINE
        self.__dict__['__code__'] = code_mock

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            pass
        raise StopAsyncIteration
<|endoftext|>"""Main entry point"""

import sys
if sys.argv[0].endswith("__main__.py"):
    import os.path
    # We change sys.argv[0] to make help message more useful
    # use executable without path, unquoted
    # (it's just a hint anyway)
    # (if you have spaces in your executable you get what you deserve!)
    executable = os.path.basename(sys.executable)
    sys.argv[0] = executable + " -m unittest"
    del os

__unittest = True

from .main import main

main(module=None)
<|endoftext|>"""Test result object"""

import io
import sys
import traceback

from . import util
from functools import wraps

__unittest = True

def failfast(method):
    @wraps(method)
    def inner(self, *args, **kw):
        if getattr(self, 'failfast', False):
            self.stop()
        return method(self, *args, **kw)
    return inner

STDOUT_LINE = '\nStdout:\n%s'
STDERR_LINE = '\nStderr:\n%s'


class TestResult(object):
    """Holder for test result information.

    Test results are automatically managed by the TestCase and TestSuite
    classes, and do not need to be explicitly manipulated by writers of tests.

    Each instance holds the total number of tests run, and collections of
    failures and errors that occurred among those test runs. The collections
    contain tuples of (testcase, exceptioninfo), where exceptioninfo is the
    formatted traceback of the error that occurred.
    """
    _previousTestClass = None
    _testRunEntered = False
    _moduleSetUpFailed = False
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        self.failfast = False
        self.failures = []
        self.errors = []
        self.testsRun = 0
        self.skipped = []
        self.expectedFailures = []
        self.unexpectedSuccesses = []
        self.shouldStop = False
        self.buffer = False
        self.tb_locals = False
        self._stdout_buffer = None
        self._stderr_buffer = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._mirrorOutput = False

    def printErrors(self):
        "Called by TestRunner after test run"

    def startTest(self, test):
        "Called when the given test is about to be run"
        self.testsRun += 1
        self._mirrorOutput = False
        self._setupStdout()

    def _setupStdout(self):
        if self.buffer:
            if self._stderr_buffer is None:
                self._stderr_buffer = io.StringIO()
                self._stdout_buffer = io.StringIO()
            sys.stdout = self._stdout_buffer
            sys.stderr = self._stderr_buffer

    def startTestRun(self):
        """Called once before any tests are executed.

        See startTest for a method called before each test.
        """

    def stopTest(self, test):
        """Called when the given test has been run"""
        self._restoreStdout()
        self._mirrorOutput = False

    def _restoreStdout(self):
        if self.buffer:
            if self._mirrorOutput:
                output = sys.stdout.getvalue()
                error = sys.stderr.getvalue()
                if output:
                    if not output.endswith('\n'):
                        output += '\n'
                    self._original_stdout.write(STDOUT_LINE % output)
                if error:
                    if not error.endswith('\n'):
                        error += '\n'
                    self._original_stderr.write(STDERR_LINE % error)

            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._stdout_buffer.seek(0)
            self._stdout_buffer.truncate()
            self._stderr_buffer.seek(0)
            self._stderr_buffer.truncate()

    def stopTestRun(self):
        """Called once after all tests are executed.

        See stopTest for a method called after each test.
        """

    @failfast
    def addError(self, test, err):
        """Called when an error has occurred. 'err' is a tuple of values as
        returned by sys.exc_info().
        """
        self.errors.append((test, self._exc_info_to_string(err, test)))
        self._mirrorOutput = True

    @failfast
    def addFailure(self, test, err):
        """Called when an error has occurred. 'err' is a tuple of values as
        returned by sys.exc_info()."""
        self.failures.append((test, self._exc_info_to_string(err, test)))
        self._mirrorOutput = True

    def addSubTest(self, test, subtest, err):
        """Called at the end of a subtest.
        'err' is None if the subtest ended successfully, otherwise it's a
        tuple of values as returned by sys.exc_info().
        """
        # By default, we don't do anything with successful subtests, but
        # more sophisticated test results might want to record them.
        if err is not None:
            if getattr(self, 'failfast', False):
                self.stop()
            if issubclass(err[0], test.failureException):
                errors = self.failures
            else:
                errors = self.errors
            errors.append((subtest, self._exc_info_to_string(err, test)))
            self._mirrorOutput = True

    def addSuccess(self, test):
        "Called when a test has completed successfully"
        pass

    def addSkip(self, test, reason):
        """Called when a test is skipped."""
        self.skipped.append((test, reason))

    def addExpectedFailure(self, test, err):
        """Called when an expected failure/error occurred."""
        self.expectedFailures.append(
            (test, self._exc_info_to_string(err, test)))

    @failfast
    def addUnexpectedSuccess(self, test):
        """Called when a test was expected to fail, but succeed."""
        self.unexpectedSuccesses.append(test)

    def wasSuccessful(self):
        """Tells whether or not this result was a success."""
        # The hasattr check is for test_result's OldResult test.  That
        # way this method works on objects that lack the attribute.
        # (where would such result instances come from? old stored pickles?)
        return ((len(self.failures) == len(self.errors) == 0) and
                (not hasattr(self, 'unexpectedSuccesses') or
                 len(self.unexpectedSuccesses) == 0))

    def stop(self):
        """Indicates that the tests should be aborted."""
        self.shouldStop = True

    def _exc_info_to_string(self, err, test):
        """Converts a sys.exc_info()-style tuple of values into a string."""
        exctype, value, tb = err
        tb = self._clean_tracebacks(exctype, value, tb, test)
        tb_e = traceback.TracebackException(
            exctype, value, tb,
            capture_locals=self.tb_locals, compact=True)
        msgLines = list(tb_e.format())

        if self.buffer:
            output = sys.stdout.getvalue()
            error = sys.stderr.getvalue()
            if output:
                if not output.endswith('\n'):
                    output += '\n'
                msgLines.append(STDOUT_LINE % output)
            if error:
                if not error.endswith('\n'):
                    error += '\n'
                msgLines.append(STDERR_LINE % error)
        return ''.join(msgLines)

    def _clean_tracebacks(self, exctype, value, tb, test):
        ret = None
        first = True
        excs = [(exctype, value, tb)]
        seen = {id(value)}  # Detect loops in chained exceptions.
        while excs:
            (exctype, value, tb) = excs.pop()
            # Skip test runner traceback levels
            while tb and self._is_relevant_tb_level(tb):
                tb = tb.tb_next

            # Skip assert*() traceback levels
            if exctype is test.failureException:
                self._remove_unittest_tb_frames(tb)

            if first:
                ret = tb
                first = False
            else:
                value.__traceback__ = tb

            if value is not None:
                for c in (value.__cause__, value.__context__):
                    if c is not None and id(c) not in seen:
                        excs.append((type(c), c, c.__traceback__))
                        seen.add(id(c))
        return ret

    def _is_relevant_tb_level(self, tb):
        return '__unittest' in tb.tb_frame.f_globals

    def _remove_unittest_tb_frames(self, tb):
        '''Truncates usercode tb at the first unittest frame.

        If the first frame of the traceback is in user code,
        the prefix up to the first unittest frame is returned.
        If the first frame is already in the unittest module,
        the traceback is not modified.
        '''
        prev = None
        while tb and not self._is_relevant_tb_level(tb):
            prev = tb
            tb = tb.tb_next
        if prev is not None:
            prev.tb_next = None

    def __repr__(self):
        return ("<%s run=%i errors=%i failures=%i>" %
               (util.strclass(self.__class__), self.testsRun, len(self.errors),
                len(self.failures)))
<|endoftext|>import asyncio
import inspect

from .case import TestCase


class IsolatedAsyncioTestCase(TestCase):
    # Names intentionally have a long prefix
    # to reduce a chance of clashing with user-defined attributes
    # from inherited test case
    #
    # The class doesn't call loop.run_until_complete(self.setUp()) and family
    # but uses a different approach:
    # 1. create a long-running task that reads self.setUp()
    #    awaitable from queue along with a future
    # 2. await the awaitable object passing in and set the result
    #    into the future object
    # 3. Outer code puts the awaitable and the future object into a queue
    #    with waiting for the future
    # The trick is necessary because every run_until_complete() call
    # creates a new task with embedded ContextVar context.
    # To share contextvars between setUp(), test and tearDown() we need to execute
    # them inside the same task.

    # Note: the test case modifies event loop policy if the policy was not instantiated
    # yet.
    # asyncio.get_event_loop_policy() creates a default policy on demand but never
    # returns None
    # I believe this is not an issue in user level tests but python itself for testing
    # should reset a policy in every test module
    # by calling asyncio.set_event_loop_policy(None) in tearDownModule()

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self._asyncioTestLoop = None
        self._asyncioCallsQueue = None

    async def asyncSetUp(self):
        pass

    async def asyncTearDown(self):
        pass

    def addAsyncCleanup(self, func, /, *args, **kwargs):
        # A trivial trampoline to addCleanup()
        # the function exists because it has a different semantics
        # and signature:
        # addCleanup() accepts regular functions
        # but addAsyncCleanup() accepts coroutines
        #
        # We intentionally don't add inspect.iscoroutinefunction() check
        # for func argument because there is no way
        # to check for async function reliably:
        # 1. It can be "async def func()" itself
        # 2. Class can implement "async def __call__()" method
        # 3. Regular "def func()" that returns awaitable object
        self.addCleanup(*(func, *args), **kwargs)

    def _callSetUp(self):
        self.setUp()
        self._callAsync(self.asyncSetUp)

    def _callTestMethod(self, method):
        self._callMaybeAsync(method)

    def _callTearDown(self):
        self._callAsync(self.asyncTearDown)
        self.tearDown()

    def _callCleanup(self, function, *args, **kwargs):
        self._callMaybeAsync(function, *args, **kwargs)

    def _callAsync(self, func, /, *args, **kwargs):
        assert self._asyncioTestLoop is not None, 'asyncio test loop is not initialized'
        ret = func(*args, **kwargs)
        assert inspect.isawaitable(ret), f'{func!r} returned non-awaitable'
        fut = self._asyncioTestLoop.create_future()
        self._asyncioCallsQueue.put_nowait((fut, ret))
        return self._asyncioTestLoop.run_until_complete(fut)

    def _callMaybeAsync(self, func, /, *args, **kwargs):
        assert self._asyncioTestLoop is not None, 'asyncio test loop is not initialized'
        ret = func(*args, **kwargs)
        if inspect.isawaitable(ret):
            fut = self._asyncioTestLoop.create_future()
            self._asyncioCallsQueue.put_nowait((fut, ret))
            return self._asyncioTestLoop.run_until_complete(fut)
        else:
            return ret

    async def _asyncioLoopRunner(self, fut):
        self._asyncioCallsQueue = queue = asyncio.Queue()
        fut.set_result(None)
        while True:
            query = await queue.get()
            queue.task_done()
            if query is None:
                return
            fut, awaitable = query
            try:
                ret = await awaitable
                if not fut.cancelled():
                    fut.set_result(ret)
            except (SystemExit, KeyboardInterrupt):
                raise
            except (BaseException, asyncio.CancelledError) as ex:
                if not fut.cancelled():
                    fut.set_exception(ex)

    def _setupAsyncioLoop(self):
        assert self._asyncioTestLoop is None, 'asyncio test loop already initialized'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_debug(True)
        self._asyncioTestLoop = loop
        fut = loop.create_future()
        self._asyncioCallsTask = loop.create_task(self._asyncioLoopRunner(fut))
        loop.run_until_complete(fut)

    def _tearDownAsyncioLoop(self):
        assert self._asyncioTestLoop is not None, 'asyncio test loop is not initialized'
        loop = self._asyncioTestLoop
        self._asyncioTestLoop = None
        self._asyncioCallsQueue.put_nowait(None)
        loop.run_until_complete(self._asyncioCallsQueue.join())

        try:
            # cancel all tasks
            to_cancel = asyncio.all_tasks(loop)
            if not to_cancel:
                return

            for task in to_cancel:
                task.cancel()

            loop.run_until_complete(
                asyncio.gather(*to_cancel, return_exceptions=True))

            for task in to_cancel:
                if task.cancelled():
                    continue
                if task.exception() is not None:
                    loop.call_exception_handler({
                        'message': 'unhandled exception during test shutdown',
                        'exception': task.exception(),
                        'task': task,
                    })
            # shutdown asyncgens
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            # Prevent our executor environment from leaking to future tests.
            loop.run_until_complete(loop.shutdown_default_executor())
            asyncio.set_event_loop(None)
            loop.close()

    def run(self, result=None):
        self._setupAsyncioLoop()
        try:
            return super().run(result)
        finally:
            self._tearDownAsyncioLoop()

    def debug(self):
        self._setupAsyncioLoop()
        super().debug()
        self._tearDownAsyncioLoop()

    def __del__(self):
        if self._asyncioTestLoop is not None:
            self._tearDownAsyncioLoop()
<|endoftext|>"""Python version compatibility support for minidom.

This module contains internal implementation details and
should not be imported; use xml.dom.minidom instead.
"""

# This module should only be imported using "import *".
#
# The following names are defined:
#
#   NodeList      -- lightest possible NodeList implementation
#
#   EmptyNodeList -- lightest possible NodeList that is guaranteed to
#                    remain empty (immutable)
#
#   StringTypes   -- tuple of defined string types
#
#   defproperty   -- function used in conjunction with GetattrMagic;
#                    using these together is needed to make them work
#                    as efficiently as possible in both Python 2.2+
#                    and older versions.  For example:
#
#                        class MyClass(GetattrMagic):
#                            def _get_myattr(self):
#                                return something
#
#                        defproperty(MyClass, "myattr",
#                                    "return some value")
#
#                    For Python 2.2 and newer, this will construct a
#                    property object on the class, which avoids
#                    needing to override __getattr__().  It will only
#                    work for read-only attributes.
#
#                    For older versions of Python, inheriting from
#                    GetattrMagic will use the traditional
#                    __getattr__() hackery to achieve the same effect,
#                    but less efficiently.
#
#                    defproperty() should be used for each version of
#                    the relevant _get_<property>() function.

__all__ = ["NodeList", "EmptyNodeList", "StringTypes", "defproperty"]

import xml.dom

StringTypes = (str,)


class NodeList(list):
    __slots__ = ()

    def item(self, index):
        if 0 <= index < len(self):
            return self[index]

    def _get_length(self):
        return len(self)

    def _set_length(self, value):
        raise xml.dom.NoModificationAllowedErr(
            "attempt to modify read-only attribute 'length'")

    length = property(_get_length, _set_length,
                      doc="The number of nodes in the NodeList.")

    # For backward compatibility
    def __setstate__(self, state):
        if state is None:
            state = []
        self[:] = state


class EmptyNodeList(tuple):
    __slots__ = ()

    def __add__(self, other):
        NL = NodeList()
        NL.extend(other)
        return NL

    def __radd__(self, other):
        NL = NodeList()
        NL.extend(other)
        return NL

    def item(self, index):
        return None

    def _get_length(self):
        return 0

    def _set_length(self, value):
        raise xml.dom.NoModificationAllowedErr(
            "attempt to modify read-only attribute 'length'")

    length = property(_get_length, _set_length,
                      doc="The number of nodes in the NodeList.")


def defproperty(klass, name, doc):
    get = getattr(klass, ("_get_" + name))
    def set(self, value, name=name):
        raise xml.dom.NoModificationAllowedErr(
            "attempt to modify read-only attribute " + repr(name))
    assert not hasattr(klass, "_set_" + name), \
           "expected not to find _set_" + name
    prop = property(get, set, doc=doc)
    setattr(klass, name, prop)
<|endoftext|>"""Registration facilities for DOM. This module should not be used
directly. Instead, the functions getDOMImplementation and
registerDOMImplementation should be imported from xml.dom."""

# This is a list of well-known implementations.  Well-known names
# should be published by posting to xml-sig@python.org, and are
# subsequently recorded in this file.

import sys

well_known_implementations = {
    'minidom':'xml.dom.minidom',
    '4DOM': 'xml.dom.DOMImplementation',
    }

# DOM implementations not officially registered should register
# themselves with their

registered = {}

def registerDOMImplementation(name, factory):
    """registerDOMImplementation(name, factory)

    Register the factory function with the name. The factory function
    should return an object which implements the DOMImplementation
    interface. The factory function can either return the same object,
    or a new one (e.g. if that implementation supports some
    customization)."""

    registered[name] = factory

def _good_enough(dom, features):
    "_good_enough(dom, features) -> Return 1 if the dom offers the features"
    for f,v in features:
        if not dom.hasFeature(f,v):
            return 0
    return 1

def getDOMImplementation(name=None, features=()):
    """getDOMImplementation(name = None, features = ()) -> DOM implementation.

    Return a suitable DOM implementation. The name is either
    well-known, the module name of a DOM implementation, or None. If
    it is not None, imports the corresponding module and returns
    DOMImplementation object if the import succeeds.

    If name is not given, consider the available implementations to
    find one with the required feature set. If no implementation can
    be found, raise an ImportError. The features list must be a sequence
    of (feature, version) pairs which are passed to hasFeature."""

    import os
    creator = None
    mod = well_known_implementations.get(name)
    if mod:
        mod = __import__(mod, {}, {}, ['getDOMImplementation'])
        return mod.getDOMImplementation()
    elif name:
        return registered[name]()
    elif not sys.flags.ignore_environment and "PYTHON_DOM" in os.environ:
        return getDOMImplementation(name = os.environ["PYTHON_DOM"])

    # User did not specify a name, try implementations in arbitrary
    # order, returning the one that has the required features
    if isinstance(features, str):
        features = _parse_feature_string(features)
    for creator in registered.values():
        dom = creator()
        if _good_enough(dom, features):
            return dom

    for creator in well_known_implementations.keys():
        try:
            dom = getDOMImplementation(name = creator)
        except Exception: # typically ImportError, or AttributeError
            continue
        if _good_enough(dom, features):
            return dom

    raise ImportError("no suitable DOM implementation found")

def _parse_feature_string(s):
    features = []
    parts = s.split()
    i = 0
    length = len(parts)
    while i < length:
        feature = parts[i]
        if feature[0] in "0123456789":
            raise ValueError("bad feature name: %r" % (feature,))
        i = i + 1
        version = None
        if i < length:
            v = parts[i]
            if v[0] in "0123456789":
                i = i + 1
                version = v
        features.append((feature, version))
    return tuple(features)
<|endoftext|>"""Simple implementation of the Level 1 DOM.

Namespaces and other minor Level 2 features are also supported.

parse("foo.xml")

parseString("<foo><bar/></foo>")

Todo:
=====
 * convenience methods for getting elements and text.
 * more testing
 * bring some of the writer and linearizer code into conformance with this
        interface
 * SAX 2 namespaces
"""

import io
import xml.dom

from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS

# This is used by the ID-cache invalidation checks; the list isn't
# actually complete, since the nodes being checked will never be the
# DOCUMENT_NODE or DOCUMENT_FRAGMENT_NODE.  (The node being checked is
# the node being added or removed, not the node being modified.)
#
_nodeTypes_with_children = (xml.dom.Node.ELEMENT_NODE,
                            xml.dom.Node.ENTITY_REFERENCE_NODE)


class Node(xml.dom.Node):
    namespaceURI = None # this is non-null only for elements and attributes
    parentNode = None
    ownerDocument = None
    nextSibling = None
    previousSibling = None

    prefix = EMPTY_PREFIX # non-null only for NS elements and attributes

    def __bool__(self):
        return True

    def toxml(self, encoding=None, standalone=None):
        return self.toprettyxml("", "", encoding, standalone)

    def toprettyxml(self, indent="\t", newl="\n", encoding=None,
                    standalone=None):
        if encoding is None:
            writer = io.StringIO()
        else:
            writer = io.TextIOWrapper(io.BytesIO(),
                                      encoding=encoding,
                                      errors="xmlcharrefreplace",
                                      newline='\n')
        if self.nodeType == Node.DOCUMENT_NODE:
            # Can pass encoding only to document, to put it into XML header
            self.writexml(writer, "", indent, newl, encoding, standalone)
        else:
            self.writexml(writer, "", indent, newl)
        if encoding is None:
            return writer.getvalue()
        else:
            return writer.detach().getvalue()

    def hasChildNodes(self):
        return bool(self.childNodes)

    def _get_childNodes(self):
        return self.childNodes

    def _get_firstChild(self):
        if self.childNodes:
            return self.childNodes[0]

    def _get_lastChild(self):
        if self.childNodes:
            return self.childNodes[-1]

    def insertBefore(self, newChild, refChild):
        if newChild.nodeType == self.DOCUMENT_FRAGMENT_NODE:
            for c in tuple(newChild.childNodes):
                self.insertBefore(c, refChild)
            ### The DOM does not clearly specify what to return in this case
            return newChild
        if newChild.nodeType not in self._child_node_types:
            raise xml.dom.HierarchyRequestErr(
                "%s cannot be child of %s" % (repr(newChild), repr(self)))
        if newChild.parentNode is not None:
            newChild.parentNode.removeChild(newChild)
        if refChild is None:
            self.appendChild(newChild)
        else:
            try:
                index = self.childNodes.index(refChild)
            except ValueError:
                raise xml.dom.NotFoundErr()
            if newChild.nodeType in _nodeTypes_with_children:
                _clear_id_cache(self)
            self.childNodes.insert(index, newChild)
            newChild.nextSibling = refChild
            refChild.previousSibling = newChild
            if index:
                node = self.childNodes[index-1]
                node.nextSibling = newChild
                newChild.previousSibling = node
            else:
                newChild.previousSibling = None
            newChild.parentNode = self
        return newChild

    def appendChild(self, node):
        if node.nodeType == self.DOCUMENT_FRAGMENT_NODE:
            for c in tuple(node.childNodes):
                self.appendChild(c)
            ### The DOM does not clearly specify what to return in this case
            return node
        if node.nodeType not in self._child_node_types:
            raise xml.dom.HierarchyRequestErr(
                "%s cannot be child of %s" % (repr(node), repr(self)))
        elif node.nodeType in _nodeTypes_with_children:
            _clear_id_cache(self)
        if node.parentNode is not None:
            node.parentNode.removeChild(node)
        _append_child(self, node)
        node.nextSibling = None
        return node

    def replaceChild(self, newChild, oldChild):
        if newChild.nodeType == self.DOCUMENT_FRAGMENT_NODE:
            refChild = oldChild.nextSibling
            self.removeChild(oldChild)
            return self.insertBefore(newChild, refChild)
        if newChild.nodeType not in self._child_node_types:
            raise xml.dom.HierarchyRequestErr(
                "%s cannot be child of %s" % (repr(newChild), repr(self)))
        if newChild is oldChild:
            return
        if newChild.parentNode is not None:
            newChild.parentNode.removeChild(newChild)
        try:
            index = self.childNodes.index(oldChild)
        except ValueError:
            raise xml.dom.NotFoundErr()
        self.childNodes[index] = newChild
        newChild.parentNode = self
        oldChild.parentNode = None
        if (newChild.nodeType in _nodeTypes_with_children
            or oldChild.nodeType in _nodeTypes_with_children):
            _clear_id_cache(self)
        newChild.nextSibling = oldChild.nextSibling
        newChild.previousSibling = oldChild.previousSibling
        oldChild.nextSibling = None
        oldChild.previousSibling = None
        if newChild.previousSibling:
            newChild.previousSibling.nextSibling = newChild
        if newChild.nextSibling:
            newChild.nextSibling.previousSibling = newChild
        return oldChild

    def removeChild(self, oldChild):
        try:
            self.childNodes.remove(oldChild)
        except ValueError:
            raise xml.dom.NotFoundErr()
        if oldChild.nextSibling is not None:
            oldChild.nextSibling.previousSibling = oldChild.previousSibling
        if oldChild.previousSibling is not None:
            oldChild.previousSibling.nextSibling = oldChild.nextSibling
        oldChild.nextSibling = oldChild.previousSibling = None
        if oldChild.nodeType in _nodeTypes_with_children:
            _clear_id_cache(self)

        oldChild.parentNode = None
        return oldChild

    def normalize(self):
        L = []
        for child in self.childNodes:
            if child.nodeType == Node.TEXT_NODE:
                if not child.data:
                    # empty text node; discard
                    if L:
                        L[-1].nextSibling = child.nextSibling
                    if child.nextSibling:
                        child.nextSibling.previousSibling = child.previousSibling
                    child.unlink()
                elif L and L[-1].nodeType == child.nodeType:
                    # collapse text node
                    node = L[-1]
                    node.data = node.data + child.data
                    node.nextSibling = child.nextSibling
                    if child.nextSibling:
                        child.nextSibling.previousSibling = node
                    child.unlink()
                else:
                    L.append(child)
            else:
                L.append(child)
                if child.nodeType == Node.ELEMENT_NODE:
                    child.normalize()
        self.childNodes[:] = L

    def cloneNode(self, deep):
        return _clone_node(self, deep, self.ownerDocument or self)

    def isSupported(self, feature, version):
        return self.ownerDocument.implementation.hasFeature(feature, version)

    def _get_localName(self):
        # Overridden in Element and Attr where localName can be Non-Null
        return None

    # Node interfaces from Level 3 (WD 9 April 2002)

    def isSameNode(self, other):
        return self is other

    def getInterface(self, feature):
        if self.isSupported(feature, None):
            return self
        else:
            return None

    # The "user data" functions use a dictionary that is only present
    # if some user data has been set, so be careful not to assume it
    # exists.

    def getUserData(self, key):
        try:
            return self._user_data[key][0]
        except (AttributeError, KeyError):
            return None

    def setUserData(self, key, data, handler):
        old = None
        try:
            d = self._user_data
        except AttributeError:
            d = {}
            self._user_data = d
        if key in d:
            old = d[key][0]
        if data is None:
            # ignore handlers passed for None
            handler = None
            if old is not None:
                del d[key]
        else:
            d[key] = (data, handler)
        return old

    def _call_user_data_handler(self, operation, src, dst):
        if hasattr(self, "_user_data"):
            for key, (data, handler) in list(self._user_data.items()):
                if handler is not None:
                    handler.handle(operation, key, data, src, dst)

    # minidom-specific API:

    def unlink(self):
        self.parentNode = self.ownerDocument = None
        if self.childNodes:
            for child in self.childNodes:
                child.unlink()
            self.childNodes = NodeList()
        self.previousSibling = None
        self.nextSibling = None

    # A Node is its own context manager, to ensure that an unlink() call occurs.
    # This is similar to how a file object works.
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        self.unlink()

defproperty(Node, "firstChild", doc="First child node, or None.")
defproperty(Node, "lastChild",  doc="Last child node, or None.")
defproperty(Node, "localName",  doc="Namespace-local name of this node.")


def _append_child(self, node):
    # fast path with less checks; usable by DOM builders if careful
    childNodes = self.childNodes
    if childNodes:
        last = childNodes[-1]
        node.previousSibling = last
        last.nextSibling = node
    childNodes.append(node)
    node.parentNode = self


def _write_data(writer, data):
    "Writes datachars to writer."
    if data:
        data = data.replace("&", "&amp;").replace("<", "&lt;"). \
                    replace("\"", "&quot;").replace(">", "&gt;")
        writer.write(data)

def _get_elements_by_tagName_helper(parent, name, rc):
    for node in parent.childNodes:
        if node.nodeType == Node.ELEMENT_NODE and \
            (name == "*" or node.tagName == name):
            rc.append(node)
        _get_elements_by_tagName_helper(node, name, rc)
    return rc

def _get_elements_by_tagName_ns_helper(parent, nsURI, localName, rc):
    for node in parent.childNodes:
        if node.nodeType == Node.ELEMENT_NODE:
            if ((localName == "*" or node.localName == localName) and
                (nsURI == "*" or node.namespaceURI == nsURI)):
                rc.append(node)
            _get_elements_by_tagName_ns_helper(node, nsURI, localName, rc)
    return rc

class DocumentFragment(Node):
    nodeType = Node.DOCUMENT_FRAGMENT_NODE
    nodeName = "#document-fragment"
    nodeValue = None
    attributes = None
    parentNode = None
    _child_node_types = (Node.ELEMENT_NODE,
                         Node.TEXT_NODE,
                         Node.CDATA_SECTION_NODE,
                         Node.ENTITY_REFERENCE_NODE,
                         Node.PROCESSING_INSTRUCTION_NODE,
                         Node.COMMENT_NODE,
                         Node.NOTATION_NODE)

    def __init__(self):
        self.childNodes = NodeList()


class Attr(Node):
    __slots__=('_name', '_value', 'namespaceURI',
               '_prefix', 'childNodes', '_localName', 'ownerDocument', 'ownerElement')
    nodeType = Node.ATTRIBUTE_NODE
    attributes = None
    specified = False
    _is_id = False

    _child_node_types = (Node.TEXT_NODE, Node.ENTITY_REFERENCE_NODE)

    def __init__(self, qName, namespaceURI=EMPTY_NAMESPACE, localName=None,
                 prefix=None):
        self.ownerElement = None
        self.ownerDocument = None
        self._name = qName
        self.namespaceURI = namespaceURI
        self._prefix = prefix
        if localName is not None:
            self._localName = localName
        self.childNodes = NodeList()

        # Add the single child node that represents the value of the attr
        self.childNodes.append(Text())

        # nodeValue and value are set elsewhere

    def _get_localName(self):
        try:
            return self._localName
        except AttributeError:
            return self.nodeName.split(":", 1)[-1]

    def _get_specified(self):
        return self.specified

    def _get_name(self):
        return self._name

    def _set_name(self, value):
        self._name = value
        if self.ownerElement is not None:
            _clear_id_cache(self.ownerElement)

    nodeName = name = property(_get_name, _set_name)

    def _get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = value
        self.childNodes[0].data = value
        if self.ownerElement is not None:
            _clear_id_cache(self.ownerElement)
        self.childNodes[0].data = value

    nodeValue = value = property(_get_value, _set_value)

    def _get_prefix(self):
        return self._prefix

    def _set_prefix(self, prefix):
        nsuri = self.namespaceURI
        if prefix == "xmlns":
            if nsuri and nsuri != XMLNS_NAMESPACE:
                raise xml.dom.NamespaceErr(
                    "illegal use of 'xmlns' prefix for the wrong namespace")
        self._prefix = prefix
        if prefix is None:
            newName = self.localName
        else:
            newName = "%s:%s" % (prefix, self.localName)
        if self.ownerElement:
            _clear_id_cache(self.ownerElement)
        self.name = newName

    prefix = property(_get_prefix, _set_prefix)

    def unlink(self):
        # This implementation does not call the base implementation
        # since most of that is not needed, and the expense of the
        # method call is not warranted.  We duplicate the removal of
        # children, but that's all we needed from the base class.
        elem = self.ownerElement
        if elem is not None:
            del elem._attrs[self.nodeName]
            del elem._attrsNS[(self.namespaceURI, self.localName)]
            if self._is_id:
                self._is_id = False
                elem._magic_id_nodes -= 1
                self.ownerDocument._magic_id_count -= 1
        for child in self.childNodes:
            child.unlink()
        del self.childNodes[:]

    def _get_isId(self):
        if self._is_id:
            return True
        doc = self.ownerDocument
        elem = self.ownerElement
        if doc is None or elem is None:
            return False

        info = doc._get_elem_info(elem)
        if info is None:
            return False
        if self.namespaceURI:
            return info.isIdNS(self.namespaceURI, self.localName)
        else:
            return info.isId(self.nodeName)

    def _get_schemaType(self):
        doc = self.ownerDocument
        elem = self.ownerElement
        if doc is None or elem is None:
            return _no_type

        info = doc._get_elem_info(elem)
        if info is None:
            return _no_type
        if self.namespaceURI:
            return info.getAttributeTypeNS(self.namespaceURI, self.localName)
        else:
            return info.getAttributeType(self.nodeName)

defproperty(Attr, "isId",       doc="True if this attribute is an ID.")
defproperty(Attr, "localName",  doc="Namespace-local name of this attribute.")
defproperty(Attr, "schemaType", doc="Schema type for this attribute.")


class NamedNodeMap(object):
    """The attribute list is a transient interface to the underlying
    dictionaries.  Mutations here will change the underlying element's
    dictionary.

    Ordering is imposed artificially and does not reflect the order of
    attributes as found in an input document.
    """

    __slots__ = ('_attrs', '_attrsNS', '_ownerElement')

    def __init__(self, attrs, attrsNS, ownerElement):
        self._attrs = attrs
        self._attrsNS = attrsNS
        self._ownerElement = ownerElement

    def _get_length(self):
        return len(self._attrs)

    def item(self, index):
        try:
            return self[list(self._attrs.keys())[index]]
        except IndexError:
            return None

    def items(self):
        L = []
        for node in self._attrs.values():
            L.append((node.nodeName, node.value))
        return L

    def itemsNS(self):
        L = []
        for node in self._attrs.values():
            L.append(((node.namespaceURI, node.localName), node.value))
        return L

    def __contains__(self, key):
        if isinstance(key, str):
            return key in self._attrs
        else:
            return key in self._attrsNS

    def keys(self):
        return self._attrs.keys()

    def keysNS(self):
        return self._attrsNS.keys()

    def values(self):
        return self._attrs.values()

    def get(self, name, value=None):
        return self._attrs.get(name, value)

    __len__ = _get_length

    def _cmp(self, other):
        if self._attrs is getattr(other, "_attrs", None):
            return 0
        else:
            return (id(self) > id(other)) - (id(self) < id(other))

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __getitem__(self, attname_or_tuple):
        if isinstance(attname_or_tuple, tuple):
            return self._attrsNS[attname_or_tuple]
        else:
            return self._attrs[attname_or_tuple]

    # same as set
    def __setitem__(self, attname, value):
        if isinstance(value, str):
            try:
                node = self._attrs[attname]
            except KeyError:
                node = Attr(attname)
                node.ownerDocument = self._ownerElement.ownerDocument
                self.setNamedItem(node)
            node.value = value
        else:
            if not isinstance(value, Attr):
                raise TypeError("value must be a string or Attr object")
            node = value
            self.setNamedItem(node)

    def getNamedItem(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            return None

    def getNamedItemNS(self, namespaceURI, localName):
        try:
            return self._attrsNS[(namespaceURI, localName)]
        except KeyError:
            return None

    def removeNamedItem(self, name):
        n = self.getNamedItem(name)
        if n is not None:
            _clear_id_cache(self._ownerElement)
            del self._attrs[n.nodeName]
            del self._attrsNS[(n.namespaceURI, n.localName)]
            if hasattr(n, 'ownerElement'):
                n.ownerElement = None
            return n
        else:
            raise xml.dom.NotFoundErr()

    def removeNamedItemNS(self, namespaceURI, localName):
        n = self.getNamedItemNS(namespaceURI, localName)
        if n is not None:
            _clear_id_cache(self._ownerElement)
            del self._attrsNS[(n.namespaceURI, n.localName)]
            del self._attrs[n.nodeName]
            if hasattr(n, 'ownerElement'):
                n.ownerElement = None
            return n
        else:
            raise xml.dom.NotFoundErr()

    def setNamedItem(self, node):
        if not isinstance(node, Attr):
            raise xml.dom.HierarchyRequestErr(
                "%s cannot be child of %s" % (repr(node), repr(self)))
        old = self._attrs.get(node.name)
        if old:
            old.unlink()
        self._attrs[node.name] = node
        self._attrsNS[(node.namespaceURI, node.localName)] = node
        node.ownerElement = self._ownerElement
        _clear_id_cache(node.ownerElement)
        return old

    def setNamedItemNS(self, node):
        return self.setNamedItem(node)

    def __delitem__(self, attname_or_tuple):
        node = self[attname_or_tuple]
        _clear_id_cache(node.ownerElement)
        node.unlink()

    def __getstate__(self):
        return self._attrs, self._attrsNS, self._ownerElement

    def __setstate__(self, state):
        self._attrs, self._attrsNS, self._ownerElement = state

defproperty(NamedNodeMap, "length",
            doc="Number of nodes in the NamedNodeMap.")

AttributeList = NamedNodeMap


class TypeInfo(object):
    __slots__ = 'namespace', 'name'

    def __init__(self, namespace, name):
        self.namespace = namespace
        self.name = name

    def __repr__(self):
        if self.namespace:
            return "<%s %r (from %r)>" % (self.__class__.__name__, self.name,
                                          self.namespace)
        else:
            return "<%s %r>" % (self.__class__.__name__, self.name)

    def _get_name(self):
        return self.name

    def _get_namespace(self):
        return self.namespace

_no_type = TypeInfo(None, None)

class Element(Node):
    __slots__=('ownerDocument', 'parentNode', 'tagName', 'nodeName', 'prefix',
               'namespaceURI', '_localName', 'childNodes', '_attrs', '_attrsNS',
               'nextSibling', 'previousSibling')
    nodeType = Node.ELEMENT_NODE
    nodeValue = None
    schemaType = _no_type

    _magic_id_nodes = 0

    _child_node_types = (Node.ELEMENT_NODE,
                         Node.PROCESSING_INSTRUCTION_NODE,
                         Node.COMMENT_NODE,
                         Node.TEXT_NODE,
                         Node.CDATA_SECTION_NODE,
                         Node.ENTITY_REFERENCE_NODE)

    def __init__(self, tagName, namespaceURI=EMPTY_NAMESPACE, prefix=None,
                 localName=None):
        self.ownerDocument = None
        self.parentNode = None
        self.tagName = self.nodeName = tagName
        self.prefix = prefix
        self.namespaceURI = namespaceURI
        self.childNodes = NodeList()
        self.nextSibling = self.previousSibling = None

        # Attribute dictionaries are lazily created
        # attributes are double-indexed:
        #    tagName -> Attribute
        #    URI,localName -> Attribute
        # in the future: consider lazy generation
        # of attribute objects this is too tricky
        # for now because of headaches with
        # namespaces.
        self._attrs = None
        self._attrsNS = None

    def _ensure_attributes(self):
        if self._attrs is None:
            self._attrs = {}
            self._attrsNS = {}

    def _get_localName(self):
        try:
            return self._localName
        except AttributeError:
            return self.tagName.split(":", 1)[-1]

    def _get_tagName(self):
        return self.tagName

    def unlink(self):
        if self._attrs is not None:
            for attr in list(self._attrs.values()):
                attr.unlink()
        self._attrs = None
        self._attrsNS = None
        Node.unlink(self)

    def getAttribute(self, attname):
        """Returns the value of the specified attribute.

        Returns the value of the element's attribute named attname as
        a string. An empty string is returned if the element does not
        have such an attribute. Note that an empty string may also be
        returned as an explicitly given attribute value, use the
        hasAttribute method to distinguish these two cases.
        """
        if self._attrs is None:
            return ""
        try:
            return self._attrs[attname].value
        except KeyError:
            return ""

    def getAttributeNS(self, namespaceURI, localName):
        if self._attrsNS is None:
            return ""
        try:
            return self._attrsNS[(namespaceURI, localName)].value
        except KeyError:
            return ""

    def setAttribute(self, attname, value):
        attr = self.getAttributeNode(attname)
        if attr is None:
            attr = Attr(attname)
            attr.value = value # also sets nodeValue
            attr.ownerDocument = self.ownerDocument
            self.setAttributeNode(attr)
        elif value != attr.value:
            attr.value = value
            if attr.isId:
                _clear_id_cache(self)

    def setAttributeNS(self, namespaceURI, qualifiedName, value):
        prefix, localname = _nssplit(qualifiedName)
        attr = self.getAttributeNodeNS(namespaceURI, localname)
        if attr is None:
            attr = Attr(qualifiedName, namespaceURI, localname, prefix)
            attr.value = value
            attr.ownerDocument = self.ownerDocument
            self.setAttributeNode(attr)
        else:
            if value != attr.value:
                attr.value = value
                if attr.isId:
                    _clear_id_cache(self)
            if attr.prefix != prefix:
                attr.prefix = prefix
                attr.nodeName = qualifiedName

    def getAttributeNode(self, attrname):
        if self._attrs is None:
            return None
        return self._attrs.get(attrname)

    def getAttributeNodeNS(self, namespaceURI, localName):
        if self._attrsNS is None:
            return None
        return self._attrsNS.get((namespaceURI, localName))

    def setAttributeNode(self, attr):
        if attr.ownerElement not in (None, self):
            raise xml.dom.InuseAttributeErr("attribute node already owned")
        self._ensure_attributes()
        old1 = self._attrs.get(attr.name, None)
        if old1 is not None:
            self.removeAttributeNode(old1)
        old2 = self._attrsNS.get((attr.namespaceURI, attr.localName), None)
        if old2 is not None and old2 is not old1:
            self.removeAttributeNode(old2)
        _set_attribute_node(self, attr)

        if old1 is not attr:
            # It might have already been part of this node, in which case
            # it doesn't represent a change, and should not be returned.
            return old1
        if old2 is not attr:
            return old2

    setAttributeNodeNS = setAttributeNode

    def removeAttribute(self, name):
        if self._attrsNS is None:
            raise xml.dom.NotFoundErr()
        try:
            attr = self._attrs[name]
        except KeyError:
            raise xml.dom.NotFoundErr()
        self.removeAttributeNode(attr)

    def removeAttributeNS(self, namespaceURI, localName):
        if self._attrsNS is None:
            raise xml.dom.NotFoundErr()
        try:
            attr = self._attrsNS[(namespaceURI, localName)]
        except KeyError:
            raise xml.dom.NotFoundErr()
        self.removeAttributeNode(attr)

    def removeAttributeNode(self, node):
        if node is None:
            raise xml.dom.NotFoundErr()
        try:
            self._attrs[node.name]
        except KeyError:
            raise xml.dom.NotFoundErr()
        _clear_id_cache(self)
        node.unlink()
        # Restore this since the node is still useful and otherwise
        # unlinked
        node.ownerDocument = self.ownerDocument
        return node

    removeAttributeNodeNS = removeAttributeNode

    def hasAttribute(self, name):
        """Checks whether the element has an attribute with the specified name.

        Returns True if the element has an attribute with the specified name.
        Otherwise, returns False.
        """
        if self._attrs is None:
            return False
        return name in self._attrs

    def hasAttributeNS(self, namespaceURI, localName):
        if self._attrsNS is None:
            return False
        return (namespaceURI, localName) in self._attrsNS

    def getElementsByTagName(self, name):
        """Returns all descendant elements with the given tag name.

        Returns the list of all descendant elements (not direct children
        only) with the specified tag name.
        """
        return _get_elements_by_tagName_helper(self, name, NodeList())

    def getElementsByTagNameNS(self, namespaceURI, localName):
        return _get_elements_by_tagName_ns_helper(
            self, namespaceURI, localName, NodeList())

    def __repr__(self):
        return "<DOM Element: %s at %#x>" % (self.tagName, id(self))

    def writexml(self, writer, indent="", addindent="", newl=""):
        """Write an XML element to a file-like object

        Write the element to the writer object that must provide
        a write method (e.g. a file or StringIO object).
        """
        # indent = current indentation
        # addindent = indentation to add to higher levels
        # newl = newline string
        writer.write(indent+"<" + self.tagName)

        attrs = self._get_attributes()

        for a_name in attrs.keys():
            writer.write(" %s=\"" % a_name)
            _write_data(writer, attrs[a_name].value)
            writer.write("\"")
        if self.childNodes:
            writer.write(">")
            if (len(self.childNodes) == 1 and
                self.childNodes[0].nodeType in (
                        Node.TEXT_NODE, Node.CDATA_SECTION_NODE)):
                self.childNodes[0].writexml(writer, '', '', '')
            else:
                writer.write(newl)
                for node in self.childNodes:
                    node.writexml(writer, indent+addindent, addindent, newl)
                writer.write(indent)
            writer.write("</%s>%s" % (self.tagName, newl))
        else:
            writer.write("/>%s"%(newl))

    def _get_attributes(self):
        self._ensure_attributes()
        return NamedNodeMap(self._attrs, self._attrsNS, self)

    def hasAttributes(self):
        if self._attrs:
            return True
        else:
            return False

    # DOM Level 3 attributes, based on the 22 Oct 2002 draft

    def setIdAttribute(self, name):
        idAttr = self.getAttributeNode(name)
        self.setIdAttributeNode(idAttr)

    def setIdAttributeNS(self, namespaceURI, localName):
        idAttr = self.getAttributeNodeNS(namespaceURI, localName)
        self.setIdAttributeNode(idAttr)

    def setIdAttributeNode(self, idAttr):
        if idAttr is None or not self.isSameNode(idAttr.ownerElement):
            raise xml.dom.NotFoundErr()
        if _get_containing_entref(self) is not None:
            raise xml.dom.NoModificationAllowedErr()
        if not idAttr._is_id:
            idAttr._is_id = True
            self._magic_id_nodes += 1
            self.ownerDocument._magic_id_count += 1
            _clear_id_cache(self)

defproperty(Element, "attributes",
            doc="NamedNodeMap of attributes on the element.")
defproperty(Element, "localName",
            doc="Namespace-local name of this element.")


def _set_attribute_node(element, attr):
    _clear_id_cache(element)
    element._ensure_attributes()
    element._attrs[attr.name] = attr
    element._attrsNS[(attr.namespaceURI, attr.localName)] = attr

    # This creates a circular reference, but Element.unlink()
    # breaks the cycle since the references to the attribute
    # dictionaries are tossed.
    attr.ownerElement = element

class Childless:
    """Mixin that makes childless-ness easy to implement and avoids
    the complexity of the Node methods that deal with children.
    """
    __slots__ = ()

    attributes = None
    childNodes = EmptyNodeList()
    firstChild = None
    lastChild = None

    def _get_firstChild(self):
        return None

    def _get_lastChild(self):
        return None

    def appendChild(self, node):
        raise xml.dom.HierarchyRequestErr(
            self.nodeName + " nodes cannot have children")

    def hasChildNodes(self):
        return False

    def insertBefore(self, newChild, refChild):
        raise xml.dom.HierarchyRequestErr(
            self.nodeName + " nodes do not have children")

    def removeChild(self, oldChild):
        raise xml.dom.NotFoundErr(
            self.nodeName + " nodes do not have children")

    def normalize(self):
        # For childless nodes, normalize() has nothing to do.
        pass

    def replaceChild(self, newChild, oldChild):
        raise xml.dom.HierarchyRequestErr(
            self.nodeName + " nodes do not have children")


class ProcessingInstruction(Childless, Node):
    nodeType = Node.PROCESSING_INSTRUCTION_NODE
    __slots__ = ('target', 'data')

    def __init__(self, target, data):
        self.target = target
        self.data = data

    # nodeValue is an alias for data
    def _get_nodeValue(self):
        return self.data
    def _set_nodeValue(self, value):
        self.data = value
    nodeValue = property(_get_nodeValue, _set_nodeValue)

    # nodeName is an alias for target
    def _get_nodeName(self):
        return self.target
    def _set_nodeName(self, value):
        self.target = value
    nodeName = property(_get_nodeName, _set_nodeName)

    def writexml(self, writer, indent="", addindent="", newl=""):
        writer.write("%s<?%s %s?>%s" % (indent,self.target, self.data, newl))


class CharacterData(Childless, Node):
    __slots__=('_data', 'ownerDocument','parentNode', 'previousSibling', 'nextSibling')

    def __init__(self):
        self.ownerDocument = self.parentNode = None
        self.previousSibling = self.nextSibling = None
        self._data = ''
        Node.__init__(self)

    def _get_length(self):
        return len(self.data)
    __len__ = _get_length

    def _get_data(self):
        return self._data
    def _set_data(self, data):
        self._data = data

    data = nodeValue = property(_get_data, _set_data)

    def __repr__(self):
        data = self.data
        if len(data) > 10:
            dotdotdot = "..."
        else:
            dotdotdot = ""
        return '<DOM %s node "%r%s">' % (
            self.__class__.__name__, data[0:10], dotdotdot)

    def substringData(self, offset, count):
        if offset < 0:
            raise xml.dom.IndexSizeErr("offset cannot be negative")
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr("offset cannot be beyond end of data")
        if count < 0:
            raise xml.dom.IndexSizeErr("count cannot be negative")
        return self.data[offset:offset+count]

    def appendData(self, arg):
        self.data = self.data + arg

    def insertData(self, offset, arg):
        if offset < 0:
            raise xml.dom.IndexSizeErr("offset cannot be negative")
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr("offset cannot be beyond end of data")
        if arg:
            self.data = "%s%s%s" % (
                self.data[:offset], arg, self.data[offset:])

    def deleteData(self, offset, count):
        if offset < 0:
            raise xml.dom.IndexSizeErr("offset cannot be negative")
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr("offset cannot be beyond end of data")
        if count < 0:
            raise xml.dom.IndexSizeErr("count cannot be negative")
        if count:
            self.data = self.data[:offset] + self.data[offset+count:]

    def replaceData(self, offset, count, arg):
        if offset < 0:
            raise xml.dom.IndexSizeErr("offset cannot be negative")
        if offset >= len(self.data):
            raise xml.dom.IndexSizeErr("offset cannot be beyond end of data")
        if count < 0:
            raise xml.dom.IndexSizeErr("count cannot be negative")
        if count:
            self.data = "%s%s%s" % (
                self.data[:offset], arg, self.data[offset+count:])

defproperty(CharacterData, "length", doc="Length of the string data.")


class Text(CharacterData):
    __slots__ = ()

    nodeType = Node.TEXT_NODE
    nodeName = "#text"
    attributes = None

    def splitText(self, offset):
        if offset < 0 or offset > len(self.data):
            raise xml.dom.IndexSizeErr("illegal offset value")
        newText = self.__class__()
        newText.data = self.data[offset:]
        newText.ownerDocument = self.ownerDocument
        next = self.nextSibling
        if self.parentNode and self in self.parentNode.childNodes:
            if next is None:
                self.parentNode.appendChild(newText)
            else:
                self.parentNode.insertBefore(newText, next)
        self.data = self.data[:offset]
        return newText

    def writexml(self, writer, indent="", addindent="", newl=""):
        _write_data(writer, "%s%s%s" % (indent, self.data, newl))

    # DOM Level 3 (WD 9 April 2002)

    def _get_wholeText(self):
        L = [self.data]
        n = self.previousSibling
        while n is not None:
            if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
                L.insert(0, n.data)
                n = n.previousSibling
            else:
                break
        n = self.nextSibling
        while n is not None:
            if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
                L.append(n.data)
                n = n.nextSibling
            else:
                break
        return ''.join(L)

    def replaceWholeText(self, content):
        # XXX This needs to be seriously changed if minidom ever
        # supports EntityReference nodes.
        parent = self.parentNode
        n = self.previousSibling
        while n is not None:
            if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
                next = n.previousSibling
                parent.removeChild(n)
                n = next
            else:
                break
        n = self.nextSibling
        if not content:
            parent.removeChild(self)
        while n is not None:
            if n.nodeType in (Node.TEXT_NODE, Node.CDATA_SECTION_NODE):
                next = n.nextSibling
                parent.removeChild(n)
                n = next
            else:
                break
        if content:
            self.data = content
            return self
        else:
            return None

    def _get_isWhitespaceInElementContent(self):
        if self.data.strip():
            return False
        elem = _get_containing_element(self)
        if elem is None:
            return False
        info = self.ownerDocument._get_elem_info(elem)
        if info is None:
            return False
        else:
            return info.isElementContent()

defproperty(Text, "isWhitespaceInElementContent",
            doc="True iff this text node contains only whitespace"
                " and is in element content.")
defproperty(Text, "wholeText",
            doc="The text of all logically-adjacent text nodes.")


def _get_containing_element(node):
    c = node.parentNode
    while c is not None:
        if c.nodeType == Node.ELEMENT_NODE:
            return c
        c = c.parentNode
    return None

def _get_containing_entref(node):
    c = node.parentNode
    while c is not None:
        if c.nodeType == Node.ENTITY_REFERENCE_NODE:
            return c
        c = c.parentNode
    return None


class Comment(CharacterData):
    nodeType = Node.COMMENT_NODE
    nodeName = "#comment"

    def __init__(self, data):
        CharacterData.__init__(self)
        self._data = data

    def writexml(self, writer, indent="", addindent="", newl=""):
        if "--" in self.data:
            raise ValueError("'--' is not allowed in a comment node")
        writer.write("%s<!--%s-->%s" % (indent, self.data, newl))


class CDATASection(Text):
    __slots__ = ()

    nodeType = Node.CDATA_SECTION_NODE
    nodeName = "#cdata-section"

    def writexml(self, writer, indent="", addindent="", newl=""):
        if self.data.find("]]>") >= 0:
            raise ValueError("']]>' not allowed in a CDATA section")
        writer.write("<![CDATA[%s]]>" % self.data)


class ReadOnlySequentialNamedNodeMap(object):
    __slots__ = '_seq',

    def __init__(self, seq=()):
        # seq should be a list or tuple
        self._seq = seq

    def __len__(self):
        return len(self._seq)

    def _get_length(self):
        return len(self._seq)

    def getNamedItem(self, name):
        for n in self._seq:
            if n.nodeName == name:
                return n

    def getNamedItemNS(self, namespaceURI, localName):
        for n in self._seq:
            if n.namespaceURI == namespaceURI and n.localName == localName:
                return n

    def __getitem__(self, name_or_tuple):
        if isinstance(name_or_tuple, tuple):
            node = self.getNamedItemNS(*name_or_tuple)
        else:
            node = self.getNamedItem(name_or_tuple)
        if node is None:
            raise KeyError(name_or_tuple)
        return node

    def item(self, index):
        if index < 0:
            return None
        try:
            return self._seq[index]
        except IndexError:
            return None

    def removeNamedItem(self, name):
        raise xml.dom.NoModificationAllowedErr(
            "NamedNodeMap instance is read-only")

    def removeNamedItemNS(self, namespaceURI, localName):
        raise xml.dom.NoModificationAllowedErr(
            "NamedNodeMap instance is read-only")

    def setNamedItem(self, node):
        raise xml.dom.NoModificationAllowedErr(
            "NamedNodeMap instance is read-only")

    def setNamedItemNS(self, node):
        raise xml.dom.NoModificationAllowedErr(
            "NamedNodeMap instance is read-only")

    def __getstate__(self):
        return [self._seq]

    def __setstate__(self, state):
        self._seq = state[0]

defproperty(ReadOnlySequentialNamedNodeMap, "length",
            doc="Number of entries in the NamedNodeMap.")


class Identified:
    """Mix-in class that supports the publicId and systemId attributes."""

    __slots__ = 'publicId', 'systemId'

    def _identified_mixin_init(self, publicId, systemId):
        self.publicId = publicId
        self.systemId = systemId

    def _get_publicId(self):
        return self.publicId

    def _get_systemId(self):
        return self.systemId

class DocumentType(Identified, Childless, Node):
    nodeType = Node.DOCUMENT_TYPE_NODE
    nodeValue = None
    name = None
    publicId = None
    systemId = None
    internalSubset = None

    def __init__(self, qualifiedName):
        self.entities = ReadOnlySequentialNamedNodeMap()
        self.notations = ReadOnlySequentialNamedNodeMap()
        if qualifiedName:
            prefix, localname = _nssplit(qualifiedName)
            self.name = localname
        self.nodeName = self.name

    def _get_internalSubset(self):
        return self.internalSubset

    def cloneNode(self, deep):
        if self.ownerDocument is None:
            # it's ok
            clone = DocumentType(None)
            clone.name = self.name
            clone.nodeName = self.name
            operation = xml.dom.UserDataHandler.NODE_CLONED
            if deep:
                clone.entities._seq = []
                clone.notations._seq = []
                for n in self.notations._seq:
                    notation = Notation(n.nodeName, n.publicId, n.systemId)
                    clone.notations._seq.append(notation)
                    n._call_user_data_handler(operation, n, notation)
                for e in self.entities._seq:
                    entity = Entity(e.nodeName, e.publicId, e.systemId,
                                    e.notationName)
                    entity.actualEncoding = e.actualEncoding
                    entity.encoding = e.encoding
                    entity.version = e.version
                    clone.entities._seq.append(entity)
                    e._call_user_data_handler(operation, e, entity)
            self._call_user_data_handler(operation, self, clone)
            return clone
        else:
            return None

    def writexml(self, writer, indent="", addindent="", newl=""):
        writer.write("<!DOCTYPE ")
        writer.write(self.name)
        if self.publicId:
            writer.write("%s  PUBLIC '%s'%s  '%s'"
                         % (newl, self.publicId, newl, self.systemId))
        elif self.systemId:
            writer.write("%s  SYSTEM '%s'" % (newl, self.systemId))
        if self.internalSubset is not None:
            writer.write(" [")
            writer.write(self.internalSubset)
            writer.write("]")
        writer.write(">"+newl)

class Entity(Identified, Node):
    attributes = None
    nodeType = Node.ENTITY_NODE
    nodeValue = None

    actualEncoding = None
    encoding = None
    version = None

    def __init__(self, name, publicId, systemId, notation):
        self.nodeName = name
        self.notationName = notation
        self.childNodes = NodeList()
        self._identified_mixin_init(publicId, systemId)

    def _get_actualEncoding(self):
        return self.actualEncoding

    def _get_encoding(self):
        return self.encoding

    def _get_version(self):
        return self.version

    def appendChild(self, newChild):
        raise xml.dom.HierarchyRequestErr(
            "cannot append children to an entity node")

    def insertBefore(self, newChild, refChild):
        raise xml.dom.HierarchyRequestErr(
            "cannot insert children below an entity node")

    def removeChild(self, oldChild):
        raise xml.dom.HierarchyRequestErr(
            "cannot remove children from an entity node")

    def replaceChild(self, newChild, oldChild):
        raise xml.dom.HierarchyRequestErr(
            "cannot replace children of an entity node")

class Notation(Identified, Childless, Node):
    nodeType = Node.NOTATION_NODE
    nodeValue = None

    def __init__(self, name, publicId, systemId):
        self.nodeName = name
        self._identified_mixin_init(publicId, systemId)


class DOMImplementation(DOMImplementationLS):
    _features = [("core", "1.0"),
                 ("core", "2.0"),
                 ("core", None),
                 ("xml", "1.0"),
                 ("xml", "2.0"),
                 ("xml", None),
                 ("ls-load", "3.0"),
                 ("ls-load", None),
                 ]

    def hasFeature(self, feature, version):
        if version == "":
            version = None
        return (feature.lower(), version) in self._features

    def createDocument(self, namespaceURI, qualifiedName, doctype):
        if doctype and doctype.parentNode is not None:
            raise xml.dom.WrongDocumentErr(
                "doctype object owned by another DOM tree")
        doc = self._create_document()

        add_root_element = not (namespaceURI is None
                                and qualifiedName is None
                                and doctype is None)

        if not qualifiedName and add_root_element:
            # The spec is unclear what to raise here; SyntaxErr
            # would be the other obvious candidate. Since Xerces raises
            # InvalidCharacterErr, and since SyntaxErr is not listed
            # for createDocument, that seems to be the better choice.
            # XXX: need to check for illegal characters here and in
            # createElement.

            # DOM Level III clears this up when talking about the return value
            # of this function.  If namespaceURI, qName and DocType are
            # Null the document is returned without a document element
            # Otherwise if doctype or namespaceURI are not None
            # Then we go back to the above problem
            raise xml.dom.InvalidCharacterErr("Element with no name")

        if add_root_element:
            prefix, localname = _nssplit(qualifiedName)
            if prefix == "xml" \
               and namespaceURI != "http://www.w3.org/XML/1998/namespace":
                raise xml.dom.NamespaceErr("illegal use of 'xml' prefix")
            if prefix and not namespaceURI:
                raise xml.dom.NamespaceErr(
                    "illegal use of prefix without namespaces")
            element = doc.createElementNS(namespaceURI, qualifiedName)
            if doctype:
                doc.appendChild(doctype)
            doc.appendChild(element)

        if doctype:
            doctype.parentNode = doctype.ownerDocument = doc

        doc.doctype = doctype
        doc.implementation = self
        return doc

    def createDocumentType(self, qualifiedName, publicId, systemId):
        doctype = DocumentType(qualifiedName)
        doctype.publicId = publicId
        doctype.systemId = systemId
        return doctype

    # DOM Level 3 (WD 9 April 2002)

    def getInterface(self, feature):
        if self.hasFeature(feature, None):
            return self
        else:
            return None

    # internal
    def _create_document(self):
        return Document()

class ElementInfo(object):
    """Object that represents content-model information for an element.

    This implementation is not expected to be used in practice; DOM
    builders should provide implementations which do the right thing
    using information available to it.

    """

    __slots__ = 'tagName',

    def __init__(self, name):
        self.tagName = name

    def getAttributeType(self, aname):
        return _no_type

    def getAttributeTypeNS(self, namespaceURI, localName):
        return _no_type

    def isElementContent(self):
        return False

    def isEmpty(self):
        """Returns true iff this element is declared to have an EMPTY
        content model."""
        return False

    def isId(self, aname):
        """Returns true iff the named attribute is a DTD-style ID."""
        return False

    def isIdNS(self, namespaceURI, localName):
        """Returns true iff the identified attribute is a DTD-style ID."""
        return False

    def __getstate__(self):
        return self.tagName

    def __setstate__(self, state):
        self.tagName = state

def _clear_id_cache(node):
    if node.nodeType == Node.DOCUMENT_NODE:
        node._id_cache.clear()
        node._id_search_stack = None
    elif node.ownerDocument:
        node.ownerDocument._id_cache.clear()
        node.ownerDocument._id_search_stack= None

class Document(Node, DocumentLS):
    __slots__ = ('_elem_info', 'doctype',
                 '_id_search_stack', 'childNodes', '_id_cache')
    _child_node_types = (Node.ELEMENT_NODE, Node.PROCESSING_INSTRUCTION_NODE,
                         Node.COMMENT_NODE, Node.DOCUMENT_TYPE_NODE)

    implementation = DOMImplementation()
    nodeType = Node.DOCUMENT_NODE
    nodeName = "#document"
    nodeValue = None
    attributes = None
    parentNode = None
    previousSibling = nextSibling = None


    # Document attributes from Level 3 (WD 9 April 2002)

    actualEncoding = None
    encoding = None
    standalone = None
    version = None
    strictErrorChecking = False
    errorHandler = None
    documentURI = None

    _magic_id_count = 0

    def __init__(self):
        self.doctype = None
        self.childNodes = NodeList()
        # mapping of (namespaceURI, localName) -> ElementInfo
        #        and tagName -> ElementInfo
        self._elem_info = {}
        self._id_cache = {}
        self._id_search_stack = None

    def _get_elem_info(self, element):
        if element.namespaceURI:
            key = element.namespaceURI, element.localName
        else:
            key = element.tagName
        return self._elem_info.get(key)

    def _get_actualEncoding(self):
        return self.actualEncoding

    def _get_doctype(self):
        return self.doctype

    def _get_documentURI(self):
        return self.documentURI

    def _get_encoding(self):
        return self.encoding

    def _get_errorHandler(self):
        return self.errorHandler

    def _get_standalone(self):
        return self.standalone

    def _get_strictErrorChecking(self):
        return self.strictErrorChecking

    def _get_version(self):
        return self.version

    def appendChild(self, node):
        if node.nodeType not in self._child_node_types:
            raise xml.dom.HierarchyRequestErr(
                "%s cannot be child of %s" % (repr(node), repr(self)))
        if node.parentNode is not None:
            # This needs to be done before the next test since this
            # may *be* the document element, in which case it should
            # end up re-ordered to the end.
            node.parentNode.removeChild(node)

        if node.nodeType == Node.ELEMENT_NODE \
           and self._get_documentElement():
            raise xml.dom.HierarchyRequestErr(
                "two document elements disallowed")
        return Node.appendChild(self, node)

    def removeChild(self, oldChild):
        try:
            self.childNodes.remove(oldChild)
        except ValueError:
            raise xml.dom.NotFoundErr()
        oldChild.nextSibling = oldChild.previousSibling = None
        oldChild.parentNode = None
        if self.documentElement is oldChild:
            self.documentElement = None

        return oldChild

    def _get_documentElement(self):
        for node in self.childNodes:
            if node.nodeType == Node.ELEMENT_NODE:
                return node

    def unlink(self):
        if self.doctype is not None:
            self.doctype.unlink()
            self.doctype = None
        Node.unlink(self)

    def cloneNode(self, deep):
        if not deep:
            return None
        clone = self.implementation.createDocument(None, None, None)
        clone.encoding = self.encoding
        clone.standalone = self.standalone
        clone.version = self.version
        for n in self.childNodes:
            childclone = _clone_node(n, deep, clone)
            assert childclone.ownerDocument.isSameNode(clone)
            clone.childNodes.append(childclone)
            if childclone.nodeType == Node.DOCUMENT_NODE:
                assert clone.documentElement is None
            elif childclone.nodeType == Node.DOCUMENT_TYPE_NODE:
                assert clone.doctype is None
                clone.doctype = childclone
            childclone.parentNode = clone
        self._call_user_data_handler(xml.dom.UserDataHandler.NODE_CLONED,
                                     self, clone)
        return clone

    def createDocumentFragment(self):
        d = DocumentFragment()
        d.ownerDocument = self
        return d

    def createElement(self, tagName):
        e = Element(tagName)
        e.ownerDocument = self
        return e

    def createTextNode(self, data):
        if not isinstance(data, str):
            raise TypeError("node contents must be a string")
        t = Text()
        t.data = data
        t.ownerDocument = self
        return t

    def createCDATASection(self, data):
        if not isinstance(data, str):
            raise TypeError("node contents must be a string")
        c = CDATASection()
        c.data = data
        c.ownerDocument = self
        return c

    def createComment(self, data):
        c = Comment(data)
        c.ownerDocument = self
        return c

    def createProcessingInstruction(self, target, data):
        p = ProcessingInstruction(target, data)
        p.ownerDocument = self
        return p

    def createAttribute(self, qName):
        a = Attr(qName)
        a.ownerDocument = self
        a.value = ""
        return a

    def createElementNS(self, namespaceURI, qualifiedName):
        prefix, localName = _nssplit(qualifiedName)
        e = Element(qualifiedName, namespaceURI, prefix)
        e.ownerDocument = self
        return e

    def createAttributeNS(self, namespaceURI, qualifiedName):
        prefix, localName = _nssplit(qualifiedName)
        a = Attr(qualifiedName, namespaceURI, localName, prefix)
        a.ownerDocument = self
        a.value = ""
        return a

    # A couple of implementation-specific helpers to create node types
    # not supported by the W3C DOM specs:

    def _create_entity(self, name, publicId, systemId, notationName):
        e = Entity(name, publicId, systemId, notationName)
        e.ownerDocument = self
        return e

    def _create_notation(self, name, publicId, systemId):
        n = Notation(name, publicId, systemId)
        n.ownerDocument = self
        return n

    def getElementById(self, id):
        if id in self._id_cache:
            return self._id_cache[id]
        if not (self._elem_info or self._magic_id_count):
            return None

        stack = self._id_search_stack
        if stack is None:
            # we never searched before, or the cache has been cleared
            stack = [self.documentElement]
            self._id_search_stack = stack
        elif not stack:
            # Previous search was completed and cache is still valid;
            # no matching node.
            return None

        result = None
        while stack:
            node = stack.pop()
            # add child elements to stack for continued searching
            stack.extend([child for child in node.childNodes
                          if child.nodeType in _nodeTypes_with_children])
            # check this node
            info = self._get_elem_info(node)
            if info:
                # We have to process all ID attributes before
                # returning in order to get all the attributes set to
                # be IDs using Element.setIdAttribute*().
                for attr in node.attributes.values():
                    if attr.namespaceURI:
                        if info.isIdNS(attr.namespaceURI, attr.localName):
                            self._id_cache[attr.value] = node
                            if attr.value == id:
                                result = node
                            elif not node._magic_id_nodes:
                                break
                    elif info.isId(attr.name):
                        self._id_cache[attr.value] = node
                        if attr.value == id:
                            result = node
                        elif not node._magic_id_nodes:
                            break
                    elif attr._is_id:
                        self._id_cache[attr.value] = node
                        if attr.value == id:
                            result = node
                        elif node._magic_id_nodes == 1:
                            break
            elif node._magic_id_nodes:
                for attr in node.attributes.values():
                    if attr._is_id:
                        self._id_cache[attr.value] = node
                        if attr.value == id:
                            result = node
            if result is not None:
                break
        return result

    def getElementsByTagName(self, name):
        return _get_elements_by_tagName_helper(self, name, NodeList())

    def getElementsByTagNameNS(self, namespaceURI, localName):
        return _get_elements_by_tagName_ns_helper(
            self, namespaceURI, localName, NodeList())

    def isSupported(self, feature, version):
        return self.implementation.hasFeature(feature, version)

    def importNode(self, node, deep):
        if node.nodeType == Node.DOCUMENT_NODE:
            raise xml.dom.NotSupportedErr("cannot import document nodes")
        elif node.nodeType == Node.DOCUMENT_TYPE_NODE:
            raise xml.dom.NotSupportedErr("cannot import document type nodes")
        return _clone_node(node, deep, self)

    def writexml(self, writer, indent="", addindent="", newl="", encoding=None,
                 standalone=None):
        declarations = []

        if encoding:
            declarations.append(f'encoding="{encoding}"')
        if standalone is not None:
            declarations.append(f'standalone="{"yes" if standalone else "no"}"')

        writer.write(f'<?xml version="1.0" {" ".join(declarations)}?>{newl}')

        for node in self.childNodes:
            node.writexml(writer, indent, addindent, newl)

    # DOM Level 3 (WD 9 April 2002)

    def renameNode(self, n, namespaceURI, name):
        if n.ownerDocument is not self:
            raise xml.dom.WrongDocumentErr(
                "cannot rename nodes from other documents;\n"
                "expected %s,\nfound %s" % (self, n.ownerDocument))
        if n.nodeType not in (Node.ELEMENT_NODE, Node.ATTRIBUTE_NODE):
            raise xml.dom.NotSupportedErr(
                "renameNode() only applies to element and attribute nodes")
        if namespaceURI != EMPTY_NAMESPACE:
            if ':' in name:
                prefix, localName = name.split(':', 1)
                if (  prefix == "xmlns"
                      and namespaceURI != xml.dom.XMLNS_NAMESPACE):
                    raise xml.dom.NamespaceErr(
                        "illegal use of 'xmlns' prefix")
            else:
                if (  name == "xmlns"
                      and namespaceURI != xml.dom.XMLNS_NAMESPACE
                      and n.nodeType == Node.ATTRIBUTE_NODE):
                    raise xml.dom.NamespaceErr(
                        "illegal use of the 'xmlns' attribute")
                prefix = None
                localName = name
        else:
            prefix = None
            localName = None
        if n.nodeType == Node.ATTRIBUTE_NODE:
            element = n.ownerElement
            if element is not None:
                is_id = n._is_id
                element.removeAttributeNode(n)
        else:
            element = None
        n.prefix = prefix
        n._localName = localName
        n.namespaceURI = namespaceURI
        n.nodeName = name
        if n.nodeType == Node.ELEMENT_NODE:
            n.tagName = name
        else:
            # attribute node
            n.name = name
            if element is not None:
                element.setAttributeNode(n)
                if is_id:
                    element.setIdAttributeNode(n)
        # It's not clear from a semantic perspective whether we should
        # call the user data handlers for the NODE_RENAMED event since
        # we're re-using the existing node.  The draft spec has been
        # interpreted as meaning "no, don't call the handler unless a
        # new node is created."
        return n

defproperty(Document, "documentElement",
            doc="Top-level element of this document.")


def _clone_node(node, deep, newOwnerDocument):
    """
    Clone a node and give it the new owner document.
    Called by Node.cloneNode and Document.importNode
    """
    if node.ownerDocument.isSameNode(newOwnerDocument):
        operation = xml.dom.UserDataHandler.NODE_CLONED
    else:
        operation = xml.dom.UserDataHandler.NODE_IMPORTED
    if node.nodeType == Node.ELEMENT_NODE:
        clone = newOwnerDocument.createElementNS(node.namespaceURI,
                                                 node.nodeName)
        for attr in node.attributes.values():
            clone.setAttributeNS(attr.namespaceURI, attr.nodeName, attr.value)
            a = clone.getAttributeNodeNS(attr.namespaceURI, attr.localName)
            a.specified = attr.specified

        if deep:
            for child in node.childNodes:
                c = _clone_node(child, deep, newOwnerDocument)
                clone.appendChild(c)

    elif node.nodeType == Node.DOCUMENT_FRAGMENT_NODE:
        clone = newOwnerDocument.createDocumentFragment()
        if deep:
            for child in node.childNodes:
                c = _clone_node(child, deep, newOwnerDocument)
                clone.appendChild(c)

    elif node.nodeType == Node.TEXT_NODE:
        clone = newOwnerDocument.createTextNode(node.data)
    elif node.nodeType == Node.CDATA_SECTION_NODE:
        clone = newOwnerDocument.createCDATASection(node.data)
    elif node.nodeType == Node.PROCESSING_INSTRUCTION_NODE:
        clone = newOwnerDocument.createProcessingInstruction(node.target,
                                                             node.data)
    elif node.nodeType == Node.COMMENT_NODE:
        clone = newOwnerDocument.createComment(node.data)
    elif node.nodeType == Node.ATTRIBUTE_NODE:
        clone = newOwnerDocument.createAttributeNS(node.namespaceURI,
                                                   node.nodeName)
        clone.specified = True
        clone.value = node.value
    elif node.nodeType == Node.DOCUMENT_TYPE_NODE:
        assert node.ownerDocument is not newOwnerDocument
        operation = xml.dom.UserDataHandler.NODE_IMPORTED
        clone = newOwnerDocument.implementation.createDocumentType(
            node.name, node.publicId, node.systemId)
        clone.ownerDocument = newOwnerDocument
        if deep:
            clone.entities._seq = []
            clone.notations._seq = []
            for n in node.notations._seq:
                notation = Notation(n.nodeName, n.publicId, n.systemId)
                notation.ownerDocument = newOwnerDocument
                clone.notations._seq.append(notation)
                if hasattr(n, '_call_user_data_handler'):
                    n._call_user_data_handler(operation, n, notation)
            for e in node.entities._seq:
                entity = Entity(e.nodeName, e.publicId, e.systemId,
                                e.notationName)
                entity.actualEncoding = e.actualEncoding
                entity.encoding = e.encoding
                entity.version = e.version
                entity.ownerDocument = newOwnerDocument
                clone.entities._seq.append(entity)
                if hasattr(e, '_call_user_data_handler'):
                    e._call_user_data_handler(operation, e, entity)
    else:
        # Note the cloning of Document and DocumentType nodes is
        # implementation specific.  minidom handles those cases
        # directly in the cloneNode() methods.
        raise xml.dom.NotSupportedErr("Cannot clone node %s" % repr(node))

    # Check for _call_user_data_handler() since this could conceivably
    # used with other DOM implementations (one of the FourThought
    # DOMs, perhaps?).
    if hasattr(node, '_call_user_data_handler'):
        node._call_user_data_handler(operation, node, clone)
    return clone


def _nssplit(qualifiedName):
    fields = qualifiedName.split(':', 1)
    if len(fields) == 2:
        return fields
    else:
        return (None, fields[0])


def _do_pulldom_parse(func, args, kwargs):
    events = func(*args, **kwargs)
    toktype, rootNode = events.getEvent()
    events.expandNode(rootNode)
    events.clear()
    return rootNode

def parse(file, parser=None, bufsize=None):
    """Parse a file into a DOM by filename or file object."""
    if parser is None and not bufsize:
        from xml.dom import expatbuilder
        return expatbuilder.parse(file)
    else:
        from xml.dom import pulldom
        return _do_pulldom_parse(pulldom.parse, (file,),
            {'parser': parser, 'bufsize': bufsize})

def parseString(string, parser=None):
    """Parse a file into a DOM from a string."""
    if parser is None:
        from xml.dom import expatbuilder
        return expatbuilder.parseString(string)
    else:
        from xml.dom import pulldom
        return _do_pulldom_parse(pulldom.parseString, (string,),
                                 {'parser': parser})

def getDOMImplementation(features=None):
    if features:
        if isinstance(features, str):
            features = domreg._parse_feature_string(features)
        for f, v in features:
            if not Document.implementation.hasFeature(f, v):
                return None
    return Document.implementation
<|endoftext|>import xml.sax
import xml.sax.handler

START_ELEMENT = "START_ELEMENT"
END_ELEMENT = "END_ELEMENT"
COMMENT = "COMMENT"
START_DOCUMENT = "START_DOCUMENT"
END_DOCUMENT = "END_DOCUMENT"
PROCESSING_INSTRUCTION = "PROCESSING_INSTRUCTION"
IGNORABLE_WHITESPACE = "IGNORABLE_WHITESPACE"
CHARACTERS = "CHARACTERS"

class PullDOM(xml.sax.ContentHandler):
    _locator = None
    document = None

    def __init__(self, documentFactory=None):
        from xml.dom import XML_NAMESPACE
        self.documentFactory = documentFactory
        self.firstEvent = [None, None]
        self.lastEvent = self.firstEvent
        self.elementStack = []
        self.push = self.elementStack.append
        try:
            self.pop = self.elementStack.pop
        except AttributeError:
            # use class' pop instead
            pass
        self._ns_contexts = [{XML_NAMESPACE:'xml'}] # contains uri -> prefix dicts
        self._current_context = self._ns_contexts[-1]
        self.pending_events = []

    def pop(self):
        result = self.elementStack[-1]
        del self.elementStack[-1]
        return result

    def setDocumentLocator(self, locator):
        self._locator = locator

    def startPrefixMapping(self, prefix, uri):
        if not hasattr(self, '_xmlns_attrs'):
            self._xmlns_attrs = []
        self._xmlns_attrs.append((prefix or 'xmlns', uri))
        self._ns_contexts.append(self._current_context.copy())
        self._current_context[uri] = prefix or None

    def endPrefixMapping(self, prefix):
        self._current_context = self._ns_contexts.pop()

    def startElementNS(self, name, tagName , attrs):
        # Retrieve xml namespace declaration attributes.
        xmlns_uri = 'http://www.w3.org/2000/xmlns/'
        xmlns_attrs = getattr(self, '_xmlns_attrs', None)
        if xmlns_attrs is not None:
            for aname, value in xmlns_attrs:
                attrs._attrs[(xmlns_uri, aname)] = value
            self._xmlns_attrs = []
        uri, localname = name
        if uri:
            # When using namespaces, the reader may or may not
            # provide us with the original name. If not, create
            # *a* valid tagName from the current context.
            if tagName is None:
                prefix = self._current_context[uri]
                if prefix:
                    tagName = prefix + ":" + localname
                else:
                    tagName = localname
            if self.document:
                node = self.document.createElementNS(uri, tagName)
            else:
                node = self.buildDocument(uri, tagName)
        else:
            # When the tagname is not prefixed, it just appears as
            # localname
            if self.document:
                node = self.document.createElement(localname)
            else:
                node = self.buildDocument(None, localname)

        for aname,value in attrs.items():
            a_uri, a_localname = aname
            if a_uri == xmlns_uri:
                if a_localname == 'xmlns':
                    qname = a_localname
                else:
                    qname = 'xmlns:' + a_localname
                attr = self.document.createAttributeNS(a_uri, qname)
                node.setAttributeNodeNS(attr)
            elif a_uri:
                prefix = self._current_context[a_uri]
                if prefix:
                    qname = prefix + ":" + a_localname
                else:
                    qname = a_localname
                attr = self.document.createAttributeNS(a_uri, qname)
                node.setAttributeNodeNS(attr)
            else:
                attr = self.document.createAttribute(a_localname)
                node.setAttributeNode(attr)
            attr.value = value

        self.lastEvent[1] = [(START_ELEMENT, node), None]
        self.lastEvent = self.lastEvent[1]
        self.push(node)

    def endElementNS(self, name, tagName):
        self.lastEvent[1] = [(END_ELEMENT, self.pop()), None]
        self.lastEvent = self.lastEvent[1]

    def startElement(self, name, attrs):
        if self.document:
            node = self.document.createElement(name)
        else:
            node = self.buildDocument(None, name)

        for aname,value in attrs.items():
            attr = self.document.createAttribute(aname)
            attr.value = value
            node.setAttributeNode(attr)

        self.lastEvent[1] = [(START_ELEMENT, node), None]
        self.lastEvent = self.lastEvent[1]
        self.push(node)

    def endElement(self, name):
        self.lastEvent[1] = [(END_ELEMENT, self.pop()), None]
        self.lastEvent = self.lastEvent[1]

    def comment(self, s):
        if self.document:
            node = self.document.createComment(s)
            self.lastEvent[1] = [(COMMENT, node), None]
            self.lastEvent = self.lastEvent[1]
        else:
            event = [(COMMENT, s), None]
            self.pending_events.append(event)

    def processingInstruction(self, target, data):
        if self.document:
            node = self.document.createProcessingInstruction(target, data)
            self.lastEvent[1] = [(PROCESSING_INSTRUCTION, node), None]
            self.lastEvent = self.lastEvent[1]
        else:
            event = [(PROCESSING_INSTRUCTION, target, data), None]
            self.pending_events.append(event)

    def ignorableWhitespace(self, chars):
        node = self.document.createTextNode(chars)
        self.lastEvent[1] = [(IGNORABLE_WHITESPACE, node), None]
        self.lastEvent = self.lastEvent[1]

    def characters(self, chars):
        node = self.document.createTextNode(chars)
        self.lastEvent[1] = [(CHARACTERS, node), None]
        self.lastEvent = self.lastEvent[1]

    def startDocument(self):
        if self.documentFactory is None:
            import xml.dom.minidom
            self.documentFactory = xml.dom.minidom.Document.implementation

    def buildDocument(self, uri, tagname):
        # Can't do that in startDocument, since we need the tagname
        # XXX: obtain DocumentType
        node = self.documentFactory.createDocument(uri, tagname, None)
        self.document = node
        self.lastEvent[1] = [(START_DOCUMENT, node), None]
        self.lastEvent = self.lastEvent[1]
        self.push(node)
        # Put everything we have seen so far into the document
        for e in self.pending_events:
            if e[0][0] == PROCESSING_INSTRUCTION:
                _,target,data = e[0]
                n = self.document.createProcessingInstruction(target, data)
                e[0] = (PROCESSING_INSTRUCTION, n)
            elif e[0][0] == COMMENT:
                n = self.document.createComment(e[0][1])
                e[0] = (COMMENT, n)
            else:
                raise AssertionError("Unknown pending event ",e[0][0])
            self.lastEvent[1] = e
            self.lastEvent = e
        self.pending_events = None
        return node.firstChild

    def endDocument(self):
        self.lastEvent[1] = [(END_DOCUMENT, self.document), None]
        self.pop()

    def clear(self):
        "clear(): Explicitly release parsing structures"
        self.document = None

class ErrorHandler:
    def warning(self, exception):
        print(exception)
    def error(self, exception):
        raise exception
    def fatalError(self, exception):
        raise exception

class DOMEventStream:
    def __init__(self, stream, parser, bufsize):
        self.stream = stream
        self.parser = parser
        self.bufsize = bufsize
        if not hasattr(self.parser, 'feed'):
            self.getEvent = self._slurp
        self.reset()

    def reset(self):
        self.pulldom = PullDOM()
        # This content handler relies on namespace support
        self.parser.setFeature(xml.sax.handler.feature_namespaces, 1)
        self.parser.setContentHandler(self.pulldom)

    def __getitem__(self, pos):
        import warnings
        warnings.warn(
            "DOMEventStream's __getitem__ method ignores 'pos' parameter. "
            "Use iterator protocol instead.",
            DeprecationWarning,
            stacklevel=2
        )
        rc = self.getEvent()
        if rc:
            return rc
        raise IndexError

    def __next__(self):
        rc = self.getEvent()
        if rc:
            return rc
        raise StopIteration

    def __iter__(self):
        return self

    def expandNode(self, node):
        event = self.getEvent()
        parents = [node]
        while event:
            token, cur_node = event
            if cur_node is node:
                return
            if token != END_ELEMENT:
                parents[-1].appendChild(cur_node)
            if token == START_ELEMENT:
                parents.append(cur_node)
            elif token == END_ELEMENT:
                del parents[-1]
            event = self.getEvent()

    def getEvent(self):
        # use IncrementalParser interface, so we get the desired
        # pull effect
        if not self.pulldom.firstEvent[1]:
            self.pulldom.lastEvent = self.pulldom.firstEvent
        while not self.pulldom.firstEvent[1]:
            buf = self.stream.read(self.bufsize)
            if not buf:
                self.parser.close()
                return None
            self.parser.feed(buf)
        rc = self.pulldom.firstEvent[1][0]
        self.pulldom.firstEvent[1] = self.pulldom.firstEvent[1][1]
        return rc

    def _slurp(self):
        """ Fallback replacement for getEvent() using the
            standard SAX2 interface, which means we slurp the
            SAX events into memory (no performance gain, but
            we are compatible to all SAX parsers).
        """
        self.parser.parse(self.stream)
        self.getEvent = self._emit
        return self._emit()

    def _emit(self):
        """ Fallback replacement for getEvent() that emits
            the events that _slurp() read previously.
        """
        rc = self.pulldom.firstEvent[1][0]
        self.pulldom.firstEvent[1] = self.pulldom.firstEvent[1][1]
        return rc

    def clear(self):
        """clear(): Explicitly release parsing objects"""
        self.pulldom.clear()
        del self.pulldom
        self.parser = None
        self.stream = None

class SAX2DOM(PullDOM):

    def startElementNS(self, name, tagName , attrs):
        PullDOM.startElementNS(self, name, tagName, attrs)
        curNode = self.elementStack[-1]
        parentNode = self.elementStack[-2]
        parentNode.appendChild(curNode)

    def startElement(self, name, attrs):
        PullDOM.startElement(self, name, attrs)
        curNode = self.elementStack[-1]
        parentNode = self.elementStack[-2]
        parentNode.appendChild(curNode)

    def processingInstruction(self, target, data):
        PullDOM.processingInstruction(self, target, data)
        node = self.lastEvent[0][1]
        parentNode = self.elementStack[-1]
        parentNode.appendChild(node)

    def ignorableWhitespace(self, chars):
        PullDOM.ignorableWhitespace(self, chars)
        node = self.lastEvent[0][1]
        parentNode = self.elementStack[-1]
        parentNode.appendChild(node)

    def characters(self, chars):
        PullDOM.characters(self, chars)
        node = self.lastEvent[0][1]
        parentNode = self.elementStack[-1]
        parentNode.appendChild(node)


default_bufsize = (2 ** 14) - 20

def parse(stream_or_string, parser=None, bufsize=None):
    if bufsize is None:
        bufsize = default_bufsize
    if isinstance(stream_or_string, str):
        stream = open(stream_or_string, 'rb')
    else:
        stream = stream_or_string
    if not parser:
        parser = xml.sax.make_parser()
    return DOMEventStream(stream, parser, bufsize)

def parseString(string, parser=None):
    from io import StringIO

    bufsize = len(string)
    buf = StringIO(string)
    if not parser:
        parser = xml.sax.make_parser()
    return DOMEventStream(buf, parser, bufsize)
<|endoftext|>"""W3C Document Object Model implementation for Python.

The Python mapping of the Document Object Model is documented in the
Python Library Reference in the section on the xml.dom package.

This package contains the following modules:

minidom -- A simple implementation of the Level 1 DOM with namespace
           support added (based on the Level 2 specification) and other
           minor Level 2 functionality.

pulldom -- DOM builder supporting on-demand tree-building for selected
           subtrees of the document.

"""


class Node:
    """Class giving the NodeType constants."""
    __slots__ = ()

    # DOM implementations may use this as a base class for their own
    # Node implementations.  If they don't, the constants defined here
    # should still be used as the canonical definitions as they match
    # the values given in the W3C recommendation.  Client code can
    # safely refer to these values in all tests of Node.nodeType
    # values.

    ELEMENT_NODE                = 1
    ATTRIBUTE_NODE              = 2
    TEXT_NODE                   = 3
    CDATA_SECTION_NODE          = 4
    ENTITY_REFERENCE_NODE       = 5
    ENTITY_NODE                 = 6
    PROCESSING_INSTRUCTION_NODE = 7
    COMMENT_NODE                = 8
    DOCUMENT_NODE               = 9
    DOCUMENT_TYPE_NODE          = 10
    DOCUMENT_FRAGMENT_NODE      = 11
    NOTATION_NODE               = 12


#ExceptionCode
INDEX_SIZE_ERR                 = 1
DOMSTRING_SIZE_ERR             = 2
HIERARCHY_REQUEST_ERR          = 3
WRONG_DOCUMENT_ERR             = 4
INVALID_CHARACTER_ERR          = 5
NO_DATA_ALLOWED_ERR            = 6
NO_MODIFICATION_ALLOWED_ERR    = 7
NOT_FOUND_ERR                  = 8
NOT_SUPPORTED_ERR              = 9
INUSE_ATTRIBUTE_ERR            = 10
INVALID_STATE_ERR              = 11
SYNTAX_ERR                     = 12
INVALID_MODIFICATION_ERR       = 13
NAMESPACE_ERR                  = 14
INVALID_ACCESS_ERR             = 15
VALIDATION_ERR                 = 16


class DOMException(Exception):
    """Abstract base class for DOM exceptions.
    Exceptions with specific codes are specializations of this class."""

    def __init__(self, *args, **kw):
        if self.__class__ is DOMException:
            raise RuntimeError(
                "DOMException should not be instantiated directly")
        Exception.__init__(self, *args, **kw)

    def _get_code(self):
        return self.code


class IndexSizeErr(DOMException):
    code = INDEX_SIZE_ERR

class DomstringSizeErr(DOMException):
    code = DOMSTRING_SIZE_ERR

class HierarchyRequestErr(DOMException):
    code = HIERARCHY_REQUEST_ERR

class WrongDocumentErr(DOMException):
    code = WRONG_DOCUMENT_ERR

class InvalidCharacterErr(DOMException):
    code = INVALID_CHARACTER_ERR

class NoDataAllowedErr(DOMException):
    code = NO_DATA_ALLOWED_ERR

class NoModificationAllowedErr(DOMException):
    code = NO_MODIFICATION_ALLOWED_ERR

class NotFoundErr(DOMException):
    code = NOT_FOUND_ERR

class NotSupportedErr(DOMException):
    code = NOT_SUPPORTED_ERR

class InuseAttributeErr(DOMException):
    code = INUSE_ATTRIBUTE_ERR

class InvalidStateErr(DOMException):
    code = INVALID_STATE_ERR

class SyntaxErr(DOMException):
    code = SYNTAX_ERR

class InvalidModificationErr(DOMException):
    code = INVALID_MODIFICATION_ERR

class NamespaceErr(DOMException):
    code = NAMESPACE_ERR

class InvalidAccessErr(DOMException):
    code = INVALID_ACCESS_ERR

class ValidationErr(DOMException):
    code = VALIDATION_ERR

class UserDataHandler:
    """Class giving the operation constants for UserDataHandler.handle()."""

    # Based on DOM Level 3 (WD 9 April 2002)

    NODE_CLONED   = 1
    NODE_IMPORTED = 2
    NODE_DELETED  = 3
    NODE_RENAMED  = 4

XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
XMLNS_NAMESPACE = "http://www.w3.org/2000/xmlns/"
XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
EMPTY_NAMESPACE = None
EMPTY_PREFIX = None

from .domreg import getDOMImplementation, registerDOMImplementation
<|endoftext|>"""Facility to use the Expat parser to load a minidom instance
from a string or file.

This avoids all the overhead of SAX and pulldom to gain performance.
"""

# Warning!
#
# This module is tightly bound to the implementation details of the
# minidom DOM and can't be used with other DOM implementations.  This
# is due, in part, to a lack of appropriate methods in the DOM (there is
# no way to create Entity and Notation nodes via the DOM Level 2
# interface), and for performance.  The latter is the cause of some fairly
# cryptic code.
#
# Performance hacks:
#
#   -  .character_data_handler() has an extra case in which continuing
#      data is appended to an existing Text node; this can be a
#      speedup since pyexpat can break up character data into multiple
#      callbacks even though we set the buffer_text attribute on the
#      parser.  This also gives us the advantage that we don't need a
#      separate normalization pass.
#
#   -  Determining that a node exists is done using an identity comparison
#      with None rather than a truth test; this avoids searching for and
#      calling any methods on the node object if it exists.  (A rather
#      nice speedup is achieved this way as well!)

from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter

TEXT_NODE = Node.TEXT_NODE
CDATA_SECTION_NODE = Node.CDATA_SECTION_NODE
DOCUMENT_NODE = Node.DOCUMENT_NODE

FILTER_ACCEPT = xmlbuilder.DOMBuilderFilter.FILTER_ACCEPT
FILTER_REJECT = xmlbuilder.DOMBuilderFilter.FILTER_REJECT
FILTER_SKIP = xmlbuilder.DOMBuilderFilter.FILTER_SKIP
FILTER_INTERRUPT = xmlbuilder.DOMBuilderFilter.FILTER_INTERRUPT

theDOMImplementation = minidom.getDOMImplementation()

# Expat typename -> TypeInfo
_typeinfo_map = {
    "CDATA":    minidom.TypeInfo(None, "cdata"),
    "ENUM":     minidom.TypeInfo(None, "enumeration"),
    "ENTITY":   minidom.TypeInfo(None, "entity"),
    "ENTITIES": minidom.TypeInfo(None, "entities"),
    "ID":       minidom.TypeInfo(None, "id"),
    "IDREF":    minidom.TypeInfo(None, "idref"),
    "IDREFS":   minidom.TypeInfo(None, "idrefs"),
    "NMTOKEN":  minidom.TypeInfo(None, "nmtoken"),
    "NMTOKENS": minidom.TypeInfo(None, "nmtokens"),
    }

class ElementInfo(object):
    __slots__ = '_attr_info', '_model', 'tagName'

    def __init__(self, tagName, model=None):
        self.tagName = tagName
        self._attr_info = []
        self._model = model

    def __getstate__(self):
        return self._attr_info, self._model, self.tagName

    def __setstate__(self, state):
        self._attr_info, self._model, self.tagName = state

    def getAttributeType(self, aname):
        for info in self._attr_info:
            if info[1] == aname:
                t = info[-2]
                if t[0] == "(":
                    return _typeinfo_map["ENUM"]
                else:
                    return _typeinfo_map[info[-2]]
        return minidom._no_type

    def getAttributeTypeNS(self, namespaceURI, localName):
        return minidom._no_type

    def isElementContent(self):
        if self._model:
            type = self._model[0]
            return type not in (expat.model.XML_CTYPE_ANY,
                                expat.model.XML_CTYPE_MIXED)
        else:
            return False

    def isEmpty(self):
        if self._model:
            return self._model[0] == expat.model.XML_CTYPE_EMPTY
        else:
            return False

    def isId(self, aname):
        for info in self._attr_info:
            if info[1] == aname:
                return info[-2] == "ID"
        return False

    def isIdNS(self, euri, ename, auri, aname):
        # not sure this is meaningful
        return self.isId((auri, aname))

def _intern(builder, s):
    return builder._intern_setdefault(s, s)

def _parse_ns_name(builder, name):
    assert ' ' in name
    parts = name.split(' ')
    intern = builder._intern_setdefault
    if len(parts) == 3:
        uri, localname, prefix = parts
        prefix = intern(prefix, prefix)
        qname = "%s:%s" % (prefix, localname)
        qname = intern(qname, qname)
        localname = intern(localname, localname)
    elif len(parts) == 2:
        uri, localname = parts
        prefix = EMPTY_PREFIX
        qname = localname = intern(localname, localname)
    else:
        raise ValueError("Unsupported syntax: spaces in URIs not supported: %r" % name)
    return intern(uri, uri), localname, prefix, qname


class ExpatBuilder:
    """Document builder that uses Expat to build a ParsedXML.DOM document
    instance."""

    def __init__(self, options=None):
        if options is None:
            options = xmlbuilder.Options()
        self._options = options
        if self._options.filter is not None:
            self._filter = FilterVisibilityController(self._options.filter)
        else:
            self._filter = None
            # This *really* doesn't do anything in this case, so
            # override it with something fast & minimal.
            self._finish_start_element = id
        self._parser = None
        self.reset()

    def createParser(self):
        """Create a new parser object."""
        return expat.ParserCreate()

    def getParser(self):
        """Return the parser object, creating a new one if needed."""
        if not self._parser:
            self._parser = self.createParser()
            self._intern_setdefault = self._parser.intern.setdefault
            self._parser.buffer_text = True
            self._parser.ordered_attributes = True
            self._parser.specified_attributes = True
            self.install(self._parser)
        return self._parser

    def reset(self):
        """Free all data structures used during DOM construction."""
        self.document = theDOMImplementation.createDocument(
            EMPTY_NAMESPACE, None, None)
        self.curNode = self.document
        self._elem_info = self.document._elem_info
        self._cdata = False

    def install(self, parser):
        """Install the callbacks needed to build the DOM into the parser."""
        # This creates circular references!
        parser.StartDoctypeDeclHandler = self.start_doctype_decl_handler
        parser.StartElementHandler = self.first_element_handler
        parser.EndElementHandler = self.end_element_handler
        parser.ProcessingInstructionHandler = self.pi_handler
        if self._options.entities:
            parser.EntityDeclHandler = self.entity_decl_handler
        parser.NotationDeclHandler = self.notation_decl_handler
        if self._options.comments:
            parser.CommentHandler = self.comment_handler
        if self._options.cdata_sections:
            parser.StartCdataSectionHandler = self.start_cdata_section_handler
            parser.EndCdataSectionHandler = self.end_cdata_section_handler
            parser.CharacterDataHandler = self.character_data_handler_cdata
        else:
            parser.CharacterDataHandler = self.character_data_handler
        parser.ExternalEntityRefHandler = self.external_entity_ref_handler
        parser.XmlDeclHandler = self.xml_decl_handler
        parser.ElementDeclHandler = self.element_decl_handler
        parser.AttlistDeclHandler = self.attlist_decl_handler

    def parseFile(self, file):
        """Parse a document from a file object, returning the document
        node."""
        parser = self.getParser()
        first_buffer = True
        try:
            while 1:
                buffer = file.read(16*1024)
                if not buffer:
                    break
                parser.Parse(buffer, False)
                if first_buffer and self.document.documentElement:
                    self._setup_subset(buffer)
                first_buffer = False
            parser.Parse(b"", True)
        except ParseEscape:
            pass
        doc = self.document
        self.reset()
        self._parser = None
        return doc

    def parseString(self, string):
        """Parse a document from a string, returning the document node."""
        parser = self.getParser()
        try:
            parser.Parse(string, True)
            self._setup_subset(string)
        except ParseEscape:
            pass
        doc = self.document
        self.reset()
        self._parser = None
        return doc

    def _setup_subset(self, buffer):
        """Load the internal subset if there might be one."""
        if self.document.doctype:
            extractor = InternalSubsetExtractor()
            extractor.parseString(buffer)
            subset = extractor.getSubset()
            self.document.doctype.internalSubset = subset

    def start_doctype_decl_handler(self, doctypeName, systemId, publicId,
                                   has_internal_subset):
        doctype = self.document.implementation.createDocumentType(
            doctypeName, publicId, systemId)
        doctype.ownerDocument = self.document
        _append_child(self.document, doctype)
        self.document.doctype = doctype
        if self._filter and self._filter.acceptNode(doctype) == FILTER_REJECT:
            self.document.doctype = None
            del self.document.childNodes[-1]
            doctype = None
            self._parser.EntityDeclHandler = None
            self._parser.NotationDeclHandler = None
        if has_internal_subset:
            if doctype is not None:
                doctype.entities._seq = []
                doctype.notations._seq = []
            self._parser.CommentHandler = None
            self._parser.ProcessingInstructionHandler = None
            self._parser.EndDoctypeDeclHandler = self.end_doctype_decl_handler

    def end_doctype_decl_handler(self):
        if self._options.comments:
            self._parser.CommentHandler = self.comment_handler
        self._parser.ProcessingInstructionHandler = self.pi_handler
        if not (self._elem_info or self._filter):
            self._finish_end_element = id

    def pi_handler(self, target, data):
        node = self.document.createProcessingInstruction(target, data)
        _append_child(self.curNode, node)
        if self._filter and self._filter.acceptNode(node) == FILTER_REJECT:
            self.curNode.removeChild(node)

    def character_data_handler_cdata(self, data):
        childNodes = self.curNode.childNodes
        if self._cdata:
            if (  self._cdata_continue
                  and childNodes[-1].nodeType == CDATA_SECTION_NODE):
                childNodes[-1].appendData(data)
                return
            node = self.document.createCDATASection(data)
            self._cdata_continue = True
        elif childNodes and childNodes[-1].nodeType == TEXT_NODE:
            node = childNodes[-1]
            value = node.data + data
            node.data = value
            return
        else:
            node = minidom.Text()
            node.data = data
            node.ownerDocument = self.document
        _append_child(self.curNode, node)

    def character_data_handler(self, data):
        childNodes = self.curNode.childNodes
        if childNodes and childNodes[-1].nodeType == TEXT_NODE:
            node = childNodes[-1]
            node.data = node.data + data
            return
        node = minidom.Text()
        node.data = node.data + data
        node.ownerDocument = self.document
        _append_child(self.curNode, node)

    def entity_decl_handler(self, entityName, is_parameter_entity, value,
                            base, systemId, publicId, notationName):
        if is_parameter_entity:
            # we don't care about parameter entities for the DOM
            return
        if not self._options.entities:
            return
        node = self.document._create_entity(entityName, publicId,
                                            systemId, notationName)
        if value is not None:
            # internal entity
            # node *should* be readonly, but we'll cheat
            child = self.document.createTextNode(value)
            node.childNodes.append(child)
        self.document.doctype.entities._seq.append(node)
        if self._filter and self._filter.acceptNode(node) == FILTER_REJECT:
            del self.document.doctype.entities._seq[-1]

    def notation_decl_handler(self, notationName, base, systemId, publicId):
        node = self.document._create_notation(notationName, publicId, systemId)
        self.document.doctype.notations._seq.append(node)
        if self._filter and self._filter.acceptNode(node) == FILTER_ACCEPT:
            del self.document.doctype.notations._seq[-1]

    def comment_handler(self, data):
        node = self.document.createComment(data)
        _append_child(self.curNode, node)
        if self._filter and self._filter.acceptNode(node) == FILTER_REJECT:
            self.curNode.removeChild(node)

    def start_cdata_section_handler(self):
        self._cdata = True
        self._cdata_continue = False

    def end_cdata_section_handler(self):
        self._cdata = False
        self._cdata_continue = False

    def external_entity_ref_handler(self, context, base, systemId, publicId):
        return 1

    def first_element_handler(self, name, attributes):
        if self._filter is None and not self._elem_info:
            self._finish_end_element = id
        self.getParser().StartElementHandler = self.start_element_handler
        self.start_element_handler(name, attributes)

    def start_element_handler(self, name, attributes):
        node = self.document.createElement(name)
        _append_child(self.curNode, node)
        self.curNode = node

        if attributes:
            for i in range(0, len(attributes), 2):
                a = minidom.Attr(attributes[i], EMPTY_NAMESPACE,
                                 None, EMPTY_PREFIX)
                value = attributes[i+1]
                a.value = value
                a.ownerDocument = self.document
                _set_attribute_node(node, a)

        if node is not self.document.documentElement:
            self._finish_start_element(node)

    def _finish_start_element(self, node):
        if self._filter:
            # To be general, we'd have to call isSameNode(), but this
            # is sufficient for minidom:
            if node is self.document.documentElement:
                return
            filt = self._filter.startContainer(node)
            if filt == FILTER_REJECT:
                # ignore this node & all descendents
                Rejecter(self)
            elif filt == FILTER_SKIP:
                # ignore this node, but make it's children become
                # children of the parent node
                Skipper(self)
            else:
                return
            self.curNode = node.parentNode
            node.parentNode.removeChild(node)
            node.unlink()

    # If this ever changes, Namespaces.end_element_handler() needs to
    # be changed to match.
    #
    def end_element_handler(self, name):
        curNode = self.curNode
        self.curNode = curNode.parentNode
        self._finish_end_element(curNode)

    def _finish_end_element(self, curNode):
        info = self._elem_info.get(curNode.tagName)
        if info:
            self._handle_white_text_nodes(curNode, info)
        if self._filter:
            if curNode is self.document.documentElement:
                return
            if self._filter.acceptNode(curNode) == FILTER_REJECT:
                self.curNode.removeChild(curNode)
                curNode.unlink()

    def _handle_white_text_nodes(self, node, info):
        if (self._options.whitespace_in_element_content
            or not info.isElementContent()):
            return

        # We have element type information and should remove ignorable
        # whitespace; identify for text nodes which contain only
        # whitespace.
        L = []
        for child in node.childNodes:
            if child.nodeType == TEXT_NODE and not child.data.strip():
                L.append(child)

        # Remove ignorable whitespace from the tree.
        for child in L:
            node.removeChild(child)

    def element_decl_handler(self, name, model):
        info = self._elem_info.get(name)
        if info is None:
            self._elem_info[name] = ElementInfo(name, model)
        else:
            assert info._model is None
            info._model = model

    def attlist_decl_handler(self, elem, name, type, default, required):
        info = self._elem_info.get(elem)
        if info is None:
            info = ElementInfo(elem)
            self._elem_info[elem] = info
        info._attr_info.append(
            [None, name, None, None, default, 0, type, required])

    def xml_decl_handler(self, version, encoding, standalone):
        self.document.version = version
        self.document.encoding = encoding
        # This is still a little ugly, thanks to the pyexpat API. ;-(
        if standalone >= 0:
            if standalone:
                self.document.standalone = True
            else:
                self.document.standalone = False


# Don't include FILTER_INTERRUPT, since that's checked separately
# where allowed.
_ALLOWED_FILTER_RETURNS = (FILTER_ACCEPT, FILTER_REJECT, FILTER_SKIP)

class FilterVisibilityController(object):
    """Wrapper around a DOMBuilderFilter which implements the checks
    to make the whatToShow filter attribute work."""

    __slots__ = 'filter',

    def __init__(self, filter):
        self.filter = filter

    def startContainer(self, node):
        mask = self._nodetype_mask[node.nodeType]
        if self.filter.whatToShow & mask:
            val = self.filter.startContainer(node)
            if val == FILTER_INTERRUPT:
                raise ParseEscape
            if val not in _ALLOWED_FILTER_RETURNS:
                raise ValueError(
                      "startContainer() returned illegal value: " + repr(val))
            return val
        else:
            return FILTER_ACCEPT

    def acceptNode(self, node):
        mask = self._nodetype_mask[node.nodeType]
        if self.filter.whatToShow & mask:
            val = self.filter.acceptNode(node)
            if val == FILTER_INTERRUPT:
                raise ParseEscape
            if val == FILTER_SKIP:
                # move all child nodes to the parent, and remove this node
                parent = node.parentNode
                for child in node.childNodes[:]:
                    parent.appendChild(child)
                # node is handled by the caller
                return FILTER_REJECT
            if val not in _ALLOWED_FILTER_RETURNS:
                raise ValueError(
                      "acceptNode() returned illegal value: " + repr(val))
            return val
        else:
            return FILTER_ACCEPT

    _nodetype_mask = {
        Node.ELEMENT_NODE:                NodeFilter.SHOW_ELEMENT,
        Node.ATTRIBUTE_NODE:              NodeFilter.SHOW_ATTRIBUTE,
        Node.TEXT_NODE:                   NodeFilter.SHOW_TEXT,
        Node.CDATA_SECTION_NODE:          NodeFilter.SHOW_CDATA_SECTION,
        Node.ENTITY_REFERENCE_NODE:       NodeFilter.SHOW_ENTITY_REFERENCE,
        Node.ENTITY_NODE:                 NodeFilter.SHOW_ENTITY,
        Node.PROCESSING_INSTRUCTION_NODE: NodeFilter.SHOW_PROCESSING_INSTRUCTION,
        Node.COMMENT_NODE:                NodeFilter.SHOW_COMMENT,
        Node.DOCUMENT_NODE:               NodeFilter.SHOW_DOCUMENT,
        Node.DOCUMENT_TYPE_NODE:          NodeFilter.SHOW_DOCUMENT_TYPE,
        Node.DOCUMENT_FRAGMENT_NODE:      NodeFilter.SHOW_DOCUMENT_FRAGMENT,
        Node.NOTATION_NODE:               NodeFilter.SHOW_NOTATION,
        }


class FilterCrutch(object):
    __slots__ = '_builder', '_level', '_old_start', '_old_end'

    def __init__(self, builder):
        self._level = 0
        self._builder = builder
        parser = builder._parser
        self._old_start = parser.StartElementHandler
        self._old_end = parser.EndElementHandler
        parser.StartElementHandler = self.start_element_handler
        parser.EndElementHandler = self.end_element_handler

class Rejecter(FilterCrutch):
    __slots__ = ()

    def __init__(self, builder):
        FilterCrutch.__init__(self, builder)
        parser = builder._parser
        for name in ("ProcessingInstructionHandler",
                     "CommentHandler",
                     "CharacterDataHandler",
                     "StartCdataSectionHandler",
                     "EndCdataSectionHandler",
                     "ExternalEntityRefHandler",
                     ):
            setattr(parser, name, None)

    def start_element_handler(self, *args):
        self._level = self._level + 1

    def end_element_handler(self, *args):
        if self._level == 0:
            # restore the old handlers
            parser = self._builder._parser
            self._builder.install(parser)
            parser.StartElementHandler = self._old_start
            parser.EndElementHandler = self._old_end
        else:
            self._level = self._level - 1

class Skipper(FilterCrutch):
    __slots__ = ()

    def start_element_handler(self, *args):
        node = self._builder.curNode
        self._old_start(*args)
        if self._builder.curNode is not node:
            self._level = self._level + 1

    def end_element_handler(self, *args):
        if self._level == 0:
            # We're popping back out of the node we're skipping, so we
            # shouldn't need to do anything but reset the handlers.
            self._builder._parser.StartElementHandler = self._old_start
            self._builder._parser.EndElementHandler = self._old_end
            self._builder = None
        else:
            self._level = self._level - 1
            self._old_end(*args)


# framework document used by the fragment builder.
# Takes a string for the doctype, subset string, and namespace attrs string.

_FRAGMENT_BUILDER_INTERNAL_SYSTEM_ID = \
    "http://xml.python.org/entities/fragment-builder/internal"

_FRAGMENT_BUILDER_TEMPLATE = (
    '''\
<!DOCTYPE wrapper
  %%s [
  <!ENTITY fragment-builder-internal
    SYSTEM "%s">
%%s
]>
<wrapper %%s
>&fragment-builder-internal;</wrapper>'''
    % _FRAGMENT_BUILDER_INTERNAL_SYSTEM_ID)


class FragmentBuilder(ExpatBuilder):
    """Builder which constructs document fragments given XML source
    text and a context node.

    The context node is expected to provide information about the
    namespace declarations which are in scope at the start of the
    fragment.
    """

    def __init__(self, context, options=None):
        if context.nodeType == DOCUMENT_NODE:
            self.originalDocument = context
            self.context = context
        else:
            self.originalDocument = context.ownerDocument
            self.context = context
        ExpatBuilder.__init__(self, options)

    def reset(self):
        ExpatBuilder.reset(self)
        self.fragment = None

    def parseFile(self, file):
        """Parse a document fragment from a file object, returning the
        fragment node."""
        return self.parseString(file.read())

    def parseString(self, string):
        """Parse a document fragment from a string, returning the
        fragment node."""
        self._source = string
        parser = self.getParser()
        doctype = self.originalDocument.doctype
        ident = ""
        if doctype:
            subset = doctype.internalSubset or self._getDeclarations()
            if doctype.publicId:
                ident = ('PUBLIC "%s" "%s"'
                         % (doctype.publicId, doctype.systemId))
            elif doctype.systemId:
                ident = 'SYSTEM "%s"' % doctype.systemId
        else:
            subset = ""
        nsattrs = self._getNSattrs() # get ns decls from node's ancestors
        document = _FRAGMENT_BUILDER_TEMPLATE % (ident, subset, nsattrs)
        try:
            parser.Parse(document, True)
        except:
            self.reset()
            raise
        fragment = self.fragment
        self.reset()
##         self._parser = None
        return fragment

    def _getDeclarations(self):
        """Re-create the internal subset from the DocumentType node.

        This is only needed if we don't already have the
        internalSubset as a string.
        """
        doctype = self.context.ownerDocument.doctype
        s = ""
        if doctype:
            for i in range(doctype.notations.length):
                notation = doctype.notations.item(i)
                if s:
                    s = s + "\n  "
                s = "%s<!NOTATION %s" % (s, notation.nodeName)
                if notation.publicId:
                    s = '%s PUBLIC "%s"\n             "%s">' \
                        % (s, notation.publicId, notation.systemId)
                else:
                    s = '%s SYSTEM "%s">' % (s, notation.systemId)
            for i in range(doctype.entities.length):
                entity = doctype.entities.item(i)
                if s:
                    s = s + "\n  "
                s = "%s<!ENTITY %s" % (s, entity.nodeName)
                if entity.publicId:
                    s = '%s PUBLIC "%s"\n             "%s"' \
                        % (s, entity.publicId, entity.systemId)
                elif entity.systemId:
                    s = '%s SYSTEM "%s"' % (s, entity.systemId)
                else:
                    s = '%s "%s"' % (s, entity.firstChild.data)
                if entity.notationName:
                    s = "%s NOTATION %s" % (s, entity.notationName)
                s = s + ">"
        return s

    def _getNSattrs(self):
        return ""

    def external_entity_ref_handler(self, context, base, systemId, publicId):
        if systemId == _FRAGMENT_BUILDER_INTERNAL_SYSTEM_ID:
            # this entref is the one that we made to put the subtree
            # in; all of our given input is parsed in here.
            old_document = self.document
            old_cur_node = self.curNode
            parser = self._parser.ExternalEntityParserCreate(context)
            # put the real document back, parse into the fragment to return
            self.document = self.originalDocument
            self.fragment = self.document.createDocumentFragment()
            self.curNode = self.fragment
            try:
                parser.Parse(self._source, True)
            finally:
                self.curNode = old_cur_node
                self.document = old_document
                self._source = None
            return -1
        else:
            return ExpatBuilder.external_entity_ref_handler(
                self, context, base, systemId, publicId)


class Namespaces:
    """Mix-in class for builders; adds support for namespaces."""

    def _initNamespaces(self):
        # list of (prefix, uri) ns declarations.  Namespace attrs are
        # constructed from this and added to the element's attrs.
        self._ns_ordered_prefixes = []

    def createParser(self):
        """Create a new namespace-handling parser."""
        parser = expat.ParserCreate(namespace_separator=" ")
        parser.namespace_prefixes = True
        return parser

    def install(self, parser):
        """Insert the namespace-handlers onto the parser."""
        ExpatBuilder.install(self, parser)
        if self._options.namespace_declarations:
            parser.StartNamespaceDeclHandler = (
                self.start_namespace_decl_handler)

    def start_namespace_decl_handler(self, prefix, uri):
        """Push this namespace declaration on our storage."""
        self._ns_ordered_prefixes.append((prefix, uri))

    def start_element_handler(self, name, attributes):
        if ' ' in name:
            uri, localname, prefix, qname = _parse_ns_name(self, name)
        else:
            uri = EMPTY_NAMESPACE
            qname = name
            localname = None
            prefix = EMPTY_PREFIX
        node = minidom.Element(qname, uri, prefix, localname)
        node.ownerDocument = self.document
        _append_child(self.curNode, node)
        self.curNode = node

        if self._ns_ordered_prefixes:
            for prefix, uri in self._ns_ordered_prefixes:
                if prefix:
                    a = minidom.Attr(_intern(self, 'xmlns:' + prefix),
                                     XMLNS_NAMESPACE, prefix, "xmlns")
                else:
                    a = minidom.Attr("xmlns", XMLNS_NAMESPACE,
                                     "xmlns", EMPTY_PREFIX)
                a.value = uri
                a.ownerDocument = self.document
                _set_attribute_node(node, a)
            del self._ns_ordered_prefixes[:]

        if attributes:
            node._ensure_attributes()
            _attrs = node._attrs
            _attrsNS = node._attrsNS
            for i in range(0, len(attributes), 2):
                aname = attributes[i]
                value = attributes[i+1]
                if ' ' in aname:
                    uri, localname, prefix, qname = _parse_ns_name(self, aname)
                    a = minidom.Attr(qname, uri, localname, prefix)
                    _attrs[qname] = a
                    _attrsNS[(uri, localname)] = a
                else:
                    a = minidom.Attr(aname, EMPTY_NAMESPACE,
                                     aname, EMPTY_PREFIX)
                    _attrs[aname] = a
                    _attrsNS[(EMPTY_NAMESPACE, aname)] = a
                a.ownerDocument = self.document
                a.value = value
                a.ownerElement = node

    if __debug__:
        # This only adds some asserts to the original
        # end_element_handler(), so we only define this when -O is not
        # used.  If changing one, be sure to check the other to see if
        # it needs to be changed as well.
        #
        def end_element_handler(self, name):
            curNode = self.curNode
            if ' ' in name:
                uri, localname, prefix, qname = _parse_ns_name(self, name)
                assert (curNode.namespaceURI == uri
                        and curNode.localName == localname
                        and curNode.prefix == prefix), \
                        "element stack messed up! (namespace)"
            else:
                assert curNode.nodeName == name, \
                       "element stack messed up - bad nodeName"
                assert curNode.namespaceURI == EMPTY_NAMESPACE, \
                       "element stack messed up - bad namespaceURI"
            self.curNode = curNode.parentNode
            self._finish_end_element(curNode)


class ExpatBuilderNS(Namespaces, ExpatBuilder):
    """Document builder that supports namespaces."""

    def reset(self):
        ExpatBuilder.reset(self)
        self._initNamespaces()


class FragmentBuilderNS(Namespaces, FragmentBuilder):
    """Fragment builder that supports namespaces."""

    def reset(self):
        FragmentBuilder.reset(self)
        self._initNamespaces()

    def _getNSattrs(self):
        """Return string of namespace attributes from this element and
        ancestors."""
        # XXX This needs to be re-written to walk the ancestors of the
        # context to build up the namespace information from
        # declarations, elements, and attributes found in context.
        # Otherwise we have to store a bunch more data on the DOM
        # (though that *might* be more reliable -- not clear).
        attrs = ""
        context = self.context
        L = []
        while context:
            if hasattr(context, '_ns_prefix_uri'):
                for prefix, uri in context._ns_prefix_uri.items():
                    # add every new NS decl from context to L and attrs string
                    if prefix in L:
                        continue
                    L.append(prefix)
                    if prefix:
                        declname = "xmlns:" + prefix
                    else:
                        declname = "xmlns"
                    if attrs:
                        attrs = "%s\n    %s='%s'" % (attrs, declname, uri)
                    else:
                        attrs = " %s='%s'" % (declname, uri)
            context = context.parentNode
        return attrs


class ParseEscape(Exception):
    """Exception raised to short-circuit parsing in InternalSubsetExtractor."""
    pass

class InternalSubsetExtractor(ExpatBuilder):
    """XML processor which can rip out the internal document type subset."""

    subset = None

    def getSubset(self):
        """Return the internal subset as a string."""
        return self.subset

    def parseFile(self, file):
        try:
            ExpatBuilder.parseFile(self, file)
        except ParseEscape:
            pass

    def parseString(self, string):
        try:
            ExpatBuilder.parseString(self, string)
        except ParseEscape:
            pass

    def install(self, parser):
        parser.StartDoctypeDeclHandler = self.start_doctype_decl_handler
        parser.StartElementHandler = self.start_element_handler

    def start_doctype_decl_handler(self, name, publicId, systemId,
                                   has_internal_subset):
        if has_internal_subset:
            parser = self.getParser()
            self.subset = []
            parser.DefaultHandler = self.subset.append
            parser.EndDoctypeDeclHandler = self.end_doctype_decl_handler
        else:
            raise ParseEscape()

    def end_doctype_decl_handler(self):
        s = ''.join(self.subset).replace('\r\n', '\n').replace('\r', '\n')
        self.subset = s
        raise ParseEscape()

    def start_element_handler(self, name, attrs):
        raise ParseEscape()


def parse(file, namespaces=True):
    """Parse a document, returning the resulting Document node.

    'file' may be either a file name or an open file object.
    """
    if namespaces:
        builder = ExpatBuilderNS()
    else:
        builder = ExpatBuilder()

    if isinstance(file, str):
        with open(file, 'rb') as fp:
            result = builder.parseFile(fp)
    else:
        result = builder.parseFile(file)
    return result


def parseString(string, namespaces=True):
    """Parse a document from a string, returning the resulting
    Document node.
    """
    if namespaces:
        builder = ExpatBuilderNS()
    else:
        builder = ExpatBuilder()
    return builder.parseString(string)


def parseFragment(file, context, namespaces=True):
    """Parse a fragment of a document, given the context from which it
    was originally extracted.  context should be the parent of the
    node(s) which are in the fragment.

    'file' may be either a file name or an open file object.
    """
    if namespaces:
        builder = FragmentBuilderNS(context)
    else:
        builder = FragmentBuilder(context)

    if isinstance(file, str):
        with open(file, 'rb') as fp:
            result = builder.parseFile(fp)
    else:
        result = builder.parseFile(file)
    return result


def parseFragmentString(string, context, namespaces=True):
    """Parse a fragment of a document from a string, given the context
    from which it was originally extracted.  context should be the
    parent of the node(s) which are in the fragment.
    """
    if namespaces:
        builder = FragmentBuilderNS(context)
    else:
        builder = FragmentBuilder(context)
    return builder.parseString(string)


def makeBuilder(options):
    """Create a builder based on an Options object."""
    if options.namespaces:
        return ExpatBuilderNS(options)
    else:
        return ExpatBuilder(options)
<|endoftext|>"""Implementation of the DOM Level 3 'LS-Load' feature."""

import copy
import xml.dom

from xml.dom.NodeFilter import NodeFilter


__all__ = ["DOMBuilder", "DOMEntityResolver", "DOMInputSource"]


class Options:
    """Features object that has variables set for each DOMBuilder feature.

    The DOMBuilder class uses an instance of this class to pass settings to
    the ExpatBuilder class.
    """

    # Note that the DOMBuilder class in LoadSave constrains which of these
    # values can be set using the DOM Level 3 LoadSave feature.

    namespaces = 1
    namespace_declarations = True
    validation = False
    external_parameter_entities = True
    external_general_entities = True
    external_dtd_subset = True
    validate_if_schema = False
    validate = False
    datatype_normalization = False
    create_entity_ref_nodes = True
    entities = True
    whitespace_in_element_content = True
    cdata_sections = True
    comments = True
    charset_overrides_xml_encoding = True
    infoset = False
    supported_mediatypes_only = False

    errorHandler = None
    filter = None


class DOMBuilder:
    entityResolver = None
    errorHandler = None
    filter = None

    ACTION_REPLACE = 1
    ACTION_APPEND_AS_CHILDREN = 2
    ACTION_INSERT_AFTER = 3
    ACTION_INSERT_BEFORE = 4

    _legal_actions = (ACTION_REPLACE, ACTION_APPEND_AS_CHILDREN,
                      ACTION_INSERT_AFTER, ACTION_INSERT_BEFORE)

    def __init__(self):
        self._options = Options()

    def _get_entityResolver(self):
        return self.entityResolver
    def _set_entityResolver(self, entityResolver):
        self.entityResolver = entityResolver

    def _get_errorHandler(self):
        return self.errorHandler
    def _set_errorHandler(self, errorHandler):
        self.errorHandler = errorHandler

    def _get_filter(self):
        return self.filter
    def _set_filter(self, filter):
        self.filter = filter

    def setFeature(self, name, state):
        if self.supportsFeature(name):
            state = state and 1 or 0
            try:
                settings = self._settings[(_name_xform(name), state)]
            except KeyError:
                raise xml.dom.NotSupportedErr(
                    "unsupported feature: %r" % (name,)) from None
            else:
                for name, value in settings:
                    setattr(self._options, name, value)
        else:
            raise xml.dom.NotFoundErr("unknown feature: " + repr(name))

    def supportsFeature(self, name):
        return hasattr(self._options, _name_xform(name))

    def canSetFeature(self, name, state):
        key = (_name_xform(name), state and 1 or 0)
        return key in self._settings

    # This dictionary maps from (feature,value) to a list of
    # (option,value) pairs that should be set on the Options object.
    # If a (feature,value) setting is not in this dictionary, it is
    # not supported by the DOMBuilder.
    #
    _settings = {
        ("namespace_declarations", 0): [
            ("namespace_declarations", 0)],
        ("namespace_declarations", 1): [
            ("namespace_declarations", 1)],
        ("validation", 0): [
            ("validation", 0)],
        ("external_general_entities", 0): [
            ("external_general_entities", 0)],
        ("external_general_entities", 1): [
            ("external_general_entities", 1)],
        ("external_parameter_entities", 0): [
            ("external_parameter_entities", 0)],
        ("external_parameter_entities", 1): [
            ("external_parameter_entities", 1)],
        ("validate_if_schema", 0): [
            ("validate_if_schema", 0)],
        ("create_entity_ref_nodes", 0): [
            ("create_entity_ref_nodes", 0)],
        ("create_entity_ref_nodes", 1): [
            ("create_entity_ref_nodes", 1)],
        ("entities", 0): [
            ("create_entity_ref_nodes", 0),
            ("entities", 0)],
        ("entities", 1): [
            ("entities", 1)],
        ("whitespace_in_element_content", 0): [
            ("whitespace_in_element_content", 0)],
        ("whitespace_in_element_content", 1): [
            ("whitespace_in_element_content", 1)],
        ("cdata_sections", 0): [
            ("cdata_sections", 0)],
        ("cdata_sections", 1): [
            ("cdata_sections", 1)],
        ("comments", 0): [
            ("comments", 0)],
        ("comments", 1): [
            ("comments", 1)],
        ("charset_overrides_xml_encoding", 0): [
            ("charset_overrides_xml_encoding", 0)],
        ("charset_overrides_xml_encoding", 1): [
            ("charset_overrides_xml_encoding", 1)],
        ("infoset", 0): [],
        ("infoset", 1): [
            ("namespace_declarations", 0),
            ("validate_if_schema", 0),
            ("create_entity_ref_nodes", 0),
            ("entities", 0),
            ("cdata_sections", 0),
            ("datatype_normalization", 1),
            ("whitespace_in_element_content", 1),
            ("comments", 1),
            ("charset_overrides_xml_encoding", 1)],
        ("supported_mediatypes_only", 0): [
            ("supported_mediatypes_only", 0)],
        ("namespaces", 0): [
            ("namespaces", 0)],
        ("namespaces", 1): [
            ("namespaces", 1)],
    }

    def getFeature(self, name):
        xname = _name_xform(name)
        try:
            return getattr(self._options, xname)
        except AttributeError:
            if name == "infoset":
                options = self._options
                return (options.datatype_normalization
                        and options.whitespace_in_element_content
                        and options.comments
                        and options.charset_overrides_xml_encoding
                        and not (options.namespace_declarations
                                 or options.validate_if_schema
                                 or options.create_entity_ref_nodes
                                 or options.entities
                                 or options.cdata_sections))
            raise xml.dom.NotFoundErr("feature %s not known" % repr(name))

    def parseURI(self, uri):
        if self.entityResolver:
            input = self.entityResolver.resolveEntity(None, uri)
        else:
            input = DOMEntityResolver().resolveEntity(None, uri)
        return self.parse(input)

    def parse(self, input):
        options = copy.copy(self._options)
        options.filter = self.filter
        options.errorHandler = self.errorHandler
        fp = input.byteStream
        if fp is None and options.systemId:
            import urllib.request
            fp = urllib.request.urlopen(input.systemId)
        return self._parse_bytestream(fp, options)

    def parseWithContext(self, input, cnode, action):
        if action not in self._legal_actions:
            raise ValueError("not a legal action")
        raise NotImplementedError("Haven't written this yet...")

    def _parse_bytestream(self, stream, options):
        import xml.dom.expatbuilder
        builder = xml.dom.expatbuilder.makeBuilder(options)
        return builder.parseFile(stream)


def _name_xform(name):
    return name.lower().replace('-', '_')


class DOMEntityResolver(object):
    __slots__ = '_opener',

    def resolveEntity(self, publicId, systemId):
        assert systemId is not None
        source = DOMInputSource()
        source.publicId = publicId
        source.systemId = systemId
        source.byteStream = self._get_opener().open(systemId)

        # determine the encoding if the transport provided it
        source.encoding = self._guess_media_encoding(source)

        # determine the base URI is we can
        import posixpath, urllib.parse
        parts = urllib.parse.urlparse(systemId)
        scheme, netloc, path, params, query, fragment = parts
        # XXX should we check the scheme here as well?
        if path and not path.endswith("/"):
            path = posixpath.dirname(path) + "/"
            parts = scheme, netloc, path, params, query, fragment
            source.baseURI = urllib.parse.urlunparse(parts)

        return source

    def _get_opener(self):
        try:
            return self._opener
        except AttributeError:
            self._opener = self._create_opener()
            return self._opener

    def _create_opener(self):
        import urllib.request
        return urllib.request.build_opener()

    def _guess_media_encoding(self, source):
        info = source.byteStream.info()
        if "Content-Type" in info:
            for param in info.getplist():
                if param.startswith("charset="):
                    return param.split("=", 1)[1].lower()


class DOMInputSource(object):
    __slots__ = ('byteStream', 'characterStream', 'stringData',
                 'encoding', 'publicId', 'systemId', 'baseURI')

    def __init__(self):
        self.byteStream = None
        self.characterStream = None
        self.stringData = None
        self.encoding = None
        self.publicId = None
        self.systemId = None
        self.baseURI = None

    def _get_byteStream(self):
        return self.byteStream
    def _set_byteStream(self, byteStream):
        self.byteStream = byteStream

    def _get_characterStream(self):
        return self.characterStream
    def _set_characterStream(self, characterStream):
        self.characterStream = characterStream

    def _get_stringData(self):
        return self.stringData
    def _set_stringData(self, data):
        self.stringData = data

    def _get_encoding(self):
        return self.encoding
    def _set_encoding(self, encoding):
        self.encoding = encoding

    def _get_publicId(self):
        return self.publicId
    def _set_publicId(self, publicId):
        self.publicId = publicId

    def _get_systemId(self):
        return self.systemId
    def _set_systemId(self, systemId):
        self.systemId = systemId

    def _get_baseURI(self):
        return self.baseURI
    def _set_baseURI(self, uri):
        self.baseURI = uri


class DOMBuilderFilter:
    """Element filter which can be used to tailor construction of
    a DOM instance.
    """

    # There's really no need for this class; concrete implementations
    # should just implement the endElement() and startElement()
    # methods as appropriate.  Using this makes it easy to only
    # implement one of them.

    FILTER_ACCEPT = 1
    FILTER_REJECT = 2
    FILTER_SKIP = 3
    FILTER_INTERRUPT = 4

    whatToShow = NodeFilter.SHOW_ALL

    def _get_whatToShow(self):
        return self.whatToShow

    def acceptNode(self, element):
        return self.FILTER_ACCEPT

    def startContainer(self, element):
        return self.FILTER_ACCEPT

del NodeFilter


class DocumentLS:
    """Mixin to create documents that conform to the load/save spec."""

    async_ = False

    def _get_async(self):
        return False

    def _set_async(self, flag):
        if flag:
            raise xml.dom.NotSupportedErr(
                "asynchronous document loading is not supported")

    def abort(self):
        # What does it mean to "clear" a document?  Does the
        # documentElement disappear?
        raise NotImplementedError(
            "haven't figured out what this means yet")

    def load(self, uri):
        raise NotImplementedError("haven't written this yet")

    def loadXML(self, source):
        raise NotImplementedError("haven't written this yet")

    def saveXML(self, snode):
        if snode is None:
            snode = self
        elif snode.ownerDocument is not self:
            raise xml.dom.WrongDocumentErr()
        return snode.toxml()


class DOMImplementationLS:
    MODE_SYNCHRONOUS = 1
    MODE_ASYNCHRONOUS = 2

    def createDOMBuilder(self, mode, schemaType):
        if schemaType is not None:
            raise xml.dom.NotSupportedErr(
                "schemaType not yet supported")
        if mode == self.MODE_SYNCHRONOUS:
            return DOMBuilder()
        if mode == self.MODE_ASYNCHRONOUS:
            raise xml.dom.NotSupportedErr(
                "asynchronous builders are not supported")
        raise ValueError("unknown value for mode")

    def createDOMWriter(self):
        raise NotImplementedError(
            "the writer interface hasn't been written yet!")

    def createDOMInputSource(self):
        return DOMInputSource()
<|endoftext|>"""Lightweight XML support for Python.

 XML is an inherently hierarchical data format, and the most natural way to
 represent it is with a tree.  This module has two classes for this purpose:

    1. ElementTree represents the whole XML document as a tree and

    2. Element represents a single node in this tree.

 Interactions with the whole document (reading and writing to/from files) are
 usually done on the ElementTree level.  Interactions with a single XML element
 and its sub-elements are done on the Element level.

 Element is a flexible container object designed to store hierarchical data
 structures in memory. It can be described as a cross between a list and a
 dictionary.  Each Element has a number of properties associated with it:

    'tag' - a string containing the element's name.

    'attributes' - a Python dictionary storing the element's attributes.

    'text' - a string containing the element's text content.

    'tail' - an optional string containing text after the element's end tag.

    And a number of child elements stored in a Python sequence.

 To create an element instance, use the Element constructor,
 or the SubElement factory function.

 You can also use the ElementTree class to wrap an element structure
 and convert it to and from XML.

"""

#---------------------------------------------------------------------
# Licensed to PSF under a Contributor Agreement.
# See https://www.python.org/psf/license for licensing details.
#
# ElementTree
# Copyright (c) 1999-2008 by Fredrik Lundh.  All rights reserved.
#
# fredrik@pythonware.com
# http://www.pythonware.com
# --------------------------------------------------------------------
# The ElementTree toolkit is
#
# Copyright (c) 1999-2008 by Fredrik Lundh
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Secret Labs AB or the author not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.
# --------------------------------------------------------------------

__all__ = [
    # public symbols
    "Comment",
    "dump",
    "Element", "ElementTree",
    "fromstring", "fromstringlist",
    "indent", "iselement", "iterparse",
    "parse", "ParseError",
    "PI", "ProcessingInstruction",
    "QName",
    "SubElement",
    "tostring", "tostringlist",
    "TreeBuilder",
    "VERSION",
    "XML", "XMLID",
    "XMLParser", "XMLPullParser",
    "register_namespace",
    "canonicalize", "C14NWriterTarget",
    ]

VERSION = "1.3.0"

import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib

from . import ElementPath


class ParseError(SyntaxError):
    """An error when parsing an XML document.

    In addition to its exception value, a ParseError contains
    two extra attributes:
        'code'     - the specific exception code
        'position' - the line and column of the error

    """
    pass

# --------------------------------------------------------------------


def iselement(element):
    """Return True if *element* appears to be an Element."""
    return hasattr(element, 'tag')


class Element:
    """An XML element.

    This class is the reference implementation of the Element interface.

    An element's length is its number of subelements.  That means if you
    want to check if an element is truly empty, you should check BOTH
    its length AND its text attribute.

    The element tag, attribute names, and attribute values can be either
    bytes or strings.

    *tag* is the element name.  *attrib* is an optional dictionary containing
    element attributes. *extra* are additional element attributes given as
    keyword arguments.

    Example form:
        <tag attrib>text<child/>...</tag>tail

    """

    tag = None
    """The element's name."""

    attrib = None
    """Dictionary of the element's attributes."""

    text = None
    """
    Text before first subelement. This is either a string or the value None.
    Note that if there is no text, this attribute may be either
    None or the empty string, depending on the parser.

    """

    tail = None
    """
    Text after this element's end tag, but before the next sibling element's
    start tag.  This is either a string or the value None.  Note that if there
    was no text, this attribute may be either None or an empty string,
    depending on the parser.

    """

    def __init__(self, tag, attrib={}, **extra):
        if not isinstance(attrib, dict):
            raise TypeError("attrib must be dict, not %s" % (
                attrib.__class__.__name__,))
        self.tag = tag
        self.attrib = {**attrib, **extra}
        self._children = []

    def __repr__(self):
        return "<%s %r at %#x>" % (self.__class__.__name__, self.tag, id(self))

    def makeelement(self, tag, attrib):
        """Create a new element with the same type.

        *tag* is a string containing the element name.
        *attrib* is a dictionary containing the element attributes.

        Do not call this method, use the SubElement factory function instead.

        """
        return self.__class__(tag, attrib)

    def copy(self):
        """Return copy of current element.

        This creates a shallow copy. Subelements will be shared with the
        original tree.

        """
        warnings.warn(
            "elem.copy() is deprecated. Use copy.copy(elem) instead.",
            DeprecationWarning
            )
        return self.__copy__()

    def __copy__(self):
        elem = self.makeelement(self.tag, self.attrib)
        elem.text = self.text
        elem.tail = self.tail
        elem[:] = self
        return elem

    def __len__(self):
        return len(self._children)

    def __bool__(self):
        warnings.warn(
            "The behavior of this method will change in future versions.  "
            "Use specific 'len(elem)' or 'elem is not None' test instead.",
            FutureWarning, stacklevel=2
            )
        return len(self._children) != 0 # emulate old behaviour, for now

    def __getitem__(self, index):
        return self._children[index]

    def __setitem__(self, index, element):
        if isinstance(index, slice):
            for elt in element:
                self._assert_is_element(elt)
        else:
            self._assert_is_element(element)
        self._children[index] = element

    def __delitem__(self, index):
        del self._children[index]

    def append(self, subelement):
        """Add *subelement* to the end of this element.

        The new element will appear in document order after the last existing
        subelement (or directly after the text, if it's the first subelement),
        but before the end tag for this element.

        """
        self._assert_is_element(subelement)
        self._children.append(subelement)

    def extend(self, elements):
        """Append subelements from a sequence.

        *elements* is a sequence with zero or more elements.

        """
        for element in elements:
            self._assert_is_element(element)
            self._children.append(element)

    def insert(self, index, subelement):
        """Insert *subelement* at position *index*."""
        self._assert_is_element(subelement)
        self._children.insert(index, subelement)

    def _assert_is_element(self, e):
        # Need to refer to the actual Python implementation, not the
        # shadowing C implementation.
        if not isinstance(e, _Element_Py):
            raise TypeError('expected an Element, not %s' % type(e).__name__)

    def remove(self, subelement):
        """Remove matching subelement.

        Unlike the find methods, this method compares elements based on
        identity, NOT ON tag value or contents.  To remove subelements by
        other means, the easiest way is to use a list comprehension to
        select what elements to keep, and then use slice assignment to update
        the parent element.

        ValueError is raised if a matching element could not be found.

        """
        # assert iselement(element)
        self._children.remove(subelement)

    def find(self, path, namespaces=None):
        """Find first matching element by tag name or path.

        *path* is a string having either an element tag or an XPath,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Return the first matching element, or None if no element was found.

        """
        return ElementPath.find(self, path, namespaces)

    def findtext(self, path, default=None, namespaces=None):
        """Find text for first matching element by tag name or path.

        *path* is a string having either an element tag or an XPath,
        *default* is the value to return if the element was not found,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Return text content of first matching element, or default value if
        none was found.  Note that if an element is found having no text
        content, the empty string is returned.

        """
        return ElementPath.findtext(self, path, default, namespaces)

    def findall(self, path, namespaces=None):
        """Find all matching subelements by tag name or path.

        *path* is a string having either an element tag or an XPath,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Returns list containing all matching elements in document order.

        """
        return ElementPath.findall(self, path, namespaces)

    def iterfind(self, path, namespaces=None):
        """Find all matching subelements by tag name or path.

        *path* is a string having either an element tag or an XPath,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Return an iterable yielding all matching elements in document order.

        """
        return ElementPath.iterfind(self, path, namespaces)

    def clear(self):
        """Reset element.

        This function removes all subelements, clears all attributes, and sets
        the text and tail attributes to None.

        """
        self.attrib.clear()
        self._children = []
        self.text = self.tail = None

    def get(self, key, default=None):
        """Get element attribute.

        Equivalent to attrib.get, but some implementations may handle this a
        bit more efficiently.  *key* is what attribute to look for, and
        *default* is what to return if the attribute was not found.

        Returns a string containing the attribute value, or the default if
        attribute was not found.

        """
        return self.attrib.get(key, default)

    def set(self, key, value):
        """Set element attribute.

        Equivalent to attrib[key] = value, but some implementations may handle
        this a bit more efficiently.  *key* is what attribute to set, and
        *value* is the attribute value to set it to.

        """
        self.attrib[key] = value

    def keys(self):
        """Get list of attribute names.

        Names are returned in an arbitrary order, just like an ordinary
        Python dict.  Equivalent to attrib.keys()

        """
        return self.attrib.keys()

    def items(self):
        """Get element attributes as a sequence.

        The attributes are returned in arbitrary order.  Equivalent to
        attrib.items().

        Return a list of (name, value) tuples.

        """
        return self.attrib.items()

    def iter(self, tag=None):
        """Create tree iterator.

        The iterator loops over the element and all subelements in document
        order, returning all elements with a matching tag.

        If the tree structure is modified during iteration, new or removed
        elements may or may not be included.  To get a stable set, use the
        list() function on the iterator, and loop over the resulting list.

        *tag* is what tags to look for (default is to return all elements)

        Return an iterator containing all the matching elements.

        """
        if tag == "*":
            tag = None
        if tag is None or self.tag == tag:
            yield self
        for e in self._children:
            yield from e.iter(tag)

    def itertext(self):
        """Create text iterator.

        The iterator loops over the element and all subelements in document
        order, returning all inner text.

        """
        tag = self.tag
        if not isinstance(tag, str) and tag is not None:
            return
        t = self.text
        if t:
            yield t
        for e in self:
            yield from e.itertext()
            t = e.tail
            if t:
                yield t


def SubElement(parent, tag, attrib={}, **extra):
    """Subelement factory which creates an element instance, and appends it
    to an existing parent.

    The element tag, attribute names, and attribute values can be either
    bytes or Unicode strings.

    *parent* is the parent element, *tag* is the subelements name, *attrib* is
    an optional directory containing element attributes, *extra* are
    additional attributes given as keyword arguments.

    """
    attrib = {**attrib, **extra}
    element = parent.makeelement(tag, attrib)
    parent.append(element)
    return element


def Comment(text=None):
    """Comment element factory.

    This function creates a special element which the standard serializer
    serializes as an XML comment.

    *text* is a string containing the comment string.

    """
    element = Element(Comment)
    element.text = text
    return element


def ProcessingInstruction(target, text=None):
    """Processing Instruction element factory.

    This function creates a special element which the standard serializer
    serializes as an XML comment.

    *target* is a string containing the processing instruction, *text* is a
    string containing the processing instruction contents, if any.

    """
    element = Element(ProcessingInstruction)
    element.text = target
    if text:
        element.text = element.text + " " + text
    return element

PI = ProcessingInstruction


class QName:
    """Qualified name wrapper.

    This class can be used to wrap a QName attribute value in order to get
    proper namespace handing on output.

    *text_or_uri* is a string containing the QName value either in the form
    {uri}local, or if the tag argument is given, the URI part of a QName.

    *tag* is an optional argument which if given, will make the first
    argument (text_or_uri) be interpreted as a URI, and this argument (tag)
    be interpreted as a local name.

    """
    def __init__(self, text_or_uri, tag=None):
        if tag:
            text_or_uri = "{%s}%s" % (text_or_uri, tag)
        self.text = text_or_uri
    def __str__(self):
        return self.text
    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.text)
    def __hash__(self):
        return hash(self.text)
    def __le__(self, other):
        if isinstance(other, QName):
            return self.text <= other.text
        return self.text <= other
    def __lt__(self, other):
        if isinstance(other, QName):
            return self.text < other.text
        return self.text < other
    def __ge__(self, other):
        if isinstance(other, QName):
            return self.text >= other.text
        return self.text >= other
    def __gt__(self, other):
        if isinstance(other, QName):
            return self.text > other.text
        return self.text > other
    def __eq__(self, other):
        if isinstance(other, QName):
            return self.text == other.text
        return self.text == other

# --------------------------------------------------------------------


class ElementTree:
    """An XML element hierarchy.

    This class also provides support for serialization to and from
    standard XML.

    *element* is an optional root element node,
    *file* is an optional file handle or file name of an XML file whose
    contents will be used to initialize the tree with.

    """
    def __init__(self, element=None, file=None):
        # assert element is None or iselement(element)
        self._root = element # first node
        if file:
            self.parse(file)

    def getroot(self):
        """Return root element of this tree."""
        return self._root

    def _setroot(self, element):
        """Replace root element of this tree.

        This will discard the current contents of the tree and replace it
        with the given element.  Use with care!

        """
        # assert iselement(element)
        self._root = element

    def parse(self, source, parser=None):
        """Load external XML document into element tree.

        *source* is a file name or file object, *parser* is an optional parser
        instance that defaults to XMLParser.

        ParseError is raised if the parser fails to parse the document.

        Returns the root element of the given source document.

        """
        close_source = False
        if not hasattr(source, "read"):
            source = open(source, "rb")
            close_source = True
        try:
            if parser is None:
                # If no parser was specified, create a default XMLParser
                parser = XMLParser()
                if hasattr(parser, '_parse_whole'):
                    # The default XMLParser, when it comes from an accelerator,
                    # can define an internal _parse_whole API for efficiency.
                    # It can be used to parse the whole source without feeding
                    # it with chunks.
                    self._root = parser._parse_whole(source)
                    return self._root
            while True:
                data = source.read(65536)
                if not data:
                    break
                parser.feed(data)
            self._root = parser.close()
            return self._root
        finally:
            if close_source:
                source.close()

    def iter(self, tag=None):
        """Create and return tree iterator for the root element.

        The iterator loops over all elements in this tree, in document order.

        *tag* is a string with the tag name to iterate over
        (default is to return all elements).

        """
        # assert self._root is not None
        return self._root.iter(tag)

    def find(self, path, namespaces=None):
        """Find first matching element by tag name or path.

        Same as getroot().find(path), which is Element.find()

        *path* is a string having either an element tag or an XPath,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Return the first matching element, or None if no element was found.

        """
        # assert self._root is not None
        if path[:1] == "/":
            path = "." + path
            warnings.warn(
                "This search is broken in 1.3 and earlier, and will be "
                "fixed in a future version.  If you rely on the current "
                "behaviour, change it to %r" % path,
                FutureWarning, stacklevel=2
                )
        return self._root.find(path, namespaces)

    def findtext(self, path, default=None, namespaces=None):
        """Find first matching element by tag name or path.

        Same as getroot().findtext(path),  which is Element.findtext()

        *path* is a string having either an element tag or an XPath,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Return the first matching element, or None if no element was found.

        """
        # assert self._root is not None
        if path[:1] == "/":
            path = "." + path
            warnings.warn(
                "This search is broken in 1.3 and earlier, and will be "
                "fixed in a future version.  If you rely on the current "
                "behaviour, change it to %r" % path,
                FutureWarning, stacklevel=2
                )
        return self._root.findtext(path, default, namespaces)

    def findall(self, path, namespaces=None):
        """Find all matching subelements by tag name or path.

        Same as getroot().findall(path), which is Element.findall().

        *path* is a string having either an element tag or an XPath,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Return list containing all matching elements in document order.

        """
        # assert self._root is not None
        if path[:1] == "/":
            path = "." + path
            warnings.warn(
                "This search is broken in 1.3 and earlier, and will be "
                "fixed in a future version.  If you rely on the current "
                "behaviour, change it to %r" % path,
                FutureWarning, stacklevel=2
                )
        return self._root.findall(path, namespaces)

    def iterfind(self, path, namespaces=None):
        """Find all matching subelements by tag name or path.

        Same as getroot().iterfind(path), which is element.iterfind()

        *path* is a string having either an element tag or an XPath,
        *namespaces* is an optional mapping from namespace prefix to full name.

        Return an iterable yielding all matching elements in document order.

        """
        # assert self._root is not None
        if path[:1] == "/":
            path = "." + path
            warnings.warn(
                "This search is broken in 1.3 and earlier, and will be "
                "fixed in a future version.  If you rely on the current "
                "behaviour, change it to %r" % path,
                FutureWarning, stacklevel=2
                )
        return self._root.iterfind(path, namespaces)

    def write(self, file_or_filename,
              encoding=None,
              xml_declaration=None,
              default_namespace=None,
              method=None, *,
              short_empty_elements=True):
        """Write element tree to a file as XML.

        Arguments:
          *file_or_filename* -- file name or a file object opened for writing

          *encoding* -- the output encoding (default: US-ASCII)

          *xml_declaration* -- bool indicating if an XML declaration should be
                               added to the output. If None, an XML declaration
                               is added if encoding IS NOT either of:
                               US-ASCII, UTF-8, or Unicode

          *default_namespace* -- sets the default XML namespace (for "xmlns")

          *method* -- either "xml" (default), "html, "text", or "c14n"

          *short_empty_elements* -- controls the formatting of elements
                                    that contain no content. If True (default)
                                    they are emitted as a single self-closed
                                    tag, otherwise they are emitted as a pair
                                    of start/end tags

        """
        if not method:
            method = "xml"
        elif method not in _serialize:
            raise ValueError("unknown method %r" % method)
        if not encoding:
            if method == "c14n":
                encoding = "utf-8"
            else:
                encoding = "us-ascii"
        with _get_writer(file_or_filename, encoding) as (write, declared_encoding):
            if method == "xml" and (xml_declaration or
                    (xml_declaration is None and
                     encoding.lower() != "unicode" and
                     declared_encoding.lower() not in ("utf-8", "us-ascii"))):
                write("<?xml version='1.0' encoding='%s'?>\n" % (
                    declared_encoding,))
            if method == "text":
                _serialize_text(write, self._root)
            else:
                qnames, namespaces = _namespaces(self._root, default_namespace)
                serialize = _serialize[method]
                serialize(write, self._root, qnames, namespaces,
                          short_empty_elements=short_empty_elements)

    def write_c14n(self, file):
        # lxml.etree compatibility.  use output method instead
        return self.write(file, method="c14n")

# --------------------------------------------------------------------
# serialization support

@contextlib.contextmanager
