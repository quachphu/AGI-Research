                defaults += (param.default,)
        elif kind is _POSITIONAL_OR_KEYWORD:
            args.append(name)
            if param.default is not param.empty:
                defaults += (param.default,)
        elif kind is _VAR_POSITIONAL:
            varargs = name
        elif kind is _KEYWORD_ONLY:
            kwonlyargs.append(name)
            if param.default is not param.empty:
                kwdefaults[name] = param.default
        elif kind is _VAR_KEYWORD:
            varkw = name

        if param.annotation is not param.empty:
            annotations[name] = param.annotation

    if not kwdefaults:
        # compatibility with 'func.__kwdefaults__'
        kwdefaults = None

    if not defaults:
        # compatibility with 'func.__defaults__'
        defaults = None

    return FullArgSpec(posonlyargs + args, varargs, varkw, defaults,
                       kwonlyargs, kwdefaults, annotations)


ArgInfo = namedtuple('ArgInfo', 'args varargs keywords locals')

def getargvalues(frame):
    """Get information about arguments passed into a particular frame.

    A tuple of four things is returned: (args, varargs, varkw, locals).
    'args' is a list of the argument names.
    'varargs' and 'varkw' are the names of the * and ** arguments or None.
    'locals' is the locals dictionary of the given frame."""
    args, varargs, varkw = getargs(frame.f_code)
    return ArgInfo(args, varargs, varkw, frame.f_locals)

def formatannotation(annotation, base_module=None):
    if getattr(annotation, '__module__', None) == 'typing':
        def repl(match):
            text = match.group()
            return text.removeprefix('typing.')
        return re.sub(r'[\w\.]+', repl, repr(annotation))
    if isinstance(annotation, types.GenericAlias):
        return str(annotation)
    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', base_module):
            return annotation.__qualname__
        return annotation.__module__+'.'+annotation.__qualname__
    return repr(annotation)

def formatannotationrelativeto(object):
    module = getattr(object, '__module__', None)
    def _formatannotation(annotation):
        return formatannotation(annotation, module)
    return _formatannotation

def formatargspec(args, varargs=None, varkw=None, defaults=None,
                  kwonlyargs=(), kwonlydefaults={}, annotations={},
                  formatarg=str,
                  formatvarargs=lambda name: '*' + name,
                  formatvarkw=lambda name: '**' + name,
                  formatvalue=lambda value: '=' + repr(value),
                  formatreturns=lambda text: ' -> ' + text,
                  formatannotation=formatannotation):
    """Format an argument spec from the values returned by getfullargspec.

    The first seven arguments are (args, varargs, varkw, defaults,
    kwonlyargs, kwonlydefaults, annotations).  The other five arguments
    are the corresponding optional formatting functions that are called to
    turn names and values into strings.  The last argument is an optional
    function to format the sequence of arguments.

    Deprecated since Python 3.5: use the `signature` function and `Signature`
    objects.
    """

    from warnings import warn

    warn("`formatargspec` is deprecated since Python 3.5. Use `signature` and "
         "the `Signature` object directly",
         DeprecationWarning,
         stacklevel=2)

    def formatargandannotation(arg):
        result = formatarg(arg)
        if arg in annotations:
            result += ': ' + formatannotation(annotations[arg])
        return result
    specs = []
    if defaults:
        firstdefault = len(args) - len(defaults)
    for i, arg in enumerate(args):
        spec = formatargandannotation(arg)
        if defaults and i >= firstdefault:
            spec = spec + formatvalue(defaults[i - firstdefault])
        specs.append(spec)
    if varargs is not None:
        specs.append(formatvarargs(formatargandannotation(varargs)))
    else:
        if kwonlyargs:
            specs.append('*')
    if kwonlyargs:
        for kwonlyarg in kwonlyargs:
            spec = formatargandannotation(kwonlyarg)
            if kwonlydefaults and kwonlyarg in kwonlydefaults:
                spec += formatvalue(kwonlydefaults[kwonlyarg])
            specs.append(spec)
    if varkw is not None:
        specs.append(formatvarkw(formatargandannotation(varkw)))
    result = '(' + ', '.join(specs) + ')'
    if 'return' in annotations:
        result += formatreturns(formatannotation(annotations['return']))
    return result

def formatargvalues(args, varargs, varkw, locals,
                    formatarg=str,
                    formatvarargs=lambda name: '*' + name,
                    formatvarkw=lambda name: '**' + name,
                    formatvalue=lambda value: '=' + repr(value)):
    """Format an argument spec from the 4 values returned by getargvalues.

    The first four arguments are (args, varargs, varkw, locals).  The
    next four arguments are the corresponding optional formatting functions
    that are called to turn names and values into strings.  The ninth
    argument is an optional function to format the sequence of arguments."""
    def convert(name, locals=locals,
                formatarg=formatarg, formatvalue=formatvalue):
        return formatarg(name) + formatvalue(locals[name])
    specs = []
    for i in range(len(args)):
        specs.append(convert(args[i]))
    if varargs:
        specs.append(formatvarargs(varargs) + formatvalue(locals[varargs]))
    if varkw:
        specs.append(formatvarkw(varkw) + formatvalue(locals[varkw]))
    return '(' + ', '.join(specs) + ')'

def _missing_arguments(f_name, argnames, pos, values):
    names = [repr(name) for name in argnames if name not in values]
    missing = len(names)
    if missing == 1:
        s = names[0]
    elif missing == 2:
        s = "{} and {}".format(*names)
    else:
        tail = ", {} and {}".format(*names[-2:])
        del names[-2:]
        s = ", ".join(names) + tail
    raise TypeError("%s() missing %i required %s argument%s: %s" %
                    (f_name, missing,
                      "positional" if pos else "keyword-only",
                      "" if missing == 1 else "s", s))

def _too_many(f_name, args, kwonly, varargs, defcount, given, values):
    atleast = len(args) - defcount
    kwonly_given = len([arg for arg in kwonly if arg in values])
    if varargs:
        plural = atleast != 1
        sig = "at least %d" % (atleast,)
    elif defcount:
        plural = True
        sig = "from %d to %d" % (atleast, len(args))
    else:
        plural = len(args) != 1
        sig = str(len(args))
    kwonly_sig = ""
    if kwonly_given:
        msg = " positional argument%s (and %d keyword-only argument%s)"
        kwonly_sig = (msg % ("s" if given != 1 else "", kwonly_given,
                             "s" if kwonly_given != 1 else ""))
    raise TypeError("%s() takes %s positional argument%s but %d%s %s given" %
            (f_name, sig, "s" if plural else "", given, kwonly_sig,
             "was" if given == 1 and not kwonly_given else "were"))

def getcallargs(func, /, *positional, **named):
    """Get the mapping of arguments to values.

    A dict is returned, with keys the function argument names (including the
    names of the * and ** arguments, if any), and values the respective bound
    values from 'positional' and 'named'."""
    spec = getfullargspec(func)
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, ann = spec
    f_name = func.__name__
    arg2value = {}


    if ismethod(func) and func.__self__ is not None:
        # implicit 'self' (or 'cls' for classmethods) argument
        positional = (func.__self__,) + positional
    num_pos = len(positional)
    num_args = len(args)
    num_defaults = len(defaults) if defaults else 0

    n = min(num_pos, num_args)
    for i in range(n):
        arg2value[args[i]] = positional[i]
    if varargs:
        arg2value[varargs] = tuple(positional[n:])
    possible_kwargs = set(args + kwonlyargs)
    if varkw:
        arg2value[varkw] = {}
    for kw, value in named.items():
        if kw not in possible_kwargs:
            if not varkw:
                raise TypeError("%s() got an unexpected keyword argument %r" %
                                (f_name, kw))
            arg2value[varkw][kw] = value
            continue
        if kw in arg2value:
            raise TypeError("%s() got multiple values for argument %r" %
                            (f_name, kw))
        arg2value[kw] = value
    if num_pos > num_args and not varargs:
        _too_many(f_name, args, kwonlyargs, varargs, num_defaults,
                   num_pos, arg2value)
    if num_pos < num_args:
        req = args[:num_args - num_defaults]
        for arg in req:
            if arg not in arg2value:
                _missing_arguments(f_name, req, True, arg2value)
        for i, arg in enumerate(args[num_args - num_defaults:]):
            if arg not in arg2value:
                arg2value[arg] = defaults[i]
    missing = 0
    for kwarg in kwonlyargs:
        if kwarg not in arg2value:
            if kwonlydefaults and kwarg in kwonlydefaults:
                arg2value[kwarg] = kwonlydefaults[kwarg]
            else:
                missing += 1
    if missing:
        _missing_arguments(f_name, kwonlyargs, False, arg2value)
    return arg2value

ClosureVars = namedtuple('ClosureVars', 'nonlocals globals builtins unbound')

def getclosurevars(func):
    """
    Get the mapping of free variables to their current values.

    Returns a named tuple of dicts mapping the current nonlocal, global
    and builtin references as seen by the body of the function. A final
    set of unbound names that could not be resolved is also provided.
    """

    if ismethod(func):
        func = func.__func__

    if not isfunction(func):
        raise TypeError("{!r} is not a Python function".format(func))

    code = func.__code__
    # Nonlocal references are named in co_freevars and resolved
    # by looking them up in __closure__ by positional index
    if func.__closure__ is None:
        nonlocal_vars = {}
    else:
        nonlocal_vars = {
            var : cell.cell_contents
            for var, cell in zip(code.co_freevars, func.__closure__)
       }

    # Global and builtin references are named in co_names and resolved
    # by looking them up in __globals__ or __builtins__
    global_ns = func.__globals__
    builtin_ns = global_ns.get("__builtins__", builtins.__dict__)
    if ismodule(builtin_ns):
        builtin_ns = builtin_ns.__dict__
    global_vars = {}
    builtin_vars = {}
    unbound_names = set()
    for name in code.co_names:
        if name in ("None", "True", "False"):
            # Because these used to be builtins instead of keywords, they
            # may still show up as name references. We ignore them.
            continue
        try:
            global_vars[name] = global_ns[name]
        except KeyError:
            try:
                builtin_vars[name] = builtin_ns[name]
            except KeyError:
                unbound_names.add(name)

    return ClosureVars(nonlocal_vars, global_vars,
                       builtin_vars, unbound_names)

# -------------------------------------------------- stack frame extraction

Traceback = namedtuple('Traceback', 'filename lineno function code_context index')

def getframeinfo(frame, context=1):
    """Get information about a frame or traceback object.

    A tuple of five things is returned: the filename, the line number of
    the current line, the function name, a list of lines of context from
    the source code, and the index of the current line within that list.
    The optional second argument specifies the number of lines of context
    to return, which are centered around the current line."""
    if istraceback(frame):
        lineno = frame.tb_lineno
        frame = frame.tb_frame
    else:
        lineno = frame.f_lineno
    if not isframe(frame):
        raise TypeError('{!r} is not a frame or traceback object'.format(frame))

    filename = getsourcefile(frame) or getfile(frame)
    if context > 0:
        start = lineno - 1 - context//2
        try:
            lines, lnum = findsource(frame)
        except OSError:
            lines = index = None
        else:
            start = max(0, min(start, len(lines) - context))
            lines = lines[start:start+context]
            index = lineno - 1 - start
    else:
        lines = index = None

    return Traceback(filename, lineno, frame.f_code.co_name, lines, index)

def getlineno(frame):
    """Get the line number from a frame object, allowing for optimization."""
    # FrameType.f_lineno is now a descriptor that grovels co_lnotab
    return frame.f_lineno

FrameInfo = namedtuple('FrameInfo', ('frame',) + Traceback._fields)

def getouterframes(frame, context=1):
    """Get a list of records for a frame and all higher (calling) frames.

    Each record contains a frame object, filename, line number, function
    name, a list of lines of context, and index within the context."""
    framelist = []
    while frame:
        frameinfo = (frame,) + getframeinfo(frame, context)
        framelist.append(FrameInfo(*frameinfo))
        frame = frame.f_back
    return framelist

def getinnerframes(tb, context=1):
    """Get a list of records for a traceback's frame and all lower frames.

    Each record contains a frame object, filename, line number, function
    name, a list of lines of context, and index within the context."""
    framelist = []
    while tb:
        frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)
        framelist.append(FrameInfo(*frameinfo))
        tb = tb.tb_next
    return framelist

def currentframe():
    """Return the frame of the caller or None if this is not possible."""
    return sys._getframe(1) if hasattr(sys, "_getframe") else None

def stack(context=1):
    """Return a list of records for the stack above the caller's frame."""
    return getouterframes(sys._getframe(1), context)

def trace(context=1):
    """Return a list of records for the stack below the current exception."""
    return getinnerframes(sys.exc_info()[2], context)


# ------------------------------------------------ static version of getattr

_sentinel = object()

def _static_getmro(klass):
    return type.__dict__['__mro__'].__get__(klass)

def _check_instance(obj, attr):
    instance_dict = {}
    try:
        instance_dict = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        pass
    return dict.get(instance_dict, attr, _sentinel)


def _check_class(klass, attr):
    for entry in _static_getmro(klass):
        if _shadowed_dict(type(entry)) is _sentinel:
            try:
                return entry.__dict__[attr]
            except KeyError:
                pass
    return _sentinel

def _is_type(obj):
    try:
        _static_getmro(obj)
    except TypeError:
        return False
    return True

def _shadowed_dict(klass):
    dict_attr = type.__dict__["__dict__"]
    for entry in _static_getmro(klass):
        try:
            class_dict = dict_attr.__get__(entry)["__dict__"]
        except KeyError:
            pass
        else:
            if not (type(class_dict) is types.GetSetDescriptorType and
                    class_dict.__name__ == "__dict__" and
                    class_dict.__objclass__ is entry):
                return class_dict
    return _sentinel

def getattr_static(obj, attr, default=_sentinel):
    """Retrieve attributes without triggering dynamic lookup via the
       descriptor protocol,  __getattr__ or __getattribute__.

       Note: this function may not be able to retrieve all attributes
       that getattr can fetch (like dynamically created attributes)
       and may find attributes that getattr can't (like descriptors
       that raise AttributeError). It can also return descriptor objects
       instead of instance members in some cases. See the
       documentation for details.
    """
    instance_result = _sentinel
    if not _is_type(obj):
        klass = type(obj)
        dict_attr = _shadowed_dict(klass)
        if (dict_attr is _sentinel or
            type(dict_attr) is types.MemberDescriptorType):
            instance_result = _check_instance(obj, attr)
    else:
        klass = obj

    klass_result = _check_class(klass, attr)

    if instance_result is not _sentinel and klass_result is not _sentinel:
        if (_check_class(type(klass_result), '__get__') is not _sentinel and
            _check_class(type(klass_result), '__set__') is not _sentinel):
            return klass_result

    if instance_result is not _sentinel:
        return instance_result
    if klass_result is not _sentinel:
        return klass_result

    if obj is klass:
        # for types we check the metaclass too
        for entry in _static_getmro(type(klass)):
            if _shadowed_dict(type(entry)) is _sentinel:
                try:
                    return entry.__dict__[attr]
                except KeyError:
                    pass
    if default is not _sentinel:
        return default
    raise AttributeError(attr)


# ------------------------------------------------ generator introspection

GEN_CREATED = 'GEN_CREATED'
GEN_RUNNING = 'GEN_RUNNING'
GEN_SUSPENDED = 'GEN_SUSPENDED'
GEN_CLOSED = 'GEN_CLOSED'

def getgeneratorstate(generator):
    """Get current state of a generator-iterator.

    Possible states are:
      GEN_CREATED: Waiting to start execution.
      GEN_RUNNING: Currently being executed by the interpreter.
      GEN_SUSPENDED: Currently suspended at a yield expression.
      GEN_CLOSED: Execution has completed.
    """
    if generator.gi_running:
        return GEN_RUNNING
    if generator.gi_frame is None:
        return GEN_CLOSED
    if generator.gi_frame.f_lasti == -1:
        return GEN_CREATED
    return GEN_SUSPENDED


def getgeneratorlocals(generator):
    """
    Get the mapping of generator local variables to their current values.

    A dict is returned, with the keys the local variable names and values the
    bound values."""

    if not isgenerator(generator):
        raise TypeError("{!r} is not a Python generator".format(generator))

    frame = getattr(generator, "gi_frame", None)
    if frame is not None:
        return generator.gi_frame.f_locals
    else:
        return {}


# ------------------------------------------------ coroutine introspection

CORO_CREATED = 'CORO_CREATED'
CORO_RUNNING = 'CORO_RUNNING'
CORO_SUSPENDED = 'CORO_SUSPENDED'
CORO_CLOSED = 'CORO_CLOSED'

def getcoroutinestate(coroutine):
    """Get current state of a coroutine object.

    Possible states are:
      CORO_CREATED: Waiting to start execution.
      CORO_RUNNING: Currently being executed by the interpreter.
      CORO_SUSPENDED: Currently suspended at an await expression.
      CORO_CLOSED: Execution has completed.
    """
    if coroutine.cr_running:
        return CORO_RUNNING
    if coroutine.cr_frame is None:
        return CORO_CLOSED
    if coroutine.cr_frame.f_lasti == -1:
        return CORO_CREATED
    return CORO_SUSPENDED


def getcoroutinelocals(coroutine):
    """
    Get the mapping of coroutine local variables to their current values.

    A dict is returned, with the keys the local variable names and values the
    bound values."""
    frame = getattr(coroutine, "cr_frame", None)
    if frame is not None:
        return frame.f_locals
    else:
        return {}


###############################################################################
### Function Signature Object (PEP 362)
###############################################################################


_WrapperDescriptor = type(type.__call__)
_MethodWrapper = type(all.__call__)
_ClassMethodWrapper = type(int.__dict__['from_bytes'])

_NonUserDefinedCallables = (_WrapperDescriptor,
                            _MethodWrapper,
                            _ClassMethodWrapper,
                            types.BuiltinFunctionType)


def _signature_get_user_defined_method(cls, method_name):
    """Private helper. Checks if ``cls`` has an attribute
    named ``method_name`` and returns it only if it is a
    pure python function.
    """
    try:
        meth = getattr(cls, method_name)
    except AttributeError:
        return
    else:
        if not isinstance(meth, _NonUserDefinedCallables):
            # Once '__signature__' will be added to 'C'-level
            # callables, this check won't be necessary
            return meth


def _signature_get_partial(wrapped_sig, partial, extra_args=()):
    """Private helper to calculate how 'wrapped_sig' signature will
    look like after applying a 'functools.partial' object (or alike)
    on it.
    """

    old_params = wrapped_sig.parameters
    new_params = OrderedDict(old_params.items())

    partial_args = partial.args or ()
    partial_keywords = partial.keywords or {}

    if extra_args:
        partial_args = extra_args + partial_args

    try:
        ba = wrapped_sig.bind_partial(*partial_args, **partial_keywords)
    except TypeError as ex:
        msg = 'partial object {!r} has incorrect arguments'.format(partial)
        raise ValueError(msg) from ex


    transform_to_kwonly = False
    for param_name, param in old_params.items():
        try:
            arg_value = ba.arguments[param_name]
        except KeyError:
            pass
        else:
            if param.kind is _POSITIONAL_ONLY:
                # If positional-only parameter is bound by partial,
                # it effectively disappears from the signature
                new_params.pop(param_name)
                continue

            if param.kind is _POSITIONAL_OR_KEYWORD:
                if param_name in partial_keywords:
                    # This means that this parameter, and all parameters
                    # after it should be keyword-only (and var-positional
                    # should be removed). Here's why. Consider the following
                    # function:
                    #     foo(a, b, *args, c):
                    #         pass
                    #
                    # "partial(foo, a='spam')" will have the following
                    # signature: "(*, a='spam', b, c)". Because attempting
                    # to call that partial with "(10, 20)" arguments will
                    # raise a TypeError, saying that "a" argument received
                    # multiple values.
                    transform_to_kwonly = True
                    # Set the new default value
                    new_params[param_name] = param.replace(default=arg_value)
                else:
                    # was passed as a positional argument
                    new_params.pop(param.name)
                    continue

            if param.kind is _KEYWORD_ONLY:
                # Set the new default value
                new_params[param_name] = param.replace(default=arg_value)

        if transform_to_kwonly:
            assert param.kind is not _POSITIONAL_ONLY

            if param.kind is _POSITIONAL_OR_KEYWORD:
                new_param = new_params[param_name].replace(kind=_KEYWORD_ONLY)
                new_params[param_name] = new_param
                new_params.move_to_end(param_name)
            elif param.kind in (_KEYWORD_ONLY, _VAR_KEYWORD):
                new_params.move_to_end(param_name)
            elif param.kind is _VAR_POSITIONAL:
                new_params.pop(param.name)

    return wrapped_sig.replace(parameters=new_params.values())


def _signature_bound_method(sig):
    """Private helper to transform signatures for unbound
    functions to bound methods.
    """

    params = tuple(sig.parameters.values())

    if not params or params[0].kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
        raise ValueError('invalid method signature')

    kind = params[0].kind
    if kind in (_POSITIONAL_OR_KEYWORD, _POSITIONAL_ONLY):
        # Drop first parameter:
        # '(p1, p2[, ...])' -> '(p2[, ...])'
        params = params[1:]
    else:
        if kind is not _VAR_POSITIONAL:
            # Unless we add a new parameter type we never
            # get here
            raise ValueError('invalid argument type')
        # It's a var-positional parameter.
        # Do nothing. '(*args[, ...])' -> '(*args[, ...])'

    return sig.replace(parameters=params)


def _signature_is_builtin(obj):
    """Private helper to test if `obj` is a callable that might
    support Argument Clinic's __text_signature__ protocol.
    """
    return (isbuiltin(obj) or
            ismethoddescriptor(obj) or
            isinstance(obj, _NonUserDefinedCallables) or
            # Can't test 'isinstance(type)' here, as it would
            # also be True for regular python classes
            obj in (type, object))


def _signature_is_functionlike(obj):
    """Private helper to test if `obj` is a duck type of FunctionType.
    A good example of such objects are functions compiled with
    Cython, which have all attributes that a pure Python function
    would have, but have their code statically compiled.
    """

    if not callable(obj) or isclass(obj):
        # All function-like objects are obviously callables,
        # and not classes.
        return False

    name = getattr(obj, '__name__', None)
    code = getattr(obj, '__code__', None)
    defaults = getattr(obj, '__defaults__', _void) # Important to use _void ...
    kwdefaults = getattr(obj, '__kwdefaults__', _void) # ... and not None here
    annotations = getattr(obj, '__annotations__', None)

    return (isinstance(code, types.CodeType) and
            isinstance(name, str) and
            (defaults is None or isinstance(defaults, tuple)) and
            (kwdefaults is None or isinstance(kwdefaults, dict)) and
            (isinstance(annotations, (dict)) or annotations is None) )


def _signature_get_bound_param(spec):
    """ Private helper to get first parameter name from a
    __text_signature__ of a builtin method, which should
    be in the following format: '($param1, ...)'.
    Assumptions are that the first argument won't have
    a default value or an annotation.
    """

    assert spec.startswith('($')

    pos = spec.find(',')
    if pos == -1:
        pos = spec.find(')')

    cpos = spec.find(':')
    assert cpos == -1 or cpos > pos

    cpos = spec.find('=')
    assert cpos == -1 or cpos > pos

    return spec[2:pos]


def _signature_strip_non_python_syntax(signature):
    """
    Private helper function. Takes a signature in Argument Clinic's
    extended signature format.

    Returns a tuple of three things:
      * that signature re-rendered in standard Python syntax,
      * the index of the "self" parameter (generally 0), or None if
        the function does not have a "self" parameter, and
      * the index of the last "positional only" parameter,
        or None if the signature has no positional-only parameters.
    """

    if not signature:
        return signature, None, None

    self_parameter = None
    last_positional_only = None

    lines = [l.encode('ascii') for l in signature.split('\n') if l]
    generator = iter(lines).__next__
    token_stream = tokenize.tokenize(generator)

    delayed_comma = False
    skip_next_comma = False
    text = []
    add = text.append

    current_parameter = 0
    OP = token.OP
    ERRORTOKEN = token.ERRORTOKEN

    # token stream always starts with ENCODING token, skip it
    t = next(token_stream)
    assert t.type == tokenize.ENCODING

    for t in token_stream:
        type, string = t.type, t.string

        if type == OP:
            if string == ',':
                if skip_next_comma:
                    skip_next_comma = False
                else:
                    assert not delayed_comma
                    delayed_comma = True
                    current_parameter += 1
                continue

            if string == '/':
                assert not skip_next_comma
                assert last_positional_only is None
                skip_next_comma = True
                last_positional_only = current_parameter - 1
                continue

        if (type == ERRORTOKEN) and (string == '$'):
            assert self_parameter is None
            self_parameter = current_parameter
            continue

        if delayed_comma:
            delayed_comma = False
            if not ((type == OP) and (string == ')')):
                add(', ')
        add(string)
        if (string == ','):
            add(' ')
    clean_signature = ''.join(text)
    return clean_signature, self_parameter, last_positional_only


def _signature_fromstr(cls, obj, s, skip_bound_arg=True):
    """Private helper to parse content of '__text_signature__'
    and return a Signature based on it.
    """
    # Lazy import ast because it's relatively heavy and
    # it's not used for other than this function.
    import ast

    Parameter = cls._parameter_cls

    clean_signature, self_parameter, last_positional_only = \
        _signature_strip_non_python_syntax(s)

    program = "def foo" + clean_signature + ": pass"

    try:
        module = ast.parse(program)
    except SyntaxError:
        module = None

    if not isinstance(module, ast.Module):
        raise ValueError("{!r} builtin has invalid signature".format(obj))

    f = module.body[0]

    parameters = []
    empty = Parameter.empty

    module = None
    module_dict = {}
    module_name = getattr(obj, '__module__', None)
    if module_name:
        module = sys.modules.get(module_name, None)
        if module:
            module_dict = module.__dict__
    sys_module_dict = sys.modules.copy()

    def parse_name(node):
        assert isinstance(node, ast.arg)
        if node.annotation is not None:
            raise ValueError("Annotations are not currently supported")
        return node.arg

    def wrap_value(s):
        try:
            value = eval(s, module_dict)
        except NameError:
            try:
                value = eval(s, sys_module_dict)
            except NameError:
                raise ValueError

        if isinstance(value, (str, int, float, bytes, bool, type(None))):
            return ast.Constant(value)
        raise ValueError

    class RewriteSymbolics(ast.NodeTransformer):
        def visit_Attribute(self, node):
            a = []
            n = node
            while isinstance(n, ast.Attribute):
                a.append(n.attr)
                n = n.value
            if not isinstance(n, ast.Name):
                raise ValueError
            a.append(n.id)
            value = ".".join(reversed(a))
            return wrap_value(value)

        def visit_Name(self, node):
            if not isinstance(node.ctx, ast.Load):
                raise ValueError()
            return wrap_value(node.id)

        def visit_BinOp(self, node):
            # Support constant folding of a couple simple binary operations
            # commonly used to define default values in text signatures
            left = self.visit(node.left)
            right = self.visit(node.right)
            if not isinstance(left, ast.Constant) or not isinstance(right, ast.Constant):
                raise ValueError
            if isinstance(node.op, ast.Add):
                return ast.Constant(left.value + right.value)
            elif isinstance(node.op, ast.Sub):
                return ast.Constant(left.value - right.value)
            elif isinstance(node.op, ast.BitOr):
                return ast.Constant(left.value | right.value)
            raise ValueError

    def p(name_node, default_node, default=empty):
        name = parse_name(name_node)
        if default_node and default_node is not _empty:
            try:
                default_node = RewriteSymbolics().visit(default_node)
                default = ast.literal_eval(default_node)
            except ValueError:
                raise ValueError("{!r} builtin has invalid signature".format(obj)) from None
        parameters.append(Parameter(name, kind, default=default, annotation=empty))

    # non-keyword-only parameters
    args = reversed(f.args.args)
    defaults = reversed(f.args.defaults)
    iter = itertools.zip_longest(args, defaults, fillvalue=None)
    if last_positional_only is not None:
        kind = Parameter.POSITIONAL_ONLY
    else:
        kind = Parameter.POSITIONAL_OR_KEYWORD
    for i, (name, default) in enumerate(reversed(list(iter))):
        p(name, default)
        if i == last_positional_only:
            kind = Parameter.POSITIONAL_OR_KEYWORD

    # *args
    if f.args.vararg:
        kind = Parameter.VAR_POSITIONAL
        p(f.args.vararg, empty)

    # keyword-only arguments
    kind = Parameter.KEYWORD_ONLY
    for name, default in zip(f.args.kwonlyargs, f.args.kw_defaults):
        p(name, default)

    # **kwargs
    if f.args.kwarg:
        kind = Parameter.VAR_KEYWORD
        p(f.args.kwarg, empty)

    if self_parameter is not None:
        # Possibly strip the bound argument:
        #    - We *always* strip first bound argument if
        #      it is a module.
        #    - We don't strip first bound argument if
        #      skip_bound_arg is False.
        assert parameters
        _self = getattr(obj, '__self__', None)
        self_isbound = _self is not None
        self_ismodule = ismodule(_self)
        if self_isbound and (self_ismodule or skip_bound_arg):
            parameters.pop(0)
        else:
            # for builtins, self parameter is always positional-only!
            p = parameters[0].replace(kind=Parameter.POSITIONAL_ONLY)
            parameters[0] = p

    return cls(parameters, return_annotation=cls.empty)


def _signature_from_builtin(cls, func, skip_bound_arg=True):
    """Private helper function to get signature for
    builtin callables.
    """

    if not _signature_is_builtin(func):
        raise TypeError("{!r} is not a Python builtin "
                        "function".format(func))

    s = getattr(func, "__text_signature__", None)
    if not s:
        raise ValueError("no signature found for builtin {!r}".format(func))

    return _signature_fromstr(cls, func, s, skip_bound_arg)


def _signature_from_function(cls, func, skip_bound_arg=True,
                             globals=None, locals=None, eval_str=False):
    """Private helper: constructs Signature for the given python function."""

    is_duck_function = False
    if not isfunction(func):
        if _signature_is_functionlike(func):
            is_duck_function = True
        else:
            # If it's not a pure Python function, and not a duck type
            # of pure function:
            raise TypeError('{!r} is not a Python function'.format(func))

    s = getattr(func, "__text_signature__", None)
    if s:
        return _signature_fromstr(cls, func, s, skip_bound_arg)

    Parameter = cls._parameter_cls

    # Parameter information.
    func_code = func.__code__
    pos_count = func_code.co_argcount
    arg_names = func_code.co_varnames
    posonly_count = func_code.co_posonlyargcount
    positional = arg_names[:pos_count]
    keyword_only_count = func_code.co_kwonlyargcount
    keyword_only = arg_names[pos_count:pos_count + keyword_only_count]
    annotations = get_annotations(func, globals=globals, locals=locals, eval_str=eval_str)
    defaults = func.__defaults__
    kwdefaults = func.__kwdefaults__

    if defaults:
        pos_default_count = len(defaults)
    else:
        pos_default_count = 0

    parameters = []

    non_default_count = pos_count - pos_default_count
    posonly_left = posonly_count

    # Non-keyword-only parameters w/o defaults.
    for name in positional[:non_default_count]:
        kind = _POSITIONAL_ONLY if posonly_left else _POSITIONAL_OR_KEYWORD
        annotation = annotations.get(name, _empty)
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=kind))
        if posonly_left:
            posonly_left -= 1

    # ... w/ defaults.
    for offset, name in enumerate(positional[non_default_count:]):
        kind = _POSITIONAL_ONLY if posonly_left else _POSITIONAL_OR_KEYWORD
        annotation = annotations.get(name, _empty)
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=kind,
                                    default=defaults[offset]))
        if posonly_left:
            posonly_left -= 1

    # *args
    if func_code.co_flags & CO_VARARGS:
        name = arg_names[pos_count + keyword_only_count]
        annotation = annotations.get(name, _empty)
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=_VAR_POSITIONAL))

    # Keyword-only parameters.
    for name in keyword_only:
        default = _empty
        if kwdefaults is not None:
            default = kwdefaults.get(name, _empty)

        annotation = annotations.get(name, _empty)
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=_KEYWORD_ONLY,
                                    default=default))
    # **kwargs
    if func_code.co_flags & CO_VARKEYWORDS:
        index = pos_count + keyword_only_count
        if func_code.co_flags & CO_VARARGS:
            index += 1

        name = arg_names[index]
        annotation = annotations.get(name, _empty)
        parameters.append(Parameter(name, annotation=annotation,
                                    kind=_VAR_KEYWORD))

    # Is 'func' is a pure Python function - don't validate the
    # parameters list (for correct order and defaults), it should be OK.
    return cls(parameters,
               return_annotation=annotations.get('return', _empty),
               __validate_parameters__=is_duck_function)


def _signature_from_callable(obj, *,
                             follow_wrapper_chains=True,
                             skip_bound_arg=True,
                             globals=None,
                             locals=None,
                             eval_str=False,
                             sigcls):

    """Private helper function to get signature for arbitrary
    callable objects.
    """

    _get_signature_of = functools.partial(_signature_from_callable,
                                follow_wrapper_chains=follow_wrapper_chains,
                                skip_bound_arg=skip_bound_arg,
                                globals=globals,
                                locals=locals,
                                sigcls=sigcls,
                                eval_str=eval_str)

    if not callable(obj):
        raise TypeError('{!r} is not a callable object'.format(obj))

    if isinstance(obj, types.MethodType):
        # In this case we skip the first parameter of the underlying
        # function (usually `self` or `cls`).
        sig = _get_signature_of(obj.__func__)

        if skip_bound_arg:
            return _signature_bound_method(sig)
        else:
            return sig

    # Was this function wrapped by a decorator?
    if follow_wrapper_chains:
        # Unwrap until we find an explicit signature or a MethodType (which will be
        # handled explicitly below).
        obj = unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")
                                or isinstance(f, types.MethodType)))
        if isinstance(obj, types.MethodType):
            # If the unwrapped object is a *method*, we might want to
            # skip its first parameter (self).
            # See test_signature_wrapped_bound_method for details.
            return _get_signature_of(obj)

    try:
        sig = obj.__signature__
    except AttributeError:
        pass
    else:
        if sig is not None:
            if not isinstance(sig, Signature):
                raise TypeError(
                    'unexpected object {!r} in __signature__ '
                    'attribute'.format(sig))
            return sig

    try:
        partialmethod = obj._partialmethod
    except AttributeError:
        pass
    else:
        if isinstance(partialmethod, functools.partialmethod):
            # Unbound partialmethod (see functools.partialmethod)
            # This means, that we need to calculate the signature
            # as if it's a regular partial object, but taking into
            # account that the first positional argument
            # (usually `self`, or `cls`) will not be passed
            # automatically (as for boundmethods)

            wrapped_sig = _get_signature_of(partialmethod.func)

            sig = _signature_get_partial(wrapped_sig, partialmethod, (None,))
            first_wrapped_param = tuple(wrapped_sig.parameters.values())[0]
            if first_wrapped_param.kind is Parameter.VAR_POSITIONAL:
                # First argument of the wrapped callable is `*args`, as in
                # `partialmethod(lambda *args)`.
                return sig
            else:
                sig_params = tuple(sig.parameters.values())
                assert (not sig_params or
                        first_wrapped_param is not sig_params[0])
                new_params = (first_wrapped_param,) + sig_params
                return sig.replace(parameters=new_params)

    if isfunction(obj) or _signature_is_functionlike(obj):
        # If it's a pure Python function, or an object that is duck type
        # of a Python function (Cython functions, for instance), then:
        return _signature_from_function(sigcls, obj,
                                        skip_bound_arg=skip_bound_arg,
                                        globals=globals, locals=locals, eval_str=eval_str)

    if _signature_is_builtin(obj):
        return _signature_from_builtin(sigcls, obj,
                                       skip_bound_arg=skip_bound_arg)

    if isinstance(obj, functools.partial):
        wrapped_sig = _get_signature_of(obj.func)
        return _signature_get_partial(wrapped_sig, obj)

    sig = None
    if isinstance(obj, type):
        # obj is a class or a metaclass

        # First, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = _signature_get_user_defined_method(type(obj), '__call__')
        if call is not None:
            sig = _get_signature_of(call)
        else:
            factory_method = None
            new = _signature_get_user_defined_method(obj, '__new__')
            init = _signature_get_user_defined_method(obj, '__init__')
            # Now we check if the 'obj' class has an own '__new__' method
            if '__new__' in obj.__dict__:
                factory_method = new
            # or an own '__init__' method
            elif '__init__' in obj.__dict__:
                factory_method = init
            # If not, we take inherited '__new__' or '__init__', if present
            elif new is not None:
                factory_method = new
            elif init is not None:
                factory_method = init

            if factory_method is not None:
                sig = _get_signature_of(factory_method)

        if sig is None:
            # At this point we know, that `obj` is a class, with no user-
            # defined '__init__', '__new__', or class-level '__call__'

            for base in obj.__mro__[:-1]:
                # Since '__text_signature__' is implemented as a
                # descriptor that extracts text signature from the
                # class docstring, if 'obj' is derived from a builtin
                # class, its own '__text_signature__' may be 'None'.
                # Therefore, we go through the MRO (except the last
                # class in there, which is 'object') to find the first
                # class with non-empty text signature.
                try:
                    text_sig = base.__text_signature__
                except AttributeError:
                    pass
                else:
                    if text_sig:
                        # If 'base' class has a __text_signature__ attribute:
                        # return a signature based on it
                        return _signature_fromstr(sigcls, base, text_sig)

            # No '__text_signature__' was found for the 'obj' class.
            # Last option is to check if its '__init__' is
            # object.__init__ or type.__init__.
            if type not in obj.__mro__:
                # We have a class (not metaclass), but no user-defined
                # __init__ or __new__ for it
                if (obj.__init__ is object.__init__ and
                    obj.__new__ is object.__new__):
                    # Return a signature of 'object' builtin.
                    return sigcls.from_callable(object)
                else:
                    raise ValueError(
                        'no signature found for builtin type {!r}'.format(obj))

    elif not isinstance(obj, _NonUserDefinedCallables):
        # An object with __call__
        # We also check that the 'obj' is not an instance of
        # _WrapperDescriptor or _MethodWrapper to avoid
        # infinite recursion (and even potential segfault)
        call = _signature_get_user_defined_method(type(obj), '__call__')
        if call is not None:
            try:
                sig = _get_signature_of(call)
            except ValueError as ex:
                msg = 'no signature found for {!r}'.format(obj)
                raise ValueError(msg) from ex

    if sig is not None:
        # For classes and objects we skip the first parameter of their
        # __call__, __new__, or __init__ methods
        if skip_bound_arg:
            return _signature_bound_method(sig)
        else:
            return sig

    if isinstance(obj, types.BuiltinFunctionType):
        # Raise a nicer error message for builtins
        msg = 'no signature found for builtin function {!r}'.format(obj)
        raise ValueError(msg)

    raise ValueError('callable {!r} is not supported by signature'.format(obj))


class _void:
    """A private marker - used in Parameter & Signature."""


class _empty:
    """Marker object for Signature.empty and Parameter.empty."""


class _ParameterKind(enum.IntEnum):
    POSITIONAL_ONLY = 0
    POSITIONAL_OR_KEYWORD = 1
    VAR_POSITIONAL = 2
    KEYWORD_ONLY = 3
    VAR_KEYWORD = 4

    def __str__(self):
        return self._name_

    @property
    def description(self):
        return _PARAM_NAME_MAPPING[self]

_POSITIONAL_ONLY         = _ParameterKind.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD   = _ParameterKind.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL          = _ParameterKind.VAR_POSITIONAL
_KEYWORD_ONLY            = _ParameterKind.KEYWORD_ONLY
_VAR_KEYWORD             = _ParameterKind.VAR_KEYWORD

_PARAM_NAME_MAPPING = {
    _POSITIONAL_ONLY: 'positional-only',
    _POSITIONAL_OR_KEYWORD: 'positional or keyword',
    _VAR_POSITIONAL: 'variadic positional',
    _KEYWORD_ONLY: 'keyword-only',
    _VAR_KEYWORD: 'variadic keyword'
}


class Parameter:
    """Represents a parameter in a function signature.

    Has the following public attributes:

    * name : str
        The name of the parameter as a string.
    * default : object
        The default value for the parameter if specified.  If the
        parameter has no default value, this attribute is set to
        `Parameter.empty`.
    * annotation
        The annotation for the parameter if specified.  If the
        parameter has no annotation, this attribute is set to
        `Parameter.empty`.
    * kind : str
        Describes how argument values are bound to the parameter.
        Possible values: `Parameter.POSITIONAL_ONLY`,
        `Parameter.POSITIONAL_OR_KEYWORD`, `Parameter.VAR_POSITIONAL`,
        `Parameter.KEYWORD_ONLY`, `Parameter.VAR_KEYWORD`.
    """

    __slots__ = ('_name', '_kind', '_default', '_annotation')

    POSITIONAL_ONLY         = _POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD   = _POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL          = _VAR_POSITIONAL
    KEYWORD_ONLY            = _KEYWORD_ONLY
    VAR_KEYWORD             = _VAR_KEYWORD

    empty = _empty

    def __init__(self, name, kind, *, default=_empty, annotation=_empty):
        try:
            self._kind = _ParameterKind(kind)
        except ValueError:
            raise ValueError(f'value {kind!r} is not a valid Parameter.kind')
        if default is not _empty:
            if self._kind in (_VAR_POSITIONAL, _VAR_KEYWORD):
                msg = '{} parameters cannot have default values'
                msg = msg.format(self._kind.description)
                raise ValueError(msg)
        self._default = default
        self._annotation = annotation

        if name is _empty:
            raise ValueError('name is a required attribute for Parameter')

        if not isinstance(name, str):
            msg = 'name must be a str, not a {}'.format(type(name).__name__)
            raise TypeError(msg)

        if name[0] == '.' and name[1:].isdigit():
            # These are implicit arguments generated by comprehensions. In
            # order to provide a friendlier interface to users, we recast
            # their name as "implicitN" and treat them as positional-only.
            # See issue 19611.
            if self._kind != _POSITIONAL_OR_KEYWORD:
                msg = (
                    'implicit arguments must be passed as '
                    'positional or keyword arguments, not {}'
                )
                msg = msg.format(self._kind.description)
                raise ValueError(msg)
            self._kind = _POSITIONAL_ONLY
            name = 'implicit{}'.format(name[1:])

        if not name.isidentifier():
            raise ValueError('{!r} is not a valid parameter name'.format(name))

        self._name = name

    def __reduce__(self):
        return (type(self),
                (self._name, self._kind),
                {'_default': self._default,
                 '_annotation': self._annotation})

    def __setstate__(self, state):
        self._default = state['_default']
        self._annotation = state['_annotation']

    @property
    def name(self):
        return self._name

    @property
    def default(self):
        return self._default

    @property
    def annotation(self):
        return self._annotation

    @property
    def kind(self):
        return self._kind

    def replace(self, *, name=_void, kind=_void,
                annotation=_void, default=_void):
        """Creates a customized copy of the Parameter."""

        if name is _void:
            name = self._name

        if kind is _void:
            kind = self._kind

        if annotation is _void:
            annotation = self._annotation

        if default is _void:
            default = self._default

        return type(self)(name, kind, default=default, annotation=annotation)

    def __str__(self):
        kind = self.kind
        formatted = self._name

        # Add annotation and default value
        if self._annotation is not _empty:
            formatted = '{}: {}'.format(formatted,
                                       formatannotation(self._annotation))

        if self._default is not _empty:
            if self._annotation is not _empty:
                formatted = '{} = {}'.format(formatted, repr(self._default))
            else:
                formatted = '{}={}'.format(formatted, repr(self._default))

        if kind == _VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == _VAR_KEYWORD:
            formatted = '**' + formatted

        return formatted

    def __repr__(self):
        return '<{} "{}">'.format(self.__class__.__name__, self)

    def __hash__(self):
        return hash((self.name, self.kind, self.annotation, self.default))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Parameter):
            return NotImplemented
        return (self._name == other._name and
                self._kind == other._kind and
                self._default == other._default and
                self._annotation == other._annotation)


class BoundArguments:
    """Result of `Signature.bind` call.  Holds the mapping of arguments
    to the function's parameters.

    Has the following public attributes:

    * arguments : dict
        An ordered mutable mapping of parameters' names to arguments' values.
        Does not contain arguments' default values.
    * signature : Signature
        The Signature object that created this instance.
    * args : tuple
        Tuple of positional arguments values.
    * kwargs : dict
        Dict of keyword arguments values.
    """

    __slots__ = ('arguments', '_signature', '__weakref__')

    def __init__(self, signature, arguments):
        self.arguments = arguments
        self._signature = signature

    @property
    def signature(self):
        return self._signature

    @property
    def args(self):
        args = []
        for param_name, param in self._signature.parameters.items():
            if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                break

            try:
                arg = self.arguments[param_name]
            except KeyError:
                # We're done here. Other arguments
                # will be mapped in 'BoundArguments.kwargs'
                break
            else:
                if param.kind == _VAR_POSITIONAL:
                    # *args
                    args.extend(arg)
                else:
                    # plain argument
                    args.append(arg)

        return tuple(args)

    @property
    def kwargs(self):
        kwargs = {}
        kwargs_started = False
        for param_name, param in self._signature.parameters.items():
            if not kwargs_started:
                if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                    kwargs_started = True
                else:
                    if param_name not in self.arguments:
                        kwargs_started = True
                        continue

            if not kwargs_started:
                continue

            try:
                arg = self.arguments[param_name]
            except KeyError:
                pass
            else:
                if param.kind == _VAR_KEYWORD:
                    # **kwargs
                    kwargs.update(arg)
                else:
                    # plain keyword argument
                    kwargs[param_name] = arg

        return kwargs

    def apply_defaults(self):
        """Set default values for missing arguments.

        For variable-positional arguments (*args) the default is an
        empty tuple.

        For variable-keyword arguments (**kwargs) the default is an
        empty dict.
        """
        arguments = self.arguments
        new_arguments = []
        for name, param in self._signature.parameters.items():
            try:
                new_arguments.append((name, arguments[name]))
            except KeyError:
                if param.default is not _empty:
                    val = param.default
                elif param.kind is _VAR_POSITIONAL:
                    val = ()
                elif param.kind is _VAR_KEYWORD:
                    val = {}
                else:
                    # This BoundArguments was likely produced by
                    # Signature.bind_partial().
                    continue
                new_arguments.append((name, val))
        self.arguments = dict(new_arguments)

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, BoundArguments):
            return NotImplemented
        return (self.signature == other.signature and
                self.arguments == other.arguments)

    def __setstate__(self, state):
        self._signature = state['_signature']
        self.arguments = state['arguments']

    def __getstate__(self):
        return {'_signature': self._signature, 'arguments': self.arguments}

    def __repr__(self):
        args = []
        for arg, value in self.arguments.items():
            args.append('{}={!r}'.format(arg, value))
        return '<{} ({})>'.format(self.__class__.__name__, ', '.join(args))


class Signature:
    """A Signature object represents the overall signature of a function.
    It stores a Parameter object for each parameter accepted by the
    function, as well as information specific to the function itself.

    A Signature object has the following public attributes and methods:

    * parameters : OrderedDict
        An ordered mapping of parameters' names to the corresponding
        Parameter objects (keyword-only arguments are in the same order
        as listed in `code.co_varnames`).
    * return_annotation : object
        The annotation for the return type of the function if specified.
        If the function has no annotation for its return type, this
        attribute is set to `Signature.empty`.
    * bind(*args, **kwargs) -> BoundArguments
        Creates a mapping from positional and keyword arguments to
        parameters.
    * bind_partial(*args, **kwargs) -> BoundArguments
        Creates a partial mapping from positional and keyword arguments
        to parameters (simulating 'functools.partial' behavior.)
    """

    __slots__ = ('_return_annotation', '_parameters')

    _parameter_cls = Parameter
    _bound_arguments_cls = BoundArguments

    empty = _empty

    def __init__(self, parameters=None, *, return_annotation=_empty,
                 __validate_parameters__=True):
        """Constructs Signature from the given list of Parameter
        objects and 'return_annotation'.  All arguments are optional.
        """

        if parameters is None:
            params = OrderedDict()
        else:
            if __validate_parameters__:
                params = OrderedDict()
                top_kind = _POSITIONAL_ONLY
                kind_defaults = False

                for param in parameters:
                    kind = param.kind
                    name = param.name

                    if kind < top_kind:
                        msg = (
                            'wrong parameter order: {} parameter before {} '
                            'parameter'
                        )
                        msg = msg.format(top_kind.description,
                                         kind.description)
                        raise ValueError(msg)
                    elif kind > top_kind:
                        kind_defaults = False
                        top_kind = kind

                    if kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD):
                        if param.default is _empty:
                            if kind_defaults:
                                # No default for this parameter, but the
                                # previous parameter of the same kind had
                                # a default
                                msg = 'non-default argument follows default ' \
                                      'argument'
                                raise ValueError(msg)
                        else:
                            # There is a default for this parameter.
                            kind_defaults = True

                    if name in params:
                        msg = 'duplicate parameter name: {!r}'.format(name)
                        raise ValueError(msg)

                    params[name] = param
            else:
                params = OrderedDict((param.name, param) for param in parameters)

        self._parameters = types.MappingProxyType(params)
        self._return_annotation = return_annotation

    @classmethod
    def from_function(cls, func):
        """Constructs Signature for the given python function.

        Deprecated since Python 3.5, use `Signature.from_callable()`.
        """

        warnings.warn("inspect.Signature.from_function() is deprecated since "
                      "Python 3.5, use Signature.from_callable()",
                      DeprecationWarning, stacklevel=2)
        return _signature_from_function(cls, func)

    @classmethod
    def from_builtin(cls, func):
        """Constructs Signature for the given builtin function.

        Deprecated since Python 3.5, use `Signature.from_callable()`.
        """

        warnings.warn("inspect.Signature.from_builtin() is deprecated since "
                      "Python 3.5, use Signature.from_callable()",
                      DeprecationWarning, stacklevel=2)
        return _signature_from_builtin(cls, func)

    @classmethod
    def from_callable(cls, obj, *,
                      follow_wrapped=True, globals=None, locals=None, eval_str=False):
        """Constructs Signature for the given callable object."""
        return _signature_from_callable(obj, sigcls=cls,
                                        follow_wrapper_chains=follow_wrapped,
                                        globals=globals, locals=locals, eval_str=eval_str)

    @property
    def parameters(self):
        return self._parameters

    @property
    def return_annotation(self):
        return self._return_annotation

    def replace(self, *, parameters=_void, return_annotation=_void):
        """Creates a customized copy of the Signature.
        Pass 'parameters' and/or 'return_annotation' arguments
        to override them in the new copy.
        """

        if parameters is _void:
            parameters = self.parameters.values()

        if return_annotation is _void:
            return_annotation = self._return_annotation

        return type(self)(parameters,
                          return_annotation=return_annotation)

    def _hash_basis(self):
        params = tuple(param for param in self.parameters.values()
                             if param.kind != _KEYWORD_ONLY)

        kwo_params = {param.name: param for param in self.parameters.values()
                                        if param.kind == _KEYWORD_ONLY}

        return params, kwo_params, self.return_annotation

    def __hash__(self):
        params, kwo_params, return_annotation = self._hash_basis()
        kwo_params = frozenset(kwo_params.values())
        return hash((params, kwo_params, return_annotation))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Signature):
            return NotImplemented
        return self._hash_basis() == other._hash_basis()

    def _bind(self, args, kwargs, *, partial=False):
        """Private method. Don't use directly."""

        arguments = {}

        parameters = iter(self.parameters.values())
        parameters_ex = ()
        arg_vals = iter(args)

        while True:
            # Let's iterate through the positional arguments and corresponding
            # parameters
            try:
                arg_val = next(arg_vals)
            except StopIteration:
                # No more positional arguments
                try:
                    param = next(parameters)
                except StopIteration:
                    # No more parameters. That's it. Just need to check that
                    # we have no `kwargs` after this while loop
                    break
                else:
                    if param.kind == _VAR_POSITIONAL:
                        # That's OK, just empty *args.  Let's start parsing
                        # kwargs
                        break
                    elif param.name in kwargs:
                        if param.kind == _POSITIONAL_ONLY:
                            msg = '{arg!r} parameter is positional only, ' \
                                  'but was passed as a keyword'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg) from None
                        parameters_ex = (param,)
                        break
                    elif (param.kind == _VAR_KEYWORD or
                                                param.default is not _empty):
                        # That's fine too - we have a default value for this
                        # parameter.  So, lets start parsing `kwargs`, starting
                        # with the current parameter
                        parameters_ex = (param,)
                        break
                    else:
                        # No default, not VAR_KEYWORD, not VAR_POSITIONAL,
                        # not in `kwargs`
                        if partial:
                            parameters_ex = (param,)
                            break
                        else:
                            msg = 'missing a required argument: {arg!r}'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg) from None
            else:
                # We have a positional argument to process
                try:
                    param = next(parameters)
                except StopIteration:
                    raise TypeError('too many positional arguments') from None
                else:
                    if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                        # Looks like we have no parameter for this positional
                        # argument
                        raise TypeError(
                            'too many positional arguments') from None

                    if param.kind == _VAR_POSITIONAL:
                        # We have an '*args'-like argument, let's fill it with
                        # all positional arguments we have left and move on to
                        # the next phase
                        values = [arg_val]
                        values.extend(arg_vals)
                        arguments[param.name] = tuple(values)
                        break

                    if param.name in kwargs and param.kind != _POSITIONAL_ONLY:
                        raise TypeError(
                            'multiple values for argument {arg!r}'.format(
                                arg=param.name)) from None

                    arguments[param.name] = arg_val

        # Now, we iterate through the remaining parameters to process
        # keyword arguments
        kwargs_param = None
        for param in itertools.chain(parameters_ex, parameters):
            if param.kind == _VAR_KEYWORD:
                # Memorize that we have a '**kwargs'-like parameter
                kwargs_param = param
                continue

            if param.kind == _VAR_POSITIONAL:
                # Named arguments don't refer to '*args'-like parameters.
                # We only arrive here if the positional arguments ended
                # before reaching the last parameter before *args.
                continue

            param_name = param.name
            try:
                arg_val = kwargs.pop(param_name)
            except KeyError:
                # We have no value for this parameter.  It's fine though,
                # if it has a default value, or it is an '*args'-like
                # parameter, left alone by the processing of positional
                # arguments.
                if (not partial and param.kind != _VAR_POSITIONAL and
                                                    param.default is _empty):
                    raise TypeError('missing a required argument: {arg!r}'. \
                                    format(arg=param_name)) from None

            else:
                if param.kind == _POSITIONAL_ONLY:
                    # This should never happen in case of a properly built
                    # Signature object (but let's have this check here
                    # to ensure correct behaviour just in case)
                    raise TypeError('{arg!r} parameter is positional only, '
                                    'but was passed as a keyword'. \
                                    format(arg=param.name))

                arguments[param_name] = arg_val

        if kwargs:
            if kwargs_param is not None:
                # Process our '**kwargs'-like parameter
                arguments[kwargs_param.name] = kwargs
            else:
                raise TypeError(
                    'got an unexpected keyword argument {arg!r}'.format(
                        arg=next(iter(kwargs))))

        return self._bound_arguments_cls(self, arguments)

    def bind(self, /, *args, **kwargs):
        """Get a BoundArguments object, that maps the passed `args`
        and `kwargs` to the function's signature.  Raises `TypeError`
        if the passed arguments can not be bound.
        """
        return self._bind(args, kwargs)

    def bind_partial(self, /, *args, **kwargs):
        """Get a BoundArguments object, that partially maps the
        passed `args` and `kwargs` to the function's signature.
        Raises `TypeError` if the passed arguments can not be bound.
        """
        return self._bind(args, kwargs, partial=True)

    def __reduce__(self):
        return (type(self),
                (tuple(self._parameters.values()),),
                {'_return_annotation': self._return_annotation})

    def __setstate__(self, state):
        self._return_annotation = state['_return_annotation']

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, self)

    def __str__(self):
        result = []
        render_pos_only_separator = False
        render_kw_only_separator = True
        for param in self.parameters.values():
            formatted = str(param)

            kind = param.kind

            if kind == _POSITIONAL_ONLY:
                render_pos_only_separator = True
            elif render_pos_only_separator:
                # It's not a positional-only parameter, and the flag
                # is set to 'True' (there were pos-only params before.)
                result.append('/')
                render_pos_only_separator = False

            if kind == _VAR_POSITIONAL:
                # OK, we have an '*args'-like parameter, so we won't need
                # a '*' to separate keyword-only arguments
                render_kw_only_separator = False
            elif kind == _KEYWORD_ONLY and render_kw_only_separator:
                # We have a keyword-only parameter to render and we haven't
                # rendered an '*args'-like parameter before, so add a '*'
                # separator to the parameters list ("foo(arg1, *, arg2)" case)
                result.append('*')
                # This condition should be only triggered once, so
                # reset the flag
                render_kw_only_separator = False

            result.append(formatted)

        if render_pos_only_separator:
            # There were only positional-only parameters, hence the
            # flag was not reset to 'False'
            result.append('/')

        rendered = '({})'.format(', '.join(result))

        if self.return_annotation is not _empty:
            anno = formatannotation(self.return_annotation)
            rendered += ' -> {}'.format(anno)

        return rendered


def signature(obj, *, follow_wrapped=True, globals=None, locals=None, eval_str=False):
    """Get a signature object for the passed callable."""
    return Signature.from_callable(obj, follow_wrapped=follow_wrapped,
                                   globals=globals, locals=locals, eval_str=eval_str)


def _main():
    """ Logic for inspecting an object given at command line """
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'object',
         help="The object to be analysed. "
              "It supports the 'module:qualname' syntax")
    parser.add_argument(
        '-d', '--details', action='store_true',
        help='Display info about the module rather than its source code')

    args = parser.parse_args()

    target = args.object
    mod_name, has_attrs, attrs = target.partition(":")
    try:
        obj = module = importlib.import_module(mod_name)
    except Exception as exc:
        msg = "Failed to import {} ({}: {})".format(mod_name,
                                                    type(exc).__name__,
                                                    exc)
        print(msg, file=sys.stderr)
        sys.exit(2)

    if has_attrs:
        parts = attrs.split(".")
        obj = module
        for part in parts:
            obj = getattr(obj, part)

    if module.__name__ in sys.builtin_module_names:
        print("Can't get info for builtin modules.", file=sys.stderr)
        sys.exit(1)

    if args.details:
        print('Target: {}'.format(target))
        print('Origin: {}'.format(getsourcefile(module)))
        print('Cached: {}'.format(module.__cached__))
        if obj is module:
            print('Loader: {}'.format(repr(module.__loader__)))
            if hasattr(module, '__path__'):
                print('Submodule search path: {}'.format(module.__path__))
        else:
            try:
                __, lineno = findsource(obj)
            except Exception:
                pass
            else:
                print('Line: {}'.format(lineno))

        print('\n')
    else:
        print(getsource(obj))


if __name__ == "__main__":
    _main()
<|endoftext|>"""Generic socket server classes.

This module tries to capture the various aspects of defining a server:

For socket-based servers:

- address family:
        - AF_INET{,6}: IP (Internet Protocol) sockets (default)
        - AF_UNIX: Unix domain sockets
        - others, e.g. AF_DECNET are conceivable (see <socket.h>
- socket type:
        - SOCK_STREAM (reliable stream, e.g. TCP)
        - SOCK_DGRAM (datagrams, e.g. UDP)

For request-based servers (including socket-based):

- client address verification before further looking at the request
        (This is actually a hook for any processing that needs to look
         at the request before anything else, e.g. logging)
- how to handle multiple requests:
        - synchronous (one request is handled at a time)
        - forking (each request is handled by a new process)
        - threading (each request is handled by a new thread)

The classes in this module favor the server type that is simplest to
write: a synchronous TCP/IP server.  This is bad class design, but
saves some typing.  (There's also the issue that a deep class hierarchy
slows down method lookups.)

There are five classes in an inheritance diagram, four of which represent
synchronous servers of four types:

        +------------+
        | BaseServer |
        +------------+
              |
              v
        +-----------+        +------------------+
        | TCPServer |------->| UnixStreamServer |
        +-----------+        +------------------+
              |
              v
        +-----------+        +--------------------+
        | UDPServer |------->| UnixDatagramServer |
        +-----------+        +--------------------+

Note that UnixDatagramServer derives from UDPServer, not from
UnixStreamServer -- the only difference between an IP and a Unix
stream server is the address family, which is simply repeated in both
unix server classes.

Forking and threading versions of each type of server can be created
using the ForkingMixIn and ThreadingMixIn mix-in classes.  For
instance, a threading UDP server class is created as follows:

        class ThreadingUDPServer(ThreadingMixIn, UDPServer): pass

The Mix-in class must come first, since it overrides a method defined
in UDPServer! Setting the various member variables also changes
the behavior of the underlying server mechanism.

To implement a service, you must derive a class from
BaseRequestHandler and redefine its handle() method.  You can then run
various versions of the service by combining one of the server classes
with your request handler class.

The request handler class must be different for datagram or stream
services.  This can be hidden by using the request handler
subclasses StreamRequestHandler or DatagramRequestHandler.

Of course, you still have to use your head!

For instance, it makes no sense to use a forking server if the service
contains state in memory that can be modified by requests (since the
modifications in the child process would never reach the initial state
kept in the parent process and passed to each child).  In this case,
you can use a threading server, but you will probably have to use
locks to avoid two requests that come in nearly simultaneous to apply
conflicting changes to the server state.

On the other hand, if you are building e.g. an HTTP server, where all
data is stored externally (e.g. in the file system), a synchronous
class will essentially render the service "deaf" while one request is
being handled -- which may be for a very long time if a client is slow
to read all the data it has requested.  Here a threading or forking
server is appropriate.

In some cases, it may be appropriate to process part of a request
synchronously, but to finish processing in a forked child depending on
the request data.  This can be implemented by using a synchronous
server and doing an explicit fork in the request handler class
handle() method.

Another approach to handling multiple simultaneous requests in an
environment that supports neither threads nor fork (or where these are
too expensive or inappropriate for the service) is to maintain an
explicit table of partially finished requests and to use a selector to
decide which request to work on next (or whether to handle a new
incoming request).  This is particularly important for stream services
where each client can potentially be connected for a long time (if
threads or subprocesses cannot be used).

Future work:
- Standard classes for Sun RPC (which uses either UDP or TCP)
- Standard mix-in classes to implement various authentication
  and encryption schemes

XXX Open problems:
- What to do with out-of-band data?

BaseServer:
- split generic "request" functionality out into BaseServer class.
  Copyright (C) 2000  Luke Kenneth Casson Leighton <lkcl@samba.org>

  example: read entries from a SQL database (requires overriding
  get_request() to return a table entry from the database).
  entry is processed by a RequestHandlerClass.

"""

# Author of the BaseServer patch: Luke Kenneth Casson Leighton

__version__ = "0.4"


import socket
import selectors
import os
import sys
import threading
from io import BufferedIOBase
from time import monotonic as time

__all__ = ["BaseServer", "TCPServer", "UDPServer",
           "ThreadingUDPServer", "ThreadingTCPServer",
           "BaseRequestHandler", "StreamRequestHandler",
           "DatagramRequestHandler", "ThreadingMixIn"]
if hasattr(os, "fork"):
    __all__.extend(["ForkingUDPServer","ForkingTCPServer", "ForkingMixIn"])
if hasattr(socket, "AF_UNIX"):
    __all__.extend(["UnixStreamServer","UnixDatagramServer",
                    "ThreadingUnixStreamServer",
                    "ThreadingUnixDatagramServer"])

# poll/select have the advantage of not requiring any extra file descriptor,
# contrarily to epoll/kqueue (also, they require a single syscall).
if hasattr(selectors, 'PollSelector'):
    _ServerSelector = selectors.PollSelector
else:
    _ServerSelector = selectors.SelectSelector


class BaseServer:

    """Base class for server classes.

    Methods for the caller:

    - __init__(server_address, RequestHandlerClass)
    - serve_forever(poll_interval=0.5)
    - shutdown()
    - handle_request()  # if you do not use serve_forever()
    - fileno() -> int   # for selector

    Methods that may be overridden:

    - server_bind()
    - server_activate()
    - get_request() -> request, client_address
    - handle_timeout()
    - verify_request(request, client_address)
    - server_close()
    - process_request(request, client_address)
    - shutdown_request(request)
    - close_request(request)
    - service_actions()
    - handle_error()

    Methods for derived classes:

    - finish_request(request, client_address)

    Class variables that may be overridden by derived classes or
    instances:

    - timeout
    - address_family
    - socket_type
    - allow_reuse_address

    Instance variables:

    - RequestHandlerClass
    - socket

    """

    timeout = None

    def __init__(self, server_address, RequestHandlerClass):
        """Constructor.  May be extended, do not override."""
        self.server_address = server_address
        self.RequestHandlerClass = RequestHandlerClass
        self.__is_shut_down = threading.Event()
        self.__shutdown_request = False

    def server_activate(self):
        """Called by constructor to activate the server.

        May be overridden.

        """
        pass

    def serve_forever(self, poll_interval=0.5):
        """Handle one request at a time until shutdown.

        Polls for shutdown every poll_interval seconds. Ignores
        self.timeout. If you need to do periodic tasks, do them in
        another thread.
        """
        self.__is_shut_down.clear()
        try:
            # XXX: Consider using another file descriptor or connecting to the
            # socket to wake this up instead of polling. Polling reduces our
            # responsiveness to a shutdown request and wastes cpu at all other
            # times.
            with _ServerSelector() as selector:
                selector.register(self, selectors.EVENT_READ)

                while not self.__shutdown_request:
                    ready = selector.select(poll_interval)
                    # bpo-35017: shutdown() called during select(), exit immediately.
                    if self.__shutdown_request:
                        break
                    if ready:
                        self._handle_request_noblock()

                    self.service_actions()
        finally:
            self.__shutdown_request = False
            self.__is_shut_down.set()

    def shutdown(self):
        """Stops the serve_forever loop.

        Blocks until the loop has finished. This must be called while
        serve_forever() is running in another thread, or it will
        deadlock.
        """
        self.__shutdown_request = True
        self.__is_shut_down.wait()

    def service_actions(self):
        """Called by the serve_forever() loop.

        May be overridden by a subclass / Mixin to implement any code that
        needs to be run during the loop.
        """
        pass

    # The distinction between handling, getting, processing and finishing a
    # request is fairly arbitrary.  Remember:
    #
    # - handle_request() is the top-level call.  It calls selector.select(),
    #   get_request(), verify_request() and process_request()
    # - get_request() is different for stream or datagram sockets
    # - process_request() is the place that may fork a new process or create a
    #   new thread to finish the request
    # - finish_request() instantiates the request handler class; this
    #   constructor will handle the request all by itself

    def handle_request(self):
        """Handle one request, possibly blocking.

        Respects self.timeout.
        """
        # Support people who used socket.settimeout() to escape
        # handle_request before self.timeout was available.
        timeout = self.socket.gettimeout()
        if timeout is None:
            timeout = self.timeout
        elif self.timeout is not None:
            timeout = min(timeout, self.timeout)
        if timeout is not None:
            deadline = time() + timeout

        # Wait until a request arrives or the timeout expires - the loop is
        # necessary to accommodate early wakeups due to EINTR.
        with _ServerSelector() as selector:
            selector.register(self, selectors.EVENT_READ)

            while True:
                ready = selector.select(timeout)
                if ready:
                    return self._handle_request_noblock()
                else:
                    if timeout is not None:
                        timeout = deadline - time()
                        if timeout < 0:
                            return self.handle_timeout()

    def _handle_request_noblock(self):
        """Handle one request, without blocking.

        I assume that selector.select() has returned that the socket is
        readable before this function was called, so there should be no risk of
        blocking in get_request().
        """
        try:
            request, client_address = self.get_request()
        except OSError:
            return
        if self.verify_request(request, client_address):
            try:
                self.process_request(request, client_address)
            except Exception:
                self.handle_error(request, client_address)
                self.shutdown_request(request)
            except:
                self.shutdown_request(request)
                raise
        else:
            self.shutdown_request(request)

    def handle_timeout(self):
        """Called if no new request arrives within self.timeout.

        Overridden by ForkingMixIn.
        """
        pass

    def verify_request(self, request, client_address):
        """Verify the request.  May be overridden.

        Return True if we should proceed with this request.

        """
        return True

    def process_request(self, request, client_address):
        """Call finish_request.

        Overridden by ForkingMixIn and ThreadingMixIn.

        """
        self.finish_request(request, client_address)
        self.shutdown_request(request)

    def server_close(self):
        """Called to clean-up the server.

        May be overridden.

        """
        pass

    def finish_request(self, request, client_address):
        """Finish one request by instantiating RequestHandlerClass."""
        self.RequestHandlerClass(request, client_address, self)

    def shutdown_request(self, request):
        """Called to shutdown and close an individual request."""
        self.close_request(request)

    def close_request(self, request):
        """Called to clean up an individual request."""
        pass

    def handle_error(self, request, client_address):
        """Handle an error gracefully.  May be overridden.

        The default is to print a traceback and continue.

        """
        print('-'*40, file=sys.stderr)
        print('Exception occurred during processing of request from',
            client_address, file=sys.stderr)
        import traceback
        traceback.print_exc()
        print('-'*40, file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.server_close()


class TCPServer(BaseServer):

    """Base class for various socket-based server classes.

    Defaults to synchronous IP stream (i.e., TCP).

    Methods for the caller:

    - __init__(server_address, RequestHandlerClass, bind_and_activate=True)
    - serve_forever(poll_interval=0.5)
    - shutdown()
    - handle_request()  # if you don't use serve_forever()
    - fileno() -> int   # for selector

    Methods that may be overridden:

    - server_bind()
    - server_activate()
    - get_request() -> request, client_address
    - handle_timeout()
    - verify_request(request, client_address)
    - process_request(request, client_address)
    - shutdown_request(request)
    - close_request(request)
    - handle_error()

    Methods for derived classes:

    - finish_request(request, client_address)

    Class variables that may be overridden by derived classes or
    instances:

    - timeout
    - address_family
    - socket_type
    - request_queue_size (only for stream sockets)
    - allow_reuse_address

    Instance variables:

    - server_address
    - RequestHandlerClass
    - socket

    """

    address_family = socket.AF_INET

    socket_type = socket.SOCK_STREAM

    request_queue_size = 5

    allow_reuse_address = False

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        """Constructor.  May be extended, do not override."""
        BaseServer.__init__(self, server_address, RequestHandlerClass)
        self.socket = socket.socket(self.address_family,
                                    self.socket_type)
        if bind_and_activate:
            try:
                self.server_bind()
                self.server_activate()
            except:
                self.server_close()
                raise

    def server_bind(self):
        """Called by constructor to bind the socket.

        May be overridden.

        """
        if self.allow_reuse_address:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()

    def server_activate(self):
        """Called by constructor to activate the server.

        May be overridden.

        """
        self.socket.listen(self.request_queue_size)

    def server_close(self):
        """Called to clean-up the server.

        May be overridden.

        """
        self.socket.close()

    def fileno(self):
        """Return socket file number.

        Interface required by selector.

        """
        return self.socket.fileno()

    def get_request(self):
        """Get the request and client address from the socket.

        May be overridden.

        """
        return self.socket.accept()

    def shutdown_request(self, request):
        """Called to shutdown and close an individual request."""
        try:
            #explicitly shutdown.  socket.close() merely releases
            #the socket and waits for GC to perform the actual close.
            request.shutdown(socket.SHUT_WR)
        except OSError:
            pass #some platforms may raise ENOTCONN here
        self.close_request(request)

    def close_request(self, request):
        """Called to clean up an individual request."""
        request.close()


class UDPServer(TCPServer):

    """UDP server class."""

    allow_reuse_address = False

    socket_type = socket.SOCK_DGRAM

    max_packet_size = 8192

    def get_request(self):
        data, client_addr = self.socket.recvfrom(self.max_packet_size)
        return (data, self.socket), client_addr

    def server_activate(self):
        # No need to call listen() for UDP.
        pass

    def shutdown_request(self, request):
        # No need to shutdown anything.
        self.close_request(request)

    def close_request(self, request):
        # No need to close anything.
        pass

if hasattr(os, "fork"):
    class ForkingMixIn:
        """Mix-in class to handle each request in a new process."""

        timeout = 300
        active_children = None
        max_children = 40
        # If true, server_close() waits until all child processes complete.
        block_on_close = True

        def collect_children(self, *, blocking=False):
            """Internal routine to wait for children that have exited."""
            if self.active_children is None:
                return

            # If we're above the max number of children, wait and reap them until
            # we go back below threshold. Note that we use waitpid(-1) below to be
            # able to collect children in size(<defunct children>) syscalls instead
            # of size(<children>): the downside is that this might reap children
            # which we didn't spawn, which is why we only resort to this when we're
            # above max_children.
            while len(self.active_children) >= self.max_children:
                try:
                    pid, _ = os.waitpid(-1, 0)
                    self.active_children.discard(pid)
                except ChildProcessError:
                    # we don't have any children, we're done
                    self.active_children.clear()
                except OSError:
                    break

            # Now reap all defunct children.
            for pid in self.active_children.copy():
                try:
                    flags = 0 if blocking else os.WNOHANG
                    pid, _ = os.waitpid(pid, flags)
                    # if the child hasn't exited yet, pid will be 0 and ignored by
                    # discard() below
                    self.active_children.discard(pid)
                except ChildProcessError:
                    # someone else reaped it
                    self.active_children.discard(pid)
                except OSError:
                    pass

        def handle_timeout(self):
            """Wait for zombies after self.timeout seconds of inactivity.

            May be extended, do not override.
            """
            self.collect_children()

        def service_actions(self):
            """Collect the zombie child processes regularly in the ForkingMixIn.

            service_actions is called in the BaseServer's serve_forever loop.
            """
            self.collect_children()

        def process_request(self, request, client_address):
            """Fork a new subprocess to process the request."""
            pid = os.fork()
            if pid:
                # Parent process
                if self.active_children is None:
                    self.active_children = set()
                self.active_children.add(pid)
                self.close_request(request)
                return
            else:
                # Child process.
                # This must never return, hence os._exit()!
                status = 1
                try:
                    self.finish_request(request, client_address)
                    status = 0
                except Exception:
                    self.handle_error(request, client_address)
                finally:
                    try:
                        self.shutdown_request(request)
                    finally:
                        os._exit(status)

        def server_close(self):
            super().server_close()
            self.collect_children(blocking=self.block_on_close)


class _Threads(list):
    """
    Joinable list of all non-daemon threads.
    """
    def append(self, thread):
        self.reap()
        if thread.daemon:
            return
        super().append(thread)

    def pop_all(self):
        self[:], result = [], self[:]
        return result

    def join(self):
        for thread in self.pop_all():
            thread.join()

    def reap(self):
        self[:] = (thread for thread in self if thread.is_alive())


class _NoThreads:
    """
    Degenerate version of _Threads.
    """
    def append(self, thread):
        pass

    def join(self):
        pass


class ThreadingMixIn:
    """Mix-in class to handle each request in a new thread."""

    # Decides how threads will act upon termination of the
    # main process
    daemon_threads = False
    # If true, server_close() waits until all non-daemonic threads terminate.
    block_on_close = True
    # Threads object
    # used by server_close() to wait for all threads completion.
    _threads = _NoThreads()

    def process_request_thread(self, request, client_address):
        """Same as in BaseServer but as a thread.

        In addition, exception handling is done here.

        """
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def process_request(self, request, client_address):
        """Start a new thread to process the request."""
        if self.block_on_close:
            vars(self).setdefault('_threads', _Threads())
        t = threading.Thread(target = self.process_request_thread,
                             args = (request, client_address))
        t.daemon = self.daemon_threads
        self._threads.append(t)
        t.start()

    def server_close(self):
        super().server_close()
        self._threads.join()


if hasattr(os, "fork"):
    class ForkingUDPServer(ForkingMixIn, UDPServer): pass
    class ForkingTCPServer(ForkingMixIn, TCPServer): pass

class ThreadingUDPServer(ThreadingMixIn, UDPServer): pass
class ThreadingTCPServer(ThreadingMixIn, TCPServer): pass

if hasattr(socket, 'AF_UNIX'):

    class UnixStreamServer(TCPServer):
        address_family = socket.AF_UNIX

    class UnixDatagramServer(UDPServer):
        address_family = socket.AF_UNIX

    class ThreadingUnixStreamServer(ThreadingMixIn, UnixStreamServer): pass

    class ThreadingUnixDatagramServer(ThreadingMixIn, UnixDatagramServer): pass

class BaseRequestHandler:

    """Base class for request handler classes.

    This class is instantiated for each request to be handled.  The
    constructor sets the instance variables request, client_address
    and server, and then calls the handle() method.  To implement a
    specific service, all you need to do is to derive a class which
    defines a handle() method.

    The handle() method can find the request as self.request, the
    client address as self.client_address, and the server (in case it
    needs access to per-server information) as self.server.  Since a
    separate instance is created for each request, the handle() method
    can define other arbitrary instance variables.

    """

    def __init__(self, request, client_address, server):
        self.request = request
        self.client_address = client_address
        self.server = server
        self.setup()
        try:
            self.handle()
        finally:
            self.finish()

    def setup(self):
        pass

    def handle(self):
        pass

    def finish(self):
        pass


# The following two classes make it possible to use the same service
# class for stream or datagram servers.
# Each class sets up these instance variables:
# - rfile: a file object from which receives the request is read
# - wfile: a file object to which the reply is written
# When the handle() method returns, wfile is flushed properly


class StreamRequestHandler(BaseRequestHandler):

    """Define self.rfile and self.wfile for stream sockets."""

    # Default buffer sizes for rfile, wfile.
    # We default rfile to buffered because otherwise it could be
    # really slow for large data (a getc() call per byte); we make
    # wfile unbuffered because (a) often after a write() we want to
    # read and we need to flush the line; (b) big writes to unbuffered
    # files are typically optimized by stdio even when big reads
    # aren't.
    rbufsize = -1
    wbufsize = 0

    # A timeout to apply to the request socket, if not None.
    timeout = None

    # Disable nagle algorithm for this socket, if True.
    # Use only when wbufsize != 0, to avoid small packets.
    disable_nagle_algorithm = False

    def setup(self):
        self.connection = self.request
        if self.timeout is not None:
            self.connection.settimeout(self.timeout)
        if self.disable_nagle_algorithm:
            self.connection.setsockopt(socket.IPPROTO_TCP,
                                       socket.TCP_NODELAY, True)
        self.rfile = self.connection.makefile('rb', self.rbufsize)
        if self.wbufsize == 0:
            self.wfile = _SocketWriter(self.connection)
        else:
            self.wfile = self.connection.makefile('wb', self.wbufsize)

    def finish(self):
        if not self.wfile.closed:
            try:
                self.wfile.flush()
            except socket.error:
                # A final socket error may have occurred here, such as
                # the local error ECONNABORTED.
                pass
        self.wfile.close()
        self.rfile.close()

class _SocketWriter(BufferedIOBase):
    """Simple writable BufferedIOBase implementation for a socket

    Does not hold data in a buffer, avoiding any need to call flush()."""

    def __init__(self, sock):
        self._sock = sock

    def writable(self):
        return True

    def write(self, b):
        self._sock.sendall(b)
        with memoryview(b) as view:
            return view.nbytes

    def fileno(self):
        return self._sock.fileno()

class DatagramRequestHandler(BaseRequestHandler):

    """Define self.rfile and self.wfile for datagram sockets."""

    def setup(self):
        from io import BytesIO
        self.packet, self.socket = self.request
        self.rfile = BytesIO(self.packet)
        self.wfile = BytesIO()

    def finish(self):
        self.socket.sendto(self.wfile.getvalue(), self.client_address)
<|endoftext|>"""Random variable generators.

    bytes
    -----
           uniform bytes (values between 0 and 255)

    integers
    --------
           uniform within range

    sequences
    ---------
           pick random element
           pick random sample
           pick weighted random sample
           generate random permutation

    distributions on the real line:
    ------------------------------
           uniform
           triangular
           normal (Gaussian)
           lognormal
           negative exponential
           gamma
           beta
           pareto
           Weibull

    distributions on the circle (angles 0 to 2pi)
    ---------------------------------------------
           circular uniform
           von Mises

General notes on the underlying Mersenne Twister core generator:

* The period is 2**19937-1.
* It is one of the most extensively tested generators in existence.
* The random() method is implemented in C, executes in a single Python step,
  and is, therefore, threadsafe.

"""

# Translated by Guido van Rossum from C source provided by
# Adrian Baddeley.  Adapted by Raymond Hettinger for use with
# the Mersenne Twister  and os.urandom() core generators.

from warnings import warn as _warn
from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
from math import sqrt as _sqrt, acos as _acos, cos as _cos, sin as _sin
from math import tau as TWOPI, floor as _floor, isfinite as _isfinite
from os import urandom as _urandom
from _collections_abc import Set as _Set, Sequence as _Sequence
from operator import index as _index
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import os as _os
import _random

try:
    # hashlib is pretty heavy to load, try lean internal module first
    from _sha512 import sha512 as _sha512
except ImportError:
    # fallback to official implementation
    from hashlib import sha512 as _sha512

__all__ = [
    "Random",
    "SystemRandom",
    "betavariate",
    "choice",
    "choices",
    "expovariate",
    "gammavariate",
    "gauss",
    "getrandbits",
    "getstate",
    "lognormvariate",
    "normalvariate",
    "paretovariate",
    "randbytes",
    "randint",
    "random",
    "randrange",
    "sample",
    "seed",
    "setstate",
    "shuffle",
    "triangular",
    "uniform",
    "vonmisesvariate",
    "weibullvariate",
]

NV_MAGICCONST = 4 * _exp(-0.5) / _sqrt(2.0)
LOG4 = _log(4.0)
SG_MAGICCONST = 1.0 + _log(4.5)
BPF = 53        # Number of bits in a float
RECIP_BPF = 2 ** -BPF
_ONE = 1


class Random(_random.Random):
    """Random number generator base class used by bound module functions.

    Used to instantiate instances of Random to get generators that don't
    share state.

    Class Random can also be subclassed if you want to use a different basic
    generator of your own devising: in that case, override the following
    methods:  random(), seed(), getstate(), and setstate().
    Optionally, implement a getrandbits() method so that randrange()
    can cover arbitrarily large ranges.

    """

    VERSION = 3     # used by getstate/setstate

    def __init__(self, x=None):
        """Initialize an instance.

        Optional argument x controls seeding, as for Random.seed().
        """

        self.seed(x)
        self.gauss_next = None

    def seed(self, a=None, version=2):
        """Initialize internal state from a seed.

        The only supported seed types are None, int, float,
        str, bytes, and bytearray.

        None or no argument seeds from current time or from an operating
        system specific randomness source if available.

        If *a* is an int, all bits are used.

        For version 2 (the default), all of the bits are used if *a* is a str,
        bytes, or bytearray.  For version 1 (provided for reproducing random
        sequences from older versions of Python), the algorithm for str and
        bytes generates a narrower range of seeds.

        """

        if version == 1 and isinstance(a, (str, bytes)):
            a = a.decode('latin-1') if isinstance(a, bytes) else a
            x = ord(a[0]) << 7 if a else 0
            for c in map(ord, a):
                x = ((1000003 * x) ^ c) & 0xFFFFFFFFFFFFFFFF
            x ^= len(a)
            a = -2 if x == -1 else x

        elif version == 2 and isinstance(a, (str, bytes, bytearray)):
            if isinstance(a, str):
                a = a.encode()
            a = int.from_bytes(a + _sha512(a).digest(), 'big')

        elif not isinstance(a, (type(None), int, float, str, bytes, bytearray)):
            _warn('Seeding based on hashing is deprecated\n'
                  'since Python 3.9 and will be removed in a subsequent '
                  'version. The only \n'
                  'supported seed types are: None, '
                  'int, float, str, bytes, and bytearray.',
                  DeprecationWarning, 2)

        super().seed(a)
        self.gauss_next = None

    def getstate(self):
        """Return internal state; can be passed to setstate() later."""
        return self.VERSION, super().getstate(), self.gauss_next

    def setstate(self, state):
        """Restore internal state from object returned by getstate()."""
        version = state[0]
        if version == 3:
            version, internalstate, self.gauss_next = state
            super().setstate(internalstate)
        elif version == 2:
            version, internalstate, self.gauss_next = state
            # In version 2, the state was saved as signed ints, which causes
            #   inconsistencies between 32/64-bit systems. The state is
            #   really unsigned 32-bit ints, so we convert negative ints from
            #   version 2 to positive longs for version 3.
            try:
                internalstate = tuple(x % (2 ** 32) for x in internalstate)
            except ValueError as e:
                raise TypeError from e
            super().setstate(internalstate)
        else:
            raise ValueError("state with version %s passed to "
                             "Random.setstate() of version %s" %
                             (version, self.VERSION))


    ## -------------------------------------------------------
    ## ---- Methods below this point do not need to be overridden or extended
    ## ---- when subclassing for the purpose of using a different core generator.


    ## -------------------- pickle support  -------------------

    # Issue 17489: Since __reduce__ was defined to fix #759889 this is no
    # longer called; we leave it here because it has been here since random was
    # rewritten back in 2001 and why risk breaking something.
    def __getstate__(self):  # for pickle
        return self.getstate()

    def __setstate__(self, state):  # for pickle
        self.setstate(state)

    def __reduce__(self):
        return self.__class__, (), self.getstate()


    ## ---- internal support method for evenly distributed integers ----

    def __init_subclass__(cls, /, **kwargs):
        """Control how subclasses generate random integers.

        The algorithm a subclass can use depends on the random() and/or
        getrandbits() implementation available to it and determines
        whether it can generate random integers from arbitrarily large
        ranges.
        """

        for c in cls.__mro__:
            if '_randbelow' in c.__dict__:
                # just inherit it
                break
            if 'getrandbits' in c.__dict__:
                cls._randbelow = cls._randbelow_with_getrandbits
                break
            if 'random' in c.__dict__:
                cls._randbelow = cls._randbelow_without_getrandbits
                break

    def _randbelow_with_getrandbits(self, n):
        "Return a random int in the range [0,n).  Returns 0 if n==0."

        if not n:
            return 0
        getrandbits = self.getrandbits
        k = n.bit_length()  # don't use (n-1) here because n can be 1
        r = getrandbits(k)  # 0 <= r < 2**k
        while r >= n:
            r = getrandbits(k)
        return r

    def _randbelow_without_getrandbits(self, n, maxsize=1<<BPF):
        """Return a random int in the range [0,n).  Returns 0 if n==0.

        The implementation does not use getrandbits, but only random.
        """

        random = self.random
        if n >= maxsize:
            _warn("Underlying random() generator does not supply \n"
                "enough bits to choose from a population range this large.\n"
                "To remove the range limitation, add a getrandbits() method.")
            return _floor(random() * n)
        if n == 0:
            return 0
        rem = maxsize % n
        limit = (maxsize - rem) / maxsize   # int(limit * maxsize) % n == 0
        r = random()
        while r >= limit:
            r = random()
        return _floor(r * maxsize) % n

    _randbelow = _randbelow_with_getrandbits


    ## --------------------------------------------------------
    ## ---- Methods below this point generate custom distributions
    ## ---- based on the methods defined above.  They do not
    ## ---- directly touch the underlying generator and only
    ## ---- access randomness through the methods:  random(),
    ## ---- getrandbits(), or _randbelow().


    ## -------------------- bytes methods ---------------------

    def randbytes(self, n):
        """Generate n random bytes."""
        return self.getrandbits(n * 8).to_bytes(n, 'little')


    ## -------------------- integer methods  -------------------

    def randrange(self, start, stop=None, step=_ONE):
        """Choose a random item from range(start, stop[, step]).

        This fixes the problem with randint() which includes the
        endpoint; in Python this is usually not what you want.

        """

        # This code is a bit messy to make it fast for the
        # common case while still doing adequate error checking.
        try:
            istart = _index(start)
        except TypeError:
            istart = int(start)
            if istart != start:
                _warn('randrange() will raise TypeError in the future',
                      DeprecationWarning, 2)
                raise ValueError("non-integer arg 1 for randrange()")
            _warn('non-integer arguments to randrange() have been deprecated '
                  'since Python 3.10 and will be removed in a subsequent '
                  'version',
                  DeprecationWarning, 2)
        if stop is None:
            # We don't check for "step != 1" because it hasn't been
            # type checked and converted to an integer yet.
            if step is not _ONE:
                raise TypeError('Missing a non-None stop argument')
            if istart > 0:
                return self._randbelow(istart)
            raise ValueError("empty range for randrange()")

        # stop argument supplied.
        try:
            istop = _index(stop)
        except TypeError:
            istop = int(stop)
            if istop != stop:
                _warn('randrange() will raise TypeError in the future',
                      DeprecationWarning, 2)
                raise ValueError("non-integer stop for randrange()")
            _warn('non-integer arguments to randrange() have been deprecated '
                  'since Python 3.10 and will be removed in a subsequent '
                  'version',
                  DeprecationWarning, 2)
        width = istop - istart
        try:
            istep = _index(step)
        except TypeError:
            istep = int(step)
            if istep != step:
                _warn('randrange() will raise TypeError in the future',
                      DeprecationWarning, 2)
                raise ValueError("non-integer step for randrange()")
            _warn('non-integer arguments to randrange() have been deprecated '
                  'since Python 3.10 and will be removed in a subsequent '
                  'version',
                  DeprecationWarning, 2)
        # Fast path.
        if istep == 1:
            if width > 0:
                return istart + self._randbelow(width)
            raise ValueError("empty range for randrange() (%d, %d, %d)" % (istart, istop, width))

        # Non-unit step argument supplied.
        if istep > 0:
            n = (width + istep - 1) // istep
        elif istep < 0:
            n = (width + istep + 1) // istep
        else:
            raise ValueError("zero step for randrange()")
        if n <= 0:
            raise ValueError("empty range for randrange()")
        return istart + istep * self._randbelow(n)

    def randint(self, a, b):
        """Return random integer in range [a, b], including both end points.
        """

        return self.randrange(a, b+1)


    ## -------------------- sequence methods  -------------------

    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        # raises IndexError if seq is empty
        return seq[self._randbelow(len(seq))]

    def shuffle(self, x, random=None):
        """Shuffle list x in place, and return None.

        Optional argument random is a 0-argument function returning a
        random float in [0.0, 1.0); if it is the default None, the
        standard random.random will be used.

        """

        if random is None:
            randbelow = self._randbelow
            for i in reversed(range(1, len(x))):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = randbelow(i + 1)
                x[i], x[j] = x[j], x[i]
        else:
            _warn('The *random* parameter to shuffle() has been deprecated\n'
                  'since Python 3.9 and will be removed in a subsequent '
                  'version.',
                  DeprecationWarning, 2)
            floor = _floor
            for i in reversed(range(1, len(x))):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = floor(random() * (i + 1))
                x[i], x[j] = x[j], x[i]

    def sample(self, population, k, *, counts=None):
        """Chooses k unique random elements from a population sequence or set.

        Returns a new list containing elements from the population while
        leaving the original population unchanged.  The resulting list is
        in selection order so that all sub-slices will also be valid random
        samples.  This allows raffle winners (the sample) to be partitioned
        into grand prize and second place winners (the subslices).

        Members of the population need not be hashable or unique.  If the
        population contains repeats, then each occurrence is a possible
        selection in the sample.

        Repeated elements can be specified one at a time or with the optional
        counts parameter.  For example:

            sample(['red', 'blue'], counts=[4, 2], k=5)

        is equivalent to:

            sample(['red', 'red', 'red', 'red', 'blue', 'blue'], k=5)

        To choose a sample from a range of integers, use range() for the
        population argument.  This is especially fast and space efficient
        for sampling from a large population:

            sample(range(10000000), 60)

        """

        # Sampling without replacement entails tracking either potential
        # selections (the pool) in a list or previous selections in a set.

        # When the number of selections is small compared to the
        # population, then tracking selections is efficient, requiring
        # only a small set and an occasional reselection.  For
        # a larger number of selections, the pool tracking method is
        # preferred since the list takes less space than the
        # set and it doesn't suffer from frequent reselections.

        # The number of calls to _randbelow() is kept at or near k, the
        # theoretical minimum.  This is important because running time
        # is dominated by _randbelow() and because it extracts the
        # least entropy from the underlying random number generators.

        # Memory requirements are kept to the smaller of a k-length
        # set or an n-length list.

        # There are other sampling algorithms that do not require
        # auxiliary memory, but they were rejected because they made
        # too many calls to _randbelow(), making them slower and
        # causing them to eat more entropy than necessary.

        if not isinstance(population, _Sequence):
            if isinstance(population, _Set):
                _warn('Sampling from a set deprecated\n'
                      'since Python 3.9 and will be removed in a subsequent version.',
                      DeprecationWarning, 2)
                population = tuple(population)
            else:
                raise TypeError("Population must be a sequence.  For dicts or sets, use sorted(d).")
        n = len(population)
        if counts is not None:
            cum_counts = list(_accumulate(counts))
            if len(cum_counts) != n:
                raise ValueError('The number of counts does not match the population')
            total = cum_counts.pop()
            if not isinstance(total, int):
                raise TypeError('Counts must be integers')
            if total <= 0:
                raise ValueError('Total of counts must be greater than zero')
            selections = self.sample(range(total), k=k)
            bisect = _bisect
            return [population[bisect(cum_counts, s)] for s in selections]
        randbelow = self._randbelow
        if not 0 <= k <= n:
            raise ValueError("Sample larger than population or is negative")
        result = [None] * k
        setsize = 21        # size of a small set minus size of an empty list
        if k > 5:
            setsize += 4 ** _ceil(_log(k * 3, 4))  # table size for big sets
        if n <= setsize:
            # An n-length list is smaller than a k-length set.
            # Invariant:  non-selected at pool[0 : n-i]
            pool = list(population)
            for i in range(k):
                j = randbelow(n - i)
                result[i] = pool[j]
                pool[j] = pool[n - i - 1]  # move non-selected item into vacancy
        else:
            selected = set()
            selected_add = selected.add
            for i in range(k):
                j = randbelow(n)
                while j in selected:
                    j = randbelow(n)
                selected_add(j)
                result[i] = population[j]
        return result

    def choices(self, population, weights=None, *, cum_weights=None, k=1):
        """Return a k sized list of population elements chosen with replacement.

        If the relative weights or cumulative weights are not specified,
        the selections are made with equal probability.

        """
        random = self.random
        n = len(population)
        if cum_weights is None:
            if weights is None:
                floor = _floor
                n += 0.0    # convert to float for a small speed improvement
                return [population[floor(random() * n)] for i in _repeat(None, k)]
            try:
                cum_weights = list(_accumulate(weights))
            except TypeError:
                if not isinstance(weights, int):
                    raise
                k = weights
                raise TypeError(
                    f'The number of choices must be a keyword argument: {k=}'
                ) from None
        elif weights is not None:
            raise TypeError('Cannot specify both weights and cumulative weights')
        if len(cum_weights) != n:
            raise ValueError('The number of weights does not match the population')
        total = cum_weights[-1] + 0.0   # convert to float
        if total <= 0.0:
            raise ValueError('Total of weights must be greater than zero')
        if not _isfinite(total):
            raise ValueError('Total of weights must be finite')
        bisect = _bisect
        hi = n - 1
        return [population[bisect(cum_weights, random() * total, 0, hi)]
                for i in _repeat(None, k)]


    ## -------------------- real-valued distributions  -------------------

    def uniform(self, a, b):
        "Get a random number in the range [a, b) or [a, b] depending on rounding."
        return a + (b - a) * self.random()

    def triangular(self, low=0.0, high=1.0, mode=None):
        """Triangular distribution.

        Continuous distribution bounded by given lower and upper limits,
        and having a given mode value in-between.

        http://en.wikipedia.org/wiki/Triangular_distribution

        """
        u = self.random()
        try:
            c = 0.5 if mode is None else (mode - low) / (high - low)
        except ZeroDivisionError:
            return low
        if u > c:
            u = 1.0 - u
            c = 1.0 - c
            low, high = high, low
        return low + (high - low) * _sqrt(u * c)

    def normalvariate(self, mu, sigma):
        """Normal distribution.

        mu is the mean, and sigma is the standard deviation.

        """
        # Uses Kinderman and Monahan method. Reference: Kinderman,
        # A.J. and Monahan, J.F., "Computer generation of random
        # variables using the ratio of uniform deviates", ACM Trans
        # Math Software, 3, (1977), pp257-260.

        random = self.random
        while True:
            u1 = random()
            u2 = 1.0 - random()
            z = NV_MAGICCONST * (u1 - 0.5) / u2
            zz = z * z / 4.0
            if zz <= -_log(u2):
                break
        return mu + z * sigma

    def gauss(self, mu, sigma):
        """Gaussian distribution.

        mu is the mean, and sigma is the standard deviation.  This is
        slightly faster than the normalvariate() function.

        Not thread-safe without a lock around calls.

        """
        # When x and y are two variables from [0, 1), uniformly
        # distributed, then
        #
        #    cos(2*pi*x)*sqrt(-2*log(1-y))
        #    sin(2*pi*x)*sqrt(-2*log(1-y))
        #
        # are two *independent* variables with normal distribution
        # (mu = 0, sigma = 1).
        # (Lambert Meertens)
        # (corrected version; bug discovered by Mike Miller, fixed by LM)

        # Multithreading note: When two threads call this function
        # simultaneously, it is possible that they will receive the
        # same return value.  The window is very small though.  To
        # avoid this, you have to use a lock around all calls.  (I
        # didn't want to slow this down in the serial case by using a
        # lock here.)

        random = self.random
        z = self.gauss_next
        self.gauss_next = None
        if z is None:
            x2pi = random() * TWOPI
            g2rad = _sqrt(-2.0 * _log(1.0 - random()))
            z = _cos(x2pi) * g2rad
            self.gauss_next = _sin(x2pi) * g2rad

        return mu + z * sigma

    def lognormvariate(self, mu, sigma):
        """Log normal distribution.

        If you take the natural logarithm of this distribution, you'll get a
        normal distribution with mean mu and standard deviation sigma.
        mu can have any value, and sigma must be greater than zero.

        """
        return _exp(self.normalvariate(mu, sigma))

    def expovariate(self, lambd):
        """Exponential distribution.

        lambd is 1.0 divided by the desired mean.  It should be
        nonzero.  (The parameter would be called "lambda", but that is
        a reserved word in Python.)  Returned values range from 0 to
        positive infinity if lambd is positive, and from negative
        infinity to 0 if lambd is negative.

        """
        # lambd: rate lambd = 1/mean
        # ('lambda' is a Python reserved word)

        # we use 1-random() instead of random() to preclude the
        # possibility of taking the log of zero.
        return -_log(1.0 - self.random()) / lambd

    def vonmisesvariate(self, mu, kappa):
        """Circular data distribution.

        mu is the mean angle, expressed in radians between 0 and 2*pi, and
        kappa is the concentration parameter, which must be greater than or
        equal to zero.  If kappa is equal to zero, this distribution reduces
        to a uniform random angle over the range 0 to 2*pi.

        """
        # Based upon an algorithm published in: Fisher, N.I.,
        # "Statistical Analysis of Circular Data", Cambridge
        # University Press, 1993.

        # Thanks to Magnus Kessler for a correction to the
        # implementation of step 4.

        random = self.random
        if kappa <= 1e-6:
            return TWOPI * random()

        s = 0.5 / kappa
        r = s + _sqrt(1.0 + s * s)

        while True:
            u1 = random()
            z = _cos(_pi * u1)

            d = z / (r + z)
            u2 = random()
            if u2 < 1.0 - d * d or u2 <= (1.0 - d) * _exp(d):
                break

        q = 1.0 / r
        f = (q + z) / (1.0 + q * z)
        u3 = random()
        if u3 > 0.5:
            theta = (mu + _acos(f)) % TWOPI
        else:
            theta = (mu - _acos(f)) % TWOPI

        return theta

    def gammavariate(self, alpha, beta):
        """Gamma distribution.  Not the gamma function!

        Conditions on the parameters are alpha > 0 and beta > 0.

        The probability distribution function is:

                    x ** (alpha - 1) * math.exp(-x / beta)
          pdf(x) =  --------------------------------------
                      math.gamma(alpha) * beta ** alpha

        """
        # alpha > 0, beta > 0, mean is alpha*beta, variance is alpha*beta**2

        # Warning: a few older sources define the gamma distribution in terms
        # of alpha > -1.0
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError('gammavariate: alpha and beta must be > 0.0')

        random = self.random
        if alpha > 1.0:

            # Uses R.C.H. Cheng, "The generation of Gamma
            # variables with non-integral shape parameters",
            # Applied Statistics, (1977), 26, No. 1, p71-74

            ainv = _sqrt(2.0 * alpha - 1.0)
            bbb = alpha - LOG4
            ccc = alpha + ainv

            while True:
                u1 = random()
                if not 1e-7 < u1 < 0.9999999:
                    continue
                u2 = 1.0 - random()
                v = _log(u1 / (1.0 - u1)) / ainv
                x = alpha * _exp(v)
                z = u1 * u1 * u2
                r = bbb + ccc * v - x
                if r + SG_MAGICCONST - 4.5 * z >= 0.0 or r >= _log(z):
                    return x * beta

        elif alpha == 1.0:
            # expovariate(1/beta)
            return -_log(1.0 - random()) * beta

        else:
            # alpha is between 0 and 1 (exclusive)
            # Uses ALGORITHM GS of Statistical Computing - Kennedy & Gentle
            while True:
                u = random()
                b = (_e + alpha) / _e
                p = b * u
                if p <= 1.0:
                    x = p ** (1.0 / alpha)
                else:
                    x = -_log((b - p) / alpha)
                u1 = random()
                if p > 1.0:
                    if u1 <= x ** (alpha - 1.0):
                        break
                elif u1 <= _exp(-x):
                    break
            return x * beta

    def betavariate(self, alpha, beta):
        """Beta distribution.

        Conditions on the parameters are alpha > 0 and beta > 0.
        Returned values range between 0 and 1.

        """
        ## See
        ## http://mail.python.org/pipermail/python-bugs-list/2001-January/003752.html
        ## for Ivan Frohne's insightful analysis of why the original implementation:
        ##
        ##    def betavariate(self, alpha, beta):
        ##        # Discrete Event Simulation in C, pp 87-88.
        ##
        ##        y = self.expovariate(alpha)
        ##        z = self.expovariate(1.0/beta)
        ##        return z/(y+z)
        ##
        ## was dead wrong, and how it probably got that way.

        # This version due to Janne Sinkkonen, and matches all the std
        # texts (e.g., Knuth Vol 2 Ed 3 pg 134 "the beta distribution").
        y = self.gammavariate(alpha, 1.0)
        if y:
            return y / (y + self.gammavariate(beta, 1.0))
        return 0.0

    def paretovariate(self, alpha):
        """Pareto distribution.  alpha is the shape parameter."""
        # Jain, pg. 495

        u = 1.0 - self.random()
        return u ** (-1.0 / alpha)

    def weibullvariate(self, alpha, beta):
        """Weibull distribution.

        alpha is the scale parameter and beta is the shape parameter.

        """
        # Jain, pg. 499; bug fix courtesy Bill Arms

        u = 1.0 - self.random()
        return alpha * (-_log(u)) ** (1.0 / beta)


## ------------------------------------------------------------------
## --------------- Operating System Random Source  ------------------


class SystemRandom(Random):
    """Alternate random number generator using sources provided
    by the operating system (such as /dev/urandom on Unix or
    CryptGenRandom on Windows).

     Not available on all systems (see os.urandom() for details).

    """

    def random(self):
        """Get the next random number in the range [0.0, 1.0)."""
        return (int.from_bytes(_urandom(7), 'big') >> 3) * RECIP_BPF

    def getrandbits(self, k):
        """getrandbits(k) -> x.  Generates an int with k random bits."""
        if k < 0:
            raise ValueError('number of bits must be non-negative')
        numbytes = (k + 7) // 8                       # bits / 8 and rounded up
        x = int.from_bytes(_urandom(numbytes), 'big')
        return x >> (numbytes * 8 - k)                # trim excess bits

    def randbytes(self, n):
        """Generate n random bytes."""
        # os.urandom(n) fails with ValueError for n < 0
        # and returns an empty bytes string for n == 0.
        return _urandom(n)

    def seed(self, *args, **kwds):
        "Stub method.  Not used for a system random number generator."
        return None

    def _notimplemented(self, *args, **kwds):
        "Method should not be called for a system random number generator."
        raise NotImplementedError('System entropy source does not have state.')
    getstate = setstate = _notimplemented


# ----------------------------------------------------------------------
# Create one instance, seeded from current time, and export its methods
# as module-level functions.  The functions share state across all uses
# (both in the user's code and in the Python libraries), but that's fine
# for most programs and is easier for the casual user than making them
# instantiate their own Random() instance.

_inst = Random()
seed = _inst.seed
random = _inst.random
uniform = _inst.uniform
triangular = _inst.triangular
randint = _inst.randint
choice = _inst.choice
randrange = _inst.randrange
sample = _inst.sample
shuffle = _inst.shuffle
choices = _inst.choices
normalvariate = _inst.normalvariate
lognormvariate = _inst.lognormvariate
expovariate = _inst.expovariate
vonmisesvariate = _inst.vonmisesvariate
gammavariate = _inst.gammavariate
gauss = _inst.gauss
betavariate = _inst.betavariate
paretovariate = _inst.paretovariate
weibullvariate = _inst.weibullvariate
getstate = _inst.getstate
setstate = _inst.setstate
getrandbits = _inst.getrandbits
randbytes = _inst.randbytes


## ------------------------------------------------------
## ----------------- test program -----------------------

def _test_generator(n, func, args):
    from statistics import stdev, fmean as mean
    from time import perf_counter

    t0 = perf_counter()
    data = [func(*args) for i in _repeat(None, n)]
    t1 = perf_counter()

    xbar = mean(data)
    sigma = stdev(data, xbar)
    low = min(data)
    high = max(data)

    print(f'{t1 - t0:.3f} sec, {n} times {func.__name__}')
    print('avg %g, stddev %g, min %g, max %g\n' % (xbar, sigma, low, high))


def _test(N=2000):
    _test_generator(N, random, ())
    _test_generator(N, normalvariate, (0.0, 1.0))
    _test_generator(N, lognormvariate, (0.0, 1.0))
    _test_generator(N, vonmisesvariate, (0.0, 1.0))
    _test_generator(N, gammavariate, (0.01, 1.0))
    _test_generator(N, gammavariate, (0.1, 1.0))
    _test_generator(N, gammavariate, (0.1, 2.0))
    _test_generator(N, gammavariate, (0.5, 1.0))
    _test_generator(N, gammavariate, (0.9, 1.0))
    _test_generator(N, gammavariate, (1.0, 1.0))
    _test_generator(N, gammavariate, (2.0, 1.0))
    _test_generator(N, gammavariate, (20.0, 1.0))
    _test_generator(N, gammavariate, (200.0, 1.0))
    _test_generator(N, gauss, (0.0, 1.0))
    _test_generator(N, betavariate, (3.0, 3.0))
    _test_generator(N, triangular, (0.0, 1.0, 1.0 / 3.0))


## ------------------------------------------------------
## ------------------ fork support  ---------------------

if hasattr(_os, "fork"):
    _os.register_at_fork(after_in_child=_inst.seed)


if __name__ == '__main__':
    _test()
<|endoftext|>"""Class for printing reports on profiled python code."""

# Written by James Roskind
# Based on prior profile module by Sjoerd Mullender...
#   which was hacked somewhat by: Guido van Rossum

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


import sys
import os
import time
import marshal
import re

from enum import Enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict

__all__ = ["Stats", "SortKey", "FunctionProfile", "StatsProfile"]

class SortKey(str, Enum):
    CALLS = 'calls', 'ncalls'
    CUMULATIVE = 'cumulative', 'cumtime'
    FILENAME = 'filename', 'module'
    LINE = 'line'
    NAME = 'name'
    NFL = 'nfl'
    PCALLS = 'pcalls'
    STDNAME = 'stdname'
    TIME = 'time', 'tottime'

    def __new__(cls, *values):
        value = values[0]
        obj = str.__new__(cls, value)
        obj._value_ = value
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj


@dataclass(unsafe_hash=True)
class FunctionProfile:
    ncalls: str
    tottime: float
    percall_tottime: float
    cumtime: float
    percall_cumtime: float
    file_name: str
    line_number: int

@dataclass(unsafe_hash=True)
class StatsProfile:
    '''Class for keeping track of an item in inventory.'''
    total_tt: float
    func_profiles: Dict[str, FunctionProfile]

class Stats:
    """This class is used for creating reports from data generated by the
    Profile class.  It is a "friend" of that class, and imports data either
    by direct access to members of Profile class, or by reading in a dictionary
    that was emitted (via marshal) from the Profile class.

    The big change from the previous Profiler (in terms of raw functionality)
    is that an "add()" method has been provided to combine Stats from
    several distinct profile runs.  Both the constructor and the add()
    method now take arbitrarily many file names as arguments.

    All the print methods now take an argument that indicates how many lines
    to print.  If the arg is a floating point number between 0 and 1.0, then
    it is taken as a decimal percentage of the available lines to be printed
    (e.g., .1 means print 10% of all available lines).  If it is an integer,
    it is taken to mean the number of lines of data that you wish to have
    printed.

    The sort_stats() method now processes some additional options (i.e., in
    addition to the old -1, 0, 1, or 2 that are respectively interpreted as
    'stdname', 'calls', 'time', and 'cumulative').  It takes either an
    arbitrary number of quoted strings or SortKey enum to select the sort
    order.

    For example sort_stats('time', 'name') or sort_stats(SortKey.TIME,
    SortKey.NAME) sorts on the major key of 'internal function time', and on
    the minor key of 'the name of the function'.  Look at the two tables in
    sort_stats() and get_sort_arg_defs(self) for more examples.

    All methods return self, so you can string together commands like:
        Stats('foo', 'goo').strip_dirs().sort_stats('calls').\
                            print_stats(5).print_callers(5)
    """

    def __init__(self, *args, stream=None):
        self.stream = stream or sys.stdout
        if not len(args):
            arg = None
        else:
            arg = args[0]
            args = args[1:]
        self.init(arg)
        self.add(*args)

    def init(self, arg):
        self.all_callees = None  # calc only if needed
        self.files = []
        self.fcn_list = None
        self.total_tt = 0
        self.total_calls = 0
        self.prim_calls = 0
        self.max_name_len = 0
        self.top_level = set()
        self.stats = {}
        self.sort_arg_dict = {}
        self.load_stats(arg)
        try:
            self.get_top_level_stats()
        except Exception:
            print("Invalid timing data %s" %
                  (self.files[-1] if self.files else ''), file=self.stream)
            raise

    def load_stats(self, arg):
        if arg is None:
            self.stats = {}
            return
        elif isinstance(arg, str):
            with open(arg, 'rb') as f:
                self.stats = marshal.load(f)
            try:
                file_stats = os.stat(arg)
                arg = time.ctime(file_stats.st_mtime) + "    " + arg
            except:  # in case this is not unix
                pass
            self.files = [arg]
        elif hasattr(arg, 'create_stats'):
            arg.create_stats()
            self.stats = arg.stats
            arg.stats = {}
        if not self.stats:
            raise TypeError("Cannot create or construct a %r object from %r"
                            % (self.__class__, arg))
        return

    def get_top_level_stats(self):
        for func, (cc, nc, tt, ct, callers) in self.stats.items():
            self.total_calls += nc
            self.prim_calls  += cc
            self.total_tt    += tt
            if ("jprofile", 0, "profiler") in callers:
                self.top_level.add(func)
            if len(func_std_string(func)) > self.max_name_len:
                self.max_name_len = len(func_std_string(func))

    def add(self, *arg_list):
        if not arg_list:
            return self
        for item in reversed(arg_list):
            if type(self) != type(item):
                item = Stats(item)
            self.files += item.files
            self.total_calls += item.total_calls
            self.prim_calls += item.prim_calls
            self.total_tt += item.total_tt
            for func in item.top_level:
                self.top_level.add(func)

            if self.max_name_len < item.max_name_len:
                self.max_name_len = item.max_name_len

            self.fcn_list = None

            for func, stat in item.stats.items():
                if func in self.stats:
                    old_func_stat = self.stats[func]
                else:
                    old_func_stat = (0, 0, 0, 0, {},)
                self.stats[func] = add_func_stats(old_func_stat, stat)
        return self

    def dump_stats(self, filename):
        """Write the profile data to a file we know how to load back."""
        with open(filename, 'wb') as f:
            marshal.dump(self.stats, f)

    # list the tuple indices and directions for sorting,
    # along with some printable description
    sort_arg_dict_default = {
              "calls"     : (((1,-1),              ), "call count"),
              "ncalls"    : (((1,-1),              ), "call count"),
              "cumtime"   : (((3,-1),              ), "cumulative time"),
              "cumulative": (((3,-1),              ), "cumulative time"),
              "filename"  : (((4, 1),              ), "file name"),
              "line"      : (((5, 1),              ), "line number"),
              "module"    : (((4, 1),              ), "file name"),
              "name"      : (((6, 1),              ), "function name"),
              "nfl"       : (((6, 1),(4, 1),(5, 1),), "name/file/line"),
              "pcalls"    : (((0,-1),              ), "primitive call count"),
              "stdname"   : (((7, 1),              ), "standard name"),
              "time"      : (((2,-1),              ), "internal time"),
              "tottime"   : (((2,-1),              ), "internal time"),
              }

    def get_sort_arg_defs(self):
        """Expand all abbreviations that are unique."""
        if not self.sort_arg_dict:
            self.sort_arg_dict = dict = {}
            bad_list = {}
            for word, tup in self.sort_arg_dict_default.items():
                fragment = word
                while fragment:
                    if not fragment:
                        break
                    if fragment in dict:
                        bad_list[fragment] = 0
                        break
                    dict[fragment] = tup
                    fragment = fragment[:-1]
            for word in bad_list:
                del dict[word]
        return self.sort_arg_dict

    def sort_stats(self, *field):
        if not field:
            self.fcn_list = 0
            return self
        if len(field) == 1 and isinstance(field[0], int):
            # Be compatible with old profiler
            field = [ {-1: "stdname",
                       0:  "calls",
                       1:  "time",
                       2:  "cumulative"}[field[0]] ]
        elif len(field) >= 2:
            for arg in field[1:]:
                if type(arg) != type(field[0]):
                    raise TypeError("Can't have mixed argument type")

        sort_arg_defs = self.get_sort_arg_defs()

        sort_tuple = ()
        self.sort_type = ""
        connector = ""
        for word in field:
            if isinstance(word, SortKey):
                word = word.value
            sort_tuple = sort_tuple + sort_arg_defs[word][0]
            self.sort_type += connector + sort_arg_defs[word][1]
            connector = ", "

        stats_list = []
        for func, (cc, nc, tt, ct, callers) in self.stats.items():
            stats_list.append((cc, nc, tt, ct) + func +
                              (func_std_string(func), func))

        stats_list.sort(key=cmp_to_key(TupleComp(sort_tuple).compare))

        self.fcn_list = fcn_list = []
        for tuple in stats_list:
            fcn_list.append(tuple[-1])
        return self

    def reverse_order(self):
        if self.fcn_list:
            self.fcn_list.reverse()
        return self

    def strip_dirs(self):
        oldstats = self.stats
        self.stats = newstats = {}
        max_name_len = 0
        for func, (cc, nc, tt, ct, callers) in oldstats.items():
            newfunc = func_strip_path(func)
            if len(func_std_string(newfunc)) > max_name_len:
                max_name_len = len(func_std_string(newfunc))
            newcallers = {}
            for func2, caller in callers.items():
                newcallers[func_strip_path(func2)] = caller

            if newfunc in newstats:
                newstats[newfunc] = add_func_stats(
                                        newstats[newfunc],
                                        (cc, nc, tt, ct, newcallers))
            else:
                newstats[newfunc] = (cc, nc, tt, ct, newcallers)
        old_top = self.top_level
        self.top_level = new_top = set()
        for func in old_top:
            new_top.add(func_strip_path(func))

        self.max_name_len = max_name_len

        self.fcn_list = None
        self.all_callees = None
        return self

    def calc_callees(self):
        if self.all_callees:
            return
        self.all_callees = all_callees = {}
        for func, (cc, nc, tt, ct, callers) in self.stats.items():
            if not func in all_callees:
                all_callees[func] = {}
            for func2, caller in callers.items():
                if not func2 in all_callees:
                    all_callees[func2] = {}
                all_callees[func2][func]  = caller
        return

    #******************************************************************
    # The following functions support actual printing of reports
    #******************************************************************

    # Optional "amount" is either a line count, or a percentage of lines.

    def eval_print_amount(self, sel, list, msg):
        new_list = list
        if isinstance(sel, str):
            try:
                rex = re.compile(sel)
            except re.error:
                msg += "   <Invalid regular expression %r>\n" % sel
                return new_list, msg
            new_list = []
            for func in list:
                if rex.search(func_std_string(func)):
                    new_list.append(func)
        else:
            count = len(list)
            if isinstance(sel, float) and 0.0 <= sel < 1.0:
                count = int(count * sel + .5)
                new_list = list[:count]
            elif isinstance(sel, int) and 0 <= sel < count:
                count = sel
                new_list = list[:count]
        if len(list) != len(new_list):
            msg += "   List reduced from %r to %r due to restriction <%r>\n" % (
                len(list), len(new_list), sel)

        return new_list, msg

    def get_stats_profile(self):
        """This method returns an instance of StatsProfile, which contains a mapping
        of function names to instances of FunctionProfile. Each FunctionProfile
        instance holds information related to the function's profile such as how
        long the function took to run, how many times it was called, etc...
        """
        func_list = self.fcn_list[:] if self.fcn_list else list(self.stats.keys())
        if not func_list:
            return StatsProfile(0, {})

        total_tt = float(f8(self.total_tt))
        func_profiles = {}
        stats_profile = StatsProfile(total_tt, func_profiles)

        for func in func_list:
            cc, nc, tt, ct, callers = self.stats[func]
            file_name, line_number, func_name = func
            ncalls = str(nc) if nc == cc else (str(nc) + '/' + str(cc))
            tottime = float(f8(tt))
            percall_tottime = -1 if nc == 0 else float(f8(tt/nc))
            cumtime = float(f8(ct))
            percall_cumtime = -1 if cc == 0 else float(f8(ct/cc))
            func_profile = FunctionProfile(
                ncalls,
                tottime, # time spent in this function alone
                percall_tottime,
                cumtime, # time spent in the function plus all functions that this function called,
                percall_cumtime,
                file_name,
                line_number
            )
            func_profiles[func_name] = func_profile

        return stats_profile

    def get_print_list(self, sel_list):
        width = self.max_name_len
        if self.fcn_list:
            stat_list = self.fcn_list[:]
            msg = "   Ordered by: " + self.sort_type + '\n'
        else:
            stat_list = list(self.stats.keys())
            msg = "   Random listing order was used\n"

        for selection in sel_list:
            stat_list, msg = self.eval_print_amount(selection, stat_list, msg)

        count = len(stat_list)

        if not stat_list:
            return 0, stat_list
        print(msg, file=self.stream)
        if count < len(self.stats):
            width = 0
            for func in stat_list:
                if  len(func_std_string(func)) > width:
                    width = len(func_std_string(func))
        return width+2, stat_list

    def print_stats(self, *amount):
        for filename in self.files:
            print(filename, file=self.stream)
        if self.files:
            print(file=self.stream)
        indent = ' ' * 8
        for func in self.top_level:
            print(indent, func_get_function_name(func), file=self.stream)

        print(indent, self.total_calls, "function calls", end=' ', file=self.stream)
        if self.total_calls != self.prim_calls:
            print("(%d primitive calls)" % self.prim_calls, end=' ', file=self.stream)
        print("in %.3f seconds" % self.total_tt, file=self.stream)
        print(file=self.stream)
        width, list = self.get_print_list(amount)
        if list:
            self.print_title()
            for func in list:
                self.print_line(func)
            print(file=self.stream)
            print(file=self.stream)
        return self

    def print_callees(self, *amount):
        width, list = self.get_print_list(amount)
        if list:
            self.calc_callees()

            self.print_call_heading(width, "called...")
            for func in list:
                if func in self.all_callees:
                    self.print_call_line(width, func, self.all_callees[func])
                else:
                    self.print_call_line(width, func, {})
            print(file=self.stream)
            print(file=self.stream)
        return self

    def print_callers(self, *amount):
        width, list = self.get_print_list(amount)
        if list:
            self.print_call_heading(width, "was called by...")
            for func in list:
                cc, nc, tt, ct, callers = self.stats[func]
                self.print_call_line(width, func, callers, "<-")
            print(file=self.stream)
            print(file=self.stream)
        return self

    def print_call_heading(self, name_size, column_title):
        print("Function ".ljust(name_size) + column_title, file=self.stream)
        # print sub-header only if we have new-style callers
        subheader = False
        for cc, nc, tt, ct, callers in self.stats.values():
            if callers:
                value = next(iter(callers.values()))
                subheader = isinstance(value, tuple)
                break
        if subheader:
            print(" "*name_size + "    ncalls  tottime  cumtime", file=self.stream)

    def print_call_line(self, name_size, source, call_dict, arrow="->"):
        print(func_std_string(source).ljust(name_size) + arrow, end=' ', file=self.stream)
        if not call_dict:
            print(file=self.stream)
            return
        clist = sorted(call_dict.keys())
        indent = ""
        for func in clist:
            name = func_std_string(func)
            value = call_dict[func]
            if isinstance(value, tuple):
                nc, cc, tt, ct = value
                if nc != cc:
                    substats = '%d/%d' % (nc, cc)
                else:
                    substats = '%d' % (nc,)
                substats = '%s %s %s  %s' % (substats.rjust(7+2*len(indent)),
                                             f8(tt), f8(ct), name)
                left_width = name_size + 1
            else:
                substats = '%s(%r) %s' % (name, value, f8(self.stats[func][3]))
                left_width = name_size + 3
            print(indent*left_width + substats, file=self.stream)
            indent = " "

    def print_title(self):
        print('   ncalls  tottime  percall  cumtime  percall', end=' ', file=self.stream)
        print('filename:lineno(function)', file=self.stream)

    def print_line(self, func):  # hack: should print percentages
        cc, nc, tt, ct, callers = self.stats[func]
        c = str(nc)
        if nc != cc:
            c = c + '/' + str(cc)
        print(c.rjust(9), end=' ', file=self.stream)
        print(f8(tt), end=' ', file=self.stream)
        if nc == 0:
            print(' '*8, end=' ', file=self.stream)
        else:
            print(f8(tt/nc), end=' ', file=self.stream)
        print(f8(ct), end=' ', file=self.stream)
        if cc == 0:
            print(' '*8, end=' ', file=self.stream)
        else:
            print(f8(ct/cc), end=' ', file=self.stream)
        print(func_std_string(func), file=self.stream)

class TupleComp:
    """This class provides a generic function for comparing any two tuples.
    Each instance records a list of tuple-indices (from most significant
    to least significant), and sort direction (ascending or decending) for
    each tuple-index.  The compare functions can then be used as the function
    argument to the system sort() function when a list of tuples need to be
    sorted in the instances order."""

    def __init__(self, comp_select_list):
        self.comp_select_list = comp_select_list

    def compare (self, left, right):
        for index, direction in self.comp_select_list:
            l = left[index]
            r = right[index]
            if l < r:
                return -direction
            if l > r:
                return direction
        return 0


#**************************************************************************
# func_name is a triple (file:string, line:int, name:string)

def func_strip_path(func_name):
    filename, line, name = func_name
    return os.path.basename(filename), line, name

def func_get_function_name(func):
    return func[2]

def func_std_string(func_name): # match what old profile produced
    if func_name[:2] == ('~', 0):
        # special case for built-in functions
        name = func_name[2]
        if name.startswith('<') and name.endswith('>'):
            return '{%s}' % name[1:-1]
        else:
            return name
    else:
        return "%s:%d(%s)" % func_name

#**************************************************************************
# The following functions combine statistics for pairs functions.
# The bulk of the processing involves correctly handling "call" lists,
# such as callers and callees.
#**************************************************************************

def add_func_stats(target, source):
    """Add together all the stats for two profile entries."""
    cc, nc, tt, ct, callers = source
    t_cc, t_nc, t_tt, t_ct, t_callers = target
    return (cc+t_cc, nc+t_nc, tt+t_tt, ct+t_ct,
              add_callers(t_callers, callers))

def add_callers(target, source):
    """Combine two caller lists in a single list."""
    new_callers = {}
    for func, caller in target.items():
        new_callers[func] = caller
    for func, caller in source.items():
        if func in new_callers:
            if isinstance(caller, tuple):
                # format used by cProfile
                new_callers[func] = tuple(i + j for i, j in zip(caller, new_callers[func]))
            else:
                # format used by profile
                new_callers[func] += caller
        else:
            new_callers[func] = caller
    return new_callers

def count_calls(callers):
    """Sum the caller statistics to get total number of calls received."""
    nc = 0
    for calls in callers.values():
        nc += calls
    return nc

#**************************************************************************
# The following functions support printing of reports
#**************************************************************************

def f8(x):
    return "%8.3f" % x

#**************************************************************************
# Statistics browser added by ESR, April 2001
#**************************************************************************

if __name__ == '__main__':
    import cmd
    try:
        import readline
    except ImportError:
        pass

    class ProfileBrowser(cmd.Cmd):
        def __init__(self, profile=None):
            cmd.Cmd.__init__(self)
            self.prompt = "% "
            self.stats = None
            self.stream = sys.stdout
            if profile is not None:
                self.do_read(profile)

        def generic(self, fn, line):
            args = line.split()
            processed = []
            for term in args:
                try:
                    processed.append(int(term))
                    continue
                except ValueError:
                    pass
                try:
                    frac = float(term)
                    if frac > 1 or frac < 0:
                        print("Fraction argument must be in [0, 1]", file=self.stream)
                        continue
                    processed.append(frac)
                    continue
                except ValueError:
                    pass
                processed.append(term)
            if self.stats:
                getattr(self.stats, fn)(*processed)
            else:
                print("No statistics object is loaded.", file=self.stream)
            return 0
        def generic_help(self):
            print("Arguments may be:", file=self.stream)
            print("* An integer maximum number of entries to print.", file=self.stream)
            print("* A decimal fractional number between 0 and 1, controlling", file=self.stream)
            print("  what fraction of selected entries to print.", file=self.stream)
            print("* A regular expression; only entries with function names", file=self.stream)
            print("  that match it are printed.", file=self.stream)

        def do_add(self, line):
            if self.stats:
                try:
                    self.stats.add(line)
                except OSError as e:
                    print("Failed to load statistics for %s: %s" % (line, e), file=self.stream)
            else:
                print("No statistics object is loaded.", file=self.stream)
            return 0
        def help_add(self):
            print("Add profile info from given file to current statistics object.", file=self.stream)

        def do_callees(self, line):
            return self.generic('print_callees', line)
        def help_callees(self):
            print("Print callees statistics from the current stat object.", file=self.stream)
            self.generic_help()

        def do_callers(self, line):
            return self.generic('print_callers', line)
        def help_callers(self):
            print("Print callers statistics from the current stat object.", file=self.stream)
            self.generic_help()

        def do_EOF(self, line):
            print("", file=self.stream)
            return 1
        def help_EOF(self):
            print("Leave the profile browser.", file=self.stream)

        def do_quit(self, line):
            return 1
        def help_quit(self):
            print("Leave the profile browser.", file=self.stream)

        def do_read(self, line):
            if line:
                try:
                    self.stats = Stats(line)
                except OSError as err:
                    print(err.args[1], file=self.stream)
                    return
                except Exception as err:
                    print(err.__class__.__name__ + ':', err, file=self.stream)
                    return
                self.prompt = line + "% "
            elif len(self.prompt) > 2:
                line = self.prompt[:-2]
                self.do_read(line)
            else:
                print("No statistics object is current -- cannot reload.", file=self.stream)
            return 0
        def help_read(self):
            print("Read in profile data from a specified file.", file=self.stream)
            print("Without argument, reload the current file.", file=self.stream)

        def do_reverse(self, line):
            if self.stats:
                self.stats.reverse_order()
            else:
                print("No statistics object is loaded.", file=self.stream)
            return 0
        def help_reverse(self):
            print("Reverse the sort order of the profiling report.", file=self.stream)

        def do_sort(self, line):
            if not self.stats:
                print("No statistics object is loaded.", file=self.stream)
                return
            abbrevs = self.stats.get_sort_arg_defs()
            if line and all((x in abbrevs) for x in line.split()):
                self.stats.sort_stats(*line.split())
            else:
                print("Valid sort keys (unique prefixes are accepted):", file=self.stream)
                for (key, value) in Stats.sort_arg_dict_default.items():
                    print("%s -- %s" % (key, value[1]), file=self.stream)
            return 0
        def help_sort(self):
            print("Sort profile data according to specified keys.", file=self.stream)
            print("(Typing `sort' without arguments lists valid keys.)", file=self.stream)
        def complete_sort(self, text, *args):
            return [a for a in Stats.sort_arg_dict_default if a.startswith(text)]

        def do_stats(self, line):
            return self.generic('print_stats', line)
        def help_stats(self):
            print("Print statistics from the current stat object.", file=self.stream)
            self.generic_help()

        def do_strip(self, line):
            if self.stats:
                self.stats.strip_dirs()
            else:
                print("No statistics object is loaded.", file=self.stream)
        def help_strip(self):
            print("Strip leading path information from filenames in the report.", file=self.stream)

        def help_help(self):
            print("Show help for a given command.", file=self.stream)

        def postcmd(self, stop, line):
            if stop:
                return stop
            return None

    if len(sys.argv) > 1:
        initprofile = sys.argv[1]
    else:
        initprofile = None
    try:
        browser = ProfileBrowser(initprofile)
        for profile in sys.argv[2:]:
            browser.do_add(profile)
        print("Welcome to the profile statistics browser.", file=browser.stream)
        browser.cmdloop()
        print("Goodbye.", file=browser.stream)
    except KeyboardInterrupt:
        pass

# That's all, folks.
<|endoftext|>"""Utilities needed to emulate Python's interactive interpreter.

"""

# Inspired by similar code by Jeff Epler and Fredrik Lundh.


import sys
import traceback
from codeop import CommandCompiler, compile_command

__all__ = ["InteractiveInterpreter", "InteractiveConsole", "interact",
           "compile_command"]

class InteractiveInterpreter:
    """Base class for InteractiveConsole.

    This class deals with parsing and interpreter state (the user's
    namespace); it doesn't deal with input buffering or prompting or
    input file naming (the filename is always passed in explicitly).

    """

    def __init__(self, locals=None):
        """Constructor.

        The optional 'locals' argument specifies the dictionary in
        which code will be executed; it defaults to a newly created
        dictionary with key "__name__" set to "__console__" and key
        "__doc__" set to None.

        """
        if locals is None:
            locals = {"__name__": "__console__", "__doc__": None}
        self.locals = locals
        self.compile = CommandCompiler()

    def runsource(self, source, filename="<input>", symbol="single"):
        """Compile and run some source in the interpreter.

        Arguments are as for compile_command().

        One of several things can happen:

        1) The input is incorrect; compile_command() raised an
        exception (SyntaxError or OverflowError).  A syntax traceback
        will be printed by calling the showsyntaxerror() method.

        2) The input is incomplete, and more input is required;
        compile_command() returned None.  Nothing happens.

        3) The input is complete; compile_command() returned a code
        object.  The code is executed by calling self.runcode() (which
        also handles run-time exceptions, except for SystemExit).

        The return value is True in case 2, False in the other cases (unless
        an exception is raised).  The return value can be used to
        decide whether to use sys.ps1 or sys.ps2 to prompt the next
        line.

        """
        try:
            code = self.compile(source, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            # Case 1
            self.showsyntaxerror(filename)
            return False

        if code is None:
            # Case 2
            return True

        # Case 3
        self.runcode(code)
        return False

    def runcode(self, code):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to
        display a traceback.  All exceptions are caught except
        SystemExit, which is reraised.

        A note about KeyboardInterrupt: this exception may occur
        elsewhere in this code, and may not always be caught.  The
        caller should be prepared to deal with it.

        """
        try:
            exec(code, self.locals)
        except SystemExit:
            raise
        except:
            self.showtraceback()

    def showsyntaxerror(self, filename=None):
        """Display the syntax error that just occurred.

        This doesn't display a stack trace because there isn't one.

        If a filename is given, it is stuffed in the exception instead
        of what was there before (because Python's parser always uses
        "<string>" when reading from a string).

        The output is written by self.write(), below.

        """
        type, value, tb = sys.exc_info()
        sys.last_type = type
        sys.last_value = value
        sys.last_traceback = tb
        if filename and type is SyntaxError:
            # Work hard to stuff the correct filename in the exception
            try:
                msg, (dummy_filename, lineno, offset, line) = value.args
            except ValueError:
                # Not the format we expect; leave it alone
                pass
            else:
                # Stuff in the right filename
                value = SyntaxError(msg, (filename, lineno, offset, line))
                sys.last_value = value
        if sys.excepthook is sys.__excepthook__:
            lines = traceback.format_exception_only(type, value)
            self.write(''.join(lines))
        else:
            # If someone has set sys.excepthook, we let that take precedence
            # over self.write
            sys.excepthook(type, value, tb)

    def showtraceback(self):
        """Display the exception that just occurred.

        We remove the first stack item because it is our own code.

        The output is written by self.write(), below.

        """
        sys.last_type, sys.last_value, last_tb = ei = sys.exc_info()
        sys.last_traceback = last_tb
        try:
            lines = traceback.format_exception(ei[0], ei[1], last_tb.tb_next)
            if sys.excepthook is sys.__excepthook__:
                self.write(''.join(lines))
            else:
                # If someone has set sys.excepthook, we let that take precedence
                # over self.write
                sys.excepthook(ei[0], ei[1], last_tb)
        finally:
            last_tb = ei = None

    def write(self, data):
        """Write a string.

        The base implementation writes to sys.stderr; a subclass may
        replace this with a different implementation.

        """
        sys.stderr.write(data)


class InteractiveConsole(InteractiveInterpreter):
    """Closely emulate the behavior of the interactive Python interpreter.

    This class builds on InteractiveInterpreter and adds prompting
    using the familiar sys.ps1 and sys.ps2, and input buffering.

    """

    def __init__(self, locals=None, filename="<console>"):
        """Constructor.

        The optional locals argument will be passed to the
        InteractiveInterpreter base class.

        The optional filename argument should specify the (file)name
        of the input stream; it will show up in tracebacks.

        """
        InteractiveInterpreter.__init__(self, locals)
        self.filename = filename
        self.resetbuffer()

    def resetbuffer(self):
        """Reset the input buffer."""
        self.buffer = []

    def interact(self, banner=None, exitmsg=None):
        """Closely emulate the interactive Python console.

        The optional banner argument specifies the banner to print
        before the first interaction; by default it prints a banner
        similar to the one printed by the real Python interpreter,
        followed by the current class name in parentheses (so as not
        to confuse this with the real interpreter -- since it's so
        close!).

        The optional exitmsg argument specifies the exit message
        printed when exiting. Pass the empty string to suppress
        printing an exit message. If exitmsg is not given or None,
        a default message is printed.

        """
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = ">>> "
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = "... "
        cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
        if banner is None:
            self.write("Python %s on %s\n%s\n(%s)\n" %
                       (sys.version, sys.platform, cprt,
                        self.__class__.__name__))
        elif banner:
            self.write("%s\n" % str(banner))
        more = 0
        while 1:
            try:
                if more:
                    prompt = sys.ps2
                else:
                    prompt = sys.ps1
                try:
                    line = self.raw_input(prompt)
                except EOFError:
                    self.write("\n")
                    break
                else:
                    more = self.push(line)
            except KeyboardInterrupt:
                self.write("\nKeyboardInterrupt\n")
                self.resetbuffer()
                more = 0
        if exitmsg is None:
            self.write('now exiting %s...\n' % self.__class__.__name__)
        elif exitmsg != '':
            self.write('%s\n' % exitmsg)

    def push(self, line):
        """Push a line to the interpreter.

        The line should not have a trailing newline; it may have
        internal newlines.  The line is appended to a buffer and the
        interpreter's runsource() method is called with the
        concatenated contents of the buffer as source.  If this
        indicates that the command was executed or invalid, the buffer
        is reset; otherwise, the command is incomplete, and the buffer
        is left as it was after the line was appended.  The return
        value is 1 if more input is required, 0 if the line was dealt
        with in some way (this is the same as runsource()).

        """
        self.buffer.append(line)
        source = "\n".join(self.buffer)
        more = self.runsource(source, self.filename)
        if not more:
            self.resetbuffer()
        return more

    def raw_input(self, prompt=""):
        """Write a prompt and read a line.

        The returned line does not include the trailing newline.
        When the user enters the EOF key sequence, EOFError is raised.

        The base implementation uses the built-in function
        input(); a subclass may replace this with a different
        implementation.

        """
        return input(prompt)



def interact(banner=None, readfunc=None, local=None, exitmsg=None):
    """Closely emulate the interactive Python interpreter.

    This is a backwards compatible interface to the InteractiveConsole
    class.  When readfunc is not specified, it attempts to import the
    readline module to enable GNU readline if it is available.

    Arguments (all optional, all default to None):

    banner -- passed to InteractiveConsole.interact()
    readfunc -- if not None, replaces InteractiveConsole.raw_input()
    local -- passed to InteractiveInterpreter.__init__()
    exitmsg -- passed to InteractiveConsole.interact()

    """
    console = InteractiveConsole(local)
    if readfunc is not None:
        console.raw_input = readfunc
    else:
        try:
            import readline
        except ImportError:
            pass
    console.interact(banner, exitmsg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', action='store_true',
                       help="don't print version and copyright messages")
    args = parser.parse_args()
    if args.q or sys.flags.quiet:
        banner = ''
    else:
        banner = None
    interact(banner)
<|endoftext|>"""Tokenization help for Python programs.

tokenize(readline) is a generator that breaks a stream of bytes into
Python tokens.  It decodes the bytes according to PEP-0263 for
determining source file encoding.

It accepts a readline-like method which is called repeatedly to get the
next line of input (or b"" for EOF).  It generates 5-tuples with these
members:

    the token type (see token.py)
    the token (a string)
    the starting (row, column) indices of the token (a 2-tuple of ints)
    the ending (row, column) indices of the token (a 2-tuple of ints)
    the original line (string)

It is designed to match the working of the Python tokenizer exactly, except
that it produces COMMENT tokens for comments and gives type OP for all
operators.  Additionally, all token lists start with an ENCODING token
which tells you which encoding was used to decode the bytes stream.
"""

__author__ = 'Ka-Ping Yee <ping@lfw.org>'
__credits__ = ('GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, '
               'Skip Montanaro, Raymond Hettinger, Trent Nelson, '
               'Michael Foord')
from builtins import open as _builtin_open
from codecs import lookup, BOM_UTF8
import collections
import functools
from io import TextIOWrapper
import itertools as _itertools
import re
import sys
from token import *
from token import EXACT_TOKEN_TYPES

cookie_re = re.compile(r'^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)', re.ASCII)
blank_re = re.compile(br'^[ \t\f]*(?:[#\r\n]|$)', re.ASCII)

import token
__all__ = token.__all__ + ["tokenize", "generate_tokens", "detect_encoding",
                           "untokenize", "TokenInfo"]
del token

class TokenInfo(collections.namedtuple('TokenInfo', 'type string start end line')):
    def __repr__(self):
        annotated_type = '%d (%s)' % (self.type, tok_name[self.type])
        return ('TokenInfo(type=%s, string=%r, start=%r, end=%r, line=%r)' %
                self._replace(type=annotated_type))

    @property
    def exact_type(self):
        if self.type == OP and self.string in EXACT_TOKEN_TYPES:
            return EXACT_TOKEN_TYPES[self.string]
        else:
            return self.type

def group(*choices): return '(' + '|'.join(choices) + ')'
def any(*choices): return group(*choices) + '*'
def maybe(*choices): return group(*choices) + '?'

# Note: we use unicode matching for names ("\w") but ascii matching for
# number literals.
Whitespace = r'[ \f\t]*'
Comment = r'#[^\r\n]*'
Ignore = Whitespace + any(r'\\\r?\n' + Whitespace) + maybe(Comment)
Name = r'\w+'

Hexnumber = r'0[xX](?:_?[0-9a-fA-F])+'
Binnumber = r'0[bB](?:_?[01])+'
Octnumber = r'0[oO](?:_?[0-7])+'
Decnumber = r'(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
Exponent = r'[eE][-+]?[0-9](?:_?[0-9])*'
Pointfloat = group(r'[0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?',
                   r'\.[0-9](?:_?[0-9])*') + maybe(Exponent)
Expfloat = r'[0-9](?:_?[0-9])*' + Exponent
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group(r'[0-9](?:_?[0-9])*[jJ]', Floatnumber + r'[jJ]')
Number = group(Imagnumber, Floatnumber, Intnumber)

# Return the empty string, plus all of the valid string prefixes.
def _all_string_prefixes():
    # The valid string prefixes. Only contain the lower case versions,
    #  and don't contain any permutations (include 'fr', but not
    #  'rf'). The various permutations will be generated.
    _valid_string_prefixes = ['b', 'r', 'u', 'f', 'br', 'fr']
    # if we add binary f-strings, add: ['fb', 'fbr']
    result = {''}
    for prefix in _valid_string_prefixes:
        for t in _itertools.permutations(prefix):
            # create a list with upper and lower versions of each
            #  character
            for u in _itertools.product(*[(c, c.upper()) for c in t]):
                result.add(''.join(u))
    return result

@functools.lru_cache
def _compile(expr):
    return re.compile(expr, re.UNICODE)

# Note that since _all_string_prefixes includes the empty string,
#  StringPrefix can be the empty string (making it optional).
StringPrefix = group(*_all_string_prefixes())

# Tail end of ' string.
Single = r"[^'\\]*(?:\\.[^'\\]*)*'"
# Tail end of " string.
Double = r'[^"\\]*(?:\\.[^"\\]*)*"'
# Tail end of ''' string.
Single3 = r"[^'\\]*(?:(?:\\.|'(?!''))[^'\\]*)*'''"
# Tail end of """ string.
Double3 = r'[^"\\]*(?:(?:\\.|"(?!""))[^"\\]*)*"""'
Triple = group(StringPrefix + "'''", StringPrefix + '"""')
# Single-line ' or " string.
String = group(StringPrefix + r"'[^\n'\\]*(?:\\.[^\n'\\]*)*'",
               StringPrefix + r'"[^\n"\\]*(?:\\.[^\n"\\]*)*"')

# Sorting in reverse order puts the long operators before their prefixes.
# Otherwise if = came before ==, == would get recognized as two instances
# of =.
Special = group(*map(re.escape, sorted(EXACT_TOKEN_TYPES, reverse=True)))
Funny = group(r'\r?\n', Special)

PlainToken = group(Number, Funny, String, Name)
Token = Ignore + PlainToken

# First (or only) line of ' or " string.
ContStr = group(StringPrefix + r"'[^\n'\\]*(?:\\.[^\n'\\]*)*" +
                group("'", r'\\\r?\n'),
                StringPrefix + r'"[^\n"\\]*(?:\\.[^\n"\\]*)*' +
                group('"', r'\\\r?\n'))
PseudoExtras = group(r'\\\r?\n|\Z', Comment, Triple)
PseudoToken = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)

# For a given string prefix plus quotes, endpats maps it to a regex
#  to match the remainder of that string. _prefix can be empty, for
#  a normal single or triple quoted string (with no prefix).
endpats = {}
for _prefix in _all_string_prefixes():
    endpats[_prefix + "'"] = Single
    endpats[_prefix + '"'] = Double
    endpats[_prefix + "'''"] = Single3
    endpats[_prefix + '"""'] = Double3

# A set of all of the single and triple quoted string prefixes,
#  including the opening quotes.
single_quoted = set()
triple_quoted = set()
for t in _all_string_prefixes():
    for u in (t + '"', t + "'"):
        single_quoted.add(u)
    for u in (t + '"""', t + "'''"):
        triple_quoted.add(u)

tabsize = 8

class TokenError(Exception): pass

class StopTokenizing(Exception): pass


class Untokenizer:

    def __init__(self):
        self.tokens = []
        self.prev_row = 1
        self.prev_col = 0
        self.encoding = None

    def add_whitespace(self, start):
        row, col = start
        if row < self.prev_row or row == self.prev_row and col < self.prev_col:
            raise ValueError("start ({},{}) precedes previous end ({},{})"
                             .format(row, col, self.prev_row, self.prev_col))
        row_offset = row - self.prev_row
        if row_offset:
            self.tokens.append("\\\n" * row_offset)
            self.prev_col = 0
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(" " * col_offset)

    def untokenize(self, iterable):
        it = iter(iterable)
        indents = []
        startline = False
        for t in it:
            if len(t) == 2:
                self.compat(t, it)
                break
            tok_type, token, start, end, line = t
            if tok_type == ENCODING:
                self.encoding = token
                continue
            if tok_type == ENDMARKER:
                break
            if tok_type == INDENT:
                indents.append(token)
                continue
            elif tok_type == DEDENT:
                indents.pop()
                self.prev_row, self.prev_col = end
                continue
            elif tok_type in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                indent = indents[-1]
                if start[1] >= len(indent):
                    self.tokens.append(indent)
                    self.prev_col = len(indent)
                startline = False
            self.add_whitespace(start)
            self.tokens.append(token)
            self.prev_row, self.prev_col = end
            if tok_type in (NEWLINE, NL):
                self.prev_row += 1
                self.prev_col = 0
        return "".join(self.tokens)

    def compat(self, token, iterable):
        indents = []
        toks_append = self.tokens.append
        startline = token[0] in (NEWLINE, NL)
        prevstring = False

        for tok in _itertools.chain([token], iterable):
            toknum, tokval = tok[:2]
            if toknum == ENCODING:
                self.encoding = tokval
                continue

            if toknum in (NAME, NUMBER):
                tokval += ' '

            # Insert a space between two consecutive strings
            if toknum == STRING:
                if prevstring:
                    tokval = ' ' + tokval
                prevstring = True
            else:
                prevstring = False

            if toknum == INDENT:
                indents.append(tokval)
                continue
            elif toknum == DEDENT:
                indents.pop()
                continue
            elif toknum in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                toks_append(indents[-1])
                startline = False
            toks_append(tokval)


def untokenize(iterable):
    """Transform tokens back into Python source code.
    It returns a bytes object, encoded using the ENCODING
    token, which is the first token sequence output by tokenize.

    Each element returned by the iterable must be a token sequence
    with at least two elements, a token number and token value.  If
    only two tokens are passed, the resulting output is poor.

    Round-trip invariant for full input:
        Untokenized source will match input source exactly

    Round-trip invariant for limited input:
        # Output bytes will tokenize back to the input
        t1 = [tok[:2] for tok in tokenize(f.readline)]
        newcode = untokenize(t1)
        readline = BytesIO(newcode).readline
        t2 = [tok[:2] for tok in tokenize(readline)]
        assert t1 == t2
    """
    ut = Untokenizer()
    out = ut.untokenize(iterable)
    if ut.encoding is not None:
        out = out.encode(ut.encoding)
    return out


def _get_normal_name(orig_enc):
    """Imitates get_normal_name in tokenizer.c."""
    # Only care about the first 12 characters.
    enc = orig_enc[:12].lower().replace("_", "-")
    if enc == "utf-8" or enc.startswith("utf-8-"):
        return "utf-8"
    if enc in ("latin-1", "iso-8859-1", "iso-latin-1") or \
       enc.startswith(("latin-1-", "iso-8859-1-", "iso-latin-1-")):
        return "iso-8859-1"
    return orig_enc

def detect_encoding(readline):
    """
    The detect_encoding() function is used to detect the encoding that should
    be used to decode a Python source file.  It requires one argument, readline,
    in the same way as the tokenize() generator.

    It will call readline a maximum of twice, and return the encoding used
    (as a string) and a list of any lines (left as bytes) it has read in.

    It detects the encoding from the presence of a utf-8 bom or an encoding
    cookie as specified in pep-0263.  If both a bom and a cookie are present,
    but disagree, a SyntaxError will be raised.  If the encoding cookie is an
    invalid charset, raise a SyntaxError.  Note that if a utf-8 bom is found,
    'utf-8-sig' is returned.

    If no encoding is specified, then the default of 'utf-8' will be returned.
    """
    try:
        filename = readline.__self__.name
    except AttributeError:
        filename = None
    bom_found = False
    encoding = None
    default = 'utf-8'
    def read_or_stop():
        try:
            return readline()
        except StopIteration:
            return b''

    def find_cookie(line):
        try:
            # Decode as UTF-8. Either the line is an encoding declaration,
            # in which case it should be pure ASCII, or it must be UTF-8
            # per default encoding.
            line_string = line.decode('utf-8')
        except UnicodeDecodeError:
            msg = "invalid or missing encoding declaration"
            if filename is not None:
                msg = '{} for {!r}'.format(msg, filename)
            raise SyntaxError(msg)

        match = cookie_re.match(line_string)
        if not match:
            return None
        encoding = _get_normal_name(match.group(1))
        try:
            codec = lookup(encoding)
        except LookupError:
            # This behaviour mimics the Python interpreter
            if filename is None:
                msg = "unknown encoding: " + encoding
            else:
                msg = "unknown encoding for {!r}: {}".format(filename,
                        encoding)
            raise SyntaxError(msg)

        if bom_found:
            if encoding != 'utf-8':
                # This behaviour mimics the Python interpreter
                if filename is None:
                    msg = 'encoding problem: utf-8'
                else:
                    msg = 'encoding problem for {!r}: utf-8'.format(filename)
                raise SyntaxError(msg)
            encoding += '-sig'
        return encoding

    first = read_or_stop()
    if first.startswith(BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = 'utf-8-sig'
    if not first:
        return default, []

    encoding = find_cookie(first)
    if encoding:
        return encoding, [first]
    if not blank_re.match(first):
        return default, [first]

    second = read_or_stop()
    if not second:
        return default, [first]

    encoding = find_cookie(second)
    if encoding:
        return encoding, [first, second]

    return default, [first, second]


def open(filename):
    """Open a file in read only mode using the encoding detected by
    detect_encoding().
    """
    buffer = _builtin_open(filename, 'rb')
    try:
        encoding, lines = detect_encoding(buffer.readline)
        buffer.seek(0)
        text = TextIOWrapper(buffer, encoding, line_buffering=True)
        text.mode = 'r'
        return text
    except:
        buffer.close()
        raise


def tokenize(readline):
    """
    The tokenize() generator requires one argument, readline, which
    must be a callable object which provides the same interface as the
    readline() method of built-in file objects.  Each call to the function
    should return one line of input as bytes.  Alternatively, readline
    can be a callable function terminating with StopIteration:
        readline = open(myfile, 'rb').__next__  # Example of alternate readline

    The generator produces 5-tuples with these members: the token type; the
    token string; a 2-tuple (srow, scol) of ints specifying the row and
    column where the token begins in the source; a 2-tuple (erow, ecol) of
    ints specifying the row and column where the token ends in the source;
    and the line on which the token was found.  The line passed is the
    physical line.

    The first token sequence will always be an ENCODING token
    which tells you which encoding was used to decode the bytes stream.
    """
    encoding, consumed = detect_encoding(readline)
    empty = _itertools.repeat(b"")
    rl_gen = _itertools.chain(consumed, iter(readline, b""), empty)
    return _tokenize(rl_gen.__next__, encoding)


def _tokenize(readline, encoding):
    lnum = parenlev = continued = 0
    numchars = '0123456789'
    contstr, needcont = '', 0
    contline = None
    indents = [0]

    if encoding is not None:
        if encoding == "utf-8-sig":
            # BOM will already have been stripped.
            encoding = "utf-8"
        yield TokenInfo(ENCODING, encoding, (0, 0), (0, 0), '')
    last_line = b''
    line = b''
    while True:                                # loop over lines in stream
        try:
            # We capture the value of the line variable here because
            # readline uses the empty string '' to signal end of input,
            # hence `line` itself will always be overwritten at the end
            # of this loop.
            last_line = line
            line = readline()
        except StopIteration:
            line = b''

        if encoding is not None:
            line = line.decode(encoding)
        lnum += 1
        pos, max = 0, len(line)

        if contstr:                            # continued string
            if not line:
                raise TokenError("EOF in multi-line string", strstart)
            endmatch = endprog.match(line)
            if endmatch:
                pos = end = endmatch.end(0)
                yield TokenInfo(STRING, contstr + line[:end],
                       strstart, (lnum, end), contline + line)
                contstr, needcont = '', 0
                contline = None
            elif needcont and line[-2:] != '\\\n' and line[-3:] != '\\\r\n':
                yield TokenInfo(ERRORTOKEN, contstr + line,
                           strstart, (lnum, len(line)), contline)
                contstr = ''
                contline = None
                continue
            else:
                contstr = contstr + line
                contline = contline + line
                continue

        elif parenlev == 0 and not continued:  # new statement
            if not line: break
            column = 0
            while pos < max:                   # measure leading whitespace
                if line[pos] == ' ':
                    column += 1
                elif line[pos] == '\t':
                    column = (column//tabsize + 1)*tabsize
                elif line[pos] == '\f':
                    column = 0
                else:
                    break
                pos += 1
            if pos == max:
                break

            if line[pos] in '#\r\n':           # skip comments or blank lines
                if line[pos] == '#':
                    comment_token = line[pos:].rstrip('\r\n')
                    yield TokenInfo(COMMENT, comment_token,
                           (lnum, pos), (lnum, pos + len(comment_token)), line)
                    pos += len(comment_token)

                yield TokenInfo(NL, line[pos:],
                           (lnum, pos), (lnum, len(line)), line)
                continue

            if column > indents[-1]:           # count indents or dedents
                indents.append(column)
                yield TokenInfo(INDENT, line[:pos], (lnum, 0), (lnum, pos), line)
            while column < indents[-1]:
                if column not in indents:
                    raise IndentationError(
                        "unindent does not match any outer indentation level",
                        ("<tokenize>", lnum, pos, line))
                indents = indents[:-1]

                yield TokenInfo(DEDENT, '', (lnum, pos), (lnum, pos), line)

        else:                                  # continued statement
            if not line:
                raise TokenError("EOF in multi-line statement", (lnum, 0))
            continued = 0

        while pos < max:
            pseudomatch = _compile(PseudoToken).match(line, pos)
            if pseudomatch:                                # scan for tokens
                start, end = pseudomatch.span(1)
                spos, epos, pos = (lnum, start), (lnum, end), end
                if start == end:
                    continue
                token, initial = line[start:end], line[start]

                if (initial in numchars or                 # ordinary number
                    (initial == '.' and token != '.' and token != '...')):
                    yield TokenInfo(NUMBER, token, spos, epos, line)
                elif initial in '\r\n':
                    if parenlev > 0:
                        yield TokenInfo(NL, token, spos, epos, line)
                    else:
                        yield TokenInfo(NEWLINE, token, spos, epos, line)

                elif initial == '#':
                    assert not token.endswith("\n")
                    yield TokenInfo(COMMENT, token, spos, epos, line)

                elif token in triple_quoted:
                    endprog = _compile(endpats[token])
                    endmatch = endprog.match(line, pos)
                    if endmatch:                           # all on one line
                        pos = endmatch.end(0)
                        token = line[start:pos]
                        yield TokenInfo(STRING, token, spos, (lnum, pos), line)
                    else:
                        strstart = (lnum, start)           # multiple lines
                        contstr = line[start:]
                        contline = line
                        break

                # Check up to the first 3 chars of the token to see if
                #  they're in the single_quoted set. If so, they start
                #  a string.
                # We're using the first 3, because we're looking for
                #  "rb'" (for example) at the start of the token. If
                #  we switch to longer prefixes, this needs to be
                #  adjusted.
                # Note that initial == token[:1].
                # Also note that single quote checking must come after
                #  triple quote checking (above).
                elif (initial in single_quoted or
                      token[:2] in single_quoted or
                      token[:3] in single_quoted):
                    if token[-1] == '\n':                  # continued string
                        strstart = (lnum, start)
                        # Again, using the first 3 chars of the
                        #  token. This is looking for the matching end
                        #  regex for the correct type of quote
                        #  character. So it's really looking for
                        #  endpats["'"] or endpats['"'], by trying to
                        #  skip string prefix characters, if any.
                        endprog = _compile(endpats.get(initial) or
                                           endpats.get(token[1]) or
                                           endpats.get(token[2]))
                        contstr, needcont = line[start:], 1
                        contline = line
                        break
                    else:                                  # ordinary string
                        yield TokenInfo(STRING, token, spos, epos, line)

                elif initial.isidentifier():               # ordinary name
                    yield TokenInfo(NAME, token, spos, epos, line)
                elif initial == '\\':                      # continued stmt
                    continued = 1
                else:
                    if initial in '([{':
                        parenlev += 1
                    elif initial in ')]}':
                        parenlev -= 1
                    yield TokenInfo(OP, token, spos, epos, line)
            else:
                yield TokenInfo(ERRORTOKEN, line[pos],
                           (lnum, pos), (lnum, pos+1), line)
                pos += 1

    # Add an implicit NEWLINE if the input doesn't end in one
    if last_line and last_line[-1] not in '\r\n' and not last_line.strip().startswith("#"):
        yield TokenInfo(NEWLINE, '', (lnum - 1, len(last_line)), (lnum - 1, len(last_line) + 1), '')
    for indent in indents[1:]:                 # pop remaining indent levels
        yield TokenInfo(DEDENT, '', (lnum, 0), (lnum, 0), '')
    yield TokenInfo(ENDMARKER, '', (lnum, 0), (lnum, 0), '')


def generate_tokens(readline):
    """Tokenize a source reading Python code as unicode strings.

    This has the same API as tokenize(), except that it expects the *readline*
    callable to return str objects instead of bytes.
    """
    return _tokenize(readline, None)

def main():
    import argparse

    # Helper error handling routines
    def perror(message):
        sys.stderr.write(message)
        sys.stderr.write('\n')

    def error(message, filename=None, location=None):
        if location:
            args = (filename,) + location + (message,)
            perror("%s:%d:%d: error: %s" % args)
        elif filename:
            perror("%s: error: %s" % (filename, message))
        else:
            perror("error: %s" % message)
        sys.exit(1)

    # Parse the arguments and options
    parser = argparse.ArgumentParser(prog='python -m tokenize')
    parser.add_argument(dest='filename', nargs='?',
                        metavar='filename.py',
                        help='the file to tokenize; defaults to stdin')
    parser.add_argument('-e', '--exact', dest='exact', action='store_true',
                        help='display token names using the exact type')
    args = parser.parse_args()

    try:
        # Tokenize the input
        if args.filename:
            filename = args.filename
            with _builtin_open(filename, 'rb') as f:
                tokens = list(tokenize(f.readline))
        else:
            filename = "<stdin>"
            tokens = _tokenize(sys.stdin.readline, None)

        # Output the tokenization
        for token in tokens:
            token_type = token.type
            if args.exact:
                token_type = token.exact_type
            token_range = "%d,%d-%d,%d:" % (token.start + token.end)
            print("%-20s%-15s%-15r" %
                  (token_range, tok_name[token_type], token.string))
    except IndentationError as err:
        line, column = err.args[1][1:3]
        error(err.args[0], filename, (line, column))
    except TokenError as err:
        line, column = err.args[1]
        error(err.args[0], filename, (line, column))
    except SyntaxError as err:
        error(err, filename)
    except OSError as err:
        error(err)
    except KeyboardInterrupt:
        print("interrupted\n")
    except Exception as err:
        perror("unexpected error: %s" % err)
        raise

if __name__ == "__main__":
    main()
<|endoftext|>"""Constants/functions for interpreting results of os.stat() and os.lstat().

Suggested usage: from stat import *
"""

