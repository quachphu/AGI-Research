        signature as time.localtime() or time.gmtime(). To change it for all
        formatters, for example if you want all logging times to be shown in GMT,
        set the 'converter' attribute in the Formatter class.
        """
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime(self.default_time_format, ct)
            if self.default_msec_format:
                s = self.default_msec_format % (s, record.msecs)
        return s

    def formatException(self, ei):
        """
        Format and return the specified exception information as a string.

        This default implementation just uses
        traceback.print_exception()
        """
        sio = io.StringIO()
        tb = ei[2]
        # See issues #9427, #1553375. Commented out for now.
        #if getattr(self, 'fullstack', False):
        #    traceback.print_stack(tb.tb_frame.f_back, file=sio)
        traceback.print_exception(ei[0], ei[1], tb, None, sio)
        s = sio.getvalue()
        sio.close()
        if s[-1:] == "\n":
            s = s[:-1]
        return s

    def usesTime(self):
        """
        Check if the format uses the creation time of the record.
        """
        return self._style.usesTime()

    def formatMessage(self, record):
        return self._style.format(record)

    def formatStack(self, stack_info):
        """
        This method is provided as an extension point for specialized
        formatting of stack information.

        The input data is a string as returned from a call to
        :func:`traceback.print_stack`, but with the last trailing newline
        removed.

        The base implementation just returns the value passed in.
        """
        return stack_info

    def format(self, record):
        """
        Format the specified record as text.

        The record's attribute dictionary is used as the operand to a
        string formatting operation which yields the returned string.
        Before formatting the dictionary, a couple of preparatory steps
        are carried out. The message attribute of the record is computed
        using LogRecord.getMessage(). If the formatting string uses the
        time (as determined by a call to usesTime(), formatTime() is
        called to format the event time. If there is exception information,
        it is formatted using formatException() and appended to the message.
        """
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + record.exc_text
        if record.stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(record.stack_info)
        return s

#
#   The default formatter to use when no other is specified
#
_defaultFormatter = Formatter()

class BufferingFormatter(object):
    """
    A formatter suitable for formatting a number of records.
    """
    def __init__(self, linefmt=None):
        """
        Optionally specify a formatter which will be used to format each
        individual record.
        """
        if linefmt:
            self.linefmt = linefmt
        else:
            self.linefmt = _defaultFormatter

    def formatHeader(self, records):
        """
        Return the header string for the specified records.
        """
        return ""

    def formatFooter(self, records):
        """
        Return the footer string for the specified records.
        """
        return ""

    def format(self, records):
        """
        Format the specified records and return the result as a string.
        """
        rv = ""
        if len(records) > 0:
            rv = rv + self.formatHeader(records)
            for record in records:
                rv = rv + self.linefmt.format(record)
            rv = rv + self.formatFooter(records)
        return rv

#---------------------------------------------------------------------------
#   Filter classes and functions
#---------------------------------------------------------------------------

class Filter(object):
    """
    Filter instances are used to perform arbitrary filtering of LogRecords.

    Loggers and Handlers can optionally use Filter instances to filter
    records as desired. The base filter class only allows events which are
    below a certain point in the logger hierarchy. For example, a filter
    initialized with "A.B" will allow events logged by loggers "A.B",
    "A.B.C", "A.B.C.D", "A.B.D" etc. but not "A.BB", "B.A.B" etc. If
    initialized with the empty string, all events are passed.
    """
    def __init__(self, name=''):
        """
        Initialize a filter.

        Initialize with the name of the logger which, together with its
        children, will have its events allowed through the filter. If no
        name is specified, allow every event.
        """
        self.name = name
        self.nlen = len(name)

    def filter(self, record):
        """
        Determine if the specified record is to be logged.

        Returns True if the record should be logged, or False otherwise.
        If deemed appropriate, the record may be modified in-place.
        """
        if self.nlen == 0:
            return True
        elif self.name == record.name:
            return True
        elif record.name.find(self.name, 0, self.nlen) != 0:
            return False
        return (record.name[self.nlen] == ".")

class Filterer(object):
    """
    A base class for loggers and handlers which allows them to share
    common code.
    """
    def __init__(self):
        """
        Initialize the list of filters to be an empty list.
        """
        self.filters = []

    def addFilter(self, filter):
        """
        Add the specified filter to this handler.
        """
        if not (filter in self.filters):
            self.filters.append(filter)

    def removeFilter(self, filter):
        """
        Remove the specified filter from this handler.
        """
        if filter in self.filters:
            self.filters.remove(filter)

    def filter(self, record):
        """
        Determine if a record is loggable by consulting all the filters.

        The default is to allow the record to be logged; any filter can veto
        this and the record is then dropped. Returns a zero value if a record
        is to be dropped, else non-zero.

        .. versionchanged:: 3.2

           Allow filters to be just callables.
        """
        rv = True
        for f in self.filters:
            if hasattr(f, 'filter'):
                result = f.filter(record)
            else:
                result = f(record) # assume callable - will raise if not
            if not result:
                rv = False
                break
        return rv

#---------------------------------------------------------------------------
#   Handler classes and functions
#---------------------------------------------------------------------------

_handlers = weakref.WeakValueDictionary()  #map of handler names to handlers
_handlerList = [] # added to allow handlers to be removed in reverse of order initialized

def _removeHandlerRef(wr):
    """
    Remove a handler reference from the internal cleanup list.
    """
    # This function can be called during module teardown, when globals are
    # set to None. It can also be called from another thread. So we need to
    # pre-emptively grab the necessary globals and check if they're None,
    # to prevent race conditions and failures during interpreter shutdown.
    acquire, release, handlers = _acquireLock, _releaseLock, _handlerList
    if acquire and release and handlers:
        acquire()
        try:
            if wr in handlers:
                handlers.remove(wr)
        finally:
            release()

def _addHandlerRef(handler):
    """
    Add a handler to the internal cleanup list using a weak reference.
    """
    _acquireLock()
    try:
        _handlerList.append(weakref.ref(handler, _removeHandlerRef))
    finally:
        _releaseLock()

class Handler(Filterer):
    """
    Handler instances dispatch logging events to specific destinations.

    The base handler class. Acts as a placeholder which defines the Handler
    interface. Handlers can optionally use Formatter instances to format
    records as desired. By default, no formatter is specified; in this case,
    the 'raw' message as determined by record.message is logged.
    """
    def __init__(self, level=NOTSET):
        """
        Initializes the instance - basically setting the formatter to None
        and the filter list to empty.
        """
        Filterer.__init__(self)
        self._name = None
        self.level = _checkLevel(level)
        self.formatter = None
        self._closed = False
        # Add the handler to the global _handlerList (for cleanup on shutdown)
        _addHandlerRef(self)
        self.createLock()

    def get_name(self):
        return self._name

    def set_name(self, name):
        _acquireLock()
        try:
            if self._name in _handlers:
                del _handlers[self._name]
            self._name = name
            if name:
                _handlers[name] = self
        finally:
            _releaseLock()

    name = property(get_name, set_name)

    def createLock(self):
        """
        Acquire a thread lock for serializing access to the underlying I/O.
        """
        self.lock = threading.RLock()
        _register_at_fork_reinit_lock(self)

    def _at_fork_reinit(self):
        self.lock._at_fork_reinit()

    def acquire(self):
        """
        Acquire the I/O thread lock.
        """
        if self.lock:
            self.lock.acquire()

    def release(self):
        """
        Release the I/O thread lock.
        """
        if self.lock:
            self.lock.release()

    def setLevel(self, level):
        """
        Set the logging level of this handler.  level must be an int or a str.
        """
        self.level = _checkLevel(level)

    def format(self, record):
        """
        Format the specified record.

        If a formatter is set, use it. Otherwise, use the default formatter
        for the module.
        """
        if self.formatter:
            fmt = self.formatter
        else:
            fmt = _defaultFormatter
        return fmt.format(record)

    def emit(self, record):
        """
        Do whatever it takes to actually log the specified logging record.

        This version is intended to be implemented by subclasses and so
        raises a NotImplementedError.
        """
        raise NotImplementedError('emit must be implemented '
                                  'by Handler subclasses')

    def handle(self, record):
        """
        Conditionally emit the specified logging record.

        Emission depends on filters which may have been added to the handler.
        Wrap the actual emission of the record with acquisition/release of
        the I/O thread lock. Returns whether the filter passed the record for
        emission.
        """
        rv = self.filter(record)
        if rv:
            self.acquire()
            try:
                self.emit(record)
            finally:
                self.release()
        return rv

    def setFormatter(self, fmt):
        """
        Set the formatter for this handler.
        """
        self.formatter = fmt

    def flush(self):
        """
        Ensure all logging output has been flushed.

        This version does nothing and is intended to be implemented by
        subclasses.
        """
        pass

    def close(self):
        """
        Tidy up any resources used by the handler.

        This version removes the handler from an internal map of handlers,
        _handlers, which is used for handler lookup by name. Subclasses
        should ensure that this gets called from overridden close()
        methods.
        """
        #get the module data lock, as we're updating a shared structure.
        _acquireLock()
        try:    #unlikely to raise an exception, but you never know...
            self._closed = True
            if self._name and self._name in _handlers:
                del _handlers[self._name]
        finally:
            _releaseLock()

    def handleError(self, record):
        """
        Handle errors which occur during an emit() call.

        This method should be called from handlers when an exception is
        encountered during an emit() call. If raiseExceptions is false,
        exceptions get silently ignored. This is what is mostly wanted
        for a logging system - most users will not care about errors in
        the logging system, they are more interested in application errors.
        You could, however, replace this with a custom handler if you wish.
        The record which was being processed is passed in to this method.
        """
        if raiseExceptions and sys.stderr:  # see issue 13807
            t, v, tb = sys.exc_info()
            try:
                sys.stderr.write('--- Logging error ---\n')
                traceback.print_exception(t, v, tb, None, sys.stderr)
                sys.stderr.write('Call stack:\n')
                # Walk the stack frame up until we're out of logging,
                # so as to print the calling context.
                frame = tb.tb_frame
                while (frame and os.path.dirname(frame.f_code.co_filename) ==
                       __path__[0]):
                    frame = frame.f_back
                if frame:
                    traceback.print_stack(frame, file=sys.stderr)
                else:
                    # couldn't find the right stack frame, for some reason
                    sys.stderr.write('Logged from file %s, line %s\n' % (
                                     record.filename, record.lineno))
                # Issue 18671: output logging message and arguments
                try:
                    sys.stderr.write('Message: %r\n'
                                     'Arguments: %s\n' % (record.msg,
                                                          record.args))
                except RecursionError:  # See issue 36272
                    raise
                except Exception:
                    sys.stderr.write('Unable to print the message and arguments'
                                     ' - possible formatting error.\nUse the'
                                     ' traceback above to help find the error.\n'
                                    )
            except OSError: #pragma: no cover
                pass    # see issue 5971
            finally:
                del t, v, tb

    def __repr__(self):
        level = getLevelName(self.level)
        return '<%s (%s)>' % (self.__class__.__name__, level)

class StreamHandler(Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a stream. Note that this class does not close the stream, as
    sys.stdout or sys.stderr may be used.
    """

    terminator = '\n'

    def __init__(self, stream=None):
        """
        Initialize the handler.

        If stream is not specified, sys.stderr is used.
        """
        Handler.__init__(self)
        if stream is None:
            stream = sys.stderr
        self.stream = stream

    def flush(self):
        """
        Flushes the stream.
        """
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record):
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)

    def setStream(self, stream):
        """
        Sets the StreamHandler's stream to the specified value,
        if it is different.

        Returns the old stream, if the stream was changed, or None
        if it wasn't.
        """
        if stream is self.stream:
            result = None
        else:
            result = self.stream
            self.acquire()
            try:
                self.flush()
                self.stream = stream
            finally:
                self.release()
        return result

    def __repr__(self):
        level = getLevelName(self.level)
        name = getattr(self.stream, 'name', '')
        #  bpo-36015: name can be an int
        name = str(name)
        if name:
            name += ' '
        return '<%s %s(%s)>' % (self.__class__.__name__, name, level)


class FileHandler(StreamHandler):
    """
    A handler class which writes formatted logging records to disk files.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False, errors=None):
        """
        Open the specified file and use it as the stream for logging.
        """
        # Issue #27493: add support for Path objects to be passed in
        filename = os.fspath(filename)
        #keep the absolute path, otherwise derived classes which use this
        #may come a cropper when the current directory changes
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        if "b" not in mode:
            self.encoding = io.text_encoding(encoding)
        self.errors = errors
        self.delay = delay
        # bpo-26789: FileHandler keeps a reference to the builtin open()
        # function to be able to open or reopen the file during Python
        # finalization.
        self._builtin_open = open
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            Handler.__init__(self)
            self.stream = None
        else:
            StreamHandler.__init__(self, self._open())

    def close(self):
        """
        Closes the stream.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                # Also see Issue #42378: we also rely on
                # self._closed being set to True there
                StreamHandler.close(self)
        finally:
            self.release()

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        open_func = self._builtin_open
        return open_func(self.baseFilename, self.mode,
                         encoding=self.encoding, errors=self.errors)

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.

        If stream is not open, current mode is 'w' and `_closed=True`, record
        will not be emitted (see Issue #42378).
        """
        if self.stream is None:
            if self.mode != 'w' or not self._closed:
                self.stream = self._open()
        if self.stream:
            StreamHandler.emit(self, record)

    def __repr__(self):
        level = getLevelName(self.level)
        return '<%s %s (%s)>' % (self.__class__.__name__, self.baseFilename, level)


class _StderrHandler(StreamHandler):
    """
    This class is like a StreamHandler using sys.stderr, but always uses
    whatever sys.stderr is currently set to rather than the value of
    sys.stderr at handler construction time.
    """
    def __init__(self, level=NOTSET):
        """
        Initialize the handler.
        """
        Handler.__init__(self, level)

    @property
    def stream(self):
        return sys.stderr


_defaultLastResort = _StderrHandler(WARNING)
lastResort = _defaultLastResort

#---------------------------------------------------------------------------
#   Manager classes and functions
#---------------------------------------------------------------------------

class PlaceHolder(object):
    """
    PlaceHolder instances are used in the Manager logger hierarchy to take
    the place of nodes for which no loggers have been defined. This class is
    intended for internal use only and not as part of the public API.
    """
    def __init__(self, alogger):
        """
        Initialize with the specified logger being a child of this placeholder.
        """
        self.loggerMap = { alogger : None }

    def append(self, alogger):
        """
        Add the specified logger as a child of this placeholder.
        """
        if alogger not in self.loggerMap:
            self.loggerMap[alogger] = None

#
#   Determine which class to use when instantiating loggers.
#

def setLoggerClass(klass):
    """
    Set the class to be used when instantiating a logger. The class should
    define __init__() such that only a name argument is required, and the
    __init__() should call Logger.__init__()
    """
    if klass != Logger:
        if not issubclass(klass, Logger):
            raise TypeError("logger not derived from logging.Logger: "
                            + klass.__name__)
    global _loggerClass
    _loggerClass = klass

def getLoggerClass():
    """
    Return the class to be used when instantiating a logger.
    """
    return _loggerClass

class Manager(object):
    """
    There is [under normal circumstances] just one Manager instance, which
    holds the hierarchy of loggers.
    """
    def __init__(self, rootnode):
        """
        Initialize the manager with the root node of the logger hierarchy.
        """
        self.root = rootnode
        self.disable = 0
        self.emittedNoHandlerWarning = False
        self.loggerDict = {}
        self.loggerClass = None
        self.logRecordFactory = None

    @property
    def disable(self):
        return self._disable

    @disable.setter
    def disable(self, value):
        self._disable = _checkLevel(value)

    def getLogger(self, name):
        """
        Get a logger with the specified name (channel name), creating it
        if it doesn't yet exist. This name is a dot-separated hierarchical
        name, such as "a", "a.b", "a.b.c" or similar.

        If a PlaceHolder existed for the specified name [i.e. the logger
        didn't exist but a child of it did], replace it with the created
        logger and fix up the parent/child references which pointed to the
        placeholder to now point to the logger.
        """
        rv = None
        if not isinstance(name, str):
            raise TypeError('A logger name must be a string')
        _acquireLock()
        try:
            if name in self.loggerDict:
                rv = self.loggerDict[name]
                if isinstance(rv, PlaceHolder):
                    ph = rv
                    rv = (self.loggerClass or _loggerClass)(name)
                    rv.manager = self
                    self.loggerDict[name] = rv
                    self._fixupChildren(ph, rv)
                    self._fixupParents(rv)
            else:
                rv = (self.loggerClass or _loggerClass)(name)
                rv.manager = self
                self.loggerDict[name] = rv
                self._fixupParents(rv)
        finally:
            _releaseLock()
        return rv

    def setLoggerClass(self, klass):
        """
        Set the class to be used when instantiating a logger with this Manager.
        """
        if klass != Logger:
            if not issubclass(klass, Logger):
                raise TypeError("logger not derived from logging.Logger: "
                                + klass.__name__)
        self.loggerClass = klass

    def setLogRecordFactory(self, factory):
        """
        Set the factory to be used when instantiating a log record with this
        Manager.
        """
        self.logRecordFactory = factory

    def _fixupParents(self, alogger):
        """
        Ensure that there are either loggers or placeholders all the way
        from the specified logger to the root of the logger hierarchy.
        """
        name = alogger.name
        i = name.rfind(".")
        rv = None
        while (i > 0) and not rv:
            substr = name[:i]
            if substr not in self.loggerDict:
                self.loggerDict[substr] = PlaceHolder(alogger)
            else:
                obj = self.loggerDict[substr]
                if isinstance(obj, Logger):
                    rv = obj
                else:
                    assert isinstance(obj, PlaceHolder)
                    obj.append(alogger)
            i = name.rfind(".", 0, i - 1)
        if not rv:
            rv = self.root
        alogger.parent = rv

    def _fixupChildren(self, ph, alogger):
        """
        Ensure that children of the placeholder ph are connected to the
        specified logger.
        """
        name = alogger.name
        namelen = len(name)
        for c in ph.loggerMap.keys():
            #The if means ... if not c.parent.name.startswith(nm)
            if c.parent.name[:namelen] != name:
                alogger.parent = c.parent
                c.parent = alogger

    def _clear_cache(self):
        """
        Clear the cache for all loggers in loggerDict
        Called when level changes are made
        """

        _acquireLock()
        for logger in self.loggerDict.values():
            if isinstance(logger, Logger):
                logger._cache.clear()
        self.root._cache.clear()
        _releaseLock()

#---------------------------------------------------------------------------
#   Logger classes and functions
#---------------------------------------------------------------------------

class Logger(Filterer):
    """
    Instances of the Logger class represent a single logging channel. A
    "logging channel" indicates an area of an application. Exactly how an
    "area" is defined is up to the application developer. Since an
    application can have any number of areas, logging channels are identified
    by a unique string. Application areas can be nested (e.g. an area
    of "input processing" might include sub-areas "read CSV files", "read
    XLS files" and "read Gnumeric files"). To cater for this natural nesting,
    channel names are organized into a namespace hierarchy where levels are
    separated by periods, much like the Java or Python package namespace. So
    in the instance given above, channel names might be "input" for the upper
    level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
    There is no arbitrary limit to the depth of nesting.
    """
    def __init__(self, name, level=NOTSET):
        """
        Initialize the logger with a name and an optional level.
        """
        Filterer.__init__(self)
        self.name = name
        self.level = _checkLevel(level)
        self.parent = None
        self.propagate = True
        self.handlers = []
        self.disabled = False
        self._cache = {}

    def setLevel(self, level):
        """
        Set the logging level of this logger.  level must be an int or a str.
        """
        self.level = _checkLevel(level)
        self.manager._clear_cache()

    def debug(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'DEBUG'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.debug("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "interesting problem", exc_info=1)
        """
        if self.isEnabledFor(INFO):
            self._log(INFO, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(WARNING):
            self._log(WARNING, msg, args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        warnings.warn("The 'warn' method is deprecated, "
            "use 'warning' instead", DeprecationWarning, 2)
        self.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'ERROR'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.error("Houston, we have a %s", "major problem", exc_info=1)
        """
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Convenience method for logging an ERROR with exception information.
        """
        self.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'CRITICAL'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.critical("Houston, we have a %s", "major disaster", exc_info=1)
        """
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        """
        Don't use this method, use critical() instead.
        """
        self.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """
        Log 'msg % args' with the integer severity 'level'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.log(level, "We have a %s", "mysterious problem", exc_info=1)
        """
        if not isinstance(level, int):
            if raiseExceptions:
                raise TypeError("level must be an integer")
            else:
                return
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)

    def findCaller(self, stack_info=False, stacklevel=1):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = currentframe()
        #On some versions of IronPython, currentframe() returns None if
        #IronPython isn't run with -X:Frames.
        if f is not None:
            f = f.f_back
        orig_f = f
        while f and stacklevel > 1:
            f = f.f_back
            stacklevel -= 1
        if not f:
            f = orig_f
        rv = "(unknown file)", 0, "(unknown function)", None
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            sinfo = None
            if stack_info:
                sio = io.StringIO()
                sio.write('Stack (most recent call last):\n')
                traceback.print_stack(f, file=sio)
                sinfo = sio.getvalue()
                if sinfo[-1] == '\n':
                    sinfo = sinfo[:-1]
                sio.close()
            rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
            break
        return rv

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        """
        rv = _logRecordFactory(name, level, fn, lno, msg, args, exc_info, func,
                             sinfo)
        if extra is not None:
            for key in extra:
                if (key in ["message", "asctime"]) or (key in rv.__dict__):
                    raise KeyError("Attempt to overwrite %r in LogRecord" % key)
                rv.__dict__[key] = extra[key]
        return rv

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False,
             stacklevel=1):
        """
        Low-level logging routine which creates a LogRecord and then calls
        all the handlers of this logger to handle the record.
        """
        sinfo = None
        if _srcfile:
            #IronPython doesn't track Python frames, so findCaller raises an
            #exception on some versions of IronPython. We trap it here so that
            #IronPython can use logging.
            try:
                fn, lno, func, sinfo = self.findCaller(stack_info, stacklevel)
            except ValueError: # pragma: no cover
                fn, lno, func = "(unknown file)", 0, "(unknown function)"
        else: # pragma: no cover
            fn, lno, func = "(unknown file)", 0, "(unknown function)"
        if exc_info:
            if isinstance(exc_info, BaseException):
                exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
            elif not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
        record = self.makeRecord(self.name, level, fn, lno, msg, args,
                                 exc_info, func, extra, sinfo)
        self.handle(record)

    def handle(self, record):
        """
        Call the handlers for the specified record.

        This method is used for unpickled records received from a socket, as
        well as those created locally. Logger-level filtering is applied.
        """
        if (not self.disabled) and self.filter(record):
            self.callHandlers(record)

    def addHandler(self, hdlr):
        """
        Add the specified handler to this logger.
        """
        _acquireLock()
        try:
            if not (hdlr in self.handlers):
                self.handlers.append(hdlr)
        finally:
            _releaseLock()

    def removeHandler(self, hdlr):
        """
        Remove the specified handler from this logger.
        """
        _acquireLock()
        try:
            if hdlr in self.handlers:
                self.handlers.remove(hdlr)
        finally:
            _releaseLock()

    def hasHandlers(self):
        """
        See if this logger has any handlers configured.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. Return True if a handler was found, else False.
        Stop searching up the hierarchy whenever a logger with the "propagate"
        attribute set to zero is found - that will be the last logger which
        is checked for the existence of handlers.
        """
        c = self
        rv = False
        while c:
            if c.handlers:
                rv = True
                break
            if not c.propagate:
                break
            else:
                c = c.parent
        return rv

    def callHandlers(self, record):
        """
        Pass a record to all relevant handlers.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. If no handler was found, output a one-off error
        message to sys.stderr. Stop searching up the hierarchy whenever a
        logger with the "propagate" attribute set to zero is found - that
        will be the last logger whose handlers are called.
        """
        c = self
        found = 0
        while c:
            for hdlr in c.handlers:
                found = found + 1
                if record.levelno >= hdlr.level:
                    hdlr.handle(record)
            if not c.propagate:
                c = None    #break out
            else:
                c = c.parent
        if (found == 0):
            if lastResort:
                if record.levelno >= lastResort.level:
                    lastResort.handle(record)
            elif raiseExceptions and not self.manager.emittedNoHandlerWarning:
                sys.stderr.write("No handlers could be found for logger"
                                 " \"%s\"\n" % self.name)
                self.manager.emittedNoHandlerWarning = True

    def getEffectiveLevel(self):
        """
        Get the effective level for this logger.

        Loop through this logger and its parents in the logger hierarchy,
        looking for a non-zero logging level. Return the first one found.
        """
        logger = self
        while logger:
            if logger.level:
                return logger.level
            logger = logger.parent
        return NOTSET

    def isEnabledFor(self, level):
        """
        Is this logger enabled for level 'level'?
        """
        if self.disabled:
            return False

        try:
            return self._cache[level]
        except KeyError:
            _acquireLock()
            try:
                if self.manager.disable >= level:
                    is_enabled = self._cache[level] = False
                else:
                    is_enabled = self._cache[level] = (
                        level >= self.getEffectiveLevel()
                    )
            finally:
                _releaseLock()
            return is_enabled

    def getChild(self, suffix):
        """
        Get a logger which is a descendant to this one.

        This is a convenience method, such that

        logging.getLogger('abc').getChild('def.ghi')

        is the same as

        logging.getLogger('abc.def.ghi')

        It's useful, for example, when the parent logger is named using
        __name__ rather than a literal string.
        """
        if self.root is not self:
            suffix = '.'.join((self.name, suffix))
        return self.manager.getLogger(suffix)

    def __repr__(self):
        level = getLevelName(self.getEffectiveLevel())
        return '<%s %s (%s)>' % (self.__class__.__name__, self.name, level)

    def __reduce__(self):
        # In general, only the root logger will not be accessible via its name.
        # However, the root logger's class has its own __reduce__ method.
        if getLogger(self.name) is not self:
            import pickle
            raise pickle.PicklingError('logger cannot be pickled')
        return getLogger, (self.name,)


class RootLogger(Logger):
    """
    A root logger is not that different to any other logger, except that
    it must have a logging level and there is only one instance of it in
    the hierarchy.
    """
    def __init__(self, level):
        """
        Initialize the logger with the name "root".
        """
        Logger.__init__(self, "root", level)

    def __reduce__(self):
        return getLogger, ()

_loggerClass = Logger

class LoggerAdapter(object):
    """
    An adapter for loggers which makes it easier to specify contextual
    information in logging output.
    """

    def __init__(self, logger, extra=None):
        """
        Initialize the adapter with a logger and a dict-like object which
        provides contextual information. This constructor signature allows
        easy stacking of LoggerAdapters, if so desired.

        You can effectively pass keyword arguments as shown in the
        following example:

        adapter = LoggerAdapter(someLogger, dict(p1=v1, p2="v2"))
        """
        self.logger = logger
        self.extra = extra

    def process(self, msg, kwargs):
        """
        Process the logging message and keyword arguments passed in to
        a logging call to insert contextual information. You can either
        manipulate the message itself, the keyword args or both. Return
        the message and kwargs modified (or not) to suit your needs.

        Normally, you'll only need to override this one method in a
        LoggerAdapter subclass for your specific needs.
        """
        kwargs["extra"] = self.extra
        return msg, kwargs

    #
    # Boilerplate convenience methods
    #
    def debug(self, msg, *args, **kwargs):
        """
        Delegate a debug call to the underlying logger.
        """
        self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Delegate an info call to the underlying logger.
        """
        self.log(INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        Delegate a warning call to the underlying logger.
        """
        self.log(WARNING, msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        warnings.warn("The 'warn' method is deprecated, "
            "use 'warning' instead", DeprecationWarning, 2)
        self.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        Delegate an error call to the underlying logger.
        """
        self.log(ERROR, msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """
        Delegate an exception call to the underlying logger.
        """
        self.log(ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """
        Delegate a critical call to the underlying logger.
        """
        self.log(CRITICAL, msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """
        Delegate a log call to the underlying logger, after adding
        contextual information from this adapter instance.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)

    def isEnabledFor(self, level):
        """
        Is this logger enabled for level 'level'?
        """
        return self.logger.isEnabledFor(level)

    def setLevel(self, level):
        """
        Set the specified level on the underlying logger.
        """
        self.logger.setLevel(level)

    def getEffectiveLevel(self):
        """
        Get the effective level for the underlying logger.
        """
        return self.logger.getEffectiveLevel()

    def hasHandlers(self):
        """
        See if the underlying logger has any handlers.
        """
        return self.logger.hasHandlers()

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        """
        Low-level log implementation, proxied to allow nested logger adapters.
        """
        return self.logger._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
        )

    @property
    def manager(self):
        return self.logger.manager

    @manager.setter
    def manager(self, value):
        self.logger.manager = value

    @property
    def name(self):
        return self.logger.name

    def __repr__(self):
        logger = self.logger
        level = getLevelName(logger.getEffectiveLevel())
        return '<%s %s (%s)>' % (self.__class__.__name__, logger.name, level)

root = RootLogger(WARNING)
Logger.root = root
Logger.manager = Manager(Logger.root)

#---------------------------------------------------------------------------
# Configuration classes and functions
#---------------------------------------------------------------------------

def basicConfig(**kwargs):
    """
    Do basic configuration for the logging system.

    This function does nothing if the root logger already has handlers
    configured, unless the keyword argument *force* is set to ``True``.
    It is a convenience method intended for use by simple scripts
    to do one-shot configuration of the logging package.

    The default behaviour is to create a StreamHandler which writes to
    sys.stderr, set a formatter using the BASIC_FORMAT format string, and
    add the handler to the root logger.

    A number of optional keyword arguments may be specified, which can alter
    the default behaviour.

    filename  Specifies that a FileHandler be created, using the specified
              filename, rather than a StreamHandler.
    filemode  Specifies the mode to open the file, if filename is specified
              (if filemode is unspecified, it defaults to 'a').
    format    Use the specified format string for the handler.
    datefmt   Use the specified date/time format.
    style     If a format string is specified, use this to specify the
              type of format string (possible values '%', '{', '$', for
              %-formatting, :meth:`str.format` and :class:`string.Template`
              - defaults to '%').
    level     Set the root logger level to the specified level.
    stream    Use the specified stream to initialize the StreamHandler. Note
              that this argument is incompatible with 'filename' - if both
              are present, 'stream' is ignored.
    handlers  If specified, this should be an iterable of already created
              handlers, which will be added to the root handler. Any handler
              in the list which does not have a formatter assigned will be
              assigned the formatter created in this function.
    force     If this keyword  is specified as true, any existing handlers
              attached to the root logger are removed and closed, before
              carrying out the configuration as specified by the other
              arguments.
    encoding  If specified together with a filename, this encoding is passed to
              the created FileHandler, causing it to be used when the file is
              opened.
    errors    If specified together with a filename, this value is passed to the
              created FileHandler, causing it to be used when the file is
              opened in text mode. If not specified, the default value is
              `backslashreplace`.

    Note that you could specify a stream created using open(filename, mode)
    rather than passing the filename and mode in. However, it should be
    remembered that StreamHandler does not close its stream (since it may be
    using sys.stdout or sys.stderr), whereas FileHandler closes its stream
    when the handler is closed.

    .. versionchanged:: 3.2
       Added the ``style`` parameter.

    .. versionchanged:: 3.3
       Added the ``handlers`` parameter. A ``ValueError`` is now thrown for
       incompatible arguments (e.g. ``handlers`` specified together with
       ``filename``/``filemode``, or ``filename``/``filemode`` specified
       together with ``stream``, or ``handlers`` specified together with
       ``stream``.

    .. versionchanged:: 3.8
       Added the ``force`` parameter.

    .. versionchanged:: 3.9
       Added the ``encoding`` and ``errors`` parameters.
    """
    # Add thread safety in case someone mistakenly calls
    # basicConfig() from multiple threads
    _acquireLock()
    try:
        force = kwargs.pop('force', False)
        encoding = kwargs.pop('encoding', None)
        errors = kwargs.pop('errors', 'backslashreplace')
        if force:
            for h in root.handlers[:]:
                root.removeHandler(h)
                h.close()
        if len(root.handlers) == 0:
            handlers = kwargs.pop("handlers", None)
            if handlers is None:
                if "stream" in kwargs and "filename" in kwargs:
                    raise ValueError("'stream' and 'filename' should not be "
                                     "specified together")
            else:
                if "stream" in kwargs or "filename" in kwargs:
                    raise ValueError("'stream' or 'filename' should not be "
                                     "specified together with 'handlers'")
            if handlers is None:
                filename = kwargs.pop("filename", None)
                mode = kwargs.pop("filemode", 'a')
                if filename:
                    if 'b' in mode:
                        errors = None
                    else:
                        encoding = io.text_encoding(encoding)
                    h = FileHandler(filename, mode,
                                    encoding=encoding, errors=errors)
                else:
                    stream = kwargs.pop("stream", None)
                    h = StreamHandler(stream)
                handlers = [h]
            dfs = kwargs.pop("datefmt", None)
            style = kwargs.pop("style", '%')
            if style not in _STYLES:
                raise ValueError('Style must be one of: %s' % ','.join(
                                 _STYLES.keys()))
            fs = kwargs.pop("format", _STYLES[style][1])
            fmt = Formatter(fs, dfs, style)
            for h in handlers:
                if h.formatter is None:
                    h.setFormatter(fmt)
                root.addHandler(h)
            level = kwargs.pop("level", None)
            if level is not None:
                root.setLevel(level)
            if kwargs:
                keys = ', '.join(kwargs.keys())
                raise ValueError('Unrecognised argument(s): %s' % keys)
    finally:
        _releaseLock()

#---------------------------------------------------------------------------
# Utility functions at module level.
# Basically delegate everything to the root logger.
#---------------------------------------------------------------------------

def getLogger(name=None):
    """
    Return a logger with the specified name, creating it if necessary.

    If no name is specified, return the root logger.
    """
    if not name or isinstance(name, str) and name == root.name:
        return root
    return Logger.manager.getLogger(name)

def critical(msg, *args, **kwargs):
    """
    Log a message with severity 'CRITICAL' on the root logger. If the logger
    has no handlers, call basicConfig() to add a console handler with a
    pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.critical(msg, *args, **kwargs)

def fatal(msg, *args, **kwargs):
    """
    Don't use this function, use critical() instead.
    """
    critical(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    """
    Log a message with severity 'ERROR' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.error(msg, *args, **kwargs)

def exception(msg, *args, exc_info=True, **kwargs):
    """
    Log a message with severity 'ERROR' on the root logger, with exception
    information. If the logger has no handlers, basicConfig() is called to add
    a console handler with a pre-defined format.
    """
    error(msg, *args, exc_info=exc_info, **kwargs)

def warning(msg, *args, **kwargs):
    """
    Log a message with severity 'WARNING' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.warning(msg, *args, **kwargs)

def warn(msg, *args, **kwargs):
    warnings.warn("The 'warn' function is deprecated, "
        "use 'warning' instead", DeprecationWarning, 2)
    warning(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.info(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    """
    Log a message with severity 'DEBUG' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.debug(msg, *args, **kwargs)

def log(level, msg, *args, **kwargs):
    """
    Log 'msg % args' with the integer severity 'level' on the root logger. If
    the logger has no handlers, call basicConfig() to add a console handler
    with a pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    root.log(level, msg, *args, **kwargs)

def disable(level=CRITICAL):
    """
    Disable all logging calls of severity 'level' and below.
    """
    root.manager.disable = level
    root.manager._clear_cache()

def shutdown(handlerList=_handlerList):
    """
    Perform any cleanup actions in the logging system (e.g. flushing
    buffers).

    Should be called at application exit.
    """
    for wr in reversed(handlerList[:]):
        #errors might occur, for example, if files are locked
        #we just ignore them if raiseExceptions is not set
        try:
            h = wr()
            if h:
                try:
                    h.acquire()
                    h.flush()
                    h.close()
                except (OSError, ValueError):
                    # Ignore errors which might be caused
                    # because handlers have been closed but
                    # references to them are still around at
                    # application exit.
                    pass
                finally:
                    h.release()
        except: # ignore everything, as we're shutting down
            if raiseExceptions:
                raise
            #else, swallow

#Let's try and shutdown automatically on application exit...
import atexit
atexit.register(shutdown)

# Null handler

class NullHandler(Handler):
    """
    This handler does nothing. It's intended to be used to avoid the
    "No handlers could be found for logger XXX" one-off warning. This is
    important for library code, which may contain code to log events. If a user
    of the library does not configure logging, the one-off warning might be
    produced; to avoid this, the library developer simply needs to instantiate
    a NullHandler and add it to the top-level logger of the library module or
    package.
    """
    def handle(self, record):
        """Stub."""

    def emit(self, record):
        """Stub."""

    def createLock(self):
        self.lock = None

    def _at_fork_reinit(self):
        pass

# Warnings integration

_warnings_showwarning = None

def _showwarning(message, category, filename, lineno, file=None, line=None):
    """
    Implementation of showwarnings which redirects to logging, which will first
    check to see if the file parameter is None. If a file is specified, it will
    delegate to the original warnings implementation of showwarning. Otherwise,
    it will call warnings.formatwarning and will log the resulting string to a
    warnings logger named "py.warnings" with level logging.WARNING.
    """
    if file is not None:
        if _warnings_showwarning is not None:
            _warnings_showwarning(message, category, filename, lineno, file, line)
    else:
        s = warnings.formatwarning(message, category, filename, lineno, line)
        logger = getLogger("py.warnings")
        if not logger.handlers:
            logger.addHandler(NullHandler())
        logger.warning("%s", s)

def captureWarnings(capture):
    """
    If capture is true, redirect all warnings to the logging package.
    If capture is False, ensure that warnings are not redirected to logging
    but to their original destinations.
    """
    global _warnings_showwarning
    if capture:
        if _warnings_showwarning is None:
            _warnings_showwarning = warnings.showwarning
            warnings.showwarning = _showwarning
    else:
        if _warnings_showwarning is not None:
            warnings.showwarning = _warnings_showwarning
            _warnings_showwarning = None
<|endoftext|># Copyright 2001-2021 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
Additional handlers for the logging package for Python. The core package is
based on PEP 282 and comments thereto in comp.lang.python.

Copyright (C) 2001-2021 Vinay Sajip. All Rights Reserved.

To use, simply 'import logging.handlers' and log away!
"""

import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy

#
# Some constants...
#

DEFAULT_TCP_LOGGING_PORT    = 9020
DEFAULT_UDP_LOGGING_PORT    = 9021
DEFAULT_HTTP_LOGGING_PORT   = 9022
DEFAULT_SOAP_LOGGING_PORT   = 9023
SYSLOG_UDP_PORT             = 514
SYSLOG_TCP_PORT             = 514

_MIDNIGHT = 24 * 60 * 60  # number of seconds in a day

class BaseRotatingHandler(logging.FileHandler):
    """
    Base class for handlers that rotate log files at a certain point.
    Not meant to be instantiated directly.  Instead, use RotatingFileHandler
    or TimedRotatingFileHandler.
    """
    namer = None
    rotator = None

    def __init__(self, filename, mode, encoding=None, delay=False, errors=None):
        """
        Use the specified filename for streamed logging
        """
        logging.FileHandler.__init__(self, filename, mode=mode,
                                     encoding=encoding, delay=delay,
                                     errors=errors)
        self.mode = mode
        self.encoding = encoding
        self.errors = errors

    def emit(self, record):
        """
        Emit a record.

        Output the record to the file, catering for rollover as described
        in doRollover().
        """
        try:
            if self.shouldRollover(record):
                self.doRollover()
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)

    def rotation_filename(self, default_name):
        """
        Modify the filename of a log file when rotating.

        This is provided so that a custom filename can be provided.

        The default implementation calls the 'namer' attribute of the
        handler, if it's callable, passing the default name to
        it. If the attribute isn't callable (the default is None), the name
        is returned unchanged.

        :param default_name: The default name for the log file.
        """
        if not callable(self.namer):
            result = default_name
        else:
            result = self.namer(default_name)
        return result

    def rotate(self, source, dest):
        """
        When rotating, rotate the current log.

        The default implementation calls the 'rotator' attribute of the
        handler, if it's callable, passing the source and dest arguments to
        it. If the attribute isn't callable (the default is None), the source
        is simply renamed to the destination.

        :param source: The source filename. This is normally the base
                       filename, e.g. 'test.log'
        :param dest:   The destination filename. This is normally
                       what the source is rotated to, e.g. 'test.log.1'.
        """
        if not callable(self.rotator):
            # Issue 18940: A file may not have been created if delay is True.
            if os.path.exists(source):
                os.rename(source, dest)
        else:
            self.rotator(source, dest)

class RotatingFileHandler(BaseRotatingHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size.
    """
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0,
                 encoding=None, delay=False, errors=None):
        """
        Open the specified file and use it as the stream for logging.

        By default, the file grows indefinitely. You can specify particular
        values of maxBytes and backupCount to allow the file to rollover at
        a predetermined size.

        Rollover occurs whenever the current log file is nearly maxBytes in
        length. If backupCount is >= 1, the system will successively create
        new files with the same pathname as the base file, but with extensions
        ".1", ".2" etc. appended to it. For example, with a backupCount of 5
        and a base file name of "app.log", you would get "app.log",
        "app.log.1", "app.log.2", ... through to "app.log.5". The file being
        written to is always "app.log" - when it gets filled up, it is closed
        and renamed to "app.log.1", and if files "app.log.1", "app.log.2" etc.
        exist, then they are renamed to "app.log.2", "app.log.3" etc.
        respectively.

        If maxBytes is zero, rollover never occurs.
        """
        # If rotation/rollover is wanted, it doesn't make sense to use another
        # mode. If for example 'w' were specified, then if there were multiple
        # runs of the calling application, the logs from previous runs would be
        # lost if the 'w' is respected, because the log file would be truncated
        # on each run.
        if maxBytes > 0:
            mode = 'a'
        if "b" not in mode:
            encoding = io.text_encoding(encoding)
        BaseRotatingHandler.__init__(self, filename, mode, encoding=encoding,
                                     delay=delay, errors=errors)
        self.maxBytes = maxBytes
        self.backupCount = backupCount

    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d" % (self.baseFilename,
                                                        i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            self.rotate(self.baseFilename, dfn)
        if not self.delay:
            self.stream = self._open()

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        Basically, see if the supplied record would cause the file to exceed
        the size limit we have.
        """
        # See bpo-45401: Never rollover anything other than regular files
        if os.path.exists(self.baseFilename) and not os.path.isfile(self.baseFilename):
            return False
        if self.stream is None:                 # delay was set...
            self.stream = self._open()
        if self.maxBytes > 0:                   # are we rolling over?
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)  #due to non-posix-compliant Windows feature
            if self.stream.tell() + len(msg) >= self.maxBytes:
                return True
        return False

class TimedRotatingFileHandler(BaseRotatingHandler):
    """
    Handler for logging to a file, rotating the log file at certain timed
    intervals.

    If backupCount is > 0, when rollover is done, no more than backupCount
    files are kept - the oldest ones are deleted.
    """
    def __init__(self, filename, when='h', interval=1, backupCount=0,
                 encoding=None, delay=False, utc=False, atTime=None,
                 errors=None):
        encoding = io.text_encoding(encoding)
        BaseRotatingHandler.__init__(self, filename, 'a', encoding=encoding,
                                     delay=delay, errors=errors)
        self.when = when.upper()
        self.backupCount = backupCount
        self.utc = utc
        self.atTime = atTime
        # Calculate the real rollover interval, which is just the number of
        # seconds between rollovers.  Also set the filename suffix used when
        # a rollover occurs.  Current 'when' events supported:
        # S - Seconds
        # M - Minutes
        # H - Hours
        # D - Days
        # midnight - roll over at midnight
        # W{0-6} - roll over on a certain day; 0 - Monday
        #
        # Case of the 'when' specifier is not important; lower or upper case
        # will work.
        if self.when == 'S':
            self.interval = 1 # one second
            self.suffix = "%Y-%m-%d_%H-%M-%S"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'M':
            self.interval = 60 # one minute
            self.suffix = "%Y-%m-%d_%H-%M"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'H':
            self.interval = 60 * 60 # one hour
            self.suffix = "%Y-%m-%d_%H"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}(\.\w+)?$"
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            self.interval = 60 * 60 * 24 # one day
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when.startswith('W'):
            self.interval = 60 * 60 * 24 * 7 # one week
            if len(self.when) != 2:
                raise ValueError("You must specify a day for weekly rollover from 0 to 6 (0 is Monday): %s" % self.when)
            if self.when[1] < '0' or self.when[1] > '6':
                raise ValueError("Invalid day specified for weekly rollover: %s" % self.when)
            self.dayOfWeek = int(self.when[1])
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        else:
            raise ValueError("Invalid rollover interval specified: %s" % self.when)

        self.extMatch = re.compile(self.extMatch, re.ASCII)
        self.interval = self.interval * interval # multiply by units requested
        # The following line added because the filename passed in could be a
        # path object (see Issue #27493), but self.baseFilename will be a string
        filename = self.baseFilename
        if os.path.exists(filename):
            t = os.stat(filename)[ST_MTIME]
        else:
            t = int(time.time())
        self.rolloverAt = self.computeRollover(t)

    def computeRollover(self, currentTime):
        """
        Work out the rollover time based on the specified time.
        """
        result = currentTime + self.interval
        # If we are rolling over at midnight or weekly, then the interval is already known.
        # What we need to figure out is WHEN the next interval is.  In other words,
        # if you are rolling over at midnight, then your base interval is 1 day,
        # but you want to start that one day clock at midnight, not now.  So, we
        # have to fudge the rolloverAt value in order to trigger the first rollover
        # at the right time.  After that, the regular interval will take care of
        # the rest.  Note that this code doesn't care about leap seconds. :)
        if self.when == 'MIDNIGHT' or self.when.startswith('W'):
            # This could be done with less code, but I wanted it to be clear
            if self.utc:
                t = time.gmtime(currentTime)
            else:
                t = time.localtime(currentTime)
            currentHour = t[3]
            currentMinute = t[4]
            currentSecond = t[5]
            currentDay = t[6]
            # r is the number of seconds left between now and the next rotation
            if self.atTime is None:
                rotate_ts = _MIDNIGHT
            else:
                rotate_ts = ((self.atTime.hour * 60 + self.atTime.minute)*60 +
                    self.atTime.second)

            r = rotate_ts - ((currentHour * 60 + currentMinute) * 60 +
                currentSecond)
            if r < 0:
                # Rotate time is before the current time (for example when
                # self.rotateAt is 13:45 and it now 14:15), rotation is
                # tomorrow.
                r += _MIDNIGHT
                currentDay = (currentDay + 1) % 7
            result = currentTime + r
            # If we are rolling over on a certain day, add in the number of days until
            # the next rollover, but offset by 1 since we just calculated the time
            # until the next day starts.  There are three cases:
            # Case 1) The day to rollover is today; in this case, do nothing
            # Case 2) The day to rollover is further in the interval (i.e., today is
            #         day 2 (Wednesday) and rollover is on day 6 (Sunday).  Days to
            #         next rollover is simply 6 - 2 - 1, or 3.
            # Case 3) The day to rollover is behind us in the interval (i.e., today
            #         is day 5 (Saturday) and rollover is on day 3 (Thursday).
            #         Days to rollover is 6 - 5 + 3, or 4.  In this case, it's the
            #         number of days left in the current week (1) plus the number
            #         of days in the next week until the rollover day (3).
            # The calculations described in 2) and 3) above need to have a day added.
            # This is because the above time calculation takes us to midnight on this
            # day, i.e. the start of the next day.
            if self.when.startswith('W'):
                day = currentDay # 0 is Monday
                if day != self.dayOfWeek:
                    if day < self.dayOfWeek:
                        daysToWait = self.dayOfWeek - day
                    else:
                        daysToWait = 6 - day + self.dayOfWeek + 1
                    newRolloverAt = result + (daysToWait * (60 * 60 * 24))
                    if not self.utc:
                        dstNow = t[-1]
                        dstAtRollover = time.localtime(newRolloverAt)[-1]
                        if dstNow != dstAtRollover:
                            if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                                addend = -3600
                            else:           # DST bows out before next rollover, so we need to add an hour
                                addend = 3600
                            newRolloverAt += addend
                    result = newRolloverAt
        return result

    def shouldRollover(self, record):
        """
        Determine if rollover should occur.

        record is not used, as we are just comparing times, but it is needed so
        the method signatures are the same
        """
        t = int(time.time())
        if t >= self.rolloverAt:
            # See #89564: Never rollover anything other than regular files
            if os.path.exists(self.baseFilename) and not os.path.isfile(self.baseFilename):
                # The file is not a regular file, so do not rollover, but do
                # set the next rollover time to avoid repeated checks.
                self.rolloverAt = self.computeRollover(t)
                return False

            return True
        return False

    def getFilesToDelete(self):
        """
        Determine the files to delete when rolling over.

        More specific than the earlier method, which just used glob.glob().
        """
        dirName, baseName = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        # See bpo-44753: Don't use the extension when computing the prefix.
        n, e = os.path.splitext(baseName)
        prefix = n + '.'
        plen = len(prefix)
        for fileName in fileNames:
            if self.namer is None:
                # Our files will always start with baseName
                if not fileName.startswith(baseName):
                    continue
            else:
                # Our files could be just about anything after custom naming, but
                # likely candidates are of the form
                # foo.log.DATETIME_SUFFIX or foo.DATETIME_SUFFIX.log
                if (not fileName.startswith(baseName) and fileName.endswith(e) and
                    len(fileName) > (plen + 1) and not fileName[plen+1].isdigit()):
                    continue

            if fileName[:plen] == prefix:
                suffix = fileName[plen:]
                # See bpo-45628: The date/time suffix could be anywhere in the
                # filename
                parts = suffix.split('.')
                for part in parts:
                    if self.extMatch.match(part):
                        result.append(os.path.join(dirName, fileName))
                        break
        if len(result) < self.backupCount:
            result = []
        else:
            result.sort()
            result = result[:len(result) - self.backupCount]
        return result

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        dfn = self.rotation_filename(self.baseFilename + "." +
                                     time.strftime(self.suffix, timeTuple))
        if os.path.exists(dfn):
            os.remove(dfn)
        self.rotate(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:           # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt

class WatchedFileHandler(logging.FileHandler):
    """
    A handler for logging to a file, which watches the file
    to see if it has changed while in use. This can happen because of
    usage of programs such as newsyslog and logrotate which perform
    log file rotation. This handler, intended for use under Unix,
    watches the file to see if it has changed since the last emit.
    (A file has changed if its device or inode have changed.)
    If it has changed, the old file stream is closed, and the file
    opened to get a new stream.

    This handler is not appropriate for use under Windows, because
    under Windows open files cannot be moved or renamed - logging
    opens the files with exclusive locks - and so there is no need
    for such a handler. Furthermore, ST_INO is not supported under
    Windows; stat always returns zero for this value.

    This handler is based on a suggestion and patch by Chad J.
    Schroeder.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False,
                 errors=None):
        if "b" not in mode:
            encoding = io.text_encoding(encoding)
        logging.FileHandler.__init__(self, filename, mode=mode,
                                     encoding=encoding, delay=delay,
                                     errors=errors)
        self.dev, self.ino = -1, -1
        self._statstream()

    def _statstream(self):
        if self.stream:
            sres = os.fstat(self.stream.fileno())
            self.dev, self.ino = sres[ST_DEV], sres[ST_INO]

    def reopenIfNeeded(self):
        """
        Reopen log file if needed.

        Checks if the underlying file has changed, and if it
        has, close the old stream and reopen the file to get the
        current stream.
        """
        # Reduce the chance of race conditions by stat'ing by path only
        # once and then fstat'ing our new fd if we opened a new log stream.
        # See issue #14632: Thanks to John Mulligan for the problem report
        # and patch.
        try:
            # stat the file by path, checking for existence
            sres = os.stat(self.baseFilename)
        except FileNotFoundError:
            sres = None
        # compare file system stat with that of our stream file handle
        if not sres or sres[ST_DEV] != self.dev or sres[ST_INO] != self.ino:
            if self.stream is not None:
                # we have an open file handle, clean it up
                self.stream.flush()
                self.stream.close()
                self.stream = None  # See Issue #21742: _open () might fail.
                # open a new file handle and get new stat info from that fd
                self.stream = self._open()
                self._statstream()

    def emit(self, record):
        """
        Emit a record.

        If underlying file has changed, reopen the file before emitting the
        record to it.
        """
        self.reopenIfNeeded()
        logging.FileHandler.emit(self, record)


class SocketHandler(logging.Handler):
    """
    A handler class which writes logging records, in pickle format, to
    a streaming socket. The socket is kept open across logging calls.
    If the peer resets it, an attempt is made to reconnect on the next call.
    The pickle which is sent is that of the LogRecord's attribute dictionary
    (__dict__), so that the receiver does not need to have the logging module
    installed in order to process the logging event.

    To unpickle the record at the receiving end into a LogRecord, use the
    makeLogRecord function.
    """

    def __init__(self, host, port):
        """
        Initializes the handler with a specific host address and port.

        When the attribute *closeOnError* is set to True - if a socket error
        occurs, the socket is silently closed and then reopened on the next
        logging call.
        """
        logging.Handler.__init__(self)
        self.host = host
        self.port = port
        if port is None:
            self.address = host
        else:
            self.address = (host, port)
        self.sock = None
        self.closeOnError = False
        self.retryTime = None
        #
        # Exponential backoff parameters.
        #
        self.retryStart = 1.0
        self.retryMax = 30.0
        self.retryFactor = 2.0

    def makeSocket(self, timeout=1):
        """
        A factory method which allows subclasses to define the precise
        type of socket they want.
        """
        if self.port is not None:
            result = socket.create_connection(self.address, timeout=timeout)
        else:
            result = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            result.settimeout(timeout)
            try:
                result.connect(self.address)
            except OSError:
                result.close()  # Issue 19182
                raise
        return result

    def createSocket(self):
        """
        Try to create a socket, using an exponential backoff with
        a max retry time. Thanks to Robert Olson for the original patch
        (SF #815911) which has been slightly refactored.
        """
        now = time.time()
        # Either retryTime is None, in which case this
        # is the first time back after a disconnect, or
        # we've waited long enough.
        if self.retryTime is None:
            attempt = True
        else:
            attempt = (now >= self.retryTime)
        if attempt:
            try:
                self.sock = self.makeSocket()
                self.retryTime = None # next time, no delay before trying
            except OSError:
                #Creation failed, so set the retry time and return.
                if self.retryTime is None:
                    self.retryPeriod = self.retryStart
                else:
                    self.retryPeriod = self.retryPeriod * self.retryFactor
                    if self.retryPeriod > self.retryMax:
                        self.retryPeriod = self.retryMax
                self.retryTime = now + self.retryPeriod

    def send(self, s):
        """
        Send a pickled string to the socket.

        This function allows for partial sends which can happen when the
        network is busy.
        """
        if self.sock is None:
            self.createSocket()
        #self.sock can be None either because we haven't reached the retry
        #time yet, or because we have reached the retry time and retried,
        #but are still unable to connect.
        if self.sock:
            try:
                self.sock.sendall(s)
            except OSError: #pragma: no cover
                self.sock.close()
                self.sock = None  # so we can call createSocket next time

    def makePickle(self, record):
        """
        Pickles the record in binary format with a length prefix, and
        returns it ready for transmission across the socket.
        """
        ei = record.exc_info
        if ei:
            # just to get traceback text into record.exc_text ...
            dummy = self.format(record)
        # See issue #14436: If msg or args are objects, they may not be
        # available on the receiving end. So we convert the msg % args
        # to a string, save it as msg and zap the args.
        d = dict(record.__dict__)
        d['msg'] = record.getMessage()
        d['args'] = None
        d['exc_info'] = None
        # Issue #25685: delete 'message' if present: redundant with 'msg'
        d.pop('message', None)
        s = pickle.dumps(d, 1)
        slen = struct.pack(">L", len(s))
        return slen + s

    def handleError(self, record):
        """
        Handle an error during logging.

        An error has occurred during logging. Most likely cause -
        connection lost. Close the socket so that we can retry on the
        next event.
        """
        if self.closeOnError and self.sock:
            self.sock.close()
            self.sock = None        #try to reconnect next time
        else:
            logging.Handler.handleError(self, record)

    def emit(self, record):
        """
        Emit a record.

        Pickles the record and writes it to the socket in binary format.
        If there is an error with the socket, silently drop the packet.
        If there was a problem with the socket, re-establishes the
        socket.
        """
        try:
            s = self.makePickle(record)
            self.send(s)
        except Exception:
            self.handleError(record)

    def close(self):
        """
        Closes the socket.
        """
        self.acquire()
        try:
            sock = self.sock
            if sock:
                self.sock = None
                sock.close()
            logging.Handler.close(self)
        finally:
            self.release()

class DatagramHandler(SocketHandler):
    """
    A handler class which writes logging records, in pickle format, to
    a datagram socket.  The pickle which is sent is that of the LogRecord's
    attribute dictionary (__dict__), so that the receiver does not need to
    have the logging module installed in order to process the logging event.

    To unpickle the record at the receiving end into a LogRecord, use the
    makeLogRecord function.

    """
    def __init__(self, host, port):
        """
        Initializes the handler with a specific host address and port.
        """
        SocketHandler.__init__(self, host, port)
        self.closeOnError = False

    def makeSocket(self):
        """
        The factory method of SocketHandler is here overridden to create
        a UDP socket (SOCK_DGRAM).
        """
        if self.port is None:
            family = socket.AF_UNIX
        else:
            family = socket.AF_INET
        s = socket.socket(family, socket.SOCK_DGRAM)
        return s

    def send(self, s):
        """
        Send a pickled string to a socket.

        This function no longer allows for partial sends which can happen
        when the network is busy - UDP does not guarantee delivery and
        can deliver packets out of sequence.
        """
        if self.sock is None:
            self.createSocket()
        self.sock.sendto(s, self.address)

class SysLogHandler(logging.Handler):
    """
    A handler class which sends formatted logging records to a syslog
    server. Based on Sam Rushing's syslog module:
    http://www.nightmare.com/squirl/python-ext/misc/syslog.py
    Contributed by Nicolas Untz (after which minor refactoring changes
    have been made).
    """

    # from <linux/sys/syslog.h>:
    # ======================================================================
    # priorities/facilities are encoded into a single 32-bit quantity, where
    # the bottom 3 bits are the priority (0-7) and the top 28 bits are the
    # facility (0-big number). Both the priorities and the facilities map
    # roughly one-to-one to strings in the syslogd(8) source code.  This
    # mapping is included in this file.
    #
    # priorities (these are ordered)

    LOG_EMERG     = 0       #  system is unusable
    LOG_ALERT     = 1       #  action must be taken immediately
    LOG_CRIT      = 2       #  critical conditions
    LOG_ERR       = 3       #  error conditions
    LOG_WARNING   = 4       #  warning conditions
    LOG_NOTICE    = 5       #  normal but significant condition
    LOG_INFO      = 6       #  informational
    LOG_DEBUG     = 7       #  debug-level messages

    #  facility codes
    LOG_KERN      = 0       #  kernel messages
    LOG_USER      = 1       #  random user-level messages
    LOG_MAIL      = 2       #  mail system
    LOG_DAEMON    = 3       #  system daemons
    LOG_AUTH      = 4       #  security/authorization messages
    LOG_SYSLOG    = 5       #  messages generated internally by syslogd
    LOG_LPR       = 6       #  line printer subsystem
    LOG_NEWS      = 7       #  network news subsystem
    LOG_UUCP      = 8       #  UUCP subsystem
    LOG_CRON      = 9       #  clock daemon
    LOG_AUTHPRIV  = 10      #  security/authorization messages (private)
    LOG_FTP       = 11      #  FTP daemon
    LOG_NTP       = 12      #  NTP subsystem
    LOG_SECURITY  = 13      #  Log audit
    LOG_CONSOLE   = 14      #  Log alert
    LOG_SOLCRON   = 15      #  Scheduling daemon (Solaris)

    #  other codes through 15 reserved for system use
    LOG_LOCAL0    = 16      #  reserved for local use
    LOG_LOCAL1    = 17      #  reserved for local use
    LOG_LOCAL2    = 18      #  reserved for local use
    LOG_LOCAL3    = 19      #  reserved for local use
    LOG_LOCAL4    = 20      #  reserved for local use
    LOG_LOCAL5    = 21      #  reserved for local use
    LOG_LOCAL6    = 22      #  reserved for local use
    LOG_LOCAL7    = 23      #  reserved for local use

    priority_names = {
        "alert":    LOG_ALERT,
        "crit":     LOG_CRIT,
        "critical": LOG_CRIT,
        "debug":    LOG_DEBUG,
        "emerg":    LOG_EMERG,
        "err":      LOG_ERR,
        "error":    LOG_ERR,        #  DEPRECATED
        "info":     LOG_INFO,
        "notice":   LOG_NOTICE,
        "panic":    LOG_EMERG,      #  DEPRECATED
        "warn":     LOG_WARNING,    #  DEPRECATED
        "warning":  LOG_WARNING,
        }

    facility_names = {
        "auth":         LOG_AUTH,
        "authpriv":     LOG_AUTHPRIV,
        "console":      LOG_CONSOLE,
        "cron":         LOG_CRON,
        "daemon":       LOG_DAEMON,
        "ftp":          LOG_FTP,
        "kern":         LOG_KERN,
        "lpr":          LOG_LPR,
        "mail":         LOG_MAIL,
        "news":         LOG_NEWS,
        "ntp":          LOG_NTP,
        "security":     LOG_SECURITY,
        "solaris-cron": LOG_SOLCRON,
        "syslog":       LOG_SYSLOG,
        "user":         LOG_USER,
        "uucp":         LOG_UUCP,
        "local0":       LOG_LOCAL0,
        "local1":       LOG_LOCAL1,
        "local2":       LOG_LOCAL2,
        "local3":       LOG_LOCAL3,
        "local4":       LOG_LOCAL4,
        "local5":       LOG_LOCAL5,
        "local6":       LOG_LOCAL6,
        "local7":       LOG_LOCAL7,
        }

    #The map below appears to be trivially lowercasing the key. However,
    #there's more to it than meets the eye - in some locales, lowercasing
    #gives unexpected results. See SF #1524081: in the Turkish locale,
    #"INFO".lower() != "info"
    priority_map = {
        "DEBUG" : "debug",
        "INFO" : "info",
        "WARNING" : "warning",
        "ERROR" : "error",
        "CRITICAL" : "critical"
    }

    def __init__(self, address=('localhost', SYSLOG_UDP_PORT),
                 facility=LOG_USER, socktype=None):
        """
        Initialize a handler.

        If address is specified as a string, a UNIX socket is used. To log to a
        local syslogd, "SysLogHandler(address="/dev/log")" can be used.
        If facility is not specified, LOG_USER is used. If socktype is
        specified as socket.SOCK_DGRAM or socket.SOCK_STREAM, that specific
        socket type will be used. For Unix sockets, you can also specify a
        socktype of None, in which case socket.SOCK_DGRAM will be used, falling
        back to socket.SOCK_STREAM.
        """
        logging.Handler.__init__(self)

        self.address = address
        self.facility = facility
        self.socktype = socktype

        if isinstance(address, str):
            self.unixsocket = True
            # Syslog server may be unavailable during handler initialisation.
            # C's openlog() function also ignores connection errors.
            # Moreover, we ignore these errors while logging, so it not worse
            # to ignore it also here.
            try:
                self._connect_unixsocket(address)
            except OSError:
                pass
        else:
            self.unixsocket = False
            if socktype is None:
                socktype = socket.SOCK_DGRAM
            host, port = address
            ress = socket.getaddrinfo(host, port, 0, socktype)
            if not ress:
                raise OSError("getaddrinfo returns an empty list")
            for res in ress:
                af, socktype, proto, _, sa = res
                err = sock = None
                try:
                    sock = socket.socket(af, socktype, proto)
                    if socktype == socket.SOCK_STREAM:
                        sock.connect(sa)
                    break
                except OSError as exc:
                    err = exc
                    if sock is not None:
                        sock.close()
            if err is not None:
                raise err
            self.socket = sock
            self.socktype = socktype

    def _connect_unixsocket(self, address):
        use_socktype = self.socktype
        if use_socktype is None:
            use_socktype = socket.SOCK_DGRAM
        self.socket = socket.socket(socket.AF_UNIX, use_socktype)
        try:
            self.socket.connect(address)
            # it worked, so set self.socktype to the used type
            self.socktype = use_socktype
        except OSError:
            self.socket.close()
            if self.socktype is not None:
                # user didn't specify falling back, so fail
                raise
            use_socktype = socket.SOCK_STREAM
            self.socket = socket.socket(socket.AF_UNIX, use_socktype)
            try:
                self.socket.connect(address)
                # it worked, so set self.socktype to the used type
                self.socktype = use_socktype
            except OSError:
                self.socket.close()
                raise

    def encodePriority(self, facility, priority):
        """
        Encode the facility and priority. You can pass in strings or
        integers - if strings are passed, the facility_names and
        priority_names mapping dictionaries are used to convert them to
        integers.
        """
        if isinstance(facility, str):
            facility = self.facility_names[facility]
        if isinstance(priority, str):
            priority = self.priority_names[priority]
        return (facility << 3) | priority

    def close(self):
        """
        Closes the socket.
        """
        self.acquire()
        try:
            self.socket.close()
            logging.Handler.close(self)
        finally:
            self.release()

    def mapPriority(self, levelName):
        """
        Map a logging level name to a key in the priority_names map.
        This is useful in two scenarios: when custom levels are being
        used, and in the case where you can't do a straightforward
        mapping by lowercasing the logging level name because of locale-
        specific issues (see SF #1524081).
        """
        return self.priority_map.get(levelName, "warning")

    ident = ''          # prepended to all messages
    append_nul = True   # some old syslog daemons expect a NUL terminator

    def emit(self, record):
        """
        Emit a record.

        The record is formatted, and then sent to the syslog server. If
        exception information is present, it is NOT sent to the server.
        """
        try:
            msg = self.format(record)
            if self.ident:
                msg = self.ident + msg
            if self.append_nul:
                msg += '\000'

            # We need to convert record level to lowercase, maybe this will
            # change in the future.
            prio = '<%d>' % self.encodePriority(self.facility,
                                                self.mapPriority(record.levelname))
            prio = prio.encode('utf-8')
            # Message is a string. Convert to bytes as required by RFC 5424
            msg = msg.encode('utf-8')
            msg = prio + msg
            if self.unixsocket:
                try:
                    self.socket.send(msg)
                except OSError:
                    self.socket.close()
                    self._connect_unixsocket(self.address)
                    self.socket.send(msg)
            elif self.socktype == socket.SOCK_DGRAM:
                self.socket.sendto(msg, self.address)
            else:
                self.socket.sendall(msg)
        except Exception:
            self.handleError(record)

class SMTPHandler(logging.Handler):
    """
    A handler class which sends an SMTP email for each logging event.
    """
    def __init__(self, mailhost, fromaddr, toaddrs, subject,
                 credentials=None, secure=None, timeout=5.0):
        """
        Initialize the handler.

        Initialize the instance with the from and to addresses and subject
        line of the email. To specify a non-standard SMTP port, use the
        (host, port) tuple format for the mailhost argument. To specify
        authentication credentials, supply a (username, password) tuple
        for the credentials argument. To specify the use of a secure
        protocol (TLS), pass in a tuple for the secure argument. This will
        only be used when authentication credentials are supplied. The tuple
        will be either an empty tuple, or a single-value tuple with the name
        of a keyfile, or a 2-value tuple with the names of the keyfile and
        certificate file. (This tuple is passed to the `starttls` method).
        A timeout in seconds can be specified for the SMTP connection (the
        default is one second).
        """
        logging.Handler.__init__(self)
        if isinstance(mailhost, (list, tuple)):
            self.mailhost, self.mailport = mailhost
        else:
            self.mailhost, self.mailport = mailhost, None
        if isinstance(credentials, (list, tuple)):
            self.username, self.password = credentials
        else:
            self.username = None
        self.fromaddr = fromaddr
        if isinstance(toaddrs, str):
            toaddrs = [toaddrs]
        self.toaddrs = toaddrs
        self.subject = subject
        self.secure = secure
        self.timeout = timeout

    def getSubject(self, record):
        """
        Determine the subject for the email.

        If you want to specify a subject line which is record-dependent,
        override this method.
        """
        return self.subject

    def emit(self, record):
        """
        Emit a record.

        Format the record and send it to the specified addressees.
        """
        try:
            import smtplib
            from email.message import EmailMessage
            import email.utils

            port = self.mailport
            if not port:
                port = smtplib.SMTP_PORT
            smtp = smtplib.SMTP(self.mailhost, port, timeout=self.timeout)
            msg = EmailMessage()
            msg['From'] = self.fromaddr
            msg['To'] = ','.join(self.toaddrs)
            msg['Subject'] = self.getSubject(record)
            msg['Date'] = email.utils.localtime()
            msg.set_content(self.format(record))
            if self.username:
                if self.secure is not None:
                    smtp.ehlo()
                    smtp.starttls(*self.secure)
                    smtp.ehlo()
                smtp.login(self.username, self.password)
            smtp.send_message(msg)
            smtp.quit()
        except Exception:
            self.handleError(record)

class NTEventLogHandler(logging.Handler):
    """
    A handler class which sends events to the NT Event Log. Adds a
    registry entry for the specified application name. If no dllname is
    provided, win32service.pyd (which contains some basic message
    placeholders) is used. Note that use of these placeholders will make
    your event logs big, as the entire message source is held in the log.
    If you want slimmer logs, you have to pass in the name of your own DLL
    which contains the message definitions you want to use in the event log.
    """
    def __init__(self, appname, dllname=None, logtype="Application"):
        logging.Handler.__init__(self)
        try:
            import win32evtlogutil, win32evtlog
            self.appname = appname
            self._welu = win32evtlogutil
            if not dllname:
                dllname = os.path.split(self._welu.__file__)
                dllname = os.path.split(dllname[0])
                dllname = os.path.join(dllname[0], r'win32service.pyd')
            self.dllname = dllname
            self.logtype = logtype
            # Administrative privileges are required to add a source to the registry.
            # This may not be available for a user that just wants to add to an
            # existing source - handle this specific case.
            try:
                self._welu.AddSourceToRegistry(appname, dllname, logtype)
            except Exception as e:
                # This will probably be a pywintypes.error. Only raise if it's not
                # an "access denied" error, else let it pass
                if getattr(e, 'winerror', None) != 5:  # not access denied
                    raise
            self.deftype = win32evtlog.EVENTLOG_ERROR_TYPE
            self.typemap = {
                logging.DEBUG   : win32evtlog.EVENTLOG_INFORMATION_TYPE,
                logging.INFO    : win32evtlog.EVENTLOG_INFORMATION_TYPE,
                logging.WARNING : win32evtlog.EVENTLOG_WARNING_TYPE,
                logging.ERROR   : win32evtlog.EVENTLOG_ERROR_TYPE,
                logging.CRITICAL: win32evtlog.EVENTLOG_ERROR_TYPE,
         }
        except ImportError:
            print("The Python Win32 extensions for NT (service, event "\
                        "logging) appear not to be available.")
            self._welu = None

    def getMessageID(self, record):
        """
        Return the message ID for the event record. If you are using your
        own messages, you could do this by having the msg passed to the
        logger being an ID rather than a formatting string. Then, in here,
        you could use a dictionary lookup to get the message ID. This
        version returns 1, which is the base message ID in win32service.pyd.
        """
        return 1

    def getEventCategory(self, record):
        """
        Return the event category for the record.

        Override this if you want to specify your own categories. This version
        returns 0.
        """
        return 0

    def getEventType(self, record):
        """
        Return the event type for the record.

        Override this if you want to specify your own types. This version does
        a mapping using the handler's typemap attribute, which is set up in
        __init__() to a dictionary which contains mappings for DEBUG, INFO,
        WARNING, ERROR and CRITICAL. If you are using your own levels you will
        either need to override this method or place a suitable dictionary in
        the handler's typemap attribute.
        """
        return self.typemap.get(record.levelno, self.deftype)

    def emit(self, record):
        """
        Emit a record.

        Determine the message ID, event category and event type. Then
        log the message in the NT event log.
        """
        if self._welu:
            try:
                id = self.getMessageID(record)
                cat = self.getEventCategory(record)
                type = self.getEventType(record)
                msg = self.format(record)
                self._welu.ReportEvent(self.appname, id, cat, type, [msg])
            except Exception:
                self.handleError(record)

    def close(self):
        """
        Clean up this handler.

        You can remove the application name from the registry as a
        source of event log entries. However, if you do this, you will
        not be able to see the events as you intended in the Event Log
        Viewer - it needs to be able to access the registry to get the
        DLL name.
        """
        #self._welu.RemoveSourceFromRegistry(self.appname, self.logtype)
        logging.Handler.close(self)

class HTTPHandler(logging.Handler):
    """
    A class which sends records to a web server, using either GET or
    POST semantics.
    """
    def __init__(self, host, url, method="GET", secure=False, credentials=None,
                 context=None):
        """
        Initialize the instance with the host, the request URL, and the method
        ("GET" or "POST")
        """
        logging.Handler.__init__(self)
        method = method.upper()
        if method not in ["GET", "POST"]:
            raise ValueError("method must be GET or POST")
        if not secure and context is not None:
            raise ValueError("context parameter only makes sense "
                             "with secure=True")
        self.host = host
        self.url = url
        self.method = method
        self.secure = secure
        self.credentials = credentials
        self.context = context

    def mapLogRecord(self, record):
        """
        Default implementation of mapping the log record into a dict
        that is sent as the CGI data. Overwrite in your class.
        Contributed by Franz Glasner.
        """
        return record.__dict__

    def getConnection(self, host, secure):
        """
        get a HTTP[S]Connection.

        Override when a custom connection is required, for example if
        there is a proxy.
        """
        import http.client
        if secure:
            connection = http.client.HTTPSConnection(host, context=self.context)
        else:
            connection = http.client.HTTPConnection(host)
        return connection

    def emit(self, record):
        """
        Emit a record.

        Send the record to the web server as a percent-encoded dictionary
        """
        try:
            import urllib.parse
            host = self.host
            h = self.getConnection(host, self.secure)
            url = self.url
            data = urllib.parse.urlencode(self.mapLogRecord(record))
            if self.method == "GET":
                if (url.find('?') >= 0):
                    sep = '&'
                else:
                    sep = '?'
                url = url + "%c%s" % (sep, data)
            h.putrequest(self.method, url)
            # support multiple hosts on one IP address...
            # need to strip optional :port from host, if present
            i = host.find(":")
            if i >= 0:
                host = host[:i]
            # See issue #30904: putrequest call above already adds this header
            # on Python 3.x.
            # h.putheader("Host", host)
            if self.method == "POST":
                h.putheader("Content-type",
                            "application/x-www-form-urlencoded")
                h.putheader("Content-length", str(len(data)))
            if self.credentials:
                import base64
                s = ('%s:%s' % self.credentials).encode('utf-8')
                s = 'Basic ' + base64.b64encode(s).strip().decode('ascii')
                h.putheader('Authorization', s)
            h.endheaders()
            if self.method == "POST":
                h.send(data.encode('utf-8'))
            h.getresponse()    #can't do anything with the result
        except Exception:
            self.handleError(record)

class BufferingHandler(logging.Handler):
    """
  A handler class which buffers logging records in memory. Whenever each
  record is added to the buffer, a check is made to see if the buffer should
  be flushed. If it should, then flush() is expected to do what's needed.
    """
    def __init__(self, capacity):
        """
        Initialize the handler with the buffer size.
        """
        logging.Handler.__init__(self)
        self.capacity = capacity
        self.buffer = []

    def shouldFlush(self, record):
        """
        Should the handler flush its buffer?

        Returns true if the buffer is up to capacity. This method can be
        overridden to implement custom flushing strategies.
        """
        return (len(self.buffer) >= self.capacity)

    def emit(self, record):
        """
        Emit a record.

        Append the record. If shouldFlush() tells us to, call flush() to process
        the buffer.
        """
        self.buffer.append(record)
        if self.shouldFlush(record):
            self.flush()

    def flush(self):
        """
        Override to implement custom flushing behaviour.

        This version just zaps the buffer to empty.
        """
        self.acquire()
        try:
            self.buffer.clear()
        finally:
            self.release()

    def close(self):
        """
        Close the handler.

        This version just flushes and chains to the parent class' close().
        """
        try:
            self.flush()
        finally:
            logging.Handler.close(self)

class MemoryHandler(BufferingHandler):
    """
    A handler class which buffers logging records in memory, periodically
    flushing them to a target handler. Flushing occurs whenever the buffer
    is full, or when an event of a certain severity or greater is seen.
    """
    def __init__(self, capacity, flushLevel=logging.ERROR, target=None,
                 flushOnClose=True):
        """
        Initialize the handler with the buffer size, the level at which
        flushing should occur and an optional target.

        Note that without a target being set either here or via setTarget(),
        a MemoryHandler is no use to anyone!

        The ``flushOnClose`` argument is ``True`` for backward compatibility
        reasons - the old behaviour is that when the handler is closed, the
        buffer is flushed, even if the flush level hasn't been exceeded nor the
        capacity exceeded. To prevent this, set ``flushOnClose`` to ``False``.
        """
        BufferingHandler.__init__(self, capacity)
        self.flushLevel = flushLevel
        self.target = target
        # See Issue #26559 for why this has been added
        self.flushOnClose = flushOnClose

    def shouldFlush(self, record):
        """
        Check for buffer full or a record at the flushLevel or higher.
        """
        return (len(self.buffer) >= self.capacity) or \
                (record.levelno >= self.flushLevel)

    def setTarget(self, target):
        """
        Set the target handler for this handler.
        """
        self.acquire()
        try:
            self.target = target
        finally:
            self.release()

    def flush(self):
        """
        For a MemoryHandler, flushing means just sending the buffered
        records to the target, if there is one. Override if you want
        different behaviour.

        The record buffer is also cleared by this operation.
        """
        self.acquire()
        try:
            if self.target:
                for record in self.buffer:
                    self.target.handle(record)
                self.buffer.clear()
        finally:
            self.release()

    def close(self):
        """
        Flush, if appropriately configured, set the target to None and lose the
        buffer.
        """
        try:
            if self.flushOnClose:
                self.flush()
        finally:
            self.acquire()
            try:
                self.target = None
                BufferingHandler.close(self)
            finally:
                self.release()


class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.

    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def enqueue(self, record):
        """
        Enqueue a record.

        The base implementation uses put_nowait. You may want to override
        this method if you want to use blocking, timeouts or custom queue
        implementations.
        """
        self.queue.put_nowait(record)

    def prepare(self, record):
        """
        Prepares a record for queuing. The object returned by this method is
        enqueued.

        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.

        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        # The format operation gets traceback text into record.exc_text
        # (if there's exception data), and also returns the formatted
        # message. We can then use this to replace the original
        # msg + args, as these might be unpickleable. We also zap the
        # exc_info, exc_text and stack_info attributes, as they are no longer
        # needed and, if not None, will typically not be pickleable.
        msg = self.format(record)
        # bpo-35726: make copy of record to avoid affecting other handlers in the chain.
        record = copy.copy(record)
        record.message = msg
        record.msg = msg
        record.args = None
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return record

    def emit(self, record):
        """
        Emit a record.

        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)


class QueueListener(object):
    """
    This class implements an internal threaded listener which watches for
    LogRecords being added to a queue, removes them and passes them to a
    list of handlers for processing.
    """
    _sentinel = None

    def __init__(self, queue, *handlers, respect_handler_level=False):
        """
        Initialise an instance with the specified queue and
        handlers.
        """
        self.queue = queue
        self.handlers = handlers
        self._thread = None
        self.respect_handler_level = respect_handler_level

    def dequeue(self, block):
        """
        Dequeue a record and return it, optionally blocking.

        The base implementation uses get. You may want to override this method
        if you want to use timeouts or work with custom queue implementations.
        """
        return self.queue.get(block)

    def start(self):
        """
        Start the listener.

        This starts up a background thread to monitor the queue for
        LogRecords to process.
        """
        self._thread = t = threading.Thread(target=self._monitor)
        t.daemon = True
        t.start()

    def prepare(self, record):
        """
        Prepare a record for handling.

        This method just returns the passed-in record. You may want to
        override this method if you need to do any custom marshalling or
        manipulation of the record before passing it to the handlers.
        """
        return record

    def handle(self, record):
        """
        Handle a record.

        This just loops through the handlers offering them the record
        to handle.
        """
        record = self.prepare(record)
        for handler in self.handlers:
            if not self.respect_handler_level:
                process = True
            else:
                process = record.levelno >= handler.level
            if process:
                handler.handle(record)

    def _monitor(self):
        """
        Monitor the queue for records, and ask the handler
        to deal with them.

        This method runs on a separate, internal thread.
        The thread will terminate if it sees a sentinel object in the queue.
        """
        q = self.queue
        has_task_done = hasattr(q, 'task_done')
        while True:
            try:
                record = self.dequeue(True)
                if record is self._sentinel:
                    if has_task_done:
                        q.task_done()
                    break
                self.handle(record)
                if has_task_done:
                    q.task_done()
            except queue.Empty:
                break

    def enqueue_sentinel(self):
        """
        This is used to enqueue the sentinel record.

        The base implementation uses put_nowait. You may want to override this
        method if you want to use timeouts or work with custom queue
        implementations.
        """
        self.queue.put_nowait(self._sentinel)

    def stop(self):
        """
        Stop the listener.

        This asks the thread to terminate, and then waits for it to do so.
        Note that if you don't call this before your application exits, there
        may be some records still left on the queue, which won't be processed.
        """
        self.enqueue_sentinel()
        self._thread.join()
        self._thread = None
<|endoftext|>import os
import sysconfig


def reset_tzpath(to=None):
    global TZPATH

    tzpaths = to
    if tzpaths is not None:
        if isinstance(tzpaths, (str, bytes)):
            raise TypeError(
                f"tzpaths must be a list or tuple, "
                + f"not {type(tzpaths)}: {tzpaths!r}"
            )

        if not all(map(os.path.isabs, tzpaths)):
            raise ValueError(_get_invalid_paths_message(tzpaths))
        base_tzpath = tzpaths
    else:
        env_var = os.environ.get("PYTHONTZPATH", None)
        if env_var is not None:
            base_tzpath = _parse_python_tzpath(env_var)
        else:
            base_tzpath = _parse_python_tzpath(
                sysconfig.get_config_var("TZPATH")
            )

    TZPATH = tuple(base_tzpath)


def _parse_python_tzpath(env_var):
    if not env_var:
        return ()

    raw_tzpath = env_var.split(os.pathsep)
    new_tzpath = tuple(filter(os.path.isabs, raw_tzpath))

    # If anything has been filtered out, we will warn about it
    if len(new_tzpath) != len(raw_tzpath):
        import warnings

        msg = _get_invalid_paths_message(raw_tzpath)

        warnings.warn(
            "Invalid paths specified in PYTHONTZPATH environment variable. "
            + msg,
            InvalidTZPathWarning,
        )

    return new_tzpath


def _get_invalid_paths_message(tzpaths):
    invalid_paths = (path for path in tzpaths if not os.path.isabs(path))

    prefix = "\n    "
    indented_str = prefix + prefix.join(invalid_paths)

    return (
        "Paths should be absolute but found the following relative paths:"
        + indented_str
    )


def find_tzfile(key):
    """Retrieve the path to a TZif file from a key."""
    _validate_tzfile_path(key)
    for search_path in TZPATH:
        filepath = os.path.join(search_path, key)
        if os.path.isfile(filepath):
            return filepath

    return None


_TEST_PATH = os.path.normpath(os.path.join("_", "_"))[:-1]


def _validate_tzfile_path(path, _base=_TEST_PATH):
    if os.path.isabs(path):
        raise ValueError(
            f"ZoneInfo keys may not be absolute paths, got: {path}"
        )

    # We only care about the kinds of path normalizations that would change the
    # length of the key - e.g. a/../b -> a/b, or a/b/ -> a/b. On Windows,
    # normpath will also change from a/b to a\b, but that would still preserve
    # the length.
    new_path = os.path.normpath(path)
    if len(new_path) != len(path):
        raise ValueError(
            f"ZoneInfo keys must be normalized relative paths, got: {path}"
        )

    resolved = os.path.normpath(os.path.join(_base, new_path))
    if not resolved.startswith(_base):
        raise ValueError(
            f"ZoneInfo keys must refer to subdirectories of TZPATH, got: {path}"
        )


del _TEST_PATH


def available_timezones():
    """Returns a set containing all available time zones.

    .. caution::

        This may attempt to open a large number of files, since the best way to
        determine if a given file on the time zone search path is to open it
        and check for the "magic string" at the beginning.
    """
    from importlib import resources

    valid_zones = set()

    # Start with loading from the tzdata package if it exists: this has a
    # pre-assembled list of zones that only requires opening one file.
    try:
        with resources.open_text("tzdata", "zones") as f:
            for zone in f:
                zone = zone.strip()
                if zone:
                    valid_zones.add(zone)
    except (ImportError, FileNotFoundError):
        pass

    def valid_key(fpath):
        try:
            with open(fpath, "rb") as f:
                return f.read(4) == b"TZif"
        except Exception:  # pragma: nocover
            return False

    for tz_root in TZPATH:
        if not os.path.exists(tz_root):
            continue

        for root, dirnames, files in os.walk(tz_root):
            if root == tz_root:
                # right/ and posix/ are special directories and shouldn't be
                # included in the output of available zones
                if "right" in dirnames:
                    dirnames.remove("right")
                if "posix" in dirnames:
                    dirnames.remove("posix")

            for file in files:
                fpath = os.path.join(root, file)

                key = os.path.relpath(fpath, start=tz_root)
                if os.sep != "/":  # pragma: nocover
                    key = key.replace(os.sep, "/")

                if not key or key in valid_zones:
                    continue

                if valid_key(fpath):
                    valid_zones.add(key)

    if "posixrules" in valid_zones:
        # posixrules is a special symlink-only time zone where it exists, it
        # should not be included in the output
        valid_zones.remove("posixrules")

    return valid_zones


class InvalidTZPathWarning(RuntimeWarning):
    """Warning raised if an invalid path is specified in PYTHONTZPATH."""


TZPATH = ()
reset_tzpath()
<|endoftext|>__all__ = [
    "ZoneInfo",
    "reset_tzpath",
    "available_timezones",
    "TZPATH",
    "ZoneInfoNotFoundError",
    "InvalidTZPathWarning",
]

from . import _tzpath
from ._common import ZoneInfoNotFoundError

try:
    from _zoneinfo import ZoneInfo
except ImportError:  # pragma: nocover
    from ._zoneinfo import ZoneInfo

reset_tzpath = _tzpath.reset_tzpath
available_timezones = _tzpath.available_timezones
InvalidTZPathWarning = _tzpath.InvalidTZPathWarning


def __getattr__(name):
    if name == "TZPATH":
        return _tzpath.TZPATH
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + ["TZPATH"])
<|endoftext|>import struct


def load_tzdata(key):
    import importlib.resources

    components = key.split("/")
    package_name = ".".join(["tzdata.zoneinfo"] + components[:-1])
    resource_name = components[-1]

    try:
        return importlib.resources.open_binary(package_name, resource_name)
    except (ImportError, FileNotFoundError, UnicodeEncodeError):
        # There are three types of exception that can be raised that all amount
        # to "we cannot find this key":
        #
        # ImportError: If package_name doesn't exist (e.g. if tzdata is not
        #   installed, or if there's an error in the folder name like
        #   Amrica/New_York)
        # FileNotFoundError: If resource_name doesn't exist in the package
        #   (e.g. Europe/Krasnoy)
        # UnicodeEncodeError: If package_name or resource_name are not UTF-8,
        #   such as keys containing a surrogate character.
        raise ZoneInfoNotFoundError(f"No time zone found with key {key}")


def load_data(fobj):
    header = _TZifHeader.from_file(fobj)

    if header.version == 1:
        time_size = 4
        time_type = "l"
    else:
        # Version 2+ has 64-bit integer transition times
        time_size = 8
        time_type = "q"

        # Version 2+ also starts with a Version 1 header and data, which
        # we need to skip now
        skip_bytes = (
            header.timecnt * 5  # Transition times and types
            + header.typecnt * 6  # Local time type records
            + header.charcnt  # Time zone designations
            + header.leapcnt * 8  # Leap second records
            + header.isstdcnt  # Standard/wall indicators
            + header.isutcnt  # UT/local indicators
        )

        fobj.seek(skip_bytes, 1)

        # Now we need to read the second header, which is not the same
        # as the first
        header = _TZifHeader.from_file(fobj)

    typecnt = header.typecnt
    timecnt = header.timecnt
    charcnt = header.charcnt

    # The data portion starts with timecnt transitions and indices
    if timecnt:
        trans_list_utc = struct.unpack(
            f">{timecnt}{time_type}", fobj.read(timecnt * time_size)
        )
        trans_idx = struct.unpack(f">{timecnt}B", fobj.read(timecnt))
    else:
        trans_list_utc = ()
        trans_idx = ()

    # Read the ttinfo struct, (utoff, isdst, abbrind)
    if typecnt:
        utcoff, isdst, abbrind = zip(
            *(struct.unpack(">lbb", fobj.read(6)) for i in range(typecnt))
        )
    else:
        utcoff = ()
        isdst = ()
        abbrind = ()

    # Now read the abbreviations. They are null-terminated strings, indexed
    # not by position in the array but by position in the unsplit
    # abbreviation string. I suppose this makes more sense in C, which uses
    # null to terminate the strings, but it's inconvenient here...
    abbr_vals = {}
    abbr_chars = fobj.read(charcnt)

    def get_abbr(idx):
        # Gets a string starting at idx and running until the next \x00
        #
        # We cannot pre-populate abbr_vals by splitting on \x00 because there
        # are some zones that use subsets of longer abbreviations, like so:
        #
        #  LMT\x00AHST\x00HDT\x00
        #
        # Where the idx to abbr mapping should be:
        #
        # {0: "LMT", 4: "AHST", 5: "HST", 9: "HDT"}
        if idx not in abbr_vals:
            span_end = abbr_chars.find(b"\x00", idx)
            abbr_vals[idx] = abbr_chars[idx:span_end].decode()

        return abbr_vals[idx]

    abbr = tuple(get_abbr(idx) for idx in abbrind)

    # The remainder of the file consists of leap seconds (currently unused) and
    # the standard/wall and ut/local indicators, which are metadata we don't need.
    # In version 2 files, we need to skip the unnecessary data to get at the TZ string:
    if header.version >= 2:
        # Each leap second record has size (time_size + 4)
        skip_bytes = header.isutcnt + header.isstdcnt + header.leapcnt * 12
        fobj.seek(skip_bytes, 1)

        c = fobj.read(1)  # Should be \n
        assert c == b"\n", c

        tz_bytes = b""
        while (c := fobj.read(1)) != b"\n":
            tz_bytes += c

        tz_str = tz_bytes
    else:
        tz_str = None

    return trans_idx, trans_list_utc, utcoff, isdst, abbr, tz_str


class _TZifHeader:
    __slots__ = [
        "version",
        "isutcnt",
        "isstdcnt",
        "leapcnt",
        "timecnt",
        "typecnt",
        "charcnt",
    ]

    def __init__(self, *args):
        for attr, val in zip(self.__slots__, args, strict=True):
            setattr(self, attr, val)

    @classmethod
    def from_file(cls, stream):
        # The header starts with a 4-byte "magic" value
        if stream.read(4) != b"TZif":
            raise ValueError("Invalid TZif file: magic not found")

        _version = stream.read(1)
        if _version == b"\x00":
            version = 1
        else:
            version = int(_version)
        stream.read(15)

        args = (version,)

        # Slots are defined in the order that the bytes are arranged
        args = args + struct.unpack(">6l", stream.read(24))

        return cls(*args)


class ZoneInfoNotFoundError(KeyError):
    """Exception raised when a ZoneInfo key is not found."""
<|endoftext|>import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo

from . import _common, _tzpath

EPOCH = datetime(1970, 1, 1)
EPOCHORDINAL = datetime(1970, 1, 1).toordinal()

# It is relatively expensive to construct new timedelta objects, and in most
# cases we're looking at the same deltas, like integer numbers of hours, etc.
# To improve speed and memory use, we'll keep a dictionary with references
# to the ones we've already used so far.
#
# Loading every time zone in the 2020a version of the time zone database
# requires 447 timedeltas, which requires approximately the amount of space
# that ZoneInfo("America/New_York") with 236 transitions takes up, so we will
# set the cache size to 512 so that in the common case we always get cache
# hits, but specifically crafted ZoneInfo objects don't leak arbitrary amounts
# of memory.
@functools.lru_cache(maxsize=512)
def _load_timedelta(seconds):
    return timedelta(seconds=seconds)


class ZoneInfo(tzinfo):
    _strong_cache_size = 8
    _strong_cache = collections.OrderedDict()
    _weak_cache = weakref.WeakValueDictionary()
    __module__ = "zoneinfo"

    def __init_subclass__(cls):
        cls._strong_cache = collections.OrderedDict()
        cls._weak_cache = weakref.WeakValueDictionary()

    def __new__(cls, key):
        instance = cls._weak_cache.get(key, None)
        if instance is None:
            instance = cls._weak_cache.setdefault(key, cls._new_instance(key))
            instance._from_cache = True

        # Update the "strong" cache
        cls._strong_cache[key] = cls._strong_cache.pop(key, instance)

        if len(cls._strong_cache) > cls._strong_cache_size:
            cls._strong_cache.popitem(last=False)

        return instance

    @classmethod
    def no_cache(cls, key):
        obj = cls._new_instance(key)
        obj._from_cache = False

        return obj

    @classmethod
    def _new_instance(cls, key):
        obj = super().__new__(cls)
        obj._key = key
        obj._file_path = obj._find_tzfile(key)

        if obj._file_path is not None:
            file_obj = open(obj._file_path, "rb")
        else:
            file_obj = _common.load_tzdata(key)

        with file_obj as f:
            obj._load_file(f)

        return obj

    @classmethod
    def from_file(cls, fobj, /, key=None):
        obj = super().__new__(cls)
        obj._key = key
        obj._file_path = None
        obj._load_file(fobj)
        obj._file_repr = repr(fobj)

        # Disable pickling for objects created from files
        obj.__reduce__ = obj._file_reduce

        return obj

    @classmethod
    def clear_cache(cls, *, only_keys=None):
        if only_keys is not None:
            for key in only_keys:
                cls._weak_cache.pop(key, None)
                cls._strong_cache.pop(key, None)

        else:
            cls._weak_cache.clear()
            cls._strong_cache.clear()

    @property
    def key(self):
        return self._key

    def utcoffset(self, dt):
        return self._find_trans(dt).utcoff

    def dst(self, dt):
        return self._find_trans(dt).dstoff

    def tzname(self, dt):
        return self._find_trans(dt).tzname

    def fromutc(self, dt):
        """Convert from datetime in UTC to datetime in local time"""

        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        timestamp = self._get_local_timestamp(dt)
        num_trans = len(self._trans_utc)

        if num_trans >= 1 and timestamp < self._trans_utc[0]:
            tti = self._tti_before
            fold = 0
        elif (
            num_trans == 0 or timestamp > self._trans_utc[-1]
        ) and not isinstance(self._tz_after, _ttinfo):
            tti, fold = self._tz_after.get_trans_info_fromutc(
                timestamp, dt.year
            )
        elif num_trans == 0:
            tti = self._tz_after
            fold = 0
        else:
            idx = bisect.bisect_right(self._trans_utc, timestamp)

            if num_trans > 1 and timestamp >= self._trans_utc[1]:
                tti_prev, tti = self._ttinfos[idx - 2 : idx]
            elif timestamp > self._trans_utc[-1]:
                tti_prev = self._ttinfos[-1]
                tti = self._tz_after
            else:
                tti_prev = self._tti_before
                tti = self._ttinfos[0]

            # Detect fold
            shift = tti_prev.utcoff - tti.utcoff
            fold = shift.total_seconds() > timestamp - self._trans_utc[idx - 1]
        dt += tti.utcoff
        if fold:
            return dt.replace(fold=1)
        else:
            return dt

    def _find_trans(self, dt):
        if dt is None:
            if self._fixed_offset:
                return self._tz_after
            else:
                return _NO_TTINFO

        ts = self._get_local_timestamp(dt)

        lt = self._trans_local[dt.fold]

        num_trans = len(lt)

        if num_trans and ts < lt[0]:
            return self._tti_before
        elif not num_trans or ts > lt[-1]:
            if isinstance(self._tz_after, _TZStr):
                return self._tz_after.get_trans_info(ts, dt.year, dt.fold)
            else:
                return self._tz_after
        else:
            # idx is the transition that occurs after this timestamp, so we
            # subtract off 1 to get the current ttinfo
            idx = bisect.bisect_right(lt, ts) - 1
            assert idx >= 0
            return self._ttinfos[idx]

    def _get_local_timestamp(self, dt):
        return (
            (dt.toordinal() - EPOCHORDINAL) * 86400
            + dt.hour * 3600
            + dt.minute * 60
            + dt.second
        )

    def __str__(self):
        if self._key is not None:
            return f"{self._key}"
        else:
            return repr(self)

    def __repr__(self):
        if self._key is not None:
            return f"{self.__class__.__name__}(key={self._key!r})"
        else:
            return f"{self.__class__.__name__}.from_file({self._file_repr})"

    def __reduce__(self):
        return (self.__class__._unpickle, (self._key, self._from_cache))

    def _file_reduce(self):
        import pickle

        raise pickle.PicklingError(
            "Cannot pickle a ZoneInfo file created from a file stream."
        )

    @classmethod
    def _unpickle(cls, key, from_cache, /):
        if from_cache:
            return cls(key)
        else:
            return cls.no_cache(key)

    def _find_tzfile(self, key):
        return _tzpath.find_tzfile(key)

    def _load_file(self, fobj):
        # Retrieve all the data as it exists in the zoneinfo file
        trans_idx, trans_utc, utcoff, isdst, abbr, tz_str = _common.load_data(
            fobj
        )

        # Infer the DST offsets (needed for .dst()) from the data
        dstoff = self._utcoff_to_dstoff(trans_idx, utcoff, isdst)

        # Convert all the transition times (UTC) into "seconds since 1970-01-01 local time"
        trans_local = self._ts_to_local(trans_idx, trans_utc, utcoff)

        # Construct `_ttinfo` objects for each transition in the file
        _ttinfo_list = [
            _ttinfo(
                _load_timedelta(utcoffset), _load_timedelta(dstoffset), tzname
            )
            for utcoffset, dstoffset, tzname in zip(utcoff, dstoff, abbr)
        ]

        self._trans_utc = trans_utc
        self._trans_local = trans_local
        self._ttinfos = [_ttinfo_list[idx] for idx in trans_idx]

        # Find the first non-DST transition
        for i in range(len(isdst)):
            if not isdst[i]:
                self._tti_before = _ttinfo_list[i]
                break
        else:
            if self._ttinfos:
                self._tti_before = self._ttinfos[0]
            else:
                self._tti_before = None

        # Set the "fallback" time zone
        if tz_str is not None and tz_str != b"":
            self._tz_after = _parse_tz_str(tz_str.decode())
        else:
            if not self._ttinfos and not _ttinfo_list:
                raise ValueError("No time zone information found.")

            if self._ttinfos:
                self._tz_after = self._ttinfos[-1]
            else:
                self._tz_after = _ttinfo_list[-1]

        # Determine if this is a "fixed offset" zone, meaning that the output
        # of the utcoffset, dst and tzname functions does not depend on the
        # specific datetime passed.
        #
        # We make three simplifying assumptions here:
        #
        # 1. If _tz_after is not a _ttinfo, it has transitions that might
        #    actually occur (it is possible to construct TZ strings that
        #    specify STD and DST but no transitions ever occur, such as
        #    AAA0BBB,0/0,J365/25).
        # 2. If _ttinfo_list contains more than one _ttinfo object, the objects
        #    represent different offsets.
        # 3. _ttinfo_list contains no unused _ttinfos (in which case an
        #    otherwise fixed-offset zone with extra _ttinfos defined may
        #    appear to *not* be a fixed offset zone).
        #
        # Violations to these assumptions would be fairly exotic, and exotic
        # zones should almost certainly not be used with datetime.time (the
        # only thing that would be affected by this).
        if len(_ttinfo_list) > 1 or not isinstance(self._tz_after, _ttinfo):
            self._fixed_offset = False
        elif not _ttinfo_list:
            self._fixed_offset = True
        else:
            self._fixed_offset = _ttinfo_list[0] == self._tz_after

    @staticmethod
    def _utcoff_to_dstoff(trans_idx, utcoffsets, isdsts):
        # Now we must transform our ttis and abbrs into `_ttinfo` objects,
        # but there is an issue: .dst() must return a timedelta with the
        # difference between utcoffset() and the "standard" offset, but
        # the "base offset" and "DST offset" are not encoded in the file;
        # we can infer what they are from the isdst flag, but it is not
        # sufficient to just look at the last standard offset, because
        # occasionally countries will shift both DST offset and base offset.

        typecnt = len(isdsts)
        dstoffs = [0] * typecnt  # Provisionally assign all to 0.
        dst_cnt = sum(isdsts)
        dst_found = 0

        for i in range(1, len(trans_idx)):
            if dst_cnt == dst_found:
                break

            idx = trans_idx[i]

            dst = isdsts[idx]

            # We're only going to look at daylight saving time
            if not dst:
                continue

            # Skip any offsets that have already been assigned
            if dstoffs[idx] != 0:
                continue

            dstoff = 0
            utcoff = utcoffsets[idx]

            comp_idx = trans_idx[i - 1]

            if not isdsts[comp_idx]:
                dstoff = utcoff - utcoffsets[comp_idx]

            if not dstoff and idx < (typecnt - 1):
                comp_idx = trans_idx[i + 1]

                # If the following transition is also DST and we couldn't
                # find the DST offset by this point, we're going to have to
                # skip it and hope this transition gets assigned later
                if isdsts[comp_idx]:
                    continue

                dstoff = utcoff - utcoffsets[comp_idx]

            if dstoff:
                dst_found += 1
                dstoffs[idx] = dstoff
        else:
            # If we didn't find a valid value for a given index, we'll end up
            # with dstoff = 0 for something where `isdst=1`. This is obviously
            # wrong - one hour will be a much better guess than 0
            for idx in range(typecnt):
                if not dstoffs[idx] and isdsts[idx]:
                    dstoffs[idx] = 3600

        return dstoffs

    @staticmethod
    def _ts_to_local(trans_idx, trans_list_utc, utcoffsets):
        """Generate number of seconds since 1970 *in the local time*.

        This is necessary to easily find the transition times in local time"""
        if not trans_list_utc:
            return [[], []]

        # Start with the timestamps and modify in-place
        trans_list_wall = [list(trans_list_utc), list(trans_list_utc)]

        if len(utcoffsets) > 1:
            offset_0 = utcoffsets[0]
            offset_1 = utcoffsets[trans_idx[0]]
            if offset_1 > offset_0:
                offset_1, offset_0 = offset_0, offset_1
        else:
            offset_0 = offset_1 = utcoffsets[0]

        trans_list_wall[0][0] += offset_0
        trans_list_wall[1][0] += offset_1

        for i in range(1, len(trans_idx)):
            offset_0 = utcoffsets[trans_idx[i - 1]]
            offset_1 = utcoffsets[trans_idx[i]]

            if offset_1 > offset_0:
                offset_1, offset_0 = offset_0, offset_1

            trans_list_wall[0][i] += offset_0
            trans_list_wall[1][i] += offset_1

        return trans_list_wall


class _ttinfo:
    __slots__ = ["utcoff", "dstoff", "tzname"]

    def __init__(self, utcoff, dstoff, tzname):
        self.utcoff = utcoff
        self.dstoff = dstoff
        self.tzname = tzname

    def __eq__(self, other):
        return (
            self.utcoff == other.utcoff
            and self.dstoff == other.dstoff
            and self.tzname == other.tzname
        )

    def __repr__(self):  # pragma: nocover
        return (
            f"{self.__class__.__name__}"
            + f"({self.utcoff}, {self.dstoff}, {self.tzname})"
        )


_NO_TTINFO = _ttinfo(None, None, None)


class _TZStr:
    __slots__ = (
        "std",
        "dst",
        "start",
        "end",
        "get_trans_info",
        "get_trans_info_fromutc",
        "dst_diff",
    )

    def __init__(
        self, std_abbr, std_offset, dst_abbr, dst_offset, start=None, end=None
    ):
        self.dst_diff = dst_offset - std_offset
        std_offset = _load_timedelta(std_offset)
        self.std = _ttinfo(
            utcoff=std_offset, dstoff=_load_timedelta(0), tzname=std_abbr
        )

        self.start = start
        self.end = end

        dst_offset = _load_timedelta(dst_offset)
        delta = _load_timedelta(self.dst_diff)
        self.dst = _ttinfo(utcoff=dst_offset, dstoff=delta, tzname=dst_abbr)

        # These are assertions because the constructor should only be called
        # by functions that would fail before passing start or end
        assert start is not None, "No transition start specified"
        assert end is not None, "No transition end specified"

        self.get_trans_info = self._get_trans_info
        self.get_trans_info_fromutc = self._get_trans_info_fromutc

    def transitions(self, year):
        start = self.start.year_to_epoch(year)
        end = self.end.year_to_epoch(year)
        return start, end

    def _get_trans_info(self, ts, year, fold):
        """Get the information about the current transition - tti"""
        start, end = self.transitions(year)

        # With fold = 0, the period (denominated in local time) with the
        # smaller offset starts at the end of the gap and ends at the end of
        # the fold; with fold = 1, it runs from the start of the gap to the
        # beginning of the fold.
        #
        # So in order to determine the DST boundaries we need to know both
        # the fold and whether DST is positive or negative (rare), and it
        # turns out that this boils down to fold XOR is_positive.
        if fold == (self.dst_diff >= 0):
            end -= self.dst_diff
        else:
            start += self.dst_diff

        if start < end:
            isdst = start <= ts < end
        else:
            isdst = not (end <= ts < start)

        return self.dst if isdst else self.std

    def _get_trans_info_fromutc(self, ts, year):
        start, end = self.transitions(year)
        start -= self.std.utcoff.total_seconds()
        end -= self.dst.utcoff.total_seconds()

        if start < end:
            isdst = start <= ts < end
        else:
            isdst = not (end <= ts < start)

        # For positive DST, the ambiguous period is one dst_diff after the end
        # of DST; for negative DST, the ambiguous period is one dst_diff before
        # the start of DST.
        if self.dst_diff > 0:
            ambig_start = end
            ambig_end = end + self.dst_diff
        else:
            ambig_start = start
            ambig_end = start - self.dst_diff

        fold = ambig_start <= ts < ambig_end

        return (self.dst if isdst else self.std, fold)


def _post_epoch_days_before_year(year):
    """Get the number of days between 1970-01-01 and YEAR-01-01"""
    y = year - 1
    return y * 365 + y // 4 - y // 100 + y // 400 - EPOCHORDINAL


class _DayOffset:
    __slots__ = ["d", "julian", "hour", "minute", "second"]

    def __init__(self, d, julian, hour=2, minute=0, second=0):
        if not (0 + julian) <= d <= 365:
            min_day = 0 + julian
            raise ValueError(f"d must be in [{min_day}, 365], not: {d}")

        self.d = d
        self.julian = julian
        self.hour = hour
        self.minute = minute
        self.second = second

    def year_to_epoch(self, year):
        days_before_year = _post_epoch_days_before_year(year)

        d = self.d
        if self.julian and d >= 59 and calendar.isleap(year):
            d += 1

        epoch = (days_before_year + d) * 86400
        epoch += self.hour * 3600 + self.minute * 60 + self.second

        return epoch


class _CalendarOffset:
    __slots__ = ["m", "w", "d", "hour", "minute", "second"]

    _DAYS_BEFORE_MONTH = (
        -1,
        0,
        31,
        59,
        90,
        120,
        151,
        181,
        212,
        243,
        273,
        304,
        334,
    )

    def __init__(self, m, w, d, hour=2, minute=0, second=0):
        if not 0 < m <= 12:
            raise ValueError("m must be in (0, 12]")

        if not 0 < w <= 5:
            raise ValueError("w must be in (0, 5]")

        if not 0 <= d <= 6:
            raise ValueError("d must be in [0, 6]")

        self.m = m
        self.w = w
        self.d = d
        self.hour = hour
        self.minute = minute
        self.second = second

    @classmethod
    def _ymd2ord(cls, year, month, day):
        return (
            _post_epoch_days_before_year(year)
            + cls._DAYS_BEFORE_MONTH[month]
            + (month > 2 and calendar.isleap(year))
            + day
        )

    # TODO: These are not actually epoch dates as they are expressed in local time
    def year_to_epoch(self, year):
        """Calculates the datetime of the occurrence from the year"""
        # We know year and month, we need to convert w, d into day of month
        #
        # Week 1 is the first week in which day `d` (where 0 = Sunday) appears.
        # Week 5 represents the last occurrence of day `d`, so we need to know
        # the range of the month.
        first_day, days_in_month = calendar.monthrange(year, self.m)

        # This equation seems magical, so I'll break it down:
        # 1. calendar says 0 = Monday, POSIX says 0 = Sunday
        #    so we need first_day + 1 to get 1 = Monday -> 7 = Sunday,
        #    which is still equivalent because this math is mod 7
        # 2. Get first day - desired day mod 7: -1 % 7 = 6, so we don't need
        #    to do anything to adjust negative numbers.
        # 3. Add 1 because month days are a 1-based index.
        month_day = (self.d - (first_day + 1)) % 7 + 1

        # Now use a 0-based index version of `w` to calculate the w-th
        # occurrence of `d`
        month_day += (self.w - 1) * 7

        # month_day will only be > days_in_month if w was 5, and `w` means
        # "last occurrence of `d`", so now we just check if we over-shot the
        # end of the month and if so knock off 1 week.
        if month_day > days_in_month:
            month_day -= 7

        ordinal = self._ymd2ord(year, self.m, month_day)
        epoch = ordinal * 86400
        epoch += self.hour * 3600 + self.minute * 60 + self.second
        return epoch


def _parse_tz_str(tz_str):
    # The tz string has the format:
    #
    # std[offset[dst[offset],start[/time],end[/time]]]
    #
    # std and dst must be 3 or more characters long and must not contain
    # a leading colon, embedded digits, commas, nor a plus or minus signs;
    # The spaces between "std" and "offset" are only for display and are
    # not actually present in the string.
    #
    # The format of the offset is ``[+|-]hh[:mm[:ss]]``

    offset_str, *start_end_str = tz_str.split(",", 1)

    # fmt: off
    parser_re = re.compile(
        r"(?P<std>[^<0-9:.+-]+|<[a-zA-Z0-9+\-]+>)" +
        r"((?P<stdoff>[+-]?\d{1,2}(:\d{2}(:\d{2})?)?)" +
            r"((?P<dst>[^0-9:.+-]+|<[a-zA-Z0-9+\-]+>)" +
                r"((?P<dstoff>[+-]?\d{1,2}(:\d{2}(:\d{2})?)?))?" +
            r")?" + # dst
        r")?$" # stdoff
    )
    # fmt: on

    m = parser_re.match(offset_str)

    if m is None:
        raise ValueError(f"{tz_str} is not a valid TZ string")

    std_abbr = m.group("std")
    dst_abbr = m.group("dst")
    dst_offset = None

    std_abbr = std_abbr.strip("<>")

    if dst_abbr:
        dst_abbr = dst_abbr.strip("<>")

    if std_offset := m.group("stdoff"):
        try:
            std_offset = _parse_tz_delta(std_offset)
        except ValueError as e:
            raise ValueError(f"Invalid STD offset in {tz_str}") from e
    else:
        std_offset = 0

    if dst_abbr is not None:
        if dst_offset := m.group("dstoff"):
            try:
                dst_offset = _parse_tz_delta(dst_offset)
            except ValueError as e:
                raise ValueError(f"Invalid DST offset in {tz_str}") from e
        else:
            dst_offset = std_offset + 3600

        if not start_end_str:
            raise ValueError(f"Missing transition rules: {tz_str}")

        start_end_strs = start_end_str[0].split(",", 1)
        try:
            start, end = (_parse_dst_start_end(x) for x in start_end_strs)
        except ValueError as e:
            raise ValueError(f"Invalid TZ string: {tz_str}") from e

        return _TZStr(std_abbr, std_offset, dst_abbr, dst_offset, start, end)
    elif start_end_str:
        raise ValueError(f"Transition rule present without DST: {tz_str}")
    else:
        # This is a static ttinfo, don't return _TZStr
        return _ttinfo(
            _load_timedelta(std_offset), _load_timedelta(0), std_abbr
        )


def _parse_dst_start_end(dststr):
    date, *time = dststr.split("/")
    if date[0] == "M":
        n_is_julian = False
        m = re.match(r"M(\d{1,2})\.(\d).(\d)$", date)
        if m is None:
            raise ValueError(f"Invalid dst start/end date: {dststr}")
        date_offset = tuple(map(int, m.groups()))
        offset = _CalendarOffset(*date_offset)
    else:
        if date[0] == "J":
            n_is_julian = True
            date = date[1:]
        else:
            n_is_julian = False

        doy = int(date)
        offset = _DayOffset(doy, n_is_julian)

    if time:
        time_components = list(map(int, time[0].split(":")))
        n_components = len(time_components)
        if n_components < 3:
            time_components.extend([0] * (3 - n_components))
        offset.hour, offset.minute, offset.second = time_components

    return offset


def _parse_tz_delta(tz_delta):
    match = re.match(
        r"(?P<sign>[+-])?(?P<h>\d{1,2})(:(?P<m>\d{2})(:(?P<s>\d{2}))?)?",
        tz_delta,
    )
    # Anything passed to this function should already have hit an equivalent
    # regular expression to find the section to parse.
    assert match is not None, tz_delta

    h, m, s = (
        int(v) if v is not None else 0
        for v in map(match.group, ("h", "m", "s"))
    )

    total = h * 3600 + m * 60 + s

    if not -86400 < total < 86400:
        raise ValueError(
            f"Offset must be strictly between -24h and +24h: {tz_delta}"
        )

    # Yes, +5 maps to an offset of -5h
    if match.group("sign") != "-":
        total *= -1

    return total
