        using the format string in self.msg (or the default one).
        """
        time = self.split()
        if self.logger is None or exc_type:
            # if there's no logger attached, or if any exception occurred in
            # the with-statement, exit without logging the time
            return
        message = self.formatTime(self.msg, time)
        # Allow log handlers to see the individual parts to facilitate things
        # like a server accumulating aggregate stats.
        msg_parts = {"msg": self.msg, "time": time}
        self.logger.log(self.level, message, msg_parts)

    def __call__(self, func_or_msg=None, **kwargs):
        """If the first argument is a function, return a decorator which runs
        the wrapped function inside Timer's context manager.
        Otherwise, treat the first argument as a 'msg' string and return an updated
        Timer instance, referencing the same logger.
        A 'level' keyword can also be passed to override self.level.
        """
        if isinstance(func_or_msg, Callable):
            func = func_or_msg
            # use the function name when no explicit 'msg' is provided
            if not self.msg:
                self.msg = "run '%s'" % func.__name__

            @wraps(func)
            def wrapper(*args, **kwds):
                with self:
                    return func(*args, **kwds)

            return wrapper
        else:
            msg = func_or_msg or kwargs.get("msg")
            level = kwargs.get("level", self.level)
            return self.__class__(self.logger, msg, level)

    def __float__(self):
        return self.elapsed

    def __int__(self):
        return int(self.elapsed)

    def __str__(self):
        return "%.3f" % self.elapsed


class ChannelsFilter(logging.Filter):
    """Provides a hierarchical filter for log entries based on channel names.

    Filters out records emitted from a list of enabled channel names,
    including their children. It works the same as the ``logging.Filter``
    class, but allows the user to specify multiple channel names.

    >>> import sys
    >>> handler = logging.StreamHandler(sys.stdout)
    >>> handler.setFormatter(logging.Formatter("%(message)s"))
    >>> filter = ChannelsFilter("A.B", "C.D")
    >>> handler.addFilter(filter)
    >>> root = logging.getLogger()
    >>> root.addHandler(handler)
    >>> root.setLevel(level=logging.DEBUG)
    >>> logging.getLogger('A.B').debug('this record passes through')
    this record passes through
    >>> logging.getLogger('A.B.C').debug('records from children also pass')
    records from children also pass
    >>> logging.getLogger('C.D').debug('this one as well')
    this one as well
    >>> logging.getLogger('A.B.').debug('also this one')
    also this one
    >>> logging.getLogger('A.F').debug('but this one does not!')
    >>> logging.getLogger('C.DE').debug('neither this one!')
    """

    def __init__(self, *names):
        self.names = names
        self.num = len(names)
        self.lengths = {n: len(n) for n in names}

    def filter(self, record):
        if self.num == 0:
            return True
        for name in self.names:
            nlen = self.lengths[name]
            if name == record.name:
                return True
            elif record.name.find(name, 0, nlen) == 0 and record.name[nlen] == ".":
                return True
        return False


class CapturingLogHandler(logging.Handler):
    def __init__(self, logger, level):
        super(CapturingLogHandler, self).__init__(level=level)
        self.records = []
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger

    def __enter__(self):
        self.original_disabled = self.logger.disabled
        self.original_level = self.logger.level
        self.original_propagate = self.logger.propagate

        self.logger.addHandler(self)
        self.logger.setLevel(self.level)
        self.logger.disabled = False
        self.logger.propagate = False

        return self

    def __exit__(self, type, value, traceback):
        self.logger.removeHandler(self)
        self.logger.setLevel(self.original_level)
        self.logger.disabled = self.original_disabled
        self.logger.propagate = self.original_propagate

        return self

    def emit(self, record):
        self.records.append(record)

    def assertRegex(self, regexp, msg=None):
        import re

        pattern = re.compile(regexp)
        for r in self.records:
            if pattern.search(r.getMessage()):
                return True
        if msg is None:
            msg = "Pattern '%s' not found in logger records" % regexp
        assert 0, msg


class LogMixin(object):
    """Mixin class that adds logging functionality to another class.

    You can define a new class that subclasses from ``LogMixin`` as well as
    other base classes through multiple inheritance.
    All instances of that class will have a ``log`` property that returns
    a ``logging.Logger`` named after their respective ``<module>.<class>``.

    For example:

    >>> class BaseClass(object):
    ...     pass
    >>> class MyClass(LogMixin, BaseClass):
    ...     pass
    >>> a = MyClass()
    >>> isinstance(a.log, logging.Logger)
    True
    >>> print(a.log.name)
    fontTools.misc.loggingTools.MyClass
    >>> class AnotherClass(MyClass):
    ...     pass
    >>> b = AnotherClass()
    >>> isinstance(b.log, logging.Logger)
    True
    >>> print(b.log.name)
    fontTools.misc.loggingTools.AnotherClass
    """

    @property
    def log(self):
        if not hasattr(self, "_log"):
            name = ".".join((self.__class__.__module__, self.__class__.__name__))
            self._log = logging.getLogger(name)
        return self._log


def deprecateArgument(name, msg, category=UserWarning):
    """Raise a warning about deprecated function argument 'name'."""
    warnings.warn("%r is deprecated; %s" % (name, msg), category=category, stacklevel=3)


def deprecateFunction(msg, category=UserWarning):
    """Decorator to raise a warning when a deprecated function is called."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                "%r is deprecated; %s" % (func.__name__, msg),
                category=category,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


if __name__ == "__main__":
    import doctest

    sys.exit(doctest.testmod(optionflags=doctest.ELLIPSIS).failed)
<|endoftext|>"""psCharStrings.py -- module implementing various kinds of CharStrings:
CFF dictionary data and Type1/Type2 CharStrings.
"""

from fontTools.misc.fixedTools import (
    fixedToFloat,
    floatToFixed,
    floatToFixedToStr,
    strToFixedToFloat,
)
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging

log = logging.getLogger(__name__)


def read_operator(self, b0, data, index):
    if b0 == 12:
        op = (b0, byteord(data[index]))
        index = index + 1
    else:
        op = b0
    try:
        operator = self.operators[op]
    except KeyError:
        return None, index
    value = self.handle_operator(operator)
    return value, index


def read_byte(self, b0, data, index):
    return b0 - 139, index


def read_smallInt1(self, b0, data, index):
    b1 = byteord(data[index])
    return (b0 - 247) * 256 + b1 + 108, index + 1


def read_smallInt2(self, b0, data, index):
    b1 = byteord(data[index])
    return -(b0 - 251) * 256 - b1 - 108, index + 1


def read_shortInt(self, b0, data, index):
    (value,) = struct.unpack(">h", data[index : index + 2])
    return value, index + 2


def read_longInt(self, b0, data, index):
    (value,) = struct.unpack(">l", data[index : index + 4])
    return value, index + 4


def read_fixed1616(self, b0, data, index):
    (value,) = struct.unpack(">l", data[index : index + 4])
    return fixedToFloat(value, precisionBits=16), index + 4


def read_reserved(self, b0, data, index):
    assert NotImplementedError
    return NotImplemented, index


def read_realNumber(self, b0, data, index):
    number = ""
    while True:
        b = byteord(data[index])
        index = index + 1
        nibble0 = (b & 0xF0) >> 4
        nibble1 = b & 0x0F
        if nibble0 == 0xF:
            break
        number = number + realNibbles[nibble0]
        if nibble1 == 0xF:
            break
        number = number + realNibbles[nibble1]
    return float(number), index


t1OperandEncoding = [None] * 256
t1OperandEncoding[0:32] = (32) * [read_operator]
t1OperandEncoding[32:247] = (247 - 32) * [read_byte]
t1OperandEncoding[247:251] = (251 - 247) * [read_smallInt1]
t1OperandEncoding[251:255] = (255 - 251) * [read_smallInt2]
t1OperandEncoding[255] = read_longInt
assert len(t1OperandEncoding) == 256

t2OperandEncoding = t1OperandEncoding[:]
t2OperandEncoding[28] = read_shortInt
t2OperandEncoding[255] = read_fixed1616

cffDictOperandEncoding = t2OperandEncoding[:]
cffDictOperandEncoding[29] = read_longInt
cffDictOperandEncoding[30] = read_realNumber
cffDictOperandEncoding[255] = read_reserved


realNibbles = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ".",
    "E",
    "E-",
    None,
    "-",
]
realNibblesDict = {v: i for i, v in enumerate(realNibbles)}

maxOpStack = 193


def buildOperatorDict(operatorList):
    oper = {}
    opc = {}
    for item in operatorList:
        if len(item) == 2:
            oper[item[0]] = item[1]
        else:
            oper[item[0]] = item[1:]
        if isinstance(item[0], tuple):
            opc[item[1]] = item[0]
        else:
            opc[item[1]] = (item[0],)
    return oper, opc


t2Operators = [
    # 	opcode		name
    (1, "hstem"),
    (3, "vstem"),
    (4, "vmoveto"),
    (5, "rlineto"),
    (6, "hlineto"),
    (7, "vlineto"),
    (8, "rrcurveto"),
    (10, "callsubr"),
    (11, "return"),
    (14, "endchar"),
    (15, "vsindex"),
    (16, "blend"),
    (18, "hstemhm"),
    (19, "hintmask"),
    (20, "cntrmask"),
    (21, "rmoveto"),
    (22, "hmoveto"),
    (23, "vstemhm"),
    (24, "rcurveline"),
    (25, "rlinecurve"),
    (26, "vvcurveto"),
    (27, "hhcurveto"),
    # 	(28,		'shortint'),  # not really an operator
    (29, "callgsubr"),
    (30, "vhcurveto"),
    (31, "hvcurveto"),
    ((12, 0), "ignore"),  # dotsection. Yes, there a few very early OTF/CFF
    # fonts with this deprecated operator. Just ignore it.
    ((12, 3), "and"),
    ((12, 4), "or"),
    ((12, 5), "not"),
    ((12, 8), "store"),
    ((12, 9), "abs"),
    ((12, 10), "add"),
    ((12, 11), "sub"),
    ((12, 12), "div"),
    ((12, 13), "load"),
    ((12, 14), "neg"),
    ((12, 15), "eq"),
    ((12, 18), "drop"),
    ((12, 20), "put"),
    ((12, 21), "get"),
    ((12, 22), "ifelse"),
    ((12, 23), "random"),
    ((12, 24), "mul"),
    ((12, 26), "sqrt"),
    ((12, 27), "dup"),
    ((12, 28), "exch"),
    ((12, 29), "index"),
    ((12, 30), "roll"),
    ((12, 34), "hflex"),
    ((12, 35), "flex"),
    ((12, 36), "hflex1"),
    ((12, 37), "flex1"),
]


def getIntEncoder(format):
    if format == "cff":
        twoByteOp = bytechr(28)
        fourByteOp = bytechr(29)
    elif format == "t1":
        twoByteOp = None
        fourByteOp = bytechr(255)
    else:
        assert format == "t2"
        twoByteOp = bytechr(28)
        fourByteOp = None

    def encodeInt(
        value,
        fourByteOp=fourByteOp,
        bytechr=bytechr,
        pack=struct.pack,
        unpack=struct.unpack,
        twoByteOp=twoByteOp,
    ):
        if -107 <= value <= 107:
            code = bytechr(value + 139)
        elif 108 <= value <= 1131:
            value = value - 108
            code = bytechr((value >> 8) + 247) + bytechr(value & 0xFF)
        elif -1131 <= value <= -108:
            value = -value - 108
            code = bytechr((value >> 8) + 251) + bytechr(value & 0xFF)
        elif twoByteOp is not None and -32768 <= value <= 32767:
            code = twoByteOp + pack(">h", value)
        elif fourByteOp is None:
            # Backwards compatible hack: due to a previous bug in FontTools,
            # 16.16 fixed numbers were written out as 4-byte ints. When
            # these numbers were small, they were wrongly written back as
            # small ints instead of 4-byte ints, breaking round-tripping.
            # This here workaround doesn't do it any better, since we can't
            # distinguish anymore between small ints that were supposed to
            # be small fixed numbers and small ints that were just small
            # ints. Hence the warning.
            log.warning(
                "4-byte T2 number got passed to the "
                "IntType handler. This should happen only when reading in "
                "old XML files.\n"
            )
            code = bytechr(255) + pack(">l", value)
        else:
            code = fourByteOp + pack(">l", value)
        return code

    return encodeInt


encodeIntCFF = getIntEncoder("cff")
encodeIntT1 = getIntEncoder("t1")
encodeIntT2 = getIntEncoder("t2")


def encodeFixed(f, pack=struct.pack):
    """For T2 only"""
    value = floatToFixed(f, precisionBits=16)
    if value & 0xFFFF == 0:  # check if the fractional part is zero
        return encodeIntT2(value >> 16)  # encode only the integer part
    else:
        return b"\xff" + pack(">l", value)  # encode the entire fixed point value


realZeroBytes = bytechr(30) + bytechr(0xF)


def encodeFloat(f):
    # For CFF only, used in cffLib
    if f == 0.0:  # 0.0 == +0.0 == -0.0
        return realZeroBytes
    # Note: 14 decimal digits seems to be the limitation for CFF real numbers
    # in macOS. However, we use 8 here to match the implementation of AFDKO.
    s = "%.8G" % f
    if s[:2] == "0.":
        s = s[1:]
    elif s[:3] == "-0.":
        s = "-" + s[2:]
    elif s.endswith("000"):
        significantDigits = s.rstrip("0")
        s = "%sE%d" % (significantDigits, len(s) - len(significantDigits))
    else:
        dotIndex = s.find(".")
        eIndex = s.find("E")
        if dotIndex != -1 and eIndex != -1:
            integerPart = s[:dotIndex]
            fractionalPart = s[dotIndex + 1 : eIndex]
            exponent = int(s[eIndex + 1 :])
            newExponent = exponent - len(fractionalPart)
            if newExponent == 1:
                s = "%s%s0" % (integerPart, fractionalPart)
            else:
                s = "%s%sE%d" % (integerPart, fractionalPart, newExponent)
    if s.startswith((".0", "-.0")):
        sign, s = s.split(".", 1)
        s = "%s%sE-%d" % (sign, s.lstrip("0"), len(s))
    nibbles = []
    while s:
        c = s[0]
        s = s[1:]
        if c == "E":
            c2 = s[:1]
            if c2 == "-":
                s = s[1:]
                c = "E-"
            elif c2 == "+":
                s = s[1:]
            if s.startswith("0"):
                s = s[1:]
        nibbles.append(realNibblesDict[c])
    nibbles.append(0xF)
    if len(nibbles) % 2:
        nibbles.append(0xF)
    d = bytechr(30)
    for i in range(0, len(nibbles), 2):
        d = d + bytechr(nibbles[i] << 4 | nibbles[i + 1])
    return d


class CharStringCompileError(Exception):
    pass


class SimpleT2Decompiler(object):
    def __init__(self, localSubrs, globalSubrs, private=None, blender=None):
        self.localSubrs = localSubrs
        self.localBias = calcSubrBias(localSubrs)
        self.globalSubrs = globalSubrs
        self.globalBias = calcSubrBias(globalSubrs)
        self.private = private
        self.blender = blender
        self.reset()

    def reset(self):
        self.callingStack = []
        self.operandStack = []
        self.hintCount = 0
        self.hintMaskBytes = 0
        self.numRegions = 0
        self.vsIndex = 0

    def execute(self, charString, *, pushToStack=None):
        self.callingStack.append(charString)
        needsDecompilation = charString.needsDecompilation()
        if needsDecompilation:
            program = []
            pushToProgram = program.append
        else:
            pushToProgram = lambda x: None
        if pushToStack is None:
            pushToStack = self.operandStack.append
        index = 0
        while True:
            token, isOperator, index = charString.getToken(index)
            if token is None:
                break  # we're done!
            pushToProgram(token)
            if isOperator:
                handlerName = "op_" + token
                handler = getattr(self, handlerName, None)
                if handler is not None:
                    rv = handler(index)
                    if rv:
                        hintMaskBytes, index = rv
                        pushToProgram(hintMaskBytes)
                else:
                    self.popall()
            else:
                pushToStack(token)
        if needsDecompilation:
            charString.setProgram(program)
        del self.callingStack[-1]

    def pop(self):
        value = self.operandStack[-1]
        del self.operandStack[-1]
        return value

    def popall(self):
        stack = self.operandStack[:]
        self.operandStack[:] = []
        return stack

    def push(self, value):
        self.operandStack.append(value)

    def op_return(self, index):
        if self.operandStack:
            pass

    def op_endchar(self, index):
        pass

    def op_ignore(self, index):
        pass

    def op_callsubr(self, index):
        subrIndex = self.pop()
        subr = self.localSubrs[subrIndex + self.localBias]
        self.execute(subr)

    def op_callgsubr(self, index):
        subrIndex = self.pop()
        subr = self.globalSubrs[subrIndex + self.globalBias]
        self.execute(subr)

    def op_hstem(self, index):
        self.countHints()

    def op_vstem(self, index):
        self.countHints()

    def op_hstemhm(self, index):
        self.countHints()

    def op_vstemhm(self, index):
        self.countHints()

    def op_hintmask(self, index):
        if not self.hintMaskBytes:
            self.countHints()
            self.hintMaskBytes = (self.hintCount + 7) // 8
        hintMaskBytes, index = self.callingStack[-1].getBytes(index, self.hintMaskBytes)
        return hintMaskBytes, index

    op_cntrmask = op_hintmask

    def countHints(self):
        args = self.popall()
        self.hintCount = self.hintCount + len(args) // 2

    # misc
    def op_and(self, index):
        raise NotImplementedError

    def op_or(self, index):
        raise NotImplementedError

    def op_not(self, index):
        raise NotImplementedError

    def op_store(self, index):
        raise NotImplementedError

    def op_abs(self, index):
        raise NotImplementedError

    def op_add(self, index):
        raise NotImplementedError

    def op_sub(self, index):
        raise NotImplementedError

    def op_div(self, index):
        raise NotImplementedError

    def op_load(self, index):
        raise NotImplementedError

    def op_neg(self, index):
        raise NotImplementedError

    def op_eq(self, index):
        raise NotImplementedError

    def op_drop(self, index):
        raise NotImplementedError

    def op_put(self, index):
        raise NotImplementedError

    def op_get(self, index):
        raise NotImplementedError

    def op_ifelse(self, index):
        raise NotImplementedError

    def op_random(self, index):
        raise NotImplementedError

    def op_mul(self, index):
        raise NotImplementedError

    def op_sqrt(self, index):
        raise NotImplementedError

    def op_dup(self, index):
        raise NotImplementedError

    def op_exch(self, index):
        raise NotImplementedError

    def op_index(self, index):
        raise NotImplementedError

    def op_roll(self, index):
        raise NotImplementedError

    def op_blend(self, index):
        if self.numRegions == 0:
            self.numRegions = self.private.getNumRegions()
        numBlends = self.pop()
        numOps = numBlends * (self.numRegions + 1)
        if self.blender is None:
            del self.operandStack[
                -(numOps - numBlends) :
            ]  # Leave the default operands on the stack.
        else:
            argi = len(self.operandStack) - numOps
            end_args = tuplei = argi + numBlends
            while argi < end_args:
                next_ti = tuplei + self.numRegions
                deltas = self.operandStack[tuplei:next_ti]
                delta = self.blender(self.vsIndex, deltas)
                self.operandStack[argi] += delta
                tuplei = next_ti
                argi += 1
            self.operandStack[end_args:] = []

    def op_vsindex(self, index):
        vi = self.pop()
        self.vsIndex = vi
        self.numRegions = self.private.getNumRegions(vi)


t1Operators = [
    # 	opcode		name
    (1, "hstem"),
    (3, "vstem"),
    (4, "vmoveto"),
    (5, "rlineto"),
    (6, "hlineto"),
    (7, "vlineto"),
    (8, "rrcurveto"),
    (9, "closepath"),
    (10, "callsubr"),
    (11, "return"),
    (13, "hsbw"),
    (14, "endchar"),
    (21, "rmoveto"),
    (22, "hmoveto"),
    (30, "vhcurveto"),
    (31, "hvcurveto"),
    ((12, 0), "dotsection"),
    ((12, 1), "vstem3"),
    ((12, 2), "hstem3"),
    ((12, 6), "seac"),
    ((12, 7), "sbw"),
    ((12, 12), "div"),
    ((12, 16), "callothersubr"),
    ((12, 17), "pop"),
    ((12, 33), "setcurrentpoint"),
]


class T2StackUseExtractor(SimpleT2Decompiler):

    def execute(self, charString):
        maxStackUse = 0

        def pushToStack(value):
            nonlocal maxStackUse
            self.operandStack.append(value)
            maxStackUse = max(maxStackUse, len(self.operandStack))

        super().execute(charString, pushToStack=pushToStack)
        return maxStackUse


class T2WidthExtractor(SimpleT2Decompiler):
    def __init__(
        self,
        localSubrs,
        globalSubrs,
        nominalWidthX,
        defaultWidthX,
        private=None,
        blender=None,
    ):
        SimpleT2Decompiler.__init__(self, localSubrs, globalSubrs, private, blender)
        self.nominalWidthX = nominalWidthX
        self.defaultWidthX = defaultWidthX

    def reset(self):
        SimpleT2Decompiler.reset(self)
        self.gotWidth = 0
        self.width = 0

    def popallWidth(self, evenOdd=0):
        args = self.popall()
        if not self.gotWidth:
            if evenOdd ^ (len(args) % 2):
                # For CFF2 charstrings, this should never happen
                assert (
                    self.defaultWidthX is not None
                ), "CFF2 CharStrings must not have an initial width value"
                self.width = self.nominalWidthX + args[0]
                args = args[1:]
            else:
                self.width = self.defaultWidthX
            self.gotWidth = 1
        return args

    def countHints(self):
        args = self.popallWidth()
        self.hintCount = self.hintCount + len(args) // 2

    def op_rmoveto(self, index):
        self.popallWidth()

    def op_hmoveto(self, index):
        self.popallWidth(1)

    def op_vmoveto(self, index):
        self.popallWidth(1)

    def op_endchar(self, index):
        self.popallWidth()


class T2OutlineExtractor(T2WidthExtractor):
    def __init__(
        self,
        pen,
        localSubrs,
        globalSubrs,
        nominalWidthX,
        defaultWidthX,
        private=None,
        blender=None,
    ):
        T2WidthExtractor.__init__(
            self,
            localSubrs,
            globalSubrs,
            nominalWidthX,
            defaultWidthX,
            private,
            blender,
        )
        self.pen = pen
        self.subrLevel = 0

    def reset(self):
        T2WidthExtractor.reset(self)
        self.currentPoint = (0, 0)
        self.sawMoveTo = 0
        self.subrLevel = 0

    def execute(self, charString):
        self.subrLevel += 1
        super().execute(charString)
        self.subrLevel -= 1
        if self.subrLevel == 0:
            self.endPath()

    def _nextPoint(self, point):
        x, y = self.currentPoint
        point = x + point[0], y + point[1]
        self.currentPoint = point
        return point

    def rMoveTo(self, point):
        self.pen.moveTo(self._nextPoint(point))
        self.sawMoveTo = 1

    def rLineTo(self, point):
        if not self.sawMoveTo:
            self.rMoveTo((0, 0))
        self.pen.lineTo(self._nextPoint(point))

    def rCurveTo(self, pt1, pt2, pt3):
        if not self.sawMoveTo:
            self.rMoveTo((0, 0))
        nextPoint = self._nextPoint
        self.pen.curveTo(nextPoint(pt1), nextPoint(pt2), nextPoint(pt3))

    def closePath(self):
        if self.sawMoveTo:
            self.pen.closePath()
        self.sawMoveTo = 0

    def endPath(self):
        # In T2 there are no open paths, so always do a closePath when
        # finishing a sub path. We avoid spurious calls to closePath()
        # because its a real T1 op we're emulating in T2 whereas
        # endPath() is just a means to that emulation
        if self.sawMoveTo:
            self.closePath()

    #
    # hint operators
    #
    # def op_hstem(self, index):
    # 	self.countHints()
    # def op_vstem(self, index):
    # 	self.countHints()
    # def op_hstemhm(self, index):
    # 	self.countHints()
    # def op_vstemhm(self, index):
    # 	self.countHints()
    # def op_hintmask(self, index):
    # 	self.countHints()
    # def op_cntrmask(self, index):
    # 	self.countHints()

    #
    # path constructors, moveto
    #
    def op_rmoveto(self, index):
        self.endPath()
        self.rMoveTo(self.popallWidth())

    def op_hmoveto(self, index):
        self.endPath()
        self.rMoveTo((self.popallWidth(1)[0], 0))

    def op_vmoveto(self, index):
        self.endPath()
        self.rMoveTo((0, self.popallWidth(1)[0]))

    def op_endchar(self, index):
        self.endPath()
        args = self.popallWidth()
        if args:
            from fontTools.encodings.StandardEncoding import StandardEncoding

            # endchar can do seac accent bulding; The T2 spec says it's deprecated,
            # but recent software that shall remain nameless does output it.
            adx, ady, bchar, achar = args
            baseGlyph = StandardEncoding[bchar]
            self.pen.addComponent(baseGlyph, (1, 0, 0, 1, 0, 0))
            accentGlyph = StandardEncoding[achar]
            self.pen.addComponent(accentGlyph, (1, 0, 0, 1, adx, ady))

    #
    # path constructors, lines
    #
    def op_rlineto(self, index):
        args = self.popall()
        for i in range(0, len(args), 2):
            point = args[i : i + 2]
            self.rLineTo(point)

    def op_hlineto(self, index):
        self.alternatingLineto(1)

    def op_vlineto(self, index):
        self.alternatingLineto(0)

    #
    # path constructors, curves
    #
    def op_rrcurveto(self, index):
        """{dxa dya dxb dyb dxc dyc}+ rrcurveto"""
        args = self.popall()
        for i in range(0, len(args), 6):
            (
                dxa,
                dya,
                dxb,
                dyb,
                dxc,
                dyc,
            ) = args[i : i + 6]
            self.rCurveTo((dxa, dya), (dxb, dyb), (dxc, dyc))

    def op_rcurveline(self, index):
        """{dxa dya dxb dyb dxc dyc}+ dxd dyd rcurveline"""
        args = self.popall()
        for i in range(0, len(args) - 2, 6):
            dxb, dyb, dxc, dyc, dxd, dyd = args[i : i + 6]
            self.rCurveTo((dxb, dyb), (dxc, dyc), (dxd, dyd))
        self.rLineTo(args[-2:])

    def op_rlinecurve(self, index):
        """{dxa dya}+ dxb dyb dxc dyc dxd dyd rlinecurve"""
        args = self.popall()
        lineArgs = args[:-6]
        for i in range(0, len(lineArgs), 2):
            self.rLineTo(lineArgs[i : i + 2])
        dxb, dyb, dxc, dyc, dxd, dyd = args[-6:]
        self.rCurveTo((dxb, dyb), (dxc, dyc), (dxd, dyd))

    def op_vvcurveto(self, index):
        "dx1? {dya dxb dyb dyc}+ vvcurveto"
        args = self.popall()
        if len(args) % 2:
            dx1 = args[0]
            args = args[1:]
        else:
            dx1 = 0
        for i in range(0, len(args), 4):
            dya, dxb, dyb, dyc = args[i : i + 4]
            self.rCurveTo((dx1, dya), (dxb, dyb), (0, dyc))
            dx1 = 0

    def op_hhcurveto(self, index):
        """dy1? {dxa dxb dyb dxc}+ hhcurveto"""
        args = self.popall()
        if len(args) % 2:
            dy1 = args[0]
            args = args[1:]
        else:
            dy1 = 0
        for i in range(0, len(args), 4):
            dxa, dxb, dyb, dxc = args[i : i + 4]
            self.rCurveTo((dxa, dy1), (dxb, dyb), (dxc, 0))
            dy1 = 0

    def op_vhcurveto(self, index):
        """dy1 dx2 dy2 dx3 {dxa dxb dyb dyc dyd dxe dye dxf}* dyf? vhcurveto (30)
        {dya dxb dyb dxc dxd dxe dye dyf}+ dxf? vhcurveto
        """
        args = self.popall()
        while args:
            args = self.vcurveto(args)
            if args:
                args = self.hcurveto(args)

    def op_hvcurveto(self, index):
        """dx1 dx2 dy2 dy3 {dya dxb dyb dxc dxd dxe dye dyf}* dxf?
        {dxa dxb dyb dyc dyd dxe dye dxf}+ dyf?
        """
        args = self.popall()
        while args:
            args = self.hcurveto(args)
            if args:
                args = self.vcurveto(args)

    #
    # path constructors, flex
    #
    def op_hflex(self, index):
        dx1, dx2, dy2, dx3, dx4, dx5, dx6 = self.popall()
        dy1 = dy3 = dy4 = dy6 = 0
        dy5 = -dy2
        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    def op_flex(self, index):
        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, dx6, dy6, fd = self.popall()
        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    def op_hflex1(self, index):
        dx1, dy1, dx2, dy2, dx3, dx4, dx5, dy5, dx6 = self.popall()
        dy3 = dy4 = 0
        dy6 = -(dy1 + dy2 + dy3 + dy4 + dy5)

        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    def op_flex1(self, index):
        dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, d6 = self.popall()
        dx = dx1 + dx2 + dx3 + dx4 + dx5
        dy = dy1 + dy2 + dy3 + dy4 + dy5
        if abs(dx) > abs(dy):
            dx6 = d6
            dy6 = -dy
        else:
            dx6 = -dx
            dy6 = d6
        self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
        self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))

    # misc
    def op_and(self, index):
        raise NotImplementedError

    def op_or(self, index):
        raise NotImplementedError

    def op_not(self, index):
        raise NotImplementedError

    def op_store(self, index):
        raise NotImplementedError

    def op_abs(self, index):
        raise NotImplementedError

    def op_add(self, index):
        raise NotImplementedError

    def op_sub(self, index):
        raise NotImplementedError

    def op_div(self, index):
        num2 = self.pop()
        num1 = self.pop()
        d1 = num1 // num2
        d2 = num1 / num2
        if d1 == d2:
            self.push(d1)
        else:
            self.push(d2)

    def op_load(self, index):
        raise NotImplementedError

    def op_neg(self, index):
        raise NotImplementedError

    def op_eq(self, index):
        raise NotImplementedError

    def op_drop(self, index):
        raise NotImplementedError

    def op_put(self, index):
        raise NotImplementedError

    def op_get(self, index):
        raise NotImplementedError

    def op_ifelse(self, index):
        raise NotImplementedError

    def op_random(self, index):
        raise NotImplementedError

    def op_mul(self, index):
        raise NotImplementedError

    def op_sqrt(self, index):
        raise NotImplementedError

    def op_dup(self, index):
        raise NotImplementedError

    def op_exch(self, index):
        raise NotImplementedError

    def op_index(self, index):
        raise NotImplementedError

    def op_roll(self, index):
        raise NotImplementedError

    #
    # miscellaneous helpers
    #
    def alternatingLineto(self, isHorizontal):
        args = self.popall()
        for arg in args:
            if isHorizontal:
                point = (arg, 0)
            else:
                point = (0, arg)
            self.rLineTo(point)
            isHorizontal = not isHorizontal

    def vcurveto(self, args):
        dya, dxb, dyb, dxc = args[:4]
        args = args[4:]
        if len(args) == 1:
            dyc = args[0]
            args = []
        else:
            dyc = 0
        self.rCurveTo((0, dya), (dxb, dyb), (dxc, dyc))
        return args

    def hcurveto(self, args):
        dxa, dxb, dyb, dyc = args[:4]
        args = args[4:]
        if len(args) == 1:
            dxc = args[0]
            args = []
        else:
            dxc = 0
        self.rCurveTo((dxa, 0), (dxb, dyb), (dxc, dyc))
        return args


class T1OutlineExtractor(T2OutlineExtractor):
    def __init__(self, pen, subrs):
        self.pen = pen
        self.subrs = subrs
        self.reset()

    def reset(self):
        self.flexing = 0
        self.width = 0
        self.sbx = 0
        T2OutlineExtractor.reset(self)

    def endPath(self):
        if self.sawMoveTo:
            self.pen.endPath()
        self.sawMoveTo = 0

    def popallWidth(self, evenOdd=0):
        return self.popall()

    def exch(self):
        stack = self.operandStack
        stack[-1], stack[-2] = stack[-2], stack[-1]

    #
    # path constructors
    #
    def op_rmoveto(self, index):
        if self.flexing:
            return
        self.endPath()
        self.rMoveTo(self.popall())

    def op_hmoveto(self, index):
        if self.flexing:
            # We must add a parameter to the stack if we are flexing
            self.push(0)
            return
        self.endPath()
        self.rMoveTo((self.popall()[0], 0))

    def op_vmoveto(self, index):
        if self.flexing:
            # We must add a parameter to the stack if we are flexing
            self.push(0)
            self.exch()
            return
        self.endPath()
        self.rMoveTo((0, self.popall()[0]))

    def op_closepath(self, index):
        self.closePath()

    def op_setcurrentpoint(self, index):
        args = self.popall()
        x, y = args
        self.currentPoint = x, y

    def op_endchar(self, index):
        self.endPath()

    def op_hsbw(self, index):
        sbx, wx = self.popall()
        self.width = wx
        self.sbx = sbx
        self.currentPoint = sbx, self.currentPoint[1]

    def op_sbw(self, index):
        self.popall()  # XXX

    #
    def op_callsubr(self, index):
        subrIndex = self.pop()
        subr = self.subrs[subrIndex]
        self.execute(subr)

    def op_callothersubr(self, index):
        subrIndex = self.pop()
        nArgs = self.pop()
        # print nArgs, subrIndex, "callothersubr"
        if subrIndex == 0 and nArgs == 3:
            self.doFlex()
            self.flexing = 0
        elif subrIndex == 1 and nArgs == 0:
            self.flexing = 1
        # ignore...

    def op_pop(self, index):
        pass  # ignore...

    def doFlex(self):
        finaly = self.pop()
        finalx = self.pop()
        self.pop()  # flex height is unused

        p3y = self.pop()
        p3x = self.pop()
        bcp4y = self.pop()
        bcp4x = self.pop()
        bcp3y = self.pop()
        bcp3x = self.pop()
        p2y = self.pop()
        p2x = self.pop()
        bcp2y = self.pop()
        bcp2x = self.pop()
        bcp1y = self.pop()
        bcp1x = self.pop()
        rpy = self.pop()
        rpx = self.pop()

        # call rrcurveto
        self.push(bcp1x + rpx)
        self.push(bcp1y + rpy)
        self.push(bcp2x)
        self.push(bcp2y)
        self.push(p2x)
        self.push(p2y)
        self.op_rrcurveto(None)

        # call rrcurveto
        self.push(bcp3x)
        self.push(bcp3y)
        self.push(bcp4x)
        self.push(bcp4y)
        self.push(p3x)
        self.push(p3y)
        self.op_rrcurveto(None)

        # Push back final coords so subr 0 can find them
        self.push(finalx)
        self.push(finaly)

    def op_dotsection(self, index):
        self.popall()  # XXX

    def op_hstem3(self, index):
        self.popall()  # XXX

    def op_seac(self, index):
        "asb adx ady bchar achar seac"
        from fontTools.encodings.StandardEncoding import StandardEncoding

        asb, adx, ady, bchar, achar = self.popall()
        baseGlyph = StandardEncoding[bchar]
        self.pen.addComponent(baseGlyph, (1, 0, 0, 1, 0, 0))
        accentGlyph = StandardEncoding[achar]
        adx = adx + self.sbx - asb  # seac weirdness
        self.pen.addComponent(accentGlyph, (1, 0, 0, 1, adx, ady))

    def op_vstem3(self, index):
        self.popall()  # XXX


class T2CharString(object):
    operandEncoding = t2OperandEncoding
    operators, opcodes = buildOperatorDict(t2Operators)
    decompilerClass = SimpleT2Decompiler
    outlineExtractor = T2OutlineExtractor

    def __init__(self, bytecode=None, program=None, private=None, globalSubrs=None):
        if program is None:
            program = []
        self.bytecode = bytecode
        self.program = program
        self.private = private
        self.globalSubrs = globalSubrs if globalSubrs is not None else []
        self._cur_vsindex = None

    def getNumRegions(self, vsindex=None):
        pd = self.private
        assert pd is not None
        if vsindex is not None:
            self._cur_vsindex = vsindex
        elif self._cur_vsindex is None:
            self._cur_vsindex = pd.vsindex if hasattr(pd, "vsindex") else 0
        return pd.getNumRegions(self._cur_vsindex)

    def __repr__(self):
        if self.bytecode is None:
            return "<%s (source) at %x>" % (self.__class__.__name__, id(self))
        else:
            return "<%s (bytecode) at %x>" % (self.__class__.__name__, id(self))

    def getIntEncoder(self):
        return encodeIntT2

    def getFixedEncoder(self):
        return encodeFixed

    def decompile(self):
        if not self.needsDecompilation():
            return
        subrs = getattr(self.private, "Subrs", [])
        decompiler = self.decompilerClass(subrs, self.globalSubrs, self.private)
        decompiler.execute(self)

    def draw(self, pen, blender=None):
        subrs = getattr(self.private, "Subrs", [])
        extractor = self.outlineExtractor(
            pen,
            subrs,
            self.globalSubrs,
            self.private.nominalWidthX,
            self.private.defaultWidthX,
            self.private,
            blender,
        )
        extractor.execute(self)
        self.width = extractor.width

    def calcBounds(self, glyphSet):
        boundsPen = BoundsPen(glyphSet)
        self.draw(boundsPen)
        return boundsPen.bounds

    def compile(self, isCFF2=False):
        if self.bytecode is not None:
            return
        opcodes = self.opcodes
        program = self.program

        if isCFF2:
            # If present, remove return and endchar operators.
            if program and program[-1] in ("return", "endchar"):
                program = program[:-1]
        elif program and not isinstance(program[-1], str):
            raise CharStringCompileError(
                "T2CharString or Subr has items on the stack after last operator."
            )

        bytecode = []
        encodeInt = self.getIntEncoder()
        encodeFixed = self.getFixedEncoder()
        i = 0
        end = len(program)
        while i < end:
            token = program[i]
            i = i + 1
            if isinstance(token, str):
                try:
                    bytecode.extend(bytechr(b) for b in opcodes[token])
                except KeyError:
                    raise CharStringCompileError("illegal operator: %s" % token)
                if token in ("hintmask", "cntrmask"):
                    bytecode.append(program[i])  # hint mask
                    i = i + 1
            elif isinstance(token, int):
                bytecode.append(encodeInt(token))
            elif isinstance(token, float):
                bytecode.append(encodeFixed(token))
            else:
                assert 0, "unsupported type: %s" % type(token)
        try:
            bytecode = bytesjoin(bytecode)
        except TypeError:
            log.error(bytecode)
            raise
        self.setBytecode(bytecode)

    def needsDecompilation(self):
        return self.bytecode is not None

    def setProgram(self, program):
        self.program = program
        self.bytecode = None

    def setBytecode(self, bytecode):
        self.bytecode = bytecode
        self.program = None

    def getToken(self, index, len=len, byteord=byteord, isinstance=isinstance):
        if self.bytecode is not None:
            if index >= len(self.bytecode):
                return None, 0, 0
            b0 = byteord(self.bytecode[index])
            index = index + 1
            handler = self.operandEncoding[b0]
            token, index = handler(self, b0, self.bytecode, index)
        else:
            if index >= len(self.program):
                return None, 0, 0
            token = self.program[index]
            index = index + 1
        isOperator = isinstance(token, str)
        return token, isOperator, index

    def getBytes(self, index, nBytes):
        if self.bytecode is not None:
            newIndex = index + nBytes
            bytes = self.bytecode[index:newIndex]
            index = newIndex
        else:
            bytes = self.program[index]
            index = index + 1
        assert len(bytes) == nBytes
        return bytes, index

    def handle_operator(self, operator):
        return operator

    def toXML(self, xmlWriter, ttFont=None):
        from fontTools.misc.textTools import num2binary

        if self.bytecode is not None:
            xmlWriter.dumphex(self.bytecode)
        else:
            index = 0
            args = []
            while True:
                token, isOperator, index = self.getToken(index)
                if token is None:
                    break
                if isOperator:
                    if token in ("hintmask", "cntrmask"):
                        hintMask, isOperator, index = self.getToken(index)
                        bits = []
                        for byte in hintMask:
                            bits.append(num2binary(byteord(byte), 8))
                        hintMask = strjoin(bits)
                        line = " ".join(args + [token, hintMask])
                    else:
                        line = " ".join(args + [token])
                    xmlWriter.write(line)
                    xmlWriter.newline()
                    args = []
                else:
                    if isinstance(token, float):
                        token = floatToFixedToStr(token, precisionBits=16)
                    else:
                        token = str(token)
                    args.append(token)
            if args:
                # NOTE: only CFF2 charstrings/subrs can have numeric arguments on
                # the stack after the last operator. Compiling this would fail if
                # this is part of CFF 1.0 table.
                line = " ".join(args)
                xmlWriter.write(line)

    def fromXML(self, name, attrs, content):
        from fontTools.misc.textTools import binary2num, readHex

        if attrs.get("raw"):
            self.setBytecode(readHex(content))
            return
        content = strjoin(content)
        content = content.split()
        program = []
        end = len(content)
        i = 0
        while i < end:
            token = content[i]
            i = i + 1
            try:
                token = int(token)
            except ValueError:
                try:
                    token = strToFixedToFloat(token, precisionBits=16)
                except ValueError:
                    program.append(token)
                    if token in ("hintmask", "cntrmask"):
                        mask = content[i]
                        maskBytes = b""
                        for j in range(0, len(mask), 8):
                            maskBytes = maskBytes + bytechr(binary2num(mask[j : j + 8]))
                        program.append(maskBytes)
                        i = i + 1
                else:
                    program.append(token)
            else:
                program.append(token)
        self.setProgram(program)


class T1CharString(T2CharString):
    operandEncoding = t1OperandEncoding
    operators, opcodes = buildOperatorDict(t1Operators)

    def __init__(self, bytecode=None, program=None, subrs=None):
        super().__init__(bytecode, program)
        self.subrs = subrs

    def getIntEncoder(self):
        return encodeIntT1

    def getFixedEncoder(self):
        def encodeFixed(value):
            raise TypeError("Type 1 charstrings don't support floating point operands")

    def decompile(self):
        if self.bytecode is None:
            return
        program = []
        index = 0
        while True:
            token, isOperator, index = self.getToken(index)
            if token is None:
                break
            program.append(token)
        self.setProgram(program)

    def draw(self, pen):
        extractor = T1OutlineExtractor(pen, self.subrs)
        extractor.execute(self)
        self.width = extractor.width


class DictDecompiler(object):
    operandEncoding = cffDictOperandEncoding

    def __init__(self, strings, parent=None):
        self.stack = []
        self.strings = strings
        self.dict = {}
        self.parent = parent

    def getDict(self):
        assert len(self.stack) == 0, "non-empty stack"
        return self.dict

    def decompile(self, data):
        index = 0
        lenData = len(data)
        push = self.stack.append
        while index < lenData:
            b0 = byteord(data[index])
            index = index + 1
            handler = self.operandEncoding[b0]
            value, index = handler(self, b0, data, index)
            if value is not None:
                push(value)

    def pop(self):
        value = self.stack[-1]
        del self.stack[-1]
        return value

    def popall(self):
        args = self.stack[:]
        del self.stack[:]
        return args

    def handle_operator(self, operator):
        operator, argType = operator
        if isinstance(argType, tuple):
            value = ()
            for i in range(len(argType) - 1, -1, -1):
                arg = argType[i]
                arghandler = getattr(self, "arg_" + arg)
                value = (arghandler(operator),) + value
        else:
            arghandler = getattr(self, "arg_" + argType)
            value = arghandler(operator)
        if operator == "blend":
            self.stack.extend(value)
        else:
            self.dict[operator] = value

    def arg_number(self, name):
        if isinstance(self.stack[0], list):
            out = self.arg_blend_number(self.stack)
        else:
            out = self.pop()
        return out

    def arg_blend_number(self, name):
        out = []
        blendArgs = self.pop()
        numMasters = len(blendArgs)
        out.append(blendArgs)
        out.append("blend")
        dummy = self.popall()
        return blendArgs

    def arg_SID(self, name):
        return self.strings[self.pop()]

    def arg_array(self, name):
        return self.popall()

    def arg_blendList(self, name):
        """
        There may be non-blend args at the top of the stack. We first calculate
        where the blend args start in the stack. These are the last
        numMasters*numBlends) +1 args.
        The blend args starts with numMasters relative coordinate values, the  BlueValues in the list from the default master font. This is followed by
        numBlends list of values. Each of  value in one of these lists is the
        Variable Font delta for the matching region.

        We re-arrange this to be a list of numMaster entries. Each entry starts with the corresponding default font relative value, and is followed by
        the delta values. We then convert the default values, the first item in each entry, to an absolute value.
        """
        vsindex = self.dict.get("vsindex", 0)
        numMasters = (
            self.parent.getNumRegions(vsindex) + 1
        )  # only a PrivateDict has blended ops.
        numBlends = self.pop()
        args = self.popall()
        numArgs = len(args)
        # The spec says that there should be no non-blended Blue Values,.
        assert numArgs == numMasters * numBlends
        value = [None] * numBlends
        numDeltas = numMasters - 1
        i = 0
        prevVal = 0
        while i < numBlends:
            newVal = args[i] + prevVal
            prevVal = newVal
            masterOffset = numBlends + (i * numDeltas)
            blendList = [newVal] + args[masterOffset : masterOffset + numDeltas]
            value[i] = blendList
            i += 1
        return value

    def arg_delta(self, name):
        valueList = self.popall()
        out = []
        if valueList and isinstance(valueList[0], list):
            # arg_blendList() has already converted these to absolute values.
            out = valueList
        else:
            current = 0
            for v in valueList:
                current = current + v
                out.append(current)
        return out


def calcSubrBias(subrs):
    nSubrs = len(subrs)
    if nSubrs < 1240:
        bias = 107
    elif nSubrs < 33900:
        bias = 1131
    else:
        bias = 32768
    return bias
<|endoftext|>"""
PostScript Type 1 fonts make use of two types of encryption: charstring
encryption and ``eexec`` encryption. Charstring encryption is used for
the charstrings themselves, while ``eexec`` is used to encrypt larger
sections of the font program, such as the ``Private`` and ``CharStrings``
dictionaries. Despite the different names, the algorithm is the same,
although ``eexec`` encryption uses a fixed initial key R=55665.

The algorithm uses cipher feedback, meaning that the ciphertext is used
to modify the key. Because of this, the routines in this module return
the new key at the end of the operation.

"""

from fontTools.misc.textTools import bytechr, bytesjoin, byteord


def _decryptChar(cipher, R):
    cipher = byteord(cipher)
    plain = ((cipher ^ (R >> 8))) & 0xFF
    R = ((cipher + R) * 52845 + 22719) & 0xFFFF
    return bytechr(plain), R


def _encryptChar(plain, R):
    plain = byteord(plain)
    cipher = ((plain ^ (R >> 8))) & 0xFF
    R = ((cipher + R) * 52845 + 22719) & 0xFFFF
    return bytechr(cipher), R


def decrypt(cipherstring, R):
    r"""
    Decrypts a string using the Type 1 encryption algorithm.

    Args:
            cipherstring: String of ciphertext.
            R: Initial key.

    Returns:
            decryptedStr: Plaintext string.
            R: Output key for subsequent decryptions.

    Examples::

            >>> testStr = b"\0\0asdadads asds\265"
            >>> decryptedStr, R = decrypt(testStr, 12321)
            >>> decryptedStr == b'0d\nh\x15\xe8\xc4\xb2\x15\x1d\x108\x1a<6\xa1'
            True
            >>> R == 36142
            True
    """
    plainList = []
    for cipher in cipherstring:
        plain, R = _decryptChar(cipher, R)
        plainList.append(plain)
    plainstring = bytesjoin(plainList)
    return plainstring, int(R)


def encrypt(plainstring, R):
    r"""
    Encrypts a string using the Type 1 encryption algorithm.

    Note that the algorithm as described in the Type 1 specification requires the
    plaintext to be prefixed with a number of random bytes. (For ``eexec`` the
    number of random bytes is set to 4.) This routine does *not* add the random
    prefix to its input.

    Args:
            plainstring: String of plaintext.
            R: Initial key.

    Returns:
            cipherstring: Ciphertext string.
            R: Output key for subsequent encryptions.

    Examples::

            >>> testStr = b"\0\0asdadads asds\265"
            >>> decryptedStr, R = decrypt(testStr, 12321)
            >>> decryptedStr == b'0d\nh\x15\xe8\xc4\xb2\x15\x1d\x108\x1a<6\xa1'
            True
            >>> R == 36142
            True

    >>> testStr = b'0d\nh\x15\xe8\xc4\xb2\x15\x1d\x108\x1a<6\xa1'
    >>> encryptedStr, R = encrypt(testStr, 12321)
    >>> encryptedStr == b"\0\0asdadads asds\265"
    True
    >>> R == 36142
    True
    """
    cipherList = []
    for plain in plainstring:
        cipher, R = _encryptChar(plain, R)
        cipherList.append(cipher)
    cipherstring = bytesjoin(cipherList)
    return cipherstring, int(R)


def hexString(s):
    import binascii

    return binascii.hexlify(s)


def deHexString(h):
    import binascii

    h = bytesjoin(h.split())
    return binascii.unhexlify(h)


if __name__ == "__main__":
    import sys
    import doctest

    sys.exit(doctest.testmod().failed)
<|endoftext|>"""xmlWriter.py -- Simple XML authoring class"""

from __future__ import annotations

from typing import BinaryIO, Callable, TextIO
from fontTools.misc.textTools import byteord, strjoin, tobytes, tostr
import sys
import os
import string
import logging
import itertools

INDENT = "  "
TTX_LOG = logging.getLogger("fontTools.ttx")
REPLACEMENT = "?"
ILLEGAL_XML_CHARS = dict.fromkeys(
    itertools.chain(
        range(0x00, 0x09),
        (0x0B, 0x0C),
        range(0x0E, 0x20),
        range(0xD800, 0xE000),
        (0xFFFE, 0xFFFF),
    ),
    REPLACEMENT,
)


class XMLWriter(object):
    def __init__(
        self,
        fileOrPath: str | os.PathLike[str] | BinaryIO | TextIO,
        indentwhite: str = INDENT,
        idlefunc: Callable[[], None] | None = None,
        encoding: str = "utf_8",
        newlinestr: str | bytes = "\n",
    ) -> None:
        if encoding.lower().replace("-", "").replace("_", "") != "utf8":
            raise Exception("Only UTF-8 encoding is supported.")
        if fileOrPath == "-":
            fileOrPath = sys.stdout
        self.filename: str | os.PathLike[str] | None
        if not hasattr(fileOrPath, "write"):
            if not isinstance(fileOrPath, (str, os.PathLike)):
                raise TypeError(
                    "fileOrPath must be a file path (str or PathLike) if it isn't an object with a `write` method."
                )
            self.filename = fileOrPath
            self.file = open(fileOrPath, "wb")
            self._closeStream = True
        else:
            self.filename = None
            # assume writable file object
            self.file = fileOrPath
            self._closeStream = False

        # Figure out if writer expects bytes or unicodes
        try:
            # The bytes check should be first.  See:
            # https://github.com/fonttools/fonttools/pull/233
            self.file.write(b"")
            self.totype = tobytes
        except TypeError:
            # This better not fail.
            self.file.write("")
            self.totype = tostr
        self.indentwhite = self.totype(indentwhite)
        if newlinestr is None:
            self.newlinestr = self.totype(os.linesep)
        else:
            self.newlinestr = self.totype(newlinestr)
        self.indentlevel = 0
        self.stack = []
        self.needindent = 1
        self.idlefunc = idlefunc
        self.idlecounter = 0
        self._writeraw('<?xml version="1.0" encoding="UTF-8"?>')
        self.newline()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def close(self) -> None:
        if self._closeStream:
            assert not isinstance(self.file, (str, os.PathLike))
            self.file.close()

    def write(self, string, indent=True):
        """Writes text."""
        self._writeraw(escape(string), indent=indent)

    def writecdata(self, string):
        """Writes text in a CDATA section."""
        # a CDATA section can't hold "]]>", so close it and open a new one around
        # the ">"; the two sections read back as the original text
        string = string.replace("]]>", "]]]]><![CDATA[>")
        self._writeraw("<![CDATA[" + string + "]]>")

    def write8bit(self, data, strip=False):
        """Writes a bytes() sequence into the XML, escaping
        non-ASCII bytes.  When this is read in xmlReader,
        the original bytes can be recovered by encoding to
        'latin-1'."""
        self._writeraw(escape8bit(data), strip=strip)

    def write_noindent(self, string):
        """Writes text without indentation."""
        self._writeraw(escape(string), indent=False)

    def _writeraw(self, data, indent=True, strip=False):
        """Writes bytes, possibly indented."""
        if indent and self.needindent:
            self.file.write(self.indentlevel * self.indentwhite)
            self.needindent = 0
        s = self.totype(data, encoding="utf_8")
        if strip:
            s = s.strip()
        self.file.write(s)

    def newline(self):
        self.file.write(self.newlinestr)
        self.needindent = 1
        idlecounter = self.idlecounter
        if not idlecounter % 100 and self.idlefunc is not None:
            self.idlefunc()
        self.idlecounter = idlecounter + 1

    def comment(self, data):
        data = escape(data)
        lines = data.split("\n")
        self._writeraw("<!-- " + lines[0])
        for line in lines[1:]:
            self.newline()
            self._writeraw("     " + line)
        self._writeraw(" -->")

    def simpletag(self, _TAG_, *args, **kwargs):
        attrdata = self.stringifyattrs(*args, **kwargs)
        data = "<%s%s/>" % (_TAG_, attrdata)
        self._writeraw(data)

    def begintag(self, _TAG_, *args, **kwargs):
        attrdata = self.stringifyattrs(*args, **kwargs)
        data = "<%s%s>" % (_TAG_, attrdata)
        self._writeraw(data)
        self.stack.append(_TAG_)
        self.indent()

    def endtag(self, _TAG_):
        assert self.stack and self.stack[-1] == _TAG_, "nonmatching endtag"
        del self.stack[-1]
        self.dedent()
        data = "</%s>" % _TAG_
        self._writeraw(data)

    def dumphex(self, data):
        linelength = 16
        hexlinelength = linelength * 2
        chunksize = 8
        for i in range(0, len(data), linelength):
            hexline = hexStr(data[i : i + linelength])
            line = ""
            white = ""
            for j in range(0, hexlinelength, chunksize):
                line = line + white + hexline[j : j + chunksize]
                white = " "
            self._writeraw(line)
            self.newline()

    def indent(self):
        self.indentlevel = self.indentlevel + 1

    def dedent(self):
        assert self.indentlevel > 0
        self.indentlevel = self.indentlevel - 1

    def stringifyattrs(self, *args, **kwargs):
        if kwargs:
            assert not args
            attributes = sorted(kwargs.items())
        elif args:
            assert len(args) == 1
            attributes = args[0]
        else:
            return ""
        data = ""
        for attr, value in attributes:
            if not isinstance(value, (bytes, str)):
                value = str(value)
            data = data + ' %s="%s"' % (attr, escapeattr(value))
        return data


def escape(data):
    """Escape characters not allowed in `XML 1.0 <https://www.w3.org/TR/xml/#NT-Char>`_."""
    data = tostr(data, "utf_8")
    data = data.replace("&", "&amp;")
    data = data.replace("<", "&lt;")
    data = data.replace(">", "&gt;")
    data = data.replace("\r", "&#13;")

    newData = data.translate(ILLEGAL_XML_CHARS)
    if newData != data:
        maxLen = 10
        preview = repr(data)
        if len(data) > maxLen:
            preview = repr(data[:maxLen])[1:-1] + "..."
        TTX_LOG.warning(
            "Illegal XML character(s) found; replacing offending string %r with %r",
            preview,
            REPLACEMENT,
        )
    return newData


def escapeattr(data):
    data = escape(data)
    data = data.replace('"', "&quot;")
    return data


def escape8bit(data):
    """Input is Unicode string."""

    def escapechar(c):
        n = ord(c)
        if 32 <= n <= 127 and c not in "<&>":
            return c
        else:
            return "&#" + repr(n) + ";"

    return strjoin(map(escapechar, data.decode("latin-1")))


def hexStr(s):
    h = string.hexdigits
    r = ""
    for c in s:
        i = byteord(c)
        r = r + h[(i >> 4) & 0xF] + h[i & 0xF]
    return r
<|endoftext|>"""Enum-related utilities, including backports for older Python versions."""

from __future__ import annotations

from enum import Enum

__all__ = ["StrEnum"]

# StrEnum is only available in Python 3.11+
try:
    from enum import StrEnum
except ImportError:

    class StrEnum(str, Enum):
        """
        Minimal backport of Python 3.11's StrEnum for older versions.

        An Enum where all members are also strings.
        """

        def __str__(self) -> str:
            return self.value
<|endoftext|>"""fontTools.misc.classifyTools.py -- tools for classifying things."""


class Classifier(object):
    """
    Main Classifier object, used to classify things into similar sets.
    """

    def __init__(self, sort=True):
        self._things = set()  # set of all things known so far
        self._sets = []  # list of class sets produced so far
        self._mapping = {}  # map from things to their class set
        self._dirty = False
        self._sort = sort

    def add(self, set_of_things):
        """
        Add a set to the classifier.  Any iterable is accepted.
        """
        if not set_of_things:
            return

        self._dirty = True

        things, sets, mapping = self._things, self._sets, self._mapping

        s = set(set_of_things)
        intersection = s.intersection(things)  # existing things
        s.difference_update(intersection)  # new things
        difference = s
        del s

        # Add new class for new things
        if difference:
            things.update(difference)
            sets.append(difference)
            for thing in difference:
                mapping[thing] = difference
        del difference

        while intersection:
            # Take one item and process the old class it belongs to
            old_class = mapping[next(iter(intersection))]
            old_class_intersection = old_class.intersection(intersection)

            # Update old class to remove items from new set
            old_class.difference_update(old_class_intersection)

            # Remove processed items from todo list
            intersection.difference_update(old_class_intersection)

            # Add new class for the intersection with old class
            sets.append(old_class_intersection)
            for thing in old_class_intersection:
                mapping[thing] = old_class_intersection
            del old_class_intersection

    def update(self, list_of_sets):
        """
        Add a a list of sets to the classifier.  Any iterable of iterables is accepted.
        """
        for s in list_of_sets:
            self.add(s)

    def _process(self):
        if not self._dirty:
            return

        # Do any deferred processing
        sets = self._sets
        self._sets = [s for s in sets if s]

        if self._sort:
            self._sets = sorted(self._sets, key=lambda s: (-len(s), sorted(s)))

        self._dirty = False

    # Output methods

    def getThings(self):
        """Returns the set of all things known so far.

        The return value belongs to the Classifier object and should NOT
        be modified while the classifier is still in use.
        """
        self._process()
        return self._things

    def getMapping(self):
        """Returns the mapping from things to their class set.

        The return value belongs to the Classifier object and should NOT
        be modified while the classifier is still in use.
        """
        self._process()
        return self._mapping

    def getClasses(self):
        """Returns the list of class sets.

        The return value belongs to the Classifier object and should NOT
        be modified while the classifier is still in use.
        """
        self._process()
        return self._sets


def classify(list_of_sets, sort=True):
    """
    Takes a iterable of iterables (list of sets from here on; but any
    iterable works.), and returns the smallest list of sets such that
    each set, is either a subset, or is disjoint from, each of the input
    sets.

    In other words, this function classifies all the things present in
    any of the input sets, into similar classes, based on which sets
    things are a member of.

    If sort=True, return class sets are sorted by decreasing size and
    their natural sort order within each class size.  Otherwise, class
    sets are returned in the order that they were identified, which is
    generally not significant.

    >>> classify([]) == ([], {})
    True
    >>> classify([[]]) == ([], {})
    True
    >>> classify([[], []]) == ([], {})
    True
    >>> classify([[1]]) == ([{1}], {1: {1}})
    True
    >>> classify([[1,2]]) == ([{1, 2}], {1: {1, 2}, 2: {1, 2}})
    True
    >>> classify([[1],[2]]) == ([{1}, {2}], {1: {1}, 2: {2}})
    True
    >>> classify([[1,2],[2]]) == ([{1}, {2}], {1: {1}, 2: {2}})
    True
    >>> classify([[1,2],[2,4]]) == ([{1}, {2}, {4}], {1: {1}, 2: {2}, 4: {4}})
    True
    >>> classify([[1,2],[2,4,5]]) == (
    ...     [{4, 5}, {1}, {2}], {1: {1}, 2: {2}, 4: {4, 5}, 5: {4, 5}})
    True
    >>> classify([[1,2],[2,4,5]], sort=False) == (
    ...     [{1}, {4, 5}, {2}], {1: {1}, 2: {2}, 4: {4, 5}, 5: {4, 5}})
    True
    >>> classify([[1,2,9],[2,4,5]], sort=False) == (
    ...     [{1, 9}, {4, 5}, {2}], {1: {1, 9}, 2: {2}, 4: {4, 5}, 5: {4, 5},
    ...     9: {1, 9}})
    True
    >>> classify([[1,2,9,15],[2,4,5]], sort=False) == (
    ...     [{1, 9, 15}, {4, 5}, {2}], {1: {1, 9, 15}, 2: {2}, 4: {4, 5},
    ...     5: {4, 5}, 9: {1, 9, 15}, 15: {1, 9, 15}})
    True
    >>> classes, mapping = classify([[1,2,9,15],[2,4,5],[15,5]], sort=False)
    >>> set([frozenset(c) for c in classes]) == set(
    ...     [frozenset(s) for s in ({1, 9}, {4}, {2}, {5}, {15})])
    True
    >>> mapping == {1: {1, 9}, 2: {2}, 4: {4}, 5: {5}, 9: {1, 9}, 15: {15}}
    True
    """
    classifier = Classifier(sort=sort)
    classifier.update(list_of_sets)
    return classifier.getClasses(), classifier.getMapping()


if __name__ == "__main__":
    import sys, doctest

    sys.exit(doctest.testmod(optionflags=doctest.ELLIPSIS).failed)
<|endoftext|>from __future__ import annotations

import typing
from pathlib import PurePosixPath

from ._errors import DirectoryNotEmpty

if typing.TYPE_CHECKING:
    from typing import IO

    from ._base import FS


def remove_empty(fs: FS, path: str):
    """Remove all empty parents."""
    path = PurePosixPath(path)
    root = PurePosixPath("/")
    try:
        while path != root:
            fs.removedir(path.as_posix())
            path = path.parent
    except DirectoryNotEmpty:
        pass


def copy_file_data(src_file: IO, dst_file: IO, chunk_size: int | None = None):
    """Copy data from one file object to another."""
    _chunk_size = 1024 * 1024 if chunk_size is None else chunk_size
    read = src_file.read
    write = dst_file.write
    # in iter(callable, sentilel), callable is called until it returns the sentinel;
    # this allows to copy `chunk_size` bytes at a time.
    for chunk in iter(lambda: read(_chunk_size) or None, None):
        write(chunk)
<|endoftext|>import os
import platform

_WINDOWS_PLATFORM = platform.system() == "Windows"


def combine(path1: str, path2) -> str:
    if not path1:
        return path2
    return "{}/{}".format(path1.rstrip("/"), path2.lstrip("/"))


def split(path: str) -> tuple[str, str]:
    if "/" not in path:
        return ("", path)
    split = path.rsplit("/", 1)
    return (split[0] or "/", split[1])


def dirname(path: str) -> str:
    return split(path)[0]


def basename(path: str) -> str:
    return split(path)[1]


def forcedir(path: str) -> str:
    # Ensure the path ends with a trailing forward slash.
    if not path.endswith("/"):
        return path + "/"
    return path


def abspath(path: str) -> str:
    # FS objects have no concept of a *current directory*. This simply
    # ensures the path starts with a forward slash.
    if not path.startswith("/"):
        return "/" + path
    return path


def isbase(path1: str, path2: str) -> bool:
    # Check if `path1` is a base or prefix of `path2`.
    _path1 = forcedir(abspath(path1))
    _path2 = forcedir(abspath(path2))
    return _path2.startswith(_path1)


def frombase(path1: str, path2: str) -> str:
    # Get the final path of `path2` that isn't in `path1`.
    if not isbase(path1, path2):
        raise ValueError(f"path1 must be a prefix of path2: {path1!r} vs {path2!r}")
    return path2[len(path1) :]


def relpath(path: str) -> str:
    return path.lstrip("/")


def normpath(path: str) -> str:
    normalized = os.path.normpath(path)
    if _WINDOWS_PLATFORM:
        # os.path.normpath converts backslashes to forward slashes on Windows
        # but we want forward slashes, so we convert them back
        normalized = normalized.replace("\\", "/")
    return normalized


def escapes_root(path: str) -> bool:
    # Check whether `path` normalizes to a location outside the filesystem root.
    # This is a purely textual test, for use where a path can be rejected before
    # anything touches the disk; OSFS._abs does the authoritative check.
    normalized = normpath(relpath(path))
    return normalized == ".." or normalized.startswith("../")
<|endoftext|>class FSError(Exception):
    pass


class CreateFailed(FSError):
    pass


class FilesystemClosed(FSError):
    pass


class MissingInfoNamespace(FSError):
    pass


class NoSysPath(FSError):
    pass


class OperationFailed(FSError):
    pass


class IllegalDestination(OperationFailed):
    pass


class ResourceError(FSError):
    pass


class ResourceNotFound(ResourceError):
    pass


class DirectoryExpected(ResourceError):
    pass


class DirectoryNotEmpty(ResourceError):
    pass


class FileExpected(ResourceError):
    pass


class DestinationExists(ResourceError):
    pass


class ResourceReadOnly(ResourceError):
    pass


class IllegalBackReference(ValueError):
    # Named after fs.errors.IllegalBackReference (which is also a ValueError and
    # not an FSError) so that code catching the upstream error keeps working.
    def __init__(self, path):
        super().__init__(f"path {path!r} resolves outside of the filesystem root")
<|endoftext|>"""Minimal, stdlib-only replacement for [`pyfilesystem2`][1] API for use by `fontTools.ufoLib`.

This package is a partial reimplementation of the `fs` package by Will McGugan, used under the
MIT license. See LICENSE.external for details.

Note this only exports a **subset** of the `pyfilesystem2` API, in particular the modules,
classes and functions that are currently used directly by `fontTools.ufoLib`.

It opportunistically tries to import the relevant modules from the upstream `fs` package
when this is available. Otherwise it falls back to the replacement modules within this package.

As of version 4.59.0, the `fonttools[ufo]` extra no longer requires the `fs` package, thus
this `fontTools.misc.filesystem` package is used by default.

Client code can either replace `import fs` with `from fontTools.misc import filesystem as fs`
if that happens to work (no guarantee), or they can continue to use `fs` but they will have
to specify it as an explicit dependency of their project.

[1]: https://github.com/PyFilesystem/pyfilesystem2
"""

from __future__ import annotations

try:
    __import__("fs")
except ImportError:
    from . import _base as base
    from . import _copy as copy
    from . import _errors as errors
    from . import _info as info
    from . import _osfs as osfs
    from . import _path as path
    from . import _subfs as subfs
    from . import _tempfs as tempfs
    from . import _tools as tools
    from . import _walk as walk
    from . import _zipfs as zipfs

    _haveFS = False
else:
    import fs.base as base
    import fs.copy as copy
    import fs.errors as errors
    import fs.info as info
    import fs.osfs as osfs
    import fs.path as path
    import fs.subfs as subfs
    import fs.tempfs as tempfs
    import fs.tools as tools
    import fs.walk as walk
    import fs.zipfs as zipfs

    _haveFS = True


__all__ = [
    "base",
    "copy",
    "errors",
    "info",
    "osfs",
    "path",
    "subfs",
    "tempfs",
    "tools",
    "walk",
    "zipfs",
]
<|endoftext|>from __future__ import annotations

import io
import os
import shutil
import stat
import typing
import zipfile
from datetime import datetime

from ._base import FS
from ._errors import (
    FileExpected,
    IllegalBackReference,
    ResourceNotFound,
    ResourceReadOnly,
)
from ._info import Info
from ._path import dirname, escapes_root, forcedir, normpath, relpath
from ._tempfs import TempFS

if typing.TYPE_CHECKING:
    from collections.abc import Collection
    from typing import IO, Any

    from ._subfs import SubFS


class ZipFS(FS):
    """Read and write zip files."""

    def __new__(
        cls, file: str | os.PathLike, write: bool = False, encoding: str = "utf-8"
    ):
        if write:
            return WriteZipFS(file, encoding)
        else:
            return ReadZipFS(file, encoding)

    if typing.TYPE_CHECKING:

        def __init__(
            self, file: str | os.PathLike, write: bool = False, encoding: str = "utf-8"
        ):
            pass


class ReadZipFS(FS):
    """A readable zip file."""

    def __init__(self, file: str | os.PathLike, encoding: str = "utf-8"):
        super().__init__()
        self._file = os.fspath(file)
        self.encoding = encoding  # unused
        self._zip = zipfile.ZipFile(file, "r")
        self._directory_fs = None

    def __repr__(self) -> str:
        return f"ReadZipFS({self._file!r})"

    def __str__(self) -> str:
        return f"<zipfs '{self._file}'>"

    def _path_to_zip_name(self, path: str) -> str:
        """Convert a path to a zip file name."""
        path = relpath(normpath(path))
        if self._directory.isdir(path):
            path = forcedir(path)
        return path

    @property
    def _directory(self) -> TempFS:
        if self._directory_fs is None:
            # Reject escaping member names up front, so the error names the zip
            # entry rather than a path the caller never passed.
            for zip_name in self._zip.namelist():
                if escapes_root(zip_name):
                    raise IllegalBackReference(zip_name)
            # Populate a local mirror and only publish it once it is complete:
            # an archive can fail partway through for reasons the pre-scan can't
            # see (a member named both as a file and as a directory, an escaping
            # name that only pathlib recognises on this platform), and a cached
            # half-built mirror would answer later calls from a silently
            # truncated view of the zip instead of failing again.
            _fs = TempFS()
            try:
                for zip_name in self._zip.namelist():
                    resource_name = zip_name
                    if resource_name.endswith("/"):
                        _fs.makedirs(resource_name, recreate=True)
                    else:
                        _fs.makedirs(dirname(resource_name), recreate=True)
                        _fs.create(resource_name)
            except Exception:
                _fs.close()
                raise
            self._directory_fs = _fs
        return self._directory_fs

    def close(self):
        super(ReadZipFS, self).close()
        self._zip.close()
        if self._directory_fs is not None:
            self._directory_fs.close()

    def getinfo(self, path: str, namespaces: Collection[str] | None = None) -> Info:
        namespaces = namespaces or ()
        raw_info = {}

        if path == "/":
            raw_info["basic"] = {"name": "", "is_dir": True}
            if "details" in namespaces:
                raw_info["details"] = {"type": stat.S_IFDIR}
        else:
            basic_info = self._directory.getinfo(path)
            raw_info["basic"] = {"name": basic_info.name, "is_dir": basic_info.is_dir}

            if "details" in namespaces:
                zip_name = self._path_to_zip_name(path)
                try:
                    zip_info = self._zip.getinfo(zip_name)
                except KeyError:
                    pass
                else:
                    if "details" in namespaces:
                        raw_info["details"] = {
                            "size": zip_info.file_size,
                            "type": int(
                                stat.S_IFDIR if basic_info.is_dir else stat.S_IFREG
                            ),
                            "modified": datetime(*zip_info.date_time).timestamp(),
                        }

        return Info(raw_info)

    def exists(self, path: str) -> bool:
        self.check()
        return self._directory.exists(path)

    def isdir(self, path: str) -> bool:
        self.check()
        return self._directory.isdir(path)

    def isfile(self, path: str) -> bool:
        self.check()
        return self._directory.isfile(path)

    def listdir(self, path: str) -> str:
        self.check()
        return self._directory.listdir(path)

    def makedir(self, path: str, recreate: bool = False) -> SubFS:
        self.check()
        raise ResourceReadOnly(path)

    def makedirs(self, path: str, recreate: bool = False) -> SubFS:
        self.check()
        raise ResourceReadOnly(path)

    def remove(self, path: str):
        self.check()
        raise ResourceReadOnly(path)

    def removedir(self, path: str):
        self.check()
        raise ResourceReadOnly(path)

    def removetree(self, path: str):
        self.check()
        raise ResourceReadOnly(path)

    def movedir(self, src: str, dst: str, create: bool = False):
        self.check()
        raise ResourceReadOnly(src)

    def readbytes(self, path: str) -> bytes:
        self.check()
        if not self._directory.isfile(path):
            raise ResourceNotFound(path)
        zip_name = self._path_to_zip_name(path)
        zip_bytes = self._zip.read(zip_name)
        return zip_bytes

    def open(self, path: str, mode: str = "rb", **kwargs) -> IO[Any]:
        self.check()
        if self._directory.isdir(path):
            raise FileExpected(f"{path!r} is a directory")

        zip_mode = mode[0]
        if zip_mode == "r" and not self._directory.exists(path):
            raise ResourceNotFound(f"No such file or directory: {path!r}")

        if any(m in mode for m in "wax+"):
            raise ResourceReadOnly(path)

        zip_name = self._path_to_zip_name(path)
        stream = self._zip.open(zip_name, zip_mode)
        if "b" in mode:
            if kwargs:
                raise ValueError("encoding args invalid for binary operation")
            return stream
        # Text mode
        return io.TextIOWrapper(stream, **kwargs)


class WriteZipFS(TempFS):
    """A writable zip file."""

    def __init__(self, file: str | os.PathLike, encoding: str = "utf-8"):
        super().__init__()
        self._file = os.fspath(file)
        self.encoding = encoding  # unused

    def __repr__(self) -> str:
        return f"WriteZipFS({self._file!r})"

    def __str__(self) -> str:
        return f"<zipfs-write '{self._file}'>"

    def close(self):
        base_name = os.path.splitext(self._file)[0]
        shutil.make_archive(base_name, format="zip", root_dir=self._temp_dir)
        if self._file != base_name + ".zip":
            shutil.move(base_name + ".zip", self._file)
        super().close()
<|endoftext|>from __future__ import annotations

import typing
from datetime import datetime, timezone

from ._errors import MissingInfoNamespace

if typing.TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def epoch_to_datetime(t: int | None) -> datetime | None:
    """Convert epoch time to a UTC datetime."""
    if t is None:
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc)


class Info:
    __slots__ = ["raw", "namespaces"]

    def __init__(self, raw_info: Mapping[str, Any]):
        self.raw = raw_info
        self.namespaces = frozenset(raw_info.keys())

    def get(self, namespace: str, key: str, default: Any | None = None) -> Any | None:
        try:
            return self.raw[namespace].get(key, default)
        except KeyError:
            raise MissingInfoNamespace(f"Namespace {namespace!r} does not exist")

    @property
    def name(self) -> str:
        return self.get("basic", "name")

    @property
    def is_dir(self) -> bool:
        return self.get("basic", "is_dir")

    @property
    def is_file(self) -> bool:
        return not self.is_dir

    @property
    def accessed(self) -> datetime | None:
        return epoch_to_datetime(self.get("details", "accessed"))

    @property
    def modified(self) -> datetime | None:
        return epoch_to_datetime(self.get("details", "modified"))

    @property
    def size(self) -> int | None:
        return self.get("details", "size")

    @property
    def type(self) -> int | None:
        return self.get("details", "type")

    @property
    def created(self) -> datetime | None:
        return epoch_to_datetime(self.get("details", "created"))

    @property
    def metadata_changed(self) -> datetime | None:
        return epoch_to_datetime(self.get("details", "metadata_changed"))

    def __str__(self) -> str:
        if self.is_dir:
            return "<dir '{}'>".format(self.name)
        else:
            return "<file '{}'>".format(self.name)

    __repr__ = __str__
<|endoftext|>from __future__ import annotations

import typing

from ._errors import IllegalDestination
from ._path import combine, frombase, isbase
from ._tools import copy_file_data

if typing.TYPE_CHECKING:
    from ._base import FS


def copy_file(src_fs: FS, src_path: str, dst_fs: FS, dst_path: str):
    if src_fs is dst_fs and src_path == dst_path:
        raise IllegalDestination(f"cannot copy {src_path!r} to itself")

    with src_fs.open(src_path, "rb") as src_file:
        with dst_fs.open(dst_path, "wb") as dst_file:
            copy_file_data(src_file, dst_file)


def copy_structure(
    src_fs: FS,
    dst_fs: FS,
    src_root: str = "/",
    dst_root: str = "/",
):
    if src_fs is dst_fs and isbase(src_root, dst_root):
        raise IllegalDestination(f"cannot copy {src_fs!r} to itself")

    dst_fs.makedirs(dst_root, recreate=True)
    for dir_path in src_fs.walk.dirs(src_root):
        dst_fs.makedir(combine(dst_root, frombase(src_root, dir_path)), recreate=True)


def copy_dir(src_fs: FS, src_path: str, dst_fs: FS, dst_path: str):
    copy_structure(src_fs, dst_fs, src_path, dst_path)

    for file_path in src_fs.walk.files(src_path):
        copy_path = combine(dst_path, frombase(src_path, file_path))
        copy_file(src_fs, file_path, dst_fs, copy_path)


def copy_fs(src_fs: FS, dst_fs: FS):
    copy_dir(src_fs, "/", dst_fs, "/")
<|endoftext|>from __future__ import annotations

import typing
from pathlib import PurePosixPath

from ._base import FS
from ._errors import DirectoryExpected, ResourceNotFound

if typing.TYPE_CHECKING:
    from collections.abc import Collection
    from typing import IO, Any

    from ._info import Info


class SubFS(FS):
    """Maps a sub-directory of another filesystem."""

    def __init__(self, parent: FS, sub_path: str):
        super().__init__()
        self._parent = parent
        self._prefix = PurePosixPath(sub_path).as_posix().rstrip("/")
        if not parent.exists(self._prefix):
            raise ResourceNotFound(f"No such file or directory: {sub_path!r}")
        elif not parent.isdir(self._prefix):
            raise DirectoryExpected(f"{sub_path!r} is not a directory")

    def delegate_fs(self):
        return self._parent

    def _full(self, rel: str) -> str:
        self.check()
        return f"{self._prefix}/{PurePosixPath(rel).as_posix()}".lstrip("/")

    def open(self, path: str, mode: str = "rb", **kwargs) -> IO[Any]:
        return self._parent.open(self._full(path), mode, **kwargs)

    def exists(self, path: str) -> bool:
        return self._parent.exists(self._full(path))

    def isdir(self, path: str) -> bool:
        return self._parent.isdir(self._full(path))

    def isfile(self, path: str) -> bool:
        return self._parent.isfile(self._full(path))

    def listdir(self, path: str) -> list[str]:
        return self._parent.listdir(self._full(path))

    def makedir(self, path: str, recreate: bool = False):
        return self._parent.makedir(self._full(path), recreate=recreate)

    def makedirs(self, path: str, recreate: bool = False):
        return self._parent.makedirs(self._full(path), recreate=recreate)

    def getinfo(self, path: str, namespaces: Collection[str] | None = None) -> Info:
        return self._parent.getinfo(self._full(path), namespaces=namespaces)

    def remove(self, path: str):
        return self._parent.remove(self._full(path))

    def removedir(self, path: str):
        return self._parent.removedir(self._full(path))

    def removetree(self, path: str):
        return self._parent.removetree(self._full(path))

    def movedir(self, src: str, dst: str, create: bool = False):
        self._parent.movedir(self._full(src), self._full(dst), create=create)

    def getsyspath(self, path: str) -> str:
        return self._parent.getsyspath(self._full(path))

    def readbytes(self, path: str) -> bytes:
        return self._parent.readbytes(self._full(path))

    def writebytes(self, path: str, data: bytes):
        self._parent.writebytes(self._full(path), data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._parent!r}, {self._prefix!r})"

    def __str__(self) -> str:
        return f"{self._parent}/{self._prefix}"


class ClosingSubFS(SubFS):
    """Like SubFS, but auto-closes the parent filesystem when closed."""

    def close(self):
        super().close()
        self._parent.close()
<|endoftext|>from __future__ import annotations

import errno
import platform
import shutil
import stat
import typing
from os import PathLike
from pathlib import Path

from ._base import FS
from ._errors import (
    CreateFailed,
    DirectoryExpected,
    DirectoryNotEmpty,
    FileExpected,
    IllegalBackReference,
    IllegalDestination,
    ResourceError,
    ResourceNotFound,
)
from ._info import Info
from ._path import isbase

if typing.TYPE_CHECKING:
    from collections.abc import Collection
    from typing import IO, Any

    from ._subfs import SubFS


_WINDOWS_PLATFORM = platform.system() == "Windows"


class OSFS(FS):
    """Filesystem for a directory on the local disk.

    A thin layer on top of `pathlib.Path`.
    """

    def __init__(self, root: str | PathLike, create: bool = False):
        super().__init__()
        self._root = Path(root).resolve()
        if create:
            self._root.mkdir(parents=True, exist_ok=True)
        else:
            if not self._root.is_dir():
                raise CreateFailed(
                    f"unable to create OSFS: {root!r} does not exist or is not a directory"
                )

    def _abs(self, rel_path: str) -> Path:
        self.check()
        abs_path = (self._root / rel_path.strip("/")).resolve()
        # resolve() expands symlinks as well as back-references, so a symlink
        # inside the root that points outside it is rejected too. That is
        # stricter than pyfilesystem2, whose OSFS normalizes textually and
        # follows symlinks out of the root, and deliberately so: paths reaching
        # here can come from untrusted data, e.g. a glyph file name read from a
        # third-party UFO's contents.plist.
        if not abs_path.is_relative_to(self._root):
            raise IllegalBackReference(rel_path)
        return abs_path

    def open(self, path: str, mode: str = "rb", **kwargs) -> IO[Any]:
        try:
            return self._abs(path).open(mode, **kwargs)
        except FileNotFoundError:
            raise ResourceNotFound(f"No such file or directory: {path!r}")

    def exists(self, path: str) -> bool:
        return self._abs(path).exists()

    def isdir(self, path: str) -> bool:
        return self._abs(path).is_dir()

    def isfile(self, path: str) -> bool:
        return self._abs(path).is_file()

    def listdir(self, path: str) -> list[str]:
        return [p.name for p in self._abs(path).iterdir()]

    def _mkdir(self, path: str, parents: bool = False, exist_ok: bool = False) -> SubFS:
        self._abs(path).mkdir(parents=parents, exist_ok=exist_ok)
        return self.opendir(path)

    def makedir(self, path: str, recreate: bool = False) -> SubFS:
        return self._mkdir(path, parents=False, exist_ok=recreate)

    def makedirs(self, path: str, recreate: bool = False) -> SubFS:
        return self._mkdir(path, parents=True, exist_ok=recreate)

    def getinfo(self, path: str, namespaces: Collection[str] | None = None) -> Info:
        path = self._abs(path)
        if not path.exists():
            raise ResourceNotFound(f"No such file or directory: {str(path)!r}")
        info = {
            "basic": {
                "name": path.name,
                "is_dir": path.is_dir(),
            }
        }
        namespaces = namespaces or ()
        if "details" in namespaces:
            stat_result = path.stat()
            details = info["details"] = {
                "accessed": stat_result.st_atime,
                "modified": stat_result.st_mtime,
                "size": stat_result.st_size,
                "type": stat.S_IFMT(stat_result.st_mode),
                "created": getattr(stat_result, "st_birthtime", None),
            }
            ctime_key = "created" if _WINDOWS_PLATFORM else "metadata_changed"
            details[ctime_key] = stat_result.st_ctime
        return Info(info)

    def remove(self, path: str):
        path = self._abs(path)
        try:
            path.unlink()
        except FileNotFoundError:
            raise ResourceNotFound(f"No such file or directory: {str(path)!r}")
        except OSError as e:
            if path.is_dir():
                raise FileExpected(f"path {str(path)!r} should be a file")
            else:
                raise ResourceError(f"unable to remove {str(path)!r}: {e}")

    def removedir(self, path: str):
        try:
            self._abs(path).rmdir()
        except NotADirectoryError:
            raise DirectoryExpected(f"path {path!r} should be a directory")
        except OSError as e:
            if e.errno == errno.ENOTEMPTY:
                raise DirectoryNotEmpty(f"Directory not empty: {path!r}")
            else:
                raise ResourceError(f"unable to remove {path!r}: {e}")

    def removetree(self, path: str):
        shutil.rmtree(self._abs(path))

    def movedir(self, src_dir: str, dst_dir: str, create: bool = False):
        if isbase(src_dir, dst_dir):
            raise IllegalDestination(f"cannot move {src_dir!r} to {dst_dir!r}")
        src_path = self._abs(src_dir)
        if not src_path.exists():
            raise ResourceNotFound(f"Source {src_dir!r} does not exist")
        elif not src_path.is_dir():
            raise DirectoryExpected(f"Source {src_dir!r} should be a directory")
        dst_path = self._abs(dst_dir)
        if not create and not dst_path.exists():
            raise ResourceNotFound(f"Destination {dst_dir!r} does not exist")
        if dst_path.is_file():
            raise DirectoryExpected(f"Destination {dst_dir!r} should be a directory")
        if create:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            if list(dst_path.iterdir()):
                raise DirectoryNotEmpty(f"Destination {dst_dir!r} is not empty")
            elif _WINDOWS_PLATFORM:
                # on Unix os.rename silently replaces an empty dst_dir whereas on
                # Windows it always raises FileExistsError, empty or not.
                dst_path.rmdir()
        src_path.rename(dst_path)

    def getsyspath(self, path: str) -> str:
        return str(self._abs(path))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._root)!r})"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__.lower()} '{self._root}'>"
<|endoftext|>from __future__ import annotations

import typing
from collections import deque
from collections.abc import Collection, Iterator

from ._path import combine

if typing.TYPE_CHECKING:
    from typing import Callable

    from ._base import FS
    from ._info import Info


class BoundWalker:
    def __init__(self, fs: FS):
        self._fs = fs

    def _iter_walk(
        self, path: str, namespaces: Collection[str] | None = None
    ) -> Iterator[tuple[str, Info | None]]:
        """Walk files using a *breadth first* search."""
        queue = deque([path])
        push = queue.appendleft
        pop = queue.pop
        _scan = self._fs.scandir
        _combine = combine

        while queue:
            dir_path = pop()
            for info in _scan(dir_path, namespaces=namespaces):
                if info.is_dir:
                    yield dir_path, info
                    push(_combine(dir_path, info.name))
                else:
                    yield dir_path, info
        yield path, None

    def _filter(
        self,
        include: Callable[[str, Info], bool] = lambda path, info: True,
        path: str = "/",
        namespaces: Collection[str] | None = None,
    ) -> Iterator[str]:
        _combine = combine
        for path, info in self._iter_walk(path, namespaces):
            if info is not None and include(path, info):
                yield _combine(path, info.name)

    def files(self, path: str = "/") -> Iterator[str]:
        yield from self._filter(lambda _, info: info.is_file, path)

    def dirs(self, path: str = "/") -> Iterator[str]:
        yield from self._filter(lambda _, info: info.is_dir, path)
<|endoftext|>from __future__ import annotations

import typing
from abc import ABC, abstractmethod

from ._copy import copy_dir, copy_file
from ._errors import (
    DestinationExists,
    DirectoryExpected,
    FileExpected,
    FilesystemClosed,
    NoSysPath,
    ResourceNotFound,
)
from ._path import dirname
from ._walk import BoundWalker

if typing.TYPE_CHECKING:
    from typing import IO, Any, Collection, Iterator, Self, Type

    from ._info import Info
    from ._subfs import SubFS


class FS(ABC):
    """Abstract base class for custom filesystems."""

    _closed: bool = False

    @abstractmethod
    def open(self, path: str, mode: str = "rb", **kwargs) -> IO[Any]: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def isdir(self, path: str) -> bool: ...

    @abstractmethod
    def isfile(self, path: str) -> bool: ...

    @abstractmethod
    def listdir(self, path: str) -> list[str]: ...

    @abstractmethod
    def makedir(self, path: str, recreate: bool = False) -> SubFS: ...

    @abstractmethod
    def makedirs(self, path: str, recreate: bool = False) -> SubFS: ...

    @abstractmethod
    def getinfo(self, path: str, namespaces: Collection[str] | None = None) -> Info: ...

    @abstractmethod
    def remove(self, path: str) -> None: ...

    @abstractmethod
    def removedir(self, path: str) -> None: ...

    @abstractmethod
    def removetree(self, path: str) -> None: ...

    @abstractmethod
    def movedir(self, src: str, dst: str, create: bool = False) -> None: ...

    def getsyspath(self, path: str) -> str:
        raise NoSysPath(f"the filesystem {self!r} has no system path")

    def close(self):
        self._closed = True

    def isclosed(self) -> bool:
        return self._closed

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # never swallow exceptions

    def check(self):
        if self._closed:
            raise FilesystemClosed(f"the filesystem {self!r} is closed")

    def opendir(self, path: str, *, factory: Type[SubFS] | None = None) -> SubFS:
        """Return a sub‑filesystem rooted at `path`."""
        if factory is None:
            from ._subfs import SubFS

            factory = SubFS
        return factory(self, path)

    def scandir(
        self, path: str, namespaces: Collection[str] | None = None
    ) -> Iterator[Info]:
        return (self.getinfo(f"{path}/{p}", namespaces) for p in self.listdir(path))

    @property
    def walk(self) -> BoundWalker:
        return BoundWalker(self)

    def readbytes(self, path: str) -> bytes:
        with self.open(path, "rb") as f:
            return f.read()

    def writebytes(self, path: str, data: bytes):
        with self.open(path, "wb") as f:
            f.write(data)

    def create(self, path: str, wipe: bool = False):
        if not wipe and self.exists(path):
            return False
        with self.open(path, "wb"):
            pass  # 'touch' empty file
        return True

    def copy(self, src_path: str, dst_path: str, overwrite=False):
        if not self.exists(src_path):
            raise ResourceNotFound(f"{src_path!r} does not exist")
        elif not self.isfile(src_path):
            raise FileExpected(f"path {src_path!r} should be a file")
        if not overwrite and self.exists(dst_path):
            raise DestinationExists(f"destination {dst_path!r} already exists")
        if not self.isdir(dirname(dst_path)):
            raise DirectoryExpected(f"path {dirname(dst_path)!r} should be a directory")
        copy_file(self, src_path, self, dst_path)

    def copydir(self, src_path: str, dst_path: str, create=False):
        if not create and not self.exists(dst_path):
            raise ResourceNotFound(f"{dst_path!r} does not exist")
        if not self.isdir(src_path):
            raise DirectoryExpected(f"path {src_path!r} should be a directory")
        copy_dir(self, src_path, self, dst_path)
<|endoftext|>from __future__ import annotations

import shutil
import tempfile

from ._errors import OperationFailed
from ._osfs import OSFS


class TempFS(OSFS):
    def __init__(self, auto_clean: bool = True, ignore_clean_errors: bool = True):
        self.auto_clean = auto_clean
        self.ignore_clean_errors = ignore_clean_errors
        self._temp_dir = tempfile.mkdtemp("__temp_fs__")
        self._cleaned = False
        super().__init__(self._temp_dir)

    def close(self):
        if self.auto_clean:
            self.clean()
        super().close()

    def clean(self):
        if self._cleaned:
            return

        try:
            shutil.rmtree(self._temp_dir)
        except Exception as e:
            if not self.ignore_clean_errors:
                raise OperationFailed(
                    f"failed to remove temporary directory: {self._temp_dir!r}"
                ) from e
        self._cleaned = True
<|endoftext|>import collections.abc
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
    IO,
)
import warnings
from io import BytesIO
from datetime import datetime
from base64 import b64encode, b64decode
from numbers import Integral
from types import SimpleNamespace
from functools import singledispatch

from fontTools.misc import etree

from fontTools.misc.textTools import tostr

# By default, we
#  - deserialize <data> elements as bytes and
#  - serialize bytes as <data> elements.
# Before, on Python 2, we
#  - deserialized <data> elements as plistlib.Data objects, in order to
#    distinguish them from the built-in str type (which is bytes on python2)
#  - serialized bytes as <string> elements (they must have only contained
#    ASCII characters in this case)
# You can pass use_builtin_types=[True|False] to the load/dump etc. functions
# to enforce a specific treatment.
# NOTE that unicode type always maps to <string> element, and plistlib.Data
# always maps to <data> element, regardless of use_builtin_types.
USE_BUILTIN_TYPES = True

XML_DECLARATION = b"""<?xml version='1.0' encoding='UTF-8'?>"""

PLIST_DOCTYPE = (
    b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
    b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
)


# Date should conform to a subset of ISO 8601:
# YYYY '-' MM '-' DD 'T' HH ':' MM ':' SS 'Z'
_date_parser = re.compile(
    r"(?P<year>\d\d\d\d)"
    r"(?:-(?P<month>\d\d)"
    r"(?:-(?P<day>\d\d)"
    r"(?:T(?P<hour>\d\d)"
    r"(?::(?P<minute>\d\d)"
    r"(?::(?P<second>\d\d))"
    r"?)?)?)?)?Z",
    re.ASCII,
)


def _date_from_string(s: str) -> datetime:
    order = ("year", "month", "day", "hour", "minute", "second")
    m = _date_parser.match(s)
    if m is None:
        raise ValueError(f"Expected ISO 8601 date string, but got '{s:r}'.")
    gd = m.groupdict()
    lst = []
    for key in order:
        val = gd[key]
        if val is None:
            break
        lst.append(int(val))
    # NOTE: mypy doesn't know that lst is 6 elements long.
    return datetime(*lst)  # type: ignore


def _date_to_string(d: datetime) -> str:
    return "%04d-%02d-%02dT%02d:%02d:%02dZ" % (
        d.year,
        d.month,
        d.day,
        d.hour,
        d.minute,
        d.second,
    )


class Data:
    """Represents binary data when ``use_builtin_types=False.``

    This class wraps binary data loaded from a plist file when the
    ``use_builtin_types`` argument to the loading function (:py:func:`fromtree`,
    :py:func:`load`, :py:func:`loads`) is false.

    The actual binary data is retrieved using the ``data`` attribute.
    """

    def __init__(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("Expected bytes, found %s" % type(data).__name__)
        self.data = data

    @classmethod
    def fromBase64(cls, data: Union[bytes, str]) -> "Data":
        return cls(b64decode(data))

    def asBase64(self, maxlinelength: int = 76, indent_level: int = 1) -> bytes:
        return _encode_base64(
            self.data, maxlinelength=maxlinelength, indent_level=indent_level
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.data == other.data
        elif isinstance(other, bytes):
            return self.data == other
        else:
            return NotImplemented

    def __repr__(self) -> str:
        return "%s(%s)" % (self.__class__.__name__, repr(self.data))


def _encode_base64(
    data: bytes, maxlinelength: Optional[int] = 76, indent_level: int = 1
) -> bytes:
    data = b64encode(data)
    if data and maxlinelength:
        # split into multiple lines right-justified to 'maxlinelength' chars
        indent = b"\n" + b"  " * indent_level
        max_length = max(16, maxlinelength - len(indent))
        chunks = []
        for i in range(0, len(data), max_length):
            chunks.append(indent)
            chunks.append(data[i : i + max_length])
        chunks.append(indent)
        data = b"".join(chunks)
    return data


# Mypy does not support recursive type aliases as of 0.782, Pylance does.
# https://github.com/python/mypy/issues/731
# https://devblogs.microsoft.com/python/pylance-introduces-five-new-features-that-enable-type-magic-for-python-developers/#1-support-for-recursive-type-aliases
PlistEncodable = Union[
    bool,
    bytes,
    Data,
    datetime,
    float,
    Integral,
    Mapping[str, Any],
    Sequence[Any],
    str,
]


class PlistTarget:
    """Event handler using the ElementTree Target API that can be
    passed to a XMLParser to produce property list objects from XML.
    It is based on the CPython plistlib module's _PlistParser class,
    but does not use the expat parser.

    >>> from fontTools.misc import etree
    >>> parser = etree.XMLParser(target=PlistTarget())
    >>> result = etree.XML(
    ...     "<dict>"
    ...     "    <key>something</key>"
    ...     "    <string>blah</string>"
    ...     "</dict>",
    ...     parser=parser)
    >>> result == {"something": "blah"}
    True

    Links:
    https://github.com/python/cpython/blob/main/Lib/plistlib.py
    http://lxml.de/parsing.html#the-target-parser-interface
    """

    def __init__(
        self,
        use_builtin_types: Optional[bool] = None,
        dict_type: Type[MutableMapping[str, Any]] = dict,
    ) -> None:
        self.stack: List[PlistEncodable] = []
        self.current_key: Optional[str] = None
        self.root: Optional[PlistEncodable] = None
        if use_builtin_types is None:
            self._use_builtin_types = USE_BUILTIN_TYPES
        else:
            if use_builtin_types is False:
                warnings.warn(
                    "Setting use_builtin_types to False is deprecated and will be "
                    "removed soon.",
                    DeprecationWarning,
                )
            self._use_builtin_types = use_builtin_types
        self._dict_type = dict_type

    def start(self, tag: str, attrib: Mapping[str, str]) -> None:
        self._data: List[str] = []
        handler = _TARGET_START_HANDLERS.get(tag)
        if handler is not None:
            handler(self)

    def end(self, tag: str) -> None:
        handler = _TARGET_END_HANDLERS.get(tag)
        if handler is not None:
            handler(self)

    def data(self, data: str) -> None:
        self._data.append(data)

    def close(self) -> PlistEncodable:
        if self.root is None:
            raise ValueError("No root set.")
        return self.root

    # helpers

    def add_object(self, value: PlistEncodable) -> None:
        if self.current_key is not None:
            stack_top = self.stack[-1]
            if not isinstance(stack_top, collections.abc.MutableMapping):
                raise ValueError("unexpected element: %r" % stack_top)
            stack_top[self.current_key] = value
            self.current_key = None
        elif not self.stack:
            # this is the root object
            self.root = value
        else:
            stack_top = self.stack[-1]
            if not isinstance(stack_top, list):
                raise ValueError("unexpected element: %r" % stack_top)
            stack_top.append(value)

    def get_data(self) -> str:
        data = "".join(self._data)
        self._data = []
        return data


# event handlers


def start_dict(self: PlistTarget) -> None:
    d = self._dict_type()
    self.add_object(d)
    self.stack.append(d)


def end_dict(self: PlistTarget) -> None:
    if self.current_key:
        raise ValueError("missing value for key '%s'" % self.current_key)
    self.stack.pop()


def end_key(self: PlistTarget) -> None:
    if self.current_key or not isinstance(self.stack[-1], collections.abc.Mapping):
        raise ValueError("unexpected key")
    self.current_key = self.get_data()


def start_array(self: PlistTarget) -> None:
    a: List[PlistEncodable] = []
    self.add_object(a)
    self.stack.append(a)


def end_array(self: PlistTarget) -> None:
    self.stack.pop()


def end_true(self: PlistTarget) -> None:
    self.add_object(True)


def end_false(self: PlistTarget) -> None:
    self.add_object(False)


def end_integer(self: PlistTarget) -> None:
    self.add_object(int(self.get_data()))


def end_real(self: PlistTarget) -> None:
    self.add_object(float(self.get_data()))


def end_string(self: PlistTarget) -> None:
    self.add_object(self.get_data())


def end_data(self: PlistTarget) -> None:
    if self._use_builtin_types:
        self.add_object(b64decode(self.get_data()))
    else:
        self.add_object(Data.fromBase64(self.get_data()))


def end_date(self: PlistTarget) -> None:
    self.add_object(_date_from_string(self.get_data()))


_TARGET_START_HANDLERS: Dict[str, Callable[[PlistTarget], None]] = {
    "dict": start_dict,
    "array": start_array,
}

_TARGET_END_HANDLERS: Dict[str, Callable[[PlistTarget], None]] = {
    "dict": end_dict,
    "array": end_array,
    "key": end_key,
    "true": end_true,
    "false": end_false,
    "integer": end_integer,
    "real": end_real,
    "string": end_string,
    "data": end_data,
    "date": end_date,
}


# functions to build element tree from plist data


def _string_element(value: str, ctx: SimpleNamespace) -> etree.Element:
    el = etree.Element("string")
    el.text = value
    return el


def _bool_element(value: bool, ctx: SimpleNamespace) -> etree.Element:
    if value:
        return etree.Element("true")
    return etree.Element("false")


def _integer_element(value: int, ctx: SimpleNamespace) -> etree.Element:
    if -1 << 63 <= value < 1 << 64:
        el = etree.Element("integer")
        el.text = "%d" % value
        return el
    raise OverflowError(value)


def _real_element(value: float, ctx: SimpleNamespace) -> etree.Element:
    el = etree.Element("real")
    el.text = repr(value)
    return el


def _dict_element(
    d: Mapping[str, PlistEncodable], ctx: SimpleNamespace
) -> etree.Element:
    el = etree.Element("dict")
    items = d.items()
    if ctx.sort_keys:
        items = sorted(items)  # type: ignore
    ctx.indent_level += 1
    for key, value in items:
        if not isinstance(key, str):
            if ctx.skipkeys:
                continue
            raise TypeError("keys must be strings")
        k = etree.SubElement(el, "key")
        k.text = tostr(key, "utf-8")
        el.append(_make_element(value, ctx))
    ctx.indent_level -= 1
    return el


def _array_element(
    array: Sequence[PlistEncodable], ctx: SimpleNamespace
) -> etree.Element:
    el = etree.Element("array")
    if len(array) == 0:
        return el
    ctx.indent_level += 1
    for value in array:
        el.append(_make_element(value, ctx))
    ctx.indent_level -= 1
    return el


def _date_element(date: datetime, ctx: SimpleNamespace) -> etree.Element:
    el = etree.Element("date")
    el.text = _date_to_string(date)
    return el


def _data_element(data: bytes, ctx: SimpleNamespace) -> etree.Element:
    el = etree.Element("data")
    # NOTE: mypy is confused about whether el.text should be str or bytes.
    el.text = _encode_base64(  # type: ignore
        data,
        maxlinelength=(76 if ctx.pretty_print else None),
        indent_level=ctx.indent_level,
    )
    return el


def _string_or_data_element(raw_bytes: bytes, ctx: SimpleNamespace) -> etree.Element:
    if ctx.use_builtin_types:
        return _data_element(raw_bytes, ctx)
    else:
        try:
            string = raw_bytes.decode(encoding="ascii", errors="strict")
        except UnicodeDecodeError:
            raise ValueError(
                "invalid non-ASCII bytes; use unicode string instead: %r" % raw_bytes
            )
        return _string_element(string, ctx)


# The following is probably not entirely correct. The signature should take `Any`
# and return `NoReturn`. At the time of this writing, neither mypy nor Pyright
# can deal with singledispatch properly and will apply the signature of the base
# function to all others. Being slightly dishonest makes it type-check and return
# usable typing information for the optimistic case.
@singledispatch
def _make_element(value: PlistEncodable, ctx: SimpleNamespace) -> etree.Element:
    raise TypeError("unsupported type: %s" % type(value))


_make_element.register(str)(_string_element)
_make_element.register(bool)(_bool_element)
_make_element.register(Integral)(_integer_element)
_make_element.register(float)(_real_element)
_make_element.register(collections.abc.Mapping)(_dict_element)
_make_element.register(list)(_array_element)
_make_element.register(tuple)(_array_element)
_make_element.register(datetime)(_date_element)
_make_element.register(bytes)(_string_or_data_element)
_make_element.register(bytearray)(_data_element)
_make_element.register(Data)(lambda v, ctx: _data_element(v.data, ctx))


# Public functions to create element tree from plist-compatible python
# data structures and viceversa, for use when (de)serializing GLIF xml.


def totree(
    value: PlistEncodable,
    sort_keys: bool = True,
    skipkeys: bool = False,
    use_builtin_types: Optional[bool] = None,
    pretty_print: bool = True,
    indent_level: int = 1,
) -> etree.Element:
    """Convert a value derived from a plist into an XML tree.

    Args:
        value: Any kind of value to be serialized to XML.
        sort_keys: Whether keys of dictionaries should be sorted.
        skipkeys (bool): Whether to silently skip non-string dictionary
            keys.
        use_builtin_types (bool): If true, byte strings will be
            encoded in Base-64 and wrapped in a ``data`` tag; if
            false, they will be either stored as ASCII strings or an
            exception raised if they cannot be decoded as such. Defaults
            to ``True`` if not present. Deprecated.
        pretty_print (bool): Whether to indent the output.
        indent_level (int): Level of indentation when serializing.

    Returns: an ``etree`` ``Element`` object.

    Raises:
        ``TypeError``
            if non-string dictionary keys are serialized
            and ``skipkeys`` is false.
        ``ValueError``
            if non-ASCII binary data is present
            and `use_builtin_types` is false.
    """
    if use_builtin_types is None:
        use_builtin_types = USE_BUILTIN_TYPES
    else:
        use_builtin_types = use_builtin_types
    context = SimpleNamespace(
        sort_keys=sort_keys,
        skipkeys=skipkeys,
        use_builtin_types=use_builtin_types,
        pretty_print=pretty_print,
        indent_level=indent_level,
    )
    return _make_element(value, context)


def fromtree(
    tree: etree.Element,
    use_builtin_types: Optional[bool] = None,
    dict_type: Type[MutableMapping[str, Any]] = dict,
) -> Any:
    """Convert an XML tree to a plist structure.

    Args:
        tree: An ``etree`` ``Element``.
        use_builtin_types: If True, binary data is deserialized to
            bytes strings. If False, it is wrapped in :py:class:`Data`
            objects. Defaults to True if not provided. Deprecated.
        dict_type: What type to use for dictionaries.

    Returns: An object (usually a dictionary).
    """
    target = PlistTarget(use_builtin_types=use_builtin_types, dict_type=dict_type)
    for action, element in etree.iterwalk(tree, events=("start", "end")):
        if action == "start":
            target.start(element.tag, element.attrib)
        elif action == "end":
            # if there are no children, parse the leaf's data
            if not len(element):
                # always pass str, not None
                target.data(element.text or "")
            target.end(element.tag)
    return target.close()


# python3 plistlib API


def load(
    fp: IO[bytes],
    use_builtin_types: Optional[bool] = None,
    dict_type: Type[MutableMapping[str, Any]] = dict,
) -> Any:
    """Load a plist file into an object.

    Args:
        fp: An opened file.
        use_builtin_types: If True, binary data is deserialized to
            bytes strings. If False, it is wrapped in :py:class:`Data`
            objects. Defaults to True if not provided. Deprecated.
        dict_type: What type to use for dictionaries.

    Returns:
        An object (usually a dictionary) representing the top level of
        the plist file.
    """

    if not hasattr(fp, "read"):
        raise AttributeError("'%s' object has no attribute 'read'" % type(fp).__name__)
    target = PlistTarget(use_builtin_types=use_builtin_types, dict_type=dict_type)
    parser = etree.XMLParser(target=target)
    result = etree.parse(fp, parser=parser)
    # lxml returns the target object directly, while ElementTree wraps
    # it as the root of an ElementTree object
    try:
        return result.getroot()
    except AttributeError:
        return result


def loads(
    value: bytes,
    use_builtin_types: Optional[bool] = None,
    dict_type: Type[MutableMapping[str, Any]] = dict,
) -> Any:
    """Load a plist file from a string into an object.

    Args:
        value: A bytes string containing a plist.
        use_builtin_types: If True, binary data is deserialized to
            bytes strings. If False, it is wrapped in :py:class:`Data`
            objects. Defaults to True if not provided. Deprecated.
        dict_type: What type to use for dictionaries.

    Returns:
        An object (usually a dictionary) representing the top level of
        the plist file.
    """

    fp = BytesIO(value)
    return load(fp, use_builtin_types=use_builtin_types, dict_type=dict_type)


def dump(
    value: PlistEncodable,
    fp: IO[bytes],
    sort_keys: bool = True,
    skipkeys: bool = False,
    use_builtin_types: Optional[bool] = None,
    pretty_print: bool = True,
) -> None:
    """Write a Python object to a plist file.

    Args:
        value: An object to write.
        fp: A file opened for writing.
        sort_keys (bool): Whether keys of dictionaries should be sorted.
        skipkeys (bool): Whether to silently skip non-string dictionary
            keys.
        use_builtin_types (bool): If true, byte strings will be
            encoded in Base-64 and wrapped in a ``data`` tag; if
            false, they will be either stored as ASCII strings or an
            exception raised if they cannot be represented. Defaults
        pretty_print (bool): Whether to indent the output.
        indent_level (int): Level of indentation when serializing.

    Raises:
        ``TypeError``
            if non-string dictionary keys are serialized
            and ``skipkeys`` is false.
        ``ValueError``
            if non-representable binary data is present
            and `use_builtin_types` is false.
    """

    if not hasattr(fp, "write"):
        raise AttributeError("'%s' object has no attribute 'write'" % type(fp).__name__)
    root = etree.Element("plist", version="1.0")
    el = totree(
        value,
        sort_keys=sort_keys,
        skipkeys=skipkeys,
        use_builtin_types=use_builtin_types,
        pretty_print=pretty_print,
    )
    root.append(el)
    tree = etree.ElementTree(root)
    # we write the doctype ourselves instead of using the 'doctype' argument
    # of 'write' method, becuse lxml will force adding a '\n' even when
    # pretty_print is False.
    if pretty_print:
        header = b"\n".join((XML_DECLARATION, PLIST_DOCTYPE, b""))
    else:
        header = XML_DECLARATION + PLIST_DOCTYPE
    fp.write(header)
    tree.write(  # type: ignore
        fp,
        encoding="utf-8",
        pretty_print=pretty_print,
        xml_declaration=False,
    )


def dumps(
    value: PlistEncodable,
    sort_keys: bool = True,
    skipkeys: bool = False,
    use_builtin_types: Optional[bool] = None,
    pretty_print: bool = True,
) -> bytes:
    """Write a Python object to a string in plist format.

    Args:
        value: An object to write.
        sort_keys (bool): Whether keys of dictionaries should be sorted.
        skipkeys (bool): Whether to silently skip non-string dictionary
            keys.
        use_builtin_types (bool): If true, byte strings will be
            encoded in Base-64 and wrapped in a ``data`` tag; if
            false, they will be either stored as strings or an
            exception raised if they cannot be represented. Defaults
        pretty_print (bool): Whether to indent the output.
        indent_level (int): Level of indentation when serializing.

    Returns:
        string: A plist representation of the Python object.

    Raises:
        ``TypeError``
            if non-string dictionary keys are serialized
            and ``skipkeys`` is false.
        ``ValueError``
            if non-representable binary data is present
            and `use_builtin_types` is false.
    """
    fp = BytesIO()
    dump(
        value,
        fp,
        sort_keys=sort_keys,
        skipkeys=skipkeys,
        use_builtin_types=use_builtin_types,
        pretty_print=pretty_print,
    )
    return fp.getvalue()
<|endoftext|>"""Private utility methods used by the subset modules"""


def _add_method(*clazzes):
    """Returns a decorator function that adds a new method to one or
    more classes."""

    def wrapper(method):
        done = []
        for clazz in clazzes:
            if clazz in done:
                continue  # Support multiple names of a clazz
            done.append(clazz)
            assert clazz.__name__ != "DefaultTable", "Oops, table class not found."
            assert not hasattr(
                clazz, method.__name__
            ), "Oops, class '%s' has method '%s'." % (clazz.__name__, method.__name__)
            setattr(clazz, method.__name__, method)
        return None

    return wrapper


def _uniq_sort(l):
    return sorted(set(l))
<|endoftext|># Copyright 2013 Google, Inc. All Rights Reserved.
#
# Google Author(s): Behdad Esfahbod

from __future__ import annotations

from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore, multiVarStore  # For monkey-patching
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor, makeName
from fontTools.unicodedata import mirrored
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType

__usage__ = "fonttools subset font-file [glyph...] [--option=value]..."

__doc__ = (
    """\
fonttools subset -- OpenType font subsetter and optimizer

fonttools subset is an OpenType font subsetter and optimizer, based on
fontTools. It accepts any TT- or CFF-flavored OpenType (.otf or .ttf)
or WOFF (.woff) font file. The subsetted glyph set is based on the
specified glyphs or characters, and specified OpenType layout features.

The tool also performs some size-reducing optimizations, aimed for using
subset fonts as webfonts.  Individual optimizations can be enabled or
disabled, and are enabled by default when they are safe.

Usage: """
    + __usage__
    + """

At least one glyph or one of --gids, --gids-file, --glyphs, --glyphs-file,
--text, --text-file, --unicodes, or --unicodes-file, must be specified.

Args:

font-file
  The input font file.
glyph
  Specify one or more glyph identifiers to include in the subset. Must be
  PS glyph names, or the special string '*' to keep the entire glyph set.

Initial glyph set specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options populate the initial glyph set. Same option can appear
multiple times, and the results are accummulated.

--gids=<NNN>[,<NNN>...]
  Specify comma/whitespace-separated list of glyph IDs or ranges as decimal
  numbers.  For example, --gids=10-12,14 adds glyphs with numbers 10, 11,
  12, and 14.

--gids-file=<path>
  Like --gids but reads from a file. Anything after a '#' on any line is
  ignored as comments.

--glyphs=<glyphname>[,<glyphname>...]
  Specify comma/whitespace-separated PS glyph names to add to the subset.
  Note that only PS glyph names are accepted, not gidNNN, U+XXXX, etc
  that are accepted on the command line.  The special string '*' will keep
  the entire glyph set.

--glyphs-file=<path>
  Like --glyphs but reads from a file. Anything after a '#' on any line
  is ignored as comments.

--text=<text>
  Specify characters to include in the subset, as UTF-8 string.

--text-file=<path>
  Like --text but reads from a file. Newline character are not added to
  the subset.

--unicodes=<XXXX>[,<XXXX>...]
  Specify comma/whitespace-separated list of Unicode codepoints or
  ranges as hex numbers, optionally prefixed with 'U+', 'u', etc.
  For example, --unicodes=41-5a,61-7a adds ASCII letters, so does
  the more verbose --unicodes=U+0041-005A,U+0061-007A.
  The special strings '*' will choose all Unicode characters mapped
  by the font.

--unicodes-file=<path>
  Like --unicodes, but reads from a file. Anything after a '#' on any
  line in the file is ignored as comments.

--ignore-missing-glyphs
  Do not fail if some requested glyphs or gids are not available in
  the font.

--no-ignore-missing-glyphs
  Stop and fail if some requested glyphs or gids are not available
  in the font. [default]

--ignore-missing-unicodes [default]
  Do not fail if some requested Unicode characters (including those
  indirectly specified using --text or --text-file) are not available
  in the font.

--no-ignore-missing-unicodes
  Stop and fail if some requested Unicode characters are not available
  in the font.
  Note the default discrepancy between ignoring missing glyphs versus
  unicodes.  This is for historical reasons and in the future
  --no-ignore-missing-unicodes might become default.

Other options
^^^^^^^^^^^^^

For the other options listed below, to see the current value of the option,
pass a value of '?' to it, with or without a '='. In some environments,
you might need to escape the question mark, like this: '--glyph-names\\?'.

Examples::

    $ fonttools subset --glyph-names?
    Current setting for 'glyph-names' is: False
    $ fonttools subset --name-IDs=?
    Current setting for 'name-IDs' is: [0, 1, 2, 3, 4, 5, 6]
    $ fonttools subset --hinting? --no-hinting --hinting?
    Current setting for 'hinting' is: True
    Current setting for 'hinting' is: False

Output options
^^^^^^^^^^^^^^

--output-file=<path>
  The output font file. If not specified, the subsetted font
  will be saved in as font-file.subset.

--flavor=<type>
  Specify flavor of output font file. May be 'woff' or 'woff2'.
  Note that WOFF2 requires the Brotli Python extension, available
  at https://github.com/google/brotli

--with-zopfli
  Use the Google Zopfli algorithm to compress WOFF. The output is 3-8 %
  smaller than pure zlib, but the compression speed is much slower.
  The Zopfli Python bindings are available at:
  https://pypi.python.org/pypi/zopfli

--harfbuzz-repacker
  By default, we serialize GPOS/GSUB using the HarfBuzz Repacker when
  uharfbuzz can be imported and is successful, otherwise fall back to
  the pure-python serializer. Set the option to force using the HarfBuzz
  Repacker (raises an error if uharfbuzz can't be found or fails).

--no-harfbuzz-repacker
  Always use the pure-python serializer even if uharfbuzz is available.

Glyph set expansion
^^^^^^^^^^^^^^^^^^^

These options control how additional glyphs are added to the subset.

--retain-gids
  Retain glyph indices; just empty glyphs not needed in-place.

--notdef-glyph
  Add the '.notdef' glyph to the subset (ie, keep it). [default]

--no-notdef-glyph
  Drop the '.notdef' glyph unless specified in the glyph set. This
  saves a few bytes, but is not possible for Postscript-flavored
  fonts, as those require '.notdef'. For TrueType-flavored fonts,
  this works fine as long as no unsupported glyphs are requested
  from the font.

--notdef-outline
  Keep the outline of '.notdef' glyph. The '.notdef' glyph outline is
  used when glyphs not supported by the font are to be shown. It is not
  needed otherwise.

--no-notdef-outline
  When including a '.notdef' glyph, remove its outline. This saves
  a few bytes. [default]

--recommended-glyphs
  Add glyphs 0, 1, 2, and 3 to the subset, as recommended for
  TrueType-flavored fonts: '.notdef', 'NULL' or '.null', 'CR', 'space'.
  Some legacy software might require this, but no modern system does.

--no-recommended-glyphs
  Do not add glyphs 0, 1, 2, and 3 to the subset, unless specified in
  glyph set. [default]

--no-layout-closure
  Do not expand glyph set to add glyphs produced by OpenType layout
  features.  Instead, OpenType layout features will be subset to only
  rules that are relevant to the otherwise-specified glyph set.

--layout-features[+|-]=<feature>[,<feature>...]
  Specify (=), add to (+=) or exclude from (-=) the comma-separated
  set of OpenType layout feature tags that will be preserved.
  Glyph variants used by the preserved features are added to the
  specified subset glyph set. By default, 'calt', 'ccmp', 'clig', 'curs',
  'dnom', 'frac', 'kern', 'liga', 'locl', 'mark', 'mkmk', 'numr', 'rclt',
  'rlig', 'rvrn', and all features required for script shaping are
  preserved. To see the full list, try '--layout-features=?'.
  Use '*' to keep all features.
  Multiple --layout-features options can be provided if necessary.
  Examples:

    --layout-features+=onum,pnum,ss01
        * Keep the default set of features and 'onum', 'pnum', 'ss01'.
    --layout-features-='mark','mkmk'
        * Keep the default set of features but drop 'mark' and 'mkmk'.
    --layout-features='kern'
        * Only keep the 'kern' feature, drop all others.
    --layout-features=''
        * Drop all features.
    --layout-features='*'
        * Keep all features.
    --layout-features+=aalt --layout-features-=vrt2
        * Keep default set of features plus 'aalt', but drop 'vrt2'.

--layout-scripts[+|-]=<script>[,<script>...]
  Specify (=), add to (+=) or exclude from (-=) the comma-separated
  set of OpenType layout script tags that will be preserved. LangSys tags
  can be appended to script tag, separated by '.', for example:
  'arab.dflt,arab.URD,latn.TRK'. By default all scripts are retained ('*').

Hinting options
^^^^^^^^^^^^^^^

--hinting
  Keep hinting [default]

--no-hinting
  Drop glyph-specific hinting and font-wide hinting tables, as well
  as remove hinting-related bits and pieces from other tables (eg. GPOS).
  See --hinting-tables for list of tables that are dropped by default.
  Instructions and hints are stripped from 'glyf' and 'CFF ' tables
  respectively. This produces (sometimes up to 30%) smaller fonts that
  are suitable for extremely high-resolution systems, like high-end
  mobile devices and retina displays.

Optimization options
^^^^^^^^^^^^^^^^^^^^

--desubroutinize
  Remove CFF use of subroutinizes.  Subroutinization is a way to make CFF
  fonts smaller.  For small subsets however, desubroutinizing might make
  the font smaller.  It has even been reported that desubroutinized CFF
  fonts compress better (produce smaller output) WOFF and WOFF2 fonts.
  Also see note under --no-hinting.

--no-desubroutinize [default]
  Leave CFF subroutinizes as is, only throw away unused subroutinizes.

Font table options
^^^^^^^^^^^^^^^^^^

--drop-tables[+|-]=<table>[,<table>...]
  Specify (=), add to (+=) or exclude from (-=) the comma-separated
  set of tables that will be be dropped.
  By default, the following tables are dropped:
  'JSTF', 'DSIG', 'EBDT', 'EBLC', 'EBSC', 'PCLT', 'LTSH'
  and Graphite tables: 'Feat', 'Glat', 'Gloc', 'Silf', 'Sill'.
  The tool will attempt to subset the remaining tables.

  Examples:

  --drop-tables-=BASE
      * Drop the default set of tables but keep 'BASE'.

  --drop-tables+=GSUB
      * Drop the default set of tables and 'GSUB'.

  --drop-tables=DSIG
      * Only drop the 'DSIG' table, keep all others.

  --drop-tables=
      * Keep all tables.

--no-subset-tables+=<table>[,<table>...]
  Add to the set of tables that will not be subsetted.
  By default, the following tables are included in this list, as
  they do not need subsetting (ignore the fact that 'loca' is listed
  here): 'gasp', 'head', 'hhea', 'maxp', 'vhea', 'OS/2', 'loca', 'name',
  'cvt ', 'fpgm', 'prep', 'VMDX', 'DSIG', 'CPAL', 'MVAR', 'cvar', 'STAT'.
  By default, tables that the tool does not know how to subset and are not
  specified here will be dropped from the font, unless --passthrough-tables
  option is passed.

  Example:

   --no-subset-tables+=FFTM
      * Keep 'FFTM' table in the font by preventing subsetting.

--passthrough-tables
  Do not drop tables that the tool does not know how to subset.

--no-passthrough-tables
  Tables that the tool does not know how to subset and are not specified
  in --no-subset-tables will be dropped from the font. [default]

--hinting-tables[-]=<table>[,<table>...]
  Specify (=), add to (+=) or exclude from (-=) the list of font-wide
  hinting tables that will be dropped if --no-hinting is specified.

  Examples:

  --hinting-tables-=VDMX
      * Drop font-wide hinting tables except 'VDMX'.
  --hinting-tables=
      * Keep all font-wide hinting tables (but strip hints from glyphs).

--legacy-kern
  Keep TrueType 'kern' table even when OpenType 'GPOS' is available.

--no-legacy-kern
  Drop TrueType 'kern' table if OpenType 'GPOS' is available. [default]

Font naming options
^^^^^^^^^^^^^^^^^^^

These options control what is retained in the 'name' table. For numerical
codes, see: http://www.microsoft.com/typography/otspec/name.htm

--name-IDs[+|-]=<nameID>[,<nameID>...]
  Specify (=), add to (+=) or exclude from (-=) the set of 'name' table
  entry nameIDs that will be preserved. By default, only nameIDs between 0
  and 6 are preserved, the rest are dropped. Use '*' to keep all entries.

  Examples:

  --name-IDs+=7,8,9
      * Also keep Trademark, Manufacturer and Designer name entries.
  --name-IDs=
      * Drop all 'name' table entries.
  --name-IDs=*
      * keep all 'name' table entries

--name-legacy
  Keep legacy (non-Unicode) 'name' table entries (0.x, 1.x etc.).
  XXX Note: This might be needed for some fonts that have no Unicode name
  entires for English. See: https://github.com/fonttools/fonttools/issues/146

--no-name-legacy
  Drop legacy (non-Unicode) 'name' table entries [default]

--name-languages[+|-]=<langID>[,<langID>]
  Specify (=), add to (+=) or exclude from (-=) the set of 'name' table
  langIDs that will be preserved. By default only records with langID
  0x0409 (English) are preserved. Use '*' to keep all langIDs.

--obfuscate-names
  Make the font unusable as a system font by replacing name IDs 1, 2, 3, 4,
  and 6 with dummy strings (it is still fully functional as webfont).

Glyph naming and encoding options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

--glyph-names
  Keep PS glyph names in TT-flavored fonts. In general glyph names are
  not needed for correct use of the font. However, some PDF generators
  and PDF viewers might rely on glyph names to extract Unicode text
  from PDF documents.
--no-glyph-names
  Drop PS glyph names in TT-flavored fonts, by using 'post' table
  version 3.0. [default]
--legacy-cmap
  Keep the legacy 'cmap' subtables (0.x, 1.x, 4.x etc.).
--no-legacy-cmap
  Drop the legacy 'cmap' subtables. [default]
--symbol-cmap
  Keep the 3.0 symbol 'cmap'.
--no-symbol-cmap
  Drop the 3.0 symbol 'cmap'. [default]

Other font-specific options
^^^^^^^^^^^^^^^^^^^^^^^^^^^

--recalc-bounds
    Recalculate font bounding boxes.
--no-recalc-bounds
    Keep original font bounding boxes. This is faster and still safe
    for all practical purposes. [default]
--recalc-timestamp
    Set font 'modified' timestamp to current time.
--no-recalc-timestamp
    Do not modify font 'modified' timestamp. [default]
--canonical-order
    Order tables as recommended in the OpenType standard. This is not
    required by the standard, nor by any known implementation.
--no-canonical-order
    Keep original order of font tables. This is faster. [default]
--prune-unicode-ranges
    Update the 'OS/2 ulUnicodeRange*' bits after subsetting. The Unicode
    ranges defined in the OpenType specification v1.7 are intersected with
    the Unicode codepoints specified in the font's Unicode 'cmap' subtables:
    when no overlap is found, the bit will be switched off. However, it will
    *not* be switched on if an intersection is found.  [default]
--no-prune-unicode-ranges
    Don't change the 'OS/2 ulUnicodeRange*' bits.
--prune-codepage-ranges
    Update the 'OS/2 ulCodePageRange*' bits after subsetting.  [default]
--no-prune-codepage-ranges
    Don't change the 'OS/2 ulCodePageRange*' bits.
--recalc-average-width
    Update the 'OS/2 xAvgCharWidth' field after subsetting.
--no-recalc-average-width
    Don't change the 'OS/2 xAvgCharWidth' field. [default]
--recalc-max-context
    Update the 'OS/2 usMaxContext' field after subsetting.
--no-recalc-max-context
    Don't change the 'OS/2 usMaxContext' field. [default]
--font-number=<number>
    Select font number for TrueType Collection (.ttc/.otc), starting from 0.
--pretty-svg
    When subsetting SVG table, use lxml pretty_print=True option to indent
    the XML output (only recommended for debugging purposes).

Application options
^^^^^^^^^^^^^^^^^^^

--verbose
    Display verbose information of the subsetting process.
--timing
    Display detailed timing information of the subsetting process.
--xml
    Display the TTX XML representation of subsetted font.

Example
^^^^^^^

Produce a subset containing the characters ' !"#$%' without performing
size-reducing optimizations::

  $ fonttools subset font.ttf --unicodes="U+0020-0025" \\
    --layout-features=* --glyph-names --symbol-cmap --legacy-cmap \\
    --notdef-glyph --notdef-outline --recommended-glyphs \\
    --name-IDs=* --name-legacy --name-languages=*
"""
)


log = logging.getLogger("fontTools.subset")


def _log_glyphs(self, glyphs, font=None):
    self.info("Glyph names: %s", sorted(glyphs))
    if font:
        reverseGlyphMap = font.getReverseGlyphMap()
        self.info("Glyph IDs:   %s", sorted(reverseGlyphMap[g] for g in glyphs))


# bind "glyphs" function to 'log' object
log.glyphs = MethodType(_log_glyphs, log)

# I use a different timing channel so I can configure it separately from the
# main module's logger
timer = Timer(logger=logging.getLogger("fontTools.subset.timer"))


def _dict_subset(d, glyphs):
    return {g: d[g] for g in glyphs}


def _list_subset(l, indices):
    count = len(l)
    return [l[i] for i in indices if i < count]


@_add_method(otTables.Coverage)
def intersect(self, glyphs):
    """Returns ascending list of matching coverage values."""
    return [i for i, g in enumerate(self.glyphs) if g in glyphs]


@_add_method(otTables.Coverage)
def intersect_glyphs(self, glyphs):
    """Returns set of intersecting glyphs."""
    return set(g for g in self.glyphs if g in glyphs)


@_add_method(otTables.Coverage)
def subset(self, glyphs):
    """Returns ascending list of remaining coverage values."""
    indices = self.intersect(glyphs)
    self.glyphs = [g for g in self.glyphs if g in glyphs]
    return indices


@_add_method(otTables.Coverage)
def remap(self, coverage_map):
    """Remaps coverage."""
    self.glyphs = [self.glyphs[i] for i in coverage_map]


@_add_method(otTables.ClassDef)
def intersect(self, glyphs):
    """Returns ascending list of matching class values."""
    return _uniq_sort(
        ([0] if any(g not in self.classDefs for g in glyphs) else [])
        + [v for g, v in self.classDefs.items() if g in glyphs]
    )


@_add_method(otTables.ClassDef)
def intersect_class(self, glyphs, klass):
    """Returns set of glyphs matching class."""
    if klass == 0:
        return set(g for g in glyphs if g not in self.classDefs)
    return set(g for g, v in self.classDefs.items() if v == klass and g in glyphs)


@_add_method(otTables.ClassDef)
def subset(self, glyphs, remap=False, useClass0=True):
    """Returns ascending list of remaining classes."""
    self.classDefs = {g: v for g, v in self.classDefs.items() if g in glyphs}
    # Note: while class 0 has the special meaning of "not matched",
    # if no glyph will ever /not match/, we can optimize class 0 out too.
    # Only do this if allowed.
    indices = _uniq_sort(
        (
            [0]
            if ((not useClass0) or any(g not in self.classDefs for g in glyphs))
            else []
        )
        + list(self.classDefs.values())
    )
    if remap:
        self.remap(indices)
    return indices


@_add_method(otTables.ClassDef)
def remap(self, class_map):
    """Remaps classes."""
    self.classDefs = {g: class_map.index(v) for g, v in self.classDefs.items()}


@_add_method(otTables.SingleSubst)
def closure_glyphs(self, s, cur_glyphs):
    s.glyphs.update(v for g, v in self.mapping.items() if g in cur_glyphs)


@_add_method(otTables.SingleSubst)
def subset_glyphs(self, s):
    self.mapping = {
        g: v for g, v in self.mapping.items() if g in s.glyphs and v in s.glyphs
    }
    return bool(self.mapping)


@_add_method(otTables.MultipleSubst)
def closure_glyphs(self, s, cur_glyphs):
    for glyph, subst in self.mapping.items():
        if glyph in cur_glyphs:
            s.glyphs.update(subst)


@_add_method(otTables.MultipleSubst)
def subset_glyphs(self, s):
    self.mapping = {
        g: v
        for g, v in self.mapping.items()
        if g in s.glyphs and all(sub in s.glyphs for sub in v)
    }
    return bool(self.mapping)


@_add_method(otTables.AlternateSubst)
def closure_glyphs(self, s, cur_glyphs):
    s.glyphs.update(*(vlist for g, vlist in self.alternates.items() if g in cur_glyphs))


@_add_method(otTables.AlternateSubst)
def subset_glyphs(self, s):
    self.alternates = {
        g: [v for v in vlist if v in s.glyphs]
        for g, vlist in self.alternates.items()
        if g in s.glyphs and any(v in s.glyphs for v in vlist)
    }
    return bool(self.alternates)


@_add_method(otTables.LigatureSubst)
def closure_glyphs(self, s, cur_glyphs):
    s.glyphs.update(
        *(
            [seq.LigGlyph for seq in seqs if all(c in s.glyphs for c in seq.Component)]
            for g, seqs in self.ligatures.items()
            if g in cur_glyphs
        )
    )


@_add_method(otTables.LigatureSubst)
def subset_glyphs(self, s):
    self.ligatures = {g: v for g, v in self.ligatures.items() if g in s.glyphs}
    self.ligatures = {
        g: [
            seq
            for seq in seqs
            if seq.LigGlyph in s.glyphs and all(c in s.glyphs for c in seq.Component)
        ]
        for g, seqs in self.ligatures.items()
    }
    self.ligatures = {g: v for g, v in self.ligatures.items() if v}
    return bool(self.ligatures)


@_add_method(otTables.ReverseChainSingleSubst)
def closure_glyphs(self, s, cur_glyphs):
    if self.Format == 1:
        indices = self.Coverage.intersect(cur_glyphs)
        if not indices or not all(
            c.intersect(s.glyphs)
            for c in self.LookAheadCoverage + self.BacktrackCoverage
        ):
            return
        s.glyphs.update(self.Substitute[i] for i in indices)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.ReverseChainSingleSubst)
def subset_glyphs(self, s):
    if self.Format == 1:
        indices = self.Coverage.subset(s.glyphs)
        self.Substitute = _list_subset(self.Substitute, indices)
        # Now drop rules generating glyphs we don't want
        indices = [i for i, sub in enumerate(self.Substitute) if sub in s.glyphs]
        self.Substitute = _list_subset(self.Substitute, indices)
        self.Coverage.remap(indices)
        self.GlyphCount = len(self.Substitute)
        return bool(
            self.GlyphCount
            and all(
                c.subset(s.glyphs)
                for c in self.LookAheadCoverage + self.BacktrackCoverage
            )
        )
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.Device)
def is_hinting(self):
    return self.DeltaFormat in (1, 2, 3)


@_add_method(otTables.ValueRecord)
def prune_hints(self):
    for name in ["XPlaDevice", "YPlaDevice", "XAdvDevice", "YAdvDevice"]:
        v = getattr(self, name, None)
        if v is not None and v.is_hinting():
            delattr(self, name)


@_add_method(otTables.SinglePos)
def subset_glyphs(self, s):
    if self.Format == 1:
        return len(self.Coverage.subset(s.glyphs))
    elif self.Format == 2:
        indices = self.Coverage.subset(s.glyphs)
        values = self.Value
        count = len(values)
        self.Value = [values[i] for i in indices if i < count]
        self.ValueCount = len(self.Value)
        return bool(self.ValueCount)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.SinglePos)
def prune_post_subset(self, font, options):
    if self.Value is None:
        assert self.ValueFormat == 0
        return True

    # Shrink ValueFormat
    if self.Format == 1:
        if not options.hinting:
            self.Value.prune_hints()
        self.ValueFormat = self.Value.getEffectiveFormat()
    elif self.Format == 2:
        if None in self.Value:
            assert self.ValueFormat == 0
            assert all(v is None for v in self.Value)
        else:
            if not options.hinting:
                for v in self.Value:
                    v.prune_hints()
            self.ValueFormat = reduce(
                int.__or__, [v.getEffectiveFormat() for v in self.Value], 0
            )

    # Downgrade to Format 1 if all ValueRecords are the same
    if self.Format == 2 and all(v == self.Value[0] for v in self.Value):
        self.Format = 1
        self.Value = self.Value[0] if self.ValueFormat != 0 else None
        del self.ValueCount

    return True


@_add_method(otTables.PairPos)
def subset_glyphs(self, s):
    if self.Format == 1:
        indices = self.Coverage.subset(s.glyphs)
        pairs = self.PairSet
        count = len(pairs)
        self.PairSet = [pairs[i] for i in indices if i < count]
        for p in self.PairSet:
            p.PairValueRecord = [
                r for r in p.PairValueRecord if r.SecondGlyph in s.glyphs
            ]
            p.PairValueCount = len(p.PairValueRecord)
        # Remove empty pairsets
        indices = [i for i, p in enumerate(self.PairSet) if p.PairValueCount]
        self.Coverage.remap(indices)
        self.PairSet = _list_subset(self.PairSet, indices)
        self.PairSetCount = len(self.PairSet)
        return bool(self.PairSetCount)
    elif self.Format == 2:
        class1_map = [
            c
            for c in self.ClassDef1.subset(
                s.glyphs.intersection(self.Coverage.glyphs), remap=True
            )
            if c < self.Class1Count
        ]
        class2_map = [
            c
            for c in self.ClassDef2.subset(s.glyphs, remap=True, useClass0=False)
            if c < self.Class2Count
        ]
        self.Class1Record = [self.Class1Record[i] for i in class1_map]
        for c in self.Class1Record:
            c.Class2Record = [c.Class2Record[i] for i in class2_map]
        self.Class1Count = len(class1_map)
        self.Class2Count = len(class2_map)
        # If only Class2 0 left, no need to keep anything.
        return bool(
            self.Class1Count
            and (self.Class2Count > 1)
            and self.Coverage.subset(s.glyphs)
        )
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.PairPos)
def prune_post_subset(self, font, options):
    if not options.hinting:
        attr1, attr2 = {
            1: ("PairSet", "PairValueRecord"),
            2: ("Class1Record", "Class2Record"),
        }[self.Format]

        self.ValueFormat1 = self.ValueFormat2 = 0
        for row in getattr(self, attr1):
            for r in getattr(row, attr2):
                if r.Value1:
                    r.Value1.prune_hints()
                    self.ValueFormat1 |= r.Value1.getEffectiveFormat()
                if r.Value2:
                    r.Value2.prune_hints()
                    self.ValueFormat2 |= r.Value2.getEffectiveFormat()

    return bool(self.ValueFormat1 | self.ValueFormat2)


@_add_method(otTables.CursivePos)
def subset_glyphs(self, s):
    if self.Format == 1:
        indices = self.Coverage.subset(s.glyphs)
        records = self.EntryExitRecord
        count = len(records)
        self.EntryExitRecord = [records[i] for i in indices if i < count]
        self.EntryExitCount = len(self.EntryExitRecord)
        return bool(self.EntryExitCount)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.Anchor)
def prune_hints(self):
    if self.Format == 2:
        self.Format = 1
    elif self.Format == 3:
        for name in ("XDeviceTable", "YDeviceTable"):
            v = getattr(self, name, None)
            if v is not None and v.is_hinting():
                setattr(self, name, None)
        if self.XDeviceTable is None and self.YDeviceTable is None:
            self.Format = 1


@_add_method(otTables.CursivePos)
def prune_post_subset(self, font, options):
    if not options.hinting:
        for rec in self.EntryExitRecord:
            if rec.EntryAnchor:
                rec.EntryAnchor.prune_hints()
            if rec.ExitAnchor:
                rec.ExitAnchor.prune_hints()
    return True


@_add_method(otTables.MarkBasePos)
def subset_glyphs(self, s):
    if self.Format == 1:
        mark_indices = self.MarkCoverage.subset(s.glyphs)
        self.MarkArray.MarkRecord = _list_subset(
            self.MarkArray.MarkRecord, mark_indices
        )
        self.MarkArray.MarkCount = len(self.MarkArray.MarkRecord)
        class_indices = _uniq_sort(v.Class for v in self.MarkArray.MarkRecord)

        intersect_base_indices = self.BaseCoverage.intersect(s.glyphs)
        base_records = self.BaseArray.BaseRecord
        num_base_records = len(base_records)
        base_indices = [
            i
            for i in intersect_base_indices
            if i < num_base_records
            and any(base_records[i].BaseAnchor[j] is not None for j in class_indices)
        ]
        if not base_indices:
            return False

        self.BaseCoverage.remap(base_indices)
        self.BaseArray.BaseRecord = _list_subset(
            self.BaseArray.BaseRecord, base_indices
        )
        self.BaseArray.BaseCount = len(self.BaseArray.BaseRecord)
        # Prune empty classes
        self.ClassCount = len(class_indices)
        for m in self.MarkArray.MarkRecord:
            m.Class = class_indices.index(m.Class)
        for b in self.BaseArray.BaseRecord:
            b.BaseAnchor = _list_subset(b.BaseAnchor, class_indices)
        return bool(
            self.ClassCount and self.MarkArray.MarkCount and self.BaseArray.BaseCount
        )
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.MarkBasePos)
def prune_post_subset(self, font, options):
    if not options.hinting:
        for m in self.MarkArray.MarkRecord:
            if m.MarkAnchor:
                m.MarkAnchor.prune_hints()
        for b in self.BaseArray.BaseRecord:
            for a in b.BaseAnchor:
                if a:
                    a.prune_hints()
    return True


@_add_method(otTables.MarkLigPos)
def subset_glyphs(self, s):
    if self.Format == 1:
        mark_indices = self.MarkCoverage.subset(s.glyphs)
        self.MarkArray.MarkRecord = _list_subset(
            self.MarkArray.MarkRecord, mark_indices
        )
        self.MarkArray.MarkCount = len(self.MarkArray.MarkRecord)
        class_indices = _uniq_sort(v.Class for v in self.MarkArray.MarkRecord)

        intersect_ligature_indices = self.LigatureCoverage.intersect(s.glyphs)
        ligature_array = self.LigatureArray.LigatureAttach
        num_ligatures = self.LigatureArray.LigatureCount

        ligature_indices = [
            i
            for i in intersect_ligature_indices
            if i < num_ligatures
            and any(
                any(component.LigatureAnchor[j] is not None for j in class_indices)
                for component in ligature_array[i].ComponentRecord
            )
        ]

        if not ligature_indices:
            return False

        self.LigatureCoverage.remap(ligature_indices)
        self.LigatureArray.LigatureAttach = _list_subset(
            self.LigatureArray.LigatureAttach, ligature_indices
        )
        self.LigatureArray.LigatureCount = len(self.LigatureArray.LigatureAttach)
        # Prune empty classes
        self.ClassCount = len(class_indices)
        for m in self.MarkArray.MarkRecord:
            m.Class = class_indices.index(m.Class)
        for l in self.LigatureArray.LigatureAttach:
            if l is None:
                continue
            for c in l.ComponentRecord:
                c.LigatureAnchor = _list_subset(c.LigatureAnchor, class_indices)
        return bool(
            self.ClassCount
            and self.MarkArray.MarkCount
            and self.LigatureArray.LigatureCount
        )
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.MarkLigPos)
def prune_post_subset(self, font, options):
    if not options.hinting:
        for m in self.MarkArray.MarkRecord:
            if m.MarkAnchor:
                m.MarkAnchor.prune_hints()
        for l in self.LigatureArray.LigatureAttach:
            if l is None:
                continue
            for c in l.ComponentRecord:
                for a in c.LigatureAnchor:
                    if a:
                        a.prune_hints()
    return True


@_add_method(otTables.MarkMarkPos)
def subset_glyphs(self, s):
    if self.Format == 1:
        mark1_indices = self.Mark1Coverage.subset(s.glyphs)
        self.Mark1Array.MarkRecord = _list_subset(
            self.Mark1Array.MarkRecord, mark1_indices
        )
        self.Mark1Array.MarkCount = len(self.Mark1Array.MarkRecord)
        class_indices = _uniq_sort(v.Class for v in self.Mark1Array.MarkRecord)

        intersect_mark2_indices = self.Mark2Coverage.intersect(s.glyphs)
        mark2_records = self.Mark2Array.Mark2Record
        num_mark2_records = len(mark2_records)
        mark2_indices = [
            i
            for i in intersect_mark2_indices
            if i < num_mark2_records
            and any(mark2_records[i].Mark2Anchor[j] is not None for j in class_indices)
        ]
        if not mark2_indices:
            return False

        self.Mark2Coverage.remap(mark2_indices)
        self.Mark2Array.Mark2Record = _list_subset(
            self.Mark2Array.Mark2Record, mark2_indices
        )
        self.Mark2Array.MarkCount = len(self.Mark2Array.Mark2Record)
        # Prune empty classes
        self.ClassCount = len(class_indices)
        for m in self.Mark1Array.MarkRecord:
            m.Class = class_indices.index(m.Class)
        for b in self.Mark2Array.Mark2Record:
            b.Mark2Anchor = _list_subset(b.Mark2Anchor, class_indices)
        return bool(
            self.ClassCount and self.Mark1Array.MarkCount and self.Mark2Array.MarkCount
        )
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.MarkMarkPos)
def prune_post_subset(self, font, options):
    if not options.hinting:
        for m in self.Mark1Array.MarkRecord:
            if m.MarkAnchor:
                m.MarkAnchor.prune_hints()
        for b in self.Mark2Array.Mark2Record:
            for m in b.Mark2Anchor:
                if m:
                    m.prune_hints()
    return True


@_add_method(
    otTables.SingleSubst,
    otTables.MultipleSubst,
    otTables.AlternateSubst,
    otTables.LigatureSubst,
    otTables.ReverseChainSingleSubst,
    otTables.SinglePos,
    otTables.PairPos,
    otTables.CursivePos,
    otTables.MarkBasePos,
    otTables.MarkLigPos,
    otTables.MarkMarkPos,
)
def subset_lookups(self, lookup_indices):
    pass


@_add_method(
    otTables.SingleSubst,
    otTables.MultipleSubst,
    otTables.AlternateSubst,
    otTables.LigatureSubst,
    otTables.ReverseChainSingleSubst,
    otTables.SinglePos,
    otTables.PairPos,
    otTables.CursivePos,
    otTables.MarkBasePos,
    otTables.MarkLigPos,
    otTables.MarkMarkPos,
)
def collect_lookups(self):
    return []


@_add_method(
    otTables.SingleSubst,
    otTables.MultipleSubst,
    otTables.AlternateSubst,
    otTables.LigatureSubst,
    otTables.ReverseChainSingleSubst,
    otTables.ContextSubst,
    otTables.ChainContextSubst,
    otTables.ContextPos,
    otTables.ChainContextPos,
)
def prune_post_subset(self, font, options):
    return True


@_add_method(
    otTables.SingleSubst, otTables.AlternateSubst, otTables.ReverseChainSingleSubst
)
def may_have_non_1to1(self):
    return False


@_add_method(
    otTables.MultipleSubst,
    otTables.LigatureSubst,
    otTables.ContextSubst,
    otTables.ChainContextSubst,
)
def may_have_non_1to1(self):
    return True


@_add_method(
    otTables.ContextSubst,
    otTables.ChainContextSubst,
    otTables.ContextPos,
    otTables.ChainContextPos,
)
def __subset_classify_context(self):
    class ContextHelper(object):
        def __init__(self, klass, Format):
            if klass.__name__.endswith("Subst"):
                Typ = "Sub"
                Type = "Subst"
            else:
                Typ = "Pos"
                Type = "Pos"
            if klass.__name__.startswith("Chain"):
                Chain = "Chain"
                InputIdx = 1
                DataLen = 3
            else:
                Chain = ""
                InputIdx = 0
                DataLen = 1
            ChainTyp = Chain + Typ

            self.Typ = Typ
            self.Type = Type
            self.Chain = Chain
            self.ChainTyp = ChainTyp
            self.InputIdx = InputIdx
            self.DataLen = DataLen

            self.LookupRecord = Type + "LookupRecord"

            if Format == 1:
                Coverage = lambda r: r.Coverage
                ChainCoverage = lambda r: r.Coverage
                ContextData = lambda r: (None,)
                ChainContextData = lambda r: (None, None, None)
                SetContextData = None
                SetChainContextData = None
                RuleData = lambda r: (r.Input,)
                ChainRuleData = lambda r: (r.Backtrack, r.Input, r.LookAhead)

                def SetRuleData(r, d):
                    (r.Input,) = d
                    (r.GlyphCount,) = (len(x) + 1 for x in d)

                def ChainSetRuleData(r, d):
                    r.Backtrack, r.Input, r.LookAhead = d
                    (
                        r.BacktrackGlyphCount,
                        r.InputGlyphCount,
                        r.LookAheadGlyphCount,
                    ) = (len(d[0]), len(d[1]) + 1, len(d[2]))

            elif Format == 2:
                Coverage = lambda r: r.Coverage
                ChainCoverage = lambda r: r.Coverage
                ContextData = lambda r: (r.ClassDef,)
                ChainContextData = lambda r: (
                    r.BacktrackClassDef,
                    r.InputClassDef,
                    r.LookAheadClassDef,
                )

                def SetContextData(r, d):
                    (r.ClassDef,) = d

                def SetChainContextData(r, d):
                    r.BacktrackClassDef, r.InputClassDef, r.LookAheadClassDef = d

                RuleData = lambda r: (r.Class,)
                ChainRuleData = lambda r: (r.Backtrack, r.Input, r.LookAhead)

                def SetRuleData(r, d):
                    (r.Class,) = d
                    (r.GlyphCount,) = (len(x) + 1 for x in d)

                def ChainSetRuleData(r, d):
                    r.Backtrack, r.Input, r.LookAhead = d
                    (
                        r.BacktrackGlyphCount,
                        r.InputGlyphCount,
                        r.LookAheadGlyphCount,
                    ) = (len(d[0]), len(d[1]) + 1, len(d[2]))

            elif Format == 3:
                Coverage = lambda r: r.Coverage[0]
                ChainCoverage = lambda r: r.InputCoverage[0]
                ContextData = None
                ChainContextData = None
                SetContextData = None
                SetChainContextData = None
                RuleData = lambda r: r.Coverage
                ChainRuleData = lambda r: (
                    r.BacktrackCoverage + r.InputCoverage + r.LookAheadCoverage
                )

                def SetRuleData(r, d):
                    (r.Coverage,) = d
                    (r.GlyphCount,) = (len(x) for x in d)

                def ChainSetRuleData(r, d):
                    r.BacktrackCoverage, r.InputCoverage, r.LookAheadCoverage = d
                    (
                        r.BacktrackGlyphCount,
                        r.InputGlyphCount,
                        r.LookAheadGlyphCount,
                    ) = (len(x) for x in d)

            else:
                assert 0, "unknown format: %s" % Format

            if Chain:
                self.Coverage = ChainCoverage
                self.ContextData = ChainContextData
                self.SetContextData = SetChainContextData
                self.RuleData = ChainRuleData
                self.SetRuleData = ChainSetRuleData
            else:
                self.Coverage = Coverage
                self.ContextData = ContextData
                self.SetContextData = SetContextData
                self.RuleData = RuleData
                self.SetRuleData = SetRuleData

            if Format == 1:
                self.Rule = ChainTyp + "Rule"
                self.RuleCount = ChainTyp + "RuleCount"
                self.RuleSet = ChainTyp + "RuleSet"
                self.RuleSetCount = ChainTyp + "RuleSetCount"
                self.Intersect = lambda glyphs, c, r: [r] if r in glyphs else []
            elif Format == 2:
                self.Rule = ChainTyp + "ClassRule"
                self.RuleCount = ChainTyp + "ClassRuleCount"
                self.RuleSet = ChainTyp + "ClassSet"
                self.RuleSetCount = ChainTyp + "ClassSetCount"
                self.Intersect = lambda glyphs, c, r: (
                    c.intersect_class(glyphs, r)
                    if c
                    else (set(glyphs) if r == 0 else set())
                )

                self.ClassDef = "InputClassDef" if Chain else "ClassDef"
                self.ClassDefIndex = 1 if Chain else 0
                self.Input = "Input" if Chain else "Class"
            elif Format == 3:
                self.Input = "InputCoverage" if Chain else "Coverage"

    if self.Format not in [1, 2, 3]:
        return None  # Don't shoot the messenger; let it go
    if not hasattr(self.__class__, "_subset__ContextHelpers"):
        self.__class__._subset__ContextHelpers = {}
    if self.Format not in self.__class__._subset__ContextHelpers:
        helper = ContextHelper(self.__class__, self.Format)
        self.__class__._subset__ContextHelpers[self.Format] = helper
    return self.__class__._subset__ContextHelpers[self.Format]


@_add_method(otTables.ContextSubst, otTables.ChainContextSubst)
def closure_glyphs(self, s, cur_glyphs):
    c = self.__subset_classify_context()

    indices = c.Coverage(self).intersect(cur_glyphs)
    if not indices:
        return []
    cur_glyphs = c.Coverage(self).intersect_glyphs(cur_glyphs)

    if self.Format == 1:
        ContextData = c.ContextData(self)
        rss = getattr(self, c.RuleSet)
        rssCount = getattr(self, c.RuleSetCount)
        for i in indices:
            if i >= rssCount or not rss[i]:
                continue
            for r in getattr(rss[i], c.Rule):
                if not r:
                    continue
                if not all(
                    all(c.Intersect(s.glyphs, cd, k) for k in klist)
                    for cd, klist in zip(ContextData, c.RuleData(r))
                ):
                    continue
                chaos = set()
                for ll in getattr(r, c.LookupRecord):
                    if not ll:
                        continue
                    seqi = ll.SequenceIndex
                    if seqi in chaos:
                        # TODO Can we improve this?
                        pos_glyphs = None
                    else:
                        if seqi == 0:
                            pos_glyphs = frozenset([c.Coverage(self).glyphs[i]])
                        else:
                            pos_glyphs = frozenset([r.Input[seqi - 1]])
                    lookup = s.table.LookupList.Lookup[ll.LookupListIndex]
                    chaos.add(seqi)
                    if lookup.may_have_non_1to1():
                        chaos.update(range(seqi, len(r.Input) + 2))
                    lookup.closure_glyphs(s, cur_glyphs=pos_glyphs)
    elif self.Format == 2:
        ClassDef = getattr(self, c.ClassDef)
        indices = ClassDef.intersect(cur_glyphs)
        ContextData = c.ContextData(self)
        rss = getattr(self, c.RuleSet)
        rssCount = getattr(self, c.RuleSetCount)
        for i in indices:
            if i >= rssCount or not rss[i]:
                continue
            for r in getattr(rss[i], c.Rule):
                if not r:
                    continue
                if not all(
                    all(c.Intersect(s.glyphs, cd, k) for k in klist)
                    for cd, klist in zip(ContextData, c.RuleData(r))
                ):
                    continue
                chaos = set()
                for ll in getattr(r, c.LookupRecord):
                    if not ll:
                        continue
                    seqi = ll.SequenceIndex
                    if seqi in chaos:
                        # TODO Can we improve this?
                        pos_glyphs = None
                    else:
                        if seqi == 0:
                            pos_glyphs = frozenset(
                                ClassDef.intersect_class(cur_glyphs, i)
                            )
                        else:
                            pos_glyphs = frozenset(
                                ClassDef.intersect_class(
                                    s.glyphs, getattr(r, c.Input)[seqi - 1]
                                )
                            )
                    lookup = s.table.LookupList.Lookup[ll.LookupListIndex]
                    chaos.add(seqi)
                    if lookup.may_have_non_1to1():
                        chaos.update(range(seqi, len(getattr(r, c.Input)) + 2))
                    lookup.closure_glyphs(s, cur_glyphs=pos_glyphs)
    elif self.Format == 3:
        if not all(x is not None and x.intersect(s.glyphs) for x in c.RuleData(self)):
            return []
        r = self
        input_coverages = getattr(r, c.Input)
        chaos = set()
        for ll in getattr(r, c.LookupRecord):
            if not ll:
                continue
            seqi = ll.SequenceIndex
            if seqi in chaos:
                # TODO Can we improve this?
                pos_glyphs = None
            else:
                if seqi == 0:
                    pos_glyphs = frozenset(cur_glyphs)
                else:
                    pos_glyphs = frozenset(
                        input_coverages[seqi].intersect_glyphs(s.glyphs)
                    )
            lookup = s.table.LookupList.Lookup[ll.LookupListIndex]
            chaos.add(seqi)
            if lookup.may_have_non_1to1():
                chaos.update(range(seqi, len(input_coverages) + 1))
            lookup.closure_glyphs(s, cur_glyphs=pos_glyphs)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(
    otTables.ContextSubst,
    otTables.ContextPos,
    otTables.ChainContextSubst,
    otTables.ChainContextPos,
)
def subset_glyphs(self, s):
    c = self.__subset_classify_context()

    if self.Format == 1:
        indices = self.Coverage.subset(s.glyphs)
        rss = getattr(self, c.RuleSet)
        rssCount = getattr(self, c.RuleSetCount)
        rss = [rss[i] for i in indices if i < rssCount]
        for rs in rss:
            if not rs:
                continue
            ss = getattr(rs, c.Rule)
            ss = [
                r
                for r in ss
                if r
                and all(all(g in s.glyphs for g in glist) for glist in c.RuleData(r))
            ]
            setattr(rs, c.Rule, ss)
            setattr(rs, c.RuleCount, len(ss))
        # Prune empty rulesets
        indices = [i for i, rs in enumerate(rss) if rs and getattr(rs, c.Rule)]
        self.Coverage.remap(indices)
        rss = _list_subset(rss, indices)
        setattr(self, c.RuleSet, rss)
        setattr(self, c.RuleSetCount, len(rss))
        return bool(rss)
    elif self.Format == 2:
        if not self.Coverage.subset(s.glyphs):
            return False
        ContextData = c.ContextData(self)
        klass_maps = [
            x.subset(s.glyphs, remap=True) if x else None for x in ContextData
        ]

        # Keep rulesets for class numbers that survived.
        indices = klass_maps[c.ClassDefIndex]
        rss = getattr(self, c.RuleSet)
        rssCount = getattr(self, c.RuleSetCount)
        rss = [rss[i] for i in indices if i < rssCount]
        del rssCount
        # Delete, but not renumber, unreachable rulesets.
        indices = getattr(self, c.ClassDef).intersect(self.Coverage.glyphs)
        rss = [rss if i in indices else None for i, rss in enumerate(rss)]

        for rs in rss:
            if not rs:
                continue
            ss = getattr(rs, c.Rule)
            ss = [
                r
                for r in ss
                if r
                and all(
                    all(k in klass_map for k in klist)
                    for klass_map, klist in zip(klass_maps, c.RuleData(r))
                )
            ]
            setattr(rs, c.Rule, ss)
            setattr(rs, c.RuleCount, len(ss))

            # Remap rule classes
            for r in ss:
                c.SetRuleData(
                    r,
                    [
                        [klass_map.index(k) for k in klist]
                        for klass_map, klist in zip(klass_maps, c.RuleData(r))
                    ],
                )

        # Prune empty rulesets
        rss = [rs if rs and getattr(rs, c.Rule) else None for rs in rss]
        while rss and rss[-1] is None:
            del rss[-1]
        setattr(self, c.RuleSet, rss)
        setattr(self, c.RuleSetCount, len(rss))

        # TODO: We can do a second round of remapping class values based
        # on classes that are actually used in at least one rule.	Right
        # now we subset classes to c.glyphs only.	Or better, rewrite
        # the above to do that.

        return bool(rss)
    elif self.Format == 3:
        return all(x is not None and x.subset(s.glyphs) for x in c.RuleData(self))
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(
    otTables.ContextSubst,
    otTables.ChainContextSubst,
    otTables.ContextPos,
    otTables.ChainContextPos,
)
def subset_lookups(self, lookup_indices):
    c = self.__subset_classify_context()

    if self.Format in [1, 2]:
        for rs in getattr(self, c.RuleSet):
            if not rs:
                continue
            for r in getattr(rs, c.Rule):
                if not r:
                    continue
                setattr(
                    r,
                    c.LookupRecord,
                    [
                        ll
                        for ll in getattr(r, c.LookupRecord)
                        if ll and ll.LookupListIndex in lookup_indices
                    ],
                )
                for ll in getattr(r, c.LookupRecord):
                    if not ll:
                        continue
                    ll.LookupListIndex = lookup_indices.index(ll.LookupListIndex)
    elif self.Format == 3:
        setattr(
            self,
            c.LookupRecord,
            [
                ll
                for ll in getattr(self, c.LookupRecord)
                if ll and ll.LookupListIndex in lookup_indices
            ],
        )
        for ll in getattr(self, c.LookupRecord):
            if not ll:
                continue
            ll.LookupListIndex = lookup_indices.index(ll.LookupListIndex)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(
    otTables.ContextSubst,
    otTables.ChainContextSubst,
    otTables.ContextPos,
    otTables.ChainContextPos,
)
def collect_lookups(self):
    c = self.__subset_classify_context()

    if self.Format in [1, 2]:
        return [
            ll.LookupListIndex
            for rs in getattr(self, c.RuleSet)
            if rs
            for r in getattr(rs, c.Rule)
            if r
            for ll in getattr(r, c.LookupRecord)
            if ll
        ]
    elif self.Format == 3:
        return [ll.LookupListIndex for ll in getattr(self, c.LookupRecord) if ll]
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.ExtensionSubst)
def closure_glyphs(self, s, cur_glyphs):
    if self.Format == 1:
        self.ExtSubTable.closure_glyphs(s, cur_glyphs)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.ExtensionSubst)
def may_have_non_1to1(self):
    if self.Format == 1:
        return self.ExtSubTable.may_have_non_1to1()
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.ExtensionSubst, otTables.ExtensionPos)
def subset_glyphs(self, s):
    if self.Format == 1:
        return self.ExtSubTable.subset_glyphs(s)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.ExtensionSubst, otTables.ExtensionPos)
def prune_post_subset(self, font, options):
    if self.Format == 1:
        return self.ExtSubTable.prune_post_subset(font, options)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.ExtensionSubst, otTables.ExtensionPos)
def subset_lookups(self, lookup_indices):
    if self.Format == 1:
        return self.ExtSubTable.subset_lookups(lookup_indices)
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.ExtensionSubst, otTables.ExtensionPos)
def collect_lookups(self):
    if self.Format == 1:
        return self.ExtSubTable.collect_lookups()
    else:
        assert 0, "unknown format: %s" % self.Format


@_add_method(otTables.Lookup)
def closure_glyphs(self, s, cur_glyphs=None):
    if cur_glyphs is None:
        cur_glyphs = frozenset(s.glyphs)

    # Memoize
    key = id(self)
    doneLookups = s._doneLookups
    count, covered = doneLookups.get(key, (0, None))
    if count != len(s.glyphs):
        count, covered = doneLookups[key] = (len(s.glyphs), set())
    if cur_glyphs.issubset(covered):
        return
    covered.update(cur_glyphs)

    for st in self.SubTable:
        if not st:
            continue
        st.closure_glyphs(s, cur_glyphs)


@_add_method(otTables.Lookup)
def subset_glyphs(self, s):
    self.SubTable = [st for st in self.SubTable if st and st.subset_glyphs(s)]
    self.SubTableCount = len(self.SubTable)
    if hasattr(self, "MarkFilteringSet") and self.MarkFilteringSet is not None:
        if self.MarkFilteringSet not in s.used_mark_sets:
            self.MarkFilteringSet = None
            self.LookupFlag &= ~0x10
            self.LookupFlag |= 0x8
        else:
            self.MarkFilteringSet = s.used_mark_sets.index(self.MarkFilteringSet)
    return bool(self.SubTableCount)


@_add_method(otTables.Lookup)
def prune_post_subset(self, font, options):
    ret = False
    for st in self.SubTable:
        if not st:
            continue
        if st.prune_post_subset(font, options):
            ret = True
    return ret


@_add_method(otTables.Lookup)
def subset_lookups(self, lookup_indices):
    for s in self.SubTable:
        s.subset_lookups(lookup_indices)


@_add_method(otTables.Lookup)
def collect_lookups(self):
    return sum((st.collect_lookups() for st in self.SubTable if st), [])


@_add_method(otTables.Lookup)
def may_have_non_1to1(self):
    return any(st.may_have_non_1to1() for st in self.SubTable if st)


@_add_method(otTables.LookupList)
def subset_glyphs(self, s):
    """Returns the indices of nonempty lookups."""
    return [i for i, l in enumerate(self.Lookup) if l and l.subset_glyphs(s)]


@_add_method(otTables.LookupList)
def prune_post_subset(self, font, options):
    ret = False
    for l in self.Lookup:
        if not l:
            continue
        if l.prune_post_subset(font, options):
            ret = True
    return ret


@_add_method(otTables.LookupList)
def subset_lookups(self, lookup_indices):
    self.ensureDecompiled()
    self.Lookup = [self.Lookup[i] for i in lookup_indices if i < self.LookupCount]
    self.LookupCount = len(self.Lookup)
    for l in self.Lookup:
        l.subset_lookups(lookup_indices)


@_add_method(otTables.LookupList)
def neuter_lookups(self, lookup_indices):
    """Sets lookups not in lookup_indices to None."""
    self.ensureDecompiled()
    self.Lookup = [
        l if i in lookup_indices else None for i, l in enumerate(self.Lookup)
    ]


@_add_method(otTables.LookupList)
def closure_lookups(self, lookup_indices):
    """Returns sorted index of all lookups reachable from lookup_indices."""
    lookup_indices = _uniq_sort(lookup_indices)
    recurse = lookup_indices
    while True:
        recurse_lookups = sum(
            (self.Lookup[i].collect_lookups() for i in recurse if i < self.LookupCount),
            [],
        )
        recurse_lookups = [
            l
            for l in recurse_lookups
            if l not in lookup_indices and l < self.LookupCount
        ]
        if not recurse_lookups:
            return _uniq_sort(lookup_indices)
        recurse_lookups = _uniq_sort(recurse_lookups)
        lookup_indices.extend(recurse_lookups)
        recurse = recurse_lookups


@_add_method(otTables.Feature)
def subset_lookups(self, lookup_indices):
    """ "Returns True if feature is non-empty afterwards."""
    self.LookupListIndex = [l for l in self.LookupListIndex if l in lookup_indices]
    # Now map them.
    self.LookupListIndex = [lookup_indices.index(l) for l in self.LookupListIndex]
    self.LookupCount = len(self.LookupListIndex)
    # keep 'size' feature even if it contains no lookups; but drop any other
    # empty feature (e.g. FeatureParams for stylistic set names)
    # https://github.com/fonttools/fonttools/issues/2324
    return self.LookupCount or isinstance(
        self.FeatureParams, otTables.FeatureParamsSize
    )


@_add_method(otTables.FeatureList)
def subset_lookups(self, lookup_indices):
    """Returns the indices of nonempty features."""
    # Note: Never ever drop feature 'pref', even if it's empty.
    # HarfBuzz chooses shaper for Khmer based on presence of this
    # feature.	See thread at:
    # http://lists.freedesktop.org/archives/harfbuzz/2012-November/002660.html
    return [
        i
        for i, f in enumerate(self.FeatureRecord)
        if (f.Feature.subset_lookups(lookup_indices) or f.FeatureTag == "pref")
    ]


@_add_method(otTables.FeatureList)
def collect_lookups(self, feature_indices):
    return sum(
        (
            self.FeatureRecord[i].Feature.LookupListIndex
            for i in feature_indices
            if i < self.FeatureCount
        ),
        [],
    )


@_add_method(otTables.FeatureList)
def subset_features(self, feature_indices):
    self.ensureDecompiled()
    self.FeatureRecord = _list_subset(self.FeatureRecord, feature_indices)
    self.FeatureCount = len(self.FeatureRecord)
    return bool(self.FeatureCount)


@_add_method(otTables.FeatureTableSubstitution)
def subset_lookups(self, lookup_indices):
    """Returns the indices of nonempty features."""
    return [
        r.FeatureIndex
        for r in self.SubstitutionRecord
        if r.Feature.subset_lookups(lookup_indices)
    ]


@_add_method(otTables.FeatureVariations)
def subset_lookups(self, lookup_indices):
    """Returns the indices of nonempty features."""
    return sum(
        (
            f.FeatureTableSubstitution.subset_lookups(lookup_indices)
            for f in self.FeatureVariationRecord
        ),
        [],
    )


@_add_method(otTables.FeatureVariations)
def collect_lookups(self, feature_indices):
    return sum(
        (
            r.Feature.LookupListIndex
            for vr in self.FeatureVariationRecord
            for r in vr.FeatureTableSubstitution.SubstitutionRecord
            if r.FeatureIndex in feature_indices
        ),
        [],
    )


@_add_method(otTables.FeatureTableSubstitution)
def subset_features(self, feature_indices):
    self.ensureDecompiled()
    self.SubstitutionRecord = [
        r for r in self.SubstitutionRecord if r.FeatureIndex in feature_indices
    ]
    # remap feature indices
    for r in self.SubstitutionRecord:
        r.FeatureIndex = feature_indices.index(r.FeatureIndex)
    self.SubstitutionCount = len(self.SubstitutionRecord)
    return bool(self.SubstitutionCount)


@_add_method(otTables.FeatureTableSubstitution)
def prune_features(self, feature_index_map):
    self.ensureDecompiled()
    self.SubstitutionRecord = [
        r for r in self.SubstitutionRecord if r.FeatureIndex in feature_index_map.keys()
    ]
    # remap feature indices
    for r in self.SubstitutionRecord:
        r.FeatureIndex = feature_index_map[r.FeatureIndex]
    self.SubstitutionCount = len(self.SubstitutionRecord)
    return bool(self.SubstitutionCount)


@_add_method(otTables.FeatureVariations)
def subset_features(self, feature_indices):
    self.ensureDecompiled()
    for r in self.FeatureVariationRecord:
        r.FeatureTableSubstitution.subset_features(feature_indices)
    # Prune empty records at the end only
    # https://github.com/fonttools/fonttools/issues/1881
    while (
        self.FeatureVariationRecord
        and not self.FeatureVariationRecord[
            -1
        ].FeatureTableSubstitution.SubstitutionCount
    ):
        self.FeatureVariationRecord.pop()
    self.FeatureVariationCount = len(self.FeatureVariationRecord)
    return bool(self.FeatureVariationCount)


@_add_method(otTables.FeatureVariations)
def prune_features(self, feature_index_map):
    self.ensureDecompiled()
    for r in self.FeatureVariationRecord:
        r.FeatureTableSubstitution.prune_features(feature_index_map)
    # Prune empty records at the end only
    # https://github.com/fonttools/fonttools/issues/1881
    while (
        self.FeatureVariationRecord
        and not self.FeatureVariationRecord[
            -1
        ].FeatureTableSubstitution.SubstitutionCount
    ):
        self.FeatureVariationRecord.pop()
    self.FeatureVariationCount = len(self.FeatureVariationRecord)
    return bool(self.FeatureVariationCount)


@_add_method(otTables.DefaultLangSys, otTables.LangSys)
def subset_features(self, feature_indices):
    if self.ReqFeatureIndex in feature_indices:
        self.ReqFeatureIndex = feature_indices.index(self.ReqFeatureIndex)
    else:
        self.ReqFeatureIndex = 65535
    self.FeatureIndex = [f for f in self.FeatureIndex if f in feature_indices]
    # Now map them.
    self.FeatureIndex = [
        feature_indices.index(f) for f in self.FeatureIndex if f in feature_indices
    ]
    self.FeatureCount = len(self.FeatureIndex)
    return bool(self.FeatureCount or self.ReqFeatureIndex != 65535)


@_add_method(otTables.DefaultLangSys, otTables.LangSys)
def prune_features(self, feature_index_map):
    self.ReqFeatureIndex = feature_index_map.get(self.ReqFeatureIndex, 65535)
    self.FeatureIndex = [
        feature_index_map[f] for f in self.FeatureIndex if f in feature_index_map.keys()
    ]
    self.FeatureCount = len(self.FeatureIndex)
    return bool(self.FeatureCount or self.ReqFeatureIndex != 65535)


@_add_method(otTables.DefaultLangSys, otTables.LangSys)
def collect_features(self):
    feature_indices = self.FeatureIndex[:]
    if self.ReqFeatureIndex != 65535:
        feature_indices.append(self.ReqFeatureIndex)
    return _uniq_sort(feature_indices)


@_add_method(otTables.Script)
def subset_features(self, feature_indices, keepEmptyDefaultLangSys=False):
    if (
        self.DefaultLangSys
        and not self.DefaultLangSys.subset_features(feature_indices)
        and not keepEmptyDefaultLangSys
    ):
        self.DefaultLangSys = None
    self.LangSysRecord = [
        l for l in self.LangSysRecord if l.LangSys.subset_features(feature_indices)
    ]
    self.LangSysCount = len(self.LangSysRecord)
    return bool(self.LangSysCount or self.DefaultLangSys)


@_add_method(otTables.Script)
def prune_features(self, feature_index_map, keepEmptyDefaultLangSys=False):
    if (
        self.DefaultLangSys
        and not self.DefaultLangSys.prune_features(feature_index_map)
        and not keepEmptyDefaultLangSys
    ):
        self.DefaultLangSys = None
    self.LangSysRecord = [
        l for l in self.LangSysRecord if l.LangSys.prune_features(feature_index_map)
    ]
    self.LangSysCount = len(self.LangSysRecord)
    return bool(self.LangSysCount or self.DefaultLangSys)


@_add_method(otTables.Script)
def collect_features(self):
    feature_indices = [l.LangSys.collect_features() for l in self.LangSysRecord]
    if self.DefaultLangSys:
        feature_indices.append(self.DefaultLangSys.collect_features())
    return _uniq_sort(sum(feature_indices, []))


@_add_method(otTables.ScriptList)
def subset_features(self, feature_indices, retain_empty):
    # https://bugzilla.mozilla.org/show_bug.cgi?id=1331737#c32
    self.ScriptRecord = [
        s
        for s in self.ScriptRecord
        if s.Script.subset_features(feature_indices, s.ScriptTag == "DFLT")
        or retain_empty
    ]
    self.ScriptCount = len(self.ScriptRecord)
    return bool(self.ScriptCount)


@_add_method(otTables.ScriptList)
def prune_features(self, feature_index_map, retain_empty):
    # https://bugzilla.mozilla.org/show_bug.cgi?id=1331737#c32
    self.ScriptRecord = [
        s
        for s in self.ScriptRecord
        if s.Script.prune_features(feature_index_map, s.ScriptTag == "DFLT")
        or retain_empty
    ]
    self.ScriptCount = len(self.ScriptRecord)
    return bool(self.ScriptCount)


@_add_method(otTables.ScriptList)
def collect_features(self):
    return _uniq_sort(sum((s.Script.collect_features() for s in self.ScriptRecord), []))


# CBLC will inherit it
@_add_method(ttLib.getTableClass("EBLC"))
def subset_glyphs(self, s):
    for strike in self.strikes:
        for indexSubTable in strike.indexSubTables:
            indexSubTable.names = [n for n in indexSubTable.names if n in s.glyphs]
        strike.indexSubTables = [i for i in strike.indexSubTables if i.names]
    self.strikes = [s for s in self.strikes if s.indexSubTables]

    return True


# CBDT will inherit it
@_add_method(ttLib.getTableClass("EBDT"))
def subset_glyphs(self, s):
    strikeData = [
        {g: strike[g] for g in s.glyphs if g in strike} for strike in self.strikeData
    ]
    # Prune empty strikes
    # https://github.com/fonttools/fonttools/issues/1633
    self.strikeData = [strike for strike in strikeData if strike]
    return True


@_add_method(ttLib.getTableClass("sbix"))
def subset_glyphs(self, s):
    for strike in self.strikes.values():
        strike.glyphs = {g: strike.glyphs[g] for g in s.glyphs if g in strike.glyphs}

    return True


@_add_method(ttLib.getTableClass("GSUB"))
def closure_glyphs(self, s):
    s.table = self.table
    if self.table.ScriptList:
        feature_indices = self.table.ScriptList.collect_features()
    else:
        feature_indices = []
    if self.table.FeatureList:
        lookup_indices = self.table.FeatureList.collect_lookups(feature_indices)
    else:
        lookup_indices = []
    if getattr(self.table, "FeatureVariations", None):
        lookup_indices += self.table.FeatureVariations.collect_lookups(feature_indices)
    lookup_indices = _uniq_sort(lookup_indices)
    if self.table.LookupList:
        s._doneLookups = {}
        while True:
            orig_glyphs = frozenset(s.glyphs)
            for i in lookup_indices:
                if i >= self.table.LookupList.LookupCount:
                    continue
                if not self.table.LookupList.Lookup[i]:
                    continue
                self.table.LookupList.Lookup[i].closure_glyphs(s)
            if orig_glyphs == s.glyphs:
                break
        del s._doneLookups
    del s.table


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def subset_glyphs(self, s):
    s.glyphs = s.glyphs_gsubed
    if self.table.LookupList:
        lookup_indices = self.table.LookupList.subset_glyphs(s)
    else:
        lookup_indices = []
    self.subset_lookups(lookup_indices)
    return True


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def retain_empty_scripts(self):
    # https://github.com/fonttools/fonttools/issues/518
    # https://bugzilla.mozilla.org/show_bug.cgi?id=1080739#c15
    return self.__class__ == ttLib.getTableClass("GSUB")


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def subset_lookups(self, lookup_indices):
    """Retains specified lookups, then removes empty features, language
    systems, and scripts."""
    if self.table.LookupList:
        self.table.LookupList.subset_lookups(lookup_indices)
    if self.table.FeatureList:
        feature_indices = self.table.FeatureList.subset_lookups(lookup_indices)
    else:
        feature_indices = []
    if getattr(self.table, "FeatureVariations", None):
        feature_indices += self.table.FeatureVariations.subset_lookups(lookup_indices)
    feature_indices = _uniq_sort(feature_indices)
    if self.table.FeatureList:
        self.table.FeatureList.subset_features(feature_indices)
    if getattr(self.table, "FeatureVariations", None):
        self.table.FeatureVariations.subset_features(feature_indices)
    if self.table.ScriptList:
        self.table.ScriptList.subset_features(
            feature_indices, self.retain_empty_scripts()
        )


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def neuter_lookups(self, lookup_indices):
    """Sets lookups not in lookup_indices to None."""
    if self.table.LookupList:
        self.table.LookupList.neuter_lookups(lookup_indices)


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def prune_lookups(self, remap=True):
    """Remove (default) or neuter unreferenced lookups"""
    if self.table.ScriptList:
        feature_indices = self.table.ScriptList.collect_features()
    else:
        feature_indices = []
    if self.table.FeatureList:
        lookup_indices = self.table.FeatureList.collect_lookups(feature_indices)
    else:
        lookup_indices = []
    if getattr(self.table, "FeatureVariations", None):
        lookup_indices += self.table.FeatureVariations.collect_lookups(feature_indices)
    lookup_indices = _uniq_sort(lookup_indices)
    if self.table.LookupList:
        lookup_indices = self.table.LookupList.closure_lookups(lookup_indices)
    else:
        lookup_indices = []
    if remap:
        self.subset_lookups(lookup_indices)
    else:
        self.neuter_lookups(lookup_indices)


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def subset_feature_tags(self, feature_tags):
    if self.table.FeatureList:
        feature_indices = [
            i
            for i, f in enumerate(self.table.FeatureList.FeatureRecord)
            if f.FeatureTag in feature_tags
        ]
        self.table.FeatureList.subset_features(feature_indices)
        if getattr(self.table, "FeatureVariations", None):
            self.table.FeatureVariations.subset_features(feature_indices)
    else:
        feature_indices = []
    if self.table.ScriptList:
        self.table.ScriptList.subset_features(
            feature_indices, self.retain_empty_scripts()
        )


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def subset_script_tags(self, tags):
    langsys = {}
    script_tags = set()
    for tag in tags:
        script_tag, lang_tag = tag.split(".") if "." in tag else (tag, "*")
        script_tags.add(script_tag.ljust(4))
        langsys.setdefault(script_tag, set()).add(lang_tag.ljust(4))

    if self.table.ScriptList:
        self.table.ScriptList.ScriptRecord = [
            s for s in self.table.ScriptList.ScriptRecord if s.ScriptTag in script_tags
        ]
        self.table.ScriptList.ScriptCount = len(self.table.ScriptList.ScriptRecord)

        for record in self.table.ScriptList.ScriptRecord:
            if record.ScriptTag in langsys and "*   " not in langsys[record.ScriptTag]:
                record.Script.LangSysRecord = [
                    l
                    for l in record.Script.LangSysRecord
                    if l.LangSysTag in langsys[record.ScriptTag]
                ]
                record.Script.LangSysCount = len(record.Script.LangSysRecord)
                if "dflt" not in langsys[record.ScriptTag]:
                    record.Script.DefaultLangSys = None


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def prune_features(self):
    """Remove unreferenced and duplicate features in FeatureList
    Remove unreferenced features and remap duplicate feature indices in ScriptList and FeatureVariations
    """
    if self.table.ScriptList:
        feature_indices = self.table.ScriptList.collect_features()
    else:
        feature_indices = []
    feature_indices, feature_index_map = self.remap_duplicate_features(feature_indices)

    if self.table.FeatureList:
        self.table.FeatureList.subset_features(feature_indices)
    if getattr(self.table, "FeatureVariations", None):
        self.table.FeatureVariations.prune_features(feature_index_map)
    if self.table.ScriptList:
        self.table.ScriptList.prune_features(
            feature_index_map, self.retain_empty_scripts()
        )


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def remap_duplicate_features(self, feature_indices):
    """Return retained feature indices(without duplicates) and remapped feature indices"""
    features = self.table.FeatureList.FeatureRecord

    unique_features = {}
    duplicate_features = {}
    for i in feature_indices:
        f = features[i]
        tag = f.FeatureTag

        same_tag_features = unique_features.get(tag)
        if same_tag_features is None:
            unique_features[tag] = set([i])
            duplicate_features[i] = i
            continue

        found = False
        for other_i in same_tag_features:
            if features[other_i] == f:
                found = True
                duplicate_features[i] = other_i
                break

        if not found:
            same_tag_features.add(i)
            duplicate_features[i] = i

    ## remap retained feature indices
    feature_map = {}
    new_idx = 0

    for i in feature_indices:
        unique_i = duplicate_features.get(i, i)
        v = feature_map.get(unique_i)
        if v is None:
            feature_map[i] = new_idx
            new_idx += 1
        else:
            feature_map[i] = v

    retained_feature_indices = _uniq_sort(
        sum((list(s) for s in unique_features.values()), [])
    )
    return (retained_feature_indices, feature_map)


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def prune_pre_subset(self, font, options):
    # Drop undesired features
    if "*" not in options.layout_scripts:
        self.subset_script_tags(options.layout_scripts)
    if "*" not in options.layout_features:
        self.subset_feature_tags(options.layout_features)
    # Neuter unreferenced lookups
    self.prune_lookups(remap=False)
    return True


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def remove_redundant_langsys(self):
    table = self.table
    if not table.ScriptList or not table.FeatureList:
        return

    features = table.FeatureList.FeatureRecord

    for s in table.ScriptList.ScriptRecord:
        d = s.Script.DefaultLangSys
        if not d:
            continue
        for lr in s.Script.LangSysRecord[:]:
            l = lr.LangSys
            # Compare d and l
            if len(d.FeatureIndex) != len(l.FeatureIndex):
                continue
            if (d.ReqFeatureIndex == 65535) != (l.ReqFeatureIndex == 65535):
                continue

            if d.ReqFeatureIndex != 65535:
                if features[d.ReqFeatureIndex] != features[l.ReqFeatureIndex]:
                    continue

            for i in range(len(d.FeatureIndex)):
                if features[d.FeatureIndex[i]] != features[l.FeatureIndex[i]]:
                    break
            else:
                # LangSys and default are equal; delete LangSys
                s.Script.LangSysRecord.remove(lr)


@_add_method(ttLib.getTableClass("GSUB"), ttLib.getTableClass("GPOS"))
def prune_post_subset(self, font, options):
    table = self.table

    self.prune_lookups()  # XXX Is this actually needed?!

    if table.LookupList:
        table.LookupList.prune_post_subset(font, options)
        # XXX Next two lines disabled because OTS is stupid and
        # doesn't like NULL offsets here.
        # if not table.LookupList.Lookup:
        # 	table.LookupList = None

    if not table.LookupList:
        table.FeatureList = None

    if table.FeatureList:
        self.remove_redundant_langsys()
        # Remove unreferenced features
        self.prune_features()

    # XXX Next two lines disabled because OTS is stupid and
    # doesn't like NULL offsets here.
    # if table.FeatureList and not table.FeatureList.FeatureRecord:
    # 	table.FeatureList = None

    # Never drop scripts themselves as them just being available
    # holds semantic significance.
    # XXX Next two lines disabled because OTS is stupid and
    # doesn't like NULL offsets here.
    # if table.ScriptList and not table.ScriptList.ScriptRecord:
    # 	table.ScriptList = None

    if hasattr(table, "FeatureVariations"):
        # drop FeatureVariations if there are no features to substitute
        if table.FeatureVariations and not (
            table.FeatureList and table.FeatureVariations.FeatureVariationRecord
        ):
            table.FeatureVariations = None

        # downgrade table version if there are no FeatureVariations
        if not table.FeatureVariations and table.Version == 0x00010001:
            table.Version = 0x00010000

    return True


@_add_method(ttLib.getTableClass("GDEF"))
def subset_glyphs(self, s):
    glyphs = s.glyphs_gsubed
    table = self.table
    if table.LigCaretList:
        indices = table.LigCaretList.Coverage.subset(glyphs)
        table.LigCaretList.LigGlyph = _list_subset(table.LigCaretList.LigGlyph, indices)
        table.LigCaretList.LigGlyphCount = len(table.LigCaretList.LigGlyph)
    if table.MarkAttachClassDef:
        table.MarkAttachClassDef.classDefs = {
            g: v for g, v in table.MarkAttachClassDef.classDefs.items() if g in glyphs
        }
    if table.GlyphClassDef:
        table.GlyphClassDef.classDefs = {
            g: v for g, v in table.GlyphClassDef.classDefs.items() if g in glyphs
        }
    if table.AttachList:
        indices = table.AttachList.Coverage.subset(glyphs)
