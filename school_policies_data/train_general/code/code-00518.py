    CalledProcessError. The CalledProcessError object will have the return code
    in the returncode attribute, and output & stderr attributes if those streams
    were captured.

    If timeout is given, and the process takes too long, a TimeoutExpired
    exception will be raised.

    There is an optional argument "input", allowing you to
    pass bytes or a string to the subprocess's stdin.  If you use this argument
    you may not also use the Popen constructor's "stdin" argument, as
    it will be used internally.

    By default, all communication is in bytes, and therefore any "input" should
    be bytes, and the stdout and stderr will be bytes. If in text mode, any
    "input" should be a string, and stdout and stderr will be strings decoded
    according to locale encoding, or by "encoding" if set. Text mode is
    triggered by setting any of text, encoding, errors or universal_newlines.

    The other arguments are the same as for the Popen constructor.
    """
    if input is not None:
        if kwargs.get('stdin') is not None:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = PIPE

    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used '
                             'with capture_output.')
        kwargs['stdout'] = PIPE
        kwargs['stderr'] = PIPE

    with Popen(*popenargs, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except TimeoutExpired as exc:
            process.kill()
            if _mswindows:
                # Windows accumulates the output in a single blocking
                # read() call run on child threads, with the timeout
                # being done in a join() on those threads.  communicate()
                # _after_ kill() is required to collect that and add it
                # to the exception.
                exc.stdout, exc.stderr = process.communicate()
            else:
                # POSIX _communicate already populated the output so
                # far into the TimeoutExpired exception.
                process.wait()
            raise
        except:  # Including KeyboardInterrupt, communicate handled that.
            process.kill()
            # We don't call process.wait() as .__exit__ does that for us.
            raise
        retcode = process.poll()
        if check and retcode:
            raise CalledProcessError(retcode, process.args,
                                     output=stdout, stderr=stderr)
    return CompletedProcess(process.args, retcode, stdout, stderr)


def list2cmdline(seq):
    """
    Translate a sequence of arguments into a command line
    string, using the same rules as the MS C runtime:

    1) Arguments are delimited by white space, which is either a
       space or a tab.

    2) A string surrounded by double quotation marks is
       interpreted as a single argument, regardless of white space
       contained within.  A quoted string can be embedded in an
       argument.

    3) A double quotation mark preceded by a backslash is
       interpreted as a literal double quotation mark.

    4) Backslashes are interpreted literally, unless they
       immediately precede a double quotation mark.

    5) If backslashes immediately precede a double quotation mark,
       every pair of backslashes is interpreted as a literal
       backslash.  If the number of backslashes is odd, the last
       backslash escapes the next double quotation mark as
       described in rule 3.
    """

    # See
    # http://msdn.microsoft.com/en-us/library/17w5ykft.aspx
    # or search http://msdn.microsoft.com for
    # "Parsing C++ Command-Line Arguments"
    result = []
    needquote = False
    for arg in map(os.fsdecode, seq):
        bs_buf = []

        # Add a space to separate this argument from the others
        if result:
            result.append(' ')

        needquote = (" " in arg) or ("\t" in arg) or not arg
        if needquote:
            result.append('"')

        for c in arg:
            if c == '\\':
                # Don't know if we need to double yet.
                bs_buf.append(c)
            elif c == '"':
                # Double backslashes.
                result.append('\\' * len(bs_buf)*2)
                bs_buf = []
                result.append('\\"')
            else:
                # Normal char
                if bs_buf:
                    result.extend(bs_buf)
                    bs_buf = []
                result.append(c)

        # Add remaining backslashes, if any.
        if bs_buf:
            result.extend(bs_buf)

        if needquote:
            result.extend(bs_buf)
            result.append('"')

    return ''.join(result)


# Various tools for executing commands and looking at their output and status.
#

def getstatusoutput(cmd):
    """Return (exitcode, output) of executing cmd in a shell.

    Execute the string 'cmd' in a shell with 'check_output' and
    return a 2-tuple (status, output). The locale encoding is used
    to decode the output and process newlines.

    A trailing newline is stripped from the output.
    The exit status for the command can be interpreted
    according to the rules for the function 'wait'. Example:

    >>> import subprocess
    >>> subprocess.getstatusoutput('ls /bin/ls')
    (0, '/bin/ls')
    >>> subprocess.getstatusoutput('cat /bin/junk')
    (1, 'cat: /bin/junk: No such file or directory')
    >>> subprocess.getstatusoutput('/bin/junk')
    (127, 'sh: /bin/junk: not found')
    >>> subprocess.getstatusoutput('/bin/kill $$')
    (-15, '')
    """
    try:
        data = check_output(cmd, shell=True, text=True, stderr=STDOUT)
        exitcode = 0
    except CalledProcessError as ex:
        data = ex.output
        exitcode = ex.returncode
    if data[-1:] == '\n':
        data = data[:-1]
    return exitcode, data

def getoutput(cmd):
    """Return output (stdout or stderr) of executing cmd in a shell.

    Like getstatusoutput(), except the exit status is ignored and the return
    value is a string containing the command's output.  Example:

    >>> import subprocess
    >>> subprocess.getoutput('ls /bin/ls')
    '/bin/ls'
    """
    return getstatusoutput(cmd)[1]


def _use_posix_spawn():
    """Check if posix_spawn() can be used for subprocess.

    subprocess requires a posix_spawn() implementation that properly reports
    errors to the parent process, & sets errno on the following failures:

    * Process attribute actions failed.
    * File actions failed.
    * exec() failed.

    Prefer an implementation which can use vfork() in some cases for best
    performance.
    """
    if _mswindows or not hasattr(os, 'posix_spawn'):
        # os.posix_spawn() is not available
        return False

    if sys.platform in ('darwin', 'sunos5'):
        # posix_spawn() is a syscall on both macOS and Solaris,
        # and properly reports errors
        return True

    # Check libc name and runtime libc version
    try:
        ver = os.confstr('CS_GNU_LIBC_VERSION')
        # parse 'glibc 2.28' as ('glibc', (2, 28))
        parts = ver.split(maxsplit=1)
        if len(parts) != 2:
            # reject unknown format
            raise ValueError
        libc = parts[0]
        version = tuple(map(int, parts[1].split('.')))

        if sys.platform == 'linux' and libc == 'glibc' and version >= (2, 24):
            # glibc 2.24 has a new Linux posix_spawn implementation using vfork
            # which properly reports errors to the parent process.
            return True
        # Note: Don't use the implementation in earlier glibc because it doesn't
        # use vfork (even if glibc 2.26 added a pipe to properly report errors
        # to the parent process).
    except (AttributeError, ValueError, OSError):
        # os.confstr() or CS_GNU_LIBC_VERSION value not available
        pass

    # By default, assume that posix_spawn() does not properly report errors.
    return False


# These are primarily fail-safe knobs for negatives. A True value does not
# guarantee the given libc/syscall API will be used.
_USE_POSIX_SPAWN = _use_posix_spawn()
_USE_VFORK = True


class Popen:
    """ Execute a child program in a new process.

    For a complete description of the arguments see the Python documentation.

    Arguments:
      args: A string, or a sequence of program arguments.

      bufsize: supplied as the buffering argument to the open() function when
          creating the stdin/stdout/stderr pipe file objects

      executable: A replacement program to execute.

      stdin, stdout and stderr: These specify the executed programs' standard
          input, standard output and standard error file handles, respectively.

      preexec_fn: (POSIX only) An object to be called in the child process
          just before the child is executed.

      close_fds: Controls closing or inheriting of file descriptors.

      shell: If true, the command will be executed through the shell.

      cwd: Sets the current directory before the child is executed.

      env: Defines the environment variables for the new process.

      text: If true, decode stdin, stdout and stderr using the given encoding
          (if set) or the system default otherwise.

      universal_newlines: Alias of text, provided for backwards compatibility.

      startupinfo and creationflags (Windows only)

      restore_signals (POSIX only)

      start_new_session (POSIX only)

      group (POSIX only)

      extra_groups (POSIX only)

      user (POSIX only)

      umask (POSIX only)

      pass_fds (POSIX only)

      encoding and errors: Text mode encoding and error handling to use for
          file objects stdin, stdout and stderr.

    Attributes:
        stdin, stdout, stderr, pid, returncode
    """
    _child_created = False  # Set here since __del__ checks it

    def __init__(self, args, bufsize=-1, executable=None,
                 stdin=None, stdout=None, stderr=None,
                 preexec_fn=None, close_fds=True,
                 shell=False, cwd=None, env=None, universal_newlines=None,
                 startupinfo=None, creationflags=0,
                 restore_signals=True, start_new_session=False,
                 pass_fds=(), *, user=None, group=None, extra_groups=None,
                 encoding=None, errors=None, text=None, umask=-1, pipesize=-1):
        """Create new Popen instance."""
        _cleanup()
        # Held while anything is calling waitpid before returncode has been
        # updated to prevent clobbering returncode if wait() or poll() are
        # called from multiple threads at once.  After acquiring the lock,
        # code must re-check self.returncode to see if another thread just
        # finished a waitpid() call.
        self._waitpid_lock = threading.Lock()

        self._input = None
        self._communication_started = False
        if bufsize is None:
            bufsize = -1  # Restore default
        if not isinstance(bufsize, int):
            raise TypeError("bufsize must be an integer")

        if pipesize is None:
            pipesize = -1  # Restore default
        if not isinstance(pipesize, int):
            raise TypeError("pipesize must be an integer")

        if _mswindows:
            if preexec_fn is not None:
                raise ValueError("preexec_fn is not supported on Windows "
                                 "platforms")
        else:
            # POSIX
            if pass_fds and not close_fds:
                warnings.warn("pass_fds overriding close_fds.", RuntimeWarning)
                close_fds = True
            if startupinfo is not None:
                raise ValueError("startupinfo is only supported on Windows "
                                 "platforms")
            if creationflags != 0:
                raise ValueError("creationflags is only supported on Windows "
                                 "platforms")

        self.args = args
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self.pid = None
        self.returncode = None
        self.encoding = encoding
        self.errors = errors
        self.pipesize = pipesize

        # Validate the combinations of text and universal_newlines
        if (text is not None and universal_newlines is not None
            and bool(universal_newlines) != bool(text)):
            raise SubprocessError('Cannot disambiguate when both text '
                                  'and universal_newlines are supplied but '
                                  'different. Pass one or the other.')

        # Input and output objects. The general principle is like
        # this:
        #
        # Parent                   Child
        # ------                   -----
        # p2cwrite   ---stdin--->  p2cread
        # c2pread    <--stdout---  c2pwrite
        # errread    <--stderr---  errwrite
        #
        # On POSIX, the child objects are file descriptors.  On
        # Windows, these are Windows file handles.  The parent objects
        # are file descriptors on both platforms.  The parent objects
        # are -1 when not using PIPEs. The child objects are -1
        # when not redirecting.

        (p2cread, p2cwrite,
         c2pread, c2pwrite,
         errread, errwrite) = self._get_handles(stdin, stdout, stderr)

        # We wrap OS handles *before* launching the child, otherwise a
        # quickly terminating child could make our fds unwrappable
        # (see #8458).

        if _mswindows:
            if p2cwrite != -1:
                p2cwrite = msvcrt.open_osfhandle(p2cwrite.Detach(), 0)
            if c2pread != -1:
                c2pread = msvcrt.open_osfhandle(c2pread.Detach(), 0)
            if errread != -1:
                errread = msvcrt.open_osfhandle(errread.Detach(), 0)

        self.text_mode = encoding or errors or text or universal_newlines

        # PEP 597: We suppress the EncodingWarning in subprocess module
        # for now (at Python 3.10), because we focus on files for now.
        # This will be changed to encoding = io.text_encoding(encoding)
        # in the future.
        if self.text_mode and encoding is None:
            self.encoding = encoding = "locale"

        # How long to resume waiting on a child after the first ^C.
        # There is no right value for this.  The purpose is to be polite
        # yet remain good for interactive users trying to exit a tool.
        self._sigint_wait_secs = 0.25  # 1/xkcd221.getRandomNumber()

        self._closed_child_pipe_fds = False

        if self.text_mode:
            if bufsize == 1:
                line_buffering = True
                # Use the default buffer size for the underlying binary streams
                # since they don't support line buffering.
                bufsize = -1
            else:
                line_buffering = False

        gid = None
        if group is not None:
            if not hasattr(os, 'setregid'):
                raise ValueError("The 'group' parameter is not supported on the "
                                 "current platform")

            elif isinstance(group, str):
                try:
                    import grp
                except ImportError:
                    raise ValueError("The group parameter cannot be a string "
                                     "on systems without the grp module")

                gid = grp.getgrnam(group).gr_gid
            elif isinstance(group, int):
                gid = group
            else:
                raise TypeError("Group must be a string or an integer, not {}"
                                .format(type(group)))

            if gid < 0:
                raise ValueError(f"Group ID cannot be negative, got {gid}")

        gids = None
        if extra_groups is not None:
            if not hasattr(os, 'setgroups'):
                raise ValueError("The 'extra_groups' parameter is not "
                                 "supported on the current platform")

            elif isinstance(extra_groups, str):
                raise ValueError("Groups must be a list, not a string")

            gids = []
            for extra_group in extra_groups:
                if isinstance(extra_group, str):
                    try:
                        import grp
                    except ImportError:
                        raise ValueError("Items in extra_groups cannot be "
                                         "strings on systems without the "
                                         "grp module")

                    gids.append(grp.getgrnam(extra_group).gr_gid)
                elif isinstance(extra_group, int):
                    gids.append(extra_group)
                else:
                    raise TypeError("Items in extra_groups must be a string "
                                    "or integer, not {}"
                                    .format(type(extra_group)))

            # make sure that the gids are all positive here so we can do less
            # checking in the C code
            for gid_check in gids:
                if gid_check < 0:
                    raise ValueError(f"Group ID cannot be negative, got {gid_check}")

        uid = None
        if user is not None:
            if not hasattr(os, 'setreuid'):
                raise ValueError("The 'user' parameter is not supported on "
                                 "the current platform")

            elif isinstance(user, str):
                try:
                    import pwd
                except ImportError:
                    raise ValueError("The user parameter cannot be a string "
                                     "on systems without the pwd module")
                uid = pwd.getpwnam(user).pw_uid
            elif isinstance(user, int):
                uid = user
            else:
                raise TypeError("User must be a string or an integer")

            if uid < 0:
                raise ValueError(f"User ID cannot be negative, got {uid}")

        try:
            if p2cwrite != -1:
                self.stdin = io.open(p2cwrite, 'wb', bufsize)
                if self.text_mode:
                    self.stdin = io.TextIOWrapper(self.stdin, write_through=True,
                            line_buffering=line_buffering,
                            encoding=encoding, errors=errors)
            if c2pread != -1:
                self.stdout = io.open(c2pread, 'rb', bufsize)
                if self.text_mode:
                    self.stdout = io.TextIOWrapper(self.stdout,
                            encoding=encoding, errors=errors)
            if errread != -1:
                self.stderr = io.open(errread, 'rb', bufsize)
                if self.text_mode:
                    self.stderr = io.TextIOWrapper(self.stderr,
                            encoding=encoding, errors=errors)

            self._execute_child(args, executable, preexec_fn, close_fds,
                                pass_fds, cwd, env,
                                startupinfo, creationflags, shell,
                                p2cread, p2cwrite,
                                c2pread, c2pwrite,
                                errread, errwrite,
                                restore_signals,
                                gid, gids, uid, umask,
                                start_new_session)
        except:
            # Cleanup if the child failed starting.
            for f in filter(None, (self.stdin, self.stdout, self.stderr)):
                try:
                    f.close()
                except OSError:
                    pass  # Ignore EBADF or other errors.

            if not self._closed_child_pipe_fds:
                to_close = []
                if stdin == PIPE:
                    to_close.append(p2cread)
                if stdout == PIPE:
                    to_close.append(c2pwrite)
                if stderr == PIPE:
                    to_close.append(errwrite)
                if hasattr(self, '_devnull'):
                    to_close.append(self._devnull)
                for fd in to_close:
                    try:
                        if _mswindows and isinstance(fd, Handle):
                            fd.Close()
                        else:
                            os.close(fd)
                    except OSError:
                        pass

            raise

    def __repr__(self):
        obj_repr = (
            f"<{self.__class__.__name__}: "
            f"returncode: {self.returncode} args: {self.args!r}>"
        )
        if len(obj_repr) > 80:
            obj_repr = obj_repr[:76] + "...>"
        return obj_repr

    __class_getitem__ = classmethod(types.GenericAlias)

    @property
    def universal_newlines(self):
        # universal_newlines as retained as an alias of text_mode for API
        # compatibility. bpo-31756
        return self.text_mode

    @universal_newlines.setter
    def universal_newlines(self, universal_newlines):
        self.text_mode = bool(universal_newlines)

    def _translate_newlines(self, data, encoding, errors):
        data = data.decode(encoding, errors)
        return data.replace("\r\n", "\n").replace("\r", "\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.stdout:
            self.stdout.close()
        if self.stderr:
            self.stderr.close()
        try:  # Flushing a BufferedWriter may raise an error
            if self.stdin:
                self.stdin.close()
        finally:
            if exc_type == KeyboardInterrupt:
                # https://bugs.python.org/issue25942
                # In the case of a KeyboardInterrupt we assume the SIGINT
                # was also already sent to our child processes.  We can't
                # block indefinitely as that is not user friendly.
                # If we have not already waited a brief amount of time in
                # an interrupted .wait() or .communicate() call, do so here
                # for consistency.
                if self._sigint_wait_secs > 0:
                    try:
                        self._wait(timeout=self._sigint_wait_secs)
                    except TimeoutExpired:
                        pass
                self._sigint_wait_secs = 0  # Note that this has been done.
                return  # resume the KeyboardInterrupt

            # Wait for the process to terminate, to avoid zombies.
            self.wait()

    def __del__(self, _maxsize=sys.maxsize, _warn=warnings.warn):
        if not self._child_created:
            # We didn't get to successfully create a child process.
            return
        if self.returncode is None:
            # Not reading subprocess exit status creates a zombie process which
            # is only destroyed at the parent python process exit
            _warn("subprocess %s is still running" % self.pid,
                  ResourceWarning, source=self)
        # In case the child hasn't been waited on, check if it's done.
        self._internal_poll(_deadstate=_maxsize)
        if self.returncode is None and _active is not None:
            # Child is still running, keep us alive until we can wait on it.
            _active.append(self)

    def _get_devnull(self):
        if not hasattr(self, '_devnull'):
            self._devnull = os.open(os.devnull, os.O_RDWR)
        return self._devnull

    def _stdin_write(self, input):
        if input:
            try:
                self.stdin.write(input)
            except BrokenPipeError:
                pass  # communicate() must ignore broken pipe errors.
            except OSError as exc:
                if exc.errno == errno.EINVAL:
                    # bpo-19612, bpo-30418: On Windows, stdin.write() fails
                    # with EINVAL if the child process exited or if the child
                    # process is still running but closed the pipe.
                    pass
                else:
                    raise

        try:
            self.stdin.close()
        except BrokenPipeError:
            pass  # communicate() must ignore broken pipe errors.
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                pass
            else:
                raise

    def communicate(self, input=None, timeout=None):
        """Interact with process: Send data to stdin and close it.
        Read data from stdout and stderr, until end-of-file is
        reached.  Wait for process to terminate.

        The optional "input" argument should be data to be sent to the
        child process, or None, if no data should be sent to the child.
        communicate() returns a tuple (stdout, stderr).

        By default, all communication is in bytes, and therefore any
        "input" should be bytes, and the (stdout, stderr) will be bytes.
        If in text mode (indicated by self.text_mode), any "input" should
        be a string, and (stdout, stderr) will be strings decoded
        according to locale encoding, or by "encoding" if set. Text mode
        is triggered by setting any of text, encoding, errors or
        universal_newlines.
        """

        if self._communication_started and input:
            raise ValueError("Cannot send input after starting communication")

        # Optimization: If we are not worried about timeouts, we haven't
        # started communicating, and we have one or zero pipes, using select()
        # or threads is unnecessary.
        if (timeout is None and not self._communication_started and
            [self.stdin, self.stdout, self.stderr].count(None) >= 2):
            stdout = None
            stderr = None
            if self.stdin:
                self._stdin_write(input)
            elif self.stdout:
                stdout = self.stdout.read()
                self.stdout.close()
            elif self.stderr:
                stderr = self.stderr.read()
                self.stderr.close()
            self.wait()
        else:
            if timeout is not None:
                endtime = _time() + timeout
            else:
                endtime = None

            try:
                stdout, stderr = self._communicate(input, endtime, timeout)
            except KeyboardInterrupt:
                # https://bugs.python.org/issue25942
                # See the detailed comment in .wait().
                if timeout is not None:
                    sigint_timeout = min(self._sigint_wait_secs,
                                         self._remaining_time(endtime))
                else:
                    sigint_timeout = self._sigint_wait_secs
                self._sigint_wait_secs = 0  # nothing else should wait.
                try:
                    self._wait(timeout=sigint_timeout)
                except TimeoutExpired:
                    pass
                raise  # resume the KeyboardInterrupt

            finally:
                self._communication_started = True

            sts = self.wait(timeout=self._remaining_time(endtime))

        return (stdout, stderr)


    def poll(self):
        """Check if child process has terminated. Set and return returncode
        attribute."""
        return self._internal_poll()


    def _remaining_time(self, endtime):
        """Convenience for _communicate when computing timeouts."""
        if endtime is None:
            return None
        else:
            return endtime - _time()


    def _check_timeout(self, endtime, orig_timeout, stdout_seq, stderr_seq,
                       skip_check_and_raise=False):
        """Convenience for checking if a timeout has expired."""
        if endtime is None:
            return
        if skip_check_and_raise or _time() > endtime:
            raise TimeoutExpired(
                    self.args, orig_timeout,
                    output=b''.join(stdout_seq) if stdout_seq else None,
                    stderr=b''.join(stderr_seq) if stderr_seq else None)


    def wait(self, timeout=None):
        """Wait for child process to terminate; returns self.returncode."""
        if timeout is not None:
            endtime = _time() + timeout
        try:
            return self._wait(timeout=timeout)
        except KeyboardInterrupt:
            # https://bugs.python.org/issue25942
            # The first keyboard interrupt waits briefly for the child to
            # exit under the common assumption that it also received the ^C
            # generated SIGINT and will exit rapidly.
            if timeout is not None:
                sigint_timeout = min(self._sigint_wait_secs,
                                     self._remaining_time(endtime))
            else:
                sigint_timeout = self._sigint_wait_secs
            self._sigint_wait_secs = 0  # nothing else should wait.
            try:
                self._wait(timeout=sigint_timeout)
            except TimeoutExpired:
                pass
            raise  # resume the KeyboardInterrupt

    def _close_pipe_fds(self,
                        p2cread, p2cwrite,
                        c2pread, c2pwrite,
                        errread, errwrite):
        # self._devnull is not always defined.
        devnull_fd = getattr(self, '_devnull', None)

        with contextlib.ExitStack() as stack:
            if _mswindows:
                if p2cread != -1:
                    stack.callback(p2cread.Close)
                if c2pwrite != -1:
                    stack.callback(c2pwrite.Close)
                if errwrite != -1:
                    stack.callback(errwrite.Close)
            else:
                if p2cread != -1 and p2cwrite != -1 and p2cread != devnull_fd:
                    stack.callback(os.close, p2cread)
                if c2pwrite != -1 and c2pread != -1 and c2pwrite != devnull_fd:
                    stack.callback(os.close, c2pwrite)
                if errwrite != -1 and errread != -1 and errwrite != devnull_fd:
                    stack.callback(os.close, errwrite)

            if devnull_fd is not None:
                stack.callback(os.close, devnull_fd)

        # Prevent a double close of these handles/fds from __init__ on error.
        self._closed_child_pipe_fds = True

    if _mswindows:
        #
        # Windows methods
        #
        def _get_handles(self, stdin, stdout, stderr):
            """Construct and return tuple with IO objects:
            p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite
            """
            if stdin is None and stdout is None and stderr is None:
                return (-1, -1, -1, -1, -1, -1)

            p2cread, p2cwrite = -1, -1
            c2pread, c2pwrite = -1, -1
            errread, errwrite = -1, -1

            if stdin is None:
                p2cread = _winapi.GetStdHandle(_winapi.STD_INPUT_HANDLE)
                if p2cread is None:
                    p2cread, _ = _winapi.CreatePipe(None, 0)
                    p2cread = Handle(p2cread)
                    _winapi.CloseHandle(_)
            elif stdin == PIPE:
                p2cread, p2cwrite = _winapi.CreatePipe(None, 0)
                p2cread, p2cwrite = Handle(p2cread), Handle(p2cwrite)
            elif stdin == DEVNULL:
                p2cread = msvcrt.get_osfhandle(self._get_devnull())
            elif isinstance(stdin, int):
                p2cread = msvcrt.get_osfhandle(stdin)
            else:
                # Assuming file-like object
                p2cread = msvcrt.get_osfhandle(stdin.fileno())
            p2cread = self._make_inheritable(p2cread)

            if stdout is None:
                c2pwrite = _winapi.GetStdHandle(_winapi.STD_OUTPUT_HANDLE)
                if c2pwrite is None:
                    _, c2pwrite = _winapi.CreatePipe(None, 0)
                    c2pwrite = Handle(c2pwrite)
                    _winapi.CloseHandle(_)
            elif stdout == PIPE:
                c2pread, c2pwrite = _winapi.CreatePipe(None, 0)
                c2pread, c2pwrite = Handle(c2pread), Handle(c2pwrite)
            elif stdout == DEVNULL:
                c2pwrite = msvcrt.get_osfhandle(self._get_devnull())
            elif isinstance(stdout, int):
                c2pwrite = msvcrt.get_osfhandle(stdout)
            else:
                # Assuming file-like object
                c2pwrite = msvcrt.get_osfhandle(stdout.fileno())
            c2pwrite = self._make_inheritable(c2pwrite)

            if stderr is None:
                errwrite = _winapi.GetStdHandle(_winapi.STD_ERROR_HANDLE)
                if errwrite is None:
                    _, errwrite = _winapi.CreatePipe(None, 0)
                    errwrite = Handle(errwrite)
                    _winapi.CloseHandle(_)
            elif stderr == PIPE:
                errread, errwrite = _winapi.CreatePipe(None, 0)
                errread, errwrite = Handle(errread), Handle(errwrite)
            elif stderr == STDOUT:
                errwrite = c2pwrite
            elif stderr == DEVNULL:
                errwrite = msvcrt.get_osfhandle(self._get_devnull())
            elif isinstance(stderr, int):
                errwrite = msvcrt.get_osfhandle(stderr)
            else:
                # Assuming file-like object
                errwrite = msvcrt.get_osfhandle(stderr.fileno())
            errwrite = self._make_inheritable(errwrite)

            return (p2cread, p2cwrite,
                    c2pread, c2pwrite,
                    errread, errwrite)


        def _make_inheritable(self, handle):
            """Return a duplicate of handle, which is inheritable"""
            h = _winapi.DuplicateHandle(
                _winapi.GetCurrentProcess(), handle,
                _winapi.GetCurrentProcess(), 0, 1,
                _winapi.DUPLICATE_SAME_ACCESS)
            return Handle(h)


        def _filter_handle_list(self, handle_list):
            """Filter out console handles that can't be used
            in lpAttributeList["handle_list"] and make sure the list
            isn't empty. This also removes duplicate handles."""
            # An handle with it's lowest two bits set might be a special console
            # handle that if passed in lpAttributeList["handle_list"], will
            # cause it to fail.
            return list({handle for handle in handle_list
                         if handle & 0x3 != 0x3
                         or _winapi.GetFileType(handle) !=
                            _winapi.FILE_TYPE_CHAR})


        def _execute_child(self, args, executable, preexec_fn, close_fds,
                           pass_fds, cwd, env,
                           startupinfo, creationflags, shell,
                           p2cread, p2cwrite,
                           c2pread, c2pwrite,
                           errread, errwrite,
                           unused_restore_signals,
                           unused_gid, unused_gids, unused_uid,
                           unused_umask,
                           unused_start_new_session):
            """Execute program (MS Windows version)"""

            assert not pass_fds, "pass_fds not supported on Windows."

            if isinstance(args, str):
                pass
            elif isinstance(args, bytes):
                if shell:
                    raise TypeError('bytes args is not allowed on Windows')
                args = list2cmdline([args])
            elif isinstance(args, os.PathLike):
                if shell:
                    raise TypeError('path-like args is not allowed when '
                                    'shell is true')
                args = list2cmdline([args])
            else:
                args = list2cmdline(args)

            if executable is not None:
                executable = os.fsdecode(executable)

            # Process startup details
            if startupinfo is None:
                startupinfo = STARTUPINFO()
            else:
                # bpo-34044: Copy STARTUPINFO since it is modified above,
                # so the caller can reuse it multiple times.
                startupinfo = startupinfo.copy()

            use_std_handles = -1 not in (p2cread, c2pwrite, errwrite)
            if use_std_handles:
                startupinfo.dwFlags |= _winapi.STARTF_USESTDHANDLES
                startupinfo.hStdInput = p2cread
                startupinfo.hStdOutput = c2pwrite
                startupinfo.hStdError = errwrite

            attribute_list = startupinfo.lpAttributeList
            have_handle_list = bool(attribute_list and
                                    "handle_list" in attribute_list and
                                    attribute_list["handle_list"])

            # If we were given an handle_list or need to create one
            if have_handle_list or (use_std_handles and close_fds):
                if attribute_list is None:
                    attribute_list = startupinfo.lpAttributeList = {}
                handle_list = attribute_list["handle_list"] = \
                    list(attribute_list.get("handle_list", []))

                if use_std_handles:
                    handle_list += [int(p2cread), int(c2pwrite), int(errwrite)]

                handle_list[:] = self._filter_handle_list(handle_list)

                if handle_list:
                    if not close_fds:
                        warnings.warn("startupinfo.lpAttributeList['handle_list'] "
                                      "overriding close_fds", RuntimeWarning)

                    # When using the handle_list we always request to inherit
                    # handles but the only handles that will be inherited are
                    # the ones in the handle_list
                    close_fds = False

            if shell:
                startupinfo.dwFlags |= _winapi.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = _winapi.SW_HIDE
                if not executable:
                    # gh-101283: without a fully-qualified path, before Windows
                    # checks the system directories, it first looks in the
                    # application directory, and also the current directory if
                    # NeedCurrentDirectoryForExePathW(ExeName) is true, so try
                    # to avoid executing unqualified "cmd.exe".
                    comspec = os.environ.get('ComSpec')
                    if not comspec:
                        system_root = os.environ.get('SystemRoot', '')
                        comspec = os.path.join(system_root, 'System32', 'cmd.exe')
                        if not os.path.isabs(comspec):
                            raise FileNotFoundError('shell not found: neither %ComSpec% nor %SystemRoot% is set')
                    if os.path.isabs(comspec):
                        executable = comspec
                else:
                    comspec = executable

                args = '{} /c "{}"'.format (comspec, args)

            if cwd is not None:
                cwd = os.fsdecode(cwd)

            sys.audit("subprocess.Popen", executable, args, cwd, env)

            # Start the process
            try:
                hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                                         # no special security
                                         None, None,
                                         int(not close_fds),
                                         creationflags,
                                         env,
                                         cwd,
                                         startupinfo)
            finally:
                # Child is launched. Close the parent's copy of those pipe
                # handles that only the child should have open.  You need
                # to make sure that no handles to the write end of the
                # output pipe are maintained in this process or else the
                # pipe will not close when the child process exits and the
                # ReadFile will hang.
                self._close_pipe_fds(p2cread, p2cwrite,
                                     c2pread, c2pwrite,
                                     errread, errwrite)

            # Retain the process handle, but close the thread handle
            self._child_created = True
            self._handle = Handle(hp)
            self.pid = pid
            _winapi.CloseHandle(ht)

        def _internal_poll(self, _deadstate=None,
                _WaitForSingleObject=_winapi.WaitForSingleObject,
                _WAIT_OBJECT_0=_winapi.WAIT_OBJECT_0,
                _GetExitCodeProcess=_winapi.GetExitCodeProcess):
            """Check if child process has terminated.  Returns returncode
            attribute.

            This method is called by __del__, so it can only refer to objects
            in its local scope.

            """
            if self.returncode is None:
                if _WaitForSingleObject(self._handle, 0) == _WAIT_OBJECT_0:
                    self.returncode = _GetExitCodeProcess(self._handle)
            return self.returncode


        def _wait(self, timeout):
            """Internal implementation of wait() on Windows."""
            if timeout is None:
                timeout_millis = _winapi.INFINITE
            else:
                timeout_millis = int(timeout * 1000)
            if self.returncode is None:
                # API note: Returns immediately if timeout_millis == 0.
                result = _winapi.WaitForSingleObject(self._handle,
                                                     timeout_millis)
                if result == _winapi.WAIT_TIMEOUT:
                    raise TimeoutExpired(self.args, timeout)
                self.returncode = _winapi.GetExitCodeProcess(self._handle)
            return self.returncode


        def _readerthread(self, fh, buffer):
            buffer.append(fh.read())
            fh.close()


        def _communicate(self, input, endtime, orig_timeout):
            # Start reader threads feeding into a list hanging off of this
            # object, unless they've already been started.
            if self.stdout and not hasattr(self, "_stdout_buff"):
                self._stdout_buff = []
                self.stdout_thread = \
                        threading.Thread(target=self._readerthread,
                                         args=(self.stdout, self._stdout_buff))
                self.stdout_thread.daemon = True
                self.stdout_thread.start()
            if self.stderr and not hasattr(self, "_stderr_buff"):
                self._stderr_buff = []
                self.stderr_thread = \
                        threading.Thread(target=self._readerthread,
                                         args=(self.stderr, self._stderr_buff))
                self.stderr_thread.daemon = True
                self.stderr_thread.start()

            if self.stdin:
                self._stdin_write(input)

            # Wait for the reader threads, or time out.  If we time out, the
            # threads remain reading and the fds left open in case the user
            # calls communicate again.
            if self.stdout is not None:
                self.stdout_thread.join(self._remaining_time(endtime))
                if self.stdout_thread.is_alive():
                    raise TimeoutExpired(self.args, orig_timeout)
            if self.stderr is not None:
                self.stderr_thread.join(self._remaining_time(endtime))
                if self.stderr_thread.is_alive():
                    raise TimeoutExpired(self.args, orig_timeout)

            # Collect the output from and close both pipes, now that we know
            # both have been read successfully.
            stdout = None
            stderr = None
            if self.stdout:
                stdout = self._stdout_buff
                self.stdout.close()
            if self.stderr:
                stderr = self._stderr_buff
                self.stderr.close()

            # All data exchanged.  Translate lists into strings.
            stdout = stdout[0] if stdout else None
            stderr = stderr[0] if stderr else None

            return (stdout, stderr)

        def send_signal(self, sig):
            """Send a signal to the process."""
            # Don't signal a process that we know has already died.
            if self.returncode is not None:
                return
            if sig == signal.SIGTERM:
                self.terminate()
            elif sig == signal.CTRL_C_EVENT:
                os.kill(self.pid, signal.CTRL_C_EVENT)
            elif sig == signal.CTRL_BREAK_EVENT:
                os.kill(self.pid, signal.CTRL_BREAK_EVENT)
            else:
                raise ValueError("Unsupported signal: {}".format(sig))

        def terminate(self):
            """Terminates the process."""
            # Don't terminate a process that we know has already died.
            if self.returncode is not None:
                return
            try:
                _winapi.TerminateProcess(self._handle, 1)
            except PermissionError:
                # ERROR_ACCESS_DENIED (winerror 5) is received when the
                # process already died.
                rc = _winapi.GetExitCodeProcess(self._handle)
                if rc == _winapi.STILL_ACTIVE:
                    raise
                self.returncode = rc

        kill = terminate

    else:
        #
        # POSIX methods
        #
        def _get_handles(self, stdin, stdout, stderr):
            """Construct and return tuple with IO objects:
            p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite
            """
            p2cread, p2cwrite = -1, -1
            c2pread, c2pwrite = -1, -1
            errread, errwrite = -1, -1

            if stdin is None:
                pass
            elif stdin == PIPE:
                p2cread, p2cwrite = os.pipe()
                if self.pipesize > 0 and hasattr(fcntl, "F_SETPIPE_SZ"):
                    fcntl.fcntl(p2cwrite, fcntl.F_SETPIPE_SZ, self.pipesize)
            elif stdin == DEVNULL:
                p2cread = self._get_devnull()
            elif isinstance(stdin, int):
                p2cread = stdin
            else:
                # Assuming file-like object
                p2cread = stdin.fileno()

            if stdout is None:
                pass
            elif stdout == PIPE:
                c2pread, c2pwrite = os.pipe()
                if self.pipesize > 0 and hasattr(fcntl, "F_SETPIPE_SZ"):
                    fcntl.fcntl(c2pwrite, fcntl.F_SETPIPE_SZ, self.pipesize)
            elif stdout == DEVNULL:
                c2pwrite = self._get_devnull()
            elif isinstance(stdout, int):
                c2pwrite = stdout
            else:
                # Assuming file-like object
                c2pwrite = stdout.fileno()

            if stderr is None:
                pass
            elif stderr == PIPE:
                errread, errwrite = os.pipe()
                if self.pipesize > 0 and hasattr(fcntl, "F_SETPIPE_SZ"):
                    fcntl.fcntl(errwrite, fcntl.F_SETPIPE_SZ, self.pipesize)
            elif stderr == STDOUT:
                if c2pwrite != -1:
                    errwrite = c2pwrite
                else: # child's stdout is not set, use parent's stdout
                    errwrite = sys.__stdout__.fileno()
            elif stderr == DEVNULL:
                errwrite = self._get_devnull()
            elif isinstance(stderr, int):
                errwrite = stderr
            else:
                # Assuming file-like object
                errwrite = stderr.fileno()

            return (p2cread, p2cwrite,
                    c2pread, c2pwrite,
                    errread, errwrite)


        def _posix_spawn(self, args, executable, env, restore_signals,
                         p2cread, p2cwrite,
                         c2pread, c2pwrite,
                         errread, errwrite):
            """Execute program using os.posix_spawn()."""
            if env is None:
                env = os.environ

            kwargs = {}
            if restore_signals:
                # See _Py_RestoreSignals() in Python/pylifecycle.c
                sigset = []
                for signame in ('SIGPIPE', 'SIGXFZ', 'SIGXFSZ'):
                    signum = getattr(signal, signame, None)
                    if signum is not None:
                        sigset.append(signum)
                kwargs['setsigdef'] = sigset

            file_actions = []
            for fd in (p2cwrite, c2pread, errread):
                if fd != -1:
                    file_actions.append((os.POSIX_SPAWN_CLOSE, fd))
            for fd, fd2 in (
                (p2cread, 0),
                (c2pwrite, 1),
                (errwrite, 2),
            ):
                if fd != -1:
                    file_actions.append((os.POSIX_SPAWN_DUP2, fd, fd2))
            if file_actions:
                kwargs['file_actions'] = file_actions

            self.pid = os.posix_spawn(executable, args, env, **kwargs)
            self._child_created = True

            self._close_pipe_fds(p2cread, p2cwrite,
                                 c2pread, c2pwrite,
                                 errread, errwrite)

        def _execute_child(self, args, executable, preexec_fn, close_fds,
                           pass_fds, cwd, env,
                           startupinfo, creationflags, shell,
                           p2cread, p2cwrite,
                           c2pread, c2pwrite,
                           errread, errwrite,
                           restore_signals,
                           gid, gids, uid, umask,
                           start_new_session):
            """Execute program (POSIX version)"""

            if isinstance(args, (str, bytes)):
                args = [args]
            elif isinstance(args, os.PathLike):
                if shell:
                    raise TypeError('path-like args is not allowed when '
                                    'shell is true')
                args = [args]
            else:
                args = list(args)

            if shell:
                # On Android the default shell is at '/system/bin/sh'.
                unix_shell = ('/system/bin/sh' if
                          hasattr(sys, 'getandroidapilevel') else '/bin/sh')
                args = [unix_shell, "-c"] + args
                if executable:
                    args[0] = executable

            if executable is None:
                executable = args[0]

            sys.audit("subprocess.Popen", executable, args, cwd, env)

            if (_USE_POSIX_SPAWN
                    and os.path.dirname(executable)
                    and preexec_fn is None
                    and not close_fds
                    and not pass_fds
                    and cwd is None
                    and (p2cread == -1 or p2cread > 2)
                    and (c2pwrite == -1 or c2pwrite > 2)
                    and (errwrite == -1 or errwrite > 2)
                    and not start_new_session
                    and gid is None
                    and gids is None
                    and uid is None
                    and umask < 0):
                self._posix_spawn(args, executable, env, restore_signals,
                                  p2cread, p2cwrite,
                                  c2pread, c2pwrite,
                                  errread, errwrite)
                return

            orig_executable = executable

            # For transferring possible exec failure from child to parent.
            # Data format: "exception name:hex errno:description"
            # Pickle is not used; it is complex and involves memory allocation.
            errpipe_read, errpipe_write = os.pipe()
            # errpipe_write must not be in the standard io 0, 1, or 2 fd range.
            low_fds_to_close = []
            while errpipe_write < 3:
                low_fds_to_close.append(errpipe_write)
                errpipe_write = os.dup(errpipe_write)
            for low_fd in low_fds_to_close:
                os.close(low_fd)
            try:
                try:
                    # We must avoid complex work that could involve
                    # malloc or free in the child process to avoid
                    # potential deadlocks, thus we do all this here.
                    # and pass it to fork_exec()

                    if env is not None:
                        env_list = []
                        for k, v in env.items():
                            k = os.fsencode(k)
                            if b'=' in k:
                                raise ValueError("illegal environment variable name")
                            env_list.append(k + b'=' + os.fsencode(v))
                    else:
                        env_list = None  # Use execv instead of execve.
                    executable = os.fsencode(executable)
                    if os.path.dirname(executable):
                        executable_list = (executable,)
                    else:
                        # This matches the behavior of os._execvpe().
                        executable_list = tuple(
                            os.path.join(os.fsencode(dir), executable)
                            for dir in os.get_exec_path(env))
                    fds_to_keep = set(pass_fds)
                    fds_to_keep.add(errpipe_write)
                    self.pid = _posixsubprocess.fork_exec(
                            args, executable_list,
                            close_fds, tuple(sorted(map(int, fds_to_keep))),
                            cwd, env_list,
                            p2cread, p2cwrite, c2pread, c2pwrite,
                            errread, errwrite,
                            errpipe_read, errpipe_write,
                            restore_signals, start_new_session,
                            gid, gids, uid, umask,
                            preexec_fn)
                    self._child_created = True
                finally:
                    # be sure the FD is closed no matter what
                    os.close(errpipe_write)

                self._close_pipe_fds(p2cread, p2cwrite,
                                     c2pread, c2pwrite,
                                     errread, errwrite)

                # Wait for exec to fail or succeed; possibly raising an
                # exception (limited in size)
                errpipe_data = bytearray()
                while True:
                    part = os.read(errpipe_read, 50000)
                    errpipe_data += part
                    if not part or len(errpipe_data) > 50000:
                        break
            finally:
                # be sure the FD is closed no matter what
                os.close(errpipe_read)

            if errpipe_data:
                try:
                    pid, sts = os.waitpid(self.pid, 0)
                    if pid == self.pid:
                        self._handle_exitstatus(sts)
                    else:
                        self.returncode = sys.maxsize
                except ChildProcessError:
                    pass

                try:
                    exception_name, hex_errno, err_msg = (
                            errpipe_data.split(b':', 2))
                    # The encoding here should match the encoding
                    # written in by the subprocess implementations
                    # like _posixsubprocess
                    err_msg = err_msg.decode()
                except ValueError:
                    exception_name = b'SubprocessError'
                    hex_errno = b'0'
                    err_msg = 'Bad exception data from child: {!r}'.format(
                                  bytes(errpipe_data))
                child_exception_type = getattr(
                        builtins, exception_name.decode('ascii'),
                        SubprocessError)
                if issubclass(child_exception_type, OSError) and hex_errno:
                    errno_num = int(hex_errno, 16)
                    child_exec_never_called = (err_msg == "noexec")
                    if child_exec_never_called:
                        err_msg = ""
                        # The error must be from chdir(cwd).
                        err_filename = cwd
                    else:
                        err_filename = orig_executable
                    if errno_num != 0:
                        err_msg = os.strerror(errno_num)
                    raise child_exception_type(errno_num, err_msg, err_filename)
                raise child_exception_type(err_msg)


        def _handle_exitstatus(self, sts,
                               waitstatus_to_exitcode=os.waitstatus_to_exitcode,
                               _WIFSTOPPED=os.WIFSTOPPED,
                               _WSTOPSIG=os.WSTOPSIG):
            """All callers to this function MUST hold self._waitpid_lock."""
            # This method is called (indirectly) by __del__, so it cannot
            # refer to anything outside of its local scope.
            if _WIFSTOPPED(sts):
                self.returncode = -_WSTOPSIG(sts)
            else:
                self.returncode = waitstatus_to_exitcode(sts)

        def _internal_poll(self, _deadstate=None, _waitpid=os.waitpid,
                _WNOHANG=os.WNOHANG, _ECHILD=errno.ECHILD):
            """Check if child process has terminated.  Returns returncode
            attribute.

            This method is called by __del__, so it cannot reference anything
            outside of the local scope (nor can any methods it calls).

            """
            if self.returncode is None:
                if not self._waitpid_lock.acquire(False):
                    # Something else is busy calling waitpid.  Don't allow two
                    # at once.  We know nothing yet.
                    return None
                try:
                    if self.returncode is not None:
                        return self.returncode  # Another thread waited.
                    pid, sts = _waitpid(self.pid, _WNOHANG)
                    if pid == self.pid:
                        self._handle_exitstatus(sts)
                except OSError as e:
                    if _deadstate is not None:
                        self.returncode = _deadstate
                    elif e.errno == _ECHILD:
                        # This happens if SIGCLD is set to be ignored or
                        # waiting for child processes has otherwise been
                        # disabled for our process.  This child is dead, we
                        # can't get the status.
                        # http://bugs.python.org/issue15756
                        self.returncode = 0
                finally:
                    self._waitpid_lock.release()
            return self.returncode


        def _try_wait(self, wait_flags):
            """All callers to this function MUST hold self._waitpid_lock."""
            try:
                (pid, sts) = os.waitpid(self.pid, wait_flags)
            except ChildProcessError:
                # This happens if SIGCLD is set to be ignored or waiting
                # for child processes has otherwise been disabled for our
                # process.  This child is dead, we can't get the status.
                pid = self.pid
                sts = 0
            return (pid, sts)


        def _wait(self, timeout):
            """Internal implementation of wait() on POSIX."""
            if self.returncode is not None:
                return self.returncode

            if timeout is not None:
                endtime = _time() + timeout
                # Enter a busy loop if we have a timeout.  This busy loop was
                # cribbed from Lib/threading.py in Thread.wait() at r71065.
                delay = 0.0005 # 500 us -> initial delay of 1 ms
                while True:
                    if self._waitpid_lock.acquire(False):
                        try:
                            if self.returncode is not None:
                                break  # Another thread waited.
                            (pid, sts) = self._try_wait(os.WNOHANG)
                            assert pid == self.pid or pid == 0
                            if pid == self.pid:
                                self._handle_exitstatus(sts)
                                break
                        finally:
                            self._waitpid_lock.release()
                    remaining = self._remaining_time(endtime)
                    if remaining <= 0:
                        raise TimeoutExpired(self.args, timeout)
                    delay = min(delay * 2, remaining, .05)
                    time.sleep(delay)
            else:
                while self.returncode is None:
                    with self._waitpid_lock:
                        if self.returncode is not None:
                            break  # Another thread waited.
                        (pid, sts) = self._try_wait(0)
                        # Check the pid and loop as waitpid has been known to
                        # return 0 even without WNOHANG in odd situations.
                        # http://bugs.python.org/issue14396.
                        if pid == self.pid:
                            self._handle_exitstatus(sts)
            return self.returncode


        def _communicate(self, input, endtime, orig_timeout):
            if self.stdin and not self._communication_started:
                # Flush stdio buffer.  This might block, if the user has
                # been writing to .stdin in an uncontrolled fashion.
                try:
                    self.stdin.flush()
                except BrokenPipeError:
                    pass  # communicate() must ignore BrokenPipeError.
                if not input:
                    try:
                        self.stdin.close()
                    except BrokenPipeError:
                        pass  # communicate() must ignore BrokenPipeError.

            stdout = None
            stderr = None

            # Only create this mapping if we haven't already.
            if not self._communication_started:
                self._fileobj2output = {}
                if self.stdout:
                    self._fileobj2output[self.stdout] = []
                if self.stderr:
                    self._fileobj2output[self.stderr] = []

            if self.stdout:
                stdout = self._fileobj2output[self.stdout]
            if self.stderr:
                stderr = self._fileobj2output[self.stderr]

            self._save_input(input)

            if self._input:
                input_view = memoryview(self._input)

            with _PopenSelector() as selector:
                if self.stdin and input:
                    selector.register(self.stdin, selectors.EVENT_WRITE)
                if self.stdout and not self.stdout.closed:
                    selector.register(self.stdout, selectors.EVENT_READ)
                if self.stderr and not self.stderr.closed:
                    selector.register(self.stderr, selectors.EVENT_READ)

                while selector.get_map():
                    timeout = self._remaining_time(endtime)
                    if timeout is not None and timeout < 0:
                        self._check_timeout(endtime, orig_timeout,
                                            stdout, stderr,
                                            skip_check_and_raise=True)
                        raise RuntimeError(  # Impossible :)
                            '_check_timeout(..., skip_check_and_raise=True) '
                            'failed to raise TimeoutExpired.')

                    ready = selector.select(timeout)
                    self._check_timeout(endtime, orig_timeout, stdout, stderr)

                    # XXX Rewrite these to use non-blocking I/O on the file
                    # objects; they are no longer using C stdio!

                    for key, events in ready:
                        if key.fileobj is self.stdin:
                            chunk = input_view[self._input_offset :
                                               self._input_offset + _PIPE_BUF]
                            try:
                                self._input_offset += os.write(key.fd, chunk)
                            except BrokenPipeError:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                            else:
                                if self._input_offset >= len(self._input):
                                    selector.unregister(key.fileobj)
                                    key.fileobj.close()
                        elif key.fileobj in (self.stdout, self.stderr):
                            data = os.read(key.fd, 32768)
                            if not data:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                            self._fileobj2output[key.fileobj].append(data)

            self.wait(timeout=self._remaining_time(endtime))

            # All data exchanged.  Translate lists into strings.
            if stdout is not None:
                stdout = b''.join(stdout)
            if stderr is not None:
                stderr = b''.join(stderr)

            # Translate newlines, if requested.
            # This also turns bytes into strings.
            if self.text_mode:
                if stdout is not None:
                    stdout = self._translate_newlines(stdout,
                                                      self.stdout.encoding,
                                                      self.stdout.errors)
                if stderr is not None:
                    stderr = self._translate_newlines(stderr,
                                                      self.stderr.encoding,
                                                      self.stderr.errors)

            return (stdout, stderr)


        def _save_input(self, input):
            # This method is called from the _communicate_with_*() methods
            # so that if we time out while communicating, we can continue
            # sending input if we retry.
            if self.stdin and self._input is None:
                self._input_offset = 0
                self._input = input
                if input is not None and self.text_mode:
                    self._input = self._input.encode(self.stdin.encoding,
                                                     self.stdin.errors)


        def send_signal(self, sig):
            """Send a signal to the process."""
            # bpo-38630: Polling reduces the risk of sending a signal to the
            # wrong process if the process completed, the Popen.returncode
            # attribute is still None, and the pid has been reassigned
            # (recycled) to a new different process. This race condition can
            # happens in two cases.
            #
            # Case 1. Thread A calls Popen.poll(), thread B calls
            # Popen.send_signal(). In thread A, waitpid() succeed and returns
            # the exit status. Thread B calls kill() because poll() in thread A
            # did not set returncode yet. Calling poll() in thread B prevents
            # the race condition thanks to Popen._waitpid_lock.
            #
            # Case 2. waitpid(pid, 0) has been called directly, without
            # using Popen methods: returncode is still None is this case.
            # Calling Popen.poll() will set returncode to a default value,
            # since waitpid() fails with ProcessLookupError.
            self.poll()
            if self.returncode is not None:
                # Skip signalling a process that we know has already died.
                return

            # The race condition can still happen if the race condition
            # described above happens between the returncode test
            # and the kill() call.
            try:
                os.kill(self.pid, sig)
            except ProcessLookupError:
                # Supress the race condition error; bpo-40550.
                pass

        def terminate(self):
            """Terminate the process with SIGTERM
            """
            self.send_signal(signal.SIGTERM)

        def kill(self):
            """Kill the process with SIGKILL
            """
            self.send_signal(signal.SIGKILL)
<|endoftext|>
try:
    from _decimal import *
    from _decimal import __doc__
    from _decimal import __version__
    from _decimal import __libmpdec_version__
except ImportError:
    from _pydecimal import *
    from _pydecimal import __doc__
    from _pydecimal import __version__
    from _pydecimal import __libmpdec_version__
<|endoftext|>"""Append module search paths for third-party packages to sys.path.

****************************************************************
* This module is automatically imported during initialization. *
****************************************************************

This will append site-specific paths to the module search path.  On
Unix (including Mac OSX), it starts with sys.prefix and
sys.exec_prefix (if different) and appends
lib/python3/dist-packages.
On other platforms (such as Windows), it tries each of the
prefixes directly, as well as with lib/site-packages appended.  The
resulting directories, if they exist, are appended to sys.path, and
also inspected for path configuration files.

For Debian and derivatives, this sys.path is augmented with directories
for packages distributed within the distribution. Local addons go
into /usr/local/lib/python<version>/dist-packages, Debian addons
install into /usr/lib/python3/dist-packages.
/usr/lib/python<version>/site-packages is not used.

If a file named "pyvenv.cfg" exists one directory above sys.executable,
sys.prefix and sys.exec_prefix are set to that directory and
it is also checked for site-packages (sys.base_prefix and
sys.base_exec_prefix will always be the "real" prefixes of the Python
installation). If "pyvenv.cfg" (a bootstrap configuration file) contains
the key "include-system-site-packages" set to anything other than "false"
(case-insensitive), the system-level prefixes will still also be
searched for site-packages; otherwise they won't.

All of the resulting site-specific directories, if they exist, are
appended to sys.path, and also inspected for path configuration
files.

A path configuration file is a file whose name has the form
<package>.pth; its contents are additional directories (one per line)
to be added to sys.path.  Non-existing directories (or
non-directories) are never added to sys.path; no directory is added to
sys.path more than once.  Blank lines and lines beginning with
'#' are skipped. Lines starting with 'import' are executed.

For example, suppose sys.prefix and sys.exec_prefix are set to
/usr/local and there is a directory /usr/local/lib/python2.5/site-packages
with three subdirectories, foo, bar and spam, and two path
configuration files, foo.pth and bar.pth.  Assume foo.pth contains the
following:

  # foo package configuration
  foo
  bar
  bletch

and bar.pth contains:

  # bar package configuration
  bar

Then the following directories are added to sys.path, in this order:

  /usr/local/lib/python2.5/site-packages/bar
  /usr/local/lib/python2.5/site-packages/foo

Note that bletch is omitted because it doesn't exist; bar precedes foo
because bar.pth comes alphabetically before foo.pth; and spam is
omitted because it is not mentioned in either path configuration file.

The readline module is also automatically configured to enable
completion for systems that support it.  This can be overridden in
sitecustomize, usercustomize or PYTHONSTARTUP.  Starting Python in
isolated mode (-I) disables automatic readline configuration.

After these operations, an attempt is made to import a module
named sitecustomize, which can perform arbitrary additional
site-specific customizations.  If this import fails with an
ImportError exception, it is silently ignored.
"""

import sys
import os
import builtins
import _sitebuiltins
import io

# Prefixes for site-packages; add additional prefixes like /usr/local here
PREFIXES = [sys.prefix, sys.exec_prefix]
# Enable per user site-packages directory
# set it to False to disable the feature or True to force the feature
ENABLE_USER_SITE = None

# for distutils.commands.install
# These values are initialized by the getuserbase() and getusersitepackages()
# functions, through the main() function when Python starts.
USER_SITE = None
USER_BASE = None


def _trace(message):
    if sys.flags.verbose:
        print(message, file=sys.stderr)


def makepath(*paths):
    dir = os.path.join(*paths)
    try:
        dir = os.path.abspath(dir)
    except OSError:
        pass
    return dir, os.path.normcase(dir)


def abs_paths():
    """Set all module __file__ and __cached__ attributes to an absolute path"""
    for m in set(sys.modules.values()):
        loader_module = None
        try:
            loader_module = m.__loader__.__module__
        except AttributeError:
            try:
                loader_module = m.__spec__.loader.__module__
            except AttributeError:
                pass
        if loader_module not in {'_frozen_importlib', '_frozen_importlib_external'}:
            continue   # don't mess with a PEP 302-supplied __file__
        try:
            m.__file__ = os.path.abspath(m.__file__)
        except (AttributeError, OSError, TypeError):
            pass
        try:
            m.__cached__ = os.path.abspath(m.__cached__)
        except (AttributeError, OSError, TypeError):
            pass


def removeduppaths():
    """ Remove duplicate entries from sys.path along with making them
    absolute"""
    # This ensures that the initial path provided by the interpreter contains
    # only absolute pathnames, even if we're running from the build directory.
    L = []
    known_paths = set()
    for dir in sys.path:
        # Filter out duplicate paths (on case-insensitive file systems also
        # if they only differ in case); turn relative paths into absolute
        # paths.
        dir, dircase = makepath(dir)
        if dircase not in known_paths:
            L.append(dir)
            known_paths.add(dircase)
    sys.path[:] = L
    return known_paths


def _init_pathinfo():
    """Return a set containing all existing file system items from sys.path."""
    d = set()
    for item in sys.path:
        try:
            if os.path.exists(item):
                _, itemcase = makepath(item)
                d.add(itemcase)
        except TypeError:
            continue
    return d


def addpackage(sitedir, name, known_paths):
    """Process a .pth file within the site-packages directory:
       For each line in the file, either combine it with sitedir to a path
       and add that to known_paths, or execute it if it starts with 'import '.
    """
    if known_paths is None:
        known_paths = _init_pathinfo()
        reset = True
    else:
        reset = False
    fullname = os.path.join(sitedir, name)
    _trace(f"Processing .pth file: {fullname!r}")
    try:
        # locale encoding is not ideal especially on Windows. But we have used
        # it for a long time. setuptools uses the locale encoding too.
        f = io.TextIOWrapper(io.open_code(fullname), encoding="locale")
    except OSError:
        return
    with f:
        for n, line in enumerate(f):
            if line.startswith("#"):
                continue
            if line.strip() == "":
                continue
            try:
                if line.startswith(("import ", "import\t")):
                    exec(line)
                    continue
                line = line.rstrip()
                dir, dircase = makepath(sitedir, line)
                if not dircase in known_paths and os.path.exists(dir):
                    sys.path.append(dir)
                    known_paths.add(dircase)
            except Exception:
                print("Error processing line {:d} of {}:\n".format(n+1, fullname),
                      file=sys.stderr)
                import traceback
                for record in traceback.format_exception(*sys.exc_info()):
                    for line in record.splitlines():
                        print('  '+line, file=sys.stderr)
                print("\nRemainder of file ignored", file=sys.stderr)
                break
    if reset:
        known_paths = None
    return known_paths


def addsitedir(sitedir, known_paths=None):
    """Add 'sitedir' argument to sys.path if missing and handle .pth files in
    'sitedir'"""
    _trace(f"Adding directory: {sitedir!r}")
    if known_paths is None:
        known_paths = _init_pathinfo()
        reset = True
    else:
        reset = False
    sitedir, sitedircase = makepath(sitedir)
    if not sitedircase in known_paths:
        sys.path.append(sitedir)        # Add path component
        known_paths.add(sitedircase)
    try:
        names = os.listdir(sitedir)
    except OSError:
        return
    names = [name for name in names if name.endswith(".pth")]
    for name in sorted(names):
        addpackage(sitedir, name, known_paths)
    if reset:
        known_paths = None
    return known_paths


def check_enableusersite():
    """Check if user site directory is safe for inclusion

    The function tests for the command line flag (including environment var),
    process uid/gid equal to effective uid/gid.

    None: Disabled for security reasons
    False: Disabled by user (command line option)
    True: Safe and enabled
    """
    if sys.flags.no_user_site:
        return False

    if hasattr(os, "getuid") and hasattr(os, "geteuid"):
        # check process uid == effective uid
        if os.geteuid() != os.getuid():
            return None
    if hasattr(os, "getgid") and hasattr(os, "getegid"):
        # check process gid == effective gid
        if os.getegid() != os.getgid():
            return None

    return True


# NOTE: sysconfig and it's dependencies are relatively large but site module
# needs very limited part of them.
# To speedup startup time, we have copy of them.
#
# See https://bugs.python.org/issue29585

# Copy of sysconfig._getuserbase()
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
                        "%d.%d" % sys.version_info[:2])

    return joinuser("~", ".local")


# Same to sysconfig.get_path('purelib', os.name+'_user')
def _get_path(userbase):
    version = sys.version_info

    if os.name == 'nt':
        ver_nodot = sys.winver.replace('.', '')
        return f'{userbase}\\Python{ver_nodot}\\site-packages'

    if sys.platform == 'darwin' and sys._framework:
        return f'{userbase}/lib/python/site-packages'

    return f'{userbase}/lib/python{version[0]}.{version[1]}/site-packages'


def getuserbase():
    """Returns the `user base` directory path.

    The `user base` directory can be used to store data. If the global
    variable ``USER_BASE`` is not initialized yet, this function will also set
    it.
    """
    global USER_BASE
    if USER_BASE is None:
        USER_BASE = _getuserbase()
    return USER_BASE


def getusersitepackages():
    """Returns the user-specific site-packages directory path.

    If the global variable ``USER_SITE`` is not initialized yet, this
    function will also set it.
    """
    global USER_SITE, ENABLE_USER_SITE
    userbase = getuserbase() # this will also set USER_BASE

    if USER_SITE is None:
        if userbase is None:
            ENABLE_USER_SITE = False # disable user site and return None
        else:
            USER_SITE = _get_path(userbase)

    return USER_SITE

def addusersitepackages(known_paths):
    """Add a per user site-package to sys.path

    Each user has its own python directory with site-packages in the
    home directory.
    """
    # get the per user site-package path
    # this call will also make sure USER_BASE and USER_SITE are set
    _trace("Processing user site-packages")
    user_site = getusersitepackages()

    if ENABLE_USER_SITE and os.path.isdir(user_site):
        addsitedir(user_site, known_paths)
    return known_paths

def getsitepackages(prefixes=None):
    """Returns a list containing all global site-packages directories.

    For each directory present in ``prefixes`` (or the global ``PREFIXES``),
    this function will find its `site-packages` subdirectory depending on the
    system environment, and will return a list of full paths.
    """
    sitepackages = []
    seen = set()

    if prefixes is None:
        prefixes = PREFIXES

    is_virtual_environment = sys.base_prefix != sys.prefix or hasattr(sys, "real_prefix")

    for prefix in prefixes:
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)

        libdirs = [sys.platlibdir]
        if sys.platlibdir != "lib":
            libdirs.append("lib")

        if os.sep == '/':
            if is_virtual_environment:
                sitepackages.append(os.path.join(prefix, "lib",
                                                 "python%d.%d" % sys.version_info[:2],
                                                 "site-packages"))
            sitepackages.append(os.path.join(prefix, "local/lib",
                                             "python%d.%d" % sys.version_info[:2],
                                             "dist-packages"))
            sitepackages.append(os.path.join(prefix, "lib",
                                             "python3",
                                             "dist-packages"))
            # this one is deprecated for Debian
            for libdir in libdirs:
                path = os.path.join(prefix, libdir,
                                    "python%d.%d" % sys.version_info[:2],
                                    "dist-packages")
                sitepackages.append(path)
        else:
            sitepackages.append(prefix)

            for libdir in libdirs:
                path = os.path.join(prefix, libdir, "site-packages")
                sitepackages.append(path)
    return sitepackages

def addsitepackages(known_paths, prefixes=None):
    """Add site-packages to sys.path"""
    _trace("Processing global site-packages")
    for sitedir in getsitepackages(prefixes):
        if os.path.isdir(sitedir):
            addsitedir(sitedir, known_paths)

    return known_paths

def setquit():
    """Define new builtins 'quit' and 'exit'.

    These are objects which make the interpreter exit when called.
    The repr of each object contains a hint at how it works.

    """
    if os.sep == '\\':
        eof = 'Ctrl-Z plus Return'
    else:
        eof = 'Ctrl-D (i.e. EOF)'

    builtins.quit = _sitebuiltins.Quitter('quit', eof)
    builtins.exit = _sitebuiltins.Quitter('exit', eof)


def setcopyright():
    """Set 'copyright' and 'credits' in builtins"""
    builtins.copyright = _sitebuiltins._Printer("copyright", sys.copyright)
    if sys.platform[:4] == 'java':
        builtins.credits = _sitebuiltins._Printer(
            "credits",
            "Jython is maintained by the Jython developers (www.jython.org).")
    else:
        builtins.credits = _sitebuiltins._Printer("credits", """\
    Thanks to CWI, CNRI, BeOpen.com, Zope Corporation and a cast of thousands
    for supporting Python development.  See www.python.org for more information.""")
    files, dirs = [], []
    # Not all modules are required to have a __file__ attribute.  See
    # PEP 420 for more details.
    if hasattr(os, '__file__'):
        here = os.path.dirname(os.__file__)
        files.extend(["LICENSE.txt", "LICENSE"])
        dirs.extend([os.path.join(here, os.pardir), here, os.curdir])
    builtins.license = _sitebuiltins._Printer(
        "license",
        "See https://www.python.org/psf/license/",
        files, dirs)


def sethelper():
    builtins.help = _sitebuiltins._Helper()

def enablerlcompleter():
    """Enable default readline configuration on interactive prompts, by
    registering a sys.__interactivehook__.

    If the readline module can be imported, the hook will set the Tab key
    as completion key and register ~/.python_history as history file.
    This can be overridden in the sitecustomize or usercustomize module,
    or in a PYTHONSTARTUP file.
    """
    def register_readline():
        import atexit
        try:
            import readline
            import rlcompleter
        except ImportError:
            return

        # Reading the initialization (config) file may not be enough to set a
        # completion key, so we set one first and then read the file.
        readline_doc = getattr(readline, '__doc__', '')
        if readline_doc is not None and 'libedit' in readline_doc:
            readline.parse_and_bind('bind ^I rl_complete')
        else:
            readline.parse_and_bind('tab: complete')

        try:
            readline.read_init_file()
        except OSError:
            # An OSError here could have many causes, but the most likely one
            # is that there's no .inputrc file (or .editrc file in the case of
            # Mac OS X + libedit) in the expected location.  In that case, we
            # want to ignore the exception.
            pass

        if readline.get_current_history_length() == 0:
            # If no history was loaded, default to .python_history.
            # The guard is necessary to avoid doubling history size at
            # each interpreter exit when readline was already configured
            # through a PYTHONSTARTUP hook, see:
            # http://bugs.python.org/issue5845#msg198636
            history = os.path.join(os.path.expanduser('~'),
                                   '.python_history')
            try:
                readline.read_history_file(history)
            except OSError:
                pass

            def write_history():
                try:
                    readline.write_history_file(history)
                except OSError:
                    # bpo-19891, bpo-41193: Home directory does not exist
                    # or is not writable, or the filesystem is read-only.
                    pass

            atexit.register(write_history)

    sys.__interactivehook__ = register_readline

def venv(known_paths):
    global PREFIXES, ENABLE_USER_SITE

    env = os.environ
    if sys.platform == 'darwin' and '__PYVENV_LAUNCHER__' in env:
        executable = sys._base_executable = os.environ['__PYVENV_LAUNCHER__']
    else:
        executable = sys.executable
    exe_dir, _ = os.path.split(os.path.abspath(executable))
    site_prefix = os.path.dirname(exe_dir)
    sys._home = None
    conf_basename = 'pyvenv.cfg'
    candidate_confs = [
        conffile for conffile in (
            os.path.join(exe_dir, conf_basename),
            os.path.join(site_prefix, conf_basename)
            )
        if os.path.isfile(conffile)
        ]

    if candidate_confs:
        virtual_conf = candidate_confs[0]
        system_site = "true"
        # Issue 25185: Use UTF-8, as that's what the venv module uses when
        # writing the file.
        with open(virtual_conf, encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip().lower()
                    value = value.strip()
                    if key == 'include-system-site-packages':
                        system_site = value.lower()
                    elif key == 'home':
                        sys._home = value

        sys.prefix = sys.exec_prefix = site_prefix

        # Doing this here ensures venv takes precedence over user-site
        addsitepackages(known_paths, [sys.prefix])

        # addsitepackages will process site_prefix again if its in PREFIXES,
        # but that's ok; known_paths will prevent anything being added twice
        if system_site == "true":
            PREFIXES.insert(0, sys.prefix)
        else:
            PREFIXES = [sys.prefix]
            ENABLE_USER_SITE = False

    return known_paths


def execsitecustomize():
    """Run custom site specific code, if available."""
    try:
        try:
            import sitecustomize
        except ImportError as exc:
            if exc.name == 'sitecustomize':
                pass
            else:
                raise
    except Exception as err:
        if sys.flags.verbose:
            sys.excepthook(*sys.exc_info())
        else:
            sys.stderr.write(
                "Error in sitecustomize; set PYTHONVERBOSE for traceback:\n"
                "%s: %s\n" %
                (err.__class__.__name__, err))


def execusercustomize():
    """Run custom user specific code, if available."""
    try:
        try:
            import usercustomize
        except ImportError as exc:
            if exc.name == 'usercustomize':
                pass
            else:
                raise
    except Exception as err:
        if sys.flags.verbose:
            sys.excepthook(*sys.exc_info())
        else:
            sys.stderr.write(
                "Error in usercustomize; set PYTHONVERBOSE for traceback:\n"
                "%s: %s\n" %
                (err.__class__.__name__, err))


def main():
    """Add standard site-specific directories to the module search path.

    This function is called automatically when this module is imported,
    unless the python interpreter was started with the -S flag.
    """
    global ENABLE_USER_SITE

    orig_path = sys.path[:]
    known_paths = removeduppaths()
    if orig_path != sys.path:
        # removeduppaths() might make sys.path absolute.
        # fix __file__ and __cached__ of already imported modules too.
        abs_paths()

    known_paths = venv(known_paths)
    if ENABLE_USER_SITE is None:
        ENABLE_USER_SITE = check_enableusersite()
    known_paths = addusersitepackages(known_paths)
    known_paths = addsitepackages(known_paths)
    setquit()
    setcopyright()
    sethelper()
    if not sys.flags.isolated:
        enablerlcompleter()
    execsitecustomize()
    if ENABLE_USER_SITE:
        execusercustomize()

# Prevent extending of sys.path when python was started with -S and
# site is imported later.
if not sys.flags.no_site:
    main()

def _script():
    help = """\
    %s [--user-base] [--user-site]

    Without arguments print some useful information
    With arguments print the value of USER_BASE and/or USER_SITE separated
    by '%s'.

    Exit codes with --user-base or --user-site:
      0 - user site directory is enabled
      1 - user site directory is disabled by user
      2 - user site directory is disabled by super user
          or for security reasons
     >2 - unknown error
    """
    args = sys.argv[1:]
    if not args:
        user_base = getuserbase()
        user_site = getusersitepackages()
        print("sys.path = [")
        for dir in sys.path:
            print("    %r," % (dir,))
        print("]")
        def exists(path):
            if path is not None and os.path.isdir(path):
                return "exists"
            else:
                return "doesn't exist"
        print(f"USER_BASE: {user_base!r} ({exists(user_base)})")
        print(f"USER_SITE: {user_site!r} ({exists(user_site)})")
        print(f"ENABLE_USER_SITE: {ENABLE_USER_SITE!r}")
        sys.exit(0)

    buffer = []
    if '--user-base' in args:
        buffer.append(USER_BASE)
    if '--user-site' in args:
        buffer.append(USER_SITE)

    if buffer:
        print(os.pathsep.join(buffer))
        if ENABLE_USER_SITE:
            sys.exit(0)
        elif ENABLE_USER_SITE is False:
            sys.exit(1)
        elif ENABLE_USER_SITE is None:
            sys.exit(2)
        else:
            sys.exit(3)
    else:
        import textwrap
        print(textwrap.dedent(help % (sys.argv[0], os.pathsep)))
        sys.exit(10)

if __name__ == '__main__':
    _script()
<|endoftext|>import sys
from types import MappingProxyType, DynamicClassAttribute


__all__ = [
        'EnumMeta',
        'Enum', 'IntEnum', 'Flag', 'IntFlag',
        'auto', 'unique',
        ]


def _is_descriptor(obj):
    """
    Returns True if obj is a descriptor, False otherwise.
    """
    return (
            hasattr(obj, '__get__') or
            hasattr(obj, '__set__') or
            hasattr(obj, '__delete__')
            )

def _is_dunder(name):
    """
    Returns True if a __dunder__ name, False otherwise.
    """
    return (
            len(name) > 4 and
            name[:2] == name[-2:] == '__' and
            name[2] != '_' and
            name[-3] != '_'
            )

def _is_sunder(name):
    """
    Returns True if a _sunder_ name, False otherwise.
    """
    return (
            len(name) > 2 and
            name[0] == name[-1] == '_' and
            name[1:2] != '_' and
            name[-2:-1] != '_'
            )

def _is_private(cls_name, name):
    # do not use `re` as `re` imports `enum`
    pattern = '_%s__' % (cls_name, )
    pat_len = len(pattern)
    if (
            len(name) > pat_len
            and name.startswith(pattern)
            and name[pat_len:pat_len+1] != ['_']
            and (name[-1] != '_' or name[-2] != '_')
        ):
        return True
    else:
        return False

def _make_class_unpicklable(cls):
    """
    Make the given class un-picklable.
    """
    def _break_on_call_reduce(self, proto):
        raise TypeError('%r cannot be pickled' % self)
    cls.__reduce_ex__ = _break_on_call_reduce
    cls.__module__ = '<unknown>'

_auto_null = object()
class auto:
    """
    Instances are replaced with an appropriate value in Enum class suites.
    """
    value = _auto_null


class _EnumDict(dict):
    """
    Track enum member order and ensure member names are not reused.

    EnumMeta will use the names found in self._member_names as the
    enumeration member names.
    """
    def __init__(self):
        super().__init__()
        self._member_names = []
        self._last_values = []
        self._ignore = []
        self._auto_called = False

    def __setitem__(self, key, value):
        """
        Changes anything not dundered or not a descriptor.

        If an enum member name is used twice, an error is raised; duplicate
        values are not checked for.

        Single underscore (sunder) names are reserved.
        """
        if _is_private(self._cls_name, key):
            import warnings
            warnings.warn(
                    "private variables, such as %r, will be normal attributes in 3.11"
                        % (key, ),
                    DeprecationWarning,
                    stacklevel=2,
                    )
        if _is_sunder(key):
            if key not in (
                    '_order_', '_create_pseudo_member_',
                    '_generate_next_value_', '_missing_', '_ignore_',
                    ):
                raise ValueError('_names_ are reserved for future Enum use')
            if key == '_generate_next_value_':
                # check if members already defined as auto()
                if self._auto_called:
                    raise TypeError("_generate_next_value_ must be defined before members")
                setattr(self, '_generate_next_value', value)
            elif key == '_ignore_':
                if isinstance(value, str):
                    value = value.replace(',',' ').split()
                else:
                    value = list(value)
                self._ignore = value
                already = set(value) & set(self._member_names)
                if already:
                    raise ValueError(
                            '_ignore_ cannot specify already set names: %r'
                            % (already, )
                            )
        elif _is_dunder(key):
            if key == '__order__':
                key = '_order_'
        elif key in self._member_names:
            # descriptor overwriting an enum?
            raise TypeError('Attempted to reuse key: %r' % key)
        elif key in self._ignore:
            pass
        elif not _is_descriptor(value):
            if key in self:
                # enum overwriting a descriptor?
                raise TypeError('%r already defined as: %r' % (key, self[key]))
            if isinstance(value, auto):
                if value.value == _auto_null:
                    value.value = self._generate_next_value(
                            key,
                            1,
                            len(self._member_names),
                            self._last_values[:],
                            )
                    self._auto_called = True
                value = value.value
            self._member_names.append(key)
            self._last_values.append(value)
        super().__setitem__(key, value)


# Dummy value for Enum as EnumMeta explicitly checks for it, but of course
# until EnumMeta finishes running the first time the Enum class doesn't exist.
# This is also why there are checks in EnumMeta like `if Enum is not None`
Enum = None

class EnumMeta(type):
    """
    Metaclass for Enum
    """
    @classmethod
    def __prepare__(metacls, cls, bases, **kwds):
        # check that previous enum members do not exist
        metacls._check_for_existing_members(cls, bases)
        # create the namespace dict
        enum_dict = _EnumDict()
        enum_dict._cls_name = cls
        # inherit previous flags and _generate_next_value_ function
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        if first_enum is not None:
            enum_dict['_generate_next_value_'] = getattr(
                    first_enum, '_generate_next_value_', None,
                    )
        return enum_dict

    def __new__(metacls, cls, bases, classdict, **kwds):
        # an Enum class is final once enumeration items have been defined; it
        # cannot be mixed with other types (int, float, etc.) if it has an
        # inherited __new__ unless a new __new__ is defined (or the resulting
        # class will fail).
        #
        # remove any keys listed in _ignore_
        classdict.setdefault('_ignore_', []).append('_ignore_')
        ignore = classdict['_ignore_']
        for key in ignore:
            classdict.pop(key, None)
        member_type, first_enum = metacls._get_mixins_(cls, bases)
        __new__, save_new, use_args = metacls._find_new_(
                classdict, member_type, first_enum,
                )

        # save enum items into separate mapping so they don't get baked into
        # the new class
        enum_members = {k: classdict[k] for k in classdict._member_names}
        for name in classdict._member_names:
            del classdict[name]

        # adjust the sunders
        _order_ = classdict.pop('_order_', None)

        # check for illegal enum names (any others?)
        invalid_names = set(enum_members) & {'mro', ''}
        if invalid_names:
            raise ValueError('Invalid enum member name: {0}'.format(
                ','.join(invalid_names)))

        # create a default docstring if one has not been provided
        if '__doc__' not in classdict:
            classdict['__doc__'] = 'An enumeration.'

        enum_class = super().__new__(metacls, cls, bases, classdict, **kwds)
        enum_class._member_names_ = []               # names in definition order
        enum_class._member_map_ = {}                 # name->value map
        enum_class._member_type_ = member_type

        # save DynamicClassAttribute attributes from super classes so we know
        # if we can take the shortcut of storing members in the class dict
        dynamic_attributes = {
                k for c in enum_class.mro()
                for k, v in c.__dict__.items()
                if isinstance(v, DynamicClassAttribute)
                }

        # Reverse value->name map for hashable values.
        enum_class._value2member_map_ = {}

        # If a custom type is mixed into the Enum, and it does not know how
        # to pickle itself, pickle.dumps will succeed but pickle.loads will
        # fail.  Rather than have the error show up later and possibly far
        # from the source, sabotage the pickle protocol for this class so
        # that pickle.dumps also fails.
        #
        # However, if the new class implements its own __reduce_ex__, do not
        # sabotage -- it's on them to make sure it works correctly.  We use
        # __reduce_ex__ instead of any of the others as it is preferred by
        # pickle over __reduce__, and it handles all pickle protocols.
        if '__reduce_ex__' not in classdict:
            if member_type is not object:
                methods = ('__getnewargs_ex__', '__getnewargs__',
                        '__reduce_ex__', '__reduce__')
                if not any(m in member_type.__dict__ for m in methods):
                    if '__new__' in classdict:
                        # too late, sabotage
                        _make_class_unpicklable(enum_class)
                    else:
                        # final attempt to verify that pickling would work:
                        # travel mro until __new__ is found, checking for
                        # __reduce__ and friends along the way -- if any of them
                        # are found before/when __new__ is found, pickling should
                        # work
                        sabotage = None
                        for chain in bases:
                            for base in chain.__mro__:
                                if base is object:
                                    continue
                                elif any(m in base.__dict__ for m in methods):
                                    # found one, we're good
                                    sabotage = False
                                    break
                                elif '__new__' in base.__dict__:
                                    # not good
                                    sabotage = True
                                    break
                            if sabotage is not None:
                                break
                        if sabotage:
                            _make_class_unpicklable(enum_class)
        # instantiate them, checking for duplicates as we go
        # we instantiate first instead of checking for duplicates first in case
        # a custom __new__ is doing something funky with the values -- such as
        # auto-numbering ;)
        for member_name in classdict._member_names:
            value = enum_members[member_name]
            if not isinstance(value, tuple):
                args = (value, )
            else:
                args = value
            if member_type is tuple:   # special case for tuple enums
                args = (args, )     # wrap it one more time
            if not use_args:
                enum_member = __new__(enum_class)
                if not hasattr(enum_member, '_value_'):
                    enum_member._value_ = value
            else:
                enum_member = __new__(enum_class, *args)
                if not hasattr(enum_member, '_value_'):
                    if member_type is object:
                        enum_member._value_ = value
                    else:
                        enum_member._value_ = member_type(*args)
            value = enum_member._value_
            enum_member._name_ = member_name
            enum_member.__objclass__ = enum_class
            enum_member.__init__(*args)
            # If another member with the same value was already defined, the
            # new member becomes an alias to the existing one.
            for name, canonical_member in enum_class._member_map_.items():
                if canonical_member._value_ == enum_member._value_:
                    enum_member = canonical_member
                    break
            else:
                # Aliases don't appear in member names (only in __members__).
                enum_class._member_names_.append(member_name)
            # performance boost for any member that would not shadow
            # a DynamicClassAttribute
            if member_name not in dynamic_attributes:
                setattr(enum_class, member_name, enum_member)
            # now add to _member_map_
            enum_class._member_map_[member_name] = enum_member
            try:
                # This may fail if value is not hashable. We can't add the value
                # to the map, and by-value lookups for this value will be
                # linear.
                enum_class._value2member_map_[value] = enum_member
            except TypeError:
                pass

        # double check that repr and friends are not the mixin's or various
        # things break (such as pickle)
        # however, if the method is defined in the Enum itself, don't replace
        # it
        for name in ('__repr__', '__str__', '__format__', '__reduce_ex__'):
            if name in classdict:
                continue
            class_method = getattr(enum_class, name)
            obj_method = getattr(member_type, name, None)
            enum_method = getattr(first_enum, name, None)
            if obj_method is not None and obj_method is class_method:
                setattr(enum_class, name, enum_method)

        # replace any other __new__ with our own (as long as Enum is not None,
        # anyway) -- again, this is to support pickle
        if Enum is not None:
            # if the user defined their own __new__, save it before it gets
            # clobbered in case they subclass later
            if save_new:
                enum_class.__new_member__ = __new__
            enum_class.__new__ = Enum.__new__

        # py3 support for definition order (helps keep py2/py3 code in sync)
        if _order_ is not None:
            if isinstance(_order_, str):
                _order_ = _order_.replace(',', ' ').split()
            if _order_ != enum_class._member_names_:
                raise TypeError('member order does not match _order_')

        return enum_class

    def __bool__(self):
        """
        classes/types should always be True.
        """
        return True

    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        """
        Either returns an existing member, or creates a new enum class.

        This method is used both when an enum class is given a value to match
        to an enumeration member (i.e. Color(3)) and for the functional API
        (i.e. Color = Enum('Color', names='RED GREEN BLUE')).

        When used for the functional API:

        `value` will be the name of the new class.

        `names` should be either a string of white-space/comma delimited names
        (values will start at `start`), or an iterator/mapping of name, value pairs.

        `module` should be set to the module this class is being created in;
        if it is not set, an attempt to find that module will be made, but if
        it fails the class will not be picklable.

        `qualname` should be set to the actual location this class can be found
        at in its module; by default it is set to the global scope.  If this is
        not correct, unpickling will fail in some circumstances.

        `type`, if set, will be mixed in as the first base class.
        """
        if names is None:  # simple value lookup
            return cls.__new__(cls, value)
        # otherwise, functional API: we're creating a new Enum type
        return cls._create_(
                value,
                names,
                module=module,
                qualname=qualname,
                type=type,
                start=start,
                )

    def __contains__(cls, obj):
        if not isinstance(obj, Enum):
            import warnings
            warnings.warn(
                    "in 3.12 __contains__ will no longer raise TypeError, but will return True if\n"
                    "obj is a member or a member's value",
                    DeprecationWarning,
                    stacklevel=2,
                    )
            raise TypeError(
                "unsupported operand type(s) for 'in': '%s' and '%s'" % (
                    type(obj).__qualname__, cls.__class__.__qualname__))
        return isinstance(obj, cls) and obj._name_ in cls._member_map_

    def __delattr__(cls, attr):
        # nicer error message when someone tries to delete an attribute
        # (see issue19025).
        if attr in cls._member_map_:
            raise AttributeError("%s: cannot delete Enum member." % cls.__name__)
        super().__delattr__(attr)

    def __dir__(self):
        return (
                ['__class__', '__doc__', '__members__', '__module__']
                + self._member_names_
                )

    def __getattr__(cls, name):
        """
        Return the enum member matching `name`

        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.
        """
        if _is_dunder(name):
            raise AttributeError(name)
        try:
            return cls._member_map_[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(cls, name):
        return cls._member_map_[name]

    def __iter__(cls):
        """
        Returns members in definition order.
        """
        return (cls._member_map_[name] for name in cls._member_names_)

    def __len__(cls):
        return len(cls._member_names_)

    @property
    def __members__(cls):
        """
        Returns a mapping of member name->value.

        This mapping lists all enum members, including aliases. Note that this
        is a read-only view of the internal mapping.
        """
        return MappingProxyType(cls._member_map_)

    def __repr__(cls):
        return "<enum %r>" % cls.__name__

    def __reversed__(cls):
        """
        Returns members in reverse definition order.
        """
        return (cls._member_map_[name] for name in reversed(cls._member_names_))

    def __setattr__(cls, name, value):
        """
        Block attempts to reassign Enum members.

        A simple assignment to the class namespace only changes one of the
        several possible ways to get an Enum member from the Enum class,
        resulting in an inconsistent Enumeration.
        """
        member_map = cls.__dict__.get('_member_map_', {})
        if name in member_map:
            raise AttributeError('Cannot reassign members.')
        super().__setattr__(name, value)

    def _create_(cls, class_name, names, *, module=None, qualname=None, type=None, start=1):
        """
        Convenience method to create a new Enum class.

        `names` can be:

        * A string containing member names, separated either with spaces or
          commas.  Values are incremented by 1 from `start`.
        * An iterable of member names.  Values are incremented by 1 from `start`.
        * An iterable of (member name, value) pairs.
        * A mapping of member name -> value pairs.
        """
        metacls = cls.__class__
        bases = (cls, ) if type is None else (type, cls)
        _, first_enum = cls._get_mixins_(cls, bases)
        classdict = metacls.__prepare__(class_name, bases)

        # special processing needed for names?
        if isinstance(names, str):
            names = names.replace(',', ' ').split()
        if isinstance(names, (tuple, list)) and names and isinstance(names[0], str):
            original_names, names = names, []
            last_values = []
            for count, name in enumerate(original_names):
                value = first_enum._generate_next_value_(name, start, count, last_values[:])
                last_values.append(value)
                names.append((name, value))

        # Here, names is either an iterable of (name, value) or a mapping.
        for item in names:
            if isinstance(item, str):
                member_name, member_value = item, names[item]
            else:
                member_name, member_value = item
            classdict[member_name] = member_value
        enum_class = metacls.__new__(metacls, class_name, bases, classdict)

        # TODO: replace the frame hack if a blessed way to know the calling
        # module is ever developed
        if module is None:
            try:
                module = sys._getframe(2).f_globals['__name__']
            except (AttributeError, ValueError, KeyError):
                pass
        if module is None:
            _make_class_unpicklable(enum_class)
        else:
            enum_class.__module__ = module
        if qualname is not None:
            enum_class.__qualname__ = qualname

        return enum_class

    def _convert_(cls, name, module, filter, source=None):
        """
        Create a new Enum subclass that replaces a collection of global constants
        """
        # convert all constants from source (or module) that pass filter() to
        # a new Enum called name, and export the enum and its members back to
        # module;
        # also, replace the __reduce_ex__ method so unpickling works in
        # previous Python versions
        module_globals = vars(sys.modules[module])
        if source:
            source = vars(source)
        else:
            source = module_globals
        # _value2member_map_ is populated in the same order every time
        # for a consistent reverse mapping of number to name when there
        # are multiple names for the same number.
        members = [
                (name, value)
                for name, value in source.items()
                if filter(name)]
        try:
            # sort by value
            members.sort(key=lambda t: (t[1], t[0]))
        except TypeError:
            # unless some values aren't comparable, in which case sort by name
            members.sort(key=lambda t: t[0])
        cls = cls(name, members, module=module)
        cls.__reduce_ex__ = _reduce_ex_by_name
        module_globals.update(cls.__members__)
        module_globals[name] = cls
        return cls

    @staticmethod
    def _check_for_existing_members(class_name, bases):
        for chain in bases:
            for base in chain.__mro__:
                if issubclass(base, Enum) and base._member_names_:
                    raise TypeError(
                            "%s: cannot extend enumeration %r"
                            % (class_name, base.__name__)
                            )

    @staticmethod
    def _get_mixins_(class_name, bases):
        """
        Returns the type for creating enum members, and the first inherited
        enum class.

        bases: the tuple of bases that was given to __new__
        """
        if not bases:
            return object, Enum

        def _find_data_type(bases):
            data_types = set()
            for chain in bases:
                candidate = None
                for base in chain.__mro__:
                    if base is object:
                        continue
                    elif issubclass(base, Enum):
                        if base._member_type_ is not object:
                            data_types.add(base._member_type_)
                            break
                    elif '__new__' in base.__dict__:
                        if issubclass(base, Enum):
                            continue
                        data_types.add(candidate or base)
                        break
                    else:
                        candidate = candidate or base
            if len(data_types) > 1:
                raise TypeError('%r: too many data types: %r' % (class_name, data_types))
            elif data_types:
                return data_types.pop()
            else:
                return None

        # ensure final parent class is an Enum derivative, find any concrete
        # data type, and check that Enum has no members
        first_enum = bases[-1]
        if not issubclass(first_enum, Enum):
            raise TypeError("new enumerations should be created as "
                    "`EnumName([mixin_type, ...] [data_type,] enum_type)`")
        member_type = _find_data_type(bases) or object
        if first_enum._member_names_:
            raise TypeError("Cannot extend enumerations")
        return member_type, first_enum

    @staticmethod
    def _find_new_(classdict, member_type, first_enum):
        """
        Returns the __new__ to be used for creating the enum members.

        classdict: the class dictionary given to __new__
        member_type: the data type whose __new__ will be used by default
        first_enum: enumeration to check for an overriding __new__
        """
        # now find the correct __new__, checking to see of one was defined
        # by the user; also check earlier enum classes in case a __new__ was
        # saved as __new_member__
        __new__ = classdict.get('__new__', None)

        # should __new__ be saved as __new_member__ later?
        save_new = __new__ is not None

        if __new__ is None:
            # check all possibles for __new_member__ before falling back to
            # __new__
            for method in ('__new_member__', '__new__'):
                for possible in (member_type, first_enum):
                    target = getattr(possible, method, None)
                    if target not in {
                            None,
                            None.__new__,
                            object.__new__,
                            Enum.__new__,
                            }:
                        __new__ = target
                        break
                if __new__ is not None:
                    break
            else:
                __new__ = object.__new__

        # if a non-object.__new__ is used then whatever value/tuple was
        # assigned to the enum member name will be passed to __new__ and to the
        # new enum member's __init__
        if __new__ is object.__new__:
            use_args = False
        else:
            use_args = True
        return __new__, save_new, use_args


class Enum(metaclass=EnumMeta):
    """
    Generic enumeration.

    Derive from this class to define new enumerations.
    """
    def __new__(cls, value):
        # all enum instances are actually created during class construction
        # without calling this method; this method is called by the metaclass'
        # __call__ (i.e. Color(3) ), and by pickle
        if type(value) is cls:
            # For lookups like Color(Color.RED)
            return value
        # by-value search for a matching enum member
        # see if it's in the reverse mapping (for hashable values)
        try:
            return cls._value2member_map_[value]
        except KeyError:
            # Not found, no need to do long O(n) search
            pass
        except TypeError:
            # not there, now do long search -- O(n) behavior
            for member in cls._member_map_.values():
                if member._value_ == value:
                    return member
        # still not found -- try _missing_ hook
        try:
            exc = None
            result = cls._missing_(value)
        except Exception as e:
            exc = e
            result = None
        try:
            if isinstance(result, cls):
                return result
            else:
                ve_exc = ValueError("%r is not a valid %s" % (value, cls.__qualname__))
                if result is None and exc is None:
                    raise ve_exc
                elif exc is None:
                    exc = TypeError(
                            'error in %s._missing_: returned %r instead of None or a valid member'
                            % (cls.__name__, result)
                            )
                if not isinstance(exc, ValueError):
                    exc.__context__ = ve_exc
                raise exc
        finally:
            # ensure all variables that could hold an exception are destroyed
            exc = None
            ve_exc = None

    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.

        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_value: the last value assigned or None
        """
        for last_value in reversed(last_values):
            try:
                return last_value + 1
            except TypeError:
                pass
        else:
            return start

    @classmethod
    def _missing_(cls, value):
        return None

    def __repr__(self):
        return "<%s.%s: %r>" % (
                self.__class__.__name__, self._name_, self._value_)

    def __str__(self):
        return "%s.%s" % (self.__class__.__name__, self._name_)

    def __dir__(self):
        """
        Returns all members and all public methods
        """
        added_behavior = [
                m
                for cls in self.__class__.mro()
                for m in cls.__dict__
                if m[0] != '_' and m not in self._member_map_
                ] + [m for m in self.__dict__ if m[0] != '_']
        return (['__class__', '__doc__', '__module__'] + added_behavior)

    def __format__(self, format_spec):
        """
        Returns format using actual value type unless __str__ has been overridden.
        """
        # mixed-in Enums should use the mixed-in type's __format__, otherwise
        # we can get strange results with the Enum name showing up instead of
        # the value

        # pure Enum branch, or branch with __str__ explicitly overridden
        str_overridden = type(self).__str__ not in (Enum.__str__, Flag.__str__)
        if self._member_type_ is object or str_overridden:
            cls = str
            val = str(self)
        # mix-in branch
        else:
            cls = self._member_type_
            val = self._value_
        return cls.__format__(val, format_spec)

    def __hash__(self):
        return hash(self._name_)

    def __reduce_ex__(self, proto):
        return self.__class__, (self._value_, )

    # DynamicClassAttribute is used to provide access to the `name` and
    # `value` properties of enum members while keeping some measure of
    # protection from modification, while still allowing for an enumeration
    # to have members named `name` and `value`.  This works because enumeration
    # members are not set directly on the enum class -- __getattr__ is
    # used to look them up.

    @DynamicClassAttribute
    def name(self):
        """The name of the Enum member."""
        return self._name_

    @DynamicClassAttribute
    def value(self):
        """The value of the Enum member."""
        return self._value_


class IntEnum(int, Enum):
    """Enum where members are also (and must be) ints"""


def _reduce_ex_by_name(self, proto):
    return self.name

class Flag(Enum):
    """
    Support for flags
    """

    def _generate_next_value_(name, start, count, last_values):
        """
        Generate the next value when not given.

        name: the name of the member
        start: the initial start value or None
        count: the number of existing members
        last_value: the last value assigned or None
        """
        if not count:
            return start if start is not None else 1
        for last_value in reversed(last_values):
            try:
                high_bit = _high_bit(last_value)
                break
            except Exception:
                raise TypeError('Invalid Flag value: %r' % last_value) from None
        return 2 ** (high_bit+1)

    @classmethod
    def _missing_(cls, value):
        """
        Returns member (possibly creating it) if one can be found for value.
        """
        original_value = value
        if value < 0:
            value = ~value
        possible_member = cls._create_pseudo_member_(value)
        if original_value < 0:
            possible_member = ~possible_member
        return possible_member

    @classmethod
    def _create_pseudo_member_(cls, value):
        """
        Create a composite member iff value contains only members.
        """
        pseudo_member = cls._value2member_map_.get(value, None)
        if pseudo_member is None:
            # verify all bits are accounted for
            _, extra_flags = _decompose(cls, value)
            if extra_flags:
                raise ValueError("%r is not a valid %s" % (value, cls.__qualname__))
            # construct a singleton enum pseudo-member
            pseudo_member = object.__new__(cls)
            pseudo_member._name_ = None
            pseudo_member._value_ = value
            # use setdefault in case another thread already created a composite
            # with this value
            pseudo_member = cls._value2member_map_.setdefault(value, pseudo_member)
        return pseudo_member

    def __contains__(self, other):
        """
        Returns True if self has at least the same flags set as other.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                "unsupported operand type(s) for 'in': '%s' and '%s'" % (
                    type(other).__qualname__, self.__class__.__qualname__))
        return other._value_ & self._value_ == other._value_

    def __repr__(self):
        cls = self.__class__
        if self._name_ is not None:
            return '<%s.%s: %r>' % (cls.__name__, self._name_, self._value_)
        members, uncovered = _decompose(cls, self._value_)
        return '<%s.%s: %r>' % (
                cls.__name__,
                '|'.join([str(m._name_ or m._value_) for m in members]),
                self._value_,
                )

    def __str__(self):
        cls = self.__class__
        if self._name_ is not None:
            return '%s.%s' % (cls.__name__, self._name_)
        members, uncovered = _decompose(cls, self._value_)
        if len(members) == 1 and members[0]._name_ is None:
            return '%s.%r' % (cls.__name__, members[0]._value_)
        else:
            return '%s.%s' % (
                    cls.__name__,
                    '|'.join([str(m._name_ or m._value_) for m in members]),
                    )

    def __bool__(self):
        return bool(self._value_)

    def __or__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__(self._value_ | other._value_)

    def __and__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__(self._value_ & other._value_)

    def __xor__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__class__(self._value_ ^ other._value_)

    def __invert__(self):
        members, uncovered = _decompose(self.__class__, self._value_)
        inverted = self.__class__(0)
        for m in self.__class__:
            if m not in members and not (m._value_ & self._value_):
                inverted = inverted | m
        return self.__class__(inverted)


class IntFlag(int, Flag):
    """
    Support for integer-based Flags
    """

    @classmethod
    def _missing_(cls, value):
        """
        Returns member (possibly creating it) if one can be found for value.
        """
        if not isinstance(value, int):
            raise ValueError("%r is not a valid %s" % (value, cls.__qualname__))
        new_member = cls._create_pseudo_member_(value)
        return new_member

    @classmethod
    def _create_pseudo_member_(cls, value):
        """
        Create a composite member iff value contains only members.
        """
        pseudo_member = cls._value2member_map_.get(value, None)
        if pseudo_member is None:
            need_to_create = [value]
            # get unaccounted for bits
            _, extra_flags = _decompose(cls, value)
            # timer = 10
            while extra_flags:
                # timer -= 1
                bit = _high_bit(extra_flags)
                flag_value = 2 ** bit
                if (flag_value not in cls._value2member_map_ and
                        flag_value not in need_to_create
                        ):
                    need_to_create.append(flag_value)
                if extra_flags == -flag_value:
                    extra_flags = 0
                else:
                    extra_flags ^= flag_value
            for value in reversed(need_to_create):
                # construct singleton pseudo-members
                pseudo_member = int.__new__(cls, value)
                pseudo_member._name_ = None
                pseudo_member._value_ = value
                # use setdefault in case another thread already created a composite
                # with this value
                pseudo_member = cls._value2member_map_.setdefault(value, pseudo_member)
        return pseudo_member

    def __or__(self, other):
        if not isinstance(other, (self.__class__, int)):
            return NotImplemented
        result = self.__class__(self._value_ | self.__class__(other)._value_)
        return result

    def __and__(self, other):
        if not isinstance(other, (self.__class__, int)):
            return NotImplemented
        return self.__class__(self._value_ & self.__class__(other)._value_)

    def __xor__(self, other):
        if not isinstance(other, (self.__class__, int)):
            return NotImplemented
        return self.__class__(self._value_ ^ self.__class__(other)._value_)

    __ror__ = __or__
    __rand__ = __and__
    __rxor__ = __xor__

    def __invert__(self):
        result = self.__class__(~self._value_)
        return result


def _high_bit(value):
    """
    returns index of highest bit, or -1 if value is zero or negative
    """
    return value.bit_length() - 1

def unique(enumeration):
    """
    Class decorator for enumerations ensuring unique member values.
    """
    duplicates = []
    for name, member in enumeration.__members__.items():
        if name != member.name:
            duplicates.append((name, member.name))
    if duplicates:
        alias_details = ', '.join(
                ["%s -> %s" % (alias, name) for (alias, name) in duplicates])
        raise ValueError('duplicate values found in %r: %s' %
                (enumeration, alias_details))
    return enumeration

def _decompose(flag, value):
    """
    Extract all members from the value.
    """
    # _decompose is only called if the value is not named
    not_covered = value
    negative = value < 0
    members = []
    for member in flag:
        member_value = member.value
        if member_value and member_value & value == member_value:
            members.append(member)
            not_covered &= ~member_value
    if not negative:
        tmp = not_covered
        while tmp:
            flag_value = 2 ** _high_bit(tmp)
            if flag_value in flag._value2member_map_:
                members.append(flag._value2member_map_[flag_value])
                not_covered &= ~flag_value
            tmp &= ~flag_value
    if not members and value in flag._value2member_map_:
        members.append(flag._value2member_map_[value])
    members.sort(key=lambda m: m._value_, reverse=True)
    if len(members) > 1 and members[0].value == value:
        # we have the breakdown, don't need the value member itself
        members.pop(0)
    return members, not_covered
<|endoftext|>"""Response classes used by urllib.

The base class, addbase, defines a minimal file-like interface,
including read() and readline().  The typical response object is an
addinfourl instance, which defines an info() method that returns
headers and a geturl() method that returns the url.
"""

import tempfile

__all__ = ['addbase', 'addclosehook', 'addinfo', 'addinfourl']


class addbase(tempfile._TemporaryFileWrapper):
    """Base class for addinfo and addclosehook. Is a good idea for garbage collection."""

    # XXX Add a method to expose the timeout on the underlying socket?

    def __init__(self, fp):
        super(addbase,  self).__init__(fp, '<urllib response>', delete=False)
        # Keep reference around as this was part of the original API.
        self.fp = fp

    def __repr__(self):
        return '<%s at %r whose fp = %r>' % (self.__class__.__name__,
                                             id(self), self.file)

    def __enter__(self):
        if self.fp.closed:
            raise ValueError("I/O operation on closed file")
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class addclosehook(addbase):
    """Class to add a close hook to an open file."""

    def __init__(self, fp, closehook, *hookargs):
        super(addclosehook, self).__init__(fp)
        self.closehook = closehook
        self.hookargs = hookargs

    def close(self):
        try:
            closehook = self.closehook
            hookargs = self.hookargs
            if closehook:
                self.closehook = None
                self.hookargs = None
                closehook(*hookargs)
        finally:
            super(addclosehook, self).close()


class addinfo(addbase):
    """class to add an info() method to an open file."""

    def __init__(self, fp, headers):
        super(addinfo, self).__init__(fp)
        self.headers = headers

    def info(self):
        return self.headers


class addinfourl(addinfo):
    """class to add info() and geturl() methods to an open file."""

    def __init__(self, fp, headers, url, code=None):
        super(addinfourl, self).__init__(fp, headers)
        self.url = url
        self.code = code

    @property
    def status(self):
        return self.code

    def getcode(self):
        return self.code

    def geturl(self):
        return self.url
<|endoftext|>"""Parse (absolute and relative) URLs.

urlparse module is based upon the following RFC specifications.

RFC 3986 (STD66): "Uniform Resource Identifiers" by T. Berners-Lee, R. Fielding
and L.  Masinter, January 2005.

RFC 2732 : "Format for Literal IPv6 Addresses in URL's by R.Hinden, B.Carpenter
and L.Masinter, December 1999.

RFC 2396:  "Uniform Resource Identifiers (URI)": Generic Syntax by T.
Berners-Lee, R. Fielding, and L. Masinter, August 1998.

RFC 2368: "The mailto URL scheme", by P.Hoffman , L Masinter, J. Zawinski, July 1998.

RFC 1808: "Relative Uniform Resource Locators", by R. Fielding, UC Irvine, June
1995.

RFC 1738: "Uniform Resource Locators (URL)" by T. Berners-Lee, L. Masinter, M.
McCahill, December 1994

RFC 3986 is considered the current standard and any future changes to
urlparse module should conform with it.  The urlparse module is
currently not entirely compliant with this RFC due to defacto
scenarios for parsing, and for backward compatibility purposes, some
parsing quirks from older RFCs are retained. The testcases in
test_urlparse.py provides a good indicator of parsing behavior.

The WHATWG URL Parser spec should also be considered.  We are not compliant with
it either due to existing user code API behavior expectations (Hyrum's Law).
It serves as a useful guide when making changes.
"""

import re
import sys
import types
import collections
import warnings
import ipaddress

__all__ = ["urlparse", "urlunparse", "urljoin", "urldefrag",
           "urlsplit", "urlunsplit", "urlencode", "parse_qs",
           "parse_qsl", "quote", "quote_plus", "quote_from_bytes",
           "unquote", "unquote_plus", "unquote_to_bytes",
           "DefragResult", "ParseResult", "SplitResult",
           "DefragResultBytes", "ParseResultBytes", "SplitResultBytes"]

# A classification of schemes.
# The empty string classifies URLs with no scheme specified,
# being the default value returned by “urlsplit” and “urlparse”.

uses_relative = ['', 'ftp', 'http', 'gopher', 'nntp', 'imap',
                 'wais', 'file', 'https', 'shttp', 'mms',
                 'prospero', 'rtsp', 'rtspu', 'sftp',
                 'svn', 'svn+ssh', 'ws', 'wss']

uses_netloc = ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet',
               'imap', 'wais', 'file', 'mms', 'https', 'shttp',
               'snews', 'prospero', 'rtsp', 'rtspu', 'rsync',
               'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh',
               'ws', 'wss']

uses_params = ['', 'ftp', 'hdl', 'prospero', 'http', 'imap',
               'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips',
               'mms', 'sftp', 'tel']

# These are not actually used anymore, but should stay for backwards
# compatibility.  (They are undocumented, but have a public-looking name.)

non_hierarchical = ['gopher', 'hdl', 'mailto', 'news',
                    'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']

uses_query = ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms',
              'gopher', 'rtsp', 'rtspu', 'sip', 'sips']

uses_fragment = ['', 'ftp', 'hdl', 'http', 'gopher', 'news',
                 'nntp', 'wais', 'https', 'shttp', 'snews',
                 'file', 'prospero']

# Characters valid in scheme names
scheme_chars = ('abcdefghijklmnopqrstuvwxyz'
                'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                '0123456789'
                '+-.')

# Leading and trailing C0 control and space to be stripped per WHATWG spec.
# == "".join([chr(i) for i in range(0, 0x20 + 1)])
_WHATWG_C0_CONTROL_OR_SPACE = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f '

# Unsafe bytes to be removed per WHATWG spec
_UNSAFE_URL_BYTES_TO_REMOVE = ['\t', '\r', '\n']

# XXX: Consider replacing with functools.lru_cache
MAX_CACHE_SIZE = 20
_parse_cache = {}

def clear_cache():
    """Clear the parse cache and the quoters cache."""
    _parse_cache.clear()
    _safe_quoters.clear()


# Helpers for bytes handling
# For 3.2, we deliberately require applications that
# handle improperly quoted URLs to do their own
# decoding and encoding. If valid use cases are
# presented, we may relax this by using latin-1
# decoding internally for 3.3
_implicit_encoding = 'ascii'
_implicit_errors = 'strict'

def _noop(obj):
    return obj

def _encode_result(obj, encoding=_implicit_encoding,
                        errors=_implicit_errors):
    return obj.encode(encoding, errors)

def _decode_args(args, encoding=_implicit_encoding,
                       errors=_implicit_errors):
    return tuple(x.decode(encoding, errors) if x else '' for x in args)

def _coerce_args(*args):
    # Invokes decode if necessary to create str args
    # and returns the coerced inputs along with
    # an appropriate result coercion function
    #   - noop for str inputs
    #   - encoding function otherwise
    str_input = isinstance(args[0], str)
    for arg in args[1:]:
        # We special-case the empty string to support the
        # "scheme=''" default argument to some functions
        if arg and isinstance(arg, str) != str_input:
            raise TypeError("Cannot mix str and non-str arguments")
    if str_input:
        return args + (_noop,)
    return _decode_args(args) + (_encode_result,)

# Result objects are more helpful than simple tuples
class _ResultMixinStr(object):
    """Standard approach to encoding parsed results from str to bytes"""
    __slots__ = ()

    def encode(self, encoding='ascii', errors='strict'):
        return self._encoded_counterpart(*(x.encode(encoding, errors) for x in self))


class _ResultMixinBytes(object):
    """Standard approach to decoding parsed results from bytes to str"""
    __slots__ = ()

    def decode(self, encoding='ascii', errors='strict'):
        return self._decoded_counterpart(*(x.decode(encoding, errors) for x in self))


class _NetlocResultMixinBase(object):
    """Shared methods for the parsed result objects containing a netloc element"""
    __slots__ = ()

    @property
    def username(self):
        return self._userinfo[0]

    @property
    def password(self):
        return self._userinfo[1]

    @property
    def hostname(self):
        hostname = self._hostinfo[0]
        if not hostname:
            return None
        # Scoped IPv6 address may have zone info, which must not be lowercased
        # like http://[fe80::822a:a8ff:fe49:470c%tESt]:1234/keys
        separator = '%' if isinstance(hostname, str) else b'%'
        hostname, percent, zone = hostname.partition(separator)
        return hostname.lower() + percent + zone

    @property
    def port(self):
        port = self._hostinfo[1]
        if port is not None:
            if port.isdigit() and port.isascii():
                port = int(port)
            else:
                raise ValueError(f"Port could not be cast to integer value as {port!r}")
            if not (0 <= port <= 65535):
                raise ValueError("Port out of range 0-65535")
        return port

    __class_getitem__ = classmethod(types.GenericAlias)


class _NetlocResultMixinStr(_NetlocResultMixinBase, _ResultMixinStr):
    __slots__ = ()

    @property
    def _userinfo(self):
        netloc = self.netloc
        userinfo, have_info, hostinfo = netloc.rpartition('@')
        if have_info:
            username, have_password, password = userinfo.partition(':')
            if not have_password:
                password = None
        else:
            username = password = None
        return username, password

    @property
    def _hostinfo(self):
        netloc = self.netloc
        _, _, hostinfo = netloc.rpartition('@')
        _, have_open_br, bracketed = hostinfo.partition('[')
        if have_open_br:
            hostname, _, port = bracketed.partition(']')
            _, _, port = port.partition(':')
        else:
            hostname, _, port = hostinfo.partition(':')
        if not port:
            port = None
        return hostname, port


class _NetlocResultMixinBytes(_NetlocResultMixinBase, _ResultMixinBytes):
    __slots__ = ()

    @property
    def _userinfo(self):
        netloc = self.netloc
        userinfo, have_info, hostinfo = netloc.rpartition(b'@')
        if have_info:
            username, have_password, password = userinfo.partition(b':')
            if not have_password:
                password = None
        else:
            username = password = None
        return username, password

    @property
    def _hostinfo(self):
        netloc = self.netloc
        _, _, hostinfo = netloc.rpartition(b'@')
        _, have_open_br, bracketed = hostinfo.partition(b'[')
        if have_open_br:
            hostname, _, port = bracketed.partition(b']')
            _, _, port = port.partition(b':')
        else:
            hostname, _, port = hostinfo.partition(b':')
        if not port:
            port = None
        return hostname, port


from collections import namedtuple

_DefragResultBase = namedtuple('DefragResult', 'url fragment')
_SplitResultBase = namedtuple(
    'SplitResult', 'scheme netloc path query fragment')
_ParseResultBase = namedtuple(
    'ParseResult', 'scheme netloc path params query fragment')

_DefragResultBase.__doc__ = """
DefragResult(url, fragment)

A 2-tuple that contains the url without fragment identifier and the fragment
identifier as a separate argument.
"""

_DefragResultBase.url.__doc__ = """The URL with no fragment identifier."""

_DefragResultBase.fragment.__doc__ = """
Fragment identifier separated from URL, that allows indirect identification of a
secondary resource by reference to a primary resource and additional identifying
information.
"""

_SplitResultBase.__doc__ = """
SplitResult(scheme, netloc, path, query, fragment)

A 5-tuple that contains the different components of a URL. Similar to
ParseResult, but does not split params.
"""

_SplitResultBase.scheme.__doc__ = """Specifies URL scheme for the request."""

_SplitResultBase.netloc.__doc__ = """
Network location where the request is made to.
"""

_SplitResultBase.path.__doc__ = """
The hierarchical path, such as the path to a file to download.
"""

_SplitResultBase.query.__doc__ = """
The query component, that contains non-hierarchical data, that along with data
in path component, identifies a resource in the scope of URI's scheme and
network location.
"""

_SplitResultBase.fragment.__doc__ = """
Fragment identifier, that allows indirect identification of a secondary resource
by reference to a primary resource and additional identifying information.
"""

_ParseResultBase.__doc__ = """
ParseResult(scheme, netloc, path, params, query, fragment)

A 6-tuple that contains components of a parsed URL.
"""

_ParseResultBase.scheme.__doc__ = _SplitResultBase.scheme.__doc__
_ParseResultBase.netloc.__doc__ = _SplitResultBase.netloc.__doc__
_ParseResultBase.path.__doc__ = _SplitResultBase.path.__doc__
_ParseResultBase.params.__doc__ = """
Parameters for last path element used to dereference the URI in order to provide
access to perform some operation on the resource.
"""

_ParseResultBase.query.__doc__ = _SplitResultBase.query.__doc__
_ParseResultBase.fragment.__doc__ = _SplitResultBase.fragment.__doc__


# For backwards compatibility, alias _NetlocResultMixinStr
# ResultBase is no longer part of the documented API, but it is
# retained since deprecating it isn't worth the hassle
ResultBase = _NetlocResultMixinStr

# Structured result objects for string data
class DefragResult(_DefragResultBase, _ResultMixinStr):
    __slots__ = ()
    def geturl(self):
        if self.fragment:
            return self.url + '#' + self.fragment
        else:
            return self.url

class SplitResult(_SplitResultBase, _NetlocResultMixinStr):
    __slots__ = ()
    def geturl(self):
        return urlunsplit(self)

class ParseResult(_ParseResultBase, _NetlocResultMixinStr):
    __slots__ = ()
    def geturl(self):
        return urlunparse(self)

# Structured result objects for bytes data
class DefragResultBytes(_DefragResultBase, _ResultMixinBytes):
    __slots__ = ()
    def geturl(self):
        if self.fragment:
            return self.url + b'#' + self.fragment
        else:
            return self.url

class SplitResultBytes(_SplitResultBase, _NetlocResultMixinBytes):
    __slots__ = ()
    def geturl(self):
        return urlunsplit(self)

class ParseResultBytes(_ParseResultBase, _NetlocResultMixinBytes):
    __slots__ = ()
    def geturl(self):
        return urlunparse(self)

# Set up the encode/decode result pairs
def _fix_result_transcoding():
    _result_pairs = (
        (DefragResult, DefragResultBytes),
        (SplitResult, SplitResultBytes),
        (ParseResult, ParseResultBytes),
    )
    for _decoded, _encoded in _result_pairs:
        _decoded._encoded_counterpart = _encoded
        _encoded._decoded_counterpart = _decoded

_fix_result_transcoding()
del _fix_result_transcoding

def urlparse(url, scheme='', allow_fragments=True):
    """Parse a URL into 6 components:
    <scheme>://<netloc>/<path>;<params>?<query>#<fragment>

    The result is a named 6-tuple with fields corresponding to the
    above. It is either a ParseResult or ParseResultBytes object,
    depending on the type of the url parameter.

    The username, password, hostname, and port sub-components of netloc
    can also be accessed as attributes of the returned object.

    The scheme argument provides the default value of the scheme
    component when no scheme is found in url.

    If allow_fragments is False, no attempt is made to separate the
    fragment component from the previous component, which can be either
    path or query.

    Note that % escapes are not expanded.
    """
    url, scheme, _coerce_result = _coerce_args(url, scheme)
    splitresult = urlsplit(url, scheme, allow_fragments)
    scheme, netloc, url, query, fragment = splitresult
    if scheme in uses_params and ';' in url:
        url, params = _splitparams(url)
    else:
        params = ''
    result = ParseResult(scheme, netloc, url, params, query, fragment)
    return _coerce_result(result)

def _splitparams(url):
    if '/'  in url:
        i = url.find(';', url.rfind('/'))
        if i < 0:
            return url, ''
    else:
        i = url.find(';')
    return url[:i], url[i+1:]

def _splitnetloc(url, start=0):
    delim = len(url)   # position of end of domain part of url, default is end
    for c in '/?#':    # look for delimiters; the order is NOT important
        wdelim = url.find(c, start)        # find first of this delim
        if wdelim >= 0:                    # if found
            delim = min(delim, wdelim)     # use earliest delim position
    return url[start:delim], url[delim:]   # return (domain, rest)

def _checknetloc(netloc):
    if not netloc or netloc.isascii():
        return
    # looking for characters like \u2100 that expand to 'a/c'
    # IDNA uses NFKC equivalence, so normalize for this check
    import unicodedata
    n = netloc.replace('@', '')   # ignore characters already included
    n = n.replace(':', '')        # but not the surrounding text
    n = n.replace('#', '')
    n = n.replace('?', '')
    netloc2 = unicodedata.normalize('NFKC', n)
    if n == netloc2:
        return
    for c in '/?#@:':
        if c in netloc2:
            raise ValueError("netloc '" + netloc + "' contains invalid " +
                             "characters under NFKC normalization")

def _check_bracketed_netloc(netloc):
    # Note that this function must mirror the splitting
    # done in NetlocResultMixins._hostinfo().
    hostname_and_port = netloc.rpartition('@')[2]
    before_bracket, have_open_br, bracketed = hostname_and_port.partition('[')
    if have_open_br:
        # No data is allowed before a bracket.
        if before_bracket:
            raise ValueError("Invalid IPv6 URL")
        hostname, _, port = bracketed.partition(']')
        # No data is allowed after the bracket but before the port delimiter.
        if port and not port.startswith(":"):
            raise ValueError("Invalid IPv6 URL")
    else:
        hostname, _, port = hostname_and_port.partition(':')
    _check_bracketed_host(hostname)

# Valid bracketed hosts are defined in
# https://www.rfc-editor.org/rfc/rfc3986#page-49 and https://url.spec.whatwg.org/
def _check_bracketed_host(hostname):
    if hostname.startswith('v'):
        if not re.match(r"\Av[a-fA-F0-9]+\..+\Z", hostname):
            raise ValueError(f"IPvFuture address is invalid")
    else:
        ip = ipaddress.ip_address(hostname) # Throws Value Error if not IPv6 or IPv4
        if isinstance(ip, ipaddress.IPv4Address):
            raise ValueError(f"An IPv4 address cannot be in brackets")

def urlsplit(url, scheme='', allow_fragments=True):
    """Parse a URL into 5 components:
    <scheme>://<netloc>/<path>?<query>#<fragment>

    The result is a named 5-tuple with fields corresponding to the
    above. It is either a SplitResult or SplitResultBytes object,
    depending on the type of the url parameter.

    The username, password, hostname, and port sub-components of netloc
    can also be accessed as attributes of the returned object.

    The scheme argument provides the default value of the scheme
    component when no scheme is found in url.

    If allow_fragments is False, no attempt is made to separate the
    fragment component from the previous component, which can be either
    path or query.

    Note that % escapes are not expanded.
    """

    url, scheme, _coerce_result = _coerce_args(url, scheme)
    # Only lstrip url as some applications rely on preserving trailing space.
    # (https://url.spec.whatwg.org/#concept-basic-url-parser would strip both)
    url = url.lstrip(_WHATWG_C0_CONTROL_OR_SPACE)
    scheme = scheme.strip(_WHATWG_C0_CONTROL_OR_SPACE)

    for b in _UNSAFE_URL_BYTES_TO_REMOVE:
        url = url.replace(b, "")
        scheme = scheme.replace(b, "")

    allow_fragments = bool(allow_fragments)
    key = url, scheme, allow_fragments, type(url), type(scheme)
    cached = _parse_cache.get(key, None)
    if cached:
        return _coerce_result(cached)
    if len(_parse_cache) >= MAX_CACHE_SIZE: # avoid runaway growth
        clear_cache()
    netloc = query = fragment = ''
    i = url.find(':')
    if i > 0:
        for c in url[:i]:
            if c not in scheme_chars:
                break
        else:
            scheme, url = url[:i].lower(), url[i+1:]
    if url[:2] == '//':
        netloc, url = _splitnetloc(url, 2)
        if (('[' in netloc and ']' not in netloc) or
                (']' in netloc and '[' not in netloc)):
            raise ValueError("Invalid IPv6 URL")
        if '[' in netloc and ']' in netloc:
            _check_bracketed_netloc(netloc)
    if allow_fragments and '#' in url:
        url, fragment = url.split('#', 1)
    if '?' in url:
        url, query = url.split('?', 1)
    _checknetloc(netloc)
    v = SplitResult(scheme, netloc, url, query, fragment)
    _parse_cache[key] = v
    return _coerce_result(v)

def urlunparse(components):
    """Put a parsed URL back together again.  This may result in a
    slightly different, but equivalent URL, if the URL that was parsed
    originally had redundant delimiters, e.g. a ? with an empty query
    (the draft states that these are equivalent)."""
    scheme, netloc, url, params, query, fragment, _coerce_result = (
                                                  _coerce_args(*components))
    if params:
        url = "%s;%s" % (url, params)
    return _coerce_result(urlunsplit((scheme, netloc, url, query, fragment)))

def urlunsplit(components):
    """Combine the elements of a tuple as returned by urlsplit() into a
    complete URL as a string. The data argument can be any five-item iterable.
    This may result in a slightly different, but equivalent URL, if the URL that
    was parsed originally had unnecessary delimiters (for example, a ? with an
    empty query; the RFC states that these are equivalent)."""
    scheme, netloc, url, query, fragment, _coerce_result = (
                                          _coerce_args(*components))
    if netloc or (scheme and scheme in uses_netloc and url[:2] != '//'):
        if url and url[:1] != '/': url = '/' + url
        url = '//' + (netloc or '') + url
    if scheme:
        url = scheme + ':' + url
    if query:
        url = url + '?' + query
    if fragment:
        url = url + '#' + fragment
    return _coerce_result(url)

def urljoin(base, url, allow_fragments=True):
    """Join a base URL and a possibly relative URL to form an absolute
    interpretation of the latter."""
    if not base:
        return url
    if not url:
        return base

    base, url, _coerce_result = _coerce_args(base, url)
    bscheme, bnetloc, bpath, bparams, bquery, bfragment = \
            urlparse(base, '', allow_fragments)
    scheme, netloc, path, params, query, fragment = \
            urlparse(url, bscheme, allow_fragments)

    if scheme != bscheme or scheme not in uses_relative:
        return _coerce_result(url)
    if scheme in uses_netloc:
        if netloc:
            return _coerce_result(urlunparse((scheme, netloc, path,
                                              params, query, fragment)))
        netloc = bnetloc

    if not path and not params:
        path = bpath
        params = bparams
        if not query:
            query = bquery
        return _coerce_result(urlunparse((scheme, netloc, path,
                                          params, query, fragment)))

    base_parts = bpath.split('/')
    if base_parts[-1] != '':
        # the last item is not a directory, so will not be taken into account
        # in resolving the relative path
        del base_parts[-1]

    # for rfc3986, ignore all base path should the first character be root.
    if path[:1] == '/':
        segments = path.split('/')
    else:
        segments = base_parts + path.split('/')
        # filter out elements that would cause redundant slashes on re-joining
        # the resolved_path
        segments[1:-1] = filter(None, segments[1:-1])

    resolved_path = []

    for seg in segments:
        if seg == '..':
            try:
                resolved_path.pop()
            except IndexError:
                # ignore any .. segments that would otherwise cause an IndexError
                # when popped from resolved_path if resolving for rfc3986
                pass
        elif seg == '.':
            continue
        else:
            resolved_path.append(seg)

    if segments[-1] in ('.', '..'):
        # do some post-processing here. if the last segment was a relative dir,
        # then we need to append the trailing '/'
        resolved_path.append('')

    return _coerce_result(urlunparse((scheme, netloc, '/'.join(
        resolved_path) or '/', params, query, fragment)))


def urldefrag(url):
    """Removes any existing fragment from URL.

    Returns a tuple of the defragmented URL and the fragment.  If
    the URL contained no fragments, the second element is the
    empty string.
    """
    url, _coerce_result = _coerce_args(url)
    if '#' in url:
        s, n, p, a, q, frag = urlparse(url)
        defrag = urlunparse((s, n, p, a, q, ''))
    else:
        frag = ''
        defrag = url
    return _coerce_result(DefragResult(defrag, frag))

_hexdig = '0123456789ABCDEFabcdef'
_hextobyte = None

def unquote_to_bytes(string):
    """unquote_to_bytes('abc%20def') -> b'abc def'."""
    # Note: strings are encoded as UTF-8. This is only an issue if it contains
    # unescaped non-ASCII characters, which URIs should not.
    if not string:
        # Is it a string-like object?
        string.split
        return b''
    if isinstance(string, str):
        string = string.encode('utf-8')
    bits = string.split(b'%')
    if len(bits) == 1:
        return string
    res = [bits[0]]
    append = res.append
    # Delay the initialization of the table to not waste memory
    # if the function is never called
    global _hextobyte
    if _hextobyte is None:
        _hextobyte = {(a + b).encode(): bytes.fromhex(a + b)
                      for a in _hexdig for b in _hexdig}
    for item in bits[1:]:
        try:
            append(_hextobyte[item[:2]])
            append(item[2:])
        except KeyError:
            append(b'%')
            append(item)
    return b''.join(res)

_asciire = re.compile('([\x00-\x7f]+)')

def unquote(string, encoding='utf-8', errors='replace'):
    """Replace %xx escapes by their single-character equivalent. The optional
    encoding and errors parameters specify how to decode percent-encoded
    sequences into Unicode characters, as accepted by the bytes.decode()
    method.
    By default, percent-encoded sequences are decoded with UTF-8, and invalid
    sequences are replaced by a placeholder character.

    unquote('abc%20def') -> 'abc def'.
    """
    if isinstance(string, bytes):
        return unquote_to_bytes(string).decode(encoding, errors)
    if '%' not in string:
        string.split
        return string
    if encoding is None:
        encoding = 'utf-8'
    if errors is None:
        errors = 'replace'
    bits = _asciire.split(string)
    res = [bits[0]]
    append = res.append
    for i in range(1, len(bits), 2):
        append(unquote_to_bytes(bits[i]).decode(encoding, errors))
        append(bits[i + 1])
    return ''.join(res)


def parse_qs(qs, keep_blank_values=False, strict_parsing=False,
             encoding='utf-8', errors='replace', max_num_fields=None, separator='&'):
    """Parse a query given as a string argument.

        Arguments:

        qs: percent-encoded query string to be parsed

        keep_blank_values: flag indicating whether blank values in
            percent-encoded queries should be treated as blank strings.
            A true value indicates that blanks should be retained as
            blank strings.  The default false value indicates that
            blank values are to be ignored and treated as if they were
            not included.

        strict_parsing: flag indicating what to do with parsing errors.
            If false (the default), errors are silently ignored.
            If true, errors raise a ValueError exception.

        encoding and errors: specify how to decode percent-encoded sequences
            into Unicode characters, as accepted by the bytes.decode() method.

        max_num_fields: int. If set, then throws a ValueError if there
            are more than n fields read by parse_qsl().

        separator: str. The symbol to use for separating the query arguments.
            Defaults to &.

        Returns a dictionary.
    """
    parsed_result = {}
    pairs = parse_qsl(qs, keep_blank_values, strict_parsing,
                      encoding=encoding, errors=errors,
                      max_num_fields=max_num_fields, separator=separator)
    for name, value in pairs:
        if name in parsed_result:
            parsed_result[name].append(value)
        else:
            parsed_result[name] = [value]
    return parsed_result


def parse_qsl(qs, keep_blank_values=False, strict_parsing=False,
              encoding='utf-8', errors='replace', max_num_fields=None, separator='&'):
    """Parse a query given as a string argument.

        Arguments:

        qs: percent-encoded query string to be parsed

        keep_blank_values: flag indicating whether blank values in
            percent-encoded queries should be treated as blank strings.
            A true value indicates that blanks should be retained as blank
            strings.  The default false value indicates that blank values
            are to be ignored and treated as if they were  not included.

        strict_parsing: flag indicating what to do with parsing errors. If
            false (the default), errors are silently ignored. If true,
            errors raise a ValueError exception.

        encoding and errors: specify how to decode percent-encoded sequences
            into Unicode characters, as accepted by the bytes.decode() method.

        max_num_fields: int. If set, then throws a ValueError
            if there are more than n fields read by parse_qsl().

        separator: str. The symbol to use for separating the query arguments.
            Defaults to &.

        Returns a list, as G-d intended.
    """
    qs, _coerce_result = _coerce_args(qs)
    separator, _ = _coerce_args(separator)

    if not separator or (not isinstance(separator, (str, bytes))):
        raise ValueError("Separator must be of type string or bytes.")

    # If max_num_fields is defined then check that the number of fields
    # is less than max_num_fields. This prevents a memory exhaustion DOS
    # attack via post bodies with many fields.
    if max_num_fields is not None:
        num_fields = 1 + qs.count(separator)
        if max_num_fields < num_fields:
            raise ValueError('Max number of fields exceeded')

    r = []
    for name_value in qs.split(separator):
        if not name_value and not strict_parsing:
            continue
        nv = name_value.split('=', 1)
        if len(nv) != 2:
            if strict_parsing:
                raise ValueError("bad query field: %r" % (name_value,))
            # Handle case of a control-name with no equal sign
            if keep_blank_values:
                nv.append('')
            else:
                continue
        if len(nv[1]) or keep_blank_values:
            name = nv[0].replace('+', ' ')
            name = unquote(name, encoding=encoding, errors=errors)
            name = _coerce_result(name)
            value = nv[1].replace('+', ' ')
            value = unquote(value, encoding=encoding, errors=errors)
            value = _coerce_result(value)
            r.append((name, value))
    return r

def unquote_plus(string, encoding='utf-8', errors='replace'):
    """Like unquote(), but also replace plus signs by spaces, as required for
    unquoting HTML form values.

    unquote_plus('%7e/abc+def') -> '~/abc def'
    """
    string = string.replace('+', ' ')
    return unquote(string, encoding, errors)

_ALWAYS_SAFE = frozenset(b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                         b'abcdefghijklmnopqrstuvwxyz'
                         b'0123456789'
                         b'_.-~')
_ALWAYS_SAFE_BYTES = bytes(_ALWAYS_SAFE)
_safe_quoters = {}

class Quoter(collections.defaultdict):
    """A mapping from bytes (in range(0,256)) to strings.

    String values are percent-encoded byte values, unless the key < 128, and
    in the "safe" set (either the specified safe set, or default set).
    """
    # Keeps a cache internally, using defaultdict, for efficiency (lookups
    # of cached keys don't call Python code at all).
    def __init__(self, safe):
        """safe: bytes object."""
        self.safe = _ALWAYS_SAFE.union(safe)

    def __repr__(self):
        # Without this, will just display as a defaultdict
        return "<%s %r>" % (self.__class__.__name__, dict(self))

    def __missing__(self, b):
        # Handle a cache miss. Store quoted string in cache and return.
        res = chr(b) if b in self.safe else '%{:02X}'.format(b)
        self[b] = res
        return res

def quote(string, safe='/', encoding=None, errors=None):
    """quote('abc def') -> 'abc%20def'

    Each part of a URL, e.g. the path info, the query, etc., has a
    different set of reserved characters that must be quoted. The
    quote function offers a cautious (not minimal) way to quote a
    string for most of these parts.

    RFC 3986 Uniform Resource Identifier (URI): Generic Syntax lists
    the following (un)reserved characters.

    unreserved    = ALPHA / DIGIT / "-" / "." / "_" / "~"
    reserved      = gen-delims / sub-delims
    gen-delims    = ":" / "/" / "?" / "#" / "[" / "]" / "@"
    sub-delims    = "!" / "$" / "&" / "'" / "(" / ")"
                  / "*" / "+" / "," / ";" / "="

    Each of the reserved characters is reserved in some component of a URL,
    but not necessarily in all of them.

    The quote function %-escapes all characters that are neither in the
    unreserved chars ("always safe") nor the additional chars set via the
    safe arg.

    The default for the safe arg is '/'. The character is reserved, but in
    typical usage the quote function is being called on a path where the
    existing slash characters are to be preserved.

    Python 3.7 updates from using RFC 2396 to RFC 3986 to quote URL strings.
    Now, "~" is included in the set of unreserved characters.

    string and safe may be either str or bytes objects. encoding and errors
    must not be specified if string is a bytes object.

    The optional encoding and errors parameters specify how to deal with
    non-ASCII characters, as accepted by the str.encode method.
    By default, encoding='utf-8' (characters are encoded with UTF-8), and
    errors='strict' (unsupported characters raise a UnicodeEncodeError).
    """
    if isinstance(string, str):
        if not string:
            return string
        if encoding is None:
            encoding = 'utf-8'
        if errors is None:
            errors = 'strict'
        string = string.encode(encoding, errors)
    else:
        if encoding is not None:
            raise TypeError("quote() doesn't support 'encoding' for bytes")
        if errors is not None:
            raise TypeError("quote() doesn't support 'errors' for bytes")
    return quote_from_bytes(string, safe)

def quote_plus(string, safe='', encoding=None, errors=None):
    """Like quote(), but also replace ' ' with '+', as required for quoting
    HTML form values. Plus signs in the original string are escaped unless
    they are included in safe. It also does not have safe default to '/'.
    """
    # Check if ' ' in string, where string may either be a str or bytes.  If
    # there are no spaces, the regular quote will produce the right answer.
    if ((isinstance(string, str) and ' ' not in string) or
        (isinstance(string, bytes) and b' ' not in string)):
        return quote(string, safe, encoding, errors)
    if isinstance(safe, str):
        space = ' '
    else:
        space = b' '
    string = quote(string, safe + space, encoding, errors)
    return string.replace(' ', '+')

def quote_from_bytes(bs, safe='/'):
    """Like quote(), but accepts a bytes object rather than a str, and does
    not perform string-to-bytes encoding.  It always returns an ASCII string.
    quote_from_bytes(b'abc def\x3f') -> 'abc%20def%3f'
    """
    if not isinstance(bs, (bytes, bytearray)):
        raise TypeError("quote_from_bytes() expected bytes")
    if not bs:
        return ''
    if isinstance(safe, str):
        # Normalize 'safe' by converting to bytes and removing non-ASCII chars
        safe = safe.encode('ascii', 'ignore')
    else:
        safe = bytes([c for c in safe if c < 128])
    if not bs.rstrip(_ALWAYS_SAFE_BYTES + safe):
        return bs.decode()
    try:
        quoter = _safe_quoters[safe]
    except KeyError:
        _safe_quoters[safe] = quoter = Quoter(safe).__getitem__
    return ''.join([quoter(char) for char in bs])

def urlencode(query, doseq=False, safe='', encoding=None, errors=None,
              quote_via=quote_plus):
    """Encode a dict or sequence of two-element tuples into a URL query string.

    If any values in the query arg are sequences and doseq is true, each
    sequence element is converted to a separate parameter.

    If the query arg is a sequence of two-element tuples, the order of the
    parameters in the output will match the order of parameters in the
    input.

    The components of a query arg may each be either a string or a bytes type.

    The safe, encoding, and errors parameters are passed down to the function
    specified by quote_via (encoding and errors only if a component is a str).
    """

    if hasattr(query, "items"):
        query = query.items()
    else:
        # It's a bother at times that strings and string-like objects are
        # sequences.
        try:
            # non-sequence items should not work with len()
            # non-empty strings will fail this
            if len(query) and not isinstance(query[0], tuple):
                raise TypeError
            # Zero-length sequences of all types will get here and succeed,
            # but that's a minor nit.  Since the original implementation
            # allowed empty dicts that type of behavior probably should be
            # preserved for consistency
        except TypeError:
            ty, va, tb = sys.exc_info()
            raise TypeError("not a valid non-string sequence "
                            "or mapping object").with_traceback(tb)

    l = []
    if not doseq:
        for k, v in query:
            if isinstance(k, bytes):
                k = quote_via(k, safe)
            else:
                k = quote_via(str(k), safe, encoding, errors)

            if isinstance(v, bytes):
                v = quote_via(v, safe)
            else:
                v = quote_via(str(v), safe, encoding, errors)
            l.append(k + '=' + v)
    else:
        for k, v in query:
            if isinstance(k, bytes):
                k = quote_via(k, safe)
            else:
                k = quote_via(str(k), safe, encoding, errors)

            if isinstance(v, bytes):
                v = quote_via(v, safe)
                l.append(k + '=' + v)
            elif isinstance(v, str):
                v = quote_via(v, safe, encoding, errors)
                l.append(k + '=' + v)
            else:
                try:
                    # Is this a sufficient test for sequence-ness?
                    x = len(v)
                except TypeError:
                    # not a sequence
                    v = quote_via(str(v), safe, encoding, errors)
                    l.append(k + '=' + v)
                else:
                    # loop over the sequence
                    for elt in v:
                        if isinstance(elt, bytes):
                            elt = quote_via(elt, safe)
                        else:
                            elt = quote_via(str(elt), safe, encoding, errors)
                        l.append(k + '=' + elt)
    return '&'.join(l)


def to_bytes(url):
    warnings.warn("urllib.parse.to_bytes() is deprecated as of 3.8",
                  DeprecationWarning, stacklevel=2)
    return _to_bytes(url)


def _to_bytes(url):
    """to_bytes(u"URL") --> 'URL'."""
    # Most URL schemes require ASCII. If that changes, the conversion
    # can be relaxed.
    # XXX get rid of to_bytes()
    if isinstance(url, str):
        try:
            url = url.encode("ASCII").decode()
        except UnicodeError:
            raise UnicodeError("URL " + repr(url) +
                               " contains non-ASCII characters")
    return url


def unwrap(url):
    """Transform a string like '<URL:scheme://host/path>' into 'scheme://host/path'.

    The string is returned unchanged if it's not a wrapped URL.
    """
    url = str(url).strip()
    if url[:1] == '<' and url[-1:] == '>':
        url = url[1:-1].strip()
    if url[:4] == 'URL:':
        url = url[4:].strip()
    return url


def splittype(url):
    warnings.warn("urllib.parse.splittype() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splittype(url)


_typeprog = None
def _splittype(url):
    """splittype('type:opaquestring') --> 'type', 'opaquestring'."""
    global _typeprog
    if _typeprog is None:
        _typeprog = re.compile('([^/:]+):(.*)', re.DOTALL)

    match = _typeprog.match(url)
    if match:
        scheme, data = match.groups()
        return scheme.lower(), data
    return None, url


def splithost(url):
    warnings.warn("urllib.parse.splithost() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splithost(url)


_hostprog = None
def _splithost(url):
    """splithost('//host[:port]/path') --> 'host[:port]', '/path'."""
    global _hostprog
    if _hostprog is None:
        _hostprog = re.compile('//([^/#?]*)(.*)', re.DOTALL)

    match = _hostprog.match(url)
    if match:
        host_port, path = match.groups()
        if path and path[0] != '/':
            path = '/' + path
        return host_port, path
    return None, url


def splituser(host):
    warnings.warn("urllib.parse.splituser() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splituser(host)


def _splituser(host):
    """splituser('user[:passwd]@host[:port]') --> 'user[:passwd]', 'host[:port]'."""
    user, delim, host = host.rpartition('@')
    return (user if delim else None), host


def splitpasswd(user):
    warnings.warn("urllib.parse.splitpasswd() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splitpasswd(user)


def _splitpasswd(user):
    """splitpasswd('user:passwd') -> 'user', 'passwd'."""
    user, delim, passwd = user.partition(':')
    return user, (passwd if delim else None)


def splitport(host):
    warnings.warn("urllib.parse.splitport() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splitport(host)


# splittag('/path#tag') --> '/path', 'tag'
_portprog = None
def _splitport(host):
    """splitport('host:port') --> 'host', 'port'."""
    global _portprog
    if _portprog is None:
        _portprog = re.compile('(.*):([0-9]*)', re.DOTALL)

    match = _portprog.fullmatch(host)
    if match:
        host, port = match.groups()
        if port:
            return host, port
    return host, None


def splitnport(host, defport=-1):
    warnings.warn("urllib.parse.splitnport() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splitnport(host, defport)


def _splitnport(host, defport=-1):
    """Split host and port, returning numeric port.
    Return given default port if no ':' found; defaults to -1.
    Return numerical port if a valid number is found after ':'.
    Return None if ':' but not a valid number."""
    host, delim, port = host.rpartition(':')
    if not delim:
        host = port
    elif port:
        if port.isdigit() and port.isascii():
            nport = int(port)
        else:
            nport = None
        return host, nport
    return host, defport


def splitquery(url):
    warnings.warn("urllib.parse.splitquery() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splitquery(url)


def _splitquery(url):
    """splitquery('/path?query') --> '/path', 'query'."""
    path, delim, query = url.rpartition('?')
    if delim:
        return path, query
    return url, None


def splittag(url):
    warnings.warn("urllib.parse.splittag() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splittag(url)


def _splittag(url):
    """splittag('/path#tag') --> '/path', 'tag'."""
    path, delim, tag = url.rpartition('#')
    if delim:
        return path, tag
    return url, None


def splitattr(url):
    warnings.warn("urllib.parse.splitattr() is deprecated as of 3.8, "
                  "use urllib.parse.urlparse() instead",
                  DeprecationWarning, stacklevel=2)
    return _splitattr(url)


def _splitattr(url):
    """splitattr('/path;attr1=value1;attr2=value2;...') ->
        '/path', ['attr1=value1', 'attr2=value2', ...]."""
    words = url.split(';')
    return words[0], words[1:]


def splitvalue(attr):
    warnings.warn("urllib.parse.splitvalue() is deprecated as of 3.8, "
                  "use urllib.parse.parse_qsl() instead",
                  DeprecationWarning, stacklevel=2)
    return _splitvalue(attr)


def _splitvalue(attr):
    """splitvalue('attr=value') --> 'attr', 'value'."""
    attr, delim, value = attr.partition('=')
    return attr, (value if delim else None)
<|endoftext|>""" robotparser.py

    Copyright (C) 2000  Bastian Kleineidam

    You can choose between two licenses when using this package:
    1) GNU GPLv2
    2) PSF license for Python 2.2

    The robots.txt Exclusion Protocol is implemented as specified in
    http://www.robotstxt.org/norobots-rfc.txt
"""

import collections
import urllib.parse
import urllib.request

__all__ = ["RobotFileParser"]

RequestRate = collections.namedtuple("RequestRate", "requests seconds")


class RobotFileParser:
    """ This class provides a set of methods to read, parse and answer
    questions about a single robots.txt file.

    """

    def __init__(self, url=''):
        self.entries = []
        self.sitemaps = []
        self.default_entry = None
        self.disallow_all = False
        self.allow_all = False
        self.set_url(url)
        self.last_checked = 0

    def mtime(self):
        """Returns the time the robots.txt file was last fetched.

        This is useful for long-running web spiders that need to
        check for new robots.txt files periodically.

        """
        return self.last_checked

    def modified(self):
        """Sets the time the robots.txt file was last fetched to the
        current time.

        """
        import time
        self.last_checked = time.time()

    def set_url(self, url):
        """Sets the URL referring to a robots.txt file."""
        self.url = url
        self.host, self.path = urllib.parse.urlparse(url)[1:3]

    def read(self):
        """Reads the robots.txt URL and feeds it to the parser."""
        try:
            f = urllib.request.urlopen(self.url)
        except urllib.error.HTTPError as err:
            if err.code in (401, 403):
                self.disallow_all = True
            elif err.code >= 400 and err.code < 500:
                self.allow_all = True
        else:
            raw = f.read()
            self.parse(raw.decode("utf-8").splitlines())

    def _add_entry(self, entry):
        if "*" in entry.useragents:
            # the default entry is considered last
            if self.default_entry is None:
                # the first default entry wins
                self.default_entry = entry
        else:
            self.entries.append(entry)

    def parse(self, lines):
        """Parse the input lines from a robots.txt file.

        We allow that a user-agent: line is not preceded by
        one or more blank lines.
        """
        # states:
        #   0: start state
        #   1: saw user-agent line
        #   2: saw an allow or disallow line
        state = 0
        entry = Entry()

        self.modified()
        for line in lines:
            if not line:
                if state == 1:
                    entry = Entry()
                    state = 0
                elif state == 2:
                    self._add_entry(entry)
                    entry = Entry()
                    state = 0
            # remove optional comment and strip line
            i = line.find('#')
            if i >= 0:
                line = line[:i]
            line = line.strip()
            if not line:
                continue
            line = line.split(':', 1)
            if len(line) == 2:
                line[0] = line[0].strip().lower()
                line[1] = urllib.parse.unquote(line[1].strip())
                if line[0] == "user-agent":
                    if state == 2:
                        self._add_entry(entry)
                        entry = Entry()
                    entry.useragents.append(line[1])
                    state = 1
                elif line[0] == "disallow":
                    if state != 0:
                        entry.rulelines.append(RuleLine(line[1], False))
                        state = 2
                elif line[0] == "allow":
                    if state != 0:
                        entry.rulelines.append(RuleLine(line[1], True))
                        state = 2
                elif line[0] == "crawl-delay":
                    if state != 0:
                        # before trying to convert to int we need to make
                        # sure that robots.txt has valid syntax otherwise
                        # it will crash
                        if line[1].strip().isdigit():
                            entry.delay = int(line[1])
                        state = 2
                elif line[0] == "request-rate":
                    if state != 0:
                        numbers = line[1].split('/')
                        # check if all values are sane
                        if (len(numbers) == 2 and numbers[0].strip().isdigit()
                            and numbers[1].strip().isdigit()):
                            entry.req_rate = RequestRate(int(numbers[0]), int(numbers[1]))
                        state = 2
                elif line[0] == "sitemap":
                    # According to http://www.sitemaps.org/protocol.html
                    # "This directive is independent of the user-agent line,
                    #  so it doesn't matter where you place it in your file."
                    # Therefore we do not change the state of the parser.
                    self.sitemaps.append(line[1])
        if state == 2:
            self._add_entry(entry)

    def can_fetch(self, useragent, url):
        """using the parsed robots.txt decide if useragent can fetch url"""
        if self.disallow_all:
            return False
        if self.allow_all:
            return True
        # Until the robots.txt file has been read or found not
        # to exist, we must assume that no url is allowable.
        # This prevents false positives when a user erroneously
        # calls can_fetch() before calling read().
        if not self.last_checked:
            return False
        # search for given user agent matches
        # the first match counts
        parsed_url = urllib.parse.urlparse(urllib.parse.unquote(url))
        url = urllib.parse.urlunparse(('','',parsed_url.path,
            parsed_url.params,parsed_url.query, parsed_url.fragment))
        url = urllib.parse.quote(url)
        if not url:
            url = "/"
        for entry in self.entries:
            if entry.applies_to(useragent):
                return entry.allowance(url)
        # try the default entry last
        if self.default_entry:
            return self.default_entry.allowance(url)
        # agent not found ==> access granted
        return True

    def crawl_delay(self, useragent):
        if not self.mtime():
            return None
        for entry in self.entries:
            if entry.applies_to(useragent):
                return entry.delay
        if self.default_entry:
            return self.default_entry.delay
        return None

    def request_rate(self, useragent):
        if not self.mtime():
            return None
        for entry in self.entries:
            if entry.applies_to(useragent):
                return entry.req_rate
        if self.default_entry:
            return self.default_entry.req_rate
        return None

    def site_maps(self):
        if not self.sitemaps:
            return None
        return self.sitemaps

    def __str__(self):
        entries = self.entries
        if self.default_entry is not None:
            entries = entries + [self.default_entry]
        return '\n\n'.join(map(str, entries))


class RuleLine:
    """A rule line is a single "Allow:" (allowance==True) or "Disallow:"
       (allowance==False) followed by a path."""
    def __init__(self, path, allowance):
        if path == '' and not allowance:
            # an empty value means allow all
            allowance = True
        path = urllib.parse.urlunparse(urllib.parse.urlparse(path))
        self.path = urllib.parse.quote(path)
        self.allowance = allowance

    def applies_to(self, filename):
        return self.path == "*" or filename.startswith(self.path)

    def __str__(self):
        return ("Allow" if self.allowance else "Disallow") + ": " + self.path


class Entry:
    """An entry has one or more user-agents and zero or more rulelines"""
    def __init__(self):
        self.useragents = []
        self.rulelines = []
        self.delay = None
        self.req_rate = None

    def __str__(self):
        ret = []
        for agent in self.useragents:
            ret.append(f"User-agent: {agent}")
        if self.delay is not None:
            ret.append(f"Crawl-delay: {self.delay}")
        if self.req_rate is not None:
            rate = self.req_rate
            ret.append(f"Request-rate: {rate.requests}/{rate.seconds}")
        ret.extend(map(str, self.rulelines))
        return '\n'.join(ret)

    def applies_to(self, useragent):
        """check if this entry applies to the specified agent"""
        # split the name token and make it lower case
        useragent = useragent.split("/")[0].lower()
        for agent in self.useragents:
            if agent == '*':
                # we have the catch-all agent
                return True
            agent = agent.lower()
            if agent in useragent:
                return True
        return False

    def allowance(self, filename):
        """Preconditions:
        - our agent applies to this entry
        - filename is URL decoded"""
        for line in self.rulelines:
            if line.applies_to(filename):
                return line.allowance
        return True
<|endoftext|>"""An extensible library for opening URLs using a variety of protocols

The simplest way to use this module is to call the urlopen function,
which accepts a string containing a URL or a Request object (described
below).  It opens the URL and returns the results as file-like
object; the returned object has some extra methods described below.

The OpenerDirector manages a collection of Handler objects that do
all the actual work.  Each Handler implements a particular protocol or
option.  The OpenerDirector is a composite object that invokes the
Handlers needed to open the requested URL.  For example, the
HTTPHandler performs HTTP GET and POST requests and deals with
non-error returns.  The HTTPRedirectHandler automatically deals with
HTTP 301, 302, 303 and 307 redirect errors, and the HTTPDigestAuthHandler
deals with digest authentication.

urlopen(url, data=None) -- Basic usage is the same as original
urllib.  pass the url and optionally data to post to an HTTP URL, and
get a file-like object back.  One difference is that you can also pass
a Request instance instead of URL.  Raises a URLError (subclass of
OSError); for HTTP errors, raises an HTTPError, which can also be
treated as a valid response.

build_opener -- Function that creates a new OpenerDirector instance.
Will install the default handlers.  Accepts one or more Handlers as
arguments, either instances or Handler classes that it will
instantiate.  If one of the argument is a subclass of the default
handler, the argument will be installed instead of the default.

install_opener -- Installs a new opener as the default opener.

objects of interest:

OpenerDirector -- Sets up the User Agent as the Python-urllib client and manages
the Handler classes, while dealing with requests and responses.

Request -- An object that encapsulates the state of a request.  The
state can be as simple as the URL.  It can also include extra HTTP
headers, e.g. a User-Agent.

BaseHandler --

internals:
BaseHandler and parent
_call_chain conventions

Example usage:

import urllib.request

# set up authentication info
authinfo = urllib.request.HTTPBasicAuthHandler()
authinfo.add_password(realm='PDQ Application',
                      uri='https://mahler:8092/site-updates.py',
                      user='klem',
                      passwd='geheim$parole')

proxy_support = urllib.request.ProxyHandler({"http" : "http://ahad-haam:3128"})

# build a new opener that adds authentication and caching FTP handlers
opener = urllib.request.build_opener(proxy_support, authinfo,
                                     urllib.request.CacheFTPHandler)

# install it
urllib.request.install_opener(opener)

f = urllib.request.urlopen('https://www.python.org/')
"""

# XXX issues:
# If an authentication error handler that tries to perform
# authentication for some reason but fails, how should the error be
# signalled?  The client needs to know the HTTP error code.  But if
# the handler knows that the problem was, e.g., that it didn't know
# that hash algo that requested in the challenge, it would be good to
# pass that information along to the client, too.
# ftp errors aren't handled cleanly
# check digest against correct (i.e. non-apache) implementation

# Possible extensions:
# complex proxies  XXX not sure what exactly was meant by this
# abstract factory for opener

import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings


from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
    urlparse, urlsplit, urljoin, unwrap, quote, unquote,
    _splittype, _splithost, _splitport, _splituser, _splitpasswd,
    _splitattr, _splitquery, _splitvalue, _splittag, _to_bytes,
    unquote_to_bytes, urlunparse)
from urllib.response import addinfourl, addclosehook

# check for SSL
try:
    import ssl
except ImportError:
    _have_ssl = False
else:
    _have_ssl = True

__all__ = [
    # Classes
    'Request', 'OpenerDirector', 'BaseHandler', 'HTTPDefaultErrorHandler',
    'HTTPRedirectHandler', 'HTTPCookieProcessor', 'ProxyHandler',
    'HTTPPasswordMgr', 'HTTPPasswordMgrWithDefaultRealm',
    'HTTPPasswordMgrWithPriorAuth', 'AbstractBasicAuthHandler',
    'HTTPBasicAuthHandler', 'ProxyBasicAuthHandler', 'AbstractDigestAuthHandler',
    'HTTPDigestAuthHandler', 'ProxyDigestAuthHandler', 'HTTPHandler',
    'FileHandler', 'FTPHandler', 'CacheFTPHandler', 'DataHandler',
    'UnknownHandler', 'HTTPErrorProcessor',
    # Functions
    'urlopen', 'install_opener', 'build_opener',
    'pathname2url', 'url2pathname', 'getproxies',
    # Legacy interface
    'urlretrieve', 'urlcleanup', 'URLopener', 'FancyURLopener',
]

# used in User-Agent header sent
__version__ = '%d.%d' % sys.version_info[:2]

_opener = None
def urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
            *, cafile=None, capath=None, cadefault=False, context=None):
    '''Open the URL url, which can be either a string or a Request object.

    *data* must be an object specifying additional data to be sent to
    the server, or None if no such data is needed.  See Request for
    details.

    urllib.request module uses HTTP/1.1 and includes a "Connection:close"
    header in its HTTP requests.

    The optional *timeout* parameter specifies a timeout in seconds for
    blocking operations like the connection attempt (if not specified, the
    global default timeout setting will be used). This only works for HTTP,
    HTTPS and FTP connections.

    If *context* is specified, it must be a ssl.SSLContext instance describing
    the various SSL options. See HTTPSConnection for more details.

    The optional *cafile* and *capath* parameters specify a set of trusted CA
    certificates for HTTPS requests. cafile should point to a single file
    containing a bundle of CA certificates, whereas capath should point to a
    directory of hashed certificate files. More information can be found in
    ssl.SSLContext.load_verify_locations().

    The *cadefault* parameter is ignored.


    This function always returns an object which can work as a
    context manager and has the properties url, headers, and status.
    See urllib.response.addinfourl for more detail on these properties.

    For HTTP and HTTPS URLs, this function returns a http.client.HTTPResponse
    object slightly modified. In addition to the three new methods above, the
    msg attribute contains the same information as the reason attribute ---
    the reason phrase returned by the server --- instead of the response
    headers as it is specified in the documentation for HTTPResponse.

    For FTP, file, and data URLs and requests explicitly handled by legacy
    URLopener and FancyURLopener classes, this function returns a
    urllib.response.addinfourl object.

    Note that None may be returned if no handler handles the request (though
    the default installed global OpenerDirector uses UnknownHandler to ensure
    this never happens).

    In addition, if proxy settings are detected (for example, when a *_proxy
    environment variable like http_proxy is set), ProxyHandler is default
    installed and makes sure the requests are handled through the proxy.

    '''
    global _opener
    if cafile or capath or cadefault:
        import warnings
        warnings.warn("cafile, capath and cadefault are deprecated, use a "
                      "custom context instead.", DeprecationWarning, 2)
        if context is not None:
            raise ValueError(
                "You can't pass both context and any of cafile, capath, and "
                "cadefault"
            )
        if not _have_ssl:
            raise ValueError('SSL support not available')
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH,
                                             cafile=cafile,
                                             capath=capath)
        # send ALPN extension to indicate HTTP/1.1 protocol
        context.set_alpn_protocols(['http/1.1'])
        https_handler = HTTPSHandler(context=context)
        opener = build_opener(https_handler)
    elif context:
        https_handler = HTTPSHandler(context=context)
        opener = build_opener(https_handler)
    elif _opener is None:
        _opener = opener = build_opener()
    else:
        opener = _opener
    return opener.open(url, data, timeout)

def install_opener(opener):
    global _opener
    _opener = opener

_url_tempfiles = []
def urlretrieve(url, filename=None, reporthook=None, data=None):
    """
    Retrieve a URL into a temporary location on disk.

    Requires a URL argument. If a filename is passed, it is used as
    the temporary file location. The reporthook argument should be
    a callable that accepts a block number, a read size, and the
    total file size of the URL target. The data argument should be
    valid URL encoded data.

    If a filename is passed and the URL points to a local resource,
    the result is a copy from local file to new file.

    Returns a tuple containing the path to the newly created
    data file as well as the resulting HTTPMessage object.
    """
    url_type, path = _splittype(url)

    with contextlib.closing(urlopen(url, data)) as fp:
        headers = fp.info()

        # Just return the local path and the "headers" for file://
        # URLs. No sense in performing a copy unless requested.
        if url_type == "file" and not filename:
            return os.path.normpath(path), headers

        # Handle temporary file setup.
        if filename:
            tfp = open(filename, 'wb')
        else:
            tfp = tempfile.NamedTemporaryFile(delete=False)
            filename = tfp.name
            _url_tempfiles.append(filename)

        with tfp:
            result = filename, headers
            bs = 1024*8
            size = -1
            read = 0
            blocknum = 0
            if "content-length" in headers:
                size = int(headers["Content-Length"])

            if reporthook:
                reporthook(blocknum, bs, size)

            while True:
                block = fp.read(bs)
                if not block:
                    break
                read += len(block)
                tfp.write(block)
                blocknum += 1
                if reporthook:
                    reporthook(blocknum, bs, size)

    if size >= 0 and read < size:
        raise ContentTooShortError(
            "retrieval incomplete: got only %i out of %i bytes"
            % (read, size), result)

    return result

def urlcleanup():
    """Clean up temporary files from urlretrieve calls."""
    for temp_file in _url_tempfiles:
        try:
            os.unlink(temp_file)
        except OSError:
            pass

    del _url_tempfiles[:]
    global _opener
    if _opener:
        _opener = None

# copied from cookielib.py
_cut_port_re = re.compile(r":\d+$", re.ASCII)
def request_host(request):
    """Return request-host, as defined by RFC 2965.

    Variation from RFC: returned value is lowercased, for convenient
    comparison.

    """
    url = request.full_url
    host = urlparse(url)[1]
