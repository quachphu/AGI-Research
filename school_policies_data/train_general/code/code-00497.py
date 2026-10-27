                if isinstance(message, MHMessage):
                    self._dump_sequences(message, new_key)
            finally:
                if self._locked:
                    _unlock_file(f)
        finally:
            if not closed:
                _sync_close(f)
        return new_key

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        path = os.path.join(self._path, str(key))
        try:
            f = open(path, 'rb+')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        else:
            f.close()
            os.remove(path)

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        path = os.path.join(self._path, str(key))
        try:
            f = open(path, 'rb+')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        try:
            if self._locked:
                _lock_file(f)
            try:
                os.close(os.open(path, os.O_WRONLY | os.O_TRUNC))
                self._dump_message(message, f)
                if isinstance(message, MHMessage):
                    self._dump_sequences(message, key)
            finally:
                if self._locked:
                    _unlock_file(f)
        finally:
            _sync_close(f)

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        try:
            if self._locked:
                f = open(os.path.join(self._path, str(key)), 'rb+')
            else:
                f = open(os.path.join(self._path, str(key)), 'rb')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        with f:
            if self._locked:
                _lock_file(f)
            try:
                msg = MHMessage(f)
            finally:
                if self._locked:
                    _unlock_file(f)
        for name, key_list in self.get_sequences().items():
            if key in key_list:
                msg.add_sequence(name)
        return msg

    def get_bytes(self, key):
        """Return a bytes representation or raise a KeyError."""
        try:
            if self._locked:
                f = open(os.path.join(self._path, str(key)), 'rb+')
            else:
                f = open(os.path.join(self._path, str(key)), 'rb')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        with f:
            if self._locked:
                _lock_file(f)
            try:
                return f.read().replace(linesep, b'\n')
            finally:
                if self._locked:
                    _unlock_file(f)

    def get_file(self, key):
        """Return a file-like representation or raise a KeyError."""
        try:
            f = open(os.path.join(self._path, str(key)), 'rb')
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise KeyError('No message with key: %s' % key)
            else:
                raise
        return _ProxyFile(f)

    def iterkeys(self):
        """Return an iterator over keys."""
        return iter(sorted(int(entry) for entry in os.listdir(self._path)
                                      if entry.isdigit()))

    def __contains__(self, key):
        """Return True if the keyed message exists, False otherwise."""
        return os.path.exists(os.path.join(self._path, str(key)))

    def __len__(self):
        """Return a count of messages in the mailbox."""
        return len(list(self.iterkeys()))

    def lock(self):
        """Lock the mailbox."""
        if not self._locked:
            self._file = open(os.path.join(self._path, '.mh_sequences'), 'rb+')
            _lock_file(self._file)
            self._locked = True

    def unlock(self):
        """Unlock the mailbox if it is locked."""
        if self._locked:
            _unlock_file(self._file)
            _sync_close(self._file)
            del self._file
            self._locked = False

    def flush(self):
        """Write any pending changes to the disk."""
        return

    def close(self):
        """Flush and close the mailbox."""
        if self._locked:
            self.unlock()

    def list_folders(self):
        """Return a list of folder names."""
        result = []
        for entry in os.listdir(self._path):
            if os.path.isdir(os.path.join(self._path, entry)):
                result.append(entry)
        return result

    def get_folder(self, folder):
        """Return an MH instance for the named folder."""
        return MH(os.path.join(self._path, folder),
                  factory=self._factory, create=False)

    def add_folder(self, folder):
        """Create a folder and return an MH instance representing it."""
        return MH(os.path.join(self._path, folder),
                  factory=self._factory)

    def remove_folder(self, folder):
        """Delete the named folder, which must be empty."""
        path = os.path.join(self._path, folder)
        entries = os.listdir(path)
        if entries == ['.mh_sequences']:
            os.remove(os.path.join(path, '.mh_sequences'))
        elif entries == []:
            pass
        else:
            raise NotEmptyError('Folder not empty: %s' % self._path)
        os.rmdir(path)

    def get_sequences(self):
        """Return a name-to-key-list dictionary to define each sequence."""
        results = {}
        with open(os.path.join(self._path, '.mh_sequences'), 'r', encoding='ASCII') as f:
            all_keys = set(self.keys())
            for line in f:
                try:
                    name, contents = line.split(':')
                    keys = set()
                    for spec in contents.split():
                        if spec.isdigit():
                            keys.add(int(spec))
                        else:
                            start, stop = (int(x) for x in spec.split('-'))
                            keys.update(range(start, stop + 1))
                    results[name] = [key for key in sorted(keys) \
                                         if key in all_keys]
                    if len(results[name]) == 0:
                        del results[name]
                except ValueError:
                    raise FormatError('Invalid sequence specification: %s' %
                                      line.rstrip())
        return results

    def set_sequences(self, sequences):
        """Set sequences using the given name-to-key-list dictionary."""
        f = open(os.path.join(self._path, '.mh_sequences'), 'r+', encoding='ASCII')
        try:
            os.close(os.open(f.name, os.O_WRONLY | os.O_TRUNC))
            for name, keys in sequences.items():
                if len(keys) == 0:
                    continue
                f.write(name + ':')
                prev = None
                completing = False
                for key in sorted(set(keys)):
                    if key - 1 == prev:
                        if not completing:
                            completing = True
                            f.write('-')
                    elif completing:
                        completing = False
                        f.write('%s %s' % (prev, key))
                    else:
                        f.write(' %s' % key)
                    prev = key
                if completing:
                    f.write(str(prev) + '\n')
                else:
                    f.write('\n')
        finally:
            _sync_close(f)

    def pack(self):
        """Re-name messages to eliminate numbering gaps. Invalidates keys."""
        sequences = self.get_sequences()
        prev = 0
        changes = []
        for key in self.iterkeys():
            if key - 1 != prev:
                changes.append((key, prev + 1))
                try:
                    os.link(os.path.join(self._path, str(key)),
                            os.path.join(self._path, str(prev + 1)))
                except (AttributeError, PermissionError):
                    os.rename(os.path.join(self._path, str(key)),
                              os.path.join(self._path, str(prev + 1)))
                else:
                    os.unlink(os.path.join(self._path, str(key)))
            prev += 1
        self._next_key = prev + 1
        if len(changes) == 0:
            return
        for name, key_list in sequences.items():
            for old, new in changes:
                if old in key_list:
                    key_list[key_list.index(old)] = new
        self.set_sequences(sequences)

    def _dump_sequences(self, message, key):
        """Inspect a new MHMessage and update sequences appropriately."""
        pending_sequences = message.get_sequences()
        all_sequences = self.get_sequences()
        for name, key_list in all_sequences.items():
            if name in pending_sequences:
                key_list.append(key)
            elif key in key_list:
                del key_list[key_list.index(key)]
        for sequence in pending_sequences:
            if sequence not in all_sequences:
                all_sequences[sequence] = [key]
        self.set_sequences(all_sequences)


class Babyl(_singlefileMailbox):
    """An Rmail-style Babyl mailbox."""

    _special_labels = frozenset({'unseen', 'deleted', 'filed', 'answered',
                                 'forwarded', 'edited', 'resent'})

    def __init__(self, path, factory=None, create=True):
        """Initialize a Babyl mailbox."""
        _singlefileMailbox.__init__(self, path, factory, create)
        self._labels = {}

    def add(self, message):
        """Add message and return assigned key."""
        key = _singlefileMailbox.add(self, message)
        if isinstance(message, BabylMessage):
            self._labels[key] = message.get_labels()
        return key

    def remove(self, key):
        """Remove the keyed message; raise KeyError if it doesn't exist."""
        _singlefileMailbox.remove(self, key)
        if key in self._labels:
            del self._labels[key]

    def __setitem__(self, key, message):
        """Replace the keyed message; raise KeyError if it doesn't exist."""
        _singlefileMailbox.__setitem__(self, key, message)
        if isinstance(message, BabylMessage):
            self._labels[key] = message.get_labels()

    def get_message(self, key):
        """Return a Message representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        self._file.readline()   # Skip b'1,' line specifying labels.
        original_headers = io.BytesIO()
        while True:
            line = self._file.readline()
            if line == b'*** EOOH ***' + linesep or not line:
                break
            original_headers.write(line.replace(linesep, b'\n'))
        visible_headers = io.BytesIO()
        while True:
            line = self._file.readline()
            if line == linesep or not line:
                break
            visible_headers.write(line.replace(linesep, b'\n'))
        # Read up to the stop, or to the end
        n = stop - self._file.tell()
        assert n >= 0
        body = self._file.read(n)
        body = body.replace(linesep, b'\n')
        msg = BabylMessage(original_headers.getvalue() + body)
        msg.set_visible(visible_headers.getvalue())
        if key in self._labels:
            msg.set_labels(self._labels[key])
        return msg

    def get_bytes(self, key):
        """Return a string representation or raise a KeyError."""
        start, stop = self._lookup(key)
        self._file.seek(start)
        self._file.readline()   # Skip b'1,' line specifying labels.
        original_headers = io.BytesIO()
        while True:
            line = self._file.readline()
            if line == b'*** EOOH ***' + linesep or not line:
                break
            original_headers.write(line.replace(linesep, b'\n'))
        while True:
            line = self._file.readline()
            if line == linesep or not line:
                break
        headers = original_headers.getvalue()
        n = stop - self._file.tell()
        assert n >= 0
        data = self._file.read(n)
        data = data.replace(linesep, b'\n')
        return headers + data

    def get_file(self, key):
        """Return a file-like representation or raise a KeyError."""
        return io.BytesIO(self.get_bytes(key).replace(b'\n', linesep))

    def get_labels(self):
        """Return a list of user-defined labels in the mailbox."""
        self._lookup()
        labels = set()
        for label_list in self._labels.values():
            labels.update(label_list)
        labels.difference_update(self._special_labels)
        return list(labels)

    def _generate_toc(self):
        """Generate key-to-(start, stop) table of contents."""
        starts, stops = [], []
        self._file.seek(0)
        next_pos = 0
        label_lists = []
        while True:
            line_pos = next_pos
            line = self._file.readline()
            next_pos = self._file.tell()
            if line == b'\037\014' + linesep:
                if len(stops) < len(starts):
                    stops.append(line_pos - len(linesep))
                starts.append(next_pos)
                labels = [label.strip() for label
                                        in self._file.readline()[1:].split(b',')
                                        if label.strip()]
                label_lists.append(labels)
            elif line == b'\037' or line == b'\037' + linesep:
                if len(stops) < len(starts):
                    stops.append(line_pos - len(linesep))
            elif not line:
                stops.append(line_pos - len(linesep))
                break
        self._toc = dict(enumerate(zip(starts, stops)))
        self._labels = dict(enumerate(label_lists))
        self._next_key = len(self._toc)
        self._file.seek(0, 2)
        self._file_length = self._file.tell()

    def _pre_mailbox_hook(self, f):
        """Called before writing the mailbox to file f."""
        babyl = b'BABYL OPTIONS:' + linesep
        babyl += b'Version: 5' + linesep
        labels = self.get_labels()
        labels = (label.encode() for label in labels)
        babyl += b'Labels:' + b','.join(labels) + linesep
        babyl += b'\037'
        f.write(babyl)

    def _pre_message_hook(self, f):
        """Called before writing each message to file f."""
        f.write(b'\014' + linesep)

    def _post_message_hook(self, f):
        """Called after writing each message to file f."""
        f.write(linesep + b'\037')

    def _install_message(self, message):
        """Write message contents and return (start, stop)."""
        start = self._file.tell()
        if isinstance(message, BabylMessage):
            special_labels = []
            labels = []
            for label in message.get_labels():
                if label in self._special_labels:
                    special_labels.append(label)
                else:
                    labels.append(label)
            self._file.write(b'1')
            for label in special_labels:
                self._file.write(b', ' + label.encode())
            self._file.write(b',,')
            for label in labels:
                self._file.write(b' ' + label.encode() + b',')
            self._file.write(linesep)
        else:
            self._file.write(b'1,,' + linesep)
        if isinstance(message, email.message.Message):
            orig_buffer = io.BytesIO()
            orig_generator = email.generator.BytesGenerator(orig_buffer, False, 0)
            orig_generator.flatten(message)
            orig_buffer.seek(0)
            while True:
                line = orig_buffer.readline()
                self._file.write(line.replace(b'\n', linesep))
                if line == b'\n' or not line:
                    break
            self._file.write(b'*** EOOH ***' + linesep)
            if isinstance(message, BabylMessage):
                vis_buffer = io.BytesIO()
                vis_generator = email.generator.BytesGenerator(vis_buffer, False, 0)
                vis_generator.flatten(message.get_visible())
                while True:
                    line = vis_buffer.readline()
                    self._file.write(line.replace(b'\n', linesep))
                    if line == b'\n' or not line:
                        break
            else:
                orig_buffer.seek(0)
                while True:
                    line = orig_buffer.readline()
                    self._file.write(line.replace(b'\n', linesep))
                    if line == b'\n' or not line:
                        break
            while True:
                buffer = orig_buffer.read(4096) # Buffer size is arbitrary.
                if not buffer:
                    break
                self._file.write(buffer.replace(b'\n', linesep))
        elif isinstance(message, (bytes, str, io.StringIO)):
            if isinstance(message, io.StringIO):
                warnings.warn("Use of StringIO input is deprecated, "
                    "use BytesIO instead", DeprecationWarning, 3)
                message = message.getvalue()
            if isinstance(message, str):
                message = self._string_to_bytes(message)
            body_start = message.find(b'\n\n') + 2
            if body_start - 2 != -1:
                self._file.write(message[:body_start].replace(b'\n', linesep))
                self._file.write(b'*** EOOH ***' + linesep)
                self._file.write(message[:body_start].replace(b'\n', linesep))
                self._file.write(message[body_start:].replace(b'\n', linesep))
            else:
                self._file.write(b'*** EOOH ***' + linesep + linesep)
                self._file.write(message.replace(b'\n', linesep))
        elif hasattr(message, 'readline'):
            if hasattr(message, 'buffer'):
                warnings.warn("Use of text mode files is deprecated, "
                    "use a binary mode file instead", DeprecationWarning, 3)
                message = message.buffer
            original_pos = message.tell()
            first_pass = True
            while True:
                line = message.readline()
                # Universal newline support.
                if line.endswith(b'\r\n'):
                    line = line[:-2] + b'\n'
                elif line.endswith(b'\r'):
                    line = line[:-1] + b'\n'
                self._file.write(line.replace(b'\n', linesep))
                if line == b'\n' or not line:
                    if first_pass:
                        first_pass = False
                        self._file.write(b'*** EOOH ***' + linesep)
                        message.seek(original_pos)
                    else:
                        break
            while True:
                line = message.readline()
                if not line:
                    break
                # Universal newline support.
                if line.endswith(b'\r\n'):
                    line = line[:-2] + linesep
                elif line.endswith(b'\r'):
                    line = line[:-1] + linesep
                elif line.endswith(b'\n'):
                    line = line[:-1] + linesep
                self._file.write(line)
        else:
            raise TypeError('Invalid message type: %s' % type(message))
        stop = self._file.tell()
        return (start, stop)


class Message(email.message.Message):
    """Message with mailbox-format-specific properties."""

    def __init__(self, message=None):
        """Initialize a Message instance."""
        if isinstance(message, email.message.Message):
            self._become_message(copy.deepcopy(message))
            if isinstance(message, Message):
                message._explain_to(self)
        elif isinstance(message, bytes):
            self._become_message(email.message_from_bytes(message))
        elif isinstance(message, str):
            self._become_message(email.message_from_string(message))
        elif isinstance(message, io.TextIOWrapper):
            self._become_message(email.message_from_file(message))
        elif hasattr(message, "read"):
            self._become_message(email.message_from_binary_file(message))
        elif message is None:
            email.message.Message.__init__(self)
        else:
            raise TypeError('Invalid message type: %s' % type(message))

    def _become_message(self, message):
        """Assume the non-format-specific state of message."""
        type_specific = getattr(message, '_type_specific_attributes', [])
        for name in message.__dict__:
            if name not in type_specific:
                self.__dict__[name] = message.__dict__[name]

    def _explain_to(self, message):
        """Copy format-specific state to message insofar as possible."""
        if isinstance(message, Message):
            return  # There's nothing format-specific to explain.
        else:
            raise TypeError('Cannot convert to specified type')


class MaildirMessage(Message):
    """Message with Maildir-specific properties."""

    _type_specific_attributes = ['_subdir', '_info', '_date']

    def __init__(self, message=None):
        """Initialize a MaildirMessage instance."""
        self._subdir = 'new'
        self._info = ''
        self._date = time.time()
        Message.__init__(self, message)

    def get_subdir(self):
        """Return 'new' or 'cur'."""
        return self._subdir

    def set_subdir(self, subdir):
        """Set subdir to 'new' or 'cur'."""
        if subdir == 'new' or subdir == 'cur':
            self._subdir = subdir
        else:
            raise ValueError("subdir must be 'new' or 'cur': %s" % subdir)

    def get_flags(self):
        """Return as a string the flags that are set."""
        if self._info.startswith('2,'):
            return self._info[2:]
        else:
            return ''

    def set_flags(self, flags):
        """Set the given flags and unset all others."""
        self._info = '2,' + ''.join(sorted(flags))

    def add_flag(self, flag):
        """Set the given flag(s) without changing others."""
        self.set_flags(''.join(set(self.get_flags()) | set(flag)))

    def remove_flag(self, flag):
        """Unset the given string flag(s) without changing others."""
        if self.get_flags():
            self.set_flags(''.join(set(self.get_flags()) - set(flag)))

    def get_date(self):
        """Return delivery date of message, in seconds since the epoch."""
        return self._date

    def set_date(self, date):
        """Set delivery date of message, in seconds since the epoch."""
        try:
            self._date = float(date)
        except ValueError:
            raise TypeError("can't convert to float: %s" % date) from None

    def get_info(self):
        """Get the message's "info" as a string."""
        return self._info

    def set_info(self, info):
        """Set the message's "info" string."""
        if isinstance(info, str):
            self._info = info
        else:
            raise TypeError('info must be a string: %s' % type(info))

    def _explain_to(self, message):
        """Copy Maildir-specific state to message insofar as possible."""
        if isinstance(message, MaildirMessage):
            message.set_flags(self.get_flags())
            message.set_subdir(self.get_subdir())
            message.set_date(self.get_date())
        elif isinstance(message, _mboxMMDFMessage):
            flags = set(self.get_flags())
            if 'S' in flags:
                message.add_flag('R')
            if self.get_subdir() == 'cur':
                message.add_flag('O')
            if 'T' in flags:
                message.add_flag('D')
            if 'F' in flags:
                message.add_flag('F')
            if 'R' in flags:
                message.add_flag('A')
            message.set_from('MAILER-DAEMON', time.gmtime(self.get_date()))
        elif isinstance(message, MHMessage):
            flags = set(self.get_flags())
            if 'S' not in flags:
                message.add_sequence('unseen')
            if 'R' in flags:
                message.add_sequence('replied')
            if 'F' in flags:
                message.add_sequence('flagged')
        elif isinstance(message, BabylMessage):
            flags = set(self.get_flags())
            if 'S' not in flags:
                message.add_label('unseen')
            if 'T' in flags:
                message.add_label('deleted')
            if 'R' in flags:
                message.add_label('answered')
            if 'P' in flags:
                message.add_label('forwarded')
        elif isinstance(message, Message):
            pass
        else:
            raise TypeError('Cannot convert to specified type: %s' %
                            type(message))


class _mboxMMDFMessage(Message):
    """Message with mbox- or MMDF-specific properties."""

    _type_specific_attributes = ['_from']

    def __init__(self, message=None):
        """Initialize an mboxMMDFMessage instance."""
        self.set_from('MAILER-DAEMON', True)
        if isinstance(message, email.message.Message):
            unixfrom = message.get_unixfrom()
            if unixfrom is not None and unixfrom.startswith('From '):
                self.set_from(unixfrom[5:])
        Message.__init__(self, message)

    def get_from(self):
        """Return contents of "From " line."""
        return self._from

    def set_from(self, from_, time_=None):
        """Set "From " line, formatting and appending time_ if specified."""
        if time_ is not None:
            if time_ is True:
                time_ = time.gmtime()
            from_ += ' ' + time.asctime(time_)
        self._from = from_

    def get_flags(self):
        """Return as a string the flags that are set."""
        return self.get('Status', '') + self.get('X-Status', '')

    def set_flags(self, flags):
        """Set the given flags and unset all others."""
        flags = set(flags)
        status_flags, xstatus_flags = '', ''
        for flag in ('R', 'O'):
            if flag in flags:
                status_flags += flag
                flags.remove(flag)
        for flag in ('D', 'F', 'A'):
            if flag in flags:
                xstatus_flags += flag
                flags.remove(flag)
        xstatus_flags += ''.join(sorted(flags))
        try:
            self.replace_header('Status', status_flags)
        except KeyError:
            self.add_header('Status', status_flags)
        try:
            self.replace_header('X-Status', xstatus_flags)
        except KeyError:
            self.add_header('X-Status', xstatus_flags)

    def add_flag(self, flag):
        """Set the given flag(s) without changing others."""
        self.set_flags(''.join(set(self.get_flags()) | set(flag)))

    def remove_flag(self, flag):
        """Unset the given string flag(s) without changing others."""
        if 'Status' in self or 'X-Status' in self:
            self.set_flags(''.join(set(self.get_flags()) - set(flag)))

    def _explain_to(self, message):
        """Copy mbox- or MMDF-specific state to message insofar as possible."""
        if isinstance(message, MaildirMessage):
            flags = set(self.get_flags())
            if 'O' in flags:
                message.set_subdir('cur')
            if 'F' in flags:
                message.add_flag('F')
            if 'A' in flags:
                message.add_flag('R')
            if 'R' in flags:
                message.add_flag('S')
            if 'D' in flags:
                message.add_flag('T')
            del message['status']
            del message['x-status']
            maybe_date = ' '.join(self.get_from().split()[-5:])
            try:
                message.set_date(calendar.timegm(time.strptime(maybe_date,
                                                      '%a %b %d %H:%M:%S %Y')))
            except (ValueError, OverflowError):
                pass
        elif isinstance(message, _mboxMMDFMessage):
            message.set_flags(self.get_flags())
            message.set_from(self.get_from())
        elif isinstance(message, MHMessage):
            flags = set(self.get_flags())
            if 'R' not in flags:
                message.add_sequence('unseen')
            if 'A' in flags:
                message.add_sequence('replied')
            if 'F' in flags:
                message.add_sequence('flagged')
            del message['status']
            del message['x-status']
        elif isinstance(message, BabylMessage):
            flags = set(self.get_flags())
            if 'R' not in flags:
                message.add_label('unseen')
            if 'D' in flags:
                message.add_label('deleted')
            if 'A' in flags:
                message.add_label('answered')
            del message['status']
            del message['x-status']
        elif isinstance(message, Message):
            pass
        else:
            raise TypeError('Cannot convert to specified type: %s' %
                            type(message))


class mboxMessage(_mboxMMDFMessage):
    """Message with mbox-specific properties."""


class MHMessage(Message):
    """Message with MH-specific properties."""

    _type_specific_attributes = ['_sequences']

    def __init__(self, message=None):
        """Initialize an MHMessage instance."""
        self._sequences = []
        Message.__init__(self, message)

    def get_sequences(self):
        """Return a list of sequences that include the message."""
        return self._sequences[:]

    def set_sequences(self, sequences):
        """Set the list of sequences that include the message."""
        self._sequences = list(sequences)

    def add_sequence(self, sequence):
        """Add sequence to list of sequences including the message."""
        if isinstance(sequence, str):
            if not sequence in self._sequences:
                self._sequences.append(sequence)
        else:
            raise TypeError('sequence type must be str: %s' % type(sequence))

    def remove_sequence(self, sequence):
        """Remove sequence from the list of sequences including the message."""
        try:
            self._sequences.remove(sequence)
        except ValueError:
            pass

    def _explain_to(self, message):
        """Copy MH-specific state to message insofar as possible."""
        if isinstance(message, MaildirMessage):
            sequences = set(self.get_sequences())
            if 'unseen' in sequences:
                message.set_subdir('cur')
            else:
                message.set_subdir('cur')
                message.add_flag('S')
            if 'flagged' in sequences:
                message.add_flag('F')
            if 'replied' in sequences:
                message.add_flag('R')
        elif isinstance(message, _mboxMMDFMessage):
            sequences = set(self.get_sequences())
            if 'unseen' not in sequences:
                message.add_flag('RO')
            else:
                message.add_flag('O')
            if 'flagged' in sequences:
                message.add_flag('F')
            if 'replied' in sequences:
                message.add_flag('A')
        elif isinstance(message, MHMessage):
            for sequence in self.get_sequences():
                message.add_sequence(sequence)
        elif isinstance(message, BabylMessage):
            sequences = set(self.get_sequences())
            if 'unseen' in sequences:
                message.add_label('unseen')
            if 'replied' in sequences:
                message.add_label('answered')
        elif isinstance(message, Message):
            pass
        else:
            raise TypeError('Cannot convert to specified type: %s' %
                            type(message))


class BabylMessage(Message):
    """Message with Babyl-specific properties."""

    _type_specific_attributes = ['_labels', '_visible']

    def __init__(self, message=None):
        """Initialize a BabylMessage instance."""
        self._labels = []
        self._visible = Message()
        Message.__init__(self, message)

    def get_labels(self):
        """Return a list of labels on the message."""
        return self._labels[:]

    def set_labels(self, labels):
        """Set the list of labels on the message."""
        self._labels = list(labels)

    def add_label(self, label):
        """Add label to list of labels on the message."""
        if isinstance(label, str):
            if label not in self._labels:
                self._labels.append(label)
        else:
            raise TypeError('label must be a string: %s' % type(label))

    def remove_label(self, label):
        """Remove label from the list of labels on the message."""
        try:
            self._labels.remove(label)
        except ValueError:
            pass

    def get_visible(self):
        """Return a Message representation of visible headers."""
        return Message(self._visible)

    def set_visible(self, visible):
        """Set the Message representation of visible headers."""
        self._visible = Message(visible)

    def update_visible(self):
        """Update and/or sensibly generate a set of visible headers."""
        for header in self._visible.keys():
            if header in self:
                self._visible.replace_header(header, self[header])
            else:
                del self._visible[header]
        for header in ('Date', 'From', 'Reply-To', 'To', 'CC', 'Subject'):
            if header in self and header not in self._visible:
                self._visible[header] = self[header]

    def _explain_to(self, message):
        """Copy Babyl-specific state to message insofar as possible."""
        if isinstance(message, MaildirMessage):
            labels = set(self.get_labels())
            if 'unseen' in labels:
                message.set_subdir('cur')
            else:
                message.set_subdir('cur')
                message.add_flag('S')
            if 'forwarded' in labels or 'resent' in labels:
                message.add_flag('P')
            if 'answered' in labels:
                message.add_flag('R')
            if 'deleted' in labels:
                message.add_flag('T')
        elif isinstance(message, _mboxMMDFMessage):
            labels = set(self.get_labels())
            if 'unseen' not in labels:
                message.add_flag('RO')
            else:
                message.add_flag('O')
            if 'deleted' in labels:
                message.add_flag('D')
            if 'answered' in labels:
                message.add_flag('A')
        elif isinstance(message, MHMessage):
            labels = set(self.get_labels())
            if 'unseen' in labels:
                message.add_sequence('unseen')
            if 'answered' in labels:
                message.add_sequence('replied')
        elif isinstance(message, BabylMessage):
            message.set_visible(self.get_visible())
            for label in self.get_labels():
                message.add_label(label)
        elif isinstance(message, Message):
            pass
        else:
            raise TypeError('Cannot convert to specified type: %s' %
                            type(message))


class MMDFMessage(_mboxMMDFMessage):
    """Message with MMDF-specific properties."""


class _ProxyFile:
    """A read-only wrapper of a file."""

    def __init__(self, f, pos=None):
        """Initialize a _ProxyFile."""
        self._file = f
        if pos is None:
            self._pos = f.tell()
        else:
            self._pos = pos

    def read(self, size=None):
        """Read bytes."""
        return self._read(size, self._file.read)

    def read1(self, size=None):
        """Read bytes."""
        return self._read(size, self._file.read1)

    def readline(self, size=None):
        """Read a line."""
        return self._read(size, self._file.readline)

    def readlines(self, sizehint=None):
        """Read multiple lines."""
        result = []
        for line in self:
            result.append(line)
            if sizehint is not None:
                sizehint -= len(line)
                if sizehint <= 0:
                    break
        return result

    def __iter__(self):
        """Iterate over lines."""
        while True:
            line = self.readline()
            if not line:
                return
            yield line

    def tell(self):
        """Return the position."""
        return self._pos

    def seek(self, offset, whence=0):
        """Change position."""
        if whence == 1:
            self._file.seek(self._pos)
        self._file.seek(offset, whence)
        self._pos = self._file.tell()

    def close(self):
        """Close the file."""
        if hasattr(self, '_file'):
            try:
                if hasattr(self._file, 'close'):
                    self._file.close()
            finally:
                del self._file

    def _read(self, size, read_method):
        """Read size bytes using read_method."""
        if size is None:
            size = -1
        self._file.seek(self._pos)
        result = read_method(size)
        self._pos = self._file.tell()
        return result

    def __enter__(self):
        """Context management protocol support."""
        return self

    def __exit__(self, *exc):
        self.close()

    def readable(self):
        return self._file.readable()

    def writable(self):
        return self._file.writable()

    def seekable(self):
        return self._file.seekable()

    def flush(self):
        return self._file.flush()

    @property
    def closed(self):
        if not hasattr(self, '_file'):
            return True
        if not hasattr(self._file, 'closed'):
            return False
        return self._file.closed

    __class_getitem__ = classmethod(GenericAlias)


class _PartialFile(_ProxyFile):
    """A read-only wrapper of part of a file."""

    def __init__(self, f, start=None, stop=None):
        """Initialize a _PartialFile."""
        _ProxyFile.__init__(self, f, start)
        self._start = start
        self._stop = stop

    def tell(self):
        """Return the position with respect to start."""
        return _ProxyFile.tell(self) - self._start

    def seek(self, offset, whence=0):
        """Change position, possibly with respect to start or stop."""
        if whence == 0:
            self._pos = self._start
            whence = 1
        elif whence == 2:
            self._pos = self._stop
            whence = 1
        _ProxyFile.seek(self, offset, whence)

    def _read(self, size, read_method):
        """Read size bytes using read_method, honoring start and stop."""
        remaining = self._stop - self._pos
        if remaining <= 0:
            return b''
        if size is None or size < 0 or size > remaining:
            size = remaining
        return _ProxyFile._read(self, size, read_method)

    def close(self):
        # do *not* close the underlying file object for partial files,
        # since it's global to the mailbox object
        if hasattr(self, '_file'):
            del self._file


def _lock_file(f, dotlock=True):
    """Lock file f using lockf and dot locking."""
    dotlock_done = False
    try:
        if fcntl:
            try:
                fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EACCES, errno.EROFS):
                    raise ExternalClashError('lockf: lock unavailable: %s' %
                                             f.name)
                else:
                    raise
        if dotlock:
            try:
                pre_lock = _create_temporary(f.name + '.lock')
                pre_lock.close()
            except OSError as e:
                if e.errno in (errno.EACCES, errno.EROFS):
                    return  # Without write access, just skip dotlocking.
                else:
                    raise
            try:
                try:
                    os.link(pre_lock.name, f.name + '.lock')
                    dotlock_done = True
                except (AttributeError, PermissionError):
                    os.rename(pre_lock.name, f.name + '.lock')
                    dotlock_done = True
                else:
                    os.unlink(pre_lock.name)
            except FileExistsError:
                os.remove(pre_lock.name)
                raise ExternalClashError('dot lock unavailable: %s' %
                                         f.name)
    except:
        if fcntl:
            fcntl.lockf(f, fcntl.LOCK_UN)
        if dotlock_done:
            os.remove(f.name + '.lock')
        raise

def _unlock_file(f):
    """Unlock file f using lockf and dot locking."""
    if fcntl:
        fcntl.lockf(f, fcntl.LOCK_UN)
    if os.path.exists(f.name + '.lock'):
        os.remove(f.name + '.lock')

def _create_carefully(path):
    """Create a file if it doesn't exist and open for reading and writing."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o666)
    try:
        return open(path, 'rb+')
    finally:
        os.close(fd)

def _create_temporary(path):
    """Create a temp file based on path and open for reading and writing."""
    return _create_carefully('%s.%s.%s.%s' % (path, int(time.time()),
                                              socket.gethostname(),
                                              os.getpid()))

def _sync_flush(f):
    """Ensure changes to file f are physically on disk."""
    f.flush()
    if hasattr(os, 'fsync'):
        os.fsync(f.fileno())

def _sync_close(f):
    """Close file f, ensuring all changes are physically on disk."""
    _sync_flush(f)
    f.close()


class Error(Exception):
    """Raised for module-specific errors."""

class NoSuchMailboxError(Error):
    """The specified mailbox does not exist and won't be created."""

class NotEmptyError(Error):
    """The specified mailbox is not empty and deletion was requested."""

class ExternalClashError(Error):
    """Another process caused an action to fail."""

class FormatError(Error):
    """A file appears to have an invalid format."""
<|endoftext|>import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import warnings
from _collections_abc import Sequence
from errno import EINVAL, ENOENT, ENOTDIR, EBADF, ELOOP
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
from urllib.parse import quote_from_bytes as urlquote_from_bytes


__all__ = [
    "PurePath", "PurePosixPath", "PureWindowsPath",
    "Path", "PosixPath", "WindowsPath",
    ]

#
# Internals
#

_WINERROR_NOT_READY = 21  # drive exists but is not accessible
_WINERROR_INVALID_NAME = 123  # fix for bpo-35306
_WINERROR_CANT_RESOLVE_FILENAME = 1921  # broken symlink pointing to itself

# EBADF - guard against macOS `stat` throwing EBADF
_IGNORED_ERROS = (ENOENT, ENOTDIR, EBADF, ELOOP)

_IGNORED_WINERRORS = (
    _WINERROR_NOT_READY,
    _WINERROR_INVALID_NAME,
    _WINERROR_CANT_RESOLVE_FILENAME)

def _ignore_error(exception):
    return (getattr(exception, 'errno', None) in _IGNORED_ERROS or
            getattr(exception, 'winerror', None) in _IGNORED_WINERRORS)


def _is_wildcard_pattern(pat):
    # Whether this pattern needs actual matching using fnmatch, or can
    # be looked up directly as a file.
    return "*" in pat or "?" in pat or "[" in pat


class _Flavour(object):
    """A flavour implements a particular (platform-specific) set of path
    semantics."""

    def __init__(self):
        self.join = self.sep.join

    def parse_parts(self, parts):
        parsed = []
        sep = self.sep
        altsep = self.altsep
        drv = root = ''
        it = reversed(parts)
        for part in it:
            if not part:
                continue
            if altsep:
                part = part.replace(altsep, sep)
            drv, root, rel = self.splitroot(part)
            if sep in rel:
                for x in reversed(rel.split(sep)):
                    if x and x != '.':
                        parsed.append(sys.intern(x))
            else:
                if rel and rel != '.':
                    parsed.append(sys.intern(rel))
            if drv or root:
                if not drv:
                    # If no drive is present, try to find one in the previous
                    # parts. This makes the result of parsing e.g.
                    # ("C:", "/", "a") reasonably intuitive.
                    for part in it:
                        if not part:
                            continue
                        if altsep:
                            part = part.replace(altsep, sep)
                        drv = self.splitroot(part)[0]
                        if drv:
                            break
                break
        if drv or root:
            parsed.append(drv + root)
        parsed.reverse()
        return drv, root, parsed

    def join_parsed_parts(self, drv, root, parts, drv2, root2, parts2):
        """
        Join the two paths represented by the respective
        (drive, root, parts) tuples.  Return a new (drive, root, parts) tuple.
        """
        if root2:
            if not drv2 and drv:
                return drv, root2, [drv + root2] + parts2[1:]
        elif drv2:
            if drv2 == drv or self.casefold(drv2) == self.casefold(drv):
                # Same drive => second path is relative to the first
                return drv, root, parts + parts2[1:]
        else:
            # Second path is non-anchored (common case)
            return drv, root, parts + parts2
        return drv2, root2, parts2


class _WindowsFlavour(_Flavour):
    # Reference for Windows paths can be found at
    # http://msdn.microsoft.com/en-us/library/aa365247%28v=vs.85%29.aspx

    sep = '\\'
    altsep = '/'
    has_drv = True
    pathmod = ntpath

    is_supported = (os.name == 'nt')

    drive_letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    ext_namespace_prefix = '\\\\?\\'

    reserved_names = (
        {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'} |
        {'COM%s' % c for c in '123456789\xb9\xb2\xb3'} |
        {'LPT%s' % c for c in '123456789\xb9\xb2\xb3'}
        )

    # Interesting findings about extended paths:
    # * '\\?\c:\a' is an extended path, which bypasses normal Windows API
    #   path processing. Thus relative paths are not resolved and slash is not
    #   translated to backslash. It has the native NT path limit of 32767
    #   characters, but a bit less after resolving device symbolic links,
    #   such as '\??\C:' => '\Device\HarddiskVolume2'.
    # * '\\?\c:/a' looks for a device named 'C:/a' because slash is a
    #   regular name character in the object namespace.
    # * '\\?\c:\foo/bar' is invalid because '/' is illegal in NT filesystems.
    #   The only path separator at the filesystem level is backslash.
    # * '//?/c:\a' and '//?/c:/a' are effectively equivalent to '\\.\c:\a' and
    #   thus limited to MAX_PATH.
    # * Prior to Windows 8, ANSI API bytes paths are limited to MAX_PATH,
    #   even with the '\\?\' prefix.

    def splitroot(self, part, sep=sep):
        first = part[0:1]
        second = part[1:2]
        if (second == sep and first == sep):
            # XXX extended paths should also disable the collapsing of "."
            # components (according to MSDN docs).
            prefix, part = self._split_extended_path(part)
            first = part[0:1]
            second = part[1:2]
        else:
            prefix = ''
        third = part[2:3]
        if (second == sep and first == sep and third != sep):
            # is a UNC path:
            # vvvvvvvvvvvvvvvvvvvvv root
            # \\machine\mountpoint\directory\etc\...
            #            directory ^^^^^^^^^^^^^^
            index = part.find(sep, 2)
            if index != -1:
                index2 = part.find(sep, index + 1)
                # a UNC path can't have two slashes in a row
                # (after the initial two)
                if index2 != index + 1:
                    if index2 == -1:
                        index2 = len(part)
                    if prefix:
                        return prefix + part[1:index2], sep, part[index2+1:]
                    else:
                        return part[:index2], sep, part[index2+1:]
        drv = root = ''
        if second == ':' and first in self.drive_letters:
            drv = part[:2]
            part = part[2:]
            first = third
        if first == sep:
            root = first
            part = part.lstrip(sep)
        return prefix + drv, root, part

    def casefold(self, s):
        return s.lower()

    def casefold_parts(self, parts):
        return [p.lower() for p in parts]

    def compile_pattern(self, pattern):
        return re.compile(fnmatch.translate(pattern), re.IGNORECASE).fullmatch

    def _split_extended_path(self, s, ext_prefix=ext_namespace_prefix):
        prefix = ''
        if s.startswith(ext_prefix):
            prefix = s[:4]
            s = s[4:]
            if s.startswith('UNC\\'):
                prefix += s[:3]
                s = '\\' + s[3:]
        return prefix, s

    def is_reserved(self, parts):
        # NOTE: the rules for reserved names seem somewhat complicated
        # (e.g. r"..\NUL" is reserved but not r"foo\NUL" if "foo" does not
        # exist). We err on the side of caution and return True for paths
        # which are not considered reserved by Windows.
        if not parts:
            return False
        if parts[0].startswith('\\\\'):
            # UNC paths are never reserved
            return False
        name = parts[-1].partition('.')[0].partition(':')[0].rstrip(' ')
        return name.upper() in self.reserved_names

    def make_uri(self, path):
        # Under Windows, file URIs use the UTF-8 encoding.
        drive = path.drive
        if len(drive) == 2 and drive[1] == ':':
            # It's a path on a local drive => 'file:///c:/a/b'
            rest = path.as_posix()[2:].lstrip('/')
            return 'file:///%s/%s' % (
                drive, urlquote_from_bytes(rest.encode('utf-8')))
        else:
            # It's a path on a network drive => 'file://host/share/a/b'
            return 'file:' + urlquote_from_bytes(path.as_posix().encode('utf-8'))


class _PosixFlavour(_Flavour):
    sep = '/'
    altsep = ''
    has_drv = False
    pathmod = posixpath

    is_supported = (os.name != 'nt')

    def splitroot(self, part, sep=sep):
        if part and part[0] == sep:
            stripped_part = part.lstrip(sep)
            # According to POSIX path resolution:
            # http://pubs.opengroup.org/onlinepubs/009695399/basedefs/xbd_chap04.html#tag_04_11
            # "A pathname that begins with two successive slashes may be
            # interpreted in an implementation-defined manner, although more
            # than two leading slashes shall be treated as a single slash".
            if len(part) - len(stripped_part) == 2:
                return '', sep * 2, stripped_part
            else:
                return '', sep, stripped_part
        else:
            return '', '', part

    def casefold(self, s):
        return s

    def casefold_parts(self, parts):
        return parts

    def compile_pattern(self, pattern):
        return re.compile(fnmatch.translate(pattern)).fullmatch

    def is_reserved(self, parts):
        return False

    def make_uri(self, path):
        # We represent the path using the local filesystem encoding,
        # for portability to other applications.
        bpath = bytes(path)
        return 'file://' + urlquote_from_bytes(bpath)


_windows_flavour = _WindowsFlavour()
_posix_flavour = _PosixFlavour()


class _Accessor:
    """An accessor implements a particular (system-specific or not) way of
    accessing paths on the filesystem."""


class _NormalAccessor(_Accessor):

    stat = os.stat

    open = io.open

    listdir = os.listdir

    scandir = os.scandir

    chmod = os.chmod

    mkdir = os.mkdir

    unlink = os.unlink

    if hasattr(os, "link"):
        link = os.link
    else:
        def link(self, src, dst):
            raise NotImplementedError("os.link() not available on this system")

    rmdir = os.rmdir

    rename = os.rename

    replace = os.replace

    if hasattr(os, "symlink"):
        symlink = os.symlink
    else:
        def symlink(self, src, dst, target_is_directory=False):
            raise NotImplementedError("os.symlink() not available on this system")

    def touch(self, path, mode=0o666, exist_ok=True):
        if exist_ok:
            # First try to bump modification time
            # Implementation note: GNU touch uses the UTIME_NOW option of
            # the utimensat() / futimens() functions.
            try:
                os.utime(path, None)
            except OSError:
                # Avoid exception chaining
                pass
            else:
                return
        flags = os.O_CREAT | os.O_WRONLY
        if not exist_ok:
            flags |= os.O_EXCL
        fd = os.open(path, flags, mode)
        os.close(fd)

    if hasattr(os, "readlink"):
        readlink = os.readlink
    else:
        def readlink(self, path):
            raise NotImplementedError("os.readlink() not available on this system")

    def owner(self, path):
        try:
            import pwd
            return pwd.getpwuid(self.stat(path).st_uid).pw_name
        except ImportError:
            raise NotImplementedError("Path.owner() is unsupported on this system")

    def group(self, path):
        try:
            import grp
            return grp.getgrgid(self.stat(path).st_gid).gr_name
        except ImportError:
            raise NotImplementedError("Path.group() is unsupported on this system")

    getcwd = os.getcwd

    expanduser = staticmethod(os.path.expanduser)

    realpath = staticmethod(os.path.realpath)


_normal_accessor = _NormalAccessor()


#
# Globbing helpers
#

def _make_selector(pattern_parts, flavour):
    pat = pattern_parts[0]
    child_parts = pattern_parts[1:]
    if pat == '**':
        cls = _RecursiveWildcardSelector
    elif '**' in pat:
        raise ValueError("Invalid pattern: '**' can only be an entire path component")
    elif _is_wildcard_pattern(pat):
        cls = _WildcardSelector
    else:
        cls = _PreciseSelector
    return cls(pat, child_parts, flavour)

if hasattr(functools, "lru_cache"):
    _make_selector = functools.lru_cache()(_make_selector)


class _Selector:
    """A selector matches a specific glob pattern part against the children
    of a given path."""

    def __init__(self, child_parts, flavour):
        self.child_parts = child_parts
        if child_parts:
            self.successor = _make_selector(child_parts, flavour)
            self.dironly = True
        else:
            self.successor = _TerminatingSelector()
            self.dironly = False

    def select_from(self, parent_path):
        """Iterate over all child paths of `parent_path` matched by this
        selector.  This can contain parent_path itself."""
        path_cls = type(parent_path)
        is_dir = path_cls.is_dir
        exists = path_cls.exists
        scandir = parent_path._accessor.scandir
        if not is_dir(parent_path):
            return iter([])
        return self._select_from(parent_path, is_dir, exists, scandir)


class _TerminatingSelector:

    def _select_from(self, parent_path, is_dir, exists, scandir):
        yield parent_path


class _PreciseSelector(_Selector):

    def __init__(self, name, child_parts, flavour):
        self.name = name
        _Selector.__init__(self, child_parts, flavour)

    def _select_from(self, parent_path, is_dir, exists, scandir):
        try:
            path = parent_path._make_child_relpath(self.name)
            if (is_dir if self.dironly else exists)(path):
                for p in self.successor._select_from(path, is_dir, exists, scandir):
                    yield p
        except PermissionError:
            return


class _WildcardSelector(_Selector):

    def __init__(self, pat, child_parts, flavour):
        self.match = flavour.compile_pattern(pat)
        _Selector.__init__(self, child_parts, flavour)

    def _select_from(self, parent_path, is_dir, exists, scandir):
        try:
            with scandir(parent_path) as scandir_it:
                entries = list(scandir_it)
            for entry in entries:
                if self.dironly:
                    try:
                        # "entry.is_dir()" can raise PermissionError
                        # in some cases (see bpo-38894), which is not
                        # among the errors ignored by _ignore_error()
                        if not entry.is_dir():
                            continue
                    except OSError as e:
                        if not _ignore_error(e):
                            raise
                        continue
                name = entry.name
                if self.match(name):
                    path = parent_path._make_child_relpath(name)
                    for p in self.successor._select_from(path, is_dir, exists, scandir):
                        yield p
        except PermissionError:
            return


class _RecursiveWildcardSelector(_Selector):

    def __init__(self, pat, child_parts, flavour):
        _Selector.__init__(self, child_parts, flavour)

    def _iterate_directories(self, parent_path, is_dir, scandir):
        yield parent_path
        try:
            with scandir(parent_path) as scandir_it:
                entries = list(scandir_it)
            for entry in entries:
                entry_is_dir = False
                try:
                    entry_is_dir = entry.is_dir()
                except OSError as e:
                    if not _ignore_error(e):
                        raise
                if entry_is_dir and not entry.is_symlink():
                    path = parent_path._make_child_relpath(entry.name)
                    for p in self._iterate_directories(path, is_dir, scandir):
                        yield p
        except PermissionError:
            return

    def _select_from(self, parent_path, is_dir, exists, scandir):
        try:
            yielded = set()
            try:
                successor_select = self.successor._select_from
                for starting_point in self._iterate_directories(parent_path, is_dir, scandir):
                    for p in successor_select(starting_point, is_dir, exists, scandir):
                        if p not in yielded:
                            yield p
                            yielded.add(p)
            finally:
                yielded.clear()
        except PermissionError:
            return


#
# Public API
#

class _PathParents(Sequence):
    """This object provides sequence-like access to the logical ancestors
    of a path.  Don't try to construct it yourself."""
    __slots__ = ('_pathcls', '_drv', '_root', '_parts')

    def __init__(self, path):
        # We don't store the instance to avoid reference cycles
        self._pathcls = type(path)
        self._drv = path._drv
        self._root = path._root
        self._parts = path._parts

    def __len__(self):
        if self._drv or self._root:
            return len(self._parts) - 1
        else:
            return len(self._parts)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self))))

        if idx >= len(self) or idx < -len(self):
            raise IndexError(idx)
        if idx < 0:
            idx += len(self)
        return self._pathcls._from_parsed_parts(self._drv, self._root,
                                                self._parts[:-idx - 1])

    def __repr__(self):
        return "<{}.parents>".format(self._pathcls.__name__)


class PurePath(object):
    """Base class for manipulating paths without I/O.

    PurePath represents a filesystem path and offers operations which
    don't imply any actual filesystem I/O.  Depending on your system,
    instantiating a PurePath will return either a PurePosixPath or a
    PureWindowsPath object.  You can also instantiate either of these classes
    directly, regardless of your system.
    """
    __slots__ = (
        '_drv', '_root', '_parts',
        '_str', '_hash', '_pparts', '_cached_cparts',
    )

    def __new__(cls, *args):
        """Construct a PurePath from one or several strings and or existing
        PurePath objects.  The strings and path objects are combined so as
        to yield a canonicalized path, which is incorporated into the
        new PurePath object.
        """
        if cls is PurePath:
            cls = PureWindowsPath if os.name == 'nt' else PurePosixPath
        return cls._from_parts(args)

    def __reduce__(self):
        # Using the parts tuple helps share interned path parts
        # when pickling related paths.
        return (self.__class__, tuple(self._parts))

    @classmethod
    def _parse_args(cls, args):
        # This is useful when you don't want to create an instance, just
        # canonicalize some constructor arguments.
        parts = []
        for a in args:
            if isinstance(a, PurePath):
                parts += a._parts
            else:
                a = os.fspath(a)
                if isinstance(a, str):
                    # Force-cast str subclasses to str (issue #21127)
                    parts.append(str(a))
                else:
                    raise TypeError(
                        "argument should be a str object or an os.PathLike "
                        "object returning str, not %r"
                        % type(a))
        return cls._flavour.parse_parts(parts)

    @classmethod
    def _from_parts(cls, args):
        # We need to call _parse_args on the instance, so as to get the
        # right flavour.
        self = object.__new__(cls)
        drv, root, parts = self._parse_args(args)
        self._drv = drv
        self._root = root
        self._parts = parts
        return self

    @classmethod
    def _from_parsed_parts(cls, drv, root, parts):
        self = object.__new__(cls)
        self._drv = drv
        self._root = root
        self._parts = parts
        return self

    @classmethod
    def _format_parsed_parts(cls, drv, root, parts):
        if drv or root:
            return drv + root + cls._flavour.join(parts[1:])
        else:
            return cls._flavour.join(parts)

    def _make_child(self, args):
        drv, root, parts = self._parse_args(args)
        drv, root, parts = self._flavour.join_parsed_parts(
            self._drv, self._root, self._parts, drv, root, parts)
        return self._from_parsed_parts(drv, root, parts)

    def __str__(self):
        """Return the string representation of the path, suitable for
        passing to system calls."""
        try:
            return self._str
        except AttributeError:
            self._str = self._format_parsed_parts(self._drv, self._root,
                                                  self._parts) or '.'
            return self._str

    def __fspath__(self):
        return str(self)

    def as_posix(self):
        """Return the string representation of the path with forward (/)
        slashes."""
        f = self._flavour
        return str(self).replace(f.sep, '/')

    def __bytes__(self):
        """Return the bytes representation of the path.  This is only
        recommended to use under Unix."""
        return os.fsencode(self)

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.as_posix())

    def as_uri(self):
        """Return the path as a 'file' URI."""
        if not self.is_absolute():
            raise ValueError("relative path can't be expressed as a file URI")
        return self._flavour.make_uri(self)

    @property
    def _cparts(self):
        # Cached casefolded parts, for hashing and comparison
        try:
            return self._cached_cparts
        except AttributeError:
            self._cached_cparts = self._flavour.casefold_parts(self._parts)
            return self._cached_cparts

    def __eq__(self, other):
        if not isinstance(other, PurePath):
            return NotImplemented
        return self._cparts == other._cparts and self._flavour is other._flavour

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(tuple(self._cparts))
            return self._hash

    def __lt__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts < other._cparts

    def __le__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts <= other._cparts

    def __gt__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts > other._cparts

    def __ge__(self, other):
        if not isinstance(other, PurePath) or self._flavour is not other._flavour:
            return NotImplemented
        return self._cparts >= other._cparts

    def __class_getitem__(cls, type):
        return cls

    drive = property(attrgetter('_drv'),
                     doc="""The drive prefix (letter or UNC path), if any.""")

    root = property(attrgetter('_root'),
                    doc="""The root of the path, if any.""")

    @property
    def anchor(self):
        """The concatenation of the drive and root, or ''."""
        anchor = self._drv + self._root
        return anchor

    @property
    def name(self):
        """The final path component, if any."""
        parts = self._parts
        if len(parts) == (1 if (self._drv or self._root) else 0):
            return ''
        return parts[-1]

    @property
    def suffix(self):
        """
        The final component's last suffix, if any.

        This includes the leading period. For example: '.txt'
        """
        name = self.name
        i = name.rfind('.')
        if 0 < i < len(name) - 1:
            return name[i:]
        else:
            return ''

    @property
    def suffixes(self):
        """
        A list of the final component's suffixes, if any.

        These include the leading periods. For example: ['.tar', '.gz']
        """
        name = self.name
        if name.endswith('.'):
            return []
        name = name.lstrip('.')
        return ['.' + suffix for suffix in name.split('.')[1:]]

    @property
    def stem(self):
        """The final path component, minus its last suffix."""
        name = self.name
        i = name.rfind('.')
        if 0 < i < len(name) - 1:
            return name[:i]
        else:
            return name

    def with_name(self, name):
        """Return a new path with the file name changed."""
        if not self.name:
            raise ValueError("%r has an empty name" % (self,))
        drv, root, parts = self._flavour.parse_parts((name,))
        if (not name or name[-1] in [self._flavour.sep, self._flavour.altsep]
            or drv or root or len(parts) != 1):
            raise ValueError("Invalid name %r" % (name))
        return self._from_parsed_parts(self._drv, self._root,
                                       self._parts[:-1] + [name])

    def with_stem(self, stem):
        """Return a new path with the stem changed."""
        return self.with_name(stem + self.suffix)

    def with_suffix(self, suffix):
        """Return a new path with the file suffix changed.  If the path
        has no suffix, add given suffix.  If the given suffix is an empty
        string, remove the suffix from the path.
        """
        f = self._flavour
        if f.sep in suffix or f.altsep and f.altsep in suffix:
            raise ValueError("Invalid suffix %r" % (suffix,))
        if suffix and not suffix.startswith('.') or suffix == '.':
            raise ValueError("Invalid suffix %r" % (suffix))
        name = self.name
        if not name:
            raise ValueError("%r has an empty name" % (self,))
        old_suffix = self.suffix
        if not old_suffix:
            name = name + suffix
        else:
            name = name[:-len(old_suffix)] + suffix
        return self._from_parsed_parts(self._drv, self._root,
                                       self._parts[:-1] + [name])

    def relative_to(self, *other):
        """Return the relative path to another path identified by the passed
        arguments.  If the operation is not possible (because this is not
        a subpath of the other path), raise ValueError.
        """
        # For the purpose of this method, drive and root are considered
        # separate parts, i.e.:
        #   Path('c:/').relative_to('c:')  gives Path('/')
        #   Path('c:/').relative_to('/')   raise ValueError
        if not other:
            raise TypeError("need at least one argument")
        parts = self._parts
        drv = self._drv
        root = self._root
        if root:
            abs_parts = [drv, root] + parts[1:]
        else:
            abs_parts = parts
        to_drv, to_root, to_parts = self._parse_args(other)
        if to_root:
            to_abs_parts = [to_drv, to_root] + to_parts[1:]
        else:
            to_abs_parts = to_parts
        n = len(to_abs_parts)
        cf = self._flavour.casefold_parts
        if (root or drv) if n == 0 else cf(abs_parts[:n]) != cf(to_abs_parts):
            formatted = self._format_parsed_parts(to_drv, to_root, to_parts)
            raise ValueError("{!r} is not in the subpath of {!r}"
                    " OR one path is relative and the other is absolute."
                             .format(str(self), str(formatted)))
        return self._from_parsed_parts('', root if n == 1 else '',
                                       abs_parts[n:])

    def is_relative_to(self, *other):
        """Return True if the path is relative to another path or False.
        """
        try:
            self.relative_to(*other)
            return True
        except ValueError:
            return False

    @property
    def parts(self):
        """An object providing sequence-like access to the
        components in the filesystem path."""
        # We cache the tuple to avoid building a new one each time .parts
        # is accessed.  XXX is this necessary?
        try:
            return self._pparts
        except AttributeError:
            self._pparts = tuple(self._parts)
            return self._pparts

    def joinpath(self, *args):
        """Combine this path with one or several arguments, and return a
        new path representing either a subpath (if all arguments are relative
        paths) or a totally different path (if one of the arguments is
        anchored).
        """
        return self._make_child(args)

    def __truediv__(self, key):
        try:
            return self._make_child((key,))
        except TypeError:
            return NotImplemented

    def __rtruediv__(self, key):
        try:
            return self._from_parts([key] + self._parts)
        except TypeError:
            return NotImplemented

    @property
    def parent(self):
        """The logical parent of the path."""
        drv = self._drv
        root = self._root
        parts = self._parts
        if len(parts) == 1 and (drv or root):
            return self
        return self._from_parsed_parts(drv, root, parts[:-1])

    @property
    def parents(self):
        """A sequence of this path's logical parents."""
        return _PathParents(self)

    def is_absolute(self):
        """True if the path is absolute (has both a root and, if applicable,
        a drive)."""
        if not self._root:
            return False
        return not self._flavour.has_drv or bool(self._drv)

    def is_reserved(self):
        """Return True if the path contains one of the special names reserved
        by the system, if any."""
        return self._flavour.is_reserved(self._parts)

    def match(self, path_pattern):
        """
        Return True if this path matches the given pattern.
        """
        cf = self._flavour.casefold
        path_pattern = cf(path_pattern)
        drv, root, pat_parts = self._flavour.parse_parts((path_pattern,))
        if not pat_parts:
            raise ValueError("empty pattern")
        if drv and drv != cf(self._drv):
            return False
        if root and root != cf(self._root):
            return False
        parts = self._cparts
        if drv or root:
            if len(pat_parts) != len(parts):
                return False
            pat_parts = pat_parts[1:]
        elif len(pat_parts) > len(parts):
            return False
        for part, pat in zip(reversed(parts), reversed(pat_parts)):
            if not fnmatch.fnmatchcase(part, pat):
                return False
        return True

# Can't subclass os.PathLike from PurePath and keep the constructor
# optimizations in PurePath._parse_args().
os.PathLike.register(PurePath)


class PurePosixPath(PurePath):
    """PurePath subclass for non-Windows systems.

    On a POSIX system, instantiating a PurePath should return this object.
    However, you can also instantiate it directly on any system.
    """
    _flavour = _posix_flavour
    __slots__ = ()


class PureWindowsPath(PurePath):
    """PurePath subclass for Windows systems.

    On a Windows system, instantiating a PurePath should return this object.
    However, you can also instantiate it directly on any system.
    """
    _flavour = _windows_flavour
    __slots__ = ()


# Filesystem-accessing classes


class Path(PurePath):
    """PurePath subclass that can make system calls.

    Path represents a filesystem path but unlike PurePath, also offers
    methods to do system calls on path objects. Depending on your system,
    instantiating a Path will return either a PosixPath or a WindowsPath
    object. You can also instantiate a PosixPath or WindowsPath directly,
    but cannot instantiate a WindowsPath on a POSIX system or vice versa.
    """
    _accessor = _normal_accessor
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        if cls is Path:
            cls = WindowsPath if os.name == 'nt' else PosixPath
        self = cls._from_parts(args)
        if not self._flavour.is_supported:
            raise NotImplementedError("cannot instantiate %r on your system"
                                      % (cls.__name__,))
        return self

    def _make_child_relpath(self, part):
        # This is an optimization used for dir walking.  `part` must be
        # a single part relative to this path.
        parts = self._parts + [part]
        return self._from_parsed_parts(self._drv, self._root, parts)

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        # https://bugs.python.org/issue39682
        # In previous versions of pathlib, this method marked this path as
        # closed; subsequent attempts to perform I/O would raise an IOError.
        # This functionality was never documented, and had the effect of
        # making Path objects mutable, contrary to PEP 428. In Python 3.9 the
        # _closed attribute was removed, and this method made a no-op.
        # This method and __enter__()/__exit__() should be deprecated and
        # removed in the future.
        pass

    # Public API

    @classmethod
    def cwd(cls):
        """Return a new path pointing to the current working directory
        (as returned by os.getcwd()).
        """
        return cls(cls._accessor.getcwd())

    @classmethod
    def home(cls):
        """Return a new path pointing to the user's home directory (as
        returned by os.path.expanduser('~')).
        """
        return cls("~").expanduser()

    def samefile(self, other_path):
        """Return whether other_path is the same or not as this file
        (as returned by os.path.samefile()).
        """
        st = self.stat()
        try:
            other_st = other_path.stat()
        except AttributeError:
            other_st = self._accessor.stat(other_path)
        return os.path.samestat(st, other_st)

    def iterdir(self):
        """Iterate over the files in this directory.  Does not yield any
        result for the special paths '.' and '..'.
        """
        for name in self._accessor.listdir(self):
            if name in {'.', '..'}:
                # Yielding a path object for these makes little sense
                continue
            yield self._make_child_relpath(name)

    def glob(self, pattern):
        """Iterate over this subtree and yield all existing files (of any
        kind, including directories) matching the given relative pattern.
        """
        sys.audit("pathlib.Path.glob", self, pattern)
        if not pattern:
            raise ValueError("Unacceptable pattern: {!r}".format(pattern))
        drv, root, pattern_parts = self._flavour.parse_parts((pattern,))
        if drv or root:
            raise NotImplementedError("Non-relative patterns are unsupported")
        selector = _make_selector(tuple(pattern_parts), self._flavour)
        for p in selector.select_from(self):
            yield p

    def rglob(self, pattern):
        """Recursively yield all existing files (of any kind, including
        directories) matching the given relative pattern, anywhere in
        this subtree.
        """
        sys.audit("pathlib.Path.rglob", self, pattern)
        drv, root, pattern_parts = self._flavour.parse_parts((pattern,))
        if drv or root:
            raise NotImplementedError("Non-relative patterns are unsupported")
        selector = _make_selector(("**",) + tuple(pattern_parts), self._flavour)
        for p in selector.select_from(self):
            yield p

    def absolute(self):
        """Return an absolute version of this path.  This function works
        even if the path doesn't point to anything.

        No normalization is done, i.e. all '.' and '..' will be kept along.
        Use resolve() to get the canonical path to a file.
        """
        # XXX untested yet!
        if self.is_absolute():
            return self
        # FIXME this must defer to the specific flavour (and, under Windows,
        # use nt._getfullpathname())
        return self._from_parts([self._accessor.getcwd()] + self._parts)

    def resolve(self, strict=False):
        """
        Make the path absolute, resolving all symlinks on the way and also
        normalizing it (for example turning slashes into backslashes under
        Windows).
        """

        def check_eloop(e):
            winerror = getattr(e, 'winerror', 0)
            if e.errno == ELOOP or winerror == _WINERROR_CANT_RESOLVE_FILENAME:
                raise RuntimeError("Symlink loop from %r" % e.filename)

        try:
            s = self._accessor.realpath(self, strict=strict)
        except OSError as e:
            check_eloop(e)
            raise
        p = self._from_parts((s,))

        # In non-strict mode, realpath() doesn't raise on symlink loops.
        # Ensure we get an exception by calling stat()
        if not strict:
            try:
                p.stat()
            except OSError as e:
                check_eloop(e)
        return p

    def stat(self, *, follow_symlinks=True):
        """
        Return the result of the stat() system call on this path, like
        os.stat() does.
        """
        return self._accessor.stat(self, follow_symlinks=follow_symlinks)

    def owner(self):
        """
        Return the login name of the file owner.
        """
        return self._accessor.owner(self)

    def group(self):
        """
        Return the group name of the file gid.
        """
        return self._accessor.group(self)

    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None):
        """
        Open the file pointed by this path and return a file object, as
        the built-in open() function does.
        """
        if "b" not in mode:
            encoding = io.text_encoding(encoding)
        return self._accessor.open(self, mode, buffering, encoding, errors,
                                   newline)

    def read_bytes(self):
        """
        Open the file in bytes mode, read it, and close the file.
        """
        with self.open(mode='rb') as f:
            return f.read()

    def read_text(self, encoding=None, errors=None):
        """
        Open the file in text mode, read it, and close the file.
        """
        encoding = io.text_encoding(encoding)
        with self.open(mode='r', encoding=encoding, errors=errors) as f:
            return f.read()

    def write_bytes(self, data):
        """
        Open the file in bytes mode, write to it, and close the file.
        """
        # type-check for the buffer interface before truncating the file
        view = memoryview(data)
        with self.open(mode='wb') as f:
            return f.write(view)

    def write_text(self, data, encoding=None, errors=None, newline=None):
        """
        Open the file in text mode, write to it, and close the file.
        """
        if not isinstance(data, str):
            raise TypeError('data must be str, not %s' %
                            data.__class__.__name__)
        encoding = io.text_encoding(encoding)
        with self.open(mode='w', encoding=encoding, errors=errors, newline=newline) as f:
            return f.write(data)

    def readlink(self):
        """
        Return the path to which the symbolic link points.
        """
        path = self._accessor.readlink(self)
        return self._from_parts((path,))

    def touch(self, mode=0o666, exist_ok=True):
        """
        Create this file with the given access mode, if it doesn't exist.
        """
        self._accessor.touch(self, mode, exist_ok)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        """
        Create a new directory at this given path.
        """
        try:
            self._accessor.mkdir(self, mode)
        except FileNotFoundError:
            if not parents or self.parent == self:
                raise
            self.parent.mkdir(parents=True, exist_ok=True)
            self.mkdir(mode, parents=False, exist_ok=exist_ok)
        except OSError:
            # Cannot rely on checking for EEXIST, since the operating system
            # could give priority to other errors like EACCES or EROFS
            if not exist_ok or not self.is_dir():
                raise

    def chmod(self, mode, *, follow_symlinks=True):
        """
        Change the permissions of the path, like os.chmod().
        """
        self._accessor.chmod(self, mode, follow_symlinks=follow_symlinks)

    def lchmod(self, mode):
        """
        Like chmod(), except if the path points to a symlink, the symlink's
        permissions are changed, rather than its target's.
        """
        self.chmod(mode, follow_symlinks=False)

    def unlink(self, missing_ok=False):
        """
        Remove this file or link.
        If the path is a directory, use rmdir() instead.
        """
        try:
            self._accessor.unlink(self)
        except FileNotFoundError:
            if not missing_ok:
                raise

    def rmdir(self):
        """
        Remove this directory.  The directory must be empty.
        """
        self._accessor.rmdir(self)

    def lstat(self):
        """
        Like stat(), except if the path points to a symlink, the symlink's
        status information is returned, rather than its target's.
        """
        return self.stat(follow_symlinks=False)

    def rename(self, target):
        """
        Rename this path to the target path.

        The target path may be absolute or relative. Relative paths are
        interpreted relative to the current working directory, *not* the
        directory of the Path object.

        Returns the new Path instance pointing to the target path.
        """
        self._accessor.rename(self, target)
        return self.__class__(target)

    def replace(self, target):
        """
        Rename this path to the target path, overwriting if that path exists.

        The target path may be absolute or relative. Relative paths are
        interpreted relative to the current working directory, *not* the
        directory of the Path object.

        Returns the new Path instance pointing to the target path.
        """
        self._accessor.replace(self, target)
        return self.__class__(target)

    def symlink_to(self, target, target_is_directory=False):
        """
        Make this path a symlink pointing to the target path.
        Note the order of arguments (link, target) is the reverse of os.symlink.
        """
        self._accessor.symlink(target, self, target_is_directory)

    def hardlink_to(self, target):
        """
        Make this path a hard link pointing to the same file as *target*.

        Note the order of arguments (self, target) is the reverse of os.link's.
        """
        self._accessor.link(target, self)

    def link_to(self, target):
        """
        Make the target path a hard link pointing to this path.

        Note this function does not make this path a hard link to *target*,
        despite the implication of the function and argument names. The order
        of arguments (target, link) is the reverse of Path.symlink_to, but
        matches that of os.link.

        Deprecated since Python 3.10 and scheduled for removal in Python 3.12.
        Use `hardlink_to()` instead.
        """
        warnings.warn("pathlib.Path.link_to() is deprecated and is scheduled "
                      "for removal in Python 3.12. "
                      "Use pathlib.Path.hardlink_to() instead.",
                      DeprecationWarning, stacklevel=2)
        self._accessor.link(self, target)

    # Convenience functions for querying the stat results

    def exists(self):
        """
        Whether this path exists.
        """
        try:
            self.stat()
        except OSError as e:
            if not _ignore_error(e):
                raise
            return False
        except ValueError:
            # Non-encodable path
            return False
        return True

    def is_dir(self):
        """
        Whether this path is a directory.
        """
        try:
            return S_ISDIR(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_file(self):
        """
        Whether this path is a regular file (also True for symlinks pointing
        to regular files).
        """
        try:
            return S_ISREG(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_mount(self):
        """
        Check if this path is a POSIX mount point
        """
        # Need to exist and be a dir
        if not self.exists() or not self.is_dir():
            return False

        try:
            parent_dev = self.parent.stat().st_dev
        except OSError:
            return False

        dev = self.stat().st_dev
        if dev != parent_dev:
            return True
        ino = self.stat().st_ino
        parent_ino = self.parent.stat().st_ino
        return ino == parent_ino

    def is_symlink(self):
        """
        Whether this path is a symbolic link.
        """
        try:
            return S_ISLNK(self.lstat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_block_device(self):
        """
        Whether this path is a block device.
        """
        try:
            return S_ISBLK(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_char_device(self):
        """
        Whether this path is a character device.
        """
        try:
            return S_ISCHR(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_fifo(self):
        """
        Whether this path is a FIFO.
        """
        try:
            return S_ISFIFO(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def is_socket(self):
        """
        Whether this path is a socket.
        """
        try:
            return S_ISSOCK(self.stat().st_mode)
        except OSError as e:
            if not _ignore_error(e):
                raise
            # Path doesn't exist or is a broken symlink
            # (see http://web.archive.org/web/20200623061726/https://bitbucket.org/pitrou/pathlib/issues/12/ )
            return False
        except ValueError:
            # Non-encodable path
            return False

    def expanduser(self):
        """ Return a new path with expanded ~ and ~user constructs
        (as returned by os.path.expanduser)
        """
        if (not (self._drv or self._root) and
            self._parts and self._parts[0][:1] == '~'):
            homedir = self._accessor.expanduser(self._parts[0])
            if homedir[:1] == "~":
                raise RuntimeError("Could not determine home directory.")
            return self._from_parts([homedir] + self._parts[1:])

        return self


class PosixPath(Path, PurePosixPath):
    """Path subclass for non-Windows systems.

    On a POSIX system, instantiating a Path should return this object.
    """
    __slots__ = ()

class WindowsPath(Path, PureWindowsPath):
    """Path subclass for Windows systems.

    On a Windows system, instantiating a Path should return this object.
    """
    __slots__ = ()

    def is_mount(self):
        raise NotImplementedError("Path.is_mount() is unsupported on this system")
<|endoftext|>"""Generic (shallow and deep) copying operations.

Interface summary:

        import copy

        x = copy.copy(y)        # make a shallow copy of y
        x = copy.deepcopy(y)    # make a deep copy of y

For module specific errors, copy.Error is raised.

The difference between shallow and deep copying is only relevant for
compound objects (objects that contain other objects, like lists or
class instances).

- A shallow copy constructs a new compound object and then (to the
  extent possible) inserts *the same objects* into it that the
  original contains.

- A deep copy constructs a new compound object and then, recursively,
  inserts *copies* into it of the objects found in the original.

Two problems often exist with deep copy operations that don't exist
with shallow copy operations:

 a) recursive objects (compound objects that, directly or indirectly,
    contain a reference to themselves) may cause a recursive loop

 b) because deep copy copies *everything* it may copy too much, e.g.
    administrative data structures that should be shared even between
    copies

Python's deep copy operation avoids these problems by:

 a) keeping a table of objects already copied during the current
    copying pass

 b) letting user-defined classes override the copying operation or the
    set of components copied

This version does not copy types like module, class, function, method,
nor stack trace, stack frame, nor file, socket, window, nor any
similar types.

Classes can use the same interfaces to control copying that they use
to control pickling: they can define methods called __getinitargs__(),
__getstate__() and __setstate__().  See the documentation for module
"pickle" for information on these methods.
"""

import types
import weakref
from copyreg import dispatch_table

class Error(Exception):
    pass
error = Error   # backward compatibility

try:
    from org.python.core import PyStringMap
except ImportError:
    PyStringMap = None

__all__ = ["Error", "copy", "deepcopy"]

def copy(x):
    """Shallow copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info.
    """

    cls = type(x)

    copier = _copy_dispatch.get(cls)
    if copier:
        return copier(x)

    if issubclass(cls, type):
        # treat it as a regular class:
        return _copy_immutable(x)

    copier = getattr(cls, "__copy__", None)
    if copier is not None:
        return copier(x)

    reductor = dispatch_table.get(cls)
    if reductor is not None:
        rv = reductor(x)
    else:
        reductor = getattr(x, "__reduce_ex__", None)
        if reductor is not None:
            rv = reductor(4)
        else:
            reductor = getattr(x, "__reduce__", None)
            if reductor:
                rv = reductor()
            else:
                raise Error("un(shallow)copyable object of type %s" % cls)

    if isinstance(rv, str):
        return x
    return _reconstruct(x, None, *rv)


_copy_dispatch = d = {}

def _copy_immutable(x):
    return x
for t in (type(None), int, float, bool, complex, str, tuple,
          bytes, frozenset, type, range, slice, property,
          types.BuiltinFunctionType, type(Ellipsis), type(NotImplemented),
          types.FunctionType, weakref.ref):
    d[t] = _copy_immutable
t = getattr(types, "CodeType", None)
if t is not None:
    d[t] = _copy_immutable

d[list] = list.copy
d[dict] = dict.copy
d[set] = set.copy
d[bytearray] = bytearray.copy

if PyStringMap is not None:
    d[PyStringMap] = PyStringMap.copy

del d, t

def deepcopy(x, memo=None, _nil=[]):
    """Deep copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info.
    """

    if memo is None:
        memo = {}

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    cls = type(x)

    copier = _deepcopy_dispatch.get(cls)
    if copier is not None:
        y = copier(x, memo)
    else:
        if issubclass(cls, type):
            y = _deepcopy_atomic(x, memo)
        else:
            copier = getattr(x, "__deepcopy__", None)
            if copier is not None:
                y = copier(memo)
            else:
                reductor = dispatch_table.get(cls)
                if reductor:
                    rv = reductor(x)
                else:
                    reductor = getattr(x, "__reduce_ex__", None)
                    if reductor is not None:
                        rv = reductor(4)
                    else:
                        reductor = getattr(x, "__reduce__", None)
                        if reductor:
                            rv = reductor()
                        else:
                            raise Error(
                                "un(deep)copyable object of type %s" % cls)
                if isinstance(rv, str):
                    y = x
                else:
                    y = _reconstruct(x, memo, *rv)

    # If is its own copy, don't memoize.
    if y is not x:
        memo[d] = y
        _keep_alive(x, memo) # Make sure x lives at least as long as d
    return y

_deepcopy_dispatch = d = {}

def _deepcopy_atomic(x, memo):
    return x
d[type(None)] = _deepcopy_atomic
d[type(Ellipsis)] = _deepcopy_atomic
d[type(NotImplemented)] = _deepcopy_atomic
d[int] = _deepcopy_atomic
d[float] = _deepcopy_atomic
d[bool] = _deepcopy_atomic
d[complex] = _deepcopy_atomic
d[bytes] = _deepcopy_atomic
d[str] = _deepcopy_atomic
d[types.CodeType] = _deepcopy_atomic
d[type] = _deepcopy_atomic
d[range] = _deepcopy_atomic
d[types.BuiltinFunctionType] = _deepcopy_atomic
d[types.FunctionType] = _deepcopy_atomic
d[weakref.ref] = _deepcopy_atomic
d[property] = _deepcopy_atomic

def _deepcopy_list(x, memo, deepcopy=deepcopy):
    y = []
    memo[id(x)] = y
    append = y.append
    for a in x:
        append(deepcopy(a, memo))
    return y
d[list] = _deepcopy_list

def _deepcopy_tuple(x, memo, deepcopy=deepcopy):
    y = [deepcopy(a, memo) for a in x]
    # We're not going to put the tuple in the memo, but it's still important we
    # check for it, in case the tuple contains recursive mutable structures.
    try:
        return memo[id(x)]
    except KeyError:
        pass
    for k, j in zip(x, y):
        if k is not j:
            y = tuple(y)
            break
    else:
        y = x
    return y
d[tuple] = _deepcopy_tuple

def _deepcopy_dict(x, memo, deepcopy=deepcopy):
    y = {}
    memo[id(x)] = y
    for key, value in x.items():
        y[deepcopy(key, memo)] = deepcopy(value, memo)
    return y
d[dict] = _deepcopy_dict
if PyStringMap is not None:
    d[PyStringMap] = _deepcopy_dict

def _deepcopy_method(x, memo): # Copy instance methods
    return type(x)(x.__func__, deepcopy(x.__self__, memo))
d[types.MethodType] = _deepcopy_method

del d

def _keep_alive(x, memo):
    """Keeps a reference to the object x in the memo.

    Because we remember objects by their id, we have
    to assure that possibly temporary objects are kept
    alive by referencing them.
    We store a reference at the id of the memo, which should
    normally not be used unless someone tries to deepcopy
    the memo itself...
    """
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # aha, this is the first one :-)
        memo[id(memo)]=[x]

def _reconstruct(x, memo, func, args,
                 state=None, listiter=None, dictiter=None,
                 *, deepcopy=deepcopy):
    deep = memo is not None
    if deep and args:
        args = (deepcopy(arg, memo) for arg in args)
    y = func(*args)
    if deep:
        memo[id(x)] = y

    if state is not None:
        if deep:
            state = deepcopy(state, memo)
        if hasattr(y, '__setstate__'):
            y.__setstate__(state)
        else:
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            else:
                slotstate = None
            if state is not None:
                y.__dict__.update(state)
            if slotstate is not None:
                for key, value in slotstate.items():
                    setattr(y, key, value)

    if listiter is not None:
        if deep:
            for item in listiter:
                item = deepcopy(item, memo)
                y.append(item)
        else:
            for item in listiter:
                y.append(item)
    if dictiter is not None:
        if deep:
            for key, value in dictiter:
                key = deepcopy(key, memo)
                value = deepcopy(value, memo)
                y[key] = value
        else:
            for key, value in dictiter:
                y[key] = value
    return y

del types, weakref, PyStringMap
<|endoftext|>"""Manage shelves of pickled objects.

A "shelf" is a persistent, dictionary-like object.  The difference
with dbm databases is that the values (not the keys!) in a shelf can
be essentially arbitrary Python objects -- anything that the "pickle"
module can handle.  This includes most class instances, recursive data
types, and objects containing lots of shared sub-objects.  The keys
are ordinary strings.

To summarize the interface (key is a string, data is an arbitrary
object):

        import shelve
        d = shelve.open(filename) # open, with (g)dbm filename -- no suffix

        d[key] = data   # store data at key (overwrites old data if
                        # using an existing key)
        data = d[key]   # retrieve a COPY of the data at key (raise
                        # KeyError if no such key) -- NOTE that this
                        # access returns a *copy* of the entry!
        del d[key]      # delete data stored at key (raises KeyError
                        # if no such key)
        flag = key in d # true if the key exists
        list = d.keys() # a list of all existing keys (slow!)

        d.close()       # close it

Dependent on the implementation, closing a persistent dictionary may
or may not be necessary to flush changes to disk.

Normally, d[key] returns a COPY of the entry.  This needs care when
mutable entries are mutated: for example, if d[key] is a list,
        d[key].append(anitem)
does NOT modify the entry d[key] itself, as stored in the persistent
mapping -- it only modifies the copy, which is then immediately
discarded, so that the append has NO effect whatsoever.  To append an
item to d[key] in a way that will affect the persistent mapping, use:
        data = d[key]
        data.append(anitem)
        d[key] = data

To avoid the problem with mutable entries, you may pass the keyword
argument writeback=True in the call to shelve.open.  When you use:
        d = shelve.open(filename, writeback=True)
then d keeps a cache of all entries you access, and writes them all back
to the persistent mapping when you call d.close().  This ensures that
such usage as d[key].append(anitem) works as intended.

However, using keyword argument writeback=True may consume vast amount
of memory for the cache, and it may make d.close() very slow, if you
access many of d's entries after opening it in this way: d has no way to
check which of the entries you access are mutable and/or which ones you
actually mutate, so it must cache, and write back at close, all of the
entries that you access.  You can call d.sync() to write back all the
entries in the cache, and empty the cache (d.sync() also synchronizes
the persistent dictionary on disk, if feasible).
"""

from pickle import DEFAULT_PROTOCOL, Pickler, Unpickler
from io import BytesIO

import collections.abc

__all__ = ["Shelf", "BsdDbShelf", "DbfilenameShelf", "open"]

class _ClosedDict(collections.abc.MutableMapping):
    'Marker for a closed dict.  Access attempts raise a ValueError.'

    def closed(self, *args):
        raise ValueError('invalid operation on closed shelf')
    __iter__ = __len__ = __getitem__ = __setitem__ = __delitem__ = keys = closed

    def __repr__(self):
        return '<Closed Dictionary>'


class Shelf(collections.abc.MutableMapping):
    """Base class for shelf implementations.

    This is initialized with a dictionary-like object.
    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, dict, protocol=None, writeback=False,
                 keyencoding="utf-8"):
        self.dict = dict
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        self._protocol = protocol
        self.writeback = writeback
        self.cache = {}
        self.keyencoding = keyencoding

    def __iter__(self):
        for k in self.dict.keys():
            yield k.decode(self.keyencoding)

    def __len__(self):
        return len(self.dict)

    def __contains__(self, key):
        return key.encode(self.keyencoding) in self.dict

    def get(self, key, default=None):
        if key.encode(self.keyencoding) in self.dict:
            return self[key]
        return default

    def __getitem__(self, key):
        try:
            value = self.cache[key]
        except KeyError:
            f = BytesIO(self.dict[key.encode(self.keyencoding)])
            value = Unpickler(f).load()
            if self.writeback:
                self.cache[key] = value
        return value

    def __setitem__(self, key, value):
        if self.writeback:
            self.cache[key] = value
        f = BytesIO()
        p = Pickler(f, self._protocol)
        p.dump(value)
        self.dict[key.encode(self.keyencoding)] = f.getvalue()

    def __delitem__(self, key):
        del self.dict[key.encode(self.keyencoding)]
        try:
            del self.cache[key]
        except KeyError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.dict is None:
            return
        try:
            self.sync()
            try:
                self.dict.close()
            except AttributeError:
                pass
        finally:
            # Catch errors that may happen when close is called from __del__
            # because CPython is in interpreter shutdown.
            try:
                self.dict = _ClosedDict()
            except:
                self.dict = None

    def __del__(self):
        if not hasattr(self, 'writeback'):
            # __init__ didn't succeed, so don't bother closing
            # see http://bugs.python.org/issue1339007 for details
            return
        self.close()

    def sync(self):
        if self.writeback and self.cache:
            self.writeback = False
            for key, entry in self.cache.items():
                self[key] = entry
            self.writeback = True
            self.cache = {}
        if hasattr(self.dict, 'sync'):
            self.dict.sync()


class BsdDbShelf(Shelf):
    """Shelf implementation using the "BSD" db interface.

    This adds methods first(), next(), previous(), last() and
    set_location() that have no counterpart in [g]dbm databases.

    The actual database must be opened using one of the "bsddb"
    modules "open" routines (i.e. bsddb.hashopen, bsddb.btopen or
    bsddb.rnopen) and passed to the constructor.

    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, dict, protocol=None, writeback=False,
                 keyencoding="utf-8"):
        Shelf.__init__(self, dict, protocol, writeback, keyencoding)

    def set_location(self, key):
        (key, value) = self.dict.set_location(key)
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def next(self):
        (key, value) = next(self.dict)
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def previous(self):
        (key, value) = self.dict.previous()
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def first(self):
        (key, value) = self.dict.first()
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())

    def last(self):
        (key, value) = self.dict.last()
        f = BytesIO(value)
        return (key.decode(self.keyencoding), Unpickler(f).load())


class DbfilenameShelf(Shelf):
    """Shelf implementation using the "dbm" generic dbm interface.

    This is initialized with the filename for the dbm database.
    See the module's __doc__ string for an overview of the interface.
    """

    def __init__(self, filename, flag='c', protocol=None, writeback=False):
        import dbm
        Shelf.__init__(self, dbm.open(filename, flag), protocol, writeback)


def open(filename, flag='c', protocol=None, writeback=False):
    """Open a persistent dictionary for reading and writing.

    The filename parameter is the base filename for the underlying
    database.  As a side-effect, an extension may be added to the
    filename and more than one file may be created.  The optional flag
    parameter has the same interpretation as the flag parameter of
    dbm.open(). The optional protocol parameter specifies the
    version of the pickle protocol.

    See the module's __doc__ string for an overview of the interface.
    """

    return DbfilenameShelf(filename, flag, protocol, writeback)
<|endoftext|>"""Text wrapping and filling.
"""

# Copyright (C) 1999-2001 Gregory P. Ward.
# Copyright (C) 2002, 2003 Python Software Foundation.
# Written by Greg Ward <gward@python.net>

import re

__all__ = ['TextWrapper', 'wrap', 'fill', 'dedent', 'indent', 'shorten']

# Hardcode the recognized whitespace characters to the US-ASCII
# whitespace characters.  The main reason for doing this is that
# some Unicode spaces (like \u00a0) are non-breaking whitespaces.
_whitespace = '\t\n\x0b\x0c\r '

class TextWrapper:
    """
    Object for wrapping/filling text.  The public interface consists of
    the wrap() and fill() methods; the other methods are just there for
    subclasses to override in order to tweak the default behaviour.
    If you want to completely replace the main wrapping algorithm,
    you'll probably have to override _wrap_chunks().

    Several instance attributes control various aspects of wrapping:
      width (default: 70)
        the maximum width of wrapped lines (unless break_long_words
        is false)
      initial_indent (default: "")
        string that will be prepended to the first line of wrapped
        output.  Counts towards the line's width.
      subsequent_indent (default: "")
        string that will be prepended to all lines save the first
        of wrapped output; also counts towards each line's width.
      expand_tabs (default: true)
        Expand tabs in input text to spaces before further processing.
        Each tab will become 0 .. 'tabsize' spaces, depending on its position
        in its line.  If false, each tab is treated as a single character.
      tabsize (default: 8)
        Expand tabs in input text to 0 .. 'tabsize' spaces, unless
        'expand_tabs' is false.
      replace_whitespace (default: true)
        Replace all whitespace characters in the input text by spaces
        after tab expansion.  Note that if expand_tabs is false and
        replace_whitespace is true, every tab will be converted to a
        single space!
      fix_sentence_endings (default: false)
        Ensure that sentence-ending punctuation is always followed
        by two spaces.  Off by default because the algorithm is
        (unavoidably) imperfect.
      break_long_words (default: true)
        Break words longer than 'width'.  If false, those words will not
        be broken, and some lines might be longer than 'width'.
      break_on_hyphens (default: true)
        Allow breaking hyphenated words. If true, wrapping will occur
        preferably on whitespaces and right after hyphens part of
        compound words.
      drop_whitespace (default: true)
        Drop leading and trailing whitespace from lines.
      max_lines (default: None)
        Truncate wrapped lines.
      placeholder (default: ' [...]')
        Append to the last line of truncated text.
    """

    unicode_whitespace_trans = {}
    uspace = ord(' ')
    for x in _whitespace:
        unicode_whitespace_trans[ord(x)] = uspace

    # This funky little regex is just the trick for splitting
    # text up into word-wrappable chunks.  E.g.
    #   "Hello there -- you goof-ball, use the -b option!"
    # splits into
    #   Hello/ /there/ /--/ /you/ /goof-/ball,/ /use/ /the/ /-b/ /option!
    # (after stripping out empty strings).
    word_punct = r'[\w!"\'&.,?]'
    letter = r'[^\d\W]'
    whitespace = r'[%s]' % re.escape(_whitespace)
    nowhitespace = '[^' + whitespace[1:]
    wordsep_re = re.compile(r'''
        ( # any whitespace
          %(ws)s+
        | # em-dash between words
          (?<=%(wp)s) -{2,} (?=\w)
        | # word, possibly hyphenated
          %(nws)s+? (?:
            # hyphenated word
              -(?: (?<=%(lt)s{2}-) | (?<=%(lt)s-%(lt)s-))
              (?= %(lt)s -? %(lt)s)
            | # end of word
              (?=%(ws)s|\Z)
            | # em-dash
              (?<=%(wp)s) (?=-{2,}\w)
            )
        )''' % {'wp': word_punct, 'lt': letter,
                'ws': whitespace, 'nws': nowhitespace},
        re.VERBOSE)
    del word_punct, letter, nowhitespace

    # This less funky little regex just split on recognized spaces. E.g.
    #   "Hello there -- you goof-ball, use the -b option!"
    # splits into
    #   Hello/ /there/ /--/ /you/ /goof-ball,/ /use/ /the/ /-b/ /option!/
    wordsep_simple_re = re.compile(r'(%s+)' % whitespace)
    del whitespace

    # XXX this is not locale- or charset-aware -- string.lowercase
    # is US-ASCII only (and therefore English-only)
    sentence_end_re = re.compile(r'[a-z]'             # lowercase letter
                                 r'[\.\!\?]'          # sentence-ending punct.
                                 r'[\"\']?'           # optional end-of-quote
                                 r'\Z')               # end of chunk

    def __init__(self,
                 width=70,
                 initial_indent="",
                 subsequent_indent="",
                 expand_tabs=True,
                 replace_whitespace=True,
                 fix_sentence_endings=False,
                 break_long_words=True,
                 drop_whitespace=True,
                 break_on_hyphens=True,
                 tabsize=8,
                 *,
                 max_lines=None,
                 placeholder=' [...]'):
        self.width = width
        self.initial_indent = initial_indent
        self.subsequent_indent = subsequent_indent
        self.expand_tabs = expand_tabs
        self.replace_whitespace = replace_whitespace
        self.fix_sentence_endings = fix_sentence_endings
        self.break_long_words = break_long_words
        self.drop_whitespace = drop_whitespace
        self.break_on_hyphens = break_on_hyphens
        self.tabsize = tabsize
        self.max_lines = max_lines
        self.placeholder = placeholder


    # -- Private methods -----------------------------------------------
    # (possibly useful for subclasses to override)

    def _munge_whitespace(self, text):
        """_munge_whitespace(text : string) -> string

        Munge whitespace in text: expand tabs and convert all other
        whitespace characters to spaces.  Eg. " foo\\tbar\\n\\nbaz"
        becomes " foo    bar  baz".
        """
        if self.expand_tabs:
            text = text.expandtabs(self.tabsize)
        if self.replace_whitespace:
            text = text.translate(self.unicode_whitespace_trans)
        return text


    def _split(self, text):
        """_split(text : string) -> [string]

        Split the text to wrap into indivisible chunks.  Chunks are
        not quite the same as words; see _wrap_chunks() for full
        details.  As an example, the text
          Look, goof-ball -- use the -b option!
        breaks into the following chunks:
          'Look,', ' ', 'goof-', 'ball', ' ', '--', ' ',
          'use', ' ', 'the', ' ', '-b', ' ', 'option!'
        if break_on_hyphens is True, or in:
          'Look,', ' ', 'goof-ball', ' ', '--', ' ',
          'use', ' ', 'the', ' ', '-b', ' ', option!'
        otherwise.
        """
        if self.break_on_hyphens is True:
            chunks = self.wordsep_re.split(text)
        else:
            chunks = self.wordsep_simple_re.split(text)
        chunks = [c for c in chunks if c]
        return chunks

    def _fix_sentence_endings(self, chunks):
        """_fix_sentence_endings(chunks : [string])

        Correct for sentence endings buried in 'chunks'.  Eg. when the
        original text contains "... foo.\\nBar ...", munge_whitespace()
        and split() will convert that to [..., "foo.", " ", "Bar", ...]
        which has one too few spaces; this method simply changes the one
        space to two.
        """
        i = 0
        patsearch = self.sentence_end_re.search
        while i < len(chunks)-1:
            if chunks[i+1] == " " and patsearch(chunks[i]):
                chunks[i+1] = "  "
                i += 2
            else:
                i += 1

    def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
        """_handle_long_word(chunks : [string],
                             cur_line : [string],
                             cur_len : int, width : int)

        Handle a chunk of text (most likely a word, not whitespace) that
        is too long to fit in any line.
        """
        # Figure out when indent is larger than the specified width, and make
        # sure at least one character is stripped off on every pass
        if width < 1:
            space_left = 1
        else:
            space_left = width - cur_len

        # If we're allowed to break long words, then do so: put as much
        # of the next chunk onto the current line as will fit.
        if self.break_long_words:
            end = space_left
            chunk = reversed_chunks[-1]
            if self.break_on_hyphens and len(chunk) > space_left:
                # break after last hyphen, but only if there are
                # non-hyphens before it
                hyphen = chunk.rfind('-', 0, space_left)
                if hyphen > 0 and any(c != '-' for c in chunk[:hyphen]):
                    end = hyphen + 1
            cur_line.append(chunk[:end])
            reversed_chunks[-1] = chunk[end:]

        # Otherwise, we have to preserve the long word intact.  Only add
        # it to the current line if there's nothing already there --
        # that minimizes how much we violate the width constraint.
        elif not cur_line:
            cur_line.append(reversed_chunks.pop())

        # If we're not allowed to break long words, and there's already
        # text on the current line, do nothing.  Next time through the
        # main loop of _wrap_chunks(), we'll wind up here again, but
        # cur_len will be zero, so the next line will be entirely
        # devoted to the long word that we can't handle right now.

    def _wrap_chunks(self, chunks):
        """_wrap_chunks(chunks : [string]) -> [string]

        Wrap a sequence of text chunks and return a list of lines of
        length 'self.width' or less.  (If 'break_long_words' is false,
        some lines may be longer than this.)  Chunks correspond roughly
        to words and the whitespace between them: each chunk is
        indivisible (modulo 'break_long_words'), but a line break can
        come between any two chunks.  Chunks should not have internal
        whitespace; ie. a chunk is either all whitespace or a "word".
        Whitespace chunks will be removed from the beginning and end of
        lines, but apart from that whitespace is preserved.
        """
        lines = []
        if self.width <= 0:
            raise ValueError("invalid width %r (must be > 0)" % self.width)
        if self.max_lines is not None:
            if self.max_lines > 1:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent
            if len(indent) + len(self.placeholder.lstrip()) > self.width:
                raise ValueError("placeholder too large for max width")

        # Arrange in reverse order so items can be efficiently popped
        # from a stack of chucks.
        chunks.reverse()

        while chunks:

            # Start the list of chunks that will make up the current line.
            # cur_len is just the length of all the chunks in cur_line.
            cur_line = []
            cur_len = 0

            # Figure out which static string will prefix this line.
            if lines:
                indent = self.subsequent_indent
            else:
                indent = self.initial_indent

            # Maximum width for this line.
            width = self.width - len(indent)

            # First chunk on line is whitespace -- drop it, unless this
            # is the very beginning of the text (ie. no lines started yet).
            if self.drop_whitespace and chunks[-1].strip() == '' and lines:
                del chunks[-1]

            while chunks:
                l = len(chunks[-1])

                # Can at least squeeze this chunk onto the current line.
                if cur_len + l <= width:
                    cur_line.append(chunks.pop())
                    cur_len += l

                # Nope, this line is full.
                else:
                    break

            # The current line is full, and the next chunk is too big to
            # fit on *any* line (not just this one).
            if chunks and len(chunks[-1]) > width:
                self._handle_long_word(chunks, cur_line, cur_len, width)
                cur_len = sum(map(len, cur_line))

            # If the last chunk on this line is all whitespace, drop it.
            if self.drop_whitespace and cur_line and cur_line[-1].strip() == '':
                cur_len -= len(cur_line[-1])
                del cur_line[-1]

            if cur_line:
                if (self.max_lines is None or
                    len(lines) + 1 < self.max_lines or
                    (not chunks or
                     self.drop_whitespace and
                     len(chunks) == 1 and
                     not chunks[0].strip()) and cur_len <= width):
                    # Convert current line back to a string and store it in
                    # list of all lines (return value).
                    lines.append(indent + ''.join(cur_line))
                else:
                    while cur_line:
                        if (cur_line[-1].strip() and
                            cur_len + len(self.placeholder) <= width):
                            cur_line.append(self.placeholder)
                            lines.append(indent + ''.join(cur_line))
                            break
                        cur_len -= len(cur_line[-1])
                        del cur_line[-1]
                    else:
                        if lines:
                            prev_line = lines[-1].rstrip()
                            if (len(prev_line) + len(self.placeholder) <=
                                    self.width):
                                lines[-1] = prev_line + self.placeholder
                                break
                        lines.append(indent + self.placeholder.lstrip())
                    break

        return lines

    def _split_chunks(self, text):
        text = self._munge_whitespace(text)
        return self._split(text)

    # -- Public interface ----------------------------------------------

    def wrap(self, text):
        """wrap(text : string) -> [string]

        Reformat the single paragraph in 'text' so it fits in lines of
        no more than 'self.width' columns, and return a list of wrapped
        lines.  Tabs in 'text' are expanded with string.expandtabs(),
        and all other whitespace characters (including newline) are
        converted to space.
        """
        chunks = self._split_chunks(text)
        if self.fix_sentence_endings:
            self._fix_sentence_endings(chunks)
        return self._wrap_chunks(chunks)

    def fill(self, text):
        """fill(text : string) -> string

        Reformat the single paragraph in 'text' to fit in lines of no
        more than 'self.width' columns, and return a new string
        containing the entire wrapped paragraph.
        """
        return "\n".join(self.wrap(text))


# -- Convenience interface ---------------------------------------------

def wrap(text, width=70, **kwargs):
    """Wrap a single paragraph of text, returning a list of wrapped lines.

    Reformat the single paragraph in 'text' so it fits in lines of no
    more than 'width' columns, and return a list of wrapped lines.  By
    default, tabs in 'text' are expanded with string.expandtabs(), and
    all other whitespace characters (including newline) are converted to
    space.  See TextWrapper class for available keyword args to customize
    wrapping behaviour.
    """
    w = TextWrapper(width=width, **kwargs)
    return w.wrap(text)

def fill(text, width=70, **kwargs):
    """Fill a single paragraph of text, returning a new string.

    Reformat the single paragraph in 'text' to fit in lines of no more
    than 'width' columns, and return a new string containing the entire
    wrapped paragraph.  As with wrap(), tabs are expanded and other
    whitespace characters converted to space.  See TextWrapper class for
    available keyword args to customize wrapping behaviour.
    """
    w = TextWrapper(width=width, **kwargs)
    return w.fill(text)

def shorten(text, width, **kwargs):
    """Collapse and truncate the given text to fit in the given width.

    The text first has its whitespace collapsed.  If it then fits in
    the *width*, it is returned as is.  Otherwise, as many words
    as possible are joined and then the placeholder is appended::

        >>> textwrap.shorten("Hello  world!", width=12)
        'Hello world!'
        >>> textwrap.shorten("Hello  world!", width=11)
        'Hello [...]'
    """
    w = TextWrapper(width=width, max_lines=1, **kwargs)
    return w.fill(' '.join(text.strip().split()))


# -- Loosely related functionality -------------------------------------

_whitespace_only_re = re.compile('^[ \t]+$', re.MULTILINE)
_leading_whitespace_re = re.compile('(^[ \t]*)(?:[^ \t\n])', re.MULTILINE)

def dedent(text):
    """Remove any common leading whitespace from every line in `text`.

    This can be used to make triple-quoted strings line up with the left
    edge of the display, while still presenting them in the source code
    in indented form.

    Note that tabs and spaces are both treated as whitespace, but they
    are not equal: the lines "  hello" and "\\thello" are
    considered to have no common leading whitespace.

    Entirely blank lines are normalized to a newline character.
    """
    # Look for the longest leading string of spaces and tabs common to
    # all lines.
    margin = None
    text = _whitespace_only_re.sub('', text)
    indents = _leading_whitespace_re.findall(text)
    for indent in indents:
        if margin is None:
            margin = indent

        # Current line more deeply indented than previous winner:
        # no change (previous winner is still on top).
        elif indent.startswith(margin):
            pass

        # Current line consistent with and no deeper than previous winner:
        # it's the new winner.
        elif margin.startswith(indent):
            margin = indent

        # Find the largest common whitespace between current line and previous
        # winner.
        else:
            for i, (x, y) in enumerate(zip(margin, indent)):
                if x != y:
                    margin = margin[:i]
                    break

    # sanity check (testing/debugging only)
    if 0 and margin:
        for line in text.split("\n"):
            assert not line or line.startswith(margin), \
                   "line = %r, margin = %r" % (line, margin)

    if margin:
        text = re.sub(r'(?m)^' + margin, '', text)
    return text


def indent(text, prefix, predicate=None):
    """Adds 'prefix' to the beginning of selected lines in 'text'.

    If 'predicate' is provided, 'prefix' will only be added to the lines
    where 'predicate(line)' is True. If 'predicate' is not provided,
    it will default to adding 'prefix' to all non-empty lines that do not
    consist solely of whitespace characters.
    """
    if predicate is None:
        def predicate(line):
            return line.strip()

    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if predicate(line) else line)
    return ''.join(prefixed_lines())


if __name__ == "__main__":
    #print dedent("\tfoo\n\tbar")
    #print dedent("  \thello there\n  \t  how are you?")
    print(dedent("Hello there.\n  This is indented."))
<|endoftext|>"""Stuff to parse AIFF-C and AIFF files.

Unless explicitly stated otherwise, the description below is true
both for AIFF-C files and AIFF files.

An AIFF-C file has the following structure.

  +-----------------+
  | FORM            |
  +-----------------+
  | <size>          |
  +----+------------+
  |    | AIFC       |
  |    +------------+
  |    | <chunks>   |
  |    |    .       |
  |    |    .       |
  |    |    .       |
  +----+------------+

An AIFF file has the string "AIFF" instead of "AIFC".

A chunk consists of an identifier (4 bytes) followed by a size (4 bytes,
big endian order), followed by the data.  The size field does not include
the size of the 8 byte header.

The following chunk types are recognized.

  FVER
      <version number of AIFF-C defining document> (AIFF-C only).
  MARK
      <# of markers> (2 bytes)
      list of markers:
          <marker ID> (2 bytes, must be > 0)
          <position> (4 bytes)
          <marker name> ("pstring")
  COMM
      <# of channels> (2 bytes)
      <# of sound frames> (4 bytes)
      <size of the samples> (2 bytes)
      <sampling frequency> (10 bytes, IEEE 80-bit extended
          floating point)
      in AIFF-C files only:
      <compression type> (4 bytes)
      <human-readable version of compression type> ("pstring")
  SSND
      <offset> (4 bytes, not used by this program)
      <blocksize> (4 bytes, not used by this program)
      <sound data>

A pstring consists of 1 byte length, a string of characters, and 0 or 1
byte pad to make the total length even.

Usage.

Reading AIFF files:
  f = aifc.open(file, 'r')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods read(), seek(), and close().
In some types of audio files, if the setpos() method is not used,
the seek() method is not necessary.

This returns an instance of a class with the following public methods:
  getnchannels()  -- returns number of audio channels (1 for
             mono, 2 for stereo)
  getsampwidth()  -- returns sample width in bytes
  getframerate()  -- returns sampling frequency
  getnframes()    -- returns number of audio frames
  getcomptype()   -- returns compression type ('NONE' for AIFF files)
  getcompname()   -- returns human-readable version of
             compression type ('not compressed' for AIFF files)
  getparams() -- returns a namedtuple consisting of all of the
             above in the above order
  getmarkers()    -- get the list of marks in the audio file or None
             if there are no marks
  getmark(id) -- get mark with the specified id (raises an error
             if the mark does not exist)
  readframes(n)   -- returns at most n frames of audio
  rewind()    -- rewind to the beginning of the audio stream
  setpos(pos) -- seek to the specified position
  tell()      -- return the current position
  close()     -- close the instance (make it unusable)
The position returned by tell(), the position given to setpos() and
the position of marks are all compatible and have nothing to do with
the actual position in the file.
The close() method is called automatically when the class instance
is destroyed.

Writing AIFF files:
  f = aifc.open(file, 'w')
where file is either the name of a file or an open file pointer.
The open file pointer must have methods write(), tell(), seek(), and
close().

This returns an instance of a class with the following public methods:
  aiff()      -- create an AIFF file (AIFF-C default)
  aifc()      -- create an AIFF-C file
  setnchannels(n) -- set the number of channels
  setsampwidth(n) -- set the sample width
  setframerate(n) -- set the frame rate
  setnframes(n)   -- set the number of frames
  setcomptype(type, name)
          -- set the compression type and the
             human-readable compression type
  setparams(tuple)
          -- set all parameters at once
  setmark(id, pos, name)
          -- add specified mark to the list of marks
  tell()      -- return current position in output file (useful
             in combination with setmark())
  writeframesraw(data)
          -- write audio frames without pathing up the
             file header
  writeframes(data)
          -- write audio frames and patch up the file header
  close()     -- patch up the file header and close the
             output file
You should set the parameters before the first writeframesraw or
writeframes.  The total number of frames does not need to be set,
but when it is set to the correct value, the header does not have to
be patched up.
It is best to first set all parameters, perhaps possibly the
compression type, and then write audio frames using writeframesraw.
When all frames have been written, either call writeframes(b'') or
close() to patch up the sizes in the header.
Marks can be added anytime.  If there are any marks, you must call
close() after all frames have been written.
The close() method is called automatically when the class instance
is destroyed.

When a file is opened with the extension '.aiff', an AIFF file is
written, otherwise an AIFF-C file is written.  This default can be
changed by calling aiff() or aifc() before the first writeframes or
writeframesraw.
"""

import struct
import builtins
import warnings

__all__ = ["Error", "open"]

class Error(Exception):
    pass

_AIFC_version = 0xA2805140     # Version 1 of AIFF-C

def _read_long(file):
    try:
        return struct.unpack('>l', file.read(4))[0]
    except struct.error:
        raise EOFError from None

def _read_ulong(file):
    try:
        return struct.unpack('>L', file.read(4))[0]
    except struct.error:
        raise EOFError from None

def _read_short(file):
    try:
        return struct.unpack('>h', file.read(2))[0]
    except struct.error:
        raise EOFError from None

def _read_ushort(file):
    try:
        return struct.unpack('>H', file.read(2))[0]
    except struct.error:
        raise EOFError from None

def _read_string(file):
    length = ord(file.read(1))
    if length == 0:
        data = b''
    else:
        data = file.read(length)
    if length & 1 == 0:
        dummy = file.read(1)
    return data

_HUGE_VAL = 1.79769313486231e+308 # See <limits.h>

def _read_float(f): # 10 bytes
    expon = _read_short(f) # 2 bytes
    sign = 1
    if expon < 0:
        sign = -1
        expon = expon + 0x8000
    himant = _read_ulong(f) # 4 bytes
    lomant = _read_ulong(f) # 4 bytes
    if expon == himant == lomant == 0:
        f = 0.0
    elif expon == 0x7FFF:
        f = _HUGE_VAL
    else:
        expon = expon - 16383
        f = (himant * 0x100000000 + lomant) * pow(2.0, expon - 63)
    return sign * f

def _write_short(f, x):
    f.write(struct.pack('>h', x))

def _write_ushort(f, x):
    f.write(struct.pack('>H', x))

def _write_long(f, x):
    f.write(struct.pack('>l', x))

def _write_ulong(f, x):
    f.write(struct.pack('>L', x))

def _write_string(f, s):
    if len(s) > 255:
        raise ValueError("string exceeds maximum pstring length")
    f.write(struct.pack('B', len(s)))
    f.write(s)
    if len(s) & 1 == 0:
        f.write(b'\x00')

def _write_float(f, x):
    import math
    if x < 0:
        sign = 0x8000
        x = x * -1
    else:
        sign = 0
    if x == 0:
        expon = 0
        himant = 0
        lomant = 0
    else:
        fmant, expon = math.frexp(x)
        if expon > 16384 or fmant >= 1 or fmant != fmant: # Infinity or NaN
            expon = sign|0x7FFF
            himant = 0
            lomant = 0
        else:                   # Finite
            expon = expon + 16382
            if expon < 0:           # denormalized
                fmant = math.ldexp(fmant, expon)
                expon = 0
            expon = expon | sign
            fmant = math.ldexp(fmant, 32)
            fsmant = math.floor(fmant)
            himant = int(fsmant)
            fmant = math.ldexp(fmant - fsmant, 32)
            fsmant = math.floor(fmant)
            lomant = int(fsmant)
    _write_ushort(f, expon)
    _write_ulong(f, himant)
    _write_ulong(f, lomant)

from chunk import Chunk
from collections import namedtuple

_aifc_params = namedtuple('_aifc_params',
                          'nchannels sampwidth framerate nframes comptype compname')

_aifc_params.nchannels.__doc__ = 'Number of audio channels (1 for mono, 2 for stereo)'
_aifc_params.sampwidth.__doc__ = 'Sample width in bytes'
_aifc_params.framerate.__doc__ = 'Sampling frequency'
_aifc_params.nframes.__doc__ = 'Number of audio frames'
_aifc_params.comptype.__doc__ = 'Compression type ("NONE" for AIFF files)'
_aifc_params.compname.__doc__ = ("""\
A human-readable version of the compression type
('not compressed' for AIFF files)""")


class Aifc_read:
    # Variables used in this class:
    #
    # These variables are available to the user though appropriate
    # methods of this class:
    # _file -- the open file with methods read(), close(), and seek()
    #       set through the __init__() method
    # _nchannels -- the number of audio channels
    #       available through the getnchannels() method
    # _nframes -- the number of audio frames
    #       available through the getnframes() method
    # _sampwidth -- the number of bytes per audio sample
    #       available through the getsampwidth() method
    # _framerate -- the sampling frequency
    #       available through the getframerate() method
    # _comptype -- the AIFF-C compression type ('NONE' if AIFF)
    #       available through the getcomptype() method
    # _compname -- the human-readable AIFF-C compression type
    #       available through the getcomptype() method
    # _markers -- the marks in the audio file
    #       available through the getmarkers() and getmark()
    #       methods
    # _soundpos -- the position in the audio stream
    #       available through the tell() method, set through the
    #       setpos() method
    #
    # These variables are used internally only:
    # _version -- the AIFF-C version number
    # _decomp -- the decompressor from builtin module cl
    # _comm_chunk_read -- 1 iff the COMM chunk has been read
    # _aifc -- 1 iff reading an AIFF-C file
    # _ssnd_seek_needed -- 1 iff positioned correctly in audio
    #       file for readframes()
    # _ssnd_chunk -- instantiation of a chunk class for the SSND chunk
    # _framesize -- size of one frame in the file

    _file = None  # Set here since __del__ checks it

    def initfp(self, file):
        self._version = 0
        self._convert = None
        self._markers = []
        self._soundpos = 0
        self._file = file
        chunk = Chunk(file)
        if chunk.getname() != b'FORM':
            raise Error('file does not start with FORM id')
        formdata = chunk.read(4)
        if formdata == b'AIFF':
            self._aifc = 0
        elif formdata == b'AIFC':
            self._aifc = 1
        else:
            raise Error('not an AIFF or AIFF-C file')
        self._comm_chunk_read = 0
        self._ssnd_chunk = None
        while 1:
            self._ssnd_seek_needed = 1
            try:
                chunk = Chunk(self._file)
            except EOFError:
                break
            chunkname = chunk.getname()
            if chunkname == b'COMM':
                self._read_comm_chunk(chunk)
                self._comm_chunk_read = 1
            elif chunkname == b'SSND':
                self._ssnd_chunk = chunk
                dummy = chunk.read(8)
                self._ssnd_seek_needed = 0
            elif chunkname == b'FVER':
                self._version = _read_ulong(chunk)
            elif chunkname == b'MARK':
                self._readmark(chunk)
            chunk.skip()
        if not self._comm_chunk_read or not self._ssnd_chunk:
            raise Error('COMM chunk and/or SSND chunk missing')

    def __init__(self, f):
        if isinstance(f, str):
            file_object = builtins.open(f, 'rb')
            try:
                self.initfp(file_object)
            except:
                file_object.close()
                raise
        else:
            # assume it is an open file object already
            self.initfp(f)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    #
    # User visible methods.
    #
    def getfp(self):
        return self._file

    def rewind(self):
        self._ssnd_seek_needed = 1
        self._soundpos = 0

    def close(self):
        file = self._file
        if file is not None:
            self._file = None
            file.close()

    def tell(self):
        return self._soundpos

    def getnchannels(self):
        return self._nchannels

    def getnframes(self):
        return self._nframes

    def getsampwidth(self):
        return self._sampwidth

    def getframerate(self):
        return self._framerate

    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        return self._compname

##  def getversion(self):
##      return self._version

    def getparams(self):
        return _aifc_params(self.getnchannels(), self.getsampwidth(),
                            self.getframerate(), self.getnframes(),
                            self.getcomptype(), self.getcompname())

    def getmarkers(self):
        if len(self._markers) == 0:
            return None
        return self._markers

    def getmark(self, id):
        for marker in self._markers:
            if id == marker[0]:
                return marker
        raise Error('marker {0!r} does not exist'.format(id))

    def setpos(self, pos):
        if pos < 0 or pos > self._nframes:
            raise Error('position not in range')
        self._soundpos = pos
        self._ssnd_seek_needed = 1

    def readframes(self, nframes):
        if self._ssnd_seek_needed:
            self._ssnd_chunk.seek(0)
            dummy = self._ssnd_chunk.read(8)
            pos = self._soundpos * self._framesize
            if pos:
                self._ssnd_chunk.seek(pos + 8)
            self._ssnd_seek_needed = 0
        if nframes == 0:
            return b''
        data = self._ssnd_chunk.read(nframes * self._framesize)
        if self._convert and data:
            data = self._convert(data)
        self._soundpos = self._soundpos + len(data) // (self._nchannels
                                                        * self._sampwidth)
        return data

    #
    # Internal methods.
    #

    def _alaw2lin(self, data):
        import audioop
        return audioop.alaw2lin(data, 2)

    def _ulaw2lin(self, data):
        import audioop
        return audioop.ulaw2lin(data, 2)

    def _adpcm2lin(self, data):
        import audioop
        if not hasattr(self, '_adpcmstate'):
            # first time
            self._adpcmstate = None
        data, self._adpcmstate = audioop.adpcm2lin(data, 2, self._adpcmstate)
        return data

    def _read_comm_chunk(self, chunk):
        self._nchannels = _read_short(chunk)
        self._nframes = _read_long(chunk)
        self._sampwidth = (_read_short(chunk) + 7) // 8
        self._framerate = int(_read_float(chunk))
        if self._sampwidth <= 0:
            raise Error('bad sample width')
        if self._nchannels <= 0:
            raise Error('bad # of channels')
        self._framesize = self._nchannels * self._sampwidth
        if self._aifc:
            #DEBUG: SGI's soundeditor produces a bad size :-(
            kludge = 0
            if chunk.chunksize == 18:
                kludge = 1
                warnings.warn('Warning: bad COMM chunk size')
                chunk.chunksize = 23
            #DEBUG end
            self._comptype = chunk.read(4)
            #DEBUG start
            if kludge:
                length = ord(chunk.file.read(1))
                if length & 1 == 0:
                    length = length + 1
                chunk.chunksize = chunk.chunksize + length
                chunk.file.seek(-1, 1)
            #DEBUG end
            self._compname = _read_string(chunk)
            if self._comptype != b'NONE':
                if self._comptype == b'G722':
                    self._convert = self._adpcm2lin
                elif self._comptype in (b'ulaw', b'ULAW'):
                    self._convert = self._ulaw2lin
                elif self._comptype in (b'alaw', b'ALAW'):
                    self._convert = self._alaw2lin
                else:
                    raise Error('unsupported compression type')
                self._sampwidth = 2
        else:
            self._comptype = b'NONE'
            self._compname = b'not compressed'

    def _readmark(self, chunk):
        nmarkers = _read_short(chunk)
        # Some files appear to contain invalid counts.
        # Cope with this by testing for EOF.
        try:
            for i in range(nmarkers):
                id = _read_short(chunk)
                pos = _read_long(chunk)
                name = _read_string(chunk)
                if pos or name:
                    # some files appear to have
                    # dummy markers consisting of
                    # a position 0 and name ''
                    self._markers.append((id, pos, name))
        except EOFError:
            w = ('Warning: MARK chunk contains only %s marker%s instead of %s' %
                 (len(self._markers), '' if len(self._markers) == 1 else 's',
                  nmarkers))
            warnings.warn(w)

class Aifc_write:
    # Variables used in this class:
    #
    # These variables are user settable through appropriate methods
    # of this class:
    # _file -- the open file with methods write(), close(), tell(), seek()
    #       set through the __init__() method
    # _comptype -- the AIFF-C compression type ('NONE' in AIFF)
    #       set through the setcomptype() or setparams() method
    # _compname -- the human-readable AIFF-C compression type
    #       set through the setcomptype() or setparams() method
    # _nchannels -- the number of audio channels
    #       set through the setnchannels() or setparams() method
    # _sampwidth -- the number of bytes per audio sample
    #       set through the setsampwidth() or setparams() method
    # _framerate -- the sampling frequency
    #       set through the setframerate() or setparams() method
    # _nframes -- the number of audio frames written to the header
    #       set through the setnframes() or setparams() method
    # _aifc -- whether we're writing an AIFF-C file or an AIFF file
    #       set through the aifc() method, reset through the
    #       aiff() method
    #
    # These variables are used internally only:
    # _version -- the AIFF-C version number
    # _comp -- the compressor from builtin module cl
    # _nframeswritten -- the number of audio frames actually written
    # _datalength -- the size of the audio samples written to the header
    # _datawritten -- the size of the audio samples actually written

    _file = None  # Set here since __del__ checks it

    def __init__(self, f):
        if isinstance(f, str):
            file_object = builtins.open(f, 'wb')
            try:
                self.initfp(file_object)
            except:
                file_object.close()
                raise

            # treat .aiff file extensions as non-compressed audio
            if f.endswith('.aiff'):
                self._aifc = 0
        else:
            # assume it is an open file object already
            self.initfp(f)

    def initfp(self, file):
        self._file = file
        self._version = _AIFC_version
        self._comptype = b'NONE'
        self._compname = b'not compressed'
        self._convert = None
        self._nchannels = 0
        self._sampwidth = 0
        self._framerate = 0
        self._nframes = 0
        self._nframeswritten = 0
        self._datawritten = 0
        self._datalength = 0
        self._markers = []
        self._marklength = 0
        self._aifc = 1      # AIFF-C is default

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    #
    # User visible methods.
    #
    def aiff(self):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._aifc = 0

    def aifc(self):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._aifc = 1

    def setnchannels(self, nchannels):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if nchannels < 1:
            raise Error('bad # of channels')
        self._nchannels = nchannels

    def getnchannels(self):
        if not self._nchannels:
            raise Error('number of channels not set')
        return self._nchannels

    def setsampwidth(self, sampwidth):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if sampwidth < 1 or sampwidth > 4:
            raise Error('bad sample width')
        self._sampwidth = sampwidth

    def getsampwidth(self):
        if not self._sampwidth:
            raise Error('sample width not set')
        return self._sampwidth

    def setframerate(self, framerate):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if framerate <= 0:
            raise Error('bad frame rate')
        self._framerate = framerate

    def getframerate(self):
        if not self._framerate:
            raise Error('frame rate not set')
        return self._framerate

    def setnframes(self, nframes):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        self._nframes = nframes

    def getnframes(self):
        return self._nframeswritten

    def setcomptype(self, comptype, compname):
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if comptype not in (b'NONE', b'ulaw', b'ULAW',
                            b'alaw', b'ALAW', b'G722'):
            raise Error('unsupported compression type')
        self._comptype = comptype
        self._compname = compname

    def getcomptype(self):
        return self._comptype

    def getcompname(self):
        return self._compname

##  def setversion(self, version):
##      if self._nframeswritten:
##          raise Error, 'cannot change parameters after starting to write'
##      self._version = version

    def setparams(self, params):
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        if self._nframeswritten:
            raise Error('cannot change parameters after starting to write')
        if comptype not in (b'NONE', b'ulaw', b'ULAW',
                            b'alaw', b'ALAW', b'G722'):
            raise Error('unsupported compression type')
        self.setnchannels(nchannels)
        self.setsampwidth(sampwidth)
        self.setframerate(framerate)
        self.setnframes(nframes)
        self.setcomptype(comptype, compname)

    def getparams(self):
        if not self._nchannels or not self._sampwidth or not self._framerate:
            raise Error('not all parameters set')
        return _aifc_params(self._nchannels, self._sampwidth, self._framerate,
                            self._nframes, self._comptype, self._compname)

    def setmark(self, id, pos, name):
        if id <= 0:
            raise Error('marker ID must be > 0')
        if pos < 0:
            raise Error('marker position must be >= 0')
        if not isinstance(name, bytes):
            raise Error('marker name must be bytes')
        for i in range(len(self._markers)):
            if id == self._markers[i][0]:
                self._markers[i] = id, pos, name
                return
        self._markers.append((id, pos, name))

    def getmark(self, id):
        for marker in self._markers:
            if id == marker[0]:
                return marker
        raise Error('marker {0!r} does not exist'.format(id))

    def getmarkers(self):
        if len(self._markers) == 0:
            return None
        return self._markers

    def tell(self):
        return self._nframeswritten

    def writeframesraw(self, data):
        if not isinstance(data, (bytes, bytearray)):
            data = memoryview(data).cast('B')
        self._ensure_header_written(len(data))
        nframes = len(data) // (self._sampwidth * self._nchannels)
        if self._convert:
            data = self._convert(data)
        self._file.write(data)
        self._nframeswritten = self._nframeswritten + nframes
        self._datawritten = self._datawritten + len(data)

    def writeframes(self, data):
        self.writeframesraw(data)
        if self._nframeswritten != self._nframes or \
              self._datalength != self._datawritten:
            self._patchheader()

    def close(self):
        if self._file is None:
            return
        try:
            self._ensure_header_written(0)
            if self._datawritten & 1:
                # quick pad to even size
                self._file.write(b'\x00')
                self._datawritten = self._datawritten + 1
            self._writemarkers()
            if self._nframeswritten != self._nframes or \
                  self._datalength != self._datawritten or \
                  self._marklength:
                self._patchheader()
        finally:
            # Prevent ref cycles
            self._convert = None
            f = self._file
            self._file = None
            f.close()

    #
    # Internal methods.
    #

    def _lin2alaw(self, data):
        import audioop
        return audioop.lin2alaw(data, 2)

    def _lin2ulaw(self, data):
        import audioop
        return audioop.lin2ulaw(data, 2)

    def _lin2adpcm(self, data):
        import audioop
        if not hasattr(self, '_adpcmstate'):
            self._adpcmstate = None
        data, self._adpcmstate = audioop.lin2adpcm(data, 2, self._adpcmstate)
        return data

    def _ensure_header_written(self, datasize):
        if not self._nframeswritten:
            if self._comptype in (b'ULAW', b'ulaw', b'ALAW', b'alaw', b'G722'):
                if not self._sampwidth:
                    self._sampwidth = 2
                if self._sampwidth != 2:
                    raise Error('sample width must be 2 when compressing '
                                'with ulaw/ULAW, alaw/ALAW or G7.22 (ADPCM)')
            if not self._nchannels:
                raise Error('# channels not specified')
            if not self._sampwidth:
                raise Error('sample width not specified')
            if not self._framerate:
                raise Error('sampling rate not specified')
            self._write_header(datasize)

    def _init_compression(self):
        if self._comptype == b'G722':
            self._convert = self._lin2adpcm
        elif self._comptype in (b'ulaw', b'ULAW'):
            self._convert = self._lin2ulaw
        elif self._comptype in (b'alaw', b'ALAW'):
            self._convert = self._lin2alaw

    def _write_header(self, initlength):
        if self._aifc and self._comptype != b'NONE':
            self._init_compression()
        self._file.write(b'FORM')
        if not self._nframes:
            self._nframes = initlength // (self._nchannels * self._sampwidth)
        self._datalength = self._nframes * self._nchannels * self._sampwidth
        if self._datalength & 1:
            self._datalength = self._datalength + 1
        if self._aifc:
            if self._comptype in (b'ulaw', b'ULAW', b'alaw', b'ALAW'):
                self._datalength = self._datalength // 2
                if self._datalength & 1:
                    self._datalength = self._datalength + 1
            elif self._comptype == b'G722':
                self._datalength = (self._datalength + 3) // 4
                if self._datalength & 1:
                    self._datalength = self._datalength + 1
        try:
            self._form_length_pos = self._file.tell()
        except (AttributeError, OSError):
            self._form_length_pos = None
        commlength = self._write_form_length(self._datalength)
        if self._aifc:
            self._file.write(b'AIFC')
            self._file.write(b'FVER')
            _write_ulong(self._file, 4)
            _write_ulong(self._file, self._version)
        else:
            self._file.write(b'AIFF')
        self._file.write(b'COMM')
        _write_ulong(self._file, commlength)
        _write_short(self._file, self._nchannels)
        if self._form_length_pos is not None:
            self._nframes_pos = self._file.tell()
        _write_ulong(self._file, self._nframes)
        if self._comptype in (b'ULAW', b'ulaw', b'ALAW', b'alaw', b'G722'):
            _write_short(self._file, 8)
        else:
            _write_short(self._file, self._sampwidth * 8)
        _write_float(self._file, self._framerate)
        if self._aifc:
            self._file.write(self._comptype)
            _write_string(self._file, self._compname)
        self._file.write(b'SSND')
        if self._form_length_pos is not None:
            self._ssnd_length_pos = self._file.tell()
        _write_ulong(self._file, self._datalength + 8)
        _write_ulong(self._file, 0)
        _write_ulong(self._file, 0)

    def _write_form_length(self, datalength):
        if self._aifc:
            commlength = 18 + 5 + len(self._compname)
            if commlength & 1:
                commlength = commlength + 1
            verslength = 12
        else:
            commlength = 18
            verslength = 0
        _write_ulong(self._file, 4 + verslength + self._marklength + \
                     8 + commlength + 16 + datalength)
        return commlength

    def _patchheader(self):
        curpos = self._file.tell()
        if self._datawritten & 1:
            datalength = self._datawritten + 1
            self._file.write(b'\x00')
        else:
            datalength = self._datawritten
        if datalength == self._datalength and \
              self._nframes == self._nframeswritten and \
              self._marklength == 0:
            self._file.seek(curpos, 0)
            return
        self._file.seek(self._form_length_pos, 0)
        dummy = self._write_form_length(datalength)
        self._file.seek(self._nframes_pos, 0)
        _write_ulong(self._file, self._nframeswritten)
        self._file.seek(self._ssnd_length_pos, 0)
        _write_ulong(self._file, datalength + 8)
        self._file.seek(curpos, 0)
        self._nframes = self._nframeswritten
        self._datalength = datalength

    def _writemarkers(self):
        if len(self._markers) == 0:
            return
        self._file.write(b'MARK')
        length = 2
        for marker in self._markers:
            id, pos, name = marker
            length = length + len(name) + 1 + 6
            if len(name) & 1 == 0:
                length = length + 1
        _write_ulong(self._file, length)
        self._marklength = length + 8
        _write_short(self._file, len(self._markers))
        for marker in self._markers:
            id, pos, name = marker
            _write_short(self._file, id)
            _write_ulong(self._file, pos)
            _write_string(self._file, name)

def open(f, mode=None):
    if mode is None:
        if hasattr(f, 'mode'):
            mode = f.mode
        else:
            mode = 'rb'
    if mode in ('r', 'rb'):
        return Aifc_read(f)
    elif mode in ('w', 'wb'):
        return Aifc_write(f)
    else:
        raise Error("mode must be 'r', 'rb', 'w', or 'wb'")


if __name__ == '__main__':
    import sys
    if not sys.argv[1:]:
        sys.argv.append('/usr/demos/data/audio/bach.aiff')
    fn = sys.argv[1]
    with open(fn, 'r') as f:
        print("Reading", fn)
        print("nchannels =", f.getnchannels())
        print("nframes   =", f.getnframes())
        print("sampwidth =", f.getsampwidth())
        print("framerate =", f.getframerate())
        print("comptype  =", f.getcomptype())
        print("compname  =", f.getcompname())
        if sys.argv[2:]:
            gn = sys.argv[2]
            print("Writing", gn)
            with open(gn, 'w') as g:
                g.setparams(f.getparams())
                while 1:
                    data = f.readframes(1024)
                    if not data:
                        break
                    g.writeframes(data)
            print("Done.")
<|endoftext|># Copyright 2007 Google Inc.
#  Licensed to PSF under a Contributor Agreement.

"""A fast, lightweight IPv4/IPv6 manipulation library in Python.

This library is used to create/poke/manipulate IPv4 and IPv6 addresses
and networks.

"""

__version__ = '1.0'


import functools

IPV4LENGTH = 32
IPV6LENGTH = 128


class AddressValueError(ValueError):
    """A Value Error related to the address."""


class NetmaskValueError(ValueError):
    """A Value Error related to the netmask."""


def ip_address(address):
    """Take an IP string/int and return an object of the correct type.

    Args:
        address: A string or integer, the IP address.  Either IPv4 or
          IPv6 addresses may be supplied; integers less than 2**32 will
          be considered to be IPv4 by default.

    Returns:
        An IPv4Address or IPv6Address object.

    Raises:
        ValueError: if the *address* passed isn't either a v4 or a v6
          address

    """
    try:
        return IPv4Address(address)
    except (AddressValueError, NetmaskValueError):
        pass

    try:
        return IPv6Address(address)
    except (AddressValueError, NetmaskValueError):
        pass

    raise ValueError(f'{address!r} does not appear to be an IPv4 or IPv6 address')


def ip_network(address, strict=True):
    """Take an IP string/int and return an object of the correct type.

    Args:
        address: A string or integer, the IP network.  Either IPv4 or
          IPv6 networks may be supplied; integers less than 2**32 will
          be considered to be IPv4 by default.

    Returns:
        An IPv4Network or IPv6Network object.

    Raises:
        ValueError: if the string passed isn't either a v4 or a v6
          address. Or if the network has host bits set.

    """
    try:
        return IPv4Network(address, strict)
    except (AddressValueError, NetmaskValueError):
        pass

    try:
        return IPv6Network(address, strict)
    except (AddressValueError, NetmaskValueError):
        pass

    raise ValueError(f'{address!r} does not appear to be an IPv4 or IPv6 network')


def ip_interface(address):
    """Take an IP string/int and return an object of the correct type.

    Args:
        address: A string or integer, the IP address.  Either IPv4 or
          IPv6 addresses may be supplied; integers less than 2**32 will
          be considered to be IPv4 by default.

    Returns:
        An IPv4Interface or IPv6Interface object.

    Raises:
        ValueError: if the string passed isn't either a v4 or a v6
          address.

    Notes:
        The IPv?Interface classes describe an Address on a particular
        Network, so they're basically a combination of both the Address
        and Network classes.

    """
    try:
        return IPv4Interface(address)
    except (AddressValueError, NetmaskValueError):
        pass

    try:
        return IPv6Interface(address)
    except (AddressValueError, NetmaskValueError):
        pass

    raise ValueError(f'{address!r} does not appear to be an IPv4 or IPv6 interface')


def v4_int_to_packed(address):
    """Represent an address as 4 packed bytes in network (big-endian) order.

    Args:
        address: An integer representation of an IPv4 IP address.

    Returns:
        The integer address packed as 4 bytes in network (big-endian) order.

    Raises:
        ValueError: If the integer is negative or too large to be an
          IPv4 IP address.

    """
    try:
        return address.to_bytes(4, 'big')
    except OverflowError:
        raise ValueError("Address negative or too large for IPv4")


def v6_int_to_packed(address):
    """Represent an address as 16 packed bytes in network (big-endian) order.

    Args:
        address: An integer representation of an IPv6 IP address.

    Returns:
        The integer address packed as 16 bytes in network (big-endian) order.

    """
    try:
        return address.to_bytes(16, 'big')
    except OverflowError:
        raise ValueError("Address negative or too large for IPv6")


def _split_optional_netmask(address):
    """Helper to split the netmask and raise AddressValueError if needed"""
    addr = str(address).split('/')
    if len(addr) > 2:
        raise AddressValueError(f"Only one '/' permitted in {address!r}")
    return addr


def _find_address_range(addresses):
    """Find a sequence of sorted deduplicated IPv#Address.

    Args:
        addresses: a list of IPv#Address objects.

    Yields:
        A tuple containing the first and last IP addresses in the sequence.

    """
    it = iter(addresses)
    first = last = next(it)
    for ip in it:
        if ip._ip != last._ip + 1:
            yield first, last
            first = ip
        last = ip
    yield first, last


def _count_righthand_zero_bits(number, bits):
    """Count the number of zero bits on the right hand side.

    Args:
        number: an integer.
        bits: maximum number of bits to count.

    Returns:
        The number of zero bits on the right hand side of the number.

    """
    if number == 0:
        return bits
    return min(bits, (~number & (number-1)).bit_length())


def summarize_address_range(first, last):
    """Summarize a network range given the first and last IP addresses.

    Example:
        >>> list(summarize_address_range(IPv4Address('192.0.2.0'),
        ...                              IPv4Address('192.0.2.130')))
        ...                                #doctest: +NORMALIZE_WHITESPACE
        [IPv4Network('192.0.2.0/25'), IPv4Network('192.0.2.128/31'),
         IPv4Network('192.0.2.130/32')]

    Args:
        first: the first IPv4Address or IPv6Address in the range.
        last: the last IPv4Address or IPv6Address in the range.

    Returns:
        An iterator of the summarized IPv(4|6) network objects.

    Raise:
        TypeError:
            If the first and last objects are not IP addresses.
            If the first and last objects are not the same version.
        ValueError:
            If the last object is not greater than the first.
            If the version of the first address is not 4 or 6.

    """
    if (not (isinstance(first, _BaseAddress) and
             isinstance(last, _BaseAddress))):
        raise TypeError('first and last must be IP addresses, not networks')
    if first.version != last.version:
        raise TypeError("%s and %s are not of the same version" % (
                         first, last))
    if first > last:
        raise ValueError('last IP address must be greater than first')

    if first.version == 4:
        ip = IPv4Network
    elif first.version == 6:
        ip = IPv6Network
    else:
        raise ValueError('unknown IP version')

    ip_bits = first._max_prefixlen
    first_int = first._ip
    last_int = last._ip
    while first_int <= last_int:
        nbits = min(_count_righthand_zero_bits(first_int, ip_bits),
                    (last_int - first_int + 1).bit_length() - 1)
        net = ip((first_int, ip_bits - nbits))
        yield net
        first_int += 1 << nbits
        if first_int - 1 == ip._ALL_ONES:
            break


def _collapse_addresses_internal(addresses):
    """Loops through the addresses, collapsing concurrent netblocks.

    Example:

        ip1 = IPv4Network('192.0.2.0/26')
        ip2 = IPv4Network('192.0.2.64/26')
        ip3 = IPv4Network('192.0.2.128/26')
        ip4 = IPv4Network('192.0.2.192/26')

        _collapse_addresses_internal([ip1, ip2, ip3, ip4]) ->
          [IPv4Network('192.0.2.0/24')]

        This shouldn't be called directly; it is called via
          collapse_addresses([]).

    Args:
        addresses: A list of IPv4Network's or IPv6Network's

    Returns:
        A list of IPv4Network's or IPv6Network's depending on what we were
        passed.

    """
    # First merge
    to_merge = list(addresses)
    subnets = {}
    while to_merge:
        net = to_merge.pop()
        supernet = net.supernet()
        existing = subnets.get(supernet)
        if existing is None:
            subnets[supernet] = net
        elif existing != net:
            # Merge consecutive subnets
            del subnets[supernet]
            to_merge.append(supernet)
    # Then iterate over resulting networks, skipping subsumed subnets
    last = None
    for net in sorted(subnets.values()):
        if last is not None:
            # Since they are sorted, last.network_address <= net.network_address
            # is a given.
            if last.broadcast_address >= net.broadcast_address:
                continue
        yield net
        last = net


def collapse_addresses(addresses):
    """Collapse a list of IP objects.

    Example:
        collapse_addresses([IPv4Network('192.0.2.0/25'),
                            IPv4Network('192.0.2.128/25')]) ->
                           [IPv4Network('192.0.2.0/24')]

    Args:
        addresses: An iterator of IPv4Network or IPv6Network objects.

    Returns:
        An iterator of the collapsed IPv(4|6)Network objects.

    Raises:
        TypeError: If passed a list of mixed version objects.

    """
    addrs = []
    ips = []
    nets = []

    # split IP addresses and networks
    for ip in addresses:
        if isinstance(ip, _BaseAddress):
            if ips and ips[-1]._version != ip._version:
                raise TypeError("%s and %s are not of the same version" % (
                                 ip, ips[-1]))
            ips.append(ip)
        elif ip._prefixlen == ip._max_prefixlen:
            if ips and ips[-1]._version != ip._version:
                raise TypeError("%s and %s are not of the same version" % (
                                 ip, ips[-1]))
            try:
                ips.append(ip.ip)
            except AttributeError:
                ips.append(ip.network_address)
        else:
            if nets and nets[-1]._version != ip._version:
                raise TypeError("%s and %s are not of the same version" % (
                                 ip, nets[-1]))
            nets.append(ip)

    # sort and dedup
    ips = sorted(set(ips))

    # find consecutive address ranges in the sorted sequence and summarize them
    if ips:
        for first, last in _find_address_range(ips):
            addrs.extend(summarize_address_range(first, last))

    return _collapse_addresses_internal(addrs + nets)


def get_mixed_type_key(obj):
    """Return a key suitable for sorting between networks and addresses.

    Address and Network objects are not sortable by default; they're
    fundamentally different so the expression

        IPv4Address('192.0.2.0') <= IPv4Network('192.0.2.0/24')

    doesn't make any sense.  There are some times however, where you may wish
    to have ipaddress sort these for you anyway. If you need to do this, you
    can use this function as the key= argument to sorted().

    Args:
      obj: either a Network or Address object.
    Returns:
      appropriate key.

    """
    if isinstance(obj, _BaseNetwork):
        return obj._get_networks_key()
    elif isinstance(obj, _BaseAddress):
        return obj._get_address_key()
    return NotImplemented


class _IPAddressBase:

    """The mother class."""

    __slots__ = ()

    @property
    def exploded(self):
        """Return the longhand version of the IP address as a string."""
        return self._explode_shorthand_ip_string()

    @property
    def compressed(self):
        """Return the shorthand version of the IP address as a string."""
        return str(self)

    @property
    def reverse_pointer(self):
        """The name of the reverse DNS pointer for the IP address, e.g.:
            >>> ipaddress.ip_address("127.0.0.1").reverse_pointer
            '1.0.0.127.in-addr.arpa'
            >>> ipaddress.ip_address("2001:db8::1").reverse_pointer
            '1.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa'

        """
        return self._reverse_pointer()

    @property
    def version(self):
        msg = '%200s has no version specified' % (type(self),)
        raise NotImplementedError(msg)

    def _check_int_address(self, address):
        if address < 0:
            msg = "%d (< 0) is not permitted as an IPv%d address"
            raise AddressValueError(msg % (address, self._version))
        if address > self._ALL_ONES:
            msg = "%d (>= 2**%d) is not permitted as an IPv%d address"
            raise AddressValueError(msg % (address, self._max_prefixlen,
                                           self._version))

    def _check_packed_address(self, address, expected_len):
        address_len = len(address)
        if address_len != expected_len:
            msg = "%r (len %d != %d) is not permitted as an IPv%d address"
            raise AddressValueError(msg % (address, address_len,
                                           expected_len, self._version))

    @classmethod
    def _ip_int_from_prefix(cls, prefixlen):
        """Turn the prefix length into a bitwise netmask

        Args:
            prefixlen: An integer, the prefix length.

        Returns:
            An integer.

        """
        return cls._ALL_ONES ^ (cls._ALL_ONES >> prefixlen)

    @classmethod
    def _prefix_from_ip_int(cls, ip_int):
        """Return prefix length from the bitwise netmask.

        Args:
            ip_int: An integer, the netmask in expanded bitwise format

        Returns:
            An integer, the prefix length.

        Raises:
            ValueError: If the input intermingles zeroes & ones
        """
        trailing_zeroes = _count_righthand_zero_bits(ip_int,
                                                     cls._max_prefixlen)
        prefixlen = cls._max_prefixlen - trailing_zeroes
        leading_ones = ip_int >> trailing_zeroes
        all_ones = (1 << prefixlen) - 1
        if leading_ones != all_ones:
            byteslen = cls._max_prefixlen // 8
            details = ip_int.to_bytes(byteslen, 'big')
            msg = 'Netmask pattern %r mixes zeroes & ones'
            raise ValueError(msg % details)
        return prefixlen

    @classmethod
    def _report_invalid_netmask(cls, netmask_str):
        msg = '%r is not a valid netmask' % netmask_str
        raise NetmaskValueError(msg) from None

    @classmethod
    def _prefix_from_prefix_string(cls, prefixlen_str):
        """Return prefix length from a numeric string

        Args:
            prefixlen_str: The string to be converted

        Returns:
            An integer, the prefix length.

        Raises:
            NetmaskValueError: If the input is not a valid netmask
        """
        # int allows a leading +/- as well as surrounding whitespace,
        # so we ensure that isn't the case
        if not (prefixlen_str.isascii() and prefixlen_str.isdigit()):
            cls._report_invalid_netmask(prefixlen_str)
        try:
            prefixlen = int(prefixlen_str)
        except ValueError:
            cls._report_invalid_netmask(prefixlen_str)
        if not (0 <= prefixlen <= cls._max_prefixlen):
            cls._report_invalid_netmask(prefixlen_str)
        return prefixlen

    @classmethod
    def _prefix_from_ip_string(cls, ip_str):
        """Turn a netmask/hostmask string into a prefix length

        Args:
            ip_str: The netmask/hostmask to be converted

        Returns:
            An integer, the prefix length.

        Raises:
            NetmaskValueError: If the input is not a valid netmask/hostmask
        """
        # Parse the netmask/hostmask like an IP address.
        try:
            ip_int = cls._ip_int_from_string(ip_str)
        except AddressValueError:
            cls._report_invalid_netmask(ip_str)

        # Try matching a netmask (this would be /1*0*/ as a bitwise regexp).
        # Note that the two ambiguous cases (all-ones and all-zeroes) are
        # treated as netmasks.
        try:
            return cls._prefix_from_ip_int(ip_int)
        except ValueError:
            pass

        # Invert the bits, and try matching a /0+1+/ hostmask instead.
        ip_int ^= cls._ALL_ONES
        try:
            return cls._prefix_from_ip_int(ip_int)
        except ValueError:
            cls._report_invalid_netmask(ip_str)

    @classmethod
    def _split_addr_prefix(cls, address):
        """Helper function to parse address of Network/Interface.

        Arg:
            address: Argument of Network/Interface.

        Returns:
            (addr, prefix) tuple.
        """
        # a packed address or integer
        if isinstance(address, (bytes, int)):
            return address, cls._max_prefixlen

        if not isinstance(address, tuple):
            # Assume input argument to be string or any object representation
            # which converts into a formatted IP prefix string.
            address = _split_optional_netmask(address)

        # Constructing from a tuple (addr, [mask])
        if len(address) > 1:
            return address
        return address[0], cls._max_prefixlen

    def __reduce__(self):
        return self.__class__, (str(self),)


_address_fmt_re = None

@functools.total_ordering
class _BaseAddress(_IPAddressBase):

    """A generic IP object.

    This IP class contains the version independent methods which are
    used by single IP addresses.
    """

    __slots__ = ()

    def __int__(self):
        return self._ip

    def __eq__(self, other):
        try:
            return (self._ip == other._ip
                    and self._version == other._version)
        except AttributeError:
            return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, _BaseAddress):
            return NotImplemented
        if self._version != other._version:
            raise TypeError('%s and %s are not of the same version' % (
                             self, other))
        if self._ip != other._ip:
            return self._ip < other._ip
        return False

    # Shorthand for Integer addition and subtraction. This is not
    # meant to ever support addition/subtraction of addresses.
    def __add__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return self.__class__(int(self) + other)

    def __sub__(self, other):
        if not isinstance(other, int):
            return NotImplemented
        return self.__class__(int(self) - other)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, str(self))

    def __str__(self):
        return str(self._string_from_ip_int(self._ip))

    def __hash__(self):
        return hash(hex(int(self._ip)))

    def _get_address_key(self):
        return (self._version, self)

    def __reduce__(self):
        return self.__class__, (self._ip,)

    def __format__(self, fmt):
        """Returns an IP address as a formatted string.

        Supported presentation types are:
        's': returns the IP address as a string (default)
        'b': converts to binary and returns a zero-padded string
        'X' or 'x': converts to upper- or lower-case hex and returns a zero-padded string
        'n': the same as 'b' for IPv4 and 'x' for IPv6

        For binary and hex presentation types, the alternate form specifier
        '#' and the grouping option '_' are supported.
        """

        # Support string formatting
        if not fmt or fmt[-1] == 's':
            return format(str(self), fmt)

        # From here on down, support for 'bnXx'
        global _address_fmt_re
        if _address_fmt_re is None:
            import re
            _address_fmt_re = re.compile('(#?)(_?)([xbnX])')

        m = _address_fmt_re.fullmatch(fmt)
        if not m:
            return super().__format__(fmt)

        alternate, grouping, fmt_base = m.groups()

        # Set some defaults
        if fmt_base == 'n':
            if self._version == 4:
                fmt_base = 'b'  # Binary is default for ipv4
            else:
                fmt_base = 'x'  # Hex is default for ipv6

        if fmt_base == 'b':
            padlen = self._max_prefixlen
        else:
            padlen = self._max_prefixlen // 4

        if grouping:
            padlen += padlen // 4 - 1

        if alternate:
            padlen += 2  # 0b or 0x

        return format(int(self), f'{alternate}0{padlen}{grouping}{fmt_base}')


@functools.total_ordering
class _BaseNetwork(_IPAddressBase):
    """A generic IP network object.

    This IP class contains the version independent methods which are
    used by networks.
    """

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, str(self))

    def __str__(self):
        return '%s/%d' % (self.network_address, self.prefixlen)

    def hosts(self):
        """Generate Iterator over usable hosts in a network.

        This is like __iter__ except it doesn't return the network
        or broadcast addresses.

        """
        network = int(self.network_address)
        broadcast = int(self.broadcast_address)
        for x in range(network + 1, broadcast):
            yield self._address_class(x)

    def __iter__(self):
        network = int(self.network_address)
        broadcast = int(self.broadcast_address)
        for x in range(network, broadcast + 1):
            yield self._address_class(x)

    def __getitem__(self, n):
        network = int(self.network_address)
        broadcast = int(self.broadcast_address)
        if n >= 0:
            if network + n > broadcast:
                raise IndexError('address out of range')
            return self._address_class(network + n)
        else:
            n += 1
            if broadcast + n < network:
                raise IndexError('address out of range')
            return self._address_class(broadcast + n)

    def __lt__(self, other):
        if not isinstance(other, _BaseNetwork):
            return NotImplemented
        if self._version != other._version:
            raise TypeError('%s and %s are not of the same version' % (
                             self, other))
        if self.network_address != other.network_address:
            return self.network_address < other.network_address
        if self.netmask != other.netmask:
            return self.netmask < other.netmask
        return False

    def __eq__(self, other):
        try:
            return (self._version == other._version and
                    self.network_address == other.network_address and
                    int(self.netmask) == int(other.netmask))
        except AttributeError:
            return NotImplemented

    def __hash__(self):
        return hash(int(self.network_address) ^ int(self.netmask))

    def __contains__(self, other):
        # always false if one is v4 and the other is v6.
        if self._version != other._version:
            return False
        # dealing with another network.
        if isinstance(other, _BaseNetwork):
            return False
        # dealing with another address
        else:
            # address
            return other._ip & self.netmask._ip == self.network_address._ip

    def overlaps(self, other):
        """Tell if self is partly contained in other."""
        return self.network_address in other or (
            self.broadcast_address in other or (
                other.network_address in self or (
                    other.broadcast_address in self)))

    @functools.cached_property
    def broadcast_address(self):
        return self._address_class(int(self.network_address) |
                                   int(self.hostmask))

    @functools.cached_property
    def hostmask(self):
        return self._address_class(int(self.netmask) ^ self._ALL_ONES)

    @property
    def with_prefixlen(self):
        return '%s/%d' % (self.network_address, self._prefixlen)

    @property
    def with_netmask(self):
        return '%s/%s' % (self.network_address, self.netmask)

    @property
    def with_hostmask(self):
        return '%s/%s' % (self.network_address, self.hostmask)

    @property
    def num_addresses(self):
        """Number of hosts in the current subnet."""
        return int(self.broadcast_address) - int(self.network_address) + 1

    @property
    def _address_class(self):
        # Returning bare address objects (rather than interfaces) allows for
        # more consistent behaviour across the network address, broadcast
        # address and individual host addresses.
        msg = '%200s has no associated address class' % (type(self),)
        raise NotImplementedError(msg)

    @property
    def prefixlen(self):
        return self._prefixlen

    def address_exclude(self, other):
        """Remove an address from a larger block.

        For example:

            addr1 = ip_network('192.0.2.0/28')
            addr2 = ip_network('192.0.2.1/32')
            list(addr1.address_exclude(addr2)) =
                [IPv4Network('192.0.2.0/32'), IPv4Network('192.0.2.2/31'),
                 IPv4Network('192.0.2.4/30'), IPv4Network('192.0.2.8/29')]

        or IPv6:

            addr1 = ip_network('2001:db8::1/32')
            addr2 = ip_network('2001:db8::1/128')
            list(addr1.address_exclude(addr2)) =
                [ip_network('2001:db8::1/128'),
                 ip_network('2001:db8::2/127'),
                 ip_network('2001:db8::4/126'),
                 ip_network('2001:db8::8/125'),
                 ...
                 ip_network('2001:db8:8000::/33')]

        Args:
            other: An IPv4Network or IPv6Network object of the same type.

        Returns:
            An iterator of the IPv(4|6)Network objects which is self
            minus other.

        Raises:
            TypeError: If self and other are of differing address
              versions, or if other is not a network object.
            ValueError: If other is not completely contained by self.

        """
        if not self._version == other._version:
            raise TypeError("%s and %s are not of the same version" % (
                             self, other))

        if not isinstance(other, _BaseNetwork):
            raise TypeError("%s is not a network object" % other)

        if not other.subnet_of(self):
            raise ValueError('%s not contained in %s' % (other, self))
        if other == self:
            return

        # Make sure we're comparing the network of other.
        other = other.__class__('%s/%s' % (other.network_address,
                                           other.prefixlen))

        s1, s2 = self.subnets()
        while s1 != other and s2 != other:
            if other.subnet_of(s1):
                yield s2
                s1, s2 = s1.subnets()
            elif other.subnet_of(s2):
                yield s1
                s1, s2 = s2.subnets()
            else:
                # If we got here, there's a bug somewhere.
                raise AssertionError('Error performing exclusion: '
                                     's1: %s s2: %s other: %s' %
                                     (s1, s2, other))
        if s1 == other:
            yield s2
        elif s2 == other:
            yield s1
        else:
            # If we got here, there's a bug somewhere.
            raise AssertionError('Error performing exclusion: '
                                 's1: %s s2: %s other: %s' %
                                 (s1, s2, other))

    def compare_networks(self, other):
        """Compare two IP objects.

        This is only concerned about the comparison of the integer
        representation of the network addresses.  This means that the
        host bits aren't considered at all in this method.  If you want
        to compare host bits, you can easily enough do a
        'HostA._ip < HostB._ip'

        Args:
            other: An IP object.

        Returns:
            If the IP versions of self and other are the same, returns:

            -1 if self < other:
              eg: IPv4Network('192.0.2.0/25') < IPv4Network('192.0.2.128/25')
              IPv6Network('2001:db8::1000/124') <
                  IPv6Network('2001:db8::2000/124')
            0 if self == other
              eg: IPv4Network('192.0.2.0/24') == IPv4Network('192.0.2.0/24')
              IPv6Network('2001:db8::1000/124') ==
                  IPv6Network('2001:db8::1000/124')
            1 if self > other
              eg: IPv4Network('192.0.2.128/25') > IPv4Network('192.0.2.0/25')
                  IPv6Network('2001:db8::2000/124') >
                      IPv6Network('2001:db8::1000/124')

          Raises:
              TypeError if the IP versions are different.

        """
        # does this need to raise a ValueError?
        if self._version != other._version:
            raise TypeError('%s and %s are not of the same type' % (
                             self, other))
        # self._version == other._version below here:
        if self.network_address < other.network_address:
            return -1
        if self.network_address > other.network_address:
            return 1
        # self.network_address == other.network_address below here:
        if self.netmask < other.netmask:
            return -1
        if self.netmask > other.netmask:
            return 1
        return 0

    def _get_networks_key(self):
        """Network-only key function.

        Returns an object that identifies this address' network and
        netmask. This function is a suitable "key" argument for sorted()
        and list.sort().

        """
        return (self._version, self.network_address, self.netmask)

    def subnets(self, prefixlen_diff=1, new_prefix=None):
        """The subnets which join to make the current subnet.

        In the case that self contains only one IP
        (self._prefixlen == 32 for IPv4 or self._prefixlen == 128
        for IPv6), yield an iterator with just ourself.

        Args:
            prefixlen_diff: An integer, the amount the prefix length
              should be increased by. This should not be set if
              new_prefix is also set.
            new_prefix: The desired new prefix length. This must be a
              larger number (smaller prefix) than the existing prefix.
              This should not be set if prefixlen_diff is also set.

        Returns:
            An iterator of IPv(4|6) objects.

        Raises:
            ValueError: The prefixlen_diff is too small or too large.
                OR
            prefixlen_diff and new_prefix are both set or new_prefix
              is a smaller number than the current prefix (smaller
              number means a larger network)

        """
        if self._prefixlen == self._max_prefixlen:
            yield self
            return

        if new_prefix is not None:
            if new_prefix < self._prefixlen:
                raise ValueError('new prefix must be longer')
            if prefixlen_diff != 1:
                raise ValueError('cannot set prefixlen_diff and new_prefix')
            prefixlen_diff = new_prefix - self._prefixlen

        if prefixlen_diff < 0:
            raise ValueError('prefix length diff must be > 0')
        new_prefixlen = self._prefixlen + prefixlen_diff

        if new_prefixlen > self._max_prefixlen:
            raise ValueError(
                'prefix length diff %d is invalid for netblock %s' % (
                    new_prefixlen, self))

        start = int(self.network_address)
        end = int(self.broadcast_address) + 1
        step = (int(self.hostmask) + 1) >> prefixlen_diff
        for new_addr in range(start, end, step):
            current = self.__class__((new_addr, new_prefixlen))
            yield current

    def supernet(self, prefixlen_diff=1, new_prefix=None):
        """The supernet containing the current network.

        Args:
            prefixlen_diff: An integer, the amount the prefix length of
              the network should be decreased by.  For example, given a
              /24 network and a prefixlen_diff of 3, a supernet with a
              /21 netmask is returned.

        Returns:
            An IPv4 network object.

        Raises:
            ValueError: If self.prefixlen - prefixlen_diff < 0. I.e., you have
              a negative prefix length.
                OR
            If prefixlen_diff and new_prefix are both set or new_prefix is a
              larger number than the current prefix (larger number means a
              smaller network)

        """
        if self._prefixlen == 0:
            return self

        if new_prefix is not None:
            if new_prefix > self._prefixlen:
                raise ValueError('new prefix must be shorter')
            if prefixlen_diff != 1:
                raise ValueError('cannot set prefixlen_diff and new_prefix')
            prefixlen_diff = self._prefixlen - new_prefix

        new_prefixlen = self.prefixlen - prefixlen_diff
        if new_prefixlen < 0:
            raise ValueError(
                'current prefixlen is %d, cannot have a prefixlen_diff of %d' %
                (self.prefixlen, prefixlen_diff))
        return self.__class__((
            int(self.network_address) & (int(self.netmask) << prefixlen_diff),
            new_prefixlen
            ))

    @property
    def is_multicast(self):
        """Test if the address is reserved for multicast use.

        Returns:
            A boolean, True if the address is a multicast address.
            See RFC 2373 2.7 for details.

        """
        return (self.network_address.is_multicast and
                self.broadcast_address.is_multicast)

    @staticmethod
    def _is_subnet_of(a, b):
        try:
            # Always false if one is v4 and the other is v6.
            if a._version != b._version:
                raise TypeError(f"{a} and {b} are not of the same version")
            return (b.network_address <= a.network_address and
                    b.broadcast_address >= a.broadcast_address)
        except AttributeError:
            raise TypeError(f"Unable to test subnet containment "
                            f"between {a} and {b}")

    def subnet_of(self, other):
        """Return True if this network is a subnet of other."""
        return self._is_subnet_of(self, other)

    def supernet_of(self, other):
        """Return True if this network is a supernet of other."""
        return self._is_subnet_of(other, self)

    @property
    def is_reserved(self):
        """Test if the address is otherwise IETF reserved.

        Returns:
            A boolean, True if the address is within one of the
            reserved IPv6 Network ranges.

        """
        return (self.network_address.is_reserved and
                self.broadcast_address.is_reserved)

    @property
    def is_link_local(self):
        """Test if the address is reserved for link-local.

        Returns:
            A boolean, True if the address is reserved per RFC 4291.

        """
        return (self.network_address.is_link_local and
                self.broadcast_address.is_link_local)

    @property
    def is_private(self):
        """Test if this address is allocated for private networks.

        Returns:
            A boolean, True if the address is reserved per
            iana-ipv4-special-registry or iana-ipv6-special-registry.

        """
        return (self.network_address.is_private and
                self.broadcast_address.is_private)

    @property
    def is_global(self):
        """Test if this address is allocated for public networks.

        Returns:
            A boolean, True if the address is not reserved per
            iana-ipv4-special-registry or iana-ipv6-special-registry.

        """
        return not self.is_private

    @property
    def is_unspecified(self):
        """Test if the address is unspecified.

        Returns:
            A boolean, True if this is the unspecified address as defined in
            RFC 2373 2.5.2.

        """
        return (self.network_address.is_unspecified and
                self.broadcast_address.is_unspecified)

    @property
    def is_loopback(self):
        """Test if the address is a loopback address.

        Returns:
            A boolean, True if the address is a loopback address as defined in
            RFC 2373 2.5.3.

        """
        return (self.network_address.is_loopback and
                self.broadcast_address.is_loopback)

class _BaseV4:

    """Base IPv4 object.

    The following methods are used by IPv4 objects in both single IP
    addresses and networks.

    """

    __slots__ = ()
    _version = 4
    # Equivalent to 255.255.255.255 or 32 bits of 1's.
    _ALL_ONES = (2**IPV4LENGTH) - 1

    _max_prefixlen = IPV4LENGTH
    # There are only a handful of valid v4 netmasks, so we cache them all
    # when constructed (see _make_netmask()).
    _netmask_cache = {}

    def _explode_shorthand_ip_string(self):
        return str(self)

    @classmethod
    def _make_netmask(cls, arg):
        """Make a (netmask, prefix_len) tuple from the given argument.

        Argument can be:
        - an integer (the prefix length)
        - a string representing the prefix length (e.g. "24")
        - a string representing the prefix netmask (e.g. "255.255.255.0")
        """
        if arg not in cls._netmask_cache:
            if isinstance(arg, int):
                prefixlen = arg
                if not (0 <= prefixlen <= cls._max_prefixlen):
                    cls._report_invalid_netmask(prefixlen)
            else:
                try:
                    # Check for a netmask in prefix length form
                    prefixlen = cls._prefix_from_prefix_string(arg)
                except NetmaskValueError:
                    # Check for a netmask or hostmask in dotted-quad form.
                    # This may raise NetmaskValueError.
                    prefixlen = cls._prefix_from_ip_string(arg)
            netmask = IPv4Address(cls._ip_int_from_prefix(prefixlen))
