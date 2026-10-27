        descr = compressor_names.get(compress_type)
        if descr:
            raise NotImplementedError("compression type %d (%s)" % (compress_type, descr))
        else:
            raise NotImplementedError("compression type %d" % (compress_type,))


class _SharedFile:
    def __init__(self, file, pos, close, lock, writing):
        self._file = file
        self._pos = pos
        self._close = close
        self._lock = lock
        self._writing = writing
        self.seekable = file.seekable

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        with self._lock:
            if self._writing():
                raise ValueError("Can't reposition in the ZIP file while "
                        "there is an open writing handle on it. "
                        "Close the writing handle before trying to read.")
            self._file.seek(offset, whence)
            self._pos = self._file.tell()
            return self._pos

    def read(self, n=-1):
        with self._lock:
            if self._writing():
                raise ValueError("Can't read from the ZIP file while there "
                        "is an open writing handle on it. "
                        "Close the writing handle before trying to read.")
            self._file.seek(self._pos)
            data = self._file.read(n)
            self._pos = self._file.tell()
            return data

    def close(self):
        if self._file is not None:
            fileobj = self._file
            self._file = None
            self._close(fileobj)

# Provide the tell method for unseekable stream
class _Tellable:
    def __init__(self, fp):
        self.fp = fp
        self.offset = 0

    def write(self, data):
        n = self.fp.write(data)
        self.offset += n
        return n

    def tell(self):
        return self.offset

    def flush(self):
        self.fp.flush()

    def close(self):
        self.fp.close()


class ZipExtFile(io.BufferedIOBase):
    """File-like object for reading an archive member.
       Is returned by ZipFile.open().
    """

    # Max size supported by decompressor.
    MAX_N = 1 << 31 - 1

    # Read from compressed files in 4k blocks.
    MIN_READ_SIZE = 4096

    # Chunk size to read during seek
    MAX_SEEK_READ = 1 << 24

    def __init__(self, fileobj, mode, zipinfo, pwd=None,
                 close_fileobj=False):
        self._fileobj = fileobj
        self._pwd = pwd
        self._close_fileobj = close_fileobj

        self._compress_type = zipinfo.compress_type
        self._compress_left = zipinfo.compress_size
        self._left = zipinfo.file_size

        self._decompressor = _get_decompressor(self._compress_type)

        self._eof = False
        self._readbuffer = b''
        self._offset = 0

        self.newlines = None

        self.mode = mode
        self.name = zipinfo.filename

        if hasattr(zipinfo, 'CRC'):
            self._expected_crc = zipinfo.CRC
            self._running_crc = crc32(b'')
        else:
            self._expected_crc = None

        self._seekable = False
        try:
            if fileobj.seekable():
                self._orig_compress_start = fileobj.tell()
                self._orig_compress_size = zipinfo.compress_size
                self._orig_file_size = zipinfo.file_size
                self._orig_start_crc = self._running_crc
                self._seekable = True
        except AttributeError:
            pass

        self._decrypter = None
        if pwd:
            if zipinfo.flag_bits & 0x8:
                # compare against the file type from extended local headers
                check_byte = (zipinfo._raw_time >> 8) & 0xff
            else:
                # compare against the CRC otherwise
                check_byte = (zipinfo.CRC >> 24) & 0xff
            h = self._init_decrypter()
            if h != check_byte:
                raise RuntimeError("Bad password for file %r" % zipinfo.orig_filename)


    def _init_decrypter(self):
        self._decrypter = _ZipDecrypter(self._pwd)
        # The first 12 bytes in the cypher stream is an encryption header
        #  used to strengthen the algorithm. The first 11 bytes are
        #  completely random, while the 12th contains the MSB of the CRC,
        #  or the MSB of the file time depending on the header type
        #  and is used to check the correctness of the password.
        header = self._fileobj.read(12)
        self._compress_left -= 12
        return self._decrypter(header)[11]

    def __repr__(self):
        result = ['<%s.%s' % (self.__class__.__module__,
                              self.__class__.__qualname__)]
        if not self.closed:
            result.append(' name=%r mode=%r' % (self.name, self.mode))
            if self._compress_type != ZIP_STORED:
                result.append(' compress_type=%s' %
                              compressor_names.get(self._compress_type,
                                                   self._compress_type))
        else:
            result.append(' [closed]')
        result.append('>')
        return ''.join(result)

    def readline(self, limit=-1):
        """Read and return a line from the stream.

        If limit is specified, at most limit bytes will be read.
        """

        if limit < 0:
            # Shortcut common case - newline found in buffer.
            i = self._readbuffer.find(b'\n', self._offset) + 1
            if i > 0:
                line = self._readbuffer[self._offset: i]
                self._offset = i
                return line

        return io.BufferedIOBase.readline(self, limit)

    def peek(self, n=1):
        """Returns buffered bytes without advancing the position."""
        if n > len(self._readbuffer) - self._offset:
            chunk = self.read(n)
            if len(chunk) > self._offset:
                self._readbuffer = chunk + self._readbuffer[self._offset:]
                self._offset = 0
            else:
                self._offset -= len(chunk)

        # Return up to 512 bytes to reduce allocation overhead for tight loops.
        return self._readbuffer[self._offset: self._offset + 512]

    def readable(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return True

    def read(self, n=-1):
        """Read and return up to n bytes.
        If the argument is omitted, None, or negative, data is read and returned until EOF is reached.
        """
        if self.closed:
            raise ValueError("read from closed file.")
        if n is None or n < 0:
            buf = self._readbuffer[self._offset:]
            self._readbuffer = b''
            self._offset = 0
            while not self._eof:
                buf += self._read1(self.MAX_N)
            return buf

        end = n + self._offset
        if end < len(self._readbuffer):
            buf = self._readbuffer[self._offset:end]
            self._offset = end
            return buf

        n = end - len(self._readbuffer)
        buf = self._readbuffer[self._offset:]
        self._readbuffer = b''
        self._offset = 0
        while n > 0 and not self._eof:
            data = self._read1(n)
            if n < len(data):
                self._readbuffer = data
                self._offset = n
                buf += data[:n]
                break
            buf += data
            n -= len(data)
        return buf

    def _update_crc(self, newdata):
        # Update the CRC using the given data.
        if self._expected_crc is None:
            # No need to compute the CRC if we don't have a reference value
            return
        self._running_crc = crc32(newdata, self._running_crc)
        # Check the CRC if we're at the end of the file
        if self._eof and self._running_crc != self._expected_crc:
            raise BadZipFile("Bad CRC-32 for file %r" % self.name)

    def read1(self, n):
        """Read up to n bytes with at most one read() system call."""

        if n is None or n < 0:
            buf = self._readbuffer[self._offset:]
            self._readbuffer = b''
            self._offset = 0
            while not self._eof:
                data = self._read1(self.MAX_N)
                if data:
                    buf += data
                    break
            return buf

        end = n + self._offset
        if end < len(self._readbuffer):
            buf = self._readbuffer[self._offset:end]
            self._offset = end
            return buf

        n = end - len(self._readbuffer)
        buf = self._readbuffer[self._offset:]
        self._readbuffer = b''
        self._offset = 0
        if n > 0:
            while not self._eof:
                data = self._read1(n)
                if n < len(data):
                    self._readbuffer = data
                    self._offset = n
                    buf += data[:n]
                    break
                if data:
                    buf += data
                    break
        return buf

    def _read1(self, n):
        # Read up to n compressed bytes with at most one read() system call,
        # decrypt and decompress them.
        if self._eof or n <= 0:
            return b''

        # Read from file.
        if self._compress_type == ZIP_DEFLATED:
            ## Handle unconsumed data.
            data = self._decompressor.unconsumed_tail
            if n > len(data):
                data += self._read2(n - len(data))
        else:
            data = self._read2(n)

        if self._compress_type == ZIP_STORED:
            self._eof = self._compress_left <= 0
        elif self._compress_type == ZIP_DEFLATED:
            n = max(n, self.MIN_READ_SIZE)
            data = self._decompressor.decompress(data, n)
            self._eof = (self._decompressor.eof or
                         self._compress_left <= 0 and
                         not self._decompressor.unconsumed_tail)
            if self._eof:
                data += self._decompressor.flush()
        else:
            data = self._decompressor.decompress(data)
            self._eof = self._decompressor.eof or self._compress_left <= 0

        data = data[:self._left]
        self._left -= len(data)
        if self._left <= 0:
            self._eof = True
        self._update_crc(data)
        return data

    def _read2(self, n):
        if self._compress_left <= 0:
            return b''

        n = max(n, self.MIN_READ_SIZE)
        n = min(n, self._compress_left)

        data = self._fileobj.read(n)
        self._compress_left -= len(data)
        if not data:
            raise EOFError

        if self._decrypter is not None:
            data = self._decrypter(data)
        return data

    def close(self):
        try:
            if self._close_fileobj:
                self._fileobj.close()
        finally:
            super().close()

    def seekable(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return self._seekable

    def seek(self, offset, whence=0):
        if self.closed:
            raise ValueError("seek on closed file.")
        if not self._seekable:
            raise io.UnsupportedOperation("underlying stream is not seekable")
        curr_pos = self.tell()
        if whence == 0: # Seek from start of file
            new_pos = offset
        elif whence == 1: # Seek from current position
            new_pos = curr_pos + offset
        elif whence == 2: # Seek from EOF
            new_pos = self._orig_file_size + offset
        else:
            raise ValueError("whence must be os.SEEK_SET (0), "
                             "os.SEEK_CUR (1), or os.SEEK_END (2)")

        if new_pos > self._orig_file_size:
            new_pos = self._orig_file_size

        if new_pos < 0:
            new_pos = 0

        read_offset = new_pos - curr_pos
        buff_offset = read_offset + self._offset

        if buff_offset >= 0 and buff_offset < len(self._readbuffer):
            # Just move the _offset index if the new position is in the _readbuffer
            self._offset = buff_offset
            read_offset = 0
        elif read_offset < 0:
            # Position is before the current position. Reset the ZipExtFile
            self._fileobj.seek(self._orig_compress_start)
            self._running_crc = self._orig_start_crc
            self._compress_left = self._orig_compress_size
            self._left = self._orig_file_size
            self._readbuffer = b''
            self._offset = 0
            self._decompressor = _get_decompressor(self._compress_type)
            self._eof = False
            read_offset = new_pos
            if self._decrypter is not None:
                self._init_decrypter()

        while read_offset > 0:
            read_len = min(self.MAX_SEEK_READ, read_offset)
            self.read(read_len)
            read_offset -= read_len

        return self.tell()

    def tell(self):
        if self.closed:
            raise ValueError("tell on closed file.")
        if not self._seekable:
            raise io.UnsupportedOperation("underlying stream is not seekable")
        filepos = self._orig_file_size - self._left - len(self._readbuffer) + self._offset
        return filepos


class _ZipWriteFile(io.BufferedIOBase):
    def __init__(self, zf, zinfo, zip64):
        self._zinfo = zinfo
        self._zip64 = zip64
        self._zipfile = zf
        self._compressor = _get_compressor(zinfo.compress_type,
                                           zinfo._compresslevel)
        self._file_size = 0
        self._compress_size = 0
        self._crc = 0

    @property
    def _fileobj(self):
        return self._zipfile.fp

    def writable(self):
        return True

    def write(self, data):
        if self.closed:
            raise ValueError('I/O operation on closed file.')

        # Accept any data that supports the buffer protocol
        if isinstance(data, (bytes, bytearray)):
            nbytes = len(data)
        else:
            data = memoryview(data)
            nbytes = data.nbytes
        self._file_size += nbytes

        self._crc = crc32(data, self._crc)
        if self._compressor:
            data = self._compressor.compress(data)
            self._compress_size += len(data)
        self._fileobj.write(data)
        return nbytes

    def close(self):
        if self.closed:
            return
        try:
            super().close()
            # Flush any data from the compressor, and update header info
            if self._compressor:
                buf = self._compressor.flush()
                self._compress_size += len(buf)
                self._fileobj.write(buf)
                self._zinfo.compress_size = self._compress_size
            else:
                self._zinfo.compress_size = self._file_size
            self._zinfo.CRC = self._crc
            self._zinfo.file_size = self._file_size

            # Write updated header info
            if self._zinfo.flag_bits & 0x08:
                # Write CRC and file sizes after the file data
                fmt = '<LLQQ' if self._zip64 else '<LLLL'
                self._fileobj.write(struct.pack(fmt, _DD_SIGNATURE, self._zinfo.CRC,
                    self._zinfo.compress_size, self._zinfo.file_size))
                self._zipfile.start_dir = self._fileobj.tell()
            else:
                if not self._zip64:
                    if self._file_size > ZIP64_LIMIT:
                        raise RuntimeError(
                            'File size unexpectedly exceeded ZIP64 limit')
                    if self._compress_size > ZIP64_LIMIT:
                        raise RuntimeError(
                            'Compressed size unexpectedly exceeded ZIP64 limit')
                # Seek backwards and write file header (which will now include
                # correct CRC and file sizes)

                # Preserve current position in file
                self._zipfile.start_dir = self._fileobj.tell()
                self._fileobj.seek(self._zinfo.header_offset)
                self._fileobj.write(self._zinfo.FileHeader(self._zip64))
                self._fileobj.seek(self._zipfile.start_dir)

            # Successfully written: Add file to our caches
            self._zipfile.filelist.append(self._zinfo)
            self._zipfile.NameToInfo[self._zinfo.filename] = self._zinfo
        finally:
            self._zipfile._writing = False



class ZipFile:
    """ Class with methods to open, read, write, close, list zip files.

    z = ZipFile(file, mode="r", compression=ZIP_STORED, allowZip64=True,
                compresslevel=None)

    file: Either the path to the file, or a file-like object.
          If it is a path, the file will be opened and closed by ZipFile.
    mode: The mode can be either read 'r', write 'w', exclusive create 'x',
          or append 'a'.
    compression: ZIP_STORED (no compression), ZIP_DEFLATED (requires zlib),
                 ZIP_BZIP2 (requires bz2) or ZIP_LZMA (requires lzma).
    allowZip64: if True ZipFile will create files with ZIP64 extensions when
                needed, otherwise it will raise an exception when this would
                be necessary.
    compresslevel: None (default for the given compression type) or an integer
                   specifying the level to pass to the compressor.
                   When using ZIP_STORED or ZIP_LZMA this keyword has no effect.
                   When using ZIP_DEFLATED integers 0 through 9 are accepted.
                   When using ZIP_BZIP2 integers 1 through 9 are accepted.

    """

    fp = None                   # Set here since __del__ checks it
    _windows_illegal_name_trans_table = None

    def __init__(self, file, mode="r", compression=ZIP_STORED, allowZip64=True,
                 compresslevel=None, *, strict_timestamps=True):
        """Open the ZIP file with mode read 'r', write 'w', exclusive create 'x',
        or append 'a'."""
        if mode not in ('r', 'w', 'x', 'a'):
            raise ValueError("ZipFile requires mode 'r', 'w', 'x', or 'a'")

        _check_compression(compression)

        self._allowZip64 = allowZip64
        self._didModify = False
        self.debug = 0  # Level of printing: 0 through 3
        self.NameToInfo = {}    # Find file info given name
        self.filelist = []      # List of ZipInfo instances for archive
        self.compression = compression  # Method of compression
        self.compresslevel = compresslevel
        self.mode = mode
        self.pwd = None
        self._comment = b''
        self._strict_timestamps = strict_timestamps

        # Check if we were passed a file-like object
        if isinstance(file, os.PathLike):
            file = os.fspath(file)
        if isinstance(file, str):
            # No, it's a filename
            self._filePassed = 0
            self.filename = file
            modeDict = {'r' : 'rb', 'w': 'w+b', 'x': 'x+b', 'a' : 'r+b',
                        'r+b': 'w+b', 'w+b': 'wb', 'x+b': 'xb'}
            filemode = modeDict[mode]
            while True:
                try:
                    self.fp = io.open(file, filemode)
                except OSError:
                    if filemode in modeDict:
                        filemode = modeDict[filemode]
                        continue
                    raise
                break
        else:
            self._filePassed = 1
            self.fp = file
            self.filename = getattr(file, 'name', None)
        self._fileRefCnt = 1
        self._lock = threading.RLock()
        self._seekable = True
        self._writing = False

        try:
            if mode == 'r':
                self._RealGetContents()
            elif mode in ('w', 'x'):
                # set the modified flag so central directory gets written
                # even if no files are added to the archive
                self._didModify = True
                try:
                    self.start_dir = self.fp.tell()
                except (AttributeError, OSError):
                    self.fp = _Tellable(self.fp)
                    self.start_dir = 0
                    self._seekable = False
                else:
                    # Some file-like objects can provide tell() but not seek()
                    try:
                        self.fp.seek(self.start_dir)
                    except (AttributeError, OSError):
                        self._seekable = False
            elif mode == 'a':
                try:
                    # See if file is a zip file
                    self._RealGetContents()
                    # seek to start of directory and overwrite
                    self.fp.seek(self.start_dir)
                except BadZipFile:
                    # file is not a zip file, just append
                    self.fp.seek(0, 2)

                    # set the modified flag so central directory gets written
                    # even if no files are added to the archive
                    self._didModify = True
                    self.start_dir = self.fp.tell()
            else:
                raise ValueError("Mode must be 'r', 'w', 'x', or 'a'")
        except:
            fp = self.fp
            self.fp = None
            self._fpclose(fp)
            raise

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        result = ['<%s.%s' % (self.__class__.__module__,
                              self.__class__.__qualname__)]
        if self.fp is not None:
            if self._filePassed:
                result.append(' file=%r' % self.fp)
            elif self.filename is not None:
                result.append(' filename=%r' % self.filename)
            result.append(' mode=%r' % self.mode)
        else:
            result.append(' [closed]')
        result.append('>')
        return ''.join(result)

    def _RealGetContents(self):
        """Read in the table of contents for the ZIP file."""
        fp = self.fp
        try:
            endrec = _EndRecData(fp)
        except OSError:
            raise BadZipFile("File is not a zip file")
        if not endrec:
            raise BadZipFile("File is not a zip file")
        if self.debug > 1:
            print(endrec)
        size_cd = endrec[_ECD_SIZE]             # bytes in central directory
        offset_cd = endrec[_ECD_OFFSET]         # offset of central directory
        self._comment = endrec[_ECD_COMMENT]    # archive comment

        # "concat" is zero, unless zip was concatenated to another file
        concat = endrec[_ECD_LOCATION] - size_cd - offset_cd

        if self.debug > 2:
            inferred = concat + offset_cd
            print("given, inferred, offset", offset_cd, inferred, concat)
        # self.start_dir:  Position of start of central directory
        self.start_dir = offset_cd + concat
        if self.start_dir < 0:
            raise BadZipFile("Bad offset for central directory")
        fp.seek(self.start_dir, 0)
        data = fp.read(size_cd)
        fp = io.BytesIO(data)
        total = 0
        while total < size_cd:
            centdir = fp.read(sizeCentralDir)
            if len(centdir) != sizeCentralDir:
                raise BadZipFile("Truncated central directory")
            centdir = struct.unpack(structCentralDir, centdir)
            if centdir[_CD_SIGNATURE] != stringCentralDir:
                raise BadZipFile("Bad magic number for central directory")
            if self.debug > 2:
                print(centdir)
            filename = fp.read(centdir[_CD_FILENAME_LENGTH])
            flags = centdir[5]
            if flags & 0x800:
                # UTF-8 file names extension
                filename = filename.decode('utf-8')
            else:
                # Historical ZIP filename encoding
                filename = filename.decode('cp437')
            # Create ZipInfo instance to store file information
            x = ZipInfo(filename)
            x.extra = fp.read(centdir[_CD_EXTRA_FIELD_LENGTH])
            x.comment = fp.read(centdir[_CD_COMMENT_LENGTH])
            x.header_offset = centdir[_CD_LOCAL_HEADER_OFFSET]
            (x.create_version, x.create_system, x.extract_version, x.reserved,
             x.flag_bits, x.compress_type, t, d,
             x.CRC, x.compress_size, x.file_size) = centdir[1:12]
            if x.extract_version > MAX_EXTRACT_VERSION:
                raise NotImplementedError("zip file version %.1f" %
                                          (x.extract_version / 10))
            x.volume, x.internal_attr, x.external_attr = centdir[15:18]
            # Convert date/time code to (year, month, day, hour, min, sec)
            x._raw_time = t
            x.date_time = ( (d>>9)+1980, (d>>5)&0xF, d&0x1F,
                            t>>11, (t>>5)&0x3F, (t&0x1F) * 2 )

            x._decodeExtra()
            x.header_offset = x.header_offset + concat
            self.filelist.append(x)
            self.NameToInfo[x.filename] = x

            # update total bytes read from central directory
            total = (total + sizeCentralDir + centdir[_CD_FILENAME_LENGTH]
                     + centdir[_CD_EXTRA_FIELD_LENGTH]
                     + centdir[_CD_COMMENT_LENGTH])

            if self.debug > 2:
                print("total", total)

        end_offset = self.start_dir
        for zinfo in sorted(self.filelist,
                            key=lambda zinfo: zinfo.header_offset,
                            reverse=True):
            zinfo._end_offset = end_offset
            end_offset = zinfo.header_offset

    def namelist(self):
        """Return a list of file names in the archive."""
        return [data.filename for data in self.filelist]

    def infolist(self):
        """Return a list of class ZipInfo instances for files in the
        archive."""
        return self.filelist

    def printdir(self, file=None):
        """Print a table of contents for the zip file."""
        print("%-46s %19s %12s" % ("File Name", "Modified    ", "Size"),
              file=file)
        for zinfo in self.filelist:
            date = "%d-%02d-%02d %02d:%02d:%02d" % zinfo.date_time[:6]
            print("%-46s %s %12d" % (zinfo.filename, date, zinfo.file_size),
                  file=file)

    def testzip(self):
        """Read all the files and check the CRC."""
        chunk_size = 2 ** 20
        for zinfo in self.filelist:
            try:
                # Read by chunks, to avoid an OverflowError or a
                # MemoryError with very large embedded files.
                with self.open(zinfo.filename, "r") as f:
                    while f.read(chunk_size):     # Check CRC-32
                        pass
            except BadZipFile:
                return zinfo.filename

    def getinfo(self, name):
        """Return the instance of ZipInfo given 'name'."""
        info = self.NameToInfo.get(name)
        if info is None:
            raise KeyError(
                'There is no item named %r in the archive' % name)

        return info

    def setpassword(self, pwd):
        """Set default password for encrypted files."""
        if pwd and not isinstance(pwd, bytes):
            raise TypeError("pwd: expected bytes, got %s" % type(pwd).__name__)
        if pwd:
            self.pwd = pwd
        else:
            self.pwd = None

    @property
    def comment(self):
        """The comment text associated with the ZIP file."""
        return self._comment

    @comment.setter
    def comment(self, comment):
        if not isinstance(comment, bytes):
            raise TypeError("comment: expected bytes, got %s" % type(comment).__name__)
        # check for valid comment length
        if len(comment) > ZIP_MAX_COMMENT:
            import warnings
            warnings.warn('Archive comment is too long; truncating to %d bytes'
                          % ZIP_MAX_COMMENT, stacklevel=2)
            comment = comment[:ZIP_MAX_COMMENT]
        self._comment = comment
        self._didModify = True

    def read(self, name, pwd=None):
        """Return file bytes for name."""
        with self.open(name, "r", pwd) as fp:
            return fp.read()

    def open(self, name, mode="r", pwd=None, *, force_zip64=False):
        """Return file-like object for 'name'.

        name is a string for the file name within the ZIP file, or a ZipInfo
        object.

        mode should be 'r' to read a file already in the ZIP file, or 'w' to
        write to a file newly added to the archive.

        pwd is the password to decrypt files (only used for reading).

        When writing, if the file size is not known in advance but may exceed
        2 GiB, pass force_zip64 to use the ZIP64 format, which can handle large
        files.  If the size is known in advance, it is best to pass a ZipInfo
        instance for name, with zinfo.file_size set.
        """
        if mode not in {"r", "w"}:
            raise ValueError('open() requires mode "r" or "w"')
        if pwd and not isinstance(pwd, bytes):
            raise TypeError("pwd: expected bytes, got %s" % type(pwd).__name__)
        if pwd and (mode == "w"):
            raise ValueError("pwd is only supported for reading files")
        if not self.fp:
            raise ValueError(
                "Attempt to use ZIP archive that was already closed")

        # Make sure we have an info object
        if isinstance(name, ZipInfo):
            # 'name' is already an info object
            zinfo = name
        elif mode == 'w':
            zinfo = ZipInfo(name)
            zinfo.compress_type = self.compression
            zinfo._compresslevel = self.compresslevel
        else:
            # Get info object for name
            zinfo = self.getinfo(name)

        if mode == 'w':
            return self._open_to_write(zinfo, force_zip64=force_zip64)

        if self._writing:
            raise ValueError("Can't read from the ZIP file while there "
                    "is an open writing handle on it. "
                    "Close the writing handle before trying to read.")

        # Open for reading:
        self._fileRefCnt += 1
        zef_file = _SharedFile(self.fp, zinfo.header_offset,
                               self._fpclose, self._lock, lambda: self._writing)
        try:
            # Skip the file header:
            fheader = zef_file.read(sizeFileHeader)
            if len(fheader) != sizeFileHeader:
                raise BadZipFile("Truncated file header")
            fheader = struct.unpack(structFileHeader, fheader)
            if fheader[_FH_SIGNATURE] != stringFileHeader:
                raise BadZipFile("Bad magic number for file header")

            fname = zef_file.read(fheader[_FH_FILENAME_LENGTH])
            if fheader[_FH_EXTRA_FIELD_LENGTH]:
                zef_file.read(fheader[_FH_EXTRA_FIELD_LENGTH])

            if zinfo.flag_bits & 0x20:
                # Zip 2.7: compressed patched data
                raise NotImplementedError("compressed patched data (flag bit 5)")

            if zinfo.flag_bits & 0x40:
                # strong encryption
                raise NotImplementedError("strong encryption (flag bit 6)")

            if fheader[_FH_GENERAL_PURPOSE_FLAG_BITS] & 0x800:
                # UTF-8 filename
                fname_str = fname.decode("utf-8")
            else:
                fname_str = fname.decode("cp437")

            if fname_str != zinfo.orig_filename:
                raise BadZipFile(
                    'File name in directory %r and header %r differ.'
                    % (zinfo.orig_filename, fname))

            if (zinfo._end_offset is not None and
                zef_file.tell() + zinfo.compress_size > zinfo._end_offset):
                raise BadZipFile(f"Overlapped entries: {zinfo.orig_filename!r} (possible zip bomb)")

            # check for encrypted flag & handle password
            is_encrypted = zinfo.flag_bits & 0x1
            if is_encrypted:
                if not pwd:
                    pwd = self.pwd
                if not pwd:
                    raise RuntimeError("File %r is encrypted, password "
                                       "required for extraction" % name)
            else:
                pwd = None

            return ZipExtFile(zef_file, mode, zinfo, pwd, True)
        except:
            zef_file.close()
            raise

    def _open_to_write(self, zinfo, force_zip64=False):
        if force_zip64 and not self._allowZip64:
            raise ValueError(
                "force_zip64 is True, but allowZip64 was False when opening "
                "the ZIP file."
            )
        if self._writing:
            raise ValueError("Can't write to the ZIP file while there is "
                             "another write handle open on it. "
                             "Close the first handle before opening another.")

        # Size and CRC are overwritten with correct data after processing the file
        zinfo.compress_size = 0
        zinfo.CRC = 0

        zinfo.flag_bits = 0x00
        if zinfo.compress_type == ZIP_LZMA:
            # Compressed data includes an end-of-stream (EOS) marker
            zinfo.flag_bits |= 0x02
        if not self._seekable:
            zinfo.flag_bits |= 0x08

        if not zinfo.external_attr:
            zinfo.external_attr = 0o600 << 16  # permissions: ?rw-------

        # Compressed size can be larger than uncompressed size
        zip64 = self._allowZip64 and \
                (force_zip64 or zinfo.file_size * 1.05 > ZIP64_LIMIT)

        if self._seekable:
            self.fp.seek(self.start_dir)
        zinfo.header_offset = self.fp.tell()

        self._writecheck(zinfo)
        self._didModify = True

        self.fp.write(zinfo.FileHeader(zip64))

        self._writing = True
        return _ZipWriteFile(self, zinfo, zip64)

    def extract(self, member, path=None, pwd=None):
        """Extract a member from the archive to the current working directory,
           using its full name. Its file information is extracted as accurately
           as possible. `member' may be a filename or a ZipInfo object. You can
           specify a different directory using `path'.
        """
        if path is None:
            path = os.getcwd()
        else:
            path = os.fspath(path)

        return self._extract_member(member, path, pwd)

    def extractall(self, path=None, members=None, pwd=None):
        """Extract all members from the archive to the current working
           directory. `path' specifies a different directory to extract to.
           `members' is optional and must be a subset of the list returned
           by namelist().
        """
        if members is None:
            members = self.namelist()

        if path is None:
            path = os.getcwd()
        else:
            path = os.fspath(path)

        for zipinfo in members:
            self._extract_member(zipinfo, path, pwd)

    @classmethod
    def _sanitize_windows_name(cls, arcname, pathsep):
        """Replace bad characters and remove trailing dots from parts."""
        table = cls._windows_illegal_name_trans_table
        if not table:
            illegal = ':<>|"?*'
            table = str.maketrans(illegal, '_' * len(illegal))
            cls._windows_illegal_name_trans_table = table
        arcname = arcname.translate(table)
        # remove trailing dots
        arcname = (x.rstrip('.') for x in arcname.split(pathsep))
        # rejoin, removing empty parts.
        arcname = pathsep.join(x for x in arcname if x)
        return arcname

    def _extract_member(self, member, targetpath, pwd):
        """Extract the ZipInfo object 'member' to a physical
           file on the path targetpath.
        """
        if not isinstance(member, ZipInfo):
            member = self.getinfo(member)

        # build the destination pathname, replacing
        # forward slashes to platform specific separators.
        arcname = member.filename.replace('/', os.path.sep)

        if os.path.altsep:
            arcname = arcname.replace(os.path.altsep, os.path.sep)
        # interpret absolute pathname as relative, remove drive letter or
        # UNC path, redundant separators, "." and ".." components.
        arcname = os.path.splitdrive(arcname)[1]
        invalid_path_parts = ('', os.path.curdir, os.path.pardir)
        arcname = os.path.sep.join(x for x in arcname.split(os.path.sep)
                                   if x not in invalid_path_parts)
        if os.path.sep == '\\':
            # filter illegal characters on Windows
            arcname = self._sanitize_windows_name(arcname, os.path.sep)

        targetpath = os.path.join(targetpath, arcname)
        targetpath = os.path.normpath(targetpath)

        # Create all upper directories if necessary.
        upperdirs = os.path.dirname(targetpath)
        if upperdirs and not os.path.exists(upperdirs):
            os.makedirs(upperdirs)

        if member.is_dir():
            if not os.path.isdir(targetpath):
                os.mkdir(targetpath)
            return targetpath

        with self.open(member, pwd=pwd) as source, \
             open(targetpath, "wb") as target:
            shutil.copyfileobj(source, target)

        return targetpath

    def _writecheck(self, zinfo):
        """Check for errors before writing a file to the archive."""
        if zinfo.filename in self.NameToInfo:
            import warnings
            warnings.warn('Duplicate name: %r' % zinfo.filename, stacklevel=3)
        if self.mode not in ('w', 'x', 'a'):
            raise ValueError("write() requires mode 'w', 'x', or 'a'")
        if not self.fp:
            raise ValueError(
                "Attempt to write ZIP archive that was already closed")
        _check_compression(zinfo.compress_type)
        if not self._allowZip64:
            requires_zip64 = None
            if len(self.filelist) >= ZIP_FILECOUNT_LIMIT:
                requires_zip64 = "Files count"
            elif zinfo.file_size > ZIP64_LIMIT:
                requires_zip64 = "Filesize"
            elif zinfo.header_offset > ZIP64_LIMIT:
                requires_zip64 = "Zipfile size"
            if requires_zip64:
                raise LargeZipFile(requires_zip64 +
                                   " would require ZIP64 extensions")

    def write(self, filename, arcname=None,
              compress_type=None, compresslevel=None):
        """Put the bytes from filename into the archive under the name
        arcname."""
        if not self.fp:
            raise ValueError(
                "Attempt to write to ZIP archive that was already closed")
        if self._writing:
            raise ValueError(
                "Can't write to ZIP archive while an open writing handle exists"
            )

        zinfo = ZipInfo.from_file(filename, arcname,
                                  strict_timestamps=self._strict_timestamps)

        if zinfo.is_dir():
            zinfo.compress_size = 0
            zinfo.CRC = 0
        else:
            if compress_type is not None:
                zinfo.compress_type = compress_type
            else:
                zinfo.compress_type = self.compression

            if compresslevel is not None:
                zinfo._compresslevel = compresslevel
            else:
                zinfo._compresslevel = self.compresslevel

        if zinfo.is_dir():
            with self._lock:
                if self._seekable:
                    self.fp.seek(self.start_dir)
                zinfo.header_offset = self.fp.tell()  # Start of header bytes
                if zinfo.compress_type == ZIP_LZMA:
                # Compressed data includes an end-of-stream (EOS) marker
                    zinfo.flag_bits |= 0x02

                self._writecheck(zinfo)
                self._didModify = True

                self.filelist.append(zinfo)
                self.NameToInfo[zinfo.filename] = zinfo
                self.fp.write(zinfo.FileHeader(False))
                self.start_dir = self.fp.tell()
        else:
            with open(filename, "rb") as src, self.open(zinfo, 'w') as dest:
                shutil.copyfileobj(src, dest, 1024*8)

    def writestr(self, zinfo_or_arcname, data,
                 compress_type=None, compresslevel=None):
        """Write a file into the archive.  The contents is 'data', which
        may be either a 'str' or a 'bytes' instance; if it is a 'str',
        it is encoded as UTF-8 first.
        'zinfo_or_arcname' is either a ZipInfo instance or
        the name of the file in the archive."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(zinfo_or_arcname, ZipInfo):
            zinfo = ZipInfo(filename=zinfo_or_arcname,
                            date_time=time.localtime(time.time())[:6])
            zinfo.compress_type = self.compression
            zinfo._compresslevel = self.compresslevel
            if zinfo.filename[-1] == '/':
                zinfo.external_attr = 0o40775 << 16   # drwxrwxr-x
                zinfo.external_attr |= 0x10           # MS-DOS directory flag
            else:
                zinfo.external_attr = 0o600 << 16     # ?rw-------
        else:
            zinfo = zinfo_or_arcname

        if not self.fp:
            raise ValueError(
                "Attempt to write to ZIP archive that was already closed")
        if self._writing:
            raise ValueError(
                "Can't write to ZIP archive while an open writing handle exists."
            )

        if compress_type is not None:
            zinfo.compress_type = compress_type

        if compresslevel is not None:
            zinfo._compresslevel = compresslevel

        zinfo.file_size = len(data)            # Uncompressed size
        with self._lock:
            with self.open(zinfo, mode='w') as dest:
                dest.write(data)

    def __del__(self):
        """Call the "close()" method in case the user forgot."""
        self.close()

    def close(self):
        """Close the file, and for mode 'w', 'x' and 'a' write the ending
        records."""
        if self.fp is None:
            return

        if self._writing:
            raise ValueError("Can't close the ZIP file while there is "
                             "an open writing handle on it. "
                             "Close the writing handle before closing the zip.")

        try:
            if self.mode in ('w', 'x', 'a') and self._didModify: # write ending records
                with self._lock:
                    if self._seekable:
                        self.fp.seek(self.start_dir)
                    self._write_end_record()
        finally:
            fp = self.fp
            self.fp = None
            self._fpclose(fp)

    def _write_end_record(self):
        for zinfo in self.filelist:         # write central directory
            dt = zinfo.date_time
            dosdate = (dt[0] - 1980) << 9 | dt[1] << 5 | dt[2]
            dostime = dt[3] << 11 | dt[4] << 5 | (dt[5] // 2)
            extra = []
            if zinfo.file_size > ZIP64_LIMIT \
               or zinfo.compress_size > ZIP64_LIMIT:
                extra.append(zinfo.file_size)
                extra.append(zinfo.compress_size)
                file_size = 0xffffffff
                compress_size = 0xffffffff
            else:
                file_size = zinfo.file_size
                compress_size = zinfo.compress_size

            if zinfo.header_offset > ZIP64_LIMIT:
                extra.append(zinfo.header_offset)
                header_offset = 0xffffffff
            else:
                header_offset = zinfo.header_offset

            extra_data = zinfo.extra
            min_version = 0
            if extra:
                # Append a ZIP64 field to the extra's
                extra_data = _strip_extra(extra_data, (1,))
                extra_data = struct.pack(
                    '<HH' + 'Q'*len(extra),
                    1, 8*len(extra), *extra) + extra_data

                min_version = ZIP64_VERSION

            if zinfo.compress_type == ZIP_BZIP2:
                min_version = max(BZIP2_VERSION, min_version)
            elif zinfo.compress_type == ZIP_LZMA:
                min_version = max(LZMA_VERSION, min_version)

            extract_version = max(min_version, zinfo.extract_version)
            create_version = max(min_version, zinfo.create_version)
            filename, flag_bits = zinfo._encodeFilenameFlags()
            centdir = struct.pack(structCentralDir,
                                  stringCentralDir, create_version,
                                  zinfo.create_system, extract_version, zinfo.reserved,
                                  flag_bits, zinfo.compress_type, dostime, dosdate,
                                  zinfo.CRC, compress_size, file_size,
                                  len(filename), len(extra_data), len(zinfo.comment),
                                  0, zinfo.internal_attr, zinfo.external_attr,
                                  header_offset)
            self.fp.write(centdir)
            self.fp.write(filename)
            self.fp.write(extra_data)
            self.fp.write(zinfo.comment)

        pos2 = self.fp.tell()
        # Write end-of-zip-archive record
        centDirCount = len(self.filelist)
        centDirSize = pos2 - self.start_dir
        centDirOffset = self.start_dir
        requires_zip64 = None
        if centDirCount > ZIP_FILECOUNT_LIMIT:
            requires_zip64 = "Files count"
        elif centDirOffset > ZIP64_LIMIT:
            requires_zip64 = "Central directory offset"
        elif centDirSize > ZIP64_LIMIT:
            requires_zip64 = "Central directory size"
        if requires_zip64:
            # Need to write the ZIP64 end-of-archive records
            if not self._allowZip64:
                raise LargeZipFile(requires_zip64 +
                                   " would require ZIP64 extensions")
            zip64endrec = struct.pack(
                structEndArchive64, stringEndArchive64,
                sizeEndCentDir64 - 12, 45, 45, 0, 0, centDirCount, centDirCount,
                centDirSize, centDirOffset)
            self.fp.write(zip64endrec)

            zip64locrec = struct.pack(
                structEndArchive64Locator,
                stringEndArchive64Locator, 0, pos2, 1)
            self.fp.write(zip64locrec)
            centDirCount = min(centDirCount, 0xFFFF)
            centDirSize = min(centDirSize, 0xFFFFFFFF)
            centDirOffset = min(centDirOffset, 0xFFFFFFFF)

        endrec = struct.pack(structEndArchive, stringEndArchive,
                             0, 0, centDirCount, centDirCount,
                             centDirSize, centDirOffset, len(self._comment))
        self.fp.write(endrec)
        self.fp.write(self._comment)
        if self.mode == "a":
            self.fp.truncate()
        self.fp.flush()

    def _fpclose(self, fp):
        assert self._fileRefCnt > 0
        self._fileRefCnt -= 1
        if not self._fileRefCnt and not self._filePassed:
            fp.close()


class PyZipFile(ZipFile):
    """Class to create ZIP archives with Python library files and packages."""

    def __init__(self, file, mode="r", compression=ZIP_STORED,
                 allowZip64=True, optimize=-1):
        ZipFile.__init__(self, file, mode=mode, compression=compression,
                         allowZip64=allowZip64)
        self._optimize = optimize

    def writepy(self, pathname, basename="", filterfunc=None):
        """Add all files from "pathname" to the ZIP archive.

        If pathname is a package directory, search the directory and
        all package subdirectories recursively for all *.py and enter
        the modules into the archive.  If pathname is a plain
        directory, listdir *.py and enter all modules.  Else, pathname
        must be a Python *.py file and the module will be put into the
        archive.  Added modules are always module.pyc.
        This method will compile the module.py into module.pyc if
        necessary.
        If filterfunc(pathname) is given, it is called with every argument.
        When it is False, the file or directory is skipped.
        """
        pathname = os.fspath(pathname)
        if filterfunc and not filterfunc(pathname):
            if self.debug:
                label = 'path' if os.path.isdir(pathname) else 'file'
                print('%s %r skipped by filterfunc' % (label, pathname))
            return
        dir, name = os.path.split(pathname)
        if os.path.isdir(pathname):
            initname = os.path.join(pathname, "__init__.py")
            if os.path.isfile(initname):
                # This is a package directory, add it
                if basename:
                    basename = "%s/%s" % (basename, name)
                else:
                    basename = name
                if self.debug:
                    print("Adding package in", pathname, "as", basename)
                fname, arcname = self._get_codename(initname[0:-3], basename)
                if self.debug:
                    print("Adding", arcname)
                self.write(fname, arcname)
                dirlist = sorted(os.listdir(pathname))
                dirlist.remove("__init__.py")
                # Add all *.py files and package subdirectories
                for filename in dirlist:
                    path = os.path.join(pathname, filename)
                    root, ext = os.path.splitext(filename)
                    if os.path.isdir(path):
                        if os.path.isfile(os.path.join(path, "__init__.py")):
                            # This is a package directory, add it
                            self.writepy(path, basename,
                                         filterfunc=filterfunc)  # Recursive call
                    elif ext == ".py":
                        if filterfunc and not filterfunc(path):
                            if self.debug:
                                print('file %r skipped by filterfunc' % path)
                            continue
                        fname, arcname = self._get_codename(path[0:-3],
                                                            basename)
                        if self.debug:
                            print("Adding", arcname)
                        self.write(fname, arcname)
            else:
                # This is NOT a package directory, add its files at top level
                if self.debug:
                    print("Adding files from directory", pathname)
                for filename in sorted(os.listdir(pathname)):
                    path = os.path.join(pathname, filename)
                    root, ext = os.path.splitext(filename)
                    if ext == ".py":
                        if filterfunc and not filterfunc(path):
                            if self.debug:
                                print('file %r skipped by filterfunc' % path)
                            continue
                        fname, arcname = self._get_codename(path[0:-3],
                                                            basename)
                        if self.debug:
                            print("Adding", arcname)
                        self.write(fname, arcname)
        else:
            if pathname[-3:] != ".py":
                raise RuntimeError(
                    'Files added with writepy() must end with ".py"')
            fname, arcname = self._get_codename(pathname[0:-3], basename)
            if self.debug:
                print("Adding file", arcname)
            self.write(fname, arcname)

    def _get_codename(self, pathname, basename):
        """Return (filename, archivename) for the path.

        Given a module name path, return the correct file path and
        archive name, compiling if necessary.  For example, given
        /python/lib/string, return (/python/lib/string.pyc, string).
        """
        def _compile(file, optimize=-1):
            import py_compile
            if self.debug:
                print("Compiling", file)
            try:
                py_compile.compile(file, doraise=True, optimize=optimize)
            except py_compile.PyCompileError as err:
                print(err.msg)
                return False
            return True

        file_py  = pathname + ".py"
        file_pyc = pathname + ".pyc"
        pycache_opt0 = importlib.util.cache_from_source(file_py, optimization='')
        pycache_opt1 = importlib.util.cache_from_source(file_py, optimization=1)
        pycache_opt2 = importlib.util.cache_from_source(file_py, optimization=2)
        if self._optimize == -1:
            # legacy mode: use whatever file is present
            if (os.path.isfile(file_pyc) and
                  os.stat(file_pyc).st_mtime >= os.stat(file_py).st_mtime):
                # Use .pyc file.
                arcname = fname = file_pyc
            elif (os.path.isfile(pycache_opt0) and
                  os.stat(pycache_opt0).st_mtime >= os.stat(file_py).st_mtime):
                # Use the __pycache__/*.pyc file, but write it to the legacy pyc
                # file name in the archive.
                fname = pycache_opt0
                arcname = file_pyc
            elif (os.path.isfile(pycache_opt1) and
                  os.stat(pycache_opt1).st_mtime >= os.stat(file_py).st_mtime):
                # Use the __pycache__/*.pyc file, but write it to the legacy pyc
                # file name in the archive.
                fname = pycache_opt1
                arcname = file_pyc
            elif (os.path.isfile(pycache_opt2) and
                  os.stat(pycache_opt2).st_mtime >= os.stat(file_py).st_mtime):
                # Use the __pycache__/*.pyc file, but write it to the legacy pyc
                # file name in the archive.
                fname = pycache_opt2
                arcname = file_pyc
            else:
                # Compile py into PEP 3147 pyc file.
                if _compile(file_py):
                    if sys.flags.optimize == 0:
                        fname = pycache_opt0
                    elif sys.flags.optimize == 1:
                        fname = pycache_opt1
                    else:
                        fname = pycache_opt2
                    arcname = file_pyc
                else:
                    fname = arcname = file_py
        else:
            # new mode: use given optimization level
            if self._optimize == 0:
                fname = pycache_opt0
                arcname = file_pyc
            else:
                arcname = file_pyc
                if self._optimize == 1:
                    fname = pycache_opt1
                elif self._optimize == 2:
                    fname = pycache_opt2
                else:
                    msg = "invalid value for 'optimize': {!r}".format(self._optimize)
                    raise ValueError(msg)
            if not (os.path.isfile(fname) and
                    os.stat(fname).st_mtime >= os.stat(file_py).st_mtime):
                if not _compile(file_py, optimize=self._optimize):
                    fname = arcname = file_py
        archivename = os.path.split(arcname)[1]
        if basename:
            archivename = "%s/%s" % (basename, archivename)
        return (fname, archivename)


def _parents(path):
    """
    Given a path with elements separated by
    posixpath.sep, generate all parents of that path.

    >>> list(_parents('b/d'))
    ['b']
    >>> list(_parents('/b/d/'))
    ['/b']
    >>> list(_parents('b/d/f/'))
    ['b/d', 'b']
    >>> list(_parents('b'))
    []
    >>> list(_parents(''))
    []
    """
    return itertools.islice(_ancestry(path), 1, None)


def _ancestry(path):
    """
    Given a path with elements separated by
    posixpath.sep, generate all elements of that path.

    >>> list(_ancestry('b/d'))
    ['b/d', 'b']
    >>> list(_ancestry('/b/d/'))
    ['/b/d', '/b']
    >>> list(_ancestry('b/d/f/'))
    ['b/d/f', 'b/d', 'b']
    >>> list(_ancestry('b'))
    ['b']
    >>> list(_ancestry(''))
    []

    Multiple separators are treated like a single.

    >>> list(_ancestry('//b//d///f//'))
    ['//b//d///f', '//b//d', '//b']
    """
    path = path.rstrip(posixpath.sep)
    while path.rstrip(posixpath.sep):
        yield path
        path, tail = posixpath.split(path)


_dedupe = dict.fromkeys
"""Deduplicate an iterable in original order"""


def _difference(minuend, subtrahend):
    """
    Return items in minuend not in subtrahend, retaining order
    with O(1) lookup.
    """
    return itertools.filterfalse(set(subtrahend).__contains__, minuend)


class CompleteDirs(ZipFile):
    """
    A ZipFile subclass that ensures that implied directories
    are always included in the namelist.
    """

    @staticmethod
    def _implied_dirs(names):
        parents = itertools.chain.from_iterable(map(_parents, names))
        as_dirs = (p + posixpath.sep for p in parents)
        return _dedupe(_difference(as_dirs, names))

    def namelist(self):
        names = super(CompleteDirs, self).namelist()
        return names + list(self._implied_dirs(names))

    def _name_set(self):
        return set(self.namelist())

    def resolve_dir(self, name):
        """
        If the name represents a directory, return that name
        as a directory (with the trailing slash).
        """
        names = self._name_set()
        dirname = name + '/'
        dir_match = name not in names and dirname in names
        return dirname if dir_match else name

    def getinfo(self, name):
        """
        Supplement getinfo for implied dirs.
        """
        try:
            return super().getinfo(name)
        except KeyError:
            if not name.endswith('/') or name not in self._name_set():
                raise
            return ZipInfo(filename=name)

    @classmethod
    def make(cls, source):
        """
        Given a source (filename or zipfile), return an
        appropriate CompleteDirs subclass.
        """
        if isinstance(source, CompleteDirs):
            return source

        if not isinstance(source, ZipFile):
            return cls(source)

        # Only allow for FastLookup when supplied zipfile is read-only
        if 'r' not in source.mode:
            cls = CompleteDirs

        source.__class__ = cls
        return source


class FastLookup(CompleteDirs):
    """
    ZipFile subclass to ensure implicit
    dirs exist and are resolved rapidly.
    """

    def namelist(self):
        with contextlib.suppress(AttributeError):
            return self.__names
        self.__names = super(FastLookup, self).namelist()
        return self.__names

    def _name_set(self):
        with contextlib.suppress(AttributeError):
            return self.__lookup
        self.__lookup = super(FastLookup, self)._name_set()
        return self.__lookup


def _extract_text_encoding(encoding=None, *args, **kwargs):
    # stacklevel=3 so that the caller of the caller see any warning.
    return io.text_encoding(encoding, 3), args, kwargs


class Path:
    """
    A pathlib-compatible interface for zip files.

    Consider a zip file with this structure::

        .
        ├── a.txt
        └── b
            ├── c.txt
            └── d
                └── e.txt

    >>> data = io.BytesIO()
    >>> zf = ZipFile(data, 'w')
    >>> zf.writestr('a.txt', 'content of a')
    >>> zf.writestr('b/c.txt', 'content of c')
    >>> zf.writestr('b/d/e.txt', 'content of e')
    >>> zf.filename = 'mem/abcde.zip'

    Path accepts the zipfile object itself or a filename

    >>> root = Path(zf)

    From there, several path operations are available.

    Directory iteration (including the zip file itself):

    >>> a, b = root.iterdir()
    >>> a
    Path('mem/abcde.zip', 'a.txt')
    >>> b
    Path('mem/abcde.zip', 'b/')

    name property:

    >>> b.name
    'b'

    join with divide operator:

    >>> c = b / 'c.txt'
    >>> c
    Path('mem/abcde.zip', 'b/c.txt')
    >>> c.name
    'c.txt'

    Read text:

    >>> c.read_text()
    'content of c'

    existence:

    >>> c.exists()
    True
    >>> (b / 'missing.txt').exists()
    False

    Coercion to string:

    >>> import os
    >>> str(c).replace(os.sep, posixpath.sep)
    'mem/abcde.zip/b/c.txt'

    At the root, ``name``, ``filename``, and ``parent``
    resolve to the zipfile. Note these attributes are not
    valid and will raise a ``ValueError`` if the zipfile
    has no filename.

    >>> root.name
    'abcde.zip'
    >>> str(root.filename).replace(os.sep, posixpath.sep)
    'mem/abcde.zip'
    >>> str(root.parent)
    'mem'
    """

    __repr = "{self.__class__.__name__}({self.root.filename!r}, {self.at!r})"

    def __init__(self, root, at=""):
        """
        Construct a Path from a ZipFile or filename.

        Note: When the source is an existing ZipFile object,
        its type (__class__) will be mutated to a
        specialized type. If the caller wishes to retain the
        original type, the caller should either create a
        separate ZipFile object or pass a filename.
        """
        self.root = FastLookup.make(root)
        self.at = at

    def open(self, mode='r', *args, pwd=None, **kwargs):
        """
        Open this entry as text or binary following the semantics
        of ``pathlib.Path.open()`` by passing arguments through
        to io.TextIOWrapper().
        """
        if self.is_dir():
            raise IsADirectoryError(self)
        zip_mode = mode[0]
        if not self.exists() and zip_mode == 'r':
            raise FileNotFoundError(self)
        stream = self.root.open(self.at, zip_mode, pwd=pwd)
        if 'b' in mode:
            if args or kwargs:
                raise ValueError("encoding args invalid for binary operation")
            return stream
        # Text mode:
        encoding, args, kwargs = _extract_text_encoding(*args, **kwargs)
        return io.TextIOWrapper(stream, encoding, *args, **kwargs)

    @property
    def name(self):
        return pathlib.Path(self.at).name or self.filename.name

    @property
    def filename(self):
        return pathlib.Path(self.root.filename).joinpath(self.at)

    def read_text(self, *args, **kwargs):
        encoding, args, kwargs = _extract_text_encoding(*args, **kwargs)
        with self.open('r', encoding, *args, **kwargs) as strm:
            return strm.read()

    def read_bytes(self):
        with self.open('rb') as strm:
            return strm.read()

    def _is_child(self, path):
        return posixpath.dirname(path.at.rstrip("/")) == self.at.rstrip("/")

    def _next(self, at):
        return self.__class__(self.root, at)

    def is_dir(self):
        return not self.at or self.at.endswith("/")

    def is_file(self):
        return self.exists() and not self.is_dir()

    def exists(self):
        return self.at in self.root._name_set()

    def iterdir(self):
        if not self.is_dir():
            raise ValueError("Can't listdir a file")
        subs = map(self._next, self.root.namelist())
        return filter(self._is_child, subs)

    def __str__(self):
        return posixpath.join(self.root.filename, self.at)

    def __repr__(self):
        return self.__repr.format(self=self)

    def joinpath(self, *other):
        next = posixpath.join(self.at, *other)
        return self._next(self.root.resolve_dir(next))

    __truediv__ = joinpath

    @property
    def parent(self):
        if not self.at:
            return self.filename.parent
        parent_at = posixpath.dirname(self.at.rstrip('/'))
        if parent_at:
            parent_at += '/'
        return self._next(parent_at)


def main(args=None):
    import argparse

    description = 'A simple command-line interface for zipfile module.'
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--list', metavar='<zipfile>',
                       help='Show listing of a zipfile')
    group.add_argument('-e', '--extract', nargs=2,
                       metavar=('<zipfile>', '<output_dir>'),
                       help='Extract zipfile into target dir')
    group.add_argument('-c', '--create', nargs='+',
                       metavar=('<name>', '<file>'),
                       help='Create zipfile from sources')
    group.add_argument('-t', '--test', metavar='<zipfile>',
                       help='Test if a zipfile is valid')
    args = parser.parse_args(args)

    if args.test is not None:
        src = args.test
        with ZipFile(src, 'r') as zf:
            badfile = zf.testzip()
        if badfile:
            print("The following enclosed file is corrupted: {!r}".format(badfile))
        print("Done testing")

    elif args.list is not None:
        src = args.list
        with ZipFile(src, 'r') as zf:
            zf.printdir()

    elif args.extract is not None:
        src, curdir = args.extract
        with ZipFile(src, 'r') as zf:
            zf.extractall(curdir)

    elif args.create is not None:
        zip_name = args.create.pop(0)
        files = args.create

        def addToZip(zf, path, zippath):
            if os.path.isfile(path):
                zf.write(path, zippath, ZIP_DEFLATED)
            elif os.path.isdir(path):
                if zippath:
                    zf.write(path, zippath)
                for nm in sorted(os.listdir(path)):
                    addToZip(zf,
                             os.path.join(path, nm), os.path.join(zippath, nm))
            # else: ignore

        with ZipFile(zip_name, 'w') as zf:
            for path in files:
                zippath = os.path.basename(path)
                if not zippath:
                    zippath = os.path.basename(os.path.dirname(path))
                if zippath in ('', os.curdir, os.pardir):
                    zippath = ''
                addToZip(zf, path, zippath)


if __name__ == "__main__":
    main()
<|endoftext|>
import webbrowser
import hashlib

webbrowser.open("https://xkcd.com/353/")

def geohash(latitude, longitude, datedow):
    '''Compute geohash() using the Munroe algorithm.

    >>> geohash(37.421542, -122.085589, b'2005-05-26-10458.68')
    37.857713 -122.544543

    '''
    # https://xkcd.com/426/
    h = hashlib.md5(datedow, usedforsecurity=False).hexdigest()
    p, q = [('%f' % float.fromhex('0.' + x)) for x in (h[:16], h[16:32])]
    print('%d%s %d%s' % (latitude, p[1:], longitude, q[1:]))
<|endoftext|>"""Python part of the warnings subsystem."""

import sys


__all__ = ["warn", "warn_explicit", "showwarning",
           "formatwarning", "filterwarnings", "simplefilter",
           "resetwarnings", "catch_warnings"]

def showwarning(message, category, filename, lineno, file=None, line=None):
    """Hook to write a warning to a file; replace if you like."""
    msg = WarningMessage(message, category, filename, lineno, file, line)
    _showwarnmsg_impl(msg)

def formatwarning(message, category, filename, lineno, line=None):
    """Function to format a warning the standard way."""
    msg = WarningMessage(message, category, filename, lineno, None, line)
    return _formatwarnmsg_impl(msg)

def _showwarnmsg_impl(msg):
    file = msg.file
    if file is None:
        file = sys.stderr
        if file is None:
            # sys.stderr is None when run with pythonw.exe:
            # warnings get lost
            return
    text = _formatwarnmsg(msg)
    try:
        file.write(text)
    except OSError:
        # the file (probably stderr) is invalid - this warning gets lost.
        pass

def _formatwarnmsg_impl(msg):
    category = msg.category.__name__
    s =  f"{msg.filename}:{msg.lineno}: {category}: {msg.message}\n"

    if msg.line is None:
        try:
            import linecache
            line = linecache.getline(msg.filename, msg.lineno)
        except Exception:
            # When a warning is logged during Python shutdown, linecache
            # and the import machinery don't work anymore
            line = None
            linecache = None
    else:
        line = msg.line
    if line:
        line = line.strip()
        s += "  %s\n" % line

    if msg.source is not None:
        try:
            import tracemalloc
        # Logging a warning should not raise a new exception:
        # catch Exception, not only ImportError and RecursionError.
        except Exception:
            # don't suggest to enable tracemalloc if it's not available
            tracing = True
            tb = None
        else:
            tracing = tracemalloc.is_tracing()
            try:
                tb = tracemalloc.get_object_traceback(msg.source)
            except Exception:
                # When a warning is logged during Python shutdown, tracemalloc
                # and the import machinery don't work anymore
                tb = None

        if tb is not None:
            s += 'Object allocated at (most recent call last):\n'
            for frame in tb:
                s += ('  File "%s", lineno %s\n'
                      % (frame.filename, frame.lineno))

                try:
                    if linecache is not None:
                        line = linecache.getline(frame.filename, frame.lineno)
                    else:
                        line = None
                except Exception:
                    line = None
                if line:
                    line = line.strip()
                    s += '    %s\n' % line
        elif not tracing:
            s += (f'{category}: Enable tracemalloc to get the object '
                  f'allocation traceback\n')
    return s

# Keep a reference to check if the function was replaced
_showwarning_orig = showwarning

def _showwarnmsg(msg):
    """Hook to write a warning to a file; replace if you like."""
    try:
        sw = showwarning
    except NameError:
        pass
    else:
        if sw is not _showwarning_orig:
            # warnings.showwarning() was replaced
            if not callable(sw):
                raise TypeError("warnings.showwarning() must be set to a "
                                "function or method")

            sw(msg.message, msg.category, msg.filename, msg.lineno,
               msg.file, msg.line)
            return
    _showwarnmsg_impl(msg)

# Keep a reference to check if the function was replaced
_formatwarning_orig = formatwarning

def _formatwarnmsg(msg):
    """Function to format a warning the standard way."""
    try:
        fw = formatwarning
    except NameError:
        pass
    else:
        if fw is not _formatwarning_orig:
            # warnings.formatwarning() was replaced
            return fw(msg.message, msg.category,
                      msg.filename, msg.lineno, msg.line)
    return _formatwarnmsg_impl(msg)

def filterwarnings(action, message="", category=Warning, module="", lineno=0,
                   append=False):
    """Insert an entry into the list of warnings filters (at the front).

    'action' -- one of "error", "ignore", "always", "default", "module",
                or "once"
    'message' -- a regex that the warning message must match
    'category' -- a class that the warning must be a subclass of
    'module' -- a regex that the module name must match
    'lineno' -- an integer line number, 0 matches all warnings
    'append' -- if true, append to the list of filters
    """
    assert action in ("error", "ignore", "always", "default", "module",
                      "once"), "invalid action: %r" % (action,)
    assert isinstance(message, str), "message must be a string"
    assert isinstance(category, type), "category must be a class"
    assert issubclass(category, Warning), "category must be a Warning subclass"
    assert isinstance(module, str), "module must be a string"
    assert isinstance(lineno, int) and lineno >= 0, \
           "lineno must be an int >= 0"

    if message or module:
        import re

    if message:
        message = re.compile(message, re.I)
    else:
        message = None
    if module:
        module = re.compile(module)
    else:
        module = None

    _add_filter(action, message, category, module, lineno, append=append)

def simplefilter(action, category=Warning, lineno=0, append=False):
    """Insert a simple entry into the list of warnings filters (at the front).

    A simple filter matches all modules and messages.
    'action' -- one of "error", "ignore", "always", "default", "module",
                or "once"
    'category' -- a class that the warning must be a subclass of
    'lineno' -- an integer line number, 0 matches all warnings
    'append' -- if true, append to the list of filters
    """
    assert action in ("error", "ignore", "always", "default", "module",
                      "once"), "invalid action: %r" % (action,)
    assert isinstance(lineno, int) and lineno >= 0, \
           "lineno must be an int >= 0"
    _add_filter(action, None, category, None, lineno, append=append)

def _add_filter(*item, append):
    # Remove possible duplicate filters, so new one will be placed
    # in correct place. If append=True and duplicate exists, do nothing.
    if not append:
        try:
            filters.remove(item)
        except ValueError:
            pass
        filters.insert(0, item)
    else:
        if item not in filters:
            filters.append(item)
    _filters_mutated()

def resetwarnings():
    """Clear the list of warning filters, so that no filters are active."""
    filters[:] = []
    _filters_mutated()

class _OptionError(Exception):
    """Exception used by option processing helpers."""
    pass

# Helper to process -W options passed via sys.warnoptions
def _processoptions(args):
    for arg in args:
        try:
            _setoption(arg)
        except _OptionError as msg:
            print("Invalid -W option ignored:", msg, file=sys.stderr)

# Helper for _processoptions()
def _setoption(arg):
    parts = arg.split(':')
    if len(parts) > 5:
        raise _OptionError("too many fields (max 5): %r" % (arg,))
    while len(parts) < 5:
        parts.append('')
    action, message, category, module, lineno = [s.strip()
                                                 for s in parts]
    action = _getaction(action)
    category = _getcategory(category)
    if message or module:
        import re
    if message:
        message = re.escape(message)
    if module:
        module = re.escape(module) + r'\Z'
    if lineno:
        try:
            lineno = int(lineno)
            if lineno < 0:
                raise ValueError
        except (ValueError, OverflowError):
            raise _OptionError("invalid lineno %r" % (lineno,)) from None
    else:
        lineno = 0
    filterwarnings(action, message, category, module, lineno)

# Helper for _setoption()
def _getaction(action):
    if not action:
        return "default"
    if action == "all": return "always" # Alias
    for a in ('default', 'always', 'ignore', 'module', 'once', 'error'):
        if a.startswith(action):
            return a
    raise _OptionError("invalid action: %r" % (action,))

# Helper for _setoption()
def _getcategory(category):
    if not category:
        return Warning
    if '.' not in category:
        import builtins as m
        klass = category
    else:
        module, _, klass = category.rpartition('.')
        try:
            m = __import__(module, None, None, [klass])
        except ImportError:
            raise _OptionError("invalid module name: %r" % (module,)) from None
    try:
        cat = getattr(m, klass)
    except AttributeError:
        raise _OptionError("unknown warning category: %r" % (category,)) from None
    if not issubclass(cat, Warning):
        raise _OptionError("invalid warning category: %r" % (category,))
    return cat


def _is_internal_frame(frame):
    """Signal whether the frame is an internal CPython implementation detail."""
    filename = frame.f_code.co_filename
    return 'importlib' in filename and '_bootstrap' in filename


def _next_external_frame(frame):
    """Find the next frame that doesn't involve CPython internals."""
    frame = frame.f_back
    while frame is not None and _is_internal_frame(frame):
        frame = frame.f_back
    return frame


# Code typically replaced by _warnings
def warn(message, category=None, stacklevel=1, source=None):
    """Issue a warning, or maybe ignore it or raise an exception."""
    # Check if message is already a Warning object
    if isinstance(message, Warning):
        category = message.__class__
    # Check category argument
    if category is None:
        category = UserWarning
    if not (isinstance(category, type) and issubclass(category, Warning)):
        raise TypeError("category must be a Warning subclass, "
                        "not '{:s}'".format(type(category).__name__))
    # Get context information
    try:
        if stacklevel <= 1 or _is_internal_frame(sys._getframe(1)):
            # If frame is too small to care or if the warning originated in
            # internal code, then do not try to hide any frames.
            frame = sys._getframe(stacklevel)
        else:
            frame = sys._getframe(1)
            # Look for one frame less since the above line starts us off.
            for x in range(stacklevel-1):
                frame = _next_external_frame(frame)
                if frame is None:
                    raise ValueError
    except ValueError:
        globals = sys.__dict__
        filename = "sys"
        lineno = 1
    else:
        globals = frame.f_globals
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
    if '__name__' in globals:
        module = globals['__name__']
    else:
        module = "<string>"
    registry = globals.setdefault("__warningregistry__", {})
    warn_explicit(message, category, filename, lineno, module, registry,
                  globals, source)

def warn_explicit(message, category, filename, lineno,
                  module=None, registry=None, module_globals=None,
                  source=None):
    lineno = int(lineno)
    if module is None:
        module = filename or "<unknown>"
        if module[-3:].lower() == ".py":
            module = module[:-3] # XXX What about leading pathname?
    if registry is None:
        registry = {}
    if registry.get('version', 0) != _filters_version:
        registry.clear()
        registry['version'] = _filters_version
    if isinstance(message, Warning):
        text = str(message)
        category = message.__class__
    else:
        text = message
        message = category(message)
    key = (text, category, lineno)
    # Quick test for common case
    if registry.get(key):
        return
    # Search the filters
    for item in filters:
        action, msg, cat, mod, ln = item
        if ((msg is None or msg.match(text)) and
            issubclass(category, cat) and
            (mod is None or mod.match(module)) and
            (ln == 0 or lineno == ln)):
            break
    else:
        action = defaultaction
    # Early exit actions
    if action == "ignore":
        return

    # Prime the linecache for formatting, in case the
    # "file" is actually in a zipfile or something.
    import linecache
    linecache.getlines(filename, module_globals)

    if action == "error":
        raise message
    # Other actions
    if action == "once":
        registry[key] = 1
        oncekey = (text, category)
        if onceregistry.get(oncekey):
            return
        onceregistry[oncekey] = 1
    elif action == "always":
        pass
    elif action == "module":
        registry[key] = 1
        altkey = (text, category, 0)
        if registry.get(altkey):
            return
        registry[altkey] = 1
    elif action == "default":
        registry[key] = 1
    else:
        # Unrecognized actions are errors
        raise RuntimeError(
              "Unrecognized action (%r) in warnings.filters:\n %s" %
              (action, item))
    # Print message and context
    msg = WarningMessage(message, category, filename, lineno, source)
    _showwarnmsg(msg)


class WarningMessage(object):

    _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file",
                        "line", "source")

    def __init__(self, message, category, filename, lineno, file=None,
                 line=None, source=None):
        self.message = message
        self.category = category
        self.filename = filename
        self.lineno = lineno
        self.file = file
        self.line = line
        self.source = source
        self._category_name = category.__name__ if category else None

    def __str__(self):
        return ("{message : %r, category : %r, filename : %r, lineno : %s, "
                    "line : %r}" % (self.message, self._category_name,
                                    self.filename, self.lineno, self.line))


class catch_warnings(object):

    """A context manager that copies and restores the warnings filter upon
    exiting the context.

    The 'record' argument specifies whether warnings should be captured by a
    custom implementation of warnings.showwarning() and be appended to a list
    returned by the context manager. Otherwise None is returned by the context
    manager. The objects appended to the list are arguments whose attributes
    mirror the arguments to showwarning().

    The 'module' argument is to specify an alternative module to the module
    named 'warnings' and imported under that name. This argument is only useful
    when testing the warnings module itself.

    """

    def __init__(self, *, record=False, module=None):
        """Specify whether to record warnings and if an alternative module
        should be used other than sys.modules['warnings'].

        For compatibility with Python 3.0, please consider all arguments to be
        keyword-only.

        """
        self._record = record
        self._module = sys.modules['warnings'] if module is None else module
        self._entered = False

    def __repr__(self):
        args = []
        if self._record:
            args.append("record=True")
        if self._module is not sys.modules['warnings']:
            args.append("module=%r" % self._module)
        name = type(self).__name__
        return "%s(%s)" % (name, ", ".join(args))

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._module._filters_mutated()
        self._showwarning = self._module.showwarning
        self._showwarnmsg_impl = self._module._showwarnmsg_impl
        if self._record:
            log = []
            self._module._showwarnmsg_impl = log.append
            # Reset showwarning() to the default implementation to make sure
            # that _showwarnmsg() calls _showwarnmsg_impl()
            self._module.showwarning = self._module._showwarning_orig
            return log
        else:
            return None

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module._filters_mutated()
        self._module.showwarning = self._showwarning
        self._module._showwarnmsg_impl = self._showwarnmsg_impl


# Private utility function called by _PyErr_WarnUnawaitedCoroutine
def _warn_unawaited_coroutine(coro):
    msg_lines = [
        f"coroutine '{coro.__qualname__}' was never awaited\n"
    ]
    if coro.cr_origin is not None:
        import linecache, traceback
        def extract():
            for filename, lineno, funcname in reversed(coro.cr_origin):
                line = linecache.getline(filename, lineno)
                yield (filename, lineno, funcname, line)
        msg_lines.append("Coroutine created at (most recent call last)\n")
        msg_lines += traceback.format_list(list(extract()))
    msg = "".join(msg_lines).rstrip("\n")
    # Passing source= here means that if the user happens to have tracemalloc
    # enabled and tracking where the coroutine was created, the warning will
    # contain that traceback. This does mean that if they have *both*
    # coroutine origin tracking *and* tracemalloc enabled, they'll get two
    # partially-redundant tracebacks. If we wanted to be clever we could
    # probably detect this case and avoid it, but for now we don't bother.
    warn(msg, category=RuntimeWarning, stacklevel=2, source=coro)


# filters contains a sequence of filter 5-tuples
# The components of the 5-tuple are:
# - an action: error, ignore, always, default, module, or once
# - a compiled regex that must match the warning message
# - a class representing the warning category
# - a compiled regex that must match the module that is being warned
# - a line number for the line being warning, or 0 to mean any line
# If either if the compiled regexs are None, match anything.
try:
    from _warnings import (filters, _defaultaction, _onceregistry,
                           warn, warn_explicit, _filters_mutated)
    defaultaction = _defaultaction
    onceregistry = _onceregistry
    _warnings_defaults = True
except ImportError:
    filters = []
    defaultaction = "default"
    onceregistry = {}

    _filters_version = 1

    def _filters_mutated():
        global _filters_version
        _filters_version += 1

    _warnings_defaults = False


# Module initialization
_processoptions(sys.warnoptions)
if not _warnings_defaults:
    # Several warning categories are ignored by default in regular builds
    if not hasattr(sys, 'gettotalrefcount'):
        filterwarnings("default", category=DeprecationWarning,
                       module="__main__", append=1)
        simplefilter("ignore", category=DeprecationWarning, append=1)
        simplefilter("ignore", category=PendingDeprecationWarning, append=1)
        simplefilter("ignore", category=ImportWarning, append=1)
        simplefilter("ignore", category=ResourceWarning, append=1)

del _warnings_defaults
<|endoftext|>"""A POP3 client class.

Based on the J. Myers POP3 draft, Jan. 96
"""

# Author: David Ascher <david_ascher@brown.edu>
#         [heavily stealing from nntplib.py]
# Updated: Piers Lauder <piers@cs.su.oz.au> [Jul '97]
# String method conversion and test jig improvements by ESR, February 2001.
# Added the POP3_SSL class. Methods loosely based on IMAP_SSL. Hector Urtubia <urtubia@mrbook.org> Aug 2003

# Example (see the test function at the end of this file)

# Imports

import errno
import re
import socket
import sys

try:
    import ssl
    HAVE_SSL = True
except ImportError:
    HAVE_SSL = False

__all__ = ["POP3","error_proto"]

# Exception raised when an error or invalid response is received:

class error_proto(Exception): pass

# Standard Port
POP3_PORT = 110

# POP SSL PORT
POP3_SSL_PORT = 995

# Line terminators (we always output CRLF, but accept any of CRLF, LFCR, LF)
CR = b'\r'
LF = b'\n'
CRLF = CR+LF

# maximal line length when calling readline(). This is to prevent
# reading arbitrary length lines. RFC 1939 limits POP3 line length to
# 512 characters, including CRLF. We have selected 2048 just to be on
# the safe side.
_MAXLINE = 2048


class POP3:

    """This class supports both the minimal and optional command sets.
    Arguments can be strings or integers (where appropriate)
    (e.g.: retr(1) and retr('1') both work equally well.

    Minimal Command Set:
            USER name               user(name)
            PASS string             pass_(string)
            STAT                    stat()
            LIST [msg]              list(msg = None)
            RETR msg                retr(msg)
            DELE msg                dele(msg)
            NOOP                    noop()
            RSET                    rset()
            QUIT                    quit()

    Optional Commands (some servers support these):
            RPOP name               rpop(name)
            APOP name digest        apop(name, digest)
            TOP msg n               top(msg, n)
            UIDL [msg]              uidl(msg = None)
            CAPA                    capa()
            STLS                    stls()
            UTF8                    utf8()

    Raises one exception: 'error_proto'.

    Instantiate with:
            POP3(hostname, port=110)

    NB:     the POP protocol locks the mailbox from user
            authorization until QUIT, so be sure to get in, suck
            the messages, and quit, each time you access the
            mailbox.

            POP is a line-based protocol, which means large mail
            messages consume lots of python cycles reading them
            line-by-line.

            If it's available on your mail server, use IMAP4
            instead, it doesn't suffer from the two problems
            above.
    """

    encoding = 'UTF-8'

    def __init__(self, host, port=POP3_PORT,
                 timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        self.host = host
        self.port = port
        self._tls_established = False
        sys.audit("poplib.connect", self, host, port)
        self.sock = self._create_socket(timeout)
        self.file = self.sock.makefile('rb')
        self._debugging = 0
        self.welcome = self._getresp()

    def _create_socket(self, timeout):
        if timeout is not None and not timeout:
            raise ValueError('Non-blocking socket (timeout=0) is not supported')
        return socket.create_connection((self.host, self.port), timeout)

    def _putline(self, line):
        if self._debugging > 1: print('*put*', repr(line))
        sys.audit("poplib.putline", self, line)
        self.sock.sendall(line + CRLF)


    # Internal: send one command to the server (through _putline())

    def _putcmd(self, line):
        if self._debugging: print('*cmd*', repr(line))
        line = bytes(line, self.encoding)
        self._putline(line)


    # Internal: return one line from the server, stripping CRLF.
    # This is where all the CPU time of this module is consumed.
    # Raise error_proto('-ERR EOF') if the connection is closed.

    def _getline(self):
        line = self.file.readline(_MAXLINE + 1)
        if len(line) > _MAXLINE:
            raise error_proto('line too long')

        if self._debugging > 1: print('*get*', repr(line))
        if not line: raise error_proto('-ERR EOF')
        octets = len(line)
        # server can send any combination of CR & LF
        # however, 'readline()' returns lines ending in LF
        # so only possibilities are ...LF, ...CRLF, CR...LF
        if line[-2:] == CRLF:
            return line[:-2], octets
        if line[:1] == CR:
            return line[1:-1], octets
        return line[:-1], octets


    # Internal: get a response from the server.
    # Raise 'error_proto' if the response doesn't start with '+'.

    def _getresp(self):
        resp, o = self._getline()
        if self._debugging > 1: print('*resp*', repr(resp))
        if not resp.startswith(b'+'):
            raise error_proto(resp)
        return resp


    # Internal: get a response plus following text from the server.

    def _getlongresp(self):
        resp = self._getresp()
        list = []; octets = 0
        line, o = self._getline()
        while line != b'.':
            if line.startswith(b'..'):
                o = o-1
                line = line[1:]
            octets = octets + o
            list.append(line)
            line, o = self._getline()
        return resp, list, octets


    # Internal: send a command and get the response

    def _shortcmd(self, line):
        self._putcmd(line)
        return self._getresp()


    # Internal: send a command and get the response plus following text

    def _longcmd(self, line):
        self._putcmd(line)
        return self._getlongresp()


    # These can be useful:

    def getwelcome(self):
        return self.welcome


    def set_debuglevel(self, level):
        self._debugging = level


    # Here are all the POP commands:

    def user(self, user):
        """Send user name, return response

        (should indicate password required).
        """
        return self._shortcmd('USER %s' % user)


    def pass_(self, pswd):
        """Send password, return response

        (response includes message count, mailbox size).

        NB: mailbox is locked by server from here to 'quit()'
        """
        return self._shortcmd('PASS %s' % pswd)


    def stat(self):
        """Get mailbox status.

        Result is tuple of 2 ints (message count, mailbox size)
        """
        retval = self._shortcmd('STAT')
        rets = retval.split()
        if self._debugging: print('*stat*', repr(rets))
        numMessages = int(rets[1])
        sizeMessages = int(rets[2])
        return (numMessages, sizeMessages)


    def list(self, which=None):
        """Request listing, return result.

        Result without a message number argument is in form
        ['response', ['mesg_num octets', ...], octets].

        Result when a message number argument is given is a
        single response: the "scan listing" for that message.
        """
        if which is not None:
            return self._shortcmd('LIST %s' % which)
        return self._longcmd('LIST')


    def retr(self, which):
        """Retrieve whole message number 'which'.

        Result is in form ['response', ['line', ...], octets].
        """
        return self._longcmd('RETR %s' % which)


    def dele(self, which):
        """Delete message number 'which'.

        Result is 'response'.
        """
        return self._shortcmd('DELE %s' % which)


    def noop(self):
        """Does nothing.

        One supposes the response indicates the server is alive.
        """
        return self._shortcmd('NOOP')


    def rset(self):
        """Unmark all messages marked for deletion."""
        return self._shortcmd('RSET')


    def quit(self):
        """Signoff: commit changes on server, unlock mailbox, close connection."""
        resp = self._shortcmd('QUIT')
        self.close()
        return resp

    def close(self):
        """Close the connection without assuming anything about it."""
        try:
            file = self.file
            self.file = None
            if file is not None:
                file.close()
        finally:
            sock = self.sock
            self.sock = None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError as exc:
                    # The server might already have closed the connection.
                    # On Windows, this may result in WSAEINVAL (error 10022):
                    # An invalid operation was attempted.
                    if (exc.errno != errno.ENOTCONN
                       and getattr(exc, 'winerror', 0) != 10022):
                        raise
                finally:
                    sock.close()

    #__del__ = quit


    # optional commands:

    def rpop(self, user):
        """Not sure what this does."""
        return self._shortcmd('RPOP %s' % user)


    timestamp = re.compile(br'\+OK.[^<]*(<.*>)')

    def apop(self, user, password):
        """Authorisation

        - only possible if server has supplied a timestamp in initial greeting.

        Args:
                user     - mailbox user;
                password - mailbox password.

        NB: mailbox is locked by server from here to 'quit()'
        """
        secret = bytes(password, self.encoding)
        m = self.timestamp.match(self.welcome)
        if not m:
            raise error_proto('-ERR APOP not supported by server')
        import hashlib
        digest = m.group(1)+secret
        digest = hashlib.md5(digest).hexdigest()
        return self._shortcmd('APOP %s %s' % (user, digest))


    def top(self, which, howmuch):
        """Retrieve message header of message number 'which'
        and first 'howmuch' lines of message body.

        Result is in form ['response', ['line', ...], octets].
        """
        return self._longcmd('TOP %s %s' % (which, howmuch))


    def uidl(self, which=None):
        """Return message digest (unique id) list.

        If 'which', result contains unique id for that message
        in the form 'response mesgnum uid', otherwise result is
        the list ['response', ['mesgnum uid', ...], octets]
        """
        if which is not None:
            return self._shortcmd('UIDL %s' % which)
        return self._longcmd('UIDL')


    def utf8(self):
        """Try to enter UTF-8 mode (see RFC 6856). Returns server response.
        """
        return self._shortcmd('UTF8')


    def capa(self):
        """Return server capabilities (RFC 2449) as a dictionary
        >>> c=poplib.POP3('localhost')
        >>> c.capa()
        {'IMPLEMENTATION': ['Cyrus', 'POP3', 'server', 'v2.2.12'],
         'TOP': [], 'LOGIN-DELAY': ['0'], 'AUTH-RESP-CODE': [],
         'EXPIRE': ['NEVER'], 'USER': [], 'STLS': [], 'PIPELINING': [],
         'UIDL': [], 'RESP-CODES': []}
        >>>

        Really, according to RFC 2449, the cyrus folks should avoid
        having the implementation split into multiple arguments...
        """
        def _parsecap(line):
            lst = line.decode('ascii').split()
            return lst[0], lst[1:]

        caps = {}
        try:
            resp = self._longcmd('CAPA')
            rawcaps = resp[1]
            for capline in rawcaps:
                capnm, capargs = _parsecap(capline)
                caps[capnm] = capargs
        except error_proto:
            raise error_proto('-ERR CAPA not supported by server')
        return caps


    def stls(self, context=None):
        """Start a TLS session on the active connection as specified in RFC 2595.

                context - a ssl.SSLContext
        """
        if not HAVE_SSL:
            raise error_proto('-ERR TLS support missing')
        if self._tls_established:
            raise error_proto('-ERR TLS session already established')
        caps = self.capa()
        if not 'STLS' in caps:
            raise error_proto('-ERR STLS not supported by server')
        if context is None:
            context = ssl._create_stdlib_context()
        resp = self._shortcmd('STLS')
        self.sock = context.wrap_socket(self.sock,
                                        server_hostname=self.host)
        self.file = self.sock.makefile('rb')
        self._tls_established = True
        return resp


if HAVE_SSL:

    class POP3_SSL(POP3):
        """POP3 client class over SSL connection

        Instantiate with: POP3_SSL(hostname, port=995, keyfile=None, certfile=None,
                                   context=None)

               hostname - the hostname of the pop3 over ssl server
               port - port number
               keyfile - PEM formatted file that contains your private key
               certfile - PEM formatted certificate chain file
               context - a ssl.SSLContext

        See the methods of the parent class POP3 for more documentation.
        """

        def __init__(self, host, port=POP3_SSL_PORT, keyfile=None, certfile=None,
                     timeout=socket._GLOBAL_DEFAULT_TIMEOUT, context=None):
            if context is not None and keyfile is not None:
                raise ValueError("context and keyfile arguments are mutually "
                                 "exclusive")
            if context is not None and certfile is not None:
                raise ValueError("context and certfile arguments are mutually "
                                 "exclusive")
            if keyfile is not None or certfile is not None:
                import warnings
                warnings.warn("keyfile and certfile are deprecated, use a "
                              "custom context instead", DeprecationWarning, 2)
            self.keyfile = keyfile
            self.certfile = certfile
            if context is None:
                context = ssl._create_stdlib_context(certfile=certfile,
                                                     keyfile=keyfile)
            self.context = context
            POP3.__init__(self, host, port, timeout)

        def _create_socket(self, timeout):
            sock = POP3._create_socket(self, timeout)
            sock = self.context.wrap_socket(sock,
                                            server_hostname=self.host)
            return sock

        def stls(self, keyfile=None, certfile=None, context=None):
            """The method unconditionally raises an exception since the
            STLS command doesn't make any sense on an already established
            SSL/TLS session.
            """
            raise error_proto('-ERR TLS session already established')

    __all__.append("POP3_SSL")

if __name__ == "__main__":
    import sys
    a = POP3(sys.argv[1])
    print(a.getwelcome())
    a.user(sys.argv[2])
    a.pass_(sys.argv[3])
    a.list()
    (numMsgs, totalSize) = a.stat()
    for i in range(1, numMsgs + 1):
        (header, msg, octets) = a.retr(i)
        print("Message %d:" % i)
        for line in msg:
            print('   ' + line)
        print('-----------------------')
    a.quit()
<|endoftext|>"""
Python implementation of the io module.
"""

import os
import abc
import codecs
import errno
import stat
import sys
# Import _thread instead of threading to reduce startup cost
from _thread import allocate_lock as Lock
if sys.platform in {'win32', 'cygwin'}:
    from msvcrt import setmode as _setmode
else:
    _setmode = None

import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)

valid_seek_flags = {0, 1, 2}  # Hardwired values
if hasattr(os, 'SEEK_HOLE') :
    valid_seek_flags.add(os.SEEK_HOLE)
    valid_seek_flags.add(os.SEEK_DATA)

# open() uses st_blksize whenever we can
DEFAULT_BUFFER_SIZE = 8 * 1024  # bytes

# NOTE: Base classes defined here are registered with the "official" ABCs
# defined in io.py. We don't use real inheritance though, because we don't want
# to inherit the C implementations.

# Rebind for compatibility
BlockingIOError = BlockingIOError

# Does io.IOBase finalizer log the exception if the close() method fails?
# The exception is ignored silently by default in release build.
_IOBASE_EMITS_UNRAISABLE = (hasattr(sys, "gettotalrefcount") or sys.flags.dev_mode)
# Does open() check its 'errors' argument?
_CHECK_ERRORS = _IOBASE_EMITS_UNRAISABLE


def text_encoding(encoding, stacklevel=2):
    """
    A helper function to choose the text encoding.

    When encoding is not None, just return it.
    Otherwise, return the default text encoding (i.e. "locale").

    This function emits an EncodingWarning if *encoding* is None and
    sys.flags.warn_default_encoding is true.

    This can be used in APIs with an encoding=None parameter
    that pass it to TextIOWrapper or open.
    However, please consider using encoding="utf-8" for new APIs.
    """
    if encoding is None:
        encoding = "locale"
        if sys.flags.warn_default_encoding:
            import warnings
            warnings.warn("'encoding' argument not specified.",
                          EncodingWarning, stacklevel + 1)
    return encoding


# Wrapper for builtins.open
#
# Trick so that open() won't become a bound method when stored
# as a class variable (as dbm.dumb does).
#
# See init_set_builtins_open() in Python/pylifecycle.c.
@staticmethod
def open(file, mode="r", buffering=-1, encoding=None, errors=None,
         newline=None, closefd=True, opener=None):

    r"""Open file and return a stream.  Raise OSError upon failure.

    file is either a text or byte string giving the name (and the path
    if the file isn't in the current working directory) of the file to
    be opened or an integer file descriptor of the file to be
    wrapped. (If a file descriptor is given, it is closed when the
    returned I/O object is closed, unless closefd is set to False.)

    mode is an optional string that specifies the mode in which the file is
    opened. It defaults to 'r' which means open for reading in text mode. Other
    common values are 'w' for writing (truncating the file if it already
    exists), 'x' for exclusive creation of a new file, and 'a' for appending
    (which on some Unix systems, means that all writes append to the end of the
    file regardless of the current seek position). In text mode, if encoding is
    not specified the encoding used is platform dependent. (For reading and
    writing raw bytes use binary mode and leave encoding unspecified.) The
    available modes are:

    ========= ===============================================================
    Character Meaning
    --------- ---------------------------------------------------------------
    'r'       open for reading (default)
    'w'       open for writing, truncating the file first
    'x'       create a new file and open it for writing
    'a'       open for writing, appending to the end of the file if it exists
    'b'       binary mode
    't'       text mode (default)
    '+'       open a disk file for updating (reading and writing)
    'U'       universal newline mode (deprecated)
    ========= ===============================================================

    The default mode is 'rt' (open for reading text). For binary random
    access, the mode 'w+b' opens and truncates the file to 0 bytes, while
    'r+b' opens the file without truncation. The 'x' mode implies 'w' and
    raises an `FileExistsError` if the file already exists.

    Python distinguishes between files opened in binary and text modes,
    even when the underlying operating system doesn't. Files opened in
    binary mode (appending 'b' to the mode argument) return contents as
    bytes objects without any decoding. In text mode (the default, or when
    't' is appended to the mode argument), the contents of the file are
    returned as strings, the bytes having been first decoded using a
    platform-dependent encoding or using the specified encoding if given.

    'U' mode is deprecated and will raise an exception in future versions
    of Python.  It has no effect in Python 3.  Use newline to control
    universal newlines mode.

    buffering is an optional integer used to set the buffering policy.
    Pass 0 to switch buffering off (only allowed in binary mode), 1 to select
    line buffering (only usable in text mode), and an integer > 1 to indicate
    the size of a fixed-size chunk buffer.  When no buffering argument is
    given, the default buffering policy works as follows:

    * Binary files are buffered in fixed-size chunks; the size of the buffer
      is chosen using a heuristic trying to determine the underlying device's
      "block size" and falling back on `io.DEFAULT_BUFFER_SIZE`.
      On many systems, the buffer will typically be 4096 or 8192 bytes long.

    * "Interactive" text files (files for which isatty() returns True)
      use line buffering.  Other text files use the policy described above
      for binary files.

    encoding is the str name of the encoding used to decode or encode the
    file. This should only be used in text mode. The default encoding is
    platform dependent, but any encoding supported by Python can be
    passed.  See the codecs module for the list of supported encodings.

    errors is an optional string that specifies how encoding errors are to
    be handled---this argument should not be used in binary mode. Pass
    'strict' to raise a ValueError exception if there is an encoding error
    (the default of None has the same effect), or pass 'ignore' to ignore
    errors. (Note that ignoring encoding errors can lead to data loss.)
    See the documentation for codecs.register for a list of the permitted
    encoding error strings.

    newline is a string controlling how universal newlines works (it only
    applies to text mode). It can be None, '', '\n', '\r', and '\r\n'.  It works
    as follows:

    * On input, if newline is None, universal newlines mode is
      enabled. Lines in the input can end in '\n', '\r', or '\r\n', and
      these are translated into '\n' before being returned to the
      caller. If it is '', universal newline mode is enabled, but line
      endings are returned to the caller untranslated. If it has any of
      the other legal values, input lines are only terminated by the given
      string, and the line ending is returned to the caller untranslated.

    * On output, if newline is None, any '\n' characters written are
      translated to the system default line separator, os.linesep. If
      newline is '', no translation takes place. If newline is any of the
      other legal values, any '\n' characters written are translated to
      the given string.

    closedfd is a bool. If closefd is False, the underlying file descriptor will
    be kept open when the file is closed. This does not work when a file name is
    given and must be True in that case.

    The newly created file is non-inheritable.

    A custom opener can be used by passing a callable as *opener*. The
    underlying file descriptor for the file object is then obtained by calling
    *opener* with (*file*, *flags*). *opener* must return an open file
    descriptor (passing os.open as *opener* results in functionality similar to
    passing None).

    open() returns a file object whose type depends on the mode, and
    through which the standard file operations such as reading and writing
    are performed. When open() is used to open a file in a text mode ('w',
    'r', 'wt', 'rt', etc.), it returns a TextIOWrapper. When used to open
    a file in a binary mode, the returned class varies: in read binary
    mode, it returns a BufferedReader; in write binary and append binary
    modes, it returns a BufferedWriter, and in read/write mode, it returns
    a BufferedRandom.

    It is also possible to use a string or bytearray as a file for both
    reading and writing. For strings StringIO can be used like a file
    opened in a text mode, and for bytes a BytesIO can be used like a file
    opened in a binary mode.
    """
    if not isinstance(file, int):
        file = os.fspath(file)
    if not isinstance(file, (str, bytes, int)):
        raise TypeError("invalid file: %r" % file)
    if not isinstance(mode, str):
        raise TypeError("invalid mode: %r" % mode)
    if not isinstance(buffering, int):
        raise TypeError("invalid buffering: %r" % buffering)
    if encoding is not None and not isinstance(encoding, str):
        raise TypeError("invalid encoding: %r" % encoding)
    if errors is not None and not isinstance(errors, str):
        raise TypeError("invalid errors: %r" % errors)
    modes = set(mode)
    if modes - set("axrwb+tU") or len(mode) > len(modes):
        raise ValueError("invalid mode: %r" % mode)
    creating = "x" in modes
    reading = "r" in modes
    writing = "w" in modes
    appending = "a" in modes
    updating = "+" in modes
    text = "t" in modes
    binary = "b" in modes
    if "U" in modes:
        if creating or writing or appending or updating:
            raise ValueError("mode U cannot be combined with 'x', 'w', 'a', or '+'")
        import warnings
        warnings.warn("'U' mode is deprecated",
                      DeprecationWarning, 2)
        reading = True
    if text and binary:
        raise ValueError("can't have text and binary mode at once")
    if creating + reading + writing + appending > 1:
        raise ValueError("can't have read/write/append mode at once")
    if not (creating or reading or writing or appending):
        raise ValueError("must have exactly one of read/write/append mode")
    if binary and encoding is not None:
        raise ValueError("binary mode doesn't take an encoding argument")
    if binary and errors is not None:
        raise ValueError("binary mode doesn't take an errors argument")
    if binary and newline is not None:
        raise ValueError("binary mode doesn't take a newline argument")
    if binary and buffering == 1:
        import warnings
        warnings.warn("line buffering (buffering=1) isn't supported in binary "
                      "mode, the default buffer size will be used",
                      RuntimeWarning, 2)
    raw = FileIO(file,
                 (creating and "x" or "") +
                 (reading and "r" or "") +
                 (writing and "w" or "") +
                 (appending and "a" or "") +
                 (updating and "+" or ""),
                 closefd, opener=opener)
    result = raw
    try:
        line_buffering = False
        if buffering == 1 or buffering < 0 and raw.isatty():
            buffering = -1
            line_buffering = True
        if buffering < 0:
            buffering = DEFAULT_BUFFER_SIZE
            try:
                bs = os.fstat(raw.fileno()).st_blksize
            except (OSError, AttributeError):
                pass
            else:
                if bs > 1:
                    buffering = bs
        if buffering < 0:
            raise ValueError("invalid buffering size")
        if buffering == 0:
            if binary:
                return result
            raise ValueError("can't have unbuffered text I/O")
        if updating:
            buffer = BufferedRandom(raw, buffering)
        elif creating or writing or appending:
            buffer = BufferedWriter(raw, buffering)
        elif reading:
            buffer = BufferedReader(raw, buffering)
        else:
            raise ValueError("unknown mode: %r" % mode)
        result = buffer
        if binary:
            return result
        encoding = text_encoding(encoding)
        text = TextIOWrapper(buffer, encoding, errors, newline, line_buffering)
        result = text
        text.mode = mode
        return result
    except:
        result.close()
        raise

# Define a default pure-Python implementation for open_code()
# that does not allow hooks. Warn on first use. Defined for tests.
def _open_code_with_warning(path):
    """Opens the provided file with mode ``'rb'``. This function
    should be used when the intent is to treat the contents as
    executable code.

    ``path`` should be an absolute path.

    When supported by the runtime, this function can be hooked
    in order to allow embedders more control over code files.
    This functionality is not supported on the current runtime.
    """
    import warnings
    warnings.warn("_pyio.open_code() may not be using hooks",
                  RuntimeWarning, 2)
    return open(path, "rb")

try:
    open_code = io.open_code
except AttributeError:
    open_code = _open_code_with_warning


def __getattr__(name):
    if name == "OpenWrapper":
        # bpo-43680: Until Python 3.9, _pyio.open was not a static method and
        # builtins.open was set to OpenWrapper to not become a bound method
        # when set to a class variable. _io.open is a built-in function whereas
        # _pyio.open is a Python function. In Python 3.10, _pyio.open() is now
        # a static method, and builtins.open() is now io.open().
        import warnings
        warnings.warn('OpenWrapper is deprecated, use open instead',
                      DeprecationWarning, stacklevel=2)
        global OpenWrapper
        OpenWrapper = open
        return OpenWrapper
    raise AttributeError(name)


# In normal operation, both `UnsupportedOperation`s should be bound to the
# same object.
try:
    UnsupportedOperation = io.UnsupportedOperation
except AttributeError:
    class UnsupportedOperation(OSError, ValueError):
        pass


class IOBase(metaclass=abc.ABCMeta):

    """The abstract base class for all I/O classes.

    This class provides dummy implementations for many methods that
    derived classes can override selectively; the default implementations
    represent a file that cannot be read, written or seeked.

    Even though IOBase does not declare read or write because
    their signatures will vary, implementations and clients should
    consider those methods part of the interface. Also, implementations
    may raise UnsupportedOperation when operations they do not support are
    called.

    The basic type used for binary data read from or written to a file is
    bytes. Other bytes-like objects are accepted as method arguments too.
    Text I/O classes work with str data.

    Note that calling any method (even inquiries) on a closed stream is
    undefined. Implementations may raise OSError in this case.

    IOBase (and its subclasses) support the iterator protocol, meaning
    that an IOBase object can be iterated over yielding the lines in a
    stream.

    IOBase also supports the :keyword:`with` statement. In this example,
    fp is closed after the suite of the with statement is complete:

    with open('spam.txt', 'r') as fp:
        fp.write('Spam and eggs!')
    """

    ### Internal ###

    def _unsupported(self, name):
        """Internal: raise an OSError exception for unsupported operations."""
        raise UnsupportedOperation("%s.%s() not supported" %
                                   (self.__class__.__name__, name))

    ### Positioning ###

    def seek(self, pos, whence=0):
        """Change stream position.

        Change the stream position to byte offset pos. Argument pos is
        interpreted relative to the position indicated by whence.  Values
        for whence are ints:

        * 0 -- start of stream (the default); offset should be zero or positive
        * 1 -- current stream position; offset may be negative
        * 2 -- end of stream; offset is usually negative
        Some operating systems / file systems could provide additional values.

        Return an int indicating the new absolute position.
        """
        self._unsupported("seek")

    def tell(self):
        """Return an int indicating the current stream position."""
        return self.seek(0, 1)

    def truncate(self, pos=None):
        """Truncate file to size bytes.

        Size defaults to the current IO position as reported by tell().  Return
        the new size.
        """
        self._unsupported("truncate")

    ### Flush and close ###

    def flush(self):
        """Flush write buffers, if applicable.

        This is not implemented for read-only and non-blocking streams.
        """
        self._checkClosed()
        # XXX Should this return the number of bytes written???

    __closed = False

    def close(self):
        """Flush and close the IO object.

        This method has no effect if the file is already closed.
        """
        if not self.__closed:
            try:
                self.flush()
            finally:
                self.__closed = True

    def __del__(self):
        """Destructor.  Calls close()."""
        try:
            closed = self.closed
        except AttributeError:
            # If getting closed fails, then the object is probably
            # in an unusable state, so ignore.
            return

        if closed:
            return

        if _IOBASE_EMITS_UNRAISABLE:
            self.close()
        else:
            # The try/except block is in case this is called at program
            # exit time, when it's possible that globals have already been
            # deleted, and then the close() call might fail.  Since
            # there's nothing we can do about such failures and they annoy
            # the end users, we suppress the traceback.
            try:
                self.close()
            except:
                pass

    ### Inquiries ###

    def seekable(self):
        """Return a bool indicating whether object supports random access.

        If False, seek(), tell() and truncate() will raise OSError.
        This method may need to do a test seek().
        """
        return False

    def _checkSeekable(self, msg=None):
        """Internal: raise UnsupportedOperation if file is not seekable
        """
        if not self.seekable():
            raise UnsupportedOperation("File or stream is not seekable."
                                       if msg is None else msg)

    def readable(self):
        """Return a bool indicating whether object was opened for reading.

        If False, read() will raise OSError.
        """
        return False

    def _checkReadable(self, msg=None):
        """Internal: raise UnsupportedOperation if file is not readable
        """
        if not self.readable():
            raise UnsupportedOperation("File or stream is not readable."
                                       if msg is None else msg)

    def writable(self):
        """Return a bool indicating whether object was opened for writing.

        If False, write() and truncate() will raise OSError.
        """
        return False

    def _checkWritable(self, msg=None):
        """Internal: raise UnsupportedOperation if file is not writable
        """
        if not self.writable():
            raise UnsupportedOperation("File or stream is not writable."
                                       if msg is None else msg)

    @property
    def closed(self):
        """closed: bool.  True iff the file has been closed.

        For backwards compatibility, this is a property, not a predicate.
        """
        return self.__closed

    def _checkClosed(self, msg=None):
        """Internal: raise a ValueError if file is closed
        """
        if self.closed:
            raise ValueError("I/O operation on closed file."
                             if msg is None else msg)

    ### Context manager ###

    def __enter__(self):  # That's a forward reference
        """Context management protocol.  Returns self (an instance of IOBase)."""
        self._checkClosed()
        return self

    def __exit__(self, *args):
        """Context management protocol.  Calls close()"""
        self.close()

    ### Lower-level APIs ###

    # XXX Should these be present even if unimplemented?

    def fileno(self):
        """Returns underlying file descriptor (an int) if one exists.

        An OSError is raised if the IO object does not use a file descriptor.
        """
        self._unsupported("fileno")

    def isatty(self):
        """Return a bool indicating whether this is an 'interactive' stream.

        Return False if it can't be determined.
        """
        self._checkClosed()
        return False

    ### Readline[s] and writelines ###

    def readline(self, size=-1):
        r"""Read and return a line of bytes from the stream.

        If size is specified, at most size bytes will be read.
        Size should be an int.

        The line terminator is always b'\n' for binary files; for text
        files, the newlines argument to open can be used to select the line
        terminator(s) recognized.
        """
        # For backwards compatibility, a (slowish) readline().
        if hasattr(self, "peek"):
            def nreadahead():
                readahead = self.peek(1)
                if not readahead:
                    return 1
                n = (readahead.find(b"\n") + 1) or len(readahead)
                if size >= 0:
                    n = min(n, size)
                return n
        else:
            def nreadahead():
                return 1
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f"{size!r} is not an integer")
            else:
                size = size_index()
        res = bytearray()
        while size < 0 or len(res) < size:
            b = self.read(nreadahead())
            if not b:
                break
            res += b
            if res.endswith(b"\n"):
                break
        return bytes(res)

    def __iter__(self):
        self._checkClosed()
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def readlines(self, hint=None):
        """Return a list of lines from the stream.

        hint can be specified to control the number of lines read: no more
        lines will be read if the total size (in bytes/characters) of all
        lines so far exceeds hint.
        """
        if hint is None or hint <= 0:
            return list(self)
        n = 0
        lines = []
        for line in self:
            lines.append(line)
            n += len(line)
            if n >= hint:
                break
        return lines

    def writelines(self, lines):
        """Write a list of lines to the stream.

        Line separators are not added, so it is usual for each of the lines
        provided to have a line separator at the end.
        """
        self._checkClosed()
        for line in lines:
            self.write(line)

io.IOBase.register(IOBase)


class RawIOBase(IOBase):

    """Base class for raw binary I/O."""

    # The read() method is implemented by calling readinto(); derived
    # classes that want to support read() only need to implement
    # readinto() as a primitive operation.  In general, readinto() can be
    # more efficient than read().

    # (It would be tempting to also provide an implementation of
    # readinto() in terms of read(), in case the latter is a more suitable
    # primitive operation, but that would lead to nasty recursion in case
    # a subclass doesn't implement either.)

    def read(self, size=-1):
        """Read and return up to size bytes, where size is an int.

        Returns an empty bytes object on EOF, or None if the object is
        set not to block and has no data to read.
        """
        if size is None:
            size = -1
        if size < 0:
            return self.readall()
        b = bytearray(size.__index__())
        n = self.readinto(b)
        if n is None:
            return None
        del b[n:]
        return bytes(b)

    def readall(self):
        """Read until EOF, using multiple read() call."""
        res = bytearray()
        while True:
            data = self.read(DEFAULT_BUFFER_SIZE)
            if not data:
                break
            res += data
        if res:
            return bytes(res)
        else:
            # b'' or None
            return data

    def readinto(self, b):
        """Read bytes into a pre-allocated bytes-like object b.

        Returns an int representing the number of bytes read (0 for EOF), or
        None if the object is set not to block and has no data to read.
        """
        self._unsupported("readinto")

    def write(self, b):
        """Write the given buffer to the IO stream.

        Returns the number of bytes written, which may be less than the
        length of b in bytes.
        """
        self._unsupported("write")

io.RawIOBase.register(RawIOBase)
from _io import FileIO
RawIOBase.register(FileIO)


class BufferedIOBase(IOBase):

    """Base class for buffered IO objects.

    The main difference with RawIOBase is that the read() method
    supports omitting the size argument, and does not have a default
    implementation that defers to readinto().

    In addition, read(), readinto() and write() may raise
    BlockingIOError if the underlying raw stream is in non-blocking
    mode and not ready; unlike their raw counterparts, they will never
    return None.

    A typical implementation should not inherit from a RawIOBase
    implementation, but wrap one.
    """

    def read(self, size=-1):
        """Read and return up to size bytes, where size is an int.

        If the argument is omitted, None, or negative, reads and
        returns all data until EOF.

        If the argument is positive, and the underlying raw stream is
        not 'interactive', multiple raw reads may be issued to satisfy
        the byte count (unless EOF is reached first).  But for
        interactive raw streams (XXX and for pipes?), at most one raw
        read will be issued, and a short result does not imply that
        EOF is imminent.

        Returns an empty bytes array on EOF.

        Raises BlockingIOError if the underlying raw stream has no
        data at the moment.
        """
        self._unsupported("read")

    def read1(self, size=-1):
        """Read up to size bytes with at most one read() system call,
        where size is an int.
        """
        self._unsupported("read1")

    def readinto(self, b):
        """Read bytes into a pre-allocated bytes-like object b.

        Like read(), this may issue multiple reads to the underlying raw
        stream, unless the latter is 'interactive'.

        Returns an int representing the number of bytes read (0 for EOF).

        Raises BlockingIOError if the underlying raw stream has no
        data at the moment.
        """

        return self._readinto(b, read1=False)

    def readinto1(self, b):
        """Read bytes into buffer *b*, using at most one system call

        Returns an int representing the number of bytes read (0 for EOF).

        Raises BlockingIOError if the underlying raw stream has no
        data at the moment.
        """

        return self._readinto(b, read1=True)

    def _readinto(self, b, read1):
        if not isinstance(b, memoryview):
            b = memoryview(b)
        b = b.cast('B')

        if read1:
            data = self.read1(len(b))
        else:
            data = self.read(len(b))
        n = len(data)

        b[:n] = data

        return n

    def write(self, b):
        """Write the given bytes buffer to the IO stream.

        Return the number of bytes written, which is always the length of b
        in bytes.

        Raises BlockingIOError if the buffer is full and the
        underlying raw stream cannot accept more data at the moment.
        """
        self._unsupported("write")

    def detach(self):
        """
        Separate the underlying raw stream from the buffer and return it.

        After the raw stream has been detached, the buffer is in an unusable
        state.
        """
        self._unsupported("detach")

io.BufferedIOBase.register(BufferedIOBase)


class _BufferedIOMixin(BufferedIOBase):

    """A mixin implementation of BufferedIOBase with an underlying raw stream.

    This passes most requests on to the underlying raw stream.  It
    does *not* provide implementations of read(), readinto() or
    write().
    """

    def __init__(self, raw):
        self._raw = raw

    ### Positioning ###

    def seek(self, pos, whence=0):
        new_position = self.raw.seek(pos, whence)
        if new_position < 0:
            raise OSError("seek() returned an invalid position")
        return new_position

    def tell(self):
        pos = self.raw.tell()
        if pos < 0:
            raise OSError("tell() returned an invalid position")
        return pos

    def truncate(self, pos=None):
        self._checkClosed()
        self._checkWritable()

        # Flush the stream.  We're mixing buffered I/O with lower-level I/O,
        # and a flush may be necessary to synch both views of the current
        # file state.
        self.flush()

        if pos is None:
            pos = self.tell()
        # XXX: Should seek() be used, instead of passing the position
        # XXX  directly to truncate?
        return self.raw.truncate(pos)

    ### Flush and close ###

    def flush(self):
        if self.closed:
            raise ValueError("flush on closed file")
        self.raw.flush()

    def close(self):
        if self.raw is not None and not self.closed:
            try:
                # may raise BlockingIOError or BrokenPipeError etc
                self.flush()
            finally:
                self.raw.close()

    def detach(self):
        if self.raw is None:
            raise ValueError("raw stream already detached")
        self.flush()
        raw = self._raw
        self._raw = None
        return raw

    ### Inquiries ###

    def seekable(self):
        return self.raw.seekable()

    @property
    def raw(self):
        return self._raw

    @property
    def closed(self):
        return self.raw.closed

    @property
    def name(self):
        return self.raw.name

    @property
    def mode(self):
        return self.raw.mode

    def __getstate__(self):
        raise TypeError(f"cannot pickle {self.__class__.__name__!r} object")

    def __repr__(self):
        modname = self.__class__.__module__
        clsname = self.__class__.__qualname__
        try:
            name = self.name
        except AttributeError:
            return "<{}.{}>".format(modname, clsname)
        else:
            return "<{}.{} name={!r}>".format(modname, clsname, name)

    ### Lower-level APIs ###

    def fileno(self):
        return self.raw.fileno()

    def isatty(self):
        return self.raw.isatty()


class BytesIO(BufferedIOBase):

    """Buffered I/O implementation using an in-memory bytes buffer."""

    # Initialize _buffer as soon as possible since it's used by __del__()
    # which calls close()
    _buffer = None

    def __init__(self, initial_bytes=None):
        buf = bytearray()
        if initial_bytes is not None:
            buf += initial_bytes
        self._buffer = buf
        self._pos = 0

    def __getstate__(self):
        if self.closed:
            raise ValueError("__getstate__ on closed file")
        return self.__dict__.copy()

    def getvalue(self):
        """Return the bytes value (contents) of the buffer
        """
        if self.closed:
            raise ValueError("getvalue on closed file")
        return bytes(self._buffer)

    def getbuffer(self):
        """Return a readable and writable view of the buffer.
        """
        if self.closed:
            raise ValueError("getbuffer on closed file")
        return memoryview(self._buffer)

    def close(self):
        if self._buffer is not None:
            self._buffer.clear()
        super().close()

    def read(self, size=-1):
        if self.closed:
            raise ValueError("read from closed file")
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f"{size!r} is not an integer")
            else:
                size = size_index()
        if size < 0:
            size = len(self._buffer)
        if len(self._buffer) <= self._pos:
            return b""
        newpos = min(len(self._buffer), self._pos + size)
        b = self._buffer[self._pos : newpos]
        self._pos = newpos
        return bytes(b)

    def read1(self, size=-1):
        """This is the same as read.
        """
        return self.read(size)

    def write(self, b):
        if self.closed:
            raise ValueError("write to closed file")
        if isinstance(b, str):
            raise TypeError("can't write str to binary stream")
        with memoryview(b) as view:
            n = view.nbytes  # Size of any bytes-like object
        if n == 0:
            return 0
        pos = self._pos
        if pos > len(self._buffer):
            # Inserts null bytes between the current end of the file
            # and the new write position.
            padding = b'\x00' * (pos - len(self._buffer))
            self._buffer += padding
        self._buffer[pos:pos + n] = b
        self._pos += n
        return n

    def seek(self, pos, whence=0):
        if self.closed:
            raise ValueError("seek on closed file")
        try:
            pos_index = pos.__index__
        except AttributeError:
            raise TypeError(f"{pos!r} is not an integer")
        else:
            pos = pos_index()
        if whence == 0:
            if pos < 0:
                raise ValueError("negative seek position %r" % (pos,))
            self._pos = pos
        elif whence == 1:
            self._pos = max(0, self._pos + pos)
        elif whence == 2:
            self._pos = max(0, len(self._buffer) + pos)
        else:
            raise ValueError("unsupported whence value")
        return self._pos

    def tell(self):
        if self.closed:
            raise ValueError("tell on closed file")
        return self._pos

    def truncate(self, pos=None):
        if self.closed:
            raise ValueError("truncate on closed file")
        if pos is None:
            pos = self._pos
        else:
            try:
                pos_index = pos.__index__
            except AttributeError:
                raise TypeError(f"{pos!r} is not an integer")
            else:
                pos = pos_index()
            if pos < 0:
                raise ValueError("negative truncate position %r" % (pos,))
        del self._buffer[pos:]
        return pos

    def readable(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return True

    def writable(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return True

    def seekable(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return True


class BufferedReader(_BufferedIOMixin):

    """BufferedReader(raw[, buffer_size])

    A buffer for a readable, sequential BaseRawIO object.

    The constructor creates a BufferedReader for the given readable raw
    stream and buffer_size. If buffer_size is omitted, DEFAULT_BUFFER_SIZE
    is used.
    """

    def __init__(self, raw, buffer_size=DEFAULT_BUFFER_SIZE):
        """Create a new buffered reader using the given readable raw IO object.
        """
        if not raw.readable():
            raise OSError('"raw" argument must be readable.')

        _BufferedIOMixin.__init__(self, raw)
        if buffer_size <= 0:
            raise ValueError("invalid buffer size")
        self.buffer_size = buffer_size
        self._reset_read_buf()
        self._read_lock = Lock()

    def readable(self):
        return self.raw.readable()

    def _reset_read_buf(self):
        self._read_buf = b""
        self._read_pos = 0

    def read(self, size=None):
        """Read size bytes.

        Returns exactly size bytes of data unless the underlying raw IO
        stream reaches EOF or if the call would block in non-blocking
        mode. If size is negative, read until EOF or until read() would
        block.
        """
        if size is not None and size < -1:
            raise ValueError("invalid number of bytes to read")
        with self._read_lock:
            return self._read_unlocked(size)

    def _read_unlocked(self, n=None):
        nodata_val = b""
        empty_values = (b"", None)
        buf = self._read_buf
        pos = self._read_pos

        # Special case for when the number of bytes to read is unspecified.
        if n is None or n == -1:
            self._reset_read_buf()
            if hasattr(self.raw, 'readall'):
                chunk = self.raw.readall()
                if chunk is None:
                    return buf[pos:] or None
                else:
                    return buf[pos:] + chunk
            chunks = [buf[pos:]]  # Strip the consumed bytes.
            current_size = 0
            while True:
                # Read until EOF or until read() would block.
                chunk = self.raw.read()
                if chunk in empty_values:
                    nodata_val = chunk
                    break
                current_size += len(chunk)
                chunks.append(chunk)
            return b"".join(chunks) or nodata_val

        # The number of bytes to read is specified, return at most n bytes.
        avail = len(buf) - pos  # Length of the available buffered data.
        if n <= avail:
            # Fast path: the data to read is fully buffered.
            self._read_pos += n
            return buf[pos:pos+n]
        # Slow path: read from the stream until enough bytes are read,
        # or until an EOF occurs or until read() would block.
        chunks = [buf[pos:]]
        wanted = max(self.buffer_size, n)
        while avail < n:
            chunk = self.raw.read(wanted)
            if chunk in empty_values:
                nodata_val = chunk
                break
            avail += len(chunk)
            chunks.append(chunk)
        # n is more than avail only when an EOF occurred or when
        # read() would have blocked.
        n = min(n, avail)
        out = b"".join(chunks)
        self._read_buf = out[n:]  # Save the extra data in the buffer.
        self._read_pos = 0
        return out[:n] if out else nodata_val

    def peek(self, size=0):
        """Returns buffered bytes without advancing the position.

        The argument indicates a desired minimal number of bytes; we
        do at most one raw read to satisfy it.  We never return more
        than self.buffer_size.
        """
        with self._read_lock:
            return self._peek_unlocked(size)

    def _peek_unlocked(self, n=0):
        want = min(n, self.buffer_size)
        have = len(self._read_buf) - self._read_pos
        if have < want or have <= 0:
            to_read = self.buffer_size - have
            current = self.raw.read(to_read)
            if current:
                self._read_buf = self._read_buf[self._read_pos:] + current
                self._read_pos = 0
        return self._read_buf[self._read_pos:]

    def read1(self, size=-1):
        """Reads up to size bytes, with at most one read() system call."""
        # Returns up to size bytes.  If at least one byte is buffered, we
        # only return buffered bytes.  Otherwise, we do one raw read.
        if size < 0:
            size = self.buffer_size
        if size == 0:
            return b""
        with self._read_lock:
            self._peek_unlocked(1)
            return self._read_unlocked(
                min(size, len(self._read_buf) - self._read_pos))

    # Implementing readinto() and readinto1() is not strictly necessary (we
    # could rely on the base class that provides an implementation in terms of
    # read() and read1()). We do it anyway to keep the _pyio implementation
    # similar to the io implementation (which implements the methods for
    # performance reasons).
    def _readinto(self, buf, read1):
        """Read data into *buf* with at most one system call."""

        # Need to create a memoryview object of type 'b', otherwise
        # we may not be able to assign bytes to it, and slicing it
        # would create a new object.
        if not isinstance(buf, memoryview):
            buf = memoryview(buf)
        if buf.nbytes == 0:
            return 0
        buf = buf.cast('B')

        written = 0
        with self._read_lock:
            while written < len(buf):

                # First try to read from internal buffer
                avail = min(len(self._read_buf) - self._read_pos, len(buf))
                if avail:
                    buf[written:written+avail] = \
                        self._read_buf[self._read_pos:self._read_pos+avail]
                    self._read_pos += avail
                    written += avail
                    if written == len(buf):
                        break

                # If remaining space in callers buffer is larger than
                # internal buffer, read directly into callers buffer
                if len(buf) - written > self.buffer_size:
                    n = self.raw.readinto(buf[written:])
                    if not n:
                        break # eof
                    written += n

                # Otherwise refill internal buffer - unless we're
                # in read1 mode and already got some data
                elif not (read1 and written):
                    if not self._peek_unlocked(1):
                        break # eof

                # In readinto1 mode, return as soon as we have some data
                if read1 and written:
                    break

        return written

    def tell(self):
        return _BufferedIOMixin.tell(self) - len(self._read_buf) + self._read_pos

    def seek(self, pos, whence=0):
        if whence not in valid_seek_flags:
            raise ValueError("invalid whence value")
        with self._read_lock:
            if whence == 1:
                pos -= len(self._read_buf) - self._read_pos
            pos = _BufferedIOMixin.seek(self, pos, whence)
            self._reset_read_buf()
            return pos

class BufferedWriter(_BufferedIOMixin):

    """A buffer for a writeable sequential RawIO object.

    The constructor creates a BufferedWriter for the given writeable raw
    stream. If the buffer_size is not given, it defaults to
    DEFAULT_BUFFER_SIZE.
    """

    def __init__(self, raw, buffer_size=DEFAULT_BUFFER_SIZE):
        if not raw.writable():
            raise OSError('"raw" argument must be writable.')

        _BufferedIOMixin.__init__(self, raw)
        if buffer_size <= 0:
            raise ValueError("invalid buffer size")
        self.buffer_size = buffer_size
        self._write_buf = bytearray()
        self._write_lock = Lock()

    def writable(self):
        return self.raw.writable()

    def write(self, b):
        if isinstance(b, str):
            raise TypeError("can't write str to binary stream")
        with self._write_lock:
            if self.closed:
                raise ValueError("write to closed file")
            # XXX we can implement some more tricks to try and avoid
            # partial writes
            if len(self._write_buf) > self.buffer_size:
                # We're full, so let's pre-flush the buffer.  (This may
                # raise BlockingIOError with characters_written == 0.)
                self._flush_unlocked()
            before = len(self._write_buf)
            self._write_buf.extend(b)
            written = len(self._write_buf) - before
            if len(self._write_buf) > self.buffer_size:
                try:
                    self._flush_unlocked()
                except BlockingIOError as e:
                    if len(self._write_buf) > self.buffer_size:
                        # We've hit the buffer_size. We have to accept a partial
                        # write and cut back our buffer.
                        overage = len(self._write_buf) - self.buffer_size
                        written -= overage
                        self._write_buf = self._write_buf[:self.buffer_size]
                        raise BlockingIOError(e.errno, e.strerror, written)
            return written

    def truncate(self, pos=None):
        with self._write_lock:
            self._flush_unlocked()
            if pos is None:
                pos = self.raw.tell()
            return self.raw.truncate(pos)

    def flush(self):
        with self._write_lock:
            self._flush_unlocked()

    def _flush_unlocked(self):
        if self.closed:
            raise ValueError("flush on closed file")
        while self._write_buf:
            try:
                n = self.raw.write(self._write_buf)
            except BlockingIOError:
                raise RuntimeError("self.raw should implement RawIOBase: it "
                                   "should not raise BlockingIOError")
            if n is None:
                raise BlockingIOError(
                    errno.EAGAIN,
                    "write could not complete without blocking", 0)
            if n > len(self._write_buf) or n < 0:
                raise OSError("write() returned incorrect number of bytes")
            del self._write_buf[:n]

    def tell(self):
        return _BufferedIOMixin.tell(self) + len(self._write_buf)

    def seek(self, pos, whence=0):
        if whence not in valid_seek_flags:
            raise ValueError("invalid whence value")
        with self._write_lock:
            self._flush_unlocked()
            return _BufferedIOMixin.seek(self, pos, whence)

    def close(self):
        with self._write_lock:
            if self.raw is None or self.closed:
                return
        # We have to release the lock and call self.flush() (which will
        # probably just re-take the lock) in case flush has been overridden in
        # a subclass or the user set self.flush to something. This is the same
        # behavior as the C implementation.
        try:
            # may raise BlockingIOError or BrokenPipeError etc
            self.flush()
        finally:
            with self._write_lock:
                self.raw.close()


class BufferedRWPair(BufferedIOBase):

    """A buffered reader and writer object together.

    A buffered reader object and buffered writer object put together to
    form a sequential IO object that can read and write. This is typically
    used with a socket or two-way pipe.

    reader and writer are RawIOBase objects that are readable and
    writeable respectively. If the buffer_size is omitted it defaults to
    DEFAULT_BUFFER_SIZE.
    """

    # XXX The usefulness of this (compared to having two separate IO
    # objects) is questionable.

    def __init__(self, reader, writer, buffer_size=DEFAULT_BUFFER_SIZE):
        """Constructor.

        The arguments are two RawIO instances.
        """
        if not reader.readable():
            raise OSError('"reader" argument must be readable.')

        if not writer.writable():
            raise OSError('"writer" argument must be writable.')

        self.reader = BufferedReader(reader, buffer_size)
        self.writer = BufferedWriter(writer, buffer_size)

    def read(self, size=-1):
        if size is None:
            size = -1
        return self.reader.read(size)

    def readinto(self, b):
        return self.reader.readinto(b)

    def write(self, b):
        return self.writer.write(b)

    def peek(self, size=0):
        return self.reader.peek(size)

    def read1(self, size=-1):
        return self.reader.read1(size)

    def readinto1(self, b):
        return self.reader.readinto1(b)

    def readable(self):
        return self.reader.readable()

    def writable(self):
        return self.writer.writable()

    def flush(self):
        return self.writer.flush()

    def close(self):
        try:
            self.writer.close()
        finally:
            self.reader.close()

    def isatty(self):
        return self.reader.isatty() or self.writer.isatty()

    @property
    def closed(self):
        return self.writer.closed


class BufferedRandom(BufferedWriter, BufferedReader):

    """A buffered interface to random access streams.

    The constructor creates a reader and writer for a seekable stream,
    raw, given in the first argument. If the buffer_size is omitted it
    defaults to DEFAULT_BUFFER_SIZE.
    """

    def __init__(self, raw, buffer_size=DEFAULT_BUFFER_SIZE):
        raw._checkSeekable()
        BufferedReader.__init__(self, raw, buffer_size)
        BufferedWriter.__init__(self, raw, buffer_size)

    def seek(self, pos, whence=0):
        if whence not in valid_seek_flags:
            raise ValueError("invalid whence value")
        self.flush()
        if self._read_buf:
            # Undo read ahead.
            with self._read_lock:
                self.raw.seek(self._read_pos - len(self._read_buf), 1)
        # First do the raw seek, then empty the read buffer, so that
        # if the raw seek fails, we don't lose buffered data forever.
        pos = self.raw.seek(pos, whence)
        with self._read_lock:
            self._reset_read_buf()
        if pos < 0:
            raise OSError("seek() returned invalid position")
        return pos

    def tell(self):
        if self._write_buf:
            return BufferedWriter.tell(self)
        else:
            return BufferedReader.tell(self)

    def truncate(self, pos=None):
        if pos is None:
            pos = self.tell()
        # Use seek to flush the read buffer.
        return BufferedWriter.truncate(self, pos)

    def read(self, size=None):
        if size is None:
            size = -1
        self.flush()
        return BufferedReader.read(self, size)

    def readinto(self, b):
        self.flush()
        return BufferedReader.readinto(self, b)

    def peek(self, size=0):
        self.flush()
        return BufferedReader.peek(self, size)

    def read1(self, size=-1):
        self.flush()
        return BufferedReader.read1(self, size)

    def readinto1(self, b):
        self.flush()
        return BufferedReader.readinto1(self, b)

    def write(self, b):
        if self._read_buf:
            # Undo readahead
            with self._read_lock:
                self.raw.seek(self._read_pos - len(self._read_buf), 1)
                self._reset_read_buf()
        return BufferedWriter.write(self, b)


class FileIO(RawIOBase):
    _fd = -1
    _created = False
    _readable = False
    _writable = False
    _appending = False
    _seekable = None
    _closefd = True

    def __init__(self, file, mode='r', closefd=True, opener=None):
        """Open a file.  The mode can be 'r' (default), 'w', 'x' or 'a' for reading,
        writing, exclusive creation or appending.  The file will be created if it
        doesn't exist when opened for writing or appending; it will be truncated
        when opened for writing.  A FileExistsError will be raised if it already
        exists when opened for creating. Opening a file for creating implies
        writing so this mode behaves in a similar way to 'w'. Add a '+' to the mode
        to allow simultaneous reading and writing. A custom opener can be used by
        passing a callable as *opener*. The underlying file descriptor for the file
        object is then obtained by calling opener with (*name*, *flags*).
        *opener* must return an open file descriptor (passing os.open as *opener*
        results in functionality similar to passing None).
        """
        if self._fd >= 0:
            # Have to close the existing file first.
            try:
                if self._closefd:
                    os.close(self._fd)
            finally:
                self._fd = -1

        if isinstance(file, float):
            raise TypeError('integer argument expected, got float')
        if isinstance(file, int):
            fd = file
            if fd < 0:
                raise ValueError('negative file descriptor')
        else:
            fd = -1

        if not isinstance(mode, str):
            raise TypeError('invalid mode: %s' % (mode,))
        if not set(mode) <= set('xrwab+'):
            raise ValueError('invalid mode: %s' % (mode,))
        if sum(c in 'rwax' for c in mode) != 1 or mode.count('+') > 1:
            raise ValueError('Must have exactly one of create/read/write/append '
                             'mode and at most one plus')

        if 'x' in mode:
            self._created = True
            self._writable = True
            flags = os.O_EXCL | os.O_CREAT
        elif 'r' in mode:
            self._readable = True
            flags = 0
        elif 'w' in mode:
            self._writable = True
            flags = os.O_CREAT | os.O_TRUNC
        elif 'a' in mode:
            self._writable = True
            self._appending = True
            flags = os.O_APPEND | os.O_CREAT

        if '+' in mode:
            self._readable = True
            self._writable = True

        if self._readable and self._writable:
            flags |= os.O_RDWR
        elif self._readable:
            flags |= os.O_RDONLY
        else:
            flags |= os.O_WRONLY

        flags |= getattr(os, 'O_BINARY', 0)

        noinherit_flag = (getattr(os, 'O_NOINHERIT', 0) or
                          getattr(os, 'O_CLOEXEC', 0))
        flags |= noinherit_flag

        owned_fd = None
        try:
            if fd < 0:
                if not closefd:
                    raise ValueError('Cannot use closefd=False with file name')
                if opener is None:
                    fd = os.open(file, flags, 0o666)
                else:
                    fd = opener(file, flags)
                    if not isinstance(fd, int):
                        raise TypeError('expected integer from opener')
                    if fd < 0:
                        raise OSError('Negative file descriptor')
                owned_fd = fd
                if not noinherit_flag:
                    os.set_inheritable(fd, False)

            self._closefd = closefd
            fdfstat = os.fstat(fd)
            try:
                if stat.S_ISDIR(fdfstat.st_mode):
                    raise IsADirectoryError(errno.EISDIR,
                                            os.strerror(errno.EISDIR), file)
            except AttributeError:
                # Ignore the AttributeError if stat.S_ISDIR or errno.EISDIR
                # don't exist.
                pass
            self._blksize = getattr(fdfstat, 'st_blksize', 0)
            if self._blksize <= 1:
                self._blksize = DEFAULT_BUFFER_SIZE

            if _setmode:
                # don't translate newlines (\r\n <=> \n)
                _setmode(fd, os.O_BINARY)

            self.name = file
            if self._appending:
                # For consistent behaviour, we explicitly seek to the
                # end of file (otherwise, it might be done only on the
                # first write()).
                try:
                    os.lseek(fd, 0, SEEK_END)
                except OSError as e:
                    if e.errno != errno.ESPIPE:
                        raise
        except:
            if owned_fd is not None:
                os.close(owned_fd)
            raise
        self._fd = fd

    def __del__(self):
        if self._fd >= 0 and self._closefd and not self.closed:
            import warnings
            warnings.warn('unclosed file %r' % (self,), ResourceWarning,
                          stacklevel=2, source=self)
            self.close()

    def __getstate__(self):
        raise TypeError(f"cannot pickle {self.__class__.__name__!r} object")

    def __repr__(self):
        class_name = '%s.%s' % (self.__class__.__module__,
                                self.__class__.__qualname__)
        if self.closed:
            return '<%s [closed]>' % class_name
        try:
            name = self.name
        except AttributeError:
            return ('<%s fd=%d mode=%r closefd=%r>' %
                    (class_name, self._fd, self.mode, self._closefd))
        else:
            return ('<%s name=%r mode=%r closefd=%r>' %
                    (class_name, name, self.mode, self._closefd))

    def _checkReadable(self):
        if not self._readable:
            raise UnsupportedOperation('File not open for reading')

    def _checkWritable(self, msg=None):
        if not self._writable:
            raise UnsupportedOperation('File not open for writing')

    def read(self, size=None):
        """Read at most size bytes, returned as bytes.

        Only makes one system call, so less data may be returned than requested
        In non-blocking mode, returns None if no data is available.
        Return an empty bytes object at EOF.
        """
        self._checkClosed()
        self._checkReadable()
        if size is None or size < 0:
            return self.readall()
        try:
            return os.read(self._fd, size)
        except BlockingIOError:
            return None

    def readall(self):
        """Read all data from the file, returned as bytes.

        In non-blocking mode, returns as much as is immediately available,
        or None if no data is available.  Return an empty bytes object at EOF.
        """
        self._checkClosed()
        self._checkReadable()
        bufsize = DEFAULT_BUFFER_SIZE
        try:
            pos = os.lseek(self._fd, 0, SEEK_CUR)
            end = os.fstat(self._fd).st_size
            if end >= pos:
                bufsize = end - pos + 1
        except OSError:
            pass

        result = bytearray()
        while True:
            if len(result) >= bufsize:
                bufsize = len(result)
                bufsize += max(bufsize, DEFAULT_BUFFER_SIZE)
            n = bufsize - len(result)
            try:
                chunk = os.read(self._fd, n)
            except BlockingIOError:
                if result:
                    break
                return None
            if not chunk: # reached the end of the file
                break
            result += chunk

        return bytes(result)

    def readinto(self, b):
        """Same as RawIOBase.readinto()."""
        m = memoryview(b).cast('B')
        data = self.read(len(m))
        n = len(data)
        m[:n] = data
        return n

    def write(self, b):
        """Write bytes b to file, return number written.

        Only makes one system call, so not all of the data may be written.
        The number of bytes actually written is returned.  In non-blocking mode,
        returns None if the write would block.
        """
        self._checkClosed()
        self._checkWritable()
        try:
            return os.write(self._fd, b)
        except BlockingIOError:
            return None

    def seek(self, pos, whence=SEEK_SET):
        """Move to new file position.

        Argument offset is a byte count.  Optional argument whence defaults to
        SEEK_SET or 0 (offset from start of file, offset should be >= 0); other values
        are SEEK_CUR or 1 (move relative to current position, positive or negative),
        and SEEK_END or 2 (move relative to end of file, usually negative, although
        many platforms allow seeking beyond the end of a file).

        Note that not all file objects are seekable.
        """
        if isinstance(pos, float):
            raise TypeError('an integer is required')
        self._checkClosed()
        return os.lseek(self._fd, pos, whence)

    def tell(self):
        """tell() -> int.  Current file position.

        Can raise OSError for non seekable files."""
        self._checkClosed()
        return os.lseek(self._fd, 0, SEEK_CUR)

    def truncate(self, size=None):
        """Truncate the file to at most size bytes.

        Size defaults to the current file position, as returned by tell().
        The current file position is changed to the value of size.
        """
        self._checkClosed()
        self._checkWritable()
        if size is None:
            size = self.tell()
        os.ftruncate(self._fd, size)
        return size

    def close(self):
        """Close the file.

        A closed file cannot be used for further I/O operations.  close() may be
        called more than once without error.
        """
        if not self.closed:
            try:
                if self._closefd:
                    os.close(self._fd)
            finally:
                super().close()

    def seekable(self):
        """True if file supports random-access."""
        self._checkClosed()
        if self._seekable is None:
            try:
                self.tell()
            except OSError:
                self._seekable = False
            else:
                self._seekable = True
        return self._seekable

    def readable(self):
        """True if file was opened in a read mode."""
        self._checkClosed()
        return self._readable

    def writable(self):
        """True if file was opened in a write mode."""
        self._checkClosed()
        return self._writable

    def fileno(self):
        """Return the underlying file descriptor (an integer)."""
        self._checkClosed()
        return self._fd

    def isatty(self):
        """True if the file is connected to a TTY device."""
        self._checkClosed()
        return os.isatty(self._fd)

    @property
    def closefd(self):
        """True if the file descriptor will be closed by close()."""
        return self._closefd

    @property
    def mode(self):
        """String giving the file mode"""
        if self._created:
            if self._readable:
                return 'xb+'
            else:
                return 'xb'
        elif self._appending:
            if self._readable:
                return 'ab+'
            else:
                return 'ab'
        elif self._readable:
            if self._writable:
                return 'rb+'
            else:
                return 'rb'
        else:
            return 'wb'


class TextIOBase(IOBase):

    """Base class for text I/O.

    This class provides a character and line based interface to stream
    I/O.
    """

    def read(self, size=-1):
        """Read at most size characters from stream, where size is an int.

        Read from underlying buffer until we have size characters or we hit EOF.
        If size is negative or omitted, read until EOF.

        Returns a string.
        """
        self._unsupported("read")

    def write(self, s):
        """Write string s to stream and returning an int."""
        self._unsupported("write")

    def truncate(self, pos=None):
        """Truncate size to pos, where pos is an int."""
        self._unsupported("truncate")

    def readline(self):
        """Read until newline or EOF.

        Returns an empty string if EOF is hit immediately.
        """
        self._unsupported("readline")

    def detach(self):
        """
        Separate the underlying buffer from the TextIOBase and return it.

        After the underlying buffer has been detached, the TextIO is in an
        unusable state.
        """
        self._unsupported("detach")

    @property
    def encoding(self):
        """Subclasses should override."""
        return None

    @property
    def newlines(self):
        """Line endings translated so far.

        Only line endings translated during reading are considered.

        Subclasses should override.
        """
        return None

    @property
    def errors(self):
        """Error setting of the decoder or encoder.

        Subclasses should override."""
        return None

io.TextIOBase.register(TextIOBase)


class IncrementalNewlineDecoder(codecs.IncrementalDecoder):
    r"""Codec used when reading a file in universal newlines mode.  It wraps
    another incremental decoder, translating \r\n and \r into \n.  It also
    records the types of newlines encountered.  When used with
    translate=False, it ensures that the newline sequence is returned in
    one piece.
    """
    def __init__(self, decoder, translate, errors='strict'):
        codecs.IncrementalDecoder.__init__(self, errors=errors)
        self.translate = translate
        self.decoder = decoder
        self.seennl = 0
        self.pendingcr = False

    def decode(self, input, final=False):
        # decode input (with the eventual \r from a previous pass)
        if self.decoder is None:
            output = input
        else:
            output = self.decoder.decode(input, final=final)
        if self.pendingcr and (output or final):
            output = "\r" + output
            self.pendingcr = False

        # retain last \r even when not translating data:
        # then readline() is sure to get \r\n in one pass
        if output.endswith("\r") and not final:
            output = output[:-1]
            self.pendingcr = True

        # Record which newlines are read
        crlf = output.count('\r\n')
        cr = output.count('\r') - crlf
        lf = output.count('\n') - crlf
        self.seennl |= (lf and self._LF) | (cr and self._CR) \
                    | (crlf and self._CRLF)

        if self.translate:
            if crlf:
                output = output.replace("\r\n", "\n")
            if cr:
                output = output.replace("\r", "\n")

        return output

    def getstate(self):
        if self.decoder is None:
            buf = b""
            flag = 0
        else:
            buf, flag = self.decoder.getstate()
        flag <<= 1
        if self.pendingcr:
            flag |= 1
        return buf, flag

    def setstate(self, state):
        buf, flag = state
        self.pendingcr = bool(flag & 1)
        if self.decoder is not None:
            self.decoder.setstate((buf, flag >> 1))

    def reset(self):
        self.seennl = 0
        self.pendingcr = False
        if self.decoder is not None:
            self.decoder.reset()

    _LF = 1
    _CR = 2
    _CRLF = 4

    @property
    def newlines(self):
        return (None,
                "\n",
                "\r",
                ("\r", "\n"),
                "\r\n",
                ("\n", "\r\n"),
                ("\r", "\r\n"),
                ("\r", "\n", "\r\n")
               )[self.seennl]


class TextIOWrapper(TextIOBase):

    r"""Character and line based layer over a BufferedIOBase object, buffer.

    encoding gives the name of the encoding that the stream will be
    decoded or encoded with. It defaults to locale.getpreferredencoding(False).

    errors determines the strictness of encoding and decoding (see the
    codecs.register) and defaults to "strict".

    newline can be None, '', '\n', '\r', or '\r\n'.  It controls the
    handling of line endings. If it is None, universal newlines is
    enabled.  With this enabled, on input, the lines endings '\n', '\r',
    or '\r\n' are translated to '\n' before being returned to the
    caller. Conversely, on output, '\n' is translated to the system
    default line separator, os.linesep. If newline is any other of its
    legal values, that newline becomes the newline when the file is read
    and it is returned untranslated. On output, '\n' is converted to the
    newline.

    If line_buffering is True, a call to flush is implied when a call to
    write contains a newline character.
    """

    _CHUNK_SIZE = 2048

    # Initialize _buffer as soon as possible since it's used by __del__()
    # which calls close()
    _buffer = None

    # The write_through argument has no effect here since this
    # implementation always writes through.  The argument is present only
    # so that the signature can match the signature of the C version.
    def __init__(self, buffer, encoding=None, errors=None, newline=None,
                 line_buffering=False, write_through=False):
        self._check_newline(newline)
        encoding = text_encoding(encoding)

        if encoding == "locale":
            try:
                encoding = os.device_encoding(buffer.fileno()) or "locale"
            except (AttributeError, UnsupportedOperation):
                pass

        if encoding == "locale":
            try:
                import locale
            except ImportError:
                # Importing locale may fail if Python is being built
                encoding = "utf-8"
            else:
                encoding = locale.getpreferredencoding(False)

        if not isinstance(encoding, str):
            raise ValueError("invalid encoding: %r" % encoding)

        if not codecs.lookup(encoding)._is_text_encoding:
            msg = ("%r is not a text encoding; "
                   "use codecs.open() to handle arbitrary codecs")
            raise LookupError(msg % encoding)

        if errors is None:
            errors = "strict"
        else:
            if not isinstance(errors, str):
                raise ValueError("invalid errors: %r" % errors)
            if _CHECK_ERRORS:
                codecs.lookup_error(errors)

        self._buffer = buffer
        self._decoded_chars = ''  # buffer for text returned from decoder
        self._decoded_chars_used = 0  # offset into _decoded_chars for read()
        self._snapshot = None  # info for reconstructing decoder state
        self._seekable = self._telling = self.buffer.seekable()
        self._has_read1 = hasattr(self.buffer, 'read1')
        self._configure(encoding, errors, newline,
                        line_buffering, write_through)

    def _check_newline(self, newline):
        if newline is not None and not isinstance(newline, str):
            raise TypeError("illegal newline type: %r" % (type(newline),))
        if newline not in (None, "", "\n", "\r", "\r\n"):
            raise ValueError("illegal newline value: %r" % (newline,))

    def _configure(self, encoding=None, errors=None, newline=None,
                   line_buffering=False, write_through=False):
        self._encoding = encoding
        self._errors = errors
        self._encoder = None
        self._decoder = None
        self._b2cratio = 0.0

        self._readuniversal = not newline
        self._readtranslate = newline is None
        self._readnl = newline
        self._writetranslate = newline != ''
        self._writenl = newline or os.linesep

        self._line_buffering = line_buffering
        self._write_through = write_through

        # don't write a BOM in the middle of a file
        if self._seekable and self.writable():
            position = self.buffer.tell()
            if position != 0:
                try:
                    self._get_encoder().setstate(0)
                except LookupError:
                    # Sometimes the encoder doesn't exist
                    pass

    # self._snapshot is either None, or a tuple (dec_flags, next_input)
    # where dec_flags is the second (integer) item of the decoder state
    # and next_input is the chunk of input bytes that comes next after the
    # snapshot point.  We use this to reconstruct decoder states in tell().

    # Naming convention:
    #   - "bytes_..." for integer variables that count input bytes
    #   - "chars_..." for integer variables that count decoded characters

    def __repr__(self):
        result = "<{}.{}".format(self.__class__.__module__,
                                 self.__class__.__qualname__)
        try:
            name = self.name
        except AttributeError:
            pass
        else:
            result += " name={0!r}".format(name)
        try:
            mode = self.mode
        except AttributeError:
            pass
        else:
            result += " mode={0!r}".format(mode)
        return result + " encoding={0!r}>".format(self.encoding)

    @property
    def encoding(self):
        return self._encoding

    @property
    def errors(self):
        return self._errors

    @property
    def line_buffering(self):
        return self._line_buffering

    @property
    def write_through(self):
        return self._write_through

    @property
    def buffer(self):
        return self._buffer

    def reconfigure(self, *,
                    encoding=None, errors=None, newline=Ellipsis,
                    line_buffering=None, write_through=None):
        """Reconfigure the text stream with new parameters.

        This also flushes the stream.
        """
        if (self._decoder is not None
                and (encoding is not None or errors is not None
                     or newline is not Ellipsis)):
            raise UnsupportedOperation(
                "It is not possible to set the encoding or newline of stream "
                "after the first read")

        if errors is None:
            if encoding is None:
                errors = self._errors
            else:
                errors = 'strict'
        elif not isinstance(errors, str):
            raise TypeError("invalid errors: %r" % errors)

        if encoding is None:
            encoding = self._encoding
        else:
            if not isinstance(encoding, str):
                raise TypeError("invalid encoding: %r" % encoding)

        if newline is Ellipsis:
            newline = self._readnl
        self._check_newline(newline)

        if line_buffering is None:
            line_buffering = self.line_buffering
        if write_through is None:
            write_through = self.write_through

        self.flush()
        self._configure(encoding, errors, newline,
                        line_buffering, write_through)

    def seekable(self):
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        return self._seekable

    def readable(self):
        return self.buffer.readable()

    def writable(self):
        return self.buffer.writable()

    def flush(self):
        self.buffer.flush()
        self._telling = self._seekable

    def close(self):
        if self.buffer is not None and not self.closed:
            try:
                self.flush()
            finally:
                self.buffer.close()

    @property
    def closed(self):
        return self.buffer.closed

    @property
    def name(self):
        return self.buffer.name

    def fileno(self):
        return self.buffer.fileno()

    def isatty(self):
        return self.buffer.isatty()

    def write(self, s):
        'Write data, where s is a str'
        if self.closed:
            raise ValueError("write to closed file")
        if not isinstance(s, str):
            raise TypeError("can't write %s to text stream" %
                            s.__class__.__name__)
        length = len(s)
        haslf = (self._writetranslate or self._line_buffering) and "\n" in s
        if haslf and self._writetranslate and self._writenl != "\n":
            s = s.replace("\n", self._writenl)
        encoder = self._encoder or self._get_encoder()
        # XXX What if we were just reading?
        b = encoder.encode(s)
        self.buffer.write(b)
        if self._line_buffering and (haslf or "\r" in s):
            self.flush()
        self._set_decoded_chars('')
        self._snapshot = None
        if self._decoder:
            self._decoder.reset()
        return length

    def _get_encoder(self):
        make_encoder = codecs.getincrementalencoder(self._encoding)
        self._encoder = make_encoder(self._errors)
        return self._encoder

    def _get_decoder(self):
        make_decoder = codecs.getincrementaldecoder(self._encoding)
        decoder = make_decoder(self._errors)
        if self._readuniversal:
            decoder = IncrementalNewlineDecoder(decoder, self._readtranslate)
        self._decoder = decoder
        return decoder

    # The following three methods implement an ADT for _decoded_chars.
    # Text returned from the decoder is buffered here until the client
    # requests it by calling our read() or readline() method.
    def _set_decoded_chars(self, chars):
        """Set the _decoded_chars buffer."""
        self._decoded_chars = chars
        self._decoded_chars_used = 0

    def _get_decoded_chars(self, n=None):
        """Advance into the _decoded_chars buffer."""
        offset = self._decoded_chars_used
        if n is None:
            chars = self._decoded_chars[offset:]
        else:
            chars = self._decoded_chars[offset:offset + n]
        self._decoded_chars_used += len(chars)
        return chars

    def _rewind_decoded_chars(self, n):
        """Rewind the _decoded_chars buffer."""
        if self._decoded_chars_used < n:
            raise AssertionError("rewind decoded_chars out of bounds")
        self._decoded_chars_used -= n

    def _read_chunk(self):
        """
        Read and decode the next chunk of data from the BufferedReader.
        """

        # The return value is True unless EOF was reached.  The decoded
        # string is placed in self._decoded_chars (replacing its previous
        # value).  The entire input chunk is sent to the decoder, though
        # some of it may remain buffered in the decoder, yet to be
        # converted.

        if self._decoder is None:
            raise ValueError("no decoder")

        if self._telling:
            # To prepare for tell(), we need to snapshot a point in the
            # file where the decoder's input buffer is empty.

            dec_buffer, dec_flags = self._decoder.getstate()
            # Given this, we know there was a valid snapshot point
            # len(dec_buffer) bytes ago with decoder state (b'', dec_flags).

        # Read a chunk, decode it, and put the result in self._decoded_chars.
        if self._has_read1:
            input_chunk = self.buffer.read1(self._CHUNK_SIZE)
        else:
            input_chunk = self.buffer.read(self._CHUNK_SIZE)
        eof = not input_chunk
        decoded_chars = self._decoder.decode(input_chunk, eof)
        self._set_decoded_chars(decoded_chars)
        if decoded_chars:
            self._b2cratio = len(input_chunk) / len(self._decoded_chars)
        else:
            self._b2cratio = 0.0

        if self._telling:
            # At the snapshot point, len(dec_buffer) bytes before the read,
            # the next input to be decoded is dec_buffer + input_chunk.
            self._snapshot = (dec_flags, dec_buffer + input_chunk)

        return not eof

    def _pack_cookie(self, position, dec_flags=0,
                           bytes_to_feed=0, need_eof=False, chars_to_skip=0):
        # The meaning of a tell() cookie is: seek to position, set the
        # decoder flags to dec_flags, read bytes_to_feed bytes, feed them
        # into the decoder with need_eof as the EOF flag, then skip
        # chars_to_skip characters of the decoded result.  For most simple
        # decoders, tell() will often just give a byte offset in the file.
        return (position | (dec_flags<<64) | (bytes_to_feed<<128) |
               (chars_to_skip<<192) | bool(need_eof)<<256)

    def _unpack_cookie(self, bigint):
        rest, position = divmod(bigint, 1<<64)
        rest, dec_flags = divmod(rest, 1<<64)
        rest, bytes_to_feed = divmod(rest, 1<<64)
        need_eof, chars_to_skip = divmod(rest, 1<<64)
        return position, dec_flags, bytes_to_feed, bool(need_eof), chars_to_skip

    def tell(self):
        if not self._seekable:
            raise UnsupportedOperation("underlying stream is not seekable")
        if not self._telling:
            raise OSError("telling position disabled by next() call")
        self.flush()
        position = self.buffer.tell()
        decoder = self._decoder
        if decoder is None or self._snapshot is None:
            if self._decoded_chars:
                # This should never happen.
                raise AssertionError("pending decoded text")
            return position

        # Skip backward to the snapshot point (see _read_chunk).
        dec_flags, next_input = self._snapshot
        position -= len(next_input)

        # How many decoded characters have been used up since the snapshot?
        chars_to_skip = self._decoded_chars_used
        if chars_to_skip == 0:
            # We haven't moved from the snapshot point.
            return self._pack_cookie(position, dec_flags)

        # Starting from the snapshot position, we will walk the decoder
        # forward until it gives us enough decoded characters.
        saved_state = decoder.getstate()
        try:
            # Fast search for an acceptable start point, close to our
            # current pos.
            # Rationale: calling decoder.decode() has a large overhead
            # regardless of chunk size; we want the number of such calls to
            # be O(1) in most situations (common decoders, sensible input).
            # Actually, it will be exactly 1 for fixed-size codecs (all
            # 8-bit codecs, also UTF-16 and UTF-32).
            skip_bytes = int(self._b2cratio * chars_to_skip)
            skip_back = 1
            assert skip_bytes <= len(next_input)
            while skip_bytes > 0:
                decoder.setstate((b'', dec_flags))
                # Decode up to temptative start point
                n = len(decoder.decode(next_input[:skip_bytes]))
                if n <= chars_to_skip:
                    b, d = decoder.getstate()
                    if not b:
                        # Before pos and no bytes buffered in decoder => OK
                        dec_flags = d
                        chars_to_skip -= n
                        break
                    # Skip back by buffered amount and reset heuristic
                    skip_bytes -= len(b)
                    skip_back = 1
                else:
                    # We're too far ahead, skip back a bit
                    skip_bytes -= skip_back
                    skip_back = skip_back * 2
            else:
                skip_bytes = 0
                decoder.setstate((b'', dec_flags))

            # Note our initial start point.
            start_pos = position + skip_bytes
            start_flags = dec_flags
            if chars_to_skip == 0:
                # We haven't moved from the start point.
                return self._pack_cookie(start_pos, start_flags)

            # Feed the decoder one byte at a time.  As we go, note the
            # nearest "safe start point" before the current location
            # (a point where the decoder has nothing buffered, so seek()
            # can safely start from there and advance to this location).
            bytes_fed = 0
            need_eof = False
            # Chars decoded since `start_pos`
            chars_decoded = 0
            for i in range(skip_bytes, len(next_input)):
                bytes_fed += 1
                chars_decoded += len(decoder.decode(next_input[i:i+1]))
                dec_buffer, dec_flags = decoder.getstate()
                if not dec_buffer and chars_decoded <= chars_to_skip:
                    # Decoder buffer is empty, so this is a safe start point.
                    start_pos += bytes_fed
                    chars_to_skip -= chars_decoded
                    start_flags, bytes_fed, chars_decoded = dec_flags, 0, 0
                if chars_decoded >= chars_to_skip:
                    break
            else:
                # We didn't get enough decoded data; signal EOF to get more.
                chars_decoded += len(decoder.decode(b'', final=True))
                need_eof = True
                if chars_decoded < chars_to_skip:
                    raise OSError("can't reconstruct logical file position")

            # The returned cookie corresponds to the last safe start point.
            return self._pack_cookie(
                start_pos, start_flags, bytes_fed, need_eof, chars_to_skip)
        finally:
            decoder.setstate(saved_state)

    def truncate(self, pos=None):
        self.flush()
        if pos is None:
            pos = self.tell()
        return self.buffer.truncate(pos)

    def detach(self):
        if self.buffer is None:
            raise ValueError("buffer is already detached")
        self.flush()
        buffer = self._buffer
        self._buffer = None
        return buffer

    def seek(self, cookie, whence=0):
        def _reset_encoder(position):
            """Reset the encoder (merely useful for proper BOM handling)"""
            try:
                encoder = self._encoder or self._get_encoder()
            except LookupError:
                # Sometimes the encoder doesn't exist
                pass
            else:
                if position != 0:
                    encoder.setstate(0)
                else:
                    encoder.reset()

        if self.closed:
            raise ValueError("tell on closed file")
        if not self._seekable:
            raise UnsupportedOperation("underlying stream is not seekable")
        if whence == SEEK_CUR:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero cur-relative seeks")
            # Seeking to the current position should attempt to
            # sync the underlying buffer with the current position.
            whence = 0
            cookie = self.tell()
        elif whence == SEEK_END:
            if cookie != 0:
                raise UnsupportedOperation("can't do nonzero end-relative seeks")
            self.flush()
            position = self.buffer.seek(0, whence)
            self._set_decoded_chars('')
            self._snapshot = None
            if self._decoder:
                self._decoder.reset()
            _reset_encoder(position)
            return position
        if whence != 0:
            raise ValueError("unsupported whence (%r)" % (whence,))
        if cookie < 0:
            raise ValueError("negative seek position %r" % (cookie,))
        self.flush()

        # The strategy of seek() is to go back to the safe start point
        # and replay the effect of read(chars_to_skip) from there.
        start_pos, dec_flags, bytes_to_feed, need_eof, chars_to_skip = \
            self._unpack_cookie(cookie)

        # Seek back to the safe start point.
        self.buffer.seek(start_pos)
        self._set_decoded_chars('')
        self._snapshot = None

        # Restore the decoder to its state from the safe start point.
        if cookie == 0 and self._decoder:
            self._decoder.reset()
        elif self._decoder or dec_flags or chars_to_skip:
            self._decoder = self._decoder or self._get_decoder()
            self._decoder.setstate((b'', dec_flags))
            self._snapshot = (dec_flags, b'')

        if chars_to_skip:
            # Just like _read_chunk, feed the decoder and save a snapshot.
            input_chunk = self.buffer.read(bytes_to_feed)
            self._set_decoded_chars(
                self._decoder.decode(input_chunk, need_eof))
            self._snapshot = (dec_flags, input_chunk)

            # Skip chars_to_skip of the decoded characters.
            if len(self._decoded_chars) < chars_to_skip:
                raise OSError("can't restore logical file position")
            self._decoded_chars_used = chars_to_skip

        _reset_encoder(cookie)
        return cookie

    def read(self, size=None):
        self._checkReadable()
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f"{size!r} is not an integer")
            else:
                size = size_index()
        decoder = self._decoder or self._get_decoder()
        if size < 0:
            # Read everything.
            result = (self._get_decoded_chars() +
                      decoder.decode(self.buffer.read(), final=True))
            self._set_decoded_chars('')
            self._snapshot = None
            return result
        else:
            # Keep reading chunks until we have size characters to return.
            eof = False
            result = self._get_decoded_chars(size)
            while len(result) < size and not eof:
                eof = not self._read_chunk()
                result += self._get_decoded_chars(size - len(result))
            return result

    def __next__(self):
        self._telling = False
        line = self.readline()
        if not line:
            self._snapshot = None
            self._telling = self._seekable
            raise StopIteration
        return line

    def readline(self, size=None):
        if self.closed:
            raise ValueError("read from closed file")
        if size is None:
            size = -1
        else:
            try:
                size_index = size.__index__
            except AttributeError:
                raise TypeError(f"{size!r} is not an integer")
            else:
                size = size_index()

        # Grab all the decoded text (we will rewind any extra bits later).
        line = self._get_decoded_chars()

        start = 0
        # Make the decoder if it doesn't already exist.
        if not self._decoder:
            self._get_decoder()

        pos = endpos = None
        while True:
            if self._readtranslate:
                # Newlines are already translated, only search for \n
                pos = line.find('\n', start)
                if pos >= 0:
                    endpos = pos + 1
                    break
                else:
                    start = len(line)

            elif self._readuniversal:
                # Universal newline search. Find any of \r, \r\n, \n
                # The decoder ensures that \r\n are not split in two pieces

                # In C we'd look for these in parallel of course.
                nlpos = line.find("\n", start)
                crpos = line.find("\r", start)
                if crpos == -1:
                    if nlpos == -1:
                        # Nothing found
                        start = len(line)
                    else:
                        # Found \n
                        endpos = nlpos + 1
                        break
                elif nlpos == -1:
                    # Found lone \r
                    endpos = crpos + 1
                    break
                elif nlpos < crpos:
                    # Found \n
                    endpos = nlpos + 1
                    break
                elif nlpos == crpos + 1:
                    # Found \r\n
                    endpos = crpos + 2
                    break
                else:
                    # Found \r
                    endpos = crpos + 1
                    break
            else:
                # non-universal
                pos = line.find(self._readnl)
                if pos >= 0:
                    endpos = pos + len(self._readnl)
                    break

            if size >= 0 and len(line) >= size:
                endpos = size  # reached length size
                break

            # No line ending seen yet - get more data'
            while self._read_chunk():
                if self._decoded_chars:
                    break
            if self._decoded_chars:
                line += self._get_decoded_chars()
            else:
                # end of file
                self._set_decoded_chars('')
                self._snapshot = None
                return line

        if size >= 0 and endpos > size:
            endpos = size  # don't exceed size

        # Rewind _decoded_chars to just after the line ending we found.
        self._rewind_decoded_chars(len(line) - endpos)
        return line[:endpos]

    @property
    def newlines(self):
        return self._decoder.newlines if self._decoder else None


class StringIO(TextIOWrapper):
    """Text I/O implementation using an in-memory buffer.

    The initial_value argument sets the value of object.  The newline
    argument is like the one of TextIOWrapper's constructor.
    """

    def __init__(self, initial_value="", newline="\n"):
        super(StringIO, self).__init__(BytesIO(),
                                       encoding="utf-8",
                                       errors="surrogatepass",
                                       newline=newline)
        # Issue #5645: make universal newlines semantics the same as in the
        # C version, even under Windows.
        if newline is None:
            self._writetranslate = False
        if initial_value is not None:
            if not isinstance(initial_value, str):
                raise TypeError("initial_value must be str or None, not {0}"
                                .format(type(initial_value).__name__))
            self.write(initial_value)
            self.seek(0)

    def getvalue(self):
        self.flush()
        decoder = self._decoder or self._get_decoder()
        old_state = decoder.getstate()
        decoder.reset()
        try:
            return decoder.decode(self.buffer.getvalue(), final=True)
        finally:
            decoder.setstate(old_state)

    def __repr__(self):
        # TextIOWrapper tells the encoding in its repr. In StringIO,
        # that's an implementation detail.
        return object.__repr__(self)

    @property
    def errors(self):
        return None

    @property
    def encoding(self):
        return None

    def detach(self):
        # This doesn't make sense on StringIO.
        self._unsupported("detach")
<|endoftext|>"""Functions that read and write gzipped files.

The user of the file doesn't have to worry about the compression,
but random access is not allowed."""

# based on Andrew Kuchling's minigzip.py distributed with the zlib module

import struct, sys, time, os
import zlib
import builtins
import io
import _compression

__all__ = ["BadGzipFile", "GzipFile", "open", "compress", "decompress"]

FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT = 1, 2, 4, 8, 16

READ, WRITE = 1, 2

_COMPRESS_LEVEL_FAST = 1
_COMPRESS_LEVEL_TRADEOFF = 6
_COMPRESS_LEVEL_BEST = 9


def open(filename, mode="rb", compresslevel=_COMPRESS_LEVEL_BEST,
         encoding=None, errors=None, newline=None):
    """Open a gzip-compressed file in binary or text mode.

    The filename argument can be an actual filename (a str or bytes object), or
    an existing file object to read from or write to.

    The mode argument can be "r", "rb", "w", "wb", "x", "xb", "a" or "ab" for
    binary mode, or "rt", "wt", "xt" or "at" for text mode. The default mode is
    "rb", and the default compresslevel is 9.

    For binary mode, this function is equivalent to the GzipFile constructor:
    GzipFile(filename, mode, compresslevel). In this case, the encoding, errors
    and newline arguments must not be provided.

    For text mode, a GzipFile object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error handling
    behavior, and line ending(s).

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

    gz_mode = mode.replace("t", "")
    if isinstance(filename, (str, bytes, os.PathLike)):
        binary_file = GzipFile(filename, gz_mode, compresslevel)
    elif hasattr(filename, "read") or hasattr(filename, "write"):
        binary_file = GzipFile(None, gz_mode, compresslevel, filename)
    else:
        raise TypeError("filename must be a str or bytes object, or a file")

    if "t" in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file

def write32u(output, value):
    # The L format writes the bit pattern correctly whether signed
    # or unsigned.
    output.write(struct.pack("<L", value))

class _PaddedFile:
    """Minimal read-only file object that prepends a string to the contents
    of an actual file. Shouldn't be used outside of gzip.py, as it lacks
    essential functionality."""

    def __init__(self, f, prepend=b''):
        self._buffer = prepend
        self._length = len(prepend)
        self.file = f
        self._read = 0

    def read(self, size):
        if self._read is None:
            return self.file.read(size)
        if self._read + size <= self._length:
            read = self._read
            self._read += size
            return self._buffer[read:self._read]
        else:
            read = self._read
            self._read = None
            return self._buffer[read:] + \
                   self.file.read(size-self._length+read)

    def prepend(self, prepend=b''):
        if self._read is None:
            self._buffer = prepend
        else:  # Assume data was read since the last prepend() call
            self._read -= len(prepend)
            return
        self._length = len(self._buffer)
        self._read = 0

    def seek(self, off):
        self._read = None
        self._buffer = None
        return self.file.seek(off)

    def seekable(self):
        return True  # Allows fast-forwarding even in unseekable streams


class BadGzipFile(OSError):
