        raise ValueError("undiscernible mode")

    @classmethod
    def taropen(cls, name, mode="r", fileobj=None, **kwargs):
        """Open uncompressed tar archive name for reading or writing.
        """
        if mode not in ("r", "a", "w", "x"):
            raise ValueError("mode must be 'r', 'a', 'w' or 'x'")
        return cls(name, mode, fileobj, **kwargs)

    @classmethod
    def gzopen(cls, name, mode="r", fileobj=None, compresslevel=9, **kwargs):
        """Open gzip compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
        if mode not in ("r", "w", "x"):
            raise ValueError("mode must be 'r', 'w' or 'x'")

        try:
            from gzip import GzipFile
        except ImportError:
            raise CompressionError("gzip module is not available") from None

        try:
            fileobj = GzipFile(name, mode + "b", compresslevel, fileobj)
        except OSError as e:
            if fileobj is not None and mode == 'r':
                raise ReadError("not a gzip file") from e
            raise

        try:
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except OSError as e:
            fileobj.close()
            if mode == 'r':
                raise ReadError("not a gzip file") from e
            raise
        except:
            fileobj.close()
            raise
        t._extfileobj = False
        return t

    @classmethod
    def bz2open(cls, name, mode="r", fileobj=None, compresslevel=9, **kwargs):
        """Open bzip2 compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
        if mode not in ("r", "w", "x"):
            raise ValueError("mode must be 'r', 'w' or 'x'")

        try:
            from bz2 import BZ2File
        except ImportError:
            raise CompressionError("bz2 module is not available") from None

        fileobj = BZ2File(fileobj or name, mode, compresslevel=compresslevel)

        try:
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except (OSError, EOFError) as e:
            fileobj.close()
            if mode == 'r':
                raise ReadError("not a bzip2 file") from e
            raise
        except:
            fileobj.close()
            raise
        t._extfileobj = False
        return t

    @classmethod
    def xzopen(cls, name, mode="r", fileobj=None, preset=None, **kwargs):
        """Open lzma compressed tar archive name for reading or writing.
           Appending is not allowed.
        """
        if mode not in ("r", "w", "x"):
            raise ValueError("mode must be 'r', 'w' or 'x'")

        try:
            from lzma import LZMAFile, LZMAError
        except ImportError:
            raise CompressionError("lzma module is not available") from None

        fileobj = LZMAFile(fileobj or name, mode, preset=preset)

        try:
            t = cls.taropen(name, mode, fileobj, **kwargs)
        except (LZMAError, EOFError) as e:
            fileobj.close()
            if mode == 'r':
                raise ReadError("not an lzma file") from e
            raise
        except:
            fileobj.close()
            raise
        t._extfileobj = False
        return t

    # All *open() methods are registered here.
    OPEN_METH = {
        "tar": "taropen",   # uncompressed tar
        "gz":  "gzopen",    # gzip compressed tar
        "bz2": "bz2open",   # bzip2 compressed tar
        "xz":  "xzopen"     # lzma compressed tar
    }

    #--------------------------------------------------------------------------
    # The public methods which TarFile provides:

    def close(self):
        """Close the TarFile. In write-mode, two finishing zero blocks are
           appended to the archive.
        """
        if self.closed:
            return

        self.closed = True
        try:
            if self.mode in ("a", "w", "x"):
                self.fileobj.write(NUL * (BLOCKSIZE * 2))
                self.offset += (BLOCKSIZE * 2)
                # fill up the end with zero-blocks
                # (like option -b20 for tar does)
                blocks, remainder = divmod(self.offset, RECORDSIZE)
                if remainder > 0:
                    self.fileobj.write(NUL * (RECORDSIZE - remainder))
        finally:
            if not self._extfileobj:
                self.fileobj.close()

    def getmember(self, name):
        """Return a TarInfo object for member `name'. If `name' can not be
           found in the archive, KeyError is raised. If a member occurs more
           than once in the archive, its last occurrence is assumed to be the
           most up-to-date version.
        """
        tarinfo = self._getmember(name.rstrip('/'))
        if tarinfo is None:
            raise KeyError("filename %r not found" % name)
        return tarinfo

    def getmembers(self):
        """Return the members of the archive as a list of TarInfo objects. The
           list has the same order as the members in the archive.
        """
        self._check()
        if not self._loaded:    # if we want to obtain a list of
            self._load()        # all members, we first have to
                                # scan the whole archive.
        return self.members

    def getnames(self):
        """Return the members of the archive as a list of their names. It has
           the same order as the list returned by getmembers().
        """
        return [tarinfo.name for tarinfo in self.getmembers()]

    def gettarinfo(self, name=None, arcname=None, fileobj=None):
        """Create a TarInfo object from the result of os.stat or equivalent
           on an existing file. The file is either named by `name', or
           specified as a file object `fileobj' with a file descriptor. If
           given, `arcname' specifies an alternative name for the file in the
           archive, otherwise, the name is taken from the 'name' attribute of
           'fileobj', or the 'name' argument. The name should be a text
           string.
        """
        self._check("awx")

        # When fileobj is given, replace name by
        # fileobj's real name.
        if fileobj is not None:
            name = fileobj.name

        # Building the name of the member in the archive.
        # Backward slashes are converted to forward slashes,
        # Absolute paths are turned to relative paths.
        if arcname is None:
            arcname = name
        drv, arcname = os.path.splitdrive(arcname)
        arcname = arcname.replace(os.sep, "/")
        arcname = arcname.lstrip("/")

        # Now, fill the TarInfo object with
        # information specific for the file.
        tarinfo = self.tarinfo()
        tarinfo.tarfile = self  # Not needed

        # Use os.stat or os.lstat, depending on if symlinks shall be resolved.
        if fileobj is None:
            if not self.dereference:
                statres = os.lstat(name)
            else:
                statres = os.stat(name)
        else:
            statres = os.fstat(fileobj.fileno())
        linkname = ""

        stmd = statres.st_mode
        if stat.S_ISREG(stmd):
            inode = (statres.st_ino, statres.st_dev)
            if not self.dereference and statres.st_nlink > 1 and \
                    inode in self.inodes and arcname != self.inodes[inode]:
                # Is it a hardlink to an already
                # archived file?
                type = LNKTYPE
                linkname = self.inodes[inode]
            else:
                # The inode is added only if its valid.
                # For win32 it is always 0.
                type = REGTYPE
                if inode[0]:
                    self.inodes[inode] = arcname
        elif stat.S_ISDIR(stmd):
            type = DIRTYPE
        elif stat.S_ISFIFO(stmd):
            type = FIFOTYPE
        elif stat.S_ISLNK(stmd):
            type = SYMTYPE
            linkname = os.readlink(name)
        elif stat.S_ISCHR(stmd):
            type = CHRTYPE
        elif stat.S_ISBLK(stmd):
            type = BLKTYPE
        else:
            return None

        # Fill the TarInfo object with all
        # information we can get.
        tarinfo.name = arcname
        tarinfo.mode = stmd
        tarinfo.uid = statres.st_uid
        tarinfo.gid = statres.st_gid
        if type == REGTYPE:
            tarinfo.size = statres.st_size
        else:
            tarinfo.size = 0
        tarinfo.mtime = statres.st_mtime
        tarinfo.type = type
        tarinfo.linkname = linkname
        if pwd:
            try:
                tarinfo.uname = pwd.getpwuid(tarinfo.uid)[0]
            except KeyError:
                pass
        if grp:
            try:
                tarinfo.gname = grp.getgrgid(tarinfo.gid)[0]
            except KeyError:
                pass

        if type in (CHRTYPE, BLKTYPE):
            if hasattr(os, "major") and hasattr(os, "minor"):
                tarinfo.devmajor = os.major(statres.st_rdev)
                tarinfo.devminor = os.minor(statres.st_rdev)
        return tarinfo

    def list(self, verbose=True, *, members=None):
        """Print a table of contents to sys.stdout. If `verbose' is False, only
           the names of the members are printed. If it is True, an `ls -l'-like
           output is produced. `members' is optional and must be a subset of the
           list returned by getmembers().
        """
        self._check()

        if members is None:
            members = self
        for tarinfo in members:
            if verbose:
                if tarinfo.mode is None:
                    _safe_print("??????????")
                else:
                    _safe_print(stat.filemode(tarinfo.mode))
                _safe_print("%s/%s" % (tarinfo.uname or tarinfo.uid,
                                       tarinfo.gname or tarinfo.gid))
                if tarinfo.ischr() or tarinfo.isblk():
                    _safe_print("%10s" %
                            ("%d,%d" % (tarinfo.devmajor, tarinfo.devminor)))
                else:
                    _safe_print("%10d" % tarinfo.size)
                if tarinfo.mtime is None:
                    _safe_print("????-??-?? ??:??:??")
                else:
                    _safe_print("%d-%02d-%02d %02d:%02d:%02d" \
                                % time.localtime(tarinfo.mtime)[:6])

            _safe_print(tarinfo.name + ("/" if tarinfo.isdir() else ""))

            if verbose:
                if tarinfo.issym():
                    _safe_print("-> " + tarinfo.linkname)
                if tarinfo.islnk():
                    _safe_print("link to " + tarinfo.linkname)
            print()

    def add(self, name, arcname=None, recursive=True, *, filter=None):
        """Add the file `name' to the archive. `name' may be any type of file
           (directory, fifo, symbolic link, etc.). If given, `arcname'
           specifies an alternative name for the file in the archive.
           Directories are added recursively by default. This can be avoided by
           setting `recursive' to False. `filter' is a function
           that expects a TarInfo object argument and returns the changed
           TarInfo object, if it returns None the TarInfo object will be
           excluded from the archive.
        """
        self._check("awx")

        if arcname is None:
            arcname = name

        # Skip if somebody tries to archive the archive...
        if self.name is not None and os.path.abspath(name) == self.name:
            self._dbg(2, "tarfile: Skipped %r" % name)
            return

        self._dbg(1, name)

        # Create a TarInfo object from the file.
        tarinfo = self.gettarinfo(name, arcname)

        if tarinfo is None:
            self._dbg(1, "tarfile: Unsupported type %r" % name)
            return

        # Change or exclude the TarInfo object.
        if filter is not None:
            tarinfo = filter(tarinfo)
            if tarinfo is None:
                self._dbg(2, "tarfile: Excluded %r" % name)
                return

        # Append the tar header and data to the archive.
        if tarinfo.isreg():
            with bltn_open(name, "rb") as f:
                self.addfile(tarinfo, f)

        elif tarinfo.isdir():
            self.addfile(tarinfo)
            if recursive:
                for f in sorted(os.listdir(name)):
                    self.add(os.path.join(name, f), os.path.join(arcname, f),
                            recursive, filter=filter)

        else:
            self.addfile(tarinfo)

    def addfile(self, tarinfo, fileobj=None):
        """Add the TarInfo object `tarinfo' to the archive. If `fileobj' is
           given, it should be a binary file, and tarinfo.size bytes are read
           from it and added to the archive. You can create TarInfo objects
           directly, or by using gettarinfo().
        """
        self._check("awx")

        tarinfo = copy.copy(tarinfo)

        buf = tarinfo.tobuf(self.format, self.encoding, self.errors)
        self.fileobj.write(buf)
        self.offset += len(buf)
        bufsize=self.copybufsize
        # If there's data to follow, append it.
        if fileobj is not None:
            copyfileobj(fileobj, self.fileobj, tarinfo.size, bufsize=bufsize)
            blocks, remainder = divmod(tarinfo.size, BLOCKSIZE)
            if remainder > 0:
                self.fileobj.write(NUL * (BLOCKSIZE - remainder))
                blocks += 1
            self.offset += blocks * BLOCKSIZE

        self.members.append(tarinfo)

    def _get_filter_function(self, filter):
        if filter is None:
            filter = self.extraction_filter
            if filter is None:
                return fully_trusted_filter
            if isinstance(filter, str):
                raise TypeError(
                    'String names are not supported for '
                    + 'TarFile.extraction_filter. Use a function such as '
                    + 'tarfile.data_filter directly.')
            return filter
        if callable(filter):
            return filter
        try:
            return _NAMED_FILTERS[filter]
        except KeyError:
            raise ValueError(f"filter {filter!r} not found") from None

    def extractall(self, path=".", members=None, *, numeric_owner=False,
                   filter=None):
        """Extract all members from the archive to the current working
           directory and set owner, modification time and permissions on
           directories afterwards. `path' specifies a different directory
           to extract to. `members' is optional and must be a subset of the
           list returned by getmembers(). If `numeric_owner` is True, only
           the numbers for user/group names are used and not the names.

           The `filter` function will be called on each member just
           before extraction.
           It can return a changed TarInfo or None to skip the member.
           String names of common filters are accepted.
        """
        directories = []

        filter_function = self._get_filter_function(filter)
        if members is None:
            members = self

        for member in members:
            tarinfo = self._get_extract_tarinfo(member, filter_function, path)
            if tarinfo is None:
                continue
            if tarinfo.isdir():
                # For directories, delay setting attributes until later,
                # since permissions can interfere with extraction and
                # extracting contents can reset mtime.
                directories.append(tarinfo)
            self._extract_one(tarinfo, path, set_attrs=not tarinfo.isdir(),
                              numeric_owner=numeric_owner,
                              filter_function=filter_function)

        # Reverse sort directories.
        directories.sort(key=lambda a: a.name, reverse=True)

        # Set correct owner, mtime and filemode on directories.
        for tarinfo in directories:
            dirpath = os.path.join(path, tarinfo.name)
            try:
                self.chown(tarinfo, dirpath, numeric_owner=numeric_owner)
                self.utime(tarinfo, dirpath)
                self.chmod(tarinfo, dirpath)
            except ExtractError as e:
                self._handle_nonfatal_error(e)

    def extract(self, member, path="", set_attrs=True, *, numeric_owner=False,
                filter=None):
        """Extract a member from the archive to the current working directory,
           using its full name. Its file information is extracted as accurately
           as possible. `member' may be a filename or a TarInfo object. You can
           specify a different directory using `path'. File attributes (owner,
           mtime, mode) are set unless `set_attrs' is False. If `numeric_owner`
           is True, only the numbers for user/group names are used and not
           the names.

           The `filter` function will be called before extraction.
           It can return a changed TarInfo or None to skip the member.
           String names of common filters are accepted.
        """
        filter_function = self._get_filter_function(filter)
        tarinfo = self._get_extract_tarinfo(member, filter_function, path)
        if tarinfo is not None:
            self._extract_one(tarinfo, path, set_attrs, numeric_owner,
                              filter_function=filter_function)

    def _get_extract_tarinfo(self, member, filter_function, path):
        """Get filtered TarInfo (or None) from member, which might be a str"""
        if isinstance(member, str):
            tarinfo = self.getmember(member)
        else:
            tarinfo = member

        unfiltered = tarinfo
        try:
            tarinfo = filter_function(tarinfo, path)
        except (OSError, FilterError) as e:
            self._handle_fatal_error(e)
        except ExtractError as e:
            self._handle_nonfatal_error(e)
        if tarinfo is None:
            self._dbg(2, "tarfile: Excluded %r" % unfiltered.name)
            return None
        # Prepare the link target for makelink().
        if tarinfo.islnk():
            tarinfo = copy.copy(tarinfo)
            tarinfo._link_target = os.path.join(path, tarinfo.linkname)
        return tarinfo

    def _extract_one(self, tarinfo, path, set_attrs, numeric_owner,
                     filter_function=None):
        """Extract from filtered tarinfo to disk

           filter_function is only used when extracting a *different*
           member (e.g. as fallback to creating a symlink)
        """
        self._check("r")

        try:
            self._extract_member(tarinfo, os.path.join(path, tarinfo.name),
                                 set_attrs=set_attrs,
                                 numeric_owner=numeric_owner,
                                 filter_function=filter_function,
                                 extraction_root=path)
        except OSError as e:
            self._handle_fatal_error(e)
        except ExtractError as e:
            self._handle_nonfatal_error(e)

    def _handle_nonfatal_error(self, e):
        """Handle non-fatal error (ExtractError) according to errorlevel"""
        if self.errorlevel > 1:
            raise
        else:
            self._dbg(1, "tarfile: %s" % e)

    def _handle_fatal_error(self, e):
        """Handle "fatal" error according to self.errorlevel"""
        if self.errorlevel > 0:
            raise
        elif isinstance(e, OSError):
            if e.filename is None:
                self._dbg(1, "tarfile: %s" % e.strerror)
            else:
                self._dbg(1, "tarfile: %s %r" % (e.strerror, e.filename))
        else:
            self._dbg(1, "tarfile: %s %s" % (type(e).__name__, e))

    def extractfile(self, member):
        """Extract a member from the archive as a file object. `member' may be
           a filename or a TarInfo object. If `member' is a regular file or
           a link, an io.BufferedReader object is returned. For all other
           existing members, None is returned. If `member' does not appear
           in the archive, KeyError is raised.
        """
        self._check("r")

        if isinstance(member, str):
            tarinfo = self.getmember(member)
        else:
            tarinfo = member

        if tarinfo.isreg() or tarinfo.type not in SUPPORTED_TYPES:
            # Members with unknown types are treated as regular files.
            return self.fileobject(self, tarinfo)

        elif tarinfo.islnk() or tarinfo.issym():
            if isinstance(self.fileobj, _Stream):
                # A small but ugly workaround for the case that someone tries
                # to extract a (sym)link as a file-object from a non-seekable
                # stream of tar blocks.
                raise StreamError("cannot extract (sym)link as file object")
            else:
                # A (sym)link's file object is its target's file object.
                return self.extractfile(self._find_link_target(tarinfo))
        else:
            # If there's no data associated with the member (directory, chrdev,
            # blkdev, etc.), return None instead of a file object.
            return None

    def _extract_member(self, tarinfo, targetpath, set_attrs=True,
                        numeric_owner=False, filter_function=None,
                        extraction_root=None):
        """Extract the TarInfo object tarinfo to a physical
           file called targetpath.

           filter_function/extraction_root are only used when extracting
           a *different* member (e.g. as fallback to creating a symlink
           or hardlink whose target is missing from the archive).
        """
        # Fetch the TarInfo object for the given name
        # and build the destination pathname, replacing
        # forward slashes to platform specific separators.
        targetpath = targetpath.rstrip("/")
        targetpath = targetpath.replace("/", os.sep)

        # Create all upper directories.
        upperdirs = os.path.dirname(targetpath)
        if upperdirs and not os.path.exists(upperdirs):
            # Create directories that are not part of the archive with
            # default permissions.
            os.makedirs(upperdirs)

        if tarinfo.islnk() or tarinfo.issym():
            self._dbg(1, "%s -> %s" % (tarinfo.name, tarinfo.linkname))
        else:
            self._dbg(1, tarinfo.name)

        if tarinfo.isreg():
            self.makefile(tarinfo, targetpath)
        elif tarinfo.isdir():
            self.makedir(tarinfo, targetpath)
        elif tarinfo.isfifo():
            self.makefifo(tarinfo, targetpath)
        elif tarinfo.ischr() or tarinfo.isblk():
            self.makedev(tarinfo, targetpath)
        elif tarinfo.islnk() or tarinfo.issym():
            self.makelink(tarinfo, targetpath,
                          filter_function=filter_function,
                          extraction_root=extraction_root)
        elif tarinfo.type not in SUPPORTED_TYPES:
            self.makeunknown(tarinfo, targetpath)
        else:
            self.makefile(tarinfo, targetpath)

        if set_attrs:
            self.chown(tarinfo, targetpath, numeric_owner)
            if not tarinfo.issym():
                self.chmod(tarinfo, targetpath)
                self.utime(tarinfo, targetpath)

    #--------------------------------------------------------------------------
    # Below are the different file methods. They are called via
    # _extract_member() when extract() is called. They can be replaced in a
    # subclass to implement other functionality.

    def makedir(self, tarinfo, targetpath):
        """Make a directory called targetpath.
        """
        try:
            if tarinfo.mode is None:
                # Use the system's default mode
                os.mkdir(targetpath)
            else:
                # Use a safe mode for the directory, the real mode is set
                # later in _extract_member().
                os.mkdir(targetpath, 0o700)
        except FileExistsError:
            pass

    def makefile(self, tarinfo, targetpath):
        """Make a file called targetpath.
        """
        source = self.fileobj
        source.seek(tarinfo.offset_data)
        bufsize = self.copybufsize
        with bltn_open(targetpath, "wb") as target:
            if tarinfo.sparse is not None:
                for offset, size in tarinfo.sparse:
                    target.seek(offset)
                    copyfileobj(source, target, size, ReadError, bufsize)
                target.seek(tarinfo.size)
                target.truncate()
            else:
                copyfileobj(source, target, tarinfo.size, ReadError, bufsize)

    def makeunknown(self, tarinfo, targetpath):
        """Make a file from a TarInfo object with an unknown type
           at targetpath.
        """
        self.makefile(tarinfo, targetpath)
        self._dbg(1, "tarfile: Unknown file type %r, " \
                     "extracted as regular file." % tarinfo.type)

    def makefifo(self, tarinfo, targetpath):
        """Make a fifo called targetpath.
        """
        if hasattr(os, "mkfifo"):
            os.mkfifo(targetpath)
        else:
            raise ExtractError("fifo not supported by system")

    def makedev(self, tarinfo, targetpath):
        """Make a character or block device called targetpath.
        """
        if not hasattr(os, "mknod") or not hasattr(os, "makedev"):
            raise ExtractError("special devices not supported by system")

        mode = tarinfo.mode
        if mode is None:
            # Use mknod's default
            mode = 0o600
        if tarinfo.isblk():
            mode |= stat.S_IFBLK
        else:
            mode |= stat.S_IFCHR

        os.mknod(targetpath, mode,
                 os.makedev(tarinfo.devmajor, tarinfo.devminor))

    def makelink(self, tarinfo, targetpath, filter_function=None,
                extraction_root=None):
        """Make a (symbolic) link called targetpath. If it cannot be created
          (platform limitation), we try to make a copy of the referenced file
          instead of a link.
        """
        def extract_link_target():
            # CVE-2026-4360: apply the caller's extraction filter to the
            # link's target too, since it is a *different* member being
            # extracted as a fallback and would otherwise bypass
            # filtering entirely (e.g. filter='data').
            target = self._find_link_target(tarinfo)
            if filter_function is not None:
                try:
                    filtered = filter_function(target, extraction_root)
                except (OSError, FilterError) as e:
                    self._handle_fatal_error(e)
                    return
                except ExtractError as e:
                    self._handle_nonfatal_error(e)
                    return
                if filtered is None:
                    self._dbg(2, "tarfile: Excluded %r" % target.name)
                    return
                target = filtered
            self._extract_member(target, targetpath)

        try:
            # For systems that support symbolic and hard links.
            if tarinfo.issym():
                if os.path.lexists(targetpath):
                    # Avoid FileExistsError on following os.symlink.
                    os.unlink(targetpath)
                os.symlink(tarinfo.linkname, targetpath)
            else:
                if os.path.exists(tarinfo._link_target):
                    os.link(tarinfo._link_target, targetpath)
                else:
                    extract_link_target()
        except symlink_exception:
            try:
                extract_link_target()
            except KeyError:
                raise ExtractError("unable to resolve link inside archive") from None

    def chown(self, tarinfo, targetpath, numeric_owner):
        """Set owner of targetpath according to tarinfo. If numeric_owner
           is True, use .gid/.uid instead of .gname/.uname. If numeric_owner
           is False, fall back to .gid/.uid when the search based on name
           fails.
        """
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            # We have to be root to do so.
            g = tarinfo.gid
            u = tarinfo.uid
            if not numeric_owner:
                try:
                    if grp and tarinfo.gname:
                        g = grp.getgrnam(tarinfo.gname)[2]
                except KeyError:
                    pass
                try:
                    if pwd and tarinfo.uname:
                        u = pwd.getpwnam(tarinfo.uname)[2]
                except KeyError:
                    pass
            if g is None:
                g = -1
            if u is None:
                u = -1
            try:
                if tarinfo.issym() and hasattr(os, "lchown"):
                    os.lchown(targetpath, u, g)
                else:
                    os.chown(targetpath, u, g)
            except OSError as e:
                raise ExtractError("could not change owner") from e

    def chmod(self, tarinfo, targetpath):
        """Set file permissions of targetpath according to tarinfo.
        """
        if tarinfo.mode is None:
            return
        try:
            os.chmod(targetpath, tarinfo.mode)
        except OSError as e:
            raise ExtractError("could not change mode") from e

    def utime(self, tarinfo, targetpath):
        """Set modification time of targetpath according to tarinfo.
        """
        mtime = tarinfo.mtime
        if mtime is None:
            return
        if not hasattr(os, 'utime'):
            return
        try:
            os.utime(targetpath, (mtime, mtime))
        except OSError as e:
            raise ExtractError("could not change modification time") from e

    #--------------------------------------------------------------------------
    def next(self):
        """Return the next member of the archive as a TarInfo object, when
           TarFile is opened for reading. Return None if there is no more
           available.
        """
        self._check("ra")
        if self.firstmember is not None:
            m = self.firstmember
            self.firstmember = None
            return m

        # Advance the file pointer.
        if self.offset != self.fileobj.tell():
            self.fileobj.seek(self.offset - 1)
            if not self.fileobj.read(1):
                raise ReadError("unexpected end of data")

        # Read the next block.
        tarinfo = None
        while True:
            try:
                tarinfo = self.tarinfo.fromtarfile(self)
            except EOFHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, "0x%X: %s" % (self.offset, e))
                    self.offset += BLOCKSIZE
                    continue
            except InvalidHeaderError as e:
                if self.ignore_zeros:
                    self._dbg(2, "0x%X: %s" % (self.offset, e))
                    self.offset += BLOCKSIZE
                    continue
                elif self.offset == 0:
                    raise ReadError(str(e)) from None
            except EmptyHeaderError:
                if self.offset == 0:
                    raise ReadError("empty file") from None
            except TruncatedHeaderError as e:
                if self.offset == 0:
                    raise ReadError(str(e)) from None
            except SubsequentHeaderError as e:
                raise ReadError(str(e)) from None
            except Exception as e:
                try:
                    import zlib
                    if isinstance(e, zlib.error):
                        raise ReadError(f'zlib error: {e}') from None
                    else:
                        raise e
                except ImportError:
                    raise e
            break

        if tarinfo is not None:
            self.members.append(tarinfo)
        else:
            self._loaded = True

        return tarinfo

    #--------------------------------------------------------------------------
    # Little helper methods:

    def _getmember(self, name, tarinfo=None, normalize=False):
        """Find an archive member by name from bottom to top.
           If tarinfo is given, it is used as the starting point.
        """
        # Ensure that all members have been loaded.
        members = self.getmembers()

        # Limit the member search list up to tarinfo.
        skipping = False
        if tarinfo is not None:
            try:
                index = members.index(tarinfo)
            except ValueError:
                # The given starting point might be a (modified) copy.
                # We'll later skip members until we find an equivalent.
                skipping = True
            else:
                # Happy fast path
                members = members[:index]

        if normalize:
            name = os.path.normpath(name)

        for member in reversed(members):
            if skipping:
                if tarinfo.offset == member.offset:
                    skipping = False
                continue
            if normalize:
                member_name = os.path.normpath(member.name)
            else:
                member_name = member.name

            if name == member_name:
                return member

        if skipping:
            # Starting point was not found
            raise ValueError(tarinfo)

    def _load(self):
        """Read through the entire archive file and look for readable
           members.
        """
        while True:
            tarinfo = self.next()
            if tarinfo is None:
                break
        self._loaded = True

    def _check(self, mode=None):
        """Check if TarFile is still open, and if the operation's mode
           corresponds to TarFile's mode.
        """
        if self.closed:
            raise OSError("%s is closed" % self.__class__.__name__)
        if mode is not None and self.mode not in mode:
            raise OSError("bad operation for mode %r" % self.mode)

    def _find_link_target(self, tarinfo):
        """Find the target member of a symlink or hardlink member in the
           archive.
        """
        if tarinfo.issym():
            # Always search the entire archive.
            linkname = "/".join(filter(None, (os.path.dirname(tarinfo.name), tarinfo.linkname)))
            limit = None
        else:
            # Search the archive before the link, because a hard link is
            # just a reference to an already archived file.
            linkname = tarinfo.linkname
            limit = tarinfo

        member = self._getmember(linkname, tarinfo=limit, normalize=True)
        if member is None:
            raise KeyError("linkname %r not found" % linkname)
        return member

    def __iter__(self):
        """Provide an iterator object.
        """
        if self._loaded:
            yield from self.members
            return

        # Yield items using TarFile's next() method.
        # When all members have been read, set TarFile as _loaded.
        index = 0
        # Fix for SF #1100429: Under rare circumstances it can
        # happen that getmembers() is called during iteration,
        # which will have already exhausted the next() method.
        if self.firstmember is not None:
            tarinfo = self.next()
            index += 1
            yield tarinfo

        while True:
            if index < len(self.members):
                tarinfo = self.members[index]
            elif not self._loaded:
                tarinfo = self.next()
                if not tarinfo:
                    self._loaded = True
                    return
            else:
                return
            index += 1
            yield tarinfo

    def _dbg(self, level, msg):
        """Write debugging output to sys.stderr.
        """
        if level <= self.debug:
            print(msg, file=sys.stderr)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, type, value, traceback):
        if type is None:
            self.close()
        else:
            # An exception occurred. We must not call close() because
            # it would try to write end-of-archive blocks and padding.
            if not self._extfileobj:
                self.fileobj.close()
            self.closed = True

#--------------------
# exported functions
#--------------------

def is_tarfile(name):
    """Return True if name points to a tar archive that we
       are able to handle, else return False.

       'name' should be a string, file, or file-like object.
    """
    try:
        if hasattr(name, "read"):
            t = open(fileobj=name)
        else:
            t = open(name)
        t.close()
        return True
    except TarError:
        return False

open = TarFile.open


def main():
    import argparse

    description = 'A simple command-line interface for tarfile module.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose output')
    parser.add_argument('--filter', metavar='<filtername>',
                        choices=_NAMED_FILTERS,
                        help='Filter for extraction')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--list', metavar='<tarfile>',
                       help='Show listing of a tarfile')
    group.add_argument('-e', '--extract', nargs='+',
                       metavar=('<tarfile>', '<output_dir>'),
                       help='Extract tarfile into target dir')
    group.add_argument('-c', '--create', nargs='+',
                       metavar=('<name>', '<file>'),
                       help='Create tarfile from sources')
    group.add_argument('-t', '--test', metavar='<tarfile>',
                       help='Test if a tarfile is valid')

    args = parser.parse_args()

    if args.filter and args.extract is None:
        parser.exit(1, '--filter is only valid for extraction\n')

    if args.test is not None:
        src = args.test
        if is_tarfile(src):
            with open(src, 'r') as tar:
                tar.getmembers()
                print(tar.getmembers(), file=sys.stderr)
            if args.verbose:
                print('{!r} is a tar archive.'.format(src))
        else:
            parser.exit(1, '{!r} is not a tar archive.\n'.format(src))

    elif args.list is not None:
        src = args.list
        if is_tarfile(src):
            with TarFile.open(src, 'r:*') as tf:
                tf.list(verbose=args.verbose)
        else:
            parser.exit(1, '{!r} is not a tar archive.\n'.format(src))

    elif args.extract is not None:
        if len(args.extract) == 1:
            src = args.extract[0]
            curdir = os.curdir
        elif len(args.extract) == 2:
            src, curdir = args.extract
        else:
            parser.exit(1, parser.format_help())

        if is_tarfile(src):
            with TarFile.open(src, 'r:*') as tf:
                tf.extractall(path=curdir, filter=args.filter)
            if args.verbose:
                if curdir == '.':
                    msg = '{!r} file is extracted.'.format(src)
                else:
                    msg = ('{!r} file is extracted '
                           'into {!r} directory.').format(src, curdir)
                print(msg)
        else:
            parser.exit(1, '{!r} is not a tar archive.\n'.format(src))

    elif args.create is not None:
        tar_name = args.create.pop(0)
        _, ext = os.path.splitext(tar_name)
        compressions = {
            # gz
            '.gz': 'gz',
            '.tgz': 'gz',
            # xz
            '.xz': 'xz',
            '.txz': 'xz',
            # bz2
            '.bz2': 'bz2',
            '.tbz': 'bz2',
            '.tbz2': 'bz2',
            '.tb2': 'bz2',
        }
        tar_mode = 'w:' + compressions[ext] if ext in compressions else 'w'
        tar_files = args.create

        with TarFile.open(tar_name, tar_mode) as tf:
            for file_name in tar_files:
                tf.add(file_name)

        if args.verbose:
            print('{!r} file created.'.format(tar_name))

if __name__ == '__main__':
    main()
<|endoftext|># -*- Mode: Python; tab-width: 4 -*-
#       Id: asynchat.py,v 2.26 2000/09/07 22:29:26 rushing Exp
#       Author: Sam Rushing <rushing@nightmare.com>

# ======================================================================
# Copyright 1996 by Sam Rushing
#
#                         All Rights Reserved
#
# Permission to use, copy, modify, and distribute this software and
# its documentation for any purpose and without fee is hereby
# granted, provided that the above copyright notice appear in all
# copies and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of Sam
# Rushing not be used in advertising or publicity pertaining to
# distribution of the software without specific, written prior
# permission.
#
# SAM RUSHING DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN
# NO EVENT SHALL SAM RUSHING BE LIABLE FOR ANY SPECIAL, INDIRECT OR
# CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# ======================================================================

r"""A class supporting chat-style (command/response) protocols.

This class adds support for 'chat' style protocols - where one side
sends a 'command', and the other sends a response (examples would be
the common internet protocols - smtp, nntp, ftp, etc..).

The handle_read() method looks at the input stream for the current
'terminator' (usually '\r\n' for single-line responses, '\r\n.\r\n'
for multi-line output), calling self.found_terminator() on its
receipt.

for example:
Say you build an async nntp client using this class.  At the start
of the connection, you'll have self.terminator set to '\r\n', in
order to process the single-line greeting.  Just before issuing a
'LIST' command you'll set it to '\r\n.\r\n'.  The output of the LIST
command will be accumulated (using your own 'collect_incoming_data'
method) up to the terminator, and then control will be returned to
you - by calling your self.found_terminator() method.
"""
import asyncore
from collections import deque

from warnings import warn
warn(
    'The asynchat module is deprecated and will be removed in Python 3.12. '
    'The recommended replacement is asyncio',
    DeprecationWarning,
    stacklevel=2)



class async_chat(asyncore.dispatcher):
    """This is an abstract class.  You must derive from this class, and add
    the two methods collect_incoming_data() and found_terminator()"""

    # these are overridable defaults

    ac_in_buffer_size = 65536
    ac_out_buffer_size = 65536

    # we don't want to enable the use of encoding by default, because that is a
    # sign of an application bug that we don't want to pass silently

    use_encoding = 0
    encoding = 'latin-1'

    def __init__(self, sock=None, map=None):
        # for string terminator matching
        self.ac_in_buffer = b''

        # we use a list here rather than io.BytesIO for a few reasons...
        # del lst[:] is faster than bio.truncate(0)
        # lst = [] is faster than bio.truncate(0)
        self.incoming = []

        # we toss the use of the "simple producer" and replace it with
        # a pure deque, which the original fifo was a wrapping of
        self.producer_fifo = deque()
        asyncore.dispatcher.__init__(self, sock, map)

    def collect_incoming_data(self, data):
        raise NotImplementedError("must be implemented in subclass")

    def _collect_incoming_data(self, data):
        self.incoming.append(data)

    def _get_data(self):
        d = b''.join(self.incoming)
        del self.incoming[:]
        return d

    def found_terminator(self):
        raise NotImplementedError("must be implemented in subclass")

    def set_terminator(self, term):
        """Set the input delimiter.

        Can be a fixed string of any length, an integer, or None.
        """
        if isinstance(term, str) and self.use_encoding:
            term = bytes(term, self.encoding)
        elif isinstance(term, int) and term < 0:
            raise ValueError('the number of received bytes must be positive')
        self.terminator = term

    def get_terminator(self):
        return self.terminator

    # grab some more data from the socket,
    # throw it to the collector method,
    # check for the terminator,
    # if found, transition to the next state.

    def handle_read(self):

        try:
            data = self.recv(self.ac_in_buffer_size)
        except BlockingIOError:
            return
        except OSError:
            self.handle_error()
            return

        if isinstance(data, str) and self.use_encoding:
            data = bytes(str, self.encoding)
        self.ac_in_buffer = self.ac_in_buffer + data

        # Continue to search for self.terminator in self.ac_in_buffer,
        # while calling self.collect_incoming_data.  The while loop
        # is necessary because we might read several data+terminator
        # combos with a single recv(4096).

        while self.ac_in_buffer:
            lb = len(self.ac_in_buffer)
            terminator = self.get_terminator()
            if not terminator:
                # no terminator, collect it all
                self.collect_incoming_data(self.ac_in_buffer)
                self.ac_in_buffer = b''
            elif isinstance(terminator, int):
                # numeric terminator
                n = terminator
                if lb < n:
                    self.collect_incoming_data(self.ac_in_buffer)
                    self.ac_in_buffer = b''
                    self.terminator = self.terminator - lb
                else:
                    self.collect_incoming_data(self.ac_in_buffer[:n])
                    self.ac_in_buffer = self.ac_in_buffer[n:]
                    self.terminator = 0
                    self.found_terminator()
            else:
                # 3 cases:
                # 1) end of buffer matches terminator exactly:
                #    collect data, transition
                # 2) end of buffer matches some prefix:
                #    collect data to the prefix
                # 3) end of buffer does not match any prefix:
                #    collect data
                terminator_len = len(terminator)
                index = self.ac_in_buffer.find(terminator)
                if index != -1:
                    # we found the terminator
                    if index > 0:
                        # don't bother reporting the empty string
                        # (source of subtle bugs)
                        self.collect_incoming_data(self.ac_in_buffer[:index])
                    self.ac_in_buffer = self.ac_in_buffer[index+terminator_len:]
                    # This does the Right Thing if the terminator
                    # is changed here.
                    self.found_terminator()
                else:
                    # check for a prefix of the terminator
                    index = find_prefix_at_end(self.ac_in_buffer, terminator)
                    if index:
                        if index != lb:
                            # we found a prefix, collect up to the prefix
                            self.collect_incoming_data(self.ac_in_buffer[:-index])
                            self.ac_in_buffer = self.ac_in_buffer[-index:]
                        break
                    else:
                        # no prefix, collect it all
                        self.collect_incoming_data(self.ac_in_buffer)
                        self.ac_in_buffer = b''

    def handle_write(self):
        self.initiate_send()

    def handle_close(self):
        self.close()

    def push(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data argument must be byte-ish (%r)',
                            type(data))
        sabs = self.ac_out_buffer_size
        if len(data) > sabs:
            for i in range(0, len(data), sabs):
                self.producer_fifo.append(data[i:i+sabs])
        else:
            self.producer_fifo.append(data)
        self.initiate_send()

    def push_with_producer(self, producer):
        self.producer_fifo.append(producer)
        self.initiate_send()

    def readable(self):
        "predicate for inclusion in the readable for select()"
        # cannot use the old predicate, it violates the claim of the
        # set_terminator method.

        # return (len(self.ac_in_buffer) <= self.ac_in_buffer_size)
        return 1

    def writable(self):
        "predicate for inclusion in the writable for select()"
        return self.producer_fifo or (not self.connected)

    def close_when_done(self):
        "automatically close this channel once the outgoing queue is empty"
        self.producer_fifo.append(None)

    def initiate_send(self):
        while self.producer_fifo and self.connected:
            first = self.producer_fifo[0]
            # handle empty string/buffer or None entry
            if not first:
                del self.producer_fifo[0]
                if first is None:
                    self.handle_close()
                    return

            # handle classic producer behavior
            obs = self.ac_out_buffer_size
            try:
                data = first[:obs]
            except TypeError:
                data = first.more()
                if data:
                    self.producer_fifo.appendleft(data)
                else:
                    del self.producer_fifo[0]
                continue

            if isinstance(data, str) and self.use_encoding:
                data = bytes(data, self.encoding)

            # send the data
            try:
                num_sent = self.send(data)
            except OSError:
                self.handle_error()
                return

            if num_sent:
                if num_sent < len(data) or obs < len(first):
                    self.producer_fifo[0] = first[num_sent:]
                else:
                    del self.producer_fifo[0]
            # we tried to send some actual data
            return

    def discard_buffers(self):
        # Emergencies only!
        self.ac_in_buffer = b''
        del self.incoming[:]
        self.producer_fifo.clear()


class simple_producer:

    def __init__(self, data, buffer_size=512):
        self.data = data
        self.buffer_size = buffer_size

    def more(self):
        if len(self.data) > self.buffer_size:
            result = self.data[:self.buffer_size]
            self.data = self.data[self.buffer_size:]
            return result
        else:
            result = self.data
            self.data = b''
            return result


# Given 'haystack', see if any prefix of 'needle' is at its end.  This
# assumes an exact match has already been checked.  Return the number of
# characters matched.
# for example:
# f_p_a_e("qwerty\r", "\r\n") => 1
# f_p_a_e("qwertydkjf", "\r\n") => 0
# f_p_a_e("qwerty\r\n", "\r\n") => <undefined>

# this could maybe be made faster with a computed regex?
# [answer: no; circa Python-2.0, Jan 2001]
# new python:   28961/s
# old python:   18307/s
# re:        12820/s
# regex:     14035/s

def find_prefix_at_end(haystack, needle):
    l = len(needle) - 1
    while l and not haystack.endswith(needle[:l]):
        l -= 1
    return l
<|endoftext|>#! /usr/bin/python3.10

# Copyright 1994 by Lance Ellinghouse
# Cathedral City, California Republic, United States of America.
#                        All Rights Reserved
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Lance Ellinghouse
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# LANCE ELLINGHOUSE DISCLAIMS ALL WARRANTIES WITH REGARD TO
# THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS, IN NO EVENT SHALL LANCE ELLINGHOUSE CENTRUM BE LIABLE
# FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# Modified by Jack Jansen, CWI, July 1995:
# - Use binascii module to do the actual line-by-line conversion
#   between ascii and binary. This results in a 1000-fold speedup. The C
#   version is still 5 times faster, though.
# - Arguments more compliant with python standard

"""Implementation of the UUencode and UUdecode functions.

encode(in_file, out_file [,name, mode], *, backtick=False)
decode(in_file [, out_file, mode, quiet])
"""

import binascii
import os
import sys

__all__ = ["Error", "encode", "decode"]

class Error(Exception):
    pass

def encode(in_file, out_file, name=None, mode=None, *, backtick=False):
    """Uuencode file"""
    #
    # If in_file is a pathname open it and change defaults
    #
    opened_files = []
    try:
        if in_file == '-':
            in_file = sys.stdin.buffer
        elif isinstance(in_file, str):
            if name is None:
                name = os.path.basename(in_file)
            if mode is None:
                try:
                    mode = os.stat(in_file).st_mode
                except AttributeError:
                    pass
            in_file = open(in_file, 'rb')
            opened_files.append(in_file)
        #
        # Open out_file if it is a pathname
        #
        if out_file == '-':
            out_file = sys.stdout.buffer
        elif isinstance(out_file, str):
            out_file = open(out_file, 'wb')
            opened_files.append(out_file)
        #
        # Set defaults for name and mode
        #
        if name is None:
            name = '-'
        if mode is None:
            mode = 0o666

        #
        # Remove newline chars from name
        #
        name = name.replace('\n','\\n')
        name = name.replace('\r','\\r')

        #
        # Write the data
        #
        out_file.write(('begin %o %s\n' % ((mode & 0o777), name)).encode("ascii"))
        data = in_file.read(45)
        while len(data) > 0:
            out_file.write(binascii.b2a_uu(data, backtick=backtick))
            data = in_file.read(45)
        if backtick:
            out_file.write(b'`\nend\n')
        else:
            out_file.write(b' \nend\n')
    finally:
        for f in opened_files:
            f.close()


def decode(in_file, out_file=None, mode=None, quiet=False):
    """Decode uuencoded file"""
    #
    # Open the input file, if needed.
    #
    opened_files = []
    if in_file == '-':
        in_file = sys.stdin.buffer
    elif isinstance(in_file, str):
        in_file = open(in_file, 'rb')
        opened_files.append(in_file)

    try:
        #
        # Read until a begin is encountered or we've exhausted the file
        #
        while True:
            hdr = in_file.readline()
            if not hdr:
                raise Error('No valid begin line found in input file')
            if not hdr.startswith(b'begin'):
                continue
            hdrfields = hdr.split(b' ', 2)
            if len(hdrfields) == 3 and hdrfields[0] == b'begin':
                try:
                    int(hdrfields[1], 8)
                    break
                except ValueError:
                    pass
        if out_file is None:
            # If the filename isn't ASCII, what's up with that?!?
            out_file = hdrfields[2].rstrip(b' \t\r\n\f').decode("ascii")
            if os.path.exists(out_file):
                raise Error(f'Cannot overwrite existing file: {out_file}')
            if (out_file.startswith(os.sep) or
                f'..{os.sep}' in out_file or (
                    os.altsep and
                    (out_file.startswith(os.altsep) or
                     f'..{os.altsep}' in out_file))
               ):
                raise Error(f'Refusing to write to {out_file} due to directory traversal')
        if mode is None:
            mode = int(hdrfields[1], 8)
        #
        # Open the output file
        #
        if out_file == '-':
            out_file = sys.stdout.buffer
        elif isinstance(out_file, str):
            fp = open(out_file, 'wb')
            os.chmod(out_file, mode)
            out_file = fp
            opened_files.append(out_file)
        #
        # Main decoding loop
        #
        s = in_file.readline()
        while s and s.strip(b' \t\r\n\f') != b'end':
            try:
                data = binascii.a2b_uu(s)
            except binascii.Error as v:
                # Workaround for broken uuencoders by /Fredrik Lundh
                nbytes = (((s[0]-32) & 63) * 4 + 5) // 3
                data = binascii.a2b_uu(s[:nbytes])
                if not quiet:
                    sys.stderr.write("Warning: %s\n" % v)
            out_file.write(data)
            s = in_file.readline()
        if not s:
            raise Error('Truncated input file')
    finally:
        for f in opened_files:
            f.close()

def test():
    """uuencode/uudecode main program"""

    import optparse
    parser = optparse.OptionParser(usage='usage: %prog [-d] [-t] [input [output]]')
    parser.add_option('-d', '--decode', dest='decode', help='Decode (instead of encode)?', default=False, action='store_true')
    parser.add_option('-t', '--text', dest='text', help='data is text, encoded format unix-compatible text?', default=False, action='store_true')

    (options, args) = parser.parse_args()
    if len(args) > 2:
        parser.error('incorrect number of arguments')
        sys.exit(1)

    # Use the binary streams underlying stdin/stdout
    input = sys.stdin.buffer
    output = sys.stdout.buffer
    if len(args) > 0:
        input = args[0]
    if len(args) > 1:
        output = args[1]

    if options.decode:
        if options.text:
            if isinstance(output, str):
                output = open(output, 'wb')
            else:
                print(sys.argv[0], ': cannot do -t to stdout')
                sys.exit(1)
        decode(input, output)
    else:
        if options.text:
            if isinstance(input, str):
                input = open(input, 'rb')
            else:
                print(sys.argv[0], ': cannot do -t from stdin')
                sys.exit(1)
        encode(input, output)

if __name__ == '__main__':
    test()
<|endoftext|>""" codecs -- Python Codec Registry, API and helpers.


Written by Marc-Andre Lemburg (mal@lemburg.com).

(c) Copyright CNRI, All Rights Reserved. NO WARRANTY.

"""

import builtins
import sys

### Registry and builtin stateless codec functions

try:
    from _codecs import *
except ImportError as why:
    raise SystemError('Failed to load the builtin codecs: %s' % why)

__all__ = ["register", "lookup", "open", "EncodedFile", "BOM", "BOM_BE",
           "BOM_LE", "BOM32_BE", "BOM32_LE", "BOM64_BE", "BOM64_LE",
           "BOM_UTF8", "BOM_UTF16", "BOM_UTF16_LE", "BOM_UTF16_BE",
           "BOM_UTF32", "BOM_UTF32_LE", "BOM_UTF32_BE",
           "CodecInfo", "Codec", "IncrementalEncoder", "IncrementalDecoder",
           "StreamReader", "StreamWriter",
           "StreamReaderWriter", "StreamRecoder",
           "getencoder", "getdecoder", "getincrementalencoder",
           "getincrementaldecoder", "getreader", "getwriter",
           "encode", "decode", "iterencode", "iterdecode",
           "strict_errors", "ignore_errors", "replace_errors",
           "xmlcharrefreplace_errors",
           "backslashreplace_errors", "namereplace_errors",
           "register_error", "lookup_error"]

### Constants

#
# Byte Order Mark (BOM = ZERO WIDTH NO-BREAK SPACE = U+FEFF)
# and its possible byte string values
# for UTF8/UTF16/UTF32 output and little/big endian machines
#

# UTF-8
BOM_UTF8 = b'\xef\xbb\xbf'

# UTF-16, little endian
BOM_LE = BOM_UTF16_LE = b'\xff\xfe'

# UTF-16, big endian
BOM_BE = BOM_UTF16_BE = b'\xfe\xff'

# UTF-32, little endian
BOM_UTF32_LE = b'\xff\xfe\x00\x00'

# UTF-32, big endian
BOM_UTF32_BE = b'\x00\x00\xfe\xff'

if sys.byteorder == 'little':

    # UTF-16, native endianness
    BOM = BOM_UTF16 = BOM_UTF16_LE

    # UTF-32, native endianness
    BOM_UTF32 = BOM_UTF32_LE

else:

    # UTF-16, native endianness
    BOM = BOM_UTF16 = BOM_UTF16_BE

    # UTF-32, native endianness
    BOM_UTF32 = BOM_UTF32_BE

# Old broken names (don't use in new code)
BOM32_LE = BOM_UTF16_LE
BOM32_BE = BOM_UTF16_BE
BOM64_LE = BOM_UTF32_LE
BOM64_BE = BOM_UTF32_BE


### Codec base classes (defining the API)

class CodecInfo(tuple):
    """Codec details when looking up the codec registry"""

    # Private API to allow Python 3.4 to denylist the known non-Unicode
    # codecs in the standard library. A more general mechanism to
    # reliably distinguish test encodings from other codecs will hopefully
    # be defined for Python 3.5
    #
    # See http://bugs.python.org/issue19619
    _is_text_encoding = True # Assume codecs are text encodings by default

    def __new__(cls, encode, decode, streamreader=None, streamwriter=None,
        incrementalencoder=None, incrementaldecoder=None, name=None,
        *, _is_text_encoding=None):
        self = tuple.__new__(cls, (encode, decode, streamreader, streamwriter))
        self.name = name
        self.encode = encode
        self.decode = decode
        self.incrementalencoder = incrementalencoder
        self.incrementaldecoder = incrementaldecoder
        self.streamwriter = streamwriter
        self.streamreader = streamreader
        if _is_text_encoding is not None:
            self._is_text_encoding = _is_text_encoding
        return self

    def __repr__(self):
        return "<%s.%s object for encoding %s at %#x>" % \
                (self.__class__.__module__, self.__class__.__qualname__,
                 self.name, id(self))

class Codec:

    """ Defines the interface for stateless encoders/decoders.

        The .encode()/.decode() methods may use different error
        handling schemes by providing the errors argument. These
        string values are predefined:

         'strict' - raise a ValueError error (or a subclass)
         'ignore' - ignore the character and continue with the next
         'replace' - replace with a suitable replacement character;
                    Python will use the official U+FFFD REPLACEMENT
                    CHARACTER for the builtin Unicode codecs on
                    decoding and '?' on encoding.
         'surrogateescape' - replace with private code points U+DCnn.
         'xmlcharrefreplace' - Replace with the appropriate XML
                               character reference (only for encoding).
         'backslashreplace'  - Replace with backslashed escape sequences.
         'namereplace'       - Replace with \\N{...} escape sequences
                               (only for encoding).

        The set of allowed values can be extended via register_error.

    """
    def encode(self, input, errors='strict'):

        """ Encodes the object input and returns a tuple (output
            object, length consumed).

            errors defines the error handling to apply. It defaults to
            'strict' handling.

            The method may not store state in the Codec instance. Use
            StreamWriter for codecs which have to keep state in order to
            make encoding efficient.

            The encoder must be able to handle zero length input and
            return an empty object of the output object type in this
            situation.

        """
        raise NotImplementedError

    def decode(self, input, errors='strict'):

        """ Decodes the object input and returns a tuple (output
            object, length consumed).

            input must be an object which provides the bf_getreadbuf
            buffer slot. Python strings, buffer objects and memory
            mapped files are examples of objects providing this slot.

            errors defines the error handling to apply. It defaults to
            'strict' handling.

            The method may not store state in the Codec instance. Use
            StreamReader for codecs which have to keep state in order to
            make decoding efficient.

            The decoder must be able to handle zero length input and
            return an empty object of the output object type in this
            situation.

        """
        raise NotImplementedError

class IncrementalEncoder(object):
    """
    An IncrementalEncoder encodes an input in multiple steps. The input can
    be passed piece by piece to the encode() method. The IncrementalEncoder
    remembers the state of the encoding process between calls to encode().
    """
    def __init__(self, errors='strict'):
        """
        Creates an IncrementalEncoder instance.

        The IncrementalEncoder may use different error handling schemes by
        providing the errors keyword argument. See the module docstring
        for a list of possible values.
        """
        self.errors = errors
        self.buffer = ""

    def encode(self, input, final=False):
        """
        Encodes input and returns the resulting object.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the encoder to the initial state.
        """

    def getstate(self):
        """
        Return the current state of the encoder.
        """
        return 0

    def setstate(self, state):
        """
        Set the current state of the encoder. state must have been
        returned by getstate().
        """

class BufferedIncrementalEncoder(IncrementalEncoder):
    """
    This subclass of IncrementalEncoder can be used as the baseclass for an
    incremental encoder if the encoder must keep some of the output in a
    buffer between calls to encode().
    """
    def __init__(self, errors='strict'):
        IncrementalEncoder.__init__(self, errors)
        # unencoded input that is kept between calls to encode()
        self.buffer = ""

    def _buffer_encode(self, input, errors, final):
        # Overwrite this method in subclasses: It must encode input
        # and return an (output, length consumed) tuple
        raise NotImplementedError

    def encode(self, input, final=False):
        # encode input (taking the buffer into account)
        data = self.buffer + input
        (result, consumed) = self._buffer_encode(data, self.errors, final)
        # keep unencoded input until the next call
        self.buffer = data[consumed:]
        return result

    def reset(self):
        IncrementalEncoder.reset(self)
        self.buffer = ""

    def getstate(self):
        return self.buffer or 0

    def setstate(self, state):
        self.buffer = state or ""

class IncrementalDecoder(object):
    """
    An IncrementalDecoder decodes an input in multiple steps. The input can
    be passed piece by piece to the decode() method. The IncrementalDecoder
    remembers the state of the decoding process between calls to decode().
    """
    def __init__(self, errors='strict'):
        """
        Create an IncrementalDecoder instance.

        The IncrementalDecoder may use different error handling schemes by
        providing the errors keyword argument. See the module docstring
        for a list of possible values.
        """
        self.errors = errors

    def decode(self, input, final=False):
        """
        Decode input and returns the resulting object.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the decoder to the initial state.
        """

    def getstate(self):
        """
        Return the current state of the decoder.

        This must be a (buffered_input, additional_state_info) tuple.
        buffered_input must be a bytes object containing bytes that
        were passed to decode() that have not yet been converted.
        additional_state_info must be a non-negative integer
        representing the state of the decoder WITHOUT yet having
        processed the contents of buffered_input.  In the initial state
        and after reset(), getstate() must return (b"", 0).
        """
        return (b"", 0)

    def setstate(self, state):
        """
        Set the current state of the decoder.

        state must have been returned by getstate().  The effect of
        setstate((b"", 0)) must be equivalent to reset().
        """

class BufferedIncrementalDecoder(IncrementalDecoder):
    """
    This subclass of IncrementalDecoder can be used as the baseclass for an
    incremental decoder if the decoder must be able to handle incomplete
    byte sequences.
    """
    def __init__(self, errors='strict'):
        IncrementalDecoder.__init__(self, errors)
        # undecoded input that is kept between calls to decode()
        self.buffer = b""

    def _buffer_decode(self, input, errors, final):
        # Overwrite this method in subclasses: It must decode input
        # and return an (output, length consumed) tuple
        raise NotImplementedError

    def decode(self, input, final=False):
        # decode input (taking the buffer into account)
        data = self.buffer + input
        (result, consumed) = self._buffer_decode(data, self.errors, final)
        # keep undecoded input until the next call
        self.buffer = data[consumed:]
        return result

    def reset(self):
        IncrementalDecoder.reset(self)
        self.buffer = b""

    def getstate(self):
        # additional state info is always 0
        return (self.buffer, 0)

    def setstate(self, state):
        # ignore additional state info
        self.buffer = state[0]

#
# The StreamWriter and StreamReader class provide generic working
# interfaces which can be used to implement new encoding submodules
# very easily. See encodings/utf_8.py for an example on how this is
# done.
#

class StreamWriter(Codec):

    def __init__(self, stream, errors='strict'):

        """ Creates a StreamWriter instance.

            stream must be a file-like object open for writing.

            The StreamWriter may use different error handling
            schemes by providing the errors keyword argument. These
            parameters are predefined:

             'strict' - raise a ValueError (or a subclass)
             'ignore' - ignore the character and continue with the next
             'replace'- replace with a suitable replacement character
             'xmlcharrefreplace' - Replace with the appropriate XML
                                   character reference.
             'backslashreplace'  - Replace with backslashed escape
                                   sequences.
             'namereplace'       - Replace with \\N{...} escape sequences.

            The set of allowed parameter values can be extended via
            register_error.
        """
        self.stream = stream
        self.errors = errors

    def write(self, object):

        """ Writes the object's contents encoded to self.stream.
        """
        data, consumed = self.encode(object, self.errors)
        self.stream.write(data)

    def writelines(self, list):

        """ Writes the concatenated list of strings to the stream
            using .write().
        """
        self.write(''.join(list))

    def reset(self):

        """ Resets the codec buffers used for keeping internal state.

            Calling this method should ensure that the data on the
            output is put into a clean state, that allows appending
            of new fresh data without having to rescan the whole
            stream to recover state.

        """
        pass

    def seek(self, offset, whence=0):
        self.stream.seek(offset, whence)
        if whence == 0 and offset == 0:
            self.reset()

    def __getattr__(self, name,
                    getattr=getattr):

        """ Inherit all other methods from the underlying stream.
        """
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()

###

class StreamReader(Codec):

    charbuffertype = str

    def __init__(self, stream, errors='strict'):

        """ Creates a StreamReader instance.

            stream must be a file-like object open for reading.

            The StreamReader may use different error handling
            schemes by providing the errors keyword argument. These
            parameters are predefined:

             'strict' - raise a ValueError (or a subclass)
             'ignore' - ignore the character and continue with the next
             'replace'- replace with a suitable replacement character
             'backslashreplace' - Replace with backslashed escape sequences;

            The set of allowed parameter values can be extended via
            register_error.
        """
        self.stream = stream
        self.errors = errors
        self.bytebuffer = b""
        self._empty_charbuffer = self.charbuffertype()
        self.charbuffer = self._empty_charbuffer
        self.linebuffer = None

    def decode(self, input, errors='strict'):
        raise NotImplementedError

    def read(self, size=-1, chars=-1, firstline=False):

        """ Decodes data from the stream self.stream and returns the
            resulting object.

            chars indicates the number of decoded code points or bytes to
            return. read() will never return more data than requested,
            but it might return less, if there is not enough available.

            size indicates the approximate maximum number of decoded
            bytes or code points to read for decoding. The decoder
            can modify this setting as appropriate. The default value
            -1 indicates to read and decode as much as possible.  size
            is intended to prevent having to decode huge files in one
            step.

            If firstline is true, and a UnicodeDecodeError happens
            after the first line terminator in the input only the first line
            will be returned, the rest of the input will be kept until the
            next call to read().

            The method should use a greedy read strategy, meaning that
            it should read as much data as is allowed within the
            definition of the encoding and the given size, e.g.  if
            optional encoding endings or state markers are available
            on the stream, these should be read too.
        """
        # If we have lines cached, first merge them back into characters
        if self.linebuffer:
            self.charbuffer = self._empty_charbuffer.join(self.linebuffer)
            self.linebuffer = None

        if chars < 0:
            # For compatibility with other read() methods that take a
            # single argument
            chars = size

        # read until we get the required number of characters (if available)
        while True:
            # can the request be satisfied from the character buffer?
            if chars >= 0:
                if len(self.charbuffer) >= chars:
                    break
            # we need more data
            if size < 0:
                newdata = self.stream.read()
            else:
                newdata = self.stream.read(size)
            # decode bytes (those remaining from the last call included)
            data = self.bytebuffer + newdata
            if not data:
                break
            try:
                newchars, decodedbytes = self.decode(data, self.errors)
            except UnicodeDecodeError as exc:
                if firstline:
                    newchars, decodedbytes = \
                        self.decode(data[:exc.start], self.errors)
                    lines = newchars.splitlines(keepends=True)
                    if len(lines)<=1:
                        raise
                else:
                    raise
            # keep undecoded bytes until the next call
            self.bytebuffer = data[decodedbytes:]
            # put new characters in the character buffer
            self.charbuffer += newchars
            # there was no data available
            if not newdata:
                break
        if chars < 0:
            # Return everything we've got
            result = self.charbuffer
            self.charbuffer = self._empty_charbuffer
        else:
            # Return the first chars characters
            result = self.charbuffer[:chars]
            self.charbuffer = self.charbuffer[chars:]
        return result

    def readline(self, size=None, keepends=True):

        """ Read one line from the input stream and return the
            decoded data.

            size, if given, is passed as size argument to the
            read() method.

        """
        # If we have lines cached from an earlier read, return
        # them unconditionally
        if self.linebuffer:
            line = self.linebuffer[0]
            del self.linebuffer[0]
            if len(self.linebuffer) == 1:
                # revert to charbuffer mode; we might need more data
                # next time
                self.charbuffer = self.linebuffer[0]
                self.linebuffer = None
            if not keepends:
                line = line.splitlines(keepends=False)[0]
            return line

        readsize = size or 72
        line = self._empty_charbuffer
        # If size is given, we call read() only once
        while True:
            data = self.read(readsize, firstline=True)
            if data:
                # If we're at a "\r" read one extra character (which might
                # be a "\n") to get a proper line ending. If the stream is
                # temporarily exhausted we return the wrong line ending.
                if (isinstance(data, str) and data.endswith("\r")) or \
                   (isinstance(data, bytes) and data.endswith(b"\r")):
                    data += self.read(size=1, chars=1)

            line += data
            lines = line.splitlines(keepends=True)
            if lines:
                if len(lines) > 1:
                    # More than one line result; the first line is a full line
                    # to return
                    line = lines[0]
                    del lines[0]
                    if len(lines) > 1:
                        # cache the remaining lines
                        lines[-1] += self.charbuffer
                        self.linebuffer = lines
                        self.charbuffer = None
                    else:
                        # only one remaining line, put it back into charbuffer
                        self.charbuffer = lines[0] + self.charbuffer
                    if not keepends:
                        line = line.splitlines(keepends=False)[0]
                    break
                line0withend = lines[0]
                line0withoutend = lines[0].splitlines(keepends=False)[0]
                if line0withend != line0withoutend: # We really have a line end
                    # Put the rest back together and keep it until the next call
                    self.charbuffer = self._empty_charbuffer.join(lines[1:]) + \
                                      self.charbuffer
                    if keepends:
                        line = line0withend
                    else:
                        line = line0withoutend
                    break
            # we didn't get anything or this was our only try
            if not data or size is not None:
                if line and not keepends:
                    line = line.splitlines(keepends=False)[0]
                break
            if readsize < 8000:
                readsize *= 2
        return line

    def readlines(self, sizehint=None, keepends=True):

        """ Read all lines available on the input stream
            and return them as a list.

            Line breaks are implemented using the codec's decoder
            method and are included in the list entries.

            sizehint, if given, is ignored since there is no efficient
            way to finding the true end-of-line.

        """
        data = self.read()
        return data.splitlines(keepends)

    def reset(self):

        """ Resets the codec buffers used for keeping internal state.

            Note that no stream repositioning should take place.
            This method is primarily intended to be able to recover
            from decoding errors.

        """
        self.bytebuffer = b""
        self.charbuffer = self._empty_charbuffer
        self.linebuffer = None

    def seek(self, offset, whence=0):
        """ Set the input stream's current position.

            Resets the codec buffers used for keeping state.
        """
        self.stream.seek(offset, whence)
        self.reset()

    def __next__(self):

        """ Return the next decoded line from the input stream."""
        line = self.readline()
        if line:
            return line
        raise StopIteration

    def __iter__(self):
        return self

    def __getattr__(self, name,
                    getattr=getattr):

        """ Inherit all other methods from the underlying stream.
        """
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()

###

class StreamReaderWriter:

    """ StreamReaderWriter instances allow wrapping streams which
        work in both read and write modes.

        The design is such that one can use the factory functions
        returned by the codec.lookup() function to construct the
        instance.

    """
    # Optional attributes set by the file wrappers below
    encoding = 'unknown'

    def __init__(self, stream, Reader, Writer, errors='strict'):

        """ Creates a StreamReaderWriter instance.

            stream must be a Stream-like object.

            Reader, Writer must be factory functions or classes
            providing the StreamReader, StreamWriter interface resp.

            Error handling is done in the same way as defined for the
            StreamWriter/Readers.

        """
        self.stream = stream
        self.reader = Reader(stream, errors)
        self.writer = Writer(stream, errors)
        self.errors = errors

    def read(self, size=-1):

        return self.reader.read(size)

    def readline(self, size=None):

        return self.reader.readline(size)

    def readlines(self, sizehint=None):

        return self.reader.readlines(sizehint)

    def __next__(self):

        """ Return the next decoded line from the input stream."""
        return next(self.reader)

    def __iter__(self):
        return self

    def write(self, data):

        return self.writer.write(data)

    def writelines(self, list):

        return self.writer.writelines(list)

    def reset(self):

        self.reader.reset()
        self.writer.reset()

    def seek(self, offset, whence=0):
        self.stream.seek(offset, whence)
        self.reader.reset()
        if whence == 0 and offset == 0:
            self.writer.reset()

    def __getattr__(self, name,
                    getattr=getattr):

        """ Inherit all other methods from the underlying stream.
        """
        return getattr(self.stream, name)

    # these are needed to make "with StreamReaderWriter(...)" work properly

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()

###

class StreamRecoder:

    """ StreamRecoder instances translate data from one encoding to another.

        They use the complete set of APIs returned by the
        codecs.lookup() function to implement their task.

        Data written to the StreamRecoder is first decoded into an
        intermediate format (depending on the "decode" codec) and then
        written to the underlying stream using an instance of the provided
        Writer class.

        In the other direction, data is read from the underlying stream using
        a Reader instance and then encoded and returned to the caller.

    """
    # Optional attributes set by the file wrappers below
    data_encoding = 'unknown'
    file_encoding = 'unknown'

    def __init__(self, stream, encode, decode, Reader, Writer,
                 errors='strict'):

        """ Creates a StreamRecoder instance which implements a two-way
            conversion: encode and decode work on the frontend (the
            data visible to .read() and .write()) while Reader and Writer
            work on the backend (the data in stream).

            You can use these objects to do transparent
            transcodings from e.g. latin-1 to utf-8 and back.

            stream must be a file-like object.

            encode and decode must adhere to the Codec interface; Reader and
            Writer must be factory functions or classes providing the
            StreamReader and StreamWriter interfaces resp.

            Error handling is done in the same way as defined for the
            StreamWriter/Readers.

        """
        self.stream = stream
        self.encode = encode
        self.decode = decode
        self.reader = Reader(stream, errors)
        self.writer = Writer(stream, errors)
        self.errors = errors

    def read(self, size=-1):

        data = self.reader.read(size)
        data, bytesencoded = self.encode(data, self.errors)
        return data

    def readline(self, size=None):

        if size is None:
            data = self.reader.readline()
        else:
            data = self.reader.readline(size)
        data, bytesencoded = self.encode(data, self.errors)
        return data

    def readlines(self, sizehint=None):

        data = self.reader.read()
        data, bytesencoded = self.encode(data, self.errors)
        return data.splitlines(keepends=True)

    def __next__(self):

        """ Return the next decoded line from the input stream."""
        data = next(self.reader)
        data, bytesencoded = self.encode(data, self.errors)
        return data

    def __iter__(self):
        return self

    def write(self, data):

        data, bytesdecoded = self.decode(data, self.errors)
        return self.writer.write(data)

    def writelines(self, list):

        data = b''.join(list)
        data, bytesdecoded = self.decode(data, self.errors)
        return self.writer.write(data)

    def reset(self):

        self.reader.reset()
        self.writer.reset()

    def seek(self, offset, whence=0):
        # Seeks must be propagated to both the readers and writers
        # as they might need to reset their internal buffers.
        self.reader.seek(offset, whence)
        self.writer.seek(offset, whence)

    def __getattr__(self, name,
                    getattr=getattr):

        """ Inherit all other methods from the underlying stream.
        """
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stream.close()

### Shortcuts

def open(filename, mode='r', encoding=None, errors='strict', buffering=-1):

    """ Open an encoded file using the given mode and return
        a wrapped version providing transparent encoding/decoding.

        Note: The wrapped version will only accept the object format
        defined by the codecs, i.e. Unicode objects for most builtin
        codecs. Output is also codec dependent and will usually be
        Unicode as well.

        If encoding is not None, then the
        underlying encoded files are always opened in binary mode.
        The default file mode is 'r', meaning to open the file in read mode.

        encoding specifies the encoding which is to be used for the
        file.

        errors may be given to define the error handling. It defaults
        to 'strict' which causes ValueErrors to be raised in case an
        encoding error occurs.

        buffering has the same meaning as for the builtin open() API.
        It defaults to -1 which means that the default buffer size will
        be used.

        The returned wrapped file object provides an extra attribute
        .encoding which allows querying the used encoding. This
        attribute is only available if an encoding was specified as
        parameter.

    """
    if encoding is not None and \
       'b' not in mode:
        # Force opening of the file in binary mode
        mode = mode + 'b'
    file = builtins.open(filename, mode, buffering)
    if encoding is None:
        return file

    try:
        info = lookup(encoding)
        srw = StreamReaderWriter(file, info.streamreader, info.streamwriter, errors)
        # Add attributes to simplify introspection
        srw.encoding = encoding
        return srw
    except:
        file.close()
        raise

def EncodedFile(file, data_encoding, file_encoding=None, errors='strict'):

    """ Return a wrapped version of file which provides transparent
        encoding translation.

        Data written to the wrapped file is decoded according
        to the given data_encoding and then encoded to the underlying
        file using file_encoding. The intermediate data type
        will usually be Unicode but depends on the specified codecs.

        Bytes read from the file are decoded using file_encoding and then
        passed back to the caller encoded using data_encoding.

        If file_encoding is not given, it defaults to data_encoding.

        errors may be given to define the error handling. It defaults
        to 'strict' which causes ValueErrors to be raised in case an
        encoding error occurs.

        The returned wrapped file object provides two extra attributes
        .data_encoding and .file_encoding which reflect the given
        parameters of the same name. The attributes can be used for
        introspection by Python programs.

    """
    if file_encoding is None:
        file_encoding = data_encoding
    data_info = lookup(data_encoding)
    file_info = lookup(file_encoding)
    sr = StreamRecoder(file, data_info.encode, data_info.decode,
                       file_info.streamreader, file_info.streamwriter, errors)
    # Add attributes to simplify introspection
    sr.data_encoding = data_encoding
    sr.file_encoding = file_encoding
    return sr

### Helpers for codec lookup

def getencoder(encoding):

    """ Lookup up the codec for the given encoding and return
        its encoder function.

        Raises a LookupError in case the encoding cannot be found.

    """
    return lookup(encoding).encode

def getdecoder(encoding):

    """ Lookup up the codec for the given encoding and return
        its decoder function.

        Raises a LookupError in case the encoding cannot be found.

    """
    return lookup(encoding).decode

def getincrementalencoder(encoding):

    """ Lookup up the codec for the given encoding and return
        its IncrementalEncoder class or factory function.

        Raises a LookupError in case the encoding cannot be found
        or the codecs doesn't provide an incremental encoder.

    """
    encoder = lookup(encoding).incrementalencoder
    if encoder is None:
        raise LookupError(encoding)
    return encoder

def getincrementaldecoder(encoding):

    """ Lookup up the codec for the given encoding and return
        its IncrementalDecoder class or factory function.

        Raises a LookupError in case the encoding cannot be found
        or the codecs doesn't provide an incremental decoder.

    """
    decoder = lookup(encoding).incrementaldecoder
    if decoder is None:
        raise LookupError(encoding)
    return decoder

def getreader(encoding):

    """ Lookup up the codec for the given encoding and return
        its StreamReader class or factory function.

        Raises a LookupError in case the encoding cannot be found.

    """
    return lookup(encoding).streamreader

def getwriter(encoding):

    """ Lookup up the codec for the given encoding and return
        its StreamWriter class or factory function.

        Raises a LookupError in case the encoding cannot be found.

    """
    return lookup(encoding).streamwriter

def iterencode(iterator, encoding, errors='strict', **kwargs):
    """
    Encoding iterator.

    Encodes the input strings from the iterator using an IncrementalEncoder.

    errors and kwargs are passed through to the IncrementalEncoder
    constructor.
    """
    encoder = getincrementalencoder(encoding)(errors, **kwargs)
    for input in iterator:
        output = encoder.encode(input)
        if output:
            yield output
    output = encoder.encode("", True)
    if output:
        yield output

def iterdecode(iterator, encoding, errors='strict', **kwargs):
    """
    Decoding iterator.

    Decodes the input strings from the iterator using an IncrementalDecoder.

    errors and kwargs are passed through to the IncrementalDecoder
    constructor.
    """
    decoder = getincrementaldecoder(encoding)(errors, **kwargs)
    for input in iterator:
        output = decoder.decode(input)
        if output:
            yield output
    output = decoder.decode(b"", True)
    if output:
        yield output

### Helpers for charmap-based codecs

def make_identity_dict(rng):

    """ make_identity_dict(rng) -> dict

        Return a dictionary where elements of the rng sequence are
        mapped to themselves.

    """
    return {i:i for i in rng}

def make_encoding_map(decoding_map):

    """ Creates an encoding map from a decoding map.

        If a target mapping in the decoding map occurs multiple
        times, then that target is mapped to None (undefined mapping),
        causing an exception when encountered by the charmap codec
        during translation.

        One example where this happens is cp875.py which decodes
        multiple character to \\u001a.

    """
    m = {}
    for k,v in decoding_map.items():
        if not v in m:
            m[v] = k
        else:
            m[v] = None
    return m

### error handlers

try:
    strict_errors = lookup_error("strict")
    ignore_errors = lookup_error("ignore")
    replace_errors = lookup_error("replace")
    xmlcharrefreplace_errors = lookup_error("xmlcharrefreplace")
    backslashreplace_errors = lookup_error("backslashreplace")
    namereplace_errors = lookup_error("namereplace")
except LookupError:
    # In --disable-unicode builds, these error handler are missing
    strict_errors = None
    ignore_errors = None
    replace_errors = None
    xmlcharrefreplace_errors = None
    backslashreplace_errors = None
    namereplace_errors = None

# Tell modulefinder that using codecs probably needs the encodings
# package
_false = 0
if _false:
    import encodings

### Tests

if __name__ == '__main__':

    # Make stdout translate Latin-1 output into UTF-8 output
    sys.stdout = EncodedFile(sys.stdout, 'latin-1', 'utf-8')

    # Have stdin translate Latin-1 input into UTF-8 input
    sys.stdin = EncodedFile(sys.stdin, 'utf-8', 'latin-1')
<|endoftext|>"""Parser for command line options.

This module helps scripts to parse the command line arguments in
sys.argv.  It supports the same conventions as the Unix getopt()
function (including the special meanings of arguments of the form `-'
and `--').  Long options similar to those supported by GNU software
may be used as well via an optional third argument.  This module
provides two functions and an exception:

getopt() -- Parse command line options
gnu_getopt() -- Like getopt(), but allow option and non-option arguments
to be intermixed.
GetoptError -- exception (class) raised with 'opt' attribute, which is the
option involved with the exception.
"""

# Long option support added by Lars Wirzenius <liw@iki.fi>.
#
# Gerrit Holl <gerrit@nl.linux.org> moved the string-based exceptions
# to class-based exceptions.
#
# Peter Åstrand <astrand@lysator.liu.se> added gnu_getopt().
#
# TODO for gnu_getopt():
#
# - GNU getopt_long_only mechanism
# - allow the caller to specify ordering
# - RETURN_IN_ORDER option
# - GNU extension with '-' as first character of option string
# - optional arguments, specified by double colons
# - an option string with a W followed by semicolon should
#   treat "-W foo" as "--foo"

__all__ = ["GetoptError","error","getopt","gnu_getopt"]

import os
try:
    from gettext import gettext as _
except ImportError:
    # Bootstrapping Python: gettext's dependencies not built yet
    def _(s): return s

class GetoptError(Exception):
    opt = ''
    msg = ''
    def __init__(self, msg, opt=''):
        self.msg = msg
        self.opt = opt
        Exception.__init__(self, msg, opt)

    def __str__(self):
        return self.msg

error = GetoptError # backward compatibility

def getopt(args, shortopts, longopts = []):
    """getopt(args, options[, long_options]) -> opts, args

    Parses command line options and parameter list.  args is the
    argument list to be parsed, without the leading reference to the
    running program.  Typically, this means "sys.argv[1:]".  shortopts
    is the string of option letters that the script wants to
    recognize, with options that require an argument followed by a
    colon (i.e., the same format that Unix getopt() uses).  If
    specified, longopts is a list of strings with the names of the
    long options which should be supported.  The leading '--'
    characters should not be included in the option name.  Options
    which require an argument should be followed by an equal sign
    ('=').

    The return value consists of two elements: the first is a list of
    (option, value) pairs; the second is the list of program arguments
    left after the option list was stripped (this is a trailing slice
    of the first argument).  Each option-and-value pair returned has
    the option as its first element, prefixed with a hyphen (e.g.,
    '-x'), and the option argument as its second element, or an empty
    string if the option has no argument.  The options occur in the
    list in the same order in which they were found, thus allowing
    multiple occurrences.  Long and short options may be mixed.

    """

    opts = []
    if type(longopts) == type(""):
        longopts = [longopts]
    else:
        longopts = list(longopts)
    while args and args[0].startswith('-') and args[0] != '-':
        if args[0] == '--':
            args = args[1:]
            break
        if args[0].startswith('--'):
            opts, args = do_longs(opts, args[0][2:], longopts, args[1:])
        else:
            opts, args = do_shorts(opts, args[0][1:], shortopts, args[1:])

    return opts, args

def gnu_getopt(args, shortopts, longopts = []):
    """getopt(args, options[, long_options]) -> opts, args

    This function works like getopt(), except that GNU style scanning
    mode is used by default. This means that option and non-option
    arguments may be intermixed. The getopt() function stops
    processing options as soon as a non-option argument is
    encountered.

    If the first character of the option string is `+', or if the
    environment variable POSIXLY_CORRECT is set, then option
    processing stops as soon as a non-option argument is encountered.

    """

    opts = []
    prog_args = []
    if isinstance(longopts, str):
        longopts = [longopts]
    else:
        longopts = list(longopts)

    # Allow options after non-option arguments?
    if shortopts.startswith('+'):
        shortopts = shortopts[1:]
        all_options_first = True
    elif os.environ.get("POSIXLY_CORRECT"):
        all_options_first = True
    else:
        all_options_first = False

    while args:
        if args[0] == '--':
            prog_args += args[1:]
            break

        if args[0][:2] == '--':
            opts, args = do_longs(opts, args[0][2:], longopts, args[1:])
        elif args[0][:1] == '-' and args[0] != '-':
            opts, args = do_shorts(opts, args[0][1:], shortopts, args[1:])
        else:
            if all_options_first:
                prog_args += args
                break
            else:
                prog_args.append(args[0])
                args = args[1:]

    return opts, prog_args

def do_longs(opts, opt, longopts, args):
    try:
        i = opt.index('=')
    except ValueError:
        optarg = None
    else:
        opt, optarg = opt[:i], opt[i+1:]

    has_arg, opt = long_has_args(opt, longopts)
    if has_arg:
        if optarg is None:
            if not args:
                raise GetoptError(_('option --%s requires argument') % opt, opt)
            optarg, args = args[0], args[1:]
    elif optarg is not None:
        raise GetoptError(_('option --%s must not have an argument') % opt, opt)
    opts.append(('--' + opt, optarg or ''))
    return opts, args

# Return:
#   has_arg?
#   full option name
def long_has_args(opt, longopts):
    possibilities = [o for o in longopts if o.startswith(opt)]
    if not possibilities:
        raise GetoptError(_('option --%s not recognized') % opt, opt)
    # Is there an exact match?
    if opt in possibilities:
        return False, opt
    elif opt + '=' in possibilities:
        return True, opt
    # No exact match, so better be unique.
    if len(possibilities) > 1:
        # XXX since possibilities contains all valid continuations, might be
        # nice to work them into the error msg
        raise GetoptError(_('option --%s not a unique prefix') % opt, opt)
    assert len(possibilities) == 1
    unique_match = possibilities[0]
    has_arg = unique_match.endswith('=')
    if has_arg:
        unique_match = unique_match[:-1]
    return has_arg, unique_match

def do_shorts(opts, optstring, shortopts, args):
    while optstring != '':
        opt, optstring = optstring[0], optstring[1:]
        if short_has_arg(opt, shortopts):
            if optstring == '':
                if not args:
                    raise GetoptError(_('option -%s requires argument') % opt,
                                      opt)
                optstring, args = args[0], args[1:]
            optarg, optstring = optstring, ''
        else:
            optarg = ''
        opts.append(('-' + opt, optarg))
    return opts, args

def short_has_arg(opt, shortopts):
    for i in range(len(shortopts)):
        if opt == shortopts[i] != ':':
            return shortopts.startswith(':', i+1)
    raise GetoptError(_('option -%s not recognized') % opt, opt)

if __name__ == '__main__':
    import sys
    print(getopt(sys.argv[1:], "a:b", ["alpha=", "beta"]))
<|endoftext|>#! /usr/bin/python3.10

"""Conversions to/from quoted-printable transport encoding as per RFC 1521."""

# (Dec 1991 version).

__all__ = ["encode", "decode", "encodestring", "decodestring"]

ESCAPE = b'='
MAXLINESIZE = 76
HEX = b'0123456789ABCDEF'
EMPTYSTRING = b''

try:
    from binascii import a2b_qp, b2a_qp
except ImportError:
    a2b_qp = None
    b2a_qp = None


def needsquoting(c, quotetabs, header):
    """Decide whether a particular byte ordinal needs to be quoted.

    The 'quotetabs' flag indicates whether embedded tabs and spaces should be
    quoted.  Note that line-ending tabs and spaces are always encoded, as per
    RFC 1521.
    """
    assert isinstance(c, bytes)
    if c in b' \t':
        return quotetabs
    # if header, we have to escape _ because _ is used to escape space
    if c == b'_':
        return header
    return c == ESCAPE or not (b' ' <= c <= b'~')

def quote(c):
    """Quote a single character."""
    assert isinstance(c, bytes) and len(c)==1
    c = ord(c)
    return ESCAPE + bytes((HEX[c//16], HEX[c%16]))



def encode(input, output, quotetabs, header=False):
    """Read 'input', apply quoted-printable encoding, and write to 'output'.

    'input' and 'output' are binary file objects. The 'quotetabs' flag
    indicates whether embedded tabs and spaces should be quoted. Note that
    line-ending tabs and spaces are always encoded, as per RFC 1521.
    The 'header' flag indicates whether we are encoding spaces as _ as per RFC
    1522."""

    if b2a_qp is not None:
        data = input.read()
        odata = b2a_qp(data, quotetabs=quotetabs, header=header)
        output.write(odata)
        return

    def write(s, output=output, lineEnd=b'\n'):
        # RFC 1521 requires that the line ending in a space or tab must have
        # that trailing character encoded.
        if s and s[-1:] in b' \t':
            output.write(s[:-1] + quote(s[-1:]) + lineEnd)
        elif s == b'.':
            output.write(quote(s) + lineEnd)
        else:
            output.write(s + lineEnd)

    prevline = None
    while 1:
        line = input.readline()
        if not line:
            break
        outline = []
        # Strip off any readline induced trailing newline
        stripped = b''
        if line[-1:] == b'\n':
            line = line[:-1]
            stripped = b'\n'
        # Calculate the un-length-limited encoded line
        for c in line:
            c = bytes((c,))
            if needsquoting(c, quotetabs, header):
                c = quote(c)
            if header and c == b' ':
                outline.append(b'_')
            else:
                outline.append(c)
        # First, write out the previous line
        if prevline is not None:
            write(prevline)
        # Now see if we need any soft line breaks because of RFC-imposed
        # length limitations.  Then do the thisline->prevline dance.
        thisline = EMPTYSTRING.join(outline)
        while len(thisline) > MAXLINESIZE:
            # Don't forget to include the soft line break `=' sign in the
            # length calculation!
            write(thisline[:MAXLINESIZE-1], lineEnd=b'=\n')
            thisline = thisline[MAXLINESIZE-1:]
        # Write out the current line
        prevline = thisline
    # Write out the last line, without a trailing newline
    if prevline is not None:
        write(prevline, lineEnd=stripped)

def encodestring(s, quotetabs=False, header=False):
    if b2a_qp is not None:
        return b2a_qp(s, quotetabs=quotetabs, header=header)
    from io import BytesIO
    infp = BytesIO(s)
    outfp = BytesIO()
    encode(infp, outfp, quotetabs, header)
    return outfp.getvalue()



def decode(input, output, header=False):
    """Read 'input', apply quoted-printable decoding, and write to 'output'.
    'input' and 'output' are binary file objects.
    If 'header' is true, decode underscore as space (per RFC 1522)."""

    if a2b_qp is not None:
        data = input.read()
        odata = a2b_qp(data, header=header)
        output.write(odata)
        return

    new = b''
    while 1:
        line = input.readline()
        if not line: break
        i, n = 0, len(line)
        if n > 0 and line[n-1:n] == b'\n':
            partial = 0; n = n-1
            # Strip trailing whitespace
            while n > 0 and line[n-1:n] in b" \t\r":
                n = n-1
        else:
            partial = 1
        while i < n:
            c = line[i:i+1]
            if c == b'_' and header:
                new = new + b' '; i = i+1
            elif c != ESCAPE:
                new = new + c; i = i+1
            elif i+1 == n and not partial:
                partial = 1; break
            elif i+1 < n and line[i+1:i+2] == ESCAPE:
                new = new + ESCAPE; i = i+2
            elif i+2 < n and ishex(line[i+1:i+2]) and ishex(line[i+2:i+3]):
                new = new + bytes((unhex(line[i+1:i+3]),)); i = i+3
            else: # Bad escape sequence -- leave it in
                new = new + c; i = i+1
        if not partial:
            output.write(new + b'\n')
            new = b''
    if new:
        output.write(new)

def decodestring(s, header=False):
    if a2b_qp is not None:
        return a2b_qp(s, header=header)
    from io import BytesIO
    infp = BytesIO(s)
    outfp = BytesIO()
    decode(infp, outfp, header=header)
    return outfp.getvalue()



# Other helper functions
def ishex(c):
    """Return true if the byte ordinal 'c' is a hexadecimal digit in ASCII."""
    assert isinstance(c, bytes)
    return b'0' <= c <= b'9' or b'a' <= c <= b'f' or b'A' <= c <= b'F'

def unhex(s):
    """Get the integer value of a hexadecimal number."""
    bits = 0
    for c in s:
        c = bytes((c,))
        if b'0' <= c <= b'9':
            i = ord('0')
        elif b'a' <= c <= b'f':
            i = ord('a')-10
        elif b'A' <= c <= b'F':
            i = ord(b'A')-10
        else:
            assert False, "non-hex digit "+repr(c)
        bits = bits*16 + (ord(c) - i)
    return bits



def main():
    import sys
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'td')
    except getopt.error as msg:
        sys.stdout = sys.stderr
        print(msg)
        print("usage: quopri [-t | -d] [file] ...")
        print("-t: quote tabs")
        print("-d: decode; default encode")
        sys.exit(2)
    deco = False
    tabs = False
    for o, a in opts:
        if o == '-t': tabs = True
        if o == '-d': deco = True
    if tabs and deco:
        sys.stdout = sys.stderr
        print("-t and -d are mutually exclusive")
        sys.exit(2)
    if not args: args = ['-']
    sts = 0
    for file in args:
        if file == '-':
            fp = sys.stdin.buffer
        else:
            try:
                fp = open(file, "rb")
            except OSError as msg:
                sys.stderr.write("%s: can't open (%s)\n" % (file, msg))
                sts = 1
                continue
        try:
            if deco:
                decode(fp, sys.stdout.buffer)
            else:
                encode(fp, sys.stdout.buffer, tabs)
        finally:
            if file != '-':
                fp.close()
    if sts:
        sys.exit(sts)



if __name__ == '__main__':
    main()
<|endoftext|>r"""OS routines for NT or Posix depending on what system we're on.

This exports:
  - all functions from posix or nt, e.g. unlink, stat, etc.
  - os.path is either posixpath or ntpath
  - os.name is either 'posix' or 'nt'
  - os.curdir is a string representing the current directory (always '.')
  - os.pardir is a string representing the parent directory (always '..')
  - os.sep is the (or a most common) pathname separator ('/' or '\\')
  - os.extsep is the extension separator (always '.')
  - os.altsep is the alternate pathname separator (None or '/')
  - os.pathsep is the component separator used in $PATH etc
  - os.linesep is the line separator in text files ('\r' or '\n' or '\r\n')
  - os.defpath is the default search path for executables
  - os.devnull is the file path of the null device ('/dev/null', etc.)

Programs that import and use 'os' stand a better chance of being
portable between different platforms.  Of course, they must then
only use functions that are defined by all platforms (e.g., unlink
and opendir), and leave all pathname manipulation to os.path
(e.g., split and join).
"""

#'
import abc
import sys
import stat as st

from _collections_abc import _check_methods

GenericAlias = type(list[int])

_names = sys.builtin_module_names

# Note:  more names are added to __all__ later.
__all__ = ["altsep", "curdir", "pardir", "sep", "pathsep", "linesep",
           "defpath", "name", "path", "devnull", "SEEK_SET", "SEEK_CUR",
           "SEEK_END", "fsencode", "fsdecode", "get_exec_path", "fdopen",
           "extsep"]

def _exists(name):
    return name in globals()

def _get_exports_list(module):
    try:
        return list(module.__all__)
    except AttributeError:
        return [n for n in dir(module) if n[0] != '_']

# Any new dependencies of the os module and/or changes in path separator
# requires updating importlib as well.
if 'posix' in _names:
    name = 'posix'
    linesep = '\n'
    from posix import *
    try:
        from posix import _exit
        __all__.append('_exit')
    except ImportError:
        pass
    import posixpath as path

    try:
        from posix import _have_functions
    except ImportError:
        pass

    import posix
    __all__.extend(_get_exports_list(posix))
    del posix

elif 'nt' in _names:
    name = 'nt'
    linesep = '\r\n'
    from nt import *
    try:
        from nt import _exit
        __all__.append('_exit')
    except ImportError:
        pass
    import ntpath as path

    import nt
    __all__.extend(_get_exports_list(nt))
    del nt

    try:
        from nt import _have_functions
    except ImportError:
        pass

else:
    raise ImportError('no os specific module found')

sys.modules['os.path'] = path
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
    devnull)

del _names


if _exists("_have_functions"):
    _globals = globals()
    def _add(str, fn):
        if (fn in _globals) and (str in _have_functions):
            _set.add(_globals[fn])

    _set = set()
    _add("HAVE_FACCESSAT",  "access")
    _add("HAVE_FCHMODAT",   "chmod")
    _add("HAVE_FCHOWNAT",   "chown")
    _add("HAVE_FSTATAT",    "stat")
    _add("HAVE_FUTIMESAT",  "utime")
    _add("HAVE_LINKAT",     "link")
    _add("HAVE_MKDIRAT",    "mkdir")
    _add("HAVE_MKFIFOAT",   "mkfifo")
    _add("HAVE_MKNODAT",    "mknod")
    _add("HAVE_OPENAT",     "open")
    _add("HAVE_READLINKAT", "readlink")
    _add("HAVE_RENAMEAT",   "rename")
    _add("HAVE_SYMLINKAT",  "symlink")
    _add("HAVE_UNLINKAT",   "unlink")
    _add("HAVE_UNLINKAT",   "rmdir")
    _add("HAVE_UTIMENSAT",  "utime")
    supports_dir_fd = _set

    _set = set()
    _add("HAVE_FACCESSAT",  "access")
    supports_effective_ids = _set

    _set = set()
    _add("HAVE_FCHDIR",     "chdir")
    _add("HAVE_FCHMOD",     "chmod")
    _add("HAVE_FCHOWN",     "chown")
    _add("HAVE_FDOPENDIR",  "listdir")
    _add("HAVE_FDOPENDIR",  "scandir")
    _add("HAVE_FEXECVE",    "execve")
    _set.add(stat) # fstat always works
    _add("HAVE_FTRUNCATE",  "truncate")
    _add("HAVE_FUTIMENS",   "utime")
    _add("HAVE_FUTIMES",    "utime")
    _add("HAVE_FPATHCONF",  "pathconf")
    if _exists("statvfs") and _exists("fstatvfs"): # mac os x10.3
        _add("HAVE_FSTATVFS", "statvfs")
    supports_fd = _set

    _set = set()
    _add("HAVE_FACCESSAT",  "access")
    # Some platforms don't support lchmod().  Often the function exists
    # anyway, as a stub that always returns ENOSUP or perhaps EOPNOTSUPP.
    # (No, I don't know why that's a good design.)  ./configure will detect
    # this and reject it--so HAVE_LCHMOD still won't be defined on such
    # platforms.  This is Very Helpful.
    #
    # However, sometimes platforms without a working lchmod() *do* have
    # fchmodat().  (Examples: Linux kernel 3.2 with glibc 2.15,
    # OpenIndiana 3.x.)  And fchmodat() has a flag that theoretically makes
    # it behave like lchmod().  So in theory it would be a suitable
    # replacement for lchmod().  But when lchmod() doesn't work, fchmodat()'s
    # flag doesn't work *either*.  Sadly ./configure isn't sophisticated
    # enough to detect this condition--it only determines whether or not
    # fchmodat() minimally works.
    #
    # Therefore we simply ignore fchmodat() when deciding whether or not
    # os.chmod supports follow_symlinks.  Just checking lchmod() is
    # sufficient.  After all--if you have a working fchmodat(), your
    # lchmod() almost certainly works too.
    #
    # _add("HAVE_FCHMODAT",   "chmod")
    _add("HAVE_FCHOWNAT",   "chown")
    _add("HAVE_FSTATAT",    "stat")
    _add("HAVE_LCHFLAGS",   "chflags")
    _add("HAVE_LCHMOD",     "chmod")
    if _exists("lchown"): # mac os x10.3
        _add("HAVE_LCHOWN", "chown")
    _add("HAVE_LINKAT",     "link")
    _add("HAVE_LUTIMES",    "utime")
    _add("HAVE_LSTAT",      "stat")
    _add("HAVE_FSTATAT",    "stat")
    _add("HAVE_UTIMENSAT",  "utime")
    _add("MS_WINDOWS",      "stat")
    supports_follow_symlinks = _set

    del _set
    del _have_functions
    del _globals
    del _add


# Python uses fixed values for the SEEK_ constants; they are mapped
# to native constants if necessary in posixmodule.c
# Other possible SEEK values are directly imported from posixmodule.c
SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

# Super directory utilities.
# (Inspired by Eric Raymond; the doc strings are mostly his)

def makedirs(name, mode=0o777, exist_ok=False):
    """makedirs(name [, mode=0o777][, exist_ok=False])

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    """
    head, tail = path.split(name)
    if not tail:
        head, tail = path.split(head)
    if head and tail and not path.exists(head):
        try:
            makedirs(head, exist_ok=exist_ok)
        except FileExistsError:
            # Defeats race condition when another thread created the path
            pass
        cdir = curdir
        if isinstance(tail, bytes):
            cdir = bytes(curdir, 'ASCII')
        if tail == cdir:           # xxx/newdir/. exists if xxx/newdir exists
            return
    try:
        mkdir(name, mode)
    except OSError:
        # Cannot rely on checking for EEXIST, since the operating system
        # could give priority to other errors like EACCES or EROFS
        if not exist_ok or not path.isdir(name):
            raise

def removedirs(name):
    """removedirs(name)

    Super-rmdir; remove a leaf directory and all empty intermediate
    ones.  Works like rmdir except that, if the leaf directory is
    successfully removed, directories corresponding to rightmost path
    segments will be pruned away until either the whole path is
    consumed or an error occurs.  Errors during this latter phase are
    ignored -- they generally mean that a directory was not empty.

    """
    rmdir(name)
    head, tail = path.split(name)
    if not tail:
        head, tail = path.split(head)
    while head and tail:
        try:
            rmdir(head)
        except OSError:
            break
        head, tail = path.split(head)

def renames(old, new):
    """renames(old, new)

    Super-rename; create directories as necessary and delete any left
    empty.  Works like rename, except creation of any intermediate
    directories needed to make the new pathname good is attempted
    first.  After the rename, directories corresponding to rightmost
    path segments of the old name will be pruned until either the
    whole path is consumed or a nonempty directory is found.

    Note: this function can fail with the new directory structure made
    if you lack permissions needed to unlink the leaf directory or
    file.

    """
    head, tail = path.split(new)
    if head and tail and not path.exists(head):
        makedirs(head)
    rename(old, new)
    head, tail = path.split(old)
    if head and tail:
        try:
            removedirs(head)
        except OSError:
            pass

__all__.extend(["makedirs", "removedirs", "renames"])

def walk(top, topdown=True, onerror=None, followlinks=False):
    """Directory tree generator.

    For each directory in the directory tree rooted at top (including top
    itself, but excluding '.' and '..'), yields a 3-tuple

        dirpath, dirnames, filenames

    dirpath is a string, the path to the directory.  dirnames is a list of
    the names of the subdirectories in dirpath (including symlinks to directories,
    and excluding '.' and '..').
    filenames is a list of the names of the non-directory files in dirpath.
    Note that the names in the lists are just names, with no path components.
    To get a full path (which begins with top) to a file or directory in
    dirpath, do os.path.join(dirpath, name).

    If optional arg 'topdown' is true or not specified, the triple for a
    directory is generated before the triples for any of its subdirectories
    (directories are generated top down).  If topdown is false, the triple
    for a directory is generated after the triples for all of its
    subdirectories (directories are generated bottom up).

    When topdown is true, the caller can modify the dirnames list in-place
    (e.g., via del or slice assignment), and walk will only recurse into the
    subdirectories whose names remain in dirnames; this can be used to prune the
    search, or to impose a specific order of visiting.  Modifying dirnames when
    topdown is false has no effect on the behavior of os.walk(), since the
    directories in dirnames have already been generated by the time dirnames
    itself is generated. No matter the value of topdown, the list of
    subdirectories is retrieved before the tuples for the directory and its
    subdirectories are generated.

    By default errors from the os.scandir() call are ignored.  If
    optional arg 'onerror' is specified, it should be a function; it
    will be called with one argument, an OSError instance.  It can
    report the error to continue with the walk, or raise the exception
    to abort the walk.  Note that the filename is available as the
    filename attribute of the exception object.

    By default, os.walk does not follow symbolic links to subdirectories on
    systems that support them.  In order to get this functionality, set the
    optional argument 'followlinks' to true.

    Caution:  if you pass a relative pathname for top, don't change the
    current working directory between resumptions of walk.  walk never
    changes the current directory, and assumes that the client doesn't
    either.

    Example:

    import os
    from os.path import join, getsize
    for root, dirs, files in os.walk('python/Lib/email'):
        print(root, "consumes", end="")
        print(sum(getsize(join(root, name)) for name in files), end="")
        print("bytes in", len(files), "non-directory files")
        if 'CVS' in dirs:
            dirs.remove('CVS')  # don't visit CVS directories

    """
    sys.audit("os.walk", top, topdown, onerror, followlinks)
    return _walk(fspath(top), topdown, onerror, followlinks)

def _walk(top, topdown, onerror, followlinks):
    dirs = []
    nondirs = []
    walk_dirs = []

    # We may not have read permission for top, in which case we can't
    # get a list of the files the directory contains.  os.walk
    # always suppressed the exception then, rather than blow up for a
    # minor reason when (say) a thousand readable directories are still
    # left to visit.  That logic is copied here.
    try:
        # Note that scandir is global in this module due
        # to earlier import-*.
        scandir_it = scandir(top)
    except OSError as error:
        if onerror is not None:
            onerror(error)
        return

    with scandir_it:
        while True:
            try:
                try:
                    entry = next(scandir_it)
                except StopIteration:
                    break
            except OSError as error:
                if onerror is not None:
                    onerror(error)
                return

            try:
                is_dir = entry.is_dir()
            except OSError:
                # If is_dir() raises an OSError, consider that the entry is not
                # a directory, same behaviour than os.path.isdir().
                is_dir = False

            if is_dir:
                dirs.append(entry.name)
            else:
                nondirs.append(entry.name)

            if not topdown and is_dir:
                # Bottom-up: recurse into sub-directory, but exclude symlinks to
                # directories if followlinks is False
                if followlinks:
                    walk_into = True
                else:
                    try:
                        is_symlink = entry.is_symlink()
                    except OSError:
                        # If is_symlink() raises an OSError, consider that the
                        # entry is not a symbolic link, same behaviour than
                        # os.path.islink().
                        is_symlink = False
                    walk_into = not is_symlink

                if walk_into:
                    walk_dirs.append(entry.path)

    # Yield before recursion if going top down
    if topdown:
        yield top, dirs, nondirs

        # Recurse into sub-directories
        islink, join = path.islink, path.join
        for dirname in dirs:
            new_path = join(top, dirname)
            # Issue #23605: os.path.islink() is used instead of caching
            # entry.is_symlink() result during the loop on os.scandir() because
            # the caller can replace the directory entry during the "yield"
            # above.
            if followlinks or not islink(new_path):
                yield from _walk(new_path, topdown, onerror, followlinks)
    else:
        # Recurse into sub-directories
        for new_path in walk_dirs:
            yield from _walk(new_path, topdown, onerror, followlinks)
        # Yield after recursion if going bottom up
        yield top, dirs, nondirs

__all__.append("walk")

if {open, stat} <= supports_dir_fd and {scandir, stat} <= supports_fd:

    def fwalk(top=".", topdown=True, onerror=None, *, follow_symlinks=False, dir_fd=None):
        """Directory tree generator.

        This behaves exactly like walk(), except that it yields a 4-tuple

            dirpath, dirnames, filenames, dirfd

        `dirpath`, `dirnames` and `filenames` are identical to walk() output,
        and `dirfd` is a file descriptor referring to the directory `dirpath`.

        The advantage of fwalk() over walk() is that it's safe against symlink
        races (when follow_symlinks is False).

        If dir_fd is not None, it should be a file descriptor open to a directory,
          and top should be relative; top will then be relative to that directory.
          (dir_fd is always supported for fwalk.)

        Caution:
        Since fwalk() yields file descriptors, those are only valid until the
        next iteration step, so you should dup() them if you want to keep them
        for a longer period.

        Example:

        import os
        for root, dirs, files, rootfd in os.fwalk('python/Lib/email'):
            print(root, "consumes", end="")
            print(sum(os.stat(name, dir_fd=rootfd).st_size for name in files),
                  end="")
            print("bytes in", len(files), "non-directory files")
            if 'CVS' in dirs:
                dirs.remove('CVS')  # don't visit CVS directories
        """
        sys.audit("os.fwalk", top, topdown, onerror, follow_symlinks, dir_fd)
        if not isinstance(top, int) or not hasattr(top, '__index__'):
            top = fspath(top)
        # Note: To guard against symlink races, we use the standard
        # lstat()/open()/fstat() trick.
        if not follow_symlinks:
            orig_st = stat(top, follow_symlinks=False, dir_fd=dir_fd)
        topfd = open(top, O_RDONLY, dir_fd=dir_fd)
        try:
            if (follow_symlinks or (st.S_ISDIR(orig_st.st_mode) and
                                    path.samestat(orig_st, stat(topfd)))):
                yield from _fwalk(topfd, top, isinstance(top, bytes),
                                  topdown, onerror, follow_symlinks)
        finally:
            close(topfd)

    def _fwalk(topfd, toppath, isbytes, topdown, onerror, follow_symlinks):
        # Note: This uses O(depth of the directory tree) file descriptors: if
        # necessary, it can be adapted to only require O(1) FDs, see issue
        # #13734.

        scandir_it = scandir(topfd)
        dirs = []
        nondirs = []
        entries = None if topdown or follow_symlinks else []
        for entry in scandir_it:
            name = entry.name
            if isbytes:
                name = fsencode(name)
            try:
                if entry.is_dir():
                    dirs.append(name)
                    if entries is not None:
                        entries.append(entry)
                else:
                    nondirs.append(name)
            except OSError:
                try:
                    # Add dangling symlinks, ignore disappeared files
                    if entry.is_symlink():
                        nondirs.append(name)
                except OSError:
                    pass

        if topdown:
            yield toppath, dirs, nondirs, topfd

        for name in dirs if entries is None else zip(dirs, entries):
            try:
                if not follow_symlinks:
                    if topdown:
                        orig_st = stat(name, dir_fd=topfd, follow_symlinks=False)
                    else:
                        assert entries is not None
                        name, entry = name
                        orig_st = entry.stat(follow_symlinks=False)
                dirfd = open(name, O_RDONLY, dir_fd=topfd)
            except OSError as err:
                if onerror is not None:
                    onerror(err)
                continue
            try:
                if follow_symlinks or path.samestat(orig_st, stat(dirfd)):
                    dirpath = path.join(toppath, name)
                    yield from _fwalk(dirfd, dirpath, isbytes,
                                      topdown, onerror, follow_symlinks)
            finally:
                close(dirfd)

        if not topdown:
            yield toppath, dirs, nondirs, topfd

    __all__.append("fwalk")

def execl(file, *args):
    """execl(file, *args)

    Execute the executable file with argument list args, replacing the
    current process. """
    execv(file, args)

def execle(file, *args):
    """execle(file, *args, env)

    Execute the executable file with argument list args and
    environment env, replacing the current process. """
    env = args[-1]
    execve(file, args[:-1], env)

def execlp(file, *args):
    """execlp(file, *args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process. """
    execvp(file, args)

def execlpe(file, *args):
    """execlpe(file, *args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env, replacing the current
    process. """
    env = args[-1]
    execvpe(file, args[:-1], env)

def execvp(file, args):
    """execvp(file, args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process.
    args may be a list or tuple of strings. """
    _execvpe(file, args)

def execvpe(file, args, env):
    """execvpe(file, args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env, replacing the
    current process.
    args may be a list or tuple of strings. """
    _execvpe(file, args, env)

__all__.extend(["execl","execle","execlp","execlpe","execvp","execvpe"])

def _execvpe(file, args, env=None):
    if env is not None:
        exec_func = execve
        argrest = (args, env)
    else:
        exec_func = execv
        argrest = (args,)
        env = environ

    if path.dirname(file):
        exec_func(file, *argrest)
        return
    saved_exc = None
    path_list = get_exec_path(env)
    if name != 'nt':
        file = fsencode(file)
        path_list = map(fsencode, path_list)
    for dir in path_list:
        fullname = path.join(dir, file)
        try:
            exec_func(fullname, *argrest)
        except (FileNotFoundError, NotADirectoryError) as e:
            last_exc = e
        except OSError as e:
            last_exc = e
            if saved_exc is None:
                saved_exc = e
    if saved_exc is not None:
        raise saved_exc
    raise last_exc


def get_exec_path(env=None):
    """Returns the sequence of directories that will be searched for the
    named executable (similar to a shell) when launching a process.

    *env* must be an environment variable dict or None.  If *env* is None,
    os.environ will be used.
    """
    # Use a local import instead of a global import to limit the number of
    # modules loaded at startup: the os module is always loaded at startup by
    # Python. It may also avoid a bootstrap issue.
    import warnings

    if env is None:
        env = environ

    # {b'PATH': ...}.get('PATH') and {'PATH': ...}.get(b'PATH') emit a
    # BytesWarning when using python -b or python -bb: ignore the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BytesWarning)

        try:
            path_list = env.get('PATH')
        except TypeError:
            path_list = None

        if supports_bytes_environ:
            try:
                path_listb = env[b'PATH']
            except (KeyError, TypeError):
                pass
            else:
                if path_list is not None:
                    raise ValueError(
                        "env cannot contain 'PATH' and b'PATH' keys")
                path_list = path_listb

            if path_list is not None and isinstance(path_list, bytes):
                path_list = fsdecode(path_list)

    if path_list is None:
        path_list = defpath
    return path_list.split(pathsep)


# Change environ to automatically call putenv() and unsetenv()
from _collections_abc import MutableMapping, Mapping

class _Environ(MutableMapping):
    def __init__(self, data, encodekey, decodekey, encodevalue, decodevalue):
        self.encodekey = encodekey
        self.decodekey = decodekey
        self.encodevalue = encodevalue
        self.decodevalue = decodevalue
        self._data = data

    def __getitem__(self, key):
        try:
            value = self._data[self.encodekey(key)]
        except KeyError:
            # raise KeyError with the original key value
            raise KeyError(key) from None
        return self.decodevalue(value)

    def __setitem__(self, key, value):
        key = self.encodekey(key)
        value = self.encodevalue(value)
        putenv(key, value)
        self._data[key] = value

    def __delitem__(self, key):
        encodedkey = self.encodekey(key)
        unsetenv(encodedkey)
        try:
            del self._data[encodedkey]
        except KeyError:
            # raise KeyError with the original key value
            raise KeyError(key) from None

    def __iter__(self):
        # list() from dict object is an atomic operation
        keys = list(self._data)
        for key in keys:
            yield self.decodekey(key)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return 'environ({{{}}})'.format(', '.join(
            ('{!r}: {!r}'.format(self.decodekey(key), self.decodevalue(value))
            for key, value in self._data.items())))

    def copy(self):
        return dict(self)

    def setdefault(self, key, value):
        if key not in self:
            self[key] = value
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        new = dict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        new = dict(other)
        new.update(self)
        return new

def _createenviron():
    if name == 'nt':
        # Where Env Var Names Must Be UPPERCASE
        def check_str(value):
            if not isinstance(value, str):
                raise TypeError("str expected, not %s" % type(value).__name__)
            return value
        encode = check_str
        decode = str
        def encodekey(key):
            return encode(key).upper()
        data = {}
        for key, value in environ.items():
            data[encodekey(key)] = value
    else:
        # Where Env Var Names Can Be Mixed Case
        encoding = sys.getfilesystemencoding()
        def encode(value):
            if not isinstance(value, str):
                raise TypeError("str expected, not %s" % type(value).__name__)
            return value.encode(encoding, 'surrogateescape')
        def decode(value):
            return value.decode(encoding, 'surrogateescape')
        encodekey = encode
        data = environ
    return _Environ(data,
        encodekey, decode,
        encode, decode)

# unicode environ
environ = _createenviron()
del _createenviron


def getenv(key, default=None):
    """Get an environment variable, return None if it doesn't exist.
    The optional second argument can specify an alternate default.
    key, default and the result are str."""
    return environ.get(key, default)

supports_bytes_environ = (name != 'nt')
__all__.extend(("getenv", "supports_bytes_environ"))

if supports_bytes_environ:
    def _check_bytes(value):
        if not isinstance(value, bytes):
            raise TypeError("bytes expected, not %s" % type(value).__name__)
        return value

    # bytes environ
    environb = _Environ(environ._data,
        _check_bytes, bytes,
        _check_bytes, bytes)
    del _check_bytes

    def getenvb(key, default=None):
        """Get an environment variable, return None if it doesn't exist.
        The optional second argument can specify an alternate default.
        key, default and the result are bytes."""
        return environb.get(key, default)

    __all__.extend(("environb", "getenvb"))

def _fscodec():
    encoding = sys.getfilesystemencoding()
    errors = sys.getfilesystemencodeerrors()

    def fsencode(filename):
        """Encode filename (an os.PathLike, bytes, or str) to the filesystem
        encoding with 'surrogateescape' error handler, return bytes unchanged.
        On Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)  # Does type-checking of `filename`.
        if isinstance(filename, str):
            return filename.encode(encoding, errors)
        else:
            return filename

    def fsdecode(filename):
        """Decode filename (an os.PathLike, bytes, or str) from the filesystem
        encoding with 'surrogateescape' error handler, return str unchanged. On
        Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)  # Does type-checking of `filename`.
        if isinstance(filename, bytes):
            return filename.decode(encoding, errors)
        else:
            return filename

    return fsencode, fsdecode

fsencode, fsdecode = _fscodec()
del _fscodec

# Supply spawn*() (probably only for Unix)
if _exists("fork") and not _exists("spawnv") and _exists("execv"):

    P_WAIT = 0
    P_NOWAIT = P_NOWAITO = 1

    __all__.extend(["P_WAIT", "P_NOWAIT", "P_NOWAITO"])

    # XXX Should we support P_DETACH?  I suppose it could fork()**2
    # and close the std I/O streams.  Also, P_OVERLAY is the same
    # as execv*()?

    def _spawnvef(mode, file, args, env, func):
        # Internal helper; func is the exec*() function to use
        if not isinstance(args, (tuple, list)):
            raise TypeError('argv must be a tuple or a list')
        if not args or not args[0]:
            raise ValueError('argv first element cannot be empty')
        pid = fork()
        if not pid:
            # Child
            try:
                if env is None:
                    func(file, args)
                else:
                    func(file, args, env)
            except:
                _exit(127)
        else:
            # Parent
            if mode == P_NOWAIT:
                return pid # Caller is responsible for waiting!
            while 1:
                wpid, sts = waitpid(pid, 0)
                if WIFSTOPPED(sts):
                    continue

                return waitstatus_to_exitcode(sts)

    def spawnv(mode, file, args):
        """spawnv(mode, file, args) -> integer

Execute file with arguments from args in a subprocess.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, None, execv)

    def spawnve(mode, file, args, env):
        """spawnve(mode, file, args, env) -> integer

Execute file with arguments from args in a subprocess with the
specified environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, env, execve)

    # Note: spawnvp[e] isn't currently supported on Windows

    def spawnvp(mode, file, args):
        """spawnvp(mode, file, args) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, None, execvp)

    def spawnvpe(mode, file, args, env):
        """spawnvpe(mode, file, args, env) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess with the supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return _spawnvef(mode, file, args, env, execvpe)


    __all__.extend(["spawnv", "spawnve", "spawnvp", "spawnvpe"])


if _exists("spawnv"):
    # These aren't supplied by the basic Windows code
    # but can be easily implemented in Python

    def spawnl(mode, file, *args):
        """spawnl(mode, file, *args) -> integer

Execute file with arguments from args in a subprocess.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return spawnv(mode, file, args)

    def spawnle(mode, file, *args):
        """spawnle(mode, file, *args, env) -> integer

Execute file with arguments from args in a subprocess with the
supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        env = args[-1]
        return spawnve(mode, file, args[:-1], env)


    __all__.extend(["spawnl", "spawnle"])


if _exists("spawnvp"):
    # At the moment, Windows doesn't implement spawnvp[e],
    # so it won't have spawnlp[e] either.
    def spawnlp(mode, file, *args):
        """spawnlp(mode, file, *args) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess with the supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        return spawnvp(mode, file, args)

    def spawnlpe(mode, file, *args):
        """spawnlpe(mode, file, *args, env) -> integer

Execute file (which is looked for along $PATH) with arguments from
args in a subprocess with the supplied environment.
If mode == P_NOWAIT return the pid of the process.
If mode == P_WAIT return the process's exit code if it exits normally;
otherwise return -SIG, where SIG is the signal that killed it. """
        env = args[-1]
        return spawnvpe(mode, file, args[:-1], env)


    __all__.extend(["spawnlp", "spawnlpe"])

# VxWorks has no user space shell provided. As a result, running
# command in a shell can't be supported.
if sys.platform != 'vxworks':
    # Supply os.popen()
    def popen(cmd, mode="r", buffering=-1):
        if not isinstance(cmd, str):
            raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))
        if mode not in ("r", "w"):
            raise ValueError("invalid mode %r" % mode)
        if buffering == 0 or buffering is None:
            raise ValueError("popen() does not support unbuffered streams")
        import subprocess, io
        if mode == "r":
            proc = subprocess.Popen(cmd,
                                    shell=True, text=True,
                                    stdout=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(proc.stdout, proc)
        else:
            proc = subprocess.Popen(cmd,
                                    shell=True, text=True,
                                    stdin=subprocess.PIPE,
                                    bufsize=buffering)
            return _wrap_close(proc.stdin, proc)

    # Helper for popen() -- a proxy for a file whose close waits for the process
    class _wrap_close:
        def __init__(self, stream, proc):
            self._stream = stream
            self._proc = proc
        def close(self):
            self._stream.close()
            returncode = self._proc.wait()
            if returncode == 0:
                return None
            if name == 'nt':
                return returncode
            else:
                return returncode << 8  # Shift left to match old behavior
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def __getattr__(self, name):
            return getattr(self._stream, name)
        def __iter__(self):
            return iter(self._stream)

    __all__.append("popen")

# Supply os.fdopen()
def fdopen(fd, mode="r", buffering=-1, encoding=None, *args, **kwargs):
    if not isinstance(fd, int):
        raise TypeError("invalid fd type (%s, expected integer)" % type(fd))
    import io
    if "b" not in mode:
        encoding = io.text_encoding(encoding)
    return io.open(fd, mode, buffering, encoding, *args, **kwargs)


# For testing purposes, make sure the function is available when the C
# implementation exists.
def _fspath(path):
    """Return the path representation of a path-like object.

    If str or bytes is passed in, it is returned unchanged. Otherwise the
    os.PathLike interface is used to get the path representation. If the
    path representation is not str or bytes, TypeError is raised. If the
    provided path is not str, bytes, or os.PathLike, TypeError is raised.
    """
    if isinstance(path, (str, bytes)):
        return path

    # Work from the object's type to match method resolution of other magic
    # methods.
    path_type = type(path)
    try:
        path_repr = path_type.__fspath__(path)
    except AttributeError:
        if hasattr(path_type, '__fspath__'):
            raise
        else:
            raise TypeError("expected str, bytes or os.PathLike object, "
                            "not " + path_type.__name__)
    if isinstance(path_repr, (str, bytes)):
        return path_repr
    else:
        raise TypeError("expected {}.__fspath__() to return str or bytes, "
                        "not {}".format(path_type.__name__,
                                        type(path_repr).__name__))

# If there is no C implementation, make the pure Python version the
# implementation as transparently as possible.
if not _exists('fspath'):
    fspath = _fspath
    fspath.__name__ = "fspath"


class PathLike(abc.ABC):

    """Abstract base class for implementing the file system path protocol."""

    @abc.abstractmethod
    def __fspath__(self):
        """Return the file system path representation of the object."""
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        if cls is PathLike:
            return _check_methods(subclass, '__fspath__')
        return NotImplemented

    __class_getitem__ = classmethod(GenericAlias)


if name == 'nt':
    class _AddedDllDirectory:
        def __init__(self, path, cookie, remove_dll_directory):
            self.path = path
            self._cookie = cookie
            self._remove_dll_directory = remove_dll_directory
        def close(self):
            self._remove_dll_directory(self._cookie)
            self.path = None
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def __repr__(self):
            if self.path:
                return "<AddedDllDirectory({!r})>".format(self.path)
            return "<AddedDllDirectory()>"

    def add_dll_directory(path):
        """Add a path to the DLL search path.

        This search path is used when resolving dependencies for imported
        extension modules (the module itself is resolved through sys.path),
        and also by ctypes.

        Remove the directory by calling close() on the returned object or
        using it in a with statement.
        """
        import nt
        cookie = nt._add_dll_directory(path)
        return _AddedDllDirectory(
            path,
            cookie,
            nt._remove_dll_directory
        )
<|endoftext|>"""Word completion for GNU readline.

The completer completes keywords, built-ins and globals in a selectable
namespace (which defaults to __main__); when completing NAME.NAME..., it
evaluates (!) the expression up to the last dot and completes its attributes.

It's very cool to do "import sys" type "sys.", hit the completion key (twice),
and see the list of names defined by the sys module!

Tip: to use the tab key as the completion key, call

    readline.parse_and_bind("tab: complete")

Notes:

- Exceptions raised by the completer function are *ignored* (and generally cause
  the completion to fail).  This is a feature -- since readline sets the tty
  device in raw (or cbreak) mode, printing a traceback wouldn't work well
  without some complicated hoopla to save, reset and restore the tty state.

- The evaluation of the NAME.NAME... form may cause arbitrary application
  defined code to be executed if an object with a __getattr__ hook is found.
  Since it is the responsibility of the application (or the user) to enable this
  feature, I consider this an acceptable risk.  More complicated expressions
  (e.g. function calls or indexing operations) are *not* evaluated.

- When the original stdin is not a tty device, GNU readline is never
  used, and this module (and the readline module) are silently inactive.

"""

import atexit
import builtins
import inspect
import __main__

__all__ = ["Completer"]

class Completer:
    def __init__(self, namespace = None):
        """Create a new completer for the command line.

        Completer([namespace]) -> completer instance.

        If unspecified, the default namespace where completions are performed
        is __main__ (technically, __main__.__dict__). Namespaces should be
        given as dictionaries.

        Completer instances should be used as the completion mechanism of
        readline via the set_completer() call:

        readline.set_completer(Completer(my_namespace).complete)
        """

        if namespace and not isinstance(namespace, dict):
            raise TypeError('namespace must be a dictionary')

        # Don't bind to namespace quite yet, but flag whether the user wants a
        # specific namespace or to use __main__.__dict__. This will allow us
        # to bind to __main__.__dict__ at completion time, not now.
        if namespace is None:
            self.use_main_ns = 1
        else:
            self.use_main_ns = 0
            self.namespace = namespace

    def complete(self, text, state):
        """Return the next possible completion for 'text'.

        This is called successively with state == 0, 1, 2, ... until it
        returns None.  The completion should begin with 'text'.

        """
        if self.use_main_ns:
            self.namespace = __main__.__dict__

        if not text.strip():
            if state == 0:
                if _readline_available:
                    readline.insert_text('\t')
                    readline.redisplay()
                    return ''
                else:
                    return '\t'
            else:
                return None

        if state == 0:
            if "." in text:
                self.matches = self.attr_matches(text)
            else:
                self.matches = self.global_matches(text)
        try:
            return self.matches[state]
        except IndexError:
            return None

    def _callable_postfix(self, val, word):
        if callable(val):
            word += "("
            try:
                if not inspect.signature(val).parameters:
                    word += ")"
            except ValueError:
                pass

        return word

    def global_matches(self, text):
        """Compute matches when text is a simple name.

        Return a list of all keywords, built-in functions and names currently
        defined in self.namespace that match.

        """
        import keyword
        matches = []
        seen = {"__builtins__"}
        n = len(text)
        for word in keyword.kwlist:
            if word[:n] == text:
                seen.add(word)
                if word in {'finally', 'try'}:
                    word = word + ':'
                elif word not in {'False', 'None', 'True',
                                  'break', 'continue', 'pass',
                                  'else'}:
                    word = word + ' '
                matches.append(word)
        for nspace in [self.namespace, builtins.__dict__]:
            for word, val in nspace.items():
                if word[:n] == text and word not in seen:
                    seen.add(word)
                    matches.append(self._callable_postfix(val, word))
        return matches

    def attr_matches(self, text):
        """Compute matches when text contains a dot.

        Assuming the text is of the form NAME.NAME....[NAME], and is
        evaluable in self.namespace, it will be evaluated and its attributes
        (as revealed by dir()) are used as possible completions.  (For class
        instances, class members are also considered.)

        WARNING: this can still invoke arbitrary C code, if an object
        with a __getattr__ hook is evaluated.

        """
        import re
        m = re.match(r"(\w+(\.\w+)*)\.(\w*)", text)
        if not m:
            return []
        expr, attr = m.group(1, 3)
        try:
            thisobject = eval(expr, self.namespace)
        except Exception:
            return []

        # get the content of the object, except __builtins__
        words = set(dir(thisobject))
        words.discard("__builtins__")

        if hasattr(thisobject, '__class__'):
            words.add('__class__')
            words.update(get_class_members(thisobject.__class__))
        matches = []
        n = len(attr)
        if attr == '':
            noprefix = '_'
        elif attr == '_':
            noprefix = '__'
        else:
            noprefix = None
        while True:
            for word in words:
                if (word[:n] == attr and
                    not (noprefix and word[:n+1] == noprefix)):
                    match = "%s.%s" % (expr, word)
                    if isinstance(getattr(type(thisobject), word, None),
                                  property):
                        # bpo-44752: thisobject.word is a method decorated by
                        # `@property`. What follows applies a postfix if
                        # thisobject.word is callable, but know we know that
                        # this is not callable (because it is a property).
                        # Also, getattr(thisobject, word) will evaluate the
                        # property method, which is not desirable.
                        matches.append(match)
                        continue
                    if (value := getattr(thisobject, word, None)) is not None:
                        matches.append(self._callable_postfix(value, match))
                    else:
                        matches.append(match)
            if matches or not noprefix:
                break
            if noprefix == '_':
                noprefix = '__'
            else:
                noprefix = None
        matches.sort()
        return matches

def get_class_members(klass):
    ret = dir(klass)
    if hasattr(klass,'__bases__'):
        for base in klass.__bases__:
            ret = ret + get_class_members(base)
    return ret

try:
    import readline
except ImportError:
    _readline_available = False
else:
    readline.set_completer(Completer().complete)
    # Release references early at shutdown (the readline module's
    # contents are quasi-immortal, and the completer function holds a
    # reference to globals).
    atexit.register(lambda: readline.set_completer(None))
    _readline_available = True
<|endoftext|>"""functools.py - Tools for working with functions and callable objects
"""
# Python module wrapper for _functools C module
# to allow utilities written in Python to be added
# to the functools module.
# Written by Nick Coghlan <ncoghlan at gmail.com>,
# Raymond Hettinger <python at rcn.com>,
# and Łukasz Langa <lukasz at langa.pl>.
#   Copyright (C) 2006-2013 Python Software Foundation.
# See C source code for _functools credits/copyright

__all__ = ['update_wrapper', 'wraps', 'WRAPPER_ASSIGNMENTS', 'WRAPPER_UPDATES',
           'total_ordering', 'cache', 'cmp_to_key', 'lru_cache', 'reduce',
           'partial', 'partialmethod', 'singledispatch', 'singledispatchmethod',
           'cached_property']

from abc import get_cache_token
from collections import namedtuple
# import types, weakref  # Deferred to single_dispatch()
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias


################################################################################
### update_wrapper() and wraps() decorator
################################################################################

# update_wrapper() and wraps() are tools to help write
# wrapper functions that can handle naive introspection

WRAPPER_ASSIGNMENTS = ('__module__', '__name__', '__qualname__', '__doc__',
                       '__annotations__')
WRAPPER_UPDATES = ('__dict__',)
def update_wrapper(wrapper,
                   wrapped,
                   assigned = WRAPPER_ASSIGNMENTS,
                   updated = WRAPPER_UPDATES):
    """Update a wrapper function to look like the wrapped function

       wrapper is the function to be updated
       wrapped is the original function
       assigned is a tuple naming the attributes assigned directly
       from the wrapped function to the wrapper function (defaults to
       functools.WRAPPER_ASSIGNMENTS)
       updated is a tuple naming the attributes of the wrapper that
       are updated with the corresponding attribute from the wrapped
       function (defaults to functools.WRAPPER_UPDATES)
    """
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper

def wraps(wrapped,
          assigned = WRAPPER_ASSIGNMENTS,
          updated = WRAPPER_UPDATES):
    """Decorator factory to apply update_wrapper() to a wrapper function

       Returns a decorator that invokes update_wrapper() with the decorated
       function as the wrapper argument and the arguments to wraps() as the
       remaining arguments. Default arguments are as for update_wrapper().
       This is a convenience function to simplify applying partial() to
       update_wrapper().
    """
    return partial(update_wrapper, wrapped=wrapped,
                   assigned=assigned, updated=updated)


################################################################################
### total_ordering class decorator
################################################################################

# The total ordering functions all invoke the root magic method directly
# rather than using the corresponding operator.  This avoids possible
# infinite recursion that could occur when the operator dispatch logic
# detects a NotImplemented result and then calls a reflected method.

def _gt_from_lt(self, other, NotImplemented=NotImplemented):
    'Return a > b.  Computed by @total_ordering from (not a < b) and (a != b).'
    op_result = type(self).__lt__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result and self != other

def _le_from_lt(self, other, NotImplemented=NotImplemented):
    'Return a <= b.  Computed by @total_ordering from (a < b) or (a == b).'
    op_result = type(self).__lt__(self, other)
    if op_result is NotImplemented:
        return op_result
    return op_result or self == other

def _ge_from_lt(self, other, NotImplemented=NotImplemented):
    'Return a >= b.  Computed by @total_ordering from (not a < b).'
    op_result = type(self).__lt__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result

def _ge_from_le(self, other, NotImplemented=NotImplemented):
    'Return a >= b.  Computed by @total_ordering from (not a <= b) or (a == b).'
    op_result = type(self).__le__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result or self == other

def _lt_from_le(self, other, NotImplemented=NotImplemented):
    'Return a < b.  Computed by @total_ordering from (a <= b) and (a != b).'
    op_result = type(self).__le__(self, other)
    if op_result is NotImplemented:
        return op_result
    return op_result and self != other

def _gt_from_le(self, other, NotImplemented=NotImplemented):
    'Return a > b.  Computed by @total_ordering from (not a <= b).'
    op_result = type(self).__le__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result

def _lt_from_gt(self, other, NotImplemented=NotImplemented):
    'Return a < b.  Computed by @total_ordering from (not a > b) and (a != b).'
    op_result = type(self).__gt__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result and self != other

def _ge_from_gt(self, other, NotImplemented=NotImplemented):
    'Return a >= b.  Computed by @total_ordering from (a > b) or (a == b).'
    op_result = type(self).__gt__(self, other)
    if op_result is NotImplemented:
        return op_result
    return op_result or self == other

def _le_from_gt(self, other, NotImplemented=NotImplemented):
    'Return a <= b.  Computed by @total_ordering from (not a > b).'
    op_result = type(self).__gt__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result

def _le_from_ge(self, other, NotImplemented=NotImplemented):
    'Return a <= b.  Computed by @total_ordering from (not a >= b) or (a == b).'
    op_result = type(self).__ge__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result or self == other

def _gt_from_ge(self, other, NotImplemented=NotImplemented):
    'Return a > b.  Computed by @total_ordering from (a >= b) and (a != b).'
    op_result = type(self).__ge__(self, other)
    if op_result is NotImplemented:
        return op_result
    return op_result and self != other

def _lt_from_ge(self, other, NotImplemented=NotImplemented):
    'Return a < b.  Computed by @total_ordering from (not a >= b).'
    op_result = type(self).__ge__(self, other)
    if op_result is NotImplemented:
        return op_result
    return not op_result

_convert = {
    '__lt__': [('__gt__', _gt_from_lt),
               ('__le__', _le_from_lt),
               ('__ge__', _ge_from_lt)],
    '__le__': [('__ge__', _ge_from_le),
               ('__lt__', _lt_from_le),
               ('__gt__', _gt_from_le)],
    '__gt__': [('__lt__', _lt_from_gt),
               ('__ge__', _ge_from_gt),
               ('__le__', _le_from_gt)],
    '__ge__': [('__le__', _le_from_ge),
               ('__gt__', _gt_from_ge),
               ('__lt__', _lt_from_ge)]
}

def total_ordering(cls):
    """Class decorator that fills in missing ordering methods"""
    # Find user-defined comparisons (not those inherited from object).
    roots = {op for op in _convert if getattr(cls, op, None) is not getattr(object, op, None)}
    if not roots:
        raise ValueError('must define at least one ordering operation: < > <= >=')
    root = max(roots)       # prefer __lt__ to __le__ to __gt__ to __ge__
    for opname, opfunc in _convert[root]:
        if opname not in roots:
            opfunc.__name__ = opname
            setattr(cls, opname, opfunc)
    return cls


################################################################################
### cmp_to_key() function converter
################################################################################

def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K

try:
    from _functools import cmp_to_key
except ImportError:
    pass


################################################################################
### reduce() sequence to a single item
################################################################################

_initial_missing = object()

def reduce(function, sequence, initial=_initial_missing):
    """
    reduce(function, iterable[, initial]) -> value

    Apply a function of two arguments cumulatively to the items of a sequence
    or iterable, from left to right, so as to reduce the iterable to a single
    value.  For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the iterable in the calculation, and serves as a default when the
    iterable is empty.
    """

    it = iter(sequence)

    if initial is _initial_missing:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value") from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value

try:
    from _functools import reduce
except ImportError:
    pass


################################################################################
### partial() argument application
################################################################################

# Purely functional, no descriptor behaviour
class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
               self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
           (kwds is not None and not isinstance(kwds, dict)) or
           (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args) # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds

try:
    from _functools import partial
except ImportError:
    pass

# Descriptor version
class partialmethod(object):
    """Method descriptor with partial application of the given arguments
    and keywords.

    Supports wrapping existing descriptors and handles non-descriptor
    callables as instance methods.
    """

    def __init__(self, func, /, *args, **keywords):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError("{!r} is not callable or a descriptor"
                                 .format(func))

        # func could be a descriptor like classmethod which isn't callable,
        # so we can't inherit from partial (it verifies func is callable)
        if isinstance(func, partialmethod):
            # flattening is mandatory in order to place cls/self before all
            # other arguments
            # it's also more efficient since only one function will be called
            self.func = func.func
            self.args = func.args + args
            self.keywords = {**func.keywords, **keywords}
        else:
            self.func = func
            self.args = args
            self.keywords = keywords

    def __repr__(self):
        args = ", ".join(map(repr, self.args))
        keywords = ", ".join("{}={!r}".format(k, v)
                                 for k, v in self.keywords.items())
        format_string = "{module}.{cls}({func}, {args}, {keywords})"
        return format_string.format(module=self.__class__.__module__,
                                    cls=self.__class__.__qualname__,
                                    func=self.func,
                                    args=args,
                                    keywords=keywords)

    def _make_unbound_method(self):
        def _method(cls_or_self, /, *args, **keywords):
            keywords = {**self.keywords, **keywords}
            return self.func(cls_or_self, *self.args, *args, **keywords)
        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method._partialmethod = self
        return _method

    def __get__(self, obj, cls=None):
        get = getattr(self.func, "__get__", None)
        result = None
        if get is not None:
            new_func = get(obj, cls)
            if new_func is not self.func:
                # Assume __get__ returning something new indicates the
                # creation of an appropriate callable
                result = partial(new_func, *self.args, **self.keywords)
                try:
                    result.__self__ = new_func.__self__
                except AttributeError:
                    pass
        if result is None:
            # If the underlying descriptor didn't do anything, treat this
            # like an instance method
            result = self._make_unbound_method().__get__(obj, cls)
        return result

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, "__isabstractmethod__", False)

    __class_getitem__ = classmethod(GenericAlias)


# Helper functions

def _unwrap_partial(func):
    while isinstance(func, partial):
        func = func.func
    return func

################################################################################
### LRU Cache function decorator
################################################################################

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.

    """

    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue

def _make_key(args, kwds, typed,
             kwd_mark = (object(),),
             fasttypes = {int, str},
             tuple=tuple, type=type, len=len):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)

def lru_cache(maxsize=128, typed=False):
    """Least-recently-used cache decorator.

    If *maxsize* is set to None, the LRU features are disabled and the cache
    can grow without bound.

    If *typed* is True, arguments of different types will be cached separately.
    For example, f(3.0) and f(3) will be treated as distinct calls with
    distinct results.

    Arguments to the cached function must be hashable.

    View the cache statistics named tuple (hits, misses, maxsize, currsize)
    with f.cache_info().  Clear the cache and statistics with f.cache_clear().
    Access the underlying function with f.__wrapped__.

    See:  https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)

    """

    # Users should only access the lru_cache through its public API:
    #       cache_info, cache_clear, and f.__wrapped__
    # The internals of the lru_cache are encapsulated for thread safety and
    # to allow the implementation to change (including a possible C version).

    if isinstance(maxsize, int):
        # Negative maxsize is treated as 0
        if maxsize < 0:
            maxsize = 0
    elif callable(maxsize) and isinstance(typed, bool):
        # The user_function was passed in directly via the maxsize argument
        user_function, maxsize = maxsize, 128
        wrapper = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)
        wrapper.cache_parameters = lambda : {'maxsize': maxsize, 'typed': typed}
        return update_wrapper(wrapper, user_function)
    elif maxsize is not None:
        raise TypeError(
            'Expected first argument to be an integer, a callable, or None')

    def decorating_function(user_function):
        wrapper = _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo)
        wrapper.cache_parameters = lambda : {'maxsize': maxsize, 'typed': typed}
        return update_wrapper(wrapper, user_function)

    return decorating_function

def _lru_cache_wrapper(user_function, maxsize, typed, _CacheInfo):
    # Constants shared by all lru cache instances:
    sentinel = object()          # unique object used to signal cache misses
    make_key = _make_key         # build a key from the function arguments
    PREV, NEXT, KEY, RESULT = 0, 1, 2, 3   # names for the link fields

    cache = {}
    hits = misses = 0
    full = False
    cache_get = cache.get    # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = RLock()           # because linkedlist updates aren't threadsafe
    root = []                # root of the circular doubly linked list
    root[:] = [root, root, None, None]     # initialize by pointing to self

    if maxsize == 0:

        def wrapper(*args, **kwds):
            # No caching -- just a statistics update
            nonlocal misses
            misses += 1
            result = user_function(*args, **kwds)
            return result

    elif maxsize is None:

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args, kwds, typed)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = user_function(*args, **kwds)
            cache[key] = result
            return result

    else:

        def wrapper(*args, **kwds):
            # Size limited caching that tracks accesses by recency
            nonlocal root, hits, misses, full
            key = make_key(args, kwds, typed)
            with lock:
                link = cache_get(key)
                if link is not None:
                    # Move the link to the front of the circular queue
                    link_prev, link_next, _key, result = link
                    link_prev[NEXT] = link_next
                    link_next[PREV] = link_prev
                    last = root[PREV]
                    last[NEXT] = root[PREV] = link
                    link[PREV] = last
                    link[NEXT] = root
                    hits += 1
                    return result
                misses += 1
            result = user_function(*args, **kwds)
            with lock:
                if key in cache:
                    # Getting here means that this same key was added to the
                    # cache while the lock was released.  Since the link
                    # update is already done, we need only return the
                    # computed result and update the count of misses.
                    pass
                elif full:
                    # Use the old root to store the new key and result.
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    # Empty the oldest link and make it the new root.
                    # Keep a reference to the old key and old result to
                    # prevent their ref counts from going to zero during the
                    # update. That will prevent potentially arbitrary object
                    # clean-up code (i.e. __del__) from running while we're
                    # still adjusting the links.
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    oldresult = root[RESULT]
                    root[KEY] = root[RESULT] = None
                    # Now update the cache dictionary.
                    del cache[oldkey]
                    # Save the potentially reentrant cache[key] assignment
                    # for last, after the root and links have been put in
                    # a consistent state.
                    cache[key] = oldroot
                else:
                    # Put result in a new link at the front of the queue.
                    last = root[PREV]
                    link = [last, root, key, result]
                    last[NEXT] = root[PREV] = cache[key] = link
                    # Use the cache_len bound method instead of the len() function
                    # which could potentially be wrapped in an lru_cache itself.
                    full = (cache_len() >= maxsize)
            return result

    def cache_info():
        """Report cache statistics"""
        with lock:
            return _CacheInfo(hits, misses, maxsize, cache_len())

    def cache_clear():
        """Clear the cache and cache statistics"""
        nonlocal hits, misses, full
        with lock:
            cache.clear()
            root[:] = [root, root, None, None]
            hits = misses = 0
            full = False

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear
    return wrapper

try:
    from _functools import _lru_cache_wrapper
except ImportError:
    pass


################################################################################
### cache -- simplified access to the infinity cache
################################################################################

def cache(user_function, /):
    'Simple lightweight unbounded cache.  Sometimes called "memoize".'
    return lru_cache(maxsize=None)(user_function)


################################################################################
### singledispatch() - single-dispatch generic function decorator
################################################################################

def _c3_merge(sequences):
    """Merges MROs in *sequences* to a single MRO using the C3 algorithm.

    Adapted from https://www.python.org/download/releases/2.3/mro/.

    """
    result = []
    while True:
        sequences = [s for s in sequences if s]   # purge empty sequences
        if not sequences:
            return result
        for s1 in sequences:   # find merge candidates among seq heads
            candidate = s1[0]
            for s2 in sequences:
                if candidate in s2[1:]:
                    candidate = None
                    break      # reject the current head, it appears later
            else:
                break
        if candidate is None:
            raise RuntimeError("Inconsistent hierarchy")
        result.append(candidate)
        # remove the chosen candidate
        for seq in sequences:
            if seq[0] == candidate:
                del seq[0]

def _c3_mro(cls, abcs=None):
    """Computes the method resolution order using extended C3 linearization.

    If no *abcs* are given, the algorithm works exactly like the built-in C3
    linearization used for method resolution.

    If given, *abcs* is a list of abstract base classes that should be inserted
    into the resulting MRO. Unrelated ABCs are ignored and don't end up in the
    result. The algorithm inserts ABCs where their functionality is introduced,
    i.e. issubclass(cls, abc) returns True for the class itself but returns
    False for all its direct base classes. Implicit ABCs for a given class
    (either registered or inferred from the presence of a special method like
    __len__) are inserted directly after the last ABC explicitly listed in the
    MRO of said class. If two implicit ABCs end up next to each other in the
    resulting MRO, their ordering depends on the order of types in *abcs*.

    """
    for i, base in enumerate(reversed(cls.__bases__)):
        if hasattr(base, '__abstractmethods__'):
            boundary = len(cls.__bases__) - i
            break   # Bases up to the last explicit ABC are considered first.
    else:
        boundary = 0
    abcs = list(abcs) if abcs else []
    explicit_bases = list(cls.__bases__[:boundary])
    abstract_bases = []
    other_bases = list(cls.__bases__[boundary:])
    for base in abcs:
        if issubclass(cls, base) and not any(
                issubclass(b, base) for b in cls.__bases__
            ):
            # If *cls* is the class that introduces behaviour described by
            # an ABC *base*, insert said ABC to its MRO.
            abstract_bases.append(base)
    for base in abstract_bases:
        abcs.remove(base)
    explicit_c3_mros = [_c3_mro(base, abcs=abcs) for base in explicit_bases]
    abstract_c3_mros = [_c3_mro(base, abcs=abcs) for base in abstract_bases]
    other_c3_mros = [_c3_mro(base, abcs=abcs) for base in other_bases]
    return _c3_merge(
        [[cls]] +
        explicit_c3_mros + abstract_c3_mros + other_c3_mros +
        [explicit_bases] + [abstract_bases] + [other_bases]
    )

def _compose_mro(cls, types):
    """Calculates the method resolution order for a given class *cls*.

    Includes relevant abstract base classes (with their respective bases) from
    the *types* iterable. Uses a modified C3 linearization algorithm.

    """
    bases = set(cls.__mro__)
    # Remove entries which are already present in the __mro__ or unrelated.
    def is_related(typ):
        return (typ not in bases and hasattr(typ, '__mro__')
                                 and not isinstance(typ, GenericAlias)
                                 and issubclass(cls, typ))
    types = [n for n in types if is_related(n)]
    # Remove entries which are strict bases of other entries (they will end up
    # in the MRO anyway.
    def is_strict_base(typ):
        for other in types:
            if typ != other and typ in other.__mro__:
                return True
        return False
    types = [n for n in types if not is_strict_base(n)]
    # Subclasses of the ABCs in *types* which are also implemented by
    # *cls* can be used to stabilize ABC ordering.
    type_set = set(types)
    mro = []
    for typ in types:
        found = []
        for sub in typ.__subclasses__():
            if sub not in bases and issubclass(cls, sub):
                found.append([s for s in sub.__mro__ if s in type_set])
        if not found:
            mro.append(typ)
            continue
        # Favor subclasses with the biggest number of useful bases
        found.sort(key=len, reverse=True)
        for sub in found:
            for subcls in sub:
                if subcls not in mro:
                    mro.append(subcls)
    return _c3_mro(cls, abcs=mro)

def _find_impl(cls, registry):
    """Returns the best matching implementation from *registry* for type *cls*.

    Where there is no registered implementation for a specific type, its method
    resolution order is used to find a more generic implementation.

    Note: if *registry* does not contain an implementation for the base
    *object* type, this function may return None.

    """
    mro = _compose_mro(cls, registry.keys())
    match = None
    for t in mro:
        if match is not None:
            # If *match* is an implicit ABC but there is another unrelated,
            # equally matching implicit ABC, refuse the temptation to guess.
            if (t in registry and t not in cls.__mro__
                              and match not in cls.__mro__
                              and not issubclass(match, t)):
                raise RuntimeError("Ambiguous dispatch: {} or {}".format(
                    match, t))
            break
        if t in registry:
            match = t
    return registry.get(match)

def singledispatch(func):
    """Single-dispatch generic function decorator.

    Transforms a function into a generic function, which can have different
    behaviours depending upon the type of its first argument. The decorated
    function acts as the default implementation, and additional
    implementations can be registered using the register() attribute of the
    generic function.
    """
    # There are many programs that use functools without singledispatch, so we
    # trade-off making singledispatch marginally slower for the benefit of
    # making start-up of such applications slightly faster.
    import types, weakref

    registry = {}
    dispatch_cache = weakref.WeakKeyDictionary()
    cache_token = None

    def dispatch(cls):
        """generic_func.dispatch(cls) -> <function implementation>

        Runs the dispatch algorithm to return the best available implementation
        for the given *cls* registered on *generic_func*.

        """
        nonlocal cache_token
        if cache_token is not None:
            current_token = get_cache_token()
            if cache_token != current_token:
                dispatch_cache.clear()
                cache_token = current_token
        try:
            impl = dispatch_cache[cls]
        except KeyError:
            try:
                impl = registry[cls]
            except KeyError:
                impl = _find_impl(cls, registry)
            dispatch_cache[cls] = impl
        return impl

    def _is_valid_dispatch_type(cls):
        return isinstance(cls, type) and not isinstance(cls, GenericAlias)

    def register(cls, func=None):
        """generic_func.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_func*.

        """
        nonlocal cache_token
        if _is_valid_dispatch_type(cls):
            if func is None:
                return lambda f: register(cls, f)
        else:
            if func is not None:
                raise TypeError(
                    f"Invalid first argument to `register()`. "
                    f"{cls!r} is not a class."
                )
            ann = getattr(cls, '__annotations__', {})
            if not ann:
                raise TypeError(
                    f"Invalid first argument to `register()`: {cls!r}. "
                    f"Use either `@register(some_class)` or plain `@register` "
                    f"on an annotated function."
                )
            func = cls

            # only import typing if annotation parsing is necessary
            from typing import get_type_hints
            argname, cls = next(iter(get_type_hints(func).items()))
            if not _is_valid_dispatch_type(cls):
                raise TypeError(
                    f"Invalid annotation for {argname!r}. "
                    f"{cls!r} is not a class."
                )

        registry[cls] = func
        if cache_token is None and hasattr(cls, '__abstractmethods__'):
            cache_token = get_cache_token()
        dispatch_cache.clear()
        return func

    def wrapper(*args, **kw):
        if not args:
            raise TypeError(f'{funcname} requires at least '
                            '1 positional argument')

        return dispatch(args[0].__class__)(*args, **kw)

    funcname = getattr(func, '__name__', 'singledispatch function')
    registry[object] = func
    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = types.MappingProxyType(registry)
    wrapper._clear_cache = dispatch_cache.clear
    update_wrapper(wrapper, func)
    return wrapper


# Descriptor version
class singledispatchmethod:
    """Single-dispatch generic method descriptor.

    Supports wrapping existing descriptors and handles non-descriptor
    callables as instance methods.
    """

    def __init__(self, func):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError(f"{func!r} is not callable or a descriptor")

        self.dispatcher = singledispatch(func)
        self.func = func

    def register(self, cls, method=None):
        """generic_method.register(cls, func) -> func

        Registers a new implementation for the given *cls* on a *generic_method*.
        """
        return self.dispatcher.register(cls, func=method)

    def __get__(self, obj, cls=None):
        def _method(*args, **kwargs):
            method = self.dispatcher.dispatch(args[0].__class__)
            return method.__get__(obj, cls)(*args, **kwargs)

        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method.register = self.register
        update_wrapper(_method, self.func)
        return _method

    @property
    def __isabstractmethod__(self):
        return getattr(self.func, '__isabstractmethod__', False)


################################################################################
### cached_property() - computed once per instance, cached as attribute
################################################################################

_NOT_FOUND = object()


class cached_property:
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = RLock()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    try:
                        cache[self.attrname] = val
                    except TypeError:
                        msg = (
                            f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {self.attrname!r} property."
                        )
                        raise TypeError(msg) from None
        return val

    __class_getitem__ = classmethod(GenericAlias)
<|endoftext|>r"""TELNET client class.

Based on RFC 854: TELNET Protocol Specification, by J. Postel and
J. Reynolds

Example:

>>> from telnetlib import Telnet
>>> tn = Telnet('www.python.org', 79)   # connect to finger port
>>> tn.write(b'guido\r\n')
>>> print(tn.read_all())
Login       Name               TTY         Idle    When    Where
guido    Guido van Rossum      pts/2        <Dec  2 11:10> snag.cnri.reston..

>>>

Note that read_all() won't read until eof -- it just reads some data
-- but it guarantees to read at least one byte unless EOF is hit.

It is possible to pass a Telnet object to a selector in order to wait until
more data is available.  Note that in this case, read_eager() may return b''
even if there was data on the socket, because the protocol negotiation may have
eaten the data.  This is why EOFError is needed in some cases to distinguish
between "no data" and "connection closed" (since the socket also appears ready
for reading when it is closed).

To do:
- option negotiation
- timeout should be intrinsic to the connection object instead of an
  option on one of the read calls only

"""


# Imported modules
import sys
import socket
import selectors
from time import monotonic as _time

__all__ = ["Telnet"]

# Tunable parameters
DEBUGLEVEL = 0

# Telnet protocol defaults
TELNET_PORT = 23

# Telnet protocol characters (don't change)
IAC  = bytes([255]) # "Interpret As Command"
DONT = bytes([254])
DO   = bytes([253])
WONT = bytes([252])
WILL = bytes([251])
theNULL = bytes([0])

SE  = bytes([240])  # Subnegotiation End
NOP = bytes([241])  # No Operation
DM  = bytes([242])  # Data Mark
BRK = bytes([243])  # Break
IP  = bytes([244])  # Interrupt process
AO  = bytes([245])  # Abort output
AYT = bytes([246])  # Are You There
EC  = bytes([247])  # Erase Character
EL  = bytes([248])  # Erase Line
GA  = bytes([249])  # Go Ahead
SB =  bytes([250])  # Subnegotiation Begin


# Telnet protocol options code (don't change)
# These ones all come from arpa/telnet.h
BINARY = bytes([0]) # 8-bit data path
ECHO = bytes([1]) # echo
RCP = bytes([2]) # prepare to reconnect
SGA = bytes([3]) # suppress go ahead
NAMS = bytes([4]) # approximate message size
STATUS = bytes([5]) # give status
TM = bytes([6]) # timing mark
RCTE = bytes([7]) # remote controlled transmission and echo
NAOL = bytes([8]) # negotiate about output line width
NAOP = bytes([9]) # negotiate about output page size
NAOCRD = bytes([10]) # negotiate about CR disposition
NAOHTS = bytes([11]) # negotiate about horizontal tabstops
NAOHTD = bytes([12]) # negotiate about horizontal tab disposition
NAOFFD = bytes([13]) # negotiate about formfeed disposition
NAOVTS = bytes([14]) # negotiate about vertical tab stops
NAOVTD = bytes([15]) # negotiate about vertical tab disposition
NAOLFD = bytes([16]) # negotiate about output LF disposition
XASCII = bytes([17]) # extended ascii character set
LOGOUT = bytes([18]) # force logout
BM = bytes([19]) # byte macro
DET = bytes([20]) # data entry terminal
SUPDUP = bytes([21]) # supdup protocol
SUPDUPOUTPUT = bytes([22]) # supdup output
SNDLOC = bytes([23]) # send location
TTYPE = bytes([24]) # terminal type
EOR = bytes([25]) # end or record
TUID = bytes([26]) # TACACS user identification
OUTMRK = bytes([27]) # output marking
TTYLOC = bytes([28]) # terminal location number
VT3270REGIME = bytes([29]) # 3270 regime
X3PAD = bytes([30]) # X.3 PAD
NAWS = bytes([31]) # window size
TSPEED = bytes([32]) # terminal speed
LFLOW = bytes([33]) # remote flow control
LINEMODE = bytes([34]) # Linemode option
XDISPLOC = bytes([35]) # X Display Location
OLD_ENVIRON = bytes([36]) # Old - Environment variables
