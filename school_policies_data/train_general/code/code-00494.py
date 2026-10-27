            # read from the initial
            out = [self._fetch_block_cached(start_block_number)[start_pos:]]

            # intermediate blocks
            # Note: it'd be nice to combine these into one big request. However
            # that doesn't play nicely with our LRU cache.
            out.extend(
                map(
                    self._fetch_block_cached,
                    range(start_block_number + 1, end_block_number),
                )
            )

            # final block
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])

            return b"".join(out)


caches: dict[str | None, type[BaseCache]] = {
    # one custom case
    None: BaseCache,
}


def register_cache(cls: type[BaseCache], clobber: bool = False) -> None:
    """'Register' cache implementation.

    Parameters
    ----------
    clobber: bool, optional
        If set to True (default is False) - allow to overwrite existing
        entry.

    Raises
    ------
    ValueError
    """
    name = cls.name
    if not clobber and name in caches:
        raise ValueError(f"Cache with name {name!r} is already known: {caches[name]}")
    caches[name] = cls


for c in (
    BaseCache,
    MMapCache,
    BytesCache,
    ReadAheadCache,
    BlockCache,
    FirstChunkCache,
    AllBytes,
    KnownPartsOfAFile,
    BackgroundBlockCache,
):
    register_cache(c)
<|endoftext|>"""
fsspec user-defined exception classes
"""

import asyncio


class BlocksizeMismatchError(ValueError):
    """
    Raised when a cached file is opened with a different blocksize than it was
    written with
    """


class FSTimeoutError(asyncio.TimeoutError):
    """
    Raised when a fsspec function timed out occurs
    """
<|endoftext|># https://hadoop.apache.org/docs/r1.0.4/webhdfs.html

import logging
import os
import secrets
import shutil
import tempfile
import uuid
from contextlib import suppress
from datetime import datetime
from urllib.parse import quote

import requests

from ..spec import AbstractBufferedFile, AbstractFileSystem
from ..utils import infer_storage_options, tokenize

logger = logging.getLogger("webhdfs")


class WebHDFS(AbstractFileSystem):
    """
    Interface to HDFS over HTTP using the WebHDFS API. Supports also HttpFS gateways.

    Four auth mechanisms are supported:

    insecure: no auth is done, and the user is assumed to be whoever they
        say they are (parameter ``user``), or a predefined value such as
        "dr.who" if not given
    spnego: when kerberos authentication is enabled, auth is negotiated by
        requests_kerberos https://github.com/requests/requests-kerberos .
        This establishes a session based on existing kinit login and/or
        specified principal/password; parameters are passed with ``kerb_kwargs``
    token: uses an existing Hadoop delegation token from another secured
        service. Indeed, this client can also generate such tokens when
        not insecure. Note that tokens expire, but can be renewed (by a
        previously specified user) and may allow for proxying.
    basic-auth: used when both parameter ``user`` and parameter ``password``
        are provided.

    """

    tempdir = str(tempfile.gettempdir())
    protocol = "webhdfs", "webHDFS"

    def __init__(
        self,
        host,
        port=50070,
        kerberos=False,
        token=None,
        user=None,
        password=None,
        proxy_to=None,
        kerb_kwargs=None,
        data_proxy=None,
        use_https=False,
        session_cert=None,
        session_verify=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        host: str
            Name-node address
        port: int
            Port for webHDFS
        kerberos: bool
            Whether to authenticate with kerberos for this connection
        token: str or None
            If given, use this token on every call to authenticate. A user
            and user-proxy may be encoded in the token and should not be also
            given
        user: str or None
            If given, assert the user name to connect with
        password: str or None
            If given, assert the password to use for basic auth. If password
            is provided, user must be provided also
        proxy_to: str or None
            If given, the user has the authority to proxy, and this value is
            the user in who's name actions are taken
        kerb_kwargs: dict
            Any extra arguments for HTTPKerberosAuth, see
            `<https://github.com/requests/requests-kerberos/blob/master/requests_kerberos/kerberos_.py>`_
        data_proxy: dict, callable or None
            If given, map data-node addresses. This can be necessary if the
            HDFS cluster is behind a proxy, running on Docker or otherwise has
            a mismatch between the host-names given by the name-node and the
            address by which to refer to them from the client. If a dict,
            maps host names ``host->data_proxy[host]``; if a callable, full
            URLs are passed, and function must conform to
            ``url->data_proxy(url)``.
        use_https: bool
            Whether to connect to the Name-node using HTTPS instead of HTTP
        session_cert: str or Tuple[str, str] or None
            Path to a certificate file, or tuple of (cert, key) files to use
            for the requests.Session
        session_verify: str, bool or None
            Path to a certificate file to use for verifying the requests.Session.
        kwargs
        """
        if self._cached:
            return
        super().__init__(**kwargs)
        self.url = f"{'https' if use_https else 'http'}://{host}:{port}/webhdfs/v1"
        self.kerb = kerberos
        self.kerb_kwargs = kerb_kwargs or {}
        self.pars = {}
        self.proxy = data_proxy or {}
        if token is not None:
            if user is not None or proxy_to is not None:
                raise ValueError(
                    "If passing a delegation token, must not set "
                    "user or proxy_to, as these are encoded in the"
                    " token"
                )
            self.pars["delegation"] = token
        self.user = user
        self.password = password

        if password is not None:
            if user is None:
                raise ValueError(
                    "If passing a password, the user must also be"
                    "set in order to set up the basic-auth"
                )
        else:
            if user is not None:
                self.pars["user.name"] = user

        if proxy_to is not None:
            self.pars["doas"] = proxy_to
        if kerberos and user is not None:
            raise ValueError(
                "If using Kerberos auth, do not specify the "
                "user, this is handled by kinit."
            )

        self.session_cert = session_cert
        self.session_verify = session_verify

        self._connect()

        self._fsid = f"webhdfs_{tokenize(host, port)}"

    @property
    def fsid(self):
        return self._fsid

    def _connect(self):
        self.session = requests.Session()

        if self.session_cert:
            self.session.cert = self.session_cert

        self.session.verify = self.session_verify

        if self.kerb:
            from requests_kerberos import HTTPKerberosAuth

            self.session.auth = HTTPKerberosAuth(**self.kerb_kwargs)

        if self.user is not None and self.password is not None:
            from requests.auth import HTTPBasicAuth

            self.session.auth = HTTPBasicAuth(self.user, self.password)

    def _call(self, op, method="get", path=None, data=None, redirect=True, **kwargs):
        path = self._strip_protocol(path) if path is not None else ""
        url = self._apply_proxy(self.url + quote(path, safe="/="))
        args = kwargs.copy()
        args.update(self.pars)
        args["op"] = op.upper()
        logger.debug("sending %s with %s", url, method)
        out = self.session.request(
            method=method.upper(),
            url=url,
            params=args,
            data=data,
            allow_redirects=redirect,
        )
        if out.status_code in [400, 401, 403, 404, 500]:
            try:
                err = out.json()
                msg = err["RemoteException"]["message"]
                exp = err["RemoteException"]["exception"]
            except (ValueError, KeyError):
                pass
            else:
                if exp in ["IllegalArgumentException", "UnsupportedOperationException"]:
                    raise ValueError(msg)
                elif exp in ["SecurityException", "AccessControlException"]:
                    raise PermissionError(msg)
                elif exp in ["FileNotFoundException"]:
                    raise FileNotFoundError(msg)
                else:
                    raise RuntimeError(msg)
        out.raise_for_status()
        return out

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        replication=None,
        permissions=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        path: str
            File location
        mode: str
            'rb', 'wb', etc.
        block_size: int
            Client buffer size for read-ahead or write buffer
        autocommit: bool
            If False, writes to temporary file that only gets put in final
            location upon commit
        replication: int
            Number of copies of file on the cluster, write mode only
        permissions: str or int
            posix permissions, write mode only
        kwargs

        Returns
        -------
        WebHDFile instance
        """
        block_size = block_size or self.blocksize
        return WebHDFile(
            self,
            path,
            mode=mode,
            block_size=block_size,
            tempdir=self.tempdir,
            autocommit=autocommit,
            replication=replication,
            permissions=permissions,
        )

    @staticmethod
    def _process_info(info):
        info["type"] = info["type"].lower()
        info["size"] = info["length"]
        return info

    @classmethod
    def _strip_protocol(cls, path):
        return infer_storage_options(path)["path"]

    @staticmethod
    def _get_kwargs_from_urls(urlpath):
        out = infer_storage_options(urlpath)
        out.pop("path", None)
        out.pop("protocol", None)
        if "username" in out:
            out["user"] = out.pop("username")
        return out

    def info(self, path):
        out = self._call("GETFILESTATUS", path=path)
        info = out.json()["FileStatus"]
        info["name"] = path
        return self._process_info(info)

    def created(self, path):
        """Return the created timestamp of a file as a datetime.datetime"""
        # The API does not provide creation time, so we use modification time
        info = self.info(path)
        mtime = info.get("modificationTime", None)
        if mtime is not None:
            return datetime.fromtimestamp(mtime / 1000)
        raise RuntimeError("Could not retrieve creation time (modification time).")

    def modified(self, path):
        """Return the modified timestamp of a file as a datetime.datetime"""
        info = self.info(path)
        mtime = info.get("modificationTime", None)
        if mtime is not None:
            return datetime.fromtimestamp(mtime / 1000)
        raise RuntimeError("Could not retrieve modification time.")

    def ls(self, path, detail=False, **kwargs):
        out = self._call("LISTSTATUS", path=path)
        infos = out.json()["FileStatuses"]["FileStatus"]
        for info in infos:
            self._process_info(info)
            info["name"] = path.rstrip("/") + "/" + info["pathSuffix"]
        if detail:
            return sorted(infos, key=lambda i: i["name"])
        else:
            return sorted(info["name"] for info in infos)

    def content_summary(self, path):
        """Total numbers of files, directories and bytes under path"""
        out = self._call("GETCONTENTSUMMARY", path=path)
        return out.json()["ContentSummary"]

    def ukey(self, path):
        """Checksum info of file, giving method and result"""
        out = self._call("GETFILECHECKSUM", path=path, redirect=False)
        if "Location" in out.headers:
            location = self._apply_proxy(out.headers["Location"])
            out2 = self.session.get(location)
            out2.raise_for_status()
            return out2.json()["FileChecksum"]
        else:
            out.raise_for_status()
            return out.json()["FileChecksum"]

    def home_directory(self):
        """Get user's home directory"""
        out = self._call("GETHOMEDIRECTORY")
        return out.json()["Path"]

    def get_delegation_token(self, renewer=None):
        """Retrieve token which can give the same authority to other uses

        Parameters
        ----------
        renewer: str or None
            User who may use this token; if None, will be current user
        """
        if renewer:
            out = self._call("GETDELEGATIONTOKEN", renewer=renewer)
        else:
            out = self._call("GETDELEGATIONTOKEN")
        t = out.json()["Token"]
        if t is None:
            raise ValueError("No token available for this user/security context")
        return t["urlString"]

    def renew_delegation_token(self, token):
        """Make token live longer. Returns new expiry time"""
        out = self._call("RENEWDELEGATIONTOKEN", method="put", token=token)
        return out.json()["long"]

    def cancel_delegation_token(self, token):
        """Stop the token from being useful"""
        self._call("CANCELDELEGATIONTOKEN", method="put", token=token)

    def chmod(self, path, mod):
        """Set the permission at path

        Parameters
        ----------
        path: str
            location to set (file or directory)
        mod: str or int
            posix epresentation or permission, give as oct string, e.g, '777'
            or 0o777
        """
        self._call("SETPERMISSION", method="put", path=path, permission=mod)

    def chown(self, path, owner=None, group=None):
        """Change owning user and/or group"""
        kwargs = {}
        if owner is not None:
            kwargs["owner"] = owner
        if group is not None:
            kwargs["group"] = group
        self._call("SETOWNER", method="put", path=path, **kwargs)

    def set_replication(self, path, replication):
        """
        Set file replication factor

        Parameters
        ----------
        path: str
            File location (not for directories)
        replication: int
            Number of copies of file on the cluster. Should be smaller than
            number of data nodes; normally 3 on most systems.
        """
        self._call("SETREPLICATION", path=path, method="put", replication=replication)

    def mkdir(self, path, **kwargs):
        self._call("MKDIRS", method="put", path=path)

    def makedirs(self, path, exist_ok=False):
        if exist_ok is False and self.exists(path):
            raise FileExistsError(path)
        self.mkdir(path)

    def mv(self, path1, path2, **kwargs):
        self._call("RENAME", method="put", path=path1, destination=path2)

    def rm(self, path, recursive=False, **kwargs):
        self._call(
            "DELETE",
            method="delete",
            path=path,
            recursive="true" if recursive else "false",
        )

    def rm_file(self, path, **kwargs):
        self.rm(path)

    def cp_file(self, lpath, rpath, **kwargs):
        with self.open(lpath) as lstream:
            tmp_fname = "/".join([self._parent(rpath), f".tmp.{secrets.token_hex(16)}"])
            # Perform an atomic copy (stream to a temporary file and
            # move it to the actual destination).
            try:
                with self.open(tmp_fname, "wb") as rstream:
                    shutil.copyfileobj(lstream, rstream)
                self.mv(tmp_fname, rpath)
            except BaseException:
                with suppress(FileNotFoundError):
                    self.rm(tmp_fname)
                raise

    def _apply_proxy(self, location):
        if self.proxy and callable(self.proxy):
            location = self.proxy(location)
        elif self.proxy:
            # as a dict
            for k, v in self.proxy.items():
                location = location.replace(k, v, 1)
        return location


class WebHDFile(AbstractBufferedFile):
    """A file living in HDFS over webHDFS"""

    def __init__(self, fs, path, **kwargs):
        super().__init__(fs, path, **kwargs)
        kwargs = kwargs.copy()
        if kwargs.get("permissions", None) is None:
            kwargs.pop("permissions", None)
        if kwargs.get("replication", None) is None:
            kwargs.pop("replication", None)
        self.permissions = kwargs.pop("permissions", 511)
        tempdir = kwargs.pop("tempdir")
        if kwargs.pop("autocommit", False) is False:
            self.target = self.path
            self.path = os.path.join(tempdir, str(uuid.uuid4()))

    def _upload_chunk(self, final=False):
        """Write one part of a multi-block file upload

        Parameters
        ==========
        final: bool
            This is the last block, so should complete file, if
            self.autocommit is True.
        """
        out = self.fs.session.post(
            self.location,
            data=self.buffer.getvalue(),
            headers={"content-type": "application/octet-stream"},
        )
        out.raise_for_status()
        return True

    def _initiate_upload(self):
        """Create remote file/upload"""
        kwargs = self.kwargs.copy()
        if "a" in self.mode:
            op, method = "APPEND", "POST"
        else:
            op, method = "CREATE", "PUT"
            kwargs["overwrite"] = "true"
        out = self.fs._call(op, method, self.path, redirect=False, **kwargs)
        location = self.fs._apply_proxy(out.headers["Location"])
        if "w" in self.mode:
            # create empty file to append to
            out2 = self.fs.session.put(
                location, headers={"content-type": "application/octet-stream"}
            )
            out2.raise_for_status()
            # after creating empty file, change location to append to
            out2 = self.fs._call("APPEND", "POST", self.path, redirect=False, **kwargs)
            self.location = self.fs._apply_proxy(out2.headers["Location"])

    def _fetch_range(self, start, end):
        start = max(start, 0)
        end = min(self.size, end)
        if start >= end or start >= self.size:
            return b""
        out = self.fs._call(
            "OPEN", path=self.path, offset=start, length=end - start, redirect=False
        )
        out.raise_for_status()
        if "Location" in out.headers:
            location = out.headers["Location"]
            out2 = self.fs.session.get(self.fs._apply_proxy(location))
            return out2.content
        else:
            return out.content

    def commit(self):
        self.fs.mv(self.path, self.target)

    def discard(self):
        self.fs.rm(self.path)
<|endoftext|>import datetime
import io
import logging
import os
import os.path as osp
import shutil
import stat
import tempfile
from functools import lru_cache

from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.utils import isfilelike, stringify_path

logger = logging.getLogger("fsspec.local")


class LocalFileSystem(AbstractFileSystem):
    """Interface to files on local storage

    Parameters
    ----------
    auto_mkdir: bool
        Whether, when opening a file, the directory containing it should
        be created (if it doesn't already exist). This is assumed by pyarrow
        code.
    """

    root_marker = "/"
    protocol = "file", "local"
    local_file = True

    def __init__(self, auto_mkdir=False, **kwargs):
        super().__init__(**kwargs)
        self.auto_mkdir = auto_mkdir

    @property
    def fsid(self):
        return "local"

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if self.exists(path):
            raise FileExistsError(path)
        if create_parents:
            self.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path, **kwargs)

    def makedirs(self, path, exist_ok=False):
        path = self._strip_protocol(path)
        os.makedirs(path, exist_ok=exist_ok)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        os.rmdir(path)

    def ls(self, path, detail=False, **kwargs):
        path = self._strip_protocol(path)
        path_info = self.info(path)
        infos = []
        if path_info["type"] == "directory":
            with os.scandir(path) as it:
                for f in it:
                    try:
                        # Only get the info if requested since it is a bit expensive (the stat call inside)
                        # The strip_protocol is also used in info() and calls make_path_posix to always return posix paths
                        info = self.info(f) if detail else self._strip_protocol(f.path)
                        infos.append(info)
                    except FileNotFoundError:
                        pass
        else:
            infos = [path_info] if detail else [path_info["name"]]

        return infos

    def info(self, path, **kwargs):
        if isinstance(path, os.DirEntry):
            # scandir DirEntry
            out = path.stat(follow_symlinks=False)
            link = path.is_symlink()
            if path.is_dir(follow_symlinks=False):
                t = "directory"
            elif path.is_file(follow_symlinks=False):
                t = "file"
            else:
                t = "other"

            size = out.st_size
            if link:
                try:
                    out2 = path.stat(follow_symlinks=True)
                    size = out2.st_size
                except OSError:
                    size = 0
            path = self._strip_protocol(path.path)
        else:
            # str or path-like
            path = self._strip_protocol(path)
            out = os.stat(path, follow_symlinks=False)
            link = stat.S_ISLNK(out.st_mode)
            if link:
                out = os.stat(path, follow_symlinks=True)
            size = out.st_size
            if stat.S_ISDIR(out.st_mode):
                t = "directory"
            elif stat.S_ISREG(out.st_mode):
                t = "file"
            else:
                t = "other"

        # Check for the 'st_birthtime' attribute, which is not always present; fallback to st_ctime
        created_time = getattr(out, "st_birthtime", out.st_ctime)

        result = {
            "name": path,
            "size": size,
            "type": t,
            "created": created_time,
            "islink": link,
        }
        for field in ["mode", "uid", "gid", "mtime", "ino", "nlink"]:
            result[field] = getattr(out, f"st_{field}")
        if link:
            result["destination"] = os.readlink(path)
        return result

    def lexists(self, path, **kwargs):
        return osp.lexists(path)

    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        if self.auto_mkdir:
            self.makedirs(self._parent(path2), exist_ok=True)
        if self.isfile(path1):
            shutil.copyfile(path1, path2)
        elif self.isdir(path1):
            self.mkdirs(path2, exist_ok=True)
        else:
            raise FileNotFoundError(path1)

    def isfile(self, path):
        path = self._strip_protocol(path)
        return os.path.isfile(path)

    def isdir(self, path):
        path = self._strip_protocol(path)
        return os.path.isdir(path)

    def get_file(self, path1, path2, callback=None, **kwargs):
        if isfilelike(path2):
            with open(path1, "rb") as f:
                shutil.copyfileobj(f, path2)
        else:
            return self.cp_file(path1, path2, **kwargs)

    def put_file(self, path1, path2, callback=None, **kwargs):
        return self.cp_file(path1, path2, **kwargs)

    def mv(self, path1, path2, recursive: bool = True, **kwargs):
        """Move files/directories
        For the specific case of local, all ops on directories are recursive and
        the recursive= kwarg is ignored.
        """
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)

        if self.auto_mkdir:
            self.makedirs(self._parent(path2), exist_ok=True)

        shutil.move(path1, path2)

    def link(self, src, dst, **kwargs):
        src = self._strip_protocol(src)
        dst = self._strip_protocol(dst)
        os.link(src, dst, **kwargs)

    def symlink(self, src, dst, **kwargs):
        src = self._strip_protocol(src)
        dst = self._strip_protocol(dst)
        os.symlink(src, dst, **kwargs)

    def islink(self, path) -> bool:
        return os.path.islink(self._strip_protocol(path))

    def rm_file(self, path):
        os.remove(self._strip_protocol(path))

    def rm(self, path, recursive=False, maxdepth=None):
        if not isinstance(path, list):
            path = [path]

        for p in path:
            p = self._strip_protocol(p)
            if self.isdir(p) and not self.islink(p):
                if not recursive:
                    raise ValueError("Cannot delete directory, set recursive=True")
                if osp.abspath(p) == os.getcwd():
                    raise ValueError("Cannot delete current working directory")
                shutil.rmtree(p)
            else:
                os.remove(p)

    def unstrip_protocol(self, name):
        protocol = self.protocol if isinstance(self.protocol, str) else self.protocol[0]
        name = self._strip_protocol(name)  # normalise for local/win/...
        return f"{protocol}://{name}"

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        path = self._strip_protocol(path)
        if self.auto_mkdir and ("w" in mode or "x" in mode or "a" in mode):
            self.makedirs(self._parent(path), exist_ok=True)
        return LocalFileOpener(path, mode, fs=self, **kwargs)

    def touch(self, path, truncate=True, **kwargs):
        path = self._strip_protocol(path)
        if self.auto_mkdir:
            self.makedirs(self._parent(path), exist_ok=True)
        if self.exists(path):
            os.utime(path, None)
        else:
            open(path, "a").close()
        if truncate:
            os.truncate(path, 0)

    def created(self, path):
        info = self.info(path=path)
        return datetime.datetime.fromtimestamp(
            info["created"], tz=datetime.timezone.utc
        )

    def modified(self, path):
        info = self.info(path=path)
        return datetime.datetime.fromtimestamp(info["mtime"], tz=datetime.timezone.utc)

    @classmethod
    def _parent(cls, path):
        path = cls._strip_protocol(path)
        if os.sep == "/":
            # posix native
            return path.rsplit("/", 1)[0] or "/"
        else:
            # NT
            path_ = path.rsplit("/", 1)[0]
            if len(path_) <= 3:
                if path_[1:2] == ":":
                    # nt root (something like c:/)
                    return path_[0] + ":/"
            # More cases may be required here
            return path_

    @classmethod
    def _strip_protocol(cls, path):
        path = stringify_path(path)
        protos = (cls.protocol,) if isinstance(cls.protocol, str) else cls.protocol
        prefixes = (protocol + sep for protocol in protos for sep in ("://", ":"))
        for prefix in prefixes:
            if path.startswith(prefix):
                path = path.removeprefix(prefix)
                break

        path = make_path_posix(path)
        if os.sep != "/":
            # This code-path is a stripped down version of
            # > drive, path = ntpath.splitdrive(path)
            if path[1:2] == ":":
                # Absolute drive-letter path, e.g. X:\Windows
                # Relative path with drive, e.g. X:Windows
                drive, path = path[:2], path[2:]
            elif path[:2] == "//":
                # UNC drives, e.g. \\server\share or \\?\UNC\server\share
                # Device drives, e.g. \\.\device or \\?\device
                if (index1 := path.find("/", 2)) == -1 or (
                    index2 := path.find("/", index1 + 1)
                ) == -1:
                    drive, path = path, ""
                else:
                    drive, path = path[:index2], path[index2:]
            else:
                # Relative path, e.g. Windows
                drive = ""

            path = path.rstrip("/") or cls.root_marker
            return drive + path

        else:
            return path.rstrip("/") or cls.root_marker

    def _isfilestore(self):
        # Inheriting from DaskFileSystem makes this False (S3, etc. were)
        # the original motivation. But we are a posix-like file system.
        # See https://github.com/dask/dask/issues/5526
        return True

    def chmod(self, path, mode):
        path = stringify_path(path)
        return os.chmod(path, mode)


def make_path_posix(path):
    """Make path generic and absolute for current OS"""
    if not isinstance(path, str):
        if isinstance(path, (list, set, tuple)):
            return type(path)(make_path_posix(p) for p in path)
        else:
            path = stringify_path(path)
            if not isinstance(path, str):
                raise TypeError(f"could not convert {path!r} to string")
    if os.sep == "/":
        # Native posix
        if path.startswith("/"):
            # most common fast case for posix
            return path
        elif path.startswith("~"):
            return osp.expanduser(path)
        elif path.startswith("./"):
            path = path[2:]
        elif path == ".":
            path = ""
        return f"{os.getcwd()}/{path}"
    else:
        # NT handling
        if path[0:1] == "/" and path[2:3] == ":":
            # path is like "/c:/local/path"
            path = path[1:]
        if path[1:2] == ":":
            # windows full path like "C:\\local\\path"
            if len(path) <= 3:
                # nt root (something like c:/)
                return path[0] + ":/"
            path = path.replace("\\", "/")
            return path
        elif path[0:1] == "~":
            return make_path_posix(osp.expanduser(path))
        elif path.startswith(("\\\\", "//")):
            # windows UNC/DFS-style paths
            return "//" + path[2:].replace("\\", "/")
        elif path.startswith(("\\", "/")):
            # windows relative path with root
            path = path.replace("\\", "/")
            return f"{osp.splitdrive(os.getcwd())[0]}{path}"
        else:
            path = path.replace("\\", "/")
            if path.startswith("./"):
                path = path[2:]
            elif path == ".":
                path = ""
            return f"{make_path_posix(os.getcwd())}/{path}"


def trailing_sep(path):
    """Return True if the path ends with a path separator.

    A forward slash is always considered a path separator, even on Operating
    Systems that normally use a backslash.
    """
    # TODO: if all incoming paths were posix-compliant then separator would
    # always be a forward slash, simplifying this function.
    # See https://github.com/fsspec/filesystem_spec/pull/1250
    return path.endswith(os.sep) or (os.altsep is not None and path.endswith(os.altsep))


@lru_cache(maxsize=1)
def get_umask(mask: int = 0o666) -> int:
    """Get the current umask.

    Follows https://stackoverflow.com/a/44130549 to get the umask.
    Temporarily sets the umask to the given value, and then resets it to the
    original value.
    """
    value = os.umask(mask)
    os.umask(value)
    return value


class LocalFileOpener(io.IOBase):
    def __init__(
        self, path, mode, autocommit=True, fs=None, compression=None, **kwargs
    ):
        logger.debug("open file: %s", path)
        self.path = path
        self.mode = mode
        self.fs = fs
        self.f = None
        self.autocommit = autocommit
        self.compression = get_compression(path, compression)
        self.blocksize = io.DEFAULT_BUFFER_SIZE
        self._open()

    def _open(self):
        if self.f is None or self.f.closed:
            if self.autocommit or "w" not in self.mode:
                self.f = open(self.path, mode=self.mode)
                if self.compression:
                    compress = compr[self.compression]
                    self.f = compress(self.f, mode=self.mode)
            else:
                # TODO: check if path is writable?
                i, name = tempfile.mkstemp()
                os.close(i)  # we want normal open and normal buffered file
                self.temp = name
                self.f = open(name, mode=self.mode)
            if "w" not in self.mode:
                self.size = self.f.seek(0, 2)
                self.f.seek(0)
                self.f.size = self.size

    def _fetch_range(self, start, end):
        # probably only used by cached FS
        if "r" not in self.mode:
            raise ValueError
        self._open()
        self.f.seek(start)
        return self.f.read(end - start)

    def __setstate__(self, state):
        self.f = None
        loc = state.pop("loc", None)
        self.__dict__.update(state)
        if "r" in state["mode"]:
            self.f = None
            self._open()
            self.f.seek(loc)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("f")
        if "r" in self.mode:
            d["loc"] = self.f.tell()
        else:
            if not self.f.closed:
                raise ValueError("Cannot serialise open write-mode local file")
        return d

    def commit(self):
        if self.autocommit:
            raise RuntimeError("Can only commit if not already set to autocommit")
        try:
            shutil.move(self.temp, self.path)
        except PermissionError as e:
            # shutil.move raises PermissionError if os.rename
            # and the default copy2 fallback with shutil.copystats fail.
            # The file should be there nonetheless, but without copied permissions.
            # If it doesn't exist, there was no permission to create the file.
            if not os.path.exists(self.path):
                raise e
        else:
            # If PermissionError is not raised, permissions can be set.
            try:
                mask = 0o666
                os.chmod(self.path, mask & ~get_umask(mask))
            except RuntimeError:
                pass

    def discard(self):
        if self.autocommit:
            raise RuntimeError("Cannot discard if set to autocommit")
        os.remove(self.temp)

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return "r" not in self.mode

    def read(self, *args, **kwargs):
        return self.f.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self.f.write(*args, **kwargs)

    def tell(self, *args, **kwargs):
        return self.f.tell(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self.f.seek(*args, **kwargs)

    def seekable(self, *args, **kwargs):
        return self.f.seekable(*args, **kwargs)

    def readline(self, *args, **kwargs):
        return self.f.readline(*args, **kwargs)

    def readlines(self, *args, **kwargs):
        return self.f.readlines(*args, **kwargs)

    def close(self):
        return self.f.close()

    def truncate(self, size=None) -> int:
        return self.f.truncate(size)

    @property
    def closed(self):
        return self.f.closed

    def fileno(self):
        return self.raw.fileno()

    def flush(self) -> None:
        self.f.flush()

    def __iter__(self):
        return self.f.__iter__()

    def __getattr__(self, item):
        return getattr(self.f, item)

    def __enter__(self):
        self._incontext = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._incontext = False
        self.f.__exit__(exc_type, exc_value, traceback)
<|endoftext|>from contextlib import contextmanager
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_int,
    c_longlong,
    c_void_p,
    cast,
    create_string_buffer,
)

import libarchive
import libarchive.ffi as ffi

from fsspec import open_files
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.utils import DEFAULT_BLOCK_SIZE

# Libarchive requires seekable files or memory only for certain archive
# types. However, since we read the directory first to cache the contents
# and also allow random access to any file, the file-like object needs
# to be seekable no matter what.

# Seek call-backs (not provided in the libarchive python wrapper)
SEEK_CALLBACK = CFUNCTYPE(c_longlong, c_int, c_void_p, c_longlong, c_int)
read_set_seek_callback = ffi.ffi(
    "read_set_seek_callback", [ffi.c_archive_p, SEEK_CALLBACK], c_int, ffi.check_int
)
new_api = hasattr(ffi, "NO_OPEN_CB")


@contextmanager
def custom_reader(file, format_name="all", filter_name="all", block_size=ffi.page_size):
    """Read an archive from a seekable file-like object.

    The `file` object must support the standard `readinto` and 'seek' methods.
    """
    buf = create_string_buffer(block_size)
    buf_p = cast(buf, c_void_p)

    def read_func(archive_p, context, ptrptr):
        # readinto the buffer, returns number of bytes read
        length = file.readinto(buf)
        # write the address of the buffer into the pointer
        ptrptr = cast(ptrptr, POINTER(c_void_p))
        ptrptr[0] = buf_p
        # tell libarchive how much data was written into the buffer
        return length

    def seek_func(archive_p, context, offset, whence):
        file.seek(offset, whence)
        # tell libarchvie the current position
        return file.tell()

    read_cb = ffi.READ_CALLBACK(read_func)
    seek_cb = SEEK_CALLBACK(seek_func)

    if new_api:
        open_cb = ffi.NO_OPEN_CB
        close_cb = ffi.NO_CLOSE_CB
    else:
        open_cb = libarchive.read.OPEN_CALLBACK(ffi.VOID_CB)
        close_cb = libarchive.read.CLOSE_CALLBACK(ffi.VOID_CB)

    with libarchive.read.new_archive_read(format_name, filter_name) as archive_p:
        read_set_seek_callback(archive_p, seek_cb)
        ffi.read_open(archive_p, None, open_cb, read_cb, close_cb)
        yield libarchive.read.ArchiveRead(archive_p)


class LibArchiveFileSystem(AbstractArchiveFileSystem):
    """Compressed archives as a file-system (read-only)

    Supports the following formats:
    tar, pax , cpio, ISO9660, zip, mtree, shar, ar, raw, xar, lha/lzh, rar
    Microsoft CAB, 7-Zip, WARC

    See the libarchive documentation for further restrictions.
    https://www.libarchive.org/

    Keeps file object open while instance lives. It only works in seekable
    file-like objects. In case the filesystem does not support this kind of
    file object, it is recommended to cache locally.

    This class is pickleable, but not necessarily thread-safe (depends on the
    platform). See libarchive documentation for details.
    """

    root_marker = ""
    protocol = "libarchive"
    cachable = False

    def __init__(
        self,
        fo="",
        mode="r",
        target_protocol=None,
        target_options=None,
        block_size=DEFAULT_BLOCK_SIZE,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fo: str or file-like
            Contains ZIP, and must exist. If a str, will fetch file using
            :meth:`~fsspec.open_files`, which must return one file exactly.
        mode: str
            Currently, only 'r' accepted
        target_protocol: str (optional)
            If ``fo`` is a string, this value can be used to override the
            FS protocol inferred from a URL
        target_options: dict (optional)
            Kwargs passed when instantiating the target FS, if ``fo`` is
            a string.
        """
        super().__init__(self, **kwargs)
        if mode != "r":
            raise ValueError("Only read from archive files accepted")
        if isinstance(fo, str):
            files = open_files(fo, protocol=target_protocol, **(target_options or {}))
            if len(files) != 1:
                raise ValueError(
                    f'Path "{fo}" did not resolve to exactly one file: "{files}"'
                )
            fo = files[0]
        self.of = fo
        self.fo = fo.__enter__()  # the whole instance is a context
        self.block_size = block_size
        self.dir_cache = None

    @contextmanager
    def _open_archive(self):
        self.fo.seek(0)
        with custom_reader(self.fo, block_size=self.block_size) as arc:
            yield arc

    @classmethod
    def _strip_protocol(cls, path):
        # file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def _get_dirs(self):
        fields = {
            "name": "pathname",
            "size": "size",
            "created": "ctime",
            "mode": "mode",
            "uid": "uid",
            "gid": "gid",
            "mtime": "mtime",
        }

        if self.dir_cache is not None:
            return

        self.dir_cache = {}
        list_names = []
        with self._open_archive() as arc:
            for entry in arc:
                if not entry.isdir and not entry.isfile:
                    # Skip symbolic links, fifo entries, etc.
                    continue
                self.dir_cache.update(
                    {
                        dirname: {"name": dirname, "size": 0, "type": "directory"}
                        for dirname in self._all_dirnames(set(entry.name))
                    }
                )
                f = {key: getattr(entry, fields[key]) for key in fields}
                f["type"] = "directory" if entry.isdir else "file"
                list_names.append(entry.name)

                self.dir_cache[f["name"]] = f
        # libarchive does not seem to return an entry for the directories (at least
        # not in all formats), so get the directories names from the files names
        self.dir_cache.update(
            {
                dirname: {"name": dirname, "size": 0, "type": "directory"}
                for dirname in self._all_dirnames(list_names)
            }
        )

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if mode != "rb":
            raise NotImplementedError

        data = b""
        with self._open_archive() as arc:
            for entry in arc:
                if entry.pathname != path:
                    continue

                if entry.size == 0:
                    # empty file, so there are no blocks
                    break

                for block in entry.get_blocks(entry.size):
                    data = block
                    break
                else:
                    raise ValueError
        return MemoryFile(fs=self, path=path, data=data)
<|endoftext|>import base64
import io
import re

import requests

import fsspec


class JupyterFileSystem(fsspec.AbstractFileSystem):
    """View of the files as seen by a Jupyter server (notebook or lab)"""

    protocol = ("jupyter", "jlab")

    def __init__(self, url, tok=None, **kwargs):
        """

        Parameters
        ----------
        url : str
            Base URL of the server, like "http://127.0.0.1:8888". May include
            token in the string, which is given by the process when starting up
        tok : str
            If the token is obtained separately, can be given here
        kwargs
        """
        if "?" in url:
            if tok is None:
                try:
                    tok = re.findall("token=([a-z0-9]+)", url)[0]
                except IndexError as e:
                    raise ValueError("Could not determine token") from e
            url = url.split("?", 1)[0]
        self.url = url.rstrip("/") + "/api/contents"
        self.session = requests.Session()
        if tok:
            self.session.headers["Authorization"] = f"token {tok}"

        super().__init__(**kwargs)

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        r = self.session.get(f"{self.url}/{path}")
        if r.status_code == 404:
            raise FileNotFoundError(path)
        r.raise_for_status()
        out = r.json()

        if out["type"] == "directory":
            out = out["content"]
        else:
            out = [out]
        for o in out:
            o["name"] = o.pop("path")
            o.pop("content")
            if o["type"] == "notebook":
                o["type"] = "file"
        if detail:
            return out
        return [o["name"] for o in out]

    def cat_file(self, path, start=None, end=None, **kwargs):
        path = self._strip_protocol(path)
        r = self.session.get(f"{self.url}/{path}")
        if r.status_code == 404:
            raise FileNotFoundError(path)
        r.raise_for_status()
        out = r.json()
        if out["format"] == "text":
            # data should be binary
            b = out["content"].encode()
        else:
            b = base64.b64decode(out["content"])
        return b[start:end]

    def pipe_file(self, path, value, **_):
        path = self._strip_protocol(path)
        json = {
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "size": len(value),
            "content": base64.b64encode(value).decode(),
            "format": "base64",
            "type": "file",
        }
        self.session.put(f"{self.url}/{path}", json=json)

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if create_parents and "/" in path:
            self.mkdir(path.rsplit("/", 1)[0], True)
        json = {
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "size": None,
            "content": None,
            "type": "directory",
        }
        self.session.put(f"{self.url}/{path}", json=json)

    def mv(self, path1, path2, recursive=False, maxdepth=None, **kwargs):
        if path1 == path2:
            return
        self.session.patch(f"{self.url}/{path1}", json={"path": path2})

    def _rm(self, path):
        path = self._strip_protocol(path)
        self.session.delete(f"{self.url}/{path}")

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)
        if mode == "rb":
            data = self.cat_file(path)
            return io.BytesIO(data)
        else:
            return SimpleFileWriter(self, path, mode="wb")


class SimpleFileWriter(fsspec.spec.AbstractBufferedFile):
    def _upload_chunk(self, final=False):
        """Never uploads a chunk until file is done

        Not suitable for large files
        """
        if final is False:
            return False
        self.buffer.seek(0)
        data = self.buffer.read()
        self.fs.pipe_file(self.path, data)
<|endoftext|>from .. import filesystem
from ..asyn import AsyncFileSystem
from .chained import ChainedFileSystem
from .local import LocalFileSystem


def _escapes_root(path):
    """Whether a relative path would resolve above its root via ".." segments."""
    depth = 0
    for part in path.split("/"):
        if part == "..":
            depth -= 1
            if depth < 0:
                return True
        elif part and part != ".":
            depth += 1
    return False


class DirFileSystem(AsyncFileSystem, ChainedFileSystem):
    """Directory prefix filesystem

    The DirFileSystem is a filesystem-wrapper. It assumes every path it is dealing with
    is relative to the `path`. After performing the necessary paths operation it
    delegates everything to the wrapped filesystem.
    """

    protocol = "dir"

    def __init__(
        self,
        path=None,
        fs=None,
        fo=None,
        target_protocol=None,
        target_options=None,
        **storage_options,
    ):
        """
        Parameters
        ----------
        path: str
            Path to the directory.
        fs: AbstractFileSystem
            An instantiated filesystem to wrap.
        target_protocol, target_options:
            if fs is none, construct it from these
        fo: str
            Alternate for path; do not provide both
        """
        super().__init__(**storage_options)
        if fs is None:
            fs = filesystem(protocol=target_protocol, **(target_options or {}))
        path = path or fo

        if self.asynchronous and not fs.async_impl:
            raise ValueError("can't use asynchronous with non-async fs")

        if fs.async_impl and self.asynchronous != fs.asynchronous:
            raise ValueError("both dirfs and fs should be in the same sync/async mode")

        self.path = fs._strip_protocol(path)
        self.fs = fs

    def _join(self, path):
        if isinstance(path, str):
            if not self.path:
                return path
            if not path:
                return self.path
            path = self._strip_protocol(path)
            # ".." only navigates above the root on filesystems that resolve it
            # against a real directory tree; on object stores it is a literal
            # path part, so only guard the local case here.
            if isinstance(self.fs, LocalFileSystem) and _escapes_root(path):
                raise ValueError(
                    f"path {path!r} escapes the {self.path!r} root of the filesystem"
                )
            return self.fs.sep.join((self.path, path))
        if isinstance(path, dict):
            return {self._join(_path): value for _path, value in path.items()}
        return [self._join(_path) for _path in path]

    def _relpath(self, path):
        if isinstance(path, str):
            if not self.path:
                return path
            # We need to account for S3FileSystem returning paths that do not
            # start with a '/'
            if path == self.path or (
                self.path.startswith(self.fs.sep) and path == self.path[1:]
            ):
                return ""
            prefix = self.path + self.fs.sep
            if self.path.startswith(self.fs.sep) and not path.startswith(self.fs.sep):
                prefix = prefix[1:]
            assert path.startswith(prefix)
            return path[len(prefix) :]
        return [self._relpath(_path) for _path in path]

    # Wrappers below

    @property
    def sep(self):
        return self.fs.sep

    async def set_session(self, *args, **kwargs):
        return await self.fs.set_session(*args, **kwargs)

    async def _rm_file(self, path, **kwargs):
        return await self.fs._rm_file(self._join(path), **kwargs)

    def rm_file(self, path, **kwargs):
        return self.fs.rm_file(self._join(path), **kwargs)

    async def _rm(self, path, *args, **kwargs):
        return await self.fs._rm(self._join(path), *args, **kwargs)

    def rm(self, path, *args, **kwargs):
        return self.fs.rm(self._join(path), *args, **kwargs)

    def delete(self, path, recursive=False, maxdepth=None):
        return self.fs.delete(self._join(path), recursive=recursive, maxdepth=maxdepth)

    async def _cp_file(self, path1, path2, **kwargs):
        return await self.fs._cp_file(self._join(path1), self._join(path2), **kwargs)

    def cp_file(self, path1, path2, **kwargs):
        return self.fs.cp_file(self._join(path1), self._join(path2), **kwargs)

    async def _copy(
        self,
        path1,
        path2,
        *args,
        **kwargs,
    ):
        return await self.fs._copy(
            self._join(path1),
            self._join(path2),
            *args,
            **kwargs,
        )

    def copy(self, path1, path2, *args, **kwargs):
        return self.fs.copy(
            self._join(path1),
            self._join(path2),
            *args,
            **kwargs,
        )

    async def _pipe(self, path, *args, **kwargs):
        return await self.fs._pipe(self._join(path), *args, **kwargs)

    def pipe(self, path, *args, **kwargs):
        return self.fs.pipe(self._join(path), *args, **kwargs)

    async def _pipe_file(self, path, *args, **kwargs):
        return await self.fs._pipe_file(self._join(path), *args, **kwargs)

    def pipe_file(self, path, *args, **kwargs):
        return self.fs.pipe_file(self._join(path), *args, **kwargs)

    def write_text(
        self, path, value, encoding=None, errors=None, newline=None, **kwargs
    ):
        return self.fs.write_text(
            self._join(path),
            value,
            encoding=encoding,
            errors=errors,
            newline=newline,
            **kwargs,
        )

    async def _cat_file(self, path, *args, **kwargs):
        return await self.fs._cat_file(self._join(path), *args, **kwargs)

    def cat_file(self, path, *args, **kwargs):
        return self.fs.cat_file(self._join(path), *args, **kwargs)

    async def _cat(self, path, *args, **kwargs):
        ret = await self.fs._cat(
            self._join(path),
            *args,
            **kwargs,
        )

        if isinstance(ret, dict):
            return {self._relpath(key): value for key, value in ret.items()}

        return ret

    def cat(self, path, *args, **kwargs):
        ret = self.fs.cat(
            self._join(path),
            *args,
            **kwargs,
        )

        if isinstance(ret, dict):
            return {self._relpath(key): value for key, value in ret.items()}

        return ret

    async def _put_file(self, lpath, rpath, **kwargs):
        return await self.fs._put_file(lpath, self._join(rpath), **kwargs)

    def put_file(self, lpath, rpath, **kwargs):
        return self.fs.put_file(lpath, self._join(rpath), **kwargs)

    async def _put(
        self,
        lpath,
        rpath,
        *args,
        **kwargs,
    ):
        return await self.fs._put(
            lpath,
            self._join(rpath),
            *args,
            **kwargs,
        )

    def put(self, lpath, rpath, *args, **kwargs):
        return self.fs.put(
            lpath,
            self._join(rpath),
            *args,
            **kwargs,
        )

    async def _get_file(self, rpath, lpath, **kwargs):
        return await self.fs._get_file(self._join(rpath), lpath, **kwargs)

    def get_file(self, rpath, lpath, **kwargs):
        return self.fs.get_file(self._join(rpath), lpath, **kwargs)

    async def _get(self, rpath, *args, **kwargs):
        return await self.fs._get(self._join(rpath), *args, **kwargs)

    def get(self, rpath, *args, **kwargs):
        return self.fs.get(self._join(rpath), *args, **kwargs)

    async def _isfile(self, path):
        return await self.fs._isfile(self._join(path))

    def isfile(self, path):
        return self.fs.isfile(self._join(path))

    async def _isdir(self, path):
        return await self.fs._isdir(self._join(path))

    def isdir(self, path):
        return self.fs.isdir(self._join(path))

    async def _size(self, path):
        return await self.fs._size(self._join(path))

    def size(self, path):
        return self.fs.size(self._join(path))

    async def _exists(self, path):
        return await self.fs._exists(self._join(path))

    def exists(self, path):
        return self.fs.exists(self._join(path))

    async def _info(self, path, **kwargs):
        info = await self.fs._info(self._join(path), **kwargs)
        info = info.copy()
        info["name"] = self._relpath(info["name"])
        return info

    def info(self, path, **kwargs):
        info = self.fs.info(self._join(path), **kwargs)
        info = info.copy()
        info["name"] = self._relpath(info["name"])
        return info

    async def _ls(self, path, detail=True, **kwargs):
        ret = (await self.fs._ls(self._join(path), detail=detail, **kwargs)).copy()
        if detail:
            out = []
            for entry in ret:
                entry = entry.copy()
                entry["name"] = self._relpath(entry["name"])
                out.append(entry)
            return out

        return self._relpath(ret)

    def ls(self, path, detail=True, **kwargs):
        ret = self.fs.ls(self._join(path), detail=detail, **kwargs).copy()
        if detail:
            out = []
            for entry in ret:
                entry = entry.copy()
                entry["name"] = self._relpath(entry["name"])
                out.append(entry)
            return out

        return self._relpath(ret)

    async def _walk(self, path, *args, **kwargs):
        async for root, dirs, files in self.fs._walk(self._join(path), *args, **kwargs):
            yield self._relpath(root), dirs, files

    def walk(self, path, *args, **kwargs):
        for root, dirs, files in self.fs.walk(self._join(path), *args, **kwargs):
            yield self._relpath(root), dirs, files

    async def _glob(self, path, **kwargs):
        detail = kwargs.get("detail", False)
        ret = await self.fs._glob(self._join(path), **kwargs)
        if detail:
            return {self._relpath(path): info for path, info in ret.items()}
        return self._relpath(ret)

    def glob(self, path, **kwargs):
        detail = kwargs.get("detail", False)
        ret = self.fs.glob(self._join(path), **kwargs)
        if detail:
            return {self._relpath(path): info for path, info in ret.items()}
        return self._relpath(ret)

    async def _du(self, path, *args, **kwargs):
        total = kwargs.get("total", True)
        ret = await self.fs._du(self._join(path), *args, **kwargs)
        if total:
            return ret

        return {self._relpath(path): size for path, size in ret.items()}

    def du(self, path, *args, **kwargs):
        total = kwargs.get("total", True)
        ret = self.fs.du(self._join(path), *args, **kwargs)
        if total:
            return ret

        return {self._relpath(path): size for path, size in ret.items()}

    async def _find(self, path, *args, **kwargs):
        detail = kwargs.get("detail", False)
        ret = await self.fs._find(self._join(path), *args, **kwargs)
        if detail:
            return {self._relpath(path): info for path, info in ret.items()}
        return self._relpath(ret)

    def find(self, path, *args, **kwargs):
        detail = kwargs.get("detail", False)
        ret = self.fs.find(self._join(path), *args, **kwargs)
        if detail:
            return {self._relpath(path): info for path, info in ret.items()}
        return self._relpath(ret)

    async def _expand_path(self, path, *args, **kwargs):
        return self._relpath(
            await self.fs._expand_path(self._join(path), *args, **kwargs)
        )

    def expand_path(self, path, *args, **kwargs):
        return self._relpath(self.fs.expand_path(self._join(path), *args, **kwargs))

    async def _mkdir(self, path, *args, **kwargs):
        return await self.fs._mkdir(self._join(path), *args, **kwargs)

    def mkdir(self, path, *args, **kwargs):
        return self.fs.mkdir(self._join(path), *args, **kwargs)

    async def _makedirs(self, path, *args, **kwargs):
        return await self.fs._makedirs(self._join(path), *args, **kwargs)

    def makedirs(self, path, *args, **kwargs):
        return self.fs.makedirs(self._join(path), *args, **kwargs)

    def rmdir(self, path):
        return self.fs.rmdir(self._join(path))

    def mv(self, path1, path2, **kwargs):
        return self.fs.mv(
            self._join(path1),
            self._join(path2),
            **kwargs,
        )

    def touch(self, path, **kwargs):
        return self.fs.touch(self._join(path), **kwargs)

    def created(self, path):
        return self.fs.created(self._join(path))

    def modified(self, path):
        return self.fs.modified(self._join(path))

    def sign(self, path, *args, **kwargs):
        return self.fs.sign(self._join(path), *args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__qualname__}(path='{self.path}', fs={self.fs})"

    def open(
        self,
        path,
        *args,
        **kwargs,
    ):
        return self.fs.open(
            self._join(path),
            *args,
            **kwargs,
        )

    async def open_async(
        self,
        path,
        *args,
        **kwargs,
    ):
        return await self.fs.open_async(
            self._join(path),
            *args,
            **kwargs,
        )
<|endoftext|>from __future__ import annotations

import logging
from datetime import datetime, timezone
from errno import ENOTEMPTY
from io import BytesIO
from pathlib import PurePath, PureWindowsPath
from typing import Any, ClassVar

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.utils import stringify_path

logger = logging.getLogger("fsspec.memoryfs")


class MemoryFileSystem(AbstractFileSystem):
    """A filesystem based on a dict of BytesIO objects

    This is a global filesystem so instances of this class all point to the same
    in memory filesystem.
    """

    store: ClassVar[dict[str, Any]] = {}  # global, do not overwrite!
    pseudo_dirs = [""]  # global, do not overwrite!
    protocol = "memory"
    root_marker = "/"

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, PurePath):
            if isinstance(path, PureWindowsPath):
                return LocalFileSystem._strip_protocol(path)
            else:
                path = stringify_path(path)

        path = path.removeprefix("memory://")
        if "::" in path or "://" in path:
            return path.rstrip("/")
        path = path.lstrip("/").rstrip("/")
        return "/" + path if path else ""

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        # The base implementation calls ls() once per directory, and each ls()
        # scans the whole (global) store, giving O(n_dirs * n_entries) behaviour
        # for a tree. Since the store is a flat mapping of every path, the same
        # result can be produced with a single pass over it.
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")
        path = self._strip_protocol(path)
        if path in self.store:
            # path is itself a file
            if not detail:
                return [path]
            filelike = self.store[path]
            return {
                path: {
                    "name": path,
                    "size": filelike.size,
                    "type": "file",
                    "created": filelike.created.timestamp(),
                }
            }

        # Uniform prefix so that the search root "" (the filesystem root) and a
        # nested path are handled the same way; rel depth is rel.count("/") + 1.
        prefix = path + "/" if path else "/"
        out = {}
        dirs = {}

        def add_ancestor_dirs(name):
            # Register every directory implied between ``path`` and ``name`` that
            # is within maxdepth, mirroring how walk() surfaces implied dirs.
            idx = name.rfind("/")
            while idx > len(path):
                parent = name[:idx]
                if parent in dirs:
                    break
                rel = parent[len(prefix) :]
                if maxdepth is None or rel.count("/") + 1 <= maxdepth:
                    dirs[parent] = {"name": parent, "size": 0, "type": "directory"}
                idx = parent.rfind("/")

        # `store` is shared by every MemoryFileSystem instance, so iterate a
        # snapshot: a concurrent create/delete would otherwise raise
        # "dictionary changed size during iteration". ls() does the same.
        for name, filelike in tuple(self.store.items()):
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix) :]
            if withdirs:
                add_ancestor_dirs(name)
            if maxdepth is not None and rel.count("/") + 1 > maxdepth:
                continue
            out[name] = {
                "name": name,
                "size": filelike.size,
                "type": "file",
                "created": filelike.created.timestamp(),
            }

        if withdirs:
            # Explicitly-created (possibly empty) directories live in pseudo_dirs.
            for pdir in self.pseudo_dirs:
                if pdir and pdir.startswith(prefix):
                    add_ancestor_dirs(pdir + "/")
            out.update(dirs)
            # Mirror the base find(): include the search root itself when it is
            # a directory (needed for posix glob compliance).
            if path != "" and self.isdir(path):
                out[path] = self.info(path)

        names = sorted(out)
        return {name: out[name] for name in names} if detail else names

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        if path in self.store:
            # there is a key with this exact name
            if not detail:
                return [path]
            return [
                {
                    "name": path,
                    "size": self.store[path].size,
                    "type": "file",
                    "created": self.store[path].created.timestamp(),
                }
            ]
        paths = set()
        starter = path + "/"
        out = []
        for p2 in tuple(self.store):
            if p2.startswith(starter):
                if "/" not in p2[len(starter) :]:
                    # exact child
                    out.append(
                        {
                            "name": p2,
                            "size": self.store[p2].size,
                            "type": "file",
                            "created": self.store[p2].created.timestamp(),
                        }
                    )
                elif len(p2) > len(starter):
                    # implied child directory
                    ppath = starter + p2[len(starter) :].split("/", 1)[0]
                    if ppath not in paths:
                        out = out or []
                        out.append(
                            {
                                "name": ppath,
                                "size": 0,
                                "type": "directory",
                            }
                        )
                        paths.add(ppath)
        for p2 in self.pseudo_dirs:
            if p2.startswith(starter):
                if "/" not in p2[len(starter) :]:
                    # exact child pdir
                    if p2 not in paths:
                        out.append({"name": p2, "size": 0, "type": "directory"})
                        paths.add(p2)
                else:
                    # directory implied by deeper pdir
                    ppath = starter + p2[len(starter) :].split("/", 1)[0]
                    if ppath not in paths:
                        out.append({"name": ppath, "size": 0, "type": "directory"})
                        paths.add(ppath)
        if not out:
            if path in self.pseudo_dirs:
                # empty dir
                return []
            raise FileNotFoundError(path)
        if detail:
            return out
        return sorted([f["name"] for f in out])

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if path in self.store or path in self.pseudo_dirs:
            raise FileExistsError(path)
        if self._parent(path).strip("/") and self.isfile(self._parent(path)):
            raise NotADirectoryError(self._parent(path))
        if create_parents and self._parent(path).strip("/"):
            try:
                self.mkdir(self._parent(path), create_parents, **kwargs)
            except FileExistsError:
                pass
        if path and path not in self.pseudo_dirs:
            self.pseudo_dirs.append(path)

    def makedirs(self, path, exist_ok=False):
        try:
            self.mkdir(path, create_parents=True)
        except FileExistsError:
            if not exist_ok:
                raise

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        """Set the bytes of given file

        Avoids copies of the data if possible
        """
        mode = "xb" if mode == "create" else "wb"
        self.open(path, mode=mode, data=value)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        if path == "":
            # silently avoid deleting FS root
            return
        if path in self.pseudo_dirs:
            if not self.ls(path):
                self.pseudo_dirs.remove(path)
            else:
                raise OSError(ENOTEMPTY, "Directory not empty", path)
        else:
            raise FileNotFoundError(path)

    def info(self, path, **kwargs):
        logger.debug("info: %s", path)
        path = self._strip_protocol(path)
        if path in self.pseudo_dirs or any(
            p.startswith(path + "/") for p in list(self.store) + self.pseudo_dirs
        ):
            return {
                "name": path,
                "size": 0,
                "type": "directory",
            }
        elif path in self.store:
            filelike = self.store[path]
            return {
                "name": path,
                "size": filelike.size,
                "type": "file",
                "created": getattr(filelike, "created", None),
            }
        else:
            raise FileNotFoundError(path)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if "x" in mode and self.exists(path):
            raise FileExistsError
        if path in self.pseudo_dirs:
            raise IsADirectoryError(path)
        parent = path
        while len(parent) > 1:
            parent = self._parent(parent)
            if self.isfile(parent):
                raise FileExistsError(parent)
        if mode in ["rb", "ab", "r+b", "a+b"]:
            if path in self.store:
                f = self.store[path]
                if "a" in mode:
                    # position at the end of file
                    f.seek(0, 2)
                else:
                    # position at the beginning of file
                    f.seek(0)
                return f
            elif "a" in mode:
                # append modes create the file if it does not exist, matching
                # builtin open() and LocalFileSystem
                m = MemoryFile(self, path, kwargs.get("data"))
                if not self._intrans:
                    m.commit()
                # position at the end of file, like the existing-file path above
                m.seek(0, 2)
                return m
            else:
                raise FileNotFoundError(path)
        elif mode in {"wb", "w+b", "xb", "x+b"}:
            if "x" in mode and self.exists(path):
                raise FileExistsError
            m = MemoryFile(self, path, kwargs.get("data"))
            if not self._intrans:
                m.commit()
            return m
        else:
            name = self.__class__.__name__
            raise ValueError(f"unsupported file mode for {name}: {mode!r}")

    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        if self.isfile(path1):
            self.store[path2] = MemoryFile(
                self, path2, self.store[path1].getvalue()
            )  # implicit copy
        elif self.isdir(path1):
            if path2 not in self.pseudo_dirs:
                self.pseudo_dirs.append(path2)
        else:
            raise FileNotFoundError(path1)

    def cat_file(self, path, start=None, end=None, **kwargs):
        logger.debug("cat: %s", path)
        path = self._strip_protocol(path)
        try:
            return bytes(self.store[path].getbuffer()[start:end])
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def _rm(self, path):
        path = self._strip_protocol(path)
        try:
            del self.store[path]
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def modified(self, path):
        path = self._strip_protocol(path)
        try:
            return self.store[path].modified
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def created(self, path):
        path = self._strip_protocol(path)
        try:
            return self.store[path].created
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def isfile(self, path):
        path = self._strip_protocol(path)
        return path in self.store

    def rm(self, path, recursive=False, maxdepth=None):
        if isinstance(path, str):
            path = self._strip_protocol(path)
        else:
            path = [self._strip_protocol(p) for p in path]
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(paths):
            if self.isfile(p):
                self.rm_file(p)
            # If the expanded path doesn't exist, it is only because the expanded
            # path was a directory that does not exist in self.pseudo_dirs. This
            # is possible if you directly create files without making the
            # directories first.
            elif not self.exists(p):
                continue
            else:
                self.rmdir(p)


class MemoryFile(BytesIO):
    """A BytesIO which can't close and works as a context manager

    Can initialise with data. Each path should only be active once at any moment.

    No need to provide fs, path if auto-committing (default)
    """

    def __init__(self, fs=None, path=None, data=None):
        logger.debug("open file %s", path)
        self.fs = fs
        self.path = path
        self.created = datetime.now(tz=timezone.utc)
        self.modified = datetime.now(tz=timezone.utc)
        if data:
            super().__init__(data)
            self.seek(0)

    @property
    def size(self):
        return self.getbuffer().nbytes

    def __enter__(self):
        return self

    def close(self):
        pass

    def discard(self):
        pass

    def commit(self):
        self.fs.store[self.path] = self
        self.modified = datetime.now(tz=timezone.utc)
<|endoftext|>import requests

from ..spec import AbstractFileSystem
from ..utils import infer_storage_options
from .memory import MemoryFile


class GistFileSystem(AbstractFileSystem):
    """
    Interface to files in a single GitHub Gist.

    Provides read-only access to a gist's files. Gists do not contain
    subdirectories, so file listing is straightforward.

    Parameters
    ----------
    gist_id: str
        The ID of the gist you want to access (the long hex value from the URL).
    filenames: list[str] (optional)
        If provided, only make a file system representing these files, and do not fetch
        the list of all files for this gist.
    sha: str (optional)
        If provided, fetch a particular revision of the gist. If omitted,
        the latest revision is used.
    username: str (optional)
        GitHub username for authentication.
    token: str (optional)
        GitHub personal access token (required if username is given), or.
    timeout: (float, float) or float, optional
        Connect and read timeouts for requests (default 60s each).
    kwargs: dict
        Stored on `self.request_kw` and passed to `requests.get` when fetching Gist
        metadata or reading ("opening") a file.
    """

    protocol = "gist"
    gist_url = "https://api.github.com/gists/{gist_id}"
    gist_rev_url = "https://api.github.com/gists/{gist_id}/{sha}"

    def __init__(
        self,
        gist_id,
        filenames=None,
        sha=None,
        username=None,
        token=None,
        timeout=None,
        **kwargs,
    ):
        super().__init__()
        self.gist_id = gist_id
        self.filenames = filenames
        self.sha = sha  # revision of the gist (optional)
        if username is not None and token is None:
            raise ValueError("User auth requires a token")
        self.username = username
        self.token = token
        self.request_kw = kwargs
        # Default timeouts to 60s connect/read if none provided
        self.timeout = timeout if timeout is not None else (60, 60)

        # We use a single-level "directory" cache, because a gist is essentially flat
        self.dircache[""] = self._fetch_file_list()

    @property
    def kw(self):
        """Auth parameters passed to 'requests' if we have username/token."""
        kw = {
            "headers": {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        }
        kw.update(self.request_kw)
        if self.username and self.token:
            kw["auth"] = (self.username, self.token)
        elif self.token:
            kw["headers"]["Authorization"] = f"Bearer {self.token}"
        return kw

    def _fetch_gist_metadata(self):
        """
        Fetch the JSON metadata for this gist (possibly for a specific revision).
        """
        if self.sha:
            url = self.gist_rev_url.format(gist_id=self.gist_id, sha=self.sha)
        else:
            url = self.gist_url.format(gist_id=self.gist_id)

        r = requests.get(url, timeout=self.timeout, **self.kw)
        if r.status_code == 404:
            raise FileNotFoundError(
                f"Gist not found: {self.gist_id}@{self.sha or 'latest'}"
            )
        r.raise_for_status()
        return r.json()

    def _fetch_file_list(self):
        """
        Returns a list of dicts describing each file in the gist. These get stored
        in self.dircache[""].
        """
        meta = self._fetch_gist_metadata()
        if self.filenames:
            available_files = meta.get("files", {})
            files = {}
            for fn in self.filenames:
                if fn not in available_files:
                    raise FileNotFoundError(fn)
                files[fn] = available_files[fn]
        else:
            files = meta.get("files", {})

        out = []
        for fname, finfo in files.items():
            if finfo is None:
                # Occasionally GitHub returns a file entry with null if it was deleted
                continue
            # Build a directory entry
            out.append(
                {
                    "name": fname,  # file's name
                    "type": "file",  # gists have no subdirectories
                    "size": finfo.get("size", 0),  # file size in bytes
                    "raw_url": finfo.get("raw_url"),
                }
            )
        return out

    @classmethod
    def _strip_protocol(cls, path):
        """
        Remove 'gist://' from the path, if present.
        """
        # The default infer_storage_options can handle gist://username:token@id/file
        # or gist://id/file, but let's ensure we handle a normal usage too.
        # We'll just strip the protocol prefix if it exists.
        path = infer_storage_options(path).get("path", path)
        return path.lstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(path):
        """
        Parse 'gist://' style URLs into GistFileSystem constructor kwargs.
        For example:
          gist://:TOKEN@<gist_id>/file.txt
          gist://username:TOKEN@<gist_id>/file.txt
        """
        so = infer_storage_options(path)
        out = {}
        if "username" in so and so["username"]:
            out["username"] = so["username"]
        if "password" in so and so["password"]:
            out["token"] = so["password"]
        if "host" in so and so["host"]:
            # We interpret 'host' as the gist ID
            out["gist_id"] = so["host"]

        # Extract SHA and filename from path
        if "path" in so and so["path"]:
            path_parts = so["path"].rsplit("/", 2)[-2:]
            if len(path_parts) == 2:
                if path_parts[0]:  # SHA present
                    out["sha"] = path_parts[0]
                if path_parts[1]:  # filename also present
                    out["filenames"] = [path_parts[1]]

        return out

    def ls(self, path="", detail=False, **kwargs):
        """
        List files in the gist. Gists are single-level, so any 'path' is basically
        the filename, or empty for all files.

        Parameters
        ----------
        path : str, optional
            The filename to list. If empty, returns all files in the gist.
        detail : bool, default False
            If True, return a list of dicts; if False, return a list of filenames.
        """
        path = self._strip_protocol(path or "")
        # If path is empty, return all
        if path == "":
            results = self.dircache[""]
        else:
            # We want just the single file with this name
            all_files = self.dircache[""]
            results = [f for f in all_files if f["name"] == path]
            if not results:
                raise FileNotFoundError(path)
        if detail:
            return results
        else:
            return sorted(f["name"] for f in results)

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        """
        Read a single file from the gist.
        """
        if mode != "rb":
            raise NotImplementedError("GitHub Gist FS is read-only (no write).")

        path = self._strip_protocol(path)
        # Find the file entry in our dircache
        matches = [f for f in self.dircache[""] if f["name"] == path]
        if not matches:
            raise FileNotFoundError(path)
        finfo = matches[0]

        raw_url = finfo.get("raw_url")
        if not raw_url:
            raise FileNotFoundError(f"No raw_url for file: {path}")

        r = requests.get(raw_url, timeout=self.timeout, **self.kw)
        if r.status_code == 404:
            raise FileNotFoundError(path)
        r.raise_for_status()
        return MemoryFile(path, None, r.content)

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        """
        Return {path: contents} for the given file or files. If 'recursive' is True,
        and path is empty, returns all files in the gist.
        """
        paths = self.expand_path(path, recursive=recursive)
        out = {}
        for p in paths:
            try:
                with self.open(p, "rb") as f:
                    out[p] = f.read()
            except FileNotFoundError as e:
                if on_error == "raise":
                    raise e
                elif on_error == "omit":
                    pass  # skip
                else:
                    out[p] = e
        if len(paths) == 1 and paths[0] == path:
            return out[path]
        return out
<|endoftext|>import base64
import collections
import io
import itertools
import logging
import math
import os
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Literal

import fsspec.core
from fsspec.spec import AbstractBufferedFile

try:
    import ujson as json
except ImportError:
    if not TYPE_CHECKING:
        import json

from fsspec.asyn import AsyncFileSystem
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.core import filesystem, open, split_protocol
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.utils import (
    isfilelike,
    merge_offset_ranges,
    other_paths,
)

logger = logging.getLogger("fsspec.reference")


class ReferenceNotReachable(RuntimeError):
    def __init__(self, reference, target, *args):
        super().__init__(*args)
        self.reference = reference
        self.target = target

    def __str__(self):
        return f'Reference "{self.reference}" failed to fetch target {self.target}'


def _first(d):
    return next(iter(d.values()))


def _prot_in_references(path, references):
    ref = references.get(path)
    if isinstance(ref, (list, tuple)) and isinstance(ref[0], str):
        return split_protocol(ref[0])[0] if ref[0] else ref[0]


def _protocol_groups(paths, references):
    if isinstance(paths, str):
        return {_prot_in_references(paths, references): [paths]}
    out = {}
    for path in paths:
        protocol = _prot_in_references(path, references)
        out.setdefault(protocol, []).append(path)
    return out


class RefsValuesView(collections.abc.ValuesView):
    def __iter__(self):
        for val in self._mapping.zmetadata.values():
            yield json.dumps(val).encode()
        yield from self._mapping._items.values()
        for field in self._mapping.listdir():
            chunk_sizes = self._mapping._get_chunk_sizes(field)
            if len(chunk_sizes) == 0:
                yield self._mapping[field + "/0"]
                continue
            yield from self._mapping._generate_all_records(field)


class RefsItemsView(collections.abc.ItemsView):
    def __iter__(self):
        return zip(self._mapping.keys(), self._mapping.values())


def ravel_multi_index(idx, sizes):
    val = 0
    mult = 1
    for i, s in zip(idx[::-1], sizes[::-1]):
        val += i * mult
        mult *= s
    return val


class LazyReferenceMapper(collections.abc.MutableMapping):
    """This interface can be used to read/write references from Parquet stores.
    It is not intended for other types of references.
    It can be used with Kerchunk's MultiZarrToZarr method to combine
    references into a parquet store.
    Examples of this use-case can be found here:
    https://fsspec.github.io/kerchunk/advanced.html?highlight=parquet#parquet-storage"""

    # import is class level to prevent numpy dep requirement for fsspec
    @property
    def np(self):
        import numpy as np

        return np

    @property
    def pd(self):
        import pandas as pd

        return pd

    def __init__(
        self,
        root,
        fs=None,
        out_root=None,
        cache_size=128,
        categorical_threshold=10,
        engine: Literal["fastparquet", "pyarrow"] = "fastparquet",
    ):
        """

        This instance will be writable, storing changes in memory until full partitions
        are accumulated or .flush() is called.

        To create an empty lazy store, use .create()

        Parameters
        ----------
        root : str
            Root of parquet store
        fs : fsspec.AbstractFileSystem
            fsspec filesystem object, default is local filesystem.
        cache_size : int, default=128
            Maximum size of LRU cache, where cache_size*record_size denotes
            the total number of references that can be loaded in memory at once.
        categorical_threshold : int
            Encode urls as pandas.Categorical to reduce memory footprint if the ratio
            of the number of unique urls to total number of refs for each variable
            is greater than or equal to this number. (default 10)
        engine: Literal["fastparquet","pyarrow"]
            Engine choice for reading parquet files. (default is "fastparquet")
        """

        self.root = root
        self.chunk_sizes = {}
        self.cat_thresh = categorical_threshold
        self.engine = engine
        self.cache_size = cache_size
        self.url = self.root + "/{field}/refs.{record}.parq"
        # TODO: derive fs from `root`
        self.fs = fsspec.filesystem("file") if fs is None else fs
        self.out_root = self.fs.unstrip_protocol(out_root or self.root)

        from importlib.util import find_spec

        if self.engine == "pyarrow" and find_spec("pyarrow") is None:
            raise ImportError("engine choice `pyarrow` is not installed.")

        # Apply `lru_cache` decorator manually per instance.
        # This way `self` reference is not held on class level.
        # WARNING: However, this means that self and its members are not reflected
        # in the cache key, so we expect they won't be mutated once a value is cached.
        self.listdir = lru_cache()(self.listdir)
        self._key_to_record = lru_cache(maxsize=4096)(self._key_to_record)

    def __getattr__(self, item):
        if item in ("_items", "record_size", "zmetadata"):
            self.setup()
            # avoid possible recursion if setup fails somehow
            return self.__dict__[item]
        raise AttributeError(item)

    def setup(self):
        self._items = {}
        self._items[".zmetadata"] = self.fs.cat_file(
            "/".join([self.root, ".zmetadata"])
        )
        met = json.loads(self._items[".zmetadata"])
        self.record_size = met["record_size"]
        self.zmetadata = met["metadata"]

        # Define function to open and decompress refs
        @lru_cache(maxsize=self.cache_size)
        def open_refs(field, record):
            """cached parquet file loader"""
            path = self.url.format(field=field, record=record)
            data = io.BytesIO(self.fs.cat_file(path))
            try:
                df = self.pd.read_parquet(data, engine=self.engine)
                refs = {c: df[c].to_numpy(copy=True) for c in df.columns}
            except OSError:
                refs = None
            return refs

        self.open_refs = open_refs

    @staticmethod
    def create(root, storage_options=None, fs=None, record_size=10000, **kwargs):
        """Make empty parquet reference set

        First deletes the contents of the given directory, if it exists.

        Parameters
        ----------
        root: str
            Directory to contain the output; will be created
        storage_options: dict | None
            For making the filesystem to use for writing is fs is None
        fs: FileSystem | None
            Filesystem for writing
        record_size: int
            Number of references per parquet file
        kwargs: passed to __init__

        Returns
        -------
        LazyReferenceMapper instance
        """
        met = {"metadata": {}, "record_size": record_size}
        if fs is None:
            fs, root = fsspec.core.url_to_fs(root, **(storage_options or {}))
        if fs.exists(root):
            fs.rm(root, recursive=True)
        fs.makedirs(root, exist_ok=True)
        fs.pipe("/".join([root, ".zmetadata"]), json.dumps(met).encode())
        return LazyReferenceMapper(root, fs, **kwargs)

    def listdir(self):
        """List top-level directories"""
        dirs = (p.rsplit("/", 1)[0] for p in self.zmetadata if not p.startswith(".z"))
        return set(dirs)

    def ls(self, path="", detail=True):
        """Shortcut file listings"""
        path = path.rstrip("/")
        pathdash = path + "/" if path else ""
        dirnames = self.listdir()
        dirs = [
            d
            for d in dirnames
            if d.startswith(pathdash) and "/" not in d.lstrip(pathdash)
        ]
        if dirs:
            others = {
                f
                for f in chain(
                    [".zmetadata"],
                    (name for name in self.zmetadata),
                    (name for name in self._items),
                )
                if f.startswith(pathdash) and "/" not in f.lstrip(pathdash)
            }
            if detail is False:
                others.update(dirs)
                return sorted(others)
            dirinfo = [{"name": name, "type": "directory", "size": 0} for name in dirs]
            fileinfo = [
                {
                    "name": name,
                    "type": "file",
                    "size": len(
                        json.dumps(self.zmetadata[name])
                        if name in self.zmetadata
                        else self._items[name]
                    ),
                }
                for name in others
            ]
            return sorted(dirinfo + fileinfo, key=lambda s: s["name"])
        field = path
        others = set(
            [name for name in self.zmetadata if name.startswith(f"{path}/")]
            + [name for name in self._items if name.startswith(f"{path}/")]
        )
        fileinfo = [
            {
                "name": name,
                "type": "file",
                "size": len(
                    json.dumps(self.zmetadata[name])
                    if name in self.zmetadata
                    else self._items[name]
                ),
            }
            for name in others
        ]
        keys = self._keys_in_field(field)

        if detail is False:
            return list(others) + list(keys)
        recs = self._generate_all_records(field)
        recinfo = [
            {"name": name, "type": "file", "size": rec[-1]}
            for name, rec in zip(keys, recs)
            if rec[0]  # filters out path==None, deleted/missing
        ]
        return fileinfo + recinfo

    def _load_one_key(self, key):
        """Get the reference for one key

        Returns bytes, one-element list or three-element list.
        """
        if key in self._items:
            return self._items[key]
        elif key in self.zmetadata:
            return json.dumps(self.zmetadata[key]).encode()
        elif "/" not in key or self._is_meta(key):
            raise KeyError(key)
        field, _ = key.rsplit("/", 1)
        record, ri, chunk_size = self._key_to_record(key)
        maybe = self._items.get((field, record), {}).get(ri, False)
        if maybe is None:
            # explicitly deleted
            raise KeyError
        elif maybe:
            return maybe
        elif chunk_size == 0:
            return b""

        # Chunk keys can be loaded from row group and cached in LRU cache
        try:
            refs = self.open_refs(field, record)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            raise KeyError(key) from exc
        columns = ["path", "offset", "size", "raw"]
        selection = [refs[c][ri] if c in refs else None for c in columns]
        raw = selection[-1]
        if raw is not None:
            return raw
        if selection[0] is None:
            raise KeyError("This reference does not exist or has been deleted")
        if selection[1:3] == [0, 0]:
            # URL only
            return selection[:1]
        # URL, offset, size
        return selection[:3]

    def _key_to_record(self, key):
        """Details needed to construct a reference for one key"""
        field, chunk = key.rsplit("/", 1)
        chunk_sizes = self._get_chunk_sizes(field)
        if len(chunk_sizes) == 0:
            return 0, 0, 0
        chunk_idx = [int(c) for c in chunk.split(".")]
        chunk_number = ravel_multi_index(chunk_idx, chunk_sizes)
        record = chunk_number // self.record_size
        ri = chunk_number % self.record_size
        return record, ri, len(chunk_sizes)

    def _get_chunk_sizes(self, field):
        """The number of chunks along each axis for a given field"""
        if field not in self.chunk_sizes:
            zarray = self.zmetadata[f"{field}/.zarray"]
            size_ratio = [
                math.ceil(s / c) for s, c in zip(zarray["shape"], zarray["chunks"])
            ]
            self.chunk_sizes[field] = size_ratio or [1]
        return self.chunk_sizes[field]

    def _generate_record(self, field, record):
        """The references for a given parquet file of a given field"""
        refs = self.open_refs(field, record)
        it = iter(zip(*refs.values()))
        if len(refs) == 3:
            # All urls
            return (list(t) for t in it)
        elif len(refs) == 1:
            # All raws
            return refs["raw"]
        else:
            # Mix of urls and raws
            return (list(t[:3]) if not t[3] else t[3] for t in it)

    def _generate_all_records(self, field):
        """Load all the references within a field by iterating over the parquet files"""
        nrec = 1
        for ch in self._get_chunk_sizes(field):
            nrec *= ch
        nrec = math.ceil(nrec / self.record_size)
        for record in range(nrec):
            yield from self._generate_record(field, record)

    def values(self):
        return RefsValuesView(self)

    def items(self):
        return RefsItemsView(self)

    def __hash__(self):
        return id(self)

    def __getitem__(self, key):
        return self._load_one_key(key)

    def __setitem__(self, key, value):
        if "/" in key and not self._is_meta(key):
            field, chunk = key.rsplit("/", 1)
            record, i, _ = self._key_to_record(key)
            subdict = self._items.setdefault((field, record), {})
            subdict[i] = value
            if len(subdict) == self.record_size:
                self.write(field, record)
        else:
            # metadata or top-level
            if hasattr(value, "to_bytes"):
                val = value.to_bytes().decode()
            elif isinstance(value, bytes):
                val = value.decode()
            else:
                val = value
            self._items[key] = val
            new_value = json.loads(val)
            self.zmetadata[key] = {**self.zmetadata.get(key, {}), **new_value}

    @staticmethod
    def _is_meta(key):
        return key.startswith(".z") or "/.z" in key

    def __delitem__(self, key):
        if key in self._items:
            del self._items[key]
        elif key in self.zmetadata:
            del self.zmetadata[key]
        else:
            if "/" in key and not self._is_meta(key):
                field, _ = key.rsplit("/", 1)
                record, i, _ = self._key_to_record(key)
                subdict = self._items.setdefault((field, record), {})
                subdict[i] = None
                if len(subdict) == self.record_size:
                    self.write(field, record)
            else:
                # metadata or top-level
                self._items[key] = None

    def write(self, field, record, base_url=None, storage_options=None):
        # extra requirements if writing
        import kerchunk.df
        import numpy as np
        import pandas as pd

        partition = self._items[(field, record)]
        original = False
        if len(partition) < self.record_size:
            try:
                original = self.open_refs(field, record)
            except OSError:
                pass

        if original:
            paths = original["path"]
            offsets = original["offset"]
            sizes = original["size"]
            raws = original["raw"]
        else:
            paths = np.full(self.record_size, np.nan, dtype="O")
            offsets = np.zeros(self.record_size, dtype="int64")
            sizes = np.zeros(self.record_size, dtype="int64")
            raws = np.full(self.record_size, np.nan, dtype="O")
        for j, data in partition.items():
            if isinstance(data, list):
                if (
                    str(paths.dtype) == "category"
                    and data[0] not in paths.dtype.categories
                ):
                    paths = paths.add_categories(data[0])
                paths[j] = data[0]
                if len(data) > 1:
                    offsets[j] = data[1]
                    sizes[j] = data[2]
            elif data is None:
                # delete
                paths[j] = None
                offsets[j] = 0
                sizes[j] = 0
                raws[j] = None
            else:
                # this is the only call into kerchunk, could remove
                raws[j] = kerchunk.df._proc_raw(data)
        # TODO: only save needed columns
        df = pd.DataFrame(
            {
                "path": paths,
                "offset": offsets,
                "size": sizes,
                "raw": raws,
            },
            copy=False,
        )
        if df.path.count() / (df.path.nunique() or 1) > self.cat_thresh:
            df["path"] = df["path"].astype("category")
        object_encoding = {"raw": "bytes", "path": "utf8"}
        has_nulls = ["path", "raw"]

        fn = f"{base_url or self.out_root}/{field}/refs.{record}.parq"
        self.fs.mkdirs(f"{base_url or self.out_root}/{field}", exist_ok=True)

        if self.engine == "pyarrow":
            df_backend_kwargs = {"write_statistics": False}
        elif self.engine == "fastparquet":
            df_backend_kwargs = {
                "stats": False,
                "object_encoding": object_encoding,
                "has_nulls": has_nulls,
            }
        else:
            raise NotImplementedError(f"{self.engine} not supported")
        df.to_parquet(
            fn,
            engine=self.engine,
            storage_options=storage_options
            or getattr(self.fs, "storage_options", None),
            compression="zstd",
            index=False,
            **df_backend_kwargs,
        )

        partition.clear()
        self._items.pop((field, record))

    def flush(self, base_url=None, storage_options=None):
        """Output any modified or deleted keys

        Parameters
        ----------
        base_url: str
            Location of the output
        """

        # write what we have so far and clear sub chunks
        for thing in list(self._items):
            if isinstance(thing, tuple):
                field, record = thing
                self.write(
                    field,
                    record,
                    base_url=base_url,
                    storage_options=storage_options,
                )

        # gather .zmetadata from self._items and write that too
        for k in list(self._items):
            if k != ".zmetadata" and ".z" in k:
                self.zmetadata[k] = json.loads(self._items.pop(k))
        met = {"metadata": self.zmetadata, "record_size": self.record_size}
        self._items.clear()
        self._items[".zmetadata"] = json.dumps(met).encode()
        self.fs.pipe(
            "/".join([base_url or self.out_root, ".zmetadata"]),
            self._items[".zmetadata"],
        )

        # TODO: only clear those that we wrote to?
        self.open_refs.cache_clear()

    def __len__(self):
        # Caveat: This counts expected references, not actual - but is fast
        count = 0
        for field in self.listdir():
            if field.startswith("."):
                count += 1
            else:
                count += math.prod(self._get_chunk_sizes(field))
        count += len(self.zmetadata)  # all metadata keys
        # any other files not in reference partitions
        count += sum(1 for _ in self._items if not isinstance(_, tuple))
        return count

    def __iter__(self):
        # Caveat: returns only existing keys, so the number of these does not
        #  match len(self)
        metas = set(self.zmetadata)
        metas.update(self._items)
        for bit in metas:
            if isinstance(bit, str):
                yield bit
        for field in self.listdir():
            for k in self._keys_in_field(field):
                if k in self:
                    yield k

    def __contains__(self, item):
        try:
            self._load_one_key(item)
            return True
        except KeyError:
            return False

    def _keys_in_field(self, field):
        """List key names in given field

        Produces strings like "field/x.y" appropriate from the chunking of the array
        """
        chunk_sizes = self._get_chunk_sizes(field)
        if len(chunk_sizes) == 0:
            yield field + "/0"
            return
        inds = itertools.product(*(range(i) for i in chunk_sizes))
        for ind in inds:
            yield field + "/" + ".".join([str(c) for c in ind])


class ReferenceFileSystem(AsyncFileSystem):
    """View byte ranges of some other file as a file system
    Initial version: single file system target, which must support
    async, and must allow start and end args in _cat_file. Later versions
    may allow multiple arbitrary URLs for the targets.
    This FileSystem is read-only. It is designed to be used with async
    targets (for now). We do not get original file details from the target FS.
    Configuration is by passing a dict of references at init, or a URL to
    a JSON file containing the same; this dict
    can also contain concrete data for some set of paths.
    Reference dict format:
    {path0: bytes_data, path1: (target_url, offset, size)}
    https://github.com/fsspec/kerchunk/blob/main/README.md

    simple_references: if True (default), no jinja interpreting is done,
        which is the safe option.
    """

    protocol = "reference"
    cachable = False

    def __init__(
        self,
        fo,
        target=None,
        ref_storage_args=None,
        target_protocol=None,
        target_options=None,
        remote_protocol=None,
        remote_options=None,
        fs=None,
        template_overrides=None,
        simple_templates=True,
        max_gap=64_000,
        max_block=256_000_000,
        cache_size=128,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fo : dict or str
            The set of references to use for this instance, with a structure as above.
            If str referencing a JSON file, will use fsspec.open, in conjunction
            with target_options and target_protocol to open and parse JSON at this
            location. If a directory, then assume references are a set of parquet
            files to be loaded lazily.
        target : str
            For any references having target_url as None, this is the default file
            target to use
        ref_storage_args : dict
            If references is a str, use these kwargs for loading the JSON file.
            Deprecated: use target_options instead.
        target_protocol : str
            Used for loading the reference file, if it is a path. If None, protocol
            will be derived from the given path
        target_options : dict
            Extra FS options for loading the reference file ``fo``, if given as a path
        remote_protocol : str
            The protocol of the filesystem on which the references will be evaluated
            (unless fs is provided). If not given, will be derived from the first
            URL that has a protocol in the templates or in the references, in that
            order.
        remote_options : dict
            kwargs to go with remote_protocol
        fs : AbstractFileSystem | dict(str, (AbstractFileSystem | dict))
            Directly provide a file system(s):
                - a single filesystem instance
                - a dict of protocol:filesystem, where each value is either a filesystem
                  instance, or a dict of kwargs that can be used to create in
                  instance for the given protocol

            If this is given, remote_options and remote_protocol are ignored.
        template_overrides : dict
            Swap out any templates in the references file with these - useful for
            testing.
        simple_templates: bool
            Whether templates can be processed with simple replace (True) or if
            jinja  is needed (False, much slower). All reference sets produced by
            ``kerchunk`` are simple in this sense, but the spec allows for complex.
        max_gap, max_block: int
            For merging multiple concurrent requests to the same remote file.
            Neighboring byte ranges will only be merged when their
            inter-range gap is <= ``max_gap``. Default is 64KB. Set to 0
            to only merge when it requires no extra bytes. Pass a negative
            number to disable merging, appropriate for local target files.
            Neighboring byte ranges will only be merged when the size of
            the aggregated range is <= ``max_block``. Default is 256MB.
        cache_size : int
            Maximum size of LRU cache, where cache_size*record_size denotes
            the total number of references that can be loaded in memory at once.
            Only used for lazily loaded references.
        kwargs : passed to parent class
        """
        super().__init__(**kwargs)
        self.target = target
        self.template_overrides = template_overrides
        self.simple_templates = simple_templates
        self.templates = {}
        self.fss = {}
        self._dircache = {}
        self.max_gap = max_gap
        self.max_block = max_block
        if isinstance(fo, str):
            dic = dict(
                **(ref_storage_args or target_options or {}), protocol=target_protocol
            )
            ref_fs, fo2 = fsspec.core.url_to_fs(fo, **dic)
            if ".json" not in fo2 and (
                fo.endswith(("parq", "parquet", "/")) or ref_fs.isdir(fo2)
            ):
                # Lazy parquet refs
                logger.info("Open lazy reference dict from URL %s", fo)
                self.references = LazyReferenceMapper(
                    fo2,
                    fs=ref_fs,
                    cache_size=cache_size,
                )
            else:
                # text JSON
                with fsspec.open(fo, "rb", **dic) as f:
                    logger.info("Read reference from URL %s", fo)
                    text = json.load(f)
                self._process_references(text, template_overrides)
        else:
            # dictionaries
            self._process_references(fo, template_overrides)
        if isinstance(fs, dict):
            self.fss = {
                k: (
                    fsspec.filesystem(k.split(":", 1)[0], **opts)
                    if isinstance(opts, dict)
                    else opts
                )
                for k, opts in fs.items()
            }
            if None not in self.fss:
                self.fss[None] = filesystem("file")
            return
        if fs is not None:
            # single remote FS
            remote_protocol = (
                fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
            )
            self.fss[remote_protocol] = fs

        if remote_protocol is None:
            # get single protocol from any templates
            for ref in self.templates.values():
                if callable(ref):
                    ref = ref()
                protocol, _ = fsspec.core.split_protocol(ref)
                if protocol and protocol not in self.fss:
                    fs = filesystem(protocol, **(remote_options or {}))
                    self.fss[protocol] = fs
        if remote_protocol is None:
            # get single protocol from references
            # TODO: warning here, since this can be very expensive?
            for ref in self.references.values():
                if callable(ref):
                    ref = ref()
                if isinstance(ref, list) and ref[0]:
                    protocol, _ = fsspec.core.split_protocol(ref[0])
                    if protocol not in self.fss:
                        fs = filesystem(protocol, **(remote_options or {}))
                        self.fss[protocol] = fs
                        # only use first remote URL
                        break

        if remote_protocol and remote_protocol not in self.fss:
            fs = filesystem(remote_protocol, **(remote_options or {}))
            self.fss[remote_protocol] = fs

        self.fss[None] = fs or filesystem("file")  # default one
        # Wrap any non-async filesystems to ensure async methods are available below
        for k, f in self.fss.items():
            if not f.async_impl:
                self.fss[k] = AsyncFileSystemWrapper(f, asynchronous=self.asynchronous)
            elif self.asynchronous ^ f.asynchronous:
                raise ValueError(
                    "Reference-FS's target filesystem must have same value "
                    "of asynchronous"
                )

    def _cat_common(self, path, start=None, end=None):
        path = self._strip_protocol(path)
        logger.debug(f"cat: {path}")
        try:
            part = self.references[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc
        if isinstance(part, str):
            part = part.encode()
        if hasattr(part, "to_bytes"):
            part = part.to_bytes()
        if isinstance(part, bytes):
            logger.debug(f"Reference: {path}, type bytes")
            if part.startswith(b"base64:"):
                part = base64.b64decode(part[7:])
            return part, None, None

        if len(part) == 1:
            logger.debug(f"Reference: {path}, whole file => {part}")
            url = part[0]
            start1, end1 = start, end
        else:
            url, start0, size = part
            logger.debug(f"Reference: {path} => {url}, offset {start0}, size {size}")
            end0 = start0 + size

            if start is not None:
                if start >= 0:
                    start1 = start0 + start
                else:
                    start1 = end0 + start
            else:
                start1 = start0
            if end is not None:
                if end >= 0:
                    end1 = start0 + end
                else:
                    end1 = end0 + end
            else:
                end1 = end0
        if url is None:
            url = self.target
        return url, start1, end1

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path, start=start, end=end)
        if isinstance(part_or_url, bytes):
            return part_or_url[start:end]
        protocol, _ = split_protocol(part_or_url)
        try:
            return await self.fss[protocol]._cat_file(
                part_or_url, start=start0, end=end0
            )
        except Exception as e:
            raise ReferenceNotReachable(path, part_or_url) from e

    def cat_file(self, path, start=None, end=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path, start=start, end=end)
        if isinstance(part_or_url, bytes):
            return part_or_url[start:end]
        protocol, _ = split_protocol(part_or_url)
        try:
            return self.fss[protocol].cat_file(part_or_url, start=start0, end=end0)
        except Exception as e:
            raise ReferenceNotReachable(path, part_or_url) from e

    def pipe_file(self, path, value, **_):
        """Temporarily add binary data or reference as a file"""
        self.references[path] = value

    async def _get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            return os.makedirs(lpath, exist_ok=True)
        data = await self._cat_file(rpath)
        with open(lpath, "wb") as f:
            f.write(data)

    def get_file(self, rpath, lpath, callback=DEFAULT_CALLBACK, **kwargs):
        if self.isdir(rpath):
            return os.makedirs(lpath, exist_ok=True)
        data = self.cat_file(rpath, **kwargs)
        callback.set_size(len(data))
        if isfilelike(lpath):
            lpath.write(data)
        else:
            with open(lpath, "wb") as f:
                f.write(data)
        callback.absolute_update(len(data))

    def get(self, rpath, lpath, recursive=False, **kwargs):
        if recursive:
            # trigger directory build
            self.ls("")
        rpath = self.expand_path(rpath, recursive=recursive)
        fs = fsspec.filesystem("file", auto_mkdir=True)
        targets = other_paths(rpath, lpath)
        if recursive:
            data = self.cat([r for r in rpath if not self.isdir(r)])
        else:
            data = self.cat(rpath)
        for remote, local in zip(rpath, targets):
            if remote in data:
                fs.pipe_file(local, data[remote])

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        if isinstance(path, str) and recursive:
            raise NotImplementedError
        if isinstance(path, list) and (recursive or any("*" in p for p in path)):
            raise NotImplementedError
        # TODO: if references is lazy, pre-fetch all paths in batch before access
        proto_dict = _protocol_groups(path, self.references)
        out = {}
        for proto, paths in proto_dict.items():
            fs = self.fss[proto]
            urls, starts, ends, valid_paths = [], [], [], []
            for p in paths:
                # find references or label not-found. Early exit if any not
                # found and on_error is "raise"
                try:
                    u, s, e = self._cat_common(p)
                    if not isinstance(u, (bytes, str)):
                        # nan/None from parquet
                        continue
                except FileNotFoundError as err:
                    if on_error == "raise":
                        raise
                    if on_error != "omit":
                        out[p] = err
                else:
                    urls.append(u)
                    starts.append(s)
                    ends.append(e)
                    valid_paths.append(p)

            # process references into form for merging
            urls2 = []
            starts2 = []
            ends2 = []
            paths2 = []
            whole_files = set()
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                if isinstance(u, bytes):
                    # data
                    out[p] = u
                elif s is None:
                    # whole file - limits are None, None, but no further
                    # entries take for this file
                    whole_files.add(u)
                    urls2.append(u)
                    starts2.append(s)
                    ends2.append(e)
                    paths2.append(p)
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                # second run to account for files that are to be loaded whole
                if s is not None and u not in whole_files:
                    urls2.append(u)
                    starts2.append(s)
                    ends2.append(e)
                    paths2.append(p)

            # merge and fetch consolidated ranges
            new_paths, new_starts, new_ends = merge_offset_ranges(
                list(urls2),
                list(starts2),
                list(ends2),
                sort=True,
                max_gap=self.max_gap,
                max_block=self.max_block,
            )
            bytes_out = fs.cat_ranges(new_paths, new_starts, new_ends)

            # unbundle from merged bytes - simple approach
            for u, s, e, p in zip(urls, starts, ends, valid_paths):
                if p in out:
                    continue  # was bytes, already handled
                for np, ns, ne, b in zip(new_paths, new_starts, new_ends, bytes_out):
                    if np == u and (ns is None or ne is None):
                        if isinstance(b, Exception):
                            out[p] = b
                        else:
                            out[p] = b[s:e]
                    elif np == u and s >= ns and e <= ne:
                        if isinstance(b, Exception):
                            out[p] = b
                        else:
                            out[p] = b[s - ns : (e - ne) or None]

        for k, v in out.copy().items():
            # these were valid references, but fetch failed, so transform exc
            if isinstance(v, Exception) and k in self.references:
                ex = out[k]
                new_ex = ReferenceNotReachable(k, self.references[k])
                new_ex.__cause__ = ex
                if on_error == "raise":
                    raise new_ex
                elif on_error != "omit":
                    out[k] = new_ex

        if len(out) == 1 and isinstance(path, str) and "*" not in path:
            return _first(out)
        return out

    def _process_references(self, references, template_overrides=None):
        vers = references.get("version", None)
        if vers is None:
            self._process_references0(references)
        elif vers == 1:
            self._process_references1(references, template_overrides=template_overrides)
        else:
            raise ValueError(f"Unknown reference spec version: {vers}")
        # TODO: we make dircache by iterating over all entries, but for Spec >= 1,
        #  can replace with programmatic. Is it even needed for mapper interface?

    def _process_references0(self, references):
        """Make reference dict for Spec Version 0"""
        if isinstance(references, dict):
            # do not do this for lazy/parquet backend, which will not make dicts,
            # but must remain writable in the original object
            references = {
                key: json.dumps(val) if isinstance(val, dict) else val
                for key, val in references.items()
            }
        self.references = references

    def _process_references1(self, references, template_overrides=None):
        if not self.simple_templates or self.templates:
            import jinja2.sandbox
        self.references = {}
        self._process_templates(references.get("templates", {}))

        @lru_cache(1000)
        def _render_jinja(u):
            return (
                jinja2.sandbox.SandboxedEnvironment()
                .from_string(u)
                .render(**self.templates)
            )

        for k, v in references.get("refs", {}).items():
            if isinstance(v, str):
                if v.startswith("base64:"):
                    self.references[k] = base64.b64decode(v[7:])
                self.references[k] = v
            elif isinstance(v, dict):
                self.references[k] = json.dumps(v)
            elif self.templates:
                u = v[0]
                if "{{" in u:
                    if self.simple_templates:
                        u = (
                            u.replace("{{", "{")
                            .replace("}}", "}")
                            .format(**self.templates)
                        )
                    else:
                        u = _render_jinja(u)
                self.references[k] = [u] if len(v) == 1 else [u, v[1], v[2]]
            else:
                self.references[k] = v
        self.references.update(self._process_gen(references.get("gen", [])))

    def _process_templates(self, tmp):
        self.templates = {}
        if self.template_overrides is not None:
            tmp.update(self.template_overrides)
        for k, v in tmp.items():
            if "{{" in v:
                import jinja2.sandbox

                self.templates[k] = (
                    lambda temp=v, **kwargs: jinja2.sandbox.SandboxedEnvironment()
                    .from_string(temp)
                    .render(**kwargs)
                )
            else:
                self.templates[k] = v

    def _process_gen(self, gens):
        out = {}
        if self.simple_templates:
            return out
        for gen in gens:
            dimension = {
                k: (
                    v
                    if isinstance(v, list)
                    else range(v.get("start", 0), v["stop"], v.get("step", 1))
                )
                for k, v in gen["dimensions"].items()
            }
            products = (
                dict(zip(dimension.keys(), values))
                for values in itertools.product(*dimension.values())
            )
            for pr in products:
                import jinja2.sandbox

                key = (
                    jinja2.sandbox.SandboxedEnvironment()
                    .from_string(gen["key"])
                    .render(**pr, **self.templates)
                )
                url = (
                    jinja2.sandbox.SandboxedEnvironment()
                    .from_string(gen["url"])
                    .render(**pr, **self.templates)
                )
                if ("offset" in gen) and ("length" in gen):
                    offset = int(
                        jinja2.sandbox.SandboxedEnvironment()
                        .from_string(gen["offset"])
                        .render(**pr, **self.templates)
                    )
                    length = int(
                        jinja2.sandbox.SandboxedEnvironment()
                        .from_string(gen["length"])
                        .render(**pr, **self.templates)
                    )
                    out[key] = [url, offset, length]
                elif ("offset" in gen) ^ ("length" in gen):
                    raise ValueError(
                        "Both 'offset' and 'length' are required for a "
                        "reference generator entry if either is provided."
                    )
                else:
                    out[key] = [url]
        return out

    def _dircache_from_items(self):
        self.dircache = {"": []}
        it = self.references.items()
        for path, part in it:
            if isinstance(part, (bytes, str)) or hasattr(part, "to_bytes"):
                size = len(part)
            elif len(part) == 1:
                size = None
            else:
                _, _, size = part
            par = path.rsplit("/", 1)[0] if "/" in path else ""
            par0 = par
            subdirs = [par0]
            while par0 and par0 not in self.dircache:
                # collect parent directories
                par0 = self._parent(par0)
                subdirs.append(par0)

            subdirs.reverse()
            for parent, child in zip(subdirs, subdirs[1:]):
                # register newly discovered directories
                assert child not in self.dircache
                assert parent in self.dircache
                self.dircache[parent].append(
                    {"name": child, "type": "directory", "size": 0}
                )
                self.dircache[child] = []

            self.dircache[par].append({"name": path, "type": "file", "size": size})

    def _open(self, path, mode="rb", block_size=None, cache_options=None, **kwargs):
        part_or_url, start0, end0 = self._cat_common(path)
        # This logic is kept outside `ReferenceFile` to avoid unnecessary redirection.
        # That does mean `_cat_common` gets called twice if it eventually reaches `ReferenceFile`.
        if isinstance(part_or_url, bytes):
            return io.BytesIO(part_or_url[start0:end0])

        protocol, _ = split_protocol(part_or_url)
        if start0 is None and end0 is None:
            return self.fss[protocol]._open(
                part_or_url,
                mode,
                block_size=block_size,
                cache_options=cache_options,
                **kwargs,
            )

        return ReferenceFile(
            self,
            path,
            mode,
            block_size=block_size,
            cache_options=cache_options,
            **kwargs,
        )

    def ls(self, path, detail=True, **kwargs):
        logger.debug("list %s", path)
        path = self._strip_protocol(path)
        if isinstance(self.references, LazyReferenceMapper):
            try:
                return self.references.ls(path, detail)
            except KeyError:
                pass
            raise FileNotFoundError(f"'{path}' is not a known key")
        if not self.dircache:
            self._dircache_from_items()
        out = self._ls_from_cache(path)
        if out is None:
            raise FileNotFoundError(path)
        if detail:
            return out
        return [o["name"] for o in out]

    def exists(self, path, **kwargs):  # overwrite auto-sync version
        return self.isdir(path) or self.isfile(path)

    def isdir(self, path):  # overwrite auto-sync version
        if self.dircache:
            return path in self.dircache
        elif isinstance(self.references, LazyReferenceMapper):
            return path in self.references.listdir()
        else:
            # this may be faster than building dircache for single calls, but
            # by looping will be slow for many calls; could cache it?
            return any(_.startswith(f"{path}/") for _ in self.references)

    def isfile(self, path):  # overwrite auto-sync version
        return path in self.references

    async def _ls(self, path, detail=True, **kwargs):  # calls fast sync code
        return self.ls(path, detail, **kwargs)

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        if withdirs:
            return super().find(
                path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs
            )
        if path:
            path = self._strip_protocol(path)
            r = sorted(k for k in self.references if k.startswith(path))
        else:
            r = sorted(self.references)
        if detail:
            if not self.dircache:
                self._dircache_from_items()
            return {k: self._ls_from_cache(k)[0] for k in r}
        else:
            return r

    def info(self, path, **kwargs):
        out = self.references.get(path)
        if out is not None:
            if isinstance(out, (str, bytes)):
                # decode base64 here
                return {"name": path, "type": "file", "size": len(out)}
            elif len(out) > 1:
                return {"name": path, "type": "file", "size": out[2]}
            else:
                out0 = [{"name": path, "type": "file", "size": None}]
        else:
            out = self.ls(path, True)
            out0 = [o for o in out if o["name"] == path]
            if not out0:
                return {"name": path, "type": "directory", "size": 0}
        if out0[0]["size"] is None:
            # if this is a whole remote file, update size using remote FS
            prot, _ = split_protocol(self.references[path][0])
            out0[0]["size"] = self.fss[prot].size(self.references[path][0])
        return out0[0]

    async def _info(self, path, **kwargs):  # calls fast sync code
        return self.info(path)

    async def _rm_file(self, path, **kwargs):
        self.references.pop(
            path, None
        )  # ignores FileNotFound, just as well for directories
        self.dircache.clear()  # this is a bit heavy handed

    async def _pipe_file(self, path, data, mode="overwrite", **kwargs):
        if mode == "create" and self.exists(path):
            raise FileExistsError
        # can be str or bytes
        self.references[path] = data
        self.dircache.clear()  # this is a bit heavy handed

    async def _put_file(self, lpath, rpath, mode="overwrite", **kwargs):
        # puts binary
        if mode == "create" and self.exists(rpath):
            raise FileExistsError
        with open(lpath, "rb") as f:
            self.references[rpath] = f.read()
        self.dircache.clear()  # this is a bit heavy handed

    def save_json(self, url, **storage_options):
        """Write modified references into new location"""
        out = {}
        for k, v in self.references.items():
            if isinstance(v, bytes):
                try:
                    out[k] = v.decode("ascii")
                except UnicodeDecodeError:
                    out[k] = (b"base64:" + base64.b64encode(v)).decode()
            else:
                out[k] = v
        with fsspec.open(url, "wb", **storage_options) as f:
            f.write(json.dumps({"version": 1, "refs": out}).encode())


class ReferenceFile(AbstractBufferedFile):
    def __init__(
        self,
        fs,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        super().__init__(
            fs,
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            size=size,
            cache_type=cache_type,
            cache_options=cache_options,
            **kwargs,
        )
        part_or_url, self.start, self.end = self.fs._cat_common(self.path)
        protocol, _ = split_protocol(part_or_url)
        self.src_fs = self.fs.fss[protocol]
        self.src_path = part_or_url
        self._f = None

    @property
    def f(self):
        if self._f is None or self._f.closed:
            self._f = self.src_fs._open(
                self.src_path,
                mode=self.mode,
                block_size=self.blocksize,
                autocommit=self.autocommit,
                cache_type="none",
                **self.kwargs,
            )
        return self._f

    def close(self):
        if self._f is not None:
            self._f.close()
        return super().close()

    def _fetch_range(self, start, end):
        start = start + self.start
        end = min(end + self.start, self.end)
        self.f.seek(start)
        return self.f.read(end - start)
<|endoftext|>import errno
import io
import os
import secrets
import shutil
from contextlib import suppress
from functools import cached_property, wraps
from urllib.parse import parse_qs

from fsspec.spec import AbstractFileSystem
from fsspec.utils import (
    get_package_version_without_import,
    infer_storage_options,
    mirror_from,
    tokenize,
)


def wrap_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as exception:
            if not exception.args:
                raise

            message, *args = exception.args
            if isinstance(message, str) and "does not exist" in message:
                raise FileNotFoundError(errno.ENOENT, message) from exception
            else:
                raise

    return wrapper


PYARROW_VERSION = None
_EMPTY_ROOT_MARKER_FILESYSTEMS = {"gcs", "s3"}


class ArrowFSWrapper(AbstractFileSystem):
    """FSSpec-compatible wrapper of pyarrow.fs.FileSystem.

    Parameters
    ----------
    fs : pyarrow.fs.FileSystem

    """

    root_marker = "/"

    def __init__(self, fs, **kwargs):
        global PYARROW_VERSION
        PYARROW_VERSION = get_package_version_without_import("pyarrow")
        self.fs = fs
        if fs.type_name in _EMPTY_ROOT_MARKER_FILESYSTEMS:
            self.root_marker = ""
        super().__init__(**kwargs)

    @property
    def protocol(self):
        return self.fs.type_name

    @cached_property
    def fsid(self):
        return "hdfs_" + tokenize(self.fs.host, self.fs.port)

    def _parent(self, path):
        if self.root_marker:
            return super()._parent(path)

        path = self._strip_protocol(path).lstrip("/")
        return path.rsplit("/", 1)[0] if "/" in path else ""

    @classmethod
    def _strip_protocol(cls, path):
        ops = infer_storage_options(path)
        path = ops["path"]
        if path.startswith("//"):
            # special case for "hdfs://path" (without the triple slash)
            path = path[1:]
        return path

    def ls(self, path, detail=False, **kwargs):
        path = self._strip_protocol(path)
        from pyarrow.fs import FileSelector

        try:
            entries = [
                self._make_entry(entry)
                for entry in self.fs.get_file_info(FileSelector(path))
            ]
        except (FileNotFoundError, NotADirectoryError):
            entries = [self.info(path, **kwargs)]
        if detail:
            return entries
        else:
            return [entry["name"] for entry in entries]

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        [info] = self.fs.get_file_info([path])
        return self._make_entry(info)

    def exists(self, path):
        path = self._strip_protocol(path)
        try:
            self.info(path)
        except FileNotFoundError:
            return False
        else:
            return True

    def _make_entry(self, info):
        from pyarrow.fs import FileType

        if info.type is FileType.Directory:
            kind = "directory"
        elif info.type is FileType.File:
            kind = "file"
        elif info.type is FileType.NotFound:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), info.path)
        else:
            kind = "other"

        return {
            "name": info.path,
            "size": info.size,
            "type": kind,
            "mtime": info.mtime,
        }

    @wrap_exceptions
    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip("/")
        path2 = self._strip_protocol(path2).rstrip("/")

        with self._open(path1, "rb") as lstream:
            tmp_fname = f"{path2}.tmp.{secrets.token_hex(6)}"
            try:
                with self.open(tmp_fname, "wb") as rstream:
                    shutil.copyfileobj(lstream, rstream)
                self.fs.move(tmp_fname, path2)
            except BaseException:
                with suppress(FileNotFoundError):
                    self.fs.delete_file(tmp_fname)
                raise

    @wrap_exceptions
    def mv(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip("/")
        path2 = self._strip_protocol(path2).rstrip("/")
        self.fs.move(path1, path2)

    @wrap_exceptions
    def rm_file(self, path):
        path = self._strip_protocol(path)
        self.fs.delete_file(path)

    @wrap_exceptions
    def rm(self, path, recursive=False, maxdepth=None):
        path = self._strip_protocol(path).rstrip("/")
        if self.isdir(path):
            if recursive:
                self.fs.delete_dir(path)
            else:
                raise ValueError("Can't delete directories without recursive=False")
        else:
            self.fs.delete_file(path)

    @wrap_exceptions
    def _open(self, path, mode="rb", block_size=None, seekable=True, **kwargs):
        if mode == "rb":
            if seekable:
                method = self.fs.open_input_file
            else:
                method = self.fs.open_input_stream
        elif mode == "wb":
            method = self.fs.open_output_stream
        elif mode == "ab":
            method = self.fs.open_append_stream
        else:
            raise ValueError(f"unsupported mode for Arrow filesystem: {mode!r}")

        _kwargs = {}
        if mode != "rb" or not seekable:
            if int(PYARROW_VERSION.split(".")[0]) >= 4:
                # disable compression auto-detection
                _kwargs["compression"] = None
        stream = method(path, **_kwargs)

        return ArrowFile(self, stream, path, mode, block_size, **kwargs)

    @wrap_exceptions
    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if create_parents:
            self.makedirs(path, exist_ok=True)
        else:
            self.fs.create_dir(path, recursive=False)

    @wrap_exceptions
    def makedirs(self, path, exist_ok=False):
        path = self._strip_protocol(path)
        self.fs.create_dir(path, recursive=True)

    @wrap_exceptions
    def rmdir(self, path):
        path = self._strip_protocol(path)
        self.fs.delete_dir(path)

    @wrap_exceptions
    def modified(self, path):
        path = self._strip_protocol(path)
        return self.fs.get_file_info(path).mtime

    def cat_file(self, path, start=None, end=None, **kwargs):
        kwargs.setdefault("seekable", start not in [None, 0])
        return super().cat_file(path, start, end, **kwargs)

    def get_file(self, rpath, lpath, **kwargs):
        kwargs.setdefault("seekable", False)
        super().get_file(rpath, lpath, **kwargs)


@mirror_from(
    "stream",
    [
        "read",
        "seek",
        "tell",
        "write",
        "readable",
        "writable",
        "close",
        "seekable",
    ],
)
class ArrowFile(io.IOBase):
    def __init__(self, fs, stream, path, mode, block_size=None, **kwargs):
        self.path = path
        self.mode = mode

        self.fs = fs
        self.stream = stream

        self.blocksize = self.block_size = block_size
        self.kwargs = kwargs

    def __enter__(self):
        return self

    @property
    def size(self):
        if self.stream.seekable():
            return self.stream.size()
        return None

    def __exit__(self, *args):
        return self.close()


class HadoopFileSystem(ArrowFSWrapper):
    """A wrapper on top of the pyarrow.fs.HadoopFileSystem
    to connect it's interface with fsspec"""

    protocol = "hdfs"

    def __init__(
        self,
        host="default",
        port=0,
        user=None,
        kerb_ticket=None,
        replication=3,
        extra_conf=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        host: str
            Hostname, IP or "default" to try to read from Hadoop config
        port: int
            Port to connect on, or default from Hadoop config if 0
        user: str or None
            If given, connect as this username
        kerb_ticket: str or None
            If given, use this ticket for authentication
        replication: int
            set replication factor of file for write operations. default value is 3.
        extra_conf: None or dict
            Passed on to HadoopFileSystem
        """
        from pyarrow.fs import HadoopFileSystem

        fs = HadoopFileSystem(
            host=host,
            port=port,
            user=user,
            kerb_ticket=kerb_ticket,
            replication=replication,
            extra_conf=extra_conf,
        )
        super().__init__(fs=fs, **kwargs)

    @staticmethod
    def _get_kwargs_from_urls(path):
        ops = infer_storage_options(path)
        out = {}
        if ops.get("host", None):
            out["host"] = ops["host"]
        if ops.get("username", None):
            out["user"] = ops["username"]
        if ops.get("port", None):
            out["port"] = ops["port"]
        if ops.get("url_query", None):
            queries = parse_qs(ops["url_query"])
            if queries.get("replication", None):
                out["replication"] = int(queries["replication"][0])
        return out
<|endoftext|>import os
import zipfile

import fsspec
from fsspec.archive import AbstractArchiveFileSystem


class ZipFileSystem(AbstractArchiveFileSystem):
    """Read/Write contents of ZIP archive as a file-system

    Keeps file object open while instance lives.

    This class is pickleable, but not necessarily thread-safe
    """

    root_marker = ""
    protocol = "zip"
    cachable = False

    def __init__(
        self,
        fo="",
        mode="r",
        target_protocol=None,
        target_options=None,
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        compresslevel=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fo: str or file-like
            Contains ZIP, and must exist. If a str, will fetch file using
            :meth:`~fsspec.open_files`, which must return one file exactly.
        mode: str
            Accept: "r", "w", "a"
        target_protocol: str (optional)
            If ``fo`` is a string, this value can be used to override the
            FS protocol inferred from a URL
        target_options: dict (optional)
            Kwargs passed when instantiating the target FS, if ``fo`` is
            a string.
        compression, allowZip64, compresslevel: passed to ZipFile
            Only relevant when creating a ZIP
        """
        super().__init__(self, **kwargs)
        if mode not in set("rwa"):
            raise ValueError(f"mode '{mode}' no understood")
        self.mode = mode
        if isinstance(fo, (str, os.PathLike)):
            if mode == "a":
                m = "r+b"
            else:
                m = mode + "b"
            fo = fsspec.open(
                fo, mode=m, protocol=target_protocol, **(target_options or {})
            )
        self.force_zip_64 = allowZip64
        self.of = fo
        self.fo = fo.__enter__()  # the whole instance is a context
        self.zip = zipfile.ZipFile(
            self.fo,
            mode=mode,
            compression=compression,
            allowZip64=allowZip64,
            compresslevel=compresslevel,
        )
        self.dir_cache = None

    @classmethod
    def _strip_protocol(cls, path):
        # zip file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def __del__(self):
        if hasattr(self, "zip"):
            self.close()
            del self.zip
        if hasattr(self, "of") and hasattr(self.of, "__exit__"):
            self.of.__exit__(None, None, None)

    def close(self):
        """Commits any write changes to the file. Done on ``del`` too."""
        self.zip.close()

    def _get_dirs(self):
        if self.dir_cache is None or self.mode in set("wa"):
            # when writing, dir_cache is always in the ZipFile's attributes,
            # not read from the file.
            files = self.zip.infolist()
            self.dir_cache = {
                dirname.rstrip("/"): {
                    "name": dirname.rstrip("/"),
                    "size": 0,
                    "type": "directory",
                }
                for dirname in self._all_dirnames(self.zip.namelist())
            }
            for z in files:
                f = {s: getattr(z, s, None) for s in zipfile.ZipInfo.__slots__}
                f.update(
                    {
                        "name": z.filename.rstrip("/"),
                        "size": z.file_size,
                        "type": ("directory" if z.is_dir() else "file"),
                    }
                )
                self.dir_cache[f["name"]] = f

    def pipe_file(self, path, value, **kwargs):
        # override upstream, because we know the exact file size in this case
        self.zip.writestr(path, value, **kwargs)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if "r" in mode and self.mode in set("wa"):
            if self.exists(path):
                raise OSError("ZipFS can only be open for reading or writing, not both")
            raise FileNotFoundError(path)
        if "r" in self.mode and "w" in mode:
            raise OSError("ZipFS can only be open for reading or writing, not both")
        out = self.zip.open(path, mode.strip("b"), force_zip64=self.force_zip_64)
        if "r" in mode:
            info = self.info(path)
            out.size = info["size"]
            out.name = info["name"]
        return out

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        def to_parts(_path: str):
            return list(filter(None, _path.replace("\\", "/").split("/")))

        if not isinstance(path, str):
            path = str(path)

        # Remove the leading slash, as the zip file paths are always
        # given without a leading slash
        path = path.lstrip("/")
        path_parts = to_parts(path)
        path_depth = len(path_parts)

        self._get_dirs()

        result = {}
        # To match posix find, if an exact file name is given, we should
        # return only that file
        if path in self.dir_cache and self.dir_cache[path]["type"] == "file":
            result[path] = self.dir_cache[path]
            return result if detail else [path]

        for file_path, file_info in self.dir_cache.items():
            if len(file_parts := to_parts(file_path)) < path_depth or any(
                a != b for a, b in zip(path_parts, file_parts)
            ):
                # skip parent folders and mismatching paths
                continue

            if file_info["type"] == "directory":
                if withdirs and file_path not in result:
                    result[file_path.strip("/")] = file_info
                continue

            if file_path not in result:
                result[file_path] = file_info if detail else None

        if maxdepth:
            result = {
                k: v for k, v in result.items() if k.count("/") < maxdepth + path_depth
            }
        return result if detail else sorted(result)
<|endoftext|>import dask
from distributed.client import Client, _get_global_client
from distributed.worker import Worker

from fsspec import filesystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import infer_storage_options


def _get_client(client):
    if client is None:
        return _get_global_client()
    elif isinstance(client, Client):
        return client
    else:
        # e.g., connection string
        return Client(client)


def _in_worker():
    return bool(Worker._instances)


class DaskWorkerFileSystem(AbstractFileSystem):
    """View files accessible to a worker as any other remote file-system

    When instances are run on the worker, uses the real filesystem. When
    run on the client, they call the worker to provide information or data.

    **Warning** this implementation is experimental, and read-only for now.
    """

    def __init__(
        self, target_protocol=None, target_options=None, fs=None, client=None, **kwargs
    ):
        super().__init__(**kwargs)
        if not (fs is None) ^ (target_protocol is None):
            raise ValueError(
                "Please provide one of filesystem instance (fs) or"
                " target_protocol, not both"
            )
        self.target_protocol = target_protocol
        self.target_options = target_options
        self.worker = None
        self.client = client
        self.fs = fs
        self._determine_worker()

    @staticmethod
    def _get_kwargs_from_urls(path):
        so = infer_storage_options(path)
        if "host" in so and "port" in so:
            return {"client": f"{so['host']}:{so['port']}"}
        else:
            return {}

    def _determine_worker(self):
        if _in_worker():
            self.worker = True
            if self.fs is None:
                self.fs = filesystem(
                    self.target_protocol, **(self.target_options or {})
                )
        else:
            self.worker = False
            self.client = _get_client(self.client)
            self.rfs = dask.delayed(self)

    def mkdir(self, *args, **kwargs):
        if self.worker:
            self.fs.mkdir(*args, **kwargs)
        else:
            self.rfs.mkdir(*args, **kwargs).compute()

    def rm(self, *args, **kwargs):
        if self.worker:
            self.fs.rm(*args, **kwargs)
        else:
            self.rfs.rm(*args, **kwargs).compute()

    def copy(self, *args, **kwargs):
        if self.worker:
            self.fs.copy(*args, **kwargs)
        else:
            self.rfs.copy(*args, **kwargs).compute()

    def mv(self, *args, **kwargs):
        if self.worker:
            self.fs.mv(*args, **kwargs)
        else:
            self.rfs.mv(*args, **kwargs).compute()

    def ls(self, *args, **kwargs):
        if self.worker:
            return self.fs.ls(*args, **kwargs)
        else:
            return self.rfs.ls(*args, **kwargs).compute()

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        if self.worker:
            return self.fs._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )
        else:
            return DaskFile(
                fs=self,
                path=path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )

    def fetch_range(self, path, mode, start, end):
        if self.worker:
            with self._open(path, mode) as f:
                f.seek(start)
                return f.read(end - start)
        else:
            return self.rfs.fetch_range(path, mode, start, end).compute()


class DaskFile(AbstractBufferedFile):
    def __init__(self, mode="rb", **kwargs):
        if mode != "rb":
            raise ValueError('Remote dask files can only be opened in "rb" mode')
        super().__init__(**kwargs)

    def _upload_chunk(self, final=False):
        pass

    def _initiate_upload(self):
        """Create remote file/upload"""
        pass

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        return self.fs.fetch_range(self.path, self.mode, start, end)
<|endoftext|>from __future__ import annotations

import base64
import urllib

import requests
from requests.adapters import HTTPAdapter, Retry
from typing_extensions import override

from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile


class DatabricksException(Exception):
    """
    Helper class for exceptions raised in this module.
    """

    def __init__(self, error_code, message, details=None):
        """Create a new DatabricksException"""
        super().__init__(message)

        self.error_code = error_code
        self.message = message
        self.details = details


class DatabricksFileSystem(AbstractFileSystem):
    """
    Get access to the Databricks filesystem implementation over HTTP.
    Can be used inside and outside of a databricks cluster.
    """

    def __init__(self, instance, token, **kwargs):
        """
        Create a new DatabricksFileSystem.

        Parameters
        ----------
        instance: str
            The instance URL of the databricks cluster.
            For example for an Azure databricks cluster, this
            has the form adb-<some-number>.<two digits>.azuredatabricks.net.
        token: str
            Your personal token. Find out more
            here: https://docs.databricks.com/dev-tools/api/latest/authentication.html
        """
        self.instance = instance
        self.token = token
        self.session = requests.Session()
        self.retries = Retry(
            total=10,
            backoff_factor=0.05,
            status_forcelist=[408, 429, 500, 502, 503, 504],
        )

        self.session.mount("https://", HTTPAdapter(max_retries=self.retries))
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

        super().__init__(**kwargs)

    @override
    def _ls_from_cache(self, path) -> list[dict[str, str | int]] | None:
        """Check cache for listing

        Returns listing, if found (may be empty list for a directory that
        exists but contains nothing), None if not in cache.
        """
        self.dircache.pop(path.rstrip("/"), None)

        parent = self._parent(path)
        if parent in self.dircache:
            for entry in self.dircache[parent]:
                if entry["name"] == path.rstrip("/"):
                    if entry["type"] != "directory":
                        return [entry]
                    return []
            raise FileNotFoundError(path)

    def ls(self, path, detail=True, **kwargs):
        """
        List the contents of the given path.

        Parameters
        ----------
        path: str
            Absolute path
        detail: bool
            Return not only the list of filenames,
            but also additional information on file sizes
            and types.
        """
        try:
            out = self._ls_from_cache(path)
        except FileNotFoundError:
            # This happens if the `path`'s parent was cached, but `path` is not
            # there. This suggests that `path` is new since the parent was
            # cached. Attempt to invalidate parent's cache before continuing.
            self.dircache.pop(self._parent(path), None)
            out = None

        if not out:
            try:
                r = self._send_to_api(
                    method="get", endpoint="list", json={"path": path}
                )
            except DatabricksException as e:
                if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                    raise FileNotFoundError(e.message) from e

                raise
            files = r.get("files", [])
            out = [
                {
                    "name": o["path"],
                    "type": "directory" if o["is_dir"] else "file",
                    "size": o["file_size"],
                }
                for o in files
            ]
            self.dircache[path] = out

        if detail:
            return out
        return [o["name"] for o in out]

    def makedirs(self, path, exist_ok=True):
        """
        Create a given absolute path and all of its parents.

        Parameters
        ----------
        path: str
            Absolute path to create
        exist_ok: bool
            If false, checks if the folder
            exists before creating it (and raises an
            Exception if this is the case)
        """
        if not exist_ok:
            try:
                # If the following succeeds, the path is already present
                self._send_to_api(
                    method="get", endpoint="get-status", json={"path": path}
                )
                raise FileExistsError(f"Path {path} already exists")
            except DatabricksException as e:
                if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                    pass

        try:
            self._send_to_api(method="post", endpoint="mkdirs", json={"path": path})
        except DatabricksException as e:
            if e.error_code == "RESOURCE_ALREADY_EXISTS":
                raise FileExistsError(e.message) from e

            raise
        self.invalidate_cache(self._parent(path))

    def mkdir(self, path, create_parents=True, **kwargs):
        """
        Create a given absolute path and all of its parents.

        Parameters
        ----------
        path: str
            Absolute path to create
        create_parents: bool
            Whether to create all parents or not.
            "False" is not implemented so far.
        """
        if not create_parents:
            raise NotImplementedError

        self.mkdirs(path, **kwargs)

    def rm(self, path, recursive=False, **kwargs):
        """
        Remove the file or folder at the given absolute path.

        Parameters
        ----------
        path: str
            Absolute path what to remove
        recursive: bool
            Recursively delete all files in a folder.
        """
        try:
            self._send_to_api(
                method="post",
                endpoint="delete",
                json={"path": path, "recursive": recursive},
            )
        except DatabricksException as e:
            # This is not really an exception, it just means
            # not everything was deleted so far
            if e.error_code == "PARTIAL_DELETE":
                self.rm(path=path, recursive=recursive)
            elif e.error_code == "IO_ERROR":
                # Using the same exception as the os module would use here
                raise OSError(e.message) from e

            raise
        self.invalidate_cache(self._parent(path))

    def mv(
        self, source_path, destination_path, recursive=False, maxdepth=None, **kwargs
    ):
        """
        Move a source to a destination path.

        A note from the original [databricks API manual]
        (https://docs.databricks.com/dev-tools/api/latest/dbfs.html#move).

        When moving a large number of files the API call will time out after
        approximately 60s, potentially resulting in partially moved data.
        Therefore, for operations that move more than 10k files, we strongly
        discourage using the DBFS REST API.

        Parameters
        ----------
        source_path: str
            From where to move (absolute path)
        destination_path: str
            To where to move (absolute path)
        recursive: bool
            Not implemented to far.
        maxdepth:
            Not implemented to far.
        """
        if recursive:
            raise NotImplementedError
        if maxdepth:
            raise NotImplementedError

        try:
            self._send_to_api(
                method="post",
                endpoint="move",
                json={"source_path": source_path, "destination_path": destination_path},
            )
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e
            elif e.error_code == "RESOURCE_ALREADY_EXISTS":
                raise FileExistsError(e.message) from e

            raise
        self.invalidate_cache(self._parent(source_path))
        self.invalidate_cache(self._parent(destination_path))

    def _open(self, path, mode="rb", block_size="default", **kwargs):
        """
        Overwrite the base class method to make sure to create a DBFile.
        All arguments are copied from the base method.

        Only the default blocksize is allowed.
        """
        return DatabricksFile(self, path, mode=mode, block_size=block_size, **kwargs)

    def _send_to_api(self, method, endpoint, json):
        """
        Send the given json to the DBFS API
        using a get or post request (specified by the argument `method`).

        Parameters
        ----------
        method: str
            Which http method to use for communication; "get" or "post".
        endpoint: str
            Where to send the request to (last part of the API URL)
        json: dict
            Dictionary of information to send
        """
        if method == "post":
            session_call = self.session.post
        elif method == "get":
            session_call = self.session.get
        else:
            raise ValueError(f"Do not understand method {method}")

        url = urllib.parse.urljoin(f"https://{self.instance}/api/2.0/dbfs/", endpoint)

        r = session_call(url, json=json)

        # The DBFS API will return a json, also in case of an exception.
        # We want to preserve this information as good as possible.
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # try to extract json error message
            # if that fails, fall back to the original exception
            try:
                exception_json = e.response.json()
            except Exception:
                raise e from None

            raise DatabricksException(**exception_json) from e

        return r.json()

    def _create_handle(self, path, overwrite=True):
        """
        Internal function to create a handle, which can be used to
        write blocks of a file to DBFS.
        A handle has a unique identifier which needs to be passed
        whenever written during this transaction.
        The handle is active for 10 minutes - after that a new
        write transaction needs to be created.
        Make sure to close the handle after you are finished.

        Parameters
        ----------
        path: str
            Absolute path for this file.
        overwrite: bool
            If a file already exist at this location, either overwrite
            it or raise an exception.
        """
        try:
            r = self._send_to_api(
                method="post",
                endpoint="create",
                json={"path": path, "overwrite": overwrite},
            )
            return r["handle"]
        except DatabricksException as e:
            if e.error_code == "RESOURCE_ALREADY_EXISTS":
                raise FileExistsError(e.message) from e

            raise

    def _close_handle(self, handle):
        """
        Close a handle, which was opened by :func:`_create_handle`.

        Parameters
        ----------
        handle: str
            Which handle to close.
        """
        try:
            self._send_to_api(method="post", endpoint="close", json={"handle": handle})
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e

            raise

    def _add_data(self, handle, data):
        """
        Upload data to an already opened file handle
        (opened by :func:`_create_handle`).
        The maximal allowed data size is 1MB after
        conversion to base64.
        Remember to close the handle when you are finished.

        Parameters
        ----------
        handle: str
            Which handle to upload data to.
        data: bytes
            Block of data to add to the handle.
        """
        data = base64.b64encode(data).decode()
        try:
            self._send_to_api(
                method="post",
                endpoint="add-block",
                json={"handle": handle, "data": data},
            )
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e
            elif e.error_code == "MAX_BLOCK_SIZE_EXCEEDED":
                raise ValueError(e.message) from e

            raise

    def _get_data(self, path, start, end):
        """
        Download data in bytes from a given absolute path in a block
        from [start, start+length].
        The maximum number of allowed bytes to read is 1MB.

        Parameters
        ----------
        path: str
            Absolute path to download data from
        start: int
            Start position of the block
        end: int
            End position of the block
        """
        try:
            r = self._send_to_api(
                method="get",
                endpoint="read",
                json={"path": path, "offset": start, "length": end - start},
            )
            return base64.b64decode(r["data"])
        except DatabricksException as e:
            if e.error_code == "RESOURCE_DOES_NOT_EXIST":
                raise FileNotFoundError(e.message) from e
            elif e.error_code in ["INVALID_PARAMETER_VALUE", "MAX_READ_SIZE_EXCEEDED"]:
                raise ValueError(e.message) from e

            raise

    def invalidate_cache(self, path=None):
        if path is None:
            self.dircache.clear()
        else:
            self.dircache.pop(path, None)
        super().invalidate_cache(path)


class DatabricksFile(AbstractBufferedFile):
    """
    Helper class for files referenced in the DatabricksFileSystem.
    """

    DEFAULT_BLOCK_SIZE = 1 * 2**20  # only allowed block size

    def __init__(
        self,
        fs,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        **kwargs,
    ):
        """
        Create a new instance of the DatabricksFile.

        The blocksize needs to be the default one.
        """
        if block_size is None or block_size == "default":
            block_size = self.DEFAULT_BLOCK_SIZE

        if block_size != self.DEFAULT_BLOCK_SIZE:
            raise ValueError(
                f"Only the default block size is allowed, not {block_size}"
            )

        super().__init__(
            fs,
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_type=cache_type,
            cache_options=cache_options or {},
            **kwargs,
        )

    def _initiate_upload(self):
        """Internal function to start a file upload"""
        self.handle = self.fs._create_handle(self.path)

    def _upload_chunk(self, final=False):
        """Internal function to add a chunk of data to a started upload"""
        self.buffer.seek(0)
        data = self.buffer.getvalue()

        data_chunks = [
            data[start:end] for start, end in self._to_sized_blocks(len(data))
        ]

        for data_chunk in data_chunks:
            self.fs._add_data(handle=self.handle, data=data_chunk)

        if final:
            self.fs._close_handle(handle=self.handle)
            return True

    def _fetch_range(self, start, end):
        """Internal function to download a block of data"""
        return_buffer = b""
        length = end - start
        for chunk_start, chunk_end in self._to_sized_blocks(length, start):
            return_buffer += self.fs._get_data(
                path=self.path, start=chunk_start, end=chunk_end
            )

        return return_buffer

    def _to_sized_blocks(self, length, start=0):
        """Helper function to split a range from 0 to total_length into blocksizes"""
        end = start + length
        for data_chunk in range(start, end, self.blocksize):
            data_start = data_chunk
            data_end = min(end, data_chunk + self.blocksize)
            yield data_start, data_end
<|endoftext|>from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from fsspec.utils import atomic_write

try:
    import ujson as json
except ImportError:
    if not TYPE_CHECKING:
        import json

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Literal, TypeAlias

    from .cached import CachingFileSystem

    Detail: TypeAlias = dict[str, Any]


class CacheMetadata:
    """Cache metadata.

    All reading and writing of cache metadata is performed by this class,
    accessing the cached files and blocks is not.

    Metadata is stored in a single file per storage directory in JSON format.
    For backward compatibility. No longer supports pickle.
    """

    def __init__(self, storage: list[str]):
        """

        Parameters
        ----------
        storage: list[str]
            Directories containing cached files, must be at least one. Metadata
            is stored in the last of these directories by convention.
        """
        if not storage:
            raise ValueError("CacheMetadata expects at least one storage location")

        self._storage = storage
        self.cached_files: list[Detail] = [{}]

    def _load(self, fn: str) -> Detail:
        """Low-level function to load metadata from specific file"""
        with open(fn, "r") as f:
            loaded = json.load(f)
        for c in loaded.values():
            if isinstance(c.get("blocks"), list):
                c["blocks"] = set(c["blocks"])
        return loaded

    def _save(self, metadata_to_save: Detail, fn: str) -> None:
        """Low-level function to save metadata to specific file"""
        with atomic_write(fn, mode="w") as f:
            json.dump(metadata_to_save, f)

    def _scan_locations(
        self, writable_only: bool = False
    ) -> Iterator[tuple[str, str, bool]]:
        """Yield locations (filenames) where metadata is stored, and whether
        writable or not.

        Parameters
        ----------
        writable: bool
            Set to True to only yield writable locations.

        Returns
        -------
        Yields (str, str, bool)
        """
        n = len(self._storage)
        for i, storage in enumerate(self._storage):
            writable = i == n - 1
            if writable_only and not writable:
                continue
            yield os.path.join(storage, "cache"), storage, writable

    def check_file(
        self, path: str, cfs: CachingFileSystem | None
    ) -> Literal[False] | tuple[Detail, str]:
        """If path is in cache return its details, otherwise return ``False``.

        If the optional CachingFileSystem is specified then it is used to
        perform extra checks to reject possible matches, such as if they are
        too old.
        """
        for (fn, base, _), cache in zip(self._scan_locations(), self.cached_files):
            if path not in cache:
                continue
            detail = cache[path].copy()

            if cfs is not None:
                if cfs.check_files and detail["uid"] != cfs.fs.ukey(path):
                    # Wrong file as determined by hash of file properties
                    continue
                if cfs.expiry and time.time() - detail["time"] > cfs.expiry:
                    # Cached file has expired
                    continue

            fn = os.path.join(base, detail["fn"])
            if os.path.exists(fn):
                return detail, fn
        return False

    def clear_expired(self, expiry_time: int) -> tuple[list[str], bool]:
        """Remove expired metadata from the cache.

        Returns names of files corresponding to expired metadata and a boolean
        flag indicating whether the writable cache is empty. Caller is
        responsible for deleting the expired files.
        """
        expired_files = []
        for path, detail in self.cached_files[-1].copy().items():
            if time.time() - detail["time"] > expiry_time:
                fn = detail.get("fn", "")
                if not fn:
                    raise RuntimeError(
                        f"Cache metadata does not contain 'fn' for {path}"
                    )
                fn = os.path.join(self._storage[-1], fn)
                expired_files.append(fn)
                self.cached_files[-1].pop(path)

        if self.cached_files[-1]:
            cache_path = os.path.join(self._storage[-1], "cache")
            self._save(self.cached_files[-1], cache_path)

        writable_cache_empty = not self.cached_files[-1]
        return expired_files, writable_cache_empty

    def load(self) -> None:
        """Load all metadata from disk and store in ``self.cached_files``"""
        cached_files = []
        for fn, _, _ in self._scan_locations():
            if os.path.exists(fn):
                # TODO: consolidate blocks here
                cached_files.append(self._load(fn))
            else:
                cached_files.append({})
        self.cached_files = cached_files or [{}]

    def on_close_cached_file(self, f: Any, path: str) -> None:
        """Perform side-effect actions on closing a cached file.

        The actual closing of the file is the responsibility of the caller.
        """
        # File must be writable, so in self.cached_files[-1]
        c = self.cached_files[-1][path]
        if c["blocks"] is not True and len(c["blocks"]) * f.blocksize >= f.size:
            c["blocks"] = True

    def pop_file(self, path: str) -> str | None:
        """Remove metadata of cached file.

        If path is in the cache, return the filename of the cached file,
        otherwise return ``None``.  Caller is responsible for deleting the
        cached file.
        """
        details = self.check_file(path, None)
        if not details:
            return None
        _, fn = details
        if fn.startswith(self._storage[-1]):
            self.cached_files[-1].pop(path)
            self.save()
        else:
            raise PermissionError(
                "Can only delete cached file in last, writable cache location"
            )
        return fn

    def save(self) -> None:
        """Save metadata to disk"""
        for (fn, _, writable), cache in zip(self._scan_locations(), self.cached_files):
            if not writable:
                continue

            if os.path.exists(fn):
                cached_files = self._load(fn)
                for k, c in cached_files.items():
                    if k in cache:
                        if c["blocks"] is True or cache[k]["blocks"] is True:
                            c["blocks"] = True
                        else:
                            # self.cached_files[*][*]["blocks"] must continue to
                            # point to the same set object so that updates
                            # performed by MMapCache are propagated back to
                            # self.cached_files.
                            blocks = cache[k]["blocks"]
                            blocks.update(c["blocks"])
                            c["blocks"] = blocks
                        c["time"] = max(c["time"], cache[k]["time"])
                        c["uid"] = cache[k]["uid"]

                # Files can be added to cache after it was written once
                for k, c in cache.items():
                    if k not in cached_files:
                        cached_files[k] = c
            else:
                cached_files = cache
            cache = {k: v.copy() for k, v in cached_files.items()}
            for c in cache.values():
                if isinstance(c["blocks"], set):
                    c["blocks"] = list(c["blocks"])
            self._save(cache, fn)
            self.cached_files[-1] = cached_files

    def update_file(self, path: str, detail: Detail) -> None:
        """Update metadata for specific file in memory, do not save"""
        self.cached_files[-1][path] = detail
<|endoftext|>import datetime
import logging
import os
import types
import uuid
from stat import S_ISDIR, S_ISLNK

import paramiko

from .. import AbstractFileSystem
from ..utils import infer_storage_options

logger = logging.getLogger("fsspec.sftp")


class SFTPFileSystem(AbstractFileSystem):
    """Files over SFTP/SSH

    Peer-to-peer filesystem over SSH using paramiko.

    Note: if using this with the ``open`` or ``open_files``, with full URLs,
    there is no way to tell if a path is relative, so all paths are assumed
    to be absolute.
    """

    protocol = "sftp", "ssh"

    def __init__(self, host, **ssh_kwargs):
        """

        Parameters
        ----------
        host: str
            Hostname or IP as a string
        temppath: str
            Location on the server to put files, when within a transaction
        ssh_kwargs: dict
            Parameters passed on to connection. See details in
            https://docs.paramiko.org/en/3.3/api/client.html#paramiko.client.SSHClient.connect
            May include port, username, password...
        """
        if self._cached:
            return
        super().__init__(**ssh_kwargs)
        self.temppath = ssh_kwargs.pop("temppath", "/tmp")  # remote temp directory
        self.host = host
        self.ssh_kwargs = ssh_kwargs
        self._connect()

    def _connect(self):
        logger.debug("Connecting to SFTP server %s", self.host)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.host, **self.ssh_kwargs)
        self.ftp = self.client.open_sftp()

    @classmethod
    def _strip_protocol(cls, path):
        return infer_storage_options(path)["path"]

    @staticmethod
    def _get_kwargs_from_urls(urlpath):
        out = infer_storage_options(urlpath)
        out.pop("path", None)
        out.pop("protocol", None)
        return out

    def mkdir(self, path, create_parents=True, mode=511):
        path = self._strip_protocol(path)
        logger.debug("Creating folder %s", path)
        if self.exists(path):
            raise FileExistsError(f"File exists: {path}")

        if create_parents:
            self.makedirs(path)
        else:
            self.ftp.mkdir(path, mode)

    def makedirs(self, path, exist_ok=False, mode=511):
        if self.exists(path) and not exist_ok:
            raise FileExistsError(f"File exists: {path}")

        parts = path.split("/")
        new_path = "/" if path[:1] == "/" else ""

        for part in parts:
            if part:
                new_path = f"{new_path}/{part}" if new_path else part
                if not self.exists(new_path):
                    self.ftp.mkdir(new_path, mode)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        logger.debug("Removing folder %s", path)
        self.ftp.rmdir(path)

    def info(self, path):
        path = self._strip_protocol(path)
        stat = self._decode_stat(self.ftp.stat(path))
        stat["name"] = path
        return stat

    @staticmethod
    def _decode_stat(stat, parent_path=None):
        if S_ISDIR(stat.st_mode):
            t = "directory"
        elif S_ISLNK(stat.st_mode):
            t = "link"
        else:
            t = "file"
        out = {
            "name": "",
            "size": stat.st_size,
            "type": t,
            "uid": stat.st_uid,
            "gid": stat.st_gid,
            "time": datetime.datetime.fromtimestamp(
                stat.st_atime, tz=datetime.timezone.utc
            ),
            "mtime": datetime.datetime.fromtimestamp(
                stat.st_mtime, tz=datetime.timezone.utc
            ),
        }
        if parent_path:
            out["name"] = "/".join([parent_path.rstrip("/"), stat.filename])
        return out

    def ls(self, path, detail=False):
        path = self._strip_protocol(path)
        logger.debug("Listing folder %s", path)
        stats = [self._decode_stat(stat, path) for stat in self.ftp.listdir_iter(path)]
        if detail:
            return stats
        else:
            paths = [stat["name"] for stat in stats]
            return sorted(paths)

    def put_file(self, lpath, rpath, callback=None, **kwargs):
        self.mkdirs(self._parent(os.fspath(rpath)), exist_ok=True)
        logger.debug("Put file %s into %s", lpath, rpath)
        self.ftp.put(lpath, rpath)

    def get_file(self, rpath, lpath, **kwargs):
        if self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
        else:
            self.ftp.get(self._strip_protocol(rpath), lpath)

    def _open(self, path, mode="rb", block_size=None, **kwargs):
        """
        block_size: int or None
            If 0, no buffering, if 1, line buffering, if >1, buffer that many
            bytes, if None use default from paramiko.
        """
        logger.debug("Opening file %s", path)
        if kwargs.get("autocommit", True) is False:
            # writes to temporary file, move on commit
            path2 = "/".join([self.temppath, str(uuid.uuid4())])
            f = self.ftp.open(path2, mode, bufsize=block_size if block_size else -1)
            f.temppath = path2
            f.targetpath = path
            f.fs = self
            f.commit = types.MethodType(commit_a_file, f)
            f.discard = types.MethodType(discard_a_file, f)
        else:
            f = self.ftp.open(path, mode, bufsize=block_size if block_size else -1)
        return f

    def _rm(self, path):
        if self.isdir(path):
            self.ftp.rmdir(path)
        else:
            self.ftp.remove(path)

    def mv(self, old, new):
        new = self._strip_protocol(new)
        old = self._strip_protocol(old)
        logger.debug("Renaming %s into %s", old, new)
        self.ftp.posix_rename(old, new)


def commit_a_file(self):
    self.fs.mv(self.temppath, self.targetpath)


def discard_a_file(self):
    self.fs._rm(self.temppath)
<|endoftext|>import base64
import io
from urllib.parse import unquote

from fsspec import AbstractFileSystem
from fsspec.utils import stringify_path


class DataFileSystem(AbstractFileSystem):
    """A handy decoder for data-URLs

    Example
    -------
    >>> with fsspec.open("data:,Hello%2C%20World%21") as f:
    ...     print(f.read())
    b"Hello, World!"

    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs
    """

    protocol = "data"

    def __init__(self, **kwargs):
        """No parameters for this filesystem"""
        super().__init__(**kwargs)

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        path = stringify_path(path)
        if path.startswith("data://"):
            path = path[7:]
        elif path.startswith("data:"):
            path = path[5:]
        # Do NOT strip trailing slashes, as they may be meaningful base64 characters
        # or percent-encoded data content
        return path

    def cat_file(self, path, start=None, end=None, **kwargs):
        pref, data = path.split(",", 1)
        if pref.endswith("base64"):
            return base64.b64decode(data)[start:end]
        return unquote(data).encode()[start:end]

    def info(self, path, **kwargs):
        pref, name = path.split(",", 1)
        data = self.cat_file(path)
        mime = pref.split(":", 1)[1].split(";", 1)[0]
        return {"name": name, "size": len(data), "type": "file", "mimetype": mime}

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        if "r" not in mode:
            raise ValueError("Read only filesystem")
        return io.BytesIO(self.cat_file(path))

    @staticmethod
    def encode(data: bytes, mime: str | None = None):
        """Format the given data into data-URL syntax

        This version always base64 encodes, even when the data is ascii/url-safe.
        """
        return f"data:{mime or ''};base64,{base64.b64encode(data).decode()}"
<|endoftext|>import base64
import re

import requests

from ..spec import AbstractFileSystem
from ..utils import infer_storage_options
from .memory import MemoryFile


class GithubFileSystem(AbstractFileSystem):
    """Interface to files in github

    An instance of this class provides the files residing within a remote github
    repository. You may specify a point in the repos history, by SHA, branch
    or tag (default is current master).

    For files less than 1 MB in size, file content is returned directly in a
    MemoryFile. For larger files, or for files tracked by git-lfs, file content
    is returned as an HTTPFile wrapping the ``download_url`` provided by the
    GitHub API.

    When using fsspec.open, allows URIs of the form:

    - "github://path/file", in which case you must specify org, repo and
      may specify sha in the extra args
    - 'github://org:repo@/precip/catalog.yml', where the org and repo are
      part of the URI
    - 'github://org:repo@sha/precip/catalog.yml', where the sha is also included

    ``sha`` can be the full or abbreviated hex of the commit you want to fetch
    from, or a branch or tag name (so long as it doesn't contain special characters
    like "/", "?", which would have to be HTTP-encoded).

    For authorised access, you must provide username and token, which can be made
    at https://github.com/settings/tokens
    """

    url = "https://api.github.com/repos/{org}/{repo}/git/trees/{sha}"
    content_url = "https://api.github.com/repos/{org}/{repo}/contents/{path}?ref={sha}"
    protocol = "github"
    timeout = (60, 60)  # connect, read timeouts

    def __init__(
        self, org, repo, sha=None, username=None, token=None, timeout=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.org = org
        self.repo = repo
        if (username is None) ^ (token is None):
            raise ValueError("Auth required both username and token")
        self.username = username
        self.token = token
        if timeout is not None:
            self.timeout = timeout
        if sha is None:
            # look up default branch (not necessarily "master")
            u = "https://api.github.com/repos/{org}/{repo}"
            r = requests.get(
                u.format(org=org, repo=repo), timeout=self.timeout, **self.kw
            )
            r.raise_for_status()
            sha = r.json()["default_branch"]

        self.root = sha
        self.ls("")
        try:
            from .http import HTTPFileSystem

            self.http_fs = HTTPFileSystem(**kwargs)
        except ImportError:
            self.http_fs = None

    @property
    def kw(self):
        if self.username:
            return {"auth": (self.username, self.token)}
        return {}

    @classmethod
    def repos(cls, org_or_user, is_org=True):
        """List repo names for given org or user

        This may become the top level of the FS

        Parameters
        ----------
        org_or_user: str
            Name of the github org or user to query
        is_org: bool (default True)
            Whether the name is an organisation (True) or user (False)

        Returns
        -------
        List of string
        """
        r = requests.get(
            f"https://api.github.com/{['users', 'orgs'][is_org]}/{org_or_user}/repos",
            timeout=cls.timeout,
        )
        r.raise_for_status()
        return [repo["name"] for repo in r.json()]

    @property
    def tags(self):
        """Names of tags in the repo"""
        r = requests.get(
            f"https://api.github.com/repos/{self.org}/{self.repo}/tags",
            timeout=self.timeout,
            **self.kw,
        )
        r.raise_for_status()
        return [t["name"] for t in r.json()]

    @property
    def branches(self):
        """Names of branches in the repo"""
        r = requests.get(
            f"https://api.github.com/repos/{self.org}/{self.repo}/branches",
            timeout=self.timeout,
            **self.kw,
        )
        r.raise_for_status()
        return [t["name"] for t in r.json()]

    @property
    def refs(self):
        """Named references, tags and branches"""
        return {"tags": self.tags, "branches": self.branches}

    def ls(self, path, detail=False, sha=None, _sha=None, **kwargs):
        """List files at given path

        Parameters
        ----------
        path: str
            Location to list, relative to repo root
        detail: bool
            If True, returns list of dicts, one per file; if False, returns
            list of full filenames only
        sha: str (optional)
            List at the given point in the repo history, branch or tag name or commit
            SHA
        _sha: str (optional)
            List this specific tree object (used internally to descend into trees)
        """
        path = self._strip_protocol(path)
        if path == "":
            _sha = sha or self.root
        if _sha is None:
            parts = path.rstrip("/").split("/")
            so_far = ""
            _sha = sha or self.root
            for part in parts:
                out = self.ls(so_far, True, sha=sha, _sha=_sha)
                so_far += "/" + part if so_far else part
                out = [o for o in out if o["name"] == so_far]
                if not out:
                    raise FileNotFoundError(path)
                out = out[0]
                if out["type"] == "file":
                    if detail:
                        return [out]
                    else:
                        return path
                _sha = out["sha"]
        if path not in self.dircache or sha not in [self.root, None]:
            r = requests.get(
                self.url.format(org=self.org, repo=self.repo, sha=_sha),
                timeout=self.timeout,
                **self.kw,
            )
            if r.status_code == 404:
                raise FileNotFoundError(path)
            r.raise_for_status()
            types = {"blob": "file", "tree": "directory"}
            out = [
                {
                    "name": path + "/" + f["path"] if path else f["path"],
                    "mode": f["mode"],
                    "type": types[f["type"]],
                    "size": f.get("size", 0),
                    "sha": f["sha"],
                }
                for f in r.json()["tree"]
                if f["type"] in types
            ]
            if sha in [self.root, None]:
                self.dircache[path] = out
        else:
            out = self.dircache[path]
        if detail:
            return out
        else:
            return sorted([f["name"] for f in out])

    def invalidate_cache(self, path=None):
        self.dircache.clear()

    @classmethod
    def _strip_protocol(cls, path):
        opts = infer_storage_options(path)
        if "username" not in opts:
            return super()._strip_protocol(path)
        return opts["path"].lstrip("/")

    @staticmethod
    def _get_kwargs_from_urls(path):
        opts = infer_storage_options(path)
        if "username" not in opts:
            return {}
        out = {"org": opts["username"], "repo": opts["password"]}
        if opts["host"]:
            out["sha"] = opts["host"]
        return out

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        sha=None,
        **kwargs,
    ):
        if mode != "rb":
            raise NotImplementedError

        # construct a url to hit the GitHub API's repo contents API
        url = self.content_url.format(
            org=self.org, repo=self.repo, path=path, sha=sha or self.root
        )

        # make a request to this API, and parse the response as JSON
        r = requests.get(url, timeout=self.timeout, **self.kw)
        if r.status_code == 404:
            raise FileNotFoundError(path)
        r.raise_for_status()
        content_json = r.json()

        # if the response's content key is not empty, try to parse it as base64
        if content_json["content"]:
            content = base64.b64decode(content_json["content"])

            # as long as the content does not start with the string
            # "version https://git-lfs.github.com/"
            # then it is probably not a git-lfs pointer and we can just return
            # the content directly
            if not content.startswith(b"version https://git-lfs.github.com/"):
                return MemoryFile(None, None, content)

        # we land here if the content was not present in the first response
        # (regular file over 1MB or git-lfs tracked file)
        # in this case, we get let the HTTPFileSystem handle the download
        if self.http_fs is None:
            raise ImportError(
                "Please install fsspec[http] to access github files >1 MB "
                "or git-lfs tracked files."
            )
        return self.http_fs.open(
            content_json["download_url"],
            mode=mode,
            block_size=block_size,
            cache_options=cache_options,
            **kwargs,
        )

    def rm(self, path, recursive=False, maxdepth=None, message=None):
        path = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(path):
            self.rm_file(p, message=message)

    def rm_file(self, path, message=None, **kwargs):
        """
        Remove a file from a specified branch using a given commit message.

        Since Github DELETE operation requires a branch name, and we can't reliably
        determine whether the provided SHA refers to a branch, tag, or commit, we
        assume it's a branch. If it's not, the user will encounter an error when
        attempting to retrieve the file SHA or delete the file.

        Parameters
        ----------
        path: str
            The file's location relative to the repository root.
        message: str, optional
            The commit message for the deletion.
        """

        if not self.username:
            raise ValueError("Authentication required")

        path = self._strip_protocol(path)

        # Attempt to get SHA from cache or Github API
        sha = self._get_sha_from_cache(path)
        if not sha:
            url = self.content_url.format(
                org=self.org, repo=self.repo, path=path.lstrip("/"), sha=self.root
            )
            r = requests.get(url, timeout=self.timeout, **self.kw)
            if r.status_code == 404:
                raise FileNotFoundError(path)
            r.raise_for_status()
            sha = r.json()["sha"]

        # Delete the file
        delete_url = self.content_url.format(
            org=self.org, repo=self.repo, path=path, sha=self.root
        )
        branch = self.root
        data = {
            "message": message or f"Delete {path}",
            "sha": sha,
            **({"branch": branch} if branch else {}),
        }

        r = requests.delete(delete_url, json=data, timeout=self.timeout, **self.kw)
        error_message = r.json().get("message", "")
        if re.search(r"Branch .+ not found", error_message):
            error = "Remove only works when the filesystem is initialised from a branch or default (None)"
            raise ValueError(error)
        r.raise_for_status()

        self.invalidate_cache(path)

    def _get_sha_from_cache(self, path):
        for entries in self.dircache.values():
            for entry in entries:
                entry_path = entry.get("name")
                if entry_path and entry_path == path and "sha" in entry:
                    return entry["sha"]
        return None
<|endoftext|>import asyncio
import functools
import inspect

import fsspec
from fsspec.asyn import AsyncFileSystem, running_async

from .chained import ChainedFileSystem


def async_wrapper(func, obj=None, semaphore=None):
    """
    Wraps a synchronous function to make it awaitable.

    Parameters
    ----------
    func : callable
        The synchronous function to wrap.
    obj : object, optional
        The instance to bind the function to, if applicable.
    semaphore : asyncio.Semaphore, optional
        A semaphore to limit concurrent calls.

    Returns
    -------
    coroutine
        An awaitable version of the function.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if semaphore:
            async with semaphore:
                return await asyncio.to_thread(func, *args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


class AsyncFileSystemWrapper(AsyncFileSystem, ChainedFileSystem):
    """
    A wrapper class to convert a synchronous filesystem into an asynchronous one.

    This class takes an existing synchronous filesystem implementation and wraps all
    its methods to provide an asynchronous interface.

    Parameters
    ----------
    sync_fs : AbstractFileSystem
        The synchronous filesystem instance to wrap.
    """

    protocol = "asyncwrapper", "async_wrapper"
    cachable = False

    def __init__(
        self,
        fs=None,
        asynchronous=None,
        target_protocol=None,
        target_options=None,
        semaphore=None,
        max_concurrent_tasks=None,
        **kwargs,
    ):
        if asynchronous is None:
            asynchronous = running_async()
        super().__init__(asynchronous=asynchronous, **kwargs)
        if fs is not None:
            self.sync_fs = fs
        else:
            self.sync_fs = fsspec.filesystem(target_protocol, **target_options)
        self.protocol = self.sync_fs.protocol
        self.semaphore = semaphore
        self._wrap_all_sync_methods()

    @property
    def fsid(self):
        return f"async_{self.sync_fs.fsid}"

    def _wrap_all_sync_methods(self):
        """
