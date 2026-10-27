class UrlDispatcher(AbstractRouter, Mapping[str, AbstractResource]):

    NAME_SPLIT_RE = re.compile(r"[.:-]")

    def __init__(self) -> None:
        super().__init__()
        self._resources: list[AbstractResource] = []
        self._named_resources: dict[str, AbstractResource] = {}
        self._resource_index: dict[str, list[AbstractResource]] = {}
        self._matched_sub_app_resources: list[MatchedSubAppResource] = []

    async def resolve(self, request: Request) -> UrlMappingMatchInfo:
        resource_index = self._resource_index
        allowed_methods: set[str] = set()

        # MatchedSubAppResource is primarily used to match on domain names
        # (though custom rules could match on other things). This means that
        # the traversal algorithm below can't be applied, and that we likely
        # need to check these first so a sub app that defines the same path
        # as a parent app will get priority if there's a domain match.
        #
        # For most cases we do not expect there to be many of these since
        # currently they are only added by `.add_domain()`.
        for resource in self._matched_sub_app_resources:
            match_dict, allowed = await resource.resolve(request)
            if match_dict is not None:
                return match_dict
            else:
                allowed_methods |= allowed

        # Walk the url parts looking for candidates. We walk the url backwards
        # to ensure the most explicit match is found first. If there are multiple
        # candidates for a given url part because there are multiple resources
        # registered for the same canonical path, we resolve them in a linear
        # fashion to ensure registration order is respected.
        url_part = request.rel_url.path_safe
        while url_part:
            for candidate in resource_index.get(url_part, ()):
                match_dict, allowed = await candidate.resolve(request)
                if match_dict is not None:
                    return match_dict
                else:
                    allowed_methods |= allowed
            if url_part == "/":
                break
            url_part = url_part.rpartition("/")[0] or "/"

        if allowed_methods:
            return MatchInfoError(HTTPMethodNotAllowed(request.method, allowed_methods))

        return MatchInfoError(HTTPNotFound())

    def __iter__(self) -> Iterator[str]:
        return iter(self._named_resources)

    def __len__(self) -> int:
        return len(self._named_resources)

    def __contains__(self, resource: object) -> bool:
        return resource in self._named_resources

    def __getitem__(self, name: str) -> AbstractResource:
        return self._named_resources[name]

    def resources(self) -> ResourcesView:
        return ResourcesView(self._resources)

    def routes(self) -> RoutesView:
        return RoutesView(self._resources)

    def named_resources(self) -> Mapping[str, AbstractResource]:
        return MappingProxyType(self._named_resources)

    def register_resource(self, resource: AbstractResource) -> None:
        assert isinstance(
            resource, AbstractResource
        ), f"Instance of AbstractResource class is required, got {resource!r}"
        if self.frozen:
            raise RuntimeError("Cannot register a resource into frozen router.")

        name = resource.name

        if name is not None:
            parts = self.NAME_SPLIT_RE.split(name)
            for part in parts:
                if keyword.iskeyword(part):
                    raise ValueError(
                        f"Incorrect route name {name!r}, "
                        "python keywords cannot be used "
                        "for route name"
                    )
                if not part.isidentifier():
                    raise ValueError(
                        f"Incorrect route name {name!r}, "
                        "the name should be a sequence of "
                        "python identifiers separated "
                        "by dash, dot or column"
                    )
            if name in self._named_resources:
                raise ValueError(
                    f"Duplicate {name!r}, "
                    f"already handled by {self._named_resources[name]!r}"
                )
            self._named_resources[name] = resource
        self._resources.append(resource)

        if isinstance(resource, MatchedSubAppResource):
            # We cannot index match sub-app resources because they have match rules
            self._matched_sub_app_resources.append(resource)
        else:
            self.index_resource(resource)

    def _get_resource_index_key(self, resource: AbstractResource) -> str:
        """Return a key to index the resource in the resource index."""
        if "{" in (index_key := resource.canonical):
            # strip at the first { to allow for variables, and than
            # rpartition at / to allow for variable parts in the path
            # For example if the canonical path is `/core/locations{tail:.*}`
            # the index key will be `/core` since index is based on the
            # url parts split by `/`
            index_key = index_key.partition("{")[0].rpartition("/")[0]
        return index_key.rstrip("/") or "/"

    def index_resource(self, resource: AbstractResource) -> None:
        """Add a resource to the resource index."""
        resource_key = self._get_resource_index_key(resource)
        # There may be multiple resources for a canonical path
        # so we keep them in a list to ensure that registration
        # order is respected.
        self._resource_index.setdefault(resource_key, []).append(resource)

    def unindex_resource(self, resource: AbstractResource) -> None:
        """Remove a resource from the resource index."""
        resource_key = self._get_resource_index_key(resource)
        self._resource_index[resource_key].remove(resource)

    def add_resource(self, path: str, *, name: str | None = None) -> Resource:
        if path and not path.startswith("/"):
            raise ValueError("path should be started with / or be empty")
        # Reuse last added resource if path and name are the same
        if self._resources:
            resource = self._resources[-1]
            if resource.name == name and resource.raw_match(path):
                return cast(Resource, resource)
        if not ("{" in path or "}" in path or ROUTE_RE.search(path)):
            resource = PlainResource(path, name=name)
            self.register_resource(resource)
            return resource
        resource = DynamicResource(path, name=name)
        self.register_resource(resource)
        return resource

    def add_route(
        self,
        method: str,
        path: str,
        handler: Handler | type[AbstractView],
        *,
        name: str | None = None,
        expect_handler: _ExpectHandler | None = None,
    ) -> AbstractRoute:
        resource = self.add_resource(path, name=name)
        return resource.add_route(method, handler, expect_handler=expect_handler)

    def add_static(
        self,
        prefix: str,
        path: PathLike,
        *,
        name: str | None = None,
        expect_handler: _ExpectHandler | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        show_index: bool = False,
        follow_symlinks: bool = False,
        append_version: bool = False,
    ) -> AbstractResource:
        """Add static files view.

        prefix - url prefix
        path - folder with files

        """
        assert prefix.startswith("/")
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        resource = StaticResource(
            prefix,
            path,
            name=name,
            expect_handler=expect_handler,
            chunk_size=chunk_size,
            show_index=show_index,
            follow_symlinks=follow_symlinks,
            append_version=append_version,
        )
        self.register_resource(resource)
        return resource

    def add_head(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        """Shortcut for add_route with method HEAD."""
        return self.add_route(hdrs.METH_HEAD, path, handler, **kwargs)

    def add_options(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        """Shortcut for add_route with method OPTIONS."""
        return self.add_route(hdrs.METH_OPTIONS, path, handler, **kwargs)

    def add_get(
        self,
        path: str,
        handler: Handler,
        *,
        name: str | None = None,
        allow_head: bool = True,
        **kwargs: Any,
    ) -> AbstractRoute:
        """Shortcut for add_route with method GET.

        If allow_head is true, another
        route is added allowing head requests to the same endpoint.
        """
        resource = self.add_resource(path, name=name)
        if allow_head:
            resource.add_route(hdrs.METH_HEAD, handler, **kwargs)
        return resource.add_route(hdrs.METH_GET, handler, **kwargs)

    def add_post(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        """Shortcut for add_route with method POST."""
        return self.add_route(hdrs.METH_POST, path, handler, **kwargs)

    def add_put(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        """Shortcut for add_route with method PUT."""
        return self.add_route(hdrs.METH_PUT, path, handler, **kwargs)

    def add_patch(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        """Shortcut for add_route with method PATCH."""
        return self.add_route(hdrs.METH_PATCH, path, handler, **kwargs)

    def add_delete(self, path: str, handler: Handler, **kwargs: Any) -> AbstractRoute:
        """Shortcut for add_route with method DELETE."""
        return self.add_route(hdrs.METH_DELETE, path, handler, **kwargs)

    def add_view(
        self, path: str, handler: type[AbstractView], **kwargs: Any
    ) -> AbstractRoute:
        """Shortcut for add_route with ANY methods for a class-based view."""
        return self.add_route(hdrs.METH_ANY, path, handler, **kwargs)

    def freeze(self) -> None:
        super().freeze()
        for resource in self._resources:
            resource.freeze()

    def add_routes(self, routes: Iterable[AbstractRouteDef]) -> list[AbstractRoute]:
        """Append routes to route table.

        Parameter should be a sequence of RouteDef objects.

        Returns a list of registered AbstractRoute instances.
        """
        registered_routes = []
        for route_def in routes:
            registered_routes.extend(route_def.register(self))
        return registered_routes


def _quote_path(value: str) -> str:
    if YARL_VERSION < (1, 6):
        value = value.replace("%", "%25")
    return URL.build(path=value, encoded=False).raw_path


def _unquote_path_safe(value: str) -> str:
    if "%" not in value:
        return value
    return value.replace("%2F", "/").replace("%25", "%")


def _requote_path(value: str) -> str:
    # Quote non-ascii characters and other characters which must be quoted,
    # but preserve existing %-sequences.
    result = _quote_path(value)
    if "%" in value:
        result = result.replace("%25", "%")
    return result
<|endoftext|>"""Various helper functions"""

import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import suppress
from email.message import EmailMessage
from email.parser import HeaderParser
from email.policy import HTTP
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import (
    Any,
    ContextManager,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    get_args,
    overload,
)
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass

import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from propcache.api import under_cached_property as reify
from yarl import URL

from . import hdrs
from .log import client_logger

if sys.version_info >= (3, 11):
    import asyncio as async_timeout
else:
    import async_timeout

__all__ = ("BasicAuth", "ChainMapProxy", "ETag", "reify")

IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

PY_311 = sys.version_info >= (3, 11)

# This is the default size/limit for several operations.
# Matches the max size we receive from sockets:
# https://github.com/python/cpython/blob/1857a40807daeae3a1bf5efb682de9c9ae6df845/Lib/asyncio/selector_events.py#L766
DEFAULT_CHUNK_SIZE = 2**18  # 256 KiB

_T = TypeVar("_T")
_S = TypeVar("_S")

_SENTINEL = enum.Enum("_SENTINEL", "sentinel")
sentinel = _SENTINEL.sentinel

NO_EXTENSIONS = bool(os.environ.get("AIOHTTP_NO_EXTENSIONS"))

# https://datatracker.ietf.org/doc/html/rfc9112#section-6.3-2.1
EMPTY_BODY_STATUS_CODES = frozenset((204, 304, *range(100, 200)))
# https://datatracker.ietf.org/doc/html/rfc9112#section-6.3-2.1
# https://datatracker.ietf.org/doc/html/rfc9112#section-6.3-2.2
EMPTY_BODY_METHODS = frozenset({hdrs.METH_HEAD})

DEBUG = sys.flags.dev_mode or (
    not sys.flags.ignore_environment and bool(os.environ.get("PYTHONASYNCIODEBUG"))
)


CHAR = {chr(i) for i in range(0, 128)}
CTL = {chr(i) for i in range(0, 32)} | {
    chr(127),
}
SEPARATORS = {
    "(",
    ")",
    "<",
    ">",
    "@",
    ",",
    ";",
    ":",
    "\\",
    '"',
    "/",
    "[",
    "]",
    "?",
    "=",
    "{",
    "}",
    " ",
    chr(9),
}
TOKEN = CHAR ^ CTL ^ SEPARATORS


class noop:
    def __await__(self) -> Generator[None, None, None]:
        yield


def encode_basic_auth(login: str, password: str = "", encoding: str = "utf-8") -> str:
    """Encode HTTP Basic Authentication credentials as an Authorization header value.

    Returns a string of the form ``"Basic <base64>"`` suitable for use as the
    value of the ``Authorization`` (or ``Proxy-Authorization``) header.
    """
    if ":" in login:
        raise ValueError('A ":" is not allowed in login (RFC 7617#section-2)')
    creds = f"{login}:{password}".encode(encoding)
    return "Basic " + base64.b64encode(creds).decode(encoding)


class BasicAuth(namedtuple("BasicAuth", ["login", "password", "encoding"])):
    """Http basic authentication helper."""

    def __new__(
        cls, login: str, password: str = "", encoding: str = "latin1"
    ) -> "BasicAuth":
        if login is None:
            raise ValueError("None is not allowed as login value")

        if password is None:
            raise ValueError("None is not allowed as password value")

        if ":" in login:
            raise ValueError('A ":" is not allowed in login (RFC 1945#section-11.1)')

        warnings.warn(
            "BasicAuth is deprecated and will be removed in aiohttp 4.0; "
            "use aiohttp.encode_basic_auth() with "
            "headers={'Authorization': ...} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls, login, password, encoding)

    @classmethod
    def decode(cls, auth_header: str, encoding: str = "latin1") -> "BasicAuth":
        """Create a BasicAuth object from an Authorization HTTP header."""
        try:
            auth_type, encoded_credentials = auth_header.split(" ", 1)
        except ValueError:
            raise ValueError("Could not parse authorization header.")

        if auth_type.lower() != "basic":
            raise ValueError("Unknown authorization method %s" % auth_type)

        try:
            decoded = base64.b64decode(
                encoded_credentials.encode("ascii"), validate=True
            ).decode(encoding)
        except binascii.Error:
            raise ValueError("Invalid base64 encoding.")

        try:
            # RFC 2617 HTTP Authentication
            # https://www.ietf.org/rfc/rfc2617.txt
            # the colon must be present, but the username and password may be
            # otherwise blank.
            username, password = decoded.split(":", 1)
        except ValueError:
            raise ValueError("Invalid credentials.")

        return _basic_auth_no_warn(username, password, encoding)

    @classmethod
    def from_url(cls, url: URL, *, encoding: str = "latin1") -> Optional["BasicAuth"]:
        """Create BasicAuth from url."""
        if not isinstance(url, URL):
            raise TypeError("url should be yarl.URL instance")
        # Check raw_user and raw_password first as yarl is likely
        # to already have these values parsed from the netloc in the cache.
        if url.raw_user is None and url.raw_password is None:
            return None
        return _basic_auth_no_warn(url.user or "", url.password or "", encoding)

    def encode(self) -> str:
        """Encode credentials."""
        return encode_basic_auth(self.login, self.password, self.encoding)


def _basic_auth_no_warn(
    login: str, password: str = "", encoding: str = "latin1"
) -> BasicAuth:
    """Construct a BasicAuth without emitting the deprecation warning.

    For internal use only. Bypasses BasicAuth.__new__ so that aiohttp's own
    machinery doesn't trigger deprecation warnings in user code.
    """
    return tuple.__new__(BasicAuth, (login, password, encoding))


def strip_auth_from_url(url: URL) -> tuple[URL, BasicAuth | None]:
    """Remove user and password from URL if present and return BasicAuth object."""
    # Check raw_user and raw_password first as yarl is likely
    # to already have these values parsed from the netloc in the cache.
    if url.raw_user is None and url.raw_password is None:
        return url, None
    return url.with_user(None), _basic_auth_no_warn(url.user or "", url.password or "")


def netrc_from_env() -> netrc.netrc | None:
    """Load netrc from file.

    Attempt to load it from the path specified by the env-var
    NETRC or in the default location in the user's home directory.

    Returns None if it couldn't be found or fails to parse.
    """
    netrc_env = os.environ.get("NETRC")

    if netrc_env is not None:
        netrc_path = Path(netrc_env)
    else:
        try:
            home_dir = Path.home()
        except RuntimeError as e:  # pragma: no cover
            # if pathlib can't resolve home, it may raise a RuntimeError
            client_logger.debug(
                "Could not resolve home directory when "
                "trying to look for .netrc file: %s",
                e,
            )
            return None

        netrc_path = home_dir / ("_netrc" if IS_WINDOWS else ".netrc")

    try:
        return netrc.netrc(str(netrc_path))
    except netrc.NetrcParseError as e:
        client_logger.warning("Could not parse .netrc file: %s", e)
    except OSError as e:
        netrc_exists = False
        with contextlib.suppress(OSError):
            netrc_exists = netrc_path.is_file()
        # we couldn't read the file (doesn't exist, permissions, etc.)
        if netrc_env or netrc_exists:
            # only warn if the environment wanted us to load it,
            # or it appears like the default file does actually exist
            client_logger.warning("Could not read .netrc file: %s", e)

    return None


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ProxyInfo:
    proxy: URL
    proxy_auth: BasicAuth | None


def basicauth_from_netrc(netrc_obj: netrc.netrc | None, host: str) -> BasicAuth:
    """
    Return :py:class:`~aiohttp.BasicAuth` credentials for ``host`` from ``netrc_obj``.

    :raises LookupError: if ``netrc_obj`` is :py:data:`None` or if no
            entry is found for the ``host``.
    """
    if netrc_obj is None:
        raise LookupError("No .netrc file found")
    auth_from_netrc = netrc_obj.authenticators(host)

    if auth_from_netrc is None:
        raise LookupError(f"No entry for {host!s} found in the `.netrc` file.")
    login, account, password = auth_from_netrc

    # TODO(PY311): username = login or account
    # Up to python 3.10, account could be None if not specified,
    # and login will be empty string if not specified. From 3.11,
    # login and account will be empty string if not specified.
    username = login if (login or account is None) else account

    # TODO(PY311): Remove this, as password will be empty string
    # if not specified
    if password is None:
        password = ""

    return _basic_auth_no_warn(username, password)


def proxies_from_env() -> dict[str, ProxyInfo]:
    proxy_urls = {
        k: URL(v)
        for k, v in getproxies().items()
        if k in ("http", "https", "ws", "wss")
    }
    netrc_obj = netrc_from_env()
    stripped = {k: strip_auth_from_url(v) for k, v in proxy_urls.items()}
    ret = {}
    for proto, val in stripped.items():
        proxy, auth = val
        if proxy.scheme in ("https", "wss"):
            client_logger.warning(
                "%s proxies %s are not supported, ignoring", proxy.scheme.upper(), proxy
            )
            continue
        if netrc_obj and auth is None:
            if proxy.host is not None:
                try:
                    auth = basicauth_from_netrc(netrc_obj, proxy.host)
                except LookupError:
                    auth = None
        ret[proto] = ProxyInfo(proxy, auth)
    return ret


def get_env_proxy_for_url(url: URL) -> tuple[URL, BasicAuth | None]:
    """Get a permitted proxy for the given URL from the env."""
    if url.host is not None and proxy_bypass(url.host):
        raise LookupError(f"Proxying is disallowed for `{url.host!r}`")

    proxies_in_env = proxies_from_env()
    try:
        proxy_info = proxies_in_env[url.scheme]
    except KeyError:
        raise LookupError(f"No proxies found for `{url!s}` in the env")
    else:
        return proxy_info.proxy, proxy_info.proxy_auth


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MimeType:
    type: str
    subtype: str
    suffix: str
    parameters: "MultiDictProxy[str]"


@functools.lru_cache(maxsize=56)
def parse_mimetype(mimetype: str) -> MimeType:
    """Parses a MIME type into its components.

    mimetype is a MIME type string.

    Returns a MimeType object.

    Example:

    >>> parse_mimetype('text/html; charset=utf-8')
    MimeType(type='text', subtype='html', suffix='',
             parameters={'charset': 'utf-8'})

    """
    if not mimetype:
        return MimeType(
            type="", subtype="", suffix="", parameters=MultiDictProxy(MultiDict())
        )

    parts = mimetype.split(";")
    params: MultiDict[str] = MultiDict()
    for item in parts[1:]:
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        params.add(key.lower().strip(), value.strip(' "'))

    fulltype = parts[0].strip().lower()
    if fulltype == "*":
        fulltype = "*/*"

    mtype, _, stype = fulltype.partition("/")
    stype, _, suffix = stype.partition("+")

    return MimeType(
        type=mtype, subtype=stype, suffix=suffix, parameters=MultiDictProxy(params)
    )


class EnsureOctetStream(EmailMessage):
    def __init__(self) -> None:
        super().__init__()
        # https://www.rfc-editor.org/rfc/rfc9110#section-8.3-5
        self.set_default_type("application/octet-stream")

    def get_content_type(self) -> str:
        """Re-implementation from Message

        Returns application/octet-stream in place of plain/text when
        value is wrong.

        The way this class is used guarantees that content-type will
        be present so simplify the checks wrt to the base implementation.
        """
        value = self.get("content-type", "").lower()

        # Based on the implementation of _splitparam in the standard library
        ctype, _, _ = value.partition(";")
        ctype = ctype.strip()
        if ctype.count("/") != 1:
            return self.get_default_type()
        return ctype


@functools.lru_cache(maxsize=56)
def parse_content_type(raw: str) -> tuple[str, MappingProxyType[str, str]]:
    """Parse Content-Type header.

    Returns a tuple of the parsed content type and a
    MappingProxyType of parameters. The default returned value
    is `application/octet-stream`
    """
    msg = HeaderParser(EnsureOctetStream, policy=HTTP).parsestr(f"Content-Type: {raw}")
    content_type = msg.get_content_type()
    params = msg.get_params(())
    content_dict = dict(params[1:])  # First element is content type again
    return content_type, MappingProxyType(content_dict)


def guess_filename(obj: Any, default: str | None = None) -> str | None:
    name = getattr(obj, "name", None)
    if name and isinstance(name, str) and name[0] != "<" and name[-1] != ">":
        return Path(name).name
    return default


not_qtext_re = re.compile(r"[^\041\043-\133\135-\176]")
QCONTENT = {chr(i) for i in range(0x20, 0x7F)} | {"\t"}


def quoted_string(content: str) -> str:
    """Return 7-bit content as quoted-string.

    Format content into a quoted-string as defined in RFC5322 for
    Internet Message Format. Notice that this is not the 8-bit HTTP
    format, but the 7-bit email format. Content must be in usascii or
    a ValueError is raised.
    """
    if not (QCONTENT > set(content)):
        raise ValueError(f"bad content for quoted-string {content!r}")
    return not_qtext_re.sub(lambda x: "\\" + x.group(0), content)


def content_disposition_header(
    disptype: str, quote_fields: bool = True, _charset: str = "utf-8", **params: str
) -> str:
    """Sets ``Content-Disposition`` header for MIME.

    This is the MIME payload Content-Disposition header from RFC 2183
    and RFC 7579 section 4.2, not the HTTP Content-Disposition from
    RFC 6266.

    disptype is a disposition type: inline, attachment, form-data.
    Should be valid extension token (see RFC 2183)

    quote_fields performs value quoting to 7-bit MIME headers
    according to RFC 7578. Set to quote_fields to False if recipient
    can take 8-bit file names and field values.

    _charset specifies the charset to use when quote_fields is True.

    params is a dict with disposition params.
    """
    if not disptype or not (TOKEN > set(disptype)):
        raise ValueError(f"bad content disposition type {disptype!r}")

    value = disptype
    if params:
        lparams = []
        for key, val in params.items():
            if not key or not (TOKEN > set(key)):
                raise ValueError(f"bad content disposition parameter {key!r}={val!r}")
            if quote_fields:
                if key.lower() == "filename":
                    qval = quote(val, "", encoding=_charset)
                    lparams.append((key, '"%s"' % qval))
                else:
                    try:
                        qval = quoted_string(val)
                    except ValueError:
                        qval = "".join(
                            (_charset, "''", quote(val, "", encoding=_charset))
                        )
                        lparams.append((key + "*", qval))
                    else:
                        lparams.append((key, '"%s"' % qval))
            else:
                qval = val.replace("\\", "\\\\").replace('"', '\\"')
                lparams.append((key, '"%s"' % qval))
        sparams = "; ".join("=".join(pair) for pair in lparams)
        value = "; ".join((value, sparams))
    return value


def is_ip_address(host: str | None) -> bool:
    """Check if host looks like an IP Address.

    This check is only meant as a heuristic to ensure that
    a host is not a domain name.
    """
    if not host:
        return False
    # For a host to be an ipv4 address, it must be all numeric.
    # The host must contain a colon to be an IPv6 address.
    return ":" in host or host.replace(".", "").isdigit()


def is_canonical_ipv4_address(host: str) -> bool:
    """Check if host is a canonical dotted-quad IPv4 address.

    Rejects the legacy numeric forms that ``socket`` still accepts and
    maps onto an address, e.g. ``2130706433``, ``017700000001``, ``127.1``.
    """
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        # Each octet must be 1-3 ASCII digits; reject unicode digits
        # (which ``str.isdigit`` accepts but ``int`` may not), octal
        # leading zeros, and values above 255.
        if not (1 <= len(part) <= 3) or not part.isascii() or not part.isdigit():
            return False
        if part[0] == "0" and len(part) != 1:
            return False
        if int(part) > 255:
            return False
    return True


_cached_current_datetime: int | None = None
_cached_formatted_datetime = ""


def rfc822_formatted_time() -> str:
    global _cached_current_datetime
    global _cached_formatted_datetime

    now = int(time.time())
    if now != _cached_current_datetime:
        # Weekday and month names for HTTP date/time formatting;
        # always English!
        # Tuples are constants stored in codeobject!
        _weekdayname = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        _monthname = (
            "",  # Dummy so we can use 1-based month numbers
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        )

        year, month, day, hh, mm, ss, wd, *tail = time.gmtime(now)
        _cached_formatted_datetime = "%s, %02d %3s %4d %02d:%02d:%02d GMT" % (
            _weekdayname[wd],
            day,
            _monthname[month],
            year,
            hh,
            mm,
            ss,
        )
        _cached_current_datetime = now
    return _cached_formatted_datetime


def _weakref_handle(info: "tuple[weakref.ref[object], str]") -> None:
    ref, name = info
    ob = ref()
    if ob is not None:
        with suppress(Exception):
            getattr(ob, name)()


def weakref_handle(
    ob: object,
    name: str,
    timeout: float | None,
    loop: asyncio.AbstractEventLoop,
    timeout_ceil_threshold: float = 5,
) -> asyncio.TimerHandle | None:
    if timeout is not None and timeout > 0:
        when = loop.time() + timeout
        if timeout >= timeout_ceil_threshold:
            when = ceil(when)

        return loop.call_at(when, _weakref_handle, (weakref.ref(ob), name))
    return None


def call_later(
    cb: Callable[[], Any],
    timeout: float,
    loop: asyncio.AbstractEventLoop,
    timeout_ceil_threshold: float = 5,
) -> asyncio.TimerHandle | None:
    if timeout is None or timeout <= 0:
        return None
    now = loop.time()
    when = calculate_timeout_when(now, timeout, timeout_ceil_threshold)
    return loop.call_at(when, cb)


def calculate_timeout_when(
    loop_time: float,
    timeout: float,
    timeout_ceiling_threshold: float,
) -> float:
    """Calculate when to execute a timeout."""
    when = loop_time + timeout
    if timeout > timeout_ceiling_threshold:
        return ceil(when)
    return when


class TimeoutHandle:
    """Timeout handle"""

    __slots__ = ("_timeout", "_loop", "_ceil_threshold", "_callbacks")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        timeout: float | None,
        ceil_threshold: float = 5,
    ) -> None:
        self._timeout = timeout
        self._loop = loop
        self._ceil_threshold = ceil_threshold
        self._callbacks: list[
            tuple[Callable[..., None], tuple[Any, ...], dict[str, Any]]
        ] = []

    def register(
        self, callback: Callable[..., None], *args: Any, **kwargs: Any
    ) -> None:
        self._callbacks.append((callback, args, kwargs))

    def close(self) -> None:
        self._callbacks.clear()

    def start(self) -> asyncio.TimerHandle | None:
        timeout = self._timeout
        if timeout is not None and timeout > 0:
            when = self._loop.time() + timeout
            if timeout >= self._ceil_threshold:
                when = ceil(when)
            return self._loop.call_at(when, self.__call__)
        else:
            return None

    def timer(self) -> "BaseTimerContext":
        if self._timeout is not None and self._timeout > 0:
            timer = TimerContext(self._loop)
            self.register(timer.timeout)
            return timer
        else:
            return TimerNoop()

    def __call__(self) -> None:
        for cb, args, kwargs in self._callbacks:
            with suppress(Exception):
                cb(*args, **kwargs)

        self._callbacks.clear()


class BaseTimerContext(ContextManager["BaseTimerContext"]):

    __slots__ = ()

    def assert_timeout(self) -> None:
        """Raise TimeoutError if timeout has been exceeded."""


class TimerNoop(BaseTimerContext):

    __slots__ = ()

    def __enter__(self) -> BaseTimerContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return


class TimerContext(BaseTimerContext):
    """Low resolution timeout context manager"""

    __slots__ = ("_loop", "_tasks", "_cancelled", "_cancelling")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._tasks: list[asyncio.Task[Any]] = []
        self._cancelled = False
        self._cancelling = 0

    def assert_timeout(self) -> None:
        """Raise TimeoutError if timer has already been cancelled."""
        if self._cancelled:
            raise asyncio.TimeoutError from None

    def __enter__(self) -> BaseTimerContext:
        task = asyncio.current_task(loop=self._loop)
        if task is None:
            raise RuntimeError("Timeout context manager should be used inside a task")

        if sys.version_info >= (3, 11):
            # Remember if the task was already cancelling
            # so when we __exit__ we can decide if we should
            # raise asyncio.TimeoutError or let the cancellation propagate
            self._cancelling = task.cancelling()

        if self._cancelled:
            raise asyncio.TimeoutError from None

        self._tasks.append(task)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        enter_task: asyncio.Task[Any] | None = None
        if self._tasks:
            enter_task = self._tasks.pop()

        if exc_type is asyncio.CancelledError and self._cancelled:
            assert enter_task is not None
            # The timeout was hit, and the task was cancelled
            # so we need to uncancel the last task that entered the context manager
            # since the cancellation should not leak out of the context manager
            if sys.version_info >= (3, 11):
                # If the task was already cancelling don't raise
                # asyncio.TimeoutError and instead return None
                # to allow the cancellation to propagate
                if enter_task.uncancel() > self._cancelling:
                    return None
            raise asyncio.TimeoutError from exc_val
        return None

    def timeout(self) -> None:
        if not self._cancelled:
            for task in set(self._tasks):
                task.cancel()

            self._cancelled = True


def ceil_timeout(
    delay: float | None, ceil_threshold: float = 5
) -> async_timeout.Timeout:
    if delay is None or delay <= 0:
        return async_timeout.timeout(None)

    loop = asyncio.get_running_loop()
    now = loop.time()
    when = now + delay
    if delay > ceil_threshold:
        when = ceil(when)
    return async_timeout.timeout_at(when)


class HeadersMixin:
    """Mixin for handling headers."""

    ATTRS = frozenset(["_content_type", "_content_dict", "_stored_content_type"])

    _headers: MultiMapping[str]
    _content_type: str | None = None
    _content_dict: dict[str, str] | None = None
    _stored_content_type: str | None | _SENTINEL = sentinel

    def _parse_content_type(self, raw: str | None) -> None:
        self._stored_content_type = raw
        if raw is None:
            # default value according to RFC 2616
            self._content_type = "application/octet-stream"
            self._content_dict = {}
        else:
            content_type, content_mapping_proxy = parse_content_type(raw)
            self._content_type = content_type
            # _content_dict needs to be mutable so we can update it
            self._content_dict = content_mapping_proxy.copy()

    @property
    def content_type(self) -> str:
        """The value of content part for Content-Type HTTP header."""
        raw = self._headers.get(hdrs.CONTENT_TYPE)
        if self._stored_content_type != raw:
            self._parse_content_type(raw)
        assert self._content_type is not None
        return self._content_type

    @property
    def charset(self) -> str | None:
        """The value of charset part for Content-Type HTTP header."""
        raw = self._headers.get(hdrs.CONTENT_TYPE)
        if self._stored_content_type != raw:
            self._parse_content_type(raw)
        assert self._content_dict is not None
        return self._content_dict.get("charset")

    @property
    def content_length(self) -> int | None:
        """The value of Content-Length HTTP header."""
        content_length = self._headers.get(hdrs.CONTENT_LENGTH)
        return None if content_length is None else int(content_length)


def set_result(fut: "asyncio.Future[_T]", result: _T) -> None:
    if not fut.done():
        fut.set_result(result)


_EXC_SENTINEL = BaseException()


class ErrorableProtocol(Protocol):
    def set_exception(
        self,
        exc: BaseException,
        exc_cause: BaseException = ...,
    ) -> None: ...  # pragma: no cover


def set_exception(
    fut: "asyncio.Future[_T] | ErrorableProtocol",
    exc: BaseException,
    exc_cause: BaseException = _EXC_SENTINEL,
) -> None:
    """Set future exception.

    If the future is marked as complete, this function is a no-op.

    :param exc_cause: An exception that is a direct cause of ``exc``.
                      Only set if provided.
    """
    if asyncio.isfuture(fut) and fut.done():
        return

    exc_is_sentinel = exc_cause is _EXC_SENTINEL
    exc_causes_itself = exc is exc_cause
    if not exc_is_sentinel and not exc_causes_itself:
        exc.__cause__ = exc_cause

    fut.set_exception(exc)


@functools.total_ordering
class BaseKey(Generic[_T]):
    """Base for concrete context storage key classes.

    Each storage is provided with its own sub-class for the sake of some additional type safety.
    """

    __slots__ = ("_name", "_t", "__orig_class__")

    # This may be set by Python when instantiating with a generic type. We need to
    # support this, in order to support types that are not concrete classes,
    # like Iterable, which can't be passed as the second parameter to __init__.
    __orig_class__: type[object]

    def __init__(self, name: str, t: type[_T] | None = None):
        # Prefix with module name to help deduplicate key names.
        frame = inspect.currentframe()
        while frame:
            if frame.f_code.co_name == "<module>":
                module: str = frame.f_globals["__name__"]
                break
            frame = frame.f_back

        self._name = module + "." + name
        self._t = t

    def __lt__(self, other: object) -> bool:
        if isinstance(other, BaseKey):
            return self._name < other._name
        return True  # Order BaseKey above other types.

    def __repr__(self) -> str:
        t = self._t
        if t is None:
            with suppress(AttributeError):
                # Set to type arg.
                t = get_args(self.__orig_class__)[0]

        if t is None:
            t_repr = "<<Unknown>>"
        elif isinstance(t, type):
            if t.__module__ == "builtins":
                t_repr = t.__qualname__
            else:
                t_repr = f"{t.__module__}.{t.__qualname__}"
        else:
            t_repr = repr(t)
        return f"<{self.__class__.__name__}({self._name}, type={t_repr})>"


class AppKey(BaseKey[_T]):
    """Keys for static typing support in Application."""


class RequestKey(BaseKey[_T]):
    """Keys for static typing support in Request."""


class ResponseKey(BaseKey[_T]):
    """Keys for static typing support in Response."""


class ChainMapProxy(Mapping[str | AppKey[Any], Any]):
    __slots__ = ("_maps",)

    def __init__(self, maps: Iterable[Mapping[str | AppKey[Any], Any]]) -> None:
        self._maps = tuple(maps)

    def __init_subclass__(cls) -> None:
        raise TypeError(
            f"Inheritance class {cls.__name__} from ChainMapProxy is forbidden"
        )

    @overload  # type: ignore[override]
    def __getitem__(self, key: AppKey[_T]) -> _T: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: str | AppKey[_T]) -> Any:
        for mapping in self._maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        raise KeyError(key)

    @overload  # type: ignore[override]
    def get(self, key: AppKey[_T], default: _S) -> _T | _S: ...

    @overload
    def get(self, key: AppKey[_T], default: None = ...) -> _T | None: ...

    @overload
    def get(self, key: str, default: Any = ...) -> Any: ...

    def get(self, key: str | AppKey[_T], default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __len__(self) -> int:
        # reuses stored hash values if possible
        return len(set().union(*self._maps))

    def __iter__(self) -> Iterator[str | AppKey[Any]]:
        d: dict[str | AppKey[Any], Any] = {}
        for mapping in reversed(self._maps):
            # reuses stored hash values if possible
            d.update(mapping)
        return iter(d)

    def __contains__(self, key: object) -> bool:
        return any(key in m for m in self._maps)

    def __bool__(self) -> bool:
        return any(self._maps)

    def __repr__(self) -> str:
        content = ", ".join(map(repr, self._maps))
        return f"ChainMapProxy({content})"


# https://tools.ietf.org/html/rfc7232#section-2.3
_ETAGC = r"[!\x23-\x7E\x80-\xff]+"
_ETAGC_RE = re.compile(_ETAGC)
_QUOTED_ETAG = rf'(W/)?"({_ETAGC})"'
QUOTED_ETAG_RE = re.compile(_QUOTED_ETAG)
LIST_QUOTED_ETAG_RE = re.compile(rf"({_QUOTED_ETAG})(?:\s*,\s*|$)|(.)")

ETAG_ANY = "*"


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ETag:
    value: str
    is_weak: bool = False


def validate_etag_value(value: str) -> None:
    if value != ETAG_ANY and not _ETAGC_RE.fullmatch(value):
        raise ValueError(
            f"Value {value!r} is not a valid etag. Maybe it contains '\"'?"
        )


def parse_http_date(date_str: str | None) -> datetime.datetime | None:
    """Process a date string, return a datetime object"""
    if date_str is not None:
        timetuple = parsedate(date_str)
        if timetuple is not None:
            with suppress(ValueError):
                return datetime.datetime(*timetuple[:6], tzinfo=datetime.timezone.utc)
    return None


@functools.lru_cache
def must_be_empty_body(method: str, code: int) -> bool:
    """Check if a request must return an empty body."""
    return (
        code in EMPTY_BODY_STATUS_CODES
        or method in EMPTY_BODY_METHODS
        or (200 <= code < 300 and method == hdrs.METH_CONNECT)
    )


def should_remove_content_length(method: str, code: int) -> bool:
    """Check if a Content-Length header should be removed.

    This should always be a subset of must_be_empty_body
    """
    # https://www.rfc-editor.org/rfc/rfc9110.html#section-8.6-8
    # https://www.rfc-editor.org/rfc/rfc9110.html#section-15.4.5-4
    return code in EMPTY_BODY_STATUS_CODES or (
        200 <= code < 300 and method == hdrs.METH_CONNECT
    )
<|endoftext|>"""Low level HTTP server."""

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional  # noqa

from .abc import AbstractStreamWriter
from .http_parser import RawRequestMessage
from .streams import StreamReader
from .web_protocol import RequestHandler, _RequestFactory, _RequestHandler
from .web_request import BaseRequest

__all__ = ("Server",)


class Server:
    def __init__(
        self,
        handler: _RequestHandler,
        *,
        request_factory: _RequestFactory | None = None,
        handler_cancellation: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs: Any,
    ) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._connections: dict[RequestHandler, asyncio.Transport] = {}
        self._kwargs = kwargs
        # requests_count is the number of requests being processed by the server
        # for the lifetime of the server.
        self.requests_count = 0
        self.request_handler = handler
        self.request_factory = request_factory or self._make_request
        self.handler_cancellation = handler_cancellation

    @property
    def connections(self) -> list[RequestHandler]:
        return list(self._connections.keys())

    def connection_made(
        self, handler: RequestHandler, transport: asyncio.Transport
    ) -> None:
        self._connections[handler] = transport

    def connection_lost(
        self, handler: RequestHandler, exc: BaseException | None = None
    ) -> None:
        if handler in self._connections:
            if handler._task_handler:
                handler._task_handler.add_done_callback(
                    lambda f: self._connections.pop(handler, None)
                )
            else:
                del self._connections[handler]

    def _make_request(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: RequestHandler,
        writer: AbstractStreamWriter,
        task: "asyncio.Task[None]",
    ) -> BaseRequest:
        return BaseRequest(message, payload, protocol, writer, task, self._loop)

    def pre_shutdown(self) -> None:
        for conn in self._connections:
            conn.close()

    async def shutdown(self, timeout: float | None = None) -> None:
        coros = (conn.shutdown(timeout) for conn in self._connections)
        await asyncio.gather(*coros)
        self._connections.clear()

    def __call__(self) -> RequestHandler:
        try:
            return RequestHandler(self, loop=self._loop, **self._kwargs)
        except TypeError:
            # Failsafe creation: remove all custom handler_args
            kwargs = {
                k: v
                for k, v in self._kwargs.items()
                if k in ["debug", "access_log_class"]
            }
            handler = RequestHandler(self, loop=self._loop, **kwargs)
            handler.logger.warning(
                "Failed to create request handler with custom kwargs %r, "
                "falling back to filtered kwargs. This may indicate a "
                "misconfiguration.",
                self._kwargs,
            )
            return handler
<|endoftext|>import asyncio
import asyncio.streams
import sys
import traceback
import warnings
from asyncio.base_events import BaseEventLoop
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from html import escape as html_escape
from http import HTTPStatus
from logging import Logger
from typing import TYPE_CHECKING, Any, Optional, cast

import attr
import yarl
from propcache import under_cached_property

from .abc import AbstractAccessLogger, AbstractStreamWriter
from .base_protocol import BaseProtocol
from .helpers import DEFAULT_CHUNK_SIZE, ceil_timeout
from .http import (
    HttpProcessingError,
    HttpRequestParser,
    HttpVersion10,
    RawRequestMessage,
    StreamWriter,
    WebSocketReader,
)
from .http_exceptions import BadHttpMethod
from .log import access_logger, server_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .tcp_helpers import tcp_keepalive
from .web_exceptions import HTTPException, HTTPInternalServerError
from .web_log import AccessLogger
from .web_request import BaseRequest
from .web_response import Response, StreamResponse

__all__ = ("RequestHandler", "RequestPayloadError", "PayloadAccessError")

# Max parsed-but-unhandled pipelined requests buffered per connection before
# reading is paused. Bounds memory a client can pin by keeping one handler busy
# and pipelining behind it; reading resumes as the queue drains.
MAX_MSG_QUEUE_SIZE = 32

if TYPE_CHECKING:
    import ssl

    from .web_server import Server


_RequestFactory = Callable[
    [
        RawRequestMessage,
        StreamReader,
        "RequestHandler",
        AbstractStreamWriter,
        "asyncio.Task[None]",
    ],
    BaseRequest,
]

_RequestHandler = Callable[[BaseRequest], Awaitable[StreamResponse]]

ERROR = RawRequestMessage(
    "UNKNOWN",
    "/",
    HttpVersion10,
    {},  # type: ignore[arg-type]
    {},  # type: ignore[arg-type]
    True,
    None,
    False,
    False,
    yarl.URL("/"),
)


class RequestPayloadError(Exception):
    """Payload parsing error."""


class PayloadAccessError(Exception):
    """Payload was accessed after response was sent."""


_PAYLOAD_ACCESS_ERROR = PayloadAccessError()


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _ErrInfo:
    status: int
    exc: BaseException
    message: str


_MsgType = tuple[RawRequestMessage | _ErrInfo, StreamReader]


class RequestHandler(BaseProtocol):
    """HTTP protocol implementation.

    RequestHandler handles incoming HTTP request. It reads request line,
    request headers and request payload and calls handle_request() method.
    By default it always returns with 404 response.

    RequestHandler handles errors in incoming request, like bad
    status line, bad headers or incomplete payload. If any error occurs,
    connection gets closed.

    keepalive_timeout -- number of seconds before closing an idle connection.

    tcp_keepalive -- TCP keep-alive is on, default is on

    debug -- enable debug mode

    logger -- custom logger object

    access_log_class -- custom class for access_logger

    access_log -- custom logging object

    access_log_format -- access log format string

    loop -- Optional event loop

    max_line_size -- Optional maximum header line size

    max_field_size -- Optional maximum header field size

    max_headers -- Optional maximum header size

    timeout_ceil_threshold -- Optional value to specify
                              threshold to ceil() timeout
                              values

    """

    __slots__ = (
        "max_field_size",
        "max_headers",
        "max_line_size",
        "_request_count",
        "_keepalive",
        "_manager",
        "_request_handler",
        "_request_factory",
        "_tcp_keepalive",
        "_next_keepalive_close_time",
        "_keepalive_handle",
        "_keepalive_timeout",
        "_lingering_time",
        "_messages",
        "_max_msg_queue_size",
        "_msg_queue_resume_size",
        "_message_tail",
        "_read_bufsize",
        "_handler_waiter",
        "_waiter",
        "_task_handler",
        "_payload_parser",
        "_data_received_cb",
        "logger",
        "debug",
        "access_log",
        "access_logger",
        "_close",
        "_force_close",
        "_current_request",
        "_timeout_ceil_threshold",
        "_request_in_progress",
        "_logging_enabled",
        "_cache",
    )

    def __init__(
        self,
        manager: "Server",
        *,
        loop: asyncio.AbstractEventLoop,
        # Default should be high enough that it's likely longer than a reverse proxy.
        keepalive_timeout: float = 3630,
        tcp_keepalive: bool = True,
        logger: Logger = server_logger,
        access_log_class: type[AbstractAccessLogger] = AccessLogger,
        access_log: Logger = access_logger,
        access_log_format: str = AccessLogger.LOG_FORMAT,
        debug: bool = False,
        max_line_size: int = 8190,
        max_headers: int = 128,
        max_field_size: int = 8190,
        lingering_time: float = 10.0,
        read_bufsize: int = DEFAULT_CHUNK_SIZE,
        auto_decompress: bool = True,
        timeout_ceil_threshold: float = 5,
    ):
        self._max_msg_queue_size = MAX_MSG_QUEUE_SIZE
        # Low-water mark: resume reading once the queue drains to half the limit
        # so we refill in batches instead of churning pause/resume per request.
        self._msg_queue_resume_size = MAX_MSG_QUEUE_SIZE // 2
        self._read_bufsize = read_bufsize
        parser = HttpRequestParser(
            self,
            loop,
            read_bufsize,
            max_line_size=max_line_size,
            max_field_size=max_field_size,
            max_headers=max_headers,
            payload_exception=RequestPayloadError,
            auto_decompress=auto_decompress,
            max_msg_queue_size=MAX_MSG_QUEUE_SIZE,
        )
        super().__init__(loop, parser)

        # _request_count is the number of requests processed with the same connection.
        self._request_count = 0
        self._keepalive = True
        self._current_request: BaseRequest | None = None
        self._manager: Server | None = manager
        self._request_handler: _RequestHandler | None = manager.request_handler
        self._request_factory: _RequestFactory | None = manager.request_factory

        self.max_line_size = max_line_size
        self.max_headers = max_headers
        self.max_field_size = max_field_size

        self._tcp_keepalive = tcp_keepalive
        # placeholder to be replaced on keepalive timeout setup
        self._next_keepalive_close_time = 0.0
        self._keepalive_handle: asyncio.Handle | None = None
        self._keepalive_timeout = keepalive_timeout
        self._lingering_time = float(lingering_time)

        self._messages: deque[_MsgType] = deque()
        self._message_tail = b""
        self._data_received_cb: Callable[[], None] | None = None

        self._waiter: asyncio.Future[None] | None = None
        self._handler_waiter: asyncio.Future[None] | None = None
        self._task_handler: asyncio.Task[None] | None = None
        self._payload_parser: Any = None

        self._timeout_ceil_threshold: float = 5
        try:
            self._timeout_ceil_threshold = float(timeout_ceil_threshold)
        except (TypeError, ValueError):
            pass

        self.logger = logger
        self.debug = debug
        self.access_log = access_log
        if access_log:
            self.access_logger: AbstractAccessLogger | None = access_log_class(
                access_log, access_log_format
            )
            self._logging_enabled = self.access_logger.enabled
        else:
            self.access_logger = None
            self._logging_enabled = False

        self._close = False
        self._force_close = False
        self._request_in_progress = False
        self._cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        return "<{} {}>".format(
            self.__class__.__name__,
            "connected" if self.transport is not None else "disconnected",
        )

    @under_cached_property
    def ssl_context(self) -> Optional["ssl.SSLContext"]:
        """Return SSLContext if available."""
        return (
            None
            if self.transport is None
            else self.transport.get_extra_info("sslcontext")
        )

    @under_cached_property
    def peername(
        self,
    ) -> str | tuple[str, int, int, int] | tuple[str, int] | None:
        """Return peername if available."""
        return (
            None
            if self.transport is None
            else self.transport.get_extra_info("peername")
        )

    @under_cached_property
    def sockname(
        self,
    ) -> str | tuple[str, int, int, int] | tuple[str, int] | None:
        """Return sockname if available."""
        return (
            None
            if self.transport is None
            else self.transport.get_extra_info("sockname")
        )

    @property
    def keepalive_timeout(self) -> float:
        return self._keepalive_timeout

    async def shutdown(self, timeout: float | None = 15.0) -> None:
        """Do worker process exit preparations.

        We need to clean up everything and stop accepting requests.
        It is especially important for keep-alive connections.
        """
        self._force_close = True

        if self._keepalive_handle is not None:
            self._keepalive_handle.cancel()

        # Wait for graceful handler completion
        if self._request_in_progress:
            # The future is only created when we are shutting
            # down while the handler is still processing a request
            # to avoid creating a future for every request.
            self._handler_waiter = self._loop.create_future()
            try:
                async with ceil_timeout(timeout):
                    await self._handler_waiter
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._handler_waiter = None
                if (
                    sys.version_info >= (3, 11)
                    and (task := asyncio.current_task())
                    and task.cancelling()
                ):
                    raise
        # Then cancel handler and wait
        try:
            async with ceil_timeout(timeout):
                if self._current_request is not None:
                    self._current_request._cancel(asyncio.CancelledError())

                if self._task_handler is not None and not self._task_handler.done():
                    await asyncio.shield(self._task_handler)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            if (
                sys.version_info >= (3, 11)
                and (task := asyncio.current_task())
                and task.cancelling()
            ):
                raise

        # force-close non-idle handler
        if self._task_handler is not None:
            self._task_handler.cancel()

        self.force_close()

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        super().connection_made(transport)

        real_transport = cast(asyncio.Transport, transport)
        if self._tcp_keepalive:
            tcp_keepalive(real_transport)

        assert self._manager is not None
        self._manager.connection_made(self, real_transport)

        loop = self._loop
        # Need to enable keepalive timeout at start of connection, as there's no other
        # protection against a dead connection that doesn't send a request at all.
        if self._keepalive_timeout > 0:
            close_time = loop.time() + self._keepalive_timeout
            self._next_keepalive_close_time = close_time
            self._keepalive_handle = loop.call_at(close_time, self._process_keepalive)

        if sys.version_info >= (3, 14):
            if isinstance(loop, BaseEventLoop):
                task = asyncio.create_task(self.start(), eager_start=True)
            else:
                task = asyncio.Task(self.start(), loop=loop, eager_start=True)
        elif sys.version_info >= (3, 12):
            task = asyncio.Task(self.start(), loop=loop, eager_start=True)
        else:
            task = loop.create_task(self.start())
        self._task_handler = task

    def connection_lost(self, exc: BaseException | None) -> None:
        if self._manager is None:
            return
        self._manager.connection_lost(self, exc)

        # Grab value before setting _manager to None.
        handler_cancellation = self._manager.handler_cancellation

        self.force_close()
        super().connection_lost(exc)
        self._manager = None
        self._request_factory = None
        self._request_handler = None
        self._parser = None

        if self._keepalive_handle is not None:
            self._keepalive_handle.cancel()

        if self._current_request is not None:
            if exc is None:
                exc = ConnectionResetError("Connection lost")
            self._current_request._cancel(exc)

        if handler_cancellation and self._task_handler is not None:
            self._task_handler.cancel()

        self._task_handler = None

        if self._payload_parser is not None:
            self._payload_parser.feed_eof()
            self._payload_parser = None

    def set_parser(
        self,
        parser: WebSocketReader,
        data_received_cb: Callable[[], None] | None = None,
    ) -> None:
        assert self._payload_parser is None

        self._payload_parser = parser
        self._data_received_cb = data_received_cb

        if self._message_tail:
            self._payload_parser.feed_data(self._message_tail)
            self._message_tail = b""

        if self._buffer_paused:
            self._resume_reading_if_drained()

    def eof_received(self) -> None:
        pass

    def data_received(self, data: bytes) -> None:
        if self._force_close or self._close:
            return
        # parse http messages
        messages: Sequence[_MsgType]
        if self._payload_parser is None and not self._upgraded:
            assert self._parser is not None
            try:
                messages, upgraded, tail = self._parser.feed_data(data)
            except HttpProcessingError as exc:
                messages = [
                    (_ErrInfo(status=400, exc=exc, message=exc.message), EMPTY_PAYLOAD)
                ]
                upgraded = False
                tail = b""

            for msg, payload in messages:
                self._request_count += 1
                self._messages.append((msg, payload))

            waiter = self._waiter
            if messages and waiter is not None and not waiter.done():
                # don't set result twice
                waiter.set_result(None)

            # Queue full: pause the transport (the parser already stopped
            # emitting). start() resumes as it drains the queue.
            if (
                not self._buffer_paused
                and len(self._messages) >= self._max_msg_queue_size
            ):
                self._pause_reading_for_buffer()

            self._upgraded = upgraded
            if upgraded and tail:
                self._message_tail = tail

        # no parser, just store
        elif self._payload_parser is None and self._upgraded and data:
            self._message_tail += data
            if (
                not self._buffer_paused
                and len(self._message_tail) >= self._read_bufsize
            ):
                self._pause_reading_for_buffer()

        # feed payload
        elif data:
            if self._data_received_cb is not None:
                self._data_received_cb()
            eof, tail = self._payload_parser.feed_data(data)
            if eof:
                self.close()

    def _resume_reading_if_drained(self) -> None:
        # Tested empty-first so a read_bufsize of 0 cannot wedge the connection.
        if self._message_tail and len(self._message_tail) >= self._read_bufsize:
            return

        if not self._upgraded:
            # Reparse buffered pipelined requests while still marked paused so
            # a refill past the limit does not re-pause an already-paused
            # transport; only resume below once it stayed under the limit.
            self.data_received(b"")
            if len(self._messages) >= self._max_msg_queue_size:
                return
        self._resume_reading_for_buffer()

    def keep_alive(self, val: bool) -> None:
        """Set keep-alive connection mode.

        :param bool val: new state.
        """
        self._keepalive = val
        if self._keepalive_handle:
            self._keepalive_handle.cancel()
            self._keepalive_handle = None

    def close(self) -> None:
        """Close connection.

        Stop accepting new pipelining messages and close
        connection when handlers done processing messages.
        """
        self._close = True
        if self._waiter:
            self._waiter.cancel()

    def force_close(self) -> None:
        """Forcefully close connection."""
        self._force_close = True
        if self._waiter:
            self._waiter.cancel()
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    def log_access(
        self, request: BaseRequest, response: StreamResponse, time: float | None
    ) -> None:
        if self._logging_enabled and self.access_logger is not None:
            if TYPE_CHECKING:
                assert time is not None
            self.access_logger.log(request, response, self._loop.time() - time)

    def log_debug(self, *args: Any, **kw: Any) -> None:
        if self.debug:
            self.logger.debug(*args, **kw)

    def log_exception(self, *args: Any, **kw: Any) -> None:
        self.logger.exception(*args, **kw)

    def _process_keepalive(self) -> None:
        self._keepalive_handle = None
        if self._force_close or not self._keepalive:
            return

        loop = self._loop
        now = loop.time()
        close_time = self._next_keepalive_close_time
        if now < close_time:
            # Keep alive close check fired too early, reschedule
            self._keepalive_handle = loop.call_at(close_time, self._process_keepalive)
            return

        # handler in idle state
        if self._waiter and not self._waiter.done():
            self.force_close()

    async def _handle_request(
        self,
        request: BaseRequest,
        start_time: float | None,
        request_handler: Callable[[BaseRequest], Awaitable[StreamResponse]],
    ) -> tuple[StreamResponse, bool]:
        self._request_in_progress = True
        try:
            try:
                self._current_request = request
                resp = await request_handler(request)
            finally:
                self._current_request = None
        except HTTPException as exc:
            resp = exc
            resp, reset = await self.finish_response(request, resp, start_time)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as exc:
            self.log_debug("Request handler timed out.", exc_info=exc)
            resp = self.handle_error(request, 504)
            resp, reset = await self.finish_response(request, resp, start_time)
        except Exception as exc:
            resp = self.handle_error(request, 500, exc)
            resp, reset = await self.finish_response(request, resp, start_time)
        else:
            # Deprecation warning (See #2415)
            if getattr(resp, "__http_exception__", False):
                warnings.warn(
                    "returning HTTPException object is deprecated "
                    "(#2415) and will be removed, "
                    "please raise the exception instead",
                    DeprecationWarning,
                )

            resp, reset = await self.finish_response(request, resp, start_time)
        finally:
            self._request_in_progress = False
            if self._handler_waiter is not None:
                self._handler_waiter.set_result(None)

        return resp, reset

    async def start(self) -> None:
        """Process incoming request.

        It reads request line, request headers and request payload, then
        calls handle_request() method. Subclass has to override
        handle_request(). start() handles various exceptions in request
        or response handling. Connection is being closed always unless
        keep_alive(True) specified.
        """
        loop = self._loop
        manager = self._manager
        assert manager is not None
        keepalive_timeout = self._keepalive_timeout
        resp = None
        assert self._request_factory is not None
        assert self._request_handler is not None

        while not self._force_close:
            if not self._messages:
                try:
                    # wait for next request
                    self._waiter = loop.create_future()
                    await self._waiter
                finally:
                    self._waiter = None

            message, payload = self._messages.popleft()

            # Free a parser slot; resume reading once drained to low water so
            # pipelining keeps flowing while this request is handled.
            # no branch: _parser is only None after connection_lost, whose path
            # exits this loop, so the None case is not reachably exercisable.
            if self._parser is not None:  # pragma: no branch
                self._parser.message_consumed()
            if (
                self._buffer_paused
                and len(self._messages) <= self._msg_queue_resume_size
            ):
                self._resume_reading_if_drained()

            # time is only fetched if logging is enabled as otherwise
            # its thrown away and never used.
            start = loop.time() if self._logging_enabled else None

            manager.requests_count += 1
            writer = StreamWriter(self, loop)
            if isinstance(message, _ErrInfo):
                # make request_factory work
                request_handler = self._make_error_handler(message)
                message = ERROR
            else:
                request_handler = self._request_handler

            # Important don't hold a reference to the current task
            # as on traceback it will prevent the task from being
            # collected and will cause a memory leak.
            request = self._request_factory(
                message,
                payload,
                self,
                writer,
                self._task_handler or asyncio.current_task(loop),  # type: ignore[arg-type]
            )
            try:
                # a new task is used for copy context vars (#3406)
                coro = self._handle_request(request, start, request_handler)
                if sys.version_info >= (3, 14):
                    if isinstance(loop, BaseEventLoop):
                        task = asyncio.create_task(coro, eager_start=True)
                    else:
                        task = asyncio.Task(coro, loop=loop, eager_start=True)
                elif sys.version_info >= (3, 12):
                    task = asyncio.Task(coro, loop=loop, eager_start=True)
                else:
                    task = loop.create_task(coro)
                try:
                    resp, reset = await task
                except ConnectionError:
                    self.log_debug("Ignored premature client disconnection")
                    break

                # Drop the processed task from asyncio.Task.all_tasks() early
                del task
                if reset:
                    self.log_debug("Ignored premature client disconnection 2")
                    break

                # notify server about keep-alive
                self._keepalive = bool(resp.keep_alive)

                # check payload
                if not payload.is_eof():
                    lingering_time = self._lingering_time
                    if not self._force_close and lingering_time:
                        self.log_debug(
                            "Start lingering close timer for %s sec.", lingering_time
                        )

                        now = loop.time()
                        end_t = now + lingering_time

                        try:
                            while not payload.is_eof() and now < end_t:
                                async with ceil_timeout(end_t - now):
                                    # read and ignore
                                    await payload.readany()
                                now = loop.time()
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            if (
                                sys.version_info >= (3, 11)
                                and (t := asyncio.current_task())
                                and t.cancelling()
                            ):
                                raise

                    # if payload still uncompleted
                    if not payload.is_eof() and not self._force_close:
                        self.log_debug("Uncompleted request.")
                        self.close()

                payload.set_exception(_PAYLOAD_ACCESS_ERROR)

            except asyncio.CancelledError:
                self.log_debug("Ignored premature client disconnection")
                self.force_close()
                raise
            except Exception as exc:
                self.log_exception("Unhandled exception", exc_info=exc)
                self.force_close()
            except BaseException:
                self.force_close()
                raise
            finally:
                request._task = None  # type: ignore[assignment] # Break reference cycle in case of exception
                if self.transport is None and resp is not None:
                    self.log_debug("Ignored premature client disconnection.")

            if self._keepalive and not self._close and not self._force_close:
                # start keep-alive timer
                close_time = loop.time() + keepalive_timeout
                self._next_keepalive_close_time = close_time
                if self._keepalive_handle is None:
                    self._keepalive_handle = loop.call_at(
                        close_time, self._process_keepalive
                    )
            else:
                break

        # remove handler, close transport if no handlers left
        if not self._force_close:
            self._task_handler = None
            if self.transport is not None:
                self.transport.close()

    async def finish_response(
        self, request: BaseRequest, resp: StreamResponse, start_time: float | None
    ) -> tuple[StreamResponse, bool]:
        """Prepare the response and write_eof, then log access.

        This has to
        be called within the context of any exception so the access logger
        can get exception information. Returns True if the client disconnects
        prematurely.
        """
        request._finish()

        # Handle feeding the message tail following an upgrade request that
        # was declined.
        # The upgrade request is the last request before the parser paused,
        # so wait for self._messages to be empty.
        # payload_parser is not None if the upgrade was accepted.
        if (
            self._upgraded
            and not self._messages
            and self._payload_parser is None
            and self._parser is not None
        ):
            self._parser.set_upgraded(False)
            self._upgraded = False
            if self._message_tail:
                messages: Sequence[_MsgType]
                try:
                    messages, upgraded, tail = self._parser.feed_data(
                        self._message_tail
                    )
                except HttpProcessingError as parse_exc:
                    # Garbage (or an oversized request line) buffered behind the
                    # upgrade: answer 400 instead of letting the error escape
                    # and lose this response, like data_received() does.
                    messages = [
                        (
                            _ErrInfo(
                                status=400,
                                exc=parse_exc,
                                message=parse_exc.message,
                            ),
                            EMPTY_PAYLOAD,
                        )
                    ]
                    upgraded = False
                    tail = b""
                # A further upgrade request in the tail buffers its own remainder.
                self._upgraded = upgraded
                self._message_tail = tail
                for msg, payload in messages:
                    self._request_count += 1
                    self._messages.append((msg, payload))
                if len(self._messages) >= self._max_msg_queue_size:
                    # Pause the transport, like in data_received().
                    self._pause_reading_for_buffer()
                elif self._buffer_paused:
                    # Resume reading now the tail has been parsed.
                    self._resume_reading_if_drained()
                # This shouldn't be possible. If a future refactor results in this
                # failing, then the code may need to be updated to set the waiter.
                assert self._waiter is None
        try:
            prepare_meth = resp.prepare
        except AttributeError:
            if resp is None:
                self.log_exception("Missing return statement on request handler")
            else:
                self.log_exception(
                    f"Web-handler should return a response instance, got {resp!r}"
                )
            exc = HTTPInternalServerError()
            resp = Response(
                status=exc.status, reason=exc.reason, text=exc.text, headers=exc.headers
            )
            prepare_meth = resp.prepare
        try:
            await prepare_meth(request)
            await resp.write_eof()
        except ConnectionError:
            self.log_access(request, resp, start_time)
            return resp, True

        self.log_access(request, resp, start_time)
        return resp, False

    def handle_error(
        self,
        request: BaseRequest,
        status: int = 500,
        exc: BaseException | None = None,
        message: str | None = None,
    ) -> StreamResponse:
        """Handle errors.

        Returns HTTP response with specific status code. Logs additional
        information. It always closes current connection.
        """
        if self._request_count == 1 and isinstance(exc, BadHttpMethod):
            # BadHttpMethod is common when a client sends non-HTTP
            # or encrypted traffic to an HTTP port. This is expected
            # to happen when connected to the public internet so we log
            # it at the debug level as to not fill logs with noise.
            self.logger.debug(
                "Error handling request from %s", request.remote, exc_info=exc
            )
        else:
            self.log_exception(
                "Error handling request from %s", request.remote, exc_info=exc
            )

        # some data already got sent, connection is broken
        if request.writer.output_size > 0:
            raise ConnectionError(
                "Response is sent already, cannot send another response "
                "with the error message"
            )

        ct = "text/plain"
        if status == HTTPStatus.INTERNAL_SERVER_ERROR:
            title = f"{HTTPStatus.INTERNAL_SERVER_ERROR.value} {HTTPStatus.INTERNAL_SERVER_ERROR.phrase}"
            msg = HTTPStatus.INTERNAL_SERVER_ERROR.description
            tb = None
            if self.debug:
                with suppress(Exception):
                    tb = traceback.format_exc()

            if "text/html" in request.headers.get("Accept", ""):
                if tb:
                    tb = html_escape(tb)
                    msg = f"<h2>Traceback:</h2>\n<pre>{tb}</pre>"
                message = (
                    "<html><head>"
                    f"<title>{title}</title>"
                    f"</head><body>\n<h1>{title}</h1>"
                    f"\n{msg}\n</body></html>\n"
                )
                ct = "text/html"
            else:
                if tb:
                    msg = tb
                message = title + "\n\n" + msg

        resp = Response(status=status, text=message, content_type=ct)
        resp.force_close()

        return resp

    def _make_error_handler(
        self, err_info: _ErrInfo
    ) -> Callable[[BaseRequest], Awaitable[StreamResponse]]:
        async def handler(request: BaseRequest) -> StreamResponse:
            return self.handle_error(
                request, err_info.status, err_info.exc, err_info.message
            )

        return handler
<|endoftext|>import re
from typing import TYPE_CHECKING, TypeVar

from .typedefs import Handler, Middleware
from .web_exceptions import HTTPMove, HTTPPermanentRedirect
from .web_request import Request
from .web_response import StreamResponse
from .web_urldispatcher import SystemRoute

__all__ = (
    "middleware",
    "normalize_path_middleware",
)

if TYPE_CHECKING:
    from .web_app import Application

_Func = TypeVar("_Func")


async def _check_request_resolves(request: Request, path: str) -> tuple[bool, Request]:
    alt_request = request.clone(rel_url=path)

    match_info = await request.app.router.resolve(alt_request)
    alt_request._match_info = match_info

    if match_info.http_exception is None:
        return True, alt_request

    return False, request


def middleware(f: _Func) -> _Func:
    f.__middleware_version__ = 1  # type: ignore[attr-defined]
    return f


def normalize_path_middleware(
    *,
    append_slash: bool = True,
    remove_slash: bool = False,
    merge_slashes: bool = True,
    redirect_class: type[HTTPMove] = HTTPPermanentRedirect,
) -> Middleware:
    """Factory for producing a middleware that normalizes the path of a request.

    Normalizing means:
        - Add or remove a trailing slash to the path.
        - Double slashes are replaced by one.

    The middleware returns as soon as it finds a path that resolves
    correctly. The order if both merge and append/remove are enabled is
        1) merge slashes
        2) append/remove slash
        3) both merge slashes and append/remove slash.
    If the path resolves with at least one of those conditions, it will
    redirect to the new path.

    Only one of `append_slash` and `remove_slash` can be enabled. If both
    are `True` the factory will raise an assertion error

    If `append_slash` is `True` the middleware will append a slash when
    needed. If a resource is defined with trailing slash and the request
    comes without it, it will append it automatically.

    If `remove_slash` is `True`, `append_slash` must be `False`. When enabled
    the middleware will remove trailing slashes and redirect if the resource
    is defined

    If merge_slashes is True, merge multiple consecutive slashes in the
    path into one.
    """
    correct_configuration = not (append_slash and remove_slash)
    assert correct_configuration, "Cannot both remove and append slash"

    @middleware
    async def impl(request: Request, handler: Handler) -> StreamResponse:
        if isinstance(request.match_info.route, SystemRoute):
            paths_to_check = []
            if "?" in request.raw_path:
                path, query = request.raw_path.split("?", 1)
                query = "?" + query
            else:
                query = ""
                path = request.raw_path

            if merge_slashes:
                paths_to_check.append(re.sub("//+", "/", path))
            if append_slash and not request.path.endswith("/"):
                paths_to_check.append(path + "/")
            if remove_slash and request.path.endswith("/"):
                paths_to_check.append(path[:-1])
            if merge_slashes and append_slash:
                paths_to_check.append(re.sub("//+", "/", path + "/"))
            if merge_slashes and remove_slash:
                merged_slashes = re.sub("//+", "/", path)
                paths_to_check.append(merged_slashes[:-1])

            for path in paths_to_check:
                path = re.sub("^//+", "/", path)  # SECURITY: GHSA-v6wp-4m6f-gcjg
                resolves, request = await _check_request_resolves(request, path)
                if resolves:
                    raise redirect_class(request.raw_path + query)

        return await handler(request)

    return impl


def _fix_request_current_app(app: "Application") -> Middleware:
    @middleware
    async def impl(request: Request, handler: Handler) -> StreamResponse:
        match_info = request.match_info
        prev = match_info.current_app
        match_info.current_app = app
        try:
            return await handler(request)
        finally:
            match_info.current_app = prev

    return impl
<|endoftext|>import abc
import os  # noqa
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Union, overload

import attr

from . import hdrs
from .abc import AbstractView
from .typedefs import Handler, PathLike

if TYPE_CHECKING:
    from .web_request import Request
    from .web_response import StreamResponse
    from .web_urldispatcher import AbstractRoute, UrlDispatcher
else:
    Request = StreamResponse = UrlDispatcher = AbstractRoute = None


__all__ = (
    "AbstractRouteDef",
    "RouteDef",
    "StaticDef",
    "RouteTableDef",
    "head",
    "options",
    "get",
    "post",
    "patch",
    "put",
    "delete",
    "route",
    "view",
    "static",
)


class AbstractRouteDef(abc.ABC):
    __slots__ = ()

    @abc.abstractmethod
    def register(self, router: UrlDispatcher) -> list[AbstractRoute]:
        pass  # pragma: no cover


_HandlerType = Union[type[AbstractView], Handler]


@attr.s(auto_attribs=True, frozen=True, repr=False, slots=True)
class RouteDef(AbstractRouteDef):
    method: str
    path: str
    handler: _HandlerType
    kwargs: dict[str, Any]

    def __repr__(self) -> str:
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f", {name}={value!r}")
        return "<RouteDef {method} {path} -> {handler.__name__!r}{info}>".format(
            method=self.method, path=self.path, handler=self.handler, info="".join(info)
        )

    def register(self, router: UrlDispatcher) -> list[AbstractRoute]:
        if self.method in hdrs.METH_ALL:
            reg = getattr(router, "add_" + self.method.lower())
            return [reg(self.path, self.handler, **self.kwargs)]
        else:
            return [
                router.add_route(self.method, self.path, self.handler, **self.kwargs)
            ]


@attr.s(auto_attribs=True, frozen=True, repr=False, slots=True)
class StaticDef(AbstractRouteDef):
    prefix: str
    path: PathLike
    kwargs: dict[str, Any]

    def __repr__(self) -> str:
        info = []
        for name, value in sorted(self.kwargs.items()):
            info.append(f", {name}={value!r}")
        return "<StaticDef {prefix} -> {path}{info}>".format(
            prefix=self.prefix, path=self.path, info="".join(info)
        )

    def register(self, router: UrlDispatcher) -> list[AbstractRoute]:
        resource = router.add_static(self.prefix, self.path, **self.kwargs)
        routes = resource.get_info().get("routes", {})
        return list(routes.values())


def route(method: str, path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    return RouteDef(method, path, handler, kwargs)


def head(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    return route(hdrs.METH_HEAD, path, handler, **kwargs)


def options(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    return route(hdrs.METH_OPTIONS, path, handler, **kwargs)


def get(
    path: str,
    handler: _HandlerType,
    *,
    name: str | None = None,
    allow_head: bool = True,
    **kwargs: Any,
) -> RouteDef:
    return route(
        hdrs.METH_GET, path, handler, name=name, allow_head=allow_head, **kwargs
    )


def post(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    return route(hdrs.METH_POST, path, handler, **kwargs)


def put(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    return route(hdrs.METH_PUT, path, handler, **kwargs)


def patch(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    return route(hdrs.METH_PATCH, path, handler, **kwargs)


def delete(path: str, handler: _HandlerType, **kwargs: Any) -> RouteDef:
    return route(hdrs.METH_DELETE, path, handler, **kwargs)


def view(path: str, handler: type[AbstractView], **kwargs: Any) -> RouteDef:
    return route(hdrs.METH_ANY, path, handler, **kwargs)


def static(prefix: str, path: PathLike, **kwargs: Any) -> StaticDef:
    return StaticDef(prefix, path, kwargs)


_Deco = Callable[[_HandlerType], _HandlerType]


class RouteTableDef(Sequence[AbstractRouteDef]):
    """Route definition table"""

    def __init__(self) -> None:
        self._items: list[AbstractRouteDef] = []

    def __repr__(self) -> str:
        return f"<RouteTableDef count={len(self._items)}>"

    @overload
    def __getitem__(self, index: int) -> AbstractRouteDef: ...

    @overload
    def __getitem__(self, index: slice) -> list[AbstractRouteDef]: ...

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self._items[index]

    def __iter__(self) -> Iterator[AbstractRouteDef]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def route(self, method: str, path: str, **kwargs: Any) -> _Deco:
        def inner(handler: _HandlerType) -> _HandlerType:
            self._items.append(RouteDef(method, path, handler, kwargs))
            return handler

        return inner

    def head(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_HEAD, path, **kwargs)

    def get(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_GET, path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_POST, path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_PUT, path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_PATCH, path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_DELETE, path, **kwargs)

    def options(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_OPTIONS, path, **kwargs)

    def view(self, path: str, **kwargs: Any) -> _Deco:
        return self.route(hdrs.METH_ANY, path, **kwargs)

    def static(self, prefix: str, path: PathLike, **kwargs: Any) -> None:
        self._items.append(StaticDef(prefix, path, kwargs))
<|endoftext|>"""Reader for WebSocket protocol versions 13 and 8."""

from typing import TYPE_CHECKING

from ..helpers import NO_EXTENSIONS

if TYPE_CHECKING or NO_EXTENSIONS:  # pragma: no cover
    from .reader_py import (
        WebSocketDataQueue as WebSocketDataQueuePython,
        WebSocketReader as WebSocketReaderPython,
    )

    WebSocketReader = WebSocketReaderPython
    WebSocketDataQueue = WebSocketDataQueuePython
else:
    try:
        from .reader_c import (  # type: ignore[import-not-found]
            WebSocketDataQueue as WebSocketDataQueueCython,
            WebSocketReader as WebSocketReaderCython,
        )

        WebSocketReader = WebSocketReaderCython
        WebSocketDataQueue = WebSocketDataQueueCython
    except ImportError:  # pragma: no cover
        from .reader_py import (
            WebSocketDataQueue as WebSocketDataQueuePython,
            WebSocketReader as WebSocketReaderPython,
        )

        WebSocketReader = WebSocketReaderPython
        WebSocketDataQueue = WebSocketDataQueuePython
<|endoftext|>"""Reader for WebSocket protocol versions 13 and 8."""

import asyncio
import builtins
import sys
import weakref
from collections import deque
from typing import Final

from ..base_protocol import BaseProtocol
from ..compression_utils import TooManyMembersError, ZLibDecompressor
from ..helpers import _EXC_SENTINEL, set_exception
from ..log import ws_logger
from ..streams import EofStream
from .helpers import UNPACK_CLOSE_CODE, UNPACK_LEN3, websocket_mask
from .models import (
    WS_DEFLATE_TRAILING,
    WebSocketError,
    WSCloseCode,
    WSMessage,
    WSMessageTextBytes,
    WSMsgType,
)

ALLOWED_CLOSE_CODES: set[int] = {int(i) for i in WSCloseCode}

# States for the reader, used to parse the WebSocket frame
# integer values are used so they can be cythonized
READ_HEADER = 1
READ_PAYLOAD_LENGTH = 2
READ_PAYLOAD_MASK = 3
READ_PAYLOAD = 4

# Largest declared payload length the reader can represent: the compiled
# reader stores it in a Py_ssize_t, which holds 2**31-1 on the 32-bit builds
# (the win32 and armv7l wheels) and 2**63-1 everywhere else.
# TODO: Remove when we drop 32 bit support (and from reader_c.pxd).
MAX_PAYLOAD_LEN = sys.maxsize

WS_MSG_TYPE_BINARY = WSMsgType.BINARY
WS_MSG_TYPE_TEXT = WSMsgType.TEXT

# WSMsgType values unpacked so they can by cythonized to ints
OP_CODE_NOT_SET = -1
OP_CODE_CONTINUATION = WSMsgType.CONTINUATION.value
OP_CODE_TEXT = WSMsgType.TEXT.value
OP_CODE_BINARY = WSMsgType.BINARY.value
OP_CODE_CLOSE = WSMsgType.CLOSE.value
OP_CODE_PING = WSMsgType.PING.value
OP_CODE_PONG = WSMsgType.PONG.value

EMPTY_FRAME_ERROR = (True, b"")
EMPTY_FRAME = (False, b"")

COMPRESSED_NOT_SET = -1
COMPRESSED_FALSE = 0
COMPRESSED_TRUE = 1

TUPLE_NEW = tuple.__new__

# Overhead added to each message to ensure that tiny messages can't use
# unreasonable amounts of memory.
MSG_SIZE_OVERHEAD: Final[int] = 128

STALLED_READER_COLLECTED: Final[str] = (
    "WebSocketReader was garbage collected while stalled; "
    "callers of set_parser() must hold a strong reference"
)

cython_int = int  # Typed to int in Python, but cython with use a signed int in the pxd


class WebSocketDataQueue:
    """WebSocketDataQueue resumes and pauses an underlying stream.

    It is a destination for WebSocket data.
    """

    def __init__(
        self, protocol: BaseProtocol, limit: int, *, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._size = 0
        self._protocol = protocol
        self._limit = limit * 2
        self._loop = loop
        self._eof = False
        self._waiter: asyncio.Future[None] | None = None
        self._exception: BaseException | None = None
        self._buffer: deque[tuple[WSMessage | WSMessageTextBytes, int]] = deque()
        self._get_buffer = self._buffer.popleft
        self._put_buffer = self._buffer.append
        self._stalled_reader: "weakref.ref[WebSocketReader] | None" = None

    def is_eof(self) -> bool:
        return self._eof

    def exception(self) -> BaseException | None:
        return self._exception

    def set_exception(
        self,
        exc: BaseException,
        exc_cause: builtins.BaseException = _EXC_SENTINEL,
    ) -> None:
        self._eof = True
        self._exception = exc
        if (waiter := self._waiter) is not None:
            self._waiter = None
            set_exception(waiter, exc, exc_cause)

    def _release_waiter(self) -> None:
        if (waiter := self._waiter) is None:
            return
        self._waiter = None
        if not waiter.done():
            waiter.set_result(None)

    def feed_eof(self) -> None:
        self._eof = True
        self._release_waiter()
        self._exception = None  # Break cyclic references

    def feed_data(
        self, data: "WSMessage | WSMessageTextBytes", size: "cython_int"
    ) -> None:
        self._size += size + MSG_SIZE_OVERHEAD
        self._put_buffer((data, size))
        self._release_waiter()
        if self._size > self._limit and not self._protocol._reading_paused:
            self._protocol.pause_reading()

    async def read(self) -> WSMessage | WSMessageTextBytes:
        if not self._buffer and not self._eof:
            assert not self._waiter
            self._waiter = self._loop.create_future()
            try:
                await self._waiter
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._waiter = None
                raise
        return self._read_from_buffer()

    def _read_from_buffer(self) -> WSMessage | WSMessageTextBytes:
        if self._buffer:
            data, size = self._get_buffer()
            self._size -= size + MSG_SIZE_OVERHEAD
            if self._stalled_reader is not None and self._size <= self._limit // 2:
                # Resume parsing once the queue drains to the low-water mark.
                # Each resume re-slices the parser's unparsed tail, so waiting
                # for headroom makes a drain cost one copy per batch of
                # messages instead of one per message.
                if (reader := self._stalled_reader()) is not None:
                    reader.feed_data(b"")
                else:
                    # The stash died with the reader. Deliver what was already
                    # queued, then surface the contract violation on the next
                    # read instead of hanging. Log as well, since a caller that
                    # stops reading early never sees the deferred exception. A
                    # real failure that was already recorded stays the reported
                    # cause.
                    self._stalled_reader = None
                    ws_logger.warning(STALLED_READER_COLLECTED)
                    if self._exception is None:
                        self.set_exception(RuntimeError(STALLED_READER_COLLECTED))
            # Resuming the transport while a stash remains would admit
            # another socket read into the tail for every couple of messages
            # drained, moving the memory bound from the queue into the tail.
            if (
                self._stalled_reader is None
                and self._size < self._limit
                and self._protocol._reading_paused
            ):
                self._protocol.resume_reading()
            return data
        if self._exception is not None:
            raise self._exception
        raise EofStream


class WebSocketReader:
    def __init__(
        self,
        queue: WebSocketDataQueue,
        max_msg_size: int,
        compress: bool,
        decode_text: bool,
    ) -> None:
        self.queue = queue
        self._max_msg_size = max_msg_size
        self._decode_text = decode_text
        # Parked on the queue while parsing is stalled; created once so
        # stalling does not allocate.
        self._weak_self = weakref.ref(self)

        self._exc: Exception | None = None
        self._partial = bytearray()
        self._state = READ_HEADER

        self._opcode: int = OP_CODE_NOT_SET
        self._frame_fin = False
        self._frame_opcode: int = OP_CODE_NOT_SET
        # Reads of an in-flight frame, joined once when it completes.
        self._payload_fragments: list[bytes] = []
        # Fold reads into _payload_buffer past this count to bound the object
        # count (bytes are bounded by max_msg_size).
        self._max_fragments = max(1024, max_msg_size // 256) if max_msg_size else 0
        self._payload_buffer = bytearray()
        self._frame_payload_len = 0

        self._tail: bytes = b""
        self._has_mask = False
        self._frame_mask: bytes | None = None
        self._payload_bytes_to_read = 0
        self._payload_len_flag = 0
        self._compressed: int = COMPRESSED_NOT_SET
        self._decompressobj: ZLibDecompressor | None = None
        self._compress = compress

    def feed_eof(self) -> None:
        self.queue.feed_eof()

    # data can be bytearray on Windows because proactor event loop uses bytearray
    # and asyncio types this to Union[bytes, bytearray, memoryview] so we need
    # coerce data to bytes if it is not
    def feed_data(self, data: bytes | bytearray | memoryview) -> tuple[bool, bytes]:
        if type(data) is not bytes:
            data = bytes(data)

        if self._exc is not None:
            return True, data

        try:
            self._feed_data(data)
        except Exception as exc:
            self._exc = exc
            set_exception(self.queue, exc)
            return EMPTY_FRAME_ERROR

        return EMPTY_FRAME

    def _handle_frame(
        self,
        fin: bool,
        opcode: int | cython_int,  # Union intended: Cython pxd uses C int
        payload: bytes | bytearray,
        compressed: int | cython_int,  # Union intended: Cython pxd uses C int
    ) -> None:
        msg: WSMessage
        if opcode in {OP_CODE_TEXT, OP_CODE_BINARY, OP_CODE_CONTINUATION}:
            # Validate continuation frames before processing
            if opcode == OP_CODE_CONTINUATION and self._opcode == OP_CODE_NOT_SET:
                raise WebSocketError(
                    WSCloseCode.PROTOCOL_ERROR,
                    "Continuation frame for non started message",
                )

            # load text/binary
            if not fin:
                # got partial frame payload
                if opcode != OP_CODE_CONTINUATION:
                    # https://datatracker.ietf.org/doc/html/rfc6455#section-5.4
                    if self._opcode != OP_CODE_NOT_SET:
                        raise WebSocketError(
                            WSCloseCode.PROTOCOL_ERROR,
                            "The opcode in non-fin frame is expected "
                            f"to be zero, got {opcode!r}",
                        )
                    self._opcode = opcode
                self._partial += payload
                return

            has_partial = bool(self._partial)
            if opcode == OP_CODE_CONTINUATION:
                opcode = self._opcode
                self._opcode = OP_CODE_NOT_SET
            # previous frame was non finished
            # we should get continuation opcode
            elif has_partial:
                raise WebSocketError(
                    WSCloseCode.PROTOCOL_ERROR,
                    "The opcode in non-fin frame is expected "
                    f"to be zero, got {opcode!r}",
                )

            assembled_payload: bytes | bytearray
            if has_partial:
                assembled_payload = self._partial + payload
                self._partial.clear()
            else:
                assembled_payload = payload

            # Decompress process must to be done after all packets
            # received.
            if compressed:
                if not self._decompressobj:
                    self._decompressobj = ZLibDecompressor(suppress_deflate_header=True)
                # XXX: It's possible that the zlib backend (isal is known to
                # do this, maybe others too?) will return max_length bytes,
                # but internally buffer more data such that the payload is
                # >max_length, so we return one extra byte and if we're able
                # to do that, then the message is too big.
                try:
                    payload_merged = self._decompressobj.decompress_sync(
                        assembled_payload + WS_DEFLATE_TRAILING,
                        (
                            self._max_msg_size + 1
                            if self._max_msg_size
                            else self._max_msg_size
                        ),
                    )
                except TooManyMembersError as exc:
                    raise WebSocketError(
                        WSCloseCode.MESSAGE_TOO_BIG,
                        "Compressed message has too many deflate members",
                    ) from exc
                if self._max_msg_size and len(payload_merged) > self._max_msg_size:
                    raise WebSocketError(
                        WSCloseCode.MESSAGE_TOO_BIG,
                        f"Decompressed message exceeds size limit {self._max_msg_size}",
                    )
            elif type(assembled_payload) is bytes:
                payload_merged = assembled_payload
            else:
                payload_merged = bytes(assembled_payload)

            if opcode == OP_CODE_TEXT:
                if self._decode_text:
                    try:
                        text = payload_merged.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise WebSocketError(
                            WSCloseCode.INVALID_TEXT, "Invalid UTF-8 text message"
                        ) from exc

                    # XXX: The Text and Binary messages here can be a performance
                    # bottleneck, so we use tuple.__new__ to improve performance.
                    # This is not type safe, but many tests should fail in
                    # test_client_ws_functional.py if this is wrong.
                    self.queue.feed_data(
                        TUPLE_NEW(WSMessage, (WS_MSG_TYPE_TEXT, text, "")),
                        len(payload_merged),
                    )
                else:
                    # Return raw bytes for TEXT messages when decode_text=False
                    self.queue.feed_data(
                        TUPLE_NEW(
                            WSMessageTextBytes, (WS_MSG_TYPE_TEXT, payload_merged, "")
                        ),
                        len(payload_merged),
                    )
            else:
                self.queue.feed_data(
                    TUPLE_NEW(WSMessage, (WS_MSG_TYPE_BINARY, payload_merged, "")),
                    len(payload_merged),
                )
        elif opcode == OP_CODE_CLOSE:
            if len(payload) >= 2:
                close_code = UNPACK_CLOSE_CODE(payload[:2])[0]
                if close_code < 3000 and close_code not in ALLOWED_CLOSE_CODES:
                    raise WebSocketError(
                        WSCloseCode.PROTOCOL_ERROR,
                        f"Invalid close code: {close_code}",
                    )
                try:
                    close_message = payload[2:].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise WebSocketError(
                        WSCloseCode.INVALID_TEXT, "Invalid UTF-8 text message"
                    ) from exc
                msg = TUPLE_NEW(WSMessage, (WSMsgType.CLOSE, close_code, close_message))
            elif payload:
                raise WebSocketError(
                    WSCloseCode.PROTOCOL_ERROR,
                    f"Invalid close frame: {fin} {opcode} {payload!r}",
                )
            else:
                msg = TUPLE_NEW(WSMessage, (WSMsgType.CLOSE, 0, ""))

            self.queue.feed_data(msg, 0)
        elif opcode == OP_CODE_PING:
            msg = TUPLE_NEW(WSMessage, (WSMsgType.PING, payload, ""))
            self.queue.feed_data(msg, len(payload))
        elif opcode == OP_CODE_PONG:
            msg = TUPLE_NEW(WSMessage, (WSMsgType.PONG, payload, ""))
            self.queue.feed_data(msg, len(payload))
        else:
            raise WebSocketError(
                WSCloseCode.PROTOCOL_ERROR, f"Unexpected opcode={opcode!r}"
            )

    def _feed_data(self, data: bytes) -> None:
        """Return the next frame from the socket."""
        self.queue._stalled_reader = None
        if self._tail:
            data, self._tail = self._tail + data, b""

        start_pos = 0
        data_len = len(data)
        data_cstr = data

        while True:
            if start_pos < data_len and self.queue._size > self.queue._limit:
                # Over the high-water mark with unparsed bytes left: stash the
                # remainder and stall. Gating on unparsed bytes keeps a read
                # that ended on a frame boundary from arming an empty stall,
                # which would hold the transport paused with nothing to drain.
                self.queue._stalled_reader = self._weak_self
                break

            # read header
            if self._state == READ_HEADER:
                if data_len - start_pos < 2:
                    break
                first_byte = data_cstr[start_pos]
                second_byte = data_cstr[start_pos + 1]
                start_pos += 2

                fin = (first_byte >> 7) & 1
                rsv1 = (first_byte >> 6) & 1
                rsv2 = (first_byte >> 5) & 1
                rsv3 = (first_byte >> 4) & 1
                opcode = first_byte & 0xF

                # frame-fin = %x0 ; more frames of this message follow
                #           / %x1 ; final frame of this message
                # frame-rsv1 = %x0 ;
                #    1 bit, MUST be 0 unless negotiated otherwise
                # frame-rsv2 = %x0 ;
                #    1 bit, MUST be 0 unless negotiated otherwise
                # frame-rsv3 = %x0 ;
                #    1 bit, MUST be 0 unless negotiated otherwise
                #
                # Remove rsv1 from this test for deflate development
                if rsv2 or rsv3 or (rsv1 and not self._compress):
                    raise WebSocketError(
                        WSCloseCode.PROTOCOL_ERROR,
                        "Received frame with non-zero reserved bits",
                    )

                if opcode not in {
                    OP_CODE_CONTINUATION,
                    OP_CODE_TEXT,
                    OP_CODE_BINARY,
                    OP_CODE_CLOSE,
                    OP_CODE_PING,
                    OP_CODE_PONG,
                }:
                    raise WebSocketError(
                        WSCloseCode.PROTOCOL_ERROR,
                        f"Unexpected opcode={opcode!r}",
                    )

                if opcode > 0x7 and fin == 0:
                    raise WebSocketError(
                        WSCloseCode.PROTOCOL_ERROR,
                        "Received fragmented control frame",
                    )

                has_mask = (second_byte >> 7) & 1
                length = second_byte & 0x7F

                # Control frames MUST have a payload
                # length of 125 bytes or less
                if opcode > 0x7 and length > 125:
                    raise WebSocketError(
                        WSCloseCode.PROTOCOL_ERROR,
                        "Control frame payload cannot be larger than 125 bytes",
                    )

                # Control frames (opcode > 0x7) may be interleaved between the
                # fragments of a data message, so they must not touch the
                # compression state. RSV1 on them is ignored for interop:
                # some clients set it, and they are never decompressed.
                # https://datatracker.ietf.org/doc/html/rfc6455#section-5.4
                # https://datatracker.ietf.org/doc/html/rfc7692#section-6.1
                if opcode <= 0x7:
                    # Set compress status if last package is FIN
                    # OR set compress status if this is first fragment
                    # Raise error if not first fragment with rsv1 = 0x1
                    if self._frame_fin or self._compressed == COMPRESSED_NOT_SET:
                        self._compressed = COMPRESSED_TRUE if rsv1 else COMPRESSED_FALSE
                    elif rsv1:
                        raise WebSocketError(
                            WSCloseCode.PROTOCOL_ERROR,
                            "Received frame with non-zero reserved bits",
                        )
                    self._frame_fin = bool(fin)

                self._frame_opcode = opcode
                self._has_mask = bool(has_mask)
                self._payload_len_flag = length
                self._state = READ_PAYLOAD_LENGTH

            # read payload length
            if self._state == READ_PAYLOAD_LENGTH:
                len_flag = self._payload_len_flag
                if len_flag == 126:
                    if data_len - start_pos < 2:
                        break
                    first_byte = data_cstr[start_pos]
                    second_byte = data_cstr[start_pos + 1]
                    start_pos += 2
                    self._payload_bytes_to_read = first_byte << 8 | second_byte
                elif len_flag > 126:
                    if data_len - start_pos < 8:
                        break
                    # The declared length is an unsigned 64-bit integer that
                    # does not necessarily fit _payload_bytes_to_read.
                    frame_len = UNPACK_LEN3(data, start_pos)[0]
                    if frame_len > MAX_PAYLOAD_LEN:
                        raise WebSocketError(
                            WSCloseCode.MESSAGE_TOO_BIG,
                            f"Message size {int(frame_len) + len(self._partial)} "
                            f"exceeds limit {self._max_msg_size or MAX_PAYLOAD_LEN}",
                        )
                    self._payload_bytes_to_read = frame_len
                    start_pos += 8
                else:
                    self._payload_bytes_to_read = len_flag

                # Reject oversized data frames before buffering any payload
                # bytes. Control frames are capped at 125 bytes (checked in
                # READ_HEADER) so only text/binary/continuation need this.
                if self._max_msg_size and self._frame_opcode in {
                    OP_CODE_TEXT,
                    OP_CODE_BINARY,
                    OP_CODE_CONTINUATION,
                }:
                    # partial_len declared in reader_c.pxd to keep it in C.
                    partial_len = len(self._partial)
                    # payload_bytes_to_read is a signed Py_ssize_t C value,
                    # use subtraction here to avoid an integer overflow.
                    if self._payload_bytes_to_read >= self._max_msg_size - partial_len:
                        raise WebSocketError(
                            WSCloseCode.MESSAGE_TOO_BIG,
                            f"Message size {int(self._payload_bytes_to_read) + partial_len} "
                            f"exceeds limit {self._max_msg_size}",
                        )

                self._state = READ_PAYLOAD_MASK if self._has_mask else READ_PAYLOAD

            # read payload mask
            if self._state == READ_PAYLOAD_MASK:
                if data_len - start_pos < 4:
                    break
                self._frame_mask = data_cstr[start_pos : start_pos + 4]
                start_pos += 4
                self._state = READ_PAYLOAD

            if self._state == READ_PAYLOAD:
                chunk_len = data_len - start_pos
                if self._payload_bytes_to_read >= chunk_len:
                    f_end_pos = data_len
                    self._payload_bytes_to_read -= chunk_len
                else:
                    f_end_pos = start_pos + self._payload_bytes_to_read
                    self._payload_bytes_to_read = 0

                had_fragments = self._frame_payload_len
                self._frame_payload_len += f_end_pos - start_pos
                f_start_pos = start_pos
                start_pos = f_end_pos

                if self._payload_bytes_to_read != 0:
                    if f_start_pos < f_end_pos:  # skip a header-only read
                        self._payload_fragments.append(data_cstr[f_start_pos:f_end_pos])
                    if (
                        self._max_fragments
                        and len(self._payload_fragments) > self._max_fragments
                    ):
                        # Fold to bound the object count. Not a pause: nothing
                        # resumes reading until the frame is queued.
                        self._payload_buffer += b"".join(self._payload_fragments)
                        self._payload_fragments.clear()
                    break

                payload: bytes | bytearray
                if had_fragments:
                    self._payload_fragments.append(data_cstr[f_start_pos:f_end_pos])
                    if self._payload_buffer:  # folded prefix
                        self._payload_buffer += b"".join(self._payload_fragments)
                        if self._has_mask:
                            assert self._frame_mask is not None
                            websocket_mask(self._frame_mask, self._payload_buffer)
                        payload = self._payload_buffer
                        self._payload_buffer = bytearray()  # detach; payload aliases it
                    elif self._has_mask:
                        assert self._frame_mask is not None
                        payload_bytearray = bytearray(b"".join(self._payload_fragments))
                        websocket_mask(self._frame_mask, payload_bytearray)
                        payload = payload_bytearray
                    else:
                        payload = b"".join(self._payload_fragments)
                    self._payload_fragments.clear()
                elif self._has_mask:
                    assert self._frame_mask is not None
                    payload_bytearray = data_cstr[f_start_pos:f_end_pos]  # type: ignore[assignment]
                    if type(payload_bytearray) is not bytearray:
                        # Cython will do the conversion for us
                        # but we need to do it for Python and we
                        # will always get here in Python
                        payload_bytearray = bytearray(payload_bytearray)
                    websocket_mask(self._frame_mask, payload_bytearray)
                    payload = payload_bytearray
                else:
                    payload = data_cstr[f_start_pos:f_end_pos]

                self._handle_frame(
                    self._frame_fin, self._frame_opcode, payload, self._compressed
                )
                self._frame_payload_len = 0
                self._state = READ_HEADER

        # XXX: Cython needs slices to be bounded, so we can't omit the slice end here.
        self._tail = data_cstr[start_pos:data_len] if start_pos < data_len else b""
<|endoftext|>"""Models for WebSocket protocol versions 13 and 8."""

import json
from collections.abc import Callable
from enum import IntEnum
from typing import Any, Final, NamedTuple, cast

WS_DEFLATE_TRAILING: Final[bytes] = bytes([0x00, 0x00, 0xFF, 0xFF])


class WSCloseCode(IntEnum):
    OK = 1000
    GOING_AWAY = 1001
    PROTOCOL_ERROR = 1002
    UNSUPPORTED_DATA = 1003
    ABNORMAL_CLOSURE = 1006
    INVALID_TEXT = 1007
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    MANDATORY_EXTENSION = 1010
    INTERNAL_ERROR = 1011
    SERVICE_RESTART = 1012
    TRY_AGAIN_LATER = 1013
    BAD_GATEWAY = 1014


class WSMsgType(IntEnum):
    # websocket spec types
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    PING = 0x9
    PONG = 0xA
    CLOSE = 0x8

    # aiohttp specific types
    CLOSING = 0x100
    CLOSED = 0x101
    ERROR = 0x102

    text = TEXT
    binary = BINARY
    ping = PING
    pong = PONG
    close = CLOSE
    closing = CLOSING
    closed = CLOSED
    error = ERROR


class WSMessage(NamedTuple):
    type: WSMsgType
    # To type correctly, this would need some kind of tagged union for each type.
    data: Any
    extra: str | None

    def json(self, *, loads: Callable[[Any], Any] = json.loads) -> Any:
        """Return parsed JSON data.

        .. versionadded:: 0.22
        """
        return loads(self.data)


class WSMessageTextBytes(NamedTuple):
    """WebSocket TEXT message with raw bytes (no UTF-8 decoding)."""

    type: WSMsgType
    # To type correctly, this would need some kind of tagged union for each type.
    # In 4.0, we use a union of message types to properly type data, but in 3.x
    # we keep it as Any to avoid a breaking change.
    data: Any
    extra: str | None

    def json(self, *, loads: Callable[[Any], Any] = json.loads) -> Any:
        """Return parsed JSON data."""
        return loads(self.data)


# Type aliases for message types based on decode_text setting
# When decode_text=True, TEXT messages have str data (WSMessage)
# When decode_text=False, TEXT messages have bytes data (WSMessageTextBytes)
WSMessageDecodeText = WSMessage
WSMessageNoDecodeText = WSMessage | WSMessageTextBytes


# Constructing the tuple directly to avoid the overhead of
# the lambda and arg processing since NamedTuples are constructed
# with a run time built lambda
# https://github.com/python/cpython/blob/d83fcf8371f2f33c7797bc8f5423a8bca8c46e5c/Lib/collections/__init__.py#L441
WS_CLOSED_MESSAGE = tuple.__new__(WSMessage, (WSMsgType.CLOSED, None, None))
WS_CLOSING_MESSAGE = tuple.__new__(WSMessage, (WSMsgType.CLOSING, None, None))


class WebSocketError(Exception):
    """WebSocket protocol parser error."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(code, message)

    def __str__(self) -> str:
        return cast(str, self.args[1])


class WSHandshakeError(Exception):
    """WebSocket protocol handshake error."""
<|endoftext|>"""WebSocket protocol versions 13 and 8."""

import asyncio
import random
import sys
from asyncio.base_events import BaseEventLoop
from functools import partial
from typing import Final, Optional, Set

from ..base_protocol import BaseProtocol
from ..client_exceptions import ClientConnectionResetError
from ..compression_utils import ZLibBackend, ZLibCompressor
from ..helpers import DEFAULT_CHUNK_SIZE
from .helpers import (
    MASK_LEN,
    MSG_SIZE,
    PACK_CLOSE_CODE,
    PACK_LEN1,
    PACK_LEN2,
    PACK_LEN3,
    PACK_RANDBITS,
    websocket_mask,
)
from .models import WS_DEFLATE_TRAILING, WSMsgType

# WebSocket opcode boundary: opcodes 0-7 are data frames, 8-15 are control frames
# Control frames (ping, pong, close) are never compressed
WS_CONTROL_FRAME_OPCODE: Final[int] = 8

# For websockets, keeping latency low is extremely important as implementations
# generally expect to be able to send and receive messages quickly. We use a
# larger chunk size to reduce the number of executor calls and avoid task
# creation overhead, since both are significant sources of latency when chunks
# are small. A size of 16KiB was chosen as a balance between avoiding task
# overhead and not blocking the event loop too long with synchronous compression.

WEBSOCKET_MAX_SYNC_CHUNK_SIZE = 16 * 1024


class WebSocketWriter:
    """WebSocket writer.

    The writer is responsible for sending messages to the client. It is
    created by the protocol when a connection is established. The writer
    should avoid implementing any application logic and should only be
    concerned with the low-level details of the WebSocket protocol.
    """

    def __init__(
        self,
        protocol: BaseProtocol,
        transport: asyncio.Transport,
        *,
        use_mask: bool = False,
        limit: int = DEFAULT_CHUNK_SIZE,
        random: random.Random = random.Random(),
        compress: int = 0,
        notakeover: bool = False,
    ) -> None:
        """Initialize a WebSocket writer."""
        self.protocol = protocol
        self.transport = transport
        self.use_mask = use_mask
        self.get_random_bits = partial(random.getrandbits, 32)
        self.compress = compress
        self.notakeover = notakeover
        self._closing = False
        self._limit = limit
        self._output_size = 0
        self._compressobj: Optional[ZLibCompressor] = None
        self._send_lock = asyncio.Lock()
        self._background_tasks: Set[asyncio.Task[None]] = set()

    async def send_frame(
        self, message: bytes, opcode: int, compress: int | None = None
    ) -> None:
        """Send a frame over the websocket with message as its payload."""
        if self._closing and not (opcode & WSMsgType.CLOSE):
            raise ClientConnectionResetError("Cannot write to closing transport")

        if not (compress or self.compress) or opcode >= WS_CONTROL_FRAME_OPCODE:
            # Non-compressed frames don't need lock or shield
            self._write_websocket_frame(message, opcode, 0)
        elif len(message) <= WEBSOCKET_MAX_SYNC_CHUNK_SIZE:
            # Small compressed payloads - compress synchronously in event loop
            # We need the lock even though sync compression has no await points.
            # This prevents small frames from interleaving with large frames that
            # compress in the executor, avoiding compressor state corruption.
            async with self._send_lock:
                self._send_compressed_frame_sync(message, opcode, compress)
        else:
            # Large compressed frames need shield to prevent corruption
            # For large compressed frames, the entire compress+send
            # operation must be atomic. If cancelled after compression but
            # before send, the compressor state would be advanced but data
            # not sent, corrupting subsequent frames.
            # Create a task to shield from cancellation
            # The lock is acquired inside the shielded task so the entire
            # operation (lock + compress + send) completes atomically.
            # Use eager_start to avoid scheduling overhead
            coro = self._send_compressed_frame_async_locked(message, opcode, compress)
            if sys.version_info >= (3, 14):
                loop = asyncio.get_running_loop()
                if isinstance(loop, BaseEventLoop):
                    send_task = asyncio.create_task(coro, eager_start=True)
                else:
                    send_task = asyncio.Task(coro, loop=loop, eager_start=True)
            elif sys.version_info >= (3, 12):
                send_task = asyncio.Task(
                    coro, loop=asyncio.get_running_loop(), eager_start=True
                )
            else:
                send_task = asyncio.create_task(coro)
            # Keep a strong reference to prevent garbage collection
            self._background_tasks.add(send_task)
            send_task.add_done_callback(self._background_tasks.discard)
            await asyncio.shield(send_task)

        # It is safe to return control to the event loop when using compression
        # after this point as we have already sent or buffered all the data.
        # Once we have written output_size up to the limit, we call the
        # drain helper which waits for the transport to be ready to accept
        # more data. This is a flow control mechanism to prevent the buffer
        # from growing too large. The drain helper will return right away
        # if the writer is not paused.
        if self._output_size > self._limit:
            self._output_size = 0
            if self.protocol._paused:
                await self.protocol._drain_helper()

    def _write_websocket_frame(self, message: bytes, opcode: int, rsv: int) -> None:
        """
        Write a websocket frame to the transport.

        This method handles frame header construction, masking, and writing to transport.
        It does not handle compression or flow control - those are the responsibility
        of the caller.
        """
        msg_length = len(message)

        use_mask = self.use_mask
        mask_bit = 0x80 if use_mask else 0

        # Depending on the message length, the header is assembled differently.
        # The first byte is reserved for the opcode and the RSV bits.
        first_byte = 0x80 | rsv | opcode
        if msg_length < 126:
            header = PACK_LEN1(first_byte, msg_length | mask_bit)
            header_len = 2
        elif msg_length < 65536:
            header = PACK_LEN2(first_byte, 126 | mask_bit, msg_length)
            header_len = 4
        else:
            header = PACK_LEN3(first_byte, 127 | mask_bit, msg_length)
            header_len = 10

        if self.transport.is_closing():
            raise ClientConnectionResetError("Cannot write to closing transport")

        # https://datatracker.ietf.org/doc/html/rfc6455#section-5.3
        # If we are using a mask, we need to generate it randomly
        # and apply it to the message before sending it. A mask is
        # a 32-bit value that is applied to the message using a
        # bitwise XOR operation. It is used to prevent certain types
        # of attacks on the websocket protocol. The mask is only used
        # when aiohttp is acting as a client. Servers do not use a mask.
        if use_mask:
            mask = PACK_RANDBITS(self.get_random_bits())
            message_arr = bytearray(message)
            websocket_mask(mask, message_arr)
            self.transport.write(header + mask + message_arr)
            self._output_size += MASK_LEN
        elif msg_length > MSG_SIZE:
            self.transport.write(header)
            self.transport.write(message)
        else:
            self.transport.write(header + message)

        self._output_size += header_len + msg_length

    def _get_compressor(self, compress: int | None) -> ZLibCompressor:
        """Get or create a compressor object for the given compression level."""
        if compress:
            # Do not set self._compress if compressing is for this frame
            return ZLibCompressor(
                level=ZLibBackend.Z_BEST_SPEED,
                wbits=-compress,
                max_sync_chunk_size=WEBSOCKET_MAX_SYNC_CHUNK_SIZE,
            )
        if not self._compressobj:
            self._compressobj = ZLibCompressor(
                level=ZLibBackend.Z_BEST_SPEED,
                wbits=-self.compress,
                max_sync_chunk_size=WEBSOCKET_MAX_SYNC_CHUNK_SIZE,
            )
        return self._compressobj

    def _send_compressed_frame_sync(
        self, message: bytes, opcode: int, compress: int | None
    ) -> None:
        """
        Synchronous send for small compressed frames.

        This is used for small compressed payloads that compress synchronously in the event loop.
        Since there are no await points, this is inherently cancellation-safe.
        """
        # RSV are the reserved bits in the frame header. They are used to
        # indicate that the frame is using an extension.
        # https://datatracker.ietf.org/doc/html/rfc6455#section-5.2
        compressobj = self._get_compressor(compress)
        # (0x40) RSV1 is set for compressed frames
        # https://datatracker.ietf.org/doc/html/rfc7692#section-7.2.3.1
        self._write_websocket_frame(
            (
                compressobj.compress_sync(message)
                + compressobj.flush(
                    ZLibBackend.Z_FULL_FLUSH
                    if self.notakeover
                    else ZLibBackend.Z_SYNC_FLUSH
                )
            ).removesuffix(WS_DEFLATE_TRAILING),
            opcode,
            0x40,
        )

    async def _send_compressed_frame_async_locked(
        self, message: bytes, opcode: int, compress: int | None
    ) -> None:
        """
        Async send for large compressed frames with lock.

        Acquires the lock and compresses large payloads asynchronously in
        the executor. The lock is held for the entire operation to ensure
        the compressor state is not corrupted by concurrent sends.

        MUST be run shielded from cancellation. If cancelled after
        compression but before sending, the compressor state would be
        advanced but data not sent, corrupting subsequent frames.
        """
        async with self._send_lock:
            # RSV are the reserved bits in the frame header. They are used to
            # indicate that the frame is using an extension.
            # https://datatracker.ietf.org/doc/html/rfc6455#section-5.2
            compressobj = self._get_compressor(compress)
            # (0x40) RSV1 is set for compressed frames
            # https://datatracker.ietf.org/doc/html/rfc7692#section-7.2.3.1
            self._write_websocket_frame(
                (
                    await compressobj.compress(message)
                    + compressobj.flush(
                        ZLibBackend.Z_FULL_FLUSH
                        if self.notakeover
                        else ZLibBackend.Z_SYNC_FLUSH
                    )
                ).removesuffix(WS_DEFLATE_TRAILING),
                opcode,
                0x40,
            )

    async def close(self, code: int = 1000, message: bytes | str = b"") -> None:
        """Close the websocket, sending the specified code and message."""
        if isinstance(message, str):
            message = message.encode("utf-8")
        try:
            await self.send_frame(
                PACK_CLOSE_CODE(code) + message, opcode=WSMsgType.CLOSE
            )
        finally:
            self._closing = True
<|endoftext|>"""Helpers for WebSocket protocol versions 13 and 8."""

import functools
import re
from re import Pattern
from struct import Struct
from typing import TYPE_CHECKING, Final

from ..helpers import NO_EXTENSIONS
from .models import WSHandshakeError

UNPACK_LEN3 = Struct("!Q").unpack_from
UNPACK_CLOSE_CODE = Struct("!H").unpack
PACK_LEN1 = Struct("!BB").pack
PACK_LEN2 = Struct("!BBH").pack
PACK_LEN3 = Struct("!BBQ").pack
PACK_CLOSE_CODE = Struct("!H").pack
PACK_RANDBITS = Struct("!L").pack
MSG_SIZE: Final[int] = 2**14
MASK_LEN: Final[int] = 4

WS_KEY: Final[bytes] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# Used by _websocket_mask_python
@functools.lru_cache
def _xor_table() -> list[bytes]:
    return [bytes(a ^ b for a in range(256)) for b in range(256)]


def _websocket_mask_python(mask: bytes, data: bytearray) -> None:
    """Websocket masking function.

    `mask` is a `bytes` object of length 4; `data` is a `bytearray`
    object of any length. The contents of `data` are masked with `mask`,
    as specified in section 5.3 of RFC 6455.

    Note that this function mutates the `data` argument.

    This pure-python implementation may be replaced by an optimized
    version when available.

    """
    assert isinstance(data, bytearray), data
    assert len(mask) == 4, mask

    if data:
        _XOR_TABLE = _xor_table()
        a, b, c, d = (_XOR_TABLE[n] for n in mask)
        data[::4] = data[::4].translate(a)
        data[1::4] = data[1::4].translate(b)
        data[2::4] = data[2::4].translate(c)
        data[3::4] = data[3::4].translate(d)


if TYPE_CHECKING or NO_EXTENSIONS:  # pragma: no cover
    websocket_mask = _websocket_mask_python
else:
    try:
        from .mask import _websocket_mask_cython  # type: ignore[import-not-found]

        websocket_mask = _websocket_mask_cython
    except ImportError:  # pragma: no cover
        websocket_mask = _websocket_mask_python


_WS_EXT_RE: Final[Pattern[str]] = re.compile(
    r"^(?:;\s*(?:"
    r"(server_no_context_takeover)|"
    r"(client_no_context_takeover)|"
    r"(server_max_window_bits(?:=(\d+))?)|"
    r"(client_max_window_bits(?:=(\d+))?)))*$"
)

_WS_EXT_RE_SPLIT: Final[Pattern[str]] = re.compile(r"permessage-deflate([^,]+)?")


def ws_ext_parse(extstr: str | None, isserver: bool = False) -> tuple[int, bool]:
    if not extstr:
        return 0, False

    compress = 0
    notakeover = False
    for ext in _WS_EXT_RE_SPLIT.finditer(extstr):
        defext = ext.group(1)
        # Return compress = 15 when get `permessage-deflate`
        if not defext:
            compress = 15
            break
        match = _WS_EXT_RE.match(defext)
        if match:
            compress = 15
            if isserver:
                # Server never fail to detect compress handshake.
                # Server does not need to send max wbit to client
                if match.group(4):
                    compress = int(match.group(4))
                    # Group3 must match if group4 matches
                    # Compress wbit 8 does not support in zlib
                    # If compress level not support,
                    # CONTINUE to next extension
                    if compress > 15 or compress < 9:
                        compress = 0
                        continue
                if match.group(1):
                    notakeover = True
                # Ignore regex group 5 & 6 for client_max_window_bits
                break
            else:
                if match.group(6):
                    compress = int(match.group(6))
                    # Group5 must match if group6 matches
                    # Compress wbit 8 does not support in zlib
                    # If compress level not support,
                    # FAIL the parse progress
                    if compress > 15 or compress < 9:
                        raise WSHandshakeError("Invalid window size")
                if match.group(2):
                    notakeover = True
                # Ignore regex group 5 & 6 for client_max_window_bits
                break
        # Return Fail if client side and not match
        elif not isserver:
            raise WSHandshakeError("Extension for deflate not supported" + ext.group(1))

    return compress, notakeover


def ws_ext_gen(
    compress: int = 15, isserver: bool = False, server_notakeover: bool = False
) -> str:
    # client_notakeover=False not used for server
    # compress wbit 8 does not support in zlib
    if compress < 9 or compress > 15:
        raise ValueError(
            "Compress wbits must between 9 and 15, zlib does not support wbits=8"
        )
    enabledext = ["permessage-deflate"]
    if not isserver:
        enabledext.append("client_max_window_bits")

    if compress < 15:
        enabledext.append("server_max_window_bits=" + str(compress))
    if server_notakeover:
        enabledext.append("server_no_context_takeover")
    # if client_notakeover:
    #     enabledext.append('client_no_context_takeover')
    return "; ".join(enabledext)
<|endoftext|>from typing import Any

from torchgen.model import (
    Annotation,
    Argument,
    Arguments,
    BaseOperatorName,
    BaseTy,
    BaseType,
    CustomClassType,
    FunctionSchema,
    ListType,
    OperatorName,
    Return,
)


# Note: These aren't actually used in torchgen, they're some utilities for generating a schema
# from real arguments. For example, this is used to generate HigherOrderOperators' schema since
# their schemas can vary for different instances of the same HOP.


class TypeGen:
    convert_to_base_ty = {
        int: BaseTy.int,
        float: BaseTy.float,
        str: BaseTy.str,
        bool: BaseTy.bool,
    }

    @staticmethod
    def from_example(obj: Any) -> BaseType | ListType | CustomClassType:
        import torch

        if isinstance(obj, torch.fx.GraphModule):
            return BaseType(BaseTy.GraphModule)
        elif isinstance(obj, torch.Tensor):
            return BaseType(BaseTy.Tensor)
        elif isinstance(obj, torch.SymInt):
            return BaseType(BaseTy.SymInt)
        elif isinstance(obj, torch.SymBool):
            return BaseType(BaseTy.SymBool)
        elif isinstance(obj, torch.ScriptObject):
            return CustomClassType(obj._type().name())  # type: ignore[attr-defined]
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                raise AssertionError("list/tuple must be non-empty")
            all_base_tys = [TypeGen.from_example(x) for x in obj]
            if len(set(all_base_tys)) > 1:
                raise RuntimeError(
                    f"Cannot generate schema for a sequence of args of heterogeneous types: {all_base_tys}. "
                    "Consider unpacking the argument and give proper names to them if possible "
                    "instead of using *args."
                )
            return ListType(all_base_tys[0], len(obj))
        tp = type(obj)
        if tp not in TypeGen.convert_to_base_ty:
            raise RuntimeError(f"unsupported type {tp}")
        return BaseType(TypeGen.convert_to_base_ty[tp])


class ReturnGen:
    @staticmethod
    def from_example(
        name: str | None, obj: Any, annotation: Annotation | None
    ) -> Return:
        return Return(name, TypeGen.from_example(obj), annotation)


class ArgumentGen:
    @staticmethod
    def from_example(
        name: str, obj: Any, default: str | None, annotation: Annotation | None
    ) -> Argument:
        return Argument(
            name, TypeGen.from_example(obj), default=default, annotation=annotation
        )


class FunctionSchemaGen:
    @staticmethod
    def from_example(
        op_name: str,
        example_inputs: tuple[tuple[str, Any], ...],
        example_outputs: tuple[Any, ...],
    ) -> FunctionSchema:
        args = []
        for name, inp in example_inputs:
            args.append(ArgumentGen.from_example(name, inp, None, None))
        # ignore the annotations and other attributes for now, we could add more when needed.
        arguments = Arguments(
            tuple(), None, tuple(args), tuple(), None, tuple(), tuple()
        )
        returns = tuple(
            ReturnGen.from_example(None, out, None) for out in example_outputs
        )
        op_name = OperatorName(BaseOperatorName(op_name, False, False, False), "")
        return FunctionSchema(op_name, arguments, returns)
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.api import cpp, dispatcher, functionalization
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    CType,
    DispatcherSignature,
    iTensorListRefT,
    NativeSignature,
    OptionalCType,
    optionalSymIntArrayRefT,
    symIntArrayRefT,
    SymIntT,
    tensorListT,
    tensorT,
    VectorCType,
    ViewInverseSignature,
)
from torchgen.context import (
    method_with_native_function,
    native_function_manager,
    with_native_function,
    with_native_function_and,
)
from torchgen.model import (
    Argument,
    BackendIndex,
    BaseTy,
    BaseType,
    FunctionSchema,
    ListType,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    OperatorName,
    Return,
    SchemaKind,
    SelfArgument,
    TensorOptionsArguments,
)
from torchgen.native_function_generation import (
    INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY,
    MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT,
    OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY,
)
from torchgen.utils import concatMap, dataclass_repr, FileManager


if TYPE_CHECKING:
    from collections.abc import Callable

    from torchgen.selective_build.selector import SelectiveBuilder


# Note: [Mutable Ops Not Using Functionalization]
# Ops in this list currently do not work with functionalization and should be fixed.
MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION = (
    OUT_OPS_THAT_DONT_GET_GROUPED_PROPERLY
    + MUTABLE_OPS_THAT_CANNOT_GET_AN_OUT_VARIANT
    + INPLACE_OPS_THAT_DONT_GET_GROUPED_PROPERLY
    + [
        # It will be BC-breaking, but we should fix their schemas.
        # should be inplace?
        "record_stream",
        # See Note [resize_ in Functionalization]
        "resize_",
        "resize_as_",
        # This function is used as for testing purposes only.
        "_fill_mem_eff_dropout_mask_",
        # Inference-only op called behind a custom op graph break.
        "_flash_attention_forward_no_dropout_inplace",
    ]
)

# Eager cumulative out variants compute in the out dtype when dtype is omitted.
# Functionalization normally lowers mutable ops through their functional variants,
# so these need to thread the out dtype explicitly to preserve eager semantics.
CUMULATIVE_OUT_OPS_PRESERVING_OUT_DTYPE = {
    OperatorName.parse("cumsum.out"),
    OperatorName.parse("cumprod.out"),
}

# This file contains codegen that relates to the functionalization pass.
# It includes:
# - gen_functionalization_definition
#     Generates dispatcher kernel definitions for the functionalization pass.
# - gen_functionalization_registration
#     Generates dispatcher kernel registrations for the functionalization pass.
# - gen_functionalization_view_inverse_declaration
#     Generates a declaration for an "inverse view", for every view op
#     that is needed in functionalization. We manually implement their definitions.
# - gen_composite_view_copy_kernel
#     Generates view_copy() composite kernels for all view_copy operators.


# Generates the body of the default composite C++ kernel for a {view}_copy NativeFunction
# See Note [view_copy NativeFunctions]
@dataclass(frozen=True)
class GenCompositeViewCopyKernel:
    backend_index: BackendIndex

    @method_with_native_function
    def __call__(self, g: NativeFunctionsViewGroup) -> str | None:
        if g.view_copy is None:
            return None
        elif g.view_copy.func.name.name.base != f"{g.view.func.name.name}_copy":
            # If the view_copy doesn't match the standard naming scheme of <op>_copy,
            # assume it already exists and doesn't need to be generated.
            # Example: slice_inverse() with the copy variant named slice_scatter()
            # instead of slice_inverse_copy()
            return None

        metadata = self.backend_index.get_kernel(g.view_copy)
        if metadata is None:
            raise AssertionError(
                f"Expected metadata for view_copy kernel: {g.view_copy}"
            )

        # We can make view_copy work in more cases by using reshape()
        # when a normal view call would ordinarily fail.
        # This also makes LTC more efficient, because they don't need to include
        # clone() calls in their graph (which is normally needed by reshape).
        if str(g.view_copy.func.name) == "view_copy":
            if metadata.kernel != "view_copy_symint":
                raise AssertionError(
                    f"Expected kernel 'view_copy_symint', got '{metadata.kernel}'"
                )
            return """\
at::Tensor view_copy_symint(const at::Tensor & self, at::SymIntArrayRef size) {
  c10::SymDimVector shape = infer_size_dv(size, self.sym_numel());
  if (!at::detail::computeStride(self.sym_sizes(), self.sym_strides(), shape).has_value()) {
    return self.reshape_symint(size);
  } else {
    auto output = at::_ops::view::call(self, size);
    return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);
  }
}
"""
        # view_copy is a native signature, since we're generating an at::native:: kernel
        # Functionalization always operates on symints though
        view_copy_sig = NativeSignature(
            g.view_copy.func, symint=metadata.supports_symint()
        )

        # view is a dispatcher signature, since we're calling into the at::_ops API
        view_sig = DispatcherSignature(g.view.func)

        view_api_name = g.view.func.name.unambiguous_name()
        exprs = ", ".join(
            [e.expr for e in translate(view_copy_sig.arguments(), view_sig.arguments())]
        )

        # view ops today always return either a Tensor or a list of Tensors
        if len(g.view.func.returns) != 1:
            raise AssertionError(f"Expected 1 return, got {len(g.view.func.returns)}")
        if not (
            g.view.func.returns[0].type == BaseType(BaseTy.Tensor)
            or g.view.func.returns[0].type == ListType(BaseType(BaseTy.Tensor), None)
        ):
            raise AssertionError(
                f"Expected Tensor or Tensor[] return type, got {g.view.func.returns[0].type}"
            )

        if g.view.func.returns[0].type == BaseType(BaseTy.Tensor):
            return_cloned_output = """\
  return output.clone(/*memory_format=*/at::MemoryFormat::Contiguous);"""
        else:
            # If the return type is a list, we need to clone each tensor in the list.
            return_cloned_output = f"""\
  {view_copy_sig.returns_type().cpp_type()} out_clone;
  for (const auto i : c10::irange(output.size())) {{
    out_clone.push_back(output[i].clone(/*memory_format=*/at::MemoryFormat::Contiguous));
  }}
  return out_clone;"""

        # The default generated composite kernel for {view}_copy() operators just clones
        # the input tensor, and runs the underlying view on the clone.
        return f"""
{view_copy_sig.defn(name=metadata.kernel)} {{
  auto output = at::_ops::{view_api_name}::call({exprs});
  {return_cloned_output}
}}
"""


def return_str(rets: tuple[Return, ...], names: list[str]) -> str:
    if len(rets) != len(names):
        raise AssertionError(f"Expected {len(rets)} names, got {len(names)}")
    if len(rets) == 0:
        return ""
    elif len(rets) == 1:
        return f"return {names[0]};"
    else:
        return f"return {dispatcher.returns_type(rets).cpp_type()}({', '.join(names)});"


def modifies_arguments(f: NativeFunction) -> bool:
    return any(
        a.annotation is not None and a.annotation.is_write
        for a in f.func.arguments.flat_all
    )


def wrapper_name(func: FunctionSchema) -> str:
    if func.name.overload_name:
        return f"{cpp.name(func)}_{func.name.overload_name}"
    else:
        return cpp.name(func)


def is_tensor_like(a: Argument | TensorOptionsArguments | SelfArgument) -> bool:
    return isinstance(a, SelfArgument) or (
        isinstance(a, Argument) and a.type.is_tensor_like()
    )


# We need to wrap / unwrap various arguments from the op in the functionalization kernels.
# Some op schemas include non-owning types though (like TensorList),
# and when we unwrap them we expect to get out an owning type!.
# We also return a lambda that tells you how to convert the non-owning type argument into the owning type.
def get_owning_type(t: CType) -> tuple[CType, Callable[[str], str]]:
    if t == BaseCType(tensorListT):
        return VectorCType(BaseCType(tensorT)), lambda x: f"{x}.vec()"
    if t == BaseCType(iTensorListRefT):
        return VectorCType(BaseCType(tensorT)), lambda x: f"{{{x}.begin(), {x}.end()}}"
    # There are technically other non-owning types out there (like IntArrayRef),
    # but functionalization only actually cares about the ones involving tensors.
    return t, lambda x: x


# unwraps all tensor-like arguments, returning:
# (1) a string containing all of the logic that does the unwrapping
# (2) a context, to be used by translate(), with all of the relevant bindings.
def unwrap_tensor_args(
    sig: DispatcherSignature, *, is_view_op: bool
) -> tuple[str, list[Binding]]:
    context: list[Binding] = []
    unwrapped_tensor_args: list[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            # for tensor inputs, we want to unwrap them before passing them into the redispatch calls.
            unwrapped_name = f"{arg.name}_"
            # For most ops, the functionalization needs to sync any pending updates on the input tensors
            # before calling the operator, since otherwise the operator will act on stale data.
            # For view ops though, we can continue to defer syncing until the tensor is used by
            # a non-view operator.
            maybe_sync_input = (
                "" if is_view_op else f"at::functionalization::impl::sync({arg.name});"
            )
            unwrapped_type, conversion_fn = get_owning_type(
                arg.nctype.remove_const_ref().type
            )
            unwrapped_tensor_args.append(
                f"""
      {unwrapped_type.cpp_type()} {unwrapped_name};
      if (at::functionalization::impl::isFunctionalTensor({arg.name})) {{
        {maybe_sync_input}
        {unwrapped_name} = at::functionalization::impl::from_functional_tensor({arg.name});
      }} else {{
        {unwrapped_name} = {conversion_fn(arg.name)};
      }}"""
            )
            context.append(arg.with_name(unwrapped_name))
        else:
            # for non-tensor inputs, we want to pass them directly into the redispatch calls.
            context.append(arg)
    unwrap_tensor_args_str = "\n      ".join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context


# converts  all tensor-like arguments to meta tensors, which are used to compute stride info. Returns:
# (1) a string containing all of the logic that does the conversions.
# (2) a context, to be used by translate(), with all of the relevant bindings.
def convert_to_meta_tensors(sig: DispatcherSignature) -> tuple[str, list[Binding]]:
    context: list[Binding] = []
    unwrapped_tensor_args: list[str] = []
    for arg in sig.arguments():
        if is_tensor_like(arg.argument):
            # for tensor inputs, we want to unwrap them before passing them into the redispatch calls.
            a_ = arg.name
            unwrapped_name = f"{arg.name}_meta"
            unwrapped_tensor_args.append(f"auto {unwrapped_name} = to_meta({a_});")
            context.append(arg.with_name(unwrapped_name))
        else:
            # for non-tensor inputs, we want to pass them directly into the redispatch calls.
            context.append(arg)
    unwrap_tensor_args_str = "\n        ".join(unwrapped_tensor_args)
    return unwrap_tensor_args_str, context


# The functionalization codegen currently expects view op schemas to have this form:
# foo(Tensor(a), ...) -> Tensor(a) (e.g. transpose)
# foo(Tensor(a!), ...) -> Tensor(a!) (e.g. transpose_)
def assert_view_op_properties(func: FunctionSchema) -> None:
    def is_alias(a: Argument) -> bool:
        return a.annotation is not None

    args = func.arguments.flat_non_out
    # The first argument is a tensor with an alias semantics (annotations)
    if not (len(args) > 0 and args[0].type == BaseType(BaseTy.Tensor)):
        raise AssertionError(
            f"In the functionalization codegen, we expect the first argument of every view operator to be a tensor, "
            f"but found an argument of type {str(args[0].type)} for operator: {str(func.name)}."
        )
    # No other arguments have aliasing semantics
    if not (is_alias(args[0]) and not any(is_alias(a) for a in args[1:])):
        raise AssertionError(
            "In the functionalization codegen, we expect the first argument of every view "
            "operator to alias the output. View operators with multiple aliasing inputs "
            "aren't supported yet. Found an operator that doesn't satisfy this constraint"
        )


# One-liner expression for checking if an expression expr of type type has any
# symbolic values.
def emit_expr_has_symbolic_values(expr: str, type: CType) -> str:
    if type == BaseCType(SymIntT):
        return f"{expr}.is_symbolic()"

    if isinstance(type, OptionalCType):
        innerexpr = f"(*{expr})"
        return f"{expr}.has_value() ? {emit_expr_has_symbolic_values(innerexpr, type.elem)} : false"

    if type == BaseCType(optionalSymIntArrayRefT):
        return emit_expr_has_symbolic_values(
            expr, OptionalCType(BaseCType(symIntArrayRefT))
        )

    if type in (BaseCType(symIntArrayRefT), VectorCType(BaseCType(SymIntT))):
        argname = "arg"
        lambda_check = emit_expr_has_symbolic_values(argname, BaseCType(SymIntT))
        return (
            "std::any_of("
            f"{expr}.begin(), {expr}.end(), "
            f"[=](auto& {argname}) {{ return {lambda_check}; }})"
        )

    raise ValueError(
        "unsupported type for has_symbolic_values check. "
        "It should be a SymInt or a collection of those. "
        f"Got: {type.cpp_type()}"
    )


# Detects whether any of the SymInt arguments are, in fact, symbolic values.
# This is used in the constructor of ViewMeta.
def emit_has_symbolic_inputs(sig: DispatcherSignature) -> str:
    name = "has_symbolic_inputs"
    statements = [
        f"{name} = {name} | ({emit_expr_has_symbolic_values(binding.name, binding.nctype.type)});"
        for binding in sig.arguments()
        if (
            isinstance(binding.argument, Argument)
            and binding.argument.type.is_symint_like()
        )
    ]
    body = "\n      ".join(statements)
    return f"""
      bool {name} = false;
      {body}"""


# Generates the Functionalization kernel for:
# - ops that create aliases (e.g. transpose())
# - ops that are views AND mutations (e.g. transpose_())
def emit_view_functionalization_body(
    g: NativeFunctionsViewGroup, *, view_inplace: bool
) -> str:
    if view_inplace:
        # This op is both an inplace op AND a view op.
        # See Note [Functionalization Pass - Inplace View Ops] for details.
        # I currently have the view meta call into the out-of-place variant of the view, to avoid
        # having to define an extra ~20 inplace {view}_inverse_ functions.
        # Most view ops don't have NativeFunctionGroup's both, because we don't define out= variants for view ops.
        # I'm assuming that every inplace-view op has a corresponding out-of-place view op,
        # with the same name but the trailing underscore removed.
        # This is currently asserted at parse time in gen.py (see error_check_native_functions).
        if g.view_inplace is None:
            raise AssertionError(
                "Expected view_inplace to be non-None for inplace view"
            )
        f = g.view_inplace
    else:
        f = g.view

    if g.view_copy is None:
        raise AssertionError("Expected view_copy to be non-None")
    with native_function_manager(f):
        call_sig = DispatcherSignature.from_schema(g.view_copy.func)

        spec = ViewMetaSpecialization(g, f=f)

        # the "view_copy" op name that the functionalization kernels need to call
        api_name = g.view_copy.func.name.unambiguous_name()
        # Sometimes the functionalization pass needs to no-op (e.g. if it was passed non-functional tensors)
        # "no-op"ing in this context is just redispatching to the original op.
        noop_api_name = f.func.name.unambiguous_name()

        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        assert_view_op_properties(f.func)
        view_tensor_name = dispatcher_sig.arguments()[0].name

        return_type = dispatcher_sig.returns_type().remove_const_ref().cpp_type()

        unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(
            dispatcher_sig, is_view_op=True
        )
        view_redispatch_args = [
            e.expr
            for e in translate(unwrapped_args_ctx, call_sig.arguments(), method=False)
        ]

        # The meta API call should use the same arguments, but convert all tensors to meta tensors first.
        meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
        meta_call_args = [
            e.expr for e in translate(meta_call_ctx, call_sig.arguments(), method=False)
        ]

        symbolic_inputs_check = emit_has_symbolic_inputs(call_sig)

        if "inplace_view" in f.tags:
            # See Note [Functionalization Pass - Inplace View Ops] for more details
            return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{
        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.
        {unwrap_tensor_args_str}
        at::AutoDispatchSkipFunctionalize guard;
        return at::_ops::{noop_api_name}::call({", ".join(view_redispatch_args)});
      }}
      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
      auto inverse_return_mode = (
          reapply_views ? at::functionalization::InverseReturnMode::ViewOrScatterInverse
            : at::functionalization::InverseReturnMode::NeverView
      );
      {symbolic_inputs_check}
      auto view_meta = {spec.new()};
      auto compute_reference_meta =
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);
      {return_type} reference_tensor_output;
      if (compute_reference_meta && !disable_meta_reference()) {{
        {meta_conversion_str}
        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        reference_tensor_output = at::_ops::{noop_api_name}::call({", ".join(meta_call_args)});
      }}
      // This function adds the above view meta to the current tensor and replays them off the base,
      // mutating the size/stride info of the current FunctionalTensorWrapper.
      // Because of this, we need to make sure to run the reference shape function above,
      // BEFORE doing this (otherwise we'll end up running the reference function using the wrong sizes/strides)
      at::functionalization::impl::mutate_view_meta({view_tensor_name}, view_meta);
      // See  Note [Propagating strides in the functionalization pass]
      // XLA/LTC don't implement the logic to propagate strides correctly, so we need to rely
      // on a reference implementation here (instead of relying on the output from the forward lambda
      // having the correct stride info)
      if (compute_reference_meta && !disable_meta_reference()) {{
        at::functionalization::impl::set_sizes_strides_offset({view_tensor_name}, reference_tensor_output);
      }}
      return {view_tensor_name};
    }}
"""

        else:
            return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      {unwrap_tensor_args_str}
      if (!at::functionalization::impl::isFunctionalTensor({view_tensor_name})) {{
        // functionalization is re-entrant, but will no-op if it wasn't passed a FunctionalTensorWrapper.
        at::AutoDispatchSkipFunctionalize guard;
        return at::_ops::{noop_api_name}::call({", ".join(view_redispatch_args)});
      }}
      auto reapply_views = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
      auto inverse_return_mode = (
          reapply_views ? at::functionalization::InverseReturnMode::ViewOrScatterInverse
            : at::functionalization::InverseReturnMode::NeverView
      );
      auto compute_reference_meta =
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::XLABit) ||
        {view_tensor_name}.key_set().has_backend(c10::BackendComponent::LazyBit);
      {return_type} reference_tensor_output;
      if (compute_reference_meta && !disable_meta_reference()) {{
        {meta_conversion_str}
        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        reference_tensor_output = at::_ops::{noop_api_name}::call({", ".join(meta_call_args)});
      }}
      {return_type} tmp_output;
      {{
        at::AutoDispatchSkipFunctionalize guard;
        if (reapply_views) {{
          tmp_output = at::_ops::{noop_api_name}::call({", ".join(view_redispatch_args)});
        }} else {{
          tmp_output = at::_ops::{api_name}::call({", ".join(view_redispatch_args)});
        }}
      }}
      {symbolic_inputs_check}
      auto view_meta = {spec.new()};
      auto out = at::functionalization::impl::create_functional_tensor_with_view_meta(tmp_output, {view_tensor_name}, view_meta);
      // See  Note [Propagating strides in the functionalization pass]
      if (compute_reference_meta && !disable_meta_reference()) {{
        at::functionalization::impl::set_sizes_strides_offset(out, reference_tensor_output);
      }}
      return out;
    }}
"""


def maybe_create_output(f: NativeFunction, var_name: str) -> str:
    if len(f.func.returns) == 0:
        return ""
    return_type = dispatcher.returns_type(f.func.returns).remove_const_ref().cpp_type()
    return f"{return_type} {var_name} = "


# Given a NativeFunction, and a variable name corresponding to the output of redispatching on the function,
# this returns two lists of names, consisting of:
# - the names of returns corresponding to the original (mutable) inputs of the outer function
# - the names of returns corresponding to the (immutable) outputs of the inner redispatched function
def get_mutable_redispatch_return_names(
    f: NativeFunction, inner_return_var: str
) -> tuple[list[str], list[str]]:
    aliased_returns = []
    non_aliased_returns = []
    for i, name in enumerate(f.func.aliased_return_names()):
        if name is not None:
            aliased_returns.append(name)
        else:
            non_aliased_returns.append(
                inner_return_var
                if len(f.func.returns) == 1
                else f"std::get<{i}>({inner_return_var})"
            )
    return aliased_returns, non_aliased_returns


# When functionalization "no-op's" and redispatches on a mutable operator, we need to take care so that:
#  - For fresh outputs, we return the result of the redispatch (without wrapping outputs)
#  - For outputs that were aliased to inputs, we return the inputs directly (since some of them might have been wrapped)
def return_from_mutable_noop_redispatch(
    f: NativeFunction, inner_return_var: str
) -> str:
    aliased, non_aliased = get_mutable_redispatch_return_names(f, inner_return_var)
    # Just get all of the return names, and immediately return them
    return return_str(f.func.returns, aliased + non_aliased)


def wrap_propagate_mutations_and_return(
    f: NativeFunction, functional_op: NativeFunction, inner_return_var: str
) -> str:
    mutable_arg_names = f.func.arguments.mutable_arg_names()
    (
        aliased_outer_rets,
        non_aliased_outer_rets,
    ) = get_mutable_redispatch_return_names(f, inner_return_var)
    _, non_aliased_inner_rets = get_mutable_redispatch_return_names(
        functional_op, inner_return_var
    )
    # The outer function may have a mix of aliased and non-aliased outputs,
    # But the inner functional op that we're transforming to should only have non-aliased outputs
    if len(mutable_arg_names) + len(non_aliased_outer_rets) != len(
        non_aliased_inner_rets
    ):
        raise AssertionError(
            f"Expected {len(mutable_arg_names)} + {len(non_aliased_outer_rets)} == {len(non_aliased_inner_rets)}"
        )

    # First, take all of the newly created outputs from the inner call and wrap them into functional tensors
    updates = []
    non_aliased_wrapped_ret_names = []
    for i, inner_ret in enumerate(
        non_aliased_inner_rets[: len(non_aliased_outer_rets)]
    ):
        ret_name = f"output_{i}"
        updates.append(
            f"""\
  auto output_{i} = at::functionalization::impl::to_functional_tensor({inner_ret});"""
        )
        non_aliased_wrapped_ret_names.append(ret_name)

    # Next, take all of the mutated outputs from the inner call corresponding to mutated inputs,
    # and propagate the mutations
    for outer_arg, inner_ret in zip(
        mutable_arg_names, non_aliased_inner_rets[len(non_aliased_outer_rets) :]
    ):
        updates.append(
            f"""\
  auto {outer_arg}_inner = at::functionalization::impl::from_functional_tensor({outer_arg});
  at::functionalization::impl::replace_({outer_arg}, {inner_ret});
  at::functionalization::impl::commit_update({outer_arg});
  at::functionalization::impl::sync({outer_arg});
  auto {outer_arg}_inner_updated = at::functionalization::impl::from_functional_tensor({outer_arg});
  at::functionalization::impl::propagate_xla_data_direct({outer_arg}_inner, {outer_arg}_inner_updated);"""
        )

    # Finally, we return:
    # - Any mutable arguments that also returns
    # - Any immutable returns that were created wrapping the output from the inner call
    returns_str = return_str(
        f.func.returns, aliased_outer_rets + non_aliased_wrapped_ret_names
    )
    updates_str = "\n".join(updates)
    return f"""\
{updates_str}
    {returns_str}"""


def maybe_replace_cumulative_out_dtype_exprs(
    f: NativeFunction,
    functional_sig: DispatcherSignature,
    functional_exprs: list[str],
) -> list[str]:
    if (
        f.func.kind() != SchemaKind.out
        or f.func.name not in CUMULATIVE_OUT_OPS_PRESERVING_OUT_DTYPE
    ):
        return functional_exprs

    if len(f.func.arguments.out) != 1:
        raise AssertionError(
            f"Expected a single out argument for cumulative out op: {f.func.name}"
        )

    dtype_arg_idx = next(
        (i for i, arg in enumerate(functional_sig.arguments()) if arg.name == "dtype"),
        None,
    )
    if dtype_arg_idx is None:
        raise AssertionError(
            f"Expected dtype argument for cumulative out op: {f.func.name}"
        )

    adjusted_exprs = functional_exprs.copy()
    dtype_expr = adjusted_exprs[dtype_arg_idx]
    adjusted_exprs[dtype_arg_idx] = (
        f"{dtype_expr}.has_value() ? {dtype_expr} : "
        f"std::optional<at::ScalarType>({f.func.arguments.out[0].name}_.scalar_type())"
    )
    return adjusted_exprs


# Generates the Functionalization kernel for:
# - mutation ops (inplace and out= ops)
@with_native_function_and
def emit_inplace_functionalization_body(
    f: NativeFunction, g: NativeFunctionsGroup
) -> str:
    # mutation case
    if not modifies_arguments(f):
        raise AssertionError(f"Expected function to modify arguments: {f.func}")

    dispatcher_sig = DispatcherSignature.from_schema(f.func)

    unwrap_tensor_args_str, unwrapped_args_ctx = unwrap_tensor_args(
        dispatcher_sig, is_view_op=False
    )

    mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is not None
    ]
    non_mutated_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type.is_tensor_like() and a.annotation is None
    ]
    non_mutated_tensor_names = [
        a.name
        for a in f.func.arguments.flat_all
        if a.type == BaseType(BaseTy.Tensor) and a.annotation is None
    ]
    # all mutable inputs must be functional tensors in order to participate in functionalization
    check_all_mutated_args_are_functional = " && ".join(
        ["true"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in mutated_names
        ]
    )
    check_any_non_mutated_args_are_functional = " || ".join(
        ["false"]
        + [
            f"at::functionalization::impl::isFunctionalTensor({a})"
            for a in non_mutated_names
        ]
    )

    check_any_non_mutated_tensors_are_xla = " || ".join(
        ["false"]
        + [
            f"{a}.device().type() == c10::DeviceType::XLA"
            for a in non_mutated_tensor_names
        ]
    )
    # These are used in the cases where we don't functionalize and redispatch to the inplace op
    # case 1: we hit an inplace op that doesn't have an out-of-place equivalent
    # case 2: we hit an inplace ops but our inputs are not functional tensors (in which case our kernel just no-ops)
    inplace_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, dispatcher_sig.arguments(), method=False)
    ]

    # call the out-of-place variant of the op
    return_type = (
        dispatcher.returns_type(g.functional.func.returns).remove_const_ref().cpp_type()
    )
    functional_sig = DispatcherSignature.from_schema(g.functional.func)
    functional_exprs = [
        e.expr
        for e in translate(unwrapped_args_ctx, functional_sig.arguments(), method=False)
    ]
    functional_exprs = maybe_replace_cumulative_out_dtype_exprs(
        f, functional_sig, functional_exprs
    )

    meta_conversion_str, meta_call_ctx = convert_to_meta_tensors(dispatcher_sig)
    # We don't want to run the inplace meta func for ops like .set_(), because:
    # (1) they're unnecessary: inplace meta checks are only useful for ops like add_(),
    #     where broadcasting will work for the out-of-place case but should fail on the inplace call
    # (2) They'll also fail without adding extra infra: we'd need to convert the input storage argument
    #     into a meta storage
    any_storage_args = any(
        a.type == BaseType(BaseTy.Storage) for a in f.func.arguments.flat_all
    )

    return f"""
    {dispatcher_sig.defn(name=wrapper_name(f.func), is_redispatching_fn=True)} {{
      if ({str(not any_storage_args and f.func.kind() == SchemaKind.inplace).lower()} && !disable_meta_reference()) {{
        // Before converting the mutable op to its functional variant, run meta tensors through the original op.
        // This will help us catch shape errors that apply to inplace ops that wouldn't apply to their functional variants.
        // (We can only do this for inplace ops today though, because they technically all support meta tensors).
        {meta_conversion_str}
        at::AutoDispatchSkipFunctionalize func_guard;
        c10::impl::ExcludeDispatchKeyGuard guard(exclude_keys_for_meta_dispatch);
        at::_ops::{f.func.name.unambiguous_name()}::call({", ".join(a.name for a in meta_call_ctx)});
      }}
      {unwrap_tensor_args_str}
      if (!({check_all_mutated_args_are_functional})) {{
        // We want to disable this check if there are any XLA tensors.
        // cpu_tensor.copy_(xla_tensor) is valid code.
        if (!({check_any_non_mutated_tensors_are_xla}) && ({check_any_non_mutated_args_are_functional})) {{
         // case 1: trying to mutate a non functional tensor with a functional tensor is an error
         TORCH_INTERNAL_ASSERT(false,
           "mutating a non-functional tensor with a functional tensor is not allowed.",
           " Please ensure that all of your inputs are wrapped inside of a functionalize() call.");
        }} else {{
         // case 2: arguments are not functional tensors, so we no-op and redispatch.
         at::AutoDispatchSkipFunctionalize guard;
         {maybe_create_output(f, "tmp_output")}at::_ops::{f.func.name.unambiguous_name()}::call({", ".join(inplace_exprs)});
         {return_from_mutable_noop_redispatch(f, "tmp_output")}
        }}
      }} else {{
        {return_type} tmp_output;
        {{
          at::AutoDispatchSkipFunctionalize guard;
          tmp_output = at::_ops::{g.functional.func.name.unambiguous_name()}::call({", ".join(functional_exprs)});
        }}
        {wrap_propagate_mutations_and_return(f, g.functional, "tmp_output")}
      }}
    }}"""


# The below functions generate RegisterFunctionalization.cpp
# These files provide the kernels that run the functionalization pass, which can be opted into
# per backend (e.g. XLA or Vulkan), or as a composable transform (functionalize() in functorch).


# See Note [Functionalization Pass: View Inverses].
def gen_functionalization_view_inverse_declaration(
    selector: SelectiveBuilder, g: NativeFunctionsViewGroup
) -> str | None:
    # For every (non-composite) view op, we need a corresponding "inverse view" function.
    # This generates the declarations so we get a good compiler error when someone adds a new view.
    @with_native_function
    def emit_decl_helper(g: NativeFunctionsViewGroup) -> str | None:
        if g.view.has_composite_implicit_autograd_kernel:
            return None
        view_inverse_sig = ViewInverseSignature(g)
        return view_inverse_sig.decl()

    return emit_decl_helper(g)


# Helper class for generating `ViewMeta` specializations.
@dataclass
class ViewMetaSpecialization:
    g: NativeFunctionsViewGroup
    f: NativeFunction

    @property
    def is_multi_output(self) -> bool:
        return functionalization.is_multi_output(self.f.func)

    @property
    def is_as_strided(self) -> bool:
        return str(self.f.func.name) == "as_strided"

    @property
    def out_index(self) -> str:
        if self.is_multi_output:
            return functionalization.out_index_binding.name
        return "0"

    @property
    def classname(self) -> str:
        return functionalization.classname(self.f.func)

    def decl(self) -> list[str]:
        base_ctor_arguments = functionalization.base_ctor_arguments(self.f.func)
        extra_ctor_arguments = functionalization.extra_ctor_arguments(self.f.func)
        attributes = functionalization.attributes(self.f.func)

        # List of types for declaring the `SerializableTuple` type.
        serializable_tuple_args = ",\n".join(
            f"      {binding.type} /* {binding.name} */"
            for binding in (base_ctor_arguments + attributes)
        )

        # Arguments used for forwarding the tuple elements to the constructor.
        destructure_tuple_args = ", ".join(
            f"std::get<{i}>(tpl)"
            for i in range(len(base_ctor_arguments) + len(extra_ctor_arguments))
        )

        # List of constructor parameters
        ctor_parameters = ", ".join(
            binding.decl() for binding in (base_ctor_arguments + extra_ctor_arguments)
        )

        # Call the base class `ViewMeta` constructor.
        #
        # Both of `is_multi_output` and `is_as_strided` are known values, given the
        # operation schema.
        is_multi_output_str = str(self.is_multi_output).lower()
        is_as_strided_str = str(self.is_as_strided).lower()

        base_ctor_bindings = ", ".join(
            [
                # `has_symbolic_inputs` is always taken as parameter.
                functionalization.has_symbolic_inputs_binding.name,
                f"/*is_multi_output=*/{is_multi_output_str}",
                f"/*is_as_strided=*/{is_as_strided_str}",
                # `out_index` is know if the operation returns only one value. Otherwise,
                # we also take it as parameter.
                f"/*out_index=*/{self.out_index}",
            ]
        )

        # Assignments of `extra_ctor_arguments` to their corresponding fields.
        # These are extra fields to-be-declared in this specialization.
        #
        # We need to set `allow_expensive_conversions`, since we are storing owned versions
        # of the non-owning arguments.
        ctor_assignments = ",\n".join(
            f"        {e.type.name}({e.expr})"
            for e in translate(
                extra_ctor_arguments,
                attributes,
                method=False,
                allow_expensive_conversions=True,
            )
        )

        # List of arguments for constructing the `SerializableTuple` from an instance.
        tuple_arguments = ", ".join(
            binding.name for binding in (base_ctor_arguments + attributes)
        )

        # List of field declarations.
        attr_declarations = "\n".join(f"  {binding.decl()};" for binding in attributes)

        # Override `to_out_index` if this operation returns more than 1 value.
        to_out_index_decl = ""
        if self.is_multi_output:
            to_out_index_decl = (
                "  std::shared_ptr<ViewMeta> to_out_index(int64_t out_idx) override;"
            )

        return [
            f"""
struct TORCH_API {self.classname} : public ViewMeta {{
  FUNCTIONALIZATION_VIEWMETA_NAME({self.classname})
  FUNCTIONALIZATION_VIEWMETA_SERIALIZABLE_TUPLE(\n{serializable_tuple_args});

  {self.classname}(const SerializableTuple& tpl)
      : {self.classname}({destructure_tuple_args}) {{}}

  {self.classname}({ctor_parameters})
      : at::functionalization::ViewMeta({base_ctor_bindings}),
{ctor_assignments} {{}}

  Tensor forward(const Tensor& base) override;
  Tensor reverse(const Tensor& base, const Tensor& mutated_view) override;
{to_out_index_decl}

  SerializableTuple to_serializable_tuple() {{
    return std::make_tuple({tuple_arguments});
  }}

{attr_declarations}
}};
"""
        ]

    # Generate a call to the actual operation.
    def opcall(self, is_reverse: bool, reapply_views: bool) -> str:
        opname = functionalization.name(
            self.g,
            is_reverse=is_reverse,
            include_namespace=True,
            reapply_views=reapply_views,
        )

        # Expected arguments for the operation.
        if self.g.view_copy is None:
            raise AssertionError("Expected view_copy to be non-None")
        op_arguments = functionalization.op_arguments(self.g.view_copy.func, is_reverse)

        # The context is composed by the constructor arguments (which are also
        # the field variables stored in the instance), and the `base` tensor.
        context = [functionalization.base_binding]
        context += functionalization.base_ctor_arguments(self.f.func)
        context += functionalization.attributes(self.f.func)

        # If we are generating the call for the reverse function, we also have
        # access to `mutated_view` argument.
        if is_reverse:
            context.append(functionalization.mutated_view_binding)

        arguments = ", ".join(
            [e.expr for e in translate(context, op_arguments, method=False)]
        )

        # Index the result if this operation returns multiple values.
        maybe_index = ""
        if not is_reverse and self.is_multi_output:
            maybe_index = f"[{self.out_index}]"

        return f"{opname}({arguments}){maybe_index}"

    def impl(self) -> list[str]:
        functions = [
            f"""
at::Tensor {self.classname}::forward(const at::Tensor& base) {{
  if (reapply_views) {{
    return {self.opcall(is_reverse=False, reapply_views=True)};
  }} else {{
    return {self.opcall(is_reverse=False, reapply_views=False)};
  }}
}}""",
            f"""
at::Tensor {self.classname}::reverse(const at::Tensor& base, const Tensor& mutated_view) {{
  return {self.opcall(is_reverse=True, reapply_views=True)};
}}""",
        ]

        # If this operation returns multiple values, also generate a `to_out_index`
        # implementation.
        if self.is_multi_output:
            functions.append(f"""
std::shared_ptr<at::functionalization::ViewMeta> {self.classname}::to_out_index(int64_t out_index) {{
  return {self.new("out_index")};
}}
""")

        return functions

    # Create the Python binding for this specialized class.
    def binding(self) -> list[str]:
        name = functionalization.classname(self.f.func, with_namespace=True)
        return [f"  create_binding_with_pickle<{name}>(functionalization);"]

    # Generate an instantiation of this specialized class.
    def new(self, out_index: str = "0") -> str:
        name = functionalization.classname(self.f.func, with_namespace=True)
        ctor_arguments = functionalization.base_ctor_arguments(
            self.f.func
        ) + functionalization.extra_ctor_arguments(self.f.func)
        # Replace the `out_index` parameter with the given `out_index`.
        arguments = ", ".join(
            binding.name if binding.name != "out_index" else out_index
            for binding in ctor_arguments
        )
        return f"std::make_shared<{name}>({arguments})"

    # Run the function `run` for both: `view` and `view_inplace` functions.
    @staticmethod
    def map(
        g: NativeFunctionsViewGroup, run: Callable[[ViewMetaSpecialization], list[str]]
    ) -> list[str]:
        def maybe_run(f: NativeFunction | None) -> list[str]:
            if f is None:
                return []
            with native_function_manager(f):
                return run(ViewMetaSpecialization(g, f))

        return list(concatMap(maybe_run, (g.view, g.view_inplace)))


def gen_functionalization_view_meta_classes_base(
    selector: SelectiveBuilder,
    g: NativeFunctionsViewGroup,
    run: Callable[[ViewMetaSpecialization], list[str]],
) -> list[str]:
    if not selector.include_all_operators:
        return []

    if g.composite:
        return []

    return ViewMetaSpecialization.map(g, run)


def gen_functionalization_view_meta_classes_decl(
    selector: SelectiveBuilder, g: NativeFunctionsViewGroup
) -> list[str]:
    return gen_functionalization_view_meta_classes_base(
        selector, g, ViewMetaSpecialization.decl
    )


def gen_functionalization_view_meta_classes_impl(
    selector: SelectiveBuilder, g: NativeFunctionsViewGroup
) -> list[str]:
    return gen_functionalization_view_meta_classes_base(
        selector, g, ViewMetaSpecialization.impl
    )


def gen_functionalization_view_meta_classes_binding(
    selector: SelectiveBuilder, g: NativeFunctionsViewGroup
) -> list[str]:
    return gen_functionalization_view_meta_classes_base(
        selector, g, ViewMetaSpecialization.binding
    )


# Generates the Python bindings for the `ViewMeta` specialized classes.
def gen_functionalization_view_meta_classes(
    native_functions_path: str,
    tags_path: str,
    selector: SelectiveBuilder,
    install_dir: str,
    template_dir: str,
) -> None:
    from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml

    # Parse the native_functions.yaml.
    # Then, group them into `NativeFunctionsViewGroup`.
    #
    # This is the same steps we do in gen.py (ATen codegen).
    native_functions = parse_native_yaml(
        native_functions_path, tags_path
    ).native_functions
    native_functions_with_view_groups = get_grouped_by_view_native_functions(
        native_functions
    )
    view_groups = [
        g
        for g in native_functions_with_view_groups
        if isinstance(g, NativeFunctionsViewGroup)
    ]

    fm = FileManager(install_dir=install_dir, template_dir=template_dir, dry_run=False)
    fm.write(
        "ViewMetaClassesPythonBinding.cpp",
        lambda: {
            "view_meta_bindings": list(
                concatMap(
                    lambda g: gen_functionalization_view_meta_classes_binding(
                        selector, g
                    ),
                    view_groups,
                )
            ),
        },
    )


def gen_functionalization_registration(
    selector: SelectiveBuilder,
    g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
    composite_implicit_autograd_index: BackendIndex,
) -> list[str]:
    @with_native_function
    def emit_registration_helper(f: NativeFunction) -> str:
        if f.has_composite_implicit_autograd_kernel:
            metadata = composite_implicit_autograd_index.get_kernel(f)
            if metadata is None:
                raise AssertionError(
                    f"Expected metadata for composite implicit autograd kernel: {f.func}"
                )
            native_api_name = metadata.kernel
            sig = NativeSignature(f.func, symint=metadata.supports_symint())
            # Note [Composite view ops in the functionalization pass]
            # We don't need to worry about implemententing functionalization kernels for views with
            # CompositeImplicitAutograd kernels, because we can just decompose them into their base operators.
            # We can't just opt the entire Functionalization dispatch key into the composite keyset though,
            # because we don't want to decompose non-view ops that are composite, like `at::ones`.
            registration_str = (
                f"static_cast<{sig.ptr_type()}>(at::native::{native_api_name})"
            )
        else:
            # non-composite view ops (and inplace ops) get a normal registration.
            registration_str = f"TORCH_FN(functionalization::{wrapper_name(f.func)})"
        return f'm.impl("{f.func.name}", {registration_str});'

    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        # functionalization needs to register kernels for view + view_inplace ops
        # See Note [Functionalization <> torch.Tensor constructor]
        if str(g.view.func.name) == "lift_fresh":
            return []
        view_str = []
        view_str.append(emit_registration_helper(g.view))
        if g.view_inplace is not None:
            if not g.view_inplace.is_view_op:
                raise AssertionError(
                    f"Expected view_inplace to be a view op: {g.view_inplace.func}"
                )
            view_str.append(emit_registration_helper(g.view_inplace))
        return view_str

    elif isinstance(g, NativeFunctionsGroup):
        # Gets a hand-written functionalization kernel
        if g.inplace is not None and str(g.inplace.func.name) == "set_.source_Tensor":
            fns = []
        else:
            fns = list(g.functions())
    else:
        if str(g.func.name) in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION:
            return []
        fns = [g]

    registrations = []
    for f in fns:
        if f.has_composite_implicit_autograd_kernel:
            continue
        if str(f.func.name) == "lift":
            # See Note [Functionalization <> torch.Tensor constructor]
            return []
        if str(f.func.name) == "resize_":
            # See Note [resize_ in Functionalization]
            return []
        if str(f.func.name.name) != "set_":
            if f.is_view_op:
                raise AssertionError(f"Unexpected view op: {f.func}")
        # functionalization needs to generate and register kernels for inplace ops.
        # We *also* need to directly register CompositeImplicitAUtograd kernels
        # so that they decompose properly before functioanlization.
        if modifies_arguments(f):
            registrations.append(emit_registration_helper(f))
    return registrations


def gen_functionalization_definition(
    selector: SelectiveBuilder,
    # Note: Ideally this code should never have to look at NativeFunction
    # (and instead only need to operate on grouped NativeFunctions).
    # The only reason currently is because we need to emit direct dispatch registrations
    # For CompositeImplicitAutograd operators, which are potentially ungrouped.
    g: NativeFunction | NativeFunctionsGroup | NativeFunctionsViewGroup,
) -> list[str]:
    # Don't generate kernels in mobile build
    if not selector.include_all_operators:
        return []

    if isinstance(g, NativeFunctionsViewGroup):
        # Case 1: emit view -> view_copy kernels for the functionalization pass
        view_defs = []
        if not g.composite:
            # invariant: NativeFunctionsViewGroup's always have a view_copy operator
            # if the view is not composite (implicit autograd)
            if g.view_copy is None:
                raise AssertionError(
                    f"Expected view_copy to be non-None: {dataclass_repr(g, indent=1)}"
                )
            view_defs.append(emit_view_functionalization_body(g, view_inplace=False))
            if g.view_inplace is not None:
                view_defs.append(emit_view_functionalization_body(g, view_inplace=True))
        return view_defs
    elif isinstance(g, NativeFunction):
        # Invariant: all mutable operators that we need to handle in functionalization
        # should have been properly grouped up.
        # TODO: The below ops all have "problematic" schemas that prevent them from
        # getting functionalized. Instead of bending over backwards to get things to work,
        # I think we should either:
        # (1) fix their schemas (BC-breaking)
        # (2) hand-write their functionalization kernels
        if (
            str(g.func.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION
            and str(g.func.name.name) not in MUTABLE_OPS_NOT_USING_FUNCTIONALIZATION
        ):
            if not (
                g.has_composite_implicit_autograd_kernel or not modifies_arguments(g)
            ):
                raise AssertionError(
                    f"Expected composite implicit autograd kernel or non-modifying function: {g.func}"
                )
        return []
    else:
        # Case 2: emit inplace -> out-of-place kernels for the functionalization pass
        mutation_defs = []
        mutation_defs.append(emit_inplace_functionalization_body(g.out, g))
        if g.inplace is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.inplace, g))
        if g.mutable is not None:
            mutation_defs.append(emit_inplace_functionalization_body(g.mutable, g))
        return mutation_defs
    return []
<|endoftext|>from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


# Simple dynamic scoping implementation.  The name "parametrize" comes
# from Racket.
#
# WARNING WARNING: LOOKING TO EDIT THIS FILE?  Think carefully about
# why you need to add a toggle to the global behavior of code
# generation.  The parameters here should really only be used
# for "temporary" situations, where we need to temporarily change
# the codegen in some cases because we cannot conveniently update
# all call sites, and are slated to be eliminated once all call
# sites are eliminated.  If you don't have a plan for how to get there,
# DON'T add a new entry here.


class Locals(threading.local):
    use_const_ref_for_mutable_tensors: bool | None = None
    use_ilistref_for_tensor_lists: bool | None = None


_locals = Locals()


def use_const_ref_for_mutable_tensors() -> bool:
    if _locals.use_const_ref_for_mutable_tensors is None:
        raise AssertionError(
            "need to initialize local.use_const_ref_for_mutable_tensors with "
            "local.parametrize"
        )
    return _locals.use_const_ref_for_mutable_tensors


def use_ilistref_for_tensor_lists() -> bool:
    if _locals.use_ilistref_for_tensor_lists is None:
        raise AssertionError(
            "need to initialize local.use_ilistref_for_tensor_lists with local.parametrize"
        )
    return _locals.use_ilistref_for_tensor_lists


@contextmanager
def parametrize(
    *, use_const_ref_for_mutable_tensors: bool, use_ilistref_for_tensor_lists: bool
) -> Iterator[None]:
    old_use_const_ref_for_mutable_tensors = _locals.use_const_ref_for_mutable_tensors
    old_use_ilistref_for_tensor_lists = _locals.use_ilistref_for_tensor_lists
    try:
        _locals.use_const_ref_for_mutable_tensors = use_const_ref_for_mutable_tensors
        _locals.use_ilistref_for_tensor_lists = use_ilistref_for_tensor_lists
        yield
    finally:
        _locals.use_const_ref_for_mutable_tensors = (
            old_use_const_ref_for_mutable_tensors
        )
        _locals.use_ilistref_for_tensor_lists = old_use_ilistref_for_tensor_lists
<|endoftext|>from __future__ import annotations

import contextlib
import functools
from typing import Any, TYPE_CHECKING, TypeVar

import torchgen.local as local
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
from torchgen.utils import context, S, T


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


# Helper functions for defining generators on things in the model

F = TypeVar(
    "F",
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
    NativeFunction | NativeFunctionsGroup,
    NativeFunction | NativeFunctionsViewGroup,
)

F2 = TypeVar(
    "F2",
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunction | None,
    bool,
    str,
)

F3 = TypeVar("F3", tuple[NativeFunction, Any], list[NativeFunction])


@contextlib.contextmanager
def native_function_manager(
    g: NativeFunctionsGroup | NativeFunctionsViewGroup | NativeFunction,
) -> Iterator[None]:
    if isinstance(g, NativeFunctionsGroup):
        # By default, we associate all errors with structured native functions
        # with the out variant.  In some cases, it might be better to have
        # a more specific place to hang things; if so, use
        # native_function_manager again on the inside
        f = g.out
    elif isinstance(g, NativeFunctionsViewGroup):
        # We associate errors with the view operator
        f = g.view
    else:
        f = g
    with context(lambda: f"in native_functions.yaml line {f.loc}:\n  {f.func}"):
        with local.parametrize(
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=f.part_of_structured_group,
        ):
            yield


# Given a function that operates on NativeFunction, wrap it into a new function
# that sets some appropriate context managers for that native function.
# YOU MUST WRAP FUNCTIONS IN THIS for calls to api modules to be sound
# (you will get an error if we try to access the local variables without having
# set them).
def with_native_function(func: Callable[[F], T]) -> Callable[[F], T]:
    @functools.wraps(func)
    def wrapper(f: F) -> T:
        with native_function_manager(f):
            return func(f)

    return wrapper


def with_native_function_and(func: Callable[[F, F2], T]) -> Callable[[F, F2], T]:
    @functools.wraps(func)
    def wrapper(f: F, f2: F2) -> T:
        # The first native_function is assumed to be the one with the appropriate context.
        with native_function_manager(f):
            return func(f, f2)

    return wrapper


def method_with_native_function(func: Callable[[S, F], T]) -> Callable[[S, F], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F) -> T:
        with native_function_manager(f):
            return func(slf, f)

    return wrapper


def method_with_nested_native_function(
    func: Callable[[S, F3], T],
) -> Callable[[S, F3], T]:
    @functools.wraps(func)
    def wrapper(slf: S, f: F3) -> T:
        with native_function_manager(f[0]):
            return func(slf, f)

    return wrapper


# Convenience decorator for functions that explicitly take in a BackendIndex,
# instead of indirectly taking one in as a closure
def with_native_function_and_index(
    func: Callable[[F, BackendIndex], T],
) -> Callable[[F, BackendIndex], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_index: BackendIndex) -> T:
        with native_function_manager(f):
            return func(f, backend_index)

    return wrapper


# Convenience decorator for functions that explicitly take in a Dict of BackendIndices
def with_native_function_and_indices(
    func: Callable[[F, dict[DispatchKey, BackendIndex]], T],
) -> Callable[[F, dict[DispatchKey, BackendIndex]], T]:
    @functools.wraps(func)
    def wrapper(f: F, backend_indices: dict[DispatchKey, BackendIndex]) -> T:
        with native_function_manager(f):
            return func(f, backend_indices)

    return wrapper
<|endoftext|>from __future__ import annotations

import itertools
import re
import textwrap
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# match $identifier or ${identifier} and replace with value in env
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block substitution by indenting to that depth and putting each element
# of the list on its own line
# if the identifier is on a line starting with non-whitespace and a list
# then it is comma separated ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.


class CodeTemplate:
    substitution_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str
    filename: str

    @staticmethod
    def from_file(filename: str) -> CodeTemplate:
        with open(filename) as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        self.pattern = pattern
        self.filename = filename

    def substitute(
        self, env: Mapping[str, object] | None = None, **kwargs: object
    ) -> str:
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            if env is None:
                raise AssertionError("env must be non-None")
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            content = "\n".join(
                itertools.chain.from_iterable(str(e).splitlines() for e in v)
            )
            content = textwrap.indent(content, prefix=indent)
            # Remove trailing whitespace on each line
