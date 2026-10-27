    if host == "":
        host = request.get_header("Host", "")

    # remove port, if present
    host = _cut_port_re.sub("", host, 1)
    return host.lower()

class Request:

    def __init__(self, url, data=None, headers={},
                 origin_req_host=None, unverifiable=False,
                 method=None):
        self.full_url = url
        self.headers = {}
        self.unredirected_hdrs = {}
        self._data = None
        self.data = data
        self._tunnel_host = None
        for key, value in headers.items():
            self.add_header(key, value)
        if origin_req_host is None:
            origin_req_host = request_host(self)
        self.origin_req_host = origin_req_host
        self.unverifiable = unverifiable
        if method:
            self.method = method

    @property
    def full_url(self):
        if self.fragment:
            return '{}#{}'.format(self._full_url, self.fragment)
        return self._full_url

    @full_url.setter
    def full_url(self, url):
        # unwrap('<URL:type://host/path>') --> 'type://host/path'
        self._full_url = unwrap(url)
        self._full_url, self.fragment = _splittag(self._full_url)
        self._parse()

    @full_url.deleter
    def full_url(self):
        self._full_url = None
        self.fragment = None
        self.selector = ''

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data != self._data:
            self._data = data
            # issue 16464
            # if we change data we need to remove content-length header
            # (cause it's most probably calculated for previous value)
            if self.has_header("Content-length"):
                self.remove_header("Content-length")

    @data.deleter
    def data(self):
        self.data = None

    def _parse(self):
        self.type, rest = _splittype(self._full_url)
        if self.type is None:
            raise ValueError("unknown url type: %r" % self.full_url)
        self.host, self.selector = _splithost(rest)
        if self.host:
            self.host = unquote(self.host)

    def get_method(self):
        """Return a string indicating the HTTP request method."""
        default_method = "POST" if self.data is not None else "GET"
        return getattr(self, 'method', default_method)

    def get_full_url(self):
        return self.full_url

    def set_proxy(self, host, type):
        if self.type == 'https' and not self._tunnel_host:
            self._tunnel_host = self.host
        else:
            self.type= type
            self.selector = self.full_url
        self.host = host

    def has_proxy(self):
        return self.selector == self.full_url

    def add_header(self, key, val):
        # useful for something like authentication
        self.headers[key.capitalize()] = val

    def add_unredirected_header(self, key, val):
        # will not be added to a redirected request
        self.unredirected_hdrs[key.capitalize()] = val

    def has_header(self, header_name):
        return (header_name in self.headers or
                header_name in self.unredirected_hdrs)

    def get_header(self, header_name, default=None):
        return self.headers.get(
            header_name,
            self.unredirected_hdrs.get(header_name, default))

    def remove_header(self, header_name):
        self.headers.pop(header_name, None)
        self.unredirected_hdrs.pop(header_name, None)

    def header_items(self):
        hdrs = {**self.unredirected_hdrs, **self.headers}
        return list(hdrs.items())

class OpenerDirector:
    def __init__(self):
        client_version = "Python-urllib/%s" % __version__
        self.addheaders = [('User-agent', client_version)]
        # self.handlers is retained only for backward compatibility
        self.handlers = []
        # manage the individual handlers
        self.handle_open = {}
        self.handle_error = {}
        self.process_response = {}
        self.process_request = {}

    def add_handler(self, handler):
        if not hasattr(handler, "add_parent"):
            raise TypeError("expected BaseHandler instance, got %r" %
                            type(handler))

        added = False
        for meth in dir(handler):
            if meth in ["redirect_request", "do_open", "proxy_open"]:
                # oops, coincidental match
                continue

            i = meth.find("_")
            protocol = meth[:i]
            condition = meth[i+1:]

            if condition.startswith("error"):
                j = condition.find("_") + i + 1
                kind = meth[j+1:]
                try:
                    kind = int(kind)
                except ValueError:
                    pass
                lookup = self.handle_error.get(protocol, {})
                self.handle_error[protocol] = lookup
            elif condition == "open":
                kind = protocol
                lookup = self.handle_open
            elif condition == "response":
                kind = protocol
                lookup = self.process_response
            elif condition == "request":
                kind = protocol
                lookup = self.process_request
            else:
                continue

            handlers = lookup.setdefault(kind, [])
            if handlers:
                bisect.insort(handlers, handler)
            else:
                handlers.append(handler)
            added = True

        if added:
            bisect.insort(self.handlers, handler)
            handler.add_parent(self)

    def close(self):
        # Only exists for backwards compatibility.
        pass

    def _call_chain(self, chain, kind, meth_name, *args):
        # Handlers raise an exception if no one else should try to handle
        # the request, or return None if they can't but another handler
        # could.  Otherwise, they return the response.
        handlers = chain.get(kind, ())
        for handler in handlers:
            func = getattr(handler, meth_name)
            result = func(*args)
            if result is not None:
                return result

    def open(self, fullurl, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        # accept a URL or a Request object
        if isinstance(fullurl, str):
            req = Request(fullurl, data)
        else:
            req = fullurl
            if data is not None:
                req.data = data

        req.timeout = timeout
        protocol = req.type

        # pre-process request
        meth_name = protocol+"_request"
        for processor in self.process_request.get(protocol, []):
            meth = getattr(processor, meth_name)
            req = meth(req)

        sys.audit('urllib.Request', req.full_url, req.data, req.headers, req.get_method())
        response = self._open(req, data)

        # post-process response
        meth_name = protocol+"_response"
        for processor in self.process_response.get(protocol, []):
            meth = getattr(processor, meth_name)
            response = meth(req, response)

        return response

    def _open(self, req, data=None):
        result = self._call_chain(self.handle_open, 'default',
                                  'default_open', req)
        if result:
            return result

        protocol = req.type
        result = self._call_chain(self.handle_open, protocol, protocol +
                                  '_open', req)
        if result:
            return result

        return self._call_chain(self.handle_open, 'unknown',
                                'unknown_open', req)

    def error(self, proto, *args):
        if proto in ('http', 'https'):
            # XXX http[s] protocols are special-cased
            dict = self.handle_error['http'] # https is not different than http
            proto = args[2]  # YUCK!
            meth_name = 'http_error_%s' % proto
            http_err = 1
            orig_args = args
        else:
            dict = self.handle_error
            meth_name = proto + '_error'
            http_err = 0
        args = (dict, proto, meth_name) + args
        result = self._call_chain(*args)
        if result:
            return result

        if http_err:
            args = (dict, 'default', 'http_error_default') + orig_args
            return self._call_chain(*args)

# XXX probably also want an abstract factory that knows when it makes
# sense to skip a superclass in favor of a subclass and when it might
# make sense to include both

def build_opener(*handlers):
    """Create an opener object from a list of handlers.

    The opener will use several default handlers, including support
    for HTTP, FTP and when applicable HTTPS.

    If any of the handlers passed as arguments are subclasses of the
    default handlers, the default handlers will not be used.
    """
    opener = OpenerDirector()
    default_classes = [ProxyHandler, UnknownHandler, HTTPHandler,
                       HTTPDefaultErrorHandler, HTTPRedirectHandler,
                       FTPHandler, FileHandler, HTTPErrorProcessor,
                       DataHandler]
    if hasattr(http.client, "HTTPSConnection"):
        default_classes.append(HTTPSHandler)
    skip = set()
    for klass in default_classes:
        for check in handlers:
            if isinstance(check, type):
                if issubclass(check, klass):
                    skip.add(klass)
            elif isinstance(check, klass):
                skip.add(klass)
    for klass in skip:
        default_classes.remove(klass)

    for klass in default_classes:
        opener.add_handler(klass())

    for h in handlers:
        if isinstance(h, type):
            h = h()
        opener.add_handler(h)
    return opener

class BaseHandler:
    handler_order = 500

    def add_parent(self, parent):
        self.parent = parent

    def close(self):
        # Only exists for backwards compatibility
        pass

    def __lt__(self, other):
        if not hasattr(other, "handler_order"):
            # Try to preserve the old behavior of having custom classes
            # inserted after default ones (works only for custom user
            # classes which are not aware of handler_order).
            return True
        return self.handler_order < other.handler_order


class HTTPErrorProcessor(BaseHandler):
    """Process HTTP error responses."""
    handler_order = 1000  # after all other processing

    def http_response(self, request, response):
        code, msg, hdrs = response.code, response.msg, response.info()

        # According to RFC 2616, "2xx" code indicates that the client's
        # request was successfully received, understood, and accepted.
        if not (200 <= code < 300):
            response = self.parent.error(
                'http', request, response, code, msg, hdrs)

        return response

    https_response = http_response

class HTTPDefaultErrorHandler(BaseHandler):
    def http_error_default(self, req, fp, code, msg, hdrs):
        raise HTTPError(req.full_url, code, msg, hdrs, fp)

class HTTPRedirectHandler(BaseHandler):
    # maximum number of redirections to any single URL
    # this is needed because of the state that cookies introduce
    max_repeats = 4
    # maximum total number of redirections (regardless of URL) before
    # assuming we're in a loop
    max_redirections = 10

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Return a Request or None in response to a redirect.

        This is called by the http_error_30x methods when a
        redirection response is received.  If a redirection should
        take place, return a new Request to allow http_error_30x to
        perform the redirect.  Otherwise, raise HTTPError if no-one
        else should try to handle this url.  Return None if you can't
        but another Handler might.
        """
        m = req.get_method()
        if (not (code in (301, 302, 303, 307) and m in ("GET", "HEAD")
            or code in (301, 302, 303) and m == "POST")):
            raise HTTPError(req.full_url, code, msg, headers, fp)

        # Strictly (according to RFC 2616), 301 or 302 in response to
        # a POST MUST NOT cause a redirection without confirmation
        # from the user (of urllib.request, in this case).  In practice,
        # essentially all clients do redirect in this case, so we do
        # the same.

        # Be conciliant with URIs containing a space.  This is mainly
        # redundant with the more complete encoding done in http_error_302(),
        # but it is kept for compatibility with other callers.
        newurl = newurl.replace(' ', '%20')

        CONTENT_HEADERS = ("content-length", "content-type")
        newheaders = {k: v for k, v in req.headers.items()
                      if k.lower() not in CONTENT_HEADERS}
        return Request(newurl,
                       headers=newheaders,
                       origin_req_host=req.origin_req_host,
                       unverifiable=True)

    # Implementation note: To avoid the server sending us into an
    # infinite loop, the request object needs to track what URLs we
    # have already seen.  Do this by adding a handler-specific
    # attribute to the Request object.
    def http_error_302(self, req, fp, code, msg, headers):
        # Some servers (incorrectly) return multiple Location headers
        # (so probably same goes for URI).  Use first header.
        if "location" in headers:
            newurl = headers["location"]
        elif "uri" in headers:
            newurl = headers["uri"]
        else:
            return

        # fix a possible malformed URL
        urlparts = urlparse(newurl)

        # For security reasons we don't allow redirection to anything other
        # than http, https or ftp.

        if urlparts.scheme not in ('http', 'https', 'ftp', ''):
            raise HTTPError(
                newurl, code,
                "%s - Redirection to url '%s' is not allowed" % (msg, newurl),
                headers, fp)

        if not urlparts.path and urlparts.netloc:
            urlparts = list(urlparts)
            urlparts[2] = "/"
        newurl = urlunparse(urlparts)

        # http.client.parse_headers() decodes as ISO-8859-1.  Recover the
        # original bytes and percent-encode non-ASCII bytes, and any special
        # characters such as the space.
        newurl = quote(
            newurl, encoding="iso-8859-1", safe=string.punctuation)
        newurl = urljoin(req.full_url, newurl)

        # XXX Probably want to forget about the state of the current
        # request, although that might interact poorly with other
        # handlers that also use handler-specific request attributes
        new = self.redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return

        # loop detection
        # .redirect_dict has a key url if url was previously visited.
        if hasattr(req, 'redirect_dict'):
            visited = new.redirect_dict = req.redirect_dict
            if (visited.get(newurl, 0) >= self.max_repeats or
                len(visited) >= self.max_redirections):
                raise HTTPError(req.full_url, code,
                                self.inf_msg + msg, headers, fp)
        else:
            visited = new.redirect_dict = req.redirect_dict = {}
        visited[newurl] = visited.get(newurl, 0) + 1

        # Don't close the fp until we are sure that we won't use it
        # with HTTPError.
        fp.read()
        fp.close()

        return self.parent.open(new, timeout=req.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_302

    inf_msg = "The HTTP server returned a redirect error that would " \
              "lead to an infinite loop.\n" \
              "The last 30x error message was:\n"


def _parse_proxy(proxy):
    """Return (scheme, user, password, host/port) given a URL or an authority.

    If a URL is supplied, it must have an authority (host:port) component.
    According to RFC 3986, having an authority component means the URL must
    have two slashes after the scheme.
    """
    scheme, r_scheme = _splittype(proxy)
    if not r_scheme.startswith("/"):
        # authority
        scheme = None
        authority = proxy
    else:
        # URL
        if not r_scheme.startswith("//"):
            raise ValueError("proxy URL with no authority: %r" % proxy)
        # We have an authority, so for RFC 3986-compliant URLs (by ss 3.
        # and 3.3.), path is empty or starts with '/'
        if '@' in r_scheme:
            host_separator = r_scheme.find('@')
            end = r_scheme.find("/", host_separator)
        else:
            end = r_scheme.find("/", 2)
        if end == -1:
            end = None
        authority = r_scheme[2:end]
    userinfo, hostport = _splituser(authority)
    if userinfo is not None:
        user, password = _splitpasswd(userinfo)
    else:
        user = password = None
    return scheme, user, password, hostport

class ProxyHandler(BaseHandler):
    # Proxies must be in front
    handler_order = 100

    def __init__(self, proxies=None):
        if proxies is None:
            proxies = getproxies()
        assert hasattr(proxies, 'keys'), "proxies must be a mapping"
        self.proxies = proxies
        for type, url in proxies.items():
            type = type.lower()
            setattr(self, '%s_open' % type,
                    lambda r, proxy=url, type=type, meth=self.proxy_open:
                        meth(r, proxy, type))

    def proxy_open(self, req, proxy, type):
        orig_type = req.type
        proxy_type, user, password, hostport = _parse_proxy(proxy)
        if proxy_type is None:
            proxy_type = orig_type

        if req.host and proxy_bypass(req.host):
            return None

        if user and password:
            user_pass = '%s:%s' % (unquote(user),
                                   unquote(password))
            creds = base64.b64encode(user_pass.encode()).decode("ascii")
            req.add_header('Proxy-authorization', 'Basic ' + creds)
        hostport = unquote(hostport)
        req.set_proxy(hostport, proxy_type)
        if orig_type == proxy_type or orig_type == 'https':
            # let other handlers take care of it
            return None
        else:
            # need to start over, because the other handlers don't
            # grok the proxy's URL type
            # e.g. if we have a constructor arg proxies like so:
            # {'http': 'ftp://proxy.example.com'}, we may end up turning
            # a request for http://acme.example.com/a into one for
            # ftp://proxy.example.com/a
            return self.parent.open(req, timeout=req.timeout)

class HTTPPasswordMgr:

    def __init__(self):
        self.passwd = {}

    def add_password(self, realm, uri, user, passwd):
        # uri could be a single URI or a sequence
        if isinstance(uri, str):
            uri = [uri]
        if realm not in self.passwd:
            self.passwd[realm] = {}
        for default_port in True, False:
            reduced_uri = tuple(
                self.reduce_uri(u, default_port) for u in uri)
            self.passwd[realm][reduced_uri] = (user, passwd)

    def find_user_password(self, realm, authuri):
        domains = self.passwd.get(realm, {})
        for default_port in True, False:
            reduced_authuri = self.reduce_uri(authuri, default_port)
            for uris, authinfo in domains.items():
                for uri in uris:
                    if self.is_suburi(uri, reduced_authuri):
                        return authinfo
        return None, None

    def reduce_uri(self, uri, default_port=True):
        """Accept authority or URI and extract only the authority and path."""
        # note HTTP URLs do not have a userinfo component
        parts = urlsplit(uri)
        if parts[1]:
            # URI
            scheme = parts[0]
            authority = parts[1]
            path = parts[2] or '/'
        else:
            # host or host:port
            scheme = None
            authority = uri
            path = '/'
        host, port = _splitport(authority)
        if default_port and port is None and scheme is not None:
            dport = {"http": 80,
                     "https": 443,
                     }.get(scheme)
            if dport is not None:
                authority = "%s:%d" % (host, dport)
        return authority, path

    def is_suburi(self, base, test):
        """Check if test is below base in a URI tree

        Both args must be URIs in reduced form.
        """
        if base == test:
            return True
        if base[0] != test[0]:
            return False
        prefix = base[1]
        if prefix[-1:] != '/':
            prefix += '/'
        return test[1].startswith(prefix)


class HTTPPasswordMgrWithDefaultRealm(HTTPPasswordMgr):

    def find_user_password(self, realm, authuri):
        user, password = HTTPPasswordMgr.find_user_password(self, realm,
                                                            authuri)
        if user is not None:
            return user, password
        return HTTPPasswordMgr.find_user_password(self, None, authuri)


class HTTPPasswordMgrWithPriorAuth(HTTPPasswordMgrWithDefaultRealm):

    def __init__(self, *args, **kwargs):
        self.authenticated = {}
        super().__init__(*args, **kwargs)

    def add_password(self, realm, uri, user, passwd, is_authenticated=False):
        self.update_authenticated(uri, is_authenticated)
        # Add a default for prior auth requests
        if realm is not None:
            super().add_password(None, uri, user, passwd)
        super().add_password(realm, uri, user, passwd)

    def update_authenticated(self, uri, is_authenticated=False):
        # uri could be a single URI or a sequence
        if isinstance(uri, str):
            uri = [uri]

        for default_port in True, False:
            for u in uri:
                reduced_uri = self.reduce_uri(u, default_port)
                self.authenticated[reduced_uri] = is_authenticated

    def is_authenticated(self, authuri):
        for default_port in True, False:
            reduced_authuri = self.reduce_uri(authuri, default_port)
            for uri in self.authenticated:
                if self.is_suburi(uri, reduced_authuri):
                    return self.authenticated[uri]


class AbstractBasicAuthHandler:

    # XXX this allows for multiple auth-schemes, but will stupidly pick
    # the last one with a realm specified.

    # allow for double- and single-quoted realm values
    # (single quotes are a violation of the RFC, but appear in the wild)
    rx = re.compile('(?:^|,)'   # start of the string or ','
                    '[ \t]*'    # optional whitespaces
                    '([^ \t,]+)' # scheme like "Basic"
                    '[ \t]+'    # mandatory whitespaces
                    # realm=xxx
                    # realm='xxx'
                    # realm="xxx"
                    'realm=(["\']?)([^"\']*)\\2',
                    re.I)

    # XXX could pre-emptively send auth info already accepted (RFC 2617,
    # end of section 2, and section 1.2 immediately after "credentials"
    # production).

    def __init__(self, password_mgr=None):
        if password_mgr is None:
            password_mgr = HTTPPasswordMgr()
        self.passwd = password_mgr
        self.add_password = self.passwd.add_password

    def _parse_realm(self, header):
        # parse WWW-Authenticate header: accept multiple challenges per header
        found_challenge = False
        for mo in AbstractBasicAuthHandler.rx.finditer(header):
            scheme, quote, realm = mo.groups()
            if quote not in ['"', "'"]:
                warnings.warn("Basic Auth Realm was unquoted",
                              UserWarning, 3)

            yield (scheme, realm)

            found_challenge = True

        if not found_challenge:
            if header:
                scheme = header.split()[0]
            else:
                scheme = ''
            yield (scheme, None)

    def http_error_auth_reqed(self, authreq, host, req, headers):
        # host may be an authority (without userinfo) or a URL with an
        # authority
        headers = headers.get_all(authreq)
        if not headers:
            # no header found
            return

        unsupported = None
        for header in headers:
            for scheme, realm in self._parse_realm(header):
                if scheme.lower() != 'basic':
                    unsupported = scheme
                    continue

                if realm is not None:
                    # Use the first matching Basic challenge.
                    # Ignore following challenges even if they use the Basic
                    # scheme.
                    return self.retry_http_basic_auth(host, req, realm)

        if unsupported is not None:
            raise ValueError("AbstractBasicAuthHandler does not "
                             "support the following scheme: %r"
                             % (scheme,))

    def retry_http_basic_auth(self, host, req, realm):
        user, pw = self.passwd.find_user_password(realm, host)
        if pw is not None:
            raw = "%s:%s" % (user, pw)
            auth = "Basic " + base64.b64encode(raw.encode()).decode("ascii")
            if req.get_header(self.auth_header, None) == auth:
                return None
            req.add_unredirected_header(self.auth_header, auth)
            return self.parent.open(req, timeout=req.timeout)
        else:
            return None

    def http_request(self, req):
        if (not hasattr(self.passwd, 'is_authenticated') or
           not self.passwd.is_authenticated(req.full_url)):
            return req

        if not req.has_header('Authorization'):
            user, passwd = self.passwd.find_user_password(None, req.full_url)
            credentials = '{0}:{1}'.format(user, passwd).encode()
            auth_str = base64.standard_b64encode(credentials).decode()
            req.add_unredirected_header('Authorization',
                                        'Basic {}'.format(auth_str.strip()))
        return req

    def http_response(self, req, response):
        if hasattr(self.passwd, 'is_authenticated'):
            if 200 <= response.code < 300:
                self.passwd.update_authenticated(req.full_url, True)
            else:
                self.passwd.update_authenticated(req.full_url, False)
        return response

    https_request = http_request
    https_response = http_response



class HTTPBasicAuthHandler(AbstractBasicAuthHandler, BaseHandler):

    auth_header = 'Authorization'

    def http_error_401(self, req, fp, code, msg, headers):
        url = req.full_url
        response = self.http_error_auth_reqed('www-authenticate',
                                          url, req, headers)
        return response


class ProxyBasicAuthHandler(AbstractBasicAuthHandler, BaseHandler):

    auth_header = 'Proxy-authorization'

    def http_error_407(self, req, fp, code, msg, headers):
        # http_error_auth_reqed requires that there is no userinfo component in
        # authority.  Assume there isn't one, since urllib.request does not (and
        # should not, RFC 3986 s. 3.2.1) support requests for URLs containing
        # userinfo.
        authority = req.host
        response = self.http_error_auth_reqed('proxy-authenticate',
                                          authority, req, headers)
        return response


# Return n random bytes.
_randombytes = os.urandom


class AbstractDigestAuthHandler:
    # Digest authentication is specified in RFC 2617.

    # XXX The client does not inspect the Authentication-Info header
    # in a successful response.

    # XXX It should be possible to test this implementation against
    # a mock server that just generates a static set of challenges.

    # XXX qop="auth-int" supports is shaky

    def __init__(self, passwd=None):
        if passwd is None:
            passwd = HTTPPasswordMgr()
        self.passwd = passwd
        self.add_password = self.passwd.add_password
        self.retried = 0
        self.nonce_count = 0
        self.last_nonce = None

    def reset_retry_count(self):
        self.retried = 0

    def http_error_auth_reqed(self, auth_header, host, req, headers):
        authreq = headers.get(auth_header, None)
        if self.retried > 5:
            # Don't fail endlessly - if we failed once, we'll probably
            # fail a second time. Hm. Unless the Password Manager is
            # prompting for the information. Crap. This isn't great
            # but it's better than the current 'repeat until recursion
            # depth exceeded' approach <wink>
            raise HTTPError(req.full_url, 401, "digest auth failed",
                            headers, None)
        else:
            self.retried += 1
        if authreq:
            scheme = authreq.split()[0]
            if scheme.lower() == 'digest':
                return self.retry_http_digest_auth(req, authreq)
            elif scheme.lower() != 'basic':
                raise ValueError("AbstractDigestAuthHandler does not support"
                                 " the following scheme: '%s'" % scheme)

    def retry_http_digest_auth(self, req, auth):
        token, challenge = auth.split(' ', 1)
        chal = parse_keqv_list(filter(None, parse_http_list(challenge)))
        auth = self.get_authorization(req, chal)
        if auth:
            auth_val = 'Digest %s' % auth
            if req.headers.get(self.auth_header, None) == auth_val:
                return None
            req.add_unredirected_header(self.auth_header, auth_val)
            resp = self.parent.open(req, timeout=req.timeout)
            return resp

    def get_cnonce(self, nonce):
        # The cnonce-value is an opaque
        # quoted string value provided by the client and used by both client
        # and server to avoid chosen plaintext attacks, to provide mutual
        # authentication, and to provide some message integrity protection.
        # This isn't a fabulous effort, but it's probably Good Enough.
        s = "%s:%s:%s:" % (self.nonce_count, nonce, time.ctime())
        b = s.encode("ascii") + _randombytes(8)
        dig = hashlib.sha1(b).hexdigest()
        return dig[:16]

    def get_authorization(self, req, chal):
        try:
            realm = chal['realm']
            nonce = chal['nonce']
            qop = chal.get('qop')
            algorithm = chal.get('algorithm', 'MD5')
            # mod_digest doesn't send an opaque, even though it isn't
            # supposed to be optional
            opaque = chal.get('opaque', None)
        except KeyError:
            return None

        H, KD = self.get_algorithm_impls(algorithm)
        if H is None:
            return None

        user, pw = self.passwd.find_user_password(realm, req.full_url)
        if user is None:
            return None

        # XXX not implemented yet
        if req.data is not None:
            entdig = self.get_entity_digest(req.data, chal)
        else:
            entdig = None

        A1 = "%s:%s:%s" % (user, realm, pw)
        A2 = "%s:%s" % (req.get_method(),
                        # XXX selector: what about proxies and full urls
                        req.selector)
        # NOTE: As per  RFC 2617, when server sends "auth,auth-int", the client could use either `auth`
        #     or `auth-int` to the response back. we use `auth` to send the response back.
        if qop is None:
            respdig = KD(H(A1), "%s:%s" % (nonce, H(A2)))
        elif 'auth' in qop.split(','):
            if nonce == self.last_nonce:
                self.nonce_count += 1
            else:
                self.nonce_count = 1
                self.last_nonce = nonce
            ncvalue = '%08x' % self.nonce_count
            cnonce = self.get_cnonce(nonce)
            noncebit = "%s:%s:%s:%s:%s" % (nonce, ncvalue, cnonce, 'auth', H(A2))
            respdig = KD(H(A1), noncebit)
        else:
            # XXX handle auth-int.
            raise URLError("qop '%s' is not supported." % qop)

        # XXX should the partial digests be encoded too?

        base = 'username="%s", realm="%s", nonce="%s", uri="%s", ' \
               'response="%s"' % (user, realm, nonce, req.selector,
                                  respdig)
        if opaque:
            base += ', opaque="%s"' % opaque
        if entdig:
            base += ', digest="%s"' % entdig
        base += ', algorithm="%s"' % algorithm
        if qop:
            base += ', qop=auth, nc=%s, cnonce="%s"' % (ncvalue, cnonce)
        return base

    def get_algorithm_impls(self, algorithm):
        # lambdas assume digest modules are imported at the top level
        if algorithm == 'MD5':
            H = lambda x: hashlib.md5(x.encode("ascii")).hexdigest()
        elif algorithm == 'SHA':
            H = lambda x: hashlib.sha1(x.encode("ascii")).hexdigest()
        # XXX MD5-sess
        else:
            raise ValueError("Unsupported digest authentication "
                             "algorithm %r" % algorithm)
        KD = lambda s, d: H("%s:%s" % (s, d))
        return H, KD

    def get_entity_digest(self, data, chal):
        # XXX not implemented yet
        return None


class HTTPDigestAuthHandler(BaseHandler, AbstractDigestAuthHandler):
    """An authentication protocol defined by RFC 2069

    Digest authentication improves on basic authentication because it
    does not transmit passwords in the clear.
    """

    auth_header = 'Authorization'
    handler_order = 490  # before Basic auth

    def http_error_401(self, req, fp, code, msg, headers):
        host = urlparse(req.full_url)[1]
        retry = self.http_error_auth_reqed('www-authenticate',
                                           host, req, headers)
        self.reset_retry_count()
        return retry


class ProxyDigestAuthHandler(BaseHandler, AbstractDigestAuthHandler):

    auth_header = 'Proxy-Authorization'
    handler_order = 490  # before Basic auth

    def http_error_407(self, req, fp, code, msg, headers):
        host = req.host
        retry = self.http_error_auth_reqed('proxy-authenticate',
                                           host, req, headers)
        self.reset_retry_count()
        return retry

class AbstractHTTPHandler(BaseHandler):

    def __init__(self, debuglevel=0):
        self._debuglevel = debuglevel

    def set_http_debuglevel(self, level):
        self._debuglevel = level

    def _get_content_length(self, request):
        return http.client.HTTPConnection._get_content_length(
            request.data,
            request.get_method())

    def do_request_(self, request):
        host = request.host
        if not host:
            raise URLError('no host given')

        if request.data is not None:  # POST
            data = request.data
            if isinstance(data, str):
                msg = "POST data should be bytes, an iterable of bytes, " \
                      "or a file object. It cannot be of type str."
                raise TypeError(msg)
            if not request.has_header('Content-type'):
                request.add_unredirected_header(
                    'Content-type',
                    'application/x-www-form-urlencoded')
            if (not request.has_header('Content-length')
                    and not request.has_header('Transfer-encoding')):
                content_length = self._get_content_length(request)
                if content_length is not None:
                    request.add_unredirected_header(
                            'Content-length', str(content_length))
                else:
                    request.add_unredirected_header(
                            'Transfer-encoding', 'chunked')

        sel_host = host
        if request.has_proxy():
            scheme, sel = _splittype(request.selector)
            sel_host, sel_path = _splithost(sel)
        if not request.has_header('Host'):
            request.add_unredirected_header('Host', sel_host)
        for name, value in self.parent.addheaders:
            name = name.capitalize()
            if not request.has_header(name):
                request.add_unredirected_header(name, value)

        return request

    def do_open(self, http_class, req, **http_conn_args):
        """Return an HTTPResponse object for the request, using http_class.

        http_class must implement the HTTPConnection API from http.client.
        """
        host = req.host
        if not host:
            raise URLError('no host given')

        # will parse host:port
        h = http_class(host, timeout=req.timeout, **http_conn_args)
        h.set_debuglevel(self._debuglevel)

        headers = dict(req.unredirected_hdrs)
        headers.update({k: v for k, v in req.headers.items()
                        if k not in headers})

        # TODO(jhylton): Should this be redesigned to handle
        # persistent connections?

        # We want to make an HTTP/1.1 request, but the addinfourl
        # class isn't prepared to deal with a persistent connection.
        # It will try to read all remaining data from the socket,
        # which will block while the server waits for the next request.
        # So make sure the connection gets closed after the (only)
        # request.
        headers["Connection"] = "close"
        headers = {name.title(): val for name, val in headers.items()}

        if req._tunnel_host:
            tunnel_headers = {}
            proxy_auth_hdr = "Proxy-Authorization"
            if proxy_auth_hdr in headers:
                tunnel_headers[proxy_auth_hdr] = headers[proxy_auth_hdr]
                # Proxy-Authorization should not be sent to origin
                # server.
                del headers[proxy_auth_hdr]
            h.set_tunnel(req._tunnel_host, headers=tunnel_headers)

        try:
            try:
                h.request(req.get_method(), req.selector, req.data, headers,
                          encode_chunked=req.has_header('Transfer-encoding'))
            except OSError as err: # timeout error
                raise URLError(err)
            r = h.getresponse()
        except:
            h.close()
            raise

        # If the server does not send us a 'Connection: close' header,
        # HTTPConnection assumes the socket should be left open. Manually
        # mark the socket to be closed when this response object goes away.
        if h.sock:
            h.sock.close()
            h.sock = None

        r.url = req.get_full_url()
        # This line replaces the .msg attribute of the HTTPResponse
        # with .headers, because urllib clients expect the response to
        # have the reason in .msg.  It would be good to mark this
        # attribute is deprecated and get then to use info() or
        # .headers.
        r.msg = r.reason
        return r


class HTTPHandler(AbstractHTTPHandler):

    def http_open(self, req):
        return self.do_open(http.client.HTTPConnection, req)

    http_request = AbstractHTTPHandler.do_request_

if hasattr(http.client, 'HTTPSConnection'):

    class HTTPSHandler(AbstractHTTPHandler):

        def __init__(self, debuglevel=0, context=None, check_hostname=None):
            AbstractHTTPHandler.__init__(self, debuglevel)
            self._context = context
            self._check_hostname = check_hostname

        def https_open(self, req):
            return self.do_open(http.client.HTTPSConnection, req,
                context=self._context, check_hostname=self._check_hostname)

        https_request = AbstractHTTPHandler.do_request_

    __all__.append('HTTPSHandler')

class HTTPCookieProcessor(BaseHandler):
    def __init__(self, cookiejar=None):
        import http.cookiejar
        if cookiejar is None:
            cookiejar = http.cookiejar.CookieJar()
        self.cookiejar = cookiejar

    def http_request(self, request):
        self.cookiejar.add_cookie_header(request)
        return request

    def http_response(self, request, response):
        self.cookiejar.extract_cookies(response, request)
        return response

    https_request = http_request
    https_response = http_response

class UnknownHandler(BaseHandler):
    def unknown_open(self, req):
        type = req.type
        raise URLError('unknown url type: %s' % type)

def parse_keqv_list(l):
    """Parse list of key=value strings where keys are not duplicated."""
    parsed = {}
    for elt in l:
        k, v = elt.split('=', 1)
        if v[0] == '"' and v[-1] == '"':
            v = v[1:-1]
        parsed[k] = v
    return parsed

def parse_http_list(s):
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings.  A quoted-string could
    contain a comma.  A non-quoted string could have quotes in the
    middle.  Neither commas nor quotes count if they are escaped.
    Only double-quotes count, not single-quotes.
    """
    res = []
    part = ''

    escape = quote = False
    for cur in s:
        if escape:
            part += cur
            escape = False
            continue
        if quote:
            if cur == '\\':
                escape = True
                continue
            elif cur == '"':
                quote = False
            part += cur
            continue

        if cur == ',':
            res.append(part)
            part = ''
            continue

        if cur == '"':
            quote = True

        part += cur

    # append last part
    if part:
        res.append(part)

    return [part.strip() for part in res]

class FileHandler(BaseHandler):
    # Use local file or FTP depending on form of URL
    def file_open(self, req):
        url = req.selector
        if url[:2] == '//' and url[2:3] != '/' and (req.host and
                req.host != 'localhost'):
            if not req.host in self.get_names():
                raise URLError("file:// scheme is supported only on localhost")
        else:
            return self.open_local_file(req)

    # names for the localhost
    names = None
    def get_names(self):
        if FileHandler.names is None:
            try:
                FileHandler.names = tuple(
                    socket.gethostbyname_ex('localhost')[2] +
                    socket.gethostbyname_ex(socket.gethostname())[2])
            except socket.gaierror:
                FileHandler.names = (socket.gethostbyname('localhost'),)
        return FileHandler.names

    # not entirely sure what the rules are here
    def open_local_file(self, req):
        import email.utils
        import mimetypes
        host = req.host
        filename = req.selector
        localfile = url2pathname(filename)
        try:
            stats = os.stat(localfile)
            size = stats.st_size
            modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
            mtype = mimetypes.guess_type(filename)[0]
            headers = email.message_from_string(
                'Content-type: %s\nContent-length: %d\nLast-modified: %s\n' %
                (mtype or 'text/plain', size, modified))
            if host:
                host, port = _splitport(host)
            if not host or \
                (not port and _safe_gethostbyname(host) in self.get_names()):
                if host:
                    origurl = 'file://' + host + filename
                else:
                    origurl = 'file://' + filename
                return addinfourl(open(localfile, 'rb'), headers, origurl)
        except OSError as exp:
            raise URLError(exp)
        raise URLError('file not on local host')

def _safe_gethostbyname(host):
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return None

class FTPHandler(BaseHandler):
    def ftp_open(self, req):
        import ftplib
        import mimetypes
        host = req.host
        if not host:
            raise URLError('ftp error: no host given')
        host, port = _splitport(host)
        if port is None:
            port = ftplib.FTP_PORT
        else:
            port = int(port)

        # username/password handling
        user, host = _splituser(host)
        if user:
            user, passwd = _splitpasswd(user)
        else:
            passwd = None
        host = unquote(host)
        user = user or ''
        passwd = passwd or ''

        try:
            host = socket.gethostbyname(host)
        except OSError as msg:
            raise URLError(msg)
        path, attrs = _splitattr(req.selector)
        dirs = path.split('/')
        dirs = list(map(unquote, dirs))
        dirs, file = dirs[:-1], dirs[-1]
        if dirs and not dirs[0]:
            dirs = dirs[1:]
        try:
            fw = self.connect_ftp(user, passwd, host, port, dirs, req.timeout)
            type = file and 'I' or 'D'
            for attr in attrs:
                attr, value = _splitvalue(attr)
                if attr.lower() == 'type' and \
                   value in ('a', 'A', 'i', 'I', 'd', 'D'):
                    type = value.upper()
            fp, retrlen = fw.retrfile(file, type)
            headers = ""
            mtype = mimetypes.guess_type(req.full_url)[0]
            if mtype:
                headers += "Content-type: %s\n" % mtype
            if retrlen is not None and retrlen >= 0:
                headers += "Content-length: %d\n" % retrlen
            headers = email.message_from_string(headers)
            return addinfourl(fp, headers, req.full_url)
        except ftplib.all_errors as exp:
            raise URLError(exp) from exp

    def connect_ftp(self, user, passwd, host, port, dirs, timeout):
        return ftpwrapper(user, passwd, host, port, dirs, timeout,
                          persistent=False)

class CacheFTPHandler(FTPHandler):
    # XXX would be nice to have pluggable cache strategies
    # XXX this stuff is definitely not thread safe
    def __init__(self):
        self.cache = {}
        self.timeout = {}
        self.soonest = 0
        self.delay = 60
        self.max_conns = 16

    def setTimeout(self, t):
        self.delay = t

    def setMaxConns(self, m):
        self.max_conns = m

    def connect_ftp(self, user, passwd, host, port, dirs, timeout):
        key = user, host, port, '/'.join(dirs), timeout
        if key in self.cache:
            self.timeout[key] = time.time() + self.delay
        else:
            self.cache[key] = ftpwrapper(user, passwd, host, port,
                                         dirs, timeout)
            self.timeout[key] = time.time() + self.delay
        self.check_cache()
        return self.cache[key]

    def check_cache(self):
        # first check for old ones
        t = time.time()
        if self.soonest <= t:
            for k, v in list(self.timeout.items()):
                if v < t:
                    self.cache[k].close()
                    del self.cache[k]
                    del self.timeout[k]
        self.soonest = min(list(self.timeout.values()))

        # then check the size
        if len(self.cache) == self.max_conns:
            for k, v in list(self.timeout.items()):
                if v == self.soonest:
                    del self.cache[k]
                    del self.timeout[k]
                    break
            self.soonest = min(list(self.timeout.values()))

    def clear_cache(self):
        for conn in self.cache.values():
            conn.close()
        self.cache.clear()
        self.timeout.clear()

class DataHandler(BaseHandler):
    def data_open(self, req):
        # data URLs as specified in RFC 2397.
        #
        # ignores POSTed data
        #
        # syntax:
        # dataurl   := "data:" [ mediatype ] [ ";base64" ] "," data
        # mediatype := [ type "/" subtype ] *( ";" parameter )
        # data      := *urlchar
        # parameter := attribute "=" value
        url = req.full_url

        scheme, data = url.split(":",1)
        mediatype, data = data.split(",",1)

        # Disallow control characters within mediatype.
        if re.search(r"[\x00-\x1F\x7F]", mediatype):
            raise ValueError(
                "Control characters not allowed in data: mediatype")

        # even base64 encoded data URLs might be quoted so unquote in any case:
        data = unquote_to_bytes(data)
        if mediatype.endswith(";base64"):
            data = base64.decodebytes(data)
            mediatype = mediatype[:-7]

        if not mediatype:
            mediatype = "text/plain;charset=US-ASCII"

        headers = email.message_from_string("Content-type: %s\nContent-length: %d\n" %
            (mediatype, len(data)))

        return addinfourl(io.BytesIO(data), headers, url)


# Code move from the old urllib module

MAXFTPCACHE = 10        # Trim the ftp cache beyond this size

# Helper for non-unix systems
if os.name == 'nt':
    from nturl2path import url2pathname, pathname2url
else:
    def url2pathname(pathname):
        """OS-specific conversion from a relative URL of the 'file' scheme
        to a file system path; not recommended for general use."""
        return unquote(pathname)

    def pathname2url(pathname):
        """OS-specific conversion from a file system path to a relative URL
        of the 'file' scheme; not recommended for general use."""
        return quote(pathname)


ftpcache = {}


class URLopener:
    """Class to open URLs.
    This is a class rather than just a subroutine because we may need
    more than one set of global protocol-specific options.
    Note -- this is a base class for those who don't want the
    automatic handling of errors type 302 (relocated) and 401
    (authorization needed)."""

    __tempfiles = None

    version = "Python-urllib/%s" % __version__

    # Constructor
    def __init__(self, proxies=None, **x509):
        msg = "%(class)s style of invoking requests is deprecated. " \
              "Use newer urlopen functions/methods" % {'class': self.__class__.__name__}
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        if proxies is None:
            proxies = getproxies()
        assert hasattr(proxies, 'keys'), "proxies must be a mapping"
        self.proxies = proxies
        self.key_file = x509.get('key_file')
        self.cert_file = x509.get('cert_file')
        self.addheaders = [('User-Agent', self.version), ('Accept', '*/*')]
        self.__tempfiles = []
        self.__unlink = os.unlink # See cleanup()
        self.tempcache = None
        # Undocumented feature: if you assign {} to tempcache,
        # it is used to cache files retrieved with
        # self.retrieve().  This is not enabled by default
        # since it does not work for changing documents (and I
        # haven't got the logic to check expiration headers
        # yet).
        self.ftpcache = ftpcache
        # Undocumented feature: you can use a different
        # ftp cache by assigning to the .ftpcache member;
        # in case you want logically independent URL openers
        # XXX This is not threadsafe.  Bah.

    def __del__(self):
        self.close()

    def close(self):
        self.cleanup()

    def cleanup(self):
        # This code sometimes runs when the rest of this module
        # has already been deleted, so it can't use any globals
        # or import anything.
        if self.__tempfiles:
            for file in self.__tempfiles:
                try:
                    self.__unlink(file)
                except OSError:
                    pass
            del self.__tempfiles[:]
        if self.tempcache:
            self.tempcache.clear()

    def addheader(self, *args):
        """Add a header to be used by the HTTP interface only
        e.g. u.addheader('Accept', 'sound/basic')"""
        self.addheaders.append(args)

    # External interface
    def open(self, fullurl, data=None):
        """Use URLopener().open(file) instead of open(file, 'r')."""
        fullurl = unwrap(_to_bytes(fullurl))
        fullurl = quote(fullurl, safe="%/:=&?~#+!$,;'@()*[]|")
        if self.tempcache and fullurl in self.tempcache:
            filename, headers = self.tempcache[fullurl]
            fp = open(filename, 'rb')
            return addinfourl(fp, headers, fullurl)
        urltype, url = _splittype(fullurl)
        if not urltype:
            urltype = 'file'
        if urltype in self.proxies:
            proxy = self.proxies[urltype]
            urltype, proxyhost = _splittype(proxy)
            host, selector = _splithost(proxyhost)
            url = (host, fullurl) # Signal special case to open_*()
        else:
            proxy = None
        name = 'open_' + urltype
        self.type = urltype
        name = name.replace('-', '_')
        if not hasattr(self, name) or name == 'open_local_file':
            if proxy:
                return self.open_unknown_proxy(proxy, fullurl, data)
            else:
                return self.open_unknown(fullurl, data)
        try:
            if data is None:
                return getattr(self, name)(url)
            else:
                return getattr(self, name)(url, data)
        except (HTTPError, URLError):
            raise
        except OSError as msg:
            raise OSError('socket error', msg).with_traceback(sys.exc_info()[2])

    def open_unknown(self, fullurl, data=None):
        """Overridable interface to open unknown URL type."""
        type, url = _splittype(fullurl)
        raise OSError('url error', 'unknown url type', type)

    def open_unknown_proxy(self, proxy, fullurl, data=None):
        """Overridable interface to open unknown URL type."""
        type, url = _splittype(fullurl)
        raise OSError('url error', 'invalid proxy for %s' % type, proxy)

    # External interface
    def retrieve(self, url, filename=None, reporthook=None, data=None):
        """retrieve(url) returns (filename, headers) for a local object
        or (tempfilename, headers) for a remote object."""
        url = unwrap(_to_bytes(url))
        if self.tempcache and url in self.tempcache:
            return self.tempcache[url]
        type, url1 = _splittype(url)
        if filename is None and (not type or type == 'file'):
            try:
                fp = self.open_local_file(url1)
                hdrs = fp.info()
                fp.close()
                return url2pathname(_splithost(url1)[1]), hdrs
            except OSError:
                pass
        fp = self.open(url, data)
        try:
            headers = fp.info()
            if filename:
                tfp = open(filename, 'wb')
            else:
                garbage, path = _splittype(url)
                garbage, path = _splithost(path or "")
                path, garbage = _splitquery(path or "")
                path, garbage = _splitattr(path or "")
                suffix = os.path.splitext(path)[1]
                (fd, filename) = tempfile.mkstemp(suffix)
                self.__tempfiles.append(filename)
                tfp = os.fdopen(fd, 'wb')
            try:
                result = filename, headers
                if self.tempcache is not None:
                    self.tempcache[url] = result
                bs = 1024*8
                size = -1
                read = 0
                blocknum = 0
                if "content-length" in headers:
                    size = int(headers["Content-Length"])
                if reporthook:
                    reporthook(blocknum, bs, size)
                while 1:
                    block = fp.read(bs)
                    if not block:
                        break
                    read += len(block)
                    tfp.write(block)
                    blocknum += 1
                    if reporthook:
                        reporthook(blocknum, bs, size)
            finally:
                tfp.close()
        finally:
            fp.close()

        # raise exception if actual size does not match content-length header
        if size >= 0 and read < size:
            raise ContentTooShortError(
                "retrieval incomplete: got only %i out of %i bytes"
                % (read, size), result)

        return result

    # Each method named open_<type> knows how to open that type of URL

    def _open_generic_http(self, connection_factory, url, data):
        """Make an HTTP connection using connection_class.

        This is an internal method that should be called from
        open_http() or open_https().

        Arguments:
        - connection_factory should take a host name and return an
          HTTPConnection instance.
        - url is the url to retrieval or a host, relative-path pair.
        - data is payload for a POST request or None.
        """

        user_passwd = None
        proxy_passwd= None
        if isinstance(url, str):
            host, selector = _splithost(url)
            if host:
                user_passwd, host = _splituser(host)
                host = unquote(host)
            realhost = host
        else:
            host, selector = url
            # check whether the proxy contains authorization information
            proxy_passwd, host = _splituser(host)
            # now we proceed with the url we want to obtain
            urltype, rest = _splittype(selector)
            url = rest
            user_passwd = None
            if urltype.lower() != 'http':
                realhost = None
            else:
                realhost, rest = _splithost(rest)
                if realhost:
                    user_passwd, realhost = _splituser(realhost)
                if user_passwd:
                    selector = "%s://%s%s" % (urltype, realhost, rest)
                if proxy_bypass(realhost):
                    host = realhost

        if not host: raise OSError('http error', 'no host given')

        if proxy_passwd:
            proxy_passwd = unquote(proxy_passwd)
            proxy_auth = base64.b64encode(proxy_passwd.encode()).decode('ascii')
        else:
            proxy_auth = None

        if user_passwd:
            user_passwd = unquote(user_passwd)
            auth = base64.b64encode(user_passwd.encode()).decode('ascii')
        else:
            auth = None
        http_conn = connection_factory(host)
        headers = {}
        if proxy_auth:
            headers["Proxy-Authorization"] = "Basic %s" % proxy_auth
        if auth:
            headers["Authorization"] =  "Basic %s" % auth
        if realhost:
            headers["Host"] = realhost

        # Add Connection:close as we don't support persistent connections yet.
        # This helps in closing the socket and avoiding ResourceWarning

        headers["Connection"] = "close"

        for header, value in self.addheaders:
            headers[header] = value

        if data is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            http_conn.request("POST", selector, data, headers)
        else:
            http_conn.request("GET", selector, headers=headers)

        try:
            response = http_conn.getresponse()
        except http.client.BadStatusLine:
            # something went wrong with the HTTP status line
            raise URLError("http protocol error: bad status line")

        # According to RFC 2616, "2xx" code indicates that the client's
        # request was successfully received, understood, and accepted.
        if 200 <= response.status < 300:
            return addinfourl(response, response.msg, "http:" + url,
                              response.status)
        else:
            return self.http_error(
                url, response.fp,
                response.status, response.reason, response.msg, data)

    def open_http(self, url, data=None):
        """Use HTTP protocol."""
        return self._open_generic_http(http.client.HTTPConnection, url, data)

    def http_error(self, url, fp, errcode, errmsg, headers, data=None):
        """Handle http errors.

        Derived class can override this, or provide specific handlers
        named http_error_DDD where DDD is the 3-digit error code."""
        # First check if there's a specific handler for this error
        name = 'http_error_%d' % errcode
        if hasattr(self, name):
            method = getattr(self, name)
            if data is None:
                result = method(url, fp, errcode, errmsg, headers)
            else:
                result = method(url, fp, errcode, errmsg, headers, data)
            if result: return result
        return self.http_error_default(url, fp, errcode, errmsg, headers)

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        """Default error handler: close the connection and raise OSError."""
        fp.close()
        raise HTTPError(url, errcode, errmsg, headers, None)

    if _have_ssl:
        def _https_connection(self, host):
            return http.client.HTTPSConnection(host,
                                           key_file=self.key_file,
                                           cert_file=self.cert_file)

        def open_https(self, url, data=None):
            """Use HTTPS protocol."""
            return self._open_generic_http(self._https_connection, url, data)

    def open_file(self, url):
        """Use local file or FTP depending on form of URL."""
        if not isinstance(url, str):
            raise URLError('file error: proxy support for file protocol currently not implemented')
        if url[:2] == '//' and url[2:3] != '/' and url[2:12].lower() != 'localhost/':
            raise ValueError("file:// scheme is supported only on localhost")
        else:
            return self.open_local_file(url)

    def open_local_file(self, url):
        """Use local file."""
        import email.utils
        import mimetypes
        host, file = _splithost(url)
        localname = url2pathname(file)
        try:
            stats = os.stat(localname)
        except OSError as e:
            raise URLError(e.strerror, e.filename)
        size = stats.st_size
        modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
        mtype = mimetypes.guess_type(url)[0]
        headers = email.message_from_string(
            'Content-Type: %s\nContent-Length: %d\nLast-modified: %s\n' %
            (mtype or 'text/plain', size, modified))
        if not host:
            urlfile = file
            if file[:1] == '/':
                urlfile = 'file://' + file
            return addinfourl(open(localname, 'rb'), headers, urlfile)
        host, port = _splitport(host)
        if (not port
           and socket.gethostbyname(host) in ((localhost(),) + thishost())):
            urlfile = file
            if file[:1] == '/':
                urlfile = 'file://' + file
            elif file[:2] == './':
                raise ValueError("local file url may start with / or file:. Unknown url of type: %s" % url)
            return addinfourl(open(localname, 'rb'), headers, urlfile)
        raise URLError('local file error: not on local host')

    def open_ftp(self, url):
        """Use FTP protocol."""
        if not isinstance(url, str):
            raise URLError('ftp error: proxy support for ftp protocol currently not implemented')
        import mimetypes
        host, path = _splithost(url)
        if not host: raise URLError('ftp error: no host given')
        host, port = _splitport(host)
        user, host = _splituser(host)
        if user: user, passwd = _splitpasswd(user)
        else: passwd = None
        host = unquote(host)
        user = unquote(user or '')
        passwd = unquote(passwd or '')
        host = socket.gethostbyname(host)
        if not port:
            import ftplib
            port = ftplib.FTP_PORT
        else:
            port = int(port)
        path, attrs = _splitattr(path)
        path = unquote(path)
        dirs = path.split('/')
        dirs, file = dirs[:-1], dirs[-1]
        if dirs and not dirs[0]: dirs = dirs[1:]
        if dirs and not dirs[0]: dirs[0] = '/'
        key = user, host, port, '/'.join(dirs)
        # XXX thread unsafe!
        if len(self.ftpcache) > MAXFTPCACHE:
            # Prune the cache, rather arbitrarily
            for k in list(self.ftpcache):
                if k != key:
                    v = self.ftpcache[k]
                    del self.ftpcache[k]
                    v.close()
        try:
            if key not in self.ftpcache:
                self.ftpcache[key] = \
                    ftpwrapper(user, passwd, host, port, dirs)
            if not file: type = 'D'
            else: type = 'I'
            for attr in attrs:
                attr, value = _splitvalue(attr)
                if attr.lower() == 'type' and \
                   value in ('a', 'A', 'i', 'I', 'd', 'D'):
                    type = value.upper()
            (fp, retrlen) = self.ftpcache[key].retrfile(file, type)
            mtype = mimetypes.guess_type("ftp:" + url)[0]
            headers = ""
            if mtype:
                headers += "Content-Type: %s\n" % mtype
            if retrlen is not None and retrlen >= 0:
                headers += "Content-Length: %d\n" % retrlen
            headers = email.message_from_string(headers)
            return addinfourl(fp, headers, "ftp:" + url)
        except ftperrors() as exp:
            raise URLError('ftp error %r' % exp).with_traceback(sys.exc_info()[2])

    def open_data(self, url, data=None):
        """Use "data" URL."""
        if not isinstance(url, str):
            raise URLError('data error: proxy support for data protocol currently not implemented')
        # ignore POSTed data
        #
        # syntax of data URLs:
        # dataurl   := "data:" [ mediatype ] [ ";base64" ] "," data
        # mediatype := [ type "/" subtype ] *( ";" parameter )
        # data      := *urlchar
        # parameter := attribute "=" value
        try:
            [type, data] = url.split(',', 1)
        except ValueError:
            raise OSError('data error', 'bad data URL')
        if not type:
            type = 'text/plain;charset=US-ASCII'
        semi = type.rfind(';')
        if semi >= 0 and '=' not in type[semi:]:
            encoding = type[semi+1:]
            type = type[:semi]
        else:
            encoding = ''
        msg = []
        msg.append('Date: %s'%time.strftime('%a, %d %b %Y %H:%M:%S GMT',
                                            time.gmtime(time.time())))
        msg.append('Content-type: %s' % type)
        if encoding == 'base64':
            # XXX is this encoding/decoding ok?
            data = base64.decodebytes(data.encode('ascii')).decode('latin-1')
        else:
            data = unquote(data)
        msg.append('Content-Length: %d' % len(data))
        msg.append('')
        msg.append(data)
        msg = '\n'.join(msg)
        headers = email.message_from_string(msg)
        f = io.StringIO(msg)
        #f.fileno = None     # needed for addinfourl
        return addinfourl(f, headers, url)


class FancyURLopener(URLopener):
    """Derived class with handlers for errors we can handle (perhaps)."""

    def __init__(self, *args, **kwargs):
        URLopener.__init__(self, *args, **kwargs)
        self.auth_cache = {}
        self.tries = 0
        self.maxtries = 10

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        """Default error handling -- don't raise an exception."""
        return addinfourl(fp, headers, "http:" + url, errcode)

    def http_error_302(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 302 -- relocated (temporarily)."""
        self.tries += 1
        try:
            if self.maxtries and self.tries >= self.maxtries:
                if hasattr(self, "http_error_500"):
                    meth = self.http_error_500
                else:
                    meth = self.http_error_default
                return meth(url, fp, 500,
                            "Internal Server Error: Redirect Recursion",
                            headers)
            result = self.redirect_internal(url, fp, errcode, errmsg,
                                            headers, data)
            return result
        finally:
            self.tries = 0

    def redirect_internal(self, url, fp, errcode, errmsg, headers, data):
        if 'location' in headers:
            newurl = headers['location']
        elif 'uri' in headers:
            newurl = headers['uri']
        else:
            return
        fp.close()

        # In case the server sent a relative URL, join with original:
        newurl = urljoin(self.type + ":" + url, newurl)

        urlparts = urlparse(newurl)

        # For security reasons, we don't allow redirection to anything other
        # than http, https and ftp.

        # We are using newer HTTPError with older redirect_internal method
        # This older method will get deprecated in 3.3

        if urlparts.scheme not in ('http', 'https', 'ftp', ''):
            raise HTTPError(newurl, errcode,
                            errmsg +
                            " Redirection to url '%s' is not allowed." % newurl,
                            headers, fp)

        return self.open(newurl)

    def http_error_301(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 301 -- also relocated (permanently)."""
        return self.http_error_302(url, fp, errcode, errmsg, headers, data)

    def http_error_303(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 303 -- also relocated (essentially identical to 302)."""
        return self.http_error_302(url, fp, errcode, errmsg, headers, data)

    def http_error_307(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 307 -- relocated, but turn POST into error."""
        if data is None:
            return self.http_error_302(url, fp, errcode, errmsg, headers, data)
        else:
            return self.http_error_default(url, fp, errcode, errmsg, headers)

    def http_error_401(self, url, fp, errcode, errmsg, headers, data=None,
            retry=False):
        """Error 401 -- authentication required.
        This function supports Basic authentication only."""
        if 'www-authenticate' not in headers:
            URLopener.http_error_default(self, url, fp,
                                         errcode, errmsg, headers)
        stuff = headers['www-authenticate']
        match = re.match('[ \t]*([^ \t]+)[ \t]+realm="([^"]*)"', stuff)
        if not match:
            URLopener.http_error_default(self, url, fp,
                                         errcode, errmsg, headers)
        scheme, realm = match.groups()
        if scheme.lower() != 'basic':
            URLopener.http_error_default(self, url, fp,
                                         errcode, errmsg, headers)
        if not retry:
            URLopener.http_error_default(self, url, fp, errcode, errmsg,
                    headers)
        name = 'retry_' + self.type + '_basic_auth'
        if data is None:
            return getattr(self,name)(url, realm)
        else:
            return getattr(self,name)(url, realm, data)

    def http_error_407(self, url, fp, errcode, errmsg, headers, data=None,
            retry=False):
        """Error 407 -- proxy authentication required.
        This function supports Basic authentication only."""
        if 'proxy-authenticate' not in headers:
            URLopener.http_error_default(self, url, fp,
                                         errcode, errmsg, headers)
        stuff = headers['proxy-authenticate']
        match = re.match('[ \t]*([^ \t]+)[ \t]+realm="([^"]*)"', stuff)
        if not match:
            URLopener.http_error_default(self, url, fp,
                                         errcode, errmsg, headers)
        scheme, realm = match.groups()
        if scheme.lower() != 'basic':
            URLopener.http_error_default(self, url, fp,
                                         errcode, errmsg, headers)
        if not retry:
            URLopener.http_error_default(self, url, fp, errcode, errmsg,
                    headers)
        name = 'retry_proxy_' + self.type + '_basic_auth'
        if data is None:
            return getattr(self,name)(url, realm)
        else:
            return getattr(self,name)(url, realm, data)

    def retry_proxy_http_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        newurl = 'http://' + host + selector
        proxy = self.proxies['http']
        urltype, proxyhost = _splittype(proxy)
        proxyhost, proxyselector = _splithost(proxyhost)
        i = proxyhost.find('@') + 1
        proxyhost = proxyhost[i:]
        user, passwd = self.get_user_passwd(proxyhost, realm, i)
        if not (user or passwd): return None
        proxyhost = "%s:%s@%s" % (quote(user, safe=''),
                                  quote(passwd, safe=''), proxyhost)
        self.proxies['http'] = 'http://' + proxyhost + proxyselector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def retry_proxy_https_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        newurl = 'https://' + host + selector
        proxy = self.proxies['https']
        urltype, proxyhost = _splittype(proxy)
        proxyhost, proxyselector = _splithost(proxyhost)
        i = proxyhost.find('@') + 1
        proxyhost = proxyhost[i:]
        user, passwd = self.get_user_passwd(proxyhost, realm, i)
        if not (user or passwd): return None
        proxyhost = "%s:%s@%s" % (quote(user, safe=''),
                                  quote(passwd, safe=''), proxyhost)
        self.proxies['https'] = 'https://' + proxyhost + proxyselector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def retry_http_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        i = host.find('@') + 1
        host = host[i:]
        user, passwd = self.get_user_passwd(host, realm, i)
        if not (user or passwd): return None
        host = "%s:%s@%s" % (quote(user, safe=''),
                             quote(passwd, safe=''), host)
        newurl = 'http://' + host + selector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def retry_https_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        i = host.find('@') + 1
        host = host[i:]
        user, passwd = self.get_user_passwd(host, realm, i)
        if not (user or passwd): return None
        host = "%s:%s@%s" % (quote(user, safe=''),
                             quote(passwd, safe=''), host)
        newurl = 'https://' + host + selector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def get_user_passwd(self, host, realm, clear_cache=0):
        key = realm + '@' + host.lower()
        if key in self.auth_cache:
            if clear_cache:
                del self.auth_cache[key]
            else:
                return self.auth_cache[key]
        user, passwd = self.prompt_user_passwd(host, realm)
        if user or passwd: self.auth_cache[key] = (user, passwd)
        return user, passwd

    def prompt_user_passwd(self, host, realm):
        """Override this in a GUI environment!"""
        import getpass
        try:
            user = input("Enter username for %s at %s: " % (realm, host))
            passwd = getpass.getpass("Enter password for %s in %s at %s: " %
                (user, realm, host))
            return user, passwd
        except KeyboardInterrupt:
            print()
            return None, None


# Utility functions

_localhost = None
def localhost():
    """Return the IP address of the magic hostname 'localhost'."""
    global _localhost
    if _localhost is None:
        _localhost = socket.gethostbyname('localhost')
    return _localhost

_thishost = None
def thishost():
    """Return the IP addresses of the current host."""
    global _thishost
    if _thishost is None:
        try:
            _thishost = tuple(socket.gethostbyname_ex(socket.gethostname())[2])
        except socket.gaierror:
            _thishost = tuple(socket.gethostbyname_ex('localhost')[2])
    return _thishost

_ftperrors = None
def ftperrors():
    """Return the set of errors raised by the FTP class."""
    global _ftperrors
    if _ftperrors is None:
        import ftplib
        _ftperrors = ftplib.all_errors
    return _ftperrors

_noheaders = None
def noheaders():
    """Return an empty email Message object."""
    global _noheaders
    if _noheaders is None:
        _noheaders = email.message_from_string("")
    return _noheaders


# Utility classes

class ftpwrapper:
    """Class used by open_ftp() for cache of open FTP connections."""

    def __init__(self, user, passwd, host, port, dirs, timeout=None,
                 persistent=True):
        self.user = user
        self.passwd = passwd
        self.host = host
        self.port = port
        self.dirs = dirs
        self.timeout = timeout
        self.refcount = 0
        self.keepalive = persistent
        try:
            self.init()
        except:
            self.close()
            raise

    def init(self):
        import ftplib
        self.busy = 0
        self.ftp = ftplib.FTP()
        self.ftp.connect(self.host, self.port, self.timeout)
        self.ftp.login(self.user, self.passwd)
        _target = '/'.join(self.dirs)
        self.ftp.cwd(_target)

    def retrfile(self, file, type):
        import ftplib
        self.endtransfer()
        if type in ('d', 'D'): cmd = 'TYPE A'; isdir = 1
        else: cmd = 'TYPE ' + type; isdir = 0
        try:
            self.ftp.voidcmd(cmd)
        except ftplib.all_errors:
            self.init()
            self.ftp.voidcmd(cmd)
        conn = None
        if file and not isdir:
            # Try to retrieve as a file
            try:
                cmd = 'RETR ' + file
                conn, retrlen = self.ftp.ntransfercmd(cmd)
            except ftplib.error_perm as reason:
                if str(reason)[:3] != '550':
                    raise URLError('ftp error: %r' % reason).with_traceback(
                        sys.exc_info()[2])
        if not conn:
            # Set transfer mode to ASCII!
            self.ftp.voidcmd('TYPE A')
            # Try a directory listing. Verify that directory exists.
            if file:
                pwd = self.ftp.pwd()
                try:
                    try:
                        self.ftp.cwd(file)
                    except ftplib.error_perm as reason:
                        raise URLError('ftp error: %r' % reason) from reason
                finally:
                    self.ftp.cwd(pwd)
                cmd = 'LIST ' + file
            else:
                cmd = 'LIST'
            conn, retrlen = self.ftp.ntransfercmd(cmd)
        self.busy = 1

        ftpobj = addclosehook(conn.makefile('rb'), self.file_close)
        self.refcount += 1
        conn.close()
        # Pass back both a suitably decorated object and a retrieval length
        return (ftpobj, retrlen)

    def endtransfer(self):
        self.busy = 0

    def close(self):
        self.keepalive = False
        if self.refcount <= 0:
            self.real_close()

    def file_close(self):
        self.endtransfer()
        self.refcount -= 1
        if self.refcount <= 0 and not self.keepalive:
            self.real_close()

    def real_close(self):
        self.endtransfer()
        try:
            self.ftp.close()
        except ftperrors():
            pass

# Proxy handling
def getproxies_environment():
    """Return a dictionary of scheme -> proxy server URL mappings.

    Scan the environment for variables named <scheme>_proxy;
    this seems to be the standard convention.  If you need a
    different way, you can pass a proxies dictionary to the
    [Fancy]URLopener constructor.
    """
    # in order to prefer lowercase variables, process environment in
    # two passes: first matches any, second pass matches lowercase only

    # select only environment variables which end in (after making lowercase) _proxy
    proxies = {}
    environment = []
    for name in os.environ.keys():
        # fast screen underscore position before more expensive case-folding
        if len(name) > 5 and name[-6] == "_" and name[-5:].lower() == "proxy":
            value = os.environ[name]
            proxy_name = name[:-6].lower()
            environment.append((name, value, proxy_name))
            if value:
                proxies[proxy_name] = value
    # CVE-2016-1000110 - If we are running as CGI script, forget HTTP_PROXY
    # (non-all-lowercase) as it may be set from the web server by a "Proxy:"
    # header from the client
    # If "proxy" is lowercase, it will still be used thanks to the next block
    if 'REQUEST_METHOD' in os.environ:
        proxies.pop('http', None)
    for name, value, proxy_name in environment:
        # not case-folded, checking here for lower-case env vars only
        if name[-6:] == '_proxy':
            if value:
                proxies[proxy_name] = value
            else:
                proxies.pop(proxy_name, None)
    return proxies

def proxy_bypass_environment(host, proxies=None):
    """Test if proxies should not be used for a particular host.

    Checks the proxy dict for the value of no_proxy, which should
    be a list of comma separated DNS suffixes, or '*' for all hosts.

    """
    if proxies is None:
        proxies = getproxies_environment()
    # don't bypass, if no_proxy isn't specified
    try:
        no_proxy = proxies['no']
    except KeyError:
        return False
    # '*' is special case for always bypass
    if no_proxy == '*':
        return True
    host = host.lower()
    # strip port off host
    hostonly, port = _splitport(host)
    # check if the host ends with any of the DNS suffixes
    for name in no_proxy.split(','):
        name = name.strip()
        if name:
            name = name.lstrip('.')  # ignore leading dots
            name = name.lower()
            if hostonly == name or host == name:
                return True
            name = '.' + name
            if hostonly.endswith(name) or host.endswith(name):
                return True
    # otherwise, don't bypass
    return False


# This code tests an OSX specific data structure but is testable on all
# platforms
def _proxy_bypass_macosx_sysconf(host, proxy_settings):
    """
    Return True iff this host shouldn't be accessed using a proxy

    This function uses the MacOSX framework SystemConfiguration
    to fetch the proxy information.

    proxy_settings come from _scproxy._get_proxy_settings or get mocked ie:
    { 'exclude_simple': bool,
      'exceptions': ['foo.bar', '*.bar.com', '127.0.0.1', '10.1', '10.0/16']
    }
    """
    from fnmatch import fnmatch

    hostonly, port = _splitport(host)

    def ip2num(ipAddr):
        parts = ipAddr.split('.')
        parts = list(map(int, parts))
        if len(parts) != 4:
            parts = (parts + [0, 0, 0, 0])[:4]
        return (parts[0] << 24) | (parts[1] << 16) | (parts[2] << 8) | parts[3]

    # Check for simple host names:
    if '.' not in host:
        if proxy_settings['exclude_simple']:
            return True

    hostIP = None

    for value in proxy_settings.get('exceptions', ()):
        # Items in the list are strings like these: *.local, 169.254/16
        if not value: continue

        m = re.match(r"(\d+(?:\.\d+)*)(/\d+)?", value)
        if m is not None:
            if hostIP is None:
                try:
                    hostIP = socket.gethostbyname(hostonly)
                    hostIP = ip2num(hostIP)
                except OSError:
                    continue

            base = ip2num(m.group(1))
            mask = m.group(2)
            if mask is None:
                mask = 8 * (m.group(1).count('.') + 1)
            else:
                mask = int(mask[1:])

            if mask < 0 or mask > 32:
                # System libraries ignore invalid prefix lengths
                continue

            mask = 32 - mask

            if (hostIP >> mask) == (base >> mask):
                return True

        elif fnmatch(host, value):
            return True

    return False


if sys.platform == 'darwin':
    from _scproxy import _get_proxy_settings, _get_proxies

    def proxy_bypass_macosx_sysconf(host):
        proxy_settings = _get_proxy_settings()
        return _proxy_bypass_macosx_sysconf(host, proxy_settings)

    def getproxies_macosx_sysconf():
        """Return a dictionary of scheme -> proxy server URL mappings.

        This function uses the MacOSX framework SystemConfiguration
        to fetch the proxy information.
        """
        return _get_proxies()



    def proxy_bypass(host):
        """Return True, if host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or from the MacOSX framework SystemConfiguration.

        """
        proxies = getproxies_environment()
        if proxies:
            return proxy_bypass_environment(host, proxies)
        else:
            return proxy_bypass_macosx_sysconf(host)

    def getproxies():
        return getproxies_environment() or getproxies_macosx_sysconf()


elif os.name == 'nt':
    def getproxies_registry():
        """Return a dictionary of scheme -> proxy server URL mappings.

        Win32 uses the registry to store proxies.

        """
        proxies = {}
        try:
            import winreg
        except ImportError:
            # Std module, so should be around - but you never know!
            return proxies
        try:
            internetSettings = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r'Software\Microsoft\Windows\CurrentVersion\Internet Settings')
            proxyEnable = winreg.QueryValueEx(internetSettings,
                                               'ProxyEnable')[0]
            if proxyEnable:
                # Returned as Unicode but problems if not converted to ASCII
                proxyServer = str(winreg.QueryValueEx(internetSettings,
                                                       'ProxyServer')[0])
                if '=' not in proxyServer and ';' not in proxyServer:
                    # Use one setting for all protocols.
                    proxyServer = 'http={0};https={0};ftp={0}'.format(proxyServer)
                for p in proxyServer.split(';'):
                    protocol, address = p.split('=', 1)
                    # See if address has a type:// prefix
                    if not re.match('(?:[^/:]+)://', address):
                        # Add type:// prefix to address without specifying type
                        if protocol in ('http', 'https', 'ftp'):
                            # The default proxy type of Windows is HTTP
                            address = 'http://' + address
                        elif protocol == 'socks':
                            address = 'socks://' + address
                    proxies[protocol] = address
                # Use SOCKS proxy for HTTP(S) protocols
                if proxies.get('socks'):
                    # The default SOCKS proxy type of Windows is SOCKS4
                    address = re.sub(r'^socks://', 'socks4://', proxies['socks'])
                    proxies['http'] = proxies.get('http') or address
                    proxies['https'] = proxies.get('https') or address
            internetSettings.Close()
        except (OSError, ValueError, TypeError):
            # Either registry key not found etc, or the value in an
            # unexpected format.
            # proxies already set up to be empty so nothing to do
            pass
        return proxies

    def getproxies():
        """Return a dictionary of scheme -> proxy server URL mappings.

        Returns settings gathered from the environment, if specified,
        or the registry.

        """
        return getproxies_environment() or getproxies_registry()

    def proxy_bypass_registry(host):
        try:
            import winreg
        except ImportError:
            # Std modules, so should be around - but you never know!
            return 0
        try:
            internetSettings = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r'Software\Microsoft\Windows\CurrentVersion\Internet Settings')
            proxyEnable = winreg.QueryValueEx(internetSettings,
                                               'ProxyEnable')[0]
            proxyOverride = str(winreg.QueryValueEx(internetSettings,
                                                     'ProxyOverride')[0])
            # ^^^^ Returned as Unicode but problems if not converted to ASCII
        except OSError:
            return 0
        if not proxyEnable or not proxyOverride:
            return 0
        # try to make a host list from name and IP address.
        rawHost, port = _splitport(host)
        host = [rawHost]
        try:
            addr = socket.gethostbyname(rawHost)
            if addr != rawHost:
                host.append(addr)
        except OSError:
            pass
        try:
            fqdn = socket.getfqdn(rawHost)
            if fqdn != rawHost:
                host.append(fqdn)
        except OSError:
            pass
        # make a check value list from the registry entry: replace the
        # '<local>' string by the localhost entry and the corresponding
        # canonical entry.
        proxyOverride = proxyOverride.split(';')
        # now check if we match one of the registry values.
        for test in proxyOverride:
            if test == '<local>':
                if '.' not in rawHost:
                    return 1
            test = test.replace(".", r"\.")     # mask dots
            test = test.replace("*", r".*")     # change glob sequence
            test = test.replace("?", r".")      # change glob char
            for val in host:
                if re.match(test, val, re.I):
                    return 1
        return 0

    def proxy_bypass(host):
        """Return True, if host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or the registry.

        """
        proxies = getproxies_environment()
        if proxies:
            return proxy_bypass_environment(host, proxies)
        else:
            return proxy_bypass_registry(host)

else:
    # By default use environment variables
    getproxies = getproxies_environment
    proxy_bypass = proxy_bypass_environment
<|endoftext|>#
# XML-RPC CLIENT LIBRARY
# $Id$
#
# an XML-RPC client interface for Python.
#
# the marshalling and response parser code can also be used to
# implement XML-RPC servers.
#
# Notes:
# this version is designed to work with Python 2.1 or newer.
#
# History:
# 1999-01-14 fl  Created
# 1999-01-15 fl  Changed dateTime to use localtime
# 1999-01-16 fl  Added Binary/base64 element, default to RPC2 service
# 1999-01-19 fl  Fixed array data element (from Skip Montanaro)
# 1999-01-21 fl  Fixed dateTime constructor, etc.
# 1999-02-02 fl  Added fault handling, handle empty sequences, etc.
# 1999-02-10 fl  Fixed problem with empty responses (from Skip Montanaro)
# 1999-06-20 fl  Speed improvements, pluggable parsers/transports (0.9.8)
# 2000-11-28 fl  Changed boolean to check the truth value of its argument
# 2001-02-24 fl  Added encoding/Unicode/SafeTransport patches
# 2001-02-26 fl  Added compare support to wrappers (0.9.9/1.0b1)
# 2001-03-28 fl  Make sure response tuple is a singleton
# 2001-03-29 fl  Don't require empty params element (from Nicholas Riley)
# 2001-06-10 fl  Folded in _xmlrpclib accelerator support (1.0b2)
# 2001-08-20 fl  Base xmlrpclib.Error on built-in Exception (from Paul Prescod)
# 2001-09-03 fl  Allow Transport subclass to override getparser
# 2001-09-10 fl  Lazy import of urllib, cgi, xmllib (20x import speedup)
# 2001-10-01 fl  Remove containers from memo cache when done with them
# 2001-10-01 fl  Use faster escape method (80% dumps speedup)
# 2001-10-02 fl  More dumps microtuning
# 2001-10-04 fl  Make sure import expat gets a parser (from Guido van Rossum)
# 2001-10-10 sm  Allow long ints to be passed as ints if they don't overflow
# 2001-10-17 sm  Test for int and long overflow (allows use on 64-bit systems)
# 2001-11-12 fl  Use repr() to marshal doubles (from Paul Felix)
# 2002-03-17 fl  Avoid buffered read when possible (from James Rucker)
# 2002-04-07 fl  Added pythondoc comments
# 2002-04-16 fl  Added __str__ methods to datetime/binary wrappers
# 2002-05-15 fl  Added error constants (from Andrew Kuchling)
# 2002-06-27 fl  Merged with Python CVS version
# 2002-10-22 fl  Added basic authentication (based on code from Phillip Eby)
# 2003-01-22 sm  Add support for the bool type
# 2003-02-27 gvr Remove apply calls
# 2003-04-24 sm  Use cStringIO if available
# 2003-04-25 ak  Add support for nil
# 2003-06-15 gn  Add support for time.struct_time
# 2003-07-12 gp  Correct marshalling of Faults
# 2003-10-31 mvl Add multicall support
# 2004-08-20 mvl Bump minimum supported Python version to 2.1
# 2014-12-02 ch/doko  Add workaround for gzip bomb vulnerability
#
# Copyright (c) 1999-2002 by Secret Labs AB.
# Copyright (c) 1999-2002 by Fredrik Lundh.
#
# info@pythonware.com
# http://www.pythonware.com
#
# --------------------------------------------------------------------
# The XML-RPC client interface is
#
# Copyright (c) 1999-2002 by Secret Labs AB
# Copyright (c) 1999-2002 by Fredrik Lundh
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

"""
An XML-RPC client interface for Python.

The marshalling and response parser code can also be used to
implement XML-RPC servers.

Exported exceptions:

  Error          Base class for client errors
  ProtocolError  Indicates an HTTP protocol error
  ResponseError  Indicates a broken response package
  Fault          Indicates an XML-RPC fault package

Exported classes:

  ServerProxy    Represents a logical connection to an XML-RPC server

  MultiCall      Executor of boxcared xmlrpc requests
  DateTime       dateTime wrapper for an ISO 8601 string or time tuple or
                 localtime integer value to generate a "dateTime.iso8601"
                 XML-RPC value
  Binary         binary data wrapper

  Marshaller     Generate an XML-RPC params chunk from a Python data structure
  Unmarshaller   Unmarshal an XML-RPC response from incoming XML event message
  Transport      Handles an HTTP transaction to an XML-RPC server
  SafeTransport  Handles an HTTPS transaction to an XML-RPC server

Exported constants:

  (none)

Exported functions:

  getparser      Create instance of the fastest available parser & attach
                 to an unmarshalling object
  dumps          Convert an argument tuple or a Fault instance to an XML-RPC
                 request (or response, if the methodresponse option is used).
  loads          Convert an XML-RPC packet to unmarshalled data plus a method
                 name (None if not present).
"""

import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
try:
    import gzip
except ImportError:
    gzip = None #python can be built without zlib/gzip support

# --------------------------------------------------------------------
# Internal stuff

def escape(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    return s.replace(">", "&gt;",)

# used in User-Agent header sent
__version__ = '%d.%d' % sys.version_info[:2]

# xmlrpc integer limits
MAXINT =  2**31-1
MININT = -2**31

# --------------------------------------------------------------------
# Error constants (from Dan Libby's specification at
# http://xmlrpc-epi.sourceforge.net/specs/rfc.fault_codes.php)

# Ranges of errors
PARSE_ERROR       = -32700
SERVER_ERROR      = -32600
APPLICATION_ERROR = -32500
SYSTEM_ERROR      = -32400
TRANSPORT_ERROR   = -32300

# Specific errors
NOT_WELLFORMED_ERROR  = -32700
UNSUPPORTED_ENCODING  = -32701
INVALID_ENCODING_CHAR = -32702
INVALID_XMLRPC        = -32600
METHOD_NOT_FOUND      = -32601
INVALID_METHOD_PARAMS = -32602
INTERNAL_ERROR        = -32603

# --------------------------------------------------------------------
# Exceptions

##
# Base class for all kinds of client-side errors.

class Error(Exception):
    """Base class for client errors."""
    __str__ = object.__str__

##
# Indicates an HTTP-level protocol error.  This is raised by the HTTP
# transport layer, if the server returns an error code other than 200
# (OK).
#
# @param url The target URL.
# @param errcode The HTTP error code.
# @param errmsg The HTTP error message.
# @param headers The HTTP header dictionary.

class ProtocolError(Error):
    """Indicates an HTTP protocol error."""
    def __init__(self, url, errcode, errmsg, headers):
        Error.__init__(self)
        self.url = url
        self.errcode = errcode
        self.errmsg = errmsg
        self.headers = headers
    def __repr__(self):
        return (
            "<%s for %s: %s %s>" %
            (self.__class__.__name__, self.url, self.errcode, self.errmsg)
            )

##
# Indicates a broken XML-RPC response package.  This exception is
# raised by the unmarshalling layer, if the XML-RPC response is
# malformed.

class ResponseError(Error):
    """Indicates a broken response package."""
    pass

##
# Indicates an XML-RPC fault response package.  This exception is
# raised by the unmarshalling layer, if the XML-RPC response contains
# a fault string.  This exception can also be used as a class, to
# generate a fault XML-RPC message.
#
# @param faultCode The XML-RPC fault code.
# @param faultString The XML-RPC fault string.

class Fault(Error):
    """Indicates an XML-RPC fault package."""
    def __init__(self, faultCode, faultString, **extra):
        Error.__init__(self)
        self.faultCode = faultCode
        self.faultString = faultString
    def __repr__(self):
        return "<%s %s: %r>" % (self.__class__.__name__,
                                self.faultCode, self.faultString)

# --------------------------------------------------------------------
# Special values

##
# Backwards compatibility

boolean = Boolean = bool

##
# Wrapper for XML-RPC DateTime values.  This converts a time value to
# the format used by XML-RPC.
# <p>
# The value can be given as a datetime object, as a string in the
# format "yyyymmddThh:mm:ss", as a 9-item time tuple (as returned by
# time.localtime()), or an integer value (as returned by time.time()).
# The wrapper uses time.localtime() to convert an integer to a time
# tuple.
#
# @param value The time, given as a datetime object, an ISO 8601 string,
#              a time tuple, or an integer time value.


# Issue #13305: different format codes across platforms
_day0 = datetime(1, 1, 1)
def _try(fmt):
    try:
        return _day0.strftime(fmt) == '0001'
    except ValueError:
        return False
if _try('%Y'):      # Mac OS X
    def _iso8601_format(value):
        return value.strftime("%Y%m%dT%H:%M:%S")
elif _try('%4Y'):   # Linux
    def _iso8601_format(value):
        return value.strftime("%4Y%m%dT%H:%M:%S")
else:
    def _iso8601_format(value):
        return value.strftime("%Y%m%dT%H:%M:%S").zfill(17)
del _day0
del _try


def _strftime(value):
    if isinstance(value, datetime):
        return _iso8601_format(value)

    if not isinstance(value, (tuple, time.struct_time)):
        if value == 0:
            value = time.time()
        value = time.localtime(value)

    return "%04d%02d%02dT%02d:%02d:%02d" % value[:6]

class DateTime:
    """DateTime wrapper for an ISO 8601 string or time tuple or
    localtime integer value to generate 'dateTime.iso8601' XML-RPC
    value.
    """

    def __init__(self, value=0):
        if isinstance(value, str):
            self.value = value
        else:
            self.value = _strftime(value)

    def make_comparable(self, other):
        if isinstance(other, DateTime):
            s = self.value
            o = other.value
        elif isinstance(other, datetime):
            s = self.value
            o = _iso8601_format(other)
        elif isinstance(other, str):
            s = self.value
            o = other
        elif hasattr(other, "timetuple"):
            s = self.timetuple()
            o = other.timetuple()
        else:
            s = self
            o = NotImplemented
        return s, o

    def __lt__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s < o

    def __le__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s <= o

    def __gt__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s > o

    def __ge__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s >= o

    def __eq__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s == o

    def timetuple(self):
        return time.strptime(self.value, "%Y%m%dT%H:%M:%S")

    ##
    # Get date/time value.
    #
    # @return Date/time value, as an ISO 8601 string.

    def __str__(self):
        return self.value

    def __repr__(self):
        return "<%s %r at %#x>" % (self.__class__.__name__, self.value, id(self))

    def decode(self, data):
        self.value = str(data).strip()

    def encode(self, out):
        out.write("<value><dateTime.iso8601>")
        out.write(self.value)
        out.write("</dateTime.iso8601></value>\n")

def _datetime(data):
    # decode xml element contents into a DateTime structure.
    value = DateTime()
    value.decode(data)
    return value

def _datetime_type(data):
    return datetime.strptime(data, "%Y%m%dT%H:%M:%S")

##
# Wrapper for binary data.  This can be used to transport any kind
# of binary data over XML-RPC, using BASE64 encoding.
#
# @param data An 8-bit string containing arbitrary data.

class Binary:
    """Wrapper for binary data."""

    def __init__(self, data=None):
        if data is None:
            data = b""
        else:
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("expected bytes or bytearray, not %s" %
                                data.__class__.__name__)
            data = bytes(data)  # Make a copy of the bytes!
        self.data = data

    ##
    # Get buffer contents.
    #
    # @return Buffer contents, as an 8-bit string.

    def __str__(self):
        return str(self.data, "latin-1")  # XXX encoding?!

    def __eq__(self, other):
        if isinstance(other, Binary):
            other = other.data
        return self.data == other

    def decode(self, data):
        self.data = base64.decodebytes(data)

    def encode(self, out):
        out.write("<value><base64>\n")
        encoded = base64.encodebytes(self.data)
        out.write(encoded.decode('ascii'))
        out.write("</base64></value>\n")

def _binary(data):
    # decode xml element contents into a Binary structure
    value = Binary()
    value.decode(data)
    return value

WRAPPERS = (DateTime, Binary)

# --------------------------------------------------------------------
# XML parsers

class ExpatParser:
    # fast expat parser for Python 2.0 and later.
    def __init__(self, target):
        self._parser = parser = expat.ParserCreate(None, None)
        self._target = target
        parser.StartElementHandler = target.start
        parser.EndElementHandler = target.end
        parser.CharacterDataHandler = target.data
        encoding = None
        target.xml(encoding, None)

    def feed(self, data):
        self._parser.Parse(data, False)

    def close(self):
        try:
            parser = self._parser
        except AttributeError:
            pass
        else:
            del self._target, self._parser # get rid of circular references
            parser.Parse(b"", True) # end of data

# --------------------------------------------------------------------
# XML-RPC marshalling and unmarshalling code

##
# XML-RPC marshaller.
#
# @param encoding Default encoding for 8-bit strings.  The default
#     value is None (interpreted as UTF-8).
# @see dumps

class Marshaller:
    """Generate an XML-RPC params chunk from a Python data structure.

    Create a Marshaller instance for each set of parameters, and use
    the "dumps" method to convert your data (represented as a tuple)
    to an XML-RPC params chunk.  To write a fault response, pass a
    Fault instance instead.  You may prefer to use the "dumps" module
    function for this purpose.
    """

    # by the way, if you don't understand what's going on in here,
    # that's perfectly ok.

    def __init__(self, encoding=None, allow_none=False):
        self.memo = {}
        self.data = None
        self.encoding = encoding
        self.allow_none = allow_none

    dispatch = {}

    def dumps(self, values):
        out = []
        write = out.append
        dump = self.__dump
        if isinstance(values, Fault):
            # fault instance
            write("<fault>\n")
            dump({'faultCode': values.faultCode,
                  'faultString': values.faultString},
                 write)
            write("</fault>\n")
        else:
            # parameter block
            # FIXME: the xml-rpc specification allows us to leave out
            # the entire <params> block if there are no parameters.
            # however, changing this may break older code (including
            # old versions of xmlrpclib.py), so this is better left as
            # is for now.  See @XMLRPC3 for more information. /F
            write("<params>\n")
            for v in values:
                write("<param>\n")
                dump(v, write)
                write("</param>\n")
            write("</params>\n")
        result = "".join(out)
        return result

    def __dump(self, value, write):
        try:
            f = self.dispatch[type(value)]
        except KeyError:
            # check if this object can be marshalled as a structure
            if not hasattr(value, '__dict__'):
                raise TypeError("cannot marshal %s objects" % type(value))
            # check if this class is a sub-class of a basic type,
            # because we don't know how to marshal these types
            # (e.g. a string sub-class)
            for type_ in type(value).__mro__:
                if type_ in self.dispatch.keys():
                    raise TypeError("cannot marshal %s objects" % type(value))
            # XXX(twouters): using "_arbitrary_instance" as key as a quick-fix
            # for the p3yk merge, this should probably be fixed more neatly.
            f = self.dispatch["_arbitrary_instance"]
        f(self, value, write)

    def dump_nil (self, value, write):
        if not self.allow_none:
            raise TypeError("cannot marshal None unless allow_none is enabled")
        write("<value><nil/></value>")
    dispatch[type(None)] = dump_nil

    def dump_bool(self, value, write):
        write("<value><boolean>")
        write(value and "1" or "0")
        write("</boolean></value>\n")
    dispatch[bool] = dump_bool

    def dump_long(self, value, write):
        if value > MAXINT or value < MININT:
            raise OverflowError("int exceeds XML-RPC limits")
        write("<value><int>")
        write(str(int(value)))
        write("</int></value>\n")
    dispatch[int] = dump_long

    # backward compatible
    dump_int = dump_long

    def dump_double(self, value, write):
        write("<value><double>")
        write(repr(value))
        write("</double></value>\n")
    dispatch[float] = dump_double

    def dump_unicode(self, value, write, escape=escape):
        write("<value><string>")
        write(escape(value))
        write("</string></value>\n")
    dispatch[str] = dump_unicode

    def dump_bytes(self, value, write):
        write("<value><base64>\n")
        encoded = base64.encodebytes(value)
        write(encoded.decode('ascii'))
        write("</base64></value>\n")
    dispatch[bytes] = dump_bytes
    dispatch[bytearray] = dump_bytes

    def dump_array(self, value, write):
        i = id(value)
        if i in self.memo:
            raise TypeError("cannot marshal recursive sequences")
        self.memo[i] = None
        dump = self.__dump
        write("<value><array><data>\n")
        for v in value:
            dump(v, write)
        write("</data></array></value>\n")
        del self.memo[i]
    dispatch[tuple] = dump_array
    dispatch[list] = dump_array

    def dump_struct(self, value, write, escape=escape):
        i = id(value)
        if i in self.memo:
            raise TypeError("cannot marshal recursive dictionaries")
        self.memo[i] = None
        dump = self.__dump
        write("<value><struct>\n")
        for k, v in value.items():
            write("<member>\n")
            if not isinstance(k, str):
                raise TypeError("dictionary key must be string")
            write("<name>%s</name>\n" % escape(k))
            dump(v, write)
            write("</member>\n")
        write("</struct></value>\n")
        del self.memo[i]
    dispatch[dict] = dump_struct

    def dump_datetime(self, value, write):
        write("<value><dateTime.iso8601>")
        write(_strftime(value))
        write("</dateTime.iso8601></value>\n")
    dispatch[datetime] = dump_datetime

    def dump_instance(self, value, write):
        # check for special wrappers
        if value.__class__ in WRAPPERS:
            self.write = write
            value.encode(self)
            del self.write
        else:
            # store instance attributes as a struct (really?)
            self.dump_struct(value.__dict__, write)
    dispatch[DateTime] = dump_instance
    dispatch[Binary] = dump_instance
    # XXX(twouters): using "_arbitrary_instance" as key as a quick-fix
    # for the p3yk merge, this should probably be fixed more neatly.
    dispatch["_arbitrary_instance"] = dump_instance

##
# XML-RPC unmarshaller.
#
# @see loads

class Unmarshaller:
    """Unmarshal an XML-RPC response, based on incoming XML event
    messages (start, data, end).  Call close() to get the resulting
    data structure.

    Note that this reader is fairly tolerant, and gladly accepts bogus
    XML-RPC data without complaining (but not bogus XML).
    """

    # and again, if you don't understand what's going on in here,
    # that's perfectly ok.

    def __init__(self, use_datetime=False, use_builtin_types=False):
        self._type = None
        self._stack = []
        self._marks = []
        self._data = []
        self._value = False
        self._methodname = None
        self._encoding = "utf-8"
        self.append = self._stack.append
        self._use_datetime = use_builtin_types or use_datetime
        self._use_bytes = use_builtin_types

    def close(self):
        # return response tuple and target method
        if self._type is None or self._marks:
            raise ResponseError()
        if self._type == "fault":
            raise Fault(**self._stack[0])
        return tuple(self._stack)

    def getmethodname(self):
        return self._methodname

    #
    # event handlers

    def xml(self, encoding, standalone):
        self._encoding = encoding
        # FIXME: assert standalone == 1 ???

    def start(self, tag, attrs):
        # prepare to handle this element
        if ':' in tag:
            tag = tag.split(':')[-1]
        if tag == "array" or tag == "struct":
            self._marks.append(len(self._stack))
        self._data = []
        if self._value and tag not in self.dispatch:
            raise ResponseError("unknown tag %r" % tag)
        self._value = (tag == "value")

    def data(self, text):
        self._data.append(text)

    def end(self, tag):
        # call the appropriate end tag handler
        try:
            f = self.dispatch[tag]
        except KeyError:
            if ':' not in tag:
                return # unknown tag ?
            try:
                f = self.dispatch[tag.split(':')[-1]]
            except KeyError:
                return # unknown tag ?
        return f(self, "".join(self._data))

    #
    # accelerator support

    def end_dispatch(self, tag, data):
        # dispatch data
        try:
            f = self.dispatch[tag]
        except KeyError:
            if ':' not in tag:
                return # unknown tag ?
            try:
                f = self.dispatch[tag.split(':')[-1]]
            except KeyError:
                return # unknown tag ?
        return f(self, data)

    #
    # element decoders

    dispatch = {}

    def end_nil (self, data):
        self.append(None)
        self._value = 0
    dispatch["nil"] = end_nil

    def end_boolean(self, data):
        if data == "0":
            self.append(False)
        elif data == "1":
            self.append(True)
        else:
            raise TypeError("bad boolean value")
        self._value = 0
    dispatch["boolean"] = end_boolean

    def end_int(self, data):
        self.append(int(data))
        self._value = 0
    dispatch["i1"] = end_int
    dispatch["i2"] = end_int
    dispatch["i4"] = end_int
    dispatch["i8"] = end_int
    dispatch["int"] = end_int
    dispatch["biginteger"] = end_int

    def end_double(self, data):
        self.append(float(data))
        self._value = 0
    dispatch["double"] = end_double
    dispatch["float"] = end_double

    def end_bigdecimal(self, data):
        self.append(Decimal(data))
        self._value = 0
    dispatch["bigdecimal"] = end_bigdecimal

    def end_string(self, data):
        if self._encoding:
            data = data.decode(self._encoding)
        self.append(data)
        self._value = 0
    dispatch["string"] = end_string
    dispatch["name"] = end_string # struct keys are always strings

    def end_array(self, data):
        mark = self._marks.pop()
        # map arrays to Python lists
        self._stack[mark:] = [self._stack[mark:]]
        self._value = 0
    dispatch["array"] = end_array

    def end_struct(self, data):
        mark = self._marks.pop()
        # map structs to Python dictionaries
        dict = {}
        items = self._stack[mark:]
        for i in range(0, len(items), 2):
            dict[items[i]] = items[i+1]
        self._stack[mark:] = [dict]
        self._value = 0
    dispatch["struct"] = end_struct

    def end_base64(self, data):
        value = Binary()
        value.decode(data.encode("ascii"))
        if self._use_bytes:
            value = value.data
        self.append(value)
        self._value = 0
    dispatch["base64"] = end_base64

    def end_dateTime(self, data):
        value = DateTime()
        value.decode(data)
        if self._use_datetime:
            value = _datetime_type(data)
        self.append(value)
    dispatch["dateTime.iso8601"] = end_dateTime

    def end_value(self, data):
        # if we stumble upon a value element with no internal
        # elements, treat it as a string element
        if self._value:
            self.end_string(data)
    dispatch["value"] = end_value

    def end_params(self, data):
        self._type = "params"
    dispatch["params"] = end_params

    def end_fault(self, data):
        self._type = "fault"
    dispatch["fault"] = end_fault

    def end_methodName(self, data):
        if self._encoding:
            data = data.decode(self._encoding)
        self._methodname = data
        self._type = "methodName" # no params
    dispatch["methodName"] = end_methodName

## Multicall support
#

class _MultiCallMethod:
    # some lesser magic to store calls made to a MultiCall object
    # for batch execution
    def __init__(self, call_list, name):
        self.__call_list = call_list
        self.__name = name
    def __getattr__(self, name):
        return _MultiCallMethod(self.__call_list, "%s.%s" % (self.__name, name))
    def __call__(self, *args):
        self.__call_list.append((self.__name, args))

class MultiCallIterator:
    """Iterates over the results of a multicall. Exceptions are
    raised in response to xmlrpc faults."""

    def __init__(self, results):
        self.results = results

    def __getitem__(self, i):
        item = self.results[i]
        if type(item) == type({}):
            raise Fault(item['faultCode'], item['faultString'])
        elif type(item) == type([]):
            return item[0]
        else:
            raise ValueError("unexpected type in multicall result")

class MultiCall:
    """server -> an object used to boxcar method calls

    server should be a ServerProxy object.

    Methods can be added to the MultiCall using normal
    method call syntax e.g.:

    multicall = MultiCall(server_proxy)
    multicall.add(2,3)
    multicall.get_address("Guido")

    To execute the multicall, call the MultiCall object e.g.:

    add_result, address = multicall()
    """

    def __init__(self, server):
        self.__server = server
        self.__call_list = []

    def __repr__(self):
        return "<%s at %#x>" % (self.__class__.__name__, id(self))

    def __getattr__(self, name):
        return _MultiCallMethod(self.__call_list, name)

    def __call__(self):
        marshalled_list = []
        for name, args in self.__call_list:
            marshalled_list.append({'methodName' : name, 'params' : args})

        return MultiCallIterator(self.__server.system.multicall(marshalled_list))

# --------------------------------------------------------------------
# convenience functions

FastMarshaller = FastParser = FastUnmarshaller = None

##
# Create a parser object, and connect it to an unmarshalling instance.
# This function picks the fastest available XML parser.
#
# return A (parser, unmarshaller) tuple.

def getparser(use_datetime=False, use_builtin_types=False):
    """getparser() -> parser, unmarshaller

    Create an instance of the fastest available parser, and attach it
    to an unmarshalling object.  Return both objects.
    """
    if FastParser and FastUnmarshaller:
        if use_builtin_types:
            mkdatetime = _datetime_type
            mkbytes = base64.decodebytes
        elif use_datetime:
            mkdatetime = _datetime_type
            mkbytes = _binary
        else:
            mkdatetime = _datetime
            mkbytes = _binary
        target = FastUnmarshaller(True, False, mkbytes, mkdatetime, Fault)
        parser = FastParser(target)
    else:
        target = Unmarshaller(use_datetime=use_datetime, use_builtin_types=use_builtin_types)
        if FastParser:
            parser = FastParser(target)
        else:
            parser = ExpatParser(target)
    return parser, target

##
# Convert a Python tuple or a Fault instance to an XML-RPC packet.
#
# @def dumps(params, **options)
# @param params A tuple or Fault instance.
# @keyparam methodname If given, create a methodCall request for
#     this method name.
# @keyparam methodresponse If given, create a methodResponse packet.
#     If used with a tuple, the tuple must be a singleton (that is,
#     it must contain exactly one element).
# @keyparam encoding The packet encoding.
# @return A string containing marshalled data.

def dumps(params, methodname=None, methodresponse=None, encoding=None,
          allow_none=False):
    """data [,options] -> marshalled data

    Convert an argument tuple or a Fault instance to an XML-RPC
    request (or response, if the methodresponse option is used).

    In addition to the data object, the following options can be given
    as keyword arguments:

        methodname: the method name for a methodCall packet

        methodresponse: true to create a methodResponse packet.
        If this option is used with a tuple, the tuple must be
        a singleton (i.e. it can contain only one element).

        encoding: the packet encoding (default is UTF-8)

    All byte strings in the data structure are assumed to use the
    packet encoding.  Unicode strings are automatically converted,
    where necessary.
    """

    assert isinstance(params, (tuple, Fault)), "argument must be tuple or Fault instance"
    if isinstance(params, Fault):
        methodresponse = 1
    elif methodresponse and isinstance(params, tuple):
        assert len(params) == 1, "response tuple must be a singleton"

    if not encoding:
        encoding = "utf-8"

    if FastMarshaller:
        m = FastMarshaller(encoding)
    else:
        m = Marshaller(encoding, allow_none)

    data = m.dumps(params)

    if encoding != "utf-8":
        xmlheader = "<?xml version='1.0' encoding='%s'?>\n" % str(encoding)
    else:
        xmlheader = "<?xml version='1.0'?>\n" # utf-8 is default

    # standard XML-RPC wrappings
    if methodname:
        # a method call
        data = (
            xmlheader,
            "<methodCall>\n"
            "<methodName>", methodname, "</methodName>\n",
            data,
            "</methodCall>\n"
            )
    elif methodresponse:
        # a method response, or a fault structure
        data = (
            xmlheader,
            "<methodResponse>\n",
            data,
            "</methodResponse>\n"
            )
    else:
        return data # return as is
    return "".join(data)

##
# Convert an XML-RPC packet to a Python object.  If the XML-RPC packet
# represents a fault condition, this function raises a Fault exception.
#
# @param data An XML-RPC packet, given as an 8-bit string.
# @return A tuple containing the unpacked data, and the method name
#     (None if not present).
# @see Fault

def loads(data, use_datetime=False, use_builtin_types=False):
    """data -> unmarshalled data, method name

    Convert an XML-RPC packet to unmarshalled data plus a method
    name (None if not present).

    If the XML-RPC packet represents a fault condition, this function
    raises a Fault exception.
    """
    p, u = getparser(use_datetime=use_datetime, use_builtin_types=use_builtin_types)
    p.feed(data)
    p.close()
    return u.close(), u.getmethodname()

##
# Encode a string using the gzip content encoding such as specified by the
# Content-Encoding: gzip
# in the HTTP header, as described in RFC 1952
#
# @param data the unencoded data
# @return the encoded data

def gzip_encode(data):
    """data -> gzip encoded data

    Encode data using the gzip content encoding as described in RFC 1952
    """
    if not gzip:
        raise NotImplementedError
    f = BytesIO()
    with gzip.GzipFile(mode="wb", fileobj=f, compresslevel=1) as gzf:
        gzf.write(data)
    return f.getvalue()

##
# Decode a string using the gzip content encoding such as specified by the
# Content-Encoding: gzip
# in the HTTP header, as described in RFC 1952
#
# @param data The encoded data
# @keyparam max_decode Maximum bytes to decode (20 MiB default), use negative
#    values for unlimited decoding
# @return the unencoded data
# @raises ValueError if data is not correctly coded.
# @raises ValueError if max gzipped payload length exceeded

def gzip_decode(data, max_decode=20971520):
    """gzip encoded data -> unencoded data

    Decode data using the gzip content encoding as described in RFC 1952
    """
    if not gzip:
        raise NotImplementedError
    with gzip.GzipFile(mode="rb", fileobj=BytesIO(data)) as gzf:
        try:
            if max_decode < 0: # no limit
                decoded = gzf.read()
            else:
                decoded = gzf.read(max_decode + 1)
        except OSError:
            raise ValueError("invalid data")
    if max_decode >= 0 and len(decoded) > max_decode:
        raise ValueError("max gzipped payload length exceeded")
    return decoded

##
# Return a decoded file-like object for the gzip encoding
# as described in RFC 1952.
#
# @param response A stream supporting a read() method
# @return a file-like object that the decoded data can be read() from

class GzipDecodedResponse(gzip.GzipFile if gzip else object):
    """a file-like object to decode a response encoded with the gzip
    method, as described in RFC 1952.
    """
    def __init__(self, response):
        #response doesn't support tell() and read(), required by
        #GzipFile
        if not gzip:
            raise NotImplementedError
        self.io = BytesIO(response.read())
        gzip.GzipFile.__init__(self, mode="rb", fileobj=self.io)

    def close(self):
        try:
            gzip.GzipFile.close(self)
        finally:
            self.io.close()


# --------------------------------------------------------------------
# request dispatcher

class _Method:
    # some magic to bind an XML-RPC method to an RPC server.
    # supports "nested" methods (e.g. examples.getStateName)
    def __init__(self, send, name):
        self.__send = send
        self.__name = name
    def __getattr__(self, name):
        return _Method(self.__send, "%s.%s" % (self.__name, name))
    def __call__(self, *args):
        return self.__send(self.__name, args)

##
# Standard transport class for XML-RPC over HTTP.
# <p>
# You can create custom transports by subclassing this method, and
# overriding selected methods.

class Transport:
    """Handles an HTTP transaction to an XML-RPC server."""

    # client identifier (may be overridden)
    user_agent = "Python-xmlrpc/%s" % __version__

    #if true, we'll request gzip encoding
    accept_gzip_encoding = True

    # if positive, encode request using gzip if it exceeds this threshold
    # note that many servers will get confused, so only use it if you know
    # that they can decode such a request
    encode_threshold = None #None = don't encode

    def __init__(self, use_datetime=False, use_builtin_types=False,
                 *, headers=()):
        self._use_datetime = use_datetime
        self._use_builtin_types = use_builtin_types
        self._connection = (None, None)
        self._headers = list(headers)
        self._extra_headers = []

    ##
    # Send a complete request, and parse the response.
    # Retry request if a cached connection has disconnected.
    #
    # @param host Target host.
    # @param handler Target PRC handler.
    # @param request_body XML-RPC request body.
    # @param verbose Debugging flag.
    # @return Parsed response.

    def request(self, host, handler, request_body, verbose=False):
        #retry request once if cached connection has gone cold
        for i in (0, 1):
            try:
                return self.single_request(host, handler, request_body, verbose)
            except http.client.RemoteDisconnected:
                if i:
                    raise
            except OSError as e:
                if i or e.errno not in (errno.ECONNRESET, errno.ECONNABORTED,
                                        errno.EPIPE):
                    raise

    def single_request(self, host, handler, request_body, verbose=False):
        # issue XML-RPC request
        try:
            http_conn = self.send_request(host, handler, request_body, verbose)
            resp = http_conn.getresponse()
            if resp.status == 200:
                self.verbose = verbose
                return self.parse_response(resp)

        except Fault:
            raise
        except Exception:
            #All unexpected errors leave connection in
            # a strange state, so we clear it.
            self.close()
            raise

        #We got an error response.
        #Discard any response data and raise exception
        if resp.getheader("content-length", ""):
            resp.read()
        raise ProtocolError(
            host + handler,
            resp.status, resp.reason,
            dict(resp.getheaders())
            )


    ##
    # Create parser.
    #
    # @return A 2-tuple containing a parser and an unmarshaller.

    def getparser(self):
        # get parser and unmarshaller
        return getparser(use_datetime=self._use_datetime,
                         use_builtin_types=self._use_builtin_types)

    ##
    # Get authorization info from host parameter
    # Host may be a string, or a (host, x509-dict) tuple; if a string,
    # it is checked for a "user:pw@host" format, and a "Basic
    # Authentication" header is added if appropriate.
    #
    # @param host Host descriptor (URL or (URL, x509 info) tuple).
    # @return A 3-tuple containing (actual host, extra headers,
    #     x509 info).  The header and x509 fields may be None.

    def get_host_info(self, host):

        x509 = {}
        if isinstance(host, tuple):
            host, x509 = host

        auth, host = urllib.parse._splituser(host)

        if auth:
            auth = urllib.parse.unquote_to_bytes(auth)
            auth = base64.encodebytes(auth).decode("utf-8")
            auth = "".join(auth.split()) # get rid of whitespace
            extra_headers = [
                ("Authorization", "Basic " + auth)
                ]
        else:
            extra_headers = []

        return host, extra_headers, x509

    ##
    # Connect to server.
    #
    # @param host Target host.
    # @return An HTTPConnection object

    def make_connection(self, host):
        #return an existing connection if possible.  This allows
        #HTTP/1.1 keep-alive.
        if self._connection and host == self._connection[0]:
            return self._connection[1]
        # create a HTTP connection object from a host descriptor
        chost, self._extra_headers, x509 = self.get_host_info(host)
        self._connection = host, http.client.HTTPConnection(chost)
        return self._connection[1]

    ##
    # Clear any cached connection object.
    # Used in the event of socket errors.
    #
    def close(self):
        host, connection = self._connection
        if connection:
            self._connection = (None, None)
            connection.close()

    ##
    # Send HTTP request.
    #
    # @param host Host descriptor (URL or (URL, x509 info) tuple).
    # @param handler Target RPC handler (a path relative to host)
    # @param request_body The XML-RPC request body
    # @param debug Enable debugging if debug is true.
    # @return An HTTPConnection.

    def send_request(self, host, handler, request_body, debug):
        connection = self.make_connection(host)
        headers = self._headers + self._extra_headers
        if debug:
            connection.set_debuglevel(1)
        if self.accept_gzip_encoding and gzip:
            connection.putrequest("POST", handler, skip_accept_encoding=True)
            headers.append(("Accept-Encoding", "gzip"))
        else:
            connection.putrequest("POST", handler)
        headers.append(("Content-Type", "text/xml"))
        headers.append(("User-Agent", self.user_agent))
        self.send_headers(connection, headers)
        self.send_content(connection, request_body)
        return connection

    ##
    # Send request headers.
    # This function provides a useful hook for subclassing
    #
    # @param connection httpConnection.
    # @param headers list of key,value pairs for HTTP headers

    def send_headers(self, connection, headers):
        for key, val in headers:
            connection.putheader(key, val)

    ##
    # Send request body.
    # This function provides a useful hook for subclassing
    #
    # @param connection httpConnection.
    # @param request_body XML-RPC request body.

    def send_content(self, connection, request_body):
        #optionally encode the request
        if (self.encode_threshold is not None and
            self.encode_threshold < len(request_body) and
            gzip):
            connection.putheader("Content-Encoding", "gzip")
            request_body = gzip_encode(request_body)

        connection.putheader("Content-Length", str(len(request_body)))
        connection.endheaders(request_body)

    ##
    # Parse response.
    #
    # @param file Stream.
    # @return Response tuple and target method.

    def parse_response(self, response):
        # read response data from httpresponse, and parse it
        # Check for new http response object, otherwise it is a file object.
        if hasattr(response, 'getheader'):
            if response.getheader("Content-Encoding", "") == "gzip":
                stream = GzipDecodedResponse(response)
            else:
                stream = response
        else:
            stream = response

        p, u = self.getparser()

        while 1:
            data = stream.read(1024)
            if not data:
                break
            if self.verbose:
                print("body:", repr(data))
            p.feed(data)

        if stream is not response:
            stream.close()
        p.close()

        return u.close()

##
# Standard transport class for XML-RPC over HTTPS.

class SafeTransport(Transport):
    """Handles an HTTPS transaction to an XML-RPC server."""

    def __init__(self, use_datetime=False, use_builtin_types=False,
                 *, headers=(), context=None):
        super().__init__(use_datetime=use_datetime,
                         use_builtin_types=use_builtin_types,
                         headers=headers)
        self.context = context

    # FIXME: mostly untested

    def make_connection(self, host):
        if self._connection and host == self._connection[0]:
            return self._connection[1]

        if not hasattr(http.client, "HTTPSConnection"):
            raise NotImplementedError(
            "your version of http.client doesn't support HTTPS")
        # create a HTTPS connection object from a host descriptor
        # host may be a string, or a (host, x509-dict) tuple
        chost, self._extra_headers, x509 = self.get_host_info(host)
        self._connection = host, http.client.HTTPSConnection(chost,
            None, context=self.context, **(x509 or {}))
        return self._connection[1]

##
# Standard server proxy.  This class establishes a virtual connection
# to an XML-RPC server.
# <p>
# This class is available as ServerProxy and Server.  New code should
# use ServerProxy, to avoid confusion.
#
# @def ServerProxy(uri, **options)
# @param uri The connection point on the server.
# @keyparam transport A transport factory, compatible with the
#    standard transport class.
# @keyparam encoding The default encoding used for 8-bit strings
#    (default is UTF-8).
# @keyparam verbose Use a true value to enable debugging output.
#    (printed to standard output).
# @see Transport

class ServerProxy:
    """uri [,options] -> a logical connection to an XML-RPC server

    uri is the connection point on the server, given as
    scheme://host/target.

    The standard implementation always supports the "http" scheme.  If
    SSL socket support is available (Python 2.0), it also supports
    "https".

    If the target part and the slash preceding it are both omitted,
    "/RPC2" is assumed.

    The following options can be given as keyword arguments:

        transport: a transport factory
        encoding: the request encoding (default is UTF-8)

    All 8-bit strings passed to the server proxy are assumed to use
    the given encoding.
    """

    def __init__(self, uri, transport=None, encoding=None, verbose=False,
                 allow_none=False, use_datetime=False, use_builtin_types=False,
                 *, headers=(), context=None):
        # establish a "logical" server connection

        # get the url
        p = urllib.parse.urlsplit(uri)
        if p.scheme not in ("http", "https"):
            raise OSError("unsupported XML-RPC protocol")
        self.__host = p.netloc
        self.__handler = urllib.parse.urlunsplit(["", "", *p[2:]])
        if not self.__handler:
            self.__handler = "/RPC2"

        if transport is None:
            if p.scheme == "https":
                handler = SafeTransport
                extra_kwargs = {"context": context}
            else:
                handler = Transport
                extra_kwargs = {}
            transport = handler(use_datetime=use_datetime,
                                use_builtin_types=use_builtin_types,
                                headers=headers,
                                **extra_kwargs)
        self.__transport = transport

        self.__encoding = encoding or 'utf-8'
        self.__verbose = verbose
        self.__allow_none = allow_none

    def __close(self):
        self.__transport.close()

    def __request(self, methodname, params):
        # call a method on the remote server

        request = dumps(params, methodname, encoding=self.__encoding,
                        allow_none=self.__allow_none).encode(self.__encoding, 'xmlcharrefreplace')

        response = self.__transport.request(
            self.__host,
            self.__handler,
            request,
            verbose=self.__verbose
            )

        if len(response) == 1:
            response = response[0]

        return response

    def __repr__(self):
        return (
            "<%s for %s%s>" %
            (self.__class__.__name__, self.__host, self.__handler)
            )

    def __getattr__(self, name):
        # magic method dispatcher
        return _Method(self.__request, name)

    # note: to call a remote object with a non-standard name, use
    # result getattr(server, "strange-python-name")(args)

    def __call__(self, attr):
        """A workaround to get special attributes on the ServerProxy
           without interfering with the magic __getattr__
        """
        if attr == "close":
            return self.__close
        elif attr == "transport":
            return self.__transport
        raise AttributeError("Attribute %r not found" % (attr,))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.__close()

# compatibility

Server = ServerProxy

# --------------------------------------------------------------------
# test code

if __name__ == "__main__":

    # simple test program (from the XML-RPC specification)

    # local server, available from Lib/xmlrpc/server.py
    server = ServerProxy("http://localhost:8000")

    try:
        print(server.currentTime.getCurrentTime())
    except Error as v:
        print("ERROR", v)

    multi = MultiCall(server)
    multi.getData()
    multi.pow(2,9)
    multi.add(1,2)
    try:
        for response in multi():
            print(response)
    except Error as v:
        print("ERROR", v)
<|endoftext|>r"""XML-RPC Servers.

This module can be used to create simple XML-RPC servers
by creating a server and either installing functions, a
class instance, or by extending the SimpleXMLRPCServer
class.

It can also be used to handle XML-RPC requests in a CGI
environment using CGIXMLRPCRequestHandler.

The Doc* classes can be used to create XML-RPC servers that
serve pydoc-style documentation in response to HTTP
GET requests. This documentation is dynamically generated
based on the functions and methods registered with the
server.

A list of possible usage patterns follows:

1. Install functions:

server = SimpleXMLRPCServer(("localhost", 8000))
server.register_function(pow)
server.register_function(lambda x,y: x+y, 'add')
server.serve_forever()

2. Install an instance:

class MyFuncs:
    def __init__(self):
        # make all of the sys functions available through sys.func_name
        import sys
        self.sys = sys
    def _listMethods(self):
        # implement this method so that system.listMethods
        # knows to advertise the sys methods
        return list_public_methods(self) + \
                ['sys.' + method for method in list_public_methods(self.sys)]
    def pow(self, x, y): return pow(x, y)
    def add(self, x, y) : return x + y

server = SimpleXMLRPCServer(("localhost", 8000))
server.register_introspection_functions()
server.register_instance(MyFuncs())
server.serve_forever()

3. Install an instance with custom dispatch method:

class Math:
    def _listMethods(self):
        # this method must be present for system.listMethods
        # to work
        return ['add', 'pow']
    def _methodHelp(self, method):
        # this method must be present for system.methodHelp
        # to work
        if method == 'add':
            return "add(2,3) => 5"
        elif method == 'pow':
            return "pow(x, y[, z]) => number"
        else:
            # By convention, return empty
            # string if no help is available
            return ""
    def _dispatch(self, method, params):
        if method == 'pow':
            return pow(*params)
        elif method == 'add':
            return params[0] + params[1]
        else:
            raise ValueError('bad method')

server = SimpleXMLRPCServer(("localhost", 8000))
server.register_introspection_functions()
server.register_instance(Math())
server.serve_forever()

4. Subclass SimpleXMLRPCServer:

class MathServer(SimpleXMLRPCServer):
    def _dispatch(self, method, params):
        try:
            # We are forcing the 'export_' prefix on methods that are
            # callable through XML-RPC to prevent potential security
            # problems
            func = getattr(self, 'export_' + method)
        except AttributeError:
            raise Exception('method "%s" is not supported' % method)
        else:
            return func(*params)

    def export_add(self, x, y):
        return x + y

server = MathServer(("localhost", 8000))
server.serve_forever()

5. CGI script:

server = CGIXMLRPCRequestHandler()
server.register_function(pow)
server.handle_request()
"""

# Written by Brian Quinlan (brian@sweetapp.com).
# Based on code written by Fredrik Lundh.

from xmlrpc.client import Fault, dumps, loads, gzip_encode, gzip_decode
from http.server import BaseHTTPRequestHandler
from functools import partial
from inspect import signature
import html
import http.server
import socketserver
import sys
import os
import re
import pydoc
import traceback
try:
    import fcntl
except ImportError:
    fcntl = None

def resolve_dotted_attribute(obj, attr, allow_dotted_names=True):
    """resolve_dotted_attribute(a, 'b.c.d') => a.b.c.d

    Resolves a dotted attribute name to an object.  Raises
    an AttributeError if any attribute in the chain starts with a '_'.

    If the optional allow_dotted_names argument is false, dots are not
    supported and this function operates similar to getattr(obj, attr).
    """

    if allow_dotted_names:
        attrs = attr.split('.')
    else:
        attrs = [attr]

    for i in attrs:
        if i.startswith('_'):
            raise AttributeError(
                'attempt to access private attribute "%s"' % i
                )
        else:
            obj = getattr(obj,i)
    return obj

def list_public_methods(obj):
    """Returns a list of attribute strings, found in the specified
    object, which represent callable attributes"""

    return [member for member in dir(obj)
                if not member.startswith('_') and
                    callable(getattr(obj, member))]

class SimpleXMLRPCDispatcher:
    """Mix-in class that dispatches XML-RPC requests.

    This class is used to register XML-RPC method handlers
    and then to dispatch them. This class doesn't need to be
    instanced directly when used by SimpleXMLRPCServer but it
    can be instanced when used by the MultiPathXMLRPCServer
    """

    def __init__(self, allow_none=False, encoding=None,
                 use_builtin_types=False):
        self.funcs = {}
        self.instance = None
        self.allow_none = allow_none
        self.encoding = encoding or 'utf-8'
        self.use_builtin_types = use_builtin_types

    def register_instance(self, instance, allow_dotted_names=False):
        """Registers an instance to respond to XML-RPC requests.

        Only one instance can be installed at a time.

        If the registered instance has a _dispatch method then that
        method will be called with the name of the XML-RPC method and
        its parameters as a tuple
        e.g. instance._dispatch('add',(2,3))

        If the registered instance does not have a _dispatch method
        then the instance will be searched to find a matching method
        and, if found, will be called. Methods beginning with an '_'
        are considered private and will not be called by
        SimpleXMLRPCServer.

        If a registered function matches an XML-RPC request, then it
        will be called instead of the registered instance.

        If the optional allow_dotted_names argument is true and the
        instance does not have a _dispatch method, method names
        containing dots are supported and resolved, as long as none of
        the name segments start with an '_'.

            *** SECURITY WARNING: ***

            Enabling the allow_dotted_names options allows intruders
            to access your module's global variables and may allow
            intruders to execute arbitrary code on your machine.  Only
            use this option on a secure, closed network.

        """

        self.instance = instance
        self.allow_dotted_names = allow_dotted_names

    def register_function(self, function=None, name=None):
        """Registers a function to respond to XML-RPC requests.

        The optional name argument can be used to set a Unicode name
        for the function.
        """
        # decorator factory
        if function is None:
            return partial(self.register_function, name=name)

        if name is None:
            name = function.__name__
        self.funcs[name] = function

        return function

    def register_introspection_functions(self):
        """Registers the XML-RPC introspection methods in the system
        namespace.

        see http://xmlrpc.usefulinc.com/doc/reserved.html
        """

        self.funcs.update({'system.listMethods' : self.system_listMethods,
                      'system.methodSignature' : self.system_methodSignature,
                      'system.methodHelp' : self.system_methodHelp})

    def register_multicall_functions(self):
        """Registers the XML-RPC multicall method in the system
        namespace.

        see http://www.xmlrpc.com/discuss/msgReader$1208"""

        self.funcs.update({'system.multicall' : self.system_multicall})

    def _marshaled_dispatch(self, data, dispatch_method = None, path = None):
        """Dispatches an XML-RPC method from marshalled (XML) data.

        XML-RPC methods are dispatched from the marshalled (XML) data
        using the _dispatch method and the result is returned as
        marshalled data. For backwards compatibility, a dispatch
        function can be provided as an argument (see comment in
        SimpleXMLRPCRequestHandler.do_POST) but overriding the
        existing method through subclassing is the preferred means
        of changing method dispatch behavior.
        """

        try:
            params, method = loads(data, use_builtin_types=self.use_builtin_types)

            # generate response
            if dispatch_method is not None:
                response = dispatch_method(method, params)
            else:
                response = self._dispatch(method, params)
            # wrap response in a singleton tuple
            response = (response,)
            response = dumps(response, methodresponse=1,
                             allow_none=self.allow_none, encoding=self.encoding)
        except Fault as fault:
            response = dumps(fault, allow_none=self.allow_none,
                             encoding=self.encoding)
        except:
            # report exception back to server
            exc_type, exc_value, exc_tb = sys.exc_info()
            try:
                response = dumps(
                    Fault(1, "%s:%s" % (exc_type, exc_value)),
                    encoding=self.encoding, allow_none=self.allow_none,
                    )
            finally:
                # Break reference cycle
                exc_type = exc_value = exc_tb = None

        return response.encode(self.encoding, 'xmlcharrefreplace')

    def system_listMethods(self):
        """system.listMethods() => ['add', 'subtract', 'multiple']

        Returns a list of the methods supported by the server."""

        methods = set(self.funcs.keys())
        if self.instance is not None:
            # Instance can implement _listMethod to return a list of
            # methods
            if hasattr(self.instance, '_listMethods'):
                methods |= set(self.instance._listMethods())
            # if the instance has a _dispatch method then we
            # don't have enough information to provide a list
            # of methods
            elif not hasattr(self.instance, '_dispatch'):
                methods |= set(list_public_methods(self.instance))
        return sorted(methods)

    def system_methodSignature(self, method_name):
        """system.methodSignature('add') => [double, int, int]

        Returns a list describing the signature of the method. In the
        above example, the add method takes two integers as arguments
        and returns a double result.

        This server does NOT support system.methodSignature."""

        # See http://xmlrpc.usefulinc.com/doc/sysmethodsig.html

        return 'signatures not supported'

    def system_methodHelp(self, method_name):
        """system.methodHelp('add') => "Adds two integers together"

        Returns a string containing documentation for the specified method."""

        method = None
        if method_name in self.funcs:
            method = self.funcs[method_name]
        elif self.instance is not None:
            # Instance can implement _methodHelp to return help for a method
            if hasattr(self.instance, '_methodHelp'):
                return self.instance._methodHelp(method_name)
            # if the instance has a _dispatch method then we
            # don't have enough information to provide help
            elif not hasattr(self.instance, '_dispatch'):
                try:
                    method = resolve_dotted_attribute(
                                self.instance,
                                method_name,
                                self.allow_dotted_names
                                )
                except AttributeError:
                    pass

        # Note that we aren't checking that the method actually
        # be a callable object of some kind
        if method is None:
            return ""
        else:
            return pydoc.getdoc(method)

    def system_multicall(self, call_list):
        """system.multicall([{'methodName': 'add', 'params': [2, 2]}, ...]) => \
[[4], ...]

        Allows the caller to package multiple XML-RPC calls into a single
        request.

        See http://www.xmlrpc.com/discuss/msgReader$1208
        """

        results = []
        for call in call_list:
            method_name = call['methodName']
            params = call['params']

            try:
                # XXX A marshalling error in any response will fail the entire
                # multicall. If someone cares they should fix this.
                results.append([self._dispatch(method_name, params)])
            except Fault as fault:
                results.append(
                    {'faultCode' : fault.faultCode,
                     'faultString' : fault.faultString}
                    )
            except:
                exc_type, exc_value, exc_tb = sys.exc_info()
                try:
                    results.append(
                        {'faultCode' : 1,
                         'faultString' : "%s:%s" % (exc_type, exc_value)}
                        )
                finally:
                    # Break reference cycle
                    exc_type = exc_value = exc_tb = None
        return results

    def _dispatch(self, method, params):
        """Dispatches the XML-RPC method.

        XML-RPC calls are forwarded to a registered function that
        matches the called XML-RPC method name. If no such function
        exists then the call is forwarded to the registered instance,
        if available.

        If the registered instance has a _dispatch method then that
        method will be called with the name of the XML-RPC method and
        its parameters as a tuple
        e.g. instance._dispatch('add',(2,3))

        If the registered instance does not have a _dispatch method
        then the instance will be searched to find a matching method
        and, if found, will be called.

        Methods beginning with an '_' are considered private and will
        not be called.
        """

        try:
            # call the matching registered function
            func = self.funcs[method]
        except KeyError:
            pass
        else:
            if func is not None:
                return func(*params)
            raise Exception('method "%s" is not supported' % method)

        if self.instance is not None:
            if hasattr(self.instance, '_dispatch'):
                # call the `_dispatch` method on the instance
                return self.instance._dispatch(method, params)

            # call the instance's method directly
            try:
                func = resolve_dotted_attribute(
                    self.instance,
                    method,
                    self.allow_dotted_names
                )
            except AttributeError:
                pass
            else:
                if func is not None:
                    return func(*params)

        raise Exception('method "%s" is not supported' % method)

class SimpleXMLRPCRequestHandler(BaseHTTPRequestHandler):
    """Simple XML-RPC request handler class.

    Handles all HTTP POST requests and attempts to decode them as
    XML-RPC requests.
    """

    # Class attribute listing the accessible path components;
    # paths not on this list will result in a 404 error.
    rpc_paths = ('/', '/RPC2')

    #if not None, encode responses larger than this, if possible
    encode_threshold = 1400 #a common MTU

    #Override form StreamRequestHandler: full buffering of output
    #and no Nagle.
    wbufsize = -1
    disable_nagle_algorithm = True

    # a re to match a gzip Accept-Encoding
    aepattern = re.compile(r"""
                            \s* ([^\s;]+) \s*            #content-coding
                            (;\s* q \s*=\s* ([0-9\.]+))? #q
                            """, re.VERBOSE | re.IGNORECASE)

    def accept_encodings(self):
        r = {}
        ae = self.headers.get("Accept-Encoding", "")
        for e in ae.split(","):
            match = self.aepattern.match(e)
            if match:
                v = match.group(3)
                v = float(v) if v else 1.0
                r[match.group(1)] = v
        return r

    def is_rpc_path_valid(self):
        if self.rpc_paths:
            return self.path in self.rpc_paths
        else:
            # If .rpc_paths is empty, just assume all paths are legal
            return True

    def do_POST(self):
        """Handles the HTTP POST request.

        Attempts to interpret all HTTP POST requests as XML-RPC calls,
        which are forwarded to the server's _dispatch method for handling.
        """

        # Check that the path is legal
        if not self.is_rpc_path_valid():
            self.report_404()
            return

        try:
            # Get arguments by reading body of request.
            # We read this in chunks to avoid straining
            # socket.read(); around the 10 or 15Mb mark, some platforms
            # begin to have problems (bug #792570).
            max_chunk_size = 10*1024*1024
            size_remaining = int(self.headers["content-length"])
            L = []
            while size_remaining:
                chunk_size = min(size_remaining, max_chunk_size)
                chunk = self.rfile.read(chunk_size)
                if not chunk:
                    break
                L.append(chunk)
                size_remaining -= len(L[-1])
            data = b''.join(L)

            data = self.decode_request_content(data)
            if data is None:
                return #response has been sent

            # In previous versions of SimpleXMLRPCServer, _dispatch
            # could be overridden in this class, instead of in
            # SimpleXMLRPCDispatcher. To maintain backwards compatibility,
            # check to see if a subclass implements _dispatch and dispatch
            # using that method if present.
            response = self.server._marshaled_dispatch(
                    data, getattr(self, '_dispatch', None), self.path
                )
        except Exception as e: # This should only happen if the module is buggy
            # internal error, report as HTTP server error
            self.send_response(500)

            # Send information about the exception if requested
            if hasattr(self.server, '_send_traceback_header') and \
                    self.server._send_traceback_header:
                self.send_header("X-exception", str(e))
                trace = traceback.format_exc()
                trace = str(trace.encode('ASCII', 'backslashreplace'), 'ASCII')
                self.send_header("X-traceback", trace)

            self.send_header("Content-length", "0")
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/xml")
            if self.encode_threshold is not None:
                if len(response) > self.encode_threshold:
                    q = self.accept_encodings().get("gzip", 0)
                    if q:
                        try:
                            response = gzip_encode(response)
                            self.send_header("Content-Encoding", "gzip")
                        except NotImplementedError:
                            pass
            self.send_header("Content-length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    def decode_request_content(self, data):
        #support gzip encoding of request
        encoding = self.headers.get("content-encoding", "identity").lower()
        if encoding == "identity":
            return data
        if encoding == "gzip":
            try:
                return gzip_decode(data)
            except NotImplementedError:
                self.send_response(501, "encoding %r not supported" % encoding)
            except ValueError:
                self.send_response(400, "error decoding gzip content")
        else:
            self.send_response(501, "encoding %r not supported" % encoding)
        self.send_header("Content-length", "0")
        self.end_headers()

    def report_404 (self):
            # Report a 404 error
        self.send_response(404)
        response = b'No such page'
        self.send_header("Content-type", "text/plain")
        self.send_header("Content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_request(self, code='-', size='-'):
        """Selectively log an accepted request."""

        if self.server.logRequests:
            BaseHTTPRequestHandler.log_request(self, code, size)

class SimpleXMLRPCServer(socketserver.TCPServer,
                         SimpleXMLRPCDispatcher):
    """Simple XML-RPC server.

    Simple XML-RPC server that allows functions and a single instance
    to be installed to handle requests. The default implementation
    attempts to dispatch XML-RPC calls to the functions or instance
    installed in the server. Override the _dispatch method inherited
    from SimpleXMLRPCDispatcher to change this behavior.
    """

    allow_reuse_address = True

    # Warning: this is for debugging purposes only! Never set this to True in
    # production code, as will be sending out sensitive information (exception
    # and stack trace details) when exceptions are raised inside
    # SimpleXMLRPCRequestHandler.do_POST
    _send_traceback_header = False

    def __init__(self, addr, requestHandler=SimpleXMLRPCRequestHandler,
                 logRequests=True, allow_none=False, encoding=None,
                 bind_and_activate=True, use_builtin_types=False):
        self.logRequests = logRequests

        SimpleXMLRPCDispatcher.__init__(self, allow_none, encoding, use_builtin_types)
        socketserver.TCPServer.__init__(self, addr, requestHandler, bind_and_activate)


class MultiPathXMLRPCServer(SimpleXMLRPCServer):
    """Multipath XML-RPC Server
    This specialization of SimpleXMLRPCServer allows the user to create
    multiple Dispatcher instances and assign them to different
    HTTP request paths.  This makes it possible to run two or more
    'virtual XML-RPC servers' at the same port.
    Make sure that the requestHandler accepts the paths in question.
    """
    def __init__(self, addr, requestHandler=SimpleXMLRPCRequestHandler,
                 logRequests=True, allow_none=False, encoding=None,
                 bind_and_activate=True, use_builtin_types=False):

        SimpleXMLRPCServer.__init__(self, addr, requestHandler, logRequests, allow_none,
                                    encoding, bind_and_activate, use_builtin_types)
        self.dispatchers = {}
        self.allow_none = allow_none
        self.encoding = encoding or 'utf-8'

    def add_dispatcher(self, path, dispatcher):
        self.dispatchers[path] = dispatcher
        return dispatcher

    def get_dispatcher(self, path):
        return self.dispatchers[path]

    def _marshaled_dispatch(self, data, dispatch_method = None, path = None):
        try:
            response = self.dispatchers[path]._marshaled_dispatch(
               data, dispatch_method, path)
        except:
            # report low level exception back to server
            # (each dispatcher should have handled their own
            # exceptions)
            exc_type, exc_value = sys.exc_info()[:2]
            try:
                response = dumps(
                    Fault(1, "%s:%s" % (exc_type, exc_value)),
                    encoding=self.encoding, allow_none=self.allow_none)
                response = response.encode(self.encoding, 'xmlcharrefreplace')
            finally:
                # Break reference cycle
                exc_type = exc_value = None
        return response

class CGIXMLRPCRequestHandler(SimpleXMLRPCDispatcher):
    """Simple handler for XML-RPC data passed through CGI."""

    def __init__(self, allow_none=False, encoding=None, use_builtin_types=False):
        SimpleXMLRPCDispatcher.__init__(self, allow_none, encoding, use_builtin_types)

    def handle_xmlrpc(self, request_text):
        """Handle a single XML-RPC request"""

        response = self._marshaled_dispatch(request_text)

        print('Content-Type: text/xml')
        print('Content-Length: %d' % len(response))
        print()
        sys.stdout.flush()
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    def handle_get(self):
        """Handle a single HTTP GET request.

        Default implementation indicates an error because
        XML-RPC uses the POST method.
        """

        code = 400
        message, explain = BaseHTTPRequestHandler.responses[code]

        response = http.server.DEFAULT_ERROR_MESSAGE % \
            {
             'code' : code,
             'message' : message,
             'explain' : explain
            }
        response = response.encode('utf-8')
        print('Status: %d %s' % (code, message))
        print('Content-Type: %s' % http.server.DEFAULT_ERROR_CONTENT_TYPE)
        print('Content-Length: %d' % len(response))
        print()
        sys.stdout.flush()
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    def handle_request(self, request_text=None):
        """Handle a single XML-RPC request passed through a CGI post method.

        If no XML data is given then it is read from stdin. The resulting
        XML-RPC response is printed to stdout along with the correct HTTP
        headers.
        """

        if request_text is None and \
            os.environ.get('REQUEST_METHOD', None) == 'GET':
            self.handle_get()
        else:
            # POST data is normally available through stdin
            try:
                length = int(os.environ.get('CONTENT_LENGTH', None))
            except (ValueError, TypeError):
                length = -1
            if request_text is None:
                request_text = sys.stdin.read(length)

            self.handle_xmlrpc(request_text)


# -----------------------------------------------------------------------------
# Self documenting XML-RPC Server.

class ServerHTMLDoc(pydoc.HTMLDoc):
    """Class used to generate pydoc HTML document for a server"""

    def markup(self, text, escape=None, funcs={}, classes={}, methods={}):
        """Mark up some plain text, given a context of symbols to look for.
        Each context dictionary maps object names to anchor names."""
        escape = escape or self.escape
        results = []
        here = 0

        # XXX Note that this regular expression does not allow for the
        # hyperlinking of arbitrary strings being used as method
        # names. Only methods with names consisting of word characters
        # and '.'s are hyperlinked.
        pattern = re.compile(r'\b((http|https|ftp)://\S+[\w/]|'
                                r'RFC[- ]?(\d+)|'
                                r'PEP[- ]?(\d+)|'
                                r'(self\.)?((?:\w|\.)+))\b')
        while 1:
            match = pattern.search(text, here)
            if not match: break
            start, end = match.span()
            results.append(escape(text[here:start]))

            all, scheme, rfc, pep, selfdot, name = match.groups()
            if scheme:
                url = escape(all).replace('"', '&quot;')
                results.append('<a href="%s">%s</a>' % (url, url))
            elif rfc:
                url = 'http://www.rfc-editor.org/rfc/rfc%d.txt' % int(rfc)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif pep:
                url = 'https://www.python.org/dev/peps/pep-%04d/' % int(pep)
                results.append('<a href="%s">%s</a>' % (url, escape(all)))
            elif text[end:end+1] == '(':
                results.append(self.namelink(name, methods, funcs, classes))
            elif selfdot:
                results.append('self.<strong>%s</strong>' % name)
            else:
                results.append(self.namelink(name, classes))
            here = end
        results.append(escape(text[here:]))
        return ''.join(results)

    def docroutine(self, object, name, mod=None,
                   funcs={}, classes={}, methods={}, cl=None):
        """Produce HTML documentation for a function or method object."""

        anchor = (cl and cl.__name__ or '') + '-' + name
        note = ''

        title = '<a name="%s"><strong>%s</strong></a>' % (
            self.escape(anchor), self.escape(name))

        if callable(object):
            argspec = str(signature(object))
        else:
            argspec = '(...)'

        if isinstance(object, tuple):
            argspec = object[0] or argspec
            docstring = object[1] or ""
        else:
            docstring = pydoc.getdoc(object)

        decl = title + argspec + (note and self.grey(
               '<font face="helvetica, arial">%s</font>' % note))

        doc = self.markup(
            docstring, self.preformat, funcs, classes, methods)
        doc = doc and '<dd><tt>%s</tt></dd>' % doc
        return '<dl><dt>%s</dt>%s</dl>\n' % (decl, doc)

    def docserver(self, server_name, package_documentation, methods):
        """Produce HTML documentation for an XML-RPC server."""

        fdict = {}
        for key, value in methods.items():
            fdict[key] = '#-' + key
            fdict[value] = fdict[key]

        server_name = self.escape(server_name)
        head = '<big><big><strong>%s</strong></big></big>' % server_name
        result = self.heading(head, '#ffffff', '#7799ee')

        doc = self.markup(package_documentation, self.preformat, fdict)
        doc = doc and '<tt>%s</tt>' % doc
        result = result + '<p>%s</p>\n' % doc

        contents = []
        method_items = sorted(methods.items())
        for key, value in method_items:
            contents.append(self.docroutine(value, key, funcs=fdict))
        result = result + self.bigsection(
            'Methods', '#ffffff', '#eeaa77', ''.join(contents))

        return result

class XMLRPCDocGenerator:
    """Generates documentation for an XML-RPC server.

    This class is designed as mix-in and should not
    be constructed directly.
    """

    def __init__(self):
        # setup variables used for HTML documentation
        self.server_name = 'XML-RPC Server Documentation'
        self.server_documentation = \
            "This server exports the following methods through the XML-RPC "\
            "protocol."
        self.server_title = 'XML-RPC Server Documentation'

    def set_server_title(self, server_title):
        """Set the HTML title of the generated server documentation"""

        self.server_title = server_title

    def set_server_name(self, server_name):
        """Set the name of the generated HTML server documentation"""

        self.server_name = server_name

    def set_server_documentation(self, server_documentation):
        """Set the documentation string for the entire server."""

        self.server_documentation = server_documentation

    def generate_html_documentation(self):
        """generate_html_documentation() => html documentation for the server

        Generates HTML documentation for the server using introspection for
        installed functions and instances that do not implement the
        _dispatch method. Alternatively, instances can choose to implement
        the _get_method_argstring(method_name) method to provide the
        argument string used in the documentation and the
        _methodHelp(method_name) method to provide the help text used
        in the documentation."""

        methods = {}

        for method_name in self.system_listMethods():
            if method_name in self.funcs:
                method = self.funcs[method_name]
            elif self.instance is not None:
                method_info = [None, None] # argspec, documentation
                if hasattr(self.instance, '_get_method_argstring'):
                    method_info[0] = self.instance._get_method_argstring(method_name)
                if hasattr(self.instance, '_methodHelp'):
                    method_info[1] = self.instance._methodHelp(method_name)

                method_info = tuple(method_info)
                if method_info != (None, None):
                    method = method_info
                elif not hasattr(self.instance, '_dispatch'):
                    try:
                        method = resolve_dotted_attribute(
                                    self.instance,
                                    method_name
                                    )
                    except AttributeError:
                        method = method_info
                else:
                    method = method_info
            else:
                assert 0, "Could not find method in self.functions and no "\
                          "instance installed"

            methods[method_name] = method

        documenter = ServerHTMLDoc()
        documentation = documenter.docserver(
                                self.server_name,
                                self.server_documentation,
                                methods
                            )

        return documenter.page(html.escape(self.server_title), documentation)

class DocXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    """XML-RPC and documentation request handler class.

    Handles all HTTP POST requests and attempts to decode them as
    XML-RPC requests.

    Handles all HTTP GET requests and interprets them as requests
    for documentation.
    """

    def do_GET(self):
        """Handles the HTTP GET request.

        Interpret all HTTP GET requests as requests for server
        documentation.
        """
        # Check that the path is legal
        if not self.is_rpc_path_valid():
            self.report_404()
            return

        response = self.server.generate_html_documentation().encode('utf-8')
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

class DocXMLRPCServer(  SimpleXMLRPCServer,
                        XMLRPCDocGenerator):
    """XML-RPC and HTML documentation server.

    Adds the ability to serve server documentation to the capabilities
    of SimpleXMLRPCServer.
    """

    def __init__(self, addr, requestHandler=DocXMLRPCRequestHandler,
                 logRequests=True, allow_none=False, encoding=None,
                 bind_and_activate=True, use_builtin_types=False):
        SimpleXMLRPCServer.__init__(self, addr, requestHandler, logRequests,
                                    allow_none, encoding, bind_and_activate,
                                    use_builtin_types)
        XMLRPCDocGenerator.__init__(self)

class DocCGIXMLRPCRequestHandler(   CGIXMLRPCRequestHandler,
                                    XMLRPCDocGenerator):
    """Handler for XML-RPC data and documentation requests passed through
    CGI"""

    def handle_get(self):
        """Handles the HTTP GET request.

        Interpret all HTTP GET requests as requests for server
        documentation.
        """

        response = self.generate_html_documentation().encode('utf-8')

        print('Content-Type: text/html')
        print('Content-Length: %d' % len(response))
        print()
        sys.stdout.flush()
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()

    def __init__(self):
        CGIXMLRPCRequestHandler.__init__(self)
        XMLRPCDocGenerator.__init__(self)


if __name__ == '__main__':
    import datetime

    class ExampleService:
        def getData(self):
            return '42'

        class currentTime:
            @staticmethod
            def getCurrentTime():
                return datetime.datetime.now()

    with SimpleXMLRPCServer(("localhost", 8000)) as server:
        server.register_function(pow)
        server.register_function(lambda x,y: x+y, 'add')
        server.register_instance(ExampleService(), allow_dotted_names=True)
        server.register_multicall_functions()
        print('Serving XML-RPC on localhost port 8000')
        print('It is advisable to run this example server within a secure, closed network.')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            sys.exit(0)
<|endoftext|>"""
Virtual environment (venv) package for Python. Based on PEP 405.

Copyright (C) 2011-2014 Vinay Sajip.
Licensed to the PSF under a contributor agreement.
"""
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
import shlex


CORE_VENV_DEPS = ('pip', 'setuptools')
logger = logging.getLogger(__name__)


class EnvBuilder:
    """
    This class exists to allow virtual environment creation to be
    customized. The constructor parameters determine the builder's
    behaviour when called upon to create a virtual environment.

    By default, the builder makes the system (global) site-packages dir
    *un*available to the created environment.

    If invoked using the Python -m option, the default is to use copying
    on Windows platforms but symlinks elsewhere. If instantiated some
    other way, the default is to *not* use symlinks.

    :param system_site_packages: If True, the system (global) site-packages
                                 dir is available to created environments.
    :param clear: If True, delete the contents of the environment directory if
                  it already exists, before environment creation.
    :param symlinks: If True, attempt to symlink rather than copy files into
                     virtual environment.
    :param upgrade: If True, upgrade an existing virtual environment.
    :param with_pip: If True, ensure pip is installed in the virtual
                     environment
    :param prompt: Alternative terminal prefix for the environment.
    :param upgrade_deps: Update the base venv modules to the latest on PyPI
    """

    def __init__(self, system_site_packages=False, clear=False,
                 symlinks=False, upgrade=False, with_pip=False, prompt=None,
                 upgrade_deps=False):
        self.system_site_packages = system_site_packages
        self.clear = clear
        self.symlinks = symlinks
        self.upgrade = upgrade
        self.with_pip = with_pip
        if prompt == '.':  # see bpo-38901
            prompt = os.path.basename(os.getcwd())
        self.prompt = prompt
        self.upgrade_deps = upgrade_deps

    def create(self, env_dir):
        """
        Create a virtual environment in a directory.

        :param env_dir: The target directory to create an environment in.

        """
        env_dir = os.path.abspath(env_dir)
        context = self.ensure_directories(env_dir)
        # See issue 24875. We need system_site_packages to be False
        # until after pip is installed.
        true_system_site_packages = self.system_site_packages
        self.system_site_packages = False
        self.create_configuration(context)
        self.setup_python(context)
        if self.with_pip:
            self._setup_pip(context)
        if not self.upgrade:
            self.setup_scripts(context)
            self.post_setup(context)
        if true_system_site_packages:
            # We had set it to False before, now
            # restore it and rewrite the configuration
            self.system_site_packages = True
            self.create_configuration(context)
        if self.upgrade_deps:
            self.upgrade_dependencies(context)

    def clear_directory(self, path):
        for fn in os.listdir(path):
            fn = os.path.join(path, fn)
            if os.path.islink(fn) or os.path.isfile(fn):
                os.remove(fn)
            elif os.path.isdir(fn):
                shutil.rmtree(fn)

    def ensure_directories(self, env_dir):
        """
        Create the directories for the environment.

        Returns a context object which holds paths in the environment,
        for use by subsequent logic.
        """

        def create_if_needed(d):
            if not os.path.exists(d):
                os.makedirs(d)
            elif os.path.islink(d) or os.path.isfile(d):
                raise ValueError('Unable to create directory %r' % d)

        if os.path.exists(env_dir) and self.clear:
            self.clear_directory(env_dir)
        context = types.SimpleNamespace()
        context.env_dir = env_dir
        context.env_name = os.path.split(env_dir)[1]
        prompt = self.prompt if self.prompt is not None else context.env_name
        context.prompt = '(%s) ' % prompt
        create_if_needed(env_dir)
        executable = sys._base_executable
        if not executable:  # see gh-96861
            raise ValueError('Unable to determine path to the running '
                             'Python interpreter. Provide an explicit path or '
                             'check that your PATH environment variable is '
                             'correctly set.')
        dirname, exename = os.path.split(os.path.abspath(executable))
        context.executable = executable
        context.python_dir = dirname
        context.python_exe = exename
        if sys.platform == 'win32':
            binname = 'Scripts'
            incpath = 'Include'
            libpath = os.path.join(env_dir, 'Lib', 'site-packages')
        else:
            binname = 'bin'
            incpath = 'include'
            libpath = os.path.join(env_dir, 'lib',
                                   'python%d.%d' % sys.version_info[:2],
                                   'site-packages')
        context.inc_path = path = os.path.join(env_dir, incpath)
        create_if_needed(path)
        create_if_needed(libpath)
        # Issue 21197: create lib64 as a symlink to lib on 64-bit non-OS X POSIX
        if ((sys.maxsize > 2**32) and (os.name == 'posix') and
            (sys.platform != 'darwin')):
            link_path = os.path.join(env_dir, 'lib64')
            if not os.path.exists(link_path):   # Issue #21643
                os.symlink('lib', link_path)
        context.bin_path = binpath = os.path.join(env_dir, binname)
        context.bin_name = binname
        context.env_exe = os.path.join(binpath, exename)
        create_if_needed(binpath)
        # Assign and update the command to use when launching the newly created
        # environment, in case it isn't simply the executable script (e.g. bpo-45337)
        context.env_exec_cmd = context.env_exe
        if sys.platform == 'win32':
            # bpo-45337: Fix up env_exec_cmd to account for file system redirections.
            # Some redirects only apply to CreateFile and not CreateProcess
            real_env_exe = os.path.realpath(context.env_exe)
            if os.path.normcase(real_env_exe) != os.path.normcase(context.env_exe):
                logger.warning('Actual environment location may have moved due to '
                               'redirects, links or junctions.\n'
                               '  Requested location: "%s"\n'
                               '  Actual location:    "%s"',
                               context.env_exe, real_env_exe)
                context.env_exec_cmd = real_env_exe
        return context

    def create_configuration(self, context):
        """
        Create a configuration file indicating where the environment's Python
        was copied from, and whether the system site-packages should be made
        available in the environment.

        :param context: The information for the environment creation request
                        being processed.
        """
        context.cfg_path = path = os.path.join(context.env_dir, 'pyvenv.cfg')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('home = %s\n' % context.python_dir)
            if self.system_site_packages:
                incl = 'true'
            else:
                incl = 'false'
            f.write('include-system-site-packages = %s\n' % incl)
            f.write('version = %d.%d.%d\n' % sys.version_info[:3])
            if self.prompt is not None:
                f.write(f'prompt = {self.prompt!r}\n')

    if os.name != 'nt':
        def symlink_or_copy(self, src, dst, relative_symlinks_ok=False):
            """
            Try symlinking a file, and if that fails, fall back to copying.
            """
            force_copy = not self.symlinks
            if not force_copy:
                try:
                    if not os.path.islink(dst): # can't link to itself!
                        if relative_symlinks_ok:
                            assert os.path.dirname(src) == os.path.dirname(dst)
                            os.symlink(os.path.basename(src), dst)
                        else:
                            os.symlink(src, dst)
                except Exception:   # may need to use a more specific exception
                    logger.warning('Unable to symlink %r to %r', src, dst)
                    force_copy = True
            if force_copy:
                shutil.copyfile(src, dst)
    else:
        def symlink_or_copy(self, src, dst, relative_symlinks_ok=False):
            """
            Try symlinking a file, and if that fails, fall back to copying.
            """
            bad_src = os.path.lexists(src) and not os.path.exists(src)
            if self.symlinks and not bad_src and not os.path.islink(dst):
                try:
                    if relative_symlinks_ok:
                        assert os.path.dirname(src) == os.path.dirname(dst)
                        os.symlink(os.path.basename(src), dst)
                    else:
                        os.symlink(src, dst)
                    return
                except Exception:   # may need to use a more specific exception
                    logger.warning('Unable to symlink %r to %r', src, dst)

            # On Windows, we rewrite symlinks to our base python.exe into
            # copies of venvlauncher.exe
            basename, ext = os.path.splitext(os.path.basename(src))
            srcfn = os.path.join(os.path.dirname(__file__),
                                 "scripts",
                                 "nt",
                                 basename + ext)
            # Builds or venv's from builds need to remap source file
            # locations, as we do not put them into Lib/venv/scripts
            if sysconfig.is_python_build(True) or not os.path.isfile(srcfn):
                if basename.endswith('_d'):
                    ext = '_d' + ext
                    basename = basename[:-2]
                if basename == 'python':
                    basename = 'venvlauncher'
                elif basename == 'pythonw':
                    basename = 'venvwlauncher'
                src = os.path.join(os.path.dirname(src), basename + ext)
            else:
                src = srcfn
            if not os.path.exists(src):
                if not bad_src:
                    logger.warning('Unable to copy %r', src)
                return

            shutil.copyfile(src, dst)

    def setup_python(self, context):
        """
        Set up a Python executable in the environment.

        :param context: The information for the environment creation request
                        being processed.
        """
        binpath = context.bin_path
        path = context.env_exe
        copier = self.symlink_or_copy
        dirname = context.python_dir
        if os.name != 'nt':
            copier(context.executable, path)
            if not os.path.islink(path):
                os.chmod(path, 0o755)
            for suffix in ('python', 'python3', f'python3.{sys.version_info[1]}'):
                path = os.path.join(binpath, suffix)
                if not os.path.exists(path):
                    # Issue 18807: make copies if
                    # symlinks are not wanted
                    copier(context.env_exe, path, relative_symlinks_ok=True)
                    if not os.path.islink(path):
                        os.chmod(path, 0o755)
        else:
            if self.symlinks:
                # For symlinking, we need a complete copy of the root directory
                # If symlinks fail, you'll get unnecessary copies of files, but
                # we assume that if you've opted into symlinks on Windows then
                # you know what you're doing.
                suffixes = [
                    f for f in os.listdir(dirname) if
                    os.path.normcase(os.path.splitext(f)[1]) in ('.exe', '.dll')
                ]
                if sysconfig.is_python_build(True):
                    suffixes = [
                        f for f in suffixes if
                        os.path.normcase(f).startswith(('python', 'vcruntime'))
                    ]
            else:
                suffixes = {'python.exe', 'python_d.exe', 'pythonw.exe', 'pythonw_d.exe'}
                base_exe = os.path.basename(context.env_exe)
                suffixes.add(base_exe)

            for suffix in suffixes:
                src = os.path.join(dirname, suffix)
                if os.path.lexists(src):
                    copier(src, os.path.join(binpath, suffix))

            if sysconfig.is_python_build(True):
                # copy init.tcl
                for root, dirs, files in os.walk(context.python_dir):
                    if 'init.tcl' in files:
                        tcldir = os.path.basename(root)
                        tcldir = os.path.join(context.env_dir, 'Lib', tcldir)
                        if not os.path.exists(tcldir):
                            os.makedirs(tcldir)
                        src = os.path.join(root, 'init.tcl')
                        dst = os.path.join(tcldir, 'init.tcl')
                        shutil.copyfile(src, dst)
                        break

    def _call_new_python(self, context, *py_args, **kwargs):
        """Executes the newly created Python using safe-ish options"""
        # gh-98251: We do not want to just use '-I' because that masks
        # legitimate user preferences (such as not writing bytecode). All we
        # really need is to ensure that the path variables do not overrule
        # normal venv handling.
        args = [context.env_exec_cmd, *py_args]
        kwargs['env'] = env = os.environ.copy()
        env['VIRTUAL_ENV'] = context.env_dir
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        kwargs['cwd'] = context.env_dir
        kwargs['executable'] = context.env_exec_cmd
        subprocess.check_output(args, **kwargs)

    def _setup_pip(self, context):
        """Installs or upgrades pip in a virtual environment"""
        try:
            self._call_new_python(context, '-m', 'ensurepip', '--upgrade',
                                  '--default-pip', stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            stdlib = sysconfig.get_path('stdlib')
            if not os.path.exists(f'{stdlib}/ensurepip/__main__.py'):
                print("""\
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python{}-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.

Failing command: {}
""".format(sysconfig.get_python_version(), context.env_exec_cmd))
                sys.exit(1)
            raise

    def setup_scripts(self, context):
        """
        Set up scripts into the created environment from a directory.

        This method installs the default scripts into the environment
        being created. You can prevent the default installation by overriding
        this method if you really need to, or if you need to specify
        a different location for the scripts to install. By default, the
        'scripts' directory in the venv package is used as the source of
        scripts to install.
        """
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(path, 'scripts')
        self.install_scripts(context, path)

    def post_setup(self, context):
        """
        Hook for post-setup modification of the venv. Subclasses may install
        additional packages or scripts here, add activation shell scripts, etc.

        :param context: The information for the environment creation request
                        being processed.
        """
        pass

    def replace_variables(self, text, context):
        """
        Replace variable placeholders in script text with context-specific
        variables.

        Return the text passed in , but with variables replaced.

        :param text: The text in which to replace placeholder variables.
        :param context: The information for the environment creation request
                        being processed.
        """
        replacements = {
            '__VENV_DIR__': context.env_dir,
            '__VENV_NAME__': context.env_name,
            '__VENV_PROMPT__': context.prompt,
            '__VENV_BIN_NAME__': context.bin_name,
            '__VENV_PYTHON__': context.env_exe,
        }

        def quote_ps1(s):
            """
            This should satisfy PowerShell quoting rules [1], unless the quoted
            string is passed directly to Windows native commands [2].
            [1]: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_quoting_rules
            [2]: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing#passing-arguments-that-contain-quote-characters
            """
            s = s.replace("'", "''")
            return f"'{s}'"

        def quote_bat(s):
            return s

        # gh-124651: need to quote the template strings properly
        quote = shlex.quote
        script_path = context.script_path
        if script_path.endswith('.ps1'):
            quote = quote_ps1
        elif script_path.endswith('.bat'):
            quote = quote_bat
        else:
            # fallbacks to POSIX shell compliant quote
            quote = shlex.quote

        replacements = {key: quote(s) for key, s in replacements.items()}
        for key, quoted in replacements.items():
            text = text.replace(key, quoted)
        return text

    def install_scripts(self, context, path):
        """
        Install scripts into the created environment from a directory.

        :param context: The information for the environment creation request
                        being processed.
        :param path:    Absolute pathname of a directory containing script.
                        Scripts in the 'common' subdirectory of this directory,
                        and those in the directory named for the platform
                        being run on, are installed in the created environment.
                        Placeholder variables are replaced with environment-
                        specific values.
        """
        binpath = context.bin_path
        plen = len(path)
        for root, dirs, files in os.walk(path):
            if root == path: # at top-level, remove irrelevant dirs
                for d in dirs[:]:
                    if d not in ('common', os.name):
                        dirs.remove(d)
                continue # ignore files in top level
            for f in files:
                if (os.name == 'nt' and f.startswith('python')
                        and f.endswith(('.exe', '.pdb'))):
                    continue
                srcfile = os.path.join(root, f)
                suffix = root[plen:].split(os.sep)[2:]
                if not suffix:
                    dstdir = binpath
                else:
                    dstdir = os.path.join(binpath, *suffix)
                if not os.path.exists(dstdir):
                    os.makedirs(dstdir)
                dstfile = os.path.join(dstdir, f)
                with open(srcfile, 'rb') as f:
                    data = f.read()
                if not srcfile.endswith(('.exe', '.pdb')):
                    context.script_path = srcfile
                    try:
                        data = data.decode('utf-8')
                        data = self.replace_variables(data, context)
                        data = data.encode('utf-8')
                    except UnicodeError as e:
                        data = None
                        logger.warning('unable to copy script %r, '
                                       'may be binary: %s', srcfile, e)
                if data is not None:
                    with open(dstfile, 'wb') as f:
                        f.write(data)
                    shutil.copymode(srcfile, dstfile)

    def upgrade_dependencies(self, context):
        logger.debug(
            f'Upgrading {CORE_VENV_DEPS} packages in {context.bin_path}'
        )
        self._call_new_python(context, '-m', 'pip', 'install', '--upgrade',
                              *CORE_VENV_DEPS)


def create(env_dir, system_site_packages=False, clear=False,
           symlinks=False, with_pip=False, prompt=None, upgrade_deps=False):
    """Create a virtual environment in a directory."""
    builder = EnvBuilder(system_site_packages=system_site_packages,
                         clear=clear, symlinks=symlinks, with_pip=with_pip,
                         prompt=prompt, upgrade_deps=upgrade_deps)
    builder.create(env_dir)

def main(args=None):
    compatible = True
    if sys.version_info < (3, 3):
        compatible = False
    elif not hasattr(sys, 'base_prefix'):
        compatible = False
    if not compatible:
        raise ValueError('This script is only for use with Python >= 3.3')
    else:
        import argparse

        parser = argparse.ArgumentParser(prog=__name__,
                                         description='Creates virtual Python '
                                                     'environments in one or '
                                                     'more target '
                                                     'directories.',
                                         epilog='Once an environment has been '
                                                'created, you may wish to '
                                                'activate it, e.g. by '
                                                'sourcing an activate script '
                                                'in its bin directory.')
        parser.add_argument('dirs', metavar='ENV_DIR', nargs='+',
                            help='A directory to create the environment in.')
        parser.add_argument('--system-site-packages', default=False,
                            action='store_true', dest='system_site',
                            help='Give the virtual environment access to the '
                                 'system site-packages dir.')
        if os.name == 'nt':
            use_symlinks = False
        else:
            use_symlinks = True
        group = parser.add_mutually_exclusive_group()
