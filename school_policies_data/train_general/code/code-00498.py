            cls._netmask_cache[arg] = netmask, prefixlen
        return cls._netmask_cache[arg]

    @classmethod
    def _ip_int_from_string(cls, ip_str):
        """Turn the given IP string into an integer for comparison.

        Args:
            ip_str: A string, the IP ip_str.

        Returns:
            The IP ip_str as an integer.

        Raises:
            AddressValueError: if ip_str isn't a valid IPv4 Address.

        """
        if not ip_str:
            raise AddressValueError('Address cannot be empty')

        octets = ip_str.split('.')
        if len(octets) != 4:
            raise AddressValueError("Expected 4 octets in %r" % ip_str)

        try:
            return int.from_bytes(map(cls._parse_octet, octets), 'big')
        except ValueError as exc:
            raise AddressValueError("%s in %r" % (exc, ip_str)) from None

    @classmethod
    def _parse_octet(cls, octet_str):
        """Convert a decimal octet into an integer.

        Args:
            octet_str: A string, the number to parse.

        Returns:
            The octet as an integer.

        Raises:
            ValueError: if the octet isn't strictly a decimal from [0..255].

        """
        if not octet_str:
            raise ValueError("Empty octet not permitted")
        # Reject non-ASCII digits.
        if not (octet_str.isascii() and octet_str.isdigit()):
            msg = "Only decimal digits permitted in %r"
            raise ValueError(msg % octet_str)
        # We do the length check second, since the invalid character error
        # is likely to be more informative for the user
        if len(octet_str) > 3:
            msg = "At most 3 characters permitted in %r"
            raise ValueError(msg % octet_str)
        # Handle leading zeros as strict as glibc's inet_pton()
        # See security bug bpo-36384
        if octet_str != '0' and octet_str[0] == '0':
            msg = "Leading zeros are not permitted in %r"
            raise ValueError(msg % octet_str)
        # Convert to integer (we know digits are legal)
        octet_int = int(octet_str, 10)
        if octet_int > 255:
            raise ValueError("Octet %d (> 255) not permitted" % octet_int)
        return octet_int

    @classmethod
    def _string_from_ip_int(cls, ip_int):
        """Turns a 32-bit integer into dotted decimal notation.

        Args:
            ip_int: An integer, the IP address.

        Returns:
            The IP address as a string in dotted decimal notation.

        """
        return '.'.join(map(str, ip_int.to_bytes(4, 'big')))

    def _reverse_pointer(self):
        """Return the reverse DNS pointer name for the IPv4 address.

        This implements the method described in RFC1035 3.5.

        """
        reverse_octets = str(self).split('.')[::-1]
        return '.'.join(reverse_octets) + '.in-addr.arpa'

    @property
    def max_prefixlen(self):
        return self._max_prefixlen

    @property
    def version(self):
        return self._version


class IPv4Address(_BaseV4, _BaseAddress):

    """Represent and manipulate single IPv4 Addresses."""

    __slots__ = ('_ip', '__weakref__')

    def __init__(self, address):

        """
        Args:
            address: A string or integer representing the IP

              Additionally, an integer can be passed, so
              IPv4Address('192.0.2.1') == IPv4Address(3221225985).
              or, more generally
              IPv4Address(int(IPv4Address('192.0.2.1'))) ==
                IPv4Address('192.0.2.1')

        Raises:
            AddressValueError: If ipaddress isn't a valid IPv4 address.

        """
        # Efficient constructor from integer.
        if isinstance(address, int):
            self._check_int_address(address)
            self._ip = address
            return

        # Constructing from a packed address
        if isinstance(address, bytes):
            self._check_packed_address(address, 4)
            self._ip = int.from_bytes(address, 'big')
            return

        # Assume input argument to be string or any object representation
        # which converts into a formatted IP string.
        addr_str = str(address)
        if '/' in addr_str:
            raise AddressValueError(f"Unexpected '/' in {address!r}")
        self._ip = self._ip_int_from_string(addr_str)

    @property
    def packed(self):
        """The binary representation of this address."""
        return v4_int_to_packed(self._ip)

    @property
    def is_reserved(self):
        """Test if the address is otherwise IETF reserved.

         Returns:
             A boolean, True if the address is within the
             reserved IPv4 Network range.

        """
        return self in self._constants._reserved_network

    @property
    @functools.lru_cache()
    def is_private(self):
        """``True`` if the address is defined as not globally reachable by
        iana-ipv4-special-registry_ (for IPv4) or iana-ipv6-special-registry_
        (for IPv6) with the following exceptions:

        * ``is_private`` is ``False`` for ``100.64.0.0/10``
        * For IPv4-mapped IPv6-addresses the ``is_private`` value is determined by the
            semantics of the underlying IPv4 addresses and the following condition holds
            (see :attr:`IPv6Address.ipv4_mapped`)::

                address.is_private == address.ipv4_mapped.is_private

        ``is_private`` has value opposite to :attr:`is_global`, except for the ``100.64.0.0/10``
        IPv4 range where they are both ``False``.
        """
        return (
            any(self in net for net in self._constants._private_networks)
            and all(self not in net for net in self._constants._private_networks_exceptions)
        )

    @property
    @functools.lru_cache()
    def is_global(self):
        """``True`` if the address is defined as globally reachable by
        iana-ipv4-special-registry_ (for IPv4) or iana-ipv6-special-registry_
        (for IPv6) with the following exception:

        For IPv4-mapped IPv6-addresses the ``is_private`` value is determined by the
        semantics of the underlying IPv4 addresses and the following condition holds
        (see :attr:`IPv6Address.ipv4_mapped`)::

            address.is_global == address.ipv4_mapped.is_global

        ``is_global`` has value opposite to :attr:`is_private`, except for the ``100.64.0.0/10``
        IPv4 range where they are both ``False``.
        """
        return self not in self._constants._public_network and not self.is_private

    @property
    def is_multicast(self):
        """Test if the address is reserved for multicast use.

        Returns:
            A boolean, True if the address is multicast.
            See RFC 3171 for details.

        """
        return self in self._constants._multicast_network

    @property
    def is_unspecified(self):
        """Test if the address is unspecified.

        Returns:
            A boolean, True if this is the unspecified address as defined in
            RFC 5735 3.

        """
        return self == self._constants._unspecified_address

    @property
    def is_loopback(self):
        """Test if the address is a loopback address.

        Returns:
            A boolean, True if the address is a loopback per RFC 3330.

        """
        return self in self._constants._loopback_network

    @property
    def is_link_local(self):
        """Test if the address is reserved for link-local.

        Returns:
            A boolean, True if the address is link-local per RFC 3927.

        """
        return self in self._constants._linklocal_network


class IPv4Interface(IPv4Address):

    def __init__(self, address):
        addr, mask = self._split_addr_prefix(address)

        IPv4Address.__init__(self, addr)
        self.network = IPv4Network((addr, mask), strict=False)
        self.netmask = self.network.netmask
        self._prefixlen = self.network._prefixlen

    @functools.cached_property
    def hostmask(self):
        return self.network.hostmask

    def __str__(self):
        return '%s/%d' % (self._string_from_ip_int(self._ip),
                          self._prefixlen)

    def __eq__(self, other):
        address_equal = IPv4Address.__eq__(self, other)
        if address_equal is NotImplemented or not address_equal:
            return address_equal
        try:
            return self.network == other.network
        except AttributeError:
            # An interface with an associated network is NOT the
            # same as an unassociated address. That's why the hash
            # takes the extra info into account.
            return False

    def __lt__(self, other):
        address_less = IPv4Address.__lt__(self, other)
        if address_less is NotImplemented:
            return NotImplemented
        try:
            return (self.network < other.network or
                    self.network == other.network and address_less)
        except AttributeError:
            # We *do* allow addresses and interfaces to be sorted. The
            # unassociated address is considered less than all interfaces.
            return False

    def __hash__(self):
        return hash((self._ip, self._prefixlen, int(self.network.network_address)))

    __reduce__ = _IPAddressBase.__reduce__

    @property
    def ip(self):
        return IPv4Address(self._ip)

    @property
    def with_prefixlen(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip),
                          self._prefixlen)

    @property
    def with_netmask(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip),
                          self.netmask)

    @property
    def with_hostmask(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip),
                          self.hostmask)


class IPv4Network(_BaseV4, _BaseNetwork):

    """This class represents and manipulates 32-bit IPv4 network + addresses..

    Attributes: [examples for IPv4Network('192.0.2.0/27')]
        .network_address: IPv4Address('192.0.2.0')
        .hostmask: IPv4Address('0.0.0.31')
        .broadcast_address: IPv4Address('192.0.2.32')
        .netmask: IPv4Address('255.255.255.224')
        .prefixlen: 27

    """
    # Class to use when creating address objects
    _address_class = IPv4Address

    def __init__(self, address, strict=True):
        """Instantiate a new IPv4 network object.

        Args:
            address: A string or integer representing the IP [& network].
              '192.0.2.0/24'
              '192.0.2.0/255.255.255.0'
              '192.0.2.0/0.0.0.255'
              are all functionally the same in IPv4. Similarly,
              '192.0.2.1'
              '192.0.2.1/255.255.255.255'
              '192.0.2.1/32'
              are also functionally equivalent. That is to say, failing to
              provide a subnetmask will create an object with a mask of /32.

              If the mask (portion after the / in the argument) is given in
              dotted quad form, it is treated as a netmask if it starts with a
              non-zero field (e.g. /255.0.0.0 == /8) and as a hostmask if it
              starts with a zero field (e.g. 0.255.255.255 == /8), with the
              single exception of an all-zero mask which is treated as a
              netmask == /0. If no mask is given, a default of /32 is used.

              Additionally, an integer can be passed, so
              IPv4Network('192.0.2.1') == IPv4Network(3221225985)
              or, more generally
              IPv4Interface(int(IPv4Interface('192.0.2.1'))) ==
                IPv4Interface('192.0.2.1')

        Raises:
            AddressValueError: If ipaddress isn't a valid IPv4 address.
            NetmaskValueError: If the netmask isn't valid for
              an IPv4 address.
            ValueError: If strict is True and a network address is not
              supplied.
        """
        addr, mask = self._split_addr_prefix(address)

        self.network_address = IPv4Address(addr)
        self.netmask, self._prefixlen = self._make_netmask(mask)
        packed = int(self.network_address)
        if packed & int(self.netmask) != packed:
            if strict:
                raise ValueError('%s has host bits set' % self)
            else:
                self.network_address = IPv4Address(packed &
                                                   int(self.netmask))

        if self._prefixlen == (self._max_prefixlen - 1):
            self.hosts = self.__iter__
        elif self._prefixlen == (self._max_prefixlen):
            self.hosts = lambda: [IPv4Address(addr)]

    @property
    @functools.lru_cache()
    def is_global(self):
        """Test if this address is allocated for public networks.

        Returns:
            A boolean, True if the address is not reserved per
            iana-ipv4-special-registry.

        """
        return (not (self.network_address in IPv4Network('100.64.0.0/10') and
                    self.broadcast_address in IPv4Network('100.64.0.0/10')) and
                not self.is_private)


class _IPv4Constants:
    _linklocal_network = IPv4Network('169.254.0.0/16')

    _loopback_network = IPv4Network('127.0.0.0/8')

    _multicast_network = IPv4Network('224.0.0.0/4')

    _public_network = IPv4Network('100.64.0.0/10')

    # Not globally reachable address blocks listed on
    # https://www.iana.org/assignments/iana-ipv4-special-registry/iana-ipv4-special-registry.xhtml
    _private_networks = [
        IPv4Network('0.0.0.0/8'),
        IPv4Network('10.0.0.0/8'),
        IPv4Network('127.0.0.0/8'),
        IPv4Network('169.254.0.0/16'),
        IPv4Network('172.16.0.0/12'),
        IPv4Network('192.0.0.0/24'),
        IPv4Network('192.0.0.170/31'),
        IPv4Network('192.0.2.0/24'),
        IPv4Network('192.168.0.0/16'),
        IPv4Network('198.18.0.0/15'),
        IPv4Network('198.51.100.0/24'),
        IPv4Network('203.0.113.0/24'),
        IPv4Network('240.0.0.0/4'),
        IPv4Network('255.255.255.255/32'),
        ]

    _private_networks_exceptions = [
        IPv4Network('192.0.0.9/32'),
        IPv4Network('192.0.0.10/32'),
    ]

    _reserved_network = IPv4Network('240.0.0.0/4')

    _unspecified_address = IPv4Address('0.0.0.0')


IPv4Address._constants = _IPv4Constants


class _BaseV6:

    """Base IPv6 object.

    The following methods are used by IPv6 objects in both single IP
    addresses and networks.

    """

    __slots__ = ()
    _version = 6
    _ALL_ONES = (2**IPV6LENGTH) - 1
    _HEXTET_COUNT = 8
    _HEX_DIGITS = frozenset('0123456789ABCDEFabcdef')
    _max_prefixlen = IPV6LENGTH

    # There are only a bunch of valid v6 netmasks, so we cache them all
    # when constructed (see _make_netmask()).
    _netmask_cache = {}

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
                prefixlen = cls._prefix_from_prefix_string(arg)
            netmask = IPv6Address(cls._ip_int_from_prefix(prefixlen))
            cls._netmask_cache[arg] = netmask, prefixlen
        return cls._netmask_cache[arg]

    @classmethod
    def _ip_int_from_string(cls, ip_str):
        """Turn an IPv6 ip_str into an integer.

        Args:
            ip_str: A string, the IPv6 ip_str.

        Returns:
            An int, the IPv6 address

        Raises:
            AddressValueError: if ip_str isn't a valid IPv6 Address.

        """
        if not ip_str:
            raise AddressValueError('Address cannot be empty')

        parts = ip_str.split(':')

        # An IPv6 address needs at least 2 colons (3 parts).
        _min_parts = 3
        if len(parts) < _min_parts:
            msg = "At least %d parts expected in %r" % (_min_parts, ip_str)
            raise AddressValueError(msg)

        # If the address has an IPv4-style suffix, convert it to hexadecimal.
        if '.' in parts[-1]:
            try:
                ipv4_int = IPv4Address(parts.pop())._ip
            except AddressValueError as exc:
                raise AddressValueError("%s in %r" % (exc, ip_str)) from None
            parts.append('%x' % ((ipv4_int >> 16) & 0xFFFF))
            parts.append('%x' % (ipv4_int & 0xFFFF))

        # An IPv6 address can't have more than 8 colons (9 parts).
        # The extra colon comes from using the "::" notation for a single
        # leading or trailing zero part.
        _max_parts = cls._HEXTET_COUNT + 1
        if len(parts) > _max_parts:
            msg = "At most %d colons permitted in %r" % (_max_parts-1, ip_str)
            raise AddressValueError(msg)

        # Disregarding the endpoints, find '::' with nothing in between.
        # This indicates that a run of zeroes has been skipped.
        skip_index = None
        for i in range(1, len(parts) - 1):
            if not parts[i]:
                if skip_index is not None:
                    # Can't have more than one '::'
                    msg = "At most one '::' permitted in %r" % ip_str
                    raise AddressValueError(msg)
                skip_index = i

        # parts_hi is the number of parts to copy from above/before the '::'
        # parts_lo is the number of parts to copy from below/after the '::'
        if skip_index is not None:
            # If we found a '::', then check if it also covers the endpoints.
            parts_hi = skip_index
            parts_lo = len(parts) - skip_index - 1
            if not parts[0]:
                parts_hi -= 1
                if parts_hi:
                    msg = "Leading ':' only permitted as part of '::' in %r"
                    raise AddressValueError(msg % ip_str)  # ^: requires ^::
            if not parts[-1]:
                parts_lo -= 1
                if parts_lo:
                    msg = "Trailing ':' only permitted as part of '::' in %r"
                    raise AddressValueError(msg % ip_str)  # :$ requires ::$
            parts_skipped = cls._HEXTET_COUNT - (parts_hi + parts_lo)
            if parts_skipped < 1:
                msg = "Expected at most %d other parts with '::' in %r"
                raise AddressValueError(msg % (cls._HEXTET_COUNT-1, ip_str))
        else:
            # Otherwise, allocate the entire address to parts_hi.  The
            # endpoints could still be empty, but _parse_hextet() will check
            # for that.
            if len(parts) != cls._HEXTET_COUNT:
                msg = "Exactly %d parts expected without '::' in %r"
                raise AddressValueError(msg % (cls._HEXTET_COUNT, ip_str))
            if not parts[0]:
                msg = "Leading ':' only permitted as part of '::' in %r"
                raise AddressValueError(msg % ip_str)  # ^: requires ^::
            if not parts[-1]:
                msg = "Trailing ':' only permitted as part of '::' in %r"
                raise AddressValueError(msg % ip_str)  # :$ requires ::$
            parts_hi = len(parts)
            parts_lo = 0
            parts_skipped = 0

        try:
            # Now, parse the hextets into a 128-bit integer.
            ip_int = 0
            for i in range(parts_hi):
                ip_int <<= 16
                ip_int |= cls._parse_hextet(parts[i])
            ip_int <<= 16 * parts_skipped
            for i in range(-parts_lo, 0):
                ip_int <<= 16
                ip_int |= cls._parse_hextet(parts[i])
            return ip_int
        except ValueError as exc:
            raise AddressValueError("%s in %r" % (exc, ip_str)) from None

    @classmethod
    def _parse_hextet(cls, hextet_str):
        """Convert an IPv6 hextet string into an integer.

        Args:
            hextet_str: A string, the number to parse.

        Returns:
            The hextet as an integer.

        Raises:
            ValueError: if the input isn't strictly a hex number from
              [0..FFFF].

        """
        # Reject non-ASCII digits.
        if not cls._HEX_DIGITS.issuperset(hextet_str):
            raise ValueError("Only hex digits permitted in %r" % hextet_str)
        # We do the length check second, since the invalid character error
        # is likely to be more informative for the user
        if len(hextet_str) > 4:
            msg = "At most 4 characters permitted in %r"
            raise ValueError(msg % hextet_str)
        # Length check means we can skip checking the integer value
        return int(hextet_str, 16)

    @classmethod
    def _compress_hextets(cls, hextets):
        """Compresses a list of hextets.

        Compresses a list of strings, replacing the longest continuous
        sequence of "0" in the list with "" and adding empty strings at
        the beginning or at the end of the string such that subsequently
        calling ":".join(hextets) will produce the compressed version of
        the IPv6 address.

        Args:
            hextets: A list of strings, the hextets to compress.

        Returns:
            A list of strings.

        """
        best_doublecolon_start = -1
        best_doublecolon_len = 0
        doublecolon_start = -1
        doublecolon_len = 0
        for index, hextet in enumerate(hextets):
            if hextet == '0':
                doublecolon_len += 1
                if doublecolon_start == -1:
                    # Start of a sequence of zeros.
                    doublecolon_start = index
                if doublecolon_len > best_doublecolon_len:
                    # This is the longest sequence of zeros so far.
                    best_doublecolon_len = doublecolon_len
                    best_doublecolon_start = doublecolon_start
            else:
                doublecolon_len = 0
                doublecolon_start = -1

        if best_doublecolon_len > 1:
            best_doublecolon_end = (best_doublecolon_start +
                                    best_doublecolon_len)
            # For zeros at the end of the address.
            if best_doublecolon_end == len(hextets):
                hextets += ['']
            hextets[best_doublecolon_start:best_doublecolon_end] = ['']
            # For zeros at the beginning of the address.
            if best_doublecolon_start == 0:
                hextets = [''] + hextets

        return hextets

    @classmethod
    def _string_from_ip_int(cls, ip_int=None):
        """Turns a 128-bit integer into hexadecimal notation.

        Args:
            ip_int: An integer, the IP address.

        Returns:
            A string, the hexadecimal representation of the address.

        Raises:
            ValueError: The address is bigger than 128 bits of all ones.

        """
        if ip_int is None:
            ip_int = int(cls._ip)

        if ip_int > cls._ALL_ONES:
            raise ValueError('IPv6 address is too large')

        hex_str = '%032x' % ip_int
        hextets = ['%x' % int(hex_str[x:x+4], 16) for x in range(0, 32, 4)]

        hextets = cls._compress_hextets(hextets)
        return ':'.join(hextets)

    def _explode_shorthand_ip_string(self):
        """Expand a shortened IPv6 address.

        Args:
            ip_str: A string, the IPv6 address.

        Returns:
            A string, the expanded IPv6 address.

        """
        if isinstance(self, IPv6Network):
            ip_str = str(self.network_address)
        elif isinstance(self, IPv6Interface):
            ip_str = str(self.ip)
        else:
            ip_str = str(self)

        ip_int = self._ip_int_from_string(ip_str)
        hex_str = '%032x' % ip_int
        parts = [hex_str[x:x+4] for x in range(0, 32, 4)]
        if isinstance(self, (_BaseNetwork, IPv6Interface)):
            return '%s/%d' % (':'.join(parts), self._prefixlen)
        return ':'.join(parts)

    def _reverse_pointer(self):
        """Return the reverse DNS pointer name for the IPv6 address.

        This implements the method described in RFC3596 2.5.

        """
        reverse_chars = self.exploded[::-1].replace(':', '')
        return '.'.join(reverse_chars) + '.ip6.arpa'

    @staticmethod
    def _split_scope_id(ip_str):
        """Helper function to parse IPv6 string address with scope id.

        See RFC 4007 for details.

        Args:
            ip_str: A string, the IPv6 address.

        Returns:
            (addr, scope_id) tuple.

        """
        addr, sep, scope_id = ip_str.partition('%')
        if not sep:
            scope_id = None
        elif not scope_id or '%' in scope_id:
            raise AddressValueError('Invalid IPv6 address: "%r"' % ip_str)
        return addr, scope_id

    @property
    def max_prefixlen(self):
        return self._max_prefixlen

    @property
    def version(self):
        return self._version


class IPv6Address(_BaseV6, _BaseAddress):

    """Represent and manipulate single IPv6 Addresses."""

    __slots__ = ('_ip', '_scope_id', '__weakref__')

    def __init__(self, address):
        """Instantiate a new IPv6 address object.

        Args:
            address: A string or integer representing the IP

              Additionally, an integer can be passed, so
              IPv6Address('2001:db8::') ==
                IPv6Address(42540766411282592856903984951653826560)
              or, more generally
              IPv6Address(int(IPv6Address('2001:db8::'))) ==
                IPv6Address('2001:db8::')

        Raises:
            AddressValueError: If address isn't a valid IPv6 address.

        """
        # Efficient constructor from integer.
        if isinstance(address, int):
            self._check_int_address(address)
            self._ip = address
            self._scope_id = None
            return

        # Constructing from a packed address
        if isinstance(address, bytes):
            self._check_packed_address(address, 16)
            self._ip = int.from_bytes(address, 'big')
            self._scope_id = None
            return

        # Assume input argument to be string or any object representation
        # which converts into a formatted IP string.
        addr_str = str(address)
        if '/' in addr_str:
            raise AddressValueError(f"Unexpected '/' in {address!r}")
        addr_str, self._scope_id = self._split_scope_id(addr_str)

        self._ip = self._ip_int_from_string(addr_str)

    def __str__(self):
        ip_str = super().__str__()
        return ip_str + '%' + self._scope_id if self._scope_id else ip_str

    def __hash__(self):
        return hash((self._ip, self._scope_id))

    def __eq__(self, other):
        address_equal = super().__eq__(other)
        if address_equal is NotImplemented:
            return NotImplemented
        if not address_equal:
            return False
        return self._scope_id == getattr(other, '_scope_id', None)

    @property
    def scope_id(self):
        """Identifier of a particular zone of the address's scope.

        See RFC 4007 for details.

        Returns:
            A string identifying the zone of the address if specified, else None.

        """
        return self._scope_id

    @property
    def packed(self):
        """The binary representation of this address."""
        return v6_int_to_packed(self._ip)

    @property
    def is_multicast(self):
        """Test if the address is reserved for multicast use.

        Returns:
            A boolean, True if the address is a multicast address.
            See RFC 2373 2.7 for details.

        """
        return self in self._constants._multicast_network

    @property
    def is_reserved(self):
        """Test if the address is otherwise IETF reserved.

        Returns:
            A boolean, True if the address is within one of the
            reserved IPv6 Network ranges.

        """
        return any(self in x for x in self._constants._reserved_networks)

    @property
    def is_link_local(self):
        """Test if the address is reserved for link-local.

        Returns:
            A boolean, True if the address is reserved per RFC 4291.

        """
        return self in self._constants._linklocal_network

    @property
    def is_site_local(self):
        """Test if the address is reserved for site-local.

        Note that the site-local address space has been deprecated by RFC 3879.
        Use is_private to test if this address is in the space of unique local
        addresses as defined by RFC 4193.

        Returns:
            A boolean, True if the address is reserved per RFC 3513 2.5.6.

        """
        return self in self._constants._sitelocal_network

    @property
    @functools.lru_cache()
    def is_private(self):
        """``True`` if the address is defined as not globally reachable by
        iana-ipv4-special-registry_ (for IPv4) or iana-ipv6-special-registry_
        (for IPv6) with the following exceptions:

        * ``is_private`` is ``False`` for ``100.64.0.0/10``
        * For IPv4-mapped IPv6-addresses the ``is_private`` value is determined by the
            semantics of the underlying IPv4 addresses and the following condition holds
            (see :attr:`IPv6Address.ipv4_mapped`)::

                address.is_private == address.ipv4_mapped.is_private

        ``is_private`` has value opposite to :attr:`is_global`, except for the ``100.64.0.0/10``
        IPv4 range where they are both ``False``.
        """
        ipv4_mapped = self.ipv4_mapped
        if ipv4_mapped is not None:
            return ipv4_mapped.is_private
        return (
            any(self in net for net in self._constants._private_networks)
            and all(self not in net for net in self._constants._private_networks_exceptions)
        )

    @property
    def is_global(self):
        """``True`` if the address is defined as globally reachable by
        iana-ipv4-special-registry_ (for IPv4) or iana-ipv6-special-registry_
        (for IPv6) with the following exception:

        For IPv4-mapped IPv6-addresses the ``is_private`` value is determined by the
        semantics of the underlying IPv4 addresses and the following condition holds
        (see :attr:`IPv6Address.ipv4_mapped`)::

            address.is_global == address.ipv4_mapped.is_global

        ``is_global`` has value opposite to :attr:`is_private`, except for the ``100.64.0.0/10``
        IPv4 range where they are both ``False``.
        """
        return not self.is_private

    @property
    def is_unspecified(self):
        """Test if the address is unspecified.

        Returns:
            A boolean, True if this is the unspecified address as defined in
            RFC 2373 2.5.2.

        """
        return self._ip == 0

    @property
    def is_loopback(self):
        """Test if the address is a loopback address.

        Returns:
            A boolean, True if the address is a loopback address as defined in
            RFC 2373 2.5.3.

        """
        return self._ip == 1

    @property
    def ipv4_mapped(self):
        """Return the IPv4 mapped address.

        Returns:
            If the IPv6 address is a v4 mapped address, return the
            IPv4 mapped address. Return None otherwise.

        """
        if (self._ip >> 32) != 0xFFFF:
            return None
        return IPv4Address(self._ip & 0xFFFFFFFF)

    @property
    def teredo(self):
        """Tuple of embedded teredo IPs.

        Returns:
            Tuple of the (server, client) IPs or None if the address
            doesn't appear to be a teredo address (doesn't start with
            2001::/32)

        """
        if (self._ip >> 96) != 0x20010000:
            return None
        return (IPv4Address((self._ip >> 64) & 0xFFFFFFFF),
                IPv4Address(~self._ip & 0xFFFFFFFF))

    @property
    def sixtofour(self):
        """Return the IPv4 6to4 embedded address.

        Returns:
            The IPv4 6to4-embedded address if present or None if the
            address doesn't appear to contain a 6to4 embedded address.

        """
        if (self._ip >> 112) != 0x2002:
            return None
        return IPv4Address((self._ip >> 80) & 0xFFFFFFFF)


class IPv6Interface(IPv6Address):

    def __init__(self, address):
        addr, mask = self._split_addr_prefix(address)

        IPv6Address.__init__(self, addr)
        self.network = IPv6Network((addr, mask), strict=False)
        self.netmask = self.network.netmask
        self._prefixlen = self.network._prefixlen

    @functools.cached_property
    def hostmask(self):
        return self.network.hostmask

    def __str__(self):
        return '%s/%d' % (super().__str__(),
                          self._prefixlen)

    def __eq__(self, other):
        address_equal = IPv6Address.__eq__(self, other)
        if address_equal is NotImplemented or not address_equal:
            return address_equal
        try:
            return self.network == other.network
        except AttributeError:
            # An interface with an associated network is NOT the
            # same as an unassociated address. That's why the hash
            # takes the extra info into account.
            return False

    def __lt__(self, other):
        address_less = IPv6Address.__lt__(self, other)
        if address_less is NotImplemented:
            return address_less
        try:
            return (self.network < other.network or
                    self.network == other.network and address_less)
        except AttributeError:
            # We *do* allow addresses and interfaces to be sorted. The
            # unassociated address is considered less than all interfaces.
            return False

    def __hash__(self):
        return hash((self._ip, self._prefixlen, int(self.network.network_address)))

    __reduce__ = _IPAddressBase.__reduce__

    @property
    def ip(self):
        return IPv6Address(self._ip)

    @property
    def with_prefixlen(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip),
                          self._prefixlen)

    @property
    def with_netmask(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip),
                          self.netmask)

    @property
    def with_hostmask(self):
        return '%s/%s' % (self._string_from_ip_int(self._ip),
                          self.hostmask)

    @property
    def is_unspecified(self):
        return self._ip == 0 and self.network.is_unspecified

    @property
    def is_loopback(self):
        return self._ip == 1 and self.network.is_loopback


class IPv6Network(_BaseV6, _BaseNetwork):

    """This class represents and manipulates 128-bit IPv6 networks.

    Attributes: [examples for IPv6('2001:db8::1000/124')]
        .network_address: IPv6Address('2001:db8::1000')
        .hostmask: IPv6Address('::f')
        .broadcast_address: IPv6Address('2001:db8::100f')
        .netmask: IPv6Address('ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff0')
        .prefixlen: 124

    """

    # Class to use when creating address objects
    _address_class = IPv6Address

    def __init__(self, address, strict=True):
        """Instantiate a new IPv6 Network object.

        Args:
            address: A string or integer representing the IPv6 network or the
              IP and prefix/netmask.
              '2001:db8::/128'
              '2001:db8:0000:0000:0000:0000:0000:0000/128'
              '2001:db8::'
              are all functionally the same in IPv6.  That is to say,
              failing to provide a subnetmask will create an object with
              a mask of /128.

              Additionally, an integer can be passed, so
              IPv6Network('2001:db8::') ==
                IPv6Network(42540766411282592856903984951653826560)
              or, more generally
              IPv6Network(int(IPv6Network('2001:db8::'))) ==
                IPv6Network('2001:db8::')

            strict: A boolean. If true, ensure that we have been passed
              A true network address, eg, 2001:db8::1000/124 and not an
              IP address on a network, eg, 2001:db8::1/124.

        Raises:
            AddressValueError: If address isn't a valid IPv6 address.
            NetmaskValueError: If the netmask isn't valid for
              an IPv6 address.
            ValueError: If strict was True and a network address was not
              supplied.
        """
        addr, mask = self._split_addr_prefix(address)

        self.network_address = IPv6Address(addr)
        self.netmask, self._prefixlen = self._make_netmask(mask)
        packed = int(self.network_address)
        if packed & int(self.netmask) != packed:
            if strict:
                raise ValueError('%s has host bits set' % self)
            else:
                self.network_address = IPv6Address(packed &
                                                   int(self.netmask))

        if self._prefixlen == (self._max_prefixlen - 1):
            self.hosts = self.__iter__
        elif self._prefixlen == self._max_prefixlen:
            self.hosts = lambda: [IPv6Address(addr)]

    def hosts(self):
        """Generate Iterator over usable hosts in a network.

          This is like __iter__ except it doesn't return the
          Subnet-Router anycast address.

        """
        network = int(self.network_address)
        broadcast = int(self.broadcast_address)
        for x in range(network + 1, broadcast + 1):
            yield self._address_class(x)

    @property
    def is_site_local(self):
        """Test if the address is reserved for site-local.

        Note that the site-local address space has been deprecated by RFC 3879.
        Use is_private to test if this address is in the space of unique local
        addresses as defined by RFC 4193.

        Returns:
            A boolean, True if the address is reserved per RFC 3513 2.5.6.

        """
        return (self.network_address.is_site_local and
                self.broadcast_address.is_site_local)


class _IPv6Constants:

    _linklocal_network = IPv6Network('fe80::/10')

    _multicast_network = IPv6Network('ff00::/8')

    # Not globally reachable address blocks listed on
    # https://www.iana.org/assignments/iana-ipv6-special-registry/iana-ipv6-special-registry.xhtml
    _private_networks = [
        IPv6Network('::1/128'),
        IPv6Network('::/128'),
        IPv6Network('::ffff:0:0/96'),
        IPv6Network('64:ff9b:1::/48'),
        IPv6Network('100::/64'),
        IPv6Network('2001::/23'),
        IPv6Network('2001:db8::/32'),
        # IANA says N/A, let's consider it not globally reachable to be safe
        IPv6Network('2002::/16'),
        IPv6Network('fc00::/7'),
        IPv6Network('fe80::/10'),
        ]

    _private_networks_exceptions = [
        IPv6Network('2001:1::1/128'),
        IPv6Network('2001:1::2/128'),
        IPv6Network('2001:3::/32'),
        IPv6Network('2001:4:112::/48'),
        IPv6Network('2001:20::/28'),
        IPv6Network('2001:30::/28'),
    ]

    _reserved_networks = [
        IPv6Network('::/8'), IPv6Network('100::/8'),
        IPv6Network('200::/7'), IPv6Network('400::/6'),
        IPv6Network('800::/5'), IPv6Network('1000::/4'),
        IPv6Network('4000::/3'), IPv6Network('6000::/3'),
        IPv6Network('8000::/3'), IPv6Network('A000::/3'),
        IPv6Network('C000::/3'), IPv6Network('E000::/4'),
        IPv6Network('F000::/5'), IPv6Network('F800::/6'),
        IPv6Network('FE00::/9'),
    ]

    _sitelocal_network = IPv6Network('fec0::/10')


IPv6Address._constants = _IPv6Constants
<|endoftext|>"""Create portable serialized representations of Python objects.

See module copyreg for a mechanism for registering custom picklers.
See module pickletools source for extensive comments.

Classes:

    Pickler
    Unpickler

Functions:

    dump(object, file)
    dumps(object) -> string
    load(file) -> object
    loads(bytes) -> object

Misc variables:

    __version__
    format_version
    compatible_formats

"""

from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle

__all__ = ["PickleError", "PicklingError", "UnpicklingError", "Pickler",
           "Unpickler", "dump", "dumps", "load", "loads"]

try:
    from _pickle import PickleBuffer
    __all__.append("PickleBuffer")
    _HAVE_PICKLE_BUFFER = True
except ImportError:
    _HAVE_PICKLE_BUFFER = False


# Shortcut for use in isinstance testing
bytes_types = (bytes, bytearray)

# These are purely informational; no code uses these.
format_version = "4.0"                  # File format version we write
compatible_formats = ["1.0",            # Original protocol 0
                      "1.1",            # Protocol 0 with INST added
                      "1.2",            # Original protocol 1
                      "1.3",            # Protocol 1 with BINFLOAT added
                      "2.0",            # Protocol 2
                      "3.0",            # Protocol 3
                      "4.0",            # Protocol 4
                      "5.0",            # Protocol 5
                      ]                 # Old format versions we can read

# This is the highest protocol number we know how to read.
HIGHEST_PROTOCOL = 5

# The protocol we write by default.  May be less than HIGHEST_PROTOCOL.
# Only bump this if the oldest still supported version of Python already
# includes it.
DEFAULT_PROTOCOL = 4

class PickleError(Exception):
    """A common base class for the other pickling exceptions."""
    pass

class PicklingError(PickleError):
    """This exception is raised when an unpicklable object is passed to the
    dump() method.

    """
    pass

class UnpicklingError(PickleError):
    """This exception is raised when there is a problem unpickling an object,
    such as a security violation.

    Note that other exceptions may also be raised during unpickling, including
    (but not necessarily limited to) AttributeError, EOFError, ImportError,
    and IndexError.

    """
    pass

# An instance of _Stop is raised by Unpickler.load_stop() in response to
# the STOP opcode, passing the object that is the result of unpickling.
class _Stop(Exception):
    def __init__(self, value):
        self.value = value

# Jython has PyStringMap; it's a dict subclass with string keys
try:
    from org.python.core import PyStringMap
except ImportError:
    PyStringMap = None

# Pickle opcodes.  See pickletools.py for extensive docs.  The listing
# here is in kind-of alphabetical order of 1-character pickle code.
# pickletools groups them by purpose.

MARK           = b'('   # push special markobject on stack
STOP           = b'.'   # every pickle ends with STOP
POP            = b'0'   # discard topmost stack item
POP_MARK       = b'1'   # discard stack top through topmost markobject
DUP            = b'2'   # duplicate top stack item
FLOAT          = b'F'   # push float object; decimal string argument
INT            = b'I'   # push integer or bool; decimal string argument
BININT         = b'J'   # push four-byte signed int
BININT1        = b'K'   # push 1-byte unsigned int
LONG           = b'L'   # push long; decimal string argument
BININT2        = b'M'   # push 2-byte unsigned int
NONE           = b'N'   # push None
PERSID         = b'P'   # push persistent object; id is taken from string arg
BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
REDUCE         = b'R'   # apply callable to argtuple, both on stack
STRING         = b'S'   # push string; NL-terminated string argument
BINSTRING      = b'T'   # push string; counted binary string argument
SHORT_BINSTRING= b'U'   #  "     "   ;    "      "       "      " < 256 bytes
UNICODE        = b'V'   # push Unicode string; raw-unicode-escaped'd argument
BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
APPEND         = b'a'   # append stack top to list below it
BUILD          = b'b'   # call __setstate__ or __dict__.update()
GLOBAL         = b'c'   # push self.find_class(modname, name); 2 string args
DICT           = b'd'   # build a dict from stack items
EMPTY_DICT     = b'}'   # push empty dict
APPENDS        = b'e'   # extend list on stack by topmost stack slice
GET            = b'g'   # push item from memo on stack; index is string arg
BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
INST           = b'i'   # build & push class instance
LONG_BINGET    = b'j'   # push item from memo on stack; index is 4-byte arg
LIST           = b'l'   # build list from topmost stack items
EMPTY_LIST     = b']'   # push empty list
OBJ            = b'o'   # build & push class instance
PUT            = b'p'   # store stack top in memo; index is string arg
BINPUT         = b'q'   #   "     "    "   "   " ;   "    " 1-byte arg
LONG_BINPUT    = b'r'   #   "     "    "   "   " ;   "    " 4-byte arg
SETITEM        = b's'   # add key+value pair to dict
TUPLE          = b't'   # build tuple from topmost stack items
EMPTY_TUPLE    = b')'   # push empty tuple
SETITEMS       = b'u'   # modify dict by adding topmost key+value pairs
BINFLOAT       = b'G'   # push float; arg is 8-byte float encoding

TRUE           = b'I01\n'  # not an opcode; see INT docs in pickletools.py
FALSE          = b'I00\n'  # not an opcode; see INT docs in pickletools.py

# Protocol 2

PROTO          = b'\x80'  # identify pickle protocol
NEWOBJ         = b'\x81'  # build object by applying cls.__new__ to argtuple
EXT1           = b'\x82'  # push object from extension registry; 1-byte index
EXT2           = b'\x83'  # ditto, but 2-byte index
EXT4           = b'\x84'  # ditto, but 4-byte index
TUPLE1         = b'\x85'  # build 1-tuple from stack top
TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
TUPLE3         = b'\x87'  # build 3-tuple from three topmost stack items
NEWTRUE        = b'\x88'  # push True
NEWFALSE       = b'\x89'  # push False
LONG1          = b'\x8a'  # push long from < 256 bytes
LONG4          = b'\x8b'  # push really big long

_tuplesize2code = [EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3]

# Protocol 3 (Python 3.x)

BINBYTES       = b'B'   # push bytes; counted binary string argument
SHORT_BINBYTES = b'C'   #  "     "   ;    "      "       "      " < 256 bytes

# Protocol 4

SHORT_BINUNICODE = b'\x8c'  # push short string; UTF-8 length < 256 bytes
BINUNICODE8      = b'\x8d'  # push very long string
BINBYTES8        = b'\x8e'  # push very long bytes string
EMPTY_SET        = b'\x8f'  # push empty set on the stack
ADDITEMS         = b'\x90'  # modify set by adding topmost stack items
FROZENSET        = b'\x91'  # build frozenset from topmost stack items
NEWOBJ_EX        = b'\x92'  # like NEWOBJ but work with keyword only arguments
STACK_GLOBAL     = b'\x93'  # same as GLOBAL but using names on the stacks
MEMOIZE          = b'\x94'  # store top of the stack in memo
FRAME            = b'\x95'  # indicate the beginning of a new frame

# Protocol 5

BYTEARRAY8       = b'\x96'  # push bytearray
NEXT_BUFFER      = b'\x97'  # push next out-of-band buffer
READONLY_BUFFER  = b'\x98'  # make top of stack readonly

__all__.extend([x for x in dir() if re.match("[A-Z][A-Z0-9_]+$", x)])


class _Framer:

    _FRAME_SIZE_MIN = 4
    _FRAME_SIZE_TARGET = 64 * 1024

    def __init__(self, file_write):
        self.file_write = file_write
        self.current_frame = None

    def start_framing(self):
        self.current_frame = io.BytesIO()

    def end_framing(self):
        if self.current_frame and self.current_frame.tell() > 0:
            self.commit_frame(force=True)
            self.current_frame = None

    def commit_frame(self, force=False):
        if self.current_frame:
            f = self.current_frame
            if f.tell() >= self._FRAME_SIZE_TARGET or force:
                data = f.getbuffer()
                write = self.file_write
                if len(data) >= self._FRAME_SIZE_MIN:
                    # Issue a single call to the write method of the underlying
                    # file object for the frame opcode with the size of the
                    # frame. The concatenation is expected to be less expensive
                    # than issuing an additional call to write.
                    write(FRAME + pack("<Q", len(data)))

                # Issue a separate call to write to append the frame
                # contents without concatenation to the above to avoid a
                # memory copy.
                write(data)

                # Start the new frame with a new io.BytesIO instance so that
                # the file object can have delayed access to the previous frame
                # contents via an unreleased memoryview of the previous
                # io.BytesIO instance.
                self.current_frame = io.BytesIO()

    def write(self, data):
        if self.current_frame:
            return self.current_frame.write(data)
        else:
            return self.file_write(data)

    def write_large_bytes(self, header, payload):
        write = self.file_write
        if self.current_frame:
            # Terminate the current frame and flush it to the file.
            self.commit_frame(force=True)

        # Perform direct write of the header and payload of the large binary
        # object. Be careful not to concatenate the header and the payload
        # prior to calling 'write' as we do not want to allocate a large
        # temporary bytes object.
        # We intentionally do not insert a protocol 4 frame opcode to make
        # it possible to optimize file.read calls in the loader.
        write(header)
        write(payload)


class _Unframer:

    def __init__(self, file_read, file_readline, file_tell=None):
        self.file_read = file_read
        self.file_readline = file_readline
        self.current_frame = None

    def readinto(self, buf):
        if self.current_frame:
            n = self.current_frame.readinto(buf)
            if n == 0 and len(buf) != 0:
                self.current_frame = None
                n = len(buf)
                buf[:] = self.file_read(n)
                return n
            if n < len(buf):
                raise UnpicklingError(
                    "pickle exhausted before end of frame")
            return n
        else:
            n = len(buf)
            buf[:] = self.file_read(n)
            return n

    def read(self, n):
        if self.current_frame:
            data = self.current_frame.read(n)
            if not data and n != 0:
                self.current_frame = None
                return self.file_read(n)
            if len(data) < n:
                raise UnpicklingError(
                    "pickle exhausted before end of frame")
            return data
        else:
            return self.file_read(n)

    def readline(self):
        if self.current_frame:
            data = self.current_frame.readline()
            if not data:
                self.current_frame = None
                return self.file_readline()
            if data[-1] != b'\n'[0]:
                raise UnpicklingError(
                    "pickle exhausted before end of frame")
            return data
        else:
            return self.file_readline()

    def load_frame(self, frame_size):
        if self.current_frame and self.current_frame.read() != b'':
            raise UnpicklingError(
                "beginning of a new frame before end of current frame")
        self.current_frame = io.BytesIO(self.file_read(frame_size))


# Tools used for pickling.

def _getattribute(obj, name):
    for subpath in name.split('.'):
        if subpath == '<locals>':
            raise AttributeError("Can't get local attribute {!r} on {!r}"
                                 .format(name, obj))
        try:
            parent = obj
            obj = getattr(obj, subpath)
        except AttributeError:
            raise AttributeError("Can't get attribute {!r} on {!r}"
                                 .format(name, obj)) from None
    return obj, parent

def whichmodule(obj, name):
    """Find the module an object belong to."""
    module_name = getattr(obj, '__module__', None)
    if module_name is not None:
        return module_name
    # Protect the iteration by using a list copy of sys.modules against dynamic
    # modules that trigger imports of other modules upon calls to getattr.
    for module_name, module in sys.modules.copy().items():
        if (module_name == '__main__'
            or module_name == '__mp_main__'  # bpo-42406
            or module is None):
            continue
        try:
            if _getattribute(module, name)[0] is obj:
                return module_name
        except AttributeError:
            pass
    return '__main__'

def encode_long(x):
    r"""Encode a long to a two's complement little-endian binary string.
    Note that 0 is a special case, returning an empty string, to save a
    byte in the LONG1 pickling context.

    >>> encode_long(0)
    b''
    >>> encode_long(255)
    b'\xff\x00'
    >>> encode_long(32767)
    b'\xff\x7f'
    >>> encode_long(-256)
    b'\x00\xff'
    >>> encode_long(-32768)
    b'\x00\x80'
    >>> encode_long(-128)
    b'\x80'
    >>> encode_long(127)
    b'\x7f'
    >>>
    """
    if x == 0:
        return b''
    nbytes = (x.bit_length() >> 3) + 1
    result = x.to_bytes(nbytes, byteorder='little', signed=True)
    if x < 0 and nbytes > 1:
        if result[-1] == 0xff and (result[-2] & 0x80) != 0:
            result = result[:-1]
    return result

def decode_long(data):
    r"""Decode a long from a two's complement little-endian binary string.

    >>> decode_long(b'')
    0
    >>> decode_long(b"\xff\x00")
    255
    >>> decode_long(b"\xff\x7f")
    32767
    >>> decode_long(b"\x00\xff")
    -256
    >>> decode_long(b"\x00\x80")
    -32768
    >>> decode_long(b"\x80")
    -128
    >>> decode_long(b"\x7f")
    127
    """
    return int.from_bytes(data, byteorder='little', signed=True)


# Pickling machinery

class _Pickler:

    def __init__(self, file, protocol=None, *, fix_imports=True,
                 buffer_callback=None):
        """This takes a binary file for writing a pickle data stream.

        The optional *protocol* argument tells the pickler to use the
        given protocol; supported protocols are 0, 1, 2, 3, 4 and 5.
        The default protocol is 4. It was introduced in Python 3.4, and
        is incompatible with previous versions.

        Specifying a negative protocol version selects the highest
        protocol version supported.  The higher the protocol used, the
        more recent the version of Python needed to read the pickle
        produced.

        The *file* argument must have a write() method that accepts a
        single bytes argument. It can thus be a file object opened for
        binary writing, an io.BytesIO instance, or any other custom
        object that meets this interface.

        If *fix_imports* is True and *protocol* is less than 3, pickle
        will try to map the new Python 3 names to the old module names
        used in Python 2, so that the pickle data stream is readable
        with Python 2.

        If *buffer_callback* is None (the default), buffer views are
        serialized into *file* as part of the pickle stream.

        If *buffer_callback* is not None, then it can be called any number
        of times with a buffer view.  If the callback returns a false value
        (such as None), the given buffer is out-of-band; otherwise the
        buffer is serialized in-band, i.e. inside the pickle stream.

        It is an error if *buffer_callback* is not None and *protocol*
        is None or smaller than 5.
        """
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        if protocol < 0:
            protocol = HIGHEST_PROTOCOL
        elif not 0 <= protocol <= HIGHEST_PROTOCOL:
            raise ValueError("pickle protocol must be <= %d" % HIGHEST_PROTOCOL)
        if buffer_callback is not None and protocol < 5:
            raise ValueError("buffer_callback needs protocol >= 5")
        self._buffer_callback = buffer_callback
        try:
            self._file_write = file.write
        except AttributeError:
            raise TypeError("file must have a 'write' attribute")
        self.framer = _Framer(self._file_write)
        self.write = self.framer.write
        self._write_large_bytes = self.framer.write_large_bytes
        self.memo = {}
        self.proto = int(protocol)
        self.bin = protocol >= 1
        self.fast = 0
        self.fix_imports = fix_imports and protocol < 3

    def clear_memo(self):
        """Clears the pickler's "memo".

        The memo is the data structure that remembers which objects the
        pickler has already seen, so that shared or recursive objects
        are pickled by reference and not by value.  This method is
        useful when re-using picklers.
        """
        self.memo.clear()

    def dump(self, obj):
        """Write a pickled representation of obj to the open file."""
        # Check whether Pickler was initialized correctly. This is
        # only needed to mimic the behavior of _pickle.Pickler.dump().
        if not hasattr(self, "_file_write"):
            raise PicklingError("Pickler.__init__() was not called by "
                                "%s.__init__()" % (self.__class__.__name__,))
        if self.proto >= 2:
            self.write(PROTO + pack("<B", self.proto))
        if self.proto >= 4:
            self.framer.start_framing()
        self.save(obj)
        self.write(STOP)
        self.framer.end_framing()

    def memoize(self, obj):
        """Store an object in the memo."""

        # The Pickler memo is a dictionary mapping object ids to 2-tuples
        # that contain the Unpickler memo key and the object being memoized.
        # The memo key is written to the pickle and will become
        # the key in the Unpickler's memo.  The object is stored in the
        # Pickler memo so that transient objects are kept alive during
        # pickling.

        # The use of the Unpickler memo length as the memo key is just a
        # convention.  The only requirement is that the memo values be unique.
        # But there appears no advantage to any other scheme, and this
        # scheme allows the Unpickler memo to be implemented as a plain (but
        # growable) array, indexed by memo key.
        if self.fast:
            return
        assert id(obj) not in self.memo
        idx = len(self.memo)
        self.write(self.put(idx))
        self.memo[id(obj)] = idx, obj

    # Return a PUT (BINPUT, LONG_BINPUT) opcode string, with argument i.
    def put(self, idx):
        if self.proto >= 4:
            return MEMOIZE
        elif self.bin:
            if idx < 256:
                return BINPUT + pack("<B", idx)
            else:
                return LONG_BINPUT + pack("<I", idx)
        else:
            return PUT + repr(idx).encode("ascii") + b'\n'

    # Return a GET (BINGET, LONG_BINGET) opcode string, with argument i.
    def get(self, i):
        if self.bin:
            if i < 256:
                return BINGET + pack("<B", i)
            else:
                return LONG_BINGET + pack("<I", i)

        return GET + repr(i).encode("ascii") + b'\n'

    def save(self, obj, save_persistent_id=True):
        self.framer.commit_frame()

        # Check for persistent id (defined by a subclass)
        pid = self.persistent_id(obj)
        if pid is not None and save_persistent_id:
            self.save_pers(pid)
            return

        # Check the memo
        x = self.memo.get(id(obj))
        if x is not None:
            self.write(self.get(x[0]))
            return

        rv = NotImplemented
        reduce = getattr(self, "reducer_override", None)
        if reduce is not None:
            rv = reduce(obj)

        if rv is NotImplemented:
            # Check the type dispatch table
            t = type(obj)
            f = self.dispatch.get(t)
            if f is not None:
                f(self, obj)  # Call unbound method with explicit self
                return

            # Check private dispatch table if any, or else
            # copyreg.dispatch_table
            reduce = getattr(self, 'dispatch_table', dispatch_table).get(t)
            if reduce is not None:
                rv = reduce(obj)
            else:
                # Check for a class with a custom metaclass; treat as regular
                # class
                if issubclass(t, type):
                    self.save_global(obj)
                    return

                # Check for a __reduce_ex__ method, fall back to __reduce__
                reduce = getattr(obj, "__reduce_ex__", None)
                if reduce is not None:
                    rv = reduce(self.proto)
                else:
                    reduce = getattr(obj, "__reduce__", None)
                    if reduce is not None:
                        rv = reduce()
                    else:
                        raise PicklingError("Can't pickle %r object: %r" %
                                            (t.__name__, obj))

        # Check for string returned by reduce(), meaning "save as global"
        if isinstance(rv, str):
            self.save_global(obj, rv)
            return

        # Assert that reduce() returned a tuple
        if not isinstance(rv, tuple):
            raise PicklingError("%s must return string or tuple" % reduce)

        # Assert that it returned an appropriately sized tuple
        l = len(rv)
        if not (2 <= l <= 6):
            raise PicklingError("Tuple returned by %s must have "
                                "two to six elements" % reduce)

        # Save the reduce() output and finally memoize the object
        self.save_reduce(obj=obj, *rv)

    def persistent_id(self, obj):
        # This exists so a subclass can override it
        return None

    def save_pers(self, pid):
        # Save a persistent id reference
        if self.bin:
            self.save(pid, save_persistent_id=False)
            self.write(BINPERSID)
        else:
            try:
                self.write(PERSID + str(pid).encode("ascii") + b'\n')
            except UnicodeEncodeError:
                raise PicklingError(
                    "persistent IDs in protocol 0 must be ASCII strings")

    def save_reduce(self, func, args, state=None, listitems=None,
                    dictitems=None, state_setter=None, *, obj=None):
        # This API is called by some subclasses

        if not isinstance(args, tuple):
            raise PicklingError("args from save_reduce() must be a tuple")
        if not callable(func):
            raise PicklingError("func from save_reduce() must be callable")

        save = self.save
        write = self.write

        func_name = getattr(func, "__name__", "")
        if self.proto >= 2 and func_name == "__newobj_ex__":
            cls, args, kwargs = args
            if not hasattr(cls, "__new__"):
                raise PicklingError("args[0] from {} args has no __new__"
                                    .format(func_name))
            if obj is not None and cls is not obj.__class__:
                raise PicklingError("args[0] from {} args has the wrong class"
                                    .format(func_name))
            if self.proto >= 4:
                save(cls)
                save(args)
                save(kwargs)
                write(NEWOBJ_EX)
            else:
                func = partial(cls.__new__, cls, *args, **kwargs)
                save(func)
                save(())
                write(REDUCE)
        elif self.proto >= 2 and func_name == "__newobj__":
            # A __reduce__ implementation can direct protocol 2 or newer to
            # use the more efficient NEWOBJ opcode, while still
            # allowing protocol 0 and 1 to work normally.  For this to
            # work, the function returned by __reduce__ should be
            # called __newobj__, and its first argument should be a
            # class.  The implementation for __newobj__
            # should be as follows, although pickle has no way to
            # verify this:
            #
            # def __newobj__(cls, *args):
            #     return cls.__new__(cls, *args)
            #
            # Protocols 0 and 1 will pickle a reference to __newobj__,
            # while protocol 2 (and above) will pickle a reference to
            # cls, the remaining args tuple, and the NEWOBJ code,
            # which calls cls.__new__(cls, *args) at unpickling time
            # (see load_newobj below).  If __reduce__ returns a
            # three-tuple, the state from the third tuple item will be
            # pickled regardless of the protocol, calling __setstate__
            # at unpickling time (see load_build below).
            #
            # Note that no standard __newobj__ implementation exists;
            # you have to provide your own.  This is to enforce
            # compatibility with Python 2.2 (pickles written using
            # protocol 0 or 1 in Python 2.3 should be unpicklable by
            # Python 2.2).
            cls = args[0]
            if not hasattr(cls, "__new__"):
                raise PicklingError(
                    "args[0] from __newobj__ args has no __new__")
            if obj is not None and cls is not obj.__class__:
                raise PicklingError(
                    "args[0] from __newobj__ args has the wrong class")
            args = args[1:]
            save(cls)
            save(args)
            write(NEWOBJ)
        else:
            save(func)
            save(args)
            write(REDUCE)

        if obj is not None:
            # If the object is already in the memo, this means it is
            # recursive. In this case, throw away everything we put on the
            # stack, and fetch the object back from the memo.
            if id(obj) in self.memo:
                write(POP + self.get(self.memo[id(obj)][0]))
            else:
                self.memoize(obj)

        # More new special cases (that work with older protocols as
        # well): when __reduce__ returns a tuple with 4 or 5 items,
        # the 4th and 5th item should be iterators that provide list
        # items and dict items (as (key, value) tuples), or None.

        if listitems is not None:
            self._batch_appends(listitems)

        if dictitems is not None:
            self._batch_setitems(dictitems)

        if state is not None:
            if state_setter is None:
                save(state)
                write(BUILD)
            else:
                # If a state_setter is specified, call it instead of load_build
                # to update obj's with its previous state.
                # First, push state_setter and its tuple of expected arguments
                # (obj, state) onto the stack.
                save(state_setter)
                save(obj)  # simple BINGET opcode as obj is already memoized.
                save(state)
                write(TUPLE2)
                # Trigger a state_setter(obj, state) function call.
                write(REDUCE)
                # The purpose of state_setter is to carry-out an
                # inplace modification of obj. We do not care about what the
                # method might return, so its output is eventually removed from
                # the stack.
                write(POP)

    # Methods below this point are dispatched through the dispatch table

    dispatch = {}

    def save_none(self, obj):
        self.write(NONE)
    dispatch[type(None)] = save_none

    def save_bool(self, obj):
        if self.proto >= 2:
            self.write(NEWTRUE if obj else NEWFALSE)
        else:
            self.write(TRUE if obj else FALSE)
    dispatch[bool] = save_bool

    def save_long(self, obj):
        if self.bin:
            # If the int is small enough to fit in a signed 4-byte 2's-comp
            # format, we can store it more efficiently than the general
            # case.
            # First one- and two-byte unsigned ints:
            if obj >= 0:
                if obj <= 0xff:
                    self.write(BININT1 + pack("<B", obj))
                    return
                if obj <= 0xffff:
                    self.write(BININT2 + pack("<H", obj))
                    return
            # Next check for 4-byte signed ints:
            if -0x80000000 <= obj <= 0x7fffffff:
                self.write(BININT + pack("<i", obj))
                return
        if self.proto >= 2:
            encoded = encode_long(obj)
            n = len(encoded)
            if n < 256:
                self.write(LONG1 + pack("<B", n) + encoded)
            else:
                self.write(LONG4 + pack("<i", n) + encoded)
            return
        if -0x80000000 <= obj <= 0x7fffffff:
            self.write(INT + repr(obj).encode("ascii") + b'\n')
        else:
            self.write(LONG + repr(obj).encode("ascii") + b'L\n')
    dispatch[int] = save_long

    def save_float(self, obj):
        if self.bin:
            self.write(BINFLOAT + pack('>d', obj))
        else:
            self.write(FLOAT + repr(obj).encode("ascii") + b'\n')
    dispatch[float] = save_float

    def save_bytes(self, obj):
        if self.proto < 3:
            if not obj: # bytes object is empty
                self.save_reduce(bytes, (), obj=obj)
            else:
                self.save_reduce(codecs.encode,
                                 (str(obj, 'latin1'), 'latin1'), obj=obj)
            return
        n = len(obj)
        if n <= 0xff:
            self.write(SHORT_BINBYTES + pack("<B", n) + obj)
        elif n > 0xffffffff and self.proto >= 4:
            self._write_large_bytes(BINBYTES8 + pack("<Q", n), obj)
        elif n >= self.framer._FRAME_SIZE_TARGET:
            self._write_large_bytes(BINBYTES + pack("<I", n), obj)
        else:
            self.write(BINBYTES + pack("<I", n) + obj)
        self.memoize(obj)
    dispatch[bytes] = save_bytes

    def save_bytearray(self, obj):
        if self.proto < 5:
            if not obj:  # bytearray is empty
                self.save_reduce(bytearray, (), obj=obj)
            else:
                self.save_reduce(bytearray, (bytes(obj),), obj=obj)
            return
        n = len(obj)
        if n >= self.framer._FRAME_SIZE_TARGET:
            self._write_large_bytes(BYTEARRAY8 + pack("<Q", n), obj)
        else:
            self.write(BYTEARRAY8 + pack("<Q", n) + obj)
        self.memoize(obj)
    dispatch[bytearray] = save_bytearray

    if _HAVE_PICKLE_BUFFER:
        def save_picklebuffer(self, obj):
            if self.proto < 5:
                raise PicklingError("PickleBuffer can only pickled with "
                                    "protocol >= 5")
            with obj.raw() as m:
                if not m.contiguous:
                    raise PicklingError("PickleBuffer can not be pickled when "
                                        "pointing to a non-contiguous buffer")
                in_band = True
                if self._buffer_callback is not None:
                    in_band = bool(self._buffer_callback(obj))
                if in_band:
                    # Write data in-band
                    # XXX The C implementation avoids a copy here
                    if m.readonly:
                        self.save_bytes(m.tobytes())
                    else:
                        self.save_bytearray(m.tobytes())
                else:
                    # Write data out-of-band
                    self.write(NEXT_BUFFER)
                    if m.readonly:
                        self.write(READONLY_BUFFER)

        dispatch[PickleBuffer] = save_picklebuffer

    def save_str(self, obj):
        if self.bin:
            encoded = obj.encode('utf-8', 'surrogatepass')
            n = len(encoded)
            if n <= 0xff and self.proto >= 4:
                self.write(SHORT_BINUNICODE + pack("<B", n) + encoded)
            elif n > 0xffffffff and self.proto >= 4:
                self._write_large_bytes(BINUNICODE8 + pack("<Q", n), encoded)
            elif n >= self.framer._FRAME_SIZE_TARGET:
                self._write_large_bytes(BINUNICODE + pack("<I", n), encoded)
            else:
                self.write(BINUNICODE + pack("<I", n) + encoded)
        else:
            obj = obj.replace("\\", "\\u005c")
            obj = obj.replace("\0", "\\u0000")
            obj = obj.replace("\n", "\\u000a")
            obj = obj.replace("\r", "\\u000d")
            obj = obj.replace("\x1a", "\\u001a")  # EOF on DOS
            self.write(UNICODE + obj.encode('raw-unicode-escape') +
                       b'\n')
        self.memoize(obj)
    dispatch[str] = save_str

    def save_tuple(self, obj):
        if not obj: # tuple is empty
            if self.bin:
                self.write(EMPTY_TUPLE)
            else:
                self.write(MARK + TUPLE)
            return

        n = len(obj)
        save = self.save
        memo = self.memo
        if n <= 3 and self.proto >= 2:
            for element in obj:
                save(element)
            # Subtle.  Same as in the big comment below.
            if id(obj) in memo:
                get = self.get(memo[id(obj)][0])
                self.write(POP * n + get)
            else:
                self.write(_tuplesize2code[n])
                self.memoize(obj)
            return

        # proto 0 or proto 1 and tuple isn't empty, or proto > 1 and tuple
        # has more than 3 elements.
        write = self.write
        write(MARK)
        for element in obj:
            save(element)

        if id(obj) in memo:
            # Subtle.  d was not in memo when we entered save_tuple(), so
            # the process of saving the tuple's elements must have saved
            # the tuple itself:  the tuple is recursive.  The proper action
            # now is to throw away everything we put on the stack, and
            # simply GET the tuple (it's already constructed).  This check
            # could have been done in the "for element" loop instead, but
            # recursive tuples are a rare thing.
            get = self.get(memo[id(obj)][0])
            if self.bin:
                write(POP_MARK + get)
            else:   # proto 0 -- POP_MARK not available
                write(POP * (n+1) + get)
            return

        # No recursion.
        write(TUPLE)
        self.memoize(obj)

    dispatch[tuple] = save_tuple

    def save_list(self, obj):
        if self.bin:
            self.write(EMPTY_LIST)
        else:   # proto 0 -- can't use EMPTY_LIST
            self.write(MARK + LIST)

        self.memoize(obj)
        self._batch_appends(obj)

    dispatch[list] = save_list

    _BATCHSIZE = 1000

    def _batch_appends(self, items):
        # Helper to batch up APPENDS sequences
        save = self.save
        write = self.write

        if not self.bin:
            for x in items:
                save(x)
                write(APPEND)
            return

        it = iter(items)
        while True:
            tmp = list(islice(it, self._BATCHSIZE))
            n = len(tmp)
            if n > 1:
                write(MARK)
                for x in tmp:
                    save(x)
                write(APPENDS)
            elif n:
                save(tmp[0])
                write(APPEND)
            # else tmp is empty, and we're done
            if n < self._BATCHSIZE:
                return

    def save_dict(self, obj):
        if self.bin:
            self.write(EMPTY_DICT)
        else:   # proto 0 -- can't use EMPTY_DICT
            self.write(MARK + DICT)

        self.memoize(obj)
        self._batch_setitems(obj.items())

    dispatch[dict] = save_dict
    if PyStringMap is not None:
        dispatch[PyStringMap] = save_dict

    def _batch_setitems(self, items):
        # Helper to batch up SETITEMS sequences; proto >= 1 only
        save = self.save
        write = self.write

        if not self.bin:
            for k, v in items:
                save(k)
                save(v)
                write(SETITEM)
            return

        it = iter(items)
        while True:
            tmp = list(islice(it, self._BATCHSIZE))
            n = len(tmp)
            if n > 1:
                write(MARK)
                for k, v in tmp:
                    save(k)
                    save(v)
                write(SETITEMS)
            elif n:
                k, v = tmp[0]
                save(k)
                save(v)
                write(SETITEM)
            # else tmp is empty, and we're done
            if n < self._BATCHSIZE:
                return

    def save_set(self, obj):
        save = self.save
        write = self.write

        if self.proto < 4:
            self.save_reduce(set, (list(obj),), obj=obj)
            return

        write(EMPTY_SET)
        self.memoize(obj)

        it = iter(obj)
        while True:
            batch = list(islice(it, self._BATCHSIZE))
            n = len(batch)
            if n > 0:
                write(MARK)
                for item in batch:
                    save(item)
                write(ADDITEMS)
            if n < self._BATCHSIZE:
                return
    dispatch[set] = save_set

    def save_frozenset(self, obj):
        save = self.save
        write = self.write

        if self.proto < 4:
            self.save_reduce(frozenset, (list(obj),), obj=obj)
            return

        write(MARK)
        for item in obj:
            save(item)

        if id(obj) in self.memo:
            # If the object is already in the memo, this means it is
            # recursive. In this case, throw away everything we put on the
            # stack, and fetch the object back from the memo.
            write(POP_MARK + self.get(self.memo[id(obj)][0]))
            return

        write(FROZENSET)
        self.memoize(obj)
    dispatch[frozenset] = save_frozenset

    def save_global(self, obj, name=None):
        write = self.write
        memo = self.memo

        if name is None:
            name = getattr(obj, '__qualname__', None)
        if name is None:
            name = obj.__name__

        module_name = whichmodule(obj, name)
        try:
            __import__(module_name, level=0)
            module = sys.modules[module_name]
            obj2, parent = _getattribute(module, name)
        except (ImportError, KeyError, AttributeError):
            raise PicklingError(
                "Can't pickle %r: it's not found as %s.%s" %
                (obj, module_name, name)) from None
        else:
            if obj2 is not obj:
                raise PicklingError(
                    "Can't pickle %r: it's not the same object as %s.%s" %
                    (obj, module_name, name))

        if self.proto >= 2:
            code = _extension_registry.get((module_name, name))
            if code:
                assert code > 0
                if code <= 0xff:
                    write(EXT1 + pack("<B", code))
                elif code <= 0xffff:
                    write(EXT2 + pack("<H", code))
                else:
                    write(EXT4 + pack("<i", code))
                return
        lastname = name.rpartition('.')[2]
        if parent is module:
            name = lastname
        # Non-ASCII identifiers are supported only with protocols >= 3.
        if self.proto >= 4:
            self.save(module_name)
            self.save(name)
            write(STACK_GLOBAL)
        elif parent is not module:
            self.save_reduce(getattr, (parent, lastname))
        elif self.proto >= 3:
            write(GLOBAL + bytes(module_name, "utf-8") + b'\n' +
                  bytes(name, "utf-8") + b'\n')
        else:
            if self.fix_imports:
                r_name_mapping = _compat_pickle.REVERSE_NAME_MAPPING
                r_import_mapping = _compat_pickle.REVERSE_IMPORT_MAPPING
                if (module_name, name) in r_name_mapping:
                    module_name, name = r_name_mapping[(module_name, name)]
                elif module_name in r_import_mapping:
                    module_name = r_import_mapping[module_name]
            try:
                write(GLOBAL + bytes(module_name, "ascii") + b'\n' +
                      bytes(name, "ascii") + b'\n')
            except UnicodeEncodeError:
                raise PicklingError(
                    "can't pickle global identifier '%s.%s' using "
                    "pickle protocol %i" % (module, name, self.proto)) from None

        self.memoize(obj)

    def save_type(self, obj):
        if obj is type(None):
            return self.save_reduce(type, (None,), obj=obj)
        elif obj is type(NotImplemented):
            return self.save_reduce(type, (NotImplemented,), obj=obj)
        elif obj is type(...):
            return self.save_reduce(type, (...,), obj=obj)
        return self.save_global(obj)

    dispatch[FunctionType] = save_global
    dispatch[type] = save_type


# Unpickling machinery

class _Unpickler:

    def __init__(self, file, *, fix_imports=True,
                 encoding="ASCII", errors="strict", buffers=None):
        """This takes a binary file for reading a pickle data stream.

        The protocol version of the pickle is detected automatically, so
        no proto argument is needed.

        The argument *file* must have two methods, a read() method that
        takes an integer argument, and a readline() method that requires
        no arguments.  Both methods should return bytes.  Thus *file*
        can be a binary file object opened for reading, an io.BytesIO
        object, or any other custom object that meets this interface.

        The file-like object must have two methods, a read() method
        that takes an integer argument, and a readline() method that
        requires no arguments.  Both methods should return bytes.
        Thus file-like object can be a binary file object opened for
        reading, a BytesIO object, or any other custom object that
        meets this interface.

        If *buffers* is not None, it should be an iterable of buffer-enabled
        objects that is consumed each time the pickle stream references
        an out-of-band buffer view.  Such buffers have been given in order
        to the *buffer_callback* of a Pickler object.

        If *buffers* is None (the default), then the buffers are taken
        from the pickle stream, assuming they are serialized there.
        It is an error for *buffers* to be None if the pickle stream
        was produced with a non-None *buffer_callback*.

        Other optional arguments are *fix_imports*, *encoding* and
        *errors*, which are used to control compatibility support for
        pickle stream generated by Python 2.  If *fix_imports* is True,
        pickle will try to map the old Python 2 names to the new names
        used in Python 3.  The *encoding* and *errors* tell pickle how
        to decode 8-bit string instances pickled by Python 2; these
        default to 'ASCII' and 'strict', respectively. *encoding* can be
        'bytes' to read these 8-bit string instances as bytes objects.
        """
        self._buffers = iter(buffers) if buffers is not None else None
        self._file_readline = file.readline
        self._file_read = file.read
        self.memo = {}
        self.encoding = encoding
        self.errors = errors
        self.proto = 0
        self.fix_imports = fix_imports

    def load(self):
        """Read a pickled object representation from the open file.

        Return the reconstituted object hierarchy specified in the file.
        """
        # Check whether Unpickler was initialized correctly. This is
        # only needed to mimic the behavior of _pickle.Unpickler.dump().
        if not hasattr(self, "_file_read"):
            raise UnpicklingError("Unpickler.__init__() was not called by "
                                  "%s.__init__()" % (self.__class__.__name__,))
        self._unframer = _Unframer(self._file_read, self._file_readline)
        self.read = self._unframer.read
        self.readinto = self._unframer.readinto
        self.readline = self._unframer.readline
        self.metastack = []
        self.stack = []
        self.append = self.stack.append
        self.proto = 0
        read = self.read
        dispatch = self.dispatch
        try:
            while True:
                key = read(1)
                if not key:
                    raise EOFError
                assert isinstance(key, bytes_types)
                dispatch[key[0]](self)
        except _Stop as stopinst:
            return stopinst.value

    # Return a list of items pushed in the stack after last MARK instruction.
    def pop_mark(self):
        items = self.stack
        self.stack = self.metastack.pop()
        self.append = self.stack.append
        return items

    def persistent_load(self, pid):
        raise UnpicklingError("unsupported persistent id encountered")

    dispatch = {}

    def load_proto(self):
        proto = self.read(1)[0]
        if not 0 <= proto <= HIGHEST_PROTOCOL:
            raise ValueError("unsupported pickle protocol: %d" % proto)
        self.proto = proto
    dispatch[PROTO[0]] = load_proto

    def load_frame(self):
        frame_size, = unpack('<Q', self.read(8))
        if frame_size > sys.maxsize:
            raise ValueError("frame size > sys.maxsize: %d" % frame_size)
        self._unframer.load_frame(frame_size)
    dispatch[FRAME[0]] = load_frame

    def load_persid(self):
        try:
            pid = self.readline()[:-1].decode("ascii")
        except UnicodeDecodeError:
            raise UnpicklingError(
                "persistent IDs in protocol 0 must be ASCII strings")
        self.append(self.persistent_load(pid))
    dispatch[PERSID[0]] = load_persid

    def load_binpersid(self):
        pid = self.stack.pop()
        self.append(self.persistent_load(pid))
    dispatch[BINPERSID[0]] = load_binpersid

    def load_none(self):
        self.append(None)
    dispatch[NONE[0]] = load_none

    def load_false(self):
        self.append(False)
    dispatch[NEWFALSE[0]] = load_false

    def load_true(self):
        self.append(True)
    dispatch[NEWTRUE[0]] = load_true

    def load_int(self):
        data = self.readline()
        if data == FALSE[1:]:
            val = False
        elif data == TRUE[1:]:
            val = True
        else:
            val = int(data, 0)
        self.append(val)
    dispatch[INT[0]] = load_int

    def load_binint(self):
        self.append(unpack('<i', self.read(4))[0])
    dispatch[BININT[0]] = load_binint

    def load_binint1(self):
        self.append(self.read(1)[0])
    dispatch[BININT1[0]] = load_binint1

    def load_binint2(self):
        self.append(unpack('<H', self.read(2))[0])
    dispatch[BININT2[0]] = load_binint2

    def load_long(self):
        val = self.readline()[:-1]
        if val and val[-1] == b'L'[0]:
            val = val[:-1]
        self.append(int(val, 0))
    dispatch[LONG[0]] = load_long

    def load_long1(self):
        n = self.read(1)[0]
        data = self.read(n)
        self.append(decode_long(data))
    dispatch[LONG1[0]] = load_long1

    def load_long4(self):
        n, = unpack('<i', self.read(4))
        if n < 0:
            # Corrupt or hostile pickle -- we never write one like this
            raise UnpicklingError("LONG pickle has negative byte count")
        data = self.read(n)
        self.append(decode_long(data))
    dispatch[LONG4[0]] = load_long4

    def load_float(self):
        self.append(float(self.readline()[:-1]))
    dispatch[FLOAT[0]] = load_float

    def load_binfloat(self):
        self.append(unpack('>d', self.read(8))[0])
    dispatch[BINFLOAT[0]] = load_binfloat

    def _decode_string(self, value):
        # Used to allow strings from Python 2 to be decoded either as
        # bytes or Unicode strings.  This should be used only with the
        # STRING, BINSTRING and SHORT_BINSTRING opcodes.
        if self.encoding == "bytes":
            return value
        else:
            return value.decode(self.encoding, self.errors)

    def load_string(self):
        data = self.readline()[:-1]
        # Strip outermost quotes
        if len(data) >= 2 and data[0] == data[-1] and data[0] in b'"\'':
            data = data[1:-1]
        else:
            raise UnpicklingError("the STRING opcode argument must be quoted")
        self.append(self._decode_string(codecs.escape_decode(data)[0]))
    dispatch[STRING[0]] = load_string

    def load_binstring(self):
        # Deprecated BINSTRING uses signed 32-bit length
        len, = unpack('<i', self.read(4))
        if len < 0:
            raise UnpicklingError("BINSTRING pickle has negative byte count")
        data = self.read(len)
        self.append(self._decode_string(data))
    dispatch[BINSTRING[0]] = load_binstring

    def load_binbytes(self):
        len, = unpack('<I', self.read(4))
        if len > maxsize:
            raise UnpicklingError("BINBYTES exceeds system's maximum size "
                                  "of %d bytes" % maxsize)
        self.append(self.read(len))
    dispatch[BINBYTES[0]] = load_binbytes

    def load_unicode(self):
        self.append(str(self.readline()[:-1], 'raw-unicode-escape'))
    dispatch[UNICODE[0]] = load_unicode

    def load_binunicode(self):
        len, = unpack('<I', self.read(4))
        if len > maxsize:
            raise UnpicklingError("BINUNICODE exceeds system's maximum size "
                                  "of %d bytes" % maxsize)
        self.append(str(self.read(len), 'utf-8', 'surrogatepass'))
    dispatch[BINUNICODE[0]] = load_binunicode

    def load_binunicode8(self):
        len, = unpack('<Q', self.read(8))
        if len > maxsize:
            raise UnpicklingError("BINUNICODE8 exceeds system's maximum size "
                                  "of %d bytes" % maxsize)
        self.append(str(self.read(len), 'utf-8', 'surrogatepass'))
    dispatch[BINUNICODE8[0]] = load_binunicode8

    def load_binbytes8(self):
        len, = unpack('<Q', self.read(8))
        if len > maxsize:
            raise UnpicklingError("BINBYTES8 exceeds system's maximum size "
                                  "of %d bytes" % maxsize)
        self.append(self.read(len))
    dispatch[BINBYTES8[0]] = load_binbytes8

    def load_bytearray8(self):
        len, = unpack('<Q', self.read(8))
        if len > maxsize:
            raise UnpicklingError("BYTEARRAY8 exceeds system's maximum size "
                                  "of %d bytes" % maxsize)
        b = bytearray(len)
        self.readinto(b)
        self.append(b)
    dispatch[BYTEARRAY8[0]] = load_bytearray8

    def load_next_buffer(self):
        if self._buffers is None:
            raise UnpicklingError("pickle stream refers to out-of-band data "
                                  "but no *buffers* argument was given")
        try:
            buf = next(self._buffers)
        except StopIteration:
            raise UnpicklingError("not enough out-of-band buffers")
        self.append(buf)
    dispatch[NEXT_BUFFER[0]] = load_next_buffer

    def load_readonly_buffer(self):
        buf = self.stack[-1]
        with memoryview(buf) as m:
            if not m.readonly:
                self.stack[-1] = m.toreadonly()
    dispatch[READONLY_BUFFER[0]] = load_readonly_buffer

    def load_short_binstring(self):
        len = self.read(1)[0]
        data = self.read(len)
        self.append(self._decode_string(data))
    dispatch[SHORT_BINSTRING[0]] = load_short_binstring

    def load_short_binbytes(self):
        len = self.read(1)[0]
        self.append(self.read(len))
    dispatch[SHORT_BINBYTES[0]] = load_short_binbytes

    def load_short_binunicode(self):
        len = self.read(1)[0]
        self.append(str(self.read(len), 'utf-8', 'surrogatepass'))
    dispatch[SHORT_BINUNICODE[0]] = load_short_binunicode

    def load_tuple(self):
        items = self.pop_mark()
        self.append(tuple(items))
    dispatch[TUPLE[0]] = load_tuple

    def load_empty_tuple(self):
        self.append(())
    dispatch[EMPTY_TUPLE[0]] = load_empty_tuple

    def load_tuple1(self):
        self.stack[-1] = (self.stack[-1],)
    dispatch[TUPLE1[0]] = load_tuple1

    def load_tuple2(self):
        self.stack[-2:] = [(self.stack[-2], self.stack[-1])]
    dispatch[TUPLE2[0]] = load_tuple2

    def load_tuple3(self):
        self.stack[-3:] = [(self.stack[-3], self.stack[-2], self.stack[-1])]
    dispatch[TUPLE3[0]] = load_tuple3

    def load_empty_list(self):
        self.append([])
    dispatch[EMPTY_LIST[0]] = load_empty_list

    def load_empty_dictionary(self):
        self.append({})
    dispatch[EMPTY_DICT[0]] = load_empty_dictionary

    def load_empty_set(self):
        self.append(set())
    dispatch[EMPTY_SET[0]] = load_empty_set

    def load_frozenset(self):
        items = self.pop_mark()
        self.append(frozenset(items))
    dispatch[FROZENSET[0]] = load_frozenset

    def load_list(self):
        items = self.pop_mark()
        self.append(items)
    dispatch[LIST[0]] = load_list

    def load_dict(self):
        items = self.pop_mark()
        d = {items[i]: items[i+1]
             for i in range(0, len(items), 2)}
        self.append(d)
    dispatch[DICT[0]] = load_dict

    # INST and OBJ differ only in how they get a class object.  It's not
    # only sensible to do the rest in a common routine, the two routines
    # previously diverged and grew different bugs.
    # klass is the class to instantiate, and k points to the topmost mark
    # object, following which are the arguments for klass.__init__.
    def _instantiate(self, klass, args):
        if (args or not isinstance(klass, type) or
            hasattr(klass, "__getinitargs__")):
            try:
                value = klass(*args)
            except TypeError as err:
                raise TypeError("in constructor for %s: %s" %
                                (klass.__name__, str(err)), sys.exc_info()[2])
        else:
            value = klass.__new__(klass)
        self.append(value)

    def load_inst(self):
        module = self.readline()[:-1].decode("ascii")
        name = self.readline()[:-1].decode("ascii")
        klass = self.find_class(module, name)
        self._instantiate(klass, self.pop_mark())
    dispatch[INST[0]] = load_inst

    def load_obj(self):
        # Stack is ... markobject classobject arg1 arg2 ...
        args = self.pop_mark()
        cls = args.pop(0)
        self._instantiate(cls, args)
    dispatch[OBJ[0]] = load_obj

    def load_newobj(self):
        args = self.stack.pop()
        cls = self.stack.pop()
        obj = cls.__new__(cls, *args)
        self.append(obj)
    dispatch[NEWOBJ[0]] = load_newobj

    def load_newobj_ex(self):
        kwargs = self.stack.pop()
        args = self.stack.pop()
        cls = self.stack.pop()
        obj = cls.__new__(cls, *args, **kwargs)
        self.append(obj)
    dispatch[NEWOBJ_EX[0]] = load_newobj_ex

    def load_global(self):
        module = self.readline()[:-1].decode("utf-8")
        name = self.readline()[:-1].decode("utf-8")
        klass = self.find_class(module, name)
        self.append(klass)
    dispatch[GLOBAL[0]] = load_global

    def load_stack_global(self):
        name = self.stack.pop()
        module = self.stack.pop()
        if type(name) is not str or type(module) is not str:
            raise UnpicklingError("STACK_GLOBAL requires str")
        self.append(self.find_class(module, name))
    dispatch[STACK_GLOBAL[0]] = load_stack_global

    def load_ext1(self):
        code = self.read(1)[0]
        self.get_extension(code)
    dispatch[EXT1[0]] = load_ext1

    def load_ext2(self):
        code, = unpack('<H', self.read(2))
        self.get_extension(code)
    dispatch[EXT2[0]] = load_ext2

    def load_ext4(self):
        code, = unpack('<i', self.read(4))
        self.get_extension(code)
    dispatch[EXT4[0]] = load_ext4

    def get_extension(self, code):
        nil = []
        obj = _extension_cache.get(code, nil)
        if obj is not nil:
            self.append(obj)
            return
        key = _inverted_registry.get(code)
        if not key:
            if code <= 0: # note that 0 is forbidden
                # Corrupt or hostile pickle.
                raise UnpicklingError("EXT specifies code <= 0")
            raise ValueError("unregistered extension code %d" % code)
        obj = self.find_class(*key)
        _extension_cache[code] = obj
        self.append(obj)

    def find_class(self, module, name):
        # Subclasses may override this.
        sys.audit('pickle.find_class', module, name)
        if self.proto < 3 and self.fix_imports:
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        __import__(module, level=0)
        if self.proto >= 4:
            return _getattribute(sys.modules[module], name)[0]
        else:
            return getattr(sys.modules[module], name)

    def load_reduce(self):
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        stack[-1] = func(*args)
    dispatch[REDUCE[0]] = load_reduce

    def load_pop(self):
        if self.stack:
            del self.stack[-1]
        else:
            self.pop_mark()
    dispatch[POP[0]] = load_pop

    def load_pop_mark(self):
        self.pop_mark()
    dispatch[POP_MARK[0]] = load_pop_mark

    def load_dup(self):
        self.append(self.stack[-1])
    dispatch[DUP[0]] = load_dup

    def load_get(self):
        i = int(self.readline()[:-1])
        try:
            self.append(self.memo[i])
        except KeyError:
            msg = f'Memo value not found at index {i}'
            raise UnpicklingError(msg) from None
    dispatch[GET[0]] = load_get

    def load_binget(self):
        i = self.read(1)[0]
        try:
            self.append(self.memo[i])
        except KeyError as exc:
            msg = f'Memo value not found at index {i}'
            raise UnpicklingError(msg) from None
    dispatch[BINGET[0]] = load_binget

    def load_long_binget(self):
        i, = unpack('<I', self.read(4))
        try:
            self.append(self.memo[i])
        except KeyError as exc:
            msg = f'Memo value not found at index {i}'
            raise UnpicklingError(msg) from None
    dispatch[LONG_BINGET[0]] = load_long_binget

    def load_put(self):
        i = int(self.readline()[:-1])
        if i < 0:
            raise ValueError("negative PUT argument")
        self.memo[i] = self.stack[-1]
    dispatch[PUT[0]] = load_put

    def load_binput(self):
        i = self.read(1)[0]
        if i < 0:
            raise ValueError("negative BINPUT argument")
        self.memo[i] = self.stack[-1]
    dispatch[BINPUT[0]] = load_binput

    def load_long_binput(self):
        i, = unpack('<I', self.read(4))
        if i > maxsize:
            raise ValueError("negative LONG_BINPUT argument")
        self.memo[i] = self.stack[-1]
    dispatch[LONG_BINPUT[0]] = load_long_binput

    def load_memoize(self):
        memo = self.memo
        memo[len(memo)] = self.stack[-1]
    dispatch[MEMOIZE[0]] = load_memoize

    def load_append(self):
        stack = self.stack
        value = stack.pop()
        list = stack[-1]
        list.append(value)
    dispatch[APPEND[0]] = load_append

    def load_appends(self):
        items = self.pop_mark()
        list_obj = self.stack[-1]
        try:
            extend = list_obj.extend
        except AttributeError:
            pass
        else:
            extend(items)
            return
        # Even if the PEP 307 requires extend() and append() methods,
        # fall back on append() if the object has no extend() method
        # for backward compatibility.
        append = list_obj.append
        for item in items:
            append(item)
    dispatch[APPENDS[0]] = load_appends

    def load_setitem(self):
        stack = self.stack
        value = stack.pop()
        key = stack.pop()
        dict = stack[-1]
        dict[key] = value
    dispatch[SETITEM[0]] = load_setitem

    def load_setitems(self):
        items = self.pop_mark()
        dict = self.stack[-1]
        for i in range(0, len(items), 2):
            dict[items[i]] = items[i + 1]
    dispatch[SETITEMS[0]] = load_setitems

    def load_additems(self):
        items = self.pop_mark()
        set_obj = self.stack[-1]
        if isinstance(set_obj, set):
            set_obj.update(items)
        else:
            add = set_obj.add
            for item in items:
                add(item)
    dispatch[ADDITEMS[0]] = load_additems

    def load_build(self):
        stack = self.stack
        state = stack.pop()
        inst = stack[-1]
        setstate = getattr(inst, "__setstate__", None)
        if setstate is not None:
            setstate(state)
            return
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if state:
            inst_dict = inst.__dict__
            intern = sys.intern
            for k, v in state.items():
                if type(k) is str:
                    inst_dict[intern(k)] = v
                else:
                    inst_dict[k] = v
        if slotstate:
            for k, v in slotstate.items():
                setattr(inst, k, v)
    dispatch[BUILD[0]] = load_build

    def load_mark(self):
        self.metastack.append(self.stack)
        self.stack = []
        self.append = self.stack.append
    dispatch[MARK[0]] = load_mark

    def load_stop(self):
        value = self.stack.pop()
        raise _Stop(value)
    dispatch[STOP[0]] = load_stop


# Shorthands

def _dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None):
    _Pickler(file, protocol, fix_imports=fix_imports,
             buffer_callback=buffer_callback).dump(obj)

def _dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None):
    f = io.BytesIO()
    _Pickler(f, protocol, fix_imports=fix_imports,
             buffer_callback=buffer_callback).dump(obj)
    res = f.getvalue()
    assert isinstance(res, bytes_types)
    return res

def _load(file, *, fix_imports=True, encoding="ASCII", errors="strict",
          buffers=None):
    return _Unpickler(file, fix_imports=fix_imports, buffers=buffers,
                     encoding=encoding, errors=errors).load()

def _loads(s, /, *, fix_imports=True, encoding="ASCII", errors="strict",
           buffers=None):
    if isinstance(s, str):
        raise TypeError("Can't load pickle from unicode string")
    file = io.BytesIO(s)
    return _Unpickler(file, fix_imports=fix_imports, buffers=buffers,
                      encoding=encoding, errors=errors).load()

# Use the faster _pickle if possible
try:
    from _pickle import (
        PickleError,
        PicklingError,
        UnpicklingError,
        Pickler,
        Unpickler,
        dump,
        dumps,
        load,
        loads
    )
except ImportError:
    Pickler, Unpickler = _Pickler, _Unpickler
    dump, dumps, load, loads = _dump, _dumps, _load, _loads

# Doctest
def _test():
    import doctest
    return doctest.testmod()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='display contents of the pickle files')
    parser.add_argument(
        'pickle_file', type=argparse.FileType('br'),
        nargs='*', help='the pickle file')
    parser.add_argument(
        '-t', '--test', action='store_true',
        help='run self-test suite')
    parser.add_argument(
        '-v', action='store_true',
        help='run verbosely; only affects self-test run')
    args = parser.parse_args()
    if args.test:
        _test()
    else:
        if not args.pickle_file:
            parser.print_help()
        else:
            import pprint
            for f in args.pickle_file:
                obj = load(f)
                pprint.pprint(obj)
<|endoftext|>"""Utilities to support packages."""

from collections import namedtuple
from functools import singledispatch as simplegeneric
import importlib
import importlib.util
import importlib.machinery
import os
import os.path
import sys
from types import ModuleType
import warnings

__all__ = [
    'get_importer', 'iter_importers', 'get_loader', 'find_loader',
    'walk_packages', 'iter_modules', 'get_data',
    'ImpImporter', 'ImpLoader', 'read_code', 'extend_path',
    'ModuleInfo',
]


ModuleInfo = namedtuple('ModuleInfo', 'module_finder name ispkg')
ModuleInfo.__doc__ = 'A namedtuple with minimal info about a module.'


def _get_spec(finder, name):
    """Return the finder-specific module spec."""
    # Works with legacy finders.
    try:
        find_spec = finder.find_spec
    except AttributeError:
        loader = finder.find_module(name)
        if loader is None:
            return None
        return importlib.util.spec_from_loader(name, loader)
    else:
        return find_spec(name)


def read_code(stream):
    # This helper is needed in order for the PEP 302 emulation to
    # correctly handle compiled files
    import marshal

    magic = stream.read(4)
    if magic != importlib.util.MAGIC_NUMBER:
        return None

    stream.read(12) # Skip rest of the header
    return marshal.load(stream)


def walk_packages(path=None, prefix='', onerror=None):
    """Yields ModuleInfo for all modules recursively
    on path, or, if path is None, all accessible modules.

    'path' should be either None or a list of paths to look for
    modules in.

    'prefix' is a string to output on the front of every module name
    on output.

    Note that this function must import all *packages* (NOT all
    modules!) on the given path, in order to access the __path__
    attribute to find submodules.

    'onerror' is a function which gets called with one argument (the
    name of the package which was being imported) if any exception
    occurs while trying to import a package.  If no onerror function is
    supplied, ImportErrors are caught and ignored, while all other
    exceptions are propagated, terminating the search.

    Examples:

    # list all modules python can access
    walk_packages()

    # list all submodules of ctypes
    walk_packages(ctypes.__path__, ctypes.__name__+'.')
    """

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True

    for info in iter_modules(path, prefix):
        yield info

        if info.ispkg:
            try:
                __import__(info.name)
            except ImportError:
                if onerror is not None:
                    onerror(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], '__path__', None) or []

                # don't traverse path items we've seen before
                path = [p for p in path if not seen(p)]

                yield from walk_packages(path, info.name+'.', onerror)


def iter_modules(path=None, prefix=''):
    """Yields ModuleInfo for all submodules on path,
    or, if path is None, all top-level modules on sys.path.

    'path' should be either None or a list of paths to look for
    modules in.

    'prefix' is a string to output on the front of every module name
    on output.
    """
    if path is None:
        importers = iter_importers()
    elif isinstance(path, str):
        raise ValueError("path must be None or list of paths to look for "
                        "modules in")
    else:
        importers = map(get_importer, path)

    yielded = {}
    for i in importers:
        for name, ispkg in iter_importer_modules(i, prefix):
            if name not in yielded:
                yielded[name] = 1
                yield ModuleInfo(i, name, ispkg)


@simplegeneric
def iter_importer_modules(importer, prefix=''):
    if not hasattr(importer, 'iter_modules'):
        return []
    return importer.iter_modules(prefix)


# Implement a file walker for the normal importlib path hook
def _iter_file_finder_modules(importer, prefix=''):
    if importer.path is None or not os.path.isdir(importer.path):
        return

    yielded = {}
    import inspect
    try:
        filenames = os.listdir(importer.path)
    except OSError:
        # ignore unreadable directories like import does
        filenames = []
    filenames.sort()  # handle packages before same-named modules

    for fn in filenames:
        modname = inspect.getmodulename(fn)
        if modname=='__init__' or modname in yielded:
            continue

        path = os.path.join(importer.path, fn)
        ispkg = False

        if not modname and os.path.isdir(path) and '.' not in fn:
            modname = fn
            try:
                dircontents = os.listdir(path)
            except OSError:
                # ignore unreadable directories like import does
                dircontents = []
            for fn in dircontents:
                subname = inspect.getmodulename(fn)
                if subname=='__init__':
                    ispkg = True
                    break
            else:
                continue    # not a package

        if modname and '.' not in modname:
            yielded[modname] = 1
            yield prefix + modname, ispkg

iter_importer_modules.register(
    importlib.machinery.FileFinder, _iter_file_finder_modules)


def _import_imp():
    global imp
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        imp = importlib.import_module('imp')

class ImpImporter:
    """PEP 302 Finder that wraps Python's "classic" import algorithm

    ImpImporter(dirname) produces a PEP 302 finder that searches that
    directory.  ImpImporter(None) produces a PEP 302 finder that searches
    the current sys.path, plus any modules that are frozen or built-in.

    Note that ImpImporter does not currently support being used by placement
    on sys.meta_path.
    """

    def __init__(self, path=None):
        global imp
        warnings.warn("This emulation is deprecated and slated for removal "
                      "in Python 3.12; use 'importlib' instead",
             DeprecationWarning)
        _import_imp()
        self.path = path

    def find_module(self, fullname, path=None):
        # Note: we ignore 'path' argument since it is only used via meta_path
        subname = fullname.split(".")[-1]
        if subname != fullname and self.path is None:
            return None
        if self.path is None:
            path = None
        else:
            path = [os.path.realpath(self.path)]
        try:
            file, filename, etc = imp.find_module(subname, path)
        except ImportError:
            return None
        return ImpLoader(fullname, file, filename, etc)

    def iter_modules(self, prefix=''):
        if self.path is None or not os.path.isdir(self.path):
            return

        yielded = {}
        import inspect
        try:
            filenames = os.listdir(self.path)
        except OSError:
            # ignore unreadable directories like import does
            filenames = []
        filenames.sort()  # handle packages before same-named modules

        for fn in filenames:
            modname = inspect.getmodulename(fn)
            if modname=='__init__' or modname in yielded:
                continue

            path = os.path.join(self.path, fn)
            ispkg = False

            if not modname and os.path.isdir(path) and '.' not in fn:
                modname = fn
                try:
                    dircontents = os.listdir(path)
                except OSError:
                    # ignore unreadable directories like import does
                    dircontents = []
                for fn in dircontents:
                    subname = inspect.getmodulename(fn)
                    if subname=='__init__':
                        ispkg = True
                        break
                else:
                    continue    # not a package

            if modname and '.' not in modname:
                yielded[modname] = 1
                yield prefix + modname, ispkg


class ImpLoader:
    """PEP 302 Loader that wraps Python's "classic" import algorithm
    """
    code = source = None

    def __init__(self, fullname, file, filename, etc):
        warnings.warn("This emulation is deprecated and slated for removal in "
                      "Python 3.12; use 'importlib' instead",
                      DeprecationWarning)
        _import_imp()
        self.file = file
        self.filename = filename
        self.fullname = fullname
        self.etc = etc

    def load_module(self, fullname):
        self._reopen()
        try:
            mod = imp.load_module(fullname, self.file, self.filename, self.etc)
        finally:
            if self.file:
                self.file.close()
        # Note: we don't set __loader__ because we want the module to look
        # normal; i.e. this is just a wrapper for standard import machinery
        return mod

    def get_data(self, pathname):
        with open(pathname, "rb") as file:
            return file.read()

    def _reopen(self):
        if self.file and self.file.closed:
            mod_type = self.etc[2]
            if mod_type==imp.PY_SOURCE:
                self.file = open(self.filename, 'r')
            elif mod_type in (imp.PY_COMPILED, imp.C_EXTENSION):
                self.file = open(self.filename, 'rb')

    def _fix_name(self, fullname):
        if fullname is None:
            fullname = self.fullname
        elif fullname != self.fullname:
            raise ImportError("Loader for module %s cannot handle "
                              "module %s" % (self.fullname, fullname))
        return fullname

    def is_package(self, fullname):
        fullname = self._fix_name(fullname)
        return self.etc[2]==imp.PKG_DIRECTORY

    def get_code(self, fullname=None):
        fullname = self._fix_name(fullname)
        if self.code is None:
            mod_type = self.etc[2]
            if mod_type==imp.PY_SOURCE:
                source = self.get_source(fullname)
                self.code = compile(source, self.filename, 'exec')
            elif mod_type==imp.PY_COMPILED:
                self._reopen()
                try:
                    self.code = read_code(self.file)
                finally:
                    self.file.close()
            elif mod_type==imp.PKG_DIRECTORY:
                self.code = self._get_delegate().get_code()
        return self.code

    def get_source(self, fullname=None):
        fullname = self._fix_name(fullname)
        if self.source is None:
            mod_type = self.etc[2]
            if mod_type==imp.PY_SOURCE:
                self._reopen()
                try:
                    self.source = self.file.read()
                finally:
                    self.file.close()
            elif mod_type==imp.PY_COMPILED:
                if os.path.exists(self.filename[:-1]):
                    with open(self.filename[:-1], 'r') as f:
                        self.source = f.read()
            elif mod_type==imp.PKG_DIRECTORY:
                self.source = self._get_delegate().get_source()
        return self.source

    def _get_delegate(self):
        finder = ImpImporter(self.filename)
        spec = _get_spec(finder, '__init__')
        return spec.loader

    def get_filename(self, fullname=None):
        fullname = self._fix_name(fullname)
        mod_type = self.etc[2]
        if mod_type==imp.PKG_DIRECTORY:
            return self._get_delegate().get_filename()
        elif mod_type in (imp.PY_SOURCE, imp.PY_COMPILED, imp.C_EXTENSION):
            return self.filename
        return None


try:
    import zipimport
    from zipimport import zipimporter

    def iter_zipimport_modules(importer, prefix=''):
        dirlist = sorted(zipimport._zip_directory_cache[importer.archive])
        _prefix = importer.prefix
        plen = len(_prefix)
        yielded = {}
        import inspect
        for fn in dirlist:
            if not fn.startswith(_prefix):
                continue

            fn = fn[plen:].split(os.sep)

            if len(fn)==2 and fn[1].startswith('__init__.py'):
                if fn[0] not in yielded:
                    yielded[fn[0]] = 1
                    yield prefix + fn[0], True

            if len(fn)!=1:
                continue

            modname = inspect.getmodulename(fn[0])
            if modname=='__init__':
                continue

            if modname and '.' not in modname and modname not in yielded:
                yielded[modname] = 1
                yield prefix + modname, False

    iter_importer_modules.register(zipimporter, iter_zipimport_modules)

except ImportError:
    pass


def get_importer(path_item):
    """Retrieve a finder for the given path item

    The returned finder is cached in sys.path_importer_cache
    if it was newly created by a path hook.

    The cache (or part of it) can be cleared manually if a
    rescan of sys.path_hooks is necessary.
    """
    try:
        importer = sys.path_importer_cache[path_item]
    except KeyError:
        for path_hook in sys.path_hooks:
            try:
                importer = path_hook(path_item)
                sys.path_importer_cache.setdefault(path_item, importer)
                break
            except ImportError:
                pass
        else:
            importer = None
    return importer


def iter_importers(fullname=""):
    """Yield finders for the given module name

    If fullname contains a '.', the finders will be for the package
    containing fullname, otherwise they will be all registered top level
    finders (i.e. those on both sys.meta_path and sys.path_hooks).

    If the named module is in a package, that package is imported as a side
    effect of invoking this function.

    If no module name is specified, all top level finders are produced.
    """
    if fullname.startswith('.'):
        msg = "Relative module name {!r} not supported".format(fullname)
        raise ImportError(msg)
    if '.' in fullname:
        # Get the containing package's __path__
        pkg_name = fullname.rpartition(".")[0]
        pkg = importlib.import_module(pkg_name)
        path = getattr(pkg, '__path__', None)
        if path is None:
            return
    else:
        yield from sys.meta_path
        path = sys.path
    for item in path:
        yield get_importer(item)


def get_loader(module_or_name):
    """Get a "loader" object for module_or_name

    Returns None if the module cannot be found or imported.
    If the named module is not already imported, its containing package
    (if any) is imported, in order to establish the package __path__.
    """
    if module_or_name in sys.modules:
        module_or_name = sys.modules[module_or_name]
        if module_or_name is None:
            return None
    if isinstance(module_or_name, ModuleType):
        module = module_or_name
        loader = getattr(module, '__loader__', None)
        if loader is not None:
            return loader
        if getattr(module, '__spec__', None) is None:
            return None
        fullname = module.__name__
    else:
        fullname = module_or_name
    return find_loader(fullname)


def find_loader(fullname):
    """Find a "loader" object for fullname

    This is a backwards compatibility wrapper around
    importlib.util.find_spec that converts most failures to ImportError
    and only returns the loader rather than the full spec
    """
    if fullname.startswith('.'):
        msg = "Relative module name {!r} not supported".format(fullname)
        raise ImportError(msg)
    try:
        spec = importlib.util.find_spec(fullname)
    except (ImportError, AttributeError, TypeError, ValueError) as ex:
        # This hack fixes an impedance mismatch between pkgutil and
        # importlib, where the latter raises other errors for cases where
        # pkgutil previously raised ImportError
        msg = "Error while finding loader for {!r} ({}: {})"
        raise ImportError(msg.format(fullname, type(ex), ex)) from ex
    return spec.loader if spec is not None else None


def extend_path(path, name):
    """Extend a package's path.

    Intended use is to place the following code in a package's __init__.py:

        from pkgutil import extend_path
        __path__ = extend_path(__path__, __name__)

    This will add to the package's __path__ all subdirectories of
    directories on sys.path named after the package.  This is useful
    if one wants to distribute different parts of a single logical
    package as multiple directories.

    It also looks for *.pkg files beginning where * matches the name
    argument.  This feature is similar to *.pth files (see site.py),
    except that it doesn't special-case lines starting with 'import'.
    A *.pkg file is trusted at face value: apart from checking for
    duplicates, all entries found in a *.pkg file are added to the
    path, regardless of whether they are exist the filesystem.  (This
    is a feature.)

    If the input path is not a list (as is the case for frozen
    packages) it is returned unchanged.  The input path is not
    modified; an extended copy is returned.  Items are only appended
    to the copy at the end.

    It is assumed that sys.path is a sequence.  Items of sys.path that
    are not (unicode or 8-bit) strings referring to existing
    directories are ignored.  Unicode items of sys.path that cause
    errors when used as filenames may cause this function to raise an
    exception (in line with os.path.isdir() behavior).
    """

    if not isinstance(path, list):
        # This could happen e.g. when this is called from inside a
        # frozen package.  Return the path unchanged in that case.
        return path

    sname_pkg = name + ".pkg"

    path = path[:] # Start with a copy of the existing path

    parent_package, _, final_name = name.rpartition('.')
    if parent_package:
        try:
            search_path = sys.modules[parent_package].__path__
        except (KeyError, AttributeError):
            # We can't do anything: find_loader() returns None when
            # passed a dotted name.
            return path
    else:
        search_path = sys.path

    for dir in search_path:
        if not isinstance(dir, str):
            continue

        finder = get_importer(dir)
        if finder is not None:
            portions = []
            if hasattr(finder, 'find_spec'):
                spec = finder.find_spec(final_name)
                if spec is not None:
                    portions = spec.submodule_search_locations or []
            # Is this finder PEP 420 compliant?
            elif hasattr(finder, 'find_loader'):
                _, portions = finder.find_loader(final_name)

            for portion in portions:
                # XXX This may still add duplicate entries to path on
                # case-insensitive filesystems
                if portion not in path:
                    path.append(portion)

        # XXX Is this the right thing for subpackages like zope.app?
        # It looks for a file named "zope.app.pkg"
        pkgfile = os.path.join(dir, sname_pkg)
        if os.path.isfile(pkgfile):
            try:
                f = open(pkgfile)
            except OSError as msg:
                sys.stderr.write("Can't open %s: %s\n" %
                                 (pkgfile, msg))
            else:
                with f:
                    for line in f:
                        line = line.rstrip('\n')
                        if not line or line.startswith('#'):
                            continue
                        path.append(line) # Don't check for existence!

    return path


def get_data(package, resource):
    """Get a resource from a package.

    This is a wrapper round the PEP 302 loader get_data API. The package
    argument should be the name of a package, in standard module format
    (foo.bar). The resource argument should be in the form of a relative
    filename, using '/' as the path separator. The parent directory name '..'
    is not allowed, and nor is a rooted name (starting with a '/').

    The function returns a binary string, which is the contents of the
    specified resource.

    For packages located in the filesystem, which have already been imported,
    this is the rough equivalent of

        d = os.path.dirname(sys.modules[package].__file__)
        data = open(os.path.join(d, resource), 'rb').read()

    If the package cannot be located or loaded, or it uses a PEP 302 loader
    which does not support get_data(), then None is returned.
    """

    spec = importlib.util.find_spec(package)
    if spec is None:
        return None
    loader = spec.loader
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    # XXX needs test
    mod = (sys.modules.get(package) or
           importlib._bootstrap._load(spec))
    if mod is None or not hasattr(mod, '__file__'):
        return None

    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.join(*parts)
    return loader.get_data(resource_name)


_NAME_PATTERN = None

def resolve_name(name):
    """
    Resolve a name to an object.

    It is expected that `name` will be a string in one of the following
    formats, where W is shorthand for a valid Python identifier and dot stands
    for a literal period in these pseudo-regexes:

    W(.W)*
    W(.W)*:(W(.W)*)?

    The first form is intended for backward compatibility only. It assumes that
    some part of the dotted name is a package, and the rest is an object
    somewhere within that package, possibly nested inside other objects.
    Because the place where the package stops and the object hierarchy starts
    can't be inferred by inspection, repeated attempts to import must be done
    with this form.

    In the second form, the caller makes the division point clear through the
    provision of a single colon: the dotted name to the left of the colon is a
    package to be imported, and the dotted name to the right is the object
    hierarchy within that package. Only one import is needed in this form. If
    it ends with the colon, then a module object is returned.

    The function will return an object (which might be a module), or raise one
    of the following exceptions:

    ValueError - if `name` isn't in a recognised format
    ImportError - if an import failed when it shouldn't have
    AttributeError - if a failure occurred when traversing the object hierarchy
                     within the imported package to get to the desired object.
    """
    global _NAME_PATTERN
    if _NAME_PATTERN is None:
        # Lazy import to speedup Python startup time
        import re
        dotted_words = r'(?!\d)(\w+)(\.(?!\d)(\w+))*'
        _NAME_PATTERN = re.compile(f'^(?P<pkg>{dotted_words})'
                                   f'(?P<cln>:(?P<obj>{dotted_words})?)?$',
                                   re.UNICODE)

    m = _NAME_PATTERN.match(name)
    if not m:
        raise ValueError(f'invalid format: {name!r}')
    gd = m.groupdict()
    if gd.get('cln'):
        # there is a colon - a one-step import is all that's needed
        mod = importlib.import_module(gd['pkg'])
        parts = gd.get('obj')
        parts = parts.split('.') if parts else []
    else:
        # no colon - have to iterate to find the package boundary
        parts = name.split('.')
        modname = parts.pop(0)
        # first part *must* be a module/package.
        mod = importlib.import_module(modname)
        while parts:
            p = parts[0]
            s = f'{modname}.{p}'
            try:
                mod = importlib.import_module(s)
                parts.pop(0)
                modname = s
            except ImportError:
                break
    # if we reach this point, mod is the module, already imported, and
    # parts is the list of parts in the object hierarchy to be traversed, or
    # an empty list if just the module is wanted.
    result = mod
    for p in parts:
        result = getattr(result, p)
    return result
<|endoftext|>#! /usr/bin/env python3

"""Tool for measuring execution time of small code snippets.

This module avoids a number of common traps for measuring execution
times.  See also Tim Peters' introduction to the Algorithms chapter in
the Python Cookbook, published by O'Reilly.

Library usage: see the Timer class.

Command line usage:
    python timeit.py [-n N] [-r N] [-s S] [-p] [-h] [--] [statement]

Options:
  -n/--number N: how many times to execute 'statement' (default: see below)
  -r/--repeat N: how many times to repeat the timer (default 5)
  -s/--setup S: statement to be executed once initially (default 'pass').
                Execution time of this setup statement is NOT timed.
  -p/--process: use time.process_time() (default is time.perf_counter())
  -v/--verbose: print raw timing results; repeat for more digits precision
  -u/--unit: set the output time unit (nsec, usec, msec, or sec)
  -h/--help: print this usage message and exit
  --: separate options from statement, use when statement starts with -
  statement: statement to be timed (default 'pass')

A multi-line statement may be given by specifying each line as a
separate argument; indented lines are possible by enclosing an
argument in quotes and using leading spaces.  Multiple -s options are
treated similarly.

If -n is not given, a suitable number of loops is calculated by trying
increasing numbers from the sequence 1, 2, 5, 10, 20, 50, ... until the
total time is at least 0.2 seconds.

Note: there is a certain baseline overhead associated with executing a
pass statement.  It differs between versions.  The code here doesn't try
to hide it, but you should be aware of it.  The baseline overhead can be
measured by invoking the program without arguments.

Classes:

    Timer

Functions:

    timeit(string, string) -> float
    repeat(string, string) -> list
    default_timer() -> float

"""

import gc
import sys
import time
import itertools

__all__ = ["Timer", "timeit", "repeat", "default_timer"]

dummy_src_name = "<timeit-src>"
default_number = 1000000
default_repeat = 5
default_timer = time.perf_counter

_globals = globals

# Don't change the indentation of the template; the reindent() calls
# in Timer.__init__() depend on setup being indented 4 spaces and stmt
# being indented 8 spaces.
template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        {stmt}
        pass
    _t1 = _timer()
    return _t1 - _t0
"""

def reindent(src, indent):
    """Helper to reindent a multi-line statement."""
    return src.replace("\n", "\n" + " "*indent)

class Timer:
    """Class for timing execution speed of small code snippets.

    The constructor takes a statement to be timed, an additional
    statement used for setup, and a timer function.  Both statements
    default to 'pass'; the timer function is platform-dependent (see
    module doc string).  If 'globals' is specified, the code will be
    executed within that namespace (as opposed to inside timeit's
    namespace).

    To measure the execution time of the first statement, use the
    timeit() method.  The repeat() method is a convenience to call
    timeit() multiple times and return a list of results.

    The statements may contain newlines, as long as they don't contain
    multi-line string literals.
    """

    def __init__(self, stmt="pass", setup="pass", timer=default_timer,
                 globals=None):
        """Constructor.  See class doc string."""
        self.timer = timer
        local_ns = {}
        global_ns = _globals() if globals is None else globals
        init = ''
        if isinstance(setup, str):
            # Check that the code can be compiled outside a function
            compile(setup, dummy_src_name, "exec")
            stmtprefix = setup + '\n'
            setup = reindent(setup, 4)
        elif callable(setup):
            local_ns['_setup'] = setup
            init += ', _setup=_setup'
            stmtprefix = ''
            setup = '_setup()'
        else:
            raise ValueError("setup is neither a string nor callable")
        if isinstance(stmt, str):
            # Check that the code can be compiled outside a function
            compile(stmtprefix + stmt, dummy_src_name, "exec")
            stmt = reindent(stmt, 8)
        elif callable(stmt):
            local_ns['_stmt'] = stmt
            init += ', _stmt=_stmt'
            stmt = '_stmt()'
        else:
            raise ValueError("stmt is neither a string nor callable")
        src = template.format(stmt=stmt, setup=setup, init=init)
        self.src = src  # Save for traceback display
        code = compile(src, dummy_src_name, "exec")
        exec(code, global_ns, local_ns)
        self.inner = local_ns["inner"]

    def print_exc(self, file=None):
        """Helper to print a traceback from the timed code.

        Typical use:

            t = Timer(...)       # outside the try/except
            try:
                t.timeit(...)    # or t.repeat(...)
            except:
                t.print_exc()

        The advantage over the standard traceback is that source lines
        in the compiled template will be displayed.

        The optional file argument directs where the traceback is
        sent; it defaults to sys.stderr.
        """
        import linecache, traceback
        if self.src is not None:
            linecache.cache[dummy_src_name] = (len(self.src),
                                               None,
                                               self.src.split("\n"),
                                               dummy_src_name)
        # else the source is already stored somewhere else

        traceback.print_exc(file=file)

    def timeit(self, number=default_number):
        """Time 'number' executions of the main statement.

        To be precise, this executes the setup statement once, and
        then returns the time it takes to execute the main statement
        a number of times, as a float measured in seconds.  The
        argument is the number of times through the loop, defaulting
        to one million.  The main statement, the setup statement and
        the timer function to be used are passed to the constructor.
        """
        it = itertools.repeat(None, number)
        gcold = gc.isenabled()
        gc.disable()
        try:
            timing = self.inner(it, self.timer)
        finally:
            if gcold:
                gc.enable()
        return timing

    def repeat(self, repeat=default_repeat, number=default_number):
        """Call timeit() a few times.

        This is a convenience function that calls the timeit()
        repeatedly, returning a list of results.  The first argument
        specifies how many times to call timeit(), defaulting to 5;
        the second argument specifies the timer argument, defaulting
        to one million.

        Note: it's tempting to calculate mean and standard deviation
        from the result vector and report these.  However, this is not
        very useful.  In a typical case, the lowest value gives a
        lower bound for how fast your machine can run the given code
        snippet; higher values in the result vector are typically not
        caused by variability in Python's speed, but by other
        processes interfering with your timing accuracy.  So the min()
        of the result is probably the only number you should be
        interested in.  After that, you should look at the entire
        vector and apply common sense rather than statistics.
        """
        r = []
        for i in range(repeat):
            t = self.timeit(number)
            r.append(t)
        return r

    def autorange(self, callback=None):
        """Return the number of loops and time taken so that total time >= 0.2.

        Calls the timeit method with increasing numbers from the sequence
        1, 2, 5, 10, 20, 50, ... until the time taken is at least 0.2
        second.  Returns (number, time_taken).

        If *callback* is given and is not None, it will be called after
        each trial with two arguments: ``callback(number, time_taken)``.
        """
        i = 1
        while True:
            for j in 1, 2, 5:
                number = i * j
                time_taken = self.timeit(number)
                if callback:
                    callback(number, time_taken)
                if time_taken >= 0.2:
                    return (number, time_taken)
            i *= 10

def timeit(stmt="pass", setup="pass", timer=default_timer,
           number=default_number, globals=None):
    """Convenience function to create Timer object and call timeit method."""
    return Timer(stmt, setup, timer, globals).timeit(number)

def repeat(stmt="pass", setup="pass", timer=default_timer,
           repeat=default_repeat, number=default_number, globals=None):
    """Convenience function to create Timer object and call repeat method."""
    return Timer(stmt, setup, timer, globals).repeat(repeat, number)

def main(args=None, *, _wrap_timer=None):
    """Main program, used when run as a script.

    The optional 'args' argument specifies the command line to be parsed,
    defaulting to sys.argv[1:].

    The return value is an exit code to be passed to sys.exit(); it
    may be None to indicate success.

    When an exception happens during timing, a traceback is printed to
    stderr and the return value is 1.  Exceptions at other times
    (including the template compilation) are not caught.

    '_wrap_timer' is an internal interface used for unit testing.  If it
    is not None, it must be a callable that accepts a timer function
    and returns another timer function (used for unit testing).
    """
    if args is None:
        args = sys.argv[1:]
    import getopt
    try:
        opts, args = getopt.getopt(args, "n:u:s:r:tcpvh",
                                   ["number=", "setup=", "repeat=",
                                    "time", "clock", "process",
                                    "verbose", "unit=", "help"])
    except getopt.error as err:
        print(err)
        print("use -h/--help for command line help")
        return 2

    timer = default_timer
    stmt = "\n".join(args) or "pass"
    number = 0 # auto-determine
    setup = []
    repeat = default_repeat
    verbose = 0
    time_unit = None
    units = {"nsec": 1e-9, "usec": 1e-6, "msec": 1e-3, "sec": 1.0}
    precision = 3
    for o, a in opts:
        if o in ("-n", "--number"):
            number = int(a)
        if o in ("-s", "--setup"):
            setup.append(a)
        if o in ("-u", "--unit"):
            if a in units:
                time_unit = a
            else:
                print("Unrecognized unit. Please select nsec, usec, msec, or sec.",
                    file=sys.stderr)
                return 2
        if o in ("-r", "--repeat"):
            repeat = int(a)
            if repeat <= 0:
                repeat = 1
        if o in ("-p", "--process"):
            timer = time.process_time
        if o in ("-v", "--verbose"):
            if verbose:
                precision += 1
            verbose += 1
        if o in ("-h", "--help"):
            print(__doc__, end=' ')
            return 0
    setup = "\n".join(setup) or "pass"

    # Include the current directory, so that local imports work (sys.path
    # contains the directory of this script, rather than the current
    # directory)
    import os
    sys.path.insert(0, os.curdir)
    if _wrap_timer is not None:
        timer = _wrap_timer(timer)

    t = Timer(stmt, setup, timer)
    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        callback = None
        if verbose:
            def callback(number, time_taken):
                msg = "{num} loop{s} -> {secs:.{prec}g} secs"
                plural = (number != 1)
                print(msg.format(num=number, s='s' if plural else '',
                                  secs=time_taken, prec=precision))
        try:
            number, _ = t.autorange(callback)
        except:
            t.print_exc()
            return 1

        if verbose:
            print()

    try:
        raw_timings = t.repeat(repeat, number)
    except:
        t.print_exc()
        return 1

    def format_time(dt):
        unit = time_unit

        if unit is not None:
            scale = units[unit]
        else:
            scales = [(scale, unit) for unit, scale in units.items()]
            scales.sort(reverse=True)
            for scale, unit in scales:
                if dt >= scale:
                    break

        return "%.*g %s" % (precision, dt / scale, unit)

    if verbose:
        print("raw times: %s" % ", ".join(map(format_time, raw_timings)))
        print()
    timings = [dt / number for dt in raw_timings]

    best = min(timings)
    print("%d loop%s, best of %d: %s per loop"
          % (number, 's' if number != 1 else '',
             repeat, format_time(best)))

    best = min(timings)
    worst = max(timings)
    if worst >= best * 4:
        import warnings
        warnings.warn_explicit("The test results are likely unreliable. "
                               "The worst time (%s) was more than four times "
                               "slower than the best time (%s)."
                               % (format_time(worst), format_time(best)),
                               UserWarning, '', 0)
    return None

if __name__ == "__main__":
    sys.exit(main())
<|endoftext|>"""Generate cryptographically strong pseudo-random numbers suitable for
managing secrets such as account authentication, tokens, and similar.

See PEP 506 for more information.
https://www.python.org/dev/peps/pep-0506/

"""

__all__ = ['choice', 'randbelow', 'randbits', 'SystemRandom',
           'token_bytes', 'token_hex', 'token_urlsafe',
           'compare_digest',
           ]


import base64
import binascii

from hmac import compare_digest
from random import SystemRandom

_sysrand = SystemRandom()

randbits = _sysrand.getrandbits
choice = _sysrand.choice

def randbelow(exclusive_upper_bound):
    """Return a random int in the range [0, n)."""
    if exclusive_upper_bound <= 0:
        raise ValueError("Upper bound must be positive.")
    return _sysrand._randbelow(exclusive_upper_bound)

DEFAULT_ENTROPY = 32  # number of bytes to return by default

def token_bytes(nbytes=None):
    """Return a random byte string containing *nbytes* bytes.

    If *nbytes* is ``None`` or not supplied, a reasonable
    default is used.

    >>> token_bytes(16)  #doctest:+SKIP
    b'\\xebr\\x17D*t\\xae\\xd4\\xe3S\\xb6\\xe2\\xebP1\\x8b'

    """
    if nbytes is None:
        nbytes = DEFAULT_ENTROPY
    return _sysrand.randbytes(nbytes)

def token_hex(nbytes=None):
    """Return a random text string, in hexadecimal.

    The string has *nbytes* random bytes, each byte converted to two
    hex digits.  If *nbytes* is ``None`` or not supplied, a reasonable
    default is used.

    >>> token_hex(16)  #doctest:+SKIP
    'f9bf78b9a18ce6d46a0cd2b0b86df9da'

    """
    return binascii.hexlify(token_bytes(nbytes)).decode('ascii')

def token_urlsafe(nbytes=None):
    """Return a random URL-safe text string, in Base64 encoding.

    The string has *nbytes* random bytes.  If *nbytes* is ``None``
    or not supplied, a reasonable default is used.

    >>> token_urlsafe(16)  #doctest:+SKIP
    'Drmhze6EPcv0fN_81Bj-nA'

    """
    tok = token_bytes(nbytes)
    return base64.urlsafe_b64encode(tok).rstrip(b'=').decode('ascii')
<|endoftext|>"""Conversion functions between RGB and other color systems.

This modules provides two functions for each color system ABC:

  rgb_to_abc(r, g, b) --> a, b, c
  abc_to_rgb(a, b, c) --> r, g, b

All inputs and outputs are triples of floats in the range [0.0...1.0]
(with the exception of I and Q, which covers a slightly larger range).
Inputs outside the valid range may cause exceptions or invalid outputs.

Supported color systems:
RGB: Red, Green, Blue components
YIQ: Luminance, Chrominance (used by composite video signals)
HLS: Hue, Luminance, Saturation
HSV: Hue, Saturation, Value
"""

# References:
# http://en.wikipedia.org/wiki/YIQ
# http://en.wikipedia.org/wiki/HLS_color_space
# http://en.wikipedia.org/wiki/HSV_color_space

__all__ = ["rgb_to_yiq","yiq_to_rgb","rgb_to_hls","hls_to_rgb",
           "rgb_to_hsv","hsv_to_rgb"]

# Some floating point constants

ONE_THIRD = 1.0/3.0
ONE_SIXTH = 1.0/6.0
TWO_THIRD = 2.0/3.0

# YIQ: used by composite video signals (linear combinations of RGB)
# Y: perceived grey level (0.0 == black, 1.0 == white)
# I, Q: color components
#
# There are a great many versions of the constants used in these formulae.
# The ones in this library uses constants from the FCC version of NTSC.

def rgb_to_yiq(r, g, b):
    y = 0.30*r + 0.59*g + 0.11*b
    i = 0.74*(r-y) - 0.27*(b-y)
    q = 0.48*(r-y) + 0.41*(b-y)
    return (y, i, q)

def yiq_to_rgb(y, i, q):
    # r = y + (0.27*q + 0.41*i) / (0.74*0.41 + 0.27*0.48)
    # b = y + (0.74*q - 0.48*i) / (0.74*0.41 + 0.27*0.48)
    # g = y - (0.30*(r-y) + 0.11*(b-y)) / 0.59

    r = y + 0.9468822170900693*i + 0.6235565819861433*q
    g = y - 0.27478764629897834*i - 0.6356910791873801*q
    b = y - 1.1085450346420322*i + 1.7090069284064666*q

    if r < 0.0:
        r = 0.0
    if g < 0.0:
        g = 0.0
    if b < 0.0:
        b = 0.0
    if r > 1.0:
        r = 1.0
    if g > 1.0:
        g = 1.0
    if b > 1.0:
        b = 1.0
    return (r, g, b)


# HLS: Hue, Luminance, Saturation
# H: position in the spectrum
# L: color lightness
# S: color saturation

def rgb_to_hls(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    sumc = (maxc+minc)
    rangec = (maxc-minc)
    l = sumc/2.0
    if minc == maxc:
        return 0.0, l, 0.0
    if l <= 0.5:
        s = rangec / sumc
    else:
        s = rangec / (2.0-sumc)
    rc = (maxc-r) / rangec
    gc = (maxc-g) / rangec
    bc = (maxc-b) / rangec
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, l, s

def hls_to_rgb(h, l, s):
    if s == 0.0:
        return l, l, l
    if l <= 0.5:
        m2 = l * (1.0+s)
    else:
        m2 = l+s-(l*s)
    m1 = 2.0*l - m2
    return (_v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD))

def _v(m1, m2, hue):
    hue = hue % 1.0
    if hue < ONE_SIXTH:
        return m1 + (m2-m1)*hue*6.0
    if hue < 0.5:
        return m2
    if hue < TWO_THIRD:
        return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
    return m1


# HSV: Hue, Saturation, Value
# H: position in the spectrum
# S: color saturation ("purity")
# V: color brightness

def rgb_to_hsv(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0) # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    # Cannot get here
<|endoftext|>"""Get useful information from live Python objects.

This module encapsulates the interface provided by the internal special
attributes (co_*, im_*, tb_*, etc.) in a friendlier fashion.
It also provides some help for examining source code and class layout.

Here are some of the useful functions provided by this module:

    ismodule(), isclass(), ismethod(), isfunction(), isgeneratorfunction(),
        isgenerator(), istraceback(), isframe(), iscode(), isbuiltin(),
        isroutine() - check object types
    getmembers() - get members of an object that satisfy a given condition

    getfile(), getsourcefile(), getsource() - find an object's source code
    getdoc(), getcomments() - get documentation on an object
    getmodule() - determine the module that an object came from
    getclasstree() - arrange classes so as to represent their hierarchy

    getargvalues(), getcallargs() - get info about function arguments
    getfullargspec() - same, with support for Python 3 features
    formatargvalues() - format an argument spec
    getouterframes(), getinnerframes() - get info about frames
    currentframe() - get the current stack frame
    stack(), trace() - get info about frames on the stack or in a traceback

    signature() - get a Signature object for the callable

    get_annotations() - safely compute an object's annotations
"""

# This module is in the public domain.  No warranties.

__author__ = ('Ka-Ping Yee <ping@lfw.org>',
              'Yury Selivanov <yselivanov@sprymix.com>')

import abc
import ast
import dis
import collections.abc
import enum
import importlib.machinery
import itertools
import linecache
import os
import re
import sys
import tokenize
import token
import types
import warnings
import functools
import builtins
from operator import attrgetter
from collections import namedtuple, OrderedDict

# Create constants for the compiler flags in Include/code.h
# We try to get them from dis to avoid duplication
mod_dict = globals()
for k, v in dis.COMPILER_FLAG_NAMES.items():
    mod_dict["CO_" + v] = k

# See Include/object.h
TPFLAGS_IS_ABSTRACT = 1 << 20


def get_annotations(obj, *, globals=None, locals=None, eval_str=False):
    """Compute the annotations dict for an object.

    obj may be a callable, class, or module.
    Passing in an object of any other type raises TypeError.

    Returns a dict.  get_annotations() returns a new dict every time
    it's called; calling it twice on the same object will return two
    different but equivalent dicts.

    This function handles several details for you:

      * If eval_str is true, values of type str will
        be un-stringized using eval().  This is intended
        for use with stringized annotations
        ("from __future__ import annotations").
      * If obj doesn't have an annotations dict, returns an
        empty dict.  (Functions and methods always have an
        annotations dict; classes, modules, and other types of
        callables may not.)
      * Ignores inherited annotations on classes.  If a class
        doesn't have its own annotations dict, returns an empty dict.
      * All accesses to object members and dict values are done
        using getattr() and dict.get() for safety.
      * Always, always, always returns a freshly-created dict.

    eval_str controls whether or not values of type str are replaced
    with the result of calling eval() on those values:

      * If eval_str is true, eval() is called on values of type str.
      * If eval_str is false (the default), values of type str are unchanged.

    globals and locals are passed in to eval(); see the documentation
    for eval() for more information.  If either globals or locals is
    None, this function may replace that value with a context-specific
    default, contingent on type(obj):

      * If obj is a module, globals defaults to obj.__dict__.
      * If obj is a class, globals defaults to
        sys.modules[obj.__module__].__dict__ and locals
        defaults to the obj class namespace.
      * If obj is a callable, globals defaults to obj.__globals__,
        although if obj is a wrapped function (using
        functools.update_wrapper()) it is first unwrapped.
    """
    if isinstance(obj, type):
        # class
        obj_dict = getattr(obj, '__dict__', None)
        if obj_dict and hasattr(obj_dict, 'get'):
            ann = obj_dict.get('__annotations__', None)
            if isinstance(ann, types.GetSetDescriptorType):
                ann = None
        else:
            ann = None

        obj_globals = None
        module_name = getattr(obj, '__module__', None)
        if module_name:
            module = sys.modules.get(module_name, None)
            if module:
                obj_globals = getattr(module, '__dict__', None)
        obj_locals = dict(vars(obj))
        unwrap = obj
    elif isinstance(obj, types.ModuleType):
        # module
        ann = getattr(obj, '__annotations__', None)
        obj_globals = getattr(obj, '__dict__')
        obj_locals = None
        unwrap = None
    elif callable(obj):
        # this includes types.Function, types.BuiltinFunctionType,
        # types.BuiltinMethodType, functools.partial, functools.singledispatch,
        # "class funclike" from Lib/test/test_inspect... on and on it goes.
        ann = getattr(obj, '__annotations__', None)
        obj_globals = getattr(obj, '__globals__', None)
        obj_locals = None
        unwrap = obj
    else:
        raise TypeError(f"{obj!r} is not a module, class, or callable.")

    if ann is None:
        return {}

    if not isinstance(ann, dict):
        raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

    if not ann:
        return {}

    if not eval_str:
        return dict(ann)

    if unwrap is not None:
        while True:
            if hasattr(unwrap, '__wrapped__'):
                unwrap = unwrap.__wrapped__
                continue
            if isinstance(unwrap, functools.partial):
                unwrap = unwrap.func
                continue
            break
        if hasattr(unwrap, "__globals__"):
            obj_globals = unwrap.__globals__

    if globals is None:
        globals = obj_globals
    if locals is None:
        locals = obj_locals

    return_value = {key:
        value if not isinstance(value, str) else eval(value, globals, locals)
        for key, value in ann.items() }
    return return_value


# ----------------------------------------------------------- type-checking
def ismodule(object):
    """Return true if the object is a module.

    Module objects provide these attributes:
        __cached__      pathname to byte compiled file
        __doc__         documentation string
        __file__        filename (missing for built-in modules)"""
    return isinstance(object, types.ModuleType)

def isclass(object):
    """Return true if the object is a class.

    Class objects provide these attributes:
        __doc__         documentation string
        __module__      name of module in which this class was defined"""
    return isinstance(object, type)

def ismethod(object):
    """Return true if the object is an instance method.

    Instance method objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this method was defined
        __func__        function object containing implementation of method
        __self__        instance to which this method is bound"""
    return isinstance(object, types.MethodType)

def ismethoddescriptor(object):
    """Return true if the object is a method descriptor.

    But not if ismethod() or isclass() or isfunction() are true.

    This is new in Python 2.2, and, for example, is true of int.__add__.
    An object passing this test has a __get__ attribute but not a __set__
    attribute, but beyond that the set of attributes varies.  __name__ is
    usually sensible, and __doc__ often is.

    Methods implemented via descriptors that also pass one of the other
    tests return false from the ismethoddescriptor() test, simply because
    the other tests promise more -- you can, e.g., count on having the
    __func__ attribute (etc) when an object passes ismethod()."""
    if isclass(object) or ismethod(object) or isfunction(object):
        # mutual exclusion
        return False
    tp = type(object)
    return hasattr(tp, "__get__") and not hasattr(tp, "__set__")

def isdatadescriptor(object):
    """Return true if the object is a data descriptor.

    Data descriptors have a __set__ or a __delete__ attribute.  Examples are
    properties (defined in Python) and getsets and members (defined in C).
    Typically, data descriptors will also have __name__ and __doc__ attributes
    (properties, getsets, and members have both of these attributes), but this
    is not guaranteed."""
    if isclass(object) or ismethod(object) or isfunction(object):
        # mutual exclusion
        return False
    tp = type(object)
    return hasattr(tp, "__set__") or hasattr(tp, "__delete__")

if hasattr(types, 'MemberDescriptorType'):
    # CPython and equivalent
    def ismemberdescriptor(object):
        """Return true if the object is a member descriptor.

        Member descriptors are specialized descriptors defined in extension
        modules."""
        return isinstance(object, types.MemberDescriptorType)
else:
    # Other implementations
    def ismemberdescriptor(object):
        """Return true if the object is a member descriptor.

        Member descriptors are specialized descriptors defined in extension
        modules."""
        return False

if hasattr(types, 'GetSetDescriptorType'):
    # CPython and equivalent
    def isgetsetdescriptor(object):
        """Return true if the object is a getset descriptor.

        getset descriptors are specialized descriptors defined in extension
        modules."""
        return isinstance(object, types.GetSetDescriptorType)
else:
    # Other implementations
    def isgetsetdescriptor(object):
        """Return true if the object is a getset descriptor.

        getset descriptors are specialized descriptors defined in extension
        modules."""
        return False

def isfunction(object):
    """Return true if the object is a user-defined function.

    Function objects provide these attributes:
        __doc__         documentation string
        __name__        name with which this function was defined
        __code__        code object containing compiled function bytecode
        __defaults__    tuple of any default values for arguments
        __globals__     global namespace in which this function was defined
        __annotations__ dict of parameter annotations
        __kwdefaults__  dict of keyword only parameters with defaults"""
    return isinstance(object, types.FunctionType)

def _has_code_flag(f, flag):
    """Return true if ``f`` is a function (or a method or functools.partial
    wrapper wrapping a function) whose code object has the given ``flag``
    set in its flags."""
    while ismethod(f):
        f = f.__func__
    f = functools._unwrap_partial(f)
    if not (isfunction(f) or _signature_is_functionlike(f)):
        return False
    return bool(f.__code__.co_flags & flag)

def isgeneratorfunction(obj):
    """Return true if the object is a user-defined generator function.

    Generator function objects provide the same attributes as functions.
    See help(isfunction) for a list of attributes."""
    return _has_code_flag(obj, CO_GENERATOR)

def iscoroutinefunction(obj):
    """Return true if the object is a coroutine function.

    Coroutine functions are defined with "async def" syntax.
    """
    return _has_code_flag(obj, CO_COROUTINE)

def isasyncgenfunction(obj):
    """Return true if the object is an asynchronous generator function.

    Asynchronous generator functions are defined with "async def"
    syntax and have "yield" expressions in their body.
    """
    return _has_code_flag(obj, CO_ASYNC_GENERATOR)

def isasyncgen(object):
    """Return true if the object is an asynchronous generator."""
    return isinstance(object, types.AsyncGeneratorType)

def isgenerator(object):
    """Return true if the object is a generator.

    Generator objects provide these attributes:
        __iter__        defined to support iteration over container
        close           raises a new GeneratorExit exception inside the
                        generator to terminate the iteration
        gi_code         code object
        gi_frame        frame object or possibly None once the generator has
                        been exhausted
        gi_running      set to 1 when generator is executing, 0 otherwise
        next            return the next item from the container
        send            resumes the generator and "sends" a value that becomes
                        the result of the current yield-expression
        throw           used to raise an exception inside the generator"""
    return isinstance(object, types.GeneratorType)

def iscoroutine(object):
    """Return true if the object is a coroutine."""
    return isinstance(object, types.CoroutineType)

def isawaitable(object):
    """Return true if object can be passed to an ``await`` expression."""
    return (isinstance(object, types.CoroutineType) or
            isinstance(object, types.GeneratorType) and
                bool(object.gi_code.co_flags & CO_ITERABLE_COROUTINE) or
            isinstance(object, collections.abc.Awaitable))

def istraceback(object):
    """Return true if the object is a traceback.

    Traceback objects provide these attributes:
        tb_frame        frame object at this level
        tb_lasti        index of last attempted instruction in bytecode
        tb_lineno       current line number in Python source code
        tb_next         next inner traceback object (called by this level)"""
    return isinstance(object, types.TracebackType)

def isframe(object):
    """Return true if the object is a frame object.

    Frame objects provide these attributes:
        f_back          next outer frame object (this frame's caller)
        f_builtins      built-in namespace seen by this frame
        f_code          code object being executed in this frame
        f_globals       global namespace seen by this frame
        f_lasti         index of last attempted instruction in bytecode
        f_lineno        current line number in Python source code
        f_locals        local namespace seen by this frame
        f_trace         tracing function for this frame, or None"""
    return isinstance(object, types.FrameType)

def iscode(object):
    """Return true if the object is a code object.

    Code objects provide these attributes:
        co_argcount         number of arguments (not including *, ** args
                            or keyword only arguments)
        co_code             string of raw compiled bytecode
        co_cellvars         tuple of names of cell variables
        co_consts           tuple of constants used in the bytecode
        co_filename         name of file in which this code object was created
        co_firstlineno      number of first line in Python source code
        co_flags            bitmap: 1=optimized | 2=newlocals | 4=*arg | 8=**arg
                            | 16=nested | 32=generator | 64=nofree | 128=coroutine
                            | 256=iterable_coroutine | 512=async_generator
        co_freevars         tuple of names of free variables
        co_posonlyargcount  number of positional only arguments
        co_kwonlyargcount   number of keyword only arguments (not including ** arg)
        co_lnotab           encoded mapping of line numbers to bytecode indices
        co_name             name with which this code object was defined
        co_names            tuple of names other than arguments and function locals
        co_nlocals          number of local variables
        co_stacksize        virtual machine stack space required
        co_varnames         tuple of names of arguments and local variables"""
    return isinstance(object, types.CodeType)

def isbuiltin(object):
    """Return true if the object is a built-in function or method.

    Built-in functions and methods provide these attributes:
        __doc__         documentation string
        __name__        original name of this function or method
        __self__        instance to which a method is bound, or None"""
    return isinstance(object, types.BuiltinFunctionType)

def isroutine(object):
    """Return true if the object is any kind of function or method."""
    return (isbuiltin(object)
            or isfunction(object)
            or ismethod(object)
            or ismethoddescriptor(object))

def isabstract(object):
    """Return true if the object is an abstract base class (ABC)."""
    if not isinstance(object, type):
        return False
    if object.__flags__ & TPFLAGS_IS_ABSTRACT:
        return True
    if not issubclass(type(object), abc.ABCMeta):
        return False
    if hasattr(object, '__abstractmethods__'):
        # It looks like ABCMeta.__new__ has finished running;
        # TPFLAGS_IS_ABSTRACT should have been accurate.
        return False
    # It looks like ABCMeta.__new__ has not finished running yet; we're
    # probably in __init_subclass__. We'll look for abstractmethods manually.
    for name, value in object.__dict__.items():
        if getattr(value, "__isabstractmethod__", False):
            return True
    for base in object.__bases__:
        for name in getattr(base, "__abstractmethods__", ()):
            value = getattr(object, name, None)
            if getattr(value, "__isabstractmethod__", False):
                return True
    return False

def getmembers(object, predicate=None):
    """Return all members of an object as (name, value) pairs sorted by name.
    Optionally, only return members that satisfy a given predicate."""
    if isclass(object):
        mro = (object,) + getmro(object)
    else:
        mro = ()
    results = []
    processed = set()
    names = dir(object)
    # :dd any DynamicClassAttributes to the list of names if object is a class;
    # this may result in duplicate entries if, for example, a virtual
    # attribute with the same name as a DynamicClassAttribute exists
    try:
        for base in object.__bases__:
            for k, v in base.__dict__.items():
                if isinstance(v, types.DynamicClassAttribute):
                    names.append(k)
    except AttributeError:
        pass
    for key in names:
        # First try to get the value via getattr.  Some descriptors don't
        # like calling their __get__ (see bug #1785), so fall back to
        # looking in the __dict__.
        try:
            value = getattr(object, key)
            # handle the duplicate key
            if key in processed:
                raise AttributeError
        except AttributeError:
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                # could be a (currently) missing slot member, or a buggy
                # __dir__; discard and move on
                continue
        if not predicate or predicate(value):
            results.append((key, value))
        processed.add(key)
    results.sort(key=lambda pair: pair[0])
    return results

Attribute = namedtuple('Attribute', 'name kind defining_class object')

def classify_class_attrs(cls):
    """Return list of attribute-descriptor tuples.

    For each name in dir(cls), the return list contains a 4-tuple
    with these elements:

        0. The name (a string).

        1. The kind of attribute this is, one of these strings:
               'class method'    created via classmethod()
               'static method'   created via staticmethod()
               'property'        created via property()
               'method'          any other flavor of method or descriptor
               'data'            not a method

        2. The class which defined this attribute (a class).

        3. The object as obtained by calling getattr; if this fails, or if the
           resulting object does not live anywhere in the class' mro (including
           metaclasses) then the object is looked up in the defining class's
           dict (found by walking the mro).

    If one of the items in dir(cls) is stored in the metaclass it will now
    be discovered and not have None be listed as the class in which it was
    defined.  Any items whose home class cannot be discovered are skipped.
    """

    mro = getmro(cls)
    metamro = getmro(type(cls)) # for attributes stored in the metaclass
    metamro = tuple(cls for cls in metamro if cls not in (type, object))
    class_bases = (cls,) + mro
    all_bases = class_bases + metamro
    names = dir(cls)
    # :dd any DynamicClassAttributes to the list of names;
    # this may result in duplicate entries if, for example, a virtual
    # attribute with the same name as a DynamicClassAttribute exists.
    for base in mro:
        for k, v in base.__dict__.items():
            if isinstance(v, types.DynamicClassAttribute) and v.fget is not None:
                names.append(k)
    result = []
    processed = set()

    for name in names:
        # Get the object associated with the name, and where it was defined.
        # Normal objects will be looked up with both getattr and directly in
        # its class' dict (in case getattr fails [bug #1785], and also to look
        # for a docstring).
        # For DynamicClassAttributes on the second pass we only look in the
        # class's dict.
        #
        # Getting an obj from the __dict__ sometimes reveals more than
        # using getattr.  Static and class methods are dramatic examples.
        homecls = None
        get_obj = None
        dict_obj = None
        if name not in processed:
            try:
                if name == '__dict__':
                    raise Exception("__dict__ is special, don't want the proxy")
                get_obj = getattr(cls, name)
            except Exception as exc:
                pass
            else:
                homecls = getattr(get_obj, "__objclass__", homecls)
                if homecls not in class_bases:
                    # if the resulting object does not live somewhere in the
                    # mro, drop it and search the mro manually
                    homecls = None
                    last_cls = None
                    # first look in the classes
                    for srch_cls in class_bases:
                        srch_obj = getattr(srch_cls, name, None)
                        if srch_obj is get_obj:
                            last_cls = srch_cls
                    # then check the metaclasses
                    for srch_cls in metamro:
                        try:
                            srch_obj = srch_cls.__getattr__(cls, name)
                        except AttributeError:
                            continue
                        if srch_obj is get_obj:
                            last_cls = srch_cls
                    if last_cls is not None:
                        homecls = last_cls
        for base in all_bases:
            if name in base.__dict__:
                dict_obj = base.__dict__[name]
                if homecls not in metamro:
                    homecls = base
                break
        if homecls is None:
            # unable to locate the attribute anywhere, most likely due to
            # buggy custom __dir__; discard and move on
            continue
        obj = get_obj if get_obj is not None else dict_obj
        # Classify the object or its descriptor.
        if isinstance(dict_obj, (staticmethod, types.BuiltinMethodType)):
            kind = "static method"
            obj = dict_obj
        elif isinstance(dict_obj, (classmethod, types.ClassMethodDescriptorType)):
            kind = "class method"
            obj = dict_obj
        elif isinstance(dict_obj, property):
            kind = "property"
            obj = dict_obj
        elif isroutine(obj):
            kind = "method"
        else:
            kind = "data"
        result.append(Attribute(name, kind, homecls, obj))
        processed.add(name)
    return result

# ----------------------------------------------------------- class helpers

def getmro(cls):
    "Return tuple of base classes (including cls) in method resolution order."
    return cls.__mro__

# -------------------------------------------------------- function helpers

def unwrap(func, *, stop=None):
    """Get the object wrapped by *func*.

   Follows the chain of :attr:`__wrapped__` attributes returning the last
   object in the chain.

   *stop* is an optional callback accepting an object in the wrapper chain
   as its sole argument that allows the unwrapping to be terminated early if
   the callback returns a true value. If the callback never returns a true
   value, the last object in the chain is returned as usual. For example,
   :func:`signature` uses this to stop unwrapping if any object in the
   chain has a ``__signature__`` attribute defined.

   :exc:`ValueError` is raised if a cycle is encountered.

    """
    if stop is None:
        def _is_wrapper(f):
            return hasattr(f, '__wrapped__')
    else:
        def _is_wrapper(f):
            return hasattr(f, '__wrapped__') and not stop(f)
    f = func  # remember the original func for error reporting
    # Memoise by id to tolerate non-hashable objects, but store objects to
    # ensure they aren't destroyed, which would allow their IDs to be reused.
    memo = {id(f): f}
    recursion_limit = sys.getrecursionlimit()
    while _is_wrapper(func):
        func = func.__wrapped__
        id_func = id(func)
        if (id_func in memo) or (len(memo) >= recursion_limit):
            raise ValueError('wrapper loop when unwrapping {!r}'.format(f))
        memo[id_func] = func
    return func

# -------------------------------------------------- source code extraction
def indentsize(line):
    """Return the indent size, in spaces, at the start of a line of text."""
    expline = line.expandtabs()
    return len(expline) - len(expline.lstrip())

def _findclass(func):
    cls = sys.modules.get(func.__module__)
    if cls is None:
        return None
    for name in func.__qualname__.split('.')[:-1]:
        cls = getattr(cls, name)
    if not isclass(cls):
        return None
    return cls

def _finddoc(obj):
    if isclass(obj):
        for base in obj.__mro__:
            if base is not object:
                try:
                    doc = base.__doc__
                except AttributeError:
                    continue
                if doc is not None:
                    return doc
        return None

    if ismethod(obj):
        name = obj.__func__.__name__
        self = obj.__self__
        if (isclass(self) and
            getattr(getattr(self, name, None), '__func__') is obj.__func__):
            # classmethod
            cls = self
        else:
            cls = self.__class__
    elif isfunction(obj):
        name = obj.__name__
        cls = _findclass(obj)
        if cls is None or getattr(cls, name) is not obj:
            return None
    elif isbuiltin(obj):
        name = obj.__name__
        self = obj.__self__
        if (isclass(self) and
            self.__qualname__ + '.' + name == obj.__qualname__):
            # classmethod
            cls = self
        else:
            cls = self.__class__
    # Should be tested before isdatadescriptor().
    elif isinstance(obj, property):
        func = obj.fget
        name = func.__name__
        cls = _findclass(func)
        if cls is None or getattr(cls, name) is not obj:
            return None
    elif ismethoddescriptor(obj) or isdatadescriptor(obj):
        name = obj.__name__
        cls = obj.__objclass__
        if getattr(cls, name) is not obj:
            return None
        if ismemberdescriptor(obj):
            slots = getattr(cls, '__slots__', None)
            if isinstance(slots, dict) and name in slots:
                return slots[name]
    else:
        return None
    for base in cls.__mro__:
        try:
            doc = getattr(base, name).__doc__
        except AttributeError:
            continue
        if doc is not None:
            return doc
    return None

def getdoc(object):
    """Get the documentation string for an object.

    All tabs are expanded to spaces.  To clean up docstrings that are
    indented to line up with blocks of code, any whitespace than can be
    uniformly removed from the second line onwards is removed."""
    try:
        doc = object.__doc__
    except AttributeError:
        return None
    if doc is None:
        try:
            doc = _finddoc(object)
        except (AttributeError, TypeError):
            return None
    if not isinstance(doc, str):
        return None
    return cleandoc(doc)

def cleandoc(doc):
    """Clean up indentation from docstrings.

    Any whitespace that can be uniformly removed from the second line
    onwards is removed."""
    try:
        lines = doc.expandtabs().split('\n')
    except UnicodeError:
        return None
    else:
        # Find minimum indentation of any non-blank lines after first line.
        margin = sys.maxsize
        for line in lines[1:]:
            content = len(line.lstrip())
            if content:
                indent = len(line) - content
                margin = min(margin, indent)
        # Remove indentation.
        if lines:
            lines[0] = lines[0].lstrip()
        if margin < sys.maxsize:
            for i in range(1, len(lines)): lines[i] = lines[i][margin:]
        # Remove any trailing or leading blank lines.
        while lines and not lines[-1]:
            lines.pop()
        while lines and not lines[0]:
            lines.pop(0)
        return '\n'.join(lines)

def getfile(object):
    """Work out which source or compiled file an object was defined in."""
    if ismodule(object):
        if getattr(object, '__file__', None):
            return object.__file__
        raise TypeError('{!r} is a built-in module'.format(object))
    if isclass(object):
        if hasattr(object, '__module__'):
            module = sys.modules.get(object.__module__)
            if getattr(module, '__file__', None):
                return module.__file__
            if object.__module__ == '__main__':
                raise OSError('source code not available')
        raise TypeError('{!r} is a built-in class'.format(object))
    if ismethod(object):
        object = object.__func__
    if isfunction(object):
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        return object.co_filename
    raise TypeError('module, class, method, function, traceback, frame, or '
                    'code object was expected, got {}'.format(
                    type(object).__name__))

def getmodulename(path):
    """Return the module name for a given file, or None."""
    fname = os.path.basename(path)
    # Check for paths that look like an actual module file
    suffixes = [(-len(suffix), suffix)
                    for suffix in importlib.machinery.all_suffixes()]
    suffixes.sort() # try longest suffixes first, in case they overlap
    for neglen, suffix in suffixes:
        if fname.endswith(suffix):
            return fname[:neglen]
    return None

def getsourcefile(object):
    """Return the filename that can be used to locate an object's source.
    Return None if no way can be identified to get the source.
    """
    filename = getfile(object)
    all_bytecode_suffixes = importlib.machinery.DEBUG_BYTECODE_SUFFIXES[:]
    all_bytecode_suffixes += importlib.machinery.OPTIMIZED_BYTECODE_SUFFIXES[:]
    if any(filename.endswith(s) for s in all_bytecode_suffixes):
        filename = (os.path.splitext(filename)[0] +
                    importlib.machinery.SOURCE_SUFFIXES[0])
    elif any(filename.endswith(s) for s in
                 importlib.machinery.EXTENSION_SUFFIXES):
        return None
    if os.path.exists(filename):
        return filename
    # only return a non-existent filename if the module has a PEP 302 loader
    module = getmodule(object, filename)
    if getattr(module, '__loader__', None) is not None:
        return filename
    elif getattr(getattr(module, "__spec__", None), "loader", None) is not None:
        return filename
    # or it is in the linecache
    elif filename in linecache.cache:
        return filename

def getabsfile(object, _filename=None):
    """Return an absolute path to the source or compiled file for an object.

    The idea is for each object to have a unique origin, so this routine
    normalizes the result as much as possible."""
    if _filename is None:
        _filename = getsourcefile(object) or getfile(object)
    return os.path.normcase(os.path.abspath(_filename))

modulesbyfile = {}
_filesbymodname = {}

def getmodule(object, _filename=None):
    """Return the module an object was defined in, or None if not found."""
    if ismodule(object):
        return object
    if hasattr(object, '__module__'):
        return sys.modules.get(object.__module__)
    # Try the filename to modulename cache
    if _filename is not None and _filename in modulesbyfile:
        return sys.modules.get(modulesbyfile[_filename])
    # Try the cache again with the absolute file name
    try:
        file = getabsfile(object, _filename)
    except (TypeError, FileNotFoundError):
        return None
    if file in modulesbyfile:
        return sys.modules.get(modulesbyfile[file])
    # Update the filename to module name cache and check yet again
    # Copy sys.modules in order to cope with changes while iterating
    for modname, module in sys.modules.copy().items():
        if ismodule(module) and hasattr(module, '__file__'):
            f = module.__file__
            if f == _filesbymodname.get(modname, None):
                # Have already mapped this module, so skip it
                continue
            _filesbymodname[modname] = f
            f = getabsfile(module)
            # Always map to the name the module knows itself by
            modulesbyfile[f] = modulesbyfile[
                os.path.realpath(f)] = module.__name__
    if file in modulesbyfile:
        return sys.modules.get(modulesbyfile[file])
    # Check the main module
    main = sys.modules['__main__']
    if not hasattr(object, '__name__'):
        return None
    if hasattr(main, object.__name__):
        mainobject = getattr(main, object.__name__)
        if mainobject is object:
            return main
    # Check builtins
    builtin = sys.modules['builtins']
    if hasattr(builtin, object.__name__):
        builtinobject = getattr(builtin, object.__name__)
        if builtinobject is object:
            return builtin


class ClassFoundException(Exception):
    pass


class _ClassFinder(ast.NodeVisitor):

    def __init__(self, qualname):
        self.stack = []
        self.qualname = qualname

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.stack.append('<locals>')
        self.generic_visit(node)
        self.stack.pop()
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        if self.qualname == '.'.join(self.stack):
            # Return the decorator for the class if present
            if node.decorator_list:
                line_number = node.decorator_list[0].lineno
            else:
                line_number = node.lineno

            # decrement by one since lines starts with indexing by zero
            line_number -= 1
            raise ClassFoundException(line_number)
        self.generic_visit(node)
        self.stack.pop()


def findsource(object):
    """Return the entire source file and starting line number for an object.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of all the lines
    in the file and the line number indexes a line in that list.  An OSError
    is raised if the source code cannot be retrieved."""

    file = getsourcefile(object)
    if file:
        # Invalidate cache if needed.
        linecache.checkcache(file)
    else:
        file = getfile(object)
        # Allow filenames in form of "<something>" to pass through.
        # `doctest` monkeypatches `linecache` module to enable
        # inspection, so let `linecache.getlines` to be called.
        if not (file.startswith('<') and file.endswith('>')):
            raise OSError('source code not available')

    module = getmodule(object, file)
    if module:
        lines = linecache.getlines(file, module.__dict__)
    else:
        lines = linecache.getlines(file)
    if not lines:
        raise OSError('could not get source code')

    if ismodule(object):
        return lines, 0

    if isclass(object):
        qualname = object.__qualname__
        source = ''.join(lines)
        tree = ast.parse(source)
        class_finder = _ClassFinder(qualname)
        try:
            class_finder.visit(tree)
        except ClassFoundException as e:
            line_number = e.args[0]
            return lines, line_number
        else:
            raise OSError('could not find class definition')

    if ismethod(object):
        object = object.__func__
    if isfunction(object):
        object = object.__code__
    if istraceback(object):
        object = object.tb_frame
    if isframe(object):
        object = object.f_code
    if iscode(object):
        if not hasattr(object, 'co_firstlineno'):
            raise OSError('could not find function definition')
        lnum = object.co_firstlineno - 1
        pat = re.compile(r'^(\s*def\s)|(\s*async\s+def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)')
        while lnum > 0:
            try:
                line = lines[lnum]
            except IndexError:
                raise OSError('lineno is out of bounds')
            if pat.match(line):
                break
            lnum = lnum - 1
        return lines, lnum
    raise OSError('could not find code object')

def getcomments(object):
    """Get lines of comments immediately preceding an object's source code.

    Returns None when source can't be found.
    """
    try:
        lines, lnum = findsource(object)
    except (OSError, TypeError):
        return None

    if ismodule(object):
        # Look for a comment block at the top of the file.
        start = 0
        if lines and lines[0][:2] == '#!': start = 1
        while start < len(lines) and lines[start].strip() in ('', '#'):
            start = start + 1
        if start < len(lines) and lines[start][:1] == '#':
            comments = []
            end = start
            while end < len(lines) and lines[end][:1] == '#':
                comments.append(lines[end].expandtabs())
                end = end + 1
            return ''.join(comments)

    # Look for a preceding block of comments at the same indentation.
    elif lnum > 0:
        indent = indentsize(lines[lnum])
        end = lnum - 1
        if end >= 0 and lines[end].lstrip()[:1] == '#' and \
            indentsize(lines[end]) == indent:
            comments = [lines[end].expandtabs().lstrip()]
            if end > 0:
                end = end - 1
                comment = lines[end].expandtabs().lstrip()
                while comment[:1] == '#' and indentsize(lines[end]) == indent:
                    comments[:0] = [comment]
                    end = end - 1
                    if end < 0: break
                    comment = lines[end].expandtabs().lstrip()
            while comments and comments[0].strip() == '#':
                comments[:1] = []
            while comments and comments[-1].strip() == '#':
                comments[-1:] = []
            return ''.join(comments)

class EndOfBlock(Exception): pass

class BlockFinder:
    """Provide a tokeneater() method to detect the end of a code block."""
    def __init__(self):
        self.indent = 0
        self.islambda = False
        self.started = False
        self.passline = False
        self.indecorator = False
        self.last = 1
        self.body_col0 = None

    def tokeneater(self, type, token, srowcol, erowcol, line):
        if not self.started and not self.indecorator:
            # skip any decorators
            if token == "@":
                self.indecorator = True
            # look for the first "def", "class" or "lambda"
            elif token in ("def", "class", "lambda"):
                if token == "lambda":
                    self.islambda = True
                self.started = True
            self.passline = True    # skip to the end of the line
        elif type == tokenize.NEWLINE:
            self.passline = False   # stop skipping when a NEWLINE is seen
            self.last = srowcol[0]
            if self.islambda:       # lambdas always end at the first NEWLINE
                raise EndOfBlock
            # hitting a NEWLINE when in a decorator without args
            # ends the decorator
            if self.indecorator:
                self.indecorator = False
        elif self.passline:
            pass
        elif type == tokenize.INDENT:
            if self.body_col0 is None and self.started:
                self.body_col0 = erowcol[1]
            self.indent = self.indent + 1
            self.passline = True
        elif type == tokenize.DEDENT:
            self.indent = self.indent - 1
            # the end of matching indent/dedent pairs end a block
            # (note that this only works for "def"/"class" blocks,
            #  not e.g. for "if: else:" or "try: finally:" blocks)
            if self.indent <= 0:
                raise EndOfBlock
        elif type == tokenize.COMMENT:
            if self.body_col0 is not None and srowcol[1] >= self.body_col0:
                # Include comments if indented at least as much as the block
                self.last = srowcol[0]
        elif self.indent == 0 and type not in (tokenize.COMMENT, tokenize.NL):
            # any other token on the same indentation level end the previous
            # block as well, except the pseudo-tokens COMMENT and NL.
            raise EndOfBlock

def getblock(lines):
    """Extract the block of code at the top of the given list of lines."""
    blockfinder = BlockFinder()
    try:
        tokens = tokenize.generate_tokens(iter(lines).__next__)
        for _token in tokens:
            blockfinder.tokeneater(*_token)
    except (EndOfBlock, IndentationError):
        pass
    return lines[:blockfinder.last]

def getsourcelines(object):
    """Return a list of source lines and starting line number for an object.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a list of the lines
    corresponding to the object and the line number indicates where in the
    original source file the first line of code was found.  An OSError is
    raised if the source code cannot be retrieved."""
    object = unwrap(object)
    lines, lnum = findsource(object)

    if istraceback(object):
        object = object.tb_frame

    # for module or frame that corresponds to module, return all source lines
    if (ismodule(object) or
        (isframe(object) and object.f_code.co_name == "<module>")):
        return lines, 0
    else:
        return getblock(lines[lnum:]), lnum + 1

def getsource(object):
    """Return the text of the source code for an object.

    The argument may be a module, class, method, function, traceback, frame,
    or code object.  The source code is returned as a single string.  An
    OSError is raised if the source code cannot be retrieved."""
    lines, lnum = getsourcelines(object)
    return ''.join(lines)

# --------------------------------------------------- class tree extraction
def walktree(classes, children, parent):
    """Recursive helper function for getclasstree()."""
    results = []
    classes.sort(key=attrgetter('__module__', '__name__'))
    for c in classes:
        results.append((c, c.__bases__))
        if c in children:
            results.append(walktree(children[c], children, c))
    return results

def getclasstree(classes, unique=False):
    """Arrange the given list of classes into a hierarchy of nested lists.

    Where a nested list appears, it contains classes derived from the class
    whose entry immediately precedes the list.  Each entry is a 2-tuple
    containing a class and a tuple of its base classes.  If the 'unique'
    argument is true, exactly one entry appears in the returned structure
    for each class in the given list.  Otherwise, classes using multiple
    inheritance and their descendants will appear multiple times."""
    children = {}
    roots = []
    for c in classes:
        if c.__bases__:
            for parent in c.__bases__:
                if parent not in children:
                    children[parent] = []
                if c not in children[parent]:
                    children[parent].append(c)
                if unique and parent in classes: break
        elif c not in roots:
            roots.append(c)
    for parent in children:
        if parent not in classes:
            roots.append(parent)
    return walktree(roots, children, None)

# ------------------------------------------------ argument list extraction
Arguments = namedtuple('Arguments', 'args, varargs, varkw')

def getargs(co):
    """Get information about the arguments accepted by a code object.

    Three things are returned: (args, varargs, varkw), where
    'args' is the list of argument names. Keyword-only arguments are
    appended. 'varargs' and 'varkw' are the names of the * and **
    arguments or None."""
    if not iscode(co):
        raise TypeError('{!r} is not a code object'.format(co))

    names = co.co_varnames
    nargs = co.co_argcount
    nkwargs = co.co_kwonlyargcount
    args = list(names[:nargs])
    kwonlyargs = list(names[nargs:nargs+nkwargs])
    step = 0

    nargs += nkwargs
    varargs = None
    if co.co_flags & CO_VARARGS:
        varargs = co.co_varnames[nargs]
        nargs = nargs + 1
    varkw = None
    if co.co_flags & CO_VARKEYWORDS:
        varkw = co.co_varnames[nargs]
    return Arguments(args + kwonlyargs, varargs, varkw)

ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')

def getargspec(func):
    """Get the names and default values of a function's parameters.

    A tuple of four things is returned: (args, varargs, keywords, defaults).
    'args' is a list of the argument names, including keyword-only argument names.
    'varargs' and 'keywords' are the names of the * and ** parameters or None.
    'defaults' is an n-tuple of the default values of the last n parameters.

    This function is deprecated, as it does not support annotations or
    keyword-only parameters and will raise ValueError if either is present
    on the supplied callable.

    For a more structured introspection API, use inspect.signature() instead.

    Alternatively, use getfullargspec() for an API with a similar namedtuple
    based interface, but full support for annotations and keyword-only
    parameters.

    Deprecated since Python 3.5, use `inspect.getfullargspec()`.
    """
    warnings.warn("inspect.getargspec() is deprecated since Python 3.0, "
                  "use inspect.signature() or inspect.getfullargspec()",
                  DeprecationWarning, stacklevel=2)
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, ann = \
        getfullargspec(func)
    if kwonlyargs or ann:
        raise ValueError("Function has keyword-only parameters or annotations"
                         ", use inspect.signature() API which can support them")
    return ArgSpec(args, varargs, varkw, defaults)

FullArgSpec = namedtuple('FullArgSpec',
    'args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations')

def getfullargspec(func):
    """Get the names and default values of a callable object's parameters.

    A tuple of seven things is returned:
    (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations).
    'args' is a list of the parameter names.
    'varargs' and 'varkw' are the names of the * and ** parameters or None.
    'defaults' is an n-tuple of the default values of the last n parameters.
    'kwonlyargs' is a list of keyword-only parameter names.
    'kwonlydefaults' is a dictionary mapping names from kwonlyargs to defaults.
    'annotations' is a dictionary mapping parameter names to annotations.

    Notable differences from inspect.signature():
      - the "self" parameter is always reported, even for bound methods
      - wrapper chains defined by __wrapped__ *not* unwrapped automatically
    """
    try:
        # Re: `skip_bound_arg=False`
        #
        # There is a notable difference in behaviour between getfullargspec
        # and Signature: the former always returns 'self' parameter for bound
        # methods, whereas the Signature always shows the actual calling
        # signature of the passed object.
        #
        # To simulate this behaviour, we "unbind" bound methods, to trick
        # inspect.signature to always return their first parameter ("self",
        # usually)

        # Re: `follow_wrapper_chains=False`
        #
        # getfullargspec() historically ignored __wrapped__ attributes,
        # so we ensure that remains the case in 3.3+

        sig = _signature_from_callable(func,
                                       follow_wrapper_chains=False,
                                       skip_bound_arg=False,
                                       sigcls=Signature,
                                       eval_str=False)
    except Exception as ex:
        # Most of the times 'signature' will raise ValueError.
        # But, it can also raise AttributeError, and, maybe something
        # else. So to be fully backwards compatible, we catch all
        # possible exceptions here, and reraise a TypeError.
        raise TypeError('unsupported callable') from ex

    args = []
    varargs = None
    varkw = None
    posonlyargs = []
    kwonlyargs = []
    annotations = {}
    defaults = ()
    kwdefaults = {}

    if sig.return_annotation is not sig.empty:
        annotations['return'] = sig.return_annotation

    for param in sig.parameters.values():
        kind = param.kind
        name = param.name

        if kind is _POSITIONAL_ONLY:
            posonlyargs.append(name)
            if param.default is not param.empty:
