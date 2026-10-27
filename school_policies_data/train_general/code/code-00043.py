        req = Request(white_squares, white_types, black_squares, black_types, side)

        # Probe.
        dtm = self._tb_probe(req)
        ply, res = unpackdist(dtm)

        if res == iWMATE:
            # White mates in the stored position.
            if req.realside == 1:
                if req.is_reversed:
                    return ply
                else:
                    return -ply
            else:
                if req.is_reversed:
                    return -ply
                else:
                    return ply
        elif res == iBMATE:
            # Black mates in the stored position.
            if req.realside == 0:
                if req.is_reversed:
                    return ply
                else:
                    return -ply
            else:
                if req.is_reversed:
                    return -ply
                else:
                    return ply
        else:
            # Draw.
            return 0

    def get_dtm(self, board: chess.Board, default: Optional[int] = None) -> Optional[int]:
        try:
            return self.probe_dtm(board)
        except KeyError:
            return default

    def probe_wdl(self, board: chess.Board) -> int:
        """
        Probes for win/draw/loss information.

        Returns ``1`` if the side to move is winning, ``0`` if it is a draw,
        and ``-1`` if the side to move is losing.

        >>> import chess
        >>> import chess.gaviota
        >>>
        >>> with chess.gaviota.open_tablebase("data/gaviota") as tablebase:
        ...     board = chess.Board("8/4k3/8/B7/8/8/8/4K3 w - - 0 1")
        ...     print(tablebase.probe_wdl(board))
        ...
        0

        :raises: :exc:`KeyError` (or specifically
            :exc:`chess.gaviota.MissingTableError`) if the probe fails. Use
            :func:`~chess.gaviota.PythonTablebase.get_wdl()` if you prefer
            to get ``None`` instead of an exception.

            Note that probing a corrupted table file is undefined behavior.
        """
        dtm = self.probe_dtm(board)

        if dtm == 0:
            if board.is_checkmate():
                return -1
            else:
                return 0
        elif dtm > 0:
            return 1
        else:
            return -1

    def get_wdl(self, board: chess.Board, default: Optional[int] = None) -> Optional[int]:
        try:
            return self.probe_wdl(board)
        except KeyError:
            return default

    def _setup_tablebase(self, req: Request) -> BinaryIO:
        white_letters = "".join(chess.piece_symbol(i) for i in req.white_types)
        black_letters = "".join(chess.piece_symbol(i) for i in req.black_types)

        if (white_letters + black_letters) in self.available_tables:
            req.is_reversed = False
            req.egkey = white_letters + black_letters
            req.white_piece_squares = req.white_squares
            req.white_piece_types = req.white_types
            req.black_piece_squares = req.black_squares
            req.black_piece_types = req.black_types
        elif (black_letters + white_letters) in self.available_tables:
            req.is_reversed = True
            req.egkey = black_letters + white_letters
            req.white_piece_squares = [flip_ns(s) for s in req.black_squares]
            req.white_piece_types = req.black_types
            req.black_piece_squares = [flip_ns(s) for s in req.white_squares]
            req.black_piece_types = req.white_types
            req.side = opp(req.side)
        else:
            raise MissingTableError(f"no gaviota table available for: {white_letters.upper()}v{black_letters.upper()}")

        return self._open_tablebase(req)

    def _open_tablebase(self, req: Request) -> BinaryIO:
        stream = self.streams.get(req.egkey)

        if stream is None:
            path = self.available_tables[req.egkey]
            stream = open(path, "rb+")
            self.egtb_loadindexes(req.egkey, stream)
            self.streams[req.egkey] = stream

        return stream

    def close(self) -> None:
        """Closes all loaded tables."""
        self.available_tables.clear()

        self.zipinfo.clear()

        self.block_age = 0
        self.block_cache.clear()

        while self.streams:
            _, stream = self.streams.popitem()
            stream.close()

    def egtb_block_getnumber(self, req: Request, idx: int) -> int:
        maxindex = EGKEY[req.egkey].maxindex

        blocks_per_side = 1 + (maxindex - 1) // ENTRIES_PER_BLOCK
        block_in_side = idx // ENTRIES_PER_BLOCK

        return req.side * blocks_per_side + block_in_side

    def egtb_block_getsize(self, req: Request, idx: int) -> int:
        blocksz = ENTRIES_PER_BLOCK
        maxindex = EGKEY[req.egkey].maxindex
        block = idx // blocksz
        offset = block * blocksz

        if (offset + blocksz) > maxindex:
            return maxindex - offset  # Last block size
        else:
            return blocksz  # Size of a normal block

    def _tb_probe(self, req: Request) -> int:
        stream = self._setup_tablebase(req)
        idx = EGKEY[req.egkey].pctoi(req)
        offset, remainder = split_index(idx)

        t = self.block_cache.get((req.egkey, offset, req.side))

        if t is None:
            t = TableBlock(req.egkey, req.side, offset, self.block_age)

            block = self.egtb_block_getnumber(req, idx)
            n = self.egtb_block_getsize(req, idx)
            z = self.egtb_block_getsize_zipped(req.egkey, block)

            self.egtb_block_park(req.egkey, block, stream)
            buffer_zipped = stream.read(z)

            if buffer_zipped[0] == 0:
                # If flag is zero, plain LZMA is following.
                buffer_zipped = buffer_zipped[2:]
            else:
                # Else LZMA86. Build a fake header.
                DICTIONARY_SIZE = 4096
                POS_STATE_BITS = 2
                NUM_LITERAL_POS_STATE_BITS = 0
                NUM_LITERAL_CONTEXT_BITS = 3
                properties = bytearray(13)
                properties[0] = (POS_STATE_BITS * 5 + NUM_LITERAL_POS_STATE_BITS) * 9 + NUM_LITERAL_CONTEXT_BITS
                for i in range(4):
                    properties[1 + i] = (DICTIONARY_SIZE >> (8 * i)) & 0xFF
                for i in range(8):
                    properties[5 + i] = (n >> (8 * i)) & 0xFF

                # Concatenate the fake header with the true LZMA stream.
                buffer_zipped = properties + buffer_zipped[15:]

            buffer_packed = lzma.LZMADecompressor().decompress(buffer_zipped)

            t.pcache = egtb_block_unpack(req.side, n, buffer_packed)

            # Update LRU block cache.
            self.block_cache[(t.egkey, t.offset, t.side)] = t
            if len(self.block_cache) > 128:
                lru_cache_key = min(self.block_cache, key=lambda cache_key: self.block_cache[cache_key].age)
                del self.block_cache[lru_cache_key]
        else:
            t.age = self.block_age

        self.block_age += 1
        dtm = t.pcache[remainder]

        return dtm

    def egtb_loadindexes(self, egkey: str, stream: BinaryIO) -> ZipInfo:
        zipinfo = self.zipinfo.get(egkey)

        if zipinfo is None:
            # Get reserved bytes, blocksize, offset.
            stream.seek(0)
            HeaderStruct = struct.Struct("<10I")
            header = HeaderStruct.unpack(stream.read(HeaderStruct.size))
            offset = header[8]

            blocks = ((offset - 40) // 4) - 1
            n_idx = blocks + 1

            IndexStruct = struct.Struct("<" + "I" * n_idx)
            p = list(IndexStruct.unpack(stream.read(IndexStruct.size)))

            zipinfo = ZipInfo(extraoffset=0, totalblocks=n_idx, blockindex=p)
            self.zipinfo[egkey] = zipinfo

        return zipinfo

    def egtb_block_getsize_zipped(self, egkey: str, block: int) -> int:
        i = self.zipinfo[egkey].blockindex[block]
        j = self.zipinfo[egkey].blockindex[block + 1]
        return j - i

    def egtb_block_park(self, egkey: str, block: int, stream: BinaryIO) -> int:
        i = self.zipinfo[egkey].blockindex[block]
        i += self.zipinfo[egkey].extraoffset
        stream.seek(i)
        return i

    def __enter__(self) -> PythonTablebase:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()


class NativeTablebase:
    """
    Provides access to Gaviota tablebases via the shared library libgtb.
    Has the same interface as :class:`~chess.gaviota.PythonTablebase`.
    """

    def __init__(self, libgtb: ctypes.CDLL) -> None:
        self.paths: List[str] = []

        self.libgtb = libgtb
        self.libgtb.tb_init.restype = ctypes.c_char_p
        self.libgtb.tb_restart.restype = ctypes.c_char_p
        self.libgtb.tbpaths_getmain.restype = ctypes.c_char_p
        self.libgtb.tb_probe_hard.argtypes = [
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint)
        ]

        if self.libgtb.tb_is_initialized():
            raise RuntimeError("only one gaviota instance can be initialized at a time")

        self._tbcache_restart(1024 * 1024, 50)

    def add_directory(self, directory: str) -> None:
        if not os.path.isdir(directory):
            raise IOError(f"not a directory: {directory!r}")

        self.paths.append(directory)
        self._tb_restart()

    def _tb_restart(self) -> None:
        self.c_paths = (ctypes.c_char_p * len(self.paths))()
        self.c_paths[:] = [path.encode("utf-8") for path in self.paths]

        verbosity = ctypes.c_int(1)
        compression_scheme = ctypes.c_int(4)

        ret = self.libgtb.tb_restart(verbosity, compression_scheme, self.c_paths)
        if ret:
            LOGGER.debug(ret.decode("utf-8"))

        LOGGER.debug("Main path has been set to %r", self.libgtb.tbpaths_getmain().decode("utf-8"))

        av = self.libgtb.tb_availability()
        if av & 1:
            LOGGER.debug("Some 3-piece tables available")
        if av & 2:
            LOGGER.debug("All 3-piece tables complete")
        if av & 4:
            LOGGER.debug("Some 4-piece tables available")
        if av & 8:
            LOGGER.debug("All 4-piece tables complete")
        if av & 16:
            LOGGER.debug("Some 5-piece tables available")
        if av & 32:
            LOGGER.debug("All 5-piece tables complete")

    def _tbcache_restart(self, cache_mem: int, wdl_fraction: int) -> None:
        self.libgtb.tbcache_restart(ctypes.c_size_t(cache_mem), ctypes.c_int(wdl_fraction))

    def probe_dtm(self, board: chess.Board) -> int:
        return self._probe_hard(board)

    def probe_wdl(self, board: chess.Board) -> int:
        return self._probe_hard(board, wdl_only=True)

    def get_dtm(self, board: chess.Board, default: Optional[int] = None) -> Optional[int]:
        try:
            return self.probe_dtm(board)
        except KeyError:
            return default

    def get_wdl(self, board: chess.Board, default: Optional[int] = None) -> Optional[int]:
        try:
            return self.probe_wdl(board)
        except KeyError:
            return default

    def _probe_hard(self, board: chess.Board, wdl_only: bool = False) -> int:
        if board.is_insufficient_material():
            return 0

        if board.castling_rights:
            raise KeyError(f"gaviota tables do not contain positions with castling rights: {board.fen()}")

        if chess.popcount(board.occupied) > 5:
            raise KeyError(f"gaviota tables support up to 5 pieces, not {chess.popcount(board.occupied)}: {board.fen()}")

        stm = ctypes.c_uint(0 if board.turn == chess.WHITE else 1)
        ep_square = ctypes.c_uint(board.ep_square if board.ep_square else 64)
        castling = ctypes.c_uint(0)

        c_ws = (ctypes.c_uint * 17)()
        c_wp = (ctypes.c_ubyte * 17)()

        i = -1
        for i, square in enumerate(chess.SquareSet(board.occupied_co[chess.WHITE])):
            c_ws[i] = square
            c_wp[i] = typing.cast(chess.PieceType, board.piece_type_at(square))

        c_ws[i + 1] = 64
        c_wp[i + 1] = 0

        c_bs = (ctypes.c_uint * 17)()
        c_bp = (ctypes.c_ubyte * 17)()

        i = -1
        for i, square in enumerate(chess.SquareSet(board.occupied_co[chess.BLACK])):
            c_bs[i] = square
            c_bp[i] = typing.cast(chess.PieceType, board.piece_type_at(square))

        c_bs[i + 1] = 64
        c_bp[i + 1] = 0

        # Do a hard probe.
        info = ctypes.c_uint()
        pliestomate = ctypes.c_uint()
        if not wdl_only:
            ret = self.libgtb.tb_probe_hard(stm, ep_square, castling, c_ws, c_bs, c_wp, c_bp, ctypes.byref(info), ctypes.byref(pliestomate))
            dtm = int(pliestomate.value)
        else:
            ret = self.libgtb.tb_probe_WDL_hard(stm, ep_square, castling, c_ws, c_bs, c_wp, c_bp, ctypes.byref(info))
            dtm = 1

        # Probe forbidden.
        if info.value == 3:
            raise MissingTableError(f"gaviota table for {board.fen()} not available")

        # Draw.
        if ret and info.value == 0:
            return 0

        # White mates.
        if ret and info.value == 1:
            return dtm if board.turn == chess.WHITE else -dtm

        # Black mates.
        if ret and info.value == 2:
            return dtm if board.turn == chess.BLACK else -dtm

        raise KeyError(f"gaviota probe failed for {board.fen()}")

    def close(self) -> None:
        self.paths = []

        if self.libgtb.tb_is_initialized():
            self.libgtb.tbcache_done()
            self.libgtb.tb_done()

    def __enter__(self) -> NativeTablebase:
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.close()


def open_tablebase_native(directory: str, *, libgtb: Optional[str] = None, LibraryLoader: ctypes.LibraryLoader[ctypes.CDLL] = ctypes.cdll) -> NativeTablebase:
    """
    Opens a collection of tables for probing using libgtb.

    In most cases :func:`~chess.gaviota.open_tablebase()` should be used.
    Use this function only if you do not want to downgrade to pure Python
    tablebase probing.

    :raises: :exc:`RuntimeError` or :exc:`OSError` when libgtb can not be used.
    """
    libgtb = libgtb or ctypes.util.find_library("gtb") or "libgtb.so.1.0.1"
    tables = NativeTablebase(LibraryLoader.LoadLibrary(libgtb))
    tables.add_directory(directory)
    return tables


def open_tablebase(directory: str, *, libgtb: Optional[str] = None, LibraryLoader: ctypes.LibraryLoader[ctypes.CDLL] = ctypes.cdll) -> Union[NativeTablebase, PythonTablebase]:
    """
    Opens a collection of tables for probing.

    First native access via the shared library libgtb is tried. You can
    optionally provide a specific library name or a library loader.
    The shared library has global state and caches, so only one instance can
    be open at a time.

    Second, pure Python probing code is tried.
    """
    try:
        if LibraryLoader:
            return open_tablebase_native(directory, libgtb=libgtb, LibraryLoader=LibraryLoader)
    except (OSError, RuntimeError) as err:
        LOGGER.info("Falling back to pure Python tablebase: %r", err)

    tables = PythonTablebase()
    tables.add_directory(directory)
    return tables
<|endoftext|>"""Module for reading and writing AFM (Adobe Font Metrics) files.

Note that this has been designed to read in AFM files generated by Fontographer
and has not been tested on many other files. In particular, it does not
implement the whole Adobe AFM specification [#f1]_ but, it should read most
"common" AFM files.

Here is an example of using `afmLib` to read, modify and write an AFM file:

        >>> from fontTools.afmLib import AFM
        >>> f = AFM("Tests/afmLib/data/TestAFM.afm")
        >>>
        >>> # Accessing a pair gets you the kern value
        >>> f[("V","A")]
        -60
        >>>
        >>> # Accessing a glyph name gets you metrics
        >>> f["A"]
        (65, 668, (8, -25, 660, 666))
        >>> # (charnum, width, bounding box)
        >>>
        >>> # Accessing an attribute gets you metadata
        >>> f.FontName
        'TestFont-Regular'
        >>> f.FamilyName
        'TestFont'
        >>> f.Weight
        'Regular'
        >>> f.XHeight
        500
        >>> f.Ascender
        750
        >>>
        >>> # Attributes and items can also be set
        >>> f[("A","V")] = -150 # Tighten kerning
        >>> f.FontName = "TestFont Squished"
        >>>
        >>> # And the font written out again (remove the # in front)
        >>> #f.write("testfont-squished.afm")

.. rubric:: Footnotes

.. [#f1] `Adobe Technote 5004 <https://www.adobe.com/content/dam/acom/en/devnet/font/pdfs/5004.AFM_Spec.pdf>`_,
   Adobe Font Metrics File Format Specification.

"""

import re

# every single line starts with a "word"
identifierRE = re.compile(r"^([A-Za-z]+).*")

# regular expression to parse char lines
charRE = re.compile(
    r"(-?\d+)"  # charnum
    r"\s*;\s*WX\s+"  # ; WX
    r"(-?\d+)"  # width
    r"\s*;\s*N\s+"  # ; N
    r"([.A-Za-z0-9_]+)"  # charname
    r"\s*;\s*B\s+"  # ; B
    r"(-?\d+)"  # left
    r"\s+"
    r"(-?\d+)"  # bottom
    r"\s+"
    r"(-?\d+)"  # right
    r"\s+"
    r"(-?\d+)"  # top
    r"\s*;\s*"  # ;
)

# regular expression to parse kerning lines
kernRE = re.compile(
    r"([.A-Za-z0-9_]+)"  # leftchar
    r"\s+"
    r"([.A-Za-z0-9_]+)"  # rightchar
    r"\s+"
    r"(-?\d+)"  # value
    r"\s*"
)

# regular expressions to parse composite info lines of the form:
# Aacute 2 ; PCC A 0 0 ; PCC acute 182 211 ;
compositeRE = re.compile(
    r"([.A-Za-z0-9_]+)"  # char name
    r"\s+"
    r"(\d+)"  # number of parts
    r"\s*;\s*"
)
componentRE = re.compile(
    r"PCC\s+"  # PPC
    r"([.A-Za-z0-9_]+)"  # base char name
    r"\s+"
    r"(-?\d+)"  # x offset
    r"\s+"
    r"(-?\d+)"  # y offset
    r"\s*;\s*"
)

preferredAttributeOrder = [
    "FontName",
    "FullName",
    "FamilyName",
    "Weight",
    "ItalicAngle",
    "IsFixedPitch",
    "FontBBox",
    "UnderlinePosition",
    "UnderlineThickness",
    "Version",
    "Notice",
    "EncodingScheme",
    "CapHeight",
    "XHeight",
    "Ascender",
    "Descender",
]


class error(Exception):
    pass


class AFM(object):
    _attrs = None

    _keywords = [
        "StartFontMetrics",
        "EndFontMetrics",
        "StartCharMetrics",
        "EndCharMetrics",
        "StartKernData",
        "StartKernPairs",
        "EndKernPairs",
        "EndKernData",
        "StartComposites",
        "EndComposites",
    ]

    def __init__(self, path=None):
        """AFM file reader.

        Instantiating an object with a path name will cause the file to be opened,
        read, and parsed. Alternatively the path can be left unspecified, and a
        file can be parsed later with the :meth:`read` method."""
        self._attrs = {}
        self._chars = {}
        self._kerning = {}
        self._index = {}
        self._comments = []
        self._composites = {}
        if path is not None:
            self.read(path)

    def read(self, path):
        """Opens, reads and parses a file."""
        lines = readlines(path)
        for line in lines:
            if not line.strip():
                continue
            m = identifierRE.match(line)
            if m is None:
                raise error("syntax error in AFM file: " + repr(line))

            pos = m.regs[1][1]
            word = line[:pos]
            rest = line[pos:].strip()
            if word in self._keywords:
                continue
            if word == "C":
                self.parsechar(rest)
            elif word == "KPX":
                self.parsekernpair(rest)
            elif word == "CC":
                self.parsecomposite(rest)
            else:
                self.parseattr(word, rest)

    def parsechar(self, rest):
        m = charRE.match(rest)
        if m is None:
            raise error("syntax error in AFM file: " + repr(rest))
        things = []
        for fr, to in m.regs[1:]:
            things.append(rest[fr:to])
        charname = things[2]
        del things[2]
        charnum, width, l, b, r, t = (int(thing) for thing in things)
        self._chars[charname] = charnum, width, (l, b, r, t)

    def parsekernpair(self, rest):
        m = kernRE.match(rest)
        if m is None:
            raise error("syntax error in AFM file: " + repr(rest))
        things = []
        for fr, to in m.regs[1:]:
            things.append(rest[fr:to])
        leftchar, rightchar, value = things
        value = int(value)
        self._kerning[(leftchar, rightchar)] = value

    def parseattr(self, word, rest):
        if word == "FontBBox":
            l, b, r, t = [int(thing) for thing in rest.split()]
            self._attrs[word] = l, b, r, t
        elif word == "Comment":
            self._comments.append(rest)
        else:
            try:
                value = int(rest)
            except (ValueError, OverflowError):
                self._attrs[word] = rest
            else:
                self._attrs[word] = value

    def parsecomposite(self, rest):
        m = compositeRE.match(rest)
        if m is None:
            raise error("syntax error in AFM file: " + repr(rest))
        charname = m.group(1)
        ncomponents = int(m.group(2))
        rest = rest[m.regs[0][1] :]
        components = []
        while True:
            m = componentRE.match(rest)
            if m is None:
                raise error("syntax error in AFM file: " + repr(rest))
            basechar = m.group(1)
            xoffset = int(m.group(2))
            yoffset = int(m.group(3))
            components.append((basechar, xoffset, yoffset))
            rest = rest[m.regs[0][1] :]
            if not rest:
                break
        assert len(components) == ncomponents
        self._composites[charname] = components

    def write(self, path, sep="\r"):
        """Writes out an AFM font to the given path."""
        import time

        lines = [
            "StartFontMetrics 2.0",
            "Comment Generated by afmLib; at %s"
            % (time.strftime("%m/%d/%Y %H:%M:%S", time.localtime(time.time()))),
        ]

        # write comments, assuming (possibly wrongly!) they should
        # all appear at the top
        for comment in self._comments:
            lines.append("Comment " + comment)

        # write attributes, first the ones we know about, in
        # a preferred order
        attrs = self._attrs
        for attr in preferredAttributeOrder:
            if attr in attrs:
                value = attrs[attr]
                if attr == "FontBBox":
                    value = "%s %s %s %s" % value
                lines.append(attr + " " + str(value))
        # then write the attributes we don't know about,
        # in alphabetical order
        items = sorted(attrs.items())
        for attr, value in items:
            if attr in preferredAttributeOrder:
                continue
            lines.append(attr + " " + str(value))

        # write char metrics
        lines.append("StartCharMetrics " + repr(len(self._chars)))
        items = [
            (charnum, (charname, width, box))
            for charname, (charnum, width, box) in self._chars.items()
        ]

        def myKey(a):
            """Custom key function to make sure unencoded chars (-1)
            end up at the end of the list after sorting."""
            if a[0] == -1:
                a = (0xFFFF,) + a[1:]  # 0xffff is an arbitrary large number
            return a

        items.sort(key=myKey)

        for charnum, (charname, width, (l, b, r, t)) in items:
            lines.append(
                "C %d ; WX %d ; N %s ; B %d %d %d %d ;"
                % (charnum, width, charname, l, b, r, t)
            )
        lines.append("EndCharMetrics")

        # write kerning info
        lines.append("StartKernData")
        lines.append("StartKernPairs " + repr(len(self._kerning)))
        items = sorted(self._kerning.items())
        for (leftchar, rightchar), value in items:
            lines.append("KPX %s %s %d" % (leftchar, rightchar, value))
        lines.append("EndKernPairs")
        lines.append("EndKernData")

        if self._composites:
            composites = sorted(self._composites.items())
            lines.append("StartComposites %s" % len(self._composites))
            for charname, components in composites:
                line = "CC %s %s ;" % (charname, len(components))
                for basechar, xoffset, yoffset in components:
                    line = line + " PCC %s %s %s ;" % (basechar, xoffset, yoffset)
                lines.append(line)
            lines.append("EndComposites")

        lines.append("EndFontMetrics")

        writelines(path, lines, sep)

    def has_kernpair(self, pair):
        """Returns `True` if the given glyph pair (specified as a tuple) exists
        in the kerning dictionary."""
        return pair in self._kerning

    def kernpairs(self):
        """Returns a list of all kern pairs in the kerning dictionary."""
        return list(self._kerning.keys())

    def has_char(self, char):
        """Returns `True` if the given glyph exists in the font."""
        return char in self._chars

    def chars(self):
        """Returns a list of all glyph names in the font."""
        return list(self._chars.keys())

    def comments(self):
        """Returns all comments from the file."""
        return self._comments

    def addComment(self, comment):
        """Adds a new comment to the file."""
        self._comments.append(comment)

    def addComposite(self, glyphName, components):
        """Specifies that the glyph `glyphName` is made up of the given components.
        The components list should be of the following form::

                [
                        (glyphname, xOffset, yOffset),
                        ...
                ]

        """
        self._composites[glyphName] = components

    def __getattr__(self, attr):
        if attr in self._attrs:
            return self._attrs[attr]
        else:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        # all attrs *not* starting with "_" are consider to be AFM keywords
        if attr[:1] == "_":
            self.__dict__[attr] = value
        else:
            self._attrs[attr] = value

    def __delattr__(self, attr):
        # all attrs *not* starting with "_" are consider to be AFM keywords
        if attr[:1] == "_":
            try:
                del self.__dict__[attr]
            except KeyError:
                raise AttributeError(attr)
        else:
            try:
                del self._attrs[attr]
            except KeyError:
                raise AttributeError(attr)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            # key is a tuple, return the kernpair
            return self._kerning[key]
        else:
            # return the metrics instead
            return self._chars[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            # key is a tuple, set kernpair
            self._kerning[key] = value
        else:
            # set char metrics
            self._chars[key] = value

    def __delitem__(self, key):
        if isinstance(key, tuple):
            # key is a tuple, del kernpair
            del self._kerning[key]
        else:
            # del char metrics
            del self._chars[key]

    def __repr__(self):
        if hasattr(self, "FullName"):
            return "<AFM object for %s>" % self.FullName
        else:
            return "<AFM object at %x>" % id(self)


def readlines(path):
    with open(path, "r", encoding="ascii") as f:
        data = f.read()
    return data.splitlines()


def writelines(path, lines, sep="\r"):
    with open(path, "w", encoding="ascii", newline=sep) as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import EasyDialogs

    path = EasyDialogs.AskFileForOpen()
    if path:
        afm = AFM(path)
        char = "A"
        if afm.has_char(char):
            print(afm[char])  # print charnum, width and boundingbox
        pair = ("A", "V")
        if afm.has_kernpair(pair):
            print(afm[pair])  # print kerning value for pair
        print(afm.Version)  # various other afm entries have become attributes
        print(afm.Weight)
        # afm.comments() returns a list of all Comment lines found in the AFM
        print(afm.comments())
        # print afm.chars()
        # print afm.kernpairs()
        print(afm)
        afm.write(path + ".muck")
<|endoftext|>from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional, TypeVar, Union
from collections.abc import Callable, Sequence
from fontTools.misc.filesystem._base import FS
from os import PathLike
from xml.etree.ElementTree import Element as ElementTreeElement

if TYPE_CHECKING:
    from fontTools.ufoLib import UFOFormatVersion
    from fontTools.ufoLib.glifLib import GLIFFormatVersion
    from lxml.etree import _Element as LxmlElement


T = TypeVar("T")  # Generic type
K = TypeVar("K")  # Generic dict key type
V = TypeVar("V")  # Generic dict value type

GlyphNameToFileNameFunc = Optional[Callable[[str, set[str]], str]]
ElementType = Union[ElementTreeElement, "LxmlElement"]
FormatVersion = Union[int, tuple[int, int]]
FormatVersions = Optional[Iterable[FormatVersion]]
GLIFFormatVersionInput = Optional[Union[int, tuple[int, int], "GLIFFormatVersion"]]
UFOFormatVersionInput = Optional[Union[int, tuple[int, int], "UFOFormatVersion"]]
IntFloat = Union[int, float]
KerningPair = tuple[str, str]
KerningDict = dict[KerningPair, IntFloat]
KerningGroups = dict[str, Sequence[str]]
KerningNested = dict[str, dict[str, IntFloat]]
PathStr = Union[str, PathLike[str]]
PathOrFS = Union[PathStr, FS]
<|endoftext|>"""\
usage: ttx [options] inputfile1 [... inputfileN]

TTX -- From OpenType To XML And Back

If an input file is a TrueType or OpenType font file, it will be
decompiled to a TTX file (an XML-based text format).
If an input file is a TTX file, it will be compiled to whatever
format the data is in, a TrueType or OpenType/CFF font file.
A special input value of - means read from the standard input.

Output files are created so they are unique: an existing file is
never overwritten.

General options
===============

-h Help            print this message.
--version          show version and exit.
-d <outputfolder>  Specify a directory where the output files are
                   to be created.
-o <outputfile>    Specify a file to write the output to. A special
                   value of - would use the standard output.
-f                 Overwrite existing output file(s), ie. don't append
                   numbers.
-v                 Verbose: more messages will be written to stdout
                   about what is being done.
-q                 Quiet: No messages will be written to stdout about
                   what is being done.
-a                 allow virtual glyphs ID's on compile or decompile.

Dump options
============

-l           List table info: instead of dumping to a TTX file, list
             some minimal info about each table.
-t <table>   Specify a table to dump. Multiple -t options
             are allowed. When no -t option is specified, all tables
             will be dumped.
-x <table>   Specify a table to exclude from the dump. Multiple
             -x options are allowed. -t and -x are mutually exclusive.
-s           Split tables: save the TTX data into separate TTX files per
             table and write one small TTX file that contains references
             to the individual table dumps. This file can be used as
             input to ttx, as long as the table files are in the
             same directory.
-g           Split glyf table: Save the glyf data into separate TTX files
             per glyph and write a small TTX for the glyf table which
             contains references to the individual TTGlyph elements.
             NOTE: specifying -g implies -s (no need for -s together
             with -g)
-i           Do NOT disassemble TT instructions: when this option is
             given, all TrueType programs (glyph programs, the font
             program and the pre-program) will be written to the TTX
             file as hex data instead of assembly. This saves some time
             and makes the TTX file smaller.
-z <format>  Specify a bitmap data export option for EBDT:
             {'raw', 'row', 'bitwise', 'extfile'} or for the CBDT:
             {'raw', 'extfile'} Each option does one of the following:

             -z raw
               export the bitmap data as a hex dump
             -z row
               export each row as hex data
             -z bitwise
               export each row as binary in an ASCII art style
             -z extfile
               export the data as external files with XML references

             If no export format is specified 'raw' format is used.
-e           Don't ignore decompilation errors, but show a full traceback
             and abort.
-y <number>  Select font number for TrueType Collection (.ttc/.otc),
             starting from 0.
--unicodedata <UnicodeData.txt>
             Use custom database file to write character names in the
             comments of the cmap TTX output.
--newline <value>
             Control how line endings are written in the XML file. It
             can be 'LF', 'CR', or 'CRLF'. If not specified, the
             default platform-specific line endings are used.

Compile options
===============

-m           Merge with TrueType-input-file: specify a TrueType or
             OpenType font file to be merged with the TTX file. This
             option is only valid when at most one TTX file is specified.
-b           Don't recalc glyph bounding boxes: use the values in the
             TTX file as-is.
--recalc-timestamp
             Set font 'modified' timestamp to current time.
             By default, the modification time of the TTX file will be
             used.
--no-recalc-timestamp
             Keep the original font 'modified' timestamp.
--flavor <type>
             Specify flavor of output font file. May be 'woff' or 'woff2'.
             Note that WOFF2 requires the Brotli Python extension,
             available at https://github.com/google/brotli
--with-zopfli
             Use Zopfli instead of Zlib to compress WOFF. The Python
             extension is available at https://pypi.python.org/pypi/zopfli
--optimize-font-speed
             Enable optimizations that prioritize speed over file size.
             This mainly affects how glyf t able and gvar / VARC tables are
             compiled. The produced fonts will be larger, but rendering
             performance will be improved with HarfBuzz and other text
             layout engines.
"""

from fontTools.ttLib import OPTIMIZE_FONT_SPEED, TTFont, TTLibError
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.unicode import setUnicodeData
from fontTools.misc.textTools import Tag, tostr
from fontTools.misc.timeTools import timestampSinceEpoch
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
import os
import sys
import getopt
import re
import logging

log = logging.getLogger("fontTools.ttx")

opentypeheaderRE = re.compile("""sfntVersion=['"]OTTO["']""")


class Options(object):
    listTables = False
    outputDir = None
    outputFile = None
    overWrite = False
    verbose = False
    quiet = False
    splitTables = False
    splitGlyphs = False
    disassembleInstructions = True
    mergeFile = None
    recalcBBoxes = True
    ignoreDecompileErrors = True
    bitmapGlyphDataFormat = "raw"
    unicodedata = None
    newlinestr = "\n"
    recalcTimestamp = None
    flavor = None
    useZopfli = False
    optimizeFontSpeed = False

    def __init__(self, rawOptions, numFiles):
        self.onlyTables = []
        self.skipTables = []
        self.fontNumber = -1
        for option, value in rawOptions:
            # general options
            if option == "-h":
                print(__doc__)
                sys.exit(0)
            elif option == "--version":
                from fontTools import version

                print(version)
                sys.exit(0)
            elif option == "-d":
                if not os.path.isdir(value):
                    raise getopt.GetoptError(
                        "The -d option value must be an existing directory"
                    )
                self.outputDir = value
            elif option == "-o":
                self.outputFile = value
            elif option == "-f":
                self.overWrite = True
            elif option == "-v":
                self.verbose = True
            elif option == "-q":
                self.quiet = True
            # dump options
            elif option == "-l":
                self.listTables = True
            elif option == "-t":
                # pad with space if table tag length is less than 4
                value = value.ljust(4)
                self.onlyTables.append(value)
            elif option == "-x":
                # pad with space if table tag length is less than 4
                value = value.ljust(4)
                self.skipTables.append(value)
            elif option == "-s":
                self.splitTables = True
            elif option == "-g":
                # -g implies (and forces) splitTables
                self.splitGlyphs = True
                self.splitTables = True
            elif option == "-i":
                self.disassembleInstructions = False
            elif option == "-z":
                validOptions = ("raw", "row", "bitwise", "extfile")
                if value not in validOptions:
                    raise getopt.GetoptError(
                        "-z does not allow %s as a format. Use %s"
                        % (option, validOptions)
                    )
                self.bitmapGlyphDataFormat = value
            elif option == "-y":
                self.fontNumber = int(value)
            # compile options
            elif option == "-m":
                self.mergeFile = value
            elif option == "-b":
                self.recalcBBoxes = False
            elif option == "-e":
                self.ignoreDecompileErrors = False
            elif option == "--unicodedata":
                self.unicodedata = value
            elif option == "--newline":
                validOptions = ("LF", "CR", "CRLF")
                if value == "LF":
                    self.newlinestr = "\n"
                elif value == "CR":
                    self.newlinestr = "\r"
                elif value == "CRLF":
                    self.newlinestr = "\r\n"
                else:
                    raise getopt.GetoptError(
                        "Invalid choice for --newline: %r (choose from %s)"
                        % (value, ", ".join(map(repr, validOptions)))
                    )
            elif option == "--recalc-timestamp":
                self.recalcTimestamp = True
            elif option == "--no-recalc-timestamp":
                self.recalcTimestamp = False
            elif option == "--flavor":
                self.flavor = value
            elif option == "--with-zopfli":
                self.useZopfli = True
            elif option == "--optimize-font-speed":
                self.optimizeFontSpeed = True
        if self.verbose and self.quiet:
            raise getopt.GetoptError("-q and -v options are mutually exclusive")
        if self.verbose:
            self.logLevel = logging.DEBUG
        elif self.quiet:
            self.logLevel = logging.WARNING
        else:
            self.logLevel = logging.INFO
        if self.mergeFile and self.flavor:
            raise getopt.GetoptError("-m and --flavor options are mutually exclusive")
        if self.onlyTables and self.skipTables:
            raise getopt.GetoptError("-t and -x options are mutually exclusive")
        if self.mergeFile and numFiles > 1:
            raise getopt.GetoptError(
                "Must specify exactly one TTX source file when using -m"
            )
        if self.flavor != "woff" and self.useZopfli:
            raise getopt.GetoptError("--with-zopfli option requires --flavor 'woff'")


def ttList(input, output, options):
    ttf = TTFont(input, fontNumber=options.fontNumber, lazy=True)
    reader = ttf.reader
    tags = sorted(reader.keys())
    print('Listing table info for "%s":' % input)
    format = "    %4s  %10s  %8s  %8s"
    print(format % ("tag ", "  checksum", "  length", "  offset"))
    print(format % ("----", "----------", "--------", "--------"))
    for tag in tags:
        entry = reader.tables[tag]
        if ttf.flavor == "woff2":
            # WOFF2 doesn't store table checksums, so they must be calculated
            from fontTools.ttLib.sfnt import calcChecksum

            data = entry.loadData(reader.transformBuffer)
            checkSum = calcChecksum(data)
        else:
            checkSum = int(entry.checkSum)
        if checkSum < 0:
            checkSum = checkSum + 0x100000000
        checksum = "0x%08X" % checkSum
        print(format % (tag, checksum, entry.length, entry.offset))
    print()
    ttf.close()


@Timer(log, "Done dumping TTX in %(time).3f seconds")
def ttDump(input, output, options):
    input_name = input
    if input == "-":
        input, input_name = sys.stdin.buffer, sys.stdin.name
    output_name = output
    if output == "-":
        output, output_name = sys.stdout, sys.stdout.name
    log.info('Dumping "%s" to "%s"...', input_name, output_name)
    if options.unicodedata:
        setUnicodeData(options.unicodedata)
    ttf = TTFont(
        input,
        0,
        ignoreDecompileErrors=options.ignoreDecompileErrors,
        fontNumber=options.fontNumber,
    )
    ttf.saveXML(
        output,
        tables=options.onlyTables,
        skipTables=options.skipTables,
        splitTables=options.splitTables,
        splitGlyphs=options.splitGlyphs,
        disassembleInstructions=options.disassembleInstructions,
        bitmapGlyphDataFormat=options.bitmapGlyphDataFormat,
        newlinestr=options.newlinestr,
    )
    ttf.close()


@Timer(log, "Done compiling TTX in %(time).3f seconds")
def ttCompile(input, output, options):
    input_name = input
    if input == "-":
        input, input_name = sys.stdin, sys.stdin.name
    output_name = output
    if output == "-":
        output, output_name = sys.stdout.buffer, sys.stdout.name
    log.info('Compiling "%s" to "%s"...' % (input_name, output))
    if options.useZopfli:
        from fontTools.ttLib import sfnt

        sfnt.USE_ZOPFLI = True
    ttf = TTFont(
        options.mergeFile,
        flavor=options.flavor,
        recalcBBoxes=options.recalcBBoxes,
        recalcTimestamp=options.recalcTimestamp,
    )
    if options.optimizeFontSpeed:
        ttf.cfg[OPTIMIZE_FONT_SPEED] = options.optimizeFontSpeed
    ttf.importXML(input)

    if options.recalcTimestamp is None and "head" in ttf and input is not sys.stdin:
        # use TTX file modification time for head "modified" timestamp
        mtime = os.path.getmtime(input)
        ttf["head"].modified = timestampSinceEpoch(mtime)

    ttf.save(output)


def guessFileType(fileName):
    if fileName == "-":
        header = sys.stdin.buffer.peek(256)
        ext = ""
    else:
        base, ext = os.path.splitext(fileName)
        try:
            with open(fileName, "rb") as f:
                header = f.read(256)
        except IOError:
            return None

    if header.startswith(b"\xef\xbb\xbf<?xml"):
        header = header.lstrip(b"\xef\xbb\xbf")
    cr, tp = getMacCreatorAndType(fileName)
    if tp in ("sfnt", "FFIL"):
        return "TTF"
    if ext == ".dfont":
        return "TTF"
    head = Tag(header[:4])
    if head == "OTTO":
        return "OTF"
    elif head == "ttcf":
        return "TTC"
    elif head in ("\0\1\0\0", "true"):
        return "TTF"
    elif head == "wOFF":
        return "WOFF"
    elif head == "wOF2":
        return "WOFF2"
    elif head == "<?xm":
        # Use 'latin1' because that can't fail.
        header = tostr(header, "latin1")
        if opentypeheaderRE.search(header):
            return "OTX"
        else:
            return "TTX"
    return None


def parseOptions(args):
    rawOptions, files = getopt.gnu_getopt(
        args,
        "ld:o:fvqht:x:sgim:z:baey:",
        [
            "unicodedata=",
            "recalc-timestamp",
            "no-recalc-timestamp",
            "flavor=",
            "version",
            "with-zopfli",
            "newline=",
            "optimize-font-speed",
        ],
    )

    options = Options(rawOptions, len(files))
    jobs = []

    if not files:
        raise getopt.GetoptError("Must specify at least one input file")

    for input in files:
        if input != "-" and not os.path.isfile(input):
            raise getopt.GetoptError('File not found: "%s"' % input)
        tp = guessFileType(input)
        if tp in ("OTF", "TTF", "TTC", "WOFF", "WOFF2"):
            extension = ".ttx"
            if options.listTables:
                action = ttList
            else:
                action = ttDump
        elif tp == "TTX":
            extension = "." + options.flavor if options.flavor else ".ttf"
            action = ttCompile
        elif tp == "OTX":
            extension = "." + options.flavor if options.flavor else ".otf"
            action = ttCompile
        else:
            raise getopt.GetoptError('Unknown file type: "%s"' % input)

        if options.outputFile:
            output = options.outputFile
        else:
            if input == "-":
                raise getopt.GetoptError("Must provide -o when reading from stdin")
            output = makeOutputFileName(
                input, options.outputDir, extension, options.overWrite
            )
            # 'touch' output file to avoid race condition in choosing file names
            if action != ttList:
                open(output, "a").close()
        jobs.append((action, input, output))
    return jobs, options


def process(jobs, options):
    for action, input, output in jobs:
        action(input, output, options)


def main(args=None):
    """Convert OpenType fonts to XML and back"""
    from fontTools import configLogger

    if args is None:
        args = sys.argv[1:]
    try:
        jobs, options = parseOptions(args)
    except getopt.GetoptError as e:
        print("%s\nERROR: %s" % (__doc__, e), file=sys.stderr)
        sys.exit(2)

    configLogger(level=options.logLevel)

    try:
        process(jobs, options)
    except KeyboardInterrupt:
        log.error("(Cancelled.)")
        sys.exit(1)
    except SystemExit:
        raise
    except TTLibError as e:
        log.error(e)
        sys.exit(1)
    except:
        log.exception("Unhandled exception has occurred")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|>"""Module for reading TFM (TeX Font Metrics) files.

The TFM format is described in the TFtoPL WEB source code, whose typeset form
can be found on `CTAN <http://mirrors.ctan.org/info/knuth-pdf/texware/tftopl.pdf>`_.

        >>> from fontTools.tfmLib import TFM
        >>> tfm = TFM("Tests/tfmLib/data/cmr10.tfm")
        >>>
        >>> # Accessing an attribute gets you metadata.
        >>> tfm.checksum
        1274110073
        >>> tfm.designsize
        10.0
        >>> tfm.codingscheme
        'TeX text'
        >>> tfm.family
        'CMR'
        >>> tfm.seven_bit_safe_flag
        False
        >>> tfm.face
        234
        >>> tfm.extraheader
        {}
        >>> tfm.fontdimens
        {'SLANT': 0.0, 'SPACE': 0.33333396911621094, 'STRETCH': 0.16666698455810547, 'SHRINK': 0.11111164093017578, 'XHEIGHT': 0.4305553436279297, 'QUAD': 1.0000028610229492, 'EXTRASPACE': 0.11111164093017578}
        >>> # Accessing a character gets you its metrics.
        >>> # “width” is always available, other metrics are available only when
        >>> # applicable. All values are relative to “designsize”.
        >>> tfm.chars[ord("g")]
        {'width': 0.5000019073486328, 'height': 0.4305553436279297, 'depth': 0.1944446563720703, 'italic': 0.013888359069824219}
        >>> # Kerning and ligature can be accessed as well.
        >>> tfm.kerning[ord("c")]
        {104: -0.02777862548828125, 107: -0.02777862548828125}
        >>> tfm.ligatures[ord("f")]
        {105: ('LIG', 12), 102: ('LIG', 11), 108: ('LIG', 13)}
"""

from types import SimpleNamespace

from fontTools.misc.sstruct import calcsize, unpack, unpack2

SIZES_FORMAT = """
    >
    lf: h    # length of the entire file, in words
    lh: h    # length of the header data, in words
    bc: h    # smallest character code in the font
    ec: h    # largest character code in the font
    nw: h    # number of words in the width table
    nh: h    # number of words in the height table
    nd: h    # number of words in the depth table
    ni: h    # number of words in the italic correction table
    nl: h    # number of words in the ligature/kern table
    nk: h    # number of words in the kern table
    ne: h    # number of words in the extensible character table
    np: h    # number of font parameter words
"""

SIZES_SIZE = calcsize(SIZES_FORMAT)

FIXED_FORMAT = "12.20F"

HEADER_FORMAT1 = f"""
    >
    checksum:            L
    designsize:          {FIXED_FORMAT}
"""

HEADER_FORMAT2 = f"""
    {HEADER_FORMAT1}
    codingscheme:        40p
"""

HEADER_FORMAT3 = f"""
    {HEADER_FORMAT2}
    family:              20p
"""

HEADER_FORMAT4 = f"""
    {HEADER_FORMAT3}
    seven_bit_safe_flag: ?
    ignored:             x
    ignored:             x
    face:                B
"""

HEADER_SIZE1 = calcsize(HEADER_FORMAT1)
HEADER_SIZE2 = calcsize(HEADER_FORMAT2)
HEADER_SIZE3 = calcsize(HEADER_FORMAT3)
HEADER_SIZE4 = calcsize(HEADER_FORMAT4)

LIG_KERN_COMMAND = """
    >
    skip_byte: B
    next_char: B
    op_byte: B
    remainder: B
"""

BASE_PARAMS = [
    "SLANT",
    "SPACE",
    "STRETCH",
    "SHRINK",
    "XHEIGHT",
    "QUAD",
    "EXTRASPACE",
]

MATHSY_PARAMS = [
    "NUM1",
    "NUM2",
    "NUM3",
    "DENOM1",
    "DENOM2",
    "SUP1",
    "SUP2",
    "SUP3",
    "SUB1",
    "SUB2",
    "SUPDROP",
    "SUBDROP",
    "DELIM1",
    "DELIM2",
    "AXISHEIGHT",
]

MATHEX_PARAMS = [
    "DEFAULTRULETHICKNESS",
    "BIGOPSPACING1",
    "BIGOPSPACING2",
    "BIGOPSPACING3",
    "BIGOPSPACING4",
    "BIGOPSPACING5",
]

VANILLA = 0
MATHSY = 1
MATHEX = 2

UNREACHABLE = 0
PASSTHROUGH = 1
ACCESSABLE = 2

NO_TAG = 0
LIG_TAG = 1
LIST_TAG = 2
EXT_TAG = 3

STOP_FLAG = 128
KERN_FLAG = 128


class TFMException(Exception):
    def __init__(self, message):
        super().__init__(message)


class TFM:
    def __init__(self, file):
        self._read(file)

    def __repr__(self):
        return (
            f"<TFM"
            f" for {self.family}"
            f" in {self.codingscheme}"
            f" at {self.designsize:g}pt>"
        )

    def _read(self, file):
        if hasattr(file, "read"):
            data = file.read()
        else:
            with open(file, "rb") as fp:
                data = fp.read()

        self._data = data

        if len(data) < SIZES_SIZE:
            raise TFMException("Too short input file")

        sizes = SimpleNamespace()
        unpack2(SIZES_FORMAT, data, sizes)

        # Do some file structure sanity checks.
        # TeX and TFtoPL do additional functional checks and might even correct
        # “errors” in the input file, but we instead try to output the file as
        # it is as long as it is parsable, even if the data make no sense.

        if sizes.lf < 0:
            raise TFMException("The file claims to have negative or zero length!")

        if len(data) < sizes.lf * 4:
            raise TFMException("The file has fewer bytes than it claims!")

        for name, length in vars(sizes).items():
            if length < 0:
                raise TFMException("The subfile size: '{name}' is negative!")

        if sizes.lh < 2:
            raise TFMException(f"The header length is only {sizes.lh}!")

        if sizes.bc > sizes.ec + 1 or sizes.ec > 255:
            raise TFMException(
                f"The character code range {sizes.bc}..{sizes.ec} is illegal!"
            )

        if sizes.nw == 0 or sizes.nh == 0 or sizes.nd == 0 or sizes.ni == 0:
            raise TFMException("Incomplete subfiles for character dimensions!")

        if sizes.ne > 256:
            raise TFMException(f"There are {ne} extensible recipes!")

        if sizes.lf != (
            6
            + sizes.lh
            + (sizes.ec - sizes.bc + 1)
            + sizes.nw
            + sizes.nh
            + sizes.nd
            + sizes.ni
            + sizes.nl
            + sizes.nk
            + sizes.ne
            + sizes.np
        ):
            raise TFMException("Subfile sizes don’t add up to the stated total")

        # Subfile offsets, used in the helper function below. These all are
        # 32-bit word offsets not 8-bit byte offsets.
        char_base = 6 + sizes.lh - sizes.bc
        width_base = char_base + sizes.ec + 1
        height_base = width_base + sizes.nw
        depth_base = height_base + sizes.nh
        italic_base = depth_base + sizes.nd
        lig_kern_base = italic_base + sizes.ni
        kern_base = lig_kern_base + sizes.nl
        exten_base = kern_base + sizes.nk
        param_base = exten_base + sizes.ne

        # Helper functions for accessing individual data. If this looks
        # nonidiomatic Python, I blame the effect of reading the literate WEB
        # documentation of TFtoPL.
        def char_info(c):
            return 4 * (char_base + c)

        def width_index(c):
            return data[char_info(c)]

        def noneexistent(c):
            return c < sizes.bc or c > sizes.ec or width_index(c) == 0

        def height_index(c):
            return data[char_info(c) + 1] // 16

        def depth_index(c):
            return data[char_info(c) + 1] % 16

        def italic_index(c):
            return data[char_info(c) + 2] // 4

        def tag(c):
            return data[char_info(c) + 2] % 4

        def remainder(c):
            return data[char_info(c) + 3]

        def width(c):
            r = 4 * (width_base + width_index(c))
            return read_fixed(r, "v")["v"]

        def height(c):
            r = 4 * (height_base + height_index(c))
            return read_fixed(r, "v")["v"]

        def depth(c):
            r = 4 * (depth_base + depth_index(c))
            return read_fixed(r, "v")["v"]

        def italic(c):
            r = 4 * (italic_base + italic_index(c))
            return read_fixed(r, "v")["v"]

        def exten(c):
            return 4 * (exten_base + remainder(c))

        def lig_step(i):
            return 4 * (lig_kern_base + i)

        def lig_kern_command(i):
            command = SimpleNamespace()
            unpack2(LIG_KERN_COMMAND, data[i:], command)
            return command

        def kern(i):
            r = 4 * (kern_base + i)
            return read_fixed(r, "v")["v"]

        def param(i):
            return 4 * (param_base + i)

        def read_fixed(index, key, obj=None):
            ret = unpack2(f">;{key}:{FIXED_FORMAT}", data[index:], obj)
            return ret[0]

        # Set all attributes to empty values regardless of the header size.
        unpack(HEADER_FORMAT4, [0] * HEADER_SIZE4, self)

        offset = 24
        length = sizes.lh * 4
        self.extraheader = {}
        if length >= HEADER_SIZE4:
            rest = unpack2(HEADER_FORMAT4, data[offset:], self)[1]
            if self.face < 18:
                s = self.face % 2
                b = self.face // 2
                self.face = "MBL"[b % 3] + "RI"[s] + "RCE"[b // 3]
            for i in range(sizes.lh - HEADER_SIZE4 // 4):
                rest = unpack2(f">;HEADER{i + 18}:l", rest, self.extraheader)[1]
        elif length >= HEADER_SIZE3:
            unpack2(HEADER_FORMAT3, data[offset:], self)
        elif length >= HEADER_SIZE2:
            unpack2(HEADER_FORMAT2, data[offset:], self)
        elif length >= HEADER_SIZE1:
            unpack2(HEADER_FORMAT1, data[offset:], self)

        self.fonttype = VANILLA
        scheme = self.codingscheme.upper()
        if scheme.startswith("TEX MATH SY"):
            self.fonttype = MATHSY
        elif scheme.startswith("TEX MATH EX"):
            self.fonttype = MATHEX

        self.fontdimens = {}
        for i in range(sizes.np):
            name = f"PARAMETER{i+1}"
            if i <= 6:
                name = BASE_PARAMS[i]
            elif self.fonttype == MATHSY and i <= 21:
                name = MATHSY_PARAMS[i - 7]
            elif self.fonttype == MATHEX and i <= 12:
                name = MATHEX_PARAMS[i - 7]
            read_fixed(param(i), name, self.fontdimens)

        lig_kern_map = {}
        self.right_boundary_char = None
        self.left_boundary_char = None
        if sizes.nl > 0:
            cmd = lig_kern_command(lig_step(0))
            if cmd.skip_byte == 255:
                self.right_boundary_char = cmd.next_char

            cmd = lig_kern_command(lig_step((sizes.nl - 1)))
            if cmd.skip_byte == 255:
                self.left_boundary_char = 256
                r = 256 * cmd.op_byte + cmd.remainder
                lig_kern_map[self.left_boundary_char] = r

        self.chars = {}
        for c in range(sizes.bc, sizes.ec + 1):
            if width_index(c) > 0:
                self.chars[c] = info = {}
                info["width"] = width(c)
                if height_index(c) > 0:
                    info["height"] = height(c)
                if depth_index(c) > 0:
                    info["depth"] = depth(c)
                if italic_index(c) > 0:
                    info["italic"] = italic(c)
                char_tag = tag(c)
                if char_tag == NO_TAG:
                    pass
                elif char_tag == LIG_TAG:
                    lig_kern_map[c] = remainder(c)
                elif char_tag == LIST_TAG:
                    info["nextlarger"] = remainder(c)
                elif char_tag == EXT_TAG:
                    info["varchar"] = varchar = {}
                    for i in range(4):
                        part = data[exten(c) + i]
                        if i == 3 or part > 0:
                            name = "rep"
                            if i == 0:
                                name = "top"
                            elif i == 1:
                                name = "mid"
                            elif i == 2:
                                name = "bot"
                            if noneexistent(part):
                                varchar[name] = c
                            else:
                                varchar[name] = part

        self.ligatures = {}
        self.kerning = {}
        for c, i in sorted(lig_kern_map.items()):
            cmd = lig_kern_command(lig_step(i))
            if cmd.skip_byte > STOP_FLAG:
                i = 256 * cmd.op_byte + cmd.remainder

            while i < sizes.nl:
                cmd = lig_kern_command(lig_step(i))
                if cmd.skip_byte > STOP_FLAG:
                    pass
                else:
                    if cmd.op_byte >= KERN_FLAG:
                        r = 256 * (cmd.op_byte - KERN_FLAG) + cmd.remainder
                        self.kerning.setdefault(c, {})[cmd.next_char] = kern(r)
                    else:
                        r = cmd.op_byte
                        if r == 4 or (r > 7 and r != 11):
                            # Ligature step with nonstandard code, we output
                            # the code verbatim.
                            lig = r
                        else:
                            lig = ""
                            if r % 4 > 1:
                                lig += "/"
                            lig += "LIG"
                            if r % 2 != 0:
                                lig += "/"
                            while r > 3:
                                lig += ">"
                                r -= 4
                        self.ligatures.setdefault(c, {})[cmd.next_char] = (
                            lig,
                            cmd.remainder,
                        )

                if cmd.skip_byte >= STOP_FLAG:
                    break
                i += cmd.skip_byte + 1


if __name__ == "__main__":
    import sys

    tfm = TFM(sys.argv[1])
    print(
        "\n".join(
            x
            for x in [
                f"tfm.checksum={tfm.checksum}",
                f"tfm.designsize={tfm.designsize}",
                f"tfm.codingscheme={tfm.codingscheme}",
                f"tfm.fonttype={tfm.fonttype}",
                f"tfm.family={tfm.family}",
                f"tfm.seven_bit_safe_flag={tfm.seven_bit_safe_flag}",
                f"tfm.face={tfm.face}",
                f"tfm.extraheader={tfm.extraheader}",
                f"tfm.fontdimens={tfm.fontdimens}",
                f"tfm.right_boundary_char={tfm.right_boundary_char}",
                f"tfm.left_boundary_char={tfm.left_boundary_char}",
                f"tfm.kerning={tfm.kerning}",
                f"tfm.ligatures={tfm.ligatures}",
                f"tfm.chars={tfm.chars}",
            ]
        )
    )
    print(tfm)
<|endoftext|>def _makeunicodes(f):
    lines = iter(f.readlines())
    unicodes = {}
    for line in lines:
        if not line:
            continue
        num, name = line.split(";")[:2]
        if name[0] == "<":
            continue  # "<control>", etc.
        num = int(num, 16)
        unicodes[num] = name
    return unicodes


class _UnicodeCustom(object):
    def __init__(self, f):
        if isinstance(f, str):
            with open(f) as fd:
                codes = _makeunicodes(fd)
        else:
            codes = _makeunicodes(f)
        self.codes = codes

    def __getitem__(self, charCode):
        try:
            return self.codes[charCode]
        except KeyError:
            return "????"


class _UnicodeBuiltin(object):
    def __getitem__(self, charCode):
        try:
            # use unicodedata backport to python2, if available:
            # https://github.com/mikekap/unicodedata2
            import unicodedata2 as unicodedata
        except ImportError:
            import unicodedata
        try:
            return unicodedata.name(chr(charCode))
        except ValueError:
            return "????"


Unicode = _UnicodeBuiltin()


def setUnicodeData(f):
    global Unicode
    Unicode = _UnicodeCustom(f)
<|endoftext|>import pkgutil
import sys
import fontTools
import importlib
import os
from pathlib import Path


def main():
    """Show this help"""
    path = fontTools.__path__
    descriptions = {}
    for pkg in sorted(
        mod.name
        for mod in pkgutil.walk_packages([fontTools.__path__[0]], prefix="fontTools.")
    ):
        try:
            imports = __import__(pkg, globals(), locals(), ["main"])
        except ImportError as e:
            continue
        try:
            description = imports.main.__doc__
            # Cython modules seem to return "main()" as the docstring
            if description and description != "main()":
                pkg = pkg.replace("fontTools.", "").replace(".__main__", "")
                # show the docstring's first line only
                descriptions[pkg] = description.splitlines()[0]
        except AttributeError as e:
            pass
    for pkg, description in descriptions.items():
        print("fonttools %-25s %s" % (pkg, description), file=sys.stderr)


if __name__ == "__main__":
    print("fonttools v%s\n" % fontTools.__version__, file=sys.stderr)
    main()
<|endoftext|>__all__ = ["FontBuilder"]

"""
This module is *experimental*, meaning it still may evolve and change.

The `FontBuilder` class is a convenient helper to construct working TTF or
OTF fonts from scratch.

Note that the various setup methods cannot be called in arbitrary order,
due to various interdependencies between OpenType tables. Here is an order
that works:

    fb = FontBuilder(...)
    fb.setupGlyphOrder(...)
    fb.setupCharacterMap(...)
    fb.setupGlyf(...) --or-- fb.setupCFF(...)
    fb.setupHorizontalMetrics(...)
    fb.setupHorizontalHeader()
    fb.setupNameTable(...)
    fb.setupOS2()
    fb.addOpenTypeFeatures(...)
    fb.setupPost()
    fb.save(...)

Here is how to build a minimal TTF:

```python
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


def drawTestGlyph(pen):
    pen.moveTo((100, 100))
    pen.lineTo((100, 1000))
    pen.qCurveTo((200, 900), (400, 900), (500, 1000))
    pen.lineTo((500, 100))
    pen.closePath()


fb = FontBuilder(1024, isTTF=True)
fb.setupGlyphOrder([".notdef", ".null", "space", "A", "a"])
fb.setupCharacterMap({32: "space", 65: "A", 97: "a"})
advanceWidths = {".notdef": 600, "space": 500, "A": 600, "a": 600, ".null": 0}

familyName = "HelloTestFont"
styleName = "TotallyNormal"
version = "0.1"

nameStrings = dict(
    familyName=dict(en=familyName, nl="HalloTestFont"),
    styleName=dict(en=styleName, nl="TotaalNormaal"),
    uniqueFontIdentifier="fontBuilder: " + familyName + "." + styleName,
    fullName=familyName + "-" + styleName,
    psName=familyName + "-" + styleName,
    version="Version " + version,
)

pen = TTGlyphPen(None)
drawTestGlyph(pen)
glyph = pen.glyph()
glyphs = {".notdef": glyph, "space": glyph, "A": glyph, "a": glyph, ".null": glyph}
fb.setupGlyf(glyphs)
metrics = {}
glyphTable = fb.font["glyf"]
for gn, advanceWidth in advanceWidths.items():
    metrics[gn] = (advanceWidth, glyphTable[gn].xMin)
fb.setupHorizontalMetrics(metrics)
fb.setupHorizontalHeader(ascent=824, descent=-200)
fb.setupNameTable(nameStrings)
fb.setupOS2(sTypoAscender=824, usWinAscent=824, usWinDescent=200)
fb.setupPost()
fb.save("test.ttf")
```

And here's how to build a minimal OTF:

```python
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen


def drawTestGlyph(pen):
    pen.moveTo((100, 100))
    pen.lineTo((100, 1000))
    pen.curveTo((200, 900), (400, 900), (500, 1000))
    pen.lineTo((500, 100))
    pen.closePath()


fb = FontBuilder(1024, isTTF=False)
fb.setupGlyphOrder([".notdef", ".null", "space", "A", "a"])
fb.setupCharacterMap({32: "space", 65: "A", 97: "a"})
advanceWidths = {".notdef": 600, "space": 500, "A": 600, "a": 600, ".null": 0}

familyName = "HelloTestFont"
styleName = "TotallyNormal"
version = "0.1"

nameStrings = dict(
    familyName=dict(en=familyName, nl="HalloTestFont"),
    styleName=dict(en=styleName, nl="TotaalNormaal"),
    uniqueFontIdentifier="fontBuilder: " + familyName + "." + styleName,
    fullName=familyName + "-" + styleName,
    psName=familyName + "-" + styleName,
    version="Version " + version,
)

pen = T2CharStringPen(600, None)
drawTestGlyph(pen)
charString = pen.getCharString()
charStrings = {
    ".notdef": charString,
    "space": charString,
    "A": charString,
    "a": charString,
    ".null": charString,
}
fb.setupCFF(nameStrings["psName"], {"FullName": nameStrings["psName"]}, charStrings, {})
lsb = {gn: cs.calcBounds(None)[0] for gn, cs in charStrings.items()}
metrics = {}
for gn, advanceWidth in advanceWidths.items():
    metrics[gn] = (advanceWidth, lsb[gn])
fb.setupHorizontalMetrics(metrics)
fb.setupHorizontalHeader(ascent=824, descent=200)
fb.setupNameTable(nameStrings)
fb.setupOS2(sTypoAscender=824, usWinAscent=824, usWinDescent=200)
fb.setupPost()
fb.save("test.otf")
```
"""

from .ttLib import TTFont, newTable
from .ttLib.tables._c_m_a_p import cmap_classes
from .ttLib.tables._g_l_y_f import flagCubic
from .ttLib.tables.O_S_2f_2 import Panose
from .misc.timeTools import timestampNow
import struct
from collections import OrderedDict

_headDefaults = dict(
    tableVersion=1.0,
    fontRevision=1.0,
    checkSumAdjustment=0,
    magicNumber=0x5F0F3CF5,
    flags=0x0003,
    unitsPerEm=1000,
    created=0,
    modified=0,
    xMin=0,
    yMin=0,
    xMax=0,
    yMax=0,
    macStyle=0,
    lowestRecPPEM=3,
    fontDirectionHint=2,
    indexToLocFormat=0,
    glyphDataFormat=0,
)

_maxpDefaultsTTF = dict(
    tableVersion=0x00010000,
    numGlyphs=0,
    maxPoints=0,
    maxContours=0,
    maxCompositePoints=0,
    maxCompositeContours=0,
    maxZones=2,
    maxTwilightPoints=0,
    maxStorage=0,
    maxFunctionDefs=0,
    maxInstructionDefs=0,
    maxStackElements=0,
    maxSizeOfInstructions=0,
    maxComponentElements=0,
    maxComponentDepth=0,
)
_maxpDefaultsOTF = dict(
    tableVersion=0x00005000,
    numGlyphs=0,
)

_postDefaults = dict(
    formatType=3.0,
    italicAngle=0,
    underlinePosition=0,
    underlineThickness=0,
    isFixedPitch=0,
    minMemType42=0,
    maxMemType42=0,
    minMemType1=0,
    maxMemType1=0,
)

_hheaDefaults = dict(
    tableVersion=0x00010000,
    ascent=0,
    descent=0,
    lineGap=0,
    advanceWidthMax=0,
    minLeftSideBearing=0,
    minRightSideBearing=0,
    xMaxExtent=0,
    caretSlopeRise=1,
    caretSlopeRun=0,
    caretOffset=0,
    reserved0=0,
    reserved1=0,
    reserved2=0,
    reserved3=0,
    metricDataFormat=0,
    numberOfHMetrics=0,
)

_vheaDefaults = dict(
    tableVersion=0x00010000,
    ascent=0,
    descent=0,
    lineGap=0,
    advanceHeightMax=0,
    minTopSideBearing=0,
    minBottomSideBearing=0,
    yMaxExtent=0,
    caretSlopeRise=0,
    caretSlopeRun=0,
    reserved0=0,
    reserved1=0,
    reserved2=0,
    reserved3=0,
    reserved4=0,
    metricDataFormat=0,
    numberOfVMetrics=0,
)

_nameIDs = dict(
    copyright=0,
    familyName=1,
    styleName=2,
    uniqueFontIdentifier=3,
    fullName=4,
    version=5,
    psName=6,
    trademark=7,
    manufacturer=8,
    designer=9,
    description=10,
    vendorURL=11,
    designerURL=12,
    licenseDescription=13,
    licenseInfoURL=14,
    # reserved = 15,
    typographicFamily=16,
    typographicSubfamily=17,
    compatibleFullName=18,
    sampleText=19,
    postScriptCIDFindfontName=20,
    wwsFamilyName=21,
    wwsSubfamilyName=22,
    lightBackgroundPalette=23,
    darkBackgroundPalette=24,
    variationsPostScriptNamePrefix=25,
)

# to insert in setupNameTable doc string:
# print("\n".join(("%s (nameID %s)" % (k, v)) for k, v in sorted(_nameIDs.items(), key=lambda x: x[1])))


def _getOS2Defaults():
    return dict(
        version=3,
        xAvgCharWidth=0,
        usWeightClass=400,
        usWidthClass=5,
        fsType=0x0004,  # default: Preview & Print embedding
        ySubscriptXSize=0,
        ySubscriptYSize=0,
        ySubscriptXOffset=0,
        ySubscriptYOffset=0,
        ySuperscriptXSize=0,
        ySuperscriptYSize=0,
        ySuperscriptXOffset=0,
        ySuperscriptYOffset=0,
        yStrikeoutSize=0,
        yStrikeoutPosition=0,
        sFamilyClass=0,
        panose=Panose(),
        ulUnicodeRange1=0,
        ulUnicodeRange2=0,
        ulUnicodeRange3=0,
        ulUnicodeRange4=0,
        achVendID="????",
        fsSelection=0,
        usFirstCharIndex=0,
        usLastCharIndex=0,
        sTypoAscender=0,
        sTypoDescender=0,
        sTypoLineGap=0,
        usWinAscent=0,
        usWinDescent=0,
        ulCodePageRange1=0,
        ulCodePageRange2=0,
        sxHeight=0,
        sCapHeight=0,
        usDefaultChar=0,  # .notdef
        usBreakChar=32,  # space
        usMaxContext=0,
        usLowerOpticalPointSize=0,
        usUpperOpticalPointSize=0,
    )


class FontBuilder(object):
    def __init__(self, unitsPerEm=None, font=None, isTTF=True, glyphDataFormat=0):
        """Initialize a FontBuilder instance.

        If the `font` argument is not given, a new `TTFont` will be
        constructed, and `unitsPerEm` must be given. If `isTTF` is True,
        the font will be a glyf-based TTF; if `isTTF` is False it will be
        a CFF-based OTF.

        The `glyphDataFormat` argument corresponds to the `head` table field
        that defines the format of the TrueType `glyf` table (default=0).
        TrueType glyphs historically can only contain quadratic splines and static
        components, but there's a proposal to add support for cubic Bezier curves as well
        as variable composites/components at
        https://github.com/harfbuzz/boring-expansion-spec/blob/main/glyf1.md
        You can experiment with the new features by setting `glyphDataFormat` to 1.
        A ValueError is raised if `glyphDataFormat` is left at 0 but glyphs are added
        that contain cubic splines or varcomposites. This is to prevent accidentally
        creating fonts that are incompatible with existing TrueType implementations.

        If `font` is given, it must be a `TTFont` instance and `unitsPerEm`
        must _not_ be given. The `isTTF` and `glyphDataFormat` arguments will be ignored.
        """
        if font is None:
            self.font = TTFont(recalcTimestamp=False)
            self.isTTF = isTTF
            now = timestampNow()
            assert unitsPerEm is not None
            self.setupHead(
                unitsPerEm=unitsPerEm,
                created=now,
                modified=now,
                glyphDataFormat=glyphDataFormat,
            )
            self.setupMaxp()
        else:
            assert unitsPerEm is None
            self.font = font
            self.isTTF = "glyf" in font

    def save(self, file):
        """Save the font. The 'file' argument can be either a pathname or a
        writable file object.
        """
        self.font.save(file)

    def _initTableWithValues(self, tableTag, defaults, values):
        table = self.font[tableTag] = newTable(tableTag)
        for k, v in defaults.items():
            setattr(table, k, v)
        for k, v in values.items():
            setattr(table, k, v)
        return table

    def _updateTableWithValues(self, tableTag, values):
        table = self.font[tableTag]
        for k, v in values.items():
            setattr(table, k, v)

    def setupHead(self, **values):
        """Create a new `head` table and initialize it with default values,
        which can be overridden by keyword arguments.
        """
        self._initTableWithValues("head", _headDefaults, values)

    def updateHead(self, **values):
        """Update the head table with the fields and values passed as
        keyword arguments.
        """
        self._updateTableWithValues("head", values)

    def setupGlyphOrder(self, glyphOrder):
        """Set the glyph order for the font."""
        self.font.setGlyphOrder(glyphOrder)

    def setupCharacterMap(self, cmapping, uvs=None, allowFallback=False):
        """Build the `cmap` table for the font. The `cmapping` argument should
        be a dict mapping unicode code points as integers to glyph names.

        The `uvs` argument, when passed, must be a list of tuples, describing
        Unicode Variation Sequences. These tuples have three elements:
            (unicodeValue, variationSelector, glyphName)
        `unicodeValue` and `variationSelector` are integer code points.
        `glyphName` may be None, to indicate this is the default variation.
        Text processors will then use the cmap to find the glyph name.
        Each Unicode Variation Sequence should be an officially supported
        sequence, but this is not policed.
        """
        subTables = []
        highestUnicode = max(cmapping) if cmapping else 0
        if highestUnicode > 0xFFFF:
            cmapping_3_1 = dict((k, v) for k, v in cmapping.items() if k < 0x10000)
            subTable_3_10 = buildCmapSubTable(cmapping, 12, 3, 10)
            subTables.append(subTable_3_10)
        else:
            cmapping_3_1 = cmapping
        format = 4
        subTable_3_1 = buildCmapSubTable(cmapping_3_1, format, 3, 1)
        try:
            subTable_3_1.compile(self.font)
        except struct.error:
            # format 4 overflowed, fall back to format 12
            if not allowFallback:
                raise ValueError(
                    "cmap format 4 subtable overflowed; sort glyph order by unicode to fix."
                )
            format = 12
            subTable_3_1 = buildCmapSubTable(cmapping_3_1, format, 3, 1)
        subTables.append(subTable_3_1)
        subTable_0_3 = buildCmapSubTable(cmapping_3_1, format, 0, 3)
        subTables.append(subTable_0_3)

        if uvs is not None:
            uvsDict = {}
            for unicodeValue, variationSelector, glyphName in uvs:
                if cmapping.get(unicodeValue) == glyphName:
                    # this is a default variation
                    glyphName = None
                if variationSelector not in uvsDict:
                    uvsDict[variationSelector] = []
                uvsDict[variationSelector].append((unicodeValue, glyphName))
            uvsSubTable = buildCmapSubTable({}, 14, 0, 5)
            uvsSubTable.uvsDict = uvsDict
            subTables.append(uvsSubTable)

        self.font["cmap"] = newTable("cmap")
        self.font["cmap"].tableVersion = 0
        self.font["cmap"].tables = subTables

    def setupNameTable(self, nameStrings, windows=True, mac=True):
        """Create the `name` table for the font. The `nameStrings` argument must
        be a dict, mapping nameIDs or descriptive names for the nameIDs to name
        record values. A value is either a string, or a dict, mapping language codes
        to strings, to allow localized name table entries.

        By default, both Windows (platformID=3) and Macintosh (platformID=1) name
        records are added, unless any of `windows` or `mac` arguments is False.

        The following descriptive names are available for nameIDs:

            copyright (nameID 0)
            familyName (nameID 1)
            styleName (nameID 2)
            uniqueFontIdentifier (nameID 3)
            fullName (nameID 4)
            version (nameID 5)
            psName (nameID 6)
            trademark (nameID 7)
            manufacturer (nameID 8)
            designer (nameID 9)
            description (nameID 10)
            vendorURL (nameID 11)
            designerURL (nameID 12)
            licenseDescription (nameID 13)
            licenseInfoURL (nameID 14)
            typographicFamily (nameID 16)
            typographicSubfamily (nameID 17)
            compatibleFullName (nameID 18)
            sampleText (nameID 19)
            postScriptCIDFindfontName (nameID 20)
            wwsFamilyName (nameID 21)
            wwsSubfamilyName (nameID 22)
            lightBackgroundPalette (nameID 23)
            darkBackgroundPalette (nameID 24)
            variationsPostScriptNamePrefix (nameID 25)
        """
        nameTable = self.font["name"] = newTable("name")
        nameTable.names = []

        for nameName, nameValue in nameStrings.items():
            if isinstance(nameName, int):
                nameID = nameName
            else:
                nameID = _nameIDs[nameName]
            if isinstance(nameValue, str):
                nameValue = dict(en=nameValue)
            nameTable.addMultilingualName(
                nameValue, ttFont=self.font, nameID=nameID, windows=windows, mac=mac
            )

    def setupOS2(self, **values):
        """Create a new `OS/2` table and initialize it with default values,
        which can be overridden by keyword arguments.
        """
        self._initTableWithValues("OS/2", _getOS2Defaults(), values)
        if "xAvgCharWidth" not in values:
            assert (
                "hmtx" in self.font
            ), "the 'hmtx' table must be setup before the 'OS/2' table"
            self.font["OS/2"].recalcAvgCharWidth(self.font)
        if not (
            "ulUnicodeRange1" in values
            or "ulUnicodeRange2" in values
            or "ulUnicodeRange3" in values
            or "ulUnicodeRange3" in values
        ):
            assert (
                "cmap" in self.font
            ), "the 'cmap' table must be setup before the 'OS/2' table"
            self.font["OS/2"].recalcUnicodeRanges(self.font)

    def setupCFF(self, psName, fontInfo, charStringsDict, privateDict):
        from .cffLib import (
            CFFFontSet,
            TopDictIndex,
            TopDict,
            CharStrings,
            GlobalSubrsIndex,
            PrivateDict,
        )

        assert not self.isTTF
        self.font.sfntVersion = "OTTO"
        fontSet = CFFFontSet()
        fontSet.major = 1
        fontSet.minor = 0
        fontSet.otFont = self.font
        fontSet.fontNames = [psName]
        fontSet.topDictIndex = TopDictIndex()

        globalSubrs = GlobalSubrsIndex()
        fontSet.GlobalSubrs = globalSubrs
        private = PrivateDict()
        for key, value in privateDict.items():
            setattr(private, key, value)
        fdSelect = None
        fdArray = None

        topDict = TopDict()
        topDict.charset = self.font.getGlyphOrder()
        topDict.Private = private
        topDict.GlobalSubrs = fontSet.GlobalSubrs
        for key, value in fontInfo.items():
            setattr(topDict, key, value)
        if "FontMatrix" not in fontInfo:
            scale = 1 / self.font["head"].unitsPerEm
            topDict.FontMatrix = [scale, 0, 0, scale, 0, 0]

        charStrings = CharStrings(
            None, topDict.charset, globalSubrs, private, fdSelect, fdArray
        )
        for glyphName, charString in charStringsDict.items():
            charString.private = private
            charString.globalSubrs = globalSubrs
            charStrings[glyphName] = charString
        topDict.CharStrings = charStrings

        fontSet.topDictIndex.append(topDict)

        self.font["CFF "] = newTable("CFF ")
        self.font["CFF "].cff = fontSet

    def setupCFF2(self, charStringsDict, fdArrayList=None, regions=None):
        from .cffLib import (
            CFFFontSet,
            TopDictIndex,
            TopDict,
            CharStrings,
            GlobalSubrsIndex,
            PrivateDict,
            FDArrayIndex,
            FontDict,
        )

        assert not self.isTTF
        self.font.sfntVersion = "OTTO"
        fontSet = CFFFontSet()
        fontSet.major = 2
        fontSet.minor = 0

        cff2GetGlyphOrder = self.font.getGlyphOrder
        fontSet.topDictIndex = TopDictIndex(None, cff2GetGlyphOrder, None)

        globalSubrs = GlobalSubrsIndex()
        fontSet.GlobalSubrs = globalSubrs

        if fdArrayList is None:
            fdArrayList = [{}]
        fdSelect = None
        fdArray = FDArrayIndex()
        fdArray.strings = None
        fdArray.GlobalSubrs = globalSubrs
        for privateDict in fdArrayList:
            fontDict = FontDict()
            fontDict.setCFF2(True)
            private = PrivateDict()
            for key, value in privateDict.items():
                setattr(private, key, value)
            fontDict.Private = private
            fdArray.append(fontDict)

        topDict = TopDict()
        topDict.cff2GetGlyphOrder = cff2GetGlyphOrder
        topDict.FDArray = fdArray
        scale = 1 / self.font["head"].unitsPerEm
        topDict.FontMatrix = [scale, 0, 0, scale, 0, 0]

        private = fdArray[0].Private
        charStrings = CharStrings(None, None, globalSubrs, private, fdSelect, fdArray)
        for glyphName, charString in charStringsDict.items():
            charString.private = private
            charString.globalSubrs = globalSubrs
            charStrings[glyphName] = charString
        topDict.CharStrings = charStrings

        fontSet.topDictIndex.append(topDict)

        self.font["CFF2"] = newTable("CFF2")
        self.font["CFF2"].cff = fontSet

        if regions:
            self.setupCFF2Regions(regions)

    def setupCFF2Regions(self, regions):
        from .varLib.builder import buildVarRegionList, buildVarData, buildVarStore
        from .cffLib import VarStoreData

        assert "fvar" in self.font, "fvar must to be set up first"
        assert "CFF2" in self.font, "CFF2 must to be set up first"
        axisTags = [a.axisTag for a in self.font["fvar"].axes]
        varRegionList = buildVarRegionList(regions, axisTags)
        varData = buildVarData(list(range(len(regions))), None, optimize=False)
        varStore = buildVarStore(varRegionList, [varData])
        vstore = VarStoreData(otVarStore=varStore)
        topDict = self.font["CFF2"].cff.topDictIndex[0]
        topDict.VarStore = vstore
        for fontDict in topDict.FDArray:
            fontDict.Private.vstore = vstore

    def setupGlyf(self, glyphs, calcGlyphBounds=True, validateGlyphFormat=True):
        """Create the `glyf` table from a dict, that maps glyph names
        to `fontTools.ttLib.tables._g_l_y_f.Glyph` objects, for example
        as made by `fontTools.pens.ttGlyphPen.TTGlyphPen`.

        If `calcGlyphBounds` is True, the bounds of all glyphs will be
        calculated. Only pass False if your glyph objects already have
        their bounding box values set.

        If `validateGlyphFormat` is True, raise ValueError if any of the glyphs contains
        cubic curves or is a variable composite but head.glyphDataFormat=0.
        Set it to False to skip the check if you know in advance all the glyphs are
        compatible with the specified glyphDataFormat.
        """
        assert self.isTTF

        if validateGlyphFormat and self.font["head"].glyphDataFormat == 0:
            for name, g in glyphs.items():
                if g.numberOfContours > 0 and any(f & flagCubic for f in g.flags):
                    raise ValueError(
                        f"Glyph {name!r} has cubic Bezier outlines, but glyphDataFormat=0; "
                        "either convert to quadratics with cu2qu or set glyphDataFormat=1."
                    )

        self.font["loca"] = newTable("loca")
        self.font["glyf"] = newTable("glyf")
        self.font["glyf"].glyphs = glyphs
        if hasattr(self.font, "glyphOrder"):
            self.font["glyf"].glyphOrder = self.font.glyphOrder
        if calcGlyphBounds:
            self.calcGlyphBounds()

    def setupFvar(self, axes, instances):
        """Adds an font variations table to the font.

        Args:
            axes (list): See below.
            instances (list): See below.

        ``axes`` should be a list of axes, with each axis either supplied as
        a py:class:`.designspaceLib.AxisDescriptor` object, or a tuple in the
        format ```tupletag, minValue, defaultValue, maxValue, name``.
        The ``name`` is either a string, or a dict, mapping language codes
        to strings, to allow localized name table entries.

        ```instances`` should be a list of instances, with each instance either
        supplied as a py:class:`.designspaceLib.InstanceDescriptor` object, or a
        dict with keys ``location`` (mapping of axis tags to float values),
        ``stylename`` and (optionally) ``postscriptfontname``.
        The ``stylename`` is either a string, or a dict, mapping language codes
        to strings, to allow localized name table entries.
        """

        addFvar(self.font, axes, instances)

    def setupAvar(self, axes, mappings=None):
        """Adds an axis variations table to the font.

        Args:
            axes (list): A list of py:class:`.designspaceLib.AxisDescriptor` objects.
        """
        from .varLib import _add_avar

        if "fvar" not in self.font:
            raise KeyError("'fvar' table is missing; can't add 'avar'.")

        axisTags = [axis.axisTag for axis in self.font["fvar"].axes]
        axes = OrderedDict(enumerate(axes))  # Only values are used
        _add_avar(self.font, axes, mappings, axisTags)

    def setupGvar(self, variations):
        gvar = self.font["gvar"] = newTable("gvar")
        gvar.version = 1
        gvar.reserved = 0
        gvar.variations = variations

    def setupGVAR(self, variations):
        gvar = self.font["GVAR"] = newTable("GVAR")
        gvar.version = 1
        gvar.reserved = 0
        gvar.variations = variations

    def calcGlyphBounds(self):
        """Calculate the bounding boxes of all glyphs in the `glyf` table.
        This is usually not called explicitly by client code.
        """
        glyphTable = self.font["glyf"]
        for glyph in glyphTable.glyphs.values():
            glyph.recalcBounds(glyphTable)

    def setupHorizontalMetrics(self, metrics):
        """Create a new `hmtx` table, for horizontal metrics.

        The `metrics` argument must be a dict, mapping glyph names to
        `(width, leftSidebearing)` tuples.
        """
        self.setupMetrics("hmtx", metrics)

    def setupVerticalMetrics(self, metrics):
        """Create a new `vmtx` table, for horizontal metrics.

        The `metrics` argument must be a dict, mapping glyph names to
        `(height, topSidebearing)` tuples.
        """
        self.setupMetrics("vmtx", metrics)

    def setupMetrics(self, tableTag, metrics):
        """See `setupHorizontalMetrics()` and `setupVerticalMetrics()`."""
        assert tableTag in ("hmtx", "vmtx")
        mtxTable = self.font[tableTag] = newTable(tableTag)
        roundedMetrics = {}
        for gn in metrics:
            w, lsb = metrics[gn]
            roundedMetrics[gn] = int(round(w)), int(round(lsb))
        mtxTable.metrics = roundedMetrics

    def setupHorizontalHeader(self, **values):
        """Create a new `hhea` table initialize it with default values,
        which can be overridden by keyword arguments.
        """
        self._initTableWithValues("hhea", _hheaDefaults, values)

    def setupVerticalHeader(self, **values):
        """Create a new `vhea` table initialize it with default values,
        which can be overridden by keyword arguments.
        """
        self._initTableWithValues("vhea", _vheaDefaults, values)

    def setupVerticalOrigins(self, verticalOrigins, defaultVerticalOrigin=None):
        """Create a new `VORG` table. The `verticalOrigins` argument must be
        a dict, mapping glyph names to vertical origin values.

        The `defaultVerticalOrigin` argument should be the most common vertical
        origin value. If omitted, this value will be derived from the actual
        values in the `verticalOrigins` argument.
        """
        if defaultVerticalOrigin is None:
            # find the most frequent vorg value
            bag = {}
            for gn in verticalOrigins:
                vorg = verticalOrigins[gn]
                if vorg not in bag:
                    bag[vorg] = 1
                else:
                    bag[vorg] += 1
            defaultVerticalOrigin = sorted(
                bag, key=lambda vorg: bag[vorg], reverse=True
            )[0]
        self._initTableWithValues(
            "VORG",
            {},
            dict(VOriginRecords={}, defaultVertOriginY=defaultVerticalOrigin),
        )
        vorgTable = self.font["VORG"]
        vorgTable.majorVersion = 1
        vorgTable.minorVersion = 0
        for gn in verticalOrigins:
            vorgTable[gn] = verticalOrigins[gn]

    def setupPost(self, keepGlyphNames=True, **values):
        """Create a new `post` table and initialize it with default values,
        which can be overridden by keyword arguments.
        """
        isCFF2 = "CFF2" in self.font
        postTable = self._initTableWithValues("post", _postDefaults, values)
        if (self.isTTF or isCFF2) and keepGlyphNames:
            postTable.formatType = 2.0
            postTable.extraNames = []
            postTable.mapping = {}
        else:
            postTable.formatType = 3.0

    def setupMaxp(self):
        """Create a new `maxp` table. This is called implicitly by FontBuilder
        itself and is usually not called by client code.
        """
        if self.isTTF:
            defaults = _maxpDefaultsTTF
        else:
            defaults = _maxpDefaultsOTF
        self._initTableWithValues("maxp", defaults, {})

    def setupDummyDSIG(self):
        """This adds an empty DSIG table to the font to make some MS applications
        happy. This does not properly sign the font.
        """
        values = dict(
            ulVersion=1,
            usFlag=0,
            usNumSigs=0,
            signatureRecords=[],
        )
        self._initTableWithValues("DSIG", {}, values)

    def addOpenTypeFeatures(self, features, filename=None, tables=None, debug=False):
        """Add OpenType features to the font from a string containing
        Feature File syntax.

        The `filename` argument is used in error messages and to determine
        where to look for "include" files.

        The optional `tables` argument can be a list of OTL tables tags to
        build, allowing the caller to only build selected OTL tables. See
        `fontTools.feaLib` for details.

        The optional `debug` argument controls whether to add source debugging
        information to the font in the `Debg` table.
        """
        from .feaLib.builder import addOpenTypeFeaturesFromString

        addOpenTypeFeaturesFromString(
            self.font, features, filename=filename, tables=tables, debug=debug
        )

    def addFeatureVariations(self, conditionalSubstitutions, featureTag="rvrn"):
        """Add conditional substitutions to a Variable Font.

        See `fontTools.varLib.featureVars.addFeatureVariations`.
        """
        from .varLib import featureVars

        if "fvar" not in self.font:
            raise KeyError("'fvar' table is missing; can't add FeatureVariations.")

        featureVars.addFeatureVariations(
            self.font, conditionalSubstitutions, featureTag=featureTag
        )

    def setupCOLR(
        self,
        colorLayers,
        version=None,
        varStore=None,
        varIndexMap=None,
        clipBoxes=None,
        allowLayerReuse=True,
    ):
        """Build new COLR table using color layers dictionary.

        Cf. `fontTools.colorLib.builder.buildCOLR`.
        """
        from fontTools.colorLib.builder import buildCOLR

        glyphMap = self.font.getReverseGlyphMap()
        self.font["COLR"] = buildCOLR(
            colorLayers,
            version=version,
            glyphMap=glyphMap,
            varStore=varStore,
            varIndexMap=varIndexMap,
            clipBoxes=clipBoxes,
            allowLayerReuse=allowLayerReuse,
        )

    def setupCPAL(
        self,
        palettes,
        paletteTypes=None,
        paletteLabels=None,
        paletteEntryLabels=None,
    ):
        """Build new CPAL table using list of palettes.

        Optionally build CPAL v1 table using paletteTypes, paletteLabels and
        paletteEntryLabels.

        Cf. `fontTools.colorLib.builder.buildCPAL`.
        """
        from fontTools.colorLib.builder import buildCPAL

        self.font["CPAL"] = buildCPAL(
            palettes,
            paletteTypes=paletteTypes,
            paletteLabels=paletteLabels,
            paletteEntryLabels=paletteEntryLabels,
            nameTable=self.font.get("name"),
        )

    def setupStat(self, axes, locations=None, elidedFallbackName=2):
        """Build a new 'STAT' table.

        See `fontTools.otlLib.builder.buildStatTable` for details about
        the arguments.
        """
        from .otlLib.builder import buildStatTable

        assert "name" in self.font, "name must to be set up first"

        buildStatTable(
            self.font,
            axes,
            locations,
            elidedFallbackName,
            macNames=any(nr.platformID == 1 for nr in self.font["name"].names),
        )


def buildCmapSubTable(cmapping, format, platformID, platEncID):
    subTable = cmap_classes[format](format)
    subTable.cmap = cmapping
    subTable.platformID = platformID
    subTable.platEncID = platEncID
    subTable.language = 0
    return subTable


def addFvar(font, axes, instances):
    from .ttLib.tables._f_v_a_r import Axis, NamedInstance

    assert axes

    fvar = newTable("fvar")
    nameTable = font["name"]

    # if there are not currently any mac names don't add them here, that's inconsistent
    # https://github.com/fonttools/fonttools/issues/683
    macNames = any(nr.platformID == 1 for nr in getattr(nameTable, "names", ()))

    # we have all the best ways to express mac names
    platforms = ((3, 1, 0x409),)
    if macNames:
        platforms = ((1, 0, 0),) + platforms

    for axis_def in axes:
        axis = Axis()

        if isinstance(axis_def, tuple):
            (
                axis.axisTag,
                axis.minValue,
                axis.defaultValue,
                axis.maxValue,
                name,
            ) = axis_def
        else:
            axis.axisTag, axis.minValue, axis.defaultValue, axis.maxValue, name = (
                axis_def.tag,
                axis_def.minimum,
                axis_def.default,
                axis_def.maximum,
                axis_def.name,
            )
            if axis_def.hidden:
                axis.flags = 0x0001  # HIDDEN_AXIS

        if isinstance(name, str):
            name = dict(en=name)

        axis.axisNameID = nameTable.addMultilingualName(name, ttFont=font, mac=macNames)
        fvar.axes.append(axis)

    for instance in instances:
        if isinstance(instance, dict):
            coordinates = instance["location"]
            name = instance["stylename"]
            psname = instance.get("postscriptfontname")
        else:
            coordinates = instance.location
            name = instance.localisedStyleName or instance.styleName
            psname = instance.postScriptFontName

        if isinstance(name, str):
            name = dict(en=name)

        inst = NamedInstance()
        inst.subfamilyNameID = nameTable.addMultilingualName(
            name, ttFont=font, mac=macNames
        )
        if psname is not None:
            inst.postscriptNameID = nameTable.addName(psname, platforms=platforms)
        inst.coordinates = coordinates
        fvar.instances.append(inst)

    font["fvar"] = fvar
<|endoftext|>import sys


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # TODO Handle library-wide options. Eg.:
    # --unicodedata
    # --verbose / other logging stuff

    # TODO Allow a way to run arbitrary modules? Useful for setting
    # library-wide options and calling another library. Eg.:
    #
    #   $ fonttools --unicodedata=... fontmake ...
    #
    # This allows for a git-like command where thirdparty commands
    # can be added.  Should we just try importing the fonttools
    # module first and try without if it fails?

    if len(sys.argv) < 2:
        sys.argv.append("help")
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        sys.argv[1] = "help"
    mod = "fontTools." + sys.argv[1]
    sys.argv[1] = sys.argv[0] + " " + sys.argv[1]
    del sys.argv[0]

    import runpy

    runpy.run_module(mod, run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
<|endoftext|>"""
Generic module for reading and writing the .glif format.

More info about the .glif format (GLyphInterchangeFormat) can be found here:

        http://unifiedfontobject.org

The main class in this module is :class:`GlyphSet`. It manages a set of .glif files
in a folder. It offers two ways to read glyph data, and one way to write
glyph data. See the class doc string for details.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from warnings import warn

import fontTools.misc.filesystem as fs
from fontTools.misc import etree, plistlib
from fontTools.misc.textTools import tobytes
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from fontTools.ufoLib import UFOFormatVersion, _UFOBaseIO
from fontTools.ufoLib.errors import GlifLibError
from fontTools.ufoLib.filenames import userNameToFileName
from fontTools.ufoLib.utils import (
    BaseFormatVersion,
    normalizeFormatVersion,
    numberTypes,
)
from fontTools.ufoLib.validators import (
    anchorsValidator,
    colorValidator,
    genericTypeValidator,
    glyphLibValidator,
    guidelinesValidator,
    identifierValidator,
    imageValidator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Set
    from logging import Logger

    from fontTools.annotations import (
        ElementType,
        FormatVersion,
        FormatVersions,
        GLIFFormatVersionInput,
        GlyphNameToFileNameFunc,
        IntFloat,
        PathOrFS,
        UFOFormatVersionInput,
    )
    from fontTools.misc.filesystem._base import FS


__all__: list[str] = [
    "GlyphSet",
    "GlifLibError",
    "readGlyphFromString",
    "writeGlyphToString",
    "glyphNameToFileName",
]

logger: Logger = logging.getLogger(__name__)


# ---------
# Constants
# ---------

CONTENTS_FILENAME = "contents.plist"
LAYERINFO_FILENAME = "layerinfo.plist"


class GLIFFormatVersion(BaseFormatVersion):
    """Class representing the versions of the .glif format supported by the UFO version in use.

    For a given :mod:`fontTools.ufoLib.UFOFormatVersion`, the :func:`supported_versions` method will
    return the supported versions of the GLIF file format. If the UFO version is unspecified, the
    :func:`supported_versions` method will return all available GLIF format versions.
    """

    FORMAT_1_0 = (1, 0)
    FORMAT_2_0 = (2, 0)

    @classmethod
    def default(
        cls, ufoFormatVersion: Optional[UFOFormatVersion] = None
    ) -> GLIFFormatVersion:
        if ufoFormatVersion is not None:
            return max(cls.supported_versions(ufoFormatVersion))
        return super().default()

    @classmethod
    def supported_versions(
        cls, ufoFormatVersion: Optional[UFOFormatVersion] = None
    ) -> frozenset[GLIFFormatVersion]:
        if ufoFormatVersion is None:
            # if ufo format unspecified, return all the supported GLIF formats
            return super().supported_versions()
        # else only return the GLIF formats supported by the given UFO format
        versions = {cls.FORMAT_1_0}
        if ufoFormatVersion >= UFOFormatVersion.FORMAT_3_0:
            versions.add(cls.FORMAT_2_0)
        return frozenset(versions)


# ------------
# Simple Glyph
# ------------


class Glyph:
    """
    Minimal glyph object. It has no glyph attributes until either
    the draw() or the drawPoints() method has been called.
    """

    def __init__(self, glyphName: str, glyphSet: GlyphSet) -> None:
        self.glyphName: str = glyphName
        self.glyphSet: GlyphSet = glyphSet

    def draw(self, pen: Any, outputImpliedClosingLine: bool = False) -> None:
        """
        Draw this glyph onto a *FontTools* Pen.
        """
        pointPen = PointToSegmentPen(
            pen, outputImpliedClosingLine=outputImpliedClosingLine
        )
        self.drawPoints(pointPen)

    def drawPoints(self, pointPen: AbstractPointPen) -> None:
        """
        Draw this glyph onto a PointPen.
        """
        self.glyphSet.readGlyph(self.glyphName, self, pointPen)


# ---------
# Glyph Set
# ---------


class GlyphSet(_UFOBaseIO):
    """
    GlyphSet manages a set of .glif files inside one directory.

    GlyphSet's constructor takes a path to an existing directory as it's
    first argument. Reading glyph data can either be done through the
    readGlyph() method, or by using GlyphSet's dictionary interface, where
    the keys are glyph names and the values are (very) simple glyph objects.

    To write a glyph to the glyph set, you use the writeGlyph() method.
    The simple glyph objects returned through the dict interface do not
    support writing, they are just a convenient way to get at the glyph data.
    """

    glyphClass = Glyph

    def __init__(
        self,
        path: PathOrFS,
        glyphNameToFileNameFunc: GlyphNameToFileNameFunc = None,
        ufoFormatVersion: UFOFormatVersionInput = None,
        validateRead: bool = True,
        validateWrite: bool = True,
        expectContentsFile: bool = False,
    ) -> None:
        """
        'path' should be a path (string) to an existing local directory, or
        an instance of fs.base.FS class.

        The optional 'glyphNameToFileNameFunc' argument must be a callback
        function that takes two arguments: a glyph name and a list of all
        existing filenames (if any exist). It should return a file name
        (including the .glif extension). The glyphNameToFileName function
        is called whenever a file name is created for a given glyph name.

        ``validateRead`` will validate read operations. Its default is ``True``.
        ``validateWrite`` will validate write operations. Its default is ``True``.
        ``expectContentsFile`` will raise a GlifLibError if a contents.plist file is
        not found on the glyph set file system. This should be set to ``True`` if you
        are reading an existing UFO and ``False`` if you create a fresh	glyph set.
        """
        try:
            ufoFormatVersion = normalizeFormatVersion(
                ufoFormatVersion, UFOFormatVersion
            )
        except ValueError as e:
            from fontTools.ufoLib.errors import UnsupportedUFOFormat

            raise UnsupportedUFOFormat(
                f"Unsupported UFO format: {ufoFormatVersion!r}"
            ) from e

        if hasattr(path, "__fspath__"):  # support os.PathLike objects
            path = path.__fspath__()

        if isinstance(path, str):
            try:
                filesystem: FS = fs.osfs.OSFS(path)
            except fs.errors.CreateFailed:
                raise GlifLibError("No glyphs directory '%s'" % path)
            self._shouldClose: bool = True
        elif isinstance(path, fs.base.FS):
            filesystem = path
            try:
                filesystem.check()
            except fs.errors.FilesystemClosed:
                raise GlifLibError("the filesystem '%s' is closed" % filesystem)
            self._shouldClose = False
        else:
            raise TypeError(
                "Expected a path string or fs object, found %s" % type(path).__name__
            )
        try:
            path = filesystem.getsyspath("/")
        except fs.errors.NoSysPath:
            # network or in-memory FS may not map to the local one
            path = str(filesystem)
        # 'dirName' is kept for backward compatibility only, but it's DEPRECATED
        # as it's not guaranteed that it maps to an existing OSFS directory.
        # Client could use the FS api via the `self.fs` attribute instead.
        self.dirName: str = fs.path.basename(path)
        self.fs: FS = filesystem
        # if glyphSet contains no 'contents.plist', we consider it empty
        self._havePreviousFile: bool = filesystem.exists(CONTENTS_FILENAME)
        if expectContentsFile and not self._havePreviousFile:
            raise GlifLibError(f"{CONTENTS_FILENAME} is missing.")
        # attribute kept for backward compatibility
        self.ufoFormatVersion: int = ufoFormatVersion.major
        self.ufoFormatVersionTuple: UFOFormatVersion = ufoFormatVersion
        if glyphNameToFileNameFunc is None:
            glyphNameToFileNameFunc = glyphNameToFileName
        self.glyphNameToFileName: Callable[[str, set[str]], str] = (
            glyphNameToFileNameFunc
        )
        self._validateRead: bool = validateRead
        self._validateWrite: bool = validateWrite
        self._existingFileNames: set[str] | None = None
        self._reverseContents: Optional[dict[str, str]] = None

        self.rebuildContents()

    def rebuildContents(self, validateRead: bool = False) -> None:
        """
        Rebuild the contents dict by loading contents.plist.

        ``validateRead`` will validate the data, by default it is set to the
        class's ``validateRead`` value, can be overridden.
        """
        if validateRead is None:
            validateRead = self._validateRead
        contents = self._getPlist(CONTENTS_FILENAME, {})
        # validate the contents
        if validateRead:
            invalidFormat = False
            if not isinstance(contents, dict):
                invalidFormat = True
            else:
                for name, fileName in contents.items():
                    if not isinstance(name, str):
                        invalidFormat = True
                    if not isinstance(fileName, str):
                        invalidFormat = True
                    elif not self.fs.exists(fileName):
                        raise GlifLibError(
                            "%s references a file that does not exist: %s"
                            % (CONTENTS_FILENAME, fileName)
                        )
            if invalidFormat:
                raise GlifLibError("%s is not properly formatted" % CONTENTS_FILENAME)
        self.contents: dict[str, str] = contents
        self._existingFileNames = None
        self._reverseContents = None

    def getReverseContents(self) -> dict[str, str]:
        """
        Return a reversed dict of self.contents, mapping file names to
        glyph names. This is primarily an aid for custom glyph name to file
        name schemes that want to make sure they don't generate duplicate
        file names. The file names are converted to lowercase so we can
        reliably check for duplicates that only differ in case, which is
        important for case-insensitive file systems.
        """
        if self._reverseContents is None:
            d = {}
            for k, v in self.contents.items():
                d[v.lower()] = k
            self._reverseContents = d
        return self._reverseContents

    def writeContents(self) -> None:
        """
        Write the contents.plist file out to disk. Call this method when
        you're done writing glyphs.
        """
        self._writePlist(CONTENTS_FILENAME, self.contents)

    # layer info

    def readLayerInfo(self, info: Any, validateRead: Optional[bool] = None) -> None:
        """
        ``validateRead`` will validate the data, by default it is set to the
        class's ``validateRead`` value, can be overridden.
        """
        if validateRead is None:
            validateRead = self._validateRead
        infoDict = self._getPlist(LAYERINFO_FILENAME, {})
        if validateRead:
            if not isinstance(infoDict, dict):
                raise GlifLibError("layerinfo.plist is not properly formatted.")
            infoDict = validateLayerInfoVersion3Data(infoDict)
        # populate the object
        for attr, value in infoDict.items():
            try:
                setattr(info, attr, value)
            except AttributeError:
                raise GlifLibError(
                    "The supplied layer info object does not support setting a necessary attribute (%s)."
                    % attr
                )

    def writeLayerInfo(self, info: Any, validateWrite: Optional[bool] = None) -> None:
        """
        ``validateWrite`` will validate the data, by default it is set to the
        class's ``validateWrite`` value, can be overridden.
        """
        if validateWrite is None:
            validateWrite = self._validateWrite
        if self.ufoFormatVersionTuple.major < 3:
            raise GlifLibError(
                "layerinfo.plist is not allowed in UFO %d."
                % self.ufoFormatVersionTuple.major
            )
        # gather data
        infoData = {}
        for attr in layerInfoVersion3ValueData.keys():
            if hasattr(info, attr):
                try:
                    value = getattr(info, attr)
                except AttributeError:
                    raise GlifLibError(
                        "The supplied info object does not support getting a necessary attribute (%s)."
                        % attr
                    )
                if value is None or (attr == "lib" and not value):
                    continue
                infoData[attr] = value
        if infoData:
            # validate
            if validateWrite:
                infoData = validateLayerInfoVersion3Data(infoData)
            # write file
            self._writePlist(LAYERINFO_FILENAME, infoData)
        elif self._havePreviousFile and self.fs.exists(LAYERINFO_FILENAME):
            # data empty, remove existing file
            self.fs.remove(LAYERINFO_FILENAME)

    def getGLIF(self, glyphName: str) -> bytes:
        """
        Get the raw GLIF text for a given glyph name. This only works
        for GLIF files that are already on disk.

        This method is useful in situations when the raw XML needs to be
        read from a glyph set for a particular glyph before fully parsing
        it into an object structure via the readGlyph method.

        Raises KeyError if 'glyphName' is not in contents.plist, or
        GlifLibError if the file associated with can't be found.
        """
        fileName = self.contents[glyphName]
        try:
            return self.fs.readbytes(fileName)
        except fs.errors.ResourceNotFound:
            raise GlifLibError(
                "The file '%s' associated with glyph '%s' in contents.plist "
                "does not exist on %s" % (fileName, glyphName, self.fs)
            )

    def getGLIFModificationTime(self, glyphName: str) -> Optional[float]:
        """
        Returns the modification time for the GLIF file with 'glyphName', as
        a floating point number giving the number of seconds since the epoch.
        Return None if the associated file does not exist or the underlying
        filesystem does not support getting modified times.
        Raises KeyError if the glyphName is not in contents.plist.
        """
        fileName = self.contents[glyphName]
        return self.getFileModificationTime(fileName)

    # reading/writing API

    def readGlyph(
        self,
        glyphName: str,
        glyphObject: Optional[Any] = None,
        pointPen: Optional[AbstractPointPen] = None,
        validate: Optional[bool] = None,
    ) -> None:
        """
        Read a .glif file for 'glyphName' from the glyph set. The
        'glyphObject' argument can be any kind of object (even None);
        the readGlyph() method will attempt to set the following
        attributes on it:

        width
                the advance width of the glyph
        height
                the advance height of the glyph
        unicodes
                a list of unicode values for this glyph
        note
                a string
        lib
                a dictionary containing custom data
        image
                a dictionary containing image data
        guidelines
                a list of guideline data dictionaries
        anchors
                a list of anchor data dictionaries

        All attributes are optional, in two ways:

        1) An attribute *won't* be set if the .glif file doesn't
           contain data for it. 'glyphObject' will have to deal
           with default values itself.
        2) If setting the attribute fails with an AttributeError
           (for example if the 'glyphObject' attribute is read-
           only), readGlyph() will not propagate that exception,
           but ignore that attribute.

        To retrieve outline information, you need to pass an object
        conforming to the PointPen protocol as the 'pointPen' argument.
        This argument may be None if you don't need the outline data.

        readGlyph() will raise KeyError if the glyph is not present in
        the glyph set.

        ``validate`` will validate the data, by default it is set to the
        class's ``validateRead`` value, can be overridden.
        """
        if validate is None:
            validate = self._validateRead
        text = self.getGLIF(glyphName)
        try:
            tree = _glifTreeFromString(text)
            formatVersions = GLIFFormatVersion.supported_versions(
                self.ufoFormatVersionTuple
            )
            _readGlyphFromTree(
                tree,
                glyphObject,
                pointPen,
                formatVersions=formatVersions,
                validate=validate,
            )
        except GlifLibError as glifLibError:
            # Re-raise with a note that gives extra context, describing where
            # the error occurred.
            fileName = self.contents[glyphName]
            try:
                glifLocation = f"'{self.fs.getsyspath(fileName)}'"
            except fs.errors.NoSysPath:
                # Network or in-memory FS may not map to a local path, so use
                # the best string representation we have.
                glifLocation = f"'{fileName}' from '{str(self.fs)}'"

            glifLibError._add_note(
                f"The issue is in glyph '{glyphName}', located in {glifLocation}."
            )
            raise

    def writeGlyph(
        self,
        glyphName: str,
        glyphObject: Optional[Any] = None,
        drawPointsFunc: Optional[Callable[[AbstractPointPen], None]] = None,
        formatVersion: GLIFFormatVersionInput = None,
        validate: Optional[bool] = None,
    ) -> None:
        """
        Write a .glif file for 'glyphName' to the glyph set. The
        'glyphObject' argument can be any kind of object (even None);
        the writeGlyph() method will attempt to get the following
        attributes from it:

        width
                the advance width of the glyph
        height
                the advance height of the glyph
        unicodes
                a list of unicode values for this glyph
        note
                a string
        lib
                a dictionary containing custom data
        image
                a dictionary containing image data
        guidelines
                a list of guideline data dictionaries
        anchors
                a list of anchor data dictionaries

        All attributes are optional: if 'glyphObject' doesn't
        have the attribute, it will simply be skipped.

        To write outline data to the .glif file, writeGlyph() needs
        a function (any callable object actually) that will take one
        argument: an object that conforms to the PointPen protocol.
        The function will be called by writeGlyph(); it has to call the
        proper PointPen methods to transfer the outline to the .glif file.

        The GLIF format version will be chosen based on the ufoFormatVersion
        passed during the creation of this object. If a particular format
        version is desired, it can be passed with the formatVersion argument.
        The formatVersion argument accepts either a tuple of integers for
        (major, minor), or a single integer for the major digit only (with
        minor digit implied as 0).

        An UnsupportedGLIFFormat exception is raised if the requested GLIF
        formatVersion is not supported.

        ``validate`` will validate the data, by default it is set to the
        class's ``validateWrite`` value, can be overridden.
        """
        if formatVersion is None:
            formatVersion = GLIFFormatVersion.default(self.ufoFormatVersionTuple)
        else:
            try:
                formatVersion = normalizeFormatVersion(formatVersion, GLIFFormatVersion)
            except ValueError as e:
                from fontTools.ufoLib.errors import UnsupportedGLIFFormat

                raise UnsupportedGLIFFormat(
                    f"Unsupported GLIF format version: {formatVersion!r}"
                ) from e
        if formatVersion not in GLIFFormatVersion.supported_versions(
            self.ufoFormatVersionTuple
        ):
            from fontTools.ufoLib.errors import UnsupportedGLIFFormat

            raise UnsupportedGLIFFormat(
                f"Unsupported GLIF format version ({formatVersion!s}) "
                f"for UFO format version {self.ufoFormatVersionTuple!s}."
            )
        if validate is None:
            validate = self._validateWrite
        fileName = self.contents.get(glyphName)
        if fileName is None:
            if self._existingFileNames is None:
                self._existingFileNames = {
                    fileName.lower() for fileName in self.contents.values()
                }
            fileName = self.glyphNameToFileName(glyphName, self._existingFileNames)
            self.contents[glyphName] = fileName
            self._existingFileNames.add(fileName.lower())
            if self._reverseContents is not None:
                self._reverseContents[fileName.lower()] = glyphName
        data = _writeGlyphToBytes(
            glyphName,
            glyphObject,
            drawPointsFunc,
            formatVersion=formatVersion,
            validate=validate,
        )
        if (
            self._havePreviousFile
            and self.fs.exists(fileName)
            and data == self.fs.readbytes(fileName)
        ):
            return
        self.fs.writebytes(fileName, data)

    def deleteGlyph(self, glyphName: str) -> None:
        """Permanently delete the glyph from the glyph set on disk. Will
        raise KeyError if the glyph is not present in the glyph set.
        """
        fileName = self.contents[glyphName]
        self.fs.remove(fileName)
        if self._existingFileNames is not None:
            self._existingFileNames.remove(fileName.lower())
        if self._reverseContents is not None:
            del self._reverseContents[fileName.lower()]
        del self.contents[glyphName]

    # dict-like support

    def keys(self) -> list[str]:
        return list(self.contents.keys())

    def has_key(self, glyphName: str) -> bool:
        return glyphName in self.contents

    __contains__ = has_key

    def __len__(self) -> int:
        return len(self.contents)

    def __getitem__(self, glyphName: str) -> Any:
        if glyphName not in self.contents:
            raise KeyError(glyphName)
        return self.glyphClass(glyphName, self)

    # quickly fetch unicode values

    def getUnicodes(
        self, glyphNames: Optional[Iterable[str]] = None
    ) -> dict[str, list[int]]:
        """
        Return a dictionary that maps glyph names to lists containing
        the unicode value[s] for that glyph, if any. This parses the .glif
        files partially, so it is a lot faster than parsing all files completely.
        By default this checks all glyphs, but a subset can be passed with glyphNames.
        """
        unicodes = {}
        if glyphNames is None:
            glyphNames = self.contents.keys()
        for glyphName in glyphNames:
            text = self.getGLIF(glyphName)
            unicodes[glyphName] = _fetchUnicodes(text)
        return unicodes

    def getComponentReferences(
        self, glyphNames: Optional[Iterable[str]] = None
    ) -> dict[str, list[str]]:
        """
        Return a dictionary that maps glyph names to lists containing the
        base glyph name of components in the glyph. This parses the .glif
        files partially, so it is a lot faster than parsing all files completely.
        By default this checks all glyphs, but a subset can be passed with glyphNames.
        """
        components = {}
        if glyphNames is None:
            glyphNames = self.contents.keys()
        for glyphName in glyphNames:
            text = self.getGLIF(glyphName)
            components[glyphName] = _fetchComponentBases(text)
        return components

    def getImageReferences(
        self, glyphNames: Optional[Iterable[str]] = None
    ) -> dict[str, Optional[str]]:
        """
        Return a dictionary that maps glyph names to the file name of the image
        referenced by the glyph. This parses the .glif files partially, so it is a
        lot faster than parsing all files completely.
        By default this checks all glyphs, but a subset can be passed with glyphNames.
        """
        images = {}
        if glyphNames is None:
            glyphNames = self.contents.keys()
        for glyphName in glyphNames:
            text = self.getGLIF(glyphName)
            images[glyphName] = _fetchImageFileName(text)
        return images

    def close(self) -> None:
        if self._shouldClose:
            self.fs.close()

    def __enter__(self) -> GlyphSet:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        self.close()


# -----------------------
# Glyph Name to File Name
# -----------------------


def glyphNameToFileName(glyphName: str, existingFileNames: Optional[set[str]]) -> str:
    """
    Wrapper around the userNameToFileName function in filenames.py

    Note that existingFileNames should be a set for large glyphsets
    or performance will suffer.
    """
    if existingFileNames is None:
        existingFileNames = set()
    return userNameToFileName(glyphName, existing=existingFileNames, suffix=".glif")


# -----------------------
# GLIF To and From String
# -----------------------


def readGlyphFromString(
    aString: Union[str, bytes],
    glyphObject: Optional[Any] = None,
    pointPen: Optional[Any] = None,
    formatVersions: FormatVersions = None,
    validate: bool = True,
) -> None:
    """
    Read .glif data from a string into a glyph object.

    The 'glyphObject' argument can be any kind of object (even None);
    the readGlyphFromString() method will attempt to set the following
    attributes on it:

    width
            the advance width of the glyph
    height
            the advance height of the glyph
    unicodes
            a list of unicode values for this glyph
    note
            a string
    lib
            a dictionary containing custom data
    image
            a dictionary containing image data
    guidelines
            a list of guideline data dictionaries
    anchors
            a list of anchor data dictionaries

    All attributes are optional, in two ways:

    1) An attribute *won't* be set if the .glif file doesn't
       contain data for it. 'glyphObject' will have to deal
       with default values itself.
    2) If setting the attribute fails with an AttributeError
       (for example if the 'glyphObject' attribute is read-
       only), readGlyphFromString() will not propagate that
       exception, but ignore that attribute.

    To retrieve outline information, you need to pass an object
    conforming to the PointPen protocol as the 'pointPen' argument.
    This argument may be None if you don't need the outline data.

    The formatVersions optional argument define the GLIF format versions
    that are allowed to be read.
    The type is Optional[Iterable[tuple[int, int], int]]. It can contain
    either integers (for the major versions to be allowed, with minor
    digits defaulting to 0), or tuples of integers to specify both
    (major, minor) versions.
    By default when formatVersions is None all the GLIF format versions
    currently defined are allowed to be read.

    ``validate`` will validate the read data. It is set to ``True`` by default.
    """
    tree = _glifTreeFromString(aString)

    if formatVersions is None:
        validFormatVersions: Set[GLIFFormatVersion] = (
            GLIFFormatVersion.supported_versions()
        )
    else:
        validFormatVersions, invalidFormatVersions = set(), set()
        for v in formatVersions:
            try:
                formatVersion = normalizeFormatVersion(v, GLIFFormatVersion)
            except ValueError:
                invalidFormatVersions.add(v)
            else:
                validFormatVersions.add(formatVersion)
        if not validFormatVersions:
            raise ValueError(
                "None of the requested GLIF formatVersions are supported: "
                f"{formatVersions!r}"
            )

    _readGlyphFromTree(
        tree,
        glyphObject,
        pointPen,
        formatVersions=validFormatVersions,
        validate=validate,
    )


def _writeGlyphToBytes(
    glyphName: str,
    glyphObject: Optional[Any] = None,
    drawPointsFunc: Optional[Callable[[Any], None]] = None,
    writer: Optional[Any] = None,
    formatVersion: Optional[FormatVersion] = None,
    validate: bool = True,
) -> bytes:
    """Return .glif data for a glyph as a UTF-8 encoded bytes string."""
    try:
        formatVersion = normalizeFormatVersion(formatVersion, GLIFFormatVersion)
    except ValueError:
        from fontTools.ufoLib.errors import UnsupportedGLIFFormat

        raise UnsupportedGLIFFormat(
            "Unsupported GLIF format version: {formatVersion!r}"
        )
    # start
    if validate and not isinstance(glyphName, str):
        raise GlifLibError("The glyph name is not properly formatted.")
    if validate and len(glyphName) == 0:
        raise GlifLibError("The glyph name is empty.")
    glyphAttrs = OrderedDict(
        [("name", glyphName), ("format", repr(formatVersion.major))]
    )
    if formatVersion.minor != 0:
        glyphAttrs["formatMinor"] = repr(formatVersion.minor)
    root = etree.Element("glyph", glyphAttrs)
    identifiers: set[str] = set()
    # advance
    _writeAdvance(glyphObject, root, validate)
    # unicodes
    if getattr(glyphObject, "unicodes", None):
        _writeUnicodes(glyphObject, root, validate)
    # note
    if getattr(glyphObject, "note", None):
        _writeNote(glyphObject, root, validate)
    # image
    if formatVersion.major >= 2 and getattr(glyphObject, "image", None):
        _writeImage(glyphObject, root, validate)
    # guidelines
    if formatVersion.major >= 2 and getattr(glyphObject, "guidelines", None):
        _writeGuidelines(glyphObject, root, identifiers, validate)
    # anchors
    anchors = getattr(glyphObject, "anchors", None)
    if formatVersion.major >= 2 and anchors:
        _writeAnchors(glyphObject, root, identifiers, validate)
    # outline
    if drawPointsFunc is not None:
        outline = etree.SubElement(root, "outline")
        pen = GLIFPointPen(outline, identifiers=identifiers, validate=validate)
        drawPointsFunc(pen)
        if formatVersion.major == 1 and anchors:
            _writeAnchorsFormat1(pen, anchors, validate)
        # prevent lxml from writing self-closing tags
        if not len(outline):
            outline.text = "\n  "
    # lib
    if getattr(glyphObject, "lib", None):
        _writeLib(glyphObject, root, validate)
    # return the text
    data = etree.tostring(
        root, encoding="UTF-8", xml_declaration=True, pretty_print=True
    )
    return data


def writeGlyphToString(
    glyphName: str,
    glyphObject: Optional[Any] = None,
    drawPointsFunc: Optional[Callable[[Any], None]] = None,
    formatVersion: Optional[FormatVersion] = None,
    validate: bool = True,
) -> str:
    """
    Return .glif data for a glyph as a string. The XML declaration's
    encoding is always set to "UTF-8".
    The 'glyphObject' argument can be any kind of object (even None);
    the writeGlyphToString() method will attempt to get the following
    attributes from it:

    width
            the advance width of the glyph
    height
            the advance height of the glyph
    unicodes
            a list of unicode values for this glyph
    note
            a string
    lib
            a dictionary containing custom data
    image
            a dictionary containing image data
    guidelines
            a list of guideline data dictionaries
    anchors
            a list of anchor data dictionaries

    All attributes are optional: if 'glyphObject' doesn't
    have the attribute, it will simply be skipped.

    To write outline data to the .glif file, writeGlyphToString() needs
    a function (any callable object actually) that will take one
    argument: an object that conforms to the PointPen protocol.
    The function will be called by writeGlyphToString(); it has to call the
    proper PointPen methods to transfer the outline to the .glif file.

    The GLIF format version can be specified with the formatVersion argument.
    This accepts either a tuple of integers for (major, minor), or a single
    integer for the major digit only (with minor digit implied as 0).
    By default when formatVesion is None the latest GLIF format version will
    be used; currently it's 2.0, which is equivalent to formatVersion=(2, 0).

    An UnsupportedGLIFFormat exception is raised if the requested UFO
    formatVersion is not supported.

    ``validate`` will validate the written data. It is set to ``True`` by default.
    """
    data = _writeGlyphToBytes(
        glyphName,
        glyphObject=glyphObject,
        drawPointsFunc=drawPointsFunc,
        formatVersion=formatVersion,
        validate=validate,
    )
    return data.decode("utf-8")


def _writeAdvance(glyphObject: Any, element: ElementType, validate: bool) -> None:
    width = getattr(glyphObject, "width", None)
    if width is not None:
        if validate and not isinstance(width, numberTypes):
            raise GlifLibError("width attribute must be int or float")
        if width == 0:
            width = None
    height = getattr(glyphObject, "height", None)
    if height is not None:
        if validate and not isinstance(height, numberTypes):
            raise GlifLibError("height attribute must be int or float")
        if height == 0:
            height = None
    if width is not None and height is not None:
        etree.SubElement(
            element,
            "advance",
            OrderedDict([("height", repr(height)), ("width", repr(width))]),
        )
    elif width is not None:
        etree.SubElement(element, "advance", dict(width=repr(width)))
    elif height is not None:
        etree.SubElement(element, "advance", dict(height=repr(height)))


def _writeUnicodes(glyphObject: Any, element: ElementType, validate: bool) -> None:
    unicodes = getattr(glyphObject, "unicodes", [])
    if validate and isinstance(unicodes, int):
        unicodes = [unicodes]
    seen = set()
    for code in unicodes:
        if validate and not isinstance(code, int):
            raise GlifLibError("unicode values must be int")
        if code in seen:
            continue
        seen.add(code)
        hexCode = "%04X" % code
        etree.SubElement(element, "unicode", dict(hex=hexCode))


def _writeNote(glyphObject: Any, element: ElementType, validate: bool) -> None:
    note = getattr(glyphObject, "note", None)
    if validate and not isinstance(note, str):
        raise GlifLibError("note attribute must be str")
    if isinstance(note, str):
        note = note.strip()
        note = "\n" + note + "\n"
        etree.SubElement(element, "note").text = note


def _writeImage(glyphObject: Any, element: ElementType, validate: bool) -> None:
    image = getattr(glyphObject, "image", None)
    if image is None:
        return

    if validate and not imageValidator(image):
        raise GlifLibError(
            "image attribute must be a dict or dict-like object with the proper structure."
        )
    attrs = OrderedDict([("fileName", image["fileName"])])
    for attr, default in _transformationInfo:
        value = image.get(attr, default)
        if value != default:
            attrs[attr] = repr(value)
    color = image.get("color")
    if color is not None:
        attrs["color"] = color
    etree.SubElement(element, "image", attrs)


def _writeGuidelines(
    glyphObject: Any, element: ElementType, identifiers: set[str], validate: bool
) -> None:
    guidelines = getattr(glyphObject, "guidelines", [])
    if validate and not guidelinesValidator(guidelines):
        raise GlifLibError("guidelines attribute does not have the proper structure.")
    for guideline in guidelines:
        attrs = OrderedDict()
        x = guideline.get("x")
        if x is not None:
            attrs["x"] = repr(x)
        y = guideline.get("y")
        if y is not None:
            attrs["y"] = repr(y)
        angle = guideline.get("angle")
        if angle is not None:
            attrs["angle"] = repr(angle)
        name = guideline.get("name")
        if name is not None:
            attrs["name"] = name
        color = guideline.get("color")
        if color is not None:
            attrs["color"] = color
        identifier = guideline.get("identifier")
        if identifier is not None:
            if validate and identifier in identifiers:
                raise GlifLibError("identifier used more than once: %s" % identifier)
            attrs["identifier"] = identifier
            identifiers.add(identifier)
        etree.SubElement(element, "guideline", attrs)


def _writeAnchorsFormat1(pen: Any, anchors: Any, validate: bool) -> None:
    if validate and not anchorsValidator(anchors):
        raise GlifLibError("anchors attribute does not have the proper structure.")
    for anchor in anchors:
        attrs = {}
        x = anchor["x"]
        attrs["x"] = repr(x)
        y = anchor["y"]
        attrs["y"] = repr(y)
        name = anchor.get("name")
        if name is not None:
            attrs["name"] = name
        pen.beginPath()
        pen.addPoint((x, y), segmentType="move", name=name)
        pen.endPath()


def _writeAnchors(
    glyphObject: Any,
    element: ElementType,
    identifiers: set[str],
    validate: bool,
) -> None:
    anchors = getattr(glyphObject, "anchors", [])
    if validate and not anchorsValidator(anchors):
        raise GlifLibError("anchors attribute does not have the proper structure.")
    for anchor in anchors:
        attrs = OrderedDict()
        x = anchor["x"]
        attrs["x"] = repr(x)
        y = anchor["y"]
        attrs["y"] = repr(y)
        name = anchor.get("name")
        if name is not None:
            attrs["name"] = name
        color = anchor.get("color")
        if color is not None:
            attrs["color"] = color
        identifier = anchor.get("identifier")
        if identifier is not None:
            if validate and identifier in identifiers:
                raise GlifLibError("identifier used more than once: %s" % identifier)
            attrs["identifier"] = identifier
            identifiers.add(identifier)
        etree.SubElement(element, "anchor", attrs)


def _writeLib(glyphObject: Any, element: ElementType, validate: bool) -> None:
    lib = getattr(glyphObject, "lib", None)
    if not lib:
        # don't write empty lib
        return
    if validate:
        valid, message = glyphLibValidator(lib)
        if not valid:
            raise GlifLibError(message)
    if not isinstance(lib, dict):
        lib = dict(lib)
    # plist inside GLIF begins with 2 levels of indentation
    e = plistlib.totree(lib, indent_level=2)
    etree.SubElement(element, "lib").append(e)


# -----------------------
# layerinfo.plist Support
# -----------------------

layerInfoVersion3ValueData = {
    "color": dict(type=str, valueValidator=colorValidator),
    "lib": dict(type=dict, valueValidator=genericTypeValidator),
}


def validateLayerInfoVersion3ValueForAttribute(attr: str, value: Any) -> bool:
    """
    This performs very basic validation of the value for attribute
    following the UFO 3 fontinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the value
    is of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    if attr not in layerInfoVersion3ValueData:
        return False
    dataValidationDict = layerInfoVersion3ValueData[attr]
    valueType = dataValidationDict.get("type")
    validator = dataValidationDict.get("valueValidator")
    valueOptions = dataValidationDict.get("valueOptions")
    # have specific options for the validator
    assert callable(validator)
    if valueOptions is not None:
        isValidValue = validator(value, valueOptions)
    # no specific options
    else:
        if validator == genericTypeValidator:
            isValidValue = validator(value, valueType)
        else:
            isValidValue = validator(value)
    return isValidValue


def validateLayerInfoVersion3Data(infoData: dict[str, Any]) -> dict[str, Any]:
    """
    This performs very basic validation of the value for infoData
    following the UFO 3 layerinfo.plist specification. The results
    of this should not be interpretted as *correct* for the font
    that they are part of. This merely indicates that the values
    are of the proper type and, where the specification defines
    a set range of possible values for an attribute, that the
    value is in the accepted range.
    """
    for attr, value in infoData.items():
        if attr not in layerInfoVersion3ValueData:
            raise GlifLibError("Unknown attribute %s." % attr)
        isValidValue = validateLayerInfoVersion3ValueForAttribute(attr, value)
        if not isValidValue:
            raise GlifLibError(f"Invalid value for attribute {attr} ({value!r}).")
    return infoData


# -----------------
# GLIF Tree Support
# -----------------


def _glifTreeFromFile(aFile: Union[str, bytes, int]) -> ElementType:
    if etree._have_lxml:
        tree = etree.parse(aFile, parser=etree.XMLParser(remove_comments=True))
    else:
        tree = etree.parse(aFile)
    root = tree.getroot()
    if root.tag != "glyph":
        raise GlifLibError("The GLIF is not properly formatted.")
    if root.text and root.text.strip() != "":
        raise GlifLibError("Invalid GLIF structure.")
    return root


def _glifTreeFromString(aString: Union[str, bytes]) -> ElementType:
    data = tobytes(aString, encoding="utf-8")
    try:
        if etree._have_lxml:
            root = etree.fromstring(data, parser=etree.XMLParser(remove_comments=True))
        else:
            root = etree.fromstring(data)
    except Exception as etree_exception:
        raise GlifLibError("GLIF contains invalid XML.") from etree_exception

    if root.tag != "glyph":
        raise GlifLibError("The GLIF is not properly formatted.")
    if root.text and root.text.strip() != "":
        raise GlifLibError("Invalid GLIF structure.")
    return root


def _readGlyphFromTree(
    tree: ElementType,
    glyphObject: Optional[Any] = None,
    pointPen: Optional[AbstractPointPen] = None,
    formatVersions: Set[GLIFFormatVersion] = GLIFFormatVersion.supported_versions(),
    validate: bool = True,
) -> None:
    # check the format version
    formatVersionMajor = tree.get("format")
    if formatVersionMajor is None:
        if validate:
            raise GlifLibError("Unspecified format version in GLIF.")
        formatVersionMajor = 0
    formatVersionMinor = tree.get("formatMinor", 0)
    try:
        formatVersion = GLIFFormatVersion(
            (int(formatVersionMajor), int(formatVersionMinor))
        )
    except ValueError as e:
        msg = "Unsupported GLIF format: %s.%s" % (
            formatVersionMajor,
            formatVersionMinor,
        )
        if validate:
            from fontTools.ufoLib.errors import UnsupportedGLIFFormat

            raise UnsupportedGLIFFormat(msg) from e
        # warn but continue using the latest supported format
        formatVersion = GLIFFormatVersion.default()
        logger.warning(
            "%s. Assuming the latest supported version (%s). "
            "Some data may be skipped or parsed incorrectly.",
            msg,
            formatVersion,
        )

    if validate and formatVersion not in formatVersions:
        raise GlifLibError(f"Forbidden GLIF format version: {formatVersion!s}")

    try:
        readGlyphFromTree = _READ_GLYPH_FROM_TREE_FUNCS[formatVersion]
    except KeyError:
        raise NotImplementedError(formatVersion)

    readGlyphFromTree(
        tree=tree,
        glyphObject=glyphObject,
        pointPen=pointPen,
        validate=validate,
        formatMinor=formatVersion.minor,
    )


def _readGlyphFromTreeFormat1(
    tree: ElementType,
    glyphObject: Optional[Any] = None,
    pointPen: Optional[AbstractPointPen] = None,
    validate: bool = False,
    **kwargs: Any,
) -> None:
    # get the name
    _readName(glyphObject, tree, validate)
    # populate the sub elements
    unicodes = []
    haveSeenAdvance = haveSeenOutline = haveSeenLib = haveSeenNote = False
    for element in tree:
        if element.tag == "outline":
            if validate:
                if haveSeenOutline:
                    raise GlifLibError("The outline element occurs more than once.")
                if element.attrib:
                    raise GlifLibError(
                        "The outline element contains unknown attributes."
                    )
                if element.text and element.text.strip() != "":
                    raise GlifLibError("Invalid outline structure.")
            haveSeenOutline = True
            buildOutlineFormat1(glyphObject, pointPen, element, validate)
        elif glyphObject is None:
            # Skip remaining elements if no glyphObject, but outline is still
            # processed above to allow drawing via pointPen without a glyphObject.
            continue
        elif element.tag == "advance":
            if validate and haveSeenAdvance:
                raise GlifLibError("The advance element occurs more than once.")
            haveSeenAdvance = True
            _readAdvance(glyphObject, element)
        elif element.tag == "unicode":
            v = element.get("hex")
            if v is None:
                raise GlifLibError(
                    "A unicode element is missing its required hex attribute."
                )
            try:
                v = int(v, 16)
                if v not in unicodes:
                    unicodes.append(v)
            except ValueError:
                raise GlifLibError(
                    "Illegal value for hex attribute of unicode element."
                )
        elif element.tag == "note":
            if validate and haveSeenNote:
                raise GlifLibError("The note element occurs more than once.")
            haveSeenNote = True
            _readNote(glyphObject, element)
        elif element.tag == "lib":
            if validate and haveSeenLib:
                raise GlifLibError("The lib element occurs more than once.")
            haveSeenLib = True
            _readLib(glyphObject, element, validate)
        else:
            raise GlifLibError("Unknown element in GLIF: %s" % element)
    # set the collected unicodes
    if unicodes:
        _relaxedSetattr(glyphObject, "unicodes", unicodes)


def _readGlyphFromTreeFormat2(
    tree: ElementType,
    glyphObject: Optional[Any] = None,
    pointPen: Optional[AbstractPointPen] = None,
    validate: bool = False,
    formatMinor: int = 0,
) -> None:
    # get the name
    _readName(glyphObject, tree, validate)
    # populate the sub elements
    unicodes = []
    guidelines = []
    anchors = []
    haveSeenAdvance = haveSeenImage = haveSeenOutline = haveSeenLib = haveSeenNote = (
        False
    )
    identifiers: set[str] = set()
    for element in tree:
        if element.tag == "outline":
            if validate:
                if haveSeenOutline:
                    raise GlifLibError("The outline element occurs more than once.")
                if element.attrib:
                    raise GlifLibError(
                        "The outline element contains unknown attributes."
                    )
                if element.text and element.text.strip() != "":
                    raise GlifLibError("Invalid outline structure.")
            haveSeenOutline = True
            if pointPen is not None:
                buildOutlineFormat2(
                    glyphObject, pointPen, element, identifiers, validate
                )
        elif glyphObject is None:
            # Skip remaining elements if no glyphObject, but outline is still
            # processed above to allow drawing via pointPen without a glyphObject.
            continue
        elif element.tag == "advance":
            if validate and haveSeenAdvance:
                raise GlifLibError("The advance element occurs more than once.")
            haveSeenAdvance = True
            _readAdvance(glyphObject, element)
        elif element.tag == "unicode":
            v = element.get("hex")
            if v is None:
                raise GlifLibError(
                    "A unicode element is missing its required hex attribute."
                )
            try:
                v = int(v, 16)
                if v not in unicodes:
                    unicodes.append(v)
            except ValueError:
                raise GlifLibError(
                    "Illegal value for hex attribute of unicode element."
                )
        elif element.tag == "guideline":
            if validate and len(element):
                raise GlifLibError("Unknown children in guideline element.")
            attrib = dict(element.attrib)
            for attr in ("x", "y", "angle"):
                if attr in attrib:
                    attrib[attr] = _number(attrib[attr])
            guidelines.append(attrib)
        elif element.tag == "anchor":
            if validate and len(element):
                raise GlifLibError("Unknown children in anchor element.")
            attrib = dict(element.attrib)
            for attr in ("x", "y"):
                if attr in element.attrib:
                    attrib[attr] = _number(attrib[attr])
            anchors.append(attrib)
        elif element.tag == "image":
            if validate:
                if haveSeenImage:
                    raise GlifLibError("The image element occurs more than once.")
                if len(element):
                    raise GlifLibError("Unknown children in image element.")
            haveSeenImage = True
            _readImage(glyphObject, element, validate)
        elif element.tag == "note":
            if validate and haveSeenNote:
                raise GlifLibError("The note element occurs more than once.")
            haveSeenNote = True
            _readNote(glyphObject, element)
        elif element.tag == "lib":
            if validate and haveSeenLib:
                raise GlifLibError("The lib element occurs more than once.")
            haveSeenLib = True
            _readLib(glyphObject, element, validate)
        else:
            raise GlifLibError("Unknown element in GLIF: %s" % element)
    # set the collected unicodes
    if unicodes:
        _relaxedSetattr(glyphObject, "unicodes", unicodes)
    # set the collected guidelines
    if guidelines:
        if validate and not guidelinesValidator(guidelines, identifiers):
            raise GlifLibError("The guidelines are improperly formatted.")
        _relaxedSetattr(glyphObject, "guidelines", guidelines)
    # set the collected anchors
    if anchors:
        if validate and not anchorsValidator(anchors, identifiers):
            raise GlifLibError("The anchors are improperly formatted.")
        _relaxedSetattr(glyphObject, "anchors", anchors)


_READ_GLYPH_FROM_TREE_FUNCS: dict[GLIFFormatVersion, Callable[..., Any]] = {
    GLIFFormatVersion.FORMAT_1_0: _readGlyphFromTreeFormat1,
    GLIFFormatVersion.FORMAT_2_0: _readGlyphFromTreeFormat2,
}


def _readName(glyphObject: Optional[Any], root: ElementType, validate: bool) -> None:
    glyphName = root.get("name")
    if validate and not glyphName:
        raise GlifLibError("Empty glyph name in GLIF.")
    if glyphName and glyphObject is not None:
        _relaxedSetattr(glyphObject, "name", glyphName)


def _readAdvance(glyphObject: Optional[Any], advance: ElementType) -> None:
    width = _number(advance.get("width", 0))
    _relaxedSetattr(glyphObject, "width", width)
    height = _number(advance.get("height", 0))
    _relaxedSetattr(glyphObject, "height", height)


def _readNote(glyphObject: Optional[Any], note: ElementType) -> None:
    if note.text is None:
        return
    lines = note.text.split("\n")
    note = "\n".join(line.strip() for line in lines if line.strip())
    _relaxedSetattr(glyphObject, "note", note)


def _readLib(glyphObject: Optional[Any], lib: ElementType, validate: bool) -> None:
    assert len(lib) == 1
    child = lib[0]
    plist = plistlib.fromtree(child)
    if validate:
        valid, message = glyphLibValidator(plist)
        if not valid:
            raise GlifLibError(message)
    _relaxedSetattr(glyphObject, "lib", plist)


def _readImage(glyphObject: Optional[Any], image: ElementType, validate: bool) -> None:
    imageData = dict(image.attrib)
    for attr, default in _transformationInfo:
        value = imageData.get(attr, default)
        imageData[attr] = _number(value)
    if validate and not imageValidator(imageData):
        raise GlifLibError("The image element is not properly formatted.")
    _relaxedSetattr(glyphObject, "image", imageData)


# ----------------
# GLIF to PointPen
# ----------------

contourAttributesFormat2: set[str] = {"identifier"}
componentAttributesFormat1: set[str] = {
    "base",
    "xScale",
    "xyScale",
    "yxScale",
    "yScale",
    "xOffset",
    "yOffset",
}
componentAttributesFormat2: set[str] = componentAttributesFormat1 | {"identifier"}
pointAttributesFormat1: set[str] = {"x", "y", "type", "smooth", "name"}
pointAttributesFormat2: set[str] = pointAttributesFormat1 | {"identifier"}
pointSmoothOptions: set[str] = {"no", "yes"}
pointTypeOptions: set[str] = {"move", "line", "offcurve", "curve", "qcurve"}

# format 1


def buildOutlineFormat1(
    glyphObject: Any,
    pen: Optional[AbstractPointPen],
    outline: Iterable[ElementType],
    validate: bool,
) -> None:
    anchors = []
    for element in outline:
        if element.tag == "contour":
            if len(element) == 1:
                point = element[0]
                if point.tag == "point":
                    anchor = _buildAnchorFormat1(point, validate)
                    if anchor is not None:
                        anchors.append(anchor)
                        continue
            if pen is not None:
                _buildOutlineContourFormat1(pen, element, validate)
        elif element.tag == "component":
            if pen is not None:
                _buildOutlineComponentFormat1(pen, element, validate)
        else:
            raise GlifLibError("Unknown element in outline element: %s" % element)
    if glyphObject is not None and anchors:
        if validate and not anchorsValidator(anchors):
            raise GlifLibError("GLIF 1 anchors are not properly formatted.")
        _relaxedSetattr(glyphObject, "anchors", anchors)


def _buildAnchorFormat1(point: ElementType, validate: bool) -> Optional[dict[str, Any]]:
    if point.get("type") != "move":
        return None
    name = point.get("name")
    if name is None:
        return None
    x = point.get("x")
    y = point.get("y")
    if validate and x is None:
        raise GlifLibError("Required x attribute is missing in point element.")
    assert x is not None
    if validate and y is None:
        raise GlifLibError("Required y attribute is missing in point element.")
    assert y is not None
    x = _number(x)
    y = _number(y)
    anchor = dict(x=x, y=y, name=name)
    return anchor


def _buildOutlineContourFormat1(
    pen: AbstractPointPen, contour: ElementType, validate: bool
) -> None:
    if validate and contour.attrib:
        raise GlifLibError("Unknown attributes in contour element.")
    pen.beginPath()
    if len(contour):
        massaged = _validateAndMassagePointStructures(
            contour,
            pointAttributesFormat1,
            openContourOffCurveLeniency=True,
            validate=validate,
        )
        _buildOutlinePointsFormat1(pen, massaged)
    pen.endPath()


def _buildOutlinePointsFormat1(
    pen: AbstractPointPen, contour: list[dict[str, Any]]
) -> None:
    for point in contour:
        x = point["x"]
        y = point["y"]
        segmentType = point["segmentType"]
        smooth = point["smooth"]
        name = point["name"]
        pen.addPoint((x, y), segmentType=segmentType, smooth=smooth, name=name)


def _buildOutlineComponentFormat1(
    pen: AbstractPointPen, component: ElementType, validate: bool
) -> None:
    if validate:
        if len(component):
            raise GlifLibError("Unknown child elements of component element.")
        for attr in component.attrib.keys():
            if attr not in componentAttributesFormat1:
                raise GlifLibError("Unknown attribute in component element: %s" % attr)
    baseGlyphName = component.get("base")
    if validate and baseGlyphName is None:
        raise GlifLibError("The base attribute is not defined in the component.")
    assert baseGlyphName is not None
    transformation = tuple(
        _number(component.get(attr) or default) for attr, default in _transformationInfo
    )
    transformation = cast(
        tuple[float, float, float, float, float, float], transformation
    )
    pen.addComponent(baseGlyphName, transformation)


# format 2


def buildOutlineFormat2(
    glyphObject: Any,
    pen: AbstractPointPen,
    outline: Iterable[ElementType],
    identifiers: set[str],
    validate: bool,
) -> None:
    for element in outline:
        if element.tag == "contour":
            _buildOutlineContourFormat2(pen, element, identifiers, validate)
        elif element.tag == "component":
            _buildOutlineComponentFormat2(pen, element, identifiers, validate)
        else:
            raise GlifLibError("Unknown element in outline element: %s" % element.tag)


def _buildOutlineContourFormat2(
    pen: AbstractPointPen, contour: ElementType, identifiers: set[str], validate: bool
) -> None:
    if validate:
        for attr in contour.attrib.keys():
            if attr not in contourAttributesFormat2:
                raise GlifLibError("Unknown attribute in contour element: %s" % attr)
    identifier = contour.get("identifier")
    if identifier is not None:
        if validate:
            if identifier in identifiers:
                raise GlifLibError(
                    "The identifier %s is used more than once." % identifier
                )
            if not identifierValidator(identifier):
                raise GlifLibError(
                    "The contour identifier %s is not valid." % identifier
                )
        identifiers.add(identifier)
    try:
        pen.beginPath(identifier=identifier)
    except TypeError:
        pen.beginPath()
        warn(
            "The beginPath method needs an identifier kwarg. The contour's identifier value has been discarded.",
            DeprecationWarning,
        )
    if len(contour):
        massaged = _validateAndMassagePointStructures(
            contour, pointAttributesFormat2, validate=validate
        )
        _buildOutlinePointsFormat2(pen, massaged, identifiers, validate)
    pen.endPath()


def _buildOutlinePointsFormat2(
    pen: AbstractPointPen,
    contour: list[dict[str, Any]],
    identifiers: set[str],
    validate: bool,
) -> None:
    for point in contour:
        x = point["x"]
        y = point["y"]
        segmentType = point["segmentType"]
        smooth = point["smooth"]
        name = point["name"]
        identifier = point.get("identifier")
        if identifier is not None:
            if validate:
                if identifier in identifiers:
                    raise GlifLibError(
                        "The identifier %s is used more than once." % identifier
                    )
                if not identifierValidator(identifier):
                    raise GlifLibError("The identifier %s is not valid." % identifier)
            identifiers.add(identifier)
        try:
            pen.addPoint(
                (x, y),
                segmentType=segmentType,
                smooth=smooth,
                name=name,
                identifier=identifier,
            )
        except TypeError:
            pen.addPoint((x, y), segmentType=segmentType, smooth=smooth, name=name)
            warn(
                "The addPoint method needs an identifier kwarg. The point's identifier value has been discarded.",
                DeprecationWarning,
            )


def _buildOutlineComponentFormat2(
    pen: AbstractPointPen, component: ElementType, identifiers: set[str], validate: bool
) -> None:
    if validate:
        if len(component):
            raise GlifLibError("Unknown child elements of component element.")
        for attr in component.attrib.keys():
            if attr not in componentAttributesFormat2:
                raise GlifLibError("Unknown attribute in component element: %s" % attr)
    baseGlyphName = component.get("base")
    if validate and baseGlyphName is None:
        raise GlifLibError("The base attribute is not defined in the component.")
    assert baseGlyphName is not None
    transformation = tuple(
        _number(component.get(attr) or default) for attr, default in _transformationInfo
    )
    transformation = cast(
        tuple[float, float, float, float, float, float], transformation
    )
    identifier = component.get("identifier")
    if identifier is not None:
        if validate:
            if identifier in identifiers:
                raise GlifLibError(
                    "The identifier %s is used more than once." % identifier
                )
            if validate and not identifierValidator(identifier):
                raise GlifLibError("The identifier %s is not valid." % identifier)
        identifiers.add(identifier)
    try:
        pen.addComponent(baseGlyphName, transformation, identifier=identifier)
    except TypeError:
        pen.addComponent(baseGlyphName, transformation)
        warn(
            "The addComponent method needs an identifier kwarg. The component's identifier value has been discarded.",
            DeprecationWarning,
        )


# all formats


def _validateAndMassagePointStructures(
    contour, pointAttributes, openContourOffCurveLeniency=False, validate=True
):
    if not len(contour):
        return
    # store some data for later validation
    lastOnCurvePoint = None
    haveOffCurvePoint = False
    # validate and massage the individual point elements
    massaged = []
    for index, element in enumerate(contour):
        # not <point>
        if element.tag != "point":
            raise GlifLibError(
                "Unknown child element (%s) of contour element." % element.tag
            )
        point = dict(element.attrib)
        massaged.append(point)
        if validate:
            # unknown attributes
            for attr in point.keys():
                if attr not in pointAttributes:
                    raise GlifLibError("Unknown attribute in point element: %s" % attr)
            # search for unknown children
            if len(element):
                raise GlifLibError("Unknown child elements in point element.")
        # x and y are required
        for attr in ("x", "y"):
            try:
                point[attr] = _number(point[attr])
            except KeyError as e:
                raise GlifLibError(
                    f"Required {attr} attribute is missing in point element."
                ) from e
        # segment type
        pointType = point.pop("type", "offcurve")
        if validate and pointType not in pointTypeOptions:
            raise GlifLibError("Unknown point type: %s" % pointType)
        if pointType == "offcurve":
            pointType = None
        point["segmentType"] = pointType
        if pointType is None:
            haveOffCurvePoint = True
        else:
            lastOnCurvePoint = index
        # move can only occur as the first point
        if validate and pointType == "move" and index != 0:
            raise GlifLibError(
                "A move point occurs after the first point in the contour."
            )
        # smooth is optional
        smooth = point.get("smooth", "no")
        if validate and smooth is not None:
            if smooth not in pointSmoothOptions:
                raise GlifLibError("Unknown point smooth value: %s" % smooth)
        smooth = smooth == "yes"
        point["smooth"] = smooth
        # smooth can only be applied to curve and qcurve
        if validate and smooth and pointType is None:
            raise GlifLibError("smooth attribute set in an offcurve point.")
        # name is optional
        if "name" not in element.attrib:
            point["name"] = None
    if openContourOffCurveLeniency:
        # remove offcurves that precede a move. this is technically illegal,
        # but we let it slide because there are fonts out there in the wild like this.
        if massaged[0]["segmentType"] == "move":
            count = 0
            for point in reversed(massaged):
                if point["segmentType"] is None:
                    count += 1
                else:
                    break
            if count:
                massaged = massaged[:-count]
    # validate the off-curves in the segments
    if validate and haveOffCurvePoint and lastOnCurvePoint is not None:
        # we only care about how many offCurves there are before an onCurve
        # filter out the trailing offCurves
        offCurvesCount = len(massaged) - 1 - lastOnCurvePoint
        for point in massaged:
            segmentType = point["segmentType"]
            if segmentType is None:
                offCurvesCount += 1
            else:
                if offCurvesCount:
                    # move and line can't be preceded by off-curves
                    if segmentType == "move":
                        # this will have been filtered out already
                        raise GlifLibError("move can not have an offcurve.")
                    elif segmentType == "line":
                        raise GlifLibError("line can not have an offcurve.")
                    elif segmentType == "curve":
                        if offCurvesCount > 2:
                            raise GlifLibError("Too many offcurves defined for curve.")
                    elif segmentType == "qcurve":
                        pass
                    else:
                        # unknown segment type. it'll be caught later.
                        pass
                offCurvesCount = 0
    return massaged


# ---------------------
# Misc Helper Functions
# ---------------------


def _relaxedSetattr(object: Any, attr: str, value: Any) -> None:
    try:
        setattr(object, attr, value)
    except AttributeError:
        pass


def _number(s: Union[str, int, float]) -> IntFloat:
    """
    Given a numeric string, return an integer or a float, whichever
    the string indicates. _number("1") will return the integer 1,
    _number("1.0") will return the float 1.0.

    >>> _number("1")
    1
    >>> _number("1.0")
    1.0
    >>> _number("a")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    GlifLibError: Could not convert a to an int or float.
    """
    try:
        n: IntFloat = int(s)
        return n
    except ValueError:
        pass
    try:
        n = float(s)
        return n
    except ValueError:
        raise GlifLibError("Could not convert %s to an int or float." % s)


# --------------------
# Rapid Value Fetching
# --------------------

# base


class _DoneParsing(Exception):
    pass


class _BaseParser:
    def __init__(self) -> None:
        self._elementStack: list[str] = []

    def parse(self, text: bytes):
        from xml.parsers.expat import ParserCreate

        parser = ParserCreate()
        parser.StartElementHandler = self.startElementHandler
        parser.EndElementHandler = self.endElementHandler
        parser.Parse(text, True)

    def startElementHandler(self, name: str, attrs: Any) -> None:
        self._elementStack.append(name)

    def endElementHandler(self, name: str) -> None:
        other = self._elementStack.pop(-1)
        assert other == name


# unicodes


def _fetchUnicodes(glif: bytes) -> list[int]:
    """
    Get a list of unicodes listed in glif.
    """
    parser = _FetchUnicodesParser()
    parser.parse(glif)
    return parser.unicodes


class _FetchUnicodesParser(_BaseParser):
    def __init__(self) -> None:
        self.unicodes: list[int] = []
        super().__init__()

    def startElementHandler(self, name: str, attrs: dict[str, str]) -> None:
        if (
            name == "unicode"
            and self._elementStack
            and self._elementStack[-1] == "glyph"
        ):
            value = attrs.get("hex")
            if value is not None:
                try:
                    intValue = int(value, 16)
                    if intValue not in self.unicodes:
                        self.unicodes.append(intValue)
                except ValueError:
                    pass
        super().startElementHandler(name, attrs)


# image


def _fetchImageFileName(glif: bytes) -> Optional[str]:
    """
    The image file name (if any) from glif.
    """
    parser = _FetchImageFileNameParser()
    try:
        parser.parse(glif)
    except _DoneParsing:
        pass
    return parser.fileName


class _FetchImageFileNameParser(_BaseParser):
    def __init__(self) -> None:
        self.fileName: Optional[str] = None
        super().__init__()

    def startElementHandler(self, name: str, attrs: dict[str, str]) -> None:
        if name == "image" and self._elementStack and self._elementStack[-1] == "glyph":
            self.fileName = attrs.get("fileName")
            raise _DoneParsing
        super().startElementHandler(name, attrs)


# component references


def _fetchComponentBases(glif: bytes) -> list[str]:
    """
    Get a list of component base glyphs listed in glif.
    """
    parser = _FetchComponentBasesParser()
    try:
        parser.parse(glif)
    except _DoneParsing:
        pass
    return list(parser.bases)


class _FetchComponentBasesParser(_BaseParser):
    def __init__(self) -> None:
        self.bases: list[str] = []
        super().__init__()

    def startElementHandler(self, name: str, attrs: dict[str, str]) -> None:
        if (
            name == "component"
            and self._elementStack
            and self._elementStack[-1] == "outline"
        ):
            base = attrs.get("base")
            if base is not None:
                self.bases.append(base)
        super().startElementHandler(name, attrs)

    def endElementHandler(self, name: str) -> None:
        if name == "outline":
            raise _DoneParsing
        super().endElementHandler(name)


# --------------
# GLIF Point Pen
# --------------

_transformationInfo: list[tuple[str, int]] = [
    # field name, default value
    ("xScale", 1),
    ("xyScale", 0),
    ("yxScale", 0),
    ("yScale", 1),
    ("xOffset", 0),
    ("yOffset", 0),
]


class GLIFPointPen(AbstractPointPen):
    """
    Helper class using the PointPen protocol to write the <outline>
    part of .glif files.
    """

    def __init__(
        self,
        element: ElementType,
        formatVersion: Optional[FormatVersion] = None,
        identifiers: Optional[set[str]] = None,
        validate: bool = True,
    ) -> None:
        if identifiers is None:
            identifiers = set()
        self.formatVersion = normalizeFormatVersion(formatVersion, GLIFFormatVersion)
        self.identifiers = identifiers
        self.outline = element
        self.contour = None
        self.prevOffCurveCount = 0
        self.prevPointTypes: list[str] = []
        self.validate = validate

    def beginPath(self, identifier=None, **kwargs):
        attrs = OrderedDict()
        if identifier is not None and self.formatVersion.major >= 2:
            if self.validate:
                if identifier in self.identifiers:
                    raise GlifLibError(
                        "identifier used more than once: %s" % identifier
                    )
                if not identifierValidator(identifier):
                    raise GlifLibError(
                        "identifier not formatted properly: %s" % identifier
                    )
            attrs["identifier"] = identifier
            self.identifiers.add(identifier)
        self.contour = etree.SubElement(self.outline, "contour", attrs)
        self.prevOffCurveCount = 0

    def endPath(self):
        if self.prevPointTypes and self.prevPointTypes[0] == "move":
            if self.validate and self.prevPointTypes[-1] == "offcurve":
                raise GlifLibError("open contour has loose offcurve point")
        # prevent lxml from writing self-closing tags
        if not len(self.contour):
            self.contour.text = "\n  "
        self.contour = None
        self.prevPointType = None
        self.prevOffCurveCount = 0
        self.prevPointTypes = []

    def addPoint(
        self, pt, segmentType=None, smooth=None, name=None, identifier=None, **kwargs
    ):
        attrs = OrderedDict()
        # coordinates
        if pt is not None:
            if self.validate:
                for coord in pt:
                    if not isinstance(coord, numberTypes):
                        raise GlifLibError("coordinates must be int or float")
            attrs["x"] = repr(pt[0])
            attrs["y"] = repr(pt[1])
        # segment type
        if segmentType == "offcurve":
            segmentType = None
        if self.validate:
            if segmentType == "move" and self.prevPointTypes:
                raise GlifLibError(
                    "move occurs after a point has already been added to the contour."
                )
            if (
                segmentType in ("move", "line")
                and self.prevPointTypes
                and self.prevPointTypes[-1] == "offcurve"
            ):
                raise GlifLibError("offcurve occurs before %s point." % segmentType)
            if segmentType == "curve" and self.prevOffCurveCount > 2:
                raise GlifLibError("too many offcurve points before curve point.")
        if segmentType is not None:
            attrs["type"] = segmentType
        else:
            segmentType = "offcurve"
        if segmentType == "offcurve":
            self.prevOffCurveCount += 1
        else:
            self.prevOffCurveCount = 0
        self.prevPointTypes.append(segmentType)
        # smooth
        if smooth:
            if self.validate and segmentType == "offcurve":
                raise GlifLibError("can't set smooth in an offcurve point.")
            attrs["smooth"] = "yes"
        # name
        if name is not None:
            attrs["name"] = name
        # identifier
        if identifier is not None and self.formatVersion.major >= 2:
            if self.validate:
                if identifier in self.identifiers:
                    raise GlifLibError(
                        "identifier used more than once: %s" % identifier
                    )
                if not identifierValidator(identifier):
                    raise GlifLibError(
                        "identifier not formatted properly: %s" % identifier
                    )
            attrs["identifier"] = identifier
            self.identifiers.add(identifier)
        etree.SubElement(self.contour, "point", attrs)

    def addComponent(self, glyphName, transformation, identifier=None, **kwargs):
        attrs = OrderedDict([("base", glyphName)])
        for (attr, default), value in zip(_transformationInfo, transformation):
            if self.validate and not isinstance(value, numberTypes):
                raise GlifLibError("transformation values must be int or float")
            if value != default:
                attrs[attr] = repr(value)
        if identifier is not None and self.formatVersion.major >= 2:
            if self.validate:
                if identifier in self.identifiers:
                    raise GlifLibError(
                        "identifier used more than once: %s" % identifier
                    )
                if self.validate and not identifierValidator(identifier):
                    raise GlifLibError(
                        "identifier not formatted properly: %s" % identifier
                    )
            attrs["identifier"] = identifier
            self.identifiers.add(identifier)
        etree.SubElement(self.outline, "component", attrs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
<|endoftext|>from __future__ import annotations

from typing import Optional

from fontTools.annotations import KerningPair, KerningDict, KerningGroups, IntFloat

StrDict = dict[str, str]


def lookupKerningValue(
    pair: KerningPair,
    kerning: KerningDict,
    groups: KerningGroups,
    fallback: IntFloat = 0,
    glyphToFirstGroup: Optional[StrDict] = None,
    glyphToSecondGroup: Optional[StrDict] = None,
) -> IntFloat:
    """Retrieve the kerning value (if any) between a pair of elements.

    The elements can be either individual glyphs (by name) or kerning
    groups (by name), or any combination of the two.

    Providing the two glyph-to-group mappings accelerates group-lookups.
    These can be built manually, or with :py:func:`glyphsToGroups`.

    Args:
      pair:
          A tuple, in logical order (first, second) with respect
          to the reading direction, to query the font for kerning
          information on. Each element in the tuple can be either
          a glyph name or a kerning group name.
      kerning:
          A dictionary of kerning pairs.
      groups:
          A dictionary mapping each group name to its glyph names.
      fallback:
          The fallback value to return if no kern is found between
          the elements in ``pair``. Defaults to 0.
      glyphToFirstGroup:
          A dictionary mapping glyph names to the first-glyph kerning
          groups to which they belong. Defaults to ``None``.
      glyphToSecondGroup:
          A dictionary mapping glyph names to the second-glyph kerning
          groups to which they belong. Defaults to ``None``.

    Returns:
      The kerning value between the element pair. If no kerning for
      the pair is found, the fallback value is returned.

    Note: This function expects the ``kerning`` argument to be a flat
    dictionary of kerning pairs, not the nested structure used in a
    kerning.plist file.

    Examples::

      >>> groups = {
      ...     "public.kern1.O" : ["O", "D", "Q"],
      ...     "public.kern2.E" : ["E", "F"]
      ... }
      >>> kerning = {
      ...     ("public.kern1.O", "public.kern2.E") : -100,
      ...     ("public.kern1.O", "F") : -200,
      ...     ("D", "F") : -300
      ... }
      >>> lookupKerningValue(("D", "F"), kerning, groups)
      -300
      >>> lookupKerningValue(("O", "F"), kerning, groups)
      -200
      >>> lookupKerningValue(("O", "E"), kerning, groups)
      -100
      >>> lookupKerningValue(("O", "O"), kerning, groups)
      0
      >>> lookupKerningValue(("E", "E"), kerning, groups)
      0
      >>> lookupKerningValue(("E", "O"), kerning, groups)
      0
      >>> lookupKerningValue(("X", "X"), kerning, groups)
      0
      >>> lookupKerningValue(("public.kern1.O", "public.kern2.E"),
      ...     kerning, groups)
      -100
      >>> lookupKerningValue(("public.kern1.O", "F"), kerning, groups)
      -200
      >>> lookupKerningValue(("O", "public.kern2.E"), kerning, groups)
      -100
      >>> lookupKerningValue(("public.kern1.X", "public.kern2.X"), kerning, groups)
      0
      >>> glyphToFirstGroup, glyphToSecondGroup = glyphsToGroups(groups)
      >>> lookupKerningValue(("O", "E"), kerning, groups, glyphToFirstGroup=glyphToFirstGroup, glyphToSecondGroup=glyphToSecondGroup)
      -100
    """
    # quickly check to see if the pair is in the kerning dictionary
    if pair in kerning:
        return kerning[pair]
    # ensure both or no glyph-to-group mappings are provided
    if (glyphToFirstGroup is None) != (glyphToSecondGroup is None):
        raise ValueError(
            "Must provide both 'glyphToFirstGroup' and 'glyphToSecondGroup', or neither."
        )
    # create glyph to group mapping
    if glyphToFirstGroup is None or glyphToSecondGroup is None:
        glyphToFirstGroup, glyphToSecondGroup = glyphsToGroups(groups)
    # get group names and make sure first and second are glyph names
    first, second = pair
    firstGroup = secondGroup = None
    if first.startswith("public.kern1."):
        firstGroup = first
        firstGlyph = None
    else:
        firstGroup = glyphToFirstGroup.get(first)
        firstGlyph = first
    if second.startswith("public.kern2."):
        secondGroup = second
        secondGlyph = None
    else:
        secondGroup = glyphToSecondGroup.get(second)
        secondGlyph = second
    # make an ordered list of pairs to look up
    pairs = [
        (a, b)
        for a in (firstGlyph, firstGroup)
        for b in (secondGlyph, secondGroup)
        if a is not None and b is not None
    ]
    # look up the pairs and return any matches
    for pair in pairs:
        if pair in kerning:
            return kerning[pair]
    # use the fallback value
    return fallback


def glyphsToGroups(groups: KerningGroups) -> tuple[StrDict, StrDict]:
    """Build dictionaries mapping glyph names to their kerning groups.

    This is a reverse-mapping of the UFO kerning groups data structure.
    Groups are assumed to be compliant with the UFO specification; with
    the current implementation, if glyphs appear in more than one group
    anyway, the last group wins.

    Args:
      groups:
          A dictionary mapping each group name to its glyph names.

    Returns:
      A tuple of two dictionaries, respectively mapping glyph names to
      the first- and second-glyph kerning groups to which they belong.

    Examples::

      >>> groups = {
      ...     "public.kern1.O" : ["O", "D", "Q"],
      ...     "public.kern2.E" : ["E", "F"]
      ... }
      >>> glyphsToGroups(groups)
      ({'O': 'public.kern1.O', 'D': 'public.kern1.O', 'Q': 'public.kern1.O'}, {'E': 'public.kern2.E', 'F': 'public.kern2.E'})
    """
    glyphToFirstGroup = {}
    glyphToSecondGroup = {}
    for group, groupMembers in groups.items():
        if group.startswith("public.kern1."):
            for glyph in groupMembers:
                glyphToFirstGroup[glyph] = group
        elif group.startswith("public.kern2."):
            for glyph in groupMembers:
                glyphToSecondGroup[glyph] = group
    return (glyphToFirstGroup, glyphToSecondGroup)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
<|endoftext|>from __future__ import annotations

from collections.abc import Iterable

"""
Convert user-provided internal UFO names to spec-compliant filenames.

This module implements the algorithm for converting between a "user name" -
something that a user can choose arbitrarily inside a font editor - and a file
name suitable for use in a wide range of operating systems and filesystems.

The `UFO 3 specification <http://unifiedfontobject.org/versions/ufo3/conventions/>`_
provides an example of an algorithm for such conversion, which avoids illegal
characters, reserved file names, ambiguity between upper- and lower-case
characters, and clashes with existing files.

This code was originally copied from
`ufoLib <https://github.com/unified-font-object/ufoLib/blob/8747da7/Lib/ufoLib/filenames.py>`_
by Tal Leming and is copyright (c) 2005-2016, The RoboFab Developers:

-	Erik van Blokland
-	Tal Leming
-	Just van Rossum
"""

# Restrictions are taken mostly from
# https://docs.microsoft.com/en-gb/windows/win32/fileio/naming-a-file#naming-conventions.
#
# 1. Integer value zero, sometimes referred to as the ASCII NUL character.
# 2. Characters whose integer representations are in the range 1 to 31,
#    inclusive.
# 3. Various characters that (mostly) Windows and POSIX-y filesystems don't
#    allow, plus "(" and ")", as per the specification.
illegalCharacters: set[str] = {
    "\x00",
    "\x01",
    "\x02",
    "\x03",
    "\x04",
    "\x05",
    "\x06",
    "\x07",
    "\x08",
    "\t",
    "\n",
    "\x0b",
    "\x0c",
    "\r",
    "\x0e",
    "\x0f",
    "\x10",
    "\x11",
    "\x12",
    "\x13",
    "\x14",
    "\x15",
    "\x16",
    "\x17",
    "\x18",
    "\x19",
    "\x1a",
    "\x1b",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    '"',
    "*",
    "+",
    "/",
    ":",
    "<",
    ">",
    "?",
    "[",
    "\\",
    "]",
    "(",
    ")",
    "|",
    "\x7f",
}
reservedFileNames: set[str] = {
    "aux",
    "clock$",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "con",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
    "nul",
    "prn",
}
maxFileNameLength: int = 255


class NameTranslationError(Exception):
    pass


def userNameToFileName(
    userName: str, existing: Iterable[str] = (), prefix: str = "", suffix: str = ""
) -> str:
    """Converts from a user name to a file name.

    Takes care to avoid illegal characters, reserved file names, ambiguity between
    upper- and lower-case characters, and clashes with existing files.

    Args:
            userName (str): The input file name.
            existing: A case-insensitive list of all existing file names.
            prefix: Prefix to be prepended to the file name.
            suffix: Suffix to be appended to the file name.

    Returns:
            A suitable filename.

    Raises:
            NameTranslationError: If no suitable name could be generated.

    Examples::

            >>> userNameToFileName("a") == "a"
            True
            >>> userNameToFileName("A") == "A_"
            True
            >>> userNameToFileName("AE") == "A_E_"
            True
            >>> userNameToFileName("Ae") == "A_e"
            True
            >>> userNameToFileName("ae") == "ae"
            True
            >>> userNameToFileName("aE") == "aE_"
            True
            >>> userNameToFileName("a.alt") == "a.alt"
            True
            >>> userNameToFileName("A.alt") == "A_.alt"
            True
            >>> userNameToFileName("A.Alt") == "A_.A_lt"
            True
            >>> userNameToFileName("A.aLt") == "A_.aL_t"
            True
            >>> userNameToFileName(u"A.alT") == "A_.alT_"
            True
            >>> userNameToFileName("T_H") == "T__H_"
            True
            >>> userNameToFileName("T_h") == "T__h"
            True
            >>> userNameToFileName("t_h") == "t_h"
            True
            >>> userNameToFileName("F_F_I") == "F__F__I_"
            True
            >>> userNameToFileName("f_f_i") == "f_f_i"
            True
            >>> userNameToFileName("Aacute_V.swash") == "A_acute_V_.swash"
            True
            >>> userNameToFileName(".notdef") == "_notdef"
            True
            >>> userNameToFileName("con") == "_con"
            True
            >>> userNameToFileName("CON") == "C_O_N_"
            True
            >>> userNameToFileName("con.alt") == "_con.alt"
            True
            >>> userNameToFileName("alt.con") == "alt._con"
            True
    """
    # the incoming name must be a string
    if not isinstance(userName, str):
        raise ValueError("The value for userName must be a string.")
    # establish the prefix and suffix lengths
    prefixLength = len(prefix)
    suffixLength = len(suffix)
    # replace an initial period with an _
    # if no prefix is to be added
    if not prefix and userName[0] == ".":
        userName = "_" + userName[1:]
    # filter the user name
    filteredUserName = []
    for character in userName:
        # replace illegal characters with _
        if character in illegalCharacters:
            character = "_"
        # add _ to all non-lower characters
        elif character != character.lower():
            character += "_"
        filteredUserName.append(character)
    userName = "".join(filteredUserName)
    # clip to 255
    sliceLength = maxFileNameLength - prefixLength - suffixLength
    userName = userName[:sliceLength]
    # test for illegal files names
    parts = []
    for part in userName.split("."):
        if part.lower() in reservedFileNames:
            part = "_" + part
        parts.append(part)
    userName = ".".join(parts)
    # test for clash
    fullName = prefix + userName + suffix
    if fullName.lower() in existing:
        fullName = handleClash1(userName, existing, prefix, suffix)
    # finished
    return fullName


def handleClash1(
    userName: str, existing: Iterable[str] = [], prefix: str = "", suffix: str = ""
) -> str:
    """A helper function that resolves collisions with existing names when choosing a filename.

    This function attempts to append an unused integer counter to the filename.

        Args:
                userName (str): The input file name.
                existing: A case-insensitive list of all existing file names.
                prefix: Prefix to be prepended to the file name.
                suffix: Suffix to be appended to the file name.

        Returns:
                A suitable filename.

        >>> prefix = ("0" * 5) + "."
        >>> suffix = "." + ("0" * 10)
        >>> existing = ["a" * 5]

        >>> e = list(existing)
        >>> handleClash1(userName="A" * 5, existing=e,
        ...		prefix=prefix, suffix=suffix) == (
        ... 	'00000.AAAAA000000000000001.0000000000')
        True

        >>> e = list(existing)
        >>> e.append(prefix + "aaaaa" + "1".zfill(15) + suffix)
        >>> handleClash1(userName="A" * 5, existing=e,
        ...		prefix=prefix, suffix=suffix) == (
        ... 	'00000.AAAAA000000000000002.0000000000')
        True

        >>> e = list(existing)
        >>> e.append(prefix + "AAAAA" + "2".zfill(15) + suffix)
        >>> handleClash1(userName="A" * 5, existing=e,
        ...		prefix=prefix, suffix=suffix) == (
        ... 	'00000.AAAAA000000000000001.0000000000')
        True
    """
    # if the prefix length + user name length + suffix length + 15 is at
    # or past the maximum length, silce 15 characters off of the user name
    prefixLength = len(prefix)
    suffixLength = len(suffix)
    if prefixLength + len(userName) + suffixLength + 15 > maxFileNameLength:
        l = prefixLength + len(userName) + suffixLength + 15
        sliceLength = maxFileNameLength - l
        userName = userName[:sliceLength]
    finalName = None
    # try to add numbers to create a unique name
    counter = 1
    while finalName is None:
        name = userName + str(counter).zfill(15)
        fullName = prefix + name + suffix
        if fullName.lower() not in existing:
            finalName = fullName
            break
        else:
            counter += 1
        if counter >= 999999999999999:
            break
    # if there is a clash, go to the next fallback
    if finalName is None:
        finalName = handleClash2(existing, prefix, suffix)
    # finished
    return finalName


def handleClash2(
    existing: Iterable[str] = [], prefix: str = "", suffix: str = ""
) -> str:
    """A helper function that resolves collisions with existing names when choosing a filename.

    This function is a fallback to :func:`handleClash1`. It attempts to append an unused integer counter to the filename.

        Args:
                userName (str): The input file name.
                existing: A case-insensitive list of all existing file names.
                prefix: Prefix to be prepended to the file name.
                suffix: Suffix to be appended to the file name.

        Returns:
                A suitable filename.

        Raises:
                NameTranslationError: If no suitable name could be generated.

        Examples::

          >>> prefix = ("0" * 5) + "."
          >>> suffix = "." + ("0" * 10)
          >>> existing = [prefix + str(i) + suffix for i in range(100)]

          >>> e = list(existing)
          >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
          ... 	'00000.100.0000000000')
          True

          >>> e = list(existing)
          >>> e.remove(prefix + "1" + suffix)
          >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
          ... 	'00000.1.0000000000')
          True

          >>> e = list(existing)
          >>> e.remove(prefix + "2" + suffix)
          >>> handleClash2(existing=e, prefix=prefix, suffix=suffix) == (
          ... 	'00000.2.0000000000')
          True
    """
    # calculate the longest possible string
    maxLength = maxFileNameLength - len(prefix) - len(suffix)
    maxValue = int("9" * maxLength)
    # try to find a number
    finalName = None
    counter = 1
    while finalName is None:
        fullName = prefix + str(counter) + suffix
        if fullName.lower() not in existing:
            finalName = fullName
            break
        else:
            counter += 1
        if counter >= maxValue:
            break
    # raise an error if nothing has been found
    if finalName is None:
        raise NameTranslationError("No unique name could be found.")
    # finished
    return finalName


if __name__ == "__main__":
    import doctest

    doctest.testmod()
<|endoftext|>from __future__ import annotations


class UFOLibError(Exception):
    pass


class UnsupportedUFOFormat(UFOLibError):
    pass


class GlifLibError(UFOLibError):
    """An error raised by glifLib.

    This class is a loose backport of PEP 678, adding a :attr:`.note`
    attribute that can hold additional context for errors encountered.

    It will be maintained until only Python 3.11-and-later are supported.
    """

    def _add_note(self, note: str) -> None:
        # Loose backport of PEP 678 until we only support Python 3.11+, used for
        # adding additional context to errors.
        # TODO: Replace with https://docs.python.org/3.11/library/exceptions.html#BaseException.add_note
        message, *rest = self.args
        self.args = ((message + "\n" + note), *rest)


class UnsupportedGLIFFormat(GlifLibError):
    pass
<|endoftext|>"""This module contains miscellaneous helpers.

It is not considered part of the public ufoLib API. It does, however,
define the :py:obj:`.deprecated` decorator that is used elsewhere in
the module.
"""

from __future__ import annotations

from typing import Optional, Type, TypeVar, Union, cast
from collections.abc import Callable
import enum
import functools
import warnings

F = TypeVar("F", bound=Callable[..., object])
FormatVersion = TypeVar("FormatVersion", bound="BaseFormatVersion")
FormatVersionInput = Optional[Union[int, tuple[int, int], FormatVersion]]

numberTypes = (int, float)


def deprecated(msg: str = "") -> Callable[[F], F]:
    """Decorator factory to mark functions as deprecated with given message.

    >>> @deprecated("Enough!")
    ... def some_function():
    ...    "I just print 'hello world'."
    ...    print("hello world")
    >>> some_function()
    hello world
    >>> some_function.__doc__ == "I just print 'hello world'."
    True
    """

    def deprecated_decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} function is a deprecated. {msg}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return deprecated_decorator


def normalizeFormatVersion(
    value: FormatVersionInput, cls: Type[FormatVersion]
) -> FormatVersion:
    # Needed for type safety of UFOFormatVersion and GLIFFormatVersion input
    if value is None:
        return cls.default()
    if isinstance(value, cls):
        return value
    if isinstance(value, int):
        return cls((value, 0))
    if isinstance(value, tuple) and len(value) == 2:
        return cls(value)
    raise ValueError(f"Unsupported format version: {value!r}")


# Base class for UFOFormatVersion and GLIFFormatVersion
class BaseFormatVersion(tuple[int, int], enum.Enum):
    value: tuple[int, int]

    def __new__(cls: Type[FormatVersion], value: tuple[int, int]) -> BaseFormatVersion:
        return super().__new__(cls, value)

    @property
    def major(self) -> int:
        return self.value[0]

    @property
    def minor(self) -> int:
        return self.value[1]

    @classmethod
    def _missing_(cls, value: object) -> BaseFormatVersion:
        # allow to initialize a version enum from a single (major) integer
        if isinstance(value, int):
            return cls((value, 0))
        # or from None to obtain the current default version
        if value is None:
            return cls.default()
        raise ValueError(f"{value!r} is not a valid {cls.__name__}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    @classmethod
    def default(cls: Type[FormatVersion]) -> FormatVersion:
        # get the latest defined version (i.e. the max of all versions)
        return max(cls.__members__.values())

    @classmethod
    def supported_versions(cls: Type[FormatVersion]) -> frozenset[FormatVersion]:
        return frozenset(cls.__members__.values())


if __name__ == "__main__":
    import doctest

    doctest.testmod()
<|endoftext|>"""Various low level data validators."""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from io import open

import fontTools.misc.filesystem as fs
from typing import Any, Type, Optional, Union

from fontTools.annotations import IntFloat
from fontTools.ufoLib.utils import numberTypes

GenericDict = dict[str, tuple[Union[type, tuple[Type[Any], ...]], bool]]

# -------
# Generic
# -------


def isDictEnough(value: Any) -> bool:
    """
    Some objects will likely come in that aren't
    dicts but are dict-ish enough.
    """
    if isinstance(value, Mapping):
        return True
    for attr in ("keys", "values", "items"):
        if not hasattr(value, attr):
            return False
    return True


def genericTypeValidator(value: Any, typ: Type[Any]) -> bool:
    """
    Generic. (Added at version 2.)
    """
    return isinstance(value, typ)


def genericIntListValidator(values: Any, validValues: Sequence[int]) -> bool:
    """
    Generic. (Added at version 2.)
    """
    if not isinstance(values, (list, tuple)):
        return False
    valuesSet = set(values)
    validValuesSet = set(validValues)
    if valuesSet - validValuesSet:
        return False
    for value in values:
        if not isinstance(value, int):
            return False
    return True


def genericNonNegativeIntValidator(value: Any) -> bool:
    """
    Generic. (Added at version 3.)
    """
    if not isinstance(value, int):
        return False
    if value < 0:
        return False
    return True


def genericNonNegativeNumberValidator(value: Any) -> bool:
    """
    Generic. (Added at version 3.)
    """
    if not isinstance(value, numberTypes):
        return False
    if value < 0:
        return False
