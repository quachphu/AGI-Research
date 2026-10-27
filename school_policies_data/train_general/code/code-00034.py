        # contains I;16, we need to convert from native to image
        # byte order.
        if im.mode in ("I;16", "I;16B", "I;16L"):
            rawmode = "I;16N"

        # Pass tags as sorted list so that the tags are set in a fixed order.
        # This is required by libtiff for some tags. For example, the JPEGQUALITY
        # pseudo tag requires that the COMPRESS tag was already set.
        tags = list(atts.items())
        tags.sort()
        a = (rawmode, compression, _fp, filename, tags, types)
        encoder = Image._getencoder(im.mode, "libtiff", a, encoderconfig)
        encoder.setimage(im.im, (0, 0) + im.size)
        while True:
            errcode, data = encoder.encode(ImageFile.MAXBLOCK)[1:]
            if not _fp:
                fp.write(data)
            if errcode:
                break
        if errcode < 0:
            msg = f"encoder error {errcode} when writing image file"
            raise OSError(msg)

    else:
        for tag in blocklist:
            del ifd[tag]
        offset = ifd.save(fp)

        ImageFile._save(
            im,
            fp,
            [ImageFile._Tile("raw", (0, 0) + im.size, offset, (rawmode, stride, 1))],
        )

    # -- helper for multi-page save --
    if "_debug_multipage" in encoderinfo:
        # just to access o32 and o16 (using correct byte order)
        setattr(im, "_debug_multipage", ifd)


class AppendingTiffWriter(io.BytesIO):
    fieldSizes = [
        0,  # None
        1,  # byte
        1,  # ascii
        2,  # short
        4,  # long
        8,  # rational
        1,  # sbyte
        1,  # undefined
        2,  # sshort
        4,  # slong
        8,  # srational
        4,  # float
        8,  # double
        4,  # ifd
        2,  # unicode
        4,  # complex
        8,  # long8
    ]

    Tags = {
        273,  # StripOffsets
        288,  # FreeOffsets
        324,  # TileOffsets
        519,  # JPEGQTables
        520,  # JPEGDCTables
        521,  # JPEGACTables
    }

    def __init__(self, fn: StrOrBytesPath | IO[bytes], new: bool = False) -> None:
        self.f: IO[bytes]
        if is_path(fn):
            self.name = fn
            self.close_fp = True
            try:
                self.f = open(fn, "w+b" if new else "r+b")
            except OSError:
                self.f = open(fn, "w+b")
        else:
            self.f = cast(IO[bytes], fn)
            self.close_fp = False
        self.beginning = self.f.tell()
        self.setup()

    def setup(self) -> None:
        # Reset everything.
        self.f.seek(self.beginning, os.SEEK_SET)

        self.whereToWriteNewIFDOffset: int | None = None
        self.offsetOfNewPage = 0

        self.IIMM = iimm = self.f.read(4)
        self._bigtiff = b"\x2b" in iimm
        if not iimm:
            # empty file - first page
            self.isFirst = True
            return

        self.isFirst = False
        if iimm not in PREFIXES:
            msg = "Invalid TIFF file header"
            raise RuntimeError(msg)

        self.setEndian("<" if iimm.startswith(II) else ">")

        if self._bigtiff:
            self.f.seek(4, os.SEEK_CUR)
        self.skipIFDs()
        self.goToEnd()

    def finalize(self) -> None:
        if self.isFirst:
            return

        # fix offsets
        self.f.seek(self.offsetOfNewPage)

        iimm = self.f.read(4)
        if not iimm:
            # Make it easy to finish a frame without committing to a new one.
            return

        if iimm != self.IIMM:
            msg = "IIMM of new page doesn't match IIMM of first page"
            raise RuntimeError(msg)

        if self._bigtiff:
            self.f.seek(4, os.SEEK_CUR)
        ifd_offset = self._read(8 if self._bigtiff else 4)
        ifd_offset += self.offsetOfNewPage
        assert self.whereToWriteNewIFDOffset is not None
        self.f.seek(self.whereToWriteNewIFDOffset)
        self._write(ifd_offset, 8 if self._bigtiff else 4)
        self.f.seek(ifd_offset)
        self.fixIFD()

    def newFrame(self) -> None:
        # Call this to finish a frame.
        self.finalize()
        self.setup()

    def __enter__(self) -> AppendingTiffWriter:
        return self

    def __exit__(self, *args: object) -> None:
        if self.close_fp:
            self.close()

    def tell(self) -> int:
        return self.f.tell() - self.offsetOfNewPage

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """
        :param offset: Distance to seek.
        :param whence: Whether the distance is relative to the start,
                       end or current position.
        :returns: The resulting position, relative to the start.
        """
        if whence == os.SEEK_SET:
            offset += self.offsetOfNewPage

        self.f.seek(offset, whence)
        return self.tell()

    def goToEnd(self) -> None:
        self.f.seek(0, os.SEEK_END)
        pos = self.f.tell()

        # pad to 16 byte boundary
        pad_bytes = 16 - pos % 16
        if 0 < pad_bytes < 16:
            self.f.write(bytes(pad_bytes))
        self.offsetOfNewPage = self.f.tell()

    def setEndian(self, endian: str) -> None:
        self.endian = endian
        self.longFmt = f"{self.endian}L"
        self.shortFmt = f"{self.endian}H"
        self.tagFormat = f"{self.endian}HH" + ("Q" if self._bigtiff else "L")

    def skipIFDs(self) -> None:
        while True:
            ifd_offset = self._read(8 if self._bigtiff else 4)
            if ifd_offset == 0:
                self.whereToWriteNewIFDOffset = self.f.tell() - (
                    8 if self._bigtiff else 4
                )
                break

            self.f.seek(ifd_offset)
            num_tags = self._read(8 if self._bigtiff else 2)
            self.f.seek(num_tags * (20 if self._bigtiff else 12), os.SEEK_CUR)

    def write(self, data: Buffer, /) -> int:
        return self.f.write(data)

    def _fmt(self, field_size: int) -> str:
        try:
            return {2: "H", 4: "L", 8: "Q"}[field_size]
        except KeyError:
            msg = "offset is not supported"
            raise RuntimeError(msg)

    def _read(self, field_size: int) -> int:
        (value,) = struct.unpack(
            self.endian + self._fmt(field_size), self.f.read(field_size)
        )
        return value

    def readShort(self) -> int:
        return self._read(2)

    def readLong(self) -> int:
        return self._read(4)

    @staticmethod
    def _verify_bytes_written(bytes_written: int | None, expected: int) -> None:
        if bytes_written is not None and bytes_written != expected:
            msg = f"wrote only {bytes_written} bytes but wanted {expected}"
            raise RuntimeError(msg)

    def _rewriteLast(
        self, value: int, field_size: int, new_field_size: int = 0
    ) -> None:
        self.f.seek(-field_size, os.SEEK_CUR)
        if not new_field_size:
            new_field_size = field_size
        bytes_written = self.f.write(
            struct.pack(self.endian + self._fmt(new_field_size), value)
        )
        self._verify_bytes_written(bytes_written, new_field_size)

    def rewriteLastShortToLong(self, value: int) -> None:
        self._rewriteLast(value, 2, 4)

    def rewriteLastShort(self, value: int) -> None:
        return self._rewriteLast(value, 2)

    def rewriteLastLong(self, value: int) -> None:
        return self._rewriteLast(value, 4)

    def _write(self, value: int, field_size: int) -> None:
        bytes_written = self.f.write(
            struct.pack(self.endian + self._fmt(field_size), value)
        )
        self._verify_bytes_written(bytes_written, field_size)

    def writeShort(self, value: int) -> None:
        self._write(value, 2)

    def writeLong(self, value: int) -> None:
        self._write(value, 4)

    def close(self) -> None:
        self.finalize()
        if self.close_fp:
            self.f.close()

    def fixIFD(self) -> None:
        num_tags = self._read(8 if self._bigtiff else 2)

        for i in range(num_tags):
            tag, field_type, count = struct.unpack(
                self.tagFormat, self.f.read(12 if self._bigtiff else 8)
            )

            field_size = self.fieldSizes[field_type]
            total_size = field_size * count
            fmt_size = 8 if self._bigtiff else 4
            is_local = total_size <= fmt_size
            if not is_local:
                offset = self._read(fmt_size) + self.offsetOfNewPage
                self._rewriteLast(offset, fmt_size)

            if tag in self.Tags:
                cur_pos = self.f.tell()

                logger.debug(
                    "fixIFD: %s (%d) - type: %s (%d) - type size: %d - count: %d",
                    TiffTags.lookup(tag).name,
                    tag,
                    TYPES.get(field_type, "unknown"),
                    field_type,
                    field_size,
                    count,
                )

                if is_local:
                    self._fixOffsets(count, field_size)
                    self.f.seek(cur_pos + fmt_size)
                else:
                    self.f.seek(offset)
                    self._fixOffsets(count, field_size)
                    self.f.seek(cur_pos)

            elif is_local:
                # skip the locally stored value that is not an offset
                self.f.seek(fmt_size, os.SEEK_CUR)

    def _fixOffsets(self, count: int, field_size: int) -> None:
        for i in range(count):
            offset = self._read(field_size)
            offset += self.offsetOfNewPage

            new_field_size = 0
            if self._bigtiff and field_size in (2, 4) and offset >= 2**32:
                # offset is now too large - we must convert long to long8
                new_field_size = 8
            elif field_size == 2 and offset >= 2**16:
                # offset is now too large - we must convert short to long
                new_field_size = 4
            if new_field_size:
                if count != 1:
                    msg = "not implemented"
                    raise RuntimeError(msg)  # XXX TODO

                # simple case - the offset is just one and therefore it is
                # local (not referenced with another offset)
                self._rewriteLast(offset, field_size, new_field_size)
                # Move back past the new offset, past 'count', and before 'field_type'
                rewind = -new_field_size - 4 - 2
                self.f.seek(rewind, os.SEEK_CUR)
                self.writeShort(new_field_size)  # rewrite the type
                self.f.seek(2 - rewind, os.SEEK_CUR)
            else:
                self._rewriteLast(offset, field_size)

    def fixOffsets(
        self, count: int, isShort: bool = False, isLong: bool = False
    ) -> None:
        if isShort:
            field_size = 2
        elif isLong:
            field_size = 4
        else:
            field_size = 0
        return self._fixOffsets(count, field_size)


def _save_all(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    append_images = list(im.encoderinfo.get("append_images", []))
    if not hasattr(im, "n_frames") and not append_images:
        return _save(im, fp, filename)

    cur_idx = im.tell()
    try:
        with AppendingTiffWriter(fp) as tf:
            for ims in [im] + append_images:
                encoderinfo = ims._attach_default_encoderinfo(im)
                if not hasattr(ims, "encoderconfig"):
                    ims.encoderconfig = ()
                nfr = getattr(ims, "n_frames", 1)

                for idx in range(nfr):
                    ims.seek(idx)
                    ims.load()
                    _save(ims, tf, filename)
                    tf.newFrame()
                ims.encoderinfo = encoderinfo
    finally:
        im.seek(cur_idx)


#
# --------------------------------------------------------------------
# Register

Image.register_open(TiffImageFile.format, TiffImageFile, _accept)
Image.register_save(TiffImageFile.format, _save)
Image.register_save_all(TiffImageFile.format, _save_all)

Image.register_extensions(TiffImageFile.format, [".tif", ".tiff"])

Image.register_mime(TiffImageFile.format, "image/tiff")
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# WMF stub codec
#
# history:
# 1996-12-14 fl   Created
# 2004-02-22 fl   Turned into a stub driver
# 2004-02-23 fl   Added EMF support
#
# Copyright (c) Secret Labs AB 1997-2004.  All rights reserved.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#
# WMF/EMF reference documentation:
# https://winprotocoldoc.blob.core.windows.net/productionwindowsarchives/MS-WMF/[MS-WMF].pdf
# http://wvware.sourceforge.net/caolan/index.html
# http://wvware.sourceforge.net/caolan/ora-wmf.html
from __future__ import annotations

from typing import IO

from . import Image, ImageFile
from ._binary import i16le as word
from ._binary import si16le as short
from ._binary import si32le as _long

_handler = None


def register_handler(handler: ImageFile.StubHandler | None) -> None:
    """
    Install application-specific WMF image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler


if hasattr(Image.core, "drawwmf"):
    # install default handler (windows only)

    class WmfHandler(ImageFile.StubHandler):
        def open(self, im: ImageFile.StubImageFile) -> None:
            self.bbox = im.info["wmf_bbox"]

        def load(self, im: ImageFile.StubImageFile) -> Image.Image:
            assert im.fp is not None
            im.fp.seek(0)  # rewind
            return Image.frombytes(
                "RGB",
                im.size,
                Image.core.drawwmf(im.fp.read(), im.size, self.bbox),
                "raw",
                "BGR",
                (im.size[0] * 3 + 3) & -4,
                -1,
            )

    register_handler(WmfHandler())

#
# --------------------------------------------------------------------
# Read WMF file


def _accept(prefix: bytes) -> bool:
    return prefix.startswith((b"\xd7\xcd\xc6\x9a\x00\x00", b"\x01\x00\x00\x00"))


##
# Image plugin for Windows metafiles.


class WmfStubImageFile(ImageFile.StubImageFile):
    format = "WMF"
    format_description = "Windows Metafile"

    def _open(self) -> None:
        # check placeable header
        assert self.fp is not None
        s = self.fp.read(44)

        if s.startswith(b"\xd7\xcd\xc6\x9a\x00\x00"):
            # placeable windows metafile

            # get units per inch
            inch = word(s, 14)
            if inch == 0:
                msg = "Invalid inch"
                raise ValueError(msg)
            self._inch: tuple[float, float] = inch, inch

            # get bounding box
            x0 = short(s, 6)
            y0 = short(s, 8)
            x1 = short(s, 10)
            y1 = short(s, 12)

            # normalize size to 72 dots per inch
            self.info["dpi"] = 72
            size = (
                (x1 - x0) * self.info["dpi"] // inch,
                (y1 - y0) * self.info["dpi"] // inch,
            )

            self.info["wmf_bbox"] = x0, y0, x1, y1

            # sanity check (standard metafile header)
            if s[22:26] != b"\x01\x00\t\x00":
                msg = "Unsupported WMF file format"
                raise SyntaxError(msg)

        elif s.startswith(b"\x01\x00\x00\x00") and s[40:44] == b" EMF":
            # enhanced metafile

            # get bounding box
            x0 = _long(s, 8)
            y0 = _long(s, 12)
            x1 = _long(s, 16)
            y1 = _long(s, 20)

            # get frame (in 0.01 millimeter units)
            frame = _long(s, 24), _long(s, 28), _long(s, 32), _long(s, 36)

            size = x1 - x0, y1 - y0

            # calculate dots per inch from bbox and frame
            xdpi = 2540.0 * (x1 - x0) / (frame[2] - frame[0])
            ydpi = 2540.0 * (y1 - y0) / (frame[3] - frame[1])

            self.info["wmf_bbox"] = x0, y0, x1, y1

            if xdpi == ydpi:
                self.info["dpi"] = xdpi
            else:
                self.info["dpi"] = xdpi, ydpi
            self._inch = xdpi, ydpi

        else:
            msg = "Unsupported file format"
            raise SyntaxError(msg)

        self._mode = "RGB"
        self._size = size

    def _load(self) -> ImageFile.StubHandler | None:
        return _handler

    def load(
        self, dpi: float | tuple[float, float] | None = None
    ) -> Image.core.PixelAccess | None:
        if dpi is not None:
            self.info["dpi"] = dpi
            x0, y0, x1, y1 = self.info["wmf_bbox"]
            if not isinstance(dpi, tuple):
                dpi = dpi, dpi
            self._size = (
                int((x1 - x0) * dpi[0] / self._inch[0]),
                int((y1 - y0) * dpi[1] / self._inch[1]),
            )
        return super().load()


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if _handler is None or not hasattr(_handler, "save"):
        msg = "WMF save handler not installed"
        raise OSError(msg)
    _handler.save(im, fp, filename)


#
# --------------------------------------------------------------------
# Registry stuff


Image.register_open(WmfStubImageFile.format, WmfStubImageFile, _accept)
Image.register_save(WmfStubImageFile.format, _save)

Image.register_extensions(WmfStubImageFile.format, [".wmf", ".emf"])
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# Adobe PSD 2.5/3.0 file handling
#
# History:
# 1995-09-01 fl   Created
# 1997-01-03 fl   Read most PSD images
# 1997-01-18 fl   Fixed P and CMYK support
# 2001-10-21 fl   Added seek/tell support (for layers)
#
# Copyright (c) 1997-2001 by Secret Labs AB.
# Copyright (c) 1995-2001 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io
from functools import cached_property
from typing import IO

from . import Image, ImageFile, ImagePalette
from ._binary import i8
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import si16be as si16
from ._binary import si32be as si32
from ._util import DeferredError

MODES = {
    # (photoshop mode, bits) -> (pil mode, required channels)
    (0, 1): ("1", 1),
    (0, 8): ("L", 1),
    (1, 8): ("L", 1),
    (2, 8): ("P", 1),
    (3, 8): ("RGB", 3),
    (4, 8): ("CMYK", 4),
    (7, 8): ("L", 1),  # FIXME: multilayer
    (8, 8): ("L", 1),  # duotone
    (9, 8): ("LAB", 3),
}


# --------------------------------------------------------------------.
# read PSD images


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"8BPS")


##
# Image plugin for Photoshop images.


class PsdImageFile(ImageFile.ImageFile):
    format = "PSD"
    format_description = "Adobe Photoshop"
    _close_exclusive_fp_after_loading = False

    def _open(self) -> None:
        assert self.fp is not None
        read = self.fp.read

        #
        # header

        s = read(26)
        if not _accept(s) or i16(s, 4) != 1:
            msg = "not a PSD file"
            raise SyntaxError(msg)

        psd_bits = i16(s, 22)
        psd_channels = i16(s, 12)
        psd_mode = i16(s, 24)

        mode, channels = MODES[(psd_mode, psd_bits)]

        if channels > psd_channels:
            msg = "not enough channels"
            raise OSError(msg)
        if mode == "RGB" and psd_channels == 4:
            mode = "RGBA"
            channels = 4

        self._mode = mode
        self._size = i32(s, 18), i32(s, 14)

        #
        # color mode data

        size = i32(read(4))
        if size:
            data = read(size)
            if mode == "P" and size == 768:
                self.palette = ImagePalette.raw("RGB;L", data)

        #
        # image resources

        self.resources = []

        size = i32(read(4))
        if size:
            # load resources
            end = self.fp.tell() + size
            while self.fp.tell() < end:
                read(4)  # signature
                id = i16(read(2))
                name = read(i8(read(1)))
                if not (len(name) & 1):
                    read(1)  # padding
                data = read(i32(read(4)))
                if len(data) & 1:
                    read(1)  # padding
                self.resources.append((id, name, data))
                if id == 1039:  # ICC profile
                    self.info["icc_profile"] = data

        #
        # layer and mask information

        self._layers_position = None

        size = i32(read(4))
        if size:
            end = self.fp.tell() + size
            size = i32(read(4))
            if size:
                self._layers_position = self.fp.tell()
                self._layers_size = size
            self.fp.seek(end)
        self._n_frames: int | None = None

        #
        # image descriptor

        self.tile = _maketile(self.fp, mode, (0, 0) + self.size, channels)

        # keep the file open
        self._fp = self.fp
        self.frame = 1
        self._min_frame = 1

    @cached_property
    def layers(
        self,
    ) -> list[tuple[str, str, tuple[int, int, int, int], list[ImageFile._Tile]]]:
        layers = []
        if self._layers_position is not None:
            if isinstance(self._fp, DeferredError):
                raise self._fp.ex
            self._fp.seek(self._layers_position)
            _layer_data = io.BytesIO(ImageFile._safe_read(self._fp, self._layers_size))
            layers = _layerinfo(_layer_data, self._layers_size)
        self._n_frames = len(layers)
        return layers

    @property
    def n_frames(self) -> int:
        if self._n_frames is None:
            self._n_frames = len(self.layers)
        return self._n_frames

    @property
    def is_animated(self) -> bool:
        return len(self.layers) > 1

    def seek(self, layer: int) -> None:
        if not self._seek_check(layer):
            return
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex

        # seek to given layer (1..max)
        if layer > len(self.layers):
            msg = "no more images in PSD file"
            raise EOFError(msg)
        _, mode, _, tile = self.layers[layer - 1]
        self._mode = mode
        self.tile = tile
        self.frame = layer
        self.fp = self._fp

    def tell(self) -> int:
        # return layer number (0=image, 1..max=layers)
        return self.frame


def _layerinfo(
    fp: IO[bytes], ct_bytes: int
) -> list[tuple[str, str, tuple[int, int, int, int], list[ImageFile._Tile]]]:
    # read layerinfo block
    layers = []

    def read(size: int) -> bytes:
        return ImageFile._safe_read(fp, size)

    ct = si16(read(2))

    # sanity check
    if ct_bytes < (abs(ct) * 20):
        msg = "Layer block too short for number of layers requested"
        raise SyntaxError(msg)

    for _ in range(abs(ct)):
        # bounding box
        y0 = si32(read(4))
        x0 = si32(read(4))
        y1 = si32(read(4))
        x1 = si32(read(4))

        # image info
        bands = []
        ct_types = i16(read(2))
        if ct_types > 4:
            fp.seek(ct_types * 6 + 12, io.SEEK_CUR)
            size = i32(read(4))
            fp.seek(size, io.SEEK_CUR)
            continue

        for _ in range(ct_types):
            channel_id = i16(read(2))

            if channel_id == 65535:
                b = "A"
            elif channel_id < 4:
                b = "RGBA"[channel_id]
            else:
                b = ""

            bands.append(b)
            read(4)  # size

        # figure out the image mode
        bands.sort()
        if bands == ["R"]:
            mode = "L"
        elif bands == ["B", "G", "R"]:
            mode = "RGB"
        elif bands == ["A", "B", "G", "R"]:
            mode = "RGBA"
        else:
            mode = ""  # unknown

        # skip over blend flags and extra information
        read(12)  # filler
        name = ""
        size = i32(read(4))  # length of the extra data field
        if size:
            data_end = fp.tell() + size

            length = i32(read(4))
            if length:
                fp.seek(length - 16, io.SEEK_CUR)

            length = i32(read(4))
            if length:
                fp.seek(length, io.SEEK_CUR)

            length = i8(read(1))
            if length:
                # Don't know the proper encoding,
                # Latin-1 should be a good guess
                name = read(length).decode("latin-1", "replace")

            fp.seek(data_end)
        layers.append((name, mode, (x0, y0, x1, y1)))

    # get tiles
    layerinfo = []
    for i, (name, mode, bbox) in enumerate(layers):
        tile = []
        for m in mode:
            t = _maketile(fp, m, bbox, 1)
            if t:
                tile.extend(t)
        layerinfo.append((name, mode, bbox, tile))

    return layerinfo


def _maketile(
    file: IO[bytes], mode: str, bbox: tuple[int, int, int, int], channels: int
) -> list[ImageFile._Tile]:
    tiles = []
    read = file.read

    compression = i16(read(2))

    xsize = bbox[2] - bbox[0]
    ysize = bbox[3] - bbox[1]

    offset = file.tell()

    if compression == 0:
        #
        # raw compression
        for channel in range(channels):
            layer = mode[channel]
            if mode == "CMYK":
                layer += ";I"
            tiles.append(ImageFile._Tile("raw", bbox, offset, layer))
            offset = offset + xsize * ysize

    elif compression == 1:
        #
        # packbits compression
        i = 0
        bytecount = read(channels * ysize * 2)
        offset = file.tell()
        for channel in range(channels):
            layer = mode[channel]
            if mode == "CMYK":
                layer += ";I"
            tiles.append(ImageFile._Tile("packbits", bbox, offset, layer))
            for y in range(ysize):
                offset = offset + i16(bytecount, i)
                i += 2

    file.seek(offset)

    if offset & 1:
        read(1)  # padding

    return tiles


# --------------------------------------------------------------------
# registry


Image.register_open(PsdImageFile.format, PsdImageFile, _accept)

Image.register_extension(PsdImageFile.format, ".psd")

Image.register_mime(PsdImageFile.format, "image/vnd.adobe.photoshop")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# read files from within a tar file
#
# History:
# 95-06-18 fl   Created
# 96-05-28 fl   Open files in binary mode
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1995-96.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io

from . import ContainerIO


class TarIO(ContainerIO.ContainerIO[bytes]):
    """A file object that provides read access to a given member of a TAR file."""

    def __init__(self, tarfile: str, file: str) -> None:
        """
        Create file object.

        :param tarfile: Name of TAR file.
        :param file: Name of member file.
        """
        self.fh = open(tarfile, "rb")

        while True:
            s = self.fh.read(512)
            if len(s) != 512:
                self.fh.close()

                msg = "unexpected end of tar file"
                raise OSError(msg)

            name = s[:100].decode("utf-8")
            i = name.find("\0")
            if i == 0:
                self.fh.close()

                msg = "cannot find subfile"
                raise OSError(msg)
            if i > 0:
                name = name[:i]

            size = int(s[124:135], 8)

            if file == name:
                break

            self.fh.seek((size + 511) & (~511), io.SEEK_CUR)

        # Open region
        super().__init__(self.fh, self.fh.tell(), size)
<|endoftext|>#
# THIS IS WORK IN PROGRESS
#
# The Python Imaging Library.
# $Id$
#
# FlashPix support for PIL
#
# History:
# 97-01-25 fl   Created (reads uncompressed RGB images only)
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import olefile

from . import Image, ImageFile
from ._binary import i32le as i32

# we map from colour field tuples to (mode, rawmode) descriptors
MODES = {
    # opacity
    (0x00007FFE,): ("A", "L"),
    # monochrome
    (0x00010000,): ("L", "L"),
    (0x00018000, 0x00017FFE): ("RGBA", "LA"),
    # photo YCC
    (0x00020000, 0x00020001, 0x00020002): ("RGB", "YCC;P"),
    (0x00028000, 0x00028001, 0x00028002, 0x00027FFE): ("RGBA", "YCCA;P"),
    # standard RGB (NIFRGB)
    (0x00030000, 0x00030001, 0x00030002): ("RGB", "RGB"),
    (0x00038000, 0x00038001, 0x00038002, 0x00037FFE): ("RGBA", "RGBA"),
}


#
# --------------------------------------------------------------------


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(olefile.MAGIC)


##
# Image plugin for the FlashPix images.


class FpxImageFile(ImageFile.ImageFile):
    format = "FPX"
    format_description = "FlashPix"

    def _open(self) -> None:
        #
        # read the OLE directory and see if this is a likely
        # to be a FlashPix file

        assert self.fp is not None
        try:
            self.ole = olefile.OleFileIO(self.fp)
        except OSError as e:
            msg = "not an FPX file; invalid OLE file"
            raise SyntaxError(msg) from e

        root = self.ole.root
        if not root or root.clsid != "56616700-C154-11CE-8553-00AA00A1F95B":
            msg = "not an FPX file; bad root CLSID"
            raise SyntaxError(msg)

        self._open_index(1)

    def _open_index(self, index: int = 1) -> None:
        #
        # get the Image Contents Property Set

        prop = self.ole.getproperties(
            [f"Data Object Store {index:06d}", "\005Image Contents"]
        )

        # size (highest resolution)

        assert isinstance(prop[0x1000002], int)
        assert isinstance(prop[0x1000003], int)
        self._size = prop[0x1000002], prop[0x1000003]

        size = max(self.size)
        i = 1
        while size > 64:
            size = size // 2
            i += 1
        self.maxid = i - 1

        # mode.  instead of using a single field for this, flashpix
        # requires you to specify the mode for each channel in each
        # resolution subimage, and leaves it to the decoder to make
        # sure that they all match.  for now, we'll cheat and assume
        # that this is always the case.

        id = self.maxid << 16

        s = prop[0x2000002 | id]

        if not isinstance(s, bytes) or (bands := i32(s, 4)) > 4:
            msg = "Invalid number of bands"
            raise OSError(msg)

        # note: for now, we ignore the "uncalibrated" flag
        colors = tuple(i32(s, 8 + i * 4) & 0x7FFFFFFF for i in range(bands))

        self._mode, self.rawmode = MODES[colors]

        # load JPEG tables, if any
        self.jpeg = {}
        for i in range(256):
            id = 0x3000001 | (i << 16)
            if id in prop:
                self.jpeg[i] = prop[id]

        self._open_subimage(1, self.maxid)

    def _open_subimage(self, index: int = 1, subimage: int = 0) -> None:
        #
        # setup tile descriptors for a given subimage

        stream = [
            f"Data Object Store {index:06d}",
            f"Resolution {subimage:04d}",
            "Subimage 0000 Header",
        ]

        fp = self.ole.openstream(stream)

        # skip prefix
        fp.read(28)

        # header stream
        s = fp.read(36)

        size = i32(s, 4), i32(s, 8)
        # tilecount = i32(s, 12)
        xtile, ytile = i32(s, 16), i32(s, 20)
        if xtile != 64 or ytile != 64:
            msg = "Tile must be 64 pixels by 64 pixels"
            raise ValueError(msg)

        # channels = i32(s, 24)
        offset = i32(s, 28)
        length = i32(s, 32)

        if size != self.size:
            msg = "subimage mismatch"
            raise OSError(msg)

        # get tile descriptors
        fp.seek(28 + offset)
        s = fp.read(i32(s, 12) * length)

        x = y = 0
        xsize, ysize = size
        self.tile = []

        for i in range(0, len(s), length):
            x1 = min(xsize, x + xtile)
            y1 = min(ysize, y + ytile)

            compression = i32(s, i + 8)

            if compression == 0:
                self.tile.append(
                    ImageFile._Tile(
                        "raw",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        self.rawmode,
                    )
                )

            elif compression == 1:
                # FIXME: the fill decoder is not implemented
                self.tile.append(
                    ImageFile._Tile(
                        "fill",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        (self.rawmode, s[12:16]),
                    )
                )

            elif compression == 2:
                internal_color_conversion = s[14]
                jpeg_tables = s[15]
                rawmode = self.rawmode

                if internal_color_conversion:
                    # The image is stored as usual (usually YCbCr).
                    if rawmode == "RGBA":
                        # For "RGBA", data is stored as YCbCrA based on
                        # negative RGB. The following trick works around
                        # this problem :
                        jpegmode, rawmode = "YCbCrK", "CMYK"
                    else:
                        jpegmode = None  # let the decoder decide

                else:
                    # The image is stored as defined by rawmode
                    jpegmode = rawmode

                self.tile.append(
                    ImageFile._Tile(
                        "jpeg",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        (rawmode, jpegmode),
                    )
                )

                # FIXME: jpeg tables are tile dependent; the prefix
                # data must be placed in the tile descriptor itself!

                if jpeg_tables:
                    self.tile_prefix = self.jpeg[jpeg_tables]

            else:
                msg = "unknown/invalid compression"
                raise OSError(msg)

            x += xtile
            if x >= xsize:
                x, y = 0, y + ytile
                if y >= ysize:
                    break  # isn't really required

        assert self.fp is not None
        self.stream = stream
        self._fp = self.fp
        self.fp = None

    def load(self) -> Image.core.PixelAccess | None:
        if not self.fp:
            self.fp = self.ole.openstream(self.stream[:2] + ["Subimage 0000 Data"])

        return ImageFile.ImageFile.load(self)

    def close(self) -> None:
        self.ole.close()
        super().close()

    def __exit__(self, *args: object) -> None:
        self.ole.close()
        super().__exit__()


#
# --------------------------------------------------------------------


Image.register_open(FpxImageFile.format, FpxImageFile, _accept)

Image.register_extension(FpxImageFile.format, ".fpx")
<|endoftext|>from __future__ import annotations

import os
from io import BytesIO
from typing import IO

from . import ExifTags, Image, ImageFile, ImageSequence

try:
    from . import _avif

    SUPPORTED = True
except ImportError:
    SUPPORTED = False

# Decoder options as module globals, until there is a way to pass parameters
# to Image.open (see https://github.com/python-pillow/Pillow/issues/569)
DECODE_CODEC_CHOICE = "auto"
DEFAULT_MAX_THREADS = 0


def get_codec_version(codec_name: str) -> str | None:
    versions = _avif.codec_versions()
    for version in versions.split(", "):
        if version.split(" [")[0] == codec_name:
            return version.split(":")[-1].split(" ")[0]
    return None


def _accept(prefix: bytes) -> bool | str:
    if prefix[4:8] != b"ftyp":
        return False
    major_brand = prefix[8:12]
    if major_brand in (
        # coding brands
        b"avif",
        b"avis",
        # We accept files with AVIF container brands; we can't yet know if
        # the ftyp box has the correct compatible brands, but if it doesn't
        # then the plugin will raise a SyntaxError which Pillow will catch
        # before moving on to the next plugin that accepts the file.
        #
        # Also, because this file might not actually be an AVIF file, we
        # don't raise an error if AVIF support isn't properly compiled.
        b"mif1",
        b"msf1",
    ):
        if not SUPPORTED:
            return (
                "image file could not be identified because AVIF support not installed"
            )
        return True
    return False


def _get_default_max_threads() -> int:
    if DEFAULT_MAX_THREADS:
        return DEFAULT_MAX_THREADS
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count() or 1


class AvifImageFile(ImageFile.ImageFile):
    format = "AVIF"
    format_description = "AVIF image"
    __frame = -1

    def _open(self) -> None:
        if not SUPPORTED:
            msg = "image file could not be opened because AVIF support not installed"
            raise SyntaxError(msg)

        if DECODE_CODEC_CHOICE != "auto" and not _avif.decoder_codec_available(
            DECODE_CODEC_CHOICE
        ):
            msg = "Invalid opening codec"
            raise ValueError(msg)

        assert self.fp is not None
        self._decoder = _avif.AvifDecoder(
            self.fp.read(),
            DECODE_CODEC_CHOICE,
            _get_default_max_threads(),
        )

        # Get info from decoder
        self._size, self.n_frames, self._mode, icc, exif, exif_orientation, xmp = (
            self._decoder.get_info()
        )
        self.is_animated = self.n_frames > 1

        if icc:
            self.info["icc_profile"] = icc
        if xmp:
            self.info["xmp"] = xmp

        if exif_orientation != 1 or exif:
            exif_data = Image.Exif()
            if exif:
                exif_data.load(exif)
                original_orientation = exif_data.get(ExifTags.Base.Orientation, 1)
            else:
                original_orientation = 1
            if exif_orientation != original_orientation:
                exif_data[ExifTags.Base.Orientation] = exif_orientation
                exif = exif_data.tobytes()
        if exif:
            self.info["exif"] = exif
        self.seek(0)

    def seek(self, frame: int) -> None:
        if not self._seek_check(frame):
            return

        # Set tile
        self.__frame = frame
        self.tile = [ImageFile._Tile("raw", (0, 0) + self.size, 0, self.mode)]

    def load(self) -> Image.core.PixelAccess | None:
        if self.tile:
            # We need to load the image data for this frame
            data, timescale, pts_in_timescales, duration_in_timescales = (
                self._decoder.get_frame(self.__frame)
            )
            self.info["timestamp"] = round(1000 * (pts_in_timescales / timescale))
            self.info["duration"] = round(1000 * (duration_in_timescales / timescale))

            if self.fp and self._exclusive_fp:
                self.fp.close()
            self.fp = BytesIO(data)

        return super().load()

    def load_seek(self, pos: int) -> None:
        pass

    def tell(self) -> int:
        return self.__frame


def _save_all(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    _save(im, fp, filename, save_all=True)


def _save(
    im: Image.Image, fp: IO[bytes], filename: str | bytes, save_all: bool = False
) -> None:
    info = im.encoderinfo.copy()
    if save_all:
        append_images = list(info.get("append_images", []))
    else:
        append_images = []

    grayscale_modes = {"1", "L", "I", "I;16", "I;16L", "I;16B", "I;16N", "F"}
    grayscale = all(
        frame.mode in grayscale_modes
        for ims in [im] + append_images
        for frame in ImageSequence.Iterator(ims)
    )

    quality = info.get("quality", 75)
    if not isinstance(quality, int) or quality < 0 or quality > 100:
        msg = "Invalid quality setting"
        raise ValueError(msg)

    duration = info.get("duration", 0)
    subsampling = info.get("subsampling", "4:0:0" if grayscale else "4:2:0")
    speed = info.get("speed", 6)
    max_threads = info.get("max_threads", _get_default_max_threads())
    codec = info.get("codec", "auto")
    if codec != "auto" and not _avif.encoder_codec_available(codec):
        msg = "Invalid saving codec"
        raise ValueError(msg)
    range_ = info.get("range", "full")
    tile_rows_log2 = info.get("tile_rows", 0)
    tile_cols_log2 = info.get("tile_cols", 0)
    alpha_premultiplied = bool(info.get("alpha_premultiplied", False))
    autotiling = bool(info.get("autotiling", tile_rows_log2 == tile_cols_log2 == 0))

    icc_profile = info.get("icc_profile", im.info.get("icc_profile"))
    exif_orientation = 1
    if exif := info.get("exif"):
        if isinstance(exif, Image.Exif):
            exif_data = exif
        else:
            exif_data = Image.Exif()
            exif_data.load(exif)
        if ExifTags.Base.Orientation in exif_data:
            exif_orientation = exif_data.pop(ExifTags.Base.Orientation)
            exif = exif_data.tobytes() if exif_data else b""
        elif isinstance(exif, Image.Exif):
            exif = exif_data.tobytes()

    xmp = info.get("xmp")

    if isinstance(xmp, str):
        xmp = xmp.encode("utf-8")

    advanced = info.get("advanced")
    if advanced is not None:
        if isinstance(advanced, dict):
            advanced = advanced.items()
        try:
            advanced = tuple(advanced)
        except TypeError:
            invalid = True
        else:
            invalid = any(not isinstance(v, tuple) or len(v) != 2 for v in advanced)
        if invalid:
            msg = (
                "advanced codec options must be a dict of key-value string "
                "pairs or a series of key-value two-tuples"
            )
            raise ValueError(msg)

    # Setup the AVIF encoder
    enc = _avif.AvifEncoder(
        im.size,
        subsampling,
        quality,
        speed,
        max_threads,
        codec,
        range_,
        tile_rows_log2,
        tile_cols_log2,
        alpha_premultiplied,
        autotiling,
        icc_profile or b"",
        exif or b"",
        exif_orientation,
        xmp or b"",
        advanced,
    )

    # Add each frame
    frame_idx = 0
    frame_duration = 0
    cur_idx = im.tell()
    is_single_frame = not append_images and not getattr(im, "is_animated", False)
    try:
        for ims in [im] + append_images:
            for frame in ImageSequence.Iterator(ims):
                # Make sure image mode is supported
                rawmode = frame.mode
                if ims.mode not in {"L", "RGB", "RGBA"}:
                    if ims.has_transparency_data:
                        rawmode = "RGBA"
                    elif ims.mode in grayscale_modes:
                        rawmode = "L"
                    else:
                        rawmode = "RGB"
                    frame = frame.convert(rawmode)

                # Update frame duration
                if isinstance(duration, (list, tuple)):
                    frame_duration = duration[frame_idx]
                else:
                    frame_duration = duration

                # Append the frame to the animation encoder
                enc.add(
                    frame.tobytes("raw", rawmode),
                    frame_duration,
                    frame.size,
                    rawmode,
                    is_single_frame,
                )

                # Update frame index
                frame_idx += 1

                if not save_all:
                    break

    finally:
        im.seek(cur_idx)

    # Get the final output from the encoder
    data = enc.finish()
    if data is None:
        msg = "cannot write file as AVIF (encoder returned None)"
        raise OSError(msg)

    fp.write(data)


Image.register_open(AvifImageFile.format, AvifImageFile, _accept)
if SUPPORTED:
    Image.register_save(AvifImageFile.format, _save)
    Image.register_save_all(AvifImageFile.format, _save_all)
    Image.register_extensions(AvifImageFile.format, [".avif", ".avifs"])
    Image.register_mime(AvifImageFile.format, "image/avif")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# IFUNC IM file handling for PIL
#
# history:
# 1995-09-01 fl   Created.
# 1997-01-03 fl   Save palette images
# 1997-01-08 fl   Added sequence support
# 1997-01-23 fl   Added P and RGB save support
# 1997-05-31 fl   Read floating point images
# 1997-06-22 fl   Save floating point images
# 1997-08-27 fl   Read and save 1-bit images
# 1998-06-25 fl   Added support for RGB+LUT images
# 1998-07-02 fl   Added support for YCC images
# 1998-07-15 fl   Renamed offset attribute to avoid name clash
# 1998-12-29 fl   Added I;16 support
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.7)
# 2003-09-26 fl   Added LA/PA support
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2001 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import os
import re
from typing import IO, Any

from . import Image, ImageFile, ImagePalette
from ._util import DeferredError

# --------------------------------------------------------------------
# Standard tags

COMMENT = "Comment"
DATE = "Date"
EQUIPMENT = "Digitalization equipment"
FRAMES = "File size (no of images)"
LUT = "Lut"
NAME = "Name"
SCALE = "Scale (x,y)"
SIZE = "Image size (x*y)"
MODE = "Image type"

TAGS = {
    COMMENT: 0,
    DATE: 0,
    EQUIPMENT: 0,
    FRAMES: 0,
    LUT: 0,
    NAME: 0,
    SCALE: 0,
    SIZE: 0,
    MODE: 0,
}

OPEN = {
    # ifunc93/p3cfunc formats
    "0 1 image": ("1", "1"),
    "L 1 image": ("1", "1"),
    "Greyscale image": ("L", "L"),
    "Grayscale image": ("L", "L"),
    "RGB image": ("RGB", "RGB;L"),
    "RLB image": ("RGB", "RLB"),
    "RYB image": ("RGB", "RLB"),
    "B1 image": ("1", "1"),
    "B2 image": ("P", "P;2"),
    "B4 image": ("P", "P;4"),
    "X 24 image": ("RGB", "RGB"),
    "L 32 S image": ("I", "I;32"),
    "L 32 F image": ("F", "F;32"),
    # old p3cfunc formats
    "RGB3 image": ("RGB", "RGB;T"),
    "RYB3 image": ("RGB", "RYB;T"),
    # extensions
    "LA image": ("LA", "LA;L"),
    "PA image": ("LA", "PA;L"),
    "RGBA image": ("RGBA", "RGBA;L"),
    "RGBX image": ("RGB", "RGBX;L"),
    "CMYK image": ("CMYK", "CMYK;L"),
    "YCC image": ("YCbCr", "YCbCr;L"),
}

# ifunc95 extensions
for i in ["8", "8S", "16", "16S", "32", "32F"]:
    OPEN[f"L {i} image"] = ("F", f"F;{i}")
    OPEN[f"L*{i} image"] = ("F", f"F;{i}")
for i in ["16", "16L", "16B"]:
    OPEN[f"L {i} image"] = (f"I;{i}", f"I;{i}")
    OPEN[f"L*{i} image"] = (f"I;{i}", f"I;{i}")
for i in ["32S"]:
    OPEN[f"L {i} image"] = ("I", f"I;{i}")
    OPEN[f"L*{i} image"] = ("I", f"I;{i}")
for j in range(2, 33):
    OPEN[f"L*{j} image"] = ("F", f"F;{j}")


# --------------------------------------------------------------------
# Read IM directory

split = re.compile(rb"^([A-Za-z][^:]*):[ \t]*(.*)[ \t]*$")


def number(s: Any) -> float:
    try:
        return int(s)
    except ValueError:
        return float(s)


##
# Image plugin for the IFUNC IM file format.


class ImImageFile(ImageFile.ImageFile):
    format = "IM"
    format_description = "IFUNC Image Memory"
    _close_exclusive_fp_after_loading = False

    def _open(self) -> None:
        # Quick rejection: if there's not an LF among the first
        # 100 bytes, this is (probably) not a text header.

        assert self.fp is not None
        if b"\n" not in self.fp.read(100):
            msg = "not an IM file"
            raise SyntaxError(msg)
        self.fp.seek(0)

        n = 0

        # Default values
        self.info[MODE] = "L"
        self.info[SIZE] = (512, 512)
        self.info[FRAMES] = 1

        self.rawmode = "L"

        while True:
            s = self.fp.read(1)

            # Some versions of IFUNC uses \n\r instead of \r\n...
            if s == b"\r":
                continue

            if not s or s == b"\0" or s == b"\x1a":
                break

            # FIXME: this may read whole file if not a text file
            s = s + self.fp.readline()

            if len(s) > 100:
                msg = "not an IM file"
                raise SyntaxError(msg)

            if s.endswith(b"\r\n"):
                s = s[:-2]
            elif s.endswith(b"\n"):
                s = s[:-1]

            try:
                m = split.match(s)
            except re.error as e:
                msg = "not an IM file"
                raise SyntaxError(msg) from e

            if m:
                k, v = m.group(1, 2)

                # Don't know if this is the correct encoding,
                # but a decent guess (I guess)
                k = k.decode("latin-1", "replace")
                v = v.decode("latin-1", "replace")

                # Convert value as appropriate
                if k in [FRAMES, SCALE, SIZE]:
                    v = v.replace("*", ",")
                    v = tuple(map(number, v.split(",")))
                    if len(v) == 1:
                        v = v[0]
                elif k == MODE and v in OPEN:
                    v, self.rawmode = OPEN[v]

                # Add to dictionary. Note that COMMENT tags are
                # combined into a list of strings.
                if k == COMMENT:
                    if k in self.info:
                        self.info[k].append(v)
                    else:
                        self.info[k] = [v]
                else:
                    self.info[k] = v

                if k in TAGS:
                    n += 1

            else:
                msg = f"Syntax error in IM header: {s.decode('ascii', 'replace')}"
                raise SyntaxError(msg)

        if not n:
            msg = "Not an IM file"
            raise SyntaxError(msg)

        # Basic attributes
        self._size = self.info[SIZE]
        self._mode = self.info[MODE]

        # Skip forward to start of image data
        while s and not s.startswith(b"\x1a"):
            s = self.fp.read(1)
        if not s:
            msg = "File truncated"
            raise SyntaxError(msg)

        if LUT in self.info:
            # convert lookup table to palette or lut attribute
            palette = self.fp.read(768)
            greyscale = 1  # greyscale palette
            linear = 1  # linear greyscale palette
            for i in range(256):
                if palette[i] == palette[i + 256] == palette[i + 512]:
                    if palette[i] != i:
                        linear = 0
                else:
                    greyscale = 0
            if self.mode in ["L", "LA", "P", "PA"]:
                if greyscale:
                    if not linear:
                        self.lut = list(palette[:256])
                else:
                    if self.mode in ["L", "P"]:
                        self._mode = self.rawmode = "P"
                    elif self.mode in ["LA", "PA"]:
                        self._mode = "PA"
                        self.rawmode = "PA;L"
                    self.palette = ImagePalette.raw("RGB;L", palette)
            elif self.mode == "RGB":
                if not greyscale or not linear:
                    self.lut = list(palette)

        self.frame = 0

        self.__offset = offs = self.fp.tell()

        self._fp = self.fp  # FIXME: hack

        if self.rawmode.startswith("F;"):
            # ifunc95 formats
            try:
                # use bit decoder (if necessary)
                bits = int(self.rawmode[2:])
                if bits not in [8, 16, 32]:
                    self.tile = [
                        ImageFile._Tile(
                            "bit", (0, 0) + self.size, offs, (bits, 8, 3, 0, -1)
                        )
                    ]
                    return
            except ValueError:
                pass

        if self.rawmode in ["RGB;T", "RYB;T"]:
            # Old LabEye/3PC files.  Would be very surprised if anyone
            # ever stumbled upon such a file ;-)
            size = self.size[0] * self.size[1]
            self.tile = [
                ImageFile._Tile("raw", (0, 0) + self.size, offs, ("G", 0, -1)),
                ImageFile._Tile("raw", (0, 0) + self.size, offs + size, ("R", 0, -1)),
                ImageFile._Tile(
                    "raw", (0, 0) + self.size, offs + 2 * size, ("B", 0, -1)
                ),
            ]
        else:
            # LabEye/IFUNC files
            self.tile = [
                ImageFile._Tile("raw", (0, 0) + self.size, offs, (self.rawmode, 0, -1))
            ]

    @property
    def n_frames(self) -> int:
        return self.info[FRAMES]

    @property
    def is_animated(self) -> bool:
        return self.info[FRAMES] > 1

    def seek(self, frame: int) -> None:
        if not self._seek_check(frame):
            return
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex

        self.frame = frame

        if self.mode == "1":
            bits = 1
        else:
            bits = 8 * len(self.mode)

        size = ((self.size[0] * bits + 7) // 8) * self.size[1]
        offs = self.__offset + frame * size

        self.fp = self._fp

        self.tile = [
            ImageFile._Tile("raw", (0, 0) + self.size, offs, (self.rawmode, 0, -1))
        ]

    def tell(self) -> int:
        return self.frame


#
# --------------------------------------------------------------------
# Save IM files


SAVE = {
    # mode: (im type, raw mode)
    "1": ("0 1", "1"),
    "L": ("Greyscale", "L"),
    "LA": ("LA", "LA;L"),
    "P": ("Greyscale", "P"),
    "PA": ("LA", "PA;L"),
    "I": ("L 32S", "I;32S"),
    "I;16": ("L 16", "I;16"),
    "I;16L": ("L 16L", "I;16L"),
    "I;16B": ("L 16B", "I;16B"),
    "F": ("L 32F", "F;32F"),
    "RGB": ("RGB", "RGB;L"),
    "RGBA": ("RGBA", "RGBA;L"),
    "RGBX": ("RGBX", "RGBX;L"),
    "CMYK": ("CMYK", "CMYK;L"),
    "YCbCr": ("YCC", "YCbCr;L"),
}


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    try:
        image_type, rawmode = SAVE[im.mode]
    except KeyError as e:
        msg = f"Cannot save {im.mode} images as IM"
        raise ValueError(msg) from e

    frames = im.encoderinfo.get("frames", 1)

    fp.write(f"Image type: {image_type} image\r\n".encode("ascii"))
    if filename:
        # Each line must be 100 characters or less,
        # or: SyntaxError("not an IM file")
        # 8 characters are used for "Name: " and "\r\n"
        # Keep just the filename, ditch the potentially overlong path
        if isinstance(filename, bytes):
            filename = filename.decode("ascii")
        name, ext = os.path.splitext(os.path.basename(filename))
        name = "".join([name[: 92 - len(ext)], ext])

        fp.write(f"Name: {name}\r\n".encode("ascii"))
    fp.write(f"Image size (x*y): {im.size[0]}*{im.size[1]}\r\n".encode("ascii"))
    fp.write(f"File size (no of images): {frames}\r\n".encode("ascii"))
    if im.mode in ["P", "PA"]:
        fp.write(b"Lut: 1\r\n")
    fp.write(b"\000" * (511 - fp.tell()) + b"\032")
    if im.mode in ["P", "PA"]:
        im_palette = im.im.getpalette("RGB", "RGB;L")
        colors = len(im_palette) // 3
        palette = b""
        for i in range(3):
            palette += im_palette[colors * i : colors * (i + 1)]
            palette += b"\x00" * (256 - colors)
        fp.write(palette)  # 768 bytes
    ImageFile._save(
        im, fp, [ImageFile._Tile("raw", (0, 0) + im.size, 0, (rawmode, 0, -1))]
    )


#
# --------------------------------------------------------------------
# Registry


Image.register_open(ImImageFile.format, ImImageFile)
Image.register_save(ImImageFile.format, _save)

Image.register_extension(ImImageFile.format, ".im")
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# BUFR stub adapter
#
# Copyright (c) 1996-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import os
from typing import IO

from . import Image, ImageFile

_handler = None


def register_handler(handler: ImageFile.StubHandler | None) -> None:
    """
    Install application-specific BUFR image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler


# --------------------------------------------------------------------
# Image adapter


def _accept(prefix: bytes) -> bool:
    return prefix.startswith((b"BUFR", b"ZCZC"))


class BufrStubImageFile(ImageFile.StubImageFile):
    format = "BUFR"
    format_description = "BUFR"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(4)):
            msg = "Not a BUFR file"
            raise SyntaxError(msg)

        self.fp.seek(-4, os.SEEK_CUR)

        # make something up
        self._mode = "F"
        self._size = 1, 1

    def _load(self) -> ImageFile.StubHandler | None:
        return _handler


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if _handler is None or not hasattr(_handler, "save"):
        msg = "BUFR save handler not installed"
        raise OSError(msg)
    _handler.save(im, fp, filename)


# --------------------------------------------------------------------
# Registry

Image.register_open(BufrStubImageFile.format, BufrStubImageFile, _accept)
Image.register_save(BufrStubImageFile.format, _save)

Image.register_extension(BufrStubImageFile.format, ".bufr")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# XPM File handling
#
# History:
# 1996-12-29 fl   Created
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.7)
#
# Copyright (c) Secret Labs AB 1997-2001.
# Copyright (c) Fredrik Lundh 1996-2001.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import re

from . import Image, ImageFile, ImagePalette
from ._binary import o8

# XPM header
xpm_head = re.compile(b'"([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)')


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"/* XPM */")


##
# Image plugin for X11 pixel maps.


class XpmImageFile(ImageFile.ImageFile):
    format = "XPM"
    format_description = "X11 Pixel Map"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(9)):
            msg = "not an XPM file"
            raise SyntaxError(msg)

        # skip forward to next string
        while True:
            line = self.fp.readline()
            if not line:
                msg = "broken XPM file"
                raise SyntaxError(msg)
            m = xpm_head.match(line)
            if m:
                break

        self._size = int(m.group(1)), int(m.group(2))

        palette_length = int(m.group(3))
        bpp = int(m.group(4))

        #
        # load palette description

        palette = {}

        for _ in range(palette_length):
            line = self.fp.readline().rstrip()

            c = line[1 : bpp + 1]
            s = line[bpp + 1 : -2].split()

            for i in range(0, len(s), 2):
                if s[i] == b"c":
                    # process colour key
                    rgb = s[i + 1]
                    if rgb == b"None":
                        self.info["transparency"] = c
                    elif rgb.startswith(b"#"):
                        rgb_int = int(rgb[1:], 16)
                        palette[c] = (
                            o8((rgb_int >> 16) & 255)
                            + o8((rgb_int >> 8) & 255)
                            + o8(rgb_int & 255)
                        )
                    else:
                        # unknown colour
                        msg = "cannot read this XPM file"
                        raise ValueError(msg)
                    break

            else:
                # missing colour key
                msg = "cannot read this XPM file"
                raise ValueError(msg)

        args: tuple[int, dict[bytes, bytes] | tuple[bytes, ...]]
        if palette_length > 256:
            self._mode = "RGB"
            args = (bpp, palette)
        else:
            self._mode = "P"
            self.palette = ImagePalette.raw("RGB", b"".join(palette.values()))
            args = (bpp, tuple(palette.keys()))

        self.tile = [ImageFile._Tile("xpm", (0, 0) + self.size, self.fp.tell(), args)]

    def load_read(self, read_bytes: int) -> bytes:
        #
        # load all image data in one chunk

        xsize, ysize = self.size

        assert self.fp is not None
        s = [self.fp.readline()[1 : xsize + 1].ljust(xsize) for i in range(ysize)]

        return b"".join(s)


class XpmDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        assert self.fd is not None

        data = bytearray()
        bpp, palette = self.args
        dest_length = self.state.xsize * self.state.ysize
        if self.mode == "RGB":
            dest_length *= 3
        pixel_header = False
        while len(data) < dest_length:
            line = self.fd.readline()
            if not line:
                break
            if line.rstrip() == b"/* pixels */" and not pixel_header:
                pixel_header = True
                continue
            line = b'"'.join(line.split(b'"')[1:-1])
            for i in range(0, len(line), bpp):
                key = line[i : i + bpp]
                if self.mode == "RGB":
                    data += palette[key]
                else:
                    data += o8(palette.index(key))
        self.set_as_raw(bytes(data))
        return -1, 0


#
# Registry


Image.register_open(XpmImageFile.format, XpmImageFile, _accept)
Image.register_decoder("xpm", XpmDecoder)

Image.register_extension(XpmImageFile.format, ".xpm")

Image.register_mime(XpmImageFile.format, "image/xpm")
<|endoftext|>#
# THIS IS WORK IN PROGRESS
#
# The Python Imaging Library
# $Id$
#
# portable compiled font file parser
#
# history:
# 1997-08-19 fl   created
# 2003-09-13 fl   fixed loading of unicode fonts
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1997-2003 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io

from . import FontFile, Image
from ._binary import i8
from ._binary import i16be as b16
from ._binary import i16le as l16
from ._binary import i32be as b32
from ._binary import i32le as l32

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import BinaryIO

# --------------------------------------------------------------------
# declarations

PCF_MAGIC = 0x70636601  # "\x01fcp"

PCF_PROPERTIES = 1 << 0
PCF_ACCELERATORS = 1 << 1
PCF_METRICS = 1 << 2
PCF_BITMAPS = 1 << 3
PCF_INK_METRICS = 1 << 4
PCF_BDF_ENCODINGS = 1 << 5
PCF_SWIDTHS = 1 << 6
PCF_GLYPH_NAMES = 1 << 7
PCF_BDF_ACCELERATORS = 1 << 8

BYTES_PER_ROW: list[Callable[[int], int]] = [
    lambda bits: ((bits + 7) >> 3),
    lambda bits: ((bits + 15) >> 3) & ~1,
    lambda bits: ((bits + 31) >> 3) & ~3,
    lambda bits: ((bits + 63) >> 3) & ~7,
]


def sz(s: bytes, o: int) -> bytes:
    return s[o : s.index(b"\0", o)]


class PcfFontFile(FontFile.FontFile):
    """Font file plugin for the X11 PCF format."""

    name = "name"

    def __init__(self, fp: BinaryIO, charset_encoding: str = "iso8859-1"):
        self.charset_encoding = charset_encoding

        magic = l32(fp.read(4))
        if magic != PCF_MAGIC:
            msg = "not a PCF file"
            raise SyntaxError(msg)

        super().__init__()

        count = l32(fp.read(4))
        self.toc = {}
        for i in range(count):
            type = l32(fp.read(4))
            self.toc[type] = l32(fp.read(4)), l32(fp.read(4)), l32(fp.read(4))

        self.fp = fp

        self.info = self._load_properties()

        metrics = self._load_metrics()
        bitmaps = self._load_bitmaps(metrics)
        encoding = self._load_encoding()

        #
        # create glyph structure

        for ch, ix in enumerate(encoding):
            if ix is not None:
                (
                    xsize,
                    ysize,
                    left,
                    right,
                    width,
                    ascent,
                    descent,
                    attributes,
                ) = metrics[ix]
                self.glyph[ch] = (
                    (width, 0),
                    (left, descent - ysize, xsize + left, descent),
                    (0, 0, xsize, ysize),
                    bitmaps[ix],
                )

    def _getformat(
        self, tag: int
    ) -> tuple[BinaryIO, int, Callable[[bytes], int], Callable[[bytes], int]]:
        format, size, offset = self.toc[tag]

        fp = self.fp
        fp.seek(offset)

        format = l32(fp.read(4))

        if format & 4:
            i16, i32 = b16, b32
        else:
            i16, i32 = l16, l32

        return fp, format, i16, i32

    def _load_properties(self) -> dict[bytes, bytes | int]:
        #
        # font properties

        properties = {}

        fp, format, i16, i32 = self._getformat(PCF_PROPERTIES)

        nprops = i32(fp.read(4))

        # read property description
        p = [(i32(fp.read(4)), i8(fp.read(1)), i32(fp.read(4))) for _ in range(nprops)]

        if nprops & 3:
            fp.seek(4 - (nprops & 3), io.SEEK_CUR)  # pad

        data = fp.read(i32(fp.read(4)))

        for k, s, v in p:
            property_value: bytes | int = sz(data, v) if s else v
            properties[sz(data, k)] = property_value

        return properties

    def _load_metrics(self) -> list[tuple[int, int, int, int, int, int, int, int]]:
        #
        # font metrics

        metrics: list[tuple[int, int, int, int, int, int, int, int]] = []

        fp, format, i16, i32 = self._getformat(PCF_METRICS)

        def append(
            left: int,
            right: int,
            width: int,
            ascent: int,
            descent: int,
            attributes: int = 0,
        ) -> None:
            xsize = right - left
            ysize = ascent + descent
            Image._decompression_bomb_check((xsize, ysize))
            metrics.append(
                (xsize, ysize, left, right, width, ascent, descent, attributes)
            )

        if (format & 0xFF00) == 0x100:
            # "compressed" metrics
            for i in range(i16(fp.read(2))):
                append(*(i8(fp.read(1)) - 128 for _ in range(5)))

        else:
            # "jumbo" metrics
            for i in range(i32(fp.read(4))):
                append(*(i16(fp.read(2)) for _ in range(6)))

        return metrics

    def _load_bitmaps(
        self, metrics: list[tuple[int, int, int, int, int, int, int, int]]
    ) -> list[Image.Image]:
        #
        # bitmap data

        fp, format, i16, i32 = self._getformat(PCF_BITMAPS)

        nbitmaps = i32(fp.read(4))

        if nbitmaps != len(metrics):
            msg = "Wrong number of bitmaps"
            raise OSError(msg)

        offsets = [i32(fp.read(4)) for _ in range(nbitmaps)]

        bitmap_sizes = [i32(fp.read(4)) for _ in range(4)]

        # byteorder = format & 4  # non-zero => MSB
        bitorder = format & 8  # non-zero => MSB
        padindex = format & 3

        bitmapsize = bitmap_sizes[padindex]
        offsets.append(bitmapsize)

        data = fp.read(bitmapsize)

        pad = BYTES_PER_ROW[padindex]
        mode = "1;R"
        if bitorder:
            mode = "1"

        bitmaps = []
        for i in range(nbitmaps):
            xsize, ysize = metrics[i][:2]
            b, e = offsets[i : i + 2]
            bitmaps.append(
                Image.frombytes("1", (xsize, ysize), data[b:e], "raw", mode, pad(xsize))
            )

        return bitmaps

    def _load_encoding(self) -> list[int | None]:
        fp, format, i16, i32 = self._getformat(PCF_BDF_ENCODINGS)

        first_col, last_col = i16(fp.read(2)), i16(fp.read(2))
        first_row, last_row = i16(fp.read(2)), i16(fp.read(2))

        i16(fp.read(2))  # default

        nencoding = (last_col - first_col + 1) * (last_row - first_row + 1)

        # map character code to bitmap index
        encoding: list[int | None] = [None] * min(256, nencoding)

        encoding_offsets = [i16(fp.read(2)) for _ in range(nencoding)]

        for i in range(first_col, len(encoding)):
            try:
                encoding_offset = encoding_offsets[
                    ord(bytearray([i]).decode(self.charset_encoding))
                ]
                if encoding_offset != 0xFFFF:
                    encoding[i] = encoding_offset
            except UnicodeDecodeError:  # noqa: PERF203
                # character is not supported in selected encoding
                pass

        return encoding
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# Microsoft Image Composer support for PIL
#
# Notes:
#       uses TiffImagePlugin.py to read the actual image streams
#
# History:
#       97-01-20 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import olefile

from . import Image, TiffImagePlugin

#
# --------------------------------------------------------------------


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(olefile.MAGIC)


##
# Image plugin for Microsoft's Image Composer file format.


class MicImageFile(TiffImagePlugin.TiffImageFile):
    format = "MIC"
    format_description = "Microsoft Image Composer"
    _close_exclusive_fp_after_loading = False

    def _open(self) -> None:
        # read the OLE directory and see if this is a likely
        # to be a Microsoft Image Composer file

        try:
            self.ole = olefile.OleFileIO(self.fp)
        except OSError as e:
            msg = "not an MIC file; invalid OLE file"
            raise SyntaxError(msg) from e

        # find ACI subfiles with Image members (maybe not the
        # best way to identify MIC files, but what the... ;-)

        self.images = [
            path
            for path in self.ole.listdir()
            if path[1:] and path[0].endswith(".ACI") and path[1] == "Image"
        ]

        # if we didn't find any images, this is probably not
        # an MIC file.
        if not self.images:
            msg = "not an MIC file; no image entries"
            raise SyntaxError(msg)

        self.frame = -1
        self._n_frames = len(self.images)
        self.is_animated = self._n_frames > 1

        assert self.fp is not None
        self.__fp = self.fp
        self.seek(0)

    def seek(self, frame: int) -> None:
        if not self._seek_check(frame):
            return
        filename = self.images[frame]
        self.fp = self.ole.openstream(filename)

        TiffImagePlugin.TiffImageFile._open(self)

        self.frame = frame

    def tell(self) -> int:
        return self.frame

    def close(self) -> None:
        self.__fp.close()
        self.ole.close()
        super().close()

    def __exit__(self, *args: object) -> None:
        self.__fp.close()
        self.ole.close()
        super().__exit__()


#
# --------------------------------------------------------------------

Image.register_open(MicImageFile.format, MicImageFile, _accept)

Image.register_extension(MicImageFile.format, ".mic")
<|endoftext|>"""Find compiled module linking to Tcl / Tk libraries"""

from __future__ import annotations

import sys
import tkinter

tk = getattr(tkinter, "_tkinter")

try:
    if hasattr(sys, "pypy_find_executable"):
        TKINTER_LIB = tk.tklib_cffi.__file__
    else:
        TKINTER_LIB = tk.__file__
except AttributeError:
    # _tkinter may be compiled directly into Python, in which case __file__ is
    # not available. load_tkinter_funcs will check the binary first in any case.
    TKINTER_LIB = None

tk_version = str(tkinter.TkVersion)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# image enhancement classes
#
# For a background, see "Image Processing By Interpolation and
# Extrapolation", Paul Haeberli and Douglas Voorhies.  Available
# at http://www.graficaobscura.com/interp/index.html
#
# History:
# 1996-03-23 fl  Created
# 2009-06-16 fl  Fixed mean calculation
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import Image, ImageFilter, ImageStat


class _Enhance:
    image: Image.Image
    degenerate: Image.Image

    def enhance(self, factor: float) -> Image.Image:
        """
        Returns an enhanced image.

        :param factor: A floating point value controlling the enhancement.
                       Factor 1.0 always returns a copy of the original image,
                       lower factors mean less color (brightness, contrast,
                       etc), and higher values more. There are no restrictions
                       on this value.
        :rtype: :py:class:`~PIL.Image.Image`
        """
        return Image.blend(self.degenerate, self.image, factor)


class Color(_Enhance):
    """Adjust image color balance.

    This class can be used to adjust the colour balance of an image, in
    a manner similar to the controls on a colour TV set. An enhancement
    factor of 0.0 gives a black and white image. A factor of 1.0 gives
    the original image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.intermediate_mode = "L"
        if "A" in image.getbands():
            self.intermediate_mode = "LA"

        if self.intermediate_mode != image.mode:
            image = image.convert(self.intermediate_mode).convert(image.mode)
        self.degenerate = image


class Contrast(_Enhance):
    """Adjust image contrast.

    This class can be used to control the contrast of an image, similar
    to the contrast control on a TV set. An enhancement factor of 0.0
    gives a solid gray image. A factor of 1.0 gives the original image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        if image.mode != "L":
            image = image.convert("L")
        mean = int(ImageStat.Stat(image).mean[0] + 0.5)
        self.degenerate = Image.new("L", image.size, mean)
        if self.degenerate.mode != self.image.mode:
            self.degenerate = self.degenerate.convert(self.image.mode)

        if "A" in self.image.getbands():
            self.degenerate.putalpha(self.image.getchannel("A"))


class Brightness(_Enhance):
    """Adjust image brightness.

    This class can be used to control the brightness of an image.  An
    enhancement factor of 0.0 gives a black image. A factor of 1.0 gives the
    original image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.degenerate = Image.new(image.mode, image.size, 0)

        if "A" in image.getbands():
            self.degenerate.putalpha(image.getchannel("A"))


class Sharpness(_Enhance):
    """Adjust image sharpness.

    This class can be used to adjust the sharpness of an image. An
    enhancement factor of 0.0 gives a blurred image, a factor of 1.0 gives the
    original image, and a factor of 2.0 gives a sharpened image.
    """

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.degenerate = image.filter(ImageFilter.SMOOTH)

        if "A" in image.getbands():
            self.degenerate.putalpha(image.getchannel("A"))
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# a simple Qt image interface.
#
# history:
# 2006-06-03 fl: created
# 2006-06-04 fl: inherit from QImage instead of wrapping it
# 2006-06-05 fl: removed toimage helper; move string support to ImageQt
# 2013-11-13 fl: add support for Qt5 (aurelien.ballier@cyclonit.com)
#
# Copyright (c) 2006 by Secret Labs AB
# Copyright (c) 2006 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import sys
from io import BytesIO

from . import Image
from ._util import is_path

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from . import ImageFile

    QBuffer: type

qt_version: str | None
qt_versions = [
    ["6", "PyQt6"],
    ["side6", "PySide6"],
]

# If a version has already been imported, attempt it first
qt_versions.sort(key=lambda version: version[1] in sys.modules, reverse=True)
for version, qt_module in qt_versions:
    try:
        qRgba: Callable[[int, int, int, int], int]
        if qt_module == "PyQt6":
            from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
            from PyQt6.QtGui import QImage, QPixmap, qRgba
        elif qt_module == "PySide6":
            from PySide6.QtCore import (  # type: ignore[assignment]
                QBuffer,
                QByteArray,
                QIODevice,
            )
            from PySide6.QtGui import QImage, QPixmap, qRgba  # type: ignore[assignment]
    except (ImportError, RuntimeError):
        continue
    qt_is_installed = True
    qt_version = version
    break
else:
    qt_is_installed = False
    qt_version = None


def rgb(r: int, g: int, b: int, a: int = 255) -> int:
    """(Internal) Turns an RGB color into a Qt compatible color integer."""
    # use qRgb to pack the colors, and then turn the resulting long
    # into a negative integer with the same bitpattern.
    return qRgba(r, g, b, a) & 0xFFFFFFFF


def fromqimage(im: QImage | QPixmap) -> ImageFile.ImageFile:
    """
    :param im: QImage or PIL ImageQt object
    """
    buffer = QBuffer()
    qt_openmode: object
    if qt_version == "6":
        try:
            qt_openmode = getattr(QIODevice, "OpenModeFlag")
        except AttributeError:
            qt_openmode = getattr(QIODevice, "OpenMode")
    else:
        qt_openmode = QIODevice
    buffer.open(getattr(qt_openmode, "ReadWrite"))
    # preserve alpha channel with png
    # otherwise ppm is more friendly with Image.open
    if im.hasAlphaChannel():
        im.save(buffer, "png")
    else:
        im.save(buffer, "ppm")

    b = BytesIO()
    b.write(buffer.data())
    buffer.close()
    b.seek(0)

    return Image.open(b)


def fromqpixmap(im: QPixmap) -> ImageFile.ImageFile:
    return fromqimage(im)


def align8to32(bytes: bytes, width: int, mode: str) -> bytes:
    """
    converts each scanline of data from 8 bit to 32 bit aligned
    """

    bits_per_pixel = {"1": 1, "L": 8, "P": 8, "I;16": 16}[mode]

    # calculate bytes per line and the extra padding if needed
    bits_per_line = bits_per_pixel * width
    full_bytes_per_line, remaining_bits_per_line = divmod(bits_per_line, 8)
    bytes_per_line = full_bytes_per_line + (1 if remaining_bits_per_line else 0)

    extra_padding = -bytes_per_line % 4

    # already 32 bit aligned by luck
    if not extra_padding:
        return bytes

    new_data = [
        bytes[i * bytes_per_line : (i + 1) * bytes_per_line] + b"\x00" * extra_padding
        for i in range(len(bytes) // bytes_per_line)
    ]

    return b"".join(new_data)


def _toqclass_helper(im: Image.Image | str | QByteArray) -> dict[str, Any]:
    data = None
    colortable = None
    exclusive_fp = False

    # handle filename, if given instead of image name
    if hasattr(im, "toUtf8"):
        # FIXME - is this really the best way to do this?
        im = str(im.toUtf8(), "utf-8")
    if is_path(im):
        im = Image.open(im)
        exclusive_fp = True
    assert isinstance(im, Image.Image)

    qt_format = getattr(QImage, "Format") if qt_version == "6" else QImage
    if im.mode == "1":
        format = getattr(qt_format, "Format_Mono")
    elif im.mode == "L":
        format = getattr(qt_format, "Format_Indexed8")
        colortable = [rgb(i, i, i) for i in range(256)]
    elif im.mode == "P":
        format = getattr(qt_format, "Format_Indexed8")
        palette = im.getpalette()
        assert palette is not None
        colortable = [rgb(*palette[i : i + 3]) for i in range(0, len(palette), 3)]
    elif im.mode == "RGB":
        # Populate the 4th channel with 255
        im = im.convert("RGBA")

        data = im.tobytes("raw", "BGRA")
        format = getattr(qt_format, "Format_RGB32")
    elif im.mode == "RGBA":
        data = im.tobytes("raw", "BGRA")
        format = getattr(qt_format, "Format_ARGB32")
    elif im.mode == "I;16":
        im = im.point(lambda i: i * 256)

        format = getattr(qt_format, "Format_Grayscale16")
    else:
        if exclusive_fp:
            im.close()
        msg = f"unsupported image mode {repr(im.mode)}"
        raise ValueError(msg)

    size = im.size
    __data = data or align8to32(im.tobytes(), size[0], im.mode)
    if exclusive_fp:
        im.close()
    return {"data": __data, "size": size, "format": format, "colortable": colortable}


if qt_is_installed:

    class ImageQt(QImage):
        def __init__(self, im: Image.Image | str | QByteArray) -> None:
            """
            An PIL image wrapper for Qt.  This is a subclass of PyQt's QImage
            class.

            :param im: A PIL Image object, or a file name (given either as
                Python string or a PyQt string object).
            """
            im_data = _toqclass_helper(im)
            # must keep a reference, or Qt will crash!
            # All QImage constructors that take data operate on an existing
            # buffer, so this buffer has to hang on for the life of the image.
            # Fixes https://github.com/python-pillow/Pillow/issues/1370
            self.__data = im_data["data"]
            super().__init__(
                self.__data,
                im_data["size"][0],
                im_data["size"][1],
                im_data["format"],
            )
            if im_data["colortable"]:
                self.setColorTable(im_data["colortable"])


def toqimage(im: Image.Image | str | QByteArray) -> ImageQt:
    return ImageQt(im)


def toqpixmap(im: Image.Image | str | QByteArray) -> QPixmap:
    qimage = toqimage(im)
    pixmap = getattr(QPixmap, "fromImage")(qimage)
    if qt_version == "6":
        pixmap.detach()
    return pixmap
<|endoftext|>"""Pillow (Fork of the Python Imaging Library)

Pillow is the friendly PIL fork by Jeffrey 'Alex' Clark and contributors.
    https://github.com/python-pillow/Pillow/

Pillow is forked from PIL 1.1.7.

PIL is the Python Imaging Library by Fredrik Lundh and contributors.
Copyright (c) 1999 by Secret Labs AB.

Use PIL.__version__ for this Pillow version.

;-)
"""

from __future__ import annotations

from . import _version

# VERSION was removed in Pillow 6.0.0.
# PILLOW_VERSION was removed in Pillow 9.0.0.
# Use __version__ instead.
__version__ = _version.__version__
del _version


_plugins = [
    "AvifImagePlugin",
    "BlpImagePlugin",
    "BmpImagePlugin",
    "BufrStubImagePlugin",
    "CurImagePlugin",
    "DcxImagePlugin",
    "DdsImagePlugin",
    "EpsImagePlugin",
    "FitsImagePlugin",
    "FliImagePlugin",
    "FpxImagePlugin",
    "FtexImagePlugin",
    "GbrImagePlugin",
    "GifImagePlugin",
    "GribStubImagePlugin",
    "Hdf5StubImagePlugin",
    "IcnsImagePlugin",
    "IcoImagePlugin",
    "ImImagePlugin",
    "ImtImagePlugin",
    "IptcImagePlugin",
    "JpegImagePlugin",
    "Jpeg2KImagePlugin",
    "McIdasImagePlugin",
    "MicImagePlugin",
    "MpegImagePlugin",
    "MpoImagePlugin",
    "MspImagePlugin",
    "PalmImagePlugin",
    "PcdImagePlugin",
    "PcxImagePlugin",
    "PdfImagePlugin",
    "PixarImagePlugin",
    "PngImagePlugin",
    "PpmImagePlugin",
    "PsdImagePlugin",
    "QoiImagePlugin",
    "SgiImagePlugin",
    "SpiderImagePlugin",
    "SunImagePlugin",
    "TgaImagePlugin",
    "TiffImagePlugin",
    "WebPImagePlugin",
    "WmfImagePlugin",
    "XbmImagePlugin",
    "XpmImagePlugin",
    "XVThumbImagePlugin",
]


class UnidentifiedImageError(OSError):
    """
    Raised in :py:meth:`PIL.Image.open` if an image cannot be opened and identified.

    If a PNG image raises this error, setting :data:`.ImageFile.LOAD_TRUNCATED_IMAGES`
    to true may allow the image to be opened after all. The setting will ignore missing
    data and checksum failures.
    """

    pass
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# SGI image file handling
#
# See "The SGI Image File Format (Draft version 0.97)", Paul Haeberli.
# <ftp://ftp.sgi.com/graphics/SGIIMAGESPEC>
#
#
# History:
# 2017-22-07 mb   Add RLE decompression
# 2016-16-10 mb   Add save method without compression
# 1995-09-10 fl   Created
#
# Copyright (c) 2016 by Mickael Bonfill.
# Copyright (c) 2008 by Karsten Hiddemann.
# Copyright (c) 1997 by Secret Labs AB.
# Copyright (c) 1995 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import os
import struct
from typing import IO

from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8


def _accept(prefix: bytes) -> bool:
    return len(prefix) >= 2 and i16(prefix) == 474


MODES = {
    (1, 1, 1): "L",
    (1, 2, 1): "L",
    (2, 1, 1): "L;16B",
    (2, 2, 1): "L;16B",
    (1, 3, 3): "RGB",
    (2, 3, 3): "RGB;16B",
    (1, 3, 4): "RGBA",
    (2, 3, 4): "RGBA;16B",
}


##
# Image plugin for SGI images.
class SgiImageFile(ImageFile.ImageFile):
    format = "SGI"
    format_description = "SGI Image File Format"

    def _open(self) -> None:
        # HEAD
        assert self.fp is not None

        headlen = 512
        s = self.fp.read(headlen)

        if not _accept(s):
            msg = "Not an SGI image file"
            raise ValueError(msg)

        # compression : verbatim or RLE
        compression = s[2]

        # bpc : 1 or 2 bytes (8bits or 16bits)
        bpc = s[3]

        # dimension : 1, 2 or 3 (depending on xsize, ysize and zsize)
        dimension = i16(s, 4)

        # xsize : width
        xsize = i16(s, 6)

        # ysize : height
        ysize = i16(s, 8)

        # zsize : channels count
        zsize = i16(s, 10)

        # determine mode from bits/zsize
        try:
            rawmode = MODES[(bpc, dimension, zsize)]
        except KeyError:
            msg = "Unsupported SGI image mode"
            raise ValueError(msg)

        self._size = xsize, ysize
        self._mode = rawmode.split(";")[0]
        if self.mode == "RGB":
            self.custom_mimetype = "image/rgb"

        # orientation -1 : scanlines begins at the bottom-left corner
        orientation = -1

        # decoder info
        if compression == 0:
            pagesize = xsize * ysize * bpc
            if bpc == 2:
                self.tile = [
                    ImageFile._Tile(
                        "SGI16",
                        (0, 0) + self.size,
                        headlen,
                        (self.mode, 0, orientation),
                    )
                ]
            else:
                self.tile = []
                offset = headlen
                for layer in self.mode:
                    self.tile.append(
                        ImageFile._Tile(
                            "raw", (0, 0) + self.size, offset, (layer, 0, orientation)
                        )
                    )
                    offset += pagesize
        elif compression == 1:
            self.tile = [
                ImageFile._Tile(
                    "sgi_rle", (0, 0) + self.size, headlen, (rawmode, orientation, bpc)
                )
            ]


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode not in {"RGB", "RGBA", "L"}:
        msg = "Unsupported SGI image mode"
        raise ValueError(msg)

    # Get the keyword arguments
    info = im.encoderinfo

    # Byte-per-pixel precision, 1 = 8bits per pixel
    bpc = info.get("bpc", 1)

    if bpc not in (1, 2):
        msg = "Unsupported number of bytes per pixel"
        raise ValueError(msg)

    # Flip the image, since the origin of SGI file is the bottom-left corner
    orientation = -1
    # Define the file as SGI File Format
    magic_number = 474
    # Run-Length Encoding Compression - Unsupported at this time
    rle = 0

    # X Dimension = width / Y Dimension = height
    x, y = im.size
    # Z Dimension: Number of channels
    z = len(im.mode)
    # Number of dimensions (x,y,z)
    if im.mode == "L":
        dimension = 1 if y == 1 else 2
    else:
        dimension = 3

    # Minimum Byte value
    pinmin = 0
    # Maximum Byte value (255 = 8bits per pixel)
    pinmax = 255
    # Image name (79 characters max, truncated below in write)
    img_name = os.path.splitext(os.path.basename(filename))[0]
    if isinstance(img_name, str):
        img_name = img_name.encode("ascii", "ignore")
    # Standard representation of pixel in the file
    colormap = 0
    fp.write(struct.pack(">h", magic_number))
    fp.write(o8(rle))
    fp.write(o8(bpc))
    fp.write(struct.pack(">H", dimension))
    fp.write(struct.pack(">H", x))
    fp.write(struct.pack(">H", y))
    fp.write(struct.pack(">H", z))
    fp.write(struct.pack(">l", pinmin))
    fp.write(struct.pack(">l", pinmax))
    fp.write(struct.pack("4s", b""))  # dummy
    fp.write(struct.pack("79s", img_name))  # truncates to 79 chars
    fp.write(struct.pack("s", b""))  # force null byte after img_name
    fp.write(struct.pack(">l", colormap))
    fp.write(struct.pack("404s", b""))  # dummy

    rawmode = "L"
    if bpc == 2:
        rawmode = "L;16B"

    for channel in im.split():
        fp.write(channel.tobytes("raw", rawmode, 0, orientation))

    if hasattr(fp, "flush"):
        fp.flush()


class SGI16Decoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        assert self.fd is not None
        assert self.im is not None

        rawmode, stride, orientation = self.args
        pagesize = self.state.xsize * self.state.ysize
        zsize = len(self.mode)
        self.fd.seek(512)

        for band in range(zsize):
            channel = Image.new("L", (self.state.xsize, self.state.ysize))
            channel.frombytes(
                self.fd.read(2 * pagesize), "raw", "L;16B", stride, orientation
            )
            self.im.putband(channel.im, band)

        return -1, 0


#
# registry


Image.register_decoder("SGI16", SGI16Decoder)
Image.register_open(SgiImageFile.format, SgiImageFile, _accept)
Image.register_save(SgiImageFile.format, _save)
Image.register_mime(SgiImageFile.format, "image/sgi")

Image.register_extensions(SgiImageFile.format, [".bw", ".rgb", ".rgba", ".sgi"])

# End of file
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# WCK-style drawing interface operations
#
# History:
# 2003-12-07 fl   created
# 2005-05-15 fl   updated; added to PIL as ImageDraw2
# 2005-05-15 fl   added text support
# 2005-05-20 fl   added arc/chord/pieslice support
#
# Copyright (c) 2003-2005 by Secret Labs AB
# Copyright (c) 2003-2005 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#


"""
(Experimental) WCK-style drawing interface operations

.. seealso:: :py:mod:`PIL.ImageDraw`
"""

from __future__ import annotations

from typing import Any, AnyStr, BinaryIO

from . import Image, ImageColor, ImageDraw, ImageFont, ImagePath
from ._typing import Coords, StrOrBytesPath


class Pen:
    """Stores an outline color and width."""

    def __init__(self, color: str, width: int = 1, opacity: int = 255) -> None:
        self.color = ImageColor.getrgb(color)
        self.width = width


class Brush:
    """Stores a fill color"""

    def __init__(self, color: str, opacity: int = 255) -> None:
        self.color = ImageColor.getrgb(color)


class Font:
    """Stores a TrueType font and color"""

    def __init__(
        self, color: str, file: StrOrBytesPath | BinaryIO, size: float = 12
    ) -> None:
        # FIXME: add support for bitmap fonts
        self.color = ImageColor.getrgb(color)
        self.font = ImageFont.truetype(file, size)


class Draw:
    """
    (Experimental) WCK-style drawing interface
    """

    def __init__(
        self,
        image: Image.Image | str,
        size: tuple[int, int] | list[int] | None = None,
        color: float | tuple[float, ...] | str | None = None,
    ) -> None:
        if isinstance(image, str):
            if size is None:
                msg = "If image argument is mode string, size must be a list or tuple"
                raise ValueError(msg)
            image = Image.new(image, size, color)
        self.draw = ImageDraw.Draw(image)
        self.image = image
        self.transform: tuple[float, float, float, float, float, float] | None = None

    def flush(self) -> Image.Image:
        return self.image

    def render(
        self,
        op: str,
        xy: Coords,
        pen: Pen | Brush | None,
        brush: Brush | Pen | None = None,
        **kwargs: Any,
    ) -> None:
        # handle color arguments
        outline = fill = None
        width = 1
        if isinstance(pen, Pen):
            outline = pen.color
            width = pen.width
        elif isinstance(brush, Pen):
            outline = brush.color
            width = brush.width
        if isinstance(brush, Brush):
            fill = brush.color
        elif isinstance(pen, Brush):
            fill = pen.color
        # handle transformation
        if self.transform:
            path = ImagePath.Path(xy)
            path.transform(self.transform)
            xy = path
        # render the item
        if op in ("arc", "line"):
            kwargs.setdefault("fill", outline)
        else:
            kwargs.setdefault("fill", fill)
            kwargs.setdefault("outline", outline)
        if op == "line":
            kwargs.setdefault("width", width)
        getattr(self.draw, op)(xy, **kwargs)

    def settransform(self, offset: tuple[float, float]) -> None:
        """Sets a transformation offset."""
        xoffset, yoffset = offset
        self.transform = (1, 0, xoffset, 0, 1, yoffset)

    def arc(
        self,
        xy: Coords,
        pen: Pen | Brush | None,
        start: float,
        end: float,
        *options: Any,
    ) -> None:
        """
        Draws an arc (a portion of a circle outline) between the start and end
        angles, inside the given bounding box.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.arc`
        """
        self.render("arc", xy, pen, *options, start=start, end=end)

    def chord(
        self,
        xy: Coords,
        pen: Pen | Brush | None,
        start: float,
        end: float,
        *options: Any,
    ) -> None:
        """
        Same as :py:meth:`~PIL.ImageDraw2.Draw.arc`, but connects the end points
        with a straight line.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.chord`
        """
        self.render("chord", xy, pen, *options, start=start, end=end)

    def ellipse(self, xy: Coords, pen: Pen | Brush | None, *options: Any) -> None:
        """
        Draws an ellipse inside the given bounding box.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.ellipse`
        """
        self.render("ellipse", xy, pen, *options)

    def line(self, xy: Coords, pen: Pen | Brush | None, *options: Any) -> None:
        """
        Draws a line between the coordinates in the ``xy`` list.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.line`
        """
        self.render("line", xy, pen, *options)

    def pieslice(
        self,
        xy: Coords,
        pen: Pen | Brush | None,
        start: float,
        end: float,
        *options: Any,
    ) -> None:
        """
        Same as arc, but also draws straight lines between the end points and the
        center of the bounding box.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.pieslice`
        """
        self.render("pieslice", xy, pen, *options, start=start, end=end)

    def polygon(self, xy: Coords, pen: Pen | Brush | None, *options: Any) -> None:
        """
        Draws a polygon.

        The polygon outline consists of straight lines between the given
        coordinates, plus a straight line between the last and the first
        coordinate.


        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.polygon`
        """
        self.render("polygon", xy, pen, *options)

    def rectangle(self, xy: Coords, pen: Pen | Brush | None, *options: Any) -> None:
        """
        Draws a rectangle.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.rectangle`
        """
        self.render("rectangle", xy, pen, *options)

    def text(self, xy: tuple[float, float], text: AnyStr, font: Font) -> None:
        """
        Draws the string at the given position.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.text`
        """
        if self.transform:
            path = ImagePath.Path(xy)
            path.transform(self.transform)
            xy = path
        self.draw.text(xy, text, font=font.font, fill=font.color)

    def textbbox(
        self, xy: tuple[float, float], text: AnyStr, font: Font
    ) -> tuple[float, float, float, float]:
        """
        Returns bounding box (in pixels) of given text.

        :return: ``(left, top, right, bottom)`` bounding box

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.textbbox`
        """
        if self.transform:
            path = ImagePath.Path(xy)
            path.transform(self.transform)
            xy = path
        return self.draw.textbbox(xy, text, font=font.font)

    def textlength(self, text: AnyStr, font: Font) -> float:
        """
        Returns length (in pixels) of given text.
        This is the amount by which following text should be offset.

        .. seealso:: :py:meth:`PIL.ImageDraw.ImageDraw.textlength`
        """
        return self.draw.textlength(text, font=font.font)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# MPO file handling
#
# See "Multi-Picture Format" (CIPA DC-007-Translation 2009, Standard of the
# Camera & Imaging Products Association)
#
# The multi-picture object combines multiple JPEG images (with a modified EXIF
# data format) into a single file. While it can theoretically be used much like
# a GIF animation, it is commonly used to represent 3D photographs and is (as
# of this writing) the most commonly used format by 3D cameras.
#
# History:
# 2014-03-13 Feneric   Created
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import os
import struct
from typing import IO, Any, cast

from . import (
    Image,
    ImageFile,
    ImageSequence,
    JpegImagePlugin,
    TiffImagePlugin,
)
from ._binary import o32le
from ._util import DeferredError


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    JpegImagePlugin._save(im, fp, filename)


def _save_all(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    append_images = im.encoderinfo.get("append_images", [])
    if not append_images and not getattr(im, "is_animated", False):
        _save(im, fp, filename)
        return

    mpf_offset = 28
    offsets: list[int] = []
    im_sequences = [im, *append_images]
    total = sum(getattr(seq, "n_frames", 1) for seq in im_sequences)
    for im_sequence in im_sequences:
        for im_frame in ImageSequence.Iterator(im_sequence):
            if not offsets:
                # APP2 marker
                ifd_length = 66 + 16 * total
                im_frame.encoderinfo["extra"] = (
                    b"\xff\xe2"
                    + struct.pack(">H", 6 + ifd_length)
                    + b"MPF\0"
                    + b" " * ifd_length
                )
                if exif := im_frame.encoderinfo.get("exif"):
                    if isinstance(exif, Image.Exif):
                        exif = exif.tobytes()
                        im_frame.encoderinfo["exif"] = exif
                    mpf_offset += 4 + len(exif)

                JpegImagePlugin._save(im_frame, fp, filename)
                offsets.append(fp.tell())
            else:
                encoderinfo = im_frame._attach_default_encoderinfo(im)
                im_frame.save(fp, "JPEG")
                im_frame.encoderinfo = encoderinfo
                offsets.append(fp.tell() - offsets[-1])

    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[0xB000] = b"0100"
    ifd[0xB001] = len(offsets)

    mpentries = b""
    data_offset = 0
    for i, size in enumerate(offsets):
        if i == 0:
            mptype = 0x030000  # Baseline MP Primary Image
        else:
            mptype = 0x000000  # Undefined
        mpentries += struct.pack("<LLLHH", mptype, size, data_offset, 0, 0)
        if i == 0:
            data_offset -= mpf_offset
        data_offset += size
    ifd[0xB002] = mpentries

    fp.seek(mpf_offset)
    fp.write(b"II\x2a\x00" + o32le(8) + ifd.tobytes(8))
    fp.seek(0, os.SEEK_END)


##
# Image plugin for MPO images.


class MpoImageFile(JpegImagePlugin.JpegImageFile):
    format = "MPO"
    format_description = "MPO (CIPA DC-007)"
    _close_exclusive_fp_after_loading = False

    def _open(self) -> None:
        assert self.fp is not None
        self.fp.seek(0)  # prep the fp in order to pass the JPEG test
        JpegImagePlugin.JpegImageFile._open(self)
        self._after_jpeg_open()

    def _after_jpeg_open(self, mpheader: dict[int, Any] | None = None) -> None:
        self.mpinfo = mpheader if mpheader is not None else self._getmp()
        if self.mpinfo is None:
            msg = "Image appears to be a malformed MPO file"
            raise ValueError(msg)
        self.n_frames = self.mpinfo[0xB001]
        self.__mpoffsets = [
            mpent["DataOffset"] + self.info["mpoffset"] for mpent in self.mpinfo[0xB002]
        ]
        self.__mpoffsets[0] = 0
        # Note that the following assertion will only be invalid if something
        # gets broken within JpegImagePlugin.
        assert self.n_frames == len(self.__mpoffsets)
        del self.info["mpoffset"]  # no longer needed
        self.is_animated = self.n_frames > 1
        assert self.fp is not None
        self._fp = self.fp  # FIXME: hack
        self._fp.seek(self.__mpoffsets[0])  # get ready to read first frame
        self.__frame = 0
        self.offset = 0
        # for now we can only handle reading and individual frame extraction
        self.readonly = 1

    def load_seek(self, pos: int) -> None:
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex
        self._fp.seek(pos)

    def seek(self, frame: int) -> None:
        if not self._seek_check(frame):
            return
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex
        self.fp = self._fp
        self.offset = self.__mpoffsets[frame]

        original_exif = self.info.get("exif")
        if "exif" in self.info:
            del self.info["exif"]

        self.fp.seek(self.offset + 2)  # skip SOI marker
        if not self.fp.read(2):
            msg = "No data found for frame"
            raise ValueError(msg)
        self.fp.seek(self.offset)
        JpegImagePlugin.JpegImageFile._open(self)
        if self.info.get("exif") != original_exif:
            self._reload_exif()

        self.tile = [
            ImageFile._Tile("jpeg", (0, 0) + self.size, self.offset, self.tile[0][-1])
        ]
        self.__frame = frame

    def tell(self) -> int:
        return self.__frame

    @staticmethod
    def adopt(
        jpeg_instance: JpegImagePlugin.JpegImageFile,
        mpheader: dict[int, Any] | None = None,
    ) -> MpoImageFile:
        """
        Transform the instance of JpegImageFile into
        an instance of MpoImageFile.
        After the call, the JpegImageFile is extended
        to be an MpoImageFile.

        This is essentially useful when opening a JPEG
        file that reveals itself as an MPO, to avoid
        double call to _open.
        """
        jpeg_instance.__class__ = MpoImageFile
        mpo_instance = cast(MpoImageFile, jpeg_instance)
        mpo_instance._after_jpeg_open(mpheader)
        return mpo_instance


# ---------------------------------------------------------------------
# Registry stuff

# Note that since MPO shares a factory with JPEG, we do not need to do a
# separate registration for it here.
# Image.register_open(MpoImageFile.format,
#                     JpegImagePlugin.jpeg_factory, _accept)
Image.register_save(MpoImageFile.format, _save)
Image.register_save_all(MpoImageFile.format, _save_all)

Image.register_extension(MpoImageFile.format, ".mpo")

Image.register_mime(MpoImageFile.format, "image/mpo")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# PCD file handling
#
# History:
#       96-05-10 fl     Created
#       96-05-27 fl     Added draft mode (128x192, 256x384)
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import Image, ImageFile

##
# Image plugin for PhotoCD images.  This plugin only reads the 768x512
# image from the file; higher resolutions are encoded in a proprietary
# encoding.


class PcdImageFile(ImageFile.ImageFile):
    format = "PCD"
    format_description = "Kodak PhotoCD"

    def _open(self) -> None:
        # rough
        assert self.fp is not None

        self.fp.seek(2048)
        s = self.fp.read(1539)

        if not s.startswith(b"PCD_"):
            msg = "not a PCD file"
            raise SyntaxError(msg)

        orientation = s[1538] & 3
        self.tile_post_rotate = None
        if orientation == 1:
            self.tile_post_rotate = 90
        elif orientation == 3:
            self.tile_post_rotate = 270

        self._mode = "RGB"
        self._size = (512, 768) if orientation in (1, 3) else (768, 512)
        self.tile = [ImageFile._Tile("pcd", (0, 0, 768, 512), 96 * 2048)]

    def load_prepare(self) -> None:
        if self._im is None and self.tile_post_rotate:
            self.im = Image.core.new(self.mode, (768, 512))
        ImageFile.ImageFile.load_prepare(self)

    def load_end(self) -> None:
        if self.tile_post_rotate:
            # Handle rotated PCDs
            self.im = self.rotate(self.tile_post_rotate, expand=True).im


#
# registry

Image.register_open(PcdImageFile.format, PcdImageFile)

Image.register_extension(PcdImageFile.format, ".pcd")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# Basic McIdas support for PIL
#
# History:
# 1997-05-05 fl  Created (8-bit images only)
# 2009-03-08 fl  Added 16/32-bit support.
#
# Thanks to Richard Jones and Craig Swank for specs and samples.
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import struct

from . import Image, ImageFile


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"\x00\x00\x00\x00\x00\x00\x00\x04")


##
# Image plugin for McIdas area images.


class McIdasImageFile(ImageFile.ImageFile):
    format = "MCIDAS"
    format_description = "McIdas area file"

    def _open(self) -> None:
        # parse area file directory
        assert self.fp is not None

        s = self.fp.read(256)
        if not _accept(s) or len(s) != 256:
            msg = "not an McIdas area file"
            raise SyntaxError(msg)

        self.area_descriptor_raw = s
        self.area_descriptor = w = [0, *struct.unpack("!64i", s)]

        # get mode
        if w[11] == 1:
            mode = rawmode = "L"
        elif w[11] == 2:
            mode = rawmode = "I;16B"
        elif w[11] == 4:
            # FIXME: add memory map support
            mode = "I"
            rawmode = "I;32B"
        else:
            msg = "unsupported McIdas format"
            raise SyntaxError(msg)

        self._mode = mode
        self._size = w[10], w[9]

        offset = w[34] + w[15]
        stride = w[15] + w[10] * w[11] * w[14]

        self.tile = [
            ImageFile._Tile("raw", (0, 0) + self.size, offset, (rawmode, stride, 1))
        ]


# --------------------------------------------------------------------
# registry

Image.register_open(McIdasImageFile.format, McIdasImageFile, _accept)

# no default extension
<|endoftext|>#
# The Python Imaging Library.
#
# MSP file handling
#
# This is the format used by the Paint program in Windows 1 and 2.
#
# History:
#       95-09-05 fl     Created
#       97-01-03 fl     Read/write MSP images
#       17-02-21 es     Fixed RLE interpretation
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1995-97.
# Copyright (c) Eric Soroos 2017.
#
# See the README file for information on usage and redistribution.
#
# More info on this format: https://archive.org/details/gg243631
# Page 313:
# Figure 205. Windows Paint Version 1: "DanM" Format
# Figure 206. Windows Paint Version 2: "LinS" Format. Used in Windows V2.03
#
# See also: https://www.fileformat.info/format/mspaint/egff.htm
from __future__ import annotations

import io
import struct
from typing import IO

from . import Image, ImageFile
from ._binary import i16le as i16
from ._binary import o16le as o16

#
# read MSP files


def _accept(prefix: bytes) -> bool:
    return prefix.startswith((b"DanM", b"LinS"))


##
# Image plugin for Windows MSP images.  This plugin supports both
# uncompressed (Windows 1.0).


class MspImageFile(ImageFile.ImageFile):
    format = "MSP"
    format_description = "Windows Paint"

    def _open(self) -> None:
        # Header
        assert self.fp is not None

        s = self.fp.read(32)
        if not _accept(s):
            msg = "not an MSP file"
            raise SyntaxError(msg)

        # Header checksum
        checksum = 0
        for i in range(0, 32, 2):
            checksum = checksum ^ i16(s, i)
        if checksum != 0:
            msg = "bad MSP checksum"
            raise SyntaxError(msg)

        self._mode = "1"
        self._size = i16(s, 4), i16(s, 6)

        if s.startswith(b"DanM"):
            self.tile = [ImageFile._Tile("raw", (0, 0) + self.size, 32, "1")]
        else:
            self.tile = [ImageFile._Tile("MSP", (0, 0) + self.size, 32)]


class MspDecoder(ImageFile.PyDecoder):
    # The algo for the MSP decoder is from
    # https://www.fileformat.info/format/mspaint/egff.htm
    # cc-by-attribution -- That page references is taken from the
    # Encyclopedia of Graphics File Formats and is licensed by
    # O'Reilly under the Creative Common/Attribution license
    #
    # For RLE encoded files, the 32byte header is followed by a scan
    # line map, encoded as one 16bit word of encoded byte length per
    # line.
    #
    # NOTE: the encoded length of the line can be 0. This was not
    # handled in the previous version of this encoder, and there's no
    # mention of how to handle it in the documentation. From the few
    # examples I've seen, I've assumed that it is a fill of the
    # background color, in this case, white.
    #
    #
    # Pseudocode of the decoder:
    # Read a BYTE value as the RunType
    #  If the RunType value is zero
    #   Read next byte as the RunCount
    #   Read the next byte as the RunValue
    #   Write the RunValue byte RunCount times
    #  If the RunType value is non-zero
    #   Use this value as the RunCount
    #   Read and write the next RunCount bytes literally
    #
    #  e.g.:
    #  0x00 03 ff 05 00 01 02 03 04
    #  would yield the bytes:
    #  0xff ff ff 00 01 02 03 04
    #
    # which are then interpreted as a bit packed mode '1' image

    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        assert self.fd is not None

        img = io.BytesIO()
        blank_line = bytearray((0xFF,) * ((self.state.xsize + 7) // 8))
        try:
            self.fd.seek(32)
            rowmap = struct.unpack_from(
                f"<{self.state.ysize}H", self.fd.read(self.state.ysize * 2)
            )
        except struct.error as e:
            msg = "Truncated MSP file in row map"
            raise OSError(msg) from e

        for x, rowlen in enumerate(rowmap):
            try:
                if rowlen == 0:
                    img.write(blank_line)
                    continue
                row = self.fd.read(rowlen)
                if len(row) != rowlen:
                    msg = f"Truncated MSP file, expected {rowlen} bytes on row {x}"
                    raise OSError(msg)
                idx = 0
                while idx < rowlen:
                    runtype = row[idx]
                    idx += 1
                    if runtype == 0:
                        runcount, runval = struct.unpack_from("Bc", row, idx)
                        img.write(runval * runcount)
                        idx += 2
                    else:
                        runcount = runtype
                        img.write(row[idx : idx + runcount])
                        idx += runcount

            except struct.error as e:
                msg = f"Corrupted MSP file in row {x}"
                raise OSError(msg) from e

        self.set_as_raw(img.getvalue(), "1")

        return -1, 0


Image.register_decoder("MSP", MspDecoder)


#
# write MSP files (uncompressed only)


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode != "1":
        msg = f"cannot write mode {im.mode} as MSP"
        raise OSError(msg)

    # create MSP header
    header = [0] * 16

    header[0], header[1] = i16(b"Da"), i16(b"nM")  # version 1
    header[2], header[3] = im.size
    header[4], header[5] = 1, 1
    header[6], header[7] = 1, 1
    header[8], header[9] = im.size

    checksum = 0
    for h in header:
        checksum = checksum ^ h
    header[12] = checksum  # FIXME: is this the right field?

    # header
    for h in header:
        fp.write(o16(h))

    # image body
    ImageFile._save(im, fp, [ImageFile._Tile("raw", (0, 0) + im.size, 32, "1")])


#
# registry

Image.register_open(MspImageFile.format, MspImageFile, _accept)
Image.register_save(MspImageFile.format, _save)

Image.register_extension(MspImageFile.format, ".msp")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# PIXAR raster support for PIL
#
# history:
#       97-01-29 fl     Created
#
# notes:
#       This is incomplete; it is based on a few samples created with
#       Photoshop 2.5 and 3.0, and a summary description provided by
#       Greg Coats <gcoats@labiris.er.usgs.gov>.  Hopefully, "L" and
#       "RGBA" support will be added in future versions.
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import Image, ImageFile
from ._binary import i16le as i16

#
# helpers


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"\200\350\000\000")


##
# Image plugin for PIXAR raster images.


class PixarImageFile(ImageFile.ImageFile):
    format = "PIXAR"
    format_description = "PIXAR raster image"

    def _open(self) -> None:
        # assuming a 4-byte magic label
        assert self.fp is not None

        s = self.fp.read(4)
        if not _accept(s):
            msg = "not a PIXAR file"
            raise SyntaxError(msg)

        # read rest of header
        s = s + self.fp.read(508)

        self._size = i16(s, 418), i16(s, 416)

        # get channel/depth descriptions
        mode = i16(s, 424), i16(s, 426)

        if mode == (14, 2):
            self._mode = "RGB"
        # FIXME: to be continued...

        # create tile descriptor (assuming "dumped")
        self.tile = [ImageFile._Tile("raw", (0, 0) + self.size, 1024, self.mode)]


#
# --------------------------------------------------------------------

Image.register_open(PixarImageFile.format, PixarImageFile, _accept)

Image.register_extension(PixarImageFile.format, ".pxr")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#

##
# Image plugin for Palm pixmap images (output only).
##
from __future__ import annotations

from typing import IO

from . import Image, ImageFile
from ._binary import o8
from ._binary import o16be as o16b

# fmt: off
_Palm8BitColormapValues = (
    (255, 255, 255), (255, 204, 255), (255, 153, 255), (255, 102, 255),
    (255,  51, 255), (255,   0, 255), (255, 255, 204), (255, 204, 204),
    (255, 153, 204), (255, 102, 204), (255,  51, 204), (255,   0, 204),
    (255, 255, 153), (255, 204, 153), (255, 153, 153), (255, 102, 153),
    (255,  51, 153), (255,   0, 153), (204, 255, 255), (204, 204, 255),
    (204, 153, 255), (204, 102, 255), (204,  51, 255), (204,   0, 255),
    (204, 255, 204), (204, 204, 204), (204, 153, 204), (204, 102, 204),
    (204,  51, 204), (204,   0, 204), (204, 255, 153), (204, 204, 153),
    (204, 153, 153), (204, 102, 153), (204,  51, 153), (204,   0, 153),
    (153, 255, 255), (153, 204, 255), (153, 153, 255), (153, 102, 255),
    (153,  51, 255), (153,   0, 255), (153, 255, 204), (153, 204, 204),
    (153, 153, 204), (153, 102, 204), (153,  51, 204), (153,   0, 204),
    (153, 255, 153), (153, 204, 153), (153, 153, 153), (153, 102, 153),
    (153,  51, 153), (153,   0, 153), (102, 255, 255), (102, 204, 255),
    (102, 153, 255), (102, 102, 255), (102,  51, 255), (102,   0, 255),
    (102, 255, 204), (102, 204, 204), (102, 153, 204), (102, 102, 204),
    (102,  51, 204), (102,   0, 204), (102, 255, 153), (102, 204, 153),
    (102, 153, 153), (102, 102, 153), (102,  51, 153), (102,   0, 153),
    (51,  255, 255), (51,  204, 255), (51,  153, 255), (51,  102, 255),
    (51,   51, 255), (51,    0, 255), (51,  255, 204), (51,  204, 204),
    (51,  153, 204), (51,  102, 204), (51,   51, 204), (51,    0, 204),
    (51,  255, 153), (51,  204, 153), (51,  153, 153), (51,  102, 153),
    (51,   51, 153), (51,    0, 153), (0,   255, 255), (0,   204, 255),
    (0,   153, 255), (0,   102, 255), (0,    51, 255), (0,     0, 255),
    (0,   255, 204), (0,   204, 204), (0,   153, 204), (0,   102, 204),
    (0,    51, 204), (0,     0, 204), (0,   255, 153), (0,   204, 153),
    (0,   153, 153), (0,   102, 153), (0,    51, 153), (0,     0, 153),
    (255, 255, 102), (255, 204, 102), (255, 153, 102), (255, 102, 102),
    (255,  51, 102), (255,   0, 102), (255, 255,  51), (255, 204,  51),
    (255, 153,  51), (255, 102,  51), (255,  51,  51), (255,   0,  51),
    (255, 255,   0), (255, 204,   0), (255, 153,   0), (255, 102,   0),
    (255,  51,   0), (255,   0,   0), (204, 255, 102), (204, 204, 102),
    (204, 153, 102), (204, 102, 102), (204,  51, 102), (204,   0, 102),
    (204, 255,  51), (204, 204,  51), (204, 153,  51), (204, 102,  51),
    (204,  51,  51), (204,   0,  51), (204, 255,   0), (204, 204,   0),
    (204, 153,   0), (204, 102,   0), (204,  51,   0), (204,   0,   0),
    (153, 255, 102), (153, 204, 102), (153, 153, 102), (153, 102, 102),
    (153,  51, 102), (153,   0, 102), (153, 255,  51), (153, 204,  51),
    (153, 153,  51), (153, 102,  51), (153,  51,  51), (153,   0,  51),
    (153, 255,   0), (153, 204,   0), (153, 153,   0), (153, 102,   0),
    (153,  51,   0), (153,   0,   0), (102, 255, 102), (102, 204, 102),
    (102, 153, 102), (102, 102, 102), (102,  51, 102), (102,   0, 102),
    (102, 255,  51), (102, 204,  51), (102, 153,  51), (102, 102,  51),
    (102,  51,  51), (102,   0,  51), (102, 255,   0), (102, 204,   0),
    (102, 153,   0), (102, 102,   0), (102,  51,   0), (102,   0,   0),
    (51,  255, 102), (51,  204, 102), (51,  153, 102), (51,  102, 102),
    (51,   51, 102), (51,    0, 102), (51,  255,  51), (51,  204,  51),
    (51,  153,  51), (51,  102,  51), (51,   51,  51), (51,    0,  51),
    (51,  255,   0), (51,  204,   0), (51,  153,   0), (51,  102,   0),
    (51,   51,   0), (51,    0,   0), (0,   255, 102), (0,   204, 102),
    (0,   153, 102), (0,   102, 102), (0,    51, 102), (0,     0, 102),
    (0,   255,  51), (0,   204,  51), (0,   153,  51), (0,   102,  51),
    (0,    51,  51), (0,     0,  51), (0,   255,   0), (0,   204,   0),
    (0,   153,   0), (0,   102,   0), (0,    51,   0), (17,   17,  17),
    (34,   34,  34), (68,   68,  68), (85,   85,  85), (119, 119, 119),
    (136, 136, 136), (170, 170, 170), (187, 187, 187), (221, 221, 221),
    (238, 238, 238), (192, 192, 192), (128,   0,   0), (128,   0, 128),
    (0,   128,   0), (0,   128, 128), (0,     0,   0), (0,     0,   0),
    (0,     0,   0), (0,     0,   0), (0,     0,   0), (0,     0,   0),
    (0,     0,   0), (0,     0,   0), (0,     0,   0), (0,     0,   0),
    (0,     0,   0), (0,     0,   0), (0,     0,   0), (0,     0,   0),
    (0,     0,   0), (0,     0,   0), (0,     0,   0), (0,     0,   0),
    (0,     0,   0), (0,     0,   0), (0,     0,   0), (0,     0,   0),
    (0,     0,   0), (0,     0,   0), (0,     0,   0), (0,     0,   0))
# fmt: on


# so build a prototype image to be used for palette resampling
def build_prototype_image() -> Image.Image:
    image = Image.new("L", (1, len(_Palm8BitColormapValues)))
    image.putdata(list(range(len(_Palm8BitColormapValues))))
    palettedata: tuple[int, ...] = ()
    for colormapValue in _Palm8BitColormapValues:
        palettedata += colormapValue
    palettedata += (0, 0, 0) * (256 - len(_Palm8BitColormapValues))
    image.putpalette(palettedata)
    return image


Palm8BitColormapImage = build_prototype_image()

# OK, we now have in Palm8BitColormapImage,
# a "P"-mode image with the right palette
#
# --------------------------------------------------------------------

_FLAGS = {"custom-colormap": 0x4000, "is-compressed": 0x8000, "has-transparent": 0x2000}

_COMPRESSION_TYPES = {"none": 0xFF, "rle": 0x01, "scanline": 0x00}


#
# --------------------------------------------------------------------

##
# (Internal) Image save plugin for the Palm format.


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode == "P":
        rawmode = "P"
        bpp = 8
        version = 1

    elif im.mode == "L":
        if im.encoderinfo.get("bpp") in (1, 2, 4):
            # this is 8-bit grayscale, so we shift it to get the high-order bits,
            # and invert it because
            # Palm does grayscale from white (0) to black (1)
            bpp = im.encoderinfo["bpp"]
            maxval = (1 << bpp) - 1
            shift = 8 - bpp
            im = im.point(lambda x: maxval - (x >> shift))
        elif im.info.get("bpp") in (1, 2, 4):
            # here we assume that even though the inherent mode is 8-bit grayscale,
            # only the lower bpp bits are significant.
            # We invert them to match the Palm.
            bpp = im.info["bpp"]
            maxval = (1 << bpp) - 1
            im = im.point(lambda x: maxval - (x & maxval))
        else:
            msg = f"cannot write mode {im.mode} as Palm"
            raise OSError(msg)

        # we ignore the palette here
        im._mode = "P"
        rawmode = f"P;{bpp}"
        version = 1

    elif im.mode == "1":
        # monochrome -- write it inverted, as is the Palm standard
        rawmode = "1;I"
        bpp = 1
        version = 0

    else:
        msg = f"cannot write mode {im.mode} as Palm"
        raise OSError(msg)

    #
    # make sure image data is available
    im.load()

    # write header

    cols = im.size[0]
    rows = im.size[1]

    rowbytes = int((cols + (16 // bpp - 1)) / (16 // bpp)) * 2
    transparent_index = 0
    compression_type = _COMPRESSION_TYPES["none"]

    flags = 0
    if im.mode == "P":
        flags |= _FLAGS["custom-colormap"]
        colormap = im.im.getpalette()
        colors = len(colormap) // 3
        colormapsize = 4 * colors + 2
    else:
        colormapsize = 0

    if "offset" in im.info:
        offset = (rowbytes * rows + 16 + 3 + colormapsize) // 4
    else:
        offset = 0

    fp.write(o16b(cols) + o16b(rows) + o16b(rowbytes) + o16b(flags))
    fp.write(o8(bpp))
    fp.write(o8(version))
    fp.write(o16b(offset))
    fp.write(o8(transparent_index))
    fp.write(o8(compression_type))
    fp.write(o16b(0))  # reserved by Palm

    # now write colormap if necessary

    if colormapsize:
        fp.write(o16b(colors))
        for i in range(colors):
            fp.write(o8(i))
            fp.write(colormap[3 * i : 3 * i + 3])

    # now convert data to raw form
    ImageFile._save(
        im, fp, [ImageFile._Tile("raw", (0, 0) + im.size, 0, (rawmode, rowbytes, 1))]
    )

    if hasattr(fp, "flush"):
        fp.flush()


#
# --------------------------------------------------------------------

Image.register_save("PALM", _save)

Image.register_extension("PALM", ".palm")

Image.register_mime("PALM", "image/palm")
<|endoftext|>from __future__ import annotations

import math
import re
from typing import AnyStr, Generic, NamedTuple

from . import ImageFont
from ._typing import _Ink


class _Line(NamedTuple):
    x: float
    y: float
    anchor: str
    text: str | bytes


class _Wrap(Generic[AnyStr]):
    lines: list[AnyStr] = []
    position = 0
    offset = 0

    def __init__(
        self,
        text: Text[AnyStr],
        width: int,
        height: int | None = None,
        font: ImageFont.BaseImageFont | None = None,
    ) -> None:
        self.text: Text[AnyStr] = text
        self.width = width
        self.height = height
        self.font = font

        input_text = self.text.text
        emptystring = "" if isinstance(input_text, str) else b""
        line = emptystring

        for word in re.findall(
            r"\s*\S+" if isinstance(input_text, str) else rb"\s*\S+", input_text
        ):
            newlines = re.findall(
                r"[^\S\n]*\n" if isinstance(input_text, str) else rb"[^\S\n]*\n", word
            )
            if newlines:
                if not self.add_line(line):
                    break
                for i, line in enumerate(newlines):
                    if i != 0 and not self.add_line(emptystring):
                        break
                    self.position += len(line)
                    word = word[len(line) :]
                line = emptystring

            new_line = line + word
            if self.text._get_bbox(new_line, self.font)[2] <= width:
                # This word fits on the line
                line = new_line
                continue

            # This word does not fit on the line
            if line and not self.add_line(line):
                break

            original_length = len(word)
            word = word.lstrip()
            self.offset = original_length - len(word)

            if self.text._get_bbox(word, self.font)[2] > width:
                if font is None:
                    msg = "Word does not fit within line"
                    raise ValueError(msg)
                break
            line = word
        else:
            if line:
                self.add_line(line)
        self.remaining_text: AnyStr = input_text[self.position :]

    def add_line(self, line: AnyStr) -> bool:
        lines = self.lines + [line]
        if self.height is not None:
            last_line_y = self.text._split(lines=lines)[-1].y
            last_line_height = self.text._get_bbox(line, self.font)[3]
            if last_line_y + last_line_height > self.height:
                return False

        self.lines = lines
        self.position += len(line) + self.offset
        self.offset = 0
        return True


class Text(Generic[AnyStr]):
    def __init__(
        self,
        text: AnyStr,
        font: ImageFont.BaseImageFont | None = None,
        mode: str = "RGB",
        spacing: float = 4,
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
    ) -> None:
        """
        :param text: String to be drawn.
        :param font: Either an :py:class:`~PIL.ImageFont.ImageFont` instance,
                     :py:class:`~PIL.ImageFont.FreeTypeFont` instance,
                     :py:class:`~PIL.ImageFont.TransposedFont` instance or ``None``. If
                     ``None``, the default font from :py:meth:`.ImageFont.load_default`
                     will be used.
        :param mode: The image mode this will be used with.
        :param spacing: The number of pixels between lines.
        :param direction: Direction of the text. It can be ``"rtl"`` (right to left),
                          ``"ltr"`` (left to right) or ``"ttb"`` (top to bottom).
                          Requires libraqm.
        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional font features
                         that are not enabled by default, for example ``"dlig"`` or
                         ``"ss01"``, but can be also used to turn off default font
                         features, for example ``"-liga"`` to disable ligatures or
                         ``"-kern"`` to disable kerning.  To get all supported
                         features, see `OpenType docs`_.
                         Requires libraqm.
        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code`_.
                         Requires libraqm.
        """
        self.text: AnyStr = text
        self.font = font or ImageFont.load_default()

        self.mode = mode
        self.spacing = spacing
        self.direction = direction
        self.features = features
        self.language = language

        self.embedded_color = False

        self.stroke_width: float = 0
        self.stroke_fill: _Ink | None = None

    def embed_color(self) -> None:
        """
        Use embedded color glyphs (COLR, CBDT, SBIX).
        """
        if self.mode not in ("RGB", "RGBA"):
            msg = "Embedded color supported only in RGB and RGBA modes"
            raise ValueError(msg)
        self.embedded_color = True

    def stroke(self, width: float = 0, fill: _Ink | None = None) -> None:
        """
        :param width: The width of the text stroke.
        :param fill: Color to use for the text stroke when drawing. If not given, will
                     default to the ``fill`` parameter from
                     :py:meth:`.ImageDraw.ImageDraw.text`.
        """
        self.stroke_width = width
        self.stroke_fill = fill

    def _get_fontmode(self) -> str:
        if self.mode in ("1", "P", "I", "F"):
            return "1"
        elif self.embedded_color:
            return "RGBA"
        else:
            return "L"

    def wrap(
        self,
        width: int,
        height: int | None = None,
        scaling: str | tuple[str, int] | None = None,
    ) -> Text[AnyStr] | None:
        """
        Wrap text to fit within a given width.

        :param width: The width to fit within.
        :param height: An optional height limit. Any text that does not fit within this
                       will be returned as a new :py:class:`.Text` object.
        :param scaling: An optional directive to scale the text, either "grow" as much
                        as possible within the given dimensions, or "shrink" until it
                        fits. It can also be a tuple of (direction, limit), with an
                        integer limit to stop scaling at.

        :returns: An :py:class:`.Text` object, or None.
        """
        if isinstance(self.font, ImageFont.TransposedFont):
            msg = "TransposedFont not supported"
            raise ValueError(msg)
        if self.direction not in (None, "ltr"):
            msg = "Only ltr direction supported"
            raise ValueError(msg)

        if scaling is None:
            wrap = _Wrap(self, width, height)
        else:
            if not isinstance(self.font, ImageFont.FreeTypeFont):
                msg = "'scaling' only supports FreeTypeFont"
                raise ValueError(msg)
            if height is None:
                msg = "'scaling' requires 'height'"
                raise ValueError(msg)

            if isinstance(scaling, str):
                limit = 1
            else:
                scaling, limit = scaling

            font = self.font
            wrap = _Wrap(self, width, height, font)
            if scaling == "shrink":
                if not wrap.remaining_text:
                    return None

                size = math.ceil(font.size)
                while wrap.remaining_text:
                    if size == max(limit, 1):
                        msg = "Text could not be scaled"
                        raise ValueError(msg)
                    size -= 1
                    font = self.font.font_variant(size=size)
                    wrap = _Wrap(self, width, height, font)
                self.font = font
            else:
                if wrap.remaining_text:
                    msg = "Text could not be scaled"
                    raise ValueError(msg)

                size = math.floor(font.size)
                while not wrap.remaining_text:
                    if size == limit:
                        msg = "Text could not be scaled"
                        raise ValueError(msg)
                    size += 1
                    font = self.font.font_variant(size=size)
                    last_wrap = wrap
                    wrap = _Wrap(self, width, height, font)
                size -= 1
                if size != self.font.size:
                    self.font = self.font.font_variant(size=size)
                wrap = last_wrap

        if wrap.remaining_text:
            text = Text(
                text=wrap.remaining_text,
                font=self.font,
                mode=self.mode,
                spacing=self.spacing,
                direction=self.direction,
                features=self.features,
                language=self.language,
            )
            text.embedded_color = self.embedded_color
            text.stroke_width = self.stroke_width
            text.stroke_fill = self.stroke_fill
        else:
            text = None

        newline = "\n" if isinstance(self.text, str) else b"\n"
        self.text = newline.join(wrap.lines)
        return text

    def get_length(self) -> float:
        """
        Returns length (in pixels with 1/64 precision) of text.

        This is the amount by which following text should be offset.
        Text bounding box may extend past the length in some fonts,
        e.g. when using italics or accents.

        The result is returned as a float; it is a whole number if using basic layout.

        Note that the sum of two lengths may not equal the length of a concatenated
        string due to kerning. If you need to adjust for kerning, include the following
        character and subtract its length.

        For example, instead of::

            hello = ImageText.Text("Hello", font).get_length()
            world = ImageText.Text("World", font).get_length()
            helloworld = ImageText.Text("HelloWorld", font).get_length()
            assert hello + world == helloworld

        use::

            hello = (
                ImageText.Text("HelloW", font).get_length() -
                ImageText.Text("W", font).get_length()
            )  # adjusted for kerning
            world = ImageText.Text("World", font).get_length()
            helloworld = ImageText.Text("HelloWorld", font).get_length()
            assert hello + world == helloworld

        or disable kerning with (requires libraqm)::

            hello = ImageText.Text("Hello", font, features=["-kern"]).get_length()
            world = ImageText.Text("World", font, features=["-kern"]).get_length()
            helloworld = ImageText.Text(
                "HelloWorld", font, features=["-kern"]
            ).get_length()
            assert hello + world == helloworld

        :return: Either width for horizontal text, or height for vertical text.
        """
        if isinstance(self.text, str):
            multiline = "\n" in self.text
        else:
            multiline = b"\n" in self.text
        if multiline:
            msg = "can't measure length of multiline text"
            raise ValueError(msg)
        return self.font.getlength(
            self.text,
            self._get_fontmode(),
            self.direction,
            self.features,
            self.language,
        )

    def _split(
        self,
        xy: tuple[float, float] = (0, 0),
        anchor: str | None = None,
        align: str = "left",
        lines: list[str] | list[bytes] | None = None,
    ) -> list[_Line]:
        if anchor is None:
            anchor = "lt" if self.direction == "ttb" else "la"
        elif len(anchor) != 2:
            msg = "anchor must be a 2 character string"
            raise ValueError(msg)

        if lines is None:
            lines = (
                self.text.split("\n")
                if isinstance(self.text, str)
                else self.text.split(b"\n")
            )
        if len(lines) == 1:
            return [_Line(xy[0], xy[1], anchor, lines[0])]

        if anchor[1] in "tb" and self.direction != "ttb":
            msg = "anchor not supported for multiline text"
            raise ValueError(msg)

        fontmode = self._get_fontmode()
        line_spacing = (
            self.font.getbbox(
                "A",
                fontmode,
                None,
                self.features,
                self.language,
                self.stroke_width,
            )[3]
            + self.stroke_width
            + self.spacing
        )

        top = xy[1]
        parts = []
        if self.direction == "ttb":
            left = xy[0]
            for line in lines:
                parts.append(_Line(left, top, anchor, line))
                left += line_spacing
        else:
            widths = []
            max_width: float = 0
            for line in lines:
                line_width = self.font.getlength(
                    line, fontmode, self.direction, self.features, self.language
                )
                widths.append(line_width)
                max_width = max(max_width, line_width)

            if anchor[1] == "m":
                top -= (len(lines) - 1) * line_spacing / 2.0
            elif anchor[1] == "d":
                top -= (len(lines) - 1) * line_spacing

            idx = -1
            for line in lines:
                left = xy[0]
                idx += 1
                width_difference = max_width - widths[idx]

                # align by align parameter
                if align in ("left", "justify"):
                    pass
                elif align == "center":
                    left += width_difference / 2.0
                elif align == "right":
                    left += width_difference
                else:
                    msg = 'align must be "left", "center", "right" or "justify"'
                    raise ValueError(msg)

                if (
                    align == "justify"
                    and width_difference != 0
                    and idx != len(lines) - 1
                ):
                    words = (
                        line.split(" ") if isinstance(line, str) else line.split(b" ")
                    )
                    if len(words) > 1:
                        # align left by anchor
                        if anchor[0] == "m":
                            left -= max_width / 2.0
                        elif anchor[0] == "r":
                            left -= max_width

                        word_widths = [
                            self.font.getlength(
                                word,
                                fontmode,
                                self.direction,
                                self.features,
                                self.language,
                            )
                            for word in words
                        ]
                        word_anchor = "l" + anchor[1]
                        width_difference = max_width - sum(word_widths)
                        i = 0
                        for word in words:
                            parts.append(_Line(left, top, word_anchor, word))
                            left += word_widths[i] + width_difference / (len(words) - 1)
                            i += 1
                        top += line_spacing
                        continue

                # align left by anchor
                if anchor[0] == "m":
                    left -= width_difference / 2.0
                elif anchor[0] == "r":
                    left -= width_difference
                parts.append(_Line(left, top, anchor, line))
                top += line_spacing

        return parts

    def _get_bbox(
        self,
        text: str | bytes,
        font: ImageFont.BaseImageFont | None = None,
        anchor: str | None = None,
    ) -> tuple[float, float, float, float]:
        return (font or self.font).getbbox(
            text,
            self._get_fontmode(),
            self.direction,
            self.features,
            self.language,
            self.stroke_width,
            anchor,
        )

    def get_bbox(
        self,
        xy: tuple[float, float] = (0, 0),
        anchor: str | None = None,
        align: str = "left",
    ) -> tuple[float, float, float, float]:
        """
        Returns bounding box (in pixels) of text.

        Use :py:meth:`get_length` to get the offset of following text with 1/64 pixel
        precision. The bounding box includes extra margins for some fonts, e.g. italics
        or accents.

        :param xy: The anchor coordinates of the text.
        :param anchor: The text anchor alignment. Determines the relative location of
                       the anchor to the text. The default alignment is top left,
                       specifically ``la`` for horizontal text and ``lt`` for
                       vertical text. See :ref:`text-anchors` for details.
        :param align: For multiline text, ``"left"``, ``"center"``, ``"right"`` or
                      ``"justify"`` determines the relative alignment of lines. Use the
                      ``anchor`` parameter to specify the alignment to ``xy``.

        :return: ``(left, top, right, bottom)`` bounding box
        """
        bbox: tuple[float, float, float, float] | None = None
        for x, y, anchor, text in self._split(xy, anchor, align):
            bbox_line = self._get_bbox(text, anchor=anchor)
            bbox_line = (
                bbox_line[0] + x,
                bbox_line[1] + y,
                bbox_line[2] + x,
                bbox_line[3] + y,
            )
            if bbox is None:
                bbox = bbox_line
            else:
                bbox = (
                    min(bbox[0], bbox_line[0]),
                    min(bbox[1], bbox_line[1]),
                    max(bbox[2], bbox_line[2]),
                    max(bbox[3], bbox_line[3]),
                )

        assert bbox is not None
        return bbox
<|endoftext|># The Python Imaging Library.
# $Id$

# Optional color management support, based on Kevin Cazabon's PyCMS
# library.

# Originally released under LGPL.  Graciously donated to PIL in
# March 2009, for distribution under the standard PIL license

# History:

# 2009-03-08 fl   Added to PIL.

# Copyright (C) 2002-2003 Kevin Cazabon
# Copyright (c) 2009 by Fredrik Lundh
# Copyright (c) 2013 by Eric Soroos

# See the README file for information on usage and redistribution.  See
# below for the original description.
from __future__ import annotations

import operator
import sys
from enum import IntEnum, IntFlag
from functools import reduce
from typing import Any, Literal, SupportsFloat, SupportsInt, Union

from . import Image
from ._deprecate import deprecate
from ._typing import SupportsRead

try:
    from . import _imagingcms as core

    _CmsProfileCompatible = Union[
        str, SupportsRead[bytes], core.CmsProfile, "ImageCmsProfile"
    ]
except ImportError as ex:
    # Allow error import for doc purposes, but error out when accessing
    # anything in core.
    from ._util import DeferredError

    core = DeferredError.new(ex)

_DESCRIPTION = """
pyCMS

    a Python / PIL interface to the littleCMS ICC Color Management System
    Copyright (C) 2002-2003 Kevin Cazabon
    kevin@cazabon.com
    https://www.cazabon.com

    pyCMS home page:  https://www.cazabon.com/pyCMS
    littleCMS home page:  https://www.littlecms.com
    (littleCMS is Copyright (C) 1998-2001 Marti Maria)

    Originally released under LGPL.  Graciously donated to PIL in
    March 2009, for distribution under the standard PIL license

    The pyCMS.py module provides a "clean" interface between Python/PIL and
    pyCMSdll, taking care of some of the more complex handling of the direct
    pyCMSdll functions, as well as error-checking and making sure that all
    relevant data is kept together.

    While it is possible to call pyCMSdll functions directly, it's not highly
    recommended.

    Version History:

        1.0.0 pil       Oct 2013 Port to LCMS 2.

        0.1.0 pil mod   March 10, 2009

                        Renamed display profile to proof profile. The proof
                        profile is the profile of the device that is being
                        simulated, not the profile of the device which is
                        actually used to display/print the final simulation
                        (that'd be the output profile) - also see LCMSAPI.txt
                        input colorspace -> using 'renderingIntent' -> proof
                        colorspace -> using 'proofRenderingIntent' -> output
                        colorspace

                        Added LCMS FLAGS support.
                        Added FLAGS["SOFTPROOFING"] as default flag for
                        buildProofTransform (otherwise the proof profile/intent
                        would be ignored).

        0.1.0 pil       March 2009 - added to PIL, as PIL.ImageCms

        0.0.2 alpha     Jan 6, 2002

                        Added try/except statements around type() checks of
                        potential CObjects... Python won't let you use type()
                        on them, and raises a TypeError (stupid, if you ask
                        me!)

                        Added buildProofTransformFromOpenProfiles() function.
                        Additional fixes in DLL, see DLL code for details.

        0.0.1 alpha     first public release, Dec. 26, 2002

    Known to-do list with current version (of Python interface, not pyCMSdll):

        none

"""

_VERSION = "1.0.0 pil"


# --------------------------------------------------------------------.


#
# intent/direction values


class Intent(IntEnum):
    PERCEPTUAL = 0
    RELATIVE_COLORIMETRIC = 1
    SATURATION = 2
    ABSOLUTE_COLORIMETRIC = 3


class Direction(IntEnum):
    INPUT = 0
    OUTPUT = 1
    PROOF = 2


#
# flags


class Flags(IntFlag):
    """Flags and documentation are taken from ``lcms2.h``."""

    NONE = 0
    NOCACHE = 0x0040
    """Inhibit 1-pixel cache"""
    NOOPTIMIZE = 0x0100
    """Inhibit optimizations"""
    NULLTRANSFORM = 0x0200
    """Don't transform anyway"""
    GAMUTCHECK = 0x1000
    """Out of Gamut alarm"""
    SOFTPROOFING = 0x4000
    """Do softproofing"""
    BLACKPOINTCOMPENSATION = 0x2000
    NOWHITEONWHITEFIXUP = 0x0004
    """Don't fix scum dot"""
    HIGHRESPRECALC = 0x0400
    """Use more memory to give better accuracy"""
    LOWRESPRECALC = 0x0800
    """Use less memory to minimize resources"""
    # this should be 8BITS_DEVICELINK, but that is not a valid name in Python:
    USE_8BITS_DEVICELINK = 0x0008
    """Create 8 bits devicelinks"""
    GUESSDEVICECLASS = 0x0020
    """Guess device class (for ``transform2devicelink``)"""
    KEEP_SEQUENCE = 0x0080
    """Keep profile sequence for devicelink creation"""
    FORCE_CLUT = 0x0002
    """Force CLUT optimization"""
    CLUT_POST_LINEARIZATION = 0x0001
    """create postlinearization tables if possible"""
    CLUT_PRE_LINEARIZATION = 0x0010
    """create prelinearization tables if possible"""
    NONEGATIVES = 0x8000
    """Prevent negative numbers in floating point transforms"""
    COPY_ALPHA = 0x04000000
    """Alpha channels are copied on ``cmsDoTransform()``"""
    NODEFAULTRESOURCEDEF = 0x01000000

    _GRIDPOINTS_1 = 1 << 16
    _GRIDPOINTS_2 = 2 << 16
    _GRIDPOINTS_4 = 4 << 16
    _GRIDPOINTS_8 = 8 << 16
    _GRIDPOINTS_16 = 16 << 16
    _GRIDPOINTS_32 = 32 << 16
    _GRIDPOINTS_64 = 64 << 16
    _GRIDPOINTS_128 = 128 << 16

    @staticmethod
    def GRIDPOINTS(n: int) -> Flags:
        """
        Fine-tune control over number of gridpoints

        :param n: :py:class:`int` in range ``0 <= n <= 255``
        """
        return Flags.NONE | ((n & 0xFF) << 16)


_MAX_FLAG = reduce(operator.or_, Flags)


_FLAGS = {
    "MATRIXINPUT": 1,
    "MATRIXOUTPUT": 2,
    "MATRIXONLY": (1 | 2),
    "NOWHITEONWHITEFIXUP": 4,  # Don't hot fix scum dot
    # Don't create prelinearization tables on precalculated transforms
    # (internal use):
    "NOPRELINEARIZATION": 16,
    "GUESSDEVICECLASS": 32,  # Guess device class (for transform2devicelink)
    "NOTCACHE": 64,  # Inhibit 1-pixel cache
    "NOTPRECALC": 256,
    "NULLTRANSFORM": 512,  # Don't transform anyway
    "HIGHRESPRECALC": 1024,  # Use more memory to give better accuracy
    "LOWRESPRECALC": 2048,  # Use less memory to minimize resources
    "WHITEBLACKCOMPENSATION": 8192,
    "BLACKPOINTCOMPENSATION": 8192,
    "GAMUTCHECK": 4096,  # Out of Gamut alarm
    "SOFTPROOFING": 16384,  # Do softproofing
    "PRESERVEBLACK": 32768,  # Black preservation
    "NODEFAULTRESOURCEDEF": 16777216,  # CRD special
    "GRIDPOINTS": lambda n: (n & 0xFF) << 16,  # Gridpoints
}


# --------------------------------------------------------------------.
# Experimental PIL-level API
# --------------------------------------------------------------------.

##
# Profile.


class ImageCmsProfile:
    def __init__(self, profile: str | SupportsRead[bytes] | core.CmsProfile) -> None:
        """
        :param profile: Either a string representing a filename,
            a file like object containing a profile or a
            low-level profile object

        """
        self.filename: str | None = None

        if isinstance(profile, str):
            if sys.platform == "win32":
                profile_bytes_path = profile.encode()
                try:
                    profile_bytes_path.decode("ascii")
                except UnicodeDecodeError:
                    with open(profile, "rb") as f:
                        self.profile = core.profile_frombytes(f.read())
                    return
            self.filename = profile
            self.profile = core.profile_open(profile)
        elif hasattr(profile, "read"):
            self.profile = core.profile_frombytes(profile.read())
        elif isinstance(profile, core.CmsProfile):
            self.profile = profile
        else:
            msg = "Invalid type for Profile"  # type: ignore[unreachable]
            raise TypeError(msg)

    def __getattr__(self, name: str) -> Any:
        if name in ("product_name", "product_info"):
            deprecate(f"ImageCms.ImageCmsProfile.{name}", 13)
            return None
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    def tobytes(self) -> bytes:
        """
        Returns the profile in a format suitable for embedding in
        saved images.

        :returns: a bytes object containing the ICC profile.
        """

        return core.profile_tobytes(self.profile)


class ImageCmsTransform(Image.ImagePointHandler):
    """
    Transform.  This can be used with the procedural API, or with the standard
    :py:func:`~PIL.Image.Image.point` method.

    Will return the output profile in the ``output.info['icc_profile']``.
    """

    def __init__(
        self,
        input: ImageCmsProfile,
        output: ImageCmsProfile,
        input_mode: str,
        output_mode: str,
        intent: Intent = Intent.PERCEPTUAL,
        proof: ImageCmsProfile | None = None,
        proof_intent: Intent = Intent.ABSOLUTE_COLORIMETRIC,
        flags: Flags = Flags.NONE,
    ):
        if proof is None:
            self.transform = core.buildTransform(
                input.profile, output.profile, input_mode, output_mode, intent, flags
            )
        else:
            self.transform = core.buildProofTransform(
                input.profile,
                output.profile,
                proof.profile,
                input_mode,
                output_mode,
                intent,
                proof_intent,
                flags,
            )
        # Note: inputMode and outputMode are for pyCMS compatibility only
        self.input_mode = self.inputMode = input_mode
        self.output_mode = self.outputMode = output_mode

        self.output_profile = output

    def point(self, im: Image.Image) -> Image.Image:
        return self.apply(im)

    def apply(self, im: Image.Image, imOut: Image.Image | None = None) -> Image.Image:
        if im.mode != self.input_mode:
            msg = "mode mismatch"
            raise ValueError(msg)
        if imOut is not None:
            if imOut.mode != self.output_mode:
                msg = "mode mismatch"
                raise ValueError(msg)
        else:
            imOut = Image.new(self.output_mode, im.size, None)
        self.transform.apply(im.getim(), imOut.getim())
        imOut.info["icc_profile"] = self.output_profile.tobytes()
        return imOut

    def apply_in_place(self, im: Image.Image) -> Image.Image:
        return self.apply(im, im)


def get_display_profile(handle: SupportsInt | None = None) -> ImageCmsProfile | None:
    """
    (experimental) Fetches the profile for the current display device.

    :returns: ``None`` if the profile is not known.
    """

    if sys.platform != "win32":
        return None

    from . import ImageWin  # type: ignore[unused-ignore, unreachable]

    if isinstance(handle, ImageWin.HDC):
        profile = core.get_display_profile_win32(int(handle), 1)
    else:
        profile = core.get_display_profile_win32(int(handle or 0))
    if profile is None:
        return None
    return ImageCmsProfile(profile)


# --------------------------------------------------------------------.
# pyCMS compatible layer
# --------------------------------------------------------------------.


class PyCMSError(Exception):
    """(pyCMS) Exception class.
    This is used for all errors in the pyCMS API."""

    pass


def profileToProfile(
    im: Image.Image,
    inputProfile: _CmsProfileCompatible,
    outputProfile: _CmsProfileCompatible,
    renderingIntent: Intent = Intent.PERCEPTUAL,
    outputMode: str | None = None,
    inPlace: bool = False,
    flags: Flags = Flags.NONE,
) -> Image.Image | None:
    """
    (pyCMS) Applies an ICC transformation to a given image, mapping from
    ``inputProfile`` to ``outputProfile``.

    If the input or output profiles specified are not valid filenames, a
    :exc:`PyCMSError` will be raised.  If ``inPlace`` is ``True`` and
    ``outputMode != im.mode``, a :exc:`PyCMSError` will be raised.
    If an error occurs during application of the profiles,
    a :exc:`PyCMSError` will be raised.
    If ``outputMode`` is not a mode supported by the ``outputProfile`` (or by pyCMS),
    a :exc:`PyCMSError` will be raised.

    This function applies an ICC transformation to im from ``inputProfile``'s
    color space to ``outputProfile``'s color space using the specified rendering
    intent to decide how to handle out-of-gamut colors.

    ``outputMode`` can be used to specify that a color mode conversion is to
    be done using these profiles, but the specified profiles must be able
    to handle that mode.  I.e., if converting im from RGB to CMYK using
    profiles, the input profile must handle RGB data, and the output
    profile must handle CMYK data.

    :param im: An open :py:class:`~PIL.Image.Image` object (i.e. Image.new(...)
        or Image.open(...), etc.)
    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this image, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        profile you wish to use for this image, or a profile object
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param outputMode: A valid PIL mode for the output image (i.e. "RGB",
        "CMYK", etc.).  Note: if rendering the image "inPlace", outputMode
        MUST be the same mode as the input, or omitted completely.  If
        omitted, the outputMode will be the same as the mode of the input
        image (im.mode)
    :param inPlace: Boolean.  If ``True``, the original image is modified in-place,
        and ``None`` is returned.  If ``False`` (default), a new
        :py:class:`~PIL.Image.Image` object is returned with the transform applied.
    :param flags: Integer (0-...) specifying additional flags
    :returns: Either None or a new :py:class:`~PIL.Image.Image` object, depending on
        the value of ``inPlace``
    :exception PyCMSError:
    """

    if outputMode is None:
        outputMode = im.mode

    if not isinstance(renderingIntent, int) or not (0 <= renderingIntent <= 3):
        msg = "renderingIntent must be an integer between 0 and 3"
        raise PyCMSError(msg)

    if not isinstance(flags, int) or not (0 <= flags <= _MAX_FLAG):
        msg = f"flags must be an integer between 0 and {_MAX_FLAG}"
        raise PyCMSError(msg)

    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        transform = ImageCmsTransform(
            inputProfile,
            outputProfile,
            im.mode,
            outputMode,
            renderingIntent,
            flags=flags,
        )
        if inPlace:
            transform.apply_in_place(im)
            imOut = None
        else:
            imOut = transform.apply(im)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v

    return imOut


def getOpenProfile(
    profileFilename: str | SupportsRead[bytes] | core.CmsProfile,
) -> ImageCmsProfile:
    """
    (pyCMS) Opens an ICC profile file.

    The PyCMSProfile object can be passed back into pyCMS for use in creating
    transforms and such (as in ImageCms.buildTransformFromOpenProfiles()).

    If ``profileFilename`` is not a valid filename for an ICC profile,
    a :exc:`PyCMSError` will be raised.

    :param profileFilename: String, as a valid filename path to the ICC profile
        you wish to open, or a file-like object.
    :returns: A CmsProfile class object.
    :exception PyCMSError:
    """

    try:
        return ImageCmsProfile(profileFilename)
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def buildTransform(
    inputProfile: _CmsProfileCompatible,
    outputProfile: _CmsProfileCompatible,
    inMode: str,
    outMode: str,
    renderingIntent: Intent = Intent.PERCEPTUAL,
    flags: Flags = Flags.NONE,
) -> ImageCmsTransform:
    """
    (pyCMS) Builds an ICC transform mapping from the ``inputProfile`` to the
    ``outputProfile``. Use applyTransform to apply the transform to a given
    image.

    If the input or output profiles specified are not valid filenames, a
    :exc:`PyCMSError` will be raised. If an error occurs during creation
    of the transform, a :exc:`PyCMSError` will be raised.

    If ``inMode`` or ``outMode`` are not a mode supported by the ``outputProfile``
    (or by pyCMS), a :exc:`PyCMSError` will be raised.

    This function builds and returns an ICC transform from the ``inputProfile``
    to the ``outputProfile`` using the ``renderingIntent`` to determine what to do
    with out-of-gamut colors.  It will ONLY work for converting images that
    are in ``inMode`` to images that are in ``outMode`` color format (PIL mode,
    i.e. "RGB", "RGBA", "CMYK", etc.).

    Building the transform is a fair part of the overhead in
    ImageCms.profileToProfile(), so if you're planning on converting multiple
    images using the same input/output settings, this can save you time.
    Once you have a transform object, it can be used with
    ImageCms.applyProfile() to convert images without the need to re-compute
    the lookup table for the transform.

    The reason pyCMS returns a class object rather than a handle directly
    to the transform is that it needs to keep track of the PIL input/output
    modes that the transform is meant for.  These attributes are stored in
    the ``inMode`` and ``outMode`` attributes of the object (which can be
    manually overridden if you really want to, but I don't know of any
    time that would be of use, or would even work).

    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this transform, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        profile you wish to use for this transform, or a profile object
    :param inMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param outMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param flags: Integer (0-...) specifying additional flags
    :returns: A CmsTransform class object.
    :exception PyCMSError:
    """

    if not isinstance(renderingIntent, int) or not (0 <= renderingIntent <= 3):
        msg = "renderingIntent must be an integer between 0 and 3"
        raise PyCMSError(msg)

    if not isinstance(flags, int) or not (0 <= flags <= _MAX_FLAG):
        msg = f"flags must be an integer between 0 and {_MAX_FLAG}"
        raise PyCMSError(msg)

    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        return ImageCmsTransform(
            inputProfile, outputProfile, inMode, outMode, renderingIntent, flags=flags
        )
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def buildProofTransform(
    inputProfile: _CmsProfileCompatible,
    outputProfile: _CmsProfileCompatible,
    proofProfile: _CmsProfileCompatible,
    inMode: str,
    outMode: str,
    renderingIntent: Intent = Intent.PERCEPTUAL,
    proofRenderingIntent: Intent = Intent.ABSOLUTE_COLORIMETRIC,
    flags: Flags = Flags.SOFTPROOFING,
) -> ImageCmsTransform:
    """
    (pyCMS) Builds an ICC transform mapping from the ``inputProfile`` to the
    ``outputProfile``, but tries to simulate the result that would be
    obtained on the ``proofProfile`` device.

    If the input, output, or proof profiles specified are not valid
    filenames, a :exc:`PyCMSError` will be raised.

    If an error occurs during creation of the transform,
    a :exc:`PyCMSError` will be raised.

    If ``inMode`` or ``outMode`` are not a mode supported by the ``outputProfile``
    (or by pyCMS), a :exc:`PyCMSError` will be raised.

    This function builds and returns an ICC transform from the ``inputProfile``
    to the ``outputProfile``, but tries to simulate the result that would be
    obtained on the ``proofProfile`` device using ``renderingIntent`` and
    ``proofRenderingIntent`` to determine what to do with out-of-gamut
    colors.  This is known as "soft-proofing".  It will ONLY work for
    converting images that are in ``inMode`` to images that are in outMode
    color format (PIL mode, i.e. "RGB", "RGBA", "CMYK", etc.).

    Usage of the resulting transform object is exactly the same as with
    ImageCms.buildTransform().

    Proof profiling is generally used when using an output device to get a
    good idea of what the final printed/displayed image would look like on
    the ``proofProfile`` device when it's quicker and easier to use the
    output device for judging color.  Generally, this means that the
    output device is a monitor, or a dye-sub printer (etc.), and the simulated
    device is something more expensive, complicated, or time consuming
    (making it difficult to make a real print for color judgement purposes).

    Soft-proofing basically functions by adjusting the colors on the
    output device to match the colors of the device being simulated. However,
    when the simulated device has a much wider gamut than the output
    device, you may obtain marginal results.

    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this transform, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        (monitor, usually) profile you wish to use for this transform, or a
        profile object
    :param proofProfile: String, as a valid filename path to the ICC proof
        profile you wish to use for this transform, or a profile object
    :param inMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param outMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the input->proof (simulated) transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param proofRenderingIntent: Integer (0-3) specifying the rendering intent
        you wish to use for proof->output transform

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param flags: Integer (0-...) specifying additional flags
    :returns: A CmsTransform class object.
    :exception PyCMSError:
    """

    if not isinstance(renderingIntent, int) or not (0 <= renderingIntent <= 3):
        msg = "renderingIntent must be an integer between 0 and 3"
        raise PyCMSError(msg)

    if not isinstance(flags, int) or not (0 <= flags <= _MAX_FLAG):
        msg = f"flags must be an integer between 0 and {_MAX_FLAG}"
        raise PyCMSError(msg)

    try:
        if not isinstance(inputProfile, ImageCmsProfile):
            inputProfile = ImageCmsProfile(inputProfile)
        if not isinstance(outputProfile, ImageCmsProfile):
            outputProfile = ImageCmsProfile(outputProfile)
        if not isinstance(proofProfile, ImageCmsProfile):
            proofProfile = ImageCmsProfile(proofProfile)
        return ImageCmsTransform(
            inputProfile,
            outputProfile,
            inMode,
            outMode,
            renderingIntent,
            proofProfile,
            proofRenderingIntent,
            flags,
        )
    except (OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


buildTransformFromOpenProfiles = buildTransform
buildProofTransformFromOpenProfiles = buildProofTransform


def applyTransform(
    im: Image.Image, transform: ImageCmsTransform, inPlace: bool = False
) -> Image.Image | None:
    """
    (pyCMS) Applies a transform to a given image.

    If ``im.mode != transform.input_mode``, a :exc:`PyCMSError` is raised.

    If ``inPlace`` is ``True`` and ``transform.input_mode != transform.output_mode``, a
    :exc:`PyCMSError` is raised.

    If ``im.mode``, ``transform.input_mode`` or ``transform.output_mode`` is not
    supported by pyCMSdll or the profiles you used for the transform, a
    :exc:`PyCMSError` is raised.

    If an error occurs while the transform is being applied,
    a :exc:`PyCMSError` is raised.

    This function applies a pre-calculated transform (from
    ImageCms.buildTransform() or ImageCms.buildTransformFromOpenProfiles())
    to an image. The transform can be used for multiple images, saving
    considerable calculation time if doing the same conversion multiple times.

    If you want to modify im in-place instead of receiving a new image as
    the return value, set ``inPlace`` to ``True``.  This can only be done if
    ``transform.input_mode`` and ``transform.output_mode`` are the same, because we
    can't change the mode in-place (the buffer sizes for some modes are
    different).  The default behavior is to return a new :py:class:`~PIL.Image.Image`
    object of the same dimensions in mode ``transform.output_mode``.

    :param im: An :py:class:`~PIL.Image.Image` object, and ``im.mode`` must be the same
        as the ``input_mode`` supported by the transform.
    :param transform: A valid CmsTransform class object
    :param inPlace: Bool.  If ``True``, ``im`` is modified in place and ``None`` is
        returned, if ``False``, a new :py:class:`~PIL.Image.Image` object with the
        transform applied is returned (and ``im`` is not changed). The default is
        ``False``.
    :returns: Either ``None``, or a new :py:class:`~PIL.Image.Image` object,
        depending on the value of ``inPlace``. The profile will be returned in
        the image's ``info['icc_profile']``.
    :exception PyCMSError:
    """

    try:
        if inPlace:
            transform.apply_in_place(im)
            imOut = None
        else:
            imOut = transform.apply(im)
    except (TypeError, ValueError) as v:
        raise PyCMSError(v) from v

    return imOut


def createProfile(
    colorSpace: Literal["LAB", "XYZ", "sRGB"], colorTemp: SupportsFloat = 0
) -> core.CmsProfile:
    """
    (pyCMS) Creates a profile.

    If colorSpace not in ``["LAB", "XYZ", "sRGB"]``,
    a :exc:`PyCMSError` is raised.

    If using LAB and ``colorTemp`` is not a positive integer,
    a :exc:`PyCMSError` is raised.

    If an error occurs while creating the profile,
    a :exc:`PyCMSError` is raised.

    Use this function to create common profiles on-the-fly instead of
    having to supply a profile on disk and knowing the path to it.  It
    returns a normal CmsProfile object that can be passed to
    ImageCms.buildTransformFromOpenProfiles() to create a transform to apply
    to images.

    :param colorSpace: String, the color space of the profile you wish to
        create.
        Currently only "LAB", "XYZ", and "sRGB" are supported.
    :param colorTemp: Positive number for the white point for the profile, in
        degrees Kelvin (i.e. 5000, 6500, 9600, etc.).  The default is for D50
        illuminant if omitted (5000k).  colorTemp is ONLY applied to LAB
        profiles, and is ignored for XYZ and sRGB.
    :returns: A CmsProfile class object
    :exception PyCMSError:
    """

    if colorSpace not in ["LAB", "XYZ", "sRGB"]:
        msg = (
            f"Color space not supported for on-the-fly profile creation ({colorSpace})"
        )
        raise PyCMSError(msg)

    if colorSpace == "LAB":
        try:
            colorTemp = float(colorTemp)
        except (TypeError, ValueError) as e:
            msg = f'Color temperature must be numeric, "{colorTemp}" not valid'
            raise PyCMSError(msg) from e

    try:
        return core.createProfile(colorSpace, colorTemp)
    except (TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def getProfileName(profile: _CmsProfileCompatible) -> str:
    """

    (pyCMS) Gets the internal product name for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile,
    a :exc:`PyCMSError` is raised If an error occurs while trying
    to obtain the name tag, a :exc:`PyCMSError` is raised.

    Use this function to obtain the INTERNAL name of the profile (stored
    in an ICC tag in the profile itself), usually the one used when the
    profile was originally created.  Sometimes this tag also contains
    additional information supplied by the creator.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal name of the profile as stored
        in an ICC tag.
    :exception PyCMSError:
    """

    try:
        # add an extra newline to preserve pyCMS compatibility
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        # do it in python, not c.
        #    // name was "%s - %s" (model, manufacturer) || Description ,
        #    // but if the Model and Manufacturer were the same or the model
        #    // was long, Just the model,  in 1.x
        model = profile.profile.model
        manufacturer = profile.profile.manufacturer

        if not (model or manufacturer):
            return (profile.profile.profile_description or "") + "\n"
        if not manufacturer or (model and len(model) > 30):
            return f"{model}\n"
        return f"{model} - {manufacturer}\n"

    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def getProfileInfo(profile: _CmsProfileCompatible) -> str:
    """
    (pyCMS) Gets the internal product information for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile,
    a :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the info tag,
    a :exc:`PyCMSError` is raised.

    Use this function to obtain the information stored in the profile's
    info tag.  This often contains details about the profile, and how it
    was created, as supplied by the creator.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """

    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        # add an extra newline to preserve pyCMS compatibility
        # Python, not C. the white point bits weren't working well,
        # so skipping.
        # info was description \r\n\r\n copyright \r\n\r\n K007 tag \r\n\r\n whitepoint
        description = profile.profile.profile_description
        cpright = profile.profile.copyright
        elements = [element for element in (description, cpright) if element]
        return "\r\n\r\n".join(elements) + "\r\n\r\n"

    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def getProfileCopyright(profile: _CmsProfileCompatible) -> str:
    """
    (pyCMS) Gets the copyright for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a
    :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the copyright tag,
    a :exc:`PyCMSError` is raised.

    Use this function to obtain the information stored in the profile's
    copyright tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """
    try:
        # add an extra newline to preserve pyCMS compatibility
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.copyright or "") + "\n"
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def getProfileManufacturer(profile: _CmsProfileCompatible) -> str:
    """
    (pyCMS) Gets the manufacturer for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a
    :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the manufacturer tag, a
    :exc:`PyCMSError` is raised.

    Use this function to obtain the information stored in the profile's
    manufacturer tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """
    try:
        # add an extra newline to preserve pyCMS compatibility
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.manufacturer or "") + "\n"
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def getProfileModel(profile: _CmsProfileCompatible) -> str:
    """
    (pyCMS) Gets the model for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a
    :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the model tag,
    a :exc:`PyCMSError` is raised.

    Use this function to obtain the information stored in the profile's
    model tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """

    try:
        # add an extra newline to preserve pyCMS compatibility
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.model or "") + "\n"
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def getProfileDescription(profile: _CmsProfileCompatible) -> str:
    """
    (pyCMS) Gets the description for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a
    :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the description tag,
    a :exc:`PyCMSError` is raised.

    Use this function to obtain the information stored in the profile's
    description tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in an
        ICC tag.
    :exception PyCMSError:
    """

    try:
        # add an extra newline to preserve pyCMS compatibility
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return (profile.profile.profile_description or "") + "\n"
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def getDefaultIntent(profile: _CmsProfileCompatible) -> int:
    """
    (pyCMS) Gets the default intent name for the given profile.

    If ``profile`` isn't a valid CmsProfile object or filename to a profile, a
    :exc:`PyCMSError` is raised.

    If an error occurs while trying to obtain the default intent, a
    :exc:`PyCMSError` is raised.

    Use this function to determine the default (and usually best optimized)
    rendering intent for this profile.  Most profiles support multiple
    rendering intents, but are intended mostly for one type of conversion.
    If you wish to use a different intent than returned, use
    ImageCms.isIntentSupported() to verify it will work first.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: Integer 0-3 specifying the default rendering intent for this
        profile.

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
            they do.
    :exception PyCMSError:
    """

    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        return profile.profile.rendering_intent
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v


def isIntentSupported(
    profile: _CmsProfileCompatible, intent: Intent, direction: Direction
) -> Literal[-1, 1]:
    """
    (pyCMS) Checks if a given intent is supported.

    Use this function to verify that you can use your desired
    ``intent`` with ``profile``, and that ``profile`` can be used for the
    input/output/proof profile as you desire.

    Some profiles are created specifically for one "direction", can cannot
    be used for others. Some profiles can only be used for certain
    rendering intents, so it's best to either verify this before trying
    to create a transform with them (using this function), or catch the
    potential :exc:`PyCMSError` that will occur if they don't
    support the modes you select.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :param intent: Integer (0-3) specifying the rendering intent you wish to
        use with this profile

            ImageCms.Intent.PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.Intent.RELATIVE_COLORIMETRIC = 1
            ImageCms.Intent.SATURATION            = 2
            ImageCms.Intent.ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
            they do.
    :param direction: Integer specifying if the profile is to be used for
        input, output, or proof

            INPUT  = 0 (or use ImageCms.Direction.INPUT)
            OUTPUT = 1 (or use ImageCms.Direction.OUTPUT)
            PROOF  = 2 (or use ImageCms.Direction.PROOF)

    :returns: 1 if the intent/direction are supported, -1 if they are not.
    :exception PyCMSError:
    """

    try:
        if not isinstance(profile, ImageCmsProfile):
            profile = ImageCmsProfile(profile)
        # FIXME: I get different results for the same data w. different
        # compilers.  Bug in LittleCMS or in the binding?
        if profile.profile.is_intent_supported(intent, direction):
            return 1
        else:
            return -1
    except (AttributeError, OSError, TypeError, ValueError) as v:
        raise PyCMSError(v) from v
<|endoftext|>#
# The Python Imaging Library
#
# load a GIMP brush file
#
# History:
#       96-03-14 fl     Created
#       16-01-08 es     Version 2
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
# Copyright (c) Eric Soroos 2016.
#
# See the README file for information on usage and redistribution.
#
#
# See https://github.com/GNOME/gimp/blob/mainline/devel-docs/gbr.txt for
# format documentation.
#
# This code Interprets version 1 and 2 .gbr files.
# Version 1 files are obsolete, and should not be used for new
#   brushes.
# Version 2 files are saved by GIMP v2.8 (at least)
# Version 3 files have a format specifier of 18 for 16bit floats in
#   the color depth field. This is currently unsupported by Pillow.
from __future__ import annotations

from . import Image, ImageFile
from ._binary import i32be as i32


def _accept(prefix: bytes) -> bool:
    return len(prefix) >= 8 and i32(prefix, 0) >= 20 and i32(prefix, 4) in (1, 2)


##
# Image plugin for the GIMP brush format.


class GbrImageFile(ImageFile.ImageFile):
    format = "GBR"
    format_description = "GIMP brush file"

    def _open(self) -> None:
        assert self.fp is not None
        header_size = i32(self.fp.read(4))
        if header_size < 20:
            msg = "not a GIMP brush"
            raise SyntaxError(msg)
        version = i32(self.fp.read(4))
        if version not in (1, 2):
            msg = f"Unsupported GIMP brush version: {version}"
            raise SyntaxError(msg)

        width = i32(self.fp.read(4))
        height = i32(self.fp.read(4))
        color_depth = i32(self.fp.read(4))
        if width == 0 or height == 0:
            msg = "not a GIMP brush"
            raise SyntaxError(msg)
        if color_depth not in (1, 4):
            msg = f"Unsupported GIMP brush color depth: {color_depth}"
            raise SyntaxError(msg)

        if version == 1:
            comment_length = header_size - 20
        else:
            comment_length = header_size - 28
            magic_number = self.fp.read(4)
            if magic_number != b"GIMP":
                msg = "not a GIMP brush, bad magic number"
                raise SyntaxError(msg)
            self.info["spacing"] = i32(self.fp.read(4))

        self.info["comment"] = self.fp.read(comment_length)[:-1]

        if color_depth == 1:
            self._mode = "L"
        else:
            self._mode = "RGBA"

        self._size = width, height

        # Image might not be small
        Image._decompression_bomb_check(self.size)

        # Data is an uncompressed block of w * h * bytes/pixel
        self._data_size = width * height * color_depth

    def load(self) -> Image.core.PixelAccess | None:
        if self._im is None:
            assert self.fp is not None
            self.im = Image.core.new(self.mode, self.size)
            self.frombytes(self.fp.read(self._data_size))
        return Image.Image.load(self)


#
# registry


Image.register_open(GbrImageFile.format, GbrImageFile, _accept)
Image.register_extension(GbrImageFile.format, ".gbr")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# WAL file handling
#
# History:
# 2003-04-23 fl   created
#
# Copyright (c) 2003 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

"""
This reader is based on the specification available from:
https://www.flipcode.com/archives/Quake_2_BSP_File_Format.shtml
and has been tested with a few sample files found using google.

.. note::
    This format cannot be automatically recognized, so the reader
    is not registered for use with :py:func:`PIL.Image.open()`.
    To open a WAL file, use the :py:func:`PIL.WalImageFile.open()` function instead.
"""

from __future__ import annotations

from typing import IO

from . import Image, ImageFile
from ._binary import i32le as i32
from ._typing import StrOrBytesPath


class WalImageFile(ImageFile.ImageFile):
    format = "WAL"
    format_description = "Quake2 Texture"

    def _open(self) -> None:
        self._mode = "P"

        # read header fields
        assert self.fp is not None
        header = self.fp.read(32 + 24 + 32 + 12)
        self._size = i32(header, 32), i32(header, 36)
        Image._decompression_bomb_check(self.size)

        # load pixel data
        offset = i32(header, 40)
        self.fp.seek(offset)

        # strings are null-terminated
        self.info["name"] = header[:32].split(b"\0", 1)[0]
        if next_name := header[56 : 56 + 32].split(b"\0", 1)[0]:
            self.info["next_name"] = next_name

    def load(self) -> Image.core.PixelAccess | None:
        if self._im is None:
            assert self.fp is not None
            self.im = Image.core.new(self.mode, self.size)
            self.frombytes(self.fp.read(self.size[0] * self.size[1]))
            self.putpalette(quake2palette)
        return Image.Image.load(self)


def open(filename: StrOrBytesPath | IO[bytes]) -> WalImageFile:
    """
    Load texture from a Quake2 WAL texture file.

    By default, a Quake2 standard palette is attached to the texture.
    To override the palette, use the :py:func:`PIL.Image.Image.putpalette()` method.

    :param filename: WAL file name, or an opened file handle.
    :returns: An image instance.
    """
    return WalImageFile(filename)


quake2palette = (
    # default palette taken from piffo 0.93 by Hans Häggström
    b"\x01\x01\x01\x0b\x0b\x0b\x12\x12\x12\x17\x17\x17\x1b\x1b\x1b\x1e"
    b"\x1e\x1e\x22\x22\x22\x26\x26\x26\x29\x29\x29\x2c\x2c\x2c\x2f\x2f"
    b"\x2f\x32\x32\x32\x35\x35\x35\x37\x37\x37\x3a\x3a\x3a\x3c\x3c\x3c"
    b"\x24\x1e\x13\x22\x1c\x12\x20\x1b\x12\x1f\x1a\x10\x1d\x19\x10\x1b"
    b"\x17\x0f\x1a\x16\x0f\x18\x14\x0d\x17\x13\x0d\x16\x12\x0d\x14\x10"
    b"\x0b\x13\x0f\x0b\x10\x0d\x0a\x0f\x0b\x0a\x0d\x0b\x07\x0b\x0a\x07"
    b"\x23\x23\x26\x22\x22\x25\x22\x20\x23\x21\x1f\x22\x20\x1e\x20\x1f"
    b"\x1d\x1e\x1d\x1b\x1c\x1b\x1a\x1a\x1a\x19\x19\x18\x17\x17\x17\x16"
    b"\x16\x14\x14\x14\x13\x13\x13\x10\x10\x10\x0f\x0f\x0f\x0d\x0d\x0d"
    b"\x2d\x28\x20\x29\x24\x1c\x27\x22\x1a\x25\x1f\x17\x38\x2e\x1e\x31"
    b"\x29\x1a\x2c\x25\x17\x26\x20\x14\x3c\x30\x14\x37\x2c\x13\x33\x28"
    b"\x12\x2d\x24\x10\x28\x1f\x0f\x22\x1a\x0b\x1b\x14\x0a\x13\x0f\x07"
    b"\x31\x1a\x16\x30\x17\x13\x2e\x16\x10\x2c\x14\x0d\x2a\x12\x0b\x27"
    b"\x0f\x0a\x25\x0f\x07\x21\x0d\x01\x1e\x0b\x01\x1c\x0b\x01\x1a\x0b"
    b"\x01\x18\x0a\x01\x16\x0a\x01\x13\x0a\x01\x10\x07\x01\x0d\x07\x01"
    b"\x29\x23\x1e\x27\x21\x1c\x26\x20\x1b\x25\x1f\x1a\x23\x1d\x19\x21"
    b"\x1c\x18\x20\x1b\x17\x1e\x19\x16\x1c\x18\x14\x1b\x17\x13\x19\x14"
    b"\x10\x17\x13\x0f\x14\x10\x0d\x12\x0f\x0b\x0f\x0b\x0a\x0b\x0a\x07"
    b"\x26\x1a\x0f\x23\x19\x0f\x20\x17\x0f\x1c\x16\x0f\x19\x13\x0d\x14"
    b"\x10\x0b\x10\x0d\x0a\x0b\x0a\x07\x33\x22\x1f\x35\x29\x26\x37\x2f"
    b"\x2d\x39\x35\x34\x37\x39\x3a\x33\x37\x39\x30\x34\x36\x2b\x31\x34"
    b"\x27\x2e\x31\x22\x2b\x2f\x1d\x28\x2c\x17\x25\x2a\x0f\x20\x26\x0d"
    b"\x1e\x25\x0b\x1c\x22\x0a\x1b\x20\x07\x19\x1e\x07\x17\x1b\x07\x14"
    b"\x18\x01\x12\x16\x01\x0f\x12\x01\x0b\x0d\x01\x07\x0a\x01\x01\x01"
    b"\x2c\x21\x21\x2a\x1f\x1f\x29\x1d\x1d\x27\x1c\x1c\x26\x1a\x1a\x24"
    b"\x18\x18\x22\x17\x17\x21\x16\x16\x1e\x13\x13\x1b\x12\x12\x18\x10"
    b"\x10\x16\x0d\x0d\x12\x0b\x0b\x0d\x0a\x0a\x0a\x07\x07\x01\x01\x01"
    b"\x2e\x30\x29\x2d\x2e\x27\x2b\x2c\x26\x2a\x2a\x24\x28\x29\x23\x27"
    b"\x27\x21\x26\x26\x1f\x24\x24\x1d\x22\x22\x1c\x1f\x1f\x1a\x1c\x1c"
    b"\x18\x19\x19\x16\x17\x17\x13\x13\x13\x10\x0f\x0f\x0d\x0b\x0b\x0a"
    b"\x30\x1e\x1b\x2d\x1c\x19\x2c\x1a\x17\x2a\x19\x14\x28\x17\x13\x26"
    b"\x16\x10\x24\x13\x0f\x21\x12\x0d\x1f\x10\x0b\x1c\x0f\x0a\x19\x0d"
    b"\x0a\x16\x0b\x07\x12\x0a\x07\x0f\x07\x01\x0a\x01\x01\x01\x01\x01"
    b"\x28\x29\x38\x26\x27\x36\x25\x26\x34\x24\x24\x31\x22\x22\x2f\x20"
    b"\x21\x2d\x1e\x1f\x2a\x1d\x1d\x27\x1b\x1b\x25\x19\x19\x21\x17\x17"
    b"\x1e\x14\x14\x1b\x13\x12\x17\x10\x0f\x13\x0d\x0b\x0f\x0a\x07\x07"
    b"\x2f\x32\x29\x2d\x30\x26\x2b\x2e\x24\x29\x2c\x21\x27\x2a\x1e\x25"
    b"\x28\x1c\x23\x26\x1a\x21\x25\x18\x1e\x22\x14\x1b\x1f\x10\x19\x1c"
    b"\x0d\x17\x1a\x0a\x13\x17\x07\x10\x13\x01\x0d\x0f\x01\x0a\x0b\x01"
    b"\x01\x3f\x01\x13\x3c\x0b\x1b\x39\x10\x20\x35\x14\x23\x31\x17\x23"
    b"\x2d\x18\x23\x29\x18\x3f\x3f\x3f\x3f\x3f\x39\x3f\x3f\x31\x3f\x3f"
    b"\x2a\x3f\x3f\x20\x3f\x3f\x14\x3f\x3c\x12\x3f\x39\x0f\x3f\x35\x0b"
    b"\x3f\x32\x07\x3f\x2d\x01\x3d\x2a\x01\x3b\x26\x01\x39\x21\x01\x37"
    b"\x1d\x01\x34\x1a\x01\x32\x16\x01\x2f\x12\x01\x2d\x0f\x01\x2a\x0b"
    b"\x01\x27\x07\x01\x23\x01\x01\x1d\x01\x01\x17\x01\x01\x10\x01\x01"
    b"\x3d\x01\x01\x19\x19\x3f\x3f\x01\x01\x01\x01\x3f\x16\x16\x13\x10"
    b"\x10\x0f\x0d\x0d\x0b\x3c\x2e\x2a\x36\x27\x20\x30\x21\x18\x29\x1b"
    b"\x10\x3c\x39\x37\x37\x32\x2f\x31\x2c\x28\x2b\x26\x21\x30\x22\x20"
)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# Windows Cursor support for PIL
#
# notes:
#       uses BmpImagePlugin.py to read the bitmap data.
#
# history:
#       96-05-27 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import BmpImagePlugin, Image, ImageOps
from ._binary import i16le as i16
from ._binary import i32le as i32

#
# --------------------------------------------------------------------


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"\0\0\2\0")


##
# Image plugin for Windows Cursor files.


class CurImageFile(BmpImagePlugin.BmpImageFile):
    format = "CUR"
    format_description = "Windows Cursor"

    def _open(self) -> None:
        assert self.fp is not None
        offset = self.fp.tell()

        # check magic
        s = self.fp.read(6)
        if not _accept(s):
            msg = "not a CUR file"
            raise SyntaxError(msg)

        # pick the largest cursor in the file
        m = b""
        for i in range(i16(s, 4)):
            s = self.fp.read(16)
            if not m:
                m = s
            elif s[0] > m[0] and s[1] > m[1]:
                m = s
        if not m:
            msg = "No cursors were found"
            raise TypeError(msg)

        # load as bitmap
        self._bitmap(i32(m, 12) + offset)
        self._masked = self.mode in ("1", "L")
        if self._masked:
            self._rawmode = self.mode
            self._mode = "LA"

        # patch up the bitmap height
        self._size = self.width, self.height // 2
        if not self._masked:
            self.tile = [self.tile[0]._replace(extents=(0, 0) + self.size)]

    def load_prepare(self) -> None:
        if self._masked:
            self._mode = self._rawmode
            self._size = self.width, self.height * 2
        super().load_prepare()

    def load_end(self) -> None:
        if not self._masked:
            return
        self._mode = "LA"
        new_height = self.height // 2

        and_mask = self.im.crop((0, 0, self.width, new_height))
        xor_mask = self.im.crop((0, new_height, self.width, self.height))

        self._size = self.width, new_height
        self._im = Image.core.fill(self.mode, self.size)
        self._im.paste(
            xor_mask.convert(self.mode),
            (0, 0) + self.size,
            ImageOps.invert(Image.Image()._new(and_mask)).im,
        )


#
# --------------------------------------------------------------------

Image.register_open(CurImageFile.format, CurImageFile, _accept)

Image.register_extension(CurImageFile.format, ".cur")
<|endoftext|>from __future__ import annotations

import warnings

from . import __version__


def deprecate(
    deprecated: str,
    when: int | None,
    replacement: str | None = None,
    *,
    action: str | None = None,
    plural: bool = False,
    stacklevel: int = 3,
) -> None:
    """
    Deprecations helper.

    :param deprecated: Name of thing to be deprecated.
    :param when: Pillow major version to be removed in.
    :param replacement: Name of replacement.
    :param action: Instead of "replacement", give a custom call to action
        e.g. "Upgrade to new thing".
    :param plural: if the deprecated thing is plural, needing "are" instead of "is".

    Usually of the form:

        "[deprecated] is deprecated and will be removed in Pillow [when] (yyyy-mm-dd).
        Use [replacement] instead."

    You can leave out the replacement sentence:

        "[deprecated] is deprecated and will be removed in Pillow [when] (yyyy-mm-dd)"

    Or with another call to action:

        "[deprecated] is deprecated and will be removed in Pillow [when] (yyyy-mm-dd).
        [action]."
    """

    is_ = "are" if plural else "is"

    if when is None:
        removed = "a future version"
    elif when <= int(__version__.split(".")[0]):
        msg = f"{deprecated} {is_} deprecated and should be removed."
        raise RuntimeError(msg)
    elif when == 13:
        removed = "Pillow 13 (2026-10-15)"
    elif when == 14:
        removed = "Pillow 14 (2027-10-15)"
    else:
        msg = f"Unknown removal version: {when}. Update {__name__}?"
        raise ValueError(msg)

    if replacement and action:
        msg = "Use only one of 'replacement' and 'action'"
        raise ValueError(msg)

    if replacement:
        action = f". Use {replacement} instead."
    elif action:
        action = f". {action.rstrip('.')}."
    else:
        action = ""

    warnings.warn(
        f"{deprecated} {is_} deprecated and will be removed in {removed}{action}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
<|endoftext|>from __future__ import annotations

from io import BytesIO

from . import Image, ImageFile

try:
    from . import _webp

    SUPPORTED = True
except ImportError:
    SUPPORTED = False

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import IO, Any

_VP8_MODES_BY_IDENTIFIER = {
    b"VP8 ": "RGB",
    b"VP8X": "RGBA",
    b"VP8L": "RGBA",  # lossless
}


def _accept(prefix: bytes) -> bool | str:
    is_riff_file_format = prefix.startswith(b"RIFF")
    is_webp_file = prefix[8:12] == b"WEBP"
    is_valid_vp8_mode = prefix[12:16] in _VP8_MODES_BY_IDENTIFIER

    if is_riff_file_format and is_webp_file and is_valid_vp8_mode:
        if not SUPPORTED:
            return (
                "image file could not be identified because WEBP support not installed"
            )
        return True
    return False


class WebPImageFile(ImageFile.ImageFile):
    format = "WEBP"
    format_description = "WebP image"
    __loaded = 0
    __logical_frame = 0

    def _open(self) -> None:
        assert self.fp is not None
        s = self.fp.read()
        if not _accept(s):
            msg = "not a WEBP file"
            raise SyntaxError(msg)

        # Use the newer AnimDecoder API to parse the (possibly) animated file,
        # and access muxed chunks like ICC/EXIF/XMP.
        self._decoder = _webp.WebPAnimDecoder(s)

        # Get info from decoder
        self._size, self.info["loop"], bgcolor, self.n_frames, self.rawmode = (
            self._decoder.get_info()
        )
        self.info["background"] = (
            (bgcolor >> 16) & 0xFF,  # R
            (bgcolor >> 8) & 0xFF,  # G
            bgcolor & 0xFF,  # B
            (bgcolor >> 24) & 0xFF,  # A
        )
