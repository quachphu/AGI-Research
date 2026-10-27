def _parse_codestream(fp: IO[bytes]) -> tuple[tuple[int, int], str]:
    """Parse the JPEG 2000 codestream to extract the size and component
    count from the SIZ marker segment, returning a PIL (size, mode) tuple."""

    hdr = fp.read(2)
    lsiz = _binary.i16be(hdr)
    if lsiz < 38:
        msg = "SIZ marker length must be at least 38"
        raise ValueError(msg)
    siz = hdr + fp.read(lsiz - 2)
    lsiz, rsiz, xsiz, ysiz, xosiz, yosiz, _, _, _, _, csiz = struct.unpack_from(
        ">HHIIIIIIIIH", siz
    )

    size = (xsiz - xosiz, ysiz - yosiz)
    if csiz == 1:
        ssiz = struct.unpack_from(">B", siz, 38)
        if (ssiz[0] & 0x7F) + 1 > 8:
            mode = "I;16"
        else:
            mode = "L"
    elif csiz == 2:
        mode = "LA"
    elif csiz == 3:
        mode = "RGB"
    elif csiz == 4:
        mode = "RGBA"
    else:
        msg = "unable to determine J2K image mode"
        raise SyntaxError(msg)

    return size, mode


def _res_to_dpi(num: int, denom: int, exp: int) -> float | None:
    """Convert JPEG2000's (numerator, denominator, exponent-base-10) resolution,
    calculated as (num / denom) * 10^exp and stored in dots per meter,
    to floating-point dots per inch."""
    if denom == 0:
        return None
    return (254 * num * (10**exp)) / (10000 * denom)


def _parse_jp2_header(
    fp: IO[bytes],
) -> tuple[
    tuple[int, int],
    str,
    str | None,
    tuple[float, float] | None,
    ImagePalette.ImagePalette | None,
]:
    """Parse the JP2 header box to extract size, component count,
    color space information, and optionally DPI information,
    returning a (size, mode, mimetype, dpi) tuple."""

    # Find the JP2 header box
    reader = BoxReader(fp)
    header = None
    mimetype = None
    while reader.has_next_box():
        tbox = reader.next_box_type()

        if tbox == b"jp2h":
            header = reader.read_boxes()
            break
        elif tbox == b"ftyp":
            if reader.read_fields(">4s")[0] == b"jpx ":
                mimetype = "image/jpx"
    assert header is not None

    size = None
    mode = None
    bpc = None
    nc = None
    dpi = None  # 2-tuple of DPI info, or None
    palette = None
    colr = None

    while header.has_next_box():
        tbox = header.next_box_type()

        if tbox == b"ihdr":
            height, width, nc, bpc = header.read_fields(">IIHB")
            assert isinstance(height, int)
            assert isinstance(width, int)
            assert isinstance(bpc, int)
            size = (width, height)
            if nc == 1 and (bpc & 0x7F) > 8:
                mode = "I;16"
            elif nc == 1:
                mode = "L"
            elif nc == 2:
                mode = "LA"
            elif nc == 3:
                mode = "RGB"
            elif nc == 4:
                mode = "RGBA"
        elif tbox == b"colr":
            meth, _, _, enumcs = header.read_fields(">BBBI")
            if meth == 1:
                if enumcs in (0, 15):
                    colr = "1"
                elif enumcs == 12:
                    colr = "CMYK"
                    if nc == 4:
                        mode = "CMYK"
                elif enumcs == 17:
                    colr = "L"
        elif tbox == b"pclr" and mode in ("L", "LA") and colr not in ("1", "L"):
            ne, npc = header.read_fields(">HB")
            assert isinstance(ne, int)
            assert isinstance(npc, int)
            max_bitdepth = 0
            for bitdepth in header.read_fields(">" + ("B" * npc)):
                assert isinstance(bitdepth, int)
                if bitdepth > max_bitdepth:
                    max_bitdepth = bitdepth
            if max_bitdepth <= 8:
                if npc == 4:
                    palette_mode = "CMYK" if colr == "CMYK" else "RGBA"
                else:
                    palette_mode = "RGB"
                palette = ImagePalette.ImagePalette(palette_mode)
                for i in range(ne):
                    color: list[int] = []
                    for value in header.read_fields(">" + ("B" * npc)):
                        assert isinstance(value, int)
                        color.append(value)
                    palette.getcolor(tuple(color))
                mode = "P" if mode == "L" else "PA"
        elif tbox == b"res ":
            res = header.read_boxes()
            while res.has_next_box():
                tres = res.next_box_type()
                if tres == b"resc":
                    vrcn, vrcd, hrcn, hrcd, vrce, hrce = res.read_fields(">HHHHBB")
                    assert isinstance(vrcn, int)
                    assert isinstance(vrcd, int)
                    assert isinstance(hrcn, int)
                    assert isinstance(hrcd, int)
                    assert isinstance(vrce, int)
                    assert isinstance(hrce, int)
                    hres = _res_to_dpi(hrcn, hrcd, hrce)
                    vres = _res_to_dpi(vrcn, vrcd, vrce)
                    if hres is not None and vres is not None:
                        dpi = (hres, vres)
                    break

    if size is None or mode is None:
        msg = "Malformed JP2 header"
        raise SyntaxError(msg)

    return size, mode, mimetype, dpi, palette


##
# Image plugin for JPEG2000 images.


class Jpeg2KImageFile(ImageFile.ImageFile):
    format = "JPEG2000"
    format_description = "JPEG 2000 (ISO 15444)"

    def _open(self) -> None:
        assert self.fp is not None
        sig = self.fp.read(4)
        if sig == b"\xff\x4f\xff\x51":
            self.codec = "j2k"
            self._size, self._mode = _parse_codestream(self.fp)
            self._parse_comment()
        else:
            sig = sig + self.fp.read(8)

            if sig == b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a":
                self.codec = "jp2"
                header = _parse_jp2_header(self.fp)
                self._size, self._mode, self.custom_mimetype, dpi, self.palette = header
                if dpi is not None:
                    self.info["dpi"] = dpi
                if self.fp.read(12).endswith(b"jp2c\xff\x4f\xff\x51"):
                    hdr = self.fp.read(2)
                    length = _binary.i16be(hdr)
                    self.fp.seek(length - 2, os.SEEK_CUR)
                    self._parse_comment()
            else:
                msg = "not a JPEG 2000 file"
                raise SyntaxError(msg)

        self._reduce = 0
        self.layers = 0

        fd = -1
        length = -1

        try:
            fd = self.fp.fileno()
            length = os.fstat(fd).st_size
        except Exception:
            fd = -1
            try:
                pos = self.fp.tell()
                self.fp.seek(0, io.SEEK_END)
                length = self.fp.tell()
                self.fp.seek(pos)
            except Exception:
                length = -1

        self.tile = [
            ImageFile._Tile(
                "jpeg2k",
                (0, 0) + self.size,
                0,
                (self.codec, self._reduce, self.layers, fd, length),
            )
        ]

    def _parse_comment(self) -> None:
        assert self.fp is not None
        while True:
            marker = self.fp.read(2)
            if not marker:
                break
            typ = marker[1]
            if typ in (0x90, 0xD9):
                # Start of tile or end of codestream
                break
            hdr = self.fp.read(2)
            length = _binary.i16be(hdr)
            if length < 2:
                msg = "Marker length too small"
                raise ValueError(msg)
            if typ == 0x64:
                # Comment
                self.info["comment"] = self.fp.read(length - 2)[2:]
                break
            else:
                self.fp.seek(length - 2, os.SEEK_CUR)

    @property  # type: ignore[override]
    def reduce(
        self,
    ) -> (
        Callable[[int | tuple[int, int], tuple[int, int, int, int] | None], Image.Image]
        | int
    ):
        # https://github.com/python-pillow/Pillow/issues/4343 found that the
        # new Image 'reduce' method was shadowed by this plugin's 'reduce'
        # property. This attempts to allow for both scenarios
        return self._reduce or super().reduce

    @reduce.setter
    def reduce(self, value: int) -> None:
        self._reduce = value

    def load(self) -> Image.core.PixelAccess | None:
        if self.tile and self._reduce:
            power = 1 << self._reduce
            adjust = power >> 1
            self._size = (
                int((self.size[0] + adjust) / power),
                int((self.size[1] + adjust) / power),
            )

            # Update the reduce and layers settings
            t = self.tile[0]
            assert isinstance(t[3], tuple)
            t3 = (t[3][0], self._reduce, self.layers, t[3][3], t[3][4])
            self.tile = [ImageFile._Tile(t[0], (0, 0) + self.size, t[2], t3)]

        return ImageFile.ImageFile.load(self)


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(
        (b"\xff\x4f\xff\x51", b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a")
    )


# ------------------------------------------------------------
# Save support


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    # Get the keyword arguments
    info = im.encoderinfo

    if isinstance(filename, str):
        filename = filename.encode()
    if filename.endswith(b".j2k") or info.get("no_jp2", False):
        kind = "j2k"
    else:
        kind = "jp2"

    offset = info.get("offset", None)
    tile_offset = info.get("tile_offset", None)
    tile_size = info.get("tile_size", None)
    quality_mode = info.get("quality_mode", "rates")
    quality_layers = info.get("quality_layers", None)
    if quality_layers is not None and not (
        isinstance(quality_layers, (list, tuple))
        and all(
            isinstance(quality_layer, (int, float)) for quality_layer in quality_layers
        )
    ):
        msg = "quality_layers must be a sequence of numbers"
        raise ValueError(msg)

    num_resolutions = info.get("num_resolutions", 0)
    cblk_size = info.get("codeblock_size", None)
    precinct_size = info.get("precinct_size", None)
    irreversible = info.get("irreversible", False)
    progression = info.get("progression", "LRCP")
    cinema_mode = info.get("cinema_mode", "no")
    mct = info.get("mct", 0)
    signed = info.get("signed", False)
    comment = info.get("comment")
    if isinstance(comment, str):
        comment = comment.encode()
    plt = info.get("plt", False)

    fd = -1
    if hasattr(fp, "fileno"):
        try:
            fd = fp.fileno()
        except Exception:
            fd = -1

    im.encoderconfig = (
        offset,
        tile_offset,
        tile_size,
        quality_mode,
        quality_layers,
        num_resolutions,
        cblk_size,
        precinct_size,
        irreversible,
        progression,
        cinema_mode,
        mct,
        signed,
        fd,
        comment,
        plt,
    )

    ImageFile._save(im, fp, [ImageFile._Tile("jpeg2k", (0, 0) + im.size, 0, kind)])


# ------------------------------------------------------------
# Registry stuff


Image.register_open(Jpeg2KImageFile.format, Jpeg2KImageFile, _accept)
Image.register_save(Jpeg2KImageFile.format, _save)

Image.register_extensions(
    Jpeg2KImageFile.format, [".jp2", ".j2k", ".jpc", ".jpf", ".jpx", ".j2c"]
)

Image.register_mime(Jpeg2KImageFile.format, "image/jp2")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# a Tk display interface
#
# History:
# 96-04-08 fl   Created
# 96-09-06 fl   Added getimage method
# 96-11-01 fl   Rewritten, removed image attribute and crop method
# 97-05-09 fl   Use PyImagingPaste method instead of image type
# 97-05-12 fl   Minor tweaks to match the IFUNC95 interface
# 97-05-17 fl   Support the "pilbitmap" booster patch
# 97-06-05 fl   Added file= and data= argument to image constructors
# 98-03-09 fl   Added width and height methods to Image classes
# 98-07-02 fl   Use default mode for "P" images without palette attribute
# 98-07-02 fl   Explicitly destroy Tkinter image objects
# 99-07-24 fl   Support multiple Tk interpreters (from Greg Couch)
# 99-07-26 fl   Automatically hook into Tkinter (if possible)
# 99-08-15 fl   Hook uses _imagingtk instead of _imaging
#
# Copyright (c) 1997-1999 by Secret Labs AB
# Copyright (c) 1996-1997 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import tkinter
from io import BytesIO
from typing import Any

from . import Image, ImageFile

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ._typing import CapsuleType

# --------------------------------------------------------------------
# Check for Tkinter interface hooks


def _get_image_from_kw(kw: dict[str, Any]) -> ImageFile.ImageFile | None:
    source = None
    if "file" in kw:
        source = kw.pop("file")
    elif "data" in kw:
        source = BytesIO(kw.pop("data"))
    if not source:
        return None
    return Image.open(source)


def _pyimagingtkcall(
    command: str, photo: PhotoImage | tkinter.PhotoImage, ptr: CapsuleType
) -> None:
    tk = photo.tk
    try:
        tk.call(command, photo, repr(ptr))
    except tkinter.TclError:
        # activate Tkinter hook
        # may raise an error if it cannot attach to Tkinter
        from . import _imagingtk

        _imagingtk.tkinit(tk.interpaddr())
        tk.call(command, photo, repr(ptr))


# --------------------------------------------------------------------
# PhotoImage


class PhotoImage:
    """
    A Tkinter-compatible photo image.  This can be used
    everywhere Tkinter expects an image object.  If the image is an RGBA
    image, pixels having alpha 0 are treated as transparent.

    The constructor takes either a PIL image, or a mode and a size.
    Alternatively, you can use the ``file`` or ``data`` options to initialize
    the photo image object.

    :param image: Either a PIL image, or a mode string.  If a mode string is
                  used, a size must also be given.
    :param size: If the first argument is a mode string, this defines the size
                 of the image.
    :keyword file: A filename to load the image from (using
                   ``Image.open(file)``).
    :keyword data: An 8-bit string containing image data (as loaded from an
                   image file).
    """

    def __init__(
        self,
        image: Image.Image | str | None = None,
        size: tuple[int, int] | None = None,
        **kw: Any,
    ) -> None:
        # Tk compatibility: file or data
        if image is None:
            image = _get_image_from_kw(kw)

        if image is None:
            msg = "Image is required"
            raise ValueError(msg)
        elif isinstance(image, str):
            mode = image
            image = None

            if size is None:
                msg = "If first argument is mode, size is required"
                raise ValueError(msg)
        else:
            # got an image instead of a mode
            mode = image.mode
            if mode == "P":
                # palette mapped data
                image.apply_transparency()
                image.load()
                mode = image.palette.mode if image.palette else "RGB"
            size = image.size
            kw["width"], kw["height"] = size

        if mode not in ["1", "L", "RGB", "RGBA"]:
            mode = Image.getmodebase(mode)

        self.__mode = mode
        self.__size = size
        self.__photo = tkinter.PhotoImage(**kw)
        self.tk = self.__photo.tk
        if image:
            self.paste(image)

    def __del__(self) -> None:
        try:
            name = self.__photo.name
        except AttributeError:
            return
        self.__photo.name = None
        try:
            self.__photo.tk.call("image", "delete", name)
        except Exception:
            pass  # ignore internal errors

    def __str__(self) -> str:
        """
        Get the Tkinter photo image identifier.  This method is automatically
        called by Tkinter whenever a PhotoImage object is passed to a Tkinter
        method.

        :return: A Tkinter photo image identifier (a string).
        """
        return str(self.__photo)

    def width(self) -> int:
        """
        Get the width of the image.

        :return: The width, in pixels.
        """
        return self.__size[0]

    def height(self) -> int:
        """
        Get the height of the image.

        :return: The height, in pixels.
        """
        return self.__size[1]

    def paste(self, im: Image.Image) -> None:
        """
        Paste a PIL image into the photo image.  Note that this can
        be very slow if the photo image is displayed.

        :param im: A PIL image. The size must match the target region.  If the
                   mode does not match, the image is converted to the mode of
                   the bitmap image.
        """
        # convert to blittable
        ptr = im.getim()
        image = im.im
        if not image.isblock() or im.mode != self.__mode:
            block = Image.core.new_block(self.__mode, im.size)
            image.convert2(block, image)  # convert directly between buffers
            ptr = block.ptr

        _pyimagingtkcall("PyImagingPhoto", self.__photo, ptr)


# --------------------------------------------------------------------
# BitmapImage


class BitmapImage:
    """
    A Tkinter-compatible bitmap image.  This can be used everywhere Tkinter
    expects an image object.

    The given image must have mode "1".  Pixels having value 0 are treated as
    transparent.  Options, if any, are passed on to Tkinter.  The most commonly
    used option is ``foreground``, which is used to specify the color for the
    non-transparent parts.  See the Tkinter documentation for information on
    how to specify colours.

    :param image: A PIL image.
    """

    def __init__(self, image: Image.Image | None = None, **kw: Any) -> None:
        # Tk compatibility: file or data
        if image is None:
            image = _get_image_from_kw(kw)

        if image is None:
            msg = "Image is required"
            raise ValueError(msg)
        self.__mode = image.mode
        self.__size = image.size

        self.__photo = tkinter.BitmapImage(data=image.tobitmap(), **kw)

    def __del__(self) -> None:
        try:
            name = self.__photo.name
        except AttributeError:
            return
        self.__photo.name = None
        try:
            self.__photo.tk.call("image", "delete", name)
        except Exception:
            pass  # ignore internal errors

    def width(self) -> int:
        """
        Get the width of the image.

        :return: The width, in pixels.
        """
        return self.__size[0]

    def height(self) -> int:
        """
        Get the height of the image.

        :return: The height, in pixels.
        """
        return self.__size[1]

    def __str__(self) -> str:
        """
        Get the Tkinter bitmap image identifier.  This method is automatically
        called by Tkinter whenever a BitmapImage object is passed to a Tkinter
        method.

        :return: A Tkinter bitmap image identifier (a string).
        """
        return str(self.__photo)


def getimage(photo: PhotoImage) -> Image.Image:
    """Copies the contents of a PhotoImage to a PIL image memory."""
    im = Image.new("RGBA", (photo.width(), photo.height()))

    _pyimagingtkcall("PyImagingPhotoGet", photo, im.getim())

    return im
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# GD file handling
#
# History:
# 1996-04-12 fl   Created
#
# Copyright (c) 1997 by Secret Labs AB.
# Copyright (c) 1996 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#


"""
.. note::
    This format cannot be automatically recognized, so the
    class is not registered for use with :py:func:`PIL.Image.open()`.  To open a
    gd file, use the :py:func:`PIL.GdImageFile.open()` function instead.

.. warning::
    THE GD FORMAT IS NOT DESIGNED FOR DATA INTERCHANGE.  This
    implementation is provided for convenience and demonstrational
    purposes only.
"""

from __future__ import annotations

from typing import IO

from . import Image, ImageFile, ImagePalette, UnidentifiedImageError
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._typing import StrOrBytesPath


class GdImageFile(ImageFile.ImageFile):
    """
    Image plugin for the GD uncompressed format.  Note that this format
    is not supported by the standard :py:func:`PIL.Image.open()` function.  To use
    this plugin, you have to import the :py:mod:`PIL.GdImageFile` module and
    use the :py:func:`PIL.GdImageFile.open()` function.
    """

    format = "GD"
    format_description = "GD uncompressed images"

    def _open(self) -> None:
        # Header
        assert self.fp is not None

        s = self.fp.read(1037)

        if i16(s) not in [65534, 65535]:
            msg = "Not a valid GD 2.x .gd file"
            raise SyntaxError(msg)

        self._mode = "P"
        self._size = i16(s, 2), i16(s, 4)
        Image._decompression_bomb_check(self.size)

        true_color = s[6]
        true_color_offset = 2 if true_color else 0

        # transparency index
        tindex = i32(s, 7 + true_color_offset)
        if tindex < 256:
            self.info["transparency"] = tindex

        self.palette = ImagePalette.raw(
            "RGBX", s[7 + true_color_offset + 6 : 7 + true_color_offset + 6 + 256 * 4]
        )

        self.tile = [
            ImageFile._Tile(
                "raw",
                (0, 0) + self.size,
                7 + true_color_offset + 6 + 256 * 4,
                "L",
            )
        ]


def open(fp: StrOrBytesPath | IO[bytes], mode: str = "r") -> GdImageFile:
    """
    Load texture from a GD image file.

    :param fp: GD file name, or an opened file handle.
    :param mode: Optional mode.  In this version, if the mode argument
        is given, it must be "r".
    :returns: An image instance.
    :raises OSError: If the image could not be read.
    """
    if mode != "r":
        msg = "bad mode"
        raise ValueError(msg)

    try:
        return GdImageFile(fp)
    except SyntaxError as e:
        msg = "cannot identify this image file"
        raise UnidentifiedImageError(msg) from e
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# standard mode descriptors
#
# History:
# 2006-03-20 fl   Added
#
# Copyright (c) 2006 by Secret Labs AB.
# Copyright (c) 2006 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import sys
from functools import lru_cache
from typing import NamedTuple


class ModeDescriptor(NamedTuple):
    """Wrapper for mode strings."""

    mode: str
    bands: tuple[str, ...]
    basemode: str
    basetype: str
    typestr: str

    def __str__(self) -> str:
        return self.mode


@lru_cache
def getmode(mode: str) -> ModeDescriptor:
    """Gets a mode descriptor for the given mode."""
    endian = "<" if sys.byteorder == "little" else ">"

    modes = {
        # core modes
        # Bits need to be extended to bytes
        "1": ("L", "L", ("1",), "|b1"),
        "L": ("L", "L", ("L",), "|u1"),
        "I": ("L", "I", ("I",), f"{endian}i4"),
        "F": ("L", "F", ("F",), f"{endian}f4"),
        "P": ("P", "L", ("P",), "|u1"),
        "RGB": ("RGB", "L", ("R", "G", "B"), "|u1"),
        "RGBX": ("RGB", "L", ("R", "G", "B", "X"), "|u1"),
        "RGBA": ("RGB", "L", ("R", "G", "B", "A"), "|u1"),
        "CMYK": ("RGB", "L", ("C", "M", "Y", "K"), "|u1"),
        "YCbCr": ("RGB", "L", ("Y", "Cb", "Cr"), "|u1"),
        # UNDONE - unsigned |u1i1i1
        "LAB": ("RGB", "L", ("L", "A", "B"), "|u1"),
        "HSV": ("RGB", "L", ("H", "S", "V"), "|u1"),
        # extra experimental modes
        "RGBa": ("RGB", "L", ("R", "G", "B", "a"), "|u1"),
        "LA": ("L", "L", ("L", "A"), "|u1"),
        "La": ("L", "L", ("L", "a"), "|u1"),
        "PA": ("RGB", "L", ("P", "A"), "|u1"),
    }
    if mode in modes:
        base_mode, base_type, bands, type_str = modes[mode]
        return ModeDescriptor(mode, bands, base_mode, base_type, type_str)

    mapping_modes = {
        # I;16 == I;16L, and I;32 == I;32L
        "I;16": "<u2",
        "I;16S": "<i2",
        "I;16L": "<u2",
        "I;16LS": "<i2",
        "I;16B": ">u2",
        "I;16BS": ">i2",
        "I;16N": f"{endian}u2",
        "I;16NS": f"{endian}i2",
        "I;32": "<u4",
        "I;32B": ">u4",
        "I;32L": "<u4",
        "I;32S": "<i4",
        "I;32BS": ">i4",
        "I;32LS": "<i4",
    }

    type_str = mapping_modes[mode]
    return ModeDescriptor(mode, ("I",), "L", "L", type_str)
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# HDF5 stub adapter
#
# Copyright (c) 2000-2003 by Fredrik Lundh
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
    Install application-specific HDF5 image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler


# --------------------------------------------------------------------
# Image adapter


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"\x89HDF\r\n\x1a\n")


class HDF5StubImageFile(ImageFile.StubImageFile):
    format = "HDF5"
    format_description = "HDF5"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(8)):
            msg = "Not an HDF file"
            raise SyntaxError(msg)

        self.fp.seek(-8, os.SEEK_CUR)

        # make something up
        self._mode = "F"
        self._size = 1, 1

    def _load(self) -> ImageFile.StubHandler | None:
        return _handler


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if _handler is None or not hasattr(_handler, "save"):
        msg = "HDF5 save handler not installed"
        raise OSError(msg)
    _handler.save(im, fp, filename)


# --------------------------------------------------------------------
# Registry

Image.register_open(HDF5StubImageFile.format, HDF5StubImageFile, _accept)
Image.register_save(HDF5StubImageFile.format, _save)

Image.register_extensions(HDF5StubImageFile.format, [".h5", ".hdf"])
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# base class for image file handlers
#
# history:
# 1995-09-09 fl   Created
# 1996-03-11 fl   Fixed load mechanism.
# 1996-04-15 fl   Added pcx/xbm decoders.
# 1996-04-30 fl   Added encoders.
# 1996-12-14 fl   Added load helpers
# 1997-01-11 fl   Use encode_to_file where possible
# 1997-08-27 fl   Flush output in _save
# 1998-03-05 fl   Use memory mapping for some modes
# 1999-02-04 fl   Use memory mapping also for "I;16" and "I;16B"
# 1999-05-31 fl   Added image parser
# 2000-10-12 fl   Set readonly flag on memory-mapped images
# 2002-03-20 fl   Use better messages for common decoder errors
# 2003-04-21 fl   Fall back on mmap/map_buffer if map is not available
# 2003-10-30 fl   Added StubImageFile class
# 2004-02-25 fl   Made incremental parser more robust
#
# Copyright (c) 1997-2004 by Secret Labs AB
# Copyright (c) 1995-2004 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import abc
import io
import itertools
import logging
import os
import struct
from typing import IO, Any, NamedTuple, cast

from . import ExifTags, Image
from ._util import DeferredError, is_path

TYPE_CHECKING = False
if TYPE_CHECKING:
    from ._typing import StrOrBytesPath

logger = logging.getLogger(__name__)

MAXBLOCK = 65536
"""
By default, Pillow processes image data in blocks. This helps to prevent excessive use
of resources. Codecs may disable this behaviour with ``_pulls_fd`` or ``_pushes_fd``.

When reading an image, this is the number of bytes to read at once.

When writing an image, this is the number of bytes to write at once.
If the image width times 4 is greater, then that will be used instead.
Plugins may also set a greater number.

User code may set this to another number.
"""

SAFEBLOCK = 1024 * 1024

LOAD_TRUNCATED_IMAGES = False
"""Whether or not to load truncated image files. User code may change this."""

ERRORS = {
    -1: "image buffer overrun error",
    -2: "decoding error",
    -3: "unknown error",
    -8: "bad configuration",
    -9: "out of memory error",
}
"""
Dict of known error codes returned from :meth:`.PyDecoder.decode`,
:meth:`.PyEncoder.encode` :meth:`.PyEncoder.encode_to_pyfd` and
:meth:`.PyEncoder.encode_to_file`.
"""


#
# --------------------------------------------------------------------
# Helpers


def _get_oserror(error: int, *, encoder: bool) -> OSError:
    try:
        msg = Image.core.getcodecstatus(error)
    except AttributeError:
        msg = ERRORS.get(error)
    if not msg:
        msg = f"{'encoder' if encoder else 'decoder'} error {error}"
    msg += f" when {'writing' if encoder else 'reading'} image file"
    return OSError(msg)


def _tilesort(t: _Tile) -> int:
    # sort on offset
    return t[2]


class _Tile(NamedTuple):
    codec_name: str
    extents: tuple[int, int, int, int] | None
    offset: int = 0
    args: tuple[Any, ...] | str | None = None


#
# --------------------------------------------------------------------
# ImageFile base class


class ImageFile(Image.Image):
    """Base class for image file format handlers."""

    def __init__(
        self, fp: StrOrBytesPath | IO[bytes], filename: str | bytes | None = None
    ) -> None:
        super().__init__()

        self._min_frame = 0

        self.custom_mimetype: str | None = None

        self.tile: list[_Tile] = []
        """ A list of tile descriptors """

        self.readonly = 1  # until we know better

        self.decoderconfig: tuple[Any, ...] = ()
        self.decodermaxblock = MAXBLOCK

        self.fp: IO[bytes] | None
        self._fp: IO[bytes] | DeferredError
        if is_path(fp):
            # filename
            self.fp = open(fp, "rb")
            self.filename = os.fspath(fp)
            self._exclusive_fp = True
        else:
            # stream
            self.fp = cast(IO[bytes], fp)
            self.filename = filename if filename is not None else ""
            # can be overridden
            self._exclusive_fp = False

        try:
            try:
                self._open()

                if isinstance(self, StubImageFile):
                    if loader := self._load():
                        loader.open(self)
            except (
                IndexError,  # end of data
                TypeError,  # end of data (ord)
                KeyError,  # unsupported mode
                EOFError,  # got header but not the first frame
                struct.error,
            ) as v:
                raise SyntaxError(v) from v

            if not self.mode or self.size[0] <= 0 or self.size[1] <= 0:
                msg = "not identified by this driver"
                raise SyntaxError(msg)
        except BaseException:
            # close the file only if we have opened it this constructor
            if self._exclusive_fp:
                self.fp.close()
            raise

    def _open(self) -> None:
        pass

    # Context manager support
    def __enter__(self) -> ImageFile:
        return self

    def _close_fp(self) -> None:
        if getattr(self, "_fp", False) and not isinstance(self._fp, DeferredError):
            if self._fp != self.fp:
                self._fp.close()
            self._fp = DeferredError(ValueError("Operation on closed image"))
        if self.fp:
            self.fp.close()

    def __exit__(self, *args: object) -> None:
        if getattr(self, "_exclusive_fp", False):
            self._close_fp()
        self.fp = None

    def close(self) -> None:
        """
        Closes the file pointer, if possible.

        This operation will destroy the image core and release its memory.
        The image data will be unusable afterward.

        This function is required to close images that have multiple frames or
        have not had their file read and closed by the
        :py:meth:`~PIL.Image.Image.load` method. See :ref:`file-handling` for
        more information.
        """
        try:
            self._close_fp()
            self.fp = None
        except Exception as msg:
            logger.debug("Error closing: %s", msg)

        super().close()

    def get_child_images(self) -> list[ImageFile]:
        child_images = []
        exif = self.getexif()
        ifds = []
        if ExifTags.Base.SubIFDs in exif:
            subifd_offsets = exif[ExifTags.Base.SubIFDs]
            if subifd_offsets:
                if not isinstance(subifd_offsets, tuple):
                    subifd_offsets = (subifd_offsets,)
                ifds = [
                    (exif._get_ifd_dict(subifd_offset), subifd_offset)
                    for subifd_offset in subifd_offsets
                ]
        ifd1 = exif.get_ifd(ExifTags.IFD.IFD1)
        if ifd1 and ifd1.get(ExifTags.Base.JpegIFOffset):
            assert exif._info is not None
            ifds.append((ifd1, exif._info.next))

        offset = None
        for ifd, ifd_offset in ifds:
            assert self.fp is not None
            current_offset = self.fp.tell()
            if offset is None:
                offset = current_offset

            fp = self.fp
            if ifd is not None:
                thumbnail_offset = ifd.get(ExifTags.Base.JpegIFOffset)
                if thumbnail_offset is not None:
                    thumbnail_offset += getattr(self, "_exif_offset", 0)
                    self.fp.seek(thumbnail_offset)

                    length = ifd.get(ExifTags.Base.JpegIFByteCount)
                    assert isinstance(length, int)
                    data = self.fp.read(length)
                    fp = io.BytesIO(data)

            with Image.open(fp) as im:
                from . import TiffImagePlugin

                if thumbnail_offset is None and isinstance(
                    im, TiffImagePlugin.TiffImageFile
                ):
                    im._frame_pos = [ifd_offset]
                    im._seek(0)
                im.load()
                child_images.append(im)

        if offset is not None:
            assert self.fp is not None
            self.fp.seek(offset)
        return child_images

    def get_format_mimetype(self) -> str | None:
        if self.custom_mimetype:
            return self.custom_mimetype
        if self.format is not None:
            return Image.MIME.get(self.format.upper())
        return None

    def __getstate__(self) -> list[Any]:
        return super().__getstate__() + [self.filename]

    def __setstate__(self, state: list[Any]) -> None:
        self.tile = []
        if len(state) > 5:
            self.filename = state[5]
        super().__setstate__(state)

    def verify(self) -> None:
        """Check file integrity"""

        # raise exception if something's wrong.  must be called
        # directly after open, and closes file when finished.
        if self._exclusive_fp and self.fp:
            self.fp.close()
        self.fp = None

    def load(self) -> Image.core.PixelAccess | None:
        """Load image data based on tile list"""

        if not self.tile and self._im is None:
            msg = "cannot load this image"
            raise OSError(msg)

        pixel = Image.Image.load(self)
        if not self.tile:
            return pixel

        self.map: mmap.mmap | None = None
        use_mmap = self.filename and len(self.tile) == 1

        assert self.fp is not None
        readonly = 0

        # look for read/seek overrides
        if hasattr(self, "load_read"):
            read = self.load_read
            # don't use mmap if there are custom read/seek functions
            use_mmap = False
        else:
            read = self.fp.read

        if hasattr(self, "load_seek"):
            seek = self.load_seek
            use_mmap = False
        else:
            seek = self.fp.seek

        if use_mmap:
            # try memory mapping
            decoder_name, extents, offset, args = self.tile[0]
            if isinstance(args, str):
                args = (args, 0, 1)
            if (
                decoder_name == "raw"
                and isinstance(args, tuple)
                and len(args) >= 3
                and args[0] == self.mode
                and args[0] in Image._MAPMODES
            ):
                if offset < 0:
                    msg = "Tile offset cannot be negative"
                    raise ValueError(msg)
                try:
                    # use mmap, if possible
                    import mmap

                    with open(self.filename) as fp:
                        self.map = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                    if offset + self.size[1] * args[1] > self.map.size():
                        msg = "buffer is not large enough"
                        raise OSError(msg)
                    self.im = Image.core.map_buffer(
                        self.map, self.size, decoder_name, offset, args
                    )
                    readonly = 1
                    # After trashing self.im,
                    # we might need to reload the palette data.
                    if self.palette:
                        self.palette.dirty = 1
                except (AttributeError, OSError, ImportError):
                    self.map = None

        self.load_prepare()
        err_code = -3  # initialize to unknown error
        if not self.map:
            # sort tiles in file order
            self.tile.sort(key=_tilesort)

            # FIXME: This is a hack to handle TIFF's JpegTables tag.
            prefix = getattr(self, "tile_prefix", b"")

            # Remove consecutive duplicates that only differ by their offset
            self.tile = [
                list(tiles)[-1]
                for _, tiles in itertools.groupby(
                    self.tile, lambda tile: (tile[0], tile[1], tile[3])
                )
            ]
            for i, (decoder_name, extents, offset, args) in enumerate(self.tile):
                seek(offset)
                decoder = Image._getdecoder(
                    self.mode, decoder_name, args, self.decoderconfig
                )
                try:
                    decoder.setimage(self.im, extents)
                    if decoder.pulls_fd:
                        decoder.setfd(self.fp)
                        err_code = decoder.decode(b"")[1]
                    else:
                        b = prefix
                        while True:
                            read_bytes = self.decodermaxblock
                            if i + 1 < len(self.tile):
                                next_offset = self.tile[i + 1].offset
                                if next_offset > offset:
                                    read_bytes = next_offset - offset
                            try:
                                s = read(read_bytes)
                            except (IndexError, struct.error) as e:
                                # truncated png/gif
                                if LOAD_TRUNCATED_IMAGES:
                                    break
                                else:
                                    msg = "image file is truncated"
                                    raise OSError(msg) from e

                            if not s:  # truncated jpeg
                                if LOAD_TRUNCATED_IMAGES:
                                    break
                                else:
                                    msg = (
                                        "image file is truncated "
                                        f"({len(b)} bytes not processed)"
                                    )
                                    raise OSError(msg)

                            b = b + s
                            n, err_code = decoder.decode(b)
                            if n < 0:
                                break
                            b = b[n:]
                finally:
                    # Need to cleanup here to prevent leaks
                    decoder.cleanup()

        self.tile = []
        self.readonly = readonly

        self.load_end()

        if self._exclusive_fp and self._close_exclusive_fp_after_loading:
            self.fp.close()
        self.fp = None

        if not self.map and not LOAD_TRUNCATED_IMAGES and err_code < 0:
            # still raised if decoder fails to return anything
            raise _get_oserror(err_code, encoder=False)

        return Image.Image.load(self)

    def load_prepare(self) -> None:
        # create image memory if necessary
        if self._im is None:
            self.im = Image.core.new(self.mode, self.size)
        # create palette (optional)
        if self.mode == "P":
            Image.Image.load(self)

    def load_end(self) -> None:
        # may be overridden
        pass

    # may be defined for contained formats
    # def load_seek(self, pos: int) -> None:
    #     pass

    # may be defined for blocked formats (e.g. PNG)
    # def load_read(self, read_bytes: int) -> bytes:
    #     pass

    def _seek_check(self, frame: int) -> bool:
        if (
            frame < self._min_frame
            # Only check upper limit on frames if additional seek operations
            # are not required to do so
            or (
                not (hasattr(self, "_n_frames") and self._n_frames is None)
                and frame >= getattr(self, "n_frames") + self._min_frame
            )
        ):
            msg = "attempt to seek outside sequence"
            raise EOFError(msg)

        return self.tell() != frame


class StubHandler(abc.ABC):
    def open(self, im: StubImageFile) -> None:
        pass

    @abc.abstractmethod
    def load(self, im: StubImageFile) -> Image.Image:
        pass


class StubImageFile(ImageFile, metaclass=abc.ABCMeta):
    """
    Base class for stub image loaders.

    A stub loader is an image loader that can identify files of a
    certain format, but relies on external code to load the file.
    """

    @abc.abstractmethod
    def _open(self) -> None:
        pass

    def load(self) -> Image.core.PixelAccess | None:
        loader = self._load()
        if loader is None:
            msg = f"cannot find loader for this {self.format} file"
            raise OSError(msg)
        image = loader.load(self)
        assert image is not None
        # become the other object (!)
        self.__class__ = image.__class__  # type: ignore[assignment]
        self.__dict__ = image.__dict__
        return image.load()

    @abc.abstractmethod
    def _load(self) -> StubHandler | None:
        """(Hook) Find actual image loader."""
        pass


class Parser:
    """
    Incremental image parser.  This class implements the standard
    feed/close consumer interface.
    """

    incremental = None
    image: Image.Image | None = None
    data: bytes | None = None
    decoder: Image.core.ImagingDecoder | PyDecoder | None = None
    offset = 0
    finished = 0

    def reset(self) -> None:
        """
        (Consumer) Reset the parser.  Note that you can only call this
        method immediately after you've created a parser; parser
        instances cannot be reused.
        """
        assert self.data is None, "cannot reuse parsers"

    def feed(self, data: bytes) -> None:
        """
        (Consumer) Feed data to the parser.

        :param data: A string buffer.
        :exception OSError: If the parser failed to parse the image file.
        """
        # collect data

        if self.finished:
            return

        if self.data is None:
            self.data = data
        else:
            self.data = self.data + data

        # parse what we have
        if self.decoder:
            if self.offset > 0:
                # skip header
                skip = min(len(self.data), self.offset)
                self.data = self.data[skip:]
                self.offset = self.offset - skip
                if self.offset > 0 or not self.data:
                    return

            n, e = self.decoder.decode(self.data)

            if n < 0:
                # end of stream
                self.data = None
                self.finished = 1
                if e < 0:
                    # decoding error
                    self.image = None
                    raise _get_oserror(e, encoder=False)
                else:
                    # end of image
                    return
            self.data = self.data[n:]

        elif self.image:
            # if we end up here with no decoder, this file cannot
            # be incrementally parsed.  wait until we've gotten all
            # available data
            pass

        else:
            # attempt to open this file
            try:
                with io.BytesIO(self.data) as fp:
                    im = Image.open(fp)
            except OSError:
                pass  # not enough data
            else:
                flag = hasattr(im, "load_seek") or hasattr(im, "load_read")
                if not flag and len(im.tile) == 1:
                    # initialize decoder
                    im.load_prepare()
                    d, e, o, a = im.tile[0]
                    im.tile = []
                    self.decoder = Image._getdecoder(im.mode, d, a, im.decoderconfig)
                    self.decoder.setimage(im.im, e)

                    # calculate decoder offset
                    self.offset = o
                    if self.offset <= len(self.data):
                        self.data = self.data[self.offset :]
                        self.offset = 0

                self.image = im

    def __enter__(self) -> Parser:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> Image.Image:
        """
        (Consumer) Close the stream.

        :returns: An image object.
        :exception OSError: If the parser failed to parse the image file either
                            because it cannot be identified or cannot be
                            decoded.
        """
        # finish decoding
        if self.decoder:
            # get rid of what's left in the buffers
            self.feed(b"")
            self.data = self.decoder = None
            if not self.finished:
                msg = "image was incomplete"
                raise OSError(msg)
        if not self.image:
            msg = "cannot parse this image"
            raise OSError(msg)
        if self.data:
            # incremental parsing not possible; reopen the file
            # not that we have all data
            with io.BytesIO(self.data) as fp:
                try:
                    self.image = Image.open(fp)
                finally:
                    self.image.load()
        return self.image


# --------------------------------------------------------------------


def _save(im: Image.Image, fp: IO[bytes], tile: list[_Tile], bufsize: int = 0) -> None:
    """Helper to save image based on tile list

    :param im: Image object.
    :param fp: File object.
    :param tile: Tile list.
    :param bufsize: Optional buffer size
    """

    im.load()
    if not hasattr(im, "encoderconfig"):
        im.encoderconfig = ()
    tile.sort(key=_tilesort)
    # FIXME: make MAXBLOCK a configuration parameter
    # It would be great if we could have the encoder specify what it needs
    # But, it would need at least the image size in most cases. RawEncode is
    # a tricky case.
    bufsize = max(MAXBLOCK, bufsize, im.size[0] * 4)  # see RawEncode.c
    try:
        fh = fp.fileno()
        fp.flush()
        _encode_tile(im, fp, tile, bufsize, fh)
    except (AttributeError, io.UnsupportedOperation) as exc:
        _encode_tile(im, fp, tile, bufsize, None, exc)
    if hasattr(fp, "flush"):
        fp.flush()


def _encode_tile(
    im: Image.Image,
    fp: IO[bytes],
    tile: list[_Tile],
    bufsize: int,
    fh: int | None,
    exc: BaseException | None = None,
) -> None:
    for encoder_name, extents, offset, args in tile:
        if offset > 0:
            fp.seek(offset)
        encoder = Image._getencoder(im.mode, encoder_name, args, im.encoderconfig)
        try:
            encoder.setimage(im.im, extents)
            if encoder.pushes_fd:
                encoder.setfd(fp)
                errcode = encoder.encode_to_pyfd()[1]
            else:
                if exc:
                    # compress to Python file-compatible object
                    while True:
                        errcode, data = encoder.encode(bufsize)[1:]
                        fp.write(data)
                        if errcode:
                            break
                else:
                    # slight speedup: compress to real file object
                    assert fh is not None
                    errcode = encoder.encode_to_file(fh, bufsize)
            if errcode < 0:
                raise _get_oserror(errcode, encoder=True) from exc
        finally:
            encoder.cleanup()


def _safe_read(fp: IO[bytes], size: int) -> bytes:
    """
    Reads large blocks in a safe way.  Unlike fp.read(n), this function
    doesn't trust the user.  If the requested size is larger than
    SAFEBLOCK, the file is read block by block.

    :param fp: File handle.  Must implement a <b>read</b> method.
    :param size: Number of bytes to read.
    :returns: A string containing <i>size</i> bytes of data.

    Raises an OSError if the file is truncated and the read cannot be completed

    """
    if size <= 0:
        return b""
    if size <= SAFEBLOCK:
        data = fp.read(size)
        if len(data) < size:
            msg = "Truncated File Read"
            raise OSError(msg)
        return data
    blocks: list[bytes] = []
    remaining_size = size
    while remaining_size > 0:
        block = fp.read(min(remaining_size, SAFEBLOCK))
        if not block:
            break
        blocks.append(block)
        remaining_size -= len(block)
    if sum(len(block) for block in blocks) < size:
        msg = "Truncated File Read"
        raise OSError(msg)
    return b"".join(blocks)


class PyCodecState:
    def __init__(self) -> None:
        self.xsize = 0
        self.ysize = 0
        self.xoff = 0
        self.yoff = 0

    def extents(self) -> tuple[int, int, int, int]:
        return self.xoff, self.yoff, self.xoff + self.xsize, self.yoff + self.ysize


class PyCodec:
    fd: IO[bytes] | None

    def __init__(self, mode: str, *args: Any) -> None:
        self.im: Image.core.ImagingCore | None = None
        self.state = PyCodecState()
        self.fd = None
        self.mode = mode
        self.init(args)

    def init(self, args: tuple[Any, ...]) -> None:
        """
        Override to perform codec specific initialization

        :param args: Tuple of arg items from the tile entry
        :returns: None
        """
        self.args = args

    def cleanup(self) -> None:
        """
        Override to perform codec specific cleanup

        :returns: None
        """
        pass

    def setfd(self, fd: IO[bytes]) -> None:
        """
        Called from ImageFile to set the Python file-like object

        :param fd: A Python file-like object
        :returns: None
        """
        self.fd = fd

    def setimage(
        self,
        im: Image.core.ImagingCore,
        extents: tuple[int, int, int, int] | None = None,
    ) -> None:
        """
        Called from ImageFile to set the core output image for the codec

        :param im: A core image object
        :param extents: a 4 tuple of (x0, y0, x1, y1) defining the rectangle
            for this tile
        :returns: None
        """

        # following c code
        self.im = im

        if extents:
            x0, y0, x1, y1 = extents

            if x0 < 0 or y0 < 0 or x1 > self.im.size[0] or y1 > self.im.size[1]:
                msg = "Tile cannot extend outside image"
                raise ValueError(msg)

            self.state.xoff = x0
            self.state.yoff = y0
            self.state.xsize = x1 - x0
            self.state.ysize = y1 - y0
        else:
            self.state.xsize, self.state.ysize = self.im.size

        if self.state.xsize <= 0 or self.state.ysize <= 0:
            msg = "Size must be positive"
            raise ValueError(msg)


class PyDecoder(PyCodec):
    """
    Python implementation of a format decoder. Override this class and
    add the decoding logic in the :meth:`decode` method.

    See :ref:`Writing Your Own File Codec in Python<file-codecs-py>`
    """

    _pulls_fd = False

    @property
    def pulls_fd(self) -> bool:
        return self._pulls_fd

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        """
        Override to perform the decoding process.

        :param buffer: A bytes object with the data to be decoded.
        :returns: A tuple of ``(bytes consumed, errcode)``.
            If finished with decoding return -1 for the bytes consumed.
            Err codes are from :data:`.ImageFile.ERRORS`.
        """
        msg = "unavailable in base decoder"
        raise NotImplementedError(msg)

    def set_as_raw(
        self,
        data: bytes | bytearray,
        rawmode: str | None = None,
        extra: tuple[Any, ...] = (),
    ) -> None:
        """
        Convenience method to set the internal image from a stream of raw data

        :param data: Bytes to be set
        :param rawmode: The rawmode to be used for the decoder.
            If not specified, it will default to the mode of the image
        :param extra: Extra arguments for the decoder.
        :returns: None
        """

        if not rawmode:
            rawmode = self.mode
        d = Image._getdecoder(self.mode, "raw", rawmode, extra)
        assert self.im is not None
        d.setimage(self.im, self.state.extents())
        s = d.decode(data)

        if s[0] >= 0:
            msg = "not enough image data"
            raise ValueError(msg)
        if s[1] != 0:
            msg = "cannot decode image data"
            raise ValueError(msg)


class PyEncoder(PyCodec):
    """
    Python implementation of a format encoder. Override this class and
    add the decoding logic in the :meth:`encode` method.

    See :ref:`Writing Your Own File Codec in Python<file-codecs-py>`
    """

    _pushes_fd = False

    @property
    def pushes_fd(self) -> bool:
        return self._pushes_fd

    def encode(self, bufsize: int) -> tuple[int, int, bytes]:
        """
        Override to perform the encoding process.

        :param bufsize: Buffer size.
        :returns: A tuple of ``(bytes encoded, errcode, bytes)``.
            If finished with encoding return 1 for the error code.
            Err codes are from :data:`.ImageFile.ERRORS`.
        """
        msg = "unavailable in base encoder"
        raise NotImplementedError(msg)

    def encode_to_pyfd(self) -> tuple[int, int]:
        """
        If ``pushes_fd`` is ``True``, then this method will be used,
        and ``encode()`` will only be called once.

        :returns: A tuple of ``(bytes consumed, errcode)``.
            Err codes are from :data:`.ImageFile.ERRORS`.
        """
        if not self.pushes_fd:
            return 0, -8  # bad configuration
        bytes_consumed, errcode, data = self.encode(0)
        if data:
            assert self.fd is not None
            self.fd.write(data)
        return bytes_consumed, errcode

    def encode_to_file(self, fh: int, bufsize: int) -> int:
        """
        :param fh: File handle.
        :param bufsize: Buffer size.

        :returns: If finished successfully, return 0.
            Otherwise, return an error code. Err codes are from
            :data:`.ImageFile.ERRORS`.
        """
        errcode = 0
        while errcode == 0:
            status, errcode, buf = self.encode(bufsize)
            if status > 0:
                os.write(fh, buf[status:])
        return errcode
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# a class to read from a container file
#
# History:
# 1995-06-18 fl     Created
# 1995-09-07 fl     Added readline(), readlines()
#
# Copyright (c) 1997-2001 by Secret Labs AB
# Copyright (c) 1995 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io
from collections.abc import Iterable
from typing import IO, AnyStr, NoReturn


class ContainerIO(IO[AnyStr]):
    """
    A file object that provides read access to a part of an existing
    file (for example a TAR file).
    """

    def __init__(self, file: IO[AnyStr], offset: int, length: int) -> None:
        """
        Create file object.

        :param file: Existing file.
        :param offset: Start of region, in bytes.
        :param length: Size of region, in bytes.
        """
        self.fh: IO[AnyStr] = file
        self.pos = 0
        self.offset = offset
        self.length = length
        self.fh.seek(offset)

    ##
    # Always false.

    def isatty(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, mode: int = io.SEEK_SET) -> int:
        """
        Move file pointer.

        :param offset: Offset in bytes.
        :param mode: Starting position. Use 0 for beginning of region, 1
           for current offset, and 2 for end of region.  You cannot move
           the pointer outside the defined region.
        :returns: Offset from start of region, in bytes.
        """
        if mode == 1:
            self.pos = self.pos + offset
        elif mode == 2:
            self.pos = self.length + offset
        else:
            self.pos = offset
        # clamp
        self.pos = max(0, min(self.pos, self.length))
        self.fh.seek(self.offset + self.pos)
        return self.pos

    def tell(self) -> int:
        """
        Get current file pointer.

        :returns: Offset from start of region, in bytes.
        """
        return self.pos

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> AnyStr:
        """
        Read data.

        :param n: Number of bytes to read. If omitted, zero or negative,
            read until end of region.
        :returns: An 8-bit string.
        """
        if n > 0:
            n = min(n, self.length - self.pos)
        else:
            n = self.length - self.pos
        if n <= 0:  # EOF
            return b"" if "b" in self.fh.mode else ""  # type: ignore[return-value]
        self.pos = self.pos + n
        return self.fh.read(n)

    def readline(self, n: int = -1) -> AnyStr:
        """
        Read a line of text.

        :param n: Number of bytes to read. If omitted, zero or negative,
            read until end of line.
        :returns: An 8-bit string.
        """
        s: AnyStr = b"" if "b" in self.fh.mode else ""  # type: ignore[assignment]
        newline_character = b"\n" if "b" in self.fh.mode else "\n"
        while True:
            c = self.read(1)
            if not c:
                break
            s = s + c
            if c == newline_character or len(s) == n:
                break
        return s

    def readlines(self, n: int | None = -1) -> list[AnyStr]:
        """
        Read multiple lines of text.

        :param n: Number of lines to read. If omitted, zero, negative or None,
            read until end of region.
        :returns: A list of 8-bit strings.
        """
        lines = []
        while True:
            s = self.readline()
            if not s:
                break
            lines.append(s)
            if len(lines) == n:
                break
        return lines

    def writable(self) -> bool:
        return False

    def write(self, b: AnyStr) -> NoReturn:
        raise NotImplementedError()

    def writelines(self, lines: Iterable[AnyStr]) -> NoReturn:
        raise NotImplementedError()

    def truncate(self, size: int | None = None) -> int:
        raise NotImplementedError()

    def __enter__(self) -> ContainerIO[AnyStr]:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __iter__(self) -> ContainerIO[AnyStr]:
        return self

    def __next__(self) -> AnyStr:
        line = self.readline()
        if not line:
            msg = "end of region"
            raise StopIteration(msg)
        return line

    def fileno(self) -> int:
        return self.fh.fileno()

    def flush(self) -> None:
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# GRIB stub adapter
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
    Install application-specific GRIB image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler


# --------------------------------------------------------------------
# Image adapter


def _accept(prefix: bytes) -> bool:
    return len(prefix) >= 8 and prefix.startswith(b"GRIB") and prefix[7] == 1


class GribStubImageFile(ImageFile.StubImageFile):
    format = "GRIB"
    format_description = "GRIB"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(8)):
            msg = "Not a GRIB file"
            raise SyntaxError(msg)

        self.fp.seek(-8, os.SEEK_CUR)

        # make something up
        self._mode = "F"
        self._size = 1, 1

    def _load(self) -> ImageFile.StubHandler | None:
        return _handler


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if _handler is None or not hasattr(_handler, "save"):
        msg = "GRIB save handler not installed"
        raise OSError(msg)
    _handler.save(im, fp, filename)


# --------------------------------------------------------------------
# Registry

Image.register_open(GribStubImageFile.format, GribStubImageFile, _accept)
Image.register_save(GribStubImageFile.format, _save)

Image.register_extension(GribStubImageFile.format, ".grib")
<|endoftext|># A binary morphology add-on for the Python Imaging Library
#
# History:
#   2014-06-04 Initial version.
#
# Copyright (c) 2014 Dov Grobgeld <dov.grobgeld@gmail.com>
from __future__ import annotations

import re

from . import Image, _imagingmorph

LUT_SIZE = 1 << 9

# fmt: off
ROTATION_MATRIX = [
    6, 3, 0,
    7, 4, 1,
    8, 5, 2,
]
MIRROR_MATRIX = [
    2, 1, 0,
    5, 4, 3,
    8, 7, 6,
]
# fmt: on


class LutBuilder:
    """A class for building a MorphLut from a descriptive language

    The input patterns is a list of a strings sequences like these::

        4:(...
           .1.
           111)->1

    (whitespaces including linebreaks are ignored). The option 4
    describes a series of symmetry operations (in this case a
    4-rotation), the pattern is described by:

    - . or X - Ignore
    - 1 - Pixel is on
    - 0 - Pixel is off

    The result of the operation is described after "->" string.

    The default is to return the current pixel value, which is
    returned if no other match is found.

    Operations:

    - 4 - 4 way rotation
    - N - Negate
    - 1 - Dummy op for no other operation (an op must always be given)
    - M - Mirroring

    Example::

        lb = LutBuilder(patterns = ["4:(... .1. 111)->1"])
        lut = lb.build_lut()

    """

    def __init__(
        self, patterns: list[str] | None = None, op_name: str | None = None
    ) -> None:
        """
        :param patterns: A list of input patterns, or None.
        :param op_name: The name of a known pattern. One of "corner", "dilation4",
           "dilation8", "erosion4", "erosion8" or "edge".
        :exception Exception: If the op_name is not recognized.
        """
        self.lut: bytearray | None = None
        if op_name is not None:
            known_patterns = {
                "corner": ["1:(... ... ...)->0", "4:(00. 01. ...)->1"],
                "dilation4": ["4:(... .0. .1.)->1"],
                "dilation8": ["4:(... .0. .1.)->1", "4:(... .0. ..1)->1"],
                "erosion4": ["4:(... .1. .0.)->0"],
                "erosion8": ["4:(... .1. .0.)->0", "4:(... .1. ..0)->0"],
                "edge": [
                    "1:(... ... ...)->0",
                    "4:(.0. .1. ...)->1",
                    "4:(01. .1. ...)->1",
                ],
            }
            if op_name not in known_patterns:
                msg = f"Unknown pattern {op_name}!"
                raise Exception(msg)

            self.patterns = known_patterns[op_name]
        elif patterns is not None:
            self.patterns = patterns
        else:
            self.patterns = []

    def add_patterns(self, patterns: list[str]) -> None:
        """
        Append to list of patterns.

        :param patterns: Additional patterns.
        """
        self.patterns += patterns

    def build_default_lut(self) -> bytearray:
        """
        Set the current LUT, and return it.

        This is the default LUT that patterns will be applied against when building.
        """
        symbols = [0, 1]
        m = 1 << 4  # pos of current pixel
        self.lut = bytearray(symbols[(i & m) > 0] for i in range(LUT_SIZE))
        return self.lut

    def get_lut(self) -> bytearray | None:
        """
        Returns the current LUT
        """
        return self.lut

    def _string_permute(self, pattern: str, permutation: list[int]) -> str:
        """Takes a pattern and a permutation and returns the
        string permuted according to the permutation list.
        """
        assert len(permutation) == 9
        return "".join(pattern[p] for p in permutation)

    def _pattern_permute(
        self, basic_pattern: str, options: str, basic_result: int
    ) -> list[tuple[str, int]]:
        """Takes a basic pattern and its result and clones
        the pattern according to the modifications described in the $options
        parameter. It returns a list of all cloned patterns."""
        patterns = [(basic_pattern, basic_result)]

        # rotations
        if "4" in options:
            res = patterns[-1][1]
            for i in range(4):
                patterns.append(
                    (self._string_permute(patterns[-1][0], ROTATION_MATRIX), res)
                )
        # mirror
        if "M" in options:
            n = len(patterns)
            for pattern, res in patterns[:n]:
                patterns.append((self._string_permute(pattern, MIRROR_MATRIX), res))

        # negate
        if "N" in options:
            n = len(patterns)
            for pattern, res in patterns[:n]:
                # Swap 0 and 1
                pattern = pattern.replace("0", "Z").replace("1", "0").replace("Z", "1")
                res = 1 - int(res)
                patterns.append((pattern, res))

        return patterns

    def build_lut(self) -> bytearray:
        """Compile all patterns into a morphology LUT, and return it.

        This is the data to be passed into MorphOp."""
        self.build_default_lut()
        assert self.lut is not None
        patterns = []

        # Parse and create symmetries of the patterns strings
        for p in self.patterns:
            m = re.search(r"(\w):?\s*\((.+?)\)\s*->\s*(\d)", p.replace("\n", ""))
            if not m:
                msg = 'Syntax error in pattern "' + p + '"'
                raise Exception(msg)
            options = m.group(1)
            pattern = m.group(2)
            result = int(m.group(3))

            # Get rid of spaces
            pattern = pattern.replace(" ", "").replace("\n", "")

            patterns += self._pattern_permute(pattern, options, result)

        # Compile the patterns into regular expressions for speed
        compiled_patterns = []
        for pattern in patterns:
            p = pattern[0].replace(".", "X").replace("X", "[01]")
            compiled_patterns.append((re.compile(p), pattern[1]))

        # Step through table and find patterns that match.
        # Note that all the patterns are searched. The last one found takes priority
        for i in range(LUT_SIZE):
            # Build the bit pattern
            bitpattern = bin(i)[2:]
            bitpattern = ("0" * (9 - len(bitpattern)) + bitpattern)[::-1]

            for pattern, r in compiled_patterns:
                if pattern.match(bitpattern):
                    self.lut[i] = [0, 1][r]

        return self.lut


class MorphOp:
    """A class for binary morphological operators"""

    def __init__(
        self,
        lut: bytearray | None = None,
        op_name: str | None = None,
        patterns: list[str] | None = None,
    ) -> None:
        """Create a binary morphological operator.

        If the LUT is not provided, then it is built using LutBuilder from the op_name
        or the patterns.

        :param lut: The LUT data.
        :param patterns: A list of input patterns, or None.
        :param op_name: The name of a known pattern. One of "corner", "dilation4",
        "dilation8", "erosion4", "erosion8", "edge".
        :exception Exception: If the op_name is not recognized.
        """
        if patterns is None and op_name is None:
            self.lut = lut
        else:
            self.lut = LutBuilder(patterns, op_name).build_lut()

    def apply(self, image: Image.Image) -> tuple[int, Image.Image]:
        """Run a single morphological operation on an image.

        Returns a tuple of the number of changed pixels and the
        morphed image.

        :param image: A 1-mode or L-mode image.
        :exception Exception: If the current operator is None.
        :exception ValueError: If the image is not 1 or L mode."""
        if self.lut is None:
            msg = "No operator loaded"
            raise Exception(msg)

        if image.mode not in ("1", "L"):
            msg = "Image mode must be 1 or L"
            raise ValueError(msg)
        outimage = Image.new(image.mode, image.size)
        count = _imagingmorph.apply(bytes(self.lut), image.getim(), outimage.getim())
        return count, outimage

    def match(self, image: Image.Image) -> list[tuple[int, int]]:
        """Get a list of coordinates matching the morphological operation on
        an image.

        Returns a list of tuples of (x,y) coordinates of all matching pixels. See
        :ref:`coordinate-system`.

        :param image: A 1-mode or L-mode image.
        :exception Exception: If the current operator is None.
        :exception ValueError: If the image is not 1 or L mode."""
        if self.lut is None:
            msg = "No operator loaded"
            raise Exception(msg)

        if image.mode not in ("1", "L"):
            msg = "Image mode must be 1 or L"
            raise ValueError(msg)
        return _imagingmorph.match(bytes(self.lut), image.getim())

    def get_on_pixels(self, image: Image.Image) -> list[tuple[int, int]]:
        """Get a list of all turned on pixels in a 1 or L mode image.

        Returns a list of tuples of (x,y) coordinates of all non-empty pixels. See
        :ref:`coordinate-system`.

        :param image: A 1-mode or L-mode image.
        :exception ValueError: If the image is not 1 or L mode."""

        if image.mode not in ("1", "L"):
            msg = "Image mode must be 1 or L"
            raise ValueError(msg)
        return _imagingmorph.get_on_pixels(image.getim())

    def load_lut(self, filename: str) -> None:
        """
        Load an operator from an mrl file

        :param filename: The file to read from.
        :exception Exception: If the length of the file data is not 512.
        """
        with open(filename, "rb") as f:
            self.lut = bytearray(f.read())

        if len(self.lut) != LUT_SIZE:
            self.lut = None
            msg = "Wrong size operator file!"
            raise Exception(msg)

    def save_lut(self, filename: str) -> None:
        """
        Save an operator to an mrl file.

        :param filename: The destination file.
        :exception Exception: If the current operator is None.
        """
        if self.lut is None:
            msg = "No operator loaded"
            raise Exception(msg)
        with open(filename, "wb") as f:
            f.write(self.lut)

    def set_lut(self, lut: bytearray | None) -> None:
        """
        Set the LUT from an external source

        :param lut: A new LUT.
        """
        self.lut = lut
<|endoftext|>from warnings import warn

from .cli import *  # NOQA
from .cli import __all__  # NOQA
from .std import TqdmDeprecationWarning

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.cli.*` instead of `tqdm._main.*`",
     TqdmDeprecationWarning, stacklevel=2)
<|endoftext|>from warnings import warn

from .gui import *  # NOQA
from .gui import __all__  # NOQA
from .std import TqdmDeprecationWarning

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.gui.*` instead of `tqdm._tqdm_gui.*`",
     TqdmDeprecationWarning, stacklevel=2)
<|endoftext|>"""
IPython/Jupyter Notebook progress bar decorator for iterators.
Includes a default `range` iterator printing to `stderr`.

Usage:
>>> from tqdm.notebook import trange, tqdm
>>> for i in trange(10):
...     ...
"""
# import compatibility functions and utilities
import re
import sys
from html import escape
from weakref import proxy

# to inherit from the tqdm class
from .std import tqdm as std_tqdm

if True:  # pragma: no cover
    # import IPython/Jupyter base widget and display utilities
    IPY = 0
    try:  # IPython 4.x
        import ipywidgets  # noqa: F401, pylint: disable=unused-import
        IPY = 4
    except ImportError:  # IPython 3.x / 2.x
        IPY = 32
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message=".*The `IPython.html` package has been deprecated.*")
            try:
                import IPython.html.widgets
            except ImportError:
                pass
            else:
                ipywidgets = IPython.html.widgets

    try:  # IPython 4.x / 3.x
        if IPY == 32:
            from IPython.html.widgets import HTML
            from IPython.html.widgets import FloatProgress as IProgress
            from IPython.html.widgets import HBox
            IPY = 3
        else:
            from ipywidgets import HTML
            from ipywidgets import FloatProgress as IProgress
            from ipywidgets import HBox
    except ImportError:
        try:  # IPython 2.x
            from IPython.html.widgets import HTML
            from IPython.html.widgets import ContainerWidget as HBox
            from IPython.html.widgets import FloatProgressWidget as IProgress
            IPY = 2
        except ImportError:
            IPY = 0
            IProgress = None
            HBox = object

    try:
        from IPython.display import display  # , clear_output
    except ImportError:
        pass

__author__ = {"github.com/": ["lrq3000", "casperdcl", "alexanderkuk"]}
__all__ = ['tqdm_notebook', 'tnrange', 'tqdm', 'trange']
WARN_NOIPYW = ("IProgress not found. Please update jupyter and ipywidgets."
               " See https://ipywidgets.readthedocs.io/en/stable"
               "/user_install.html")


class TqdmHBox(HBox):
    """`ipywidgets.HBox` with a pretty representation"""
    def _json_(self, pretty=None):
        pbar = getattr(self, 'pbar', None)
        if pbar is None:
            return {}
        d = pbar.format_dict
        if pretty is not None:
            d["ascii"] = not pretty
        return d

    def __repr__(self, pretty=False):
        pbar = getattr(self, 'pbar', None)
        if pbar is None:
            return super().__repr__()
        return pbar.format_meter(**self._json_(pretty))

    def _repr_pretty_(self, pp, *_, **__):
        pp.text(self.__repr__(True))


class tqdm_notebook(std_tqdm):
    """
    Experimental IPython/Jupyter Notebook widget using tqdm!
    """
    @staticmethod
    def status_printer(_, total=None, desc=None, ncols=None):
        """
        Manage the printing of an IPython/Jupyter Notebook progress bar widget.
        """
        # Fallback to text bar if there's no total
        # DEPRECATED: replaced with an 'info' style bar
        # if not total:
        #    return super(tqdm_notebook, tqdm_notebook).status_printer(file)

        # fp = file

        # Prepare IPython progress bar
        if IProgress is None:  # #187 #451 #558 #872
            raise ImportError(WARN_NOIPYW)
        if total:
            pbar = IProgress(min=0, max=total)
        else:  # No total? Show info style bar with no progress tqdm status
            pbar = IProgress(min=0, max=1)
            pbar.value = 1
            pbar.bar_style = 'info'
            if ncols is None:
                pbar.layout.width = "20px"

        ltext = HTML()
        rtext = HTML()
        if desc:
            ltext.value = desc
        container = TqdmHBox(children=[ltext, pbar, rtext])
        # Prepare layout
        if ncols is not None:  # use default style of ipywidgets
            # ncols could be 100, "100px", "100%"
            ncols = str(ncols)  # ipywidgets only accepts string
            try:
                if int(ncols) > 0:  # isnumeric and positive
                    ncols += 'px'
            except ValueError:
                pass
            pbar.layout.flex = '2'
            container.layout.width = ncols
            container.layout.display = 'inline-flex'
            container.layout.flex_flow = 'row wrap'

        return container

    def display(self, msg=None, pos=None,
                # additional signals
                close=False, bar_style=None, check_delay=True):
        # Note: contrary to native tqdm, msg='' does NOT clear bar
        # goal is to keep all infos if error happens so user knows
        # at which iteration the loop failed.

        # Clear previous output (really necessary?)
        # clear_output(wait=1)

        if not msg and not close:
            d = self.format_dict
            # remove {bar}
            d['bar_format'] = (d['bar_format'] or "{l_bar}<bar/>{r_bar}").replace(
                "{bar}", "<bar/>")
            msg = self.format_meter(**d)

        ltext, pbar, rtext = self.container.children
        pbar.value = self.n

        if msg:
            msg = msg.replace(' ', '\u2007')  # fix html space padding
            # html escape special characters (like '&')
            if '<bar/>' in msg:
                left, right = map(escape, re.split(r'\|?<bar/>\|?', msg, maxsplit=1))
            else:
                left, right = '', escape(msg)

            # Update description
            ltext.value = left
            # never clear the bar (signal: msg='')
            if right:
                rtext.value = right

        # Change bar style
        if bar_style:
            # Hack-ish way to avoid the danger bar_style being overridden by
            # success because the bar gets closed after the error...
            if pbar.bar_style != 'danger' or bar_style != 'success':
                pbar.bar_style = bar_style

        # Special signal to close the bar
        if close and pbar.bar_style != 'danger':  # hide only if no error
            try:
                self.container.close()
            except AttributeError:
                self.container.visible = False
            self.container.layout.visibility = 'hidden'  # IPYW>=8

        if check_delay and self.delay > 0 and not self.displayed:
            display(self.container)
            self.displayed = True

    @property
    def colour(self):
        if hasattr(self, 'container'):
            return self.container.children[-2].style.bar_color

    @colour.setter
    def colour(self, bar_color):
        if hasattr(self, 'container'):
            self.container.children[-2].style.bar_color = bar_color

    def __init__(self, *args, **kwargs):
        """
        Supports the usual `tqdm.tqdm` parameters as well as those listed below.

        Parameters
        ----------
        display  : Whether to call `display(self.container)` immediately
            [default: True].
        """
        kwargs = kwargs.copy()
        # Setup default output
        file_kwarg = kwargs.get('file', sys.stderr)
        if file_kwarg is sys.stderr or file_kwarg is None:
            kwargs['file'] = sys.stdout  # avoid the red block in IPython

        # Initialize parent class + avoid printing by using gui=True
        kwargs['gui'] = True
        # convert disable = None to False
        kwargs['disable'] = bool(kwargs.get('disable', False))
        colour = kwargs.pop('colour', None)
        display_here = kwargs.pop('display', True)
        super().__init__(*args, **kwargs)
        if self.disable or not kwargs['gui']:
            self.disp = lambda *_, **__: None
            return

        # Get bar width
        self.ncols = '100%' if self.dynamic_ncols else kwargs.get("ncols", None)

        # Replace with IPython progress bar display (with correct total)
        unit_scale = 1 if self.unit_scale is True else self.unit_scale or 1
        total = self.total * unit_scale if self.total else self.total
        self.container = self.status_printer(self.fp, total, self.desc, self.ncols)
        self.container.pbar = proxy(self)
        self.displayed = False
        if display_here and self.delay <= 0:
            display(self.container)
            self.displayed = True
        self.disp = self.display
        self.colour = colour

        # Print initial bar state
        if not self.disable:
            self.display(check_delay=False)

    def __iter__(self):
        try:
            it = super().__iter__()
            yield from it
        # NB: except ... [ as ...] breaks IPython async KeyboardInterrupt
        except:  # NOQA
            self.disp(bar_style='danger')
            raise
        # NB: don't `finally: close()`
        # since this could be a shared bar which the user will `reset()`

    def update(self, n=1):
        try:
            return super().update(n=n)
        # NB: except ... [ as ...] breaks IPython async KeyboardInterrupt
        except:  # NOQA
            # cannot catch KeyboardInterrupt when using manual tqdm
            # as the interrupt will most likely happen on another statement
            self.disp(bar_style='danger')
            raise
        # NB: don't `finally: close()`
        # since this could be a shared bar which the user will `reset()`

    def close(self):
        if self.disable:
            return
        super().close()
        # Try to detect if there was an error or KeyboardInterrupt
        # in manual mode: if n < total, things probably got wrong
        if self.total and self.n < self.total:
            self.disp(bar_style='danger', check_delay=False)
        else:
            if self.leave:
                self.disp(bar_style='success', check_delay=False)
            else:
                self.disp(close=True, check_delay=False)

    def clear(self, *_, **__):
        pass

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Consider combining with `leave=True`.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if self.disable:
            return super().reset(total=total)
        _, pbar, _ = self.container.children
        pbar.bar_style = ''
        if total is not None:
            pbar.max = total
            if not self.total and self.ncols is None:  # no longer unknown total
                pbar.layout.width = None  # reset width
        return super().reset(total=total)


def tnrange(*args, **kwargs):
    """Shortcut for `tqdm.notebook.tqdm(range(*args), **kwargs)`."""
    return tqdm_notebook(range(*args), **kwargs)


# Aliases
tqdm = tqdm_notebook
trange = tnrange
<|endoftext|>from functools import partial

from dask.callbacks import Callback

from .auto import tqdm as tqdm_auto

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['TqdmCallback']


class TqdmCallback(Callback):
    """Dask callback for task progress."""
    def __init__(self, start=None, pretask=None, tqdm_class=tqdm_auto,
                 **tqdm_kwargs):
        """
        Parameters
        ----------
        tqdm_class  : optional
            `tqdm` class to use for bars [default: `tqdm.auto.tqdm`].
        tqdm_kwargs  : optional
            Any other arguments used for all bars.
        """
        super().__init__(start=start, pretask=pretask)
        if tqdm_kwargs:
            tqdm_class = partial(tqdm_class, **tqdm_kwargs)
        self.tqdm_class = tqdm_class

    def _start_state(self, _, state):
        self.pbar = self.tqdm_class(total=sum(
            len(state[k]) for k in ['ready', 'waiting', 'running', 'finished']))

    def _posttask(self, *_, **__):
        self.pbar.update()

    def _finish(self, *_, **__):
        self.pbar.close()

    def display(self):
        """Displays in the current cell in Notebooks."""
        container = getattr(self.bar, 'container', None)
        if container is None:
            return
        from .notebook import display
        display(container)
<|endoftext|>"""
Enables multiple commonly used features.

Method resolution order:

- `tqdm.autonotebook` without import warnings
- `tqdm.asyncio`
- `tqdm.std` base class

Usage:
>>> from tqdm.auto import trange, tqdm
>>> for i in trange(10):
...     ...
"""
import warnings

from .std import TqdmExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning)
    from .autonotebook import tqdm as notebook_tqdm

from .asyncio import tqdm as asyncio_tqdm
from .std import tqdm as std_tqdm

if notebook_tqdm != std_tqdm:
    class tqdm(notebook_tqdm, asyncio_tqdm):  # pylint: disable=inconsistent-mro
        pass
else:
    tqdm = asyncio_tqdm


def trange(*args, **kwargs):
    """
    A shortcut for `tqdm.auto.tqdm(range(*args), **kwargs)`.
    """
    return tqdm(range(*args), **kwargs)


__all__ = ["tqdm", "trange"]
<|endoftext|>import atexit
from threading import Event, Thread, current_thread
from time import time
from warnings import warn

__all__ = ["TMonitor", "TqdmSynchronisationWarning"]


class TqdmSynchronisationWarning(RuntimeWarning):
    """
    tqdm multi-thread/-process errors which may cause incorrect nesting
    but otherwise no adverse effects
    """


class TMonitor(Thread):
    """
    Monitoring thread for tqdm bars.
    Monitors if tqdm bars are taking too much time to display
    and readjusts miniters automatically if necessary.

    Parameters
    ----------
    tqdm_cls  : class
        tqdm class to use (can be core tqdm or a submodule).
    sleep_interval  : float
        Time to sleep between monitoring checks.
    """
    _test = {}  # internal vars for unit testing

    def __init__(self, tqdm_cls, sleep_interval):
        Thread.__init__(self, name="tqdm_monitor")
        self.daemon = True  # kill thread when main killed (KeyboardInterrupt)
        self.woken = 0  # last time woken up, to sync with monitor
        self.tqdm_cls = tqdm_cls
        self.sleep_interval = sleep_interval
        self._time = self._test.get("time", time)
        self.was_killed = self._test.get("Event", Event)()
        atexit.register(self._atexit_signal)
        self.start()

    def _atexit_signal(self):
        """
        Non-joining shutdown signal.
        Avoids deadlocks at interpreter exit from other threads, dead forks, etc.
        This daemon thread is auto-reaped on shutdown without needing a join.
        """
        self.was_killed.set()

    def exit(self):
        self.was_killed.set()
        if self is not current_thread():
            self.join()
        return self.report()

    def get_instances(self):
        # returns a copy of started `tqdm_cls` instances
        return [i for i in self.tqdm_cls._instances.copy()
                # Avoid race by checking that the instance started
                if hasattr(i, 'start_t')]

    def run(self):
        cur_t = self._time()
        while True:
            # After processing and before sleeping, notify that we woke
            # Need to be done just before sleeping
            self.woken = cur_t
            # Sleep some time...
            self.was_killed.wait(self.sleep_interval)
            # Quit if killed
            if self.was_killed.is_set():
                return
            # Then monitor!
            # Acquire lock (to access _instances)
            with self.tqdm_cls.get_lock():
                cur_t = self._time()
                # Check tqdm instances are waiting too long to print
                instances = self.get_instances()
                for instance in instances:
                    # Check event in loop to reduce blocking time on exit
                    if self.was_killed.is_set():
                        return
                    # Only if mininterval > 1 (else iterations are just slow)
                    # and last refresh exceeded maxinterval
                    if (
                        instance.miniters > 1
                        and (cur_t - instance.last_print_t) >= instance.maxinterval
                    ):
                        # force bypassing miniters on next iteration
                        # (dynamic_miniters adjusts mininterval automatically)
                        instance.miniters = 1
                        # Refresh now! (works only for manual tqdm)
                        instance.refresh(nolock=True)
                    # Remove accidental long-lived strong reference
                    del instance
                if instances != self.get_instances():  # pragma: nocover
                    warn("Set changed size during iteration" +
                         " (see https://github.com/tqdm/tqdm/issues/481)",
                         TqdmSynchronisationWarning, stacklevel=2)
                # Remove accidental long-lived strong references
                del instances

    def report(self):
        return not self.was_killed.is_set()
<|endoftext|>"""
Matplotlib GUI progress bar decorator for iterators.

Usage:
>>> from tqdm.gui import trange, tqdm
>>> for i in trange(10):
...     ...
"""
# future division is important to divide integers and get as
# a result precise floating numbers (instead of truncated int)
import re
from warnings import warn

# to inherit from the tqdm class
from .std import TqdmExperimentalWarning
from .std import tqdm as std_tqdm

# import compatibility functions and utilities

__author__ = {"github.com/": ["casperdcl", "lrq3000"]}
__all__ = ['tqdm_gui', 'tgrange', 'tqdm', 'trange']


class tqdm_gui(std_tqdm):  # pragma: no cover
    """Experimental Matplotlib GUI version of tqdm!"""
    # TODO: @classmethod: write() on GUI?
    def __init__(self, *args, **kwargs):
        from collections import deque

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        colour = kwargs.pop('colour', 'g')
        super().__init__(*args, **kwargs)

        if self.disable:
            return

        warn("GUI is experimental/alpha", TqdmExperimentalWarning, stacklevel=2)
        self.mpl = mpl
        self.plt = plt

        # Remember if external environment uses toolbars
        self.toolbar = self.mpl.rcParams['toolbar']
        self.mpl.rcParams['toolbar'] = 'None'

        self.mininterval = max(self.mininterval, 0.5)
        self.fig, ax = plt.subplots(figsize=(9, 2.2))
        # self.fig.subplots_adjust(bottom=0.2)
        total = self.__len__()  # avoids TypeError on None #971
        if total is not None:
            self.xdata = []
            self.ydata = []
            self.zdata = []
        else:
            self.xdata = deque([])
            self.ydata = deque([])
            self.zdata = deque([])
        self.line1, = ax.plot(self.xdata, self.ydata, color='b')
        self.line2, = ax.plot(self.xdata, self.zdata, color='k')
        ax.set_ylim(0, 0.001)
        if total is not None:
            ax.set_xlim(0, 100)
            ax.set_xlabel("percent")
            self.fig.legend((self.line1, self.line2), ("cur", "est"),
                            loc='center right')
            # progress bar
            self.hspan = plt.axhspan(0, 0.001, xmin=0, xmax=0, color=colour)
        else:
            # ax.set_xlim(-60, 0)
            ax.set_xlim(0, 60)
            ax.invert_xaxis()
            ax.set_xlabel("seconds")
            ax.legend(("cur", "est"), loc='lower left')
        ax.grid()
        # ax.set_xlabel('seconds')
        ax.set_ylabel((self.unit if self.unit else "it") + "/s")
        if self.unit_scale:
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_x(-0.15)

        # Remember if external environment is interactive
        self.wasion = plt.isinteractive()
        plt.ion()
        self.ax = ax

    def close(self):
        if self.disable:
            return

        self.disable = True

        with self.get_lock():
            self._instances.remove(self)

        # Restore toolbars
        self.mpl.rcParams['toolbar'] = self.toolbar
        # Return to non-interactive mode
        if not self.wasion:
            self.plt.ioff()
        if self.leave:
            self.display()
        else:
            self.plt.close(self.fig)

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        n = self.n
        cur_t = self._time()
        elapsed = cur_t - self.start_t
        delta_it = n - self.last_print_n
        delta_t = cur_t - self.last_print_t

        # Inline due to multiple calls
        total = self.total
        xdata = self.xdata
        ydata = self.ydata
        zdata = self.zdata
        ax = self.ax
        line1 = self.line1
        line2 = self.line2
        hspan = getattr(self, 'hspan', None)
        # instantaneous rate
        y = delta_it / delta_t
        # overall rate
        z = n / elapsed
        # update line data
        xdata.append(n * 100.0 / total if total else cur_t)
        ydata.append(y)
        zdata.append(z)

        # Discard old values
        # xmin, xmax = ax.get_xlim()
        # if (not total) and elapsed > xmin * 1.1:
        if (not total) and elapsed > 66:
            xdata.popleft()
            ydata.popleft()
            zdata.popleft()

        ymin, ymax = ax.get_ylim()
        if y > ymax or z > ymax:
            ymax = 1.1 * y
            ax.set_ylim(ymin, ymax)
            ax.figure.canvas.draw()

        if total:
            line1.set_data(xdata, ydata)
            line2.set_data(xdata, zdata)
            if hspan:
                hspan.set_xy((0, ymin))
                hspan.set_height(ymax - ymin)
                hspan.set_width(n / total)
        else:
            t_ago = [cur_t - i for i in xdata]
            line1.set_data(t_ago, ydata)
            line2.set_data(t_ago, zdata)

        d = self.format_dict
        # remove {bar}
        d['bar_format'] = (d['bar_format'] or "{l_bar}<bar/>{r_bar}").replace(
            "{bar}", "<bar/>")
        msg = self.format_meter(**d)
        if '<bar/>' in msg:
            msg = "".join(re.split(r'\|?<bar/>\|?', msg, maxsplit=1))
        ax.set_title(msg, fontname="DejaVu Sans Mono", fontsize=11)
        self.plt.pause(1e-9)


def tgrange(*args, **kwargs):
    """Shortcut for `tqdm.gui.tqdm(range(*args), **kwargs)`."""
    return tqdm_gui(range(*args), **kwargs)


# Aliases
tqdm = tqdm_gui
trange = tgrange
<|endoftext|>"""
General helpers required for `tqdm.std`.
"""
import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
# TODO consider using wcswidth third-party package for 0-width characters
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy

_range, _unich, _unicode, _basestring = range, chr, str, str
CUR_OS = sys.platform
IS_WIN = any(CUR_OS.startswith(i) for i in ['win32', 'cygwin'])
IS_NIX = any(CUR_OS.startswith(i) for i in ['aix', 'linux', 'darwin', 'freebsd'])
RE_ANSI = re.compile(r"\x1b\[[;\d]*[A-Za-z]")

try:
    if IS_WIN:
        import colorama
    else:
        raise ImportError
except ImportError:
    colorama = None
else:
    try:
        colorama.init(strip=False)
    except TypeError:
        colorama.init()


def envwrap(name, app="", types=None, is_method=False):
    """
    Basic (env-only) version of [envwrap](https://github.com/tqdm/envwrap).
    Install `envwrap` for config file support.
    """
    if types is None:
        types = {}
    if name[-1] == "_":
        name = name[:-1]
        warn("Trailing underscore in `name` is automatic", DeprecationWarning, stacklevel=2)
    prefixes = (name, f"{name}_{app}") if app else (name,)
    env_overrides = {}
    for prefix in prefixes:
        prefix = prefix.upper() + "_"
        i = len(prefix)
        env_overrides.update(
            (k[i:].lower(), v) for k, v in os.environ.items() if k.startswith(prefix))
    part = partialmethod if is_method else partial

    def wrap(func):
        params = signature(func).parameters
        # ignore unknown env vars
        overrides = {k: v for k, v in env_overrides.items() if k in params}
        # infer overrides' `type`s
        for k in overrides:
            param = params[k]
            if param.annotation is not param.empty:  # typehints
                for typ in getattr(param.annotation, '__args__', (param.annotation,)):
                    try:
                        overrides[k] = typ(overrides[k])
                    except Exception:  # nosec B110
                        pass
                    else:
                        break
            elif param.default is not None:  # type of default value
                overrides[k] = type(param.default)(overrides[k])
            else:
                try:  # `types` fallback
                    overrides[k] = types[k](overrides[k])
                except KeyError:  # keep unconverted (`str`)
                    pass
        return part(func, **overrides)
    return wrap


try:
    from envwrap import envwrap  # noqa: F401, F811, pylint: disable=unused-import
except ModuleNotFoundError:
    pass


class FormatReplace:
    """
    >>> a = FormatReplace('something')
    >>> f"{a:5d}"
    'something'
    """  # NOQA: P102
    def __init__(self, replace=''):
        self.replace = replace
        self.format_called = 0

    def __format__(self, _):
        self.format_called += 1
        return self.replace


class Comparable:
    """Assumes child has self._comparable attr/@property"""
    def __lt__(self, other):
        return self._comparable < other._comparable

    def __le__(self, other):
        return (self < other) or (self == other)

    def __eq__(self, other):
        return self._comparable == other._comparable

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class ObjectWrapper:
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        return setattr(self._wrapped, name, value)

    def wrapper_getattr(self, name):
        """Actual `self.getattr` rather than self._wrapped.getattr"""
        try:
            return object.__getattr__(self, name)
        except AttributeError:  # py2
            return getattr(self, name)

    def wrapper_setattr(self, name, value):
        """Actual `self.setattr` rather than self._wrapped.setattr"""
        return object.__setattr__(self, name, value)

    def __init__(self, wrapped):
        """
        Thin wrapper around a given object
        """
        self.wrapper_setattr('_wrapped', wrapped)


class SimpleTextIOWrapper(ObjectWrapper):
    """
    Change only `.write()` of the wrapped object by encoding the passed
    value and passing the result to the wrapped object's `.write()` method.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, wrapped, encoding):
        super().__init__(wrapped)
        self.wrapper_setattr('encoding', encoding)

    def write(self, s):
        """
        Encode `s` and pass to the wrapped object's `.write()` method.
        """
        return self._wrapped.write(s.encode(self.wrapper_getattr('encoding')))

    def __eq__(self, other):
        return self._wrapped == getattr(other, '_wrapped', other)


class DisableOnWriteError(ObjectWrapper):
    """
    Disable the given `tqdm_instance` upon `write()` or `flush()` errors.
    """
    @staticmethod
    def disable_on_exception(tqdm_instance, func):
        """
        Quietly set `tqdm_instance.miniters=inf` if `func` raises `errno=5`.
        """
        tqdm_instance = proxy(tqdm_instance)

        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except OSError as e:
                if e.errno != 5:
                    raise
                try:
                    tqdm_instance.miniters = float('inf')
                except ReferenceError:
                    pass
            except ValueError as e:
                if 'closed' not in str(e):
                    raise
                try:
                    tqdm_instance.miniters = float('inf')
                except ReferenceError:
                    pass
        return inner

    def __init__(self, wrapped, tqdm_instance):  # noqa: B042
        super().__init__(wrapped)
        if hasattr(wrapped, 'write'):
            self.wrapper_setattr(
                'write', self.disable_on_exception(tqdm_instance, wrapped.write))
        if hasattr(wrapped, 'flush'):
            self.wrapper_setattr(
                'flush', self.disable_on_exception(tqdm_instance, wrapped.flush))

    def __eq__(self, other):
        return self._wrapped == getattr(other, '_wrapped', other)


class CallbackIOWrapper(ObjectWrapper):
    def __init__(self, callback, stream, method="read"):
        """
        Wrap a given `file`-like object's `read()` or `write()` to report
        lengths to the given `callback`
        """
        super().__init__(stream)
        func = getattr(stream, method)
        if method == "write":
            @wraps(func)
            def write(data, *args, **kwargs):
                res = func(data, *args, **kwargs)
                callback(len(data))
                return res
            self.wrapper_setattr('write', write)
        elif method == "read":
            @wraps(func)
            def read(*args, **kwargs):
                data = func(*args, **kwargs)
                callback(len(data))
                return data
            self.wrapper_setattr('read', read)
        else:
            raise KeyError("Can only wrap read/write methods")


def _is_utf(encoding):
    try:
        '\u2588\u2589'.encode(encoding)
    except UnicodeEncodeError:
        return False
    except Exception:
        try:
            return encoding.lower().startswith('utf-') or ('U8' == encoding)
        except Exception:
            return False
    else:
        return True


def _supports_unicode(fp):
    try:
        return _is_utf(fp.encoding)
    except AttributeError:
        return False


def _is_ascii(s):
    if isinstance(s, str):
        for c in s:
            if ord(c) > 255:
                return False
        return True
    return _supports_unicode(s)


def _screen_shape_wrapper():  # pragma: no cover
    """
    Return a function which returns console dimensions (width, height).
    Supported: linux, osx, windows, cygwin.
    """
    def inner(fp):
        try:
            from os import get_terminal_size
            cols, lines = get_terminal_size(getattr(fp, 'fileno', lambda: None)())
            return cols - 1, lines - 1
        except Exception:
            return None, None

    return inner


def _environ_cols_wrapper():  # pragma: no cover
    """
    Return a function which returns console width.
    Supported: linux, osx, windows, cygwin.
    """
    warn("Use `_screen_shape_wrapper()(file)[0]` instead of"
         " `_environ_cols_wrapper()(file)`", DeprecationWarning, stacklevel=2)
    shape = _screen_shape_wrapper()
    if not shape:
        return None

    @wraps(shape)
    def inner(fp):
        return shape(fp)[0]

    return inner


def _term_move_up():  # pragma: no cover
    return '' if (os.name == 'nt') and (colorama is None) else '\x1b[A'


def _text_width(s):
    return sum(2 if east_asian_width(ch) in 'FW' else 1 for ch in str(s))


def disp_len(data):
    """
    Returns the real on-screen length of a string which may contain
    ANSI control codes and wide chars.
    """
    return _text_width(RE_ANSI.sub('', data))


def disp_trim(data, length):
    """
    Trim a string which may contain ANSI control characters.
    """
    if len(data) == disp_len(data):
        return data[:length]

    ansi_present = bool(RE_ANSI.search(data))
    while disp_len(data) > length:  # carefully delete one char at a time
        data = data[:-1]
    if ansi_present and bool(RE_ANSI.search(data)):
        # assume ANSI reset is required
        return data if data.endswith("\033[0m") else data + "\033[0m"
    return data
<|endoftext|>from warnings import warn

from .std import *  # NOQA
from .std import __all__  # NOQA
from .std import TqdmDeprecationWarning

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.std.*` instead of `tqdm._tqdm.*`",
     TqdmDeprecationWarning, stacklevel=2)
<|endoftext|>from copy import copy
from functools import partial

from .auto import tqdm as tqdm_auto

try:
    import keras
except (ImportError, AttributeError) as e:
    try:
        from tensorflow import keras
    except ImportError:
        raise e
__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['TqdmCallback']


class TqdmCallback(keras.callbacks.Callback):
    """Keras callback for epoch and batch progress."""
    @staticmethod
    def bar2callback(bar, pop=None, delta=(lambda logs: 1)):
        def callback(_, logs=None):
            n = delta(logs)
            if logs:
                if pop:
                    logs = copy(logs)
                    for i in pop:
                        logs.pop(i, 0)
                bar.set_postfix(logs, refresh=False)
            bar.update(n)

        return callback

    def __init__(self, epochs=None, data_size=None, batch_size=None, verbose=1,
                 tqdm_class=tqdm_auto, **tqdm_kwargs):
        """
        Parameters
        ----------
        epochs  : int, optional
        data_size  : int, optional
            Number of training pairs.
        batch_size  : int, optional
            Number of training pairs per batch.
        verbose  : int
            0: epoch, 1: batch (transient), 2: batch. [default: 1].
            Will be set to `0` unless both `data_size` and `batch_size`
            are given.
        tqdm_class  : optional
            `tqdm` class to use for bars [default: `tqdm.auto.tqdm`].
        tqdm_kwargs  : optional
            Any other arguments used for all bars.
        """
        if tqdm_kwargs:
            tqdm_class = partial(tqdm_class, **tqdm_kwargs)
        self.tqdm_class = tqdm_class
        self.epoch_bar = tqdm_class(total=epochs, unit='epoch')
        self.on_epoch_end = self.bar2callback(self.epoch_bar)
        if data_size and batch_size:
            self.batches = batches = (data_size + batch_size - 1) // batch_size
        else:
            self.batches = batches = None
        self.verbose = verbose
        if verbose == 1:
            self.batch_bar = tqdm_class(total=batches, unit='batch', leave=False)
            self.on_batch_end = self.bar2callback(
                self.batch_bar, pop=['batch', 'size'],
                delta=lambda logs: logs.get('size', 1))

    def on_train_begin(self, *_, **__):
        params = self.params.get
        auto_total = params('epochs', params('nb_epoch', None))
        if auto_total is not None and auto_total != self.epoch_bar.total:
            self.epoch_bar.reset(total=auto_total)

    def on_epoch_begin(self, epoch, *_, **__):
        if self.epoch_bar.n < epoch:
            ebar = self.epoch_bar
            ebar.n = ebar.last_print_n = ebar.initial = epoch
        if self.verbose:
            params = self.params.get
            total = params('samples', params(
                'nb_sample', params('steps', None))) or self.batches
            if self.verbose == 2:
                if hasattr(self, 'batch_bar'):
                    self.batch_bar.close()
                self.batch_bar = self.tqdm_class(
                    total=total, unit='batch', leave=True,
                    unit_scale=1 / (params('batch_size', 1) or 1))
                self.on_batch_end = self.bar2callback(
                    self.batch_bar, pop=['batch', 'size'],
                    delta=lambda logs: logs.get('size', 1))
            elif self.verbose == 1:
                self.batch_bar.unit_scale = 1 / (params('batch_size', 1) or 1)
                self.batch_bar.reset(total=total)
            else:
                raise KeyError('Unknown verbosity')

    def on_train_end(self, *_, **__):
        if hasattr(self, 'batch_bar'):
            self.batch_bar.close()
        self.epoch_bar.close()

    def display(self):
        """Displays in the current cell in Notebooks."""
        container = getattr(self.epoch_bar, 'container', None)
        if container is None:
            return
        from .notebook import display
        display(container)
        batch_bar = getattr(self, 'batch_bar', None)
        if batch_bar is not None:
            display(batch_bar.container)

    @staticmethod
    def _implements_train_batch_hooks():
        return True

    @staticmethod
    def _implements_test_batch_hooks():
        return True

    @staticmethod
    def _implements_predict_batch_hooks():
        return True
<|endoftext|>"""
Asynchronous progress bar decorator for iterators.
Includes a default `range` iterator printing to `stderr`.

Usage:
>>> from tqdm.asyncio import trange, tqdm
>>> async for i in trange(10):
...     ...
"""
import asyncio
from sys import version_info

from .std import tqdm as std_tqdm

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['tqdm_asyncio', 'tarange', 'tqdm', 'trange']


class tqdm_asyncio(std_tqdm):
    """
    Asynchronous-friendly version of tqdm.
    """
    def __init__(self, iterable=None, *args, **kwargs):
        super().__init__(iterable, *args, **kwargs)
        self.iterable_awaitable = False
        if iterable is not None:
            if hasattr(iterable, "__anext__"):
                self.iterable_next = iterable.__anext__
                self.iterable_awaitable = True
            elif hasattr(iterable, "__aiter__"):
                self.iterable_iterator = iterable.__aiter__()
                self.iterable_next = self.iterable_iterator.__anext__
                self.iterable_awaitable = True
            elif hasattr(iterable, "__next__"):
                self.iterable_next = iterable.__next__
            else:
                self.iterable_iterator = iter(iterable)
                self.iterable_next = self.iterable_iterator.__next__

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.iterable_awaitable:
                res = await self.iterable_next()
            else:
                res = self.iterable_next()
            self.update()
            return res
        except StopIteration:
            self.close()
            raise StopAsyncIteration
        except BaseException:
            self.close()
            raise

    def send(self, *args, **kwargs):
        return self.iterable.send(*args, **kwargs)

    @classmethod
    def as_completed(cls, fs, *, loop=None, timeout=None, total=None, **tqdm_kwargs):
        """
        Wrapper for `asyncio.as_completed`.
        """
        if total is None:
            total = len(fs)
        kwargs = {}
        if version_info[:2] < (3, 10):
            kwargs['loop'] = loop
        yield from cls(asyncio.as_completed(fs, timeout=timeout, **kwargs),
                       total=total, **tqdm_kwargs)

    @classmethod
    async def gather(cls, *fs, loop=None, timeout=None, total=None, return_exceptions=False,
                     **tqdm_kwargs):
        """
        Wrapper for `asyncio.gather`.
        """
        async def wrap_awaitable(i, f):
            try:
                return i, await f
            except Exception as e:  # noqa: BLE001
                if return_exceptions:
                    return i, e
                raise

        ifs = [wrap_awaitable(i, f) for i, f in enumerate(fs)]
        res = [await f for f in cls.as_completed(ifs, loop=loop, timeout=timeout,
                                                 total=total, **tqdm_kwargs)]
        return [i for _, i in sorted(res)]


def tarange(*args, **kwargs):
    """
    A shortcut for `tqdm.asyncio.tqdm(range(*args), **kwargs)`.
    """
    return tqdm_asyncio(range(*args), **kwargs)


# Aliases
tqdm = tqdm_asyncio
trange = tarange
<|endoftext|>import sys

__author__ = "github.com/casperdcl"
__all__ = ['tqdm_pandas']


def tqdm_pandas(tclass, **tqdm_kwargs):
    """
    Registers the given `tqdm` instance with
    `pandas.core.groupby.DataFrameGroupBy.progress_apply`.
    """
    from tqdm import TqdmDeprecationWarning

    if isinstance(tclass, type) or (getattr(tclass, '__name__', '').startswith(
            'tqdm_')):  # delayed adapter case
        TqdmDeprecationWarning(
            "Please use `tqdm.pandas(...)` instead of `tqdm_pandas(tqdm, ...)`.",
            fp_write=getattr(tqdm_kwargs.get('file', None), 'write', sys.stderr.write))
        tclass.pandas(**tqdm_kwargs)
    else:
        TqdmDeprecationWarning(
            "Please use `tqdm.pandas(...)` instead of `tqdm_pandas(tqdm(...))`.",
            fp_write=getattr(tclass.fp, 'write', sys.stderr.write))
        type(tclass).pandas(deprecated_t=tclass)
<|endoftext|>"""
Automatically choose between `tqdm.notebook` and `tqdm.std`.

Usage:
>>> from tqdm.autonotebook import trange, tqdm
>>> for i in trange(10):
...     ...
"""
import os
import sys
from warnings import warn

try:
    if 'ipykernel.zmqshell' in sys.modules:
        if any(i == 'QT_API' or i.startswith('SPYDER') for i in os.environ):
            raise ImportError("console")  # jupyter-qtconsole/spyder
        ipy = sys.modules['IPython'].get_ipython().__class__.__name__.lower()
        if 'qt' in ipy or 'spyder' in ipy:
            raise ImportError("console")  # older jupyter-qtconsole/spyder
        # jupyter-notebook/jupyterlab/vscode/binder/colab
    elif 'IPython.utils._process_emscripten' in sys.modules:
        pass  # jupyterlite (pyodide)/jupyterlite-xeus
    else:
        raise ImportError("console")  # ipython/jupyter-console
    from .notebook import WARN_NOIPYW, IProgress
    if IProgress is None:
        from .std import TqdmWarning
        warn(WARN_NOIPYW, TqdmWarning, stacklevel=2)
        raise ImportError('ipywidgets')
except Exception:
    from .std import tqdm, trange
else:  # pragma: no cover
    from .notebook import tqdm, trange
    from .std import TqdmExperimentalWarning
    warn("Using `tqdm.autonotebook.tqdm` in notebook mode."
         " Use `tqdm.tqdm` instead to force console mode"
         " (e.g. in jupyter console)", TqdmExperimentalWarning, stacklevel=2)
__all__ = ["tqdm", "trange"]
<|endoftext|>from ._monitor import TMonitor, TqdmSynchronisationWarning
from ._tqdm_pandas import tqdm_pandas
from .cli import main  # TODO: remove in v5.0.0
from .gui import tqdm as tqdm_gui  # TODO: remove in v5.0.0
from .gui import trange as tgrange  # TODO: remove in v5.0.0
from .std import (
    TqdmDeprecationWarning, TqdmExperimentalWarning, TqdmKeyError, TqdmMonitorWarning,
    TqdmTypeError, TqdmWarning, tqdm, trange)
from .version import __version__

__all__ = ['tqdm', 'tqdm_gui', 'trange', 'tgrange', 'tqdm_pandas',
           'tqdm_notebook', 'tnrange', 'main', 'TMonitor',
           'TqdmTypeError', 'TqdmKeyError',
           'TqdmWarning', 'TqdmDeprecationWarning',
           'TqdmExperimentalWarning',
           'TqdmMonitorWarning', 'TqdmSynchronisationWarning',
           '__version__']


def tqdm_notebook(*args, **kwargs):  # pragma: no cover
    """See tqdm.notebook.tqdm for full documentation"""
    from warnings import warn

    from .notebook import tqdm as _tqdm_notebook
    warn("This function will be removed in tqdm==5.0.0\n"
         "Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`",
         TqdmDeprecationWarning, stacklevel=2)
    return _tqdm_notebook(*args, **kwargs)


def tnrange(*args, **kwargs):  # pragma: no cover
    """Shortcut for `tqdm.notebook.tqdm(range(*args), **kwargs)`."""
    from warnings import warn

    from .notebook import trange as _tnrange
    warn("Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`",
         TqdmDeprecationWarning, stacklevel=2)
    return _tnrange(*args, **kwargs)
<|endoftext|>"""
Tkinter GUI progress bar decorator for iterators.

Usage:
>>> from tqdm.tk import trange, tqdm
>>> for i in trange(10):
...     ...
"""
import re
import sys
import tkinter
import tkinter.ttk as ttk
from warnings import warn

from .std import TqdmExperimentalWarning, TqdmWarning
from .std import tqdm as std_tqdm

__author__ = {"github.com/": ["richardsheridan", "casperdcl"]}
__all__ = ['tqdm_tk', 'ttkrange', 'tqdm', 'trange']


class tqdm_tk(std_tqdm):  # pragma: no cover
    """
    Experimental Tkinter GUI version of tqdm!

    Note: Window interactivity suffers if `tqdm_tk` is not running within
    a Tkinter mainloop and values are generated infrequently. In this case,
    consider calling `tqdm_tk.refresh()` frequently in the Tk thread.
    """

    # TODO: @classmethod: write()?

    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        grab  : bool, optional
            Grab the input across all windows of the process.
        tk_parent  : `tkinter.Wm`, optional
            Parent Tk window.
        cancel_callback  : Callable, optional
            Create a cancel button and set `cancel_callback` to be called
            when the cancel or window close button is clicked.
        """
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        # convert disable = None to False
        kwargs['disable'] = bool(kwargs.get('disable', False))
        self._warn_leave = 'leave' in kwargs
        grab = kwargs.pop('grab', False)
        tk_parent = kwargs.pop('tk_parent', None)
        self._cancel_callback = kwargs.pop('cancel_callback', None)
        super().__init__(*args, **kwargs)

        if self.disable:
            return

        if tk_parent is None:  # Discover parent widget
            try:
                tk_parent = tkinter._default_root
            except AttributeError:
                raise AttributeError(
                    "`tk_parent` required when using `tkinter.NoDefaultRoot()`")
            if tk_parent is None:  # use new default root window as display
                self._tk_window = tkinter.Tk()
            else:  # some other windows already exist
                self._tk_window = tkinter.Toplevel()
        else:
            self._tk_window = tkinter.Toplevel(tk_parent)

        warn("GUI is experimental/alpha", TqdmExperimentalWarning, stacklevel=2)
        self._tk_dispatching = self._tk_dispatching_helper()

        self._tk_window.protocol("WM_DELETE_WINDOW", self.cancel)
        self._tk_window.wm_title(self.desc)
        self._tk_window.wm_attributes("-topmost", 1)
        self._tk_window.after(0, lambda: self._tk_window.wm_attributes("-topmost", 0))
        self._tk_n_var = tkinter.DoubleVar(self._tk_window, value=0)
        self._tk_text_var = tkinter.StringVar(self._tk_window)
        pbar_frame = ttk.Frame(self._tk_window, padding=5)
        pbar_frame.pack()
        _tk_label = ttk.Label(pbar_frame, textvariable=self._tk_text_var,
                              wraplength=600, anchor="center", justify="center")
        _tk_label.pack()
        self._tk_pbar = ttk.Progressbar(
            pbar_frame, variable=self._tk_n_var, length=450)
        if self.total is not None:
            self._tk_pbar.configure(maximum=self.total)
        else:
            self._tk_pbar.configure(mode="indeterminate")
        self._tk_pbar.pack()
        if self._cancel_callback is not None:
            _tk_button = ttk.Button(pbar_frame, text="Cancel", command=self.cancel)
            _tk_button.pack()
        if grab:
            self._tk_window.grab_set()

    def close(self):
        if self.disable:
            return

        self.disable = True

        with self.get_lock():
            self._instances.remove(self)

        def _close():
            self._tk_window.after('idle', self._tk_window.destroy)
            if not self._tk_dispatching:
                self._tk_window.update()

        self._tk_window.protocol("WM_DELETE_WINDOW", _close)

        # if leave is set but we are self-dispatching, the left window is
        # totally unresponsive unless the user manually dispatches
        if not self.leave:
            _close()
        elif not self._tk_dispatching:
            if self._warn_leave:
                warn("leave flag ignored if not in tkinter mainloop",
                     TqdmWarning, stacklevel=2)
            _close()

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        self._tk_n_var.set(self.n)
        d = self.format_dict
        # remove {bar}
        d['bar_format'] = (d['bar_format'] or "{l_bar}<bar/>{r_bar}").replace(
            "{bar}", "<bar/>")
        msg = self.format_meter(**d)
        if '<bar/>' in msg:
            msg = "".join(re.split(r'\|?<bar/>\|?', msg, maxsplit=1))
        self._tk_text_var.set(msg)
        if not self._tk_dispatching:
            self._tk_window.update()

    def set_description(self, desc=None, refresh=True):
        self.set_description_str(desc, refresh)

    def set_description_str(self, desc=None, refresh=True):
        self.desc = desc
        if not self.disable:
            self._tk_window.wm_title(desc)
            if refresh and not self._tk_dispatching:
                self._tk_window.update()

    def cancel(self):
        """
        `cancel_callback()` followed by `close()`
        when close/cancel buttons clicked.
        """
        if self._cancel_callback is not None:
            self._cancel_callback()
        self.close()

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_tk_pbar'):
            if total is None:
                self._tk_pbar.configure(maximum=100, mode="indeterminate")
            else:
                self._tk_pbar.configure(maximum=total, mode="determinate")
        super().reset(total=total)

    @staticmethod
    def _tk_dispatching_helper():
        """determine if Tkinter mainloop is dispatching events"""
        codes = {tkinter.mainloop.__code__, tkinter.Misc.mainloop.__code__}
        for frame in sys._current_frames().values():
            while frame:
                if frame.f_code in codes:
                    return True
                frame = frame.f_back
        return False


def ttkrange(*args, **kwargs):
    """Shortcut for `tqdm.tk.tqdm(range(*args), **kwargs)`."""
    return tqdm_tk(range(*args), **kwargs)


# Aliases
tqdm = tqdm_tk
trange = ttkrange
<|endoftext|>from warnings import warn

from .notebook import *  # NOQA
from .notebook import __all__  # NOQA
from .std import TqdmDeprecationWarning

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.notebook.*` instead of `tqdm._tqdm_notebook.*`",
     TqdmDeprecationWarning, stacklevel=2)
<|endoftext|>from warnings import warn

from .std import TqdmDeprecationWarning
from .utils import (  # noqa: F401, pylint: disable=unused-import
    CUR_OS, IS_NIX, IS_WIN, RE_ANSI, Comparable, FormatReplace, SimpleTextIOWrapper,
    _environ_cols_wrapper, _is_ascii, _is_utf, _screen_shape_linux, _screen_shape_tput,
    _screen_shape_windows, _screen_shape_wrapper, _supports_unicode, _term_move_up, colorama)

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.utils.*` instead of `tqdm._utils.*`",
     TqdmDeprecationWarning, stacklevel=2)
<|endoftext|>"""
Customisable progress bar decorator for iterators.
Includes a default `range` iterator printing to `stderr`.

Usage:
>>> from tqdm import trange, tqdm
>>> for i in trange(10):
...     ...
"""
import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet

from ._monitor import TMonitor
from .utils import (
    CallbackIOWrapper, Comparable, DisableOnWriteError, FormatReplace, SimpleTextIOWrapper,
    _is_ascii, _screen_shape_wrapper, _supports_unicode, _term_move_up, disp_len, disp_trim,
    envwrap)

__author__ = "https://github.com/tqdm/tqdm#contributions"
__all__ = ['tqdm', 'trange',
           'TqdmTypeError', 'TqdmKeyError', 'TqdmWarning',
           'TqdmExperimentalWarning', 'TqdmDeprecationWarning',
           'TqdmMonitorWarning']


class TqdmTypeError(TypeError):
    pass


class TqdmKeyError(KeyError):
    pass


class TqdmWarning(Warning):
    """base class for all tqdm warnings.

    Used for non-external-code-breaking errors, such as garbled printing.
    """
    def __init__(self, msg, fp_write=None):  # noqa: B042
        if fp_write is not None:
            fp_write("\n" + self.__class__.__name__ + ": " + str(msg).rstrip() + '\n')
        else:
            super().__init__(msg)


class TqdmExperimentalWarning(TqdmWarning, FutureWarning):
    """beta feature, unstable API and behaviour"""


class TqdmDeprecationWarning(TqdmWarning, DeprecationWarning):
    """may be removed in a future release"""
    # not suppressed if raised


class TqdmMonitorWarning(TqdmWarning, RuntimeWarning):
    """tqdm monitor errors which do not affect external functionality"""


def TRLock(*args, **kwargs):
    """threading RLock"""
    try:
        from threading import RLock
        return RLock(*args, **kwargs)
    except (ImportError, OSError):  # pragma: no cover
        pass


class TqdmDefaultWriteLock:
    """
    Provide a default write lock for thread and multiprocessing safety.
    Works only on platforms supporting `fork` (so Windows is excluded).
    You must initialise a `tqdm` or `TqdmDefaultWriteLock` instance
    before forking in order for the write lock to work.
    On Windows, you need to supply the lock from the parent to the children as
    an argument to joblib or the parallelism lib you use.
    """
    # global thread lock so no setup required for multithreading.
    # NB: Do not create multiprocessing lock as it sets the multiprocessing
    # context, disallowing `spawn()`/`forkserver()`
    th_lock = TRLock()

    def __init__(self):
        # Create global parallelism locks to avoid racing issues with parallel
        # bars works only if fork available (Linux/MacOSX, but not Windows)
        cls = type(self)
        root_lock = cls.th_lock
        if root_lock is not None:
            root_lock.acquire()
        cls.create_mp_lock()
        self.locks = [lk for lk in [cls.mp_lock, cls.th_lock] if lk is not None]
        if root_lock is not None:
            root_lock.release()

    def acquire(self, *a, **k):
        for lock in self.locks:
            lock.acquire(*a, **k)

    def release(self):
        for lock in self.locks[::-1]:  # Release in inverse order of acquisition
            lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()

    @classmethod
    def create_mp_lock(cls):
        if not hasattr(cls, 'mp_lock'):
            try:
                from multiprocessing import RLock
                cls.mp_lock = RLock()
            except (ImportError, OSError):  # pragma: no cover
                cls.mp_lock = None

    @classmethod
    def create_th_lock(cls):
        assert hasattr(cls, 'th_lock')
        warn("create_th_lock not needed anymore", TqdmDeprecationWarning, stacklevel=2)


class Bar:
    """
    `str.format`-able bar with format specifiers: `[width][type]`

    - `width`
      + unspecified (default): use `self.default_len`
      + `int >= 0`: overrides `self.default_len`
      + `int < 0`: subtract from `self.default_len`
    - `type`
      + `a`: ascii (`charset=self.ASCII` override)
      + `u`: unicode (`charset=self.UTF` override)
      + `b`: blank (`charset="  "` override)
    """
    ASCII = " 123456789#"
    UTF = " " + ''.join(map(chr, range(0x258F, 0x2587, -1)))
    BLANK = "  "
    COLOUR_RESET = '\x1b[0m'
    COLOUR_RGB = '\x1b[38;2;%d;%d;%dm'
    COLOURS = {'BLACK': '\x1b[30m', 'RED': '\x1b[31m', 'GREEN': '\x1b[32m',
               'YELLOW': '\x1b[33m', 'BLUE': '\x1b[34m', 'MAGENTA': '\x1b[35m',
               'CYAN': '\x1b[36m', 'WHITE': '\x1b[37m'}

    def __init__(self, frac, default_len=10, charset=UTF, colour=None):
        if not 0 <= frac <= 1:
            warn("clamping frac to range [0, 1]", TqdmWarning, stacklevel=2)
            frac = max(0, min(1, frac))
        assert default_len > 0
        self.frac = frac
        self.default_len = default_len
        self.charset = charset
        self.colour = colour

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, value):
        if not value:
            self._colour = None
            return
        try:
            if value.upper() in self.COLOURS:
                self._colour = self.COLOURS[value.upper()]
            elif value[0] == '#' and len(value) == 7:
                self._colour = self.COLOUR_RGB % tuple(
                    int(i, 16) for i in (value[1:3], value[3:5], value[5:7]))
            else:
                raise KeyError
        except (KeyError, AttributeError):
            warn(f"Unknown colour ({value}); valid choices:"
                 f" [hex (#00ff00), {', '.join(self.COLOURS)}]", TqdmWarning, stacklevel=2)
            self._colour = None

    def __format__(self, format_spec):
        if format_spec:
            _type = format_spec[-1].lower()
            try:
                charset = {'a': self.ASCII, 'u': self.UTF, 'b': self.BLANK}[_type]
            except KeyError:
                charset = self.charset
            else:
                format_spec = format_spec[:-1]
            if format_spec:
                N_BARS = int(format_spec)
                if N_BARS < 0:
                    N_BARS += self.default_len
            else:
                N_BARS = self.default_len
        else:
            charset = self.charset
            N_BARS = self.default_len

        nsyms = len(charset) - 1
        bar_length, frac_bar_length = divmod(int(self.frac * N_BARS * nsyms), nsyms)

        res = charset[-1] * bar_length
        if bar_length < N_BARS:  # whitespace padding
            res = res + charset[frac_bar_length] + charset[0] * (N_BARS - bar_length - 1)
        return self.colour + res + self.COLOUR_RESET if self.colour else res


class EMA:
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.

    Parameters
    ----------
    smoothing  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields old value) to 1 (yields new value).
    """
    def __init__(self, smoothing=0.3):
        self.alpha = smoothing
        self.last = 0
        self.calls = 0

    def __call__(self, x=None):
        """
        Parameters
        ----------
        x  : float
            New value to include in EMA.
        """
        beta = 1 - self.alpha
        if x is not None:
            self.last = self.alpha * x + beta * self.last
            self.calls += 1
        return self.last / (1 - beta ** self.calls) if self.calls else self.last


class tqdm(Comparable):
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progress bar every time a value is requested.

    Parameters
    ----------
    iterable  : iterable, optional
        Iterable to decorate with a progress bar.
        Leave blank to manually manage the updates.
    desc  : str, optional
        Prefix for the progress bar.
    total  : int or float, optional
        The number of expected iterations. If unspecified,
        len(iterable) is used if possible. If float("inf") or as a last
        resort, only basic progress statistics are displayed
        (no ETA, no progress bar).
        If `gui` is True and this parameter needs subsequent updating,
        specify an initial arbitrary large positive number,
        e.g. 9e9.
    leave  : bool, optional
        If [default: True], keeps all traces of the progress bar
        upon termination of iteration.
        If `None`, will leave only if `position` is `0`.
    file  : `io.TextIOWrapper` or `io.StringIO`, optional
        Specifies where to output the progress messages
        (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
        methods.  For encoding, see `write_bytes`.
    ncols  : int, optional
        The width of the entire output message. If specified,
        dynamically resizes the progress bar to stay within this bound.
        If unspecified, attempts to use environment width. The
        fallback is a meter width of 10 and no limit for the counter and
        statistics. If 0, will not print any meter (only stats).
    mininterval  : float, optional
        Minimum progress display update interval [default: 0.1] seconds.
    maxinterval  : float, optional
        Maximum progress display update interval [default: 10] seconds.
        Automatically adjusts `miniters` to correspond to `mininterval`
        after long display update lag. Only works if `dynamic_miniters`
        or monitor thread is enabled.
    miniters  : int or float, optional
        Minimum progress display update interval, in iterations.
        If 0 and `dynamic_miniters`, will automatically adjust to equal
        `mininterval` (more CPU efficient, good for tight loops).
        If > 0, will skip display of specified number of iterations.
        Tweak this and `mininterval` to get very efficient loops.
        If your progress is erratic with both fast and slow iterations
        (network, skipping items, etc) you should set miniters=1.
    ascii  : bool or str, optional
        If unspecified or False, use unicode (smooth blocks) to fill
        the meter. The fallback is to use ASCII characters " 123456789#".
    disable  : bool, optional
        Whether to disable the entire progress bar wrapper
        [default: False]. If set to None, disable on non-TTY.
    unit  : str, optional
        String that will be used to define the unit of each iteration
        [default: it].
    unit_scale  : bool or int or float, optional
        If 1 or True, the number of iterations will be reduced/scaled
        automatically and a metric prefix following the
        International System of Units standard will be added
        (kilo, mega, etc.) [default: False]. If any other non-zero
        number, will scale `total` and `n`.
    dynamic_ncols  : bool, optional
        If set, constantly alters `ncols` and `nrows` to the
        environment (allowing for window resizes) [default: False].
    smoothing  : float, optional
        Exponential moving average smoothing factor for speed estimates
        (ignored in GUI mode). Ranges from 0 (average speed) to 1
        (current/instantaneous speed) [default: 0.3].
    bar_format  : str, optional
        Specify a custom bar string formatting. May impact performance.
        [default: '{l_bar}{bar}{r_bar}'], where
        l_bar='{desc}: {percentage:3.0f}%|' and
        r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
            '{rate_fmt}{postfix}]'
        Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
            percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
            rate, rate_fmt, rate_noinv, rate_noinv_fmt,
            rate_inv, rate_inv_fmt, postfix, unit_divisor,
            remaining, remaining_s, eta.
        Note that a trailing ": " is automatically removed after {desc}
        if the latter is empty.
    initial  : int or float, optional
        The initial counter value. Useful when restarting a progress
        bar [default: 0]. If using float, consider specifying `{n:.3f}`
        or similar in `bar_format`, or specifying `unit_scale`.
    position  : int, optional
        Specify the line offset to print this bar (starting from 0)
        Automatic if unspecified.
        Useful to manage multiple bars at once (eg, from threads).
    postfix  : dict or *, optional
        Specify additional stats to display at the end of the bar.
        Calls `set_postfix(**postfix)` if possible (dict).
    unit_divisor  : float, optional
        [default: 1000], ignored unless `unit_scale` is True.
    write_bytes  : bool, optional
        Whether to write bytes. If (default: False) will write unicode.
    lock_args  : tuple, optional
        Passed to `refresh` for intermediate output
        (initialisation, iterating, and updating).
    nrows  : int, optional
        The screen height. If specified, hides nested bars outside this
        bound. If unspecified, attempts to use environment height.
        The fallback is 20.
    colour  : str, optional
        Bar colour (e.g. 'green', '#00ff00').
    delay  : float, optional
        Don't display until [default: 0] seconds have elapsed.
    gui  : bool, optional
        WARNING: internal parameter - do not use.
        Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
        matplotlib animations for a graphical output [default: False].

    Returns
    -------
    out  : decorated iterator.
    """

    monitor_interval = 10  # set to 0 to disable the thread
    monitor = None
    _instances = WeakSet()

    @staticmethod
    def format_sizeof(num, suffix='', divisor=1000):
        """
        Formats a number (greater than unity) with SI Order of Magnitude
        prefixes.

        Parameters
        ----------
        num  : float
            Number ( >= 1) to format.
        suffix  : str, optional
            Post-postfix [default: ''].
        divisor  : float, optional
            Divisor between prefixes [default: 1000].

        Returns
        -------
        out  : str
            Number with Order of Magnitude SI unit postfix.
        """
        for unit in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z']:
            if abs(num) < 999.5:
                if abs(num) < 99.95:
                    if abs(num) < 9.995:
                        return f'{num:1.2f}{unit}{suffix}'
                    return f'{num:2.1f}{unit}{suffix}'
                return f'{num:3.0f}{unit}{suffix}'
            num /= divisor
        return f'{num:3.1f}Y{suffix}'

    @staticmethod
    def format_interval(t):
        """
        Formats a number of seconds as a clock time, [H:]MM:SS

        Parameters
        ----------
        t  : int
            Number of seconds.

        Returns
        -------
        out  : str
            [H:]MM:SS
        """
        sign = '-' if t < 0 else ''
        mins, s = divmod(abs(int(t)), 60)
        h, m = divmod(mins, 60)
        return f'{sign}{h:d}:{m:02d}:{s:02d}' if h else f'{sign}{m:02d}:{s:02d}'

    @staticmethod
    def format_num(n):
        """
        Intelligent scientific notation (.3g).

        Parameters
        ----------
        n  : int or float or Numeric
            A Number.

        Returns
        -------
        out  : str
            Formatted number.
        """
        f = f'{n:.3g}'.replace('e+0', 'e+').replace('e-0', 'e-')
        n = str(n)
        return f if len(f) < len(n) else n

    @staticmethod
    def status_printer(file):
        """
        Manage the printing and in-place updating of a line of characters.
        Note that if the string is longer than a line, then in-place
        updating may not work (it will print a new line at each refresh).
        """
        fp = file
        fp_flush = getattr(fp, 'flush', lambda: None)  # pragma: no cover
        if fp in (sys.stderr, sys.stdout):
            getattr(sys.stderr, 'flush', lambda: None)()
            getattr(sys.stdout, 'flush', lambda: None)()

        def fp_write(s):
            fp.write(str(s))
            fp_flush()

        last_len = [0]

        def print_status(s):
            len_s = disp_len(s)
            fp_write('\r' + s + (' ' * max(last_len[0] - len_s, 0)))
            last_len[0] = len_s

        return print_status

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='',
                     ascii=False,  # pylint: disable=redefined-builtin
                     unit='it', unit_scale=False, rate=None, bar_format=None, postfix=None,
                     unit_divisor=1000, initial=0, colour=None, **extra_kwargs):
        """
        Return a string-based progress bar given some parameters

        Parameters
        ----------
        n  : int or float
            Number of finished iterations.
        total  : int or float
            The expected total number of iterations. If meaningless (None),
            only basic progress statistics are displayed (no ETA).
        elapsed  : float
            Number of seconds passed since start.
        ncols  : int, optional
            The width of the entire output message. If specified,
            dynamically resizes `{bar}` to stay within this bound
            [default: None]. If `0`, will not print any bar (only stats).
            The fallback is `{bar:10}`.
        prefix  : str, optional
            Prefix message (included in total width) [default: ''].
            Use as {desc} in bar_format string.
        ascii  : bool, optional or str, optional
            If not set, use unicode (smooth blocks) to fill the meter
            [default: False]. The fallback is to use ASCII characters
            " 123456789#".
        unit  : str, optional
            The iteration unit [default: 'it'].
        unit_scale  : bool or int or float, optional
            If 1 or True, the number of iterations will be printed with an
            appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)
            [default: False]. If any other non-zero number, will scale
            `total` and `n`.
        rate  : float, optional
            Manual override for iteration rate.
            If [default: None], uses n/elapsed.
        bar_format  : str, optional
            Specify a custom bar string formatting. May impact performance.
            [default: '{l_bar}{bar}{r_bar}'], where
            l_bar='{desc}: {percentage:3.0f}%|' and
            r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
              '{rate_fmt}{postfix}]'
            Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
              percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
              rate, rate_fmt, rate_noinv, rate_noinv_fmt,
              rate_inv, rate_inv_fmt, postfix, unit_divisor,
              remaining, remaining_s, eta.
            Note that a trailing ": " is automatically removed after {desc}
            if the latter is empty.
        postfix  : *, optional
            Similar to `prefix`, but placed at the end
            (e.g. for additional stats).
            Note: postfix is usually a string (not a dict) for this method,
            and will if possible be set to postfix = ', ' + postfix.
            However other types are supported (#382).
        unit_divisor  : float, optional
            [default: 1000], ignored unless `unit_scale` is True.
        initial  : int or float, optional
            The initial counter value [default: 0].
        colour  : str, optional
            Bar colour (e.g. 'green', '#00ff00').

        Returns
        -------
        out  : Formatted meter and stats, ready to display.
        """

        # sanity check: total
        if total and (n >= (total + 0.5) or total == float("inf")):
            # allow float imprecision (#849) or inf (#651)
            total = None

        # apply custom scale if necessary
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= unit_scale  # by default rate = self.avg_dn / self.avg_dt
            unit_scale = False

        elapsed_str = tqdm.format_interval(elapsed)

        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = (n - initial) / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = tqdm.format_sizeof
        rate_noinv_fmt = ((format_sizeof(rate) if unit_scale else f'{rate:5.2f}')
                          if rate else '?') + unit + '/s'
        rate_inv_fmt = (
            (format_sizeof(inv_rate) if unit_scale else f'{inv_rate:5.2f}')
            if inv_rate else '?') + 's/' + unit
        rate_fmt = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = format_sizeof(total, divisor=unit_divisor) if total is not None else '?'
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'

        try:
            postfix = ', ' + postfix if postfix else ''
        except TypeError:
            pass

        remaining = (total - n) / rate if rate and total else 0
        remaining_str = tqdm.format_interval(remaining) if rate else '?'
        try:
            eta_dt = (datetime.now() + timedelta(seconds=remaining)
                      if rate and total else datetime.fromtimestamp(0, timezone.utc))
        except OverflowError:
            eta_dt = datetime.max

        # format the stats displayed to the left and right sides of the bar
        if prefix:
            # old prefix setup work around
            bool_prefix_colon_already = (prefix[-2:] == ": ")
            l_bar = prefix if bool_prefix_colon_already else prefix + ": "
        else:
            l_bar = ''

        r_bar = f'| {n_fmt}/{total_fmt} [{elapsed_str}<{remaining_str}, {rate_fmt}{postfix}]'

        # Custom bar formatting
        # Populate a dict with all available progress indicators
        format_dict = {
            # slight extension of self.format_dict
            'n': n, 'n_fmt': n_fmt, 'total': total, 'total_fmt': total_fmt,
            'elapsed': elapsed_str, 'elapsed_s': elapsed,
            'ncols': ncols, 'desc': prefix or '', 'unit': unit,
            'rate': inv_rate if inv_rate and inv_rate > 1 else rate,
            'rate_fmt': rate_fmt, 'rate_noinv': rate,
            'rate_noinv_fmt': rate_noinv_fmt, 'rate_inv': inv_rate,
            'rate_inv_fmt': rate_inv_fmt,
            'postfix': postfix, 'unit_divisor': unit_divisor,
            'colour': colour,
            # plus more useful definitions
            'remaining': remaining_str, 'remaining_s': remaining,
            'l_bar': l_bar, 'r_bar': r_bar, 'eta': eta_dt,
            **extra_kwargs}

        # total is known: we can predict some stats
        if total:
            # fractional and percentage progress
            frac = n / total
            percentage = frac * 100

            l_bar += f'{percentage:3.0f}%|'

            if ncols == 0:
                return l_bar[:-1] + r_bar[1:]

            format_dict.update(l_bar=l_bar)
            if bar_format:
                format_dict.update(percentage=percentage)

                # auto-remove colon for empty `{desc}`
                if not prefix:
                    bar_format = bar_format.replace("{desc}: ", '')
            else:
                bar_format = "{l_bar}{bar}{r_bar}"

            full_bar = FormatReplace()
            nobar = bar_format.format(bar=full_bar, **format_dict)  # no `{bar}`
            if not full_bar.format_called:
                return disp_trim(nobar, ncols) if ncols else nobar

            # Formatting progress bar space available for bar's display
            full_bar = Bar(frac,
                           max(1, ncols - disp_len(nobar)) if ncols else 10,
                           charset=Bar.ASCII if ascii is True else ascii or Bar.UTF,
                           colour=colour)
            if not _is_ascii(full_bar.charset) and _is_ascii(bar_format):
                bar_format = str(bar_format)
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res

        elif bar_format:
            # user-specified bar_format but no total
            l_bar += '|'
            format_dict.update(l_bar=l_bar, percentage=0)
            full_bar = FormatReplace()
            nobar = bar_format.format(bar=full_bar, **format_dict)
            if not full_bar.format_called:
                return disp_trim(nobar, ncols) if ncols else nobar
            full_bar = Bar(0,
                           max(1, ncols - disp_len(nobar)) if ncols else 10,
                           charset=Bar.BLANK, colour=colour)
            res = bar_format.format(bar=full_bar, **format_dict)
            return disp_trim(res, ncols) if ncols else res
        else:
            # no total: no bar & ETA, just progress stats
            res = (f'{(prefix + ": ") if prefix else ""}'
                   f'{n_fmt}{unit} [{elapsed_str}, {rate_fmt}{postfix}]')
            return disp_trim(res, ncols) if ncols else res

    def __new__(cls, *_, **__):
        instance = object.__new__(cls)
        with cls.get_lock():  # also constructs lock if non-existent
            cls._instances.add(instance)
            # create monitoring thread
            if cls.monitor_interval and (cls.monitor is None
                                         or not cls.monitor.report()):
                try:
                    cls.monitor = TMonitor(cls, cls.monitor_interval)
                except Exception as e:  # pragma: nocover
                    warn("tqdm:disabling monitor support"
                         " (monitor_interval = 0) due to:\n" + str(e),
                         TqdmMonitorWarning, stacklevel=2)
                    cls.monitor_interval = 0
        return instance

    @classmethod
    def _get_free_pos(cls, instance=None):
        """Skips specified instance."""
        positions = {abs(inst.pos) for inst in cls._instances
                     if inst is not instance and hasattr(inst, "pos")}
        return min(set(range(len(positions) + 1)).difference(positions))

    @classmethod
    def _decr_instances(cls, instance):
        """
        Remove from list and reposition another unfixed bar
        to fill the new gap.

        This means that by default (where all nested bars are unfixed),
        order is not maintained but screen flicker/blank space is minimised.
        (tqdm<=4.44.1 moved ALL subsequent unfixed bars up.)
        """
        with cls._lock:
            try:
                cls._instances.remove(instance)
            except KeyError:
                # if not instance.gui:  # pragma: no cover
                #     raise
                pass  # py2: maybe magically removed already
            # else:
            if not instance.gui:
                last = (instance.nrows or 20) - 1
                # find unfixed (`pos >= 0`) overflow (`pos >= nrows - 1`)
                instances = list(filter(
                    lambda i: hasattr(i, "pos") and last <= i.pos,
                    cls._instances))
                # set first found to current `pos`
                if instances:
                    inst = min(instances, key=lambda i: i.pos)
                    inst.clear(nolock=True)
                    inst.pos = abs(instance.pos)

    @classmethod
    def write(cls, s, file=None, end="\n", nolock=False):
        """Print a message via tqdm (without overlap with bars)."""
        fp = file if file is not None else sys.stdout
        if fp is None:
            return
        with cls.external_write_mode(file=file, nolock=nolock):
            # Write the message
            fp.write(s)
            fp.write(end)

    @classmethod
    @contextmanager
    def external_write_mode(cls, file=None, nolock=False):
        """
        Disable tqdm within context and refresh tqdm when exits.
        Useful when writing to standard output stream
        """
        fp = file if file is not None else sys.stdout
        if fp is None:
            yield
            return

        try:
            if not nolock:
                cls.get_lock().acquire()
            # Clear all bars
            inst_cleared = []
            for inst in getattr(cls, '_instances', []):
                # Clear instance if in the target output file
                # or if write output + tqdm output are both either
                # sys.stdout or sys.stderr (because both are mixed in terminal)
                if hasattr(inst, "start_t") and (inst.fp == fp or all(
                        f in (sys.stdout, sys.stderr) for f in (fp, inst.fp))):
                    inst.clear(nolock=True)
                    inst_cleared.append(inst)
            yield
            # Force refresh display of bars we cleared
            for inst in inst_cleared:
                inst.refresh(nolock=True)
        finally:
            if not nolock:
                cls._lock.release()

    @classmethod
    def set_lock(cls, lock):
        """Set the global lock."""
        cls._lock = lock

    @classmethod
    def get_lock(cls):
        """Get the global lock. Construct it if it does not exist."""
        if not hasattr(cls, '_lock'):
            cls._lock = TqdmDefaultWriteLock()
        return cls._lock

    @classmethod
    def pandas(cls, **tqdm_kwargs):
        """
        Registers the current `tqdm` class with
            pandas.core.
            ( frame.DataFrame
            | series.Series
            | groupby.(generic.)DataFrameGroupBy
            | groupby.(generic.)SeriesGroupBy
            ).progress_apply

        A new instance will be created every time `progress_apply` is called,
        and each instance will automatically `close()` upon completion.

        Parameters
        ----------
        tqdm_kwargs  : arguments for the tqdm instance

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from tqdm import tqdm
        >>> from tqdm.gui import tqdm as tqdm_gui
        >>>
        >>> df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
        >>> tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
        >>> # Now you can use `progress_apply` instead of `apply`
        >>> df.groupby(0).progress_apply(lambda x: x**2)

        References
        ----------
        <https://stackoverflow.com/questions/18603270/\
        progress-indicator-during-pandas-operations-python>
        """
        from warnings import catch_warnings, simplefilter

        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
        try:
            with catch_warnings():
                simplefilter("ignore", category=FutureWarning)
                from pandas import Panel
        except ImportError:  # pandas>=1.2.0
            Panel = None
        Rolling, Expanding = None, None
        try:  # pandas>=1.0.0
            from pandas.core.window.rolling import _Rolling_and_Expanding
        except ImportError:
            try:  # pandas>=0.18.0
                from pandas.core.window import _Rolling_and_Expanding
            except ImportError:  # pandas>=1.2.0
                try:  # pandas>=1.2.0
                    from pandas.core.window.expanding import Expanding
                    from pandas.core.window.rolling import Rolling
                    _Rolling_and_Expanding = Rolling, Expanding
                except ImportError:  # pragma: no cover
                    _Rolling_and_Expanding = None
        try:  # pandas>=0.25.0
            from pandas.core.groupby.generic import SeriesGroupBy  # , NDFrameGroupBy
            from pandas.core.groupby.generic import DataFrameGroupBy
        except ImportError:  # pragma: no cover
            try:  # pandas>=0.23.0
                from pandas.core.groupby.groupby import DataFrameGroupBy, SeriesGroupBy
            except ImportError:
                from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import GroupBy
        except ImportError:  # pragma: no cover
            from pandas.core.groupby import GroupBy

        try:  # pandas>=0.23.0
            from pandas.core.groupby.groupby import PanelGroupBy
        except ImportError:
            try:
                from pandas.core.groupby import PanelGroupBy
            except ImportError:  # pandas>=0.25.0
                PanelGroupBy = None

        tqdm_kwargs = tqdm_kwargs.copy()
        deprecated_t = [tqdm_kwargs.pop('deprecated_t', None)]

        def inner_generator(df_function='apply'):
            def inner(df, func, *args, **kwargs):
                """
                Parameters
                ----------
                df  : (DataFrame|Series)[GroupBy]
                    Data (may be grouped).
                func  : function
                    To be applied on the (grouped) data.
                **kwargs  : optional
                    Transmitted to `df.apply()`.
                """

                # Precompute total iterations
                total = tqdm_kwargs.pop("total", getattr(df, 'ngroups', None))
                if total is None:  # not grouped
                    if df_function == 'applymap':
                        total = df.size
                    elif isinstance(df, Series):
                        total = len(df)
                    elif (_Rolling_and_Expanding is None or
                          not isinstance(df, _Rolling_and_Expanding)):
                        # DataFrame or Panel
                        axis = kwargs.get('axis', 0)
                        if axis == 'index':
                            axis = 0
                        elif axis == 'columns':
                            axis = 1
                        # when axis=0, total is shape[axis1]
                        total = df.size // df.shape[axis]

                # Init bar
                if deprecated_t[0] is not None:
                    t = deprecated_t[0]
                    deprecated_t[0] = None
                else:
                    t = cls(total=total, **tqdm_kwargs)

                if len(args) > 0:
                    # *args intentionally not supported (see #244, #299)
                    TqdmDeprecationWarning(
                        "Except func, normal arguments are intentionally" +
                        " not supported by" +
                        " `(DataFrame|Series|GroupBy).progress_apply`." +
                        " Use keyword arguments instead.",
                        fp_write=getattr(t.fp, 'write', sys.stderr.write))

                try:  # pandas>=1.3.0,<3.0
                    from pandas.core.common import is_builtin_func
                except ImportError:  # pandas<1.3.0
                    is_builtin_func = getattr(df, '_is_builtin_func', lambda f: f)
                try:
                    func = is_builtin_func(func)
                except TypeError:
                    pass

                # Define bar updating wrapper
                def wrapper(*args, **kwargs):
                    # update tbar correctly
                    # it seems `pandas apply` calls `func` twice
                    # on the first column/row to decide whether it can
                    # take a fast or slow code path; so stop when t.total==t.n
                    t.update(n=1 if not t.total or t.n < t.total else 0)
                    return func(*args, **kwargs)

                # Apply the provided function (in **kwargs)
                # on the df using our wrapper (which provides bar updating)
                try:
                    return getattr(df, df_function)(wrapper, **kwargs)
                finally:
                    t.close()

            return inner

        # Monkeypatch pandas to provide easy methods
        # Enable custom tqdm progress in pandas!
        Series.progress_apply = inner_generator()
        SeriesGroupBy.progress_apply = inner_generator()
        Series.progress_map = inner_generator('map')
        SeriesGroupBy.progress_map = inner_generator('map')

        DataFrame.progress_apply = inner_generator()
        DataFrameGroupBy.progress_apply = inner_generator()
        DataFrame.progress_applymap = inner_generator('applymap')
        DataFrame.progress_map = inner_generator('map')
        DataFrameGroupBy.progress_map = inner_generator('map')

        if Panel is not None:
            Panel.progress_apply = inner_generator()
        if PanelGroupBy is not None:
            PanelGroupBy.progress_apply = inner_generator()

        GroupBy.progress_apply = inner_generator()
        GroupBy.progress_aggregate = inner_generator('aggregate')
        GroupBy.progress_transform = inner_generator('transform')

        if Rolling is not None and Expanding is not None:
            Rolling.progress_apply = inner_generator()
            Expanding.progress_apply = inner_generator()
        elif _Rolling_and_Expanding is not None:
            _Rolling_and_Expanding.progress_apply = inner_generator()

    # override defaults via env vars
    @envwrap("tqdm", is_method=True, types={'total': float, 'ncols': int, 'miniters': float,
                                            'position': int, 'nrows': int})
    def __init__(self, iterable=None, desc=None, total=None, leave=True, file=None,
                 ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
                 ascii=None,  # pylint: disable=redefined-builtin
                 disable=False, unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3,
                 bar_format=None, initial=0, position=None, postfix=None, unit_divisor=1000,
                 write_bytes=False, lock_args=None, nrows=None, colour=None, delay=0.0, gui=False,
                 **kwargs):
        """see tqdm.tqdm for arguments"""
        if file is None:
            file = sys.stderr

        if write_bytes:
            # Despite coercing unicode into bytes, py2 sys.std* streams
            # should have bytes written to them.
            file = SimpleTextIOWrapper(
                file, encoding=getattr(file, 'encoding', None) or 'utf-8')

        file = DisableOnWriteError(file, tqdm_instance=self)

        if disable is None and hasattr(file, "isatty") and not file.isatty():
            disable = True

        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                total = None
        if total == float("inf"):
            total = None  # same as unknown

        if disable:
            self.iterable = iterable
            self.disable = disable
            with self._lock:
                self.pos = self._get_free_pos(self)
                self._instances.remove(self)
            self.n = initial
            self.total = total
            self.leave = leave
            return

        if kwargs:
            self.disable = True
            with self._lock:
                self.pos = self._get_free_pos(self)
                self._instances.remove(self)
            raise (
                TqdmDeprecationWarning(
                    "`nested` is deprecated and automated.\n"
                    "Use `position` instead for manual control.\n",
                    fp_write=getattr(file, 'write', sys.stderr.write))
                if "nested" in kwargs else
                TqdmKeyError("Unknown argument(s): " + str(kwargs)))

        # Preprocess the arguments
        if (
            (ncols is None or nrows is None) and (file in (sys.stderr, sys.stdout))
        ) or dynamic_ncols:  # pragma: no cover
            if dynamic_ncols:
                dynamic_ncols = _screen_shape_wrapper()
                if dynamic_ncols:
                    ncols, nrows = dynamic_ncols(file)
            else:
                _dynamic_ncols = _screen_shape_wrapper()
                if _dynamic_ncols:
                    _ncols, _nrows = _dynamic_ncols(file)
                    if ncols is None:
                        ncols = _ncols
                    if nrows is None:
                        nrows = _nrows

        if miniters is None:
            miniters = 0
            dynamic_miniters = True
        else:
            dynamic_miniters = False

        if mininterval is None:
            mininterval = 0

        if maxinterval is None:
            maxinterval = 0

        if ascii is None:
            ascii = not _supports_unicode(file)

        if bar_format and ascii is not True and not _is_ascii(ascii):
            # Convert bar format into unicode since terminal uses unicode
            bar_format = str(bar_format)

        if smoothing is None:
            smoothing = 0

        # Store the arguments
        self.iterable = iterable
        self.desc = desc or ''
        self.total = total
        self.leave = leave
        self.fp = file
        self.ncols = ncols
        self.nrows = nrows
        self.mininterval = mininterval
        self.maxinterval = maxinterval
        self.miniters = miniters
        self.dynamic_miniters = dynamic_miniters
        self.ascii = ascii
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.initial = initial
        self.lock_args = lock_args
        self.delay = delay
        self.gui = gui
        self.dynamic_ncols = dynamic_ncols
        self.smoothing = smoothing
        self._ema_dn = EMA(smoothing)
        self._ema_dt = EMA(smoothing)
        self._ema_miniters = EMA(smoothing)
        self.bar_format = bar_format
        self.postfix = None
        self.colour = colour
        self._time = time
        if postfix:
            try:
                self.set_postfix(refresh=False, **postfix)
            except TypeError:
                self.postfix = postfix

        # Init the iterations counters
        self.last_print_n = initial
        self.n = initial

        # if nested, at initial sp() call we replace '\r' by '\n' to
        # not overwrite the outer progress bar
        with self._lock:
            # mark fixed positions as negative
            self.pos = self._get_free_pos(self) if position is None else -position

        if not gui:
            # Initialize the screen printer
            self.sp = self.status_printer(self.fp)
            if delay <= 0:
                self.refresh(lock_args=self.lock_args)

        # Init the time counter
        self.last_print_t = self._time()
        # NB: Avoid race conditions by setting start_t at the very end of init
        self.start_t = self.last_print_t

    def __bool__(self):
        if self.total is not None:
            return self.total > 0
        if self.iterable is None:
            raise TypeError('bool() undefined when iterable == total == None')
        return bool(self.iterable)

    def __len__(self):
        return (
            self.total if self.iterable is None
            else self.iterable.shape[0] if hasattr(self.iterable, "shape")
            else len(self.iterable) if hasattr(self.iterable, "__len__")
            else self.iterable.__length_hint__() if hasattr(self.iterable, "__length_hint__")
            else getattr(self, "total", None))

    def __reversed__(self):
        try:
            orig = self.iterable
        except AttributeError:
            raise TypeError("'tqdm' object is not reversible")
        else:
            self.iterable = reversed(self.iterable)
            return self.__iter__()
        finally:
            self.iterable = orig

    def __contains__(self, item):
        contains = getattr(self.iterable, '__contains__', None)
        return (contains(item) if contains is not None  # pylint: disable=not-callable
                else item in self.__iter__())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except AttributeError:
            # maybe eager thread cleanup upon external error
            if (exc_type, exc_value, traceback) == (None, None, None):
                raise
            warn("AttributeError ignored", TqdmWarning, stacklevel=2)

    def __del__(self):
        self.close()

    def __str__(self):
        return self.format_meter(**self.format_dict)

    @property
    def _comparable(self):
        return abs(getattr(self, "pos", 1 << 31))

    def __hash__(self):
        return id(self)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            yield from iterable
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj
                # Update and possibly print the progress bar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def update(self, n=1):
        """
        Manually update the progress bar, useful for streams
        such as reading files.
        E.g.:
        >>> t = tqdm(total=filesize) # Initialise
        >>> for current_buffer in stream:
        ...    ...
        ...    t.update(len(current_buffer))
        >>> t.close()
        The last line is highly recommended, but possibly not necessary if
        `t.update()` will be called in such a way that `filesize` will be
        exactly reached and printed.

        Parameters
        ----------
        n  : int or float, optional
            Increment to add to the internal counter of iterations
            [default: 1]. If using float, consider specifying `{n:.3f}`
            or similar in `bar_format`, or specifying `unit_scale`.

        Returns
        -------
        out  : bool or None
            True if a `display()` was triggered.
        """
        if self.disable:
            return

        if n < 0:
            self.last_print_n += n  # for auto-refresh logic to work
        self.n += n

        # check counter first to reduce calls to time()
        if self.n - self.last_print_n >= self.miniters:
            cur_t = self._time()
            dt = cur_t - self.last_print_t
            if dt >= self.mininterval and cur_t >= self.start_t + self.delay:
                cur_t = self._time()
                dn = self.n - self.last_print_n  # >= n
                if self.smoothing and dt and dn:
                    # EMA (not just overall average)
                    self._ema_dn(dn)
                    self._ema_dt(dt)
                self.refresh(lock_args=self.lock_args)
                if self.dynamic_miniters:
                    # If no `miniters` was specified, adjust automatically to the
                    # maximum iteration rate seen so far between two prints.
                    # e.g.: After running `tqdm.update(5)`, subsequent
                    # calls to `tqdm.update()` will only cause an update after
                    # at least 5 more iterations.
                    if self.maxinterval and dt >= self.maxinterval:
                        self.miniters = dn * (self.mininterval or self.maxinterval) / dt
                    elif self.smoothing:
                        # EMA miniters update
                        self.miniters = self._ema_miniters(
                            dn * (self.mininterval / dt if self.mininterval and dt
                                  else 1))
                    else:
                        # max iters between two prints
                        self.miniters = max(self.miniters, dn)

                # Store old values for next call
                self.last_print_n = self.n
                self.last_print_t = cur_t
                return True

    def close(self):
        """Cleanup and (if leave=False) close the progress bar."""
        if getattr(self, 'disable', True):
            return

        # Prevent multiple closures
        self.disable = True

        # decrement instance pos and remove from internal set
        pos = abs(self.pos)
        self._decr_instances(self)

        if not hasattr(self, 'last_print_t'):
            return
        if self.last_print_t < self.start_t + self.delay:
            # haven't ever displayed; nothing to clear
            return

        # GUI mode
        if getattr(self, 'sp', None) is None:
            return

        # annoyingly, _supports_unicode isn't good enough
        def fp_write(s):
            self.fp.write(str(s))

        try:
            fp_write('')
        except ValueError as e:
            if 'closed' in str(e):
                return
            raise  # pragma: no cover

        leave = pos == 0 if self.leave is None else self.leave

        with self._lock:
            if leave:
                # stats for overall rate (no weighted average)
                self._ema_dt = lambda: None
                self.display(pos=0)
                fp_write('\n')
            else:
                # clear previous display
                if self.display(msg='', pos=pos) and not pos:
                    fp_write('\r')

    def clear(self, nolock=False):
        """Clear current bar display."""
        if self.disable:
            return

        if not nolock:
            self._lock.acquire()
        pos = abs(self.pos)
        if pos < (self.nrows or 20):
            self.moveto(pos)
            self.sp('')
            self.fp.write('\r')  # place cursor back at the beginning of line
            self.moveto(-pos)
        if not nolock:
            self._lock.release()

    def refresh(self, nolock=False, lock_args=None):
        """
        Force refresh the display of this bar.

        Parameters
        ----------
        nolock  : bool, optional
            If `True`, does not lock.
            If [default: `False`]: calls `acquire()` on internal lock.
        lock_args  : tuple, optional
            Passed to internal lock's `acquire()`.
            If specified, will only `display()` if `acquire()` returns `True`.
        """
        if self.disable:
            return

        if not nolock:
            if lock_args:
                if not self._lock.acquire(*lock_args):
                    return False
            else:
                self._lock.acquire()
        self.display()
        if not nolock:
            self._lock.release()
        return True

    def unpause(self):
        """Restart tqdm timer from last print time."""
        if self.disable:
            return
        cur_t = self._time()
        self.start_t += cur_t - self.last_print_t
        self.last_print_t = cur_t

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Consider combining with `leave=True`.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        self.n = 0
        if total is not None:
            self.total = None if total == float("inf") else total
        if self.disable:
            return
        self.last_print_n = 0
        self.last_print_t = self.start_t = self._time()
        self._ema_dn = EMA(self.smoothing)
        self._ema_dt = EMA(self.smoothing)
        self._ema_miniters = EMA(self.smoothing)
        self.refresh()

    def set_description(self, desc=None, refresh=True):
        """
        Set/modify description of the progress bar.

        Parameters
        ----------
        desc  : str, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        """
        self.desc = desc + ': ' if desc else ''
        if refresh:
            self.refresh()

    def set_description_str(self, desc=None, refresh=True):
        """Set/modify description without ': ' appended."""
        self.desc = desc or ''
        if refresh:
            self.refresh()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        """
        Set/modify postfix (additional stats)
        with automatic formatting based on datatype.

        Parameters
        ----------
        ordered_dict  : dict or OrderedDict, optional
        refresh  : bool, optional
            Forces refresh [default: True].
        kwargs  : dict, optional
        """
        # Sort in alphabetical order to be more deterministic
        postfix = OrderedDict([] if ordered_dict is None else ordered_dict)
        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = self.format_num(postfix[key])
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        # Stitch together to get the final postfix
        self.postfix = ', '.join(key + '=' + postfix[key].strip()
                                 for key in postfix.keys())
        if refresh:
            self.refresh()

    def set_postfix_str(self, s='', refresh=True):
        """
        Postfix without dictionary expansion, similar to prefix handling.
        """
        self.postfix = str(s)
        if refresh:
            self.refresh()

    def moveto(self, n):
        # TODO: private method
        self.fp.write('\n' * n + _term_move_up() * -n)
        getattr(self.fp, 'flush', lambda: None)()

    @property
    def format_dict(self):
        """Public API for read-only member access."""
        if self.disable and not hasattr(self, 'unit'):
            return defaultdict(lambda: None, {
                'n': self.n, 'total': self.total, 'elapsed': 0, 'unit': 'it'})
        if self.dynamic_ncols:
            self.ncols, self.nrows = self.dynamic_ncols(self.fp)
        return {
            'n': self.n, 'total': self.total,
            'elapsed': self._time() - self.start_t if hasattr(self, 'start_t') else 0,
            'ncols': self.ncols, 'nrows': self.nrows, 'prefix': self.desc,
            'ascii': self.ascii, 'unit': self.unit, 'unit_scale': self.unit_scale,
            'rate': self._ema_dn() / self._ema_dt() if self._ema_dt() else None,
            'bar_format': self.bar_format, 'postfix': self.postfix,
            'unit_divisor': self.unit_divisor, 'initial': self.initial,
            'colour': self.colour}

    def display(self, msg=None, pos=None):
        """
        Use `self.sp` to display `msg` in the specified `pos`.

        Consider overloading this function when inheriting to use e.g.:
        `self.some_frontend(**self.format_dict)` instead of `self.sp`.

        Parameters
        ----------
        msg  : str, optional. What to display (default: `repr(self)`).
        pos  : int, optional. Position to `moveto`
          (default: `abs(self.pos)`).
        """
        if pos is None:
            pos = abs(self.pos)

        nrows = self.nrows or 20
        if pos >= nrows - 1:
            if pos >= nrows:
                return False
            if msg or msg is None:  # override at `nrows - 1`
                msg = " ... (more hidden) ..."

        if not hasattr(self, "sp"):
            raise TqdmDeprecationWarning(
                "Please use `tqdm.gui.tqdm(...)`"
                " instead of `tqdm(..., gui=True)`\n",
                fp_write=getattr(self.fp, 'write', sys.stderr.write))

        if pos:
            self.moveto(pos)
        self.sp(self.__str__() if msg is None else msg)
        if pos:
            self.moveto(-pos)
        return True

    @classmethod
    @contextmanager
    def wrapattr(cls, stream, method, total=None, bytes=True,  # pylint: disable=redefined-builtin
                 **tqdm_kwargs):
        """
        stream  : file-like object.
        method  : str, "read" or "write". The result of `read()` and
            the first argument of `write()` should have a `len()`.

        >>> with tqdm.wrapattr(file_obj, "read", total=file_obj.size) as fobj:
        ...     while True:
        ...         chunk = fobj.read(chunk_size)
        ...         if not chunk:
        ...             break
        """
        with cls(total=total, **tqdm_kwargs) as t:
            if bytes:
                t.unit = "B"
                t.unit_scale = True
                t.unit_divisor = 1024
            yield CallbackIOWrapper(t.update, stream, method)


def trange(*args, **kwargs):
    """Shortcut for tqdm(range(*args), **kwargs)."""
    return tqdm(range(*args), **kwargs)
<|endoftext|>"""
`rich.progress` decorator for iterators.

Usage:
>>> from tqdm.rich import trange, tqdm
>>> for i in trange(10):
...     ...
"""
from warnings import warn

from rich.progress import (
    BarColumn, Progress, ProgressColumn, Text, TimeElapsedColumn, TimeRemainingColumn, filesize)

from .std import TqdmExperimentalWarning
from .std import tqdm as std_tqdm

__author__ = {"github.com/": ["casperdcl"]}
__all__ = ['tqdm_rich', 'trrange', 'tqdm', 'trange']


class FractionColumn(ProgressColumn):
    """Renders completed/total, e.g. '0.5/2.3 G'."""
    def __init__(self, unit_scale=False, unit_divisor=1000):
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Calculate common unit for completed and total."""
        completed = int(task.completed)
        # total is None when the iterable has no length (e.g. a generator)
        total = int(task.total) if task.total is not None else None
        base = total if total is not None else completed
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                base,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(base, [""], 1)
        precision = 0 if unit == 1 else 1
        total_str = f"{total/unit:,.{precision}f}" if total is not None else "?"
        return Text(
            f"{completed/unit:,.{precision}f}/{total_str} {suffix}",
            style="progress.download")


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""
    def __init__(self, unit="", unit_scale=False, unit_divisor=1000):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"? {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        precision = 0 if unit == 1 else 1
        return Text(f"{speed/unit:,.{precision}f} {suffix}{self.unit}/s",
                    style="progress.data.speed")


class tqdm_rich(std_tqdm):  # pragma: no cover
    """Experimental rich.progress GUI version of tqdm!"""
    # TODO: @classmethod: write()?
    def __init__(self, *args, **kwargs):
        """
        This class accepts the following parameters *in addition* to
        the parameters accepted by `tqdm`.

        Parameters
        ----------
        progress  : tuple, optional
            arguments for `rich.progress.Progress()`.
        options  : dict, optional
            keyword arguments for `rich.progress.Progress()`.
        """
        kwargs = kwargs.copy()
        kwargs['gui'] = True
        # convert disable = None to False
        kwargs['disable'] = bool(kwargs.get('disable', False))
        progress = kwargs.pop('progress', None)
        options = kwargs.pop('options', {}).copy()
        super().__init__(*args, **kwargs)

        if self.disable:
            return

        warn("rich is experimental/alpha", TqdmExperimentalWarning, stacklevel=2)
        d = self.format_dict
        if progress is None:
            progress = (
                "[progress.description]{task.description}"
                "[progress.percentage]{task.percentage:>4.0f}%",
                BarColumn(bar_width=None),
                FractionColumn(
                    unit_scale=d['unit_scale'], unit_divisor=d['unit_divisor']),
                "[", TimeElapsedColumn(), "<", TimeRemainingColumn(),
                ",", RateColumn(unit=d['unit'], unit_scale=d['unit_scale'],
                                unit_divisor=d['unit_divisor']), "]"
            )
        options.setdefault('transient', not self.leave)
        self._prog = Progress(*progress, **options)
        self._prog.__enter__()
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        if self.disable:
            return
        self.display()  # print 100%, vis #1306
        super().close()
        self._prog.__exit__(None, None, None)

    def clear(self, *_, **__):
        pass

    def display(self, *_, **__):
        if not hasattr(self, '_task_id'):
            return
        self._prog.update(self._task_id, completed=self.n, description=self.desc)

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, '_prog'):
            self._prog.reset(total=total)
        super().reset(total=total)


def trrange(*args, **kwargs):
    """Shortcut for `tqdm.rich.tqdm(range(*args), **kwargs)`."""
    return tqdm_rich(range(*args), **kwargs)


# Aliases
tqdm = tqdm_rich
trange = trrange
<|endoftext|>"""
Module version for monitoring CLI pipes (`... | python -m tqdm | ...`).
"""
import logging
import re
import sys
from ast import literal_eval
from textwrap import indent

from .std import TqdmKeyError, TqdmTypeError, tqdm
from .version import __version__

__all__ = ["main"]
log = logging.getLogger(__name__)


def cast(val, typ):
    log.debug((val, typ))
    if " or " in typ:
        for t in typ.split(" or "):
            try:
                return cast(val, t)
            except TqdmTypeError:
                pass
        raise TqdmTypeError(f"{val} : {typ}")

    # sys.stderr.write('\ndebug | `val:type`: `' + val + ':' + typ + '`.\n')
    if typ == 'bool':
        if (val == 'True') or (val == ''):
            return True
        if val == 'False':
            return False
        raise TqdmTypeError(val + ' : ' + typ)
    if typ == 'chr':
        if len(val) == 1:
            return val.encode()
        if re.match(r"^\\\w+$", val):
            return literal_eval(f'"{val}"').encode()
        raise TqdmTypeError(f"{val} : {typ}")
    if typ == 'str':
        return val
    if typ == 'int':
        try:
            return int(val)
        except ValueError as exc:
            raise TqdmTypeError(f"{val} : {typ}") from exc
    if typ == 'float':
        try:
            return float(val)
        except ValueError as exc:
            raise TqdmTypeError(f"{val} : {typ}") from exc
    raise TqdmTypeError(f"{val} : {typ}")


def posix_pipe(fin, fout, delim=b'\\n', buf_size=256,
               callback=lambda float: None, callback_len=True):
    """
    Params
    ------
    fin  : binary file with `read(buf_size : int)` method
    fout  : binary file with `write` (and optionally `flush`) methods.
    callback  : function(float), e.g.: `tqdm.update`
    callback_len  : If (default: True) do `callback(len(buffer))`.
      Otherwise, do `callback(data) for data in buffer.split(delim)`.
    """
    fp_write = fout.write

    if not delim:
        while True:
            tmp = fin.read(buf_size)

            # flush at EOF
            if not tmp:
                getattr(fout, 'flush', lambda: None)()
                return

            fp_write(tmp)
            callback(len(tmp))
        # return

    buf = b''
    len_delim = len(delim)
    # n = 0
    while True:
        tmp = fin.read(buf_size)

        # flush at EOF
        if not tmp:
            if buf:
                fp_write(buf)
                if callback_len:
                    # n += 1 + buf.count(delim)
                    callback(1 + buf.count(delim))
                else:
                    for i in buf.split(delim):
                        callback(i)
            getattr(fout, 'flush', lambda: None)()
            return  # n

        while True:
            i = tmp.find(delim)
            if i < 0:
                buf += tmp
                break
            fp_write(buf + tmp[:i + len(delim)])
            # n += 1
            callback(1 if callback_len else (buf + tmp[:i]))
            buf = b''
            tmp = tmp[i + len_delim:]


# ((opt, type), ... )
RE_OPTS = re.compile(r'\n {4}(\S+)\s{2,}:\s*([^,]+)')
# better split method assuming no positional args
RE_SHLEX = re.compile(r'\s*(?<!\S)--?([^\s=]+)(\s+|=|$)')

# TODO: add custom support for some of the following?
UNSUPPORTED_OPTS = ('iterable', 'gui', 'out', 'file')

# The 8 leading spaces are required for consistency
CLI_EXTRA_DOC = r"""
    Extra CLI Options
    -----------------
    name  : type, optional
        TODO: find out why this is needed.
    delim  : chr, optional
        Delimiting character [default: '\n']. Use '\0' for null.
        N.B.: on Windows systems, Python converts '\n' to '\r\n'.
    buf_size  : int, optional
        String buffer size in bytes [default: 256]
        used when `delim` is specified.
    bytes  : bool, optional
        If true, will count bytes, ignore `delim`, and default
        `unit_scale` to True, `unit_divisor` to 1024, and `unit` to 'B'.
    tee  : bool, optional
        If true, passes `stdin` to both `stderr` and `stdout`.
    update  : bool, optional
        If true, will treat input as newly elapsed iterations,
        i.e. numbers to pass to `update()`. Note that this is slow
        (~2e5 it/s) since every input must be decoded as a number.
    update_to  : bool, optional
        If true, will treat input as total elapsed iterations,
        i.e. numbers to assign to `self.n`. Note that this is slow
        (~2e5 it/s) since every input must be decoded as a number.
    null  : bool, optional
        If true, will discard input (no stdout).
    manpath  : str, optional
        Directory in which to install tqdm man pages.
    comppath  : str, optional
        Directory in which to place tqdm completion.
    log  : str, optional
        CRITICAL|FATAL|ERROR|WARN(ING)|[default: 'INFO']|DEBUG|NOTSET.
"""


def main(fp=sys.stderr, argv=None):
    """
    Parameters (internal use only)
    ---------
    fp  : file-like object for tqdm
    argv  : list (default: sys.argv[1:])
    """
    if argv is None:
        argv = sys.argv[1:]
    try:
        log_idx = argv.index('--log')
    except ValueError:
        for i in argv:
            if i.startswith('--log='):
                logLevel = i[len('--log='):]
                break
        else:
            logLevel = 'INFO'
    else:
        # argv.pop(log_idx)
        # logLevel = argv.pop(log_idx)
        logLevel = argv[log_idx + 1]
    logging.basicConfig(level=getattr(logging, logLevel),
                        format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    # py<3.13 doesn't dedent docstrings
    d = (tqdm.__doc__ if sys.version_info < (3, 13)
         else indent(tqdm.__doc__, "    ")) + CLI_EXTRA_DOC

    opt_types = dict(RE_OPTS.findall(d))
    # opt_types['delim'] = 'chr'

    for o in UNSUPPORTED_OPTS:
        opt_types.pop(o)

    log.debug(sorted(opt_types.items()))

    # d = RE_OPTS.sub(r'  --\1=<\1>  : \2', d)
    split = RE_OPTS.split(d)
    opt_types_desc = zip(split[1::3], split[2::3], split[3::3])
    d = ''.join(('\n  --{0}  : {2}{3}' if otd[1] == 'bool' else
                 '\n  --{0}=<{1}>  : {2}{3}').format(
                     otd[0].replace('_', '-'), otd[0], *otd[1:])
                for otd in opt_types_desc if otd[0] not in UNSUPPORTED_OPTS)

    help_short = "Usage:\n  tqdm [--help | options]\n"
    d = help_short + """
Options:
  -h, --help     Print this help and exit.
  -v, --version  Print version and exit.
""" + d.strip('\n') + '\n'

    # opts = docopt(d, version=__version__)
    if any(v in argv for v in ('-v', '--version')):
        sys.stdout.write(__version__ + '\n')
        sys.exit(0)
    elif any(v in argv for v in ('-h', '--help')):
        sys.stdout.write(d + '\n')
        sys.exit(0)
    elif argv and argv[0][:2] != '--':
        sys.stderr.write(f"Error:Unknown argument:{argv[0]}\n{help_short}")

    argv = RE_SHLEX.split(' '.join(["tqdm"] + argv))
    opts = dict(zip(argv[1::3], argv[3::3]))

    log.debug(opts)
    opts.pop('log', True)

    tqdm_args = {'file': fp}
    try:
        for (o, v) in opts.items():
            o = o.replace('-', '_')
            try:
                tqdm_args[o] = cast(v, opt_types[o])
            except KeyError as e:
                raise TqdmKeyError(str(e))
        log.debug('args:' + str(tqdm_args))

        delim_per_char = tqdm_args.pop('bytes', False)
        update = tqdm_args.pop('update', False)
        update_to = tqdm_args.pop('update_to', False)
        if sum((delim_per_char, update, update_to)) > 1:
            raise TqdmKeyError("Can only have one of --bytes --update --update_to")
    except Exception:
        fp.write("\nError:\n" + help_short)
        stdin, stdout_write = sys.stdin, sys.stdout.write
        for i in stdin:
            stdout_write(i)
        raise
    else:
        buf_size = tqdm_args.pop('buf_size', 256)
        delim = tqdm_args.pop('delim', b'\\n')
        tee = tqdm_args.pop('tee', False)
        manpath = tqdm_args.pop('manpath', None)
        comppath = tqdm_args.pop('comppath', None)
        if tqdm_args.pop('null', False):
            class stdout:
                @staticmethod
                def write(_):
                    pass
        else:
            stdout = sys.stdout
            stdout = getattr(stdout, 'buffer', stdout)
        stdin = getattr(sys.stdin, 'buffer', sys.stdin)
        if manpath or comppath:
            try:  # py<3.9
                import importlib_resources as resources
            except ImportError:
                from importlib import resources
            from pathlib import Path

            def cp(name, dst):
                """copy resource `name` to `dst`"""
                fi = resources.files('tqdm') / name
                dst.write_bytes(fi.read_bytes())
                log.info("written:%s", dst)
            if manpath is not None:
                cp('tqdm.1', Path(manpath) / 'tqdm.1')
            if comppath is not None:
                cp('completion.sh', Path(comppath) / 'tqdm_completion.sh')
            sys.exit(0)
        if tee:
            stdout_write = stdout.write
            fp_write = getattr(fp, 'buffer', fp).write

            class stdout:  # pylint: disable=function-redefined
                @staticmethod
                def write(x):
                    with tqdm.external_write_mode(file=fp):
                        fp_write(x)
                    stdout_write(x)
        if delim_per_char:
            tqdm_args.setdefault('unit', 'B')
            tqdm_args.setdefault('unit_scale', True)
            tqdm_args.setdefault('unit_divisor', 1024)
            log.debug(tqdm_args)
            with tqdm(**tqdm_args) as t:
                posix_pipe(stdin, stdout, '', buf_size, t.update)
        elif delim == b'\\n':
            log.debug(tqdm_args)
            write = stdout.write
            if update or update_to:
                with tqdm(**tqdm_args) as t:
                    if update:
                        def callback(i):
                            t.update(literal_eval(i.decode()))
                    else:  # update_to
                        def callback(i):
                            t.update(literal_eval(i.decode()) - t.n)
                    for i in stdin:
                        write(i)
                        callback(i)
            else:
                for i in tqdm(stdin, **tqdm_args):
                    write(i)
        else:
            log.debug(tqdm_args)
            with tqdm(**tqdm_args) as t:
                callback_len = False
                if update:
                    def callback(i):
                        t.update(literal_eval(i.decode()))
                elif update_to:
                    def callback(i):
                        t.update(literal_eval(i.decode()) - t.n)
                else:
                    callback = t.update
                    callback_len = True
