        self.is_animated = self.n_frames > 1
        self._mode = "RGB" if self.rawmode == "RGBX" else self.rawmode

        # Attempt to read ICC / EXIF / XMP chunks from file
        for key, chunk_name in {
            "icc_profile": "ICCP",
            "exif": "EXIF",
            "xmp": "XMP ",
        }.items():
            if value := self._decoder.get_chunk(chunk_name):
                self.info[key] = value

        # Initialize seek state
        self._reset(reset=False)

    def _getexif(self) -> dict[int, Any] | None:
        if "exif" not in self.info:
            return None
        return self.getexif()._get_merged_dict()

    def seek(self, frame: int) -> None:
        if not self._seek_check(frame):
            return

        # Set logical frame to requested position
        self.__logical_frame = frame

    def _reset(self, reset: bool = True) -> None:
        if reset:
            self._decoder.reset()
        self.__physical_frame = 0
        self.__loaded = -1
        self.__timestamp = 0

    def _get_next(self) -> tuple[bytes, int, int]:
        # Get next frame
        ret = self._decoder.get_next()
        self.__physical_frame += 1

        # Check if an error occurred
        if ret is None:
            self._reset()  # Reset just to be safe
            self.seek(0)
            msg = "failed to decode next frame in WebP file"
            raise EOFError(msg)

        # Compute duration
        data, timestamp = ret
        duration = timestamp - self.__timestamp
        self.__timestamp = timestamp

        # libwebp gives frame end, adjust to start of frame
        timestamp -= duration
        return data, timestamp, duration

    def _seek(self, frame: int) -> None:
        if self.__physical_frame == frame:
            return  # Nothing to do
        if frame < self.__physical_frame:
            self._reset()  # Rewind to beginning
        while self.__physical_frame < frame:
            self._get_next()  # Advance to the requested frame

    def load(self) -> Image.core.PixelAccess | None:
        if self.__loaded != self.__logical_frame:
            self._seek(self.__logical_frame)

            # We need to load the image data for this frame
            data, self.info["timestamp"], self.info["duration"] = self._get_next()
            self.__loaded = self.__logical_frame

            # Set tile
            if self.fp and self._exclusive_fp:
                self.fp.close()
            self.fp = BytesIO(data)
            self.tile = [ImageFile._Tile("raw", (0, 0) + self.size, 0, self.rawmode)]

        return super().load()

    def load_seek(self, pos: int) -> None:
        pass

    def tell(self) -> int:
        return self.__logical_frame


def _convert_frame(im: Image.Image) -> Image.Image:
    # Make sure image mode is supported
    if im.mode not in ("RGBX", "RGBA", "RGB"):
        im = im.convert("RGBA" if im.has_transparency_data else "RGB")
    return im


def _save_all(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    encoderinfo = im.encoderinfo.copy()
    append_images = list(encoderinfo.get("append_images", []))

    # If total frame count is 1, then save using the legacy API, which
    # will preserve non-alpha modes
    total = 0
    for ims in [im] + append_images:
        total += getattr(ims, "n_frames", 1)
    if total == 1:
        _save(im, fp, filename)
        return

    background: int | tuple[int, ...] = (0, 0, 0, 0)
    if "background" in encoderinfo:
        background = encoderinfo["background"]
    elif "background" in im.info:
        background = im.info["background"]
        if isinstance(background, int):
            # GifImagePlugin stores a global color table index in
            # info["background"]. So it must be converted to an RGBA value
            palette = im.getpalette()
            if palette:
                r, g, b = palette[background * 3 : (background + 1) * 3]
                background = (r, g, b, 255)
            else:
                background = (background, background, background, 255)

    duration = im.encoderinfo.get("duration", im.info.get("duration", 0))
    loop = im.encoderinfo.get("loop", 0)
    minimize_size = im.encoderinfo.get("minimize_size", False)
    kmin = im.encoderinfo.get("kmin", None)
    kmax = im.encoderinfo.get("kmax", None)
    allow_mixed = im.encoderinfo.get("allow_mixed", False)
    verbose = False
    lossless = im.encoderinfo.get("lossless", False)
    quality = im.encoderinfo.get("quality", 80)
    alpha_quality = im.encoderinfo.get("alpha_quality", 100)
    method = im.encoderinfo.get("method", 0)
    icc_profile = im.encoderinfo.get("icc_profile") or ""
    exif = im.encoderinfo.get("exif", "")
    if isinstance(exif, Image.Exif):
        exif = exif.tobytes()
    xmp = im.encoderinfo.get("xmp", "")
    if allow_mixed:
        lossless = False

    # Sensible keyframe defaults are from gif2webp.c script
    if kmin is None:
        kmin = 9 if lossless else 3
    if kmax is None:
        kmax = 17 if lossless else 5

    # Validate background color
    if (
        not isinstance(background, (list, tuple))
        or len(background) != 4
        or not all(0 <= v < 256 for v in background)
    ):
        msg = f"Background color is not an RGBA tuple clamped to (0-255): {background}"
        raise OSError(msg)

    # Convert to packed uint
    bg_r, bg_g, bg_b, bg_a = background
    background = (bg_a << 24) | (bg_r << 16) | (bg_g << 8) | (bg_b << 0)

    # Setup the WebP animation encoder
    enc = _webp.WebPAnimEncoder(
        im.size,
        background,
        loop,
        minimize_size,
        kmin,
        kmax,
        allow_mixed,
        verbose,
    )

    # Add each frame
    frame_idx = 0
    timestamp = 0
    cur_idx = im.tell()
    try:
        for ims in [im] + append_images:
            # Get number of frames in this image
            nfr = getattr(ims, "n_frames", 1)

            for idx in range(nfr):
                ims.seek(idx)

                frame = _convert_frame(ims)

                # Append the frame to the animation encoder
                enc.add(
                    frame.getim(),
                    round(timestamp),
                    lossless,
                    quality,
                    alpha_quality,
                    method,
                )

                # Update timestamp and frame index
                if isinstance(duration, (list, tuple)):
                    timestamp += duration[frame_idx]
                else:
                    timestamp += duration
                frame_idx += 1

    finally:
        im.seek(cur_idx)

    # Force encoder to flush frames
    enc.add(None, round(timestamp), lossless, quality, alpha_quality, 0)

    # Get the final output from the encoder
    data = enc.assemble(icc_profile, exif, xmp)
    if data is None:
        msg = "cannot write file as WebP (encoder returned None)"
        raise OSError(msg)

    fp.write(data)


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    lossless = im.encoderinfo.get("lossless", False)
    quality = im.encoderinfo.get("quality", 80)
    alpha_quality = im.encoderinfo.get("alpha_quality", 100)
    icc_profile = im.encoderinfo.get("icc_profile") or ""
    exif = im.encoderinfo.get("exif", b"")
    if isinstance(exif, Image.Exif):
        exif = exif.tobytes()
    if exif.startswith(b"Exif\x00\x00"):
        exif = exif[6:]
    xmp = im.encoderinfo.get("xmp", "")
    method = im.encoderinfo.get("method", 4)
    exact = 1 if im.encoderinfo.get("exact") else 0

    im = _convert_frame(im)

    data = _webp.WebPEncode(
        im.getim(),
        lossless,
        float(quality),
        float(alpha_quality),
        icc_profile,
        method,
        exact,
        exif,
        xmp,
    )
    if data is None:
        msg = "cannot write file as WebP (encoder returned None)"
        raise OSError(msg)

    fp.write(data)


Image.register_open(WebPImageFile.format, WebPImageFile, _accept)
if SUPPORTED:
    Image.register_save(WebPImageFile.format, _save)
    Image.register_save_all(WebPImageFile.format, _save_all)
    Image.register_extension(WebPImageFile.format, ".webp")
    Image.register_mime(WebPImageFile.format, "image/webp")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# standard channel operations
#
# History:
# 1996-03-24 fl   Created
# 1996-08-13 fl   Added logical operations (for "1" images)
# 2000-10-12 fl   Added offset method (from Image.py)
#
# Copyright (c) 1997-2000 by Secret Labs AB
# Copyright (c) 1996-2000 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

from __future__ import annotations

from . import Image


def constant(image: Image.Image, value: int) -> Image.Image:
    """Fill a channel with a given gray level.

    :rtype: :py:class:`~PIL.Image.Image`
    """

    return Image.new("L", image.size, value)


def duplicate(image: Image.Image) -> Image.Image:
    """Copy a channel. Alias for :py:meth:`PIL.Image.Image.copy`.

    :rtype: :py:class:`~PIL.Image.Image`
    """

    return image.copy()


def invert(image: Image.Image) -> Image.Image:
    """
    Invert an image (channel). ::

        out = MAX - image

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image.load()
    return image._new(image.im.chop_invert())


def lighter(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Compares the two images, pixel by pixel, and returns a new image containing
    the lighter values. ::

        out = max(image1, image2)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_lighter(image2.im))


def darker(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Compares the two images, pixel by pixel, and returns a new image containing
    the darker values. ::

        out = min(image1, image2)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_darker(image2.im))


def difference(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Returns the absolute value of the pixel-by-pixel difference between the two
    images. ::

        out = abs(image1 - image2)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_difference(image2.im))


def multiply(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Superimposes two images on top of each other.

    If you multiply an image with a solid black image, the result is black. If
    you multiply with a solid white image, the image is unaffected. ::

        out = image1 * image2 / MAX

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_multiply(image2.im))


def screen(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Superimposes two inverted images on top of each other. ::

        out = MAX - ((MAX - image1) * (MAX - image2) / MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_screen(image2.im))


def soft_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Superimposes two images on top of each other using the Soft Light algorithm

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_soft_light(image2.im))


def hard_light(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Superimposes two images on top of each other using the Hard Light algorithm

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_hard_light(image2.im))


def overlay(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Superimposes two images on top of each other using the Overlay algorithm

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_overlay(image2.im))


def add(
    image1: Image.Image, image2: Image.Image, scale: float = 1.0, offset: float = 0
) -> Image.Image:
    """
    Adds two images, dividing the result by scale and adding the
    offset. If omitted, scale defaults to 1.0, and offset to 0.0. ::

        out = ((image1 + image2) / scale + offset)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_add(image2.im, scale, offset))


def subtract(
    image1: Image.Image, image2: Image.Image, scale: float = 1.0, offset: float = 0
) -> Image.Image:
    """
    Subtracts two images, dividing the result by scale and adding the offset.
    If omitted, scale defaults to 1.0, and offset to 0.0. ::

        out = ((image1 - image2) / scale + offset)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_subtract(image2.im, scale, offset))


def add_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Add two images, without clipping the result. ::

        out = ((image1 + image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_add_modulo(image2.im))


def subtract_modulo(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Subtract two images, without clipping the result. ::

        out = ((image1 - image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_subtract_modulo(image2.im))


def logical_and(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Logical AND between two images.

    Both of the images must have mode "1". If you would like to perform a
    logical AND on an image with a mode other than "1", try
    :py:meth:`~PIL.ImageChops.multiply` instead, using a black-and-white mask
    as the second image. ::

        out = ((image1 and image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_and(image2.im))


def logical_or(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Logical OR between two images.

    Both of the images must have mode "1". ::

        out = ((image1 or image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_or(image2.im))


def logical_xor(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Logical XOR between two images.

    Both of the images must have mode "1". ::

        out = ((bool(image1) != bool(image2)) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """

    image1.load()
    image2.load()
    return image1._new(image1.im.chop_xor(image2.im))


def blend(image1: Image.Image, image2: Image.Image, alpha: float) -> Image.Image:
    """Blend images using constant transparency weight. Alias for
    :py:func:`PIL.Image.blend`.

    :rtype: :py:class:`~PIL.Image.Image`
    """

    return Image.blend(image1, image2, alpha)


def composite(
    image1: Image.Image, image2: Image.Image, mask: Image.Image
) -> Image.Image:
    """Create composite using transparency mask. Alias for
    :py:func:`PIL.Image.composite`.

    :rtype: :py:class:`~PIL.Image.Image`
    """

    return Image.composite(image1, image2, mask)


def offset(image: Image.Image, xoffset: int, yoffset: int | None = None) -> Image.Image:
    """Returns a copy of the image where data has been offset by the given
    distances. Data wraps around the edges. If ``yoffset`` is omitted, it
    is assumed to be equal to ``xoffset``.

    :param image: Input image.
    :param xoffset: The horizontal distance.
    :param yoffset: The vertical distance.  If omitted, both
        distances are set to the same value.
    :rtype: :py:class:`~PIL.Image.Image`
    """

    if yoffset is None:
        yoffset = xoffset
    image.load()
    return image._new(image.im.offset(xoffset, yoffset))
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# BMP file handler
#
# Windows (and OS/2) native bitmap storage format.
#
# history:
# 1995-09-01 fl   Created
# 1996-04-30 fl   Added save
# 1997-08-27 fl   Fixed save of 1-bit images
# 1998-03-06 fl   Load P images as L where possible
# 1998-07-03 fl   Load P images as 1 where possible
# 1998-12-29 fl   Handle small palettes
# 2002-12-30 fl   Fixed load of 1-bit palette images
# 2003-04-21 fl   Fixed load of 1-bit monochrome images
# 2003-04-23 fl   Added limited support for BI_BITFIELDS compression
#
# Copyright (c) 1997-2003 by Secret Labs AB
# Copyright (c) 1995-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import os
from typing import IO, Any

from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32

#
# --------------------------------------------------------------------
# Read BMP file

BIT2MODE = {
    # bits => mode, rawmode
    1: ("P", "P;1"),
    4: ("P", "P;4"),
    8: ("P", "P"),
    16: ("RGB", "BGR;15"),
    24: ("RGB", "BGR"),
    32: ("RGB", "BGRX"),
}

USE_RAW_ALPHA = False


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"BM")


def _dib_accept(prefix: bytes) -> bool:
    return i32(prefix) in [12, 40, 52, 56, 64, 108, 124]


# =============================================================================
# Image plugin for the Windows BMP format.
# =============================================================================
class BmpImageFile(ImageFile.ImageFile):
    """Image plugin for the Windows Bitmap format (BMP)"""

    # ------------------------------------------------------------- Description
    format_description = "Windows Bitmap"
    format = "BMP"

    # -------------------------------------------------- BMP Compression values
    COMPRESSIONS = {"RAW": 0, "RLE8": 1, "RLE4": 2, "BITFIELDS": 3, "JPEG": 4, "PNG": 5}
    for k, v in COMPRESSIONS.items():
        vars()[k] = v

    def _bitmap(self, header: int = 0, offset: int = 0) -> None:
        """Read relevant info about the BMP"""
        assert self.fp is not None
        read, seek = self.fp.read, self.fp.seek
        if header:
            seek(header)
        # read bmp header size @offset 14 (this is part of the header size)
        file_info: dict[str, bool | int | tuple[int, ...]] = {
            "header_size": i32(read(4)),
            "direction": -1,
        }

        # -------------------- If requested, read header at a specific position
        # read the rest of the bmp header, without its size
        assert isinstance(file_info["header_size"], int)
        header_data = ImageFile._safe_read(self.fp, file_info["header_size"] - 4)

        # ------------------------------- Windows Bitmap v2, IBM OS/2 Bitmap v1
        # ----- This format has different offsets because of width/height types
        # 12: BITMAPCOREHEADER/OS21XBITMAPHEADER
        if file_info["header_size"] == 12:
            file_info["width"] = i16(header_data, 0)
            file_info["height"] = i16(header_data, 2)
            file_info["planes"] = i16(header_data, 4)
            file_info["bits"] = i16(header_data, 6)
            file_info["compression"] = self.COMPRESSIONS["RAW"]
            file_info["palette_padding"] = 3

        # --------------------------------------------- Windows Bitmap v3 to v5
        #  40: BITMAPINFOHEADER
        #  52: BITMAPV2HEADER
        #  56: BITMAPV3HEADER
        #  64: BITMAPCOREHEADER2/OS22XBITMAPHEADER
        # 108: BITMAPV4HEADER
        # 124: BITMAPV5HEADER
        elif file_info["header_size"] in (40, 52, 56, 64, 108, 124):
            file_info["y_flip"] = header_data[7] == 0xFF
            file_info["direction"] = 1 if file_info["y_flip"] else -1
            file_info["width"] = i32(header_data, 0)
            file_info["height"] = (
                i32(header_data, 4)
                if not file_info["y_flip"]
                else 2**32 - i32(header_data, 4)
            )
            file_info["planes"] = i16(header_data, 8)
            file_info["bits"] = i16(header_data, 10)
            file_info["compression"] = i32(header_data, 12)
            # byte size of pixel data
            file_info["data_size"] = i32(header_data, 16)
            file_info["pixels_per_meter"] = (
                i32(header_data, 20),
                i32(header_data, 24),
            )
            file_info["colors"] = i32(header_data, 28)
            file_info["palette_padding"] = 4
            assert isinstance(file_info["pixels_per_meter"], tuple)
            self.info["dpi"] = tuple(x / 39.3701 for x in file_info["pixels_per_meter"])
            if file_info["compression"] == self.COMPRESSIONS["BITFIELDS"]:
                masks = ["r_mask", "g_mask", "b_mask"]
                if len(header_data) >= 48:
                    if len(header_data) >= 52:
                        masks.append("a_mask")
                    else:
                        file_info["a_mask"] = 0x0
                    for idx, mask in enumerate(masks):
                        file_info[mask] = i32(header_data, 36 + idx * 4)
                else:
                    # 40 byte headers only have the three components in the
                    # bitfields masks, ref:
                    # https://msdn.microsoft.com/en-us/library/windows/desktop/dd183376(v=vs.85).aspx
                    # See also
                    # https://github.com/python-pillow/Pillow/issues/1293
                    # There is a 4th component in the RGBQuad, in the alpha
                    # location, but it is listed as a reserved component,
                    # and it is not generally an alpha channel
                    file_info["a_mask"] = 0x0
                    for mask in masks:
                        file_info[mask] = i32(read(4))
                assert isinstance(file_info["r_mask"], int)
                assert isinstance(file_info["g_mask"], int)
                assert isinstance(file_info["b_mask"], int)
                assert isinstance(file_info["a_mask"], int)
                file_info["rgb_mask"] = (
                    file_info["r_mask"],
                    file_info["g_mask"],
                    file_info["b_mask"],
                )
                file_info["rgba_mask"] = (
                    file_info["r_mask"],
                    file_info["g_mask"],
                    file_info["b_mask"],
                    file_info["a_mask"],
                )
        else:
            msg = f"Unsupported BMP header type ({file_info['header_size']})"
            raise OSError(msg)

        # ------------------ Special case : header is reported 40, which
        # ---------------------- is shorter than real size for bpp >= 16
        assert isinstance(file_info["width"], int)
        assert isinstance(file_info["height"], int)
        self._size = file_info["width"], file_info["height"]

        # ------- If color count was not found in the header, compute from bits
        assert isinstance(file_info["bits"], int)
        if not file_info.get("colors", 0):
            file_info["colors"] = 1 << file_info["bits"]
        assert isinstance(file_info["palette_padding"], int)
        assert isinstance(file_info["colors"], int)
        if offset == 14 + file_info["header_size"] and file_info["bits"] <= 8:
            offset += file_info["palette_padding"] * file_info["colors"]

        # ---------------------- Check bit depth for unusual unsupported values
        self._mode, raw_mode = BIT2MODE.get(file_info["bits"], ("", ""))
        if not self.mode:
            msg = f"Unsupported BMP pixel depth ({file_info['bits']})"
            raise OSError(msg)

        # ---------------- Process BMP with Bitfields compression (not palette)
        decoder_name = "raw"
        if file_info["compression"] == self.COMPRESSIONS["BITFIELDS"]:
            SUPPORTED: dict[int, list[tuple[int, ...]]] = {
                32: [
                    (0xFF0000, 0xFF00, 0xFF, 0x0),
                    (0xFF000000, 0xFF0000, 0xFF00, 0x0),
                    (0xFF000000, 0xFF00, 0xFF, 0x0),
                    (0xFF000000, 0xFF0000, 0xFF00, 0xFF),
                    (0xFF, 0xFF00, 0xFF0000, 0xFF000000),
                    (0xFF0000, 0xFF00, 0xFF, 0xFF000000),
                    (0xFF000000, 0xFF00, 0xFF, 0xFF0000),
                    (0x0, 0x0, 0x0, 0x0),
                ],
                24: [(0xFF0000, 0xFF00, 0xFF)],
                16: [(0xF800, 0x7E0, 0x1F), (0x7C00, 0x3E0, 0x1F)],
            }
            MASK_MODES = {
                (32, (0xFF0000, 0xFF00, 0xFF, 0x0)): "BGRX",
                (32, (0xFF000000, 0xFF0000, 0xFF00, 0x0)): "XBGR",
                (32, (0xFF000000, 0xFF00, 0xFF, 0x0)): "BGXR",
                (32, (0xFF000000, 0xFF0000, 0xFF00, 0xFF)): "ABGR",
                (32, (0xFF, 0xFF00, 0xFF0000, 0xFF000000)): "RGBA",
                (32, (0xFF0000, 0xFF00, 0xFF, 0xFF000000)): "BGRA",
                (32, (0xFF000000, 0xFF00, 0xFF, 0xFF0000)): "BGAR",
                (32, (0x0, 0x0, 0x0, 0x0)): "BGRA",
                (24, (0xFF0000, 0xFF00, 0xFF)): "BGR",
                (16, (0xF800, 0x7E0, 0x1F)): "BGR;16",
                (16, (0x7C00, 0x3E0, 0x1F)): "BGR;15",
            }
            if file_info["bits"] in SUPPORTED:
                if (
                    file_info["bits"] == 32
                    and file_info["rgba_mask"] in SUPPORTED[file_info["bits"]]
                ):
                    assert isinstance(file_info["rgba_mask"], tuple)
                    raw_mode = MASK_MODES[(file_info["bits"], file_info["rgba_mask"])]
                    self._mode = "RGBA" if "A" in raw_mode else self.mode
                elif (
                    file_info["bits"] in (24, 16)
                    and file_info["rgb_mask"] in SUPPORTED[file_info["bits"]]
                ):
                    assert isinstance(file_info["rgb_mask"], tuple)
                    raw_mode = MASK_MODES[(file_info["bits"], file_info["rgb_mask"])]
                else:
                    msg = "Unsupported BMP bitfields layout"
                    raise OSError(msg)
            else:
                msg = "Unsupported BMP bitfields layout"
                raise OSError(msg)
        elif file_info["compression"] == self.COMPRESSIONS["RAW"]:
            if file_info["bits"] == 32 and (
                header == 22 or USE_RAW_ALPHA  # 32-bit .cur offset
            ):
                raw_mode, self._mode = "BGRA", "RGBA"
        elif file_info["compression"] in (
            self.COMPRESSIONS["RLE8"],
            self.COMPRESSIONS["RLE4"],
        ):
            decoder_name = "bmp_rle"
        else:
            msg = f"Unsupported BMP compression ({file_info['compression']})"
            raise OSError(msg)

        # --------------- Once the header is processed, process the palette/LUT
        if self.mode == "P":  # Paletted for 1, 4 and 8 bit images
            # ---------------------------------------------------- 1-bit images
            if not (0 < file_info["colors"] <= 65536):
                msg = f"Unsupported BMP Palette size ({file_info['colors']})"
                raise OSError(msg)
            else:
                padding = file_info["palette_padding"]
                palette = read(padding * file_info["colors"])
                grayscale = True
                indices = (
                    (0, 255)
                    if file_info["colors"] == 2
                    else list(range(file_info["colors"]))
                )

                # ----------------- Check if grayscale and ignore palette if so
                for ind, val in enumerate(indices):
                    rgb = palette[ind * padding : ind * padding + 3]
                    if rgb != o8(val) * 3:
                        grayscale = False

                # ------- If all colors are gray, white or black, ditch palette
                if grayscale:
                    self._mode = "1" if file_info["colors"] == 2 else "L"
                    raw_mode = self.mode
                else:
                    self._mode = "P"
                    self.palette = ImagePalette.raw(
                        "BGRX" if padding == 4 else "BGR", palette
                    )

        # ---------------------------- Finally set the tile data for the plugin
        self.info["compression"] = file_info["compression"]
        args: list[Any] = [raw_mode]
        if decoder_name == "bmp_rle":
            args.append(file_info["compression"] == self.COMPRESSIONS["RLE4"])
        else:
            assert isinstance(file_info["width"], int)
            args.append(((file_info["width"] * file_info["bits"] + 31) >> 3) & (~3))
        args.append(file_info["direction"])
        self.tile = [
            ImageFile._Tile(
                decoder_name,
                (0, 0, file_info["width"], file_info["height"]),
                offset or self.fp.tell(),
                tuple(args),
            )
        ]

    def _open(self) -> None:
        """Open file, check magic number and read header"""
        # read 14 bytes: magic number, filesize, reserved, header final offset
        assert self.fp is not None
        head_data = self.fp.read(14)
        # choke if the file does not have the required magic bytes
        if not _accept(head_data):
            msg = "Not a BMP file"
            raise SyntaxError(msg)
        # read the start position of the BMP image data (u32)
        offset = i32(head_data, 10)
        # load bitmap information (offset=raster info)
        self._bitmap(offset=offset)


class BmpRleDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        assert self.fd is not None
        rle4 = self.args[1]
        data = bytearray()
        x = 0
        dest_length = self.state.xsize * self.state.ysize
        while len(data) < dest_length:
            pixels = self.fd.read(1)
            byte = self.fd.read(1)
            if not pixels or not byte:
                break
            num_pixels = pixels[0]
            if num_pixels:
                # encoded mode
                if x + num_pixels > self.state.xsize:
                    # Too much data for row
                    num_pixels = max(0, self.state.xsize - x)
                if rle4:
                    first_pixel = o8(byte[0] >> 4)
                    second_pixel = o8(byte[0] & 0x0F)
                    for index in range(num_pixels):
                        if index % 2 == 0:
                            data += first_pixel
                        else:
                            data += second_pixel
                else:
                    data += byte * num_pixels
                x += num_pixels
            else:
                if byte[0] == 0:
                    # end of line
                    while len(data) % self.state.xsize != 0:
                        data += b"\x00"
                    x = 0
                elif byte[0] == 1:
                    # end of bitmap
                    break
                elif byte[0] == 2:
                    # delta
                    bytes_read = self.fd.read(2)
                    if len(bytes_read) < 2:
                        break
                    right, up = bytes_read
                    data += b"\x00" * (right + up * self.state.xsize)
                    x = len(data) % self.state.xsize
                else:
                    # absolute mode
                    if rle4:
                        # 2 pixels per byte
                        byte_count = byte[0] // 2
                        bytes_read = self.fd.read(byte_count)
                        for byte_read in bytes_read:
                            data += o8(byte_read >> 4)
                            data += o8(byte_read & 0x0F)
                    else:
                        byte_count = byte[0]
                        bytes_read = self.fd.read(byte_count)
                        data += bytes_read
                    if len(bytes_read) < byte_count:
                        break
                    x += byte[0]

                    # align to 16-bit word boundary
                    if self.fd.tell() % 2 != 0:
                        self.fd.seek(1, os.SEEK_CUR)
        rawmode = "L" if self.mode == "L" else "P"
        self.set_as_raw(bytes(data), rawmode, (0, self.args[-1]))
        return -1, 0


# =============================================================================
# Image plugin for the DIB format (BMP alias)
# =============================================================================
class DibImageFile(BmpImageFile):
    format = "DIB"
    format_description = "Windows Bitmap"

    def _open(self) -> None:
        self._bitmap()


#
# --------------------------------------------------------------------
# Write BMP file


SAVE = {
    "1": ("1", 1, 2),
    "L": ("L", 8, 256),
    "P": ("P", 8, 256),
    "RGB": ("BGR", 24, 0),
    "RGBA": ("BGRA", 32, 0),
}


def _dib_save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    _save(im, fp, filename, False)


def _save(
    im: Image.Image, fp: IO[bytes], filename: str | bytes, bitmap_header: bool = True
) -> None:
    try:
        rawmode, bits, colors = SAVE[im.mode]
    except KeyError as e:
        msg = f"cannot write mode {im.mode} as BMP"
        raise OSError(msg) from e

    info = im.encoderinfo

    dpi = info.get("dpi", (96, 96))

    # 1 meter == 39.3701 inches
    ppm = tuple(int(x * 39.3701 + 0.5) for x in dpi)

    stride = ((im.size[0] * bits + 7) // 8 + 3) & (~3)
    header = 40  # or 64 for OS/2 version 2
    image = stride * im.size[1]

    if im.mode == "1":
        palette = b"".join(o8(i) * 3 + b"\x00" for i in (0, 255))
    elif im.mode == "L":
        palette = b"".join(o8(i) * 3 + b"\x00" for i in range(256))
    elif im.mode == "P":
        palette = im.im.getpalette("RGB", "BGRX")
        colors = len(palette) // 4
    else:
        palette = None

    # bitmap header
    if bitmap_header:
        offset = 14 + header + colors * 4
        file_size = offset + image
        if file_size > 2**32 - 1:
            msg = "File size is too large for the BMP format"
            raise ValueError(msg)
        fp.write(
            b"BM"  # file type (magic)
            + o32(file_size)  # file size
            + o32(0)  # reserved
            + o32(offset)  # image data offset
        )

    # bitmap info header
    fp.write(
        o32(header)  # info header size
        + o32(im.size[0])  # width
        + o32(im.size[1])  # height
        + o16(1)  # planes
        + o16(bits)  # depth
        + o32(0)  # compression (0=uncompressed)
        + o32(image)  # size of bitmap
        + o32(ppm[0])  # resolution
        + o32(ppm[1])  # resolution
        + o32(colors)  # colors used
        + o32(colors)  # colors important
    )

    fp.write(b"\0" * (header - 40))  # padding (for OS/2 format)

    if palette:
        fp.write(palette)

    ImageFile._save(
        im, fp, [ImageFile._Tile("raw", (0, 0) + im.size, 0, (rawmode, stride, -1))]
    )


#
# --------------------------------------------------------------------
# Registry


Image.register_open(BmpImageFile.format, BmpImageFile, _accept)
Image.register_save(BmpImageFile.format, _save)

Image.register_extension(BmpImageFile.format, ".bmp")

Image.register_mime(BmpImageFile.format, "image/bmp")

Image.register_decoder("bmp_rle", BmpRleDecoder)

Image.register_open(DibImageFile.format, DibImageFile, _dib_accept)
Image.register_save(DibImageFile.format, _dib_save)

Image.register_extension(DibImageFile.format, ".dib")

Image.register_mime(DibImageFile.format, "image/bmp")
<|endoftext|>#
# Python Imaging Library
# $Id$
#
# stuff to read GIMP palette files
#
# History:
# 1997-08-23 fl     Created
# 2004-09-07 fl     Support GIMP 2.0 palette files.
#
# Copyright (c) Secret Labs AB 1997-2004.  All rights reserved.
# Copyright (c) Fredrik Lundh 1997-2004.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import re
from io import BytesIO

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import IO


class GimpPaletteFile:
    """File handler for GIMP's palette format."""

    rawmode = "RGB"

    def _read(self, fp: IO[bytes], limit: bool = True) -> None:
        if not fp.readline().startswith(b"GIMP Palette"):
            msg = "not a GIMP palette file"
            raise SyntaxError(msg)

        palette: list[int] = []
        i = 0
        while True:
            if limit and i == 256 + 3:
                break

            i += 1
            s = fp.readline()
            if not s:
                break

            # skip fields and comment lines
            if re.match(rb"\w+:|#", s):
                continue
            if limit and len(s) > 100:
                msg = "bad palette file"
                raise SyntaxError(msg)

            v = s.split(maxsplit=3)
            if len(v) < 3:
                msg = "bad palette entry"
                raise ValueError(msg)

            palette += (int(v[i]) for i in range(3))
            if limit and len(palette) == 768:
                break

        self.palette = bytes(palette)

    def __init__(self, fp: IO[bytes]) -> None:
        self._read(fp)

    @classmethod
    def frombytes(cls, data: bytes) -> GimpPaletteFile:
        self = cls.__new__(cls)
        self._read(BytesIO(data), False)
        return self

    def getpalette(self) -> tuple[bytes, str]:
        return self.palette, self.rawmode
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# IM Tools support for PIL
#
# history:
# 1996-05-27 fl   Created (read 8-bit images only)
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.2)
#
# Copyright (c) Secret Labs AB 1997-2001.
# Copyright (c) Fredrik Lundh 1996-2001.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import re

from . import Image, ImageFile

#
# --------------------------------------------------------------------

field = re.compile(rb"([a-z]*) ([^ \r\n]*)")


##
# Image plugin for IM Tools images.


class ImtImageFile(ImageFile.ImageFile):
    format = "IMT"
    format_description = "IM Tools"

    def _open(self) -> None:
        # Quick rejection: if there's not a LF among the first
        # 100 bytes, this is (probably) not a text header.

        assert self.fp is not None

        buffer = self.fp.read(100)
        if b"\n" not in buffer:
            msg = "not an IM file"
            raise SyntaxError(msg)

        xsize = ysize = 0

        while True:
            if buffer:
                s = buffer[:1]
                buffer = buffer[1:]
            else:
                s = self.fp.read(1)
            if not s:
                break

            if s == b"\x0c":
                # image data begins
                self.tile = [
                    ImageFile._Tile(
                        "raw",
                        (0, 0) + self.size,
                        self.fp.tell() - len(buffer),
                        self.mode,
                    )
                ]

                break

            else:
                # read key/value pair
                if b"\n" not in buffer:
                    buffer += self.fp.read(100)
                lines = buffer.split(b"\n")
                s += lines.pop(0)
                buffer = b"\n".join(lines)
                if len(s) == 1 or len(s) > 100:
                    break
                if s[0] == ord(b"*"):
                    continue  # comment

                m = field.match(s)
                if not m:
                    break
                k, v = m.group(1, 2)
                if k == b"width":
                    xsize = int(v)
                    self._size = xsize, ysize
                elif k == b"height":
                    ysize = int(v)
                    self._size = xsize, ysize
                elif k == b"pixel" and v == b"n8":
                    self._mode = "L"


#
# --------------------------------------------------------------------

Image.register_open(ImtImageFile.format, ImtImageFile)

#
# no extension registered (".im" is simply too common)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# sequence support classes
#
# history:
# 1997-02-20 fl     Created
#
# Copyright (c) 1997 by Secret Labs AB.
# Copyright (c) 1997 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

##
from __future__ import annotations

from . import Image

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable


class Iterator:
    """
    This class implements an iterator object that can be used to loop
    over an image sequence.

    You can use the ``[]`` operator to access elements by index. This operator
    will raise an :py:exc:`IndexError` if you try to access a nonexistent
    frame.

    :param im: An image object.
    """

    def __init__(self, im: Image.Image) -> None:
        if not hasattr(im, "seek"):
            msg = "im must have seek method"
            raise AttributeError(msg)
        self.im = im
        self.position = getattr(self.im, "_min_frame", 0)

    def __getitem__(self, ix: int) -> Image.Image:
        try:
            self.im.seek(ix)
            return self.im
        except EOFError as e:
            msg = "end of sequence"
            raise IndexError(msg) from e

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Image.Image:
        try:
            self.im.seek(self.position)
            self.position += 1
            return self.im
        except EOFError as e:
            msg = "end of sequence"
            raise StopIteration(msg) from e


def all_frames(
    im: Image.Image | list[Image.Image],
    func: Callable[[Image.Image], Image.Image] | None = None,
) -> list[Image.Image]:
    """
    Applies a given function to all frames in an image or a list of images.
    The frames are returned as a list of separate images.

    :param im: An image, or a list of images.
    :param func: The function to apply to all of the image frames.
    :returns: A list of images.
    """
    if not isinstance(im, list):
        im = [im]

    ims = []
    for imSequence in im:
        current = imSequence.tell()

        ims += [im_frame.copy() for im_frame in Iterator(imSequence)]

        imSequence.seek(current)
    return [func(im) for im in ims] if func else ims
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# EPS file handling
#
# History:
# 1995-09-01 fl   Created (0.1)
# 1996-05-18 fl   Don't choke on "atend" fields, Ghostscript interface (0.2)
# 1996-08-22 fl   Don't choke on floating point BoundingBox values
# 1996-08-23 fl   Handle files from Macintosh (0.3)
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.4)
# 2003-09-07 fl   Check gs.close status (from Federico Di Gregorio) (0.5)
# 2014-05-07 e    Handling of EPS with binary preview and fixed resolution
#                 resizing
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
from typing import IO

from . import Image, ImageFile
from ._binary import i32le as i32

# --------------------------------------------------------------------


split = re.compile(r"^%%([^:]*):[ \t]*(.*)[ \t]*$")
field = re.compile(r"^%[%!\w]([^:]*)[ \t]*$")

gs_binary: str | bool | None = None
gs_windows_binary = None


def has_ghostscript() -> bool:
    global gs_binary, gs_windows_binary
    if gs_binary is None:
        if sys.platform.startswith("win"):
            if gs_windows_binary is None:
                import shutil

                for binary in ("gswin32c", "gswin64c", "gs"):
                    if shutil.which(binary) is not None:
                        gs_windows_binary = binary
                        break
                else:
                    gs_windows_binary = False
            gs_binary = gs_windows_binary
        else:
            try:
                subprocess.check_call(["gs", "--version"], stdout=subprocess.DEVNULL)
                gs_binary = "gs"
            except OSError:
                gs_binary = False
    return gs_binary is not False


def Ghostscript(
    tile: list[ImageFile._Tile],
    size: tuple[int, int],
    fp: IO[bytes],
    scale: int = 1,
    transparency: bool = False,
) -> Image.core.ImagingCore:
    """Render an image using Ghostscript"""
    global gs_binary
    if not has_ghostscript():
        msg = "Unable to locate Ghostscript on paths"
        raise OSError(msg)
    assert isinstance(gs_binary, str)

    # Unpack decoder tile
    args = tile[0].args
    assert isinstance(args, tuple)
    length, bbox = args

    # Hack to support hi-res rendering
    scale = int(scale) or 1
    width = size[0] * scale
    height = size[1] * scale
    # resolution is dependent on bbox and size
    res_x = 72.0 * width / (bbox[2] - bbox[0])
    res_y = 72.0 * height / (bbox[3] - bbox[1])

    out_fd, outfile = tempfile.mkstemp()
    os.close(out_fd)

    infile_temp = None
    if hasattr(fp, "name") and os.path.exists(fp.name):
        infile = fp.name
    else:
        in_fd, infile_temp = tempfile.mkstemp()
        os.close(in_fd)
        infile = infile_temp

        # Ignore length and offset!
        # Ghostscript can read it
        # Copy whole file to read in Ghostscript
        with open(infile_temp, "wb") as f:
            # fetch length of fp
            fp.seek(0, io.SEEK_END)
            fsize = fp.tell()
            # ensure start position
            # go back
            fp.seek(0)
            lengthfile = fsize
            while lengthfile > 0:
                s = fp.read(min(lengthfile, 100 * 1024))
                if not s:
                    break
                lengthfile -= len(s)
                f.write(s)

    if transparency:
        # "RGBA"
        device = "pngalpha"
    else:
        # "pnmraw" automatically chooses between
        # PBM ("1"), PGM ("L"), and PPM ("RGB").
        device = "pnmraw"

    # Build Ghostscript command
    command = [
        gs_binary,
        "-q",  # quiet mode
        f"-g{width:d}x{height:d}",  # set output geometry (pixels)
        f"-r{res_x:f}x{res_y:f}",  # set input DPI (dots per inch)
        "-dBATCH",  # exit after processing
        "-dNOPAUSE",  # don't pause between pages
        "-dSAFER",  # safe mode
        f"-sDEVICE={device}",
        f"-sOutputFile={outfile}",  # output file
        # adjust for image origin
        "-c",
        f"{-bbox[0]} {-bbox[1]} translate",
        "-f",
        infile,  # input file
        # showpage (see https://bugs.ghostscript.com/show_bug.cgi?id=698272)
        "-c",
        "showpage",
    ]

    # push data through Ghostscript
    try:
        startupinfo = None
        if sys.platform.startswith("win"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.check_call(command, startupinfo=startupinfo)
        with Image.open(outfile) as out_im:
            out_im.load()
            return out_im.im.copy()
    finally:
        try:
            os.unlink(outfile)
            if infile_temp:
                os.unlink(infile_temp)
        except OSError:
            pass


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"%!PS") or (
        len(prefix) >= 4 and i32(prefix) == 0xC6D3D0C5
    )


##
# Image plugin for Encapsulated PostScript. This plugin supports only
# a few variants of this format.


class EpsImageFile(ImageFile.ImageFile):
    """EPS File Parser for the Python Imaging Library"""

    format = "EPS"
    format_description = "Encapsulated Postscript"

    mode_map = {1: "L", 2: "LAB", 3: "RGB", 4: "CMYK"}

    def _open(self) -> None:
        assert self.fp is not None
        length, offset = self._find_offset(self.fp)

        # go to offset - start of "%!PS"
        self.fp.seek(offset)

        self._mode = "RGB"

        # When reading header comments, the first comment is used.
        # When reading trailer comments, the last comment is used.
        bounding_box: list[int] | None = None
        imagedata_size: tuple[int, int] | None = None

        byte_arr = bytearray(255)
        bytes_mv = memoryview(byte_arr)
        bytes_read = 0
        reading_header_comments = True
        reading_trailer_comments = False
        trailer_reached = False

        def check_required_header_comments() -> None:
            """
            The EPS specification requires that some headers exist.
            This should be checked when the header comments formally end,
            when image data starts, or when the file ends, whichever comes first.
            """
            if "PS-Adobe" not in self.info:
                msg = 'EPS header missing "%!PS-Adobe" comment'
                raise SyntaxError(msg)
            if "BoundingBox" not in self.info:
                msg = 'EPS header missing "%%BoundingBox" comment'
                raise SyntaxError(msg)

        def read_comment(s: str) -> bool:
            nonlocal bounding_box, reading_trailer_comments
            try:
                m = split.match(s)
            except re.error as e:
                msg = "not an EPS file"
                raise SyntaxError(msg) from e

            if not m:
                return False

            k, v = m.group(1, 2)
            self.info[k] = v
            if k == "BoundingBox":
                if v == "(atend)":
                    reading_trailer_comments = True
                elif not bounding_box or (trailer_reached and reading_trailer_comments):
                    try:
                        # Note: The DSC spec says that BoundingBox
                        # fields should be integers, but some drivers
                        # put floating point values there anyway.
                        bounding_box = [int(float(i)) for i in v.split()]
                    except Exception:
                        pass
            return True

        while True:
            byte = self.fp.read(1)
            if byte == b"":
                # if we didn't read a byte we must be at the end of the file
                if bytes_read == 0:
                    if reading_header_comments:
                        check_required_header_comments()
                    break
            elif byte in b"\r\n":
                # if we read a line ending character, ignore it and parse what
                # we have already read. if we haven't read any other characters,
                # continue reading
                if bytes_read == 0:
                    continue
            else:
                # ASCII/hexadecimal lines in an EPS file must not exceed
                # 255 characters, not including line ending characters
                if bytes_read >= 255:
                    # only enforce this for lines starting with a "%",
                    # otherwise assume it's binary data
                    if byte_arr[0] == ord("%"):
                        msg = "not an EPS file"
                        raise SyntaxError(msg)
                    else:
                        if reading_header_comments:
                            check_required_header_comments()
                            reading_header_comments = False
                        # reset bytes_read so we can keep reading
                        # data until the end of the line
                        bytes_read = 0
                byte_arr[bytes_read] = byte[0]
                bytes_read += 1
                continue

            if reading_header_comments:
                # Load EPS header

                # if this line doesn't start with a "%",
                # or does start with "%%EndComments",
                # then we've reached the end of the header/comments
                if byte_arr[0] != ord("%") or bytes_mv[:13] == b"%%EndComments":
                    check_required_header_comments()
                    reading_header_comments = False
                    continue

                s = str(bytes_mv[:bytes_read], "latin-1")
                if not read_comment(s):
                    m = field.match(s)
                    if m:
                        k = m.group(1)
                        if k.startswith("PS-Adobe"):
                            self.info["PS-Adobe"] = k[9:]
                        else:
                            self.info[k] = ""
                    elif s[0] == "%":
                        # handle non-DSC PostScript comments that some
                        # tools mistakenly put in the Comments section
                        pass
                    else:
                        msg = "bad EPS header"
                        raise OSError(msg)
            elif bytes_mv[:11] == b"%ImageData:":
                # Check for an "ImageData" descriptor
                # https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/#50577413_pgfId-1035096

                # If we've already read an "ImageData" descriptor,
                # don't read another one.
                if imagedata_size:
                    bytes_read = 0
                    continue

                # Values:
                # columns
                # rows
                # bit depth (1 or 8)
                # mode (1: L, 2: LAB, 3: RGB, 4: CMYK)
                # number of padding channels
                # block size (number of bytes per row per channel)
                # binary/ascii (1: binary, 2: ascii)
                # data start identifier (the image data follows after a single line
                #   consisting only of this quoted value)
                image_data_values = byte_arr[11:bytes_read].split(None, 7)
                columns, rows, bit_depth, mode_id = (
                    int(value) for value in image_data_values[:4]
                )

                if bit_depth == 1:
                    self._mode = "1"
                elif bit_depth == 8:
                    try:
                        self._mode = self.mode_map[mode_id]
                    except ValueError:
                        break
                else:
                    break

                # Parse the columns and rows after checking the bit depth and mode
                # in case the bit depth and/or mode are invalid.
                imagedata_size = columns, rows
            elif bytes_mv[:5] == b"%%EOF":
                break
            elif trailer_reached and reading_trailer_comments:
                # Load EPS trailer
                s = str(bytes_mv[:bytes_read], "latin-1")
                read_comment(s)
            elif bytes_mv[:9] == b"%%Trailer":
                trailer_reached = True
            elif bytes_mv[:14] == b"%%BeginBinary:":
                bytecount = int(byte_arr[14:bytes_read])
                if bytecount < 0:
                    msg = "BeginBinary bytecount cannot be negative"
                    raise ValueError(msg)
                self.fp.seek(bytecount, os.SEEK_CUR)
            bytes_read = 0

        # A "BoundingBox" is always required,
        # even if an "ImageData" descriptor size exists.
        if not bounding_box:
            msg = "cannot determine EPS bounding box"
            raise OSError(msg)

        # An "ImageData" size takes precedence over the "BoundingBox".
        self._size = imagedata_size or (
            bounding_box[2] - bounding_box[0],
            bounding_box[3] - bounding_box[1],
        )

        self.tile = [
            ImageFile._Tile("eps", (0, 0) + self.size, offset, (length, bounding_box))
        ]

    def _find_offset(self, fp: IO[bytes]) -> tuple[int, int]:
        s = fp.read(4)

        if s == b"%!PS":
            # for HEAD without binary preview
            fp.seek(0, io.SEEK_END)
            length = fp.tell()
            offset = 0
        elif i32(s) == 0xC6D3D0C5:
            # FIX for: Some EPS file not handled correctly / issue #302
            # EPS can contain binary data
            # or start directly with latin coding
            # more info see:
            # https://web.archive.org/web/20160528181353/http://partners.adobe.com/public/developer/en/ps/5002.EPSF_Spec.pdf
            s = fp.read(8)
            offset = i32(s)
            length = i32(s, 4)
        else:
            msg = "not an EPS file"
            raise SyntaxError(msg)

        return length, offset

    def load(
        self, scale: int = 1, transparency: bool = False
    ) -> Image.core.PixelAccess | None:
        # Load EPS via Ghostscript
        if self.tile:
            assert self.fp is not None
            self.im = Ghostscript(self.tile, self.size, self.fp, scale, transparency)
            self._mode = self.im.mode
            self._size = self.im.size
            self.tile = []
        return Image.Image.load(self)

    def load_seek(self, pos: int) -> None:
        # we can't incrementally load, so force ImageFile.parser to
        # use our custom load method by defining this method.
        pass


# --------------------------------------------------------------------


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes, eps: int = 1) -> None:
    """EPS Writer for the Python Imaging Library."""

    # make sure image data is available
    im.load()

    # determine PostScript image mode
    if im.mode == "L":
        operator = (8, 1, b"image")
    elif im.mode == "RGB":
        operator = (8, 3, b"false 3 colorimage")
    elif im.mode == "CMYK":
        operator = (8, 4, b"false 4 colorimage")
    else:
        msg = "image mode is not supported"
        raise ValueError(msg)

    if eps:
        # write EPS header
        fp.write(b"%!PS-Adobe-3.0 EPSF-3.0\n")
        fp.write(b"%%Creator: PIL 0.1 EpsEncode\n")
        # fp.write("%%CreationDate: %s"...)
        fp.write(b"%%%%BoundingBox: 0 0 %d %d\n" % im.size)
        fp.write(b"%%Pages: 1\n")
        fp.write(b"%%EndComments\n")
        fp.write(b"%%Page: 1 1\n")
        fp.write(b"%%ImageData: %d %d " % im.size)
        fp.write(b'%d %d 0 1 1 "%s"\n' % operator)

    # image header
    fp.write(b"gsave\n")
    fp.write(b"10 dict begin\n")
    fp.write(b"/buf %d string def\n" % (im.size[0] * operator[1]))
    fp.write(b"%d %d scale\n" % im.size)
    fp.write(b"%d %d 8\n" % im.size)  # <= bits
    fp.write(b"[%d 0 0 -%d 0 %d]\n" % (im.size[0], im.size[1], im.size[1]))
    fp.write(b"{ currentfile buf readhexstring pop } bind\n")
    fp.write(operator[2] + b"\n")
    if hasattr(fp, "flush"):
        fp.flush()

    ImageFile._save(im, fp, [ImageFile._Tile("eps", (0, 0) + im.size)])

    fp.write(b"\n%%%%EndBinary\n")
    fp.write(b"grestore end\n")
    if hasattr(fp, "flush"):
        fp.flush()


# --------------------------------------------------------------------


Image.register_open(EpsImageFile.format, EpsImageFile, _accept)

Image.register_save(EpsImageFile.format, _save)

Image.register_extensions(EpsImageFile.format, [".ps", ".eps"])

Image.register_mime(EpsImageFile.format, "application/postscript")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# PPM support for PIL
#
# History:
#       96-03-24 fl     Created
#       98-03-06 fl     Write RGBA images (as RGB, that is)
#
# Copyright (c) Secret Labs AB 1997-98.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import math
from typing import IO

from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32

#
# --------------------------------------------------------------------

b_whitespace = b"\x20\x09\x0a\x0b\x0c\x0d"

MODES = {
    # standard
    b"P1": "1",
    b"P2": "L",
    b"P3": "RGB",
    b"P4": "1",
    b"P5": "L",
    b"P6": "RGB",
    # extensions
    b"P0CMYK": "CMYK",
    b"Pf": "F",
    # PIL extensions (for test purposes only)
    b"PyP": "P",
    b"PyRGBA": "RGBA",
    b"PyCMYK": "CMYK",
}


def _accept(prefix: bytes) -> bool:
    return len(prefix) >= 2 and prefix.startswith(b"P") and prefix[1] in b"0123456fy"


##
# Image plugin for PBM, PGM, and PPM images.


class PpmImageFile(ImageFile.ImageFile):
    format = "PPM"
    format_description = "Pbmplus image"

    def _read_magic(self) -> bytes:
        assert self.fp is not None

        magic = b""
        # read until whitespace or longest available magic number
        for _ in range(6):
            c = self.fp.read(1)
            if not c or c in b_whitespace:
                break
            magic += c
        return magic

    def _read_token(self) -> bytes:
        assert self.fp is not None

        token = b""
        while len(token) <= 10:  # read until next whitespace or limit of 10 characters
            c = self.fp.read(1)
            if not c:
                break
            elif c in b_whitespace:  # token ended
                if not token:
                    # skip whitespace at start
                    continue
                break
            elif c == b"#":
                # ignores rest of the line; stops at CR, LF or EOF
                while self.fp.read(1) not in b"\r\n":
                    pass
                continue
            token += c
        if not token:
            # Token was not even 1 byte
            msg = "Reached EOF while reading header"
            raise ValueError(msg)
        elif len(token) > 10:
            msg_too_long = b"Token too long in file header: %s" % token
            raise ValueError(msg_too_long)
        return token

    def _open(self) -> None:
        assert self.fp is not None

        magic_number = self._read_magic()
        try:
            mode = MODES[magic_number]
        except KeyError:
            msg = "not a PPM file"
            raise SyntaxError(msg)
        self._mode = mode

        if magic_number in (b"P1", b"P4"):
            self.custom_mimetype = "image/x-portable-bitmap"
        elif magic_number in (b"P2", b"P5"):
            self.custom_mimetype = "image/x-portable-graymap"
        elif magic_number in (b"P3", b"P6"):
            self.custom_mimetype = "image/x-portable-pixmap"

        self._size = int(self._read_token()), int(self._read_token())

        decoder_name = "raw"
        if magic_number in (b"P1", b"P2", b"P3"):
            decoder_name = "ppm_plain"

        args: str | tuple[str | int, ...]
        if mode == "1":
            args = "1;I"
        elif mode == "F":
            scale = float(self._read_token())
            if scale == 0.0 or not math.isfinite(scale):
                msg = "scale must be finite and non-zero"
                raise ValueError(msg)
            self.info["scale"] = abs(scale)

            rawmode = "F;32F" if scale < 0 else "F;32BF"
            args = (rawmode, 0, -1)
        else:
            maxval = int(self._read_token())
            if not 0 < maxval < 65536:
                msg = "maxval must be greater than 0 and less than 65536"
                raise ValueError(msg)
            if maxval > 255 and mode == "L":
                self._mode = "I"

            rawmode = mode
            if decoder_name != "ppm_plain":
                # If maxval matches a bit depth, use the raw decoder directly
                if maxval == 65535 and mode == "L":
                    rawmode = "I;16B"
                elif maxval != 255:
                    decoder_name = "ppm"

            args = rawmode if decoder_name == "raw" else (rawmode, maxval)
        self.tile = [
            ImageFile._Tile(decoder_name, (0, 0) + self.size, self.fp.tell(), args)
        ]


#
# --------------------------------------------------------------------


class PpmPlainDecoder(ImageFile.PyDecoder):
    _pulls_fd = True
    _comment_spans: bool

    def _read_block(self) -> bytes:
        assert self.fd is not None

        return self.fd.read(ImageFile.SAFEBLOCK)

    def _find_comment_end(self, block: bytes, start: int = 0) -> int:
        a = block.find(b"\n", start)
        b = block.find(b"\r", start)
        return min(a, b) if a * b > 0 else max(a, b)  # lowest nonnegative index (or -1)

    def _ignore_comments(self, block: bytes) -> bytes:
        if self._comment_spans:
            # Finish current comment
            while block:
                comment_end = self._find_comment_end(block)
                if comment_end != -1:
                    # Comment ends in this block
                    # Delete tail of comment
                    block = block[comment_end + 1 :]
                    break
                else:
                    # Comment spans whole block
                    # So read the next block, looking for the end
                    block = self._read_block()

        # Search for any further comments
        self._comment_spans = False
        while True:
            comment_start = block.find(b"#")
            if comment_start == -1:
                # No comment found
                break
            comment_end = self._find_comment_end(block, comment_start)
            if comment_end != -1:
                # Comment ends in this block
                # Delete comment
                block = block[:comment_start] + block[comment_end + 1 :]
            else:
                # Comment continues to next block(s)
                block = block[:comment_start]
                self._comment_spans = True
                break
        return block

    def _decode_bitonal(self) -> bytearray:
        """
        This is a separate method because in the plain PBM format, all data tokens are
        exactly one byte, so the inter-token whitespace is optional.
        """
        data = bytearray()
        total_bytes = self.state.xsize * self.state.ysize

        while len(data) != total_bytes:
            block = self._read_block()  # read next block
            if not block:
                # eof
                break

            block = self._ignore_comments(block)

            tokens = b"".join(block.split())
            for token in tokens:
                if token not in (48, 49):
                    msg = b"Invalid token for this mode: %s" % bytes([token])
                    raise ValueError(msg)
            data = (data + tokens)[:total_bytes]
        invert = bytes.maketrans(b"01", b"\xff\x00")
        return data.translate(invert)

    def _decode_blocks(self, maxval: int) -> bytearray:
        data = bytearray()
        max_len = 10
        out_byte_count = 4 if self.mode == "I" else 1
        out_max = 65535 if self.mode == "I" else 255
        bands = Image.getmodebands(self.mode)
        total_bytes = self.state.xsize * self.state.ysize * bands * out_byte_count

        half_token = b""
        while len(data) != total_bytes:
            block = self._read_block()  # read next block
            if not block:
                if half_token:
                    block = b" "  # flush half_token
                else:
                    # eof
                    break

            block = self._ignore_comments(block)

            if half_token:
                block = half_token + block  # stitch half_token to new block
                half_token = b""

            tokens = block.split()

            if block and not block[-1:].isspace():  # block might split token
                half_token = tokens.pop()  # save half token for later
                if len(half_token) > max_len:  # prevent buildup of half_token
                    msg = (
                        b"Token too long found in data: %s" % half_token[: max_len + 1]
                    )
                    raise ValueError(msg)

            for token in tokens:
                if len(token) > max_len:
                    msg = b"Token too long found in data: %s" % token[: max_len + 1]
                    raise ValueError(msg)
                value = int(token)
                if value < 0:
                    msg_str = f"Channel value is negative: {value}"
                    raise ValueError(msg_str)
                if value > maxval:
                    msg_str = f"Channel value too large for this mode: {value}"
                    raise ValueError(msg_str)
                value = round(value / maxval * out_max)
                data += o32(value) if self.mode == "I" else o8(value)
                if len(data) == total_bytes:  # finished!
                    break
        return data

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        self._comment_spans = False
        if self.mode == "1":
            data = self._decode_bitonal()
            rawmode = "1;8"
        else:
            maxval = self.args[-1]
            data = self._decode_blocks(maxval)
            rawmode = "I;32" if self.mode == "I" else self.mode
        self.set_as_raw(bytes(data), rawmode)
        return -1, 0


class PpmDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        assert self.fd is not None

        data = bytearray()
        maxval = self.args[-1]
        in_byte_count = 1 if maxval < 256 else 2
        out_byte_count = 4 if self.mode == "I" else 1
        out_max = 65535 if self.mode == "I" else 255
        bands = Image.getmodebands(self.mode)
        dest_length = self.state.xsize * self.state.ysize * bands * out_byte_count
        while len(data) < dest_length:
            pixels = self.fd.read(in_byte_count * bands)
            if len(pixels) < in_byte_count * bands:
                # eof
                break
            for b in range(bands):
                value = (
                    pixels[b] if in_byte_count == 1 else i16(pixels, b * in_byte_count)
                )
                value = min(out_max, round(value / maxval * out_max))
                data += o32(value) if self.mode == "I" else o8(value)
        rawmode = "I;32" if self.mode == "I" else self.mode
        self.set_as_raw(bytes(data), rawmode)
        return -1, 0


#
# --------------------------------------------------------------------


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode == "1":
        rawmode, head = "1;I", b"P4"
    elif im.mode == "L":
        rawmode, head = "L", b"P5"
    elif im.mode in ("I", "I;16"):
        rawmode, head = "I;16B", b"P5"
    elif im.mode in ("RGB", "RGBA"):
        rawmode, head = "RGB", b"P6"
    elif im.mode == "F":
        rawmode, head = "F;32F", b"Pf"
    else:
        msg = f"cannot write mode {im.mode} as PPM"
        raise OSError(msg)
    fp.write(head + b"\n%d %d\n" % im.size)
    if head == b"P6":
        fp.write(b"255\n")
    elif head == b"P5":
        if rawmode == "L":
            fp.write(b"255\n")
        else:
            fp.write(b"65535\n")
    elif head == b"Pf":
        fp.write(b"-1.0\n")
    row_order = -1 if im.mode == "F" else 1
    ImageFile._save(
        im, fp, [ImageFile._Tile("raw", (0, 0) + im.size, 0, (rawmode, 0, row_order))]
    )


#
# --------------------------------------------------------------------


Image.register_open(PpmImageFile.format, PpmImageFile, _accept)
Image.register_save(PpmImageFile.format, _save)

Image.register_decoder("ppm", PpmDecoder)
Image.register_decoder("ppm_plain", PpmPlainDecoder)

Image.register_extensions(PpmImageFile.format, [".pbm", ".pgm", ".ppm", ".pnm", ".pfm"])

Image.register_mime(PpmImageFile.format, "image/x-portable-anymap")
<|endoftext|>from __future__ import annotations

import os

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Any, NoReturn, TypeGuard

    from ._typing import StrOrBytesPath


def is_path(f: Any) -> TypeGuard[StrOrBytesPath]:
    return isinstance(f, (bytes, str, os.PathLike))


class DeferredError:
    def __init__(self, ex: BaseException):
        self.ex = ex

    def __getattr__(self, elt: str) -> NoReturn:
        raise self.ex

    @staticmethod
    def new(ex: BaseException) -> Any:
        """
        Creates an object that raises the wrapped exception ``ex`` when used,
        and casts it to :py:obj:`~typing.Any` type.
        """
        return DeferredError(ex)
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# FITS file handling
#
# Copyright (c) 1998-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import gzip
import math

from . import Image, ImageFile


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"SIMPLE")


class FitsImageFile(ImageFile.ImageFile):
    format = "FITS"
    format_description = "FITS"

    def _open(self) -> None:
        assert self.fp is not None

        headers: dict[bytes, bytes] = {}
        header_in_progress = False
        decoder_name = ""
        while True:
            header = self.fp.read(80)
            if not header:
                msg = "Truncated FITS file"
                raise OSError(msg)
            keyword = header[:8].strip()
            if keyword in (b"SIMPLE", b"XTENSION"):
                header_in_progress = True
            elif headers and not header_in_progress:
                # This is now a data unit
                break
            elif keyword == b"END":
                # Seek to the end of the header unit
                self.fp.seek(math.ceil(self.fp.tell() / 2880) * 2880)
                if not decoder_name:
                    decoder_name, offset, args = self._parse_headers(headers)

                header_in_progress = False
                continue

            if decoder_name:
                # Keep going to read past the headers
                continue

            value = header[8:].split(b"/")[0].strip()
            if value.startswith(b"="):
                value = value[1:].strip()
            if not headers and (not _accept(keyword) or value != b"T"):
                msg = "Not a FITS file"
                raise SyntaxError(msg)
            headers[keyword] = value

        if not decoder_name:
            msg = "No image data"
            raise ValueError(msg)

        offset += self.fp.tell() - 80
        self.tile = [ImageFile._Tile(decoder_name, (0, 0) + self.size, offset, args)]

    def _get_size(
        self, headers: dict[bytes, bytes], prefix: bytes
    ) -> tuple[int, int] | None:
        naxis = int(headers[prefix + b"NAXIS"])
        if naxis == 0:
            return None

        if naxis == 1:
            return 1, int(headers[prefix + b"NAXIS1"])
        else:
            return int(headers[prefix + b"NAXIS1"]), int(headers[prefix + b"NAXIS2"])

    def _parse_headers(
        self, headers: dict[bytes, bytes]
    ) -> tuple[str, int, tuple[str | int, ...]]:
        prefix = b""
        decoder_name = "raw"
        offset = 0
        if (
            headers.get(b"XTENSION") == b"'BINTABLE'"
            and headers.get(b"ZIMAGE") == b"T"
            and headers[b"ZCMPTYPE"] == b"'GZIP_1  '"
        ):
            no_prefix_size = self._get_size(headers, prefix) or (0, 0)
            number_of_bits = int(headers[b"BITPIX"])
            offset = no_prefix_size[0] * no_prefix_size[1] * (number_of_bits // 8)

            prefix = b"Z"
            decoder_name = "fits_gzip"

        size = self._get_size(headers, prefix)
        if not size:
            return "", 0, ()

        self._size = size

        number_of_bits = int(headers[prefix + b"BITPIX"])
        if number_of_bits == 8:
            self._mode = "L"
        elif number_of_bits == 16:
            self._mode = "I;16"
        elif number_of_bits == 32:
            self._mode = "I"
        elif number_of_bits in (-32, -64):
            self._mode = "F"

        args: tuple[str | int, ...]
        if decoder_name == "raw":
            args = (self.mode, 0, -1)
        else:
            args = (number_of_bits,)
        return decoder_name, offset, args


class FitsGzipDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        assert self.fd is not None
        with gzip.open(self.fd) as fp:
            value = fp.read(self.state.xsize * self.state.ysize * 4)

            rows = []
            offset = 0
            number_of_bits = min(self.args[0] // 8, 4)
            for y in range(self.state.ysize):
                row = bytearray()
                for x in range(self.state.xsize):
                    row += value[offset + (4 - number_of_bits) : offset + 4]
                    offset += 4
                rows.append(row)
        self.set_as_raw(bytes([pixel for row in rows[::-1] for pixel in row]))
        return -1, 0


# --------------------------------------------------------------------
# Registry

Image.register_open(FitsImageFile.format, FitsImageFile, _accept)
Image.register_decoder("fits_gzip", FitsGzipDecoder)

Image.register_extensions(FitsImageFile.format, [".fit", ".fits"])
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# IPTC/NAA file handling
#
# history:
# 1995-10-01 fl   Created
# 1998-03-09 fl   Cleaned up and added to PIL
# 2002-06-18 fl   Added getiptcinfo helper
#
# Copyright (c) Secret Labs AB 1997-2002.
# Copyright (c) Fredrik Lundh 1995.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from io import BytesIO
from typing import cast

from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32

COMPRESSION = {1: "raw", 5: "jpeg"}


#
# Helpers


def _i(c: bytes) -> int:
    return i32((b"\0\0\0\0" + c)[-4:])


##
# Image plugin for IPTC/NAA datastreams.  To read IPTC/NAA fields
# from TIFF and JPEG files, use the <b>getiptcinfo</b> function.


class IptcImageFile(ImageFile.ImageFile):
    format = "IPTC"
    format_description = "IPTC/NAA"

    def getint(self, key: tuple[int, int]) -> int:
        return _i(self.info[key])

    def field(self) -> tuple[tuple[int, int] | None, int]:
        #
        # get a IPTC field header
        assert self.fp is not None
        s = self.fp.read(5)
        if not s.strip(b"\x00"):
            return None, 0

        tag = s[1], s[2]

        # syntax
        if s[0] != 0x1C or tag[0] not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 240]:
            msg = "invalid IPTC/NAA file"
            raise SyntaxError(msg)

        # field size
        size = s[3]
        if size > 132:
            msg = "illegal field length in IPTC/NAA file"
            raise OSError(msg)
        elif size == 128:
            size = 0
        elif size > 128:
            size = _i(self.fp.read(size - 128))
        else:
            size = i16(s, 3)

        return tag, size

    def _open(self) -> None:
        # load descriptive fields
        assert self.fp is not None
        while True:
            offset = self.fp.tell()
            tag, size = self.field()
            if not tag or tag == (8, 10):
                break
            if size:
                tagdata = self.fp.read(size)
            else:
                tagdata = None
            if tag in self.info:
                if isinstance(self.info[tag], list):
                    self.info[tag].append(tagdata)
                else:
                    self.info[tag] = [self.info[tag], tagdata]
            else:
                self.info[tag] = tagdata

        # mode
        layers = self.info[(3, 60)][0]
        component = self.info[(3, 60)][1]
        if layers == 1 and not component:
            self._mode = "L"
            band = None
        else:
            if layers == 3 and component:
                self._mode = "RGB"
            elif layers == 4 and component:
                self._mode = "CMYK"
            if (3, 65) in self.info:
                band = self.info[(3, 65)][0] - 1
            else:
                band = 0

        # size
        self._size = self.getint((3, 20)), self.getint((3, 30))

        # compression
        try:
            compression = COMPRESSION[self.getint((3, 120))]
        except KeyError as e:
            msg = "Unknown IPTC image compression"
            raise OSError(msg) from e

        # tile
        if tag == (8, 10):
            self.tile = [
                ImageFile._Tile("iptc", (0, 0) + self.size, offset, (compression, band))
            ]

    def load(self) -> Image.core.PixelAccess | None:
        if self.tile:
            args = self.tile[0].args
            assert isinstance(args, tuple)
            compression, band = args

            assert self.fp is not None
            self.fp.seek(self.tile[0].offset)

            # Copy image data to temporary file
            o = BytesIO()
            if compression == "raw":
                # To simplify access to the extracted file,
                # prepend a PPM header
                o.write(b"P5\n%d %d\n255\n" % self.size)
            while True:
                type, size = self.field()
                if type != (8, 10):
                    break
                while size > 0:
                    s = self.fp.read(min(size, 8192))
                    if not s:
                        break
                    o.write(s)
                    size -= len(s)

            with Image.open(o) as _im:
                if band is not None:
                    bands = [Image.new("L", _im.size)] * Image.getmodebands(self.mode)
                    bands[band] = _im
                    im = Image.merge(self.mode, bands)
                else:
                    im = _im
                    im.load()
                self.im = im.im
            self.tile = []
        return ImageFile.ImageFile.load(self)


Image.register_open(IptcImageFile.format, IptcImageFile)

Image.register_extension(IptcImageFile.format, ".iim")


def getiptcinfo(
    im: ImageFile.ImageFile,
) -> dict[tuple[int, int], bytes | list[bytes]] | None:
    """
    Get IPTC information from TIFF, JPEG, or IPTC file.

    :param im: An image containing IPTC data.
    :returns: A dictionary containing IPTC information, or None if
        no IPTC information block was found.
    """
    from . import JpegImagePlugin, TiffImagePlugin

    data = None

    if isinstance(im, IptcImageFile):
        # return info dictionary right away
        return {k: v for k, v in im.info.items() if isinstance(k, tuple)}

    elif isinstance(im, JpegImagePlugin.JpegImageFile):
        # extract the IPTC/NAA resource
        photoshop = im.info.get("photoshop")
        if photoshop:
            data = photoshop.get(0x0404)

    elif isinstance(im, TiffImagePlugin.TiffImageFile):
        # get raw data from the IPTC/NAA tag (PhotoShop tags the data
        # as 4-byte integers, so we cannot use the get method...)
        try:
            data = im.tag_v2._tagdata[TiffImagePlugin.IPTC_NAA_CHUNK]
        except KeyError:
            pass

    if data is None:
        return None  # no properties

    # create an IptcImagePlugin object without initializing it
    class FakeImage:
        pass

    fake_im = FakeImage()
    fake_im.__class__ = IptcImageFile  # type: ignore[assignment]
    iptc_im = cast(IptcImageFile, fake_im)

    # parse the IPTC information chunk
    iptc_im.info = {}
    iptc_im.fp = BytesIO(data)

    try:
        iptc_im._open()
    except (IndexError, KeyError):
        pass  # expected failure

    return {k: v for k, v in iptc_im.info.items() if isinstance(k, tuple)}
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# transform wrappers
#
# History:
# 2002-04-08 fl   Created
#
# Copyright (c) 2002 by Secret Labs AB
# Copyright (c) 2002 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import Image


class Transform(Image.ImageTransformHandler):
    """Base class for other transforms defined in :py:mod:`~PIL.ImageTransform`."""

    method: Image.Transform

    def __init__(self, data: Sequence[Any]) -> None:
        self.data = data

    def getdata(self) -> tuple[Image.Transform, Sequence[int]]:
        return self.method, self.data

    def transform(
        self,
        size: tuple[int, int],
        image: Image.Image,
        **options: Any,
    ) -> Image.Image:
        """Perform the transform. Called from :py:meth:`.Image.transform`."""
        # can be overridden
        method, data = self.getdata()
        return image.transform(size, method, data, **options)


class AffineTransform(Transform):
    """
    Define an affine image transform.

    This function takes a 6-tuple (a, b, c, d, e, f) which contain the first
    two rows from the inverse of an affine transform matrix. For each pixel
    (x, y) in the output image, the new value is taken from a position (a x +
    b y + c, d x + e y + f) in the input image, rounded to nearest pixel.

    This function can be used to scale, translate, rotate, and shear the
    original image.

    See :py:meth:`.Image.transform`

    :param matrix: A 6-tuple (a, b, c, d, e, f) containing the first two rows
        from the inverse of an affine transform matrix.
    """

    method = Image.Transform.AFFINE


class PerspectiveTransform(Transform):
    """
    Define a perspective image transform.

    This function takes an 8-tuple (a, b, c, d, e, f, g, h). For each pixel
    (x, y) in the output image, the new value is taken from a position
    ((a x + b y + c) / (g x + h y + 1), (d x + e y + f) / (g x + h y + 1)) in
    the input image, rounded to nearest pixel.

    This function can be used to scale, translate, rotate, and shear the
    original image.

    See :py:meth:`.Image.transform`

    :param matrix: An 8-tuple (a, b, c, d, e, f, g, h).
    """

    method = Image.Transform.PERSPECTIVE


class ExtentTransform(Transform):
    """
    Define a transform to extract a subregion from an image.

    Maps a rectangle (defined by two corners) from the image to a rectangle of
    the given size. The resulting image will contain data sampled from between
    the corners, such that (x0, y0) in the input image will end up at (0,0) in
    the output image, and (x1, y1) at size.

    This method can be used to crop, stretch, shrink, or mirror an arbitrary
    rectangle in the current image. It is slightly slower than crop, but about
    as fast as a corresponding resize operation.

    See :py:meth:`.Image.transform`

    :param bbox: A 4-tuple (x0, y0, x1, y1) which specifies two points in the
        input image's coordinate system. See :ref:`coordinate-system`.
    """

    method = Image.Transform.EXTENT


class QuadTransform(Transform):
    """
    Define a quad image transform.

    Maps a quadrilateral (a region defined by four corners) from the image to a
    rectangle of the given size.

    See :py:meth:`.Image.transform`

    :param xy: An 8-tuple (x0, y0, x1, y1, x2, y2, x3, y3) which contain the
        upper left, lower left, lower right, and upper right corner of the
        source quadrilateral.
    """

    method = Image.Transform.QUAD


class MeshTransform(Transform):
    """
    Define a mesh image transform.  A mesh transform consists of one or more
    individual quad transforms.

    See :py:meth:`.Image.transform`

    :param data: A list of (bbox, quad) tuples.
    """

    method = Image.Transform.MESH
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# TIFF tags
#
# This module provides clear-text names for various well-known
# TIFF tags.  the TIFF codec works just fine without it.
#
# Copyright (c) Secret Labs AB 1999.
#
# See the README file for information on usage and redistribution.
#

##
# This module provides constants and clear-text names for various
# well-known TIFF tags.
##
from __future__ import annotations

from typing import NamedTuple


class _TagInfo(NamedTuple):
    value: int | None
    name: str
    type: int | None
    length: int | None
    enum: dict[str, int]


class TagInfo(_TagInfo):
    __slots__: list[str] = []

    def __new__(
        cls,
        value: int | None = None,
        name: str = "unknown",
        type: int | None = None,
        length: int | None = None,
        enum: dict[str, int] | None = None,
    ) -> TagInfo:
        return super().__new__(cls, value, name, type, length, enum or {})

    def cvt_enum(self, value: str) -> int | str:
        # Using get will call hash(value), which can be expensive
        # for some types (e.g. Fraction). Since self.enum is rarely
        # used, it's usually better to test it first.
        return self.enum.get(value, value) if self.enum else value


def lookup(tag: int, group: int | None = None) -> TagInfo:
    """
    :param tag: Integer tag number
    :param group: Which :py:data:`~PIL.TiffTags.TAGS_V2_GROUPS` to look in

    .. versionadded:: 8.3.0

    :returns: Taginfo namedtuple, From the ``TAGS_V2`` info if possible,
        otherwise just populating the value and name from ``TAGS``.
        If the tag is not recognized, "unknown" is returned for the name

    """

    if group is not None:
        info = TAGS_V2_GROUPS[group].get(tag) if group in TAGS_V2_GROUPS else None
    else:
        info = TAGS_V2.get(tag)
    return info or TagInfo(tag, TAGS.get(tag, "unknown"))


##
# Map tag numbers to tag info.
#
#  id: (Name, Type, Length[, enum_values])
#
# The length here differs from the length in the tiff spec.  For
# numbers, the tiff spec is for the number of fields returned. We
# agree here.  For string-like types, the tiff spec uses the length of
# field in bytes.  In Pillow, we are using the number of expected
# fields, in general 1 for string-like types.


BYTE = 1
ASCII = 2
SHORT = 3
LONG = 4
RATIONAL = 5
SIGNED_BYTE = 6
UNDEFINED = 7
SIGNED_SHORT = 8
SIGNED_LONG = 9
SIGNED_RATIONAL = 10
FLOAT = 11
DOUBLE = 12
IFD = 13
LONG8 = 16

_tags_v2: dict[int, tuple[str, int, int] | tuple[str, int, int, dict[str, int]]] = {
    254: ("NewSubfileType", LONG, 1),
    255: ("SubfileType", SHORT, 1),
    256: ("ImageWidth", LONG, 1),
    257: ("ImageLength", LONG, 1),
    258: ("BitsPerSample", SHORT, 0),
    259: (
        "Compression",
        SHORT,
        1,
        {
            "Uncompressed": 1,
            "CCITT 1d": 2,
            "Group 3 Fax": 3,
            "Group 4 Fax": 4,
            "LZW": 5,
            "JPEG": 6,
            "PackBits": 32773,
        },
    ),
    262: (
        "PhotometricInterpretation",
        SHORT,
        1,
        {
            "WhiteIsZero": 0,
            "BlackIsZero": 1,
            "RGB": 2,
            "RGB Palette": 3,
            "Transparency Mask": 4,
            "CMYK": 5,
            "YCbCr": 6,
            "CieLAB": 8,
            "CFA": 32803,  # TIFF/EP, Adobe DNG
            "LinearRaw": 32892,  # Adobe DNG
        },
    ),
    263: ("Threshholding", SHORT, 1),
    264: ("CellWidth", SHORT, 1),
    265: ("CellLength", SHORT, 1),
    266: ("FillOrder", SHORT, 1),
    269: ("DocumentName", ASCII, 1),
    270: ("ImageDescription", ASCII, 1),
    271: ("Make", ASCII, 1),
    272: ("Model", ASCII, 1),
    273: ("StripOffsets", LONG, 0),
    274: ("Orientation", SHORT, 1),
    277: ("SamplesPerPixel", SHORT, 1),
    278: ("RowsPerStrip", LONG, 1),
    279: ("StripByteCounts", LONG, 0),
    280: ("MinSampleValue", SHORT, 0),
    281: ("MaxSampleValue", SHORT, 0),
    282: ("XResolution", RATIONAL, 1),
    283: ("YResolution", RATIONAL, 1),
    284: ("PlanarConfiguration", SHORT, 1, {"Contiguous": 1, "Separate": 2}),
    285: ("PageName", ASCII, 1),
    286: ("XPosition", RATIONAL, 1),
    287: ("YPosition", RATIONAL, 1),
    288: ("FreeOffsets", LONG, 1),
    289: ("FreeByteCounts", LONG, 1),
    290: ("GrayResponseUnit", SHORT, 1),
    291: ("GrayResponseCurve", SHORT, 0),
    292: ("T4Options", LONG, 1),
    293: ("T6Options", LONG, 1),
    296: ("ResolutionUnit", SHORT, 1, {"none": 1, "inch": 2, "cm": 3}),
    297: ("PageNumber", SHORT, 2),
    301: ("TransferFunction", SHORT, 0),
    305: ("Software", ASCII, 1),
    306: ("DateTime", ASCII, 1),
    315: ("Artist", ASCII, 1),
    316: ("HostComputer", ASCII, 1),
    317: ("Predictor", SHORT, 1, {"none": 1, "Horizontal Differencing": 2}),
    318: ("WhitePoint", RATIONAL, 2),
    319: ("PrimaryChromaticities", RATIONAL, 6),
    320: ("ColorMap", SHORT, 0),
    321: ("HalftoneHints", SHORT, 2),
    322: ("TileWidth", LONG, 1),
    323: ("TileLength", LONG, 1),
    324: ("TileOffsets", LONG, 0),
    325: ("TileByteCounts", LONG, 0),
    330: ("SubIFDs", LONG, 0),
    332: ("InkSet", SHORT, 1),
    333: ("InkNames", ASCII, 1),
    334: ("NumberOfInks", SHORT, 1),
    336: ("DotRange", SHORT, 0),
    337: ("TargetPrinter", ASCII, 1),
    338: ("ExtraSamples", SHORT, 0),
    339: ("SampleFormat", SHORT, 0),
    340: ("SMinSampleValue", DOUBLE, 0),
    341: ("SMaxSampleValue", DOUBLE, 0),
    342: ("TransferRange", SHORT, 6),
    347: ("JPEGTables", UNDEFINED, 1),
    # obsolete JPEG tags
    512: ("JPEGProc", SHORT, 1),
    513: ("JPEGInterchangeFormat", LONG, 1),
    514: ("JPEGInterchangeFormatLength", LONG, 1),
    515: ("JPEGRestartInterval", SHORT, 1),
    517: ("JPEGLosslessPredictors", SHORT, 0),
    518: ("JPEGPointTransforms", SHORT, 0),
    519: ("JPEGQTables", LONG, 0),
    520: ("JPEGDCTables", LONG, 0),
    521: ("JPEGACTables", LONG, 0),
    529: ("YCbCrCoefficients", RATIONAL, 3),
    530: ("YCbCrSubSampling", SHORT, 2),
    531: ("YCbCrPositioning", SHORT, 1),
    532: ("ReferenceBlackWhite", RATIONAL, 6),
    700: ("XMP", BYTE, 0),
    # Four private SGI tags
    32995: ("Matteing", SHORT, 1),
    32996: ("DataType", SHORT, 0),
    32997: ("ImageDepth", LONG, 1),
    32998: ("TileDepth", LONG, 1),
    33432: ("Copyright", ASCII, 1),
    33723: ("IptcNaaInfo", UNDEFINED, 1),
    34377: ("PhotoshopInfo", BYTE, 0),
    # FIXME add more tags here
    34665: ("ExifIFD", LONG, 1),
    34675: ("ICCProfile", UNDEFINED, 1),
    34853: ("GPSInfoIFD", LONG, 1),
    36864: ("ExifVersion", UNDEFINED, 1),
    37724: ("ImageSourceData", UNDEFINED, 1),
    40965: ("InteroperabilityIFD", LONG, 1),
    41730: ("CFAPattern", UNDEFINED, 1),
    # MPInfo
    45056: ("MPFVersion", UNDEFINED, 1),
    45057: ("NumberOfImages", LONG, 1),
    45058: ("MPEntry", UNDEFINED, 1),
    45059: ("ImageUIDList", UNDEFINED, 0),  # UNDONE, check
    45060: ("TotalFrames", LONG, 1),
    45313: ("MPIndividualNum", LONG, 1),
    45569: ("PanOrientation", LONG, 1),
    45570: ("PanOverlap_H", RATIONAL, 1),
    45571: ("PanOverlap_V", RATIONAL, 1),
    45572: ("BaseViewpointNum", LONG, 1),
    45573: ("ConvergenceAngle", SIGNED_RATIONAL, 1),
    45574: ("BaselineLength", RATIONAL, 1),
    45575: ("VerticalDivergence", SIGNED_RATIONAL, 1),
    45576: ("AxisDistance_X", SIGNED_RATIONAL, 1),
    45577: ("AxisDistance_Y", SIGNED_RATIONAL, 1),
    45578: ("AxisDistance_Z", SIGNED_RATIONAL, 1),
    45579: ("YawAngle", SIGNED_RATIONAL, 1),
    45580: ("PitchAngle", SIGNED_RATIONAL, 1),
    45581: ("RollAngle", SIGNED_RATIONAL, 1),
    40960: ("FlashPixVersion", UNDEFINED, 1),
    50741: ("MakerNoteSafety", SHORT, 1, {"Unsafe": 0, "Safe": 1}),
    50780: ("BestQualityScale", RATIONAL, 1),
    50838: ("ImageJMetaDataByteCounts", LONG, 0),  # Can be more than one
    50839: ("ImageJMetaData", UNDEFINED, 1),  # see Issue #2006
}
_tags_v2_groups = {
    # ExifIFD
    34665: {
        36864: ("ExifVersion", UNDEFINED, 1),
        40960: ("FlashPixVersion", UNDEFINED, 1),
        40965: ("InteroperabilityIFD", LONG, 1),
        41730: ("CFAPattern", UNDEFINED, 1),
    },
    # GPSInfoIFD
    34853: {
        0: ("GPSVersionID", BYTE, 4),
        1: ("GPSLatitudeRef", ASCII, 2),
        2: ("GPSLatitude", RATIONAL, 3),
        3: ("GPSLongitudeRef", ASCII, 2),
        4: ("GPSLongitude", RATIONAL, 3),
        5: ("GPSAltitudeRef", BYTE, 1),
        6: ("GPSAltitude", RATIONAL, 1),
        7: ("GPSTimeStamp", RATIONAL, 3),
        8: ("GPSSatellites", ASCII, 0),
        9: ("GPSStatus", ASCII, 2),
        10: ("GPSMeasureMode", ASCII, 2),
        11: ("GPSDOP", RATIONAL, 1),
        12: ("GPSSpeedRef", ASCII, 2),
        13: ("GPSSpeed", RATIONAL, 1),
        14: ("GPSTrackRef", ASCII, 2),
        15: ("GPSTrack", RATIONAL, 1),
        16: ("GPSImgDirectionRef", ASCII, 2),
        17: ("GPSImgDirection", RATIONAL, 1),
        18: ("GPSMapDatum", ASCII, 0),
        19: ("GPSDestLatitudeRef", ASCII, 2),
        20: ("GPSDestLatitude", RATIONAL, 3),
        21: ("GPSDestLongitudeRef", ASCII, 2),
        22: ("GPSDestLongitude", RATIONAL, 3),
        23: ("GPSDestBearingRef", ASCII, 2),
        24: ("GPSDestBearing", RATIONAL, 1),
        25: ("GPSDestDistanceRef", ASCII, 2),
        26: ("GPSDestDistance", RATIONAL, 1),
        27: ("GPSProcessingMethod", UNDEFINED, 0),
        28: ("GPSAreaInformation", UNDEFINED, 0),
        29: ("GPSDateStamp", ASCII, 11),
        30: ("GPSDifferential", SHORT, 1),
    },
    # InteroperabilityIFD
    40965: {1: ("InteropIndex", ASCII, 1), 2: ("InteropVersion", UNDEFINED, 1)},
}

# Legacy Tags structure
# these tags aren't included above, but were in the previous versions
TAGS: dict[int | tuple[int, int], str] = {
    347: "JPEGTables",
    700: "XMP",
    # Additional Exif Info
    32932: "Wang Annotation",
    33434: "ExposureTime",
    33437: "FNumber",
    33445: "MD FileTag",
    33446: "MD ScalePixel",
    33447: "MD ColorTable",
    33448: "MD LabName",
    33449: "MD SampleInfo",
    33450: "MD PrepDate",
    33451: "MD PrepTime",
    33452: "MD FileUnits",
    33550: "ModelPixelScaleTag",
    33723: "IptcNaaInfo",
    33918: "INGR Packet Data Tag",
    33919: "INGR Flag Registers",
    33920: "IrasB Transformation Matrix",
    33922: "ModelTiepointTag",
    34264: "ModelTransformationTag",
    34377: "PhotoshopInfo",
    34735: "GeoKeyDirectoryTag",
    34736: "GeoDoubleParamsTag",
    34737: "GeoAsciiParamsTag",
    34850: "ExposureProgram",
    34852: "SpectralSensitivity",
    34855: "ISOSpeedRatings",
    34856: "OECF",
    34864: "SensitivityType",
    34865: "StandardOutputSensitivity",
    34866: "RecommendedExposureIndex",
    34867: "ISOSpeed",
    34868: "ISOSpeedLatitudeyyy",
    34869: "ISOSpeedLatitudezzz",
    34908: "HylaFAX FaxRecvParams",
    34909: "HylaFAX FaxSubAddress",
    34910: "HylaFAX FaxRecvTime",
    36864: "ExifVersion",
    36867: "DateTimeOriginal",
    36868: "DateTimeDigitized",
    37121: "ComponentsConfiguration",
    37122: "CompressedBitsPerPixel",
    37724: "ImageSourceData",
    37377: "ShutterSpeedValue",
    37378: "ApertureValue",
    37379: "BrightnessValue",
    37380: "ExposureBiasValue",
    37381: "MaxApertureValue",
    37382: "SubjectDistance",
    37383: "MeteringMode",
    37384: "LightSource",
    37385: "Flash",
    37386: "FocalLength",
    37396: "SubjectArea",
    37500: "MakerNote",
    37510: "UserComment",
    37520: "SubSec",
    37521: "SubSecTimeOriginal",
    37522: "SubsecTimeDigitized",
    40960: "FlashPixVersion",
    40961: "ColorSpace",
    40962: "PixelXDimension",
    40963: "PixelYDimension",
    40964: "RelatedSoundFile",
    40965: "InteroperabilityIFD",
    41483: "FlashEnergy",
    41484: "SpatialFrequencyResponse",
    41486: "FocalPlaneXResolution",
    41487: "FocalPlaneYResolution",
    41488: "FocalPlaneResolutionUnit",
    41492: "SubjectLocation",
    41493: "ExposureIndex",
    41495: "SensingMethod",
    41728: "FileSource",
    41729: "SceneType",
    41730: "CFAPattern",
    41985: "CustomRendered",
    41986: "ExposureMode",
    41987: "WhiteBalance",
    41988: "DigitalZoomRatio",
    41989: "FocalLengthIn35mmFilm",
    41990: "SceneCaptureType",
    41991: "GainControl",
    41992: "Contrast",
    41993: "Saturation",
    41994: "Sharpness",
    41995: "DeviceSettingDescription",
    41996: "SubjectDistanceRange",
    42016: "ImageUniqueID",
    42032: "CameraOwnerName",
    42033: "BodySerialNumber",
    42034: "LensSpecification",
    42035: "LensMake",
    42036: "LensModel",
    42037: "LensSerialNumber",
    42112: "GDAL_METADATA",
    42113: "GDAL_NODATA",
    42240: "Gamma",
    50215: "Oce Scanjob Description",
    50216: "Oce Application Selector",
    50217: "Oce Identification Number",
    50218: "Oce ImageLogic Characteristics",
    # Adobe DNG
    50706: "DNGVersion",
    50707: "DNGBackwardVersion",
    50708: "UniqueCameraModel",
    50709: "LocalizedCameraModel",
    50710: "CFAPlaneColor",
    50711: "CFALayout",
    50712: "LinearizationTable",
    50713: "BlackLevelRepeatDim",
    50714: "BlackLevel",
    50715: "BlackLevelDeltaH",
    50716: "BlackLevelDeltaV",
    50717: "WhiteLevel",
    50718: "DefaultScale",
    50719: "DefaultCropOrigin",
    50720: "DefaultCropSize",
    50721: "ColorMatrix1",
    50722: "ColorMatrix2",
    50723: "CameraCalibration1",
    50724: "CameraCalibration2",
    50725: "ReductionMatrix1",
    50726: "ReductionMatrix2",
    50727: "AnalogBalance",
    50728: "AsShotNeutral",
    50729: "AsShotWhiteXY",
    50730: "BaselineExposure",
    50731: "BaselineNoise",
    50732: "BaselineSharpness",
    50733: "BayerGreenSplit",
    50734: "LinearResponseLimit",
    50735: "CameraSerialNumber",
    50736: "LensInfo",
    50737: "ChromaBlurRadius",
    50738: "AntiAliasStrength",
    50740: "DNGPrivateData",
    50778: "CalibrationIlluminant1",
    50779: "CalibrationIlluminant2",
    50784: "Alias Layer Metadata",
}

TAGS_V2: dict[int, TagInfo] = {}
TAGS_V2_GROUPS: dict[int, dict[int, TagInfo]] = {}


def _populate() -> None:
    for k, v in _tags_v2.items():
        # Populate legacy structure.
        TAGS[k] = v[0]
        if len(v) == 4:
            for sk, sv in v[3].items():
                TAGS[(k, sv)] = sk

        TAGS_V2[k] = TagInfo(k, *v)

    for group, tags in _tags_v2_groups.items():
        TAGS_V2_GROUPS[group] = {k: TagInfo(k, *v) for k, v in tags.items()}


_populate()
##
# Map type numbers to type names -- defined in ImageFileDirectory.

TYPES: dict[int, str] = {}

#
# These tags are handled by default in libtiff, without
# adding to the custom dictionary. From tif_dir.c, searching for
# case TIFFTAG in the _TIFFVSetField function:
# Line: item.
# 148: case TIFFTAG_SUBFILETYPE:
# 151: case TIFFTAG_IMAGEWIDTH:
# 154: case TIFFTAG_IMAGELENGTH:
# 157: case TIFFTAG_BITSPERSAMPLE:
# 181: case TIFFTAG_COMPRESSION:
# 202: case TIFFTAG_PHOTOMETRIC:
# 205: case TIFFTAG_THRESHHOLDING:
# 208: case TIFFTAG_FILLORDER:
# 214: case TIFFTAG_ORIENTATION:
# 221: case TIFFTAG_SAMPLESPERPIXEL:
# 228: case TIFFTAG_ROWSPERSTRIP:
# 238: case TIFFTAG_MINSAMPLEVALUE:
# 241: case TIFFTAG_MAXSAMPLEVALUE:
# 244: case TIFFTAG_SMINSAMPLEVALUE:
# 247: case TIFFTAG_SMAXSAMPLEVALUE:
# 250: case TIFFTAG_XRESOLUTION:
# 256: case TIFFTAG_YRESOLUTION:
# 262: case TIFFTAG_PLANARCONFIG:
# 268: case TIFFTAG_XPOSITION:
# 271: case TIFFTAG_YPOSITION:
# 274: case TIFFTAG_RESOLUTIONUNIT:
# 280: case TIFFTAG_PAGENUMBER:
# 284: case TIFFTAG_HALFTONEHINTS:
# 288: case TIFFTAG_COLORMAP:
# 294: case TIFFTAG_EXTRASAMPLES:
# 298: case TIFFTAG_MATTEING:
# 305: case TIFFTAG_TILEWIDTH:
# 316: case TIFFTAG_TILELENGTH:
# 327: case TIFFTAG_TILEDEPTH:
# 333: case TIFFTAG_DATATYPE:
# 344: case TIFFTAG_SAMPLEFORMAT:
# 361: case TIFFTAG_IMAGEDEPTH:
# 364: case TIFFTAG_SUBIFD:
# 376: case TIFFTAG_YCBCRPOSITIONING:
# 379: case TIFFTAG_YCBCRSUBSAMPLING:
# 383: case TIFFTAG_TRANSFERFUNCTION:
# 389: case TIFFTAG_REFERENCEBLACKWHITE:
# 393: case TIFFTAG_INKNAMES:

# Following pseudo-tags are also handled by default in libtiff:
# TIFFTAG_JPEGQUALITY 65537

# some of these are not in our TAGS_V2 dict and were included from tiff.h

# This list also exists in encode.c
LIBTIFF_CORE = {
    255,
    256,
    257,
    258,
    259,
    262,
    263,
    266,
    274,
    277,
    278,
    280,
    281,
    340,
    341,
    282,
    283,
    284,
    286,
    287,
    296,
    297,
    321,
    320,
    338,
    32995,
    322,
    323,
    32998,
    32996,
    339,
    32997,
    330,
    531,
    530,
    301,
    532,
    333,
    # as above
    269,  # this has been in our tests forever, and works
    65537,
}

LIBTIFF_CORE.remove(255)  # We don't have support for subfiletypes
LIBTIFF_CORE.remove(322)  # We don't have support for writing tiled images with libtiff
LIBTIFF_CORE.remove(323)  # Tiled images

# Note to advanced users: There may be combinations of these
# parameters and values that when added properly, will work and
# produce valid tiff images that may work in your application.
# It is safe to add and remove tags from this set from Pillow's point
# of view so long as you test against libtiff.
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# PNG support code
#
# See "PNG (Portable Network Graphics) Specification, version 1.0;
# W3C Recommendation", 1996-10-01, Thomas Boutell (ed.).
#
# history:
# 1996-05-06 fl   Created (couldn't resist it)
# 1996-12-14 fl   Upgraded, added read and verify support (0.2)
# 1996-12-15 fl   Separate PNG stream parser
# 1996-12-29 fl   Added write support, added getchunks
# 1996-12-30 fl   Eliminated circular references in decoder (0.3)
# 1998-07-12 fl   Read/write 16-bit images as mode I (0.4)
# 2001-02-08 fl   Added transparency support (from Zircon) (0.5)
# 2001-04-16 fl   Don't close data source in "open" method (0.6)
# 2004-02-24 fl   Don't even pretend to support interlaced files (0.7)
# 2004-08-31 fl   Do basic sanity check on chunk identifiers (0.8)
# 2004-09-20 fl   Added PngInfo chunk container
# 2004-12-18 fl   Added DPI read support (based on code by Niki Spahiev)
# 2008-08-13 fl   Added tRNS support for RGB images
# 2009-03-06 fl   Support for preserving ICC profiles (by Florian Hoech)
# 2009-03-08 fl   Added zTXT support (from Lowell Alleman)
# 2009-03-29 fl   Read interlaced PNG files (from Conrado Porto Lopes Gouvua)
#
# Copyright (c) 1997-2009 by Secret Labs AB
# Copyright (c) 1996 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from fractions import Fraction
from typing import IO, NamedTuple, cast

from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
from ._deprecate import deprecate
from ._util import DeferredError

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, NoReturn

    from . import _imaging

logger = logging.getLogger(__name__)

is_cid = re.compile(rb"\w\w\w\w").match


_MAGIC = b"\211PNG\r\n\032\n"


_MODES = {
    # supported bits/color combinations, and corresponding modes/rawmodes
    # Grayscale
    (1, 0): ("1", "1"),
    (2, 0): ("L", "L;2"),
    (4, 0): ("L", "L;4"),
    (8, 0): ("L", "L"),
    (16, 0): ("I;16", "I;16B"),
    # Truecolour
    (8, 2): ("RGB", "RGB"),
    (16, 2): ("RGB", "RGB;16B"),
    # Indexed-colour
    (1, 3): ("P", "P;1"),
    (2, 3): ("P", "P;2"),
    (4, 3): ("P", "P;4"),
    (8, 3): ("P", "P"),
    # Grayscale with alpha
    (8, 4): ("LA", "LA"),
    (16, 4): ("RGBA", "LA;16B"),  # LA;16B->LA not yet available
    # Truecolour with alpha
    (8, 6): ("RGBA", "RGBA"),
    (16, 6): ("RGBA", "RGBA;16B"),
}


_simple_palette = re.compile(b"^\xff*\x00\xff*$")

MAX_TEXT_CHUNK = ImageFile.SAFEBLOCK
"""
Maximum decompressed size for a iTXt or zTXt chunk.
Eliminates decompression bombs where compressed chunks can expand 1000x.
See :ref:`Text in PNG File Format<png-text>`.
"""
MAX_TEXT_MEMORY = 64 * MAX_TEXT_CHUNK
"""
Set the maximum total text chunk size.
See :ref:`Text in PNG File Format<png-text>`.
"""


# APNG frame disposal modes
class Disposal(IntEnum):
    OP_NONE = 0
    """
    No disposal is done on this frame before rendering the next frame.
    See :ref:`Saving APNG sequences<apng-saving>`.
    """
    OP_BACKGROUND = 1
    """
    This frame’s modified region is cleared to fully transparent black before rendering
    the next frame.
    See :ref:`Saving APNG sequences<apng-saving>`.
    """
    OP_PREVIOUS = 2
    """
    This frame’s modified region is reverted to the previous frame’s contents before
    rendering the next frame.
    See :ref:`Saving APNG sequences<apng-saving>`.
    """


# APNG frame blend modes
class Blend(IntEnum):
    OP_SOURCE = 0
    """
    All color components of this frame, including alpha, overwrite the previous output
    image contents.
    See :ref:`Saving APNG sequences<apng-saving>`.
    """
    OP_OVER = 1
    """
    This frame should be alpha composited with the previous output image contents.
    See :ref:`Saving APNG sequences<apng-saving>`.
    """


def _safe_zlib_decompress(s: bytes) -> bytes:
    dobj = zlib.decompressobj()
    plaintext = dobj.decompress(s, MAX_TEXT_CHUNK)
    if dobj.unconsumed_tail:
        msg = "Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK"
        raise ValueError(msg)
    return plaintext


def _crc32(data: bytes, seed: int = 0) -> int:
    return zlib.crc32(data, seed) & 0xFFFFFFFF


# --------------------------------------------------------------------
# Support classes.  Suitable for PNG and related formats like MNG etc.


class ChunkStream:
    def __init__(self, fp: IO[bytes]) -> None:
        self.fp: IO[bytes] | None = fp
        self.queue: list[tuple[bytes, int, int]] | None = []

    def read(self) -> tuple[bytes, int, int]:
        """Fetch a new chunk. Returns header information."""
        cid = None

        assert self.fp is not None
        if self.queue:
            cid, pos, length = self.queue.pop()
            self.fp.seek(pos)
        else:
            s = self.fp.read(8)
            cid = s[4:]
            pos = self.fp.tell()
            length = i32(s)

        if not is_cid(cid):
            if not ImageFile.LOAD_TRUNCATED_IMAGES:
                msg = f"broken PNG file (chunk {repr(cid)})"
                raise SyntaxError(msg)

        return cid, pos, length

    def __enter__(self) -> ChunkStream:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.queue = self.fp = None

    def push(self, cid: bytes, pos: int, length: int) -> None:
        assert self.queue is not None
        self.queue.append((cid, pos, length))

    def call(self, cid: bytes, pos: int, length: int) -> bytes:
        """Call the appropriate chunk handler"""

        logger.debug("STREAM %r %s %s", cid, pos, length)
        return getattr(self, f"chunk_{cid.decode('ascii')}")(pos, length)

    def crc(self, cid: bytes, data: bytes) -> None:
        """Read and verify checksum"""

        # Skip CRC checks for ancillary chunks if allowed to load truncated
        # images
        # 5th byte of first char is 1 [specs, section 5.4]
        if ImageFile.LOAD_TRUNCATED_IMAGES and (cid[0] >> 5 & 1):
            self.crc_skip(cid, data)
            return

        assert self.fp is not None
        try:
            crc1 = _crc32(data, _crc32(cid))
            crc2 = i32(self.fp.read(4))
            if crc1 != crc2:
                msg = f"broken PNG file (bad header checksum in {repr(cid)})"
                raise SyntaxError(msg)
        except struct.error as e:
            msg = f"broken PNG file (incomplete checksum in {repr(cid)})"
            raise SyntaxError(msg) from e

    def crc_skip(self, cid: bytes, data: bytes) -> None:
        """Read checksum"""

        assert self.fp is not None
        self.fp.read(4)

    def verify(self, endchunk: bytes = b"IEND") -> list[bytes]:
        # Simple approach; just calculate checksum for all remaining
        # blocks.  Must be called directly after open.

        cids = []

        assert self.fp is not None
        while True:
            try:
                cid, pos, length = self.read()
            except struct.error as e:
                msg = "truncated PNG file"
                raise OSError(msg) from e

            if cid == endchunk:
                break
            self.crc(cid, ImageFile._safe_read(self.fp, length))
            cids.append(cid)

        return cids


class iTXt(str):
    """
    Subclass of string to allow iTXt chunks to look like strings while
    keeping their extra information

    """

    lang: str | bytes | None
    tkey: str | bytes | None

    @staticmethod
    def __new__(
        cls, text: str, lang: str | None = None, tkey: str | None = None
    ) -> iTXt:
        """
        :param cls: the class to use when creating the instance
        :param text: value for this key
        :param lang: language code
        :param tkey: UTF-8 version of the key name
        """

        self = str.__new__(cls, text)
        self.lang = lang
        self.tkey = tkey
        return self


class PngInfo:
    """
    PNG chunk container (for use with save(pnginfo=))

    """

    def __init__(self) -> None:
        self.chunks: list[tuple[bytes, bytes, bool]] = []

    def add(self, cid: bytes, data: bytes, after_idat: bool = False) -> None:
        """Appends an arbitrary chunk. Use with caution.

        :param cid: a byte string, 4 bytes long.
        :param data: a byte string of the encoded data
        :param after_idat: for use with private chunks. Whether the chunk
                           should be written after IDAT

        """

        self.chunks.append((cid, data, after_idat))

    def add_itxt(
        self,
        key: str | bytes,
        value: str | bytes,
        lang: str | bytes = "",
        tkey: str | bytes = "",
        zip: bool = False,
    ) -> None:
        """Appends an iTXt chunk.

        :param key: latin-1 encodable text key name
        :param value: value for this key
        :param lang: language code
        :param tkey: UTF-8 version of the key name
        :param zip: compression flag

        """

        if not isinstance(key, bytes):
            key = key.encode("latin-1", "strict")
        if not isinstance(value, bytes):
            value = value.encode("utf-8", "strict")
        if not isinstance(lang, bytes):
            lang = lang.encode("utf-8", "strict")
        if not isinstance(tkey, bytes):
            tkey = tkey.encode("utf-8", "strict")

        if zip:
            self.add(
                b"iTXt",
                key + b"\0\x01\0" + lang + b"\0" + tkey + b"\0" + zlib.compress(value),
            )
        else:
            self.add(b"iTXt", key + b"\0\0\0" + lang + b"\0" + tkey + b"\0" + value)

    def add_text(
        self, key: str | bytes, value: str | bytes | iTXt, zip: bool = False
    ) -> None:
        """Appends a text chunk.

        :param key: latin-1 encodable text key name
        :param value: value for this key, text or an
           :py:class:`PIL.PngImagePlugin.iTXt` instance
        :param zip: compression flag

        """
        if isinstance(value, iTXt):
            return self.add_itxt(
                key,
                value,
                value.lang if value.lang is not None else b"",
                value.tkey if value.tkey is not None else b"",
                zip=zip,
            )

        # The tEXt chunk stores latin-1 text
        if not isinstance(value, bytes):
            try:
                value = value.encode("latin-1", "strict")
            except UnicodeError:
                return self.add_itxt(key, value, zip=zip)

        if not isinstance(key, bytes):
            key = key.encode("latin-1", "strict")

        if zip:
            self.add(b"zTXt", key + b"\0\0" + zlib.compress(value))
        else:
            self.add(b"tEXt", key + b"\0" + value)


# --------------------------------------------------------------------
# PNG image stream (IHDR/IEND)


class _RewindState(NamedTuple):
    info: dict[str | tuple[int, int], Any]
    tile: list[ImageFile._Tile]
    seq_num: int | None


class PngStream(ChunkStream):
    def __init__(self, fp: IO[bytes]) -> None:
        super().__init__(fp)

        # local copies of Image attributes
        self.im_info: dict[str | tuple[int, int], Any] = {}
        self.im_text: dict[str, str | iTXt] = {}
        self.im_size = (0, 0)
        self.im_mode = ""
        self.im_tile: list[ImageFile._Tile] = []
        self.im_palette: tuple[str, bytes] | None = None
        self.im_custom_mimetype: str | None = None
        self.im_n_frames: int | None = None
        self._seq_num: int | None = None
        self.rewind_state = _RewindState({}, [], None)

        self.text_memory = 0

    def __enter__(self) -> PngStream:
        return self

    def check_text_memory(self, chunklen: int) -> None:
        self.text_memory += chunklen
        if self.text_memory > MAX_TEXT_MEMORY:
            msg = (
                "Too much memory used in text chunks: "
                f"{self.text_memory}>MAX_TEXT_MEMORY"
            )
            raise ValueError(msg)

    def save_rewind(self) -> None:
        self.rewind_state = _RewindState(
            self.im_info.copy(),
            self.im_tile,
            self._seq_num,
        )

    def rewind(self) -> None:
        self.im_info = self.rewind_state.info.copy()
        self.im_tile = self.rewind_state.tile
        self._seq_num = self.rewind_state.seq_num

    def chunk_iCCP(self, pos: int, length: int) -> bytes:
        # ICC profile
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        # according to PNG spec, the iCCP chunk contains:
        # Profile name  1-79 bytes (character string)
        # Null separator        1 byte (null character)
        # Compression method    1 byte (0)
        # Compressed profile    n bytes (zlib with deflate compression)
        i = s.find(b"\0")
        logger.debug("iCCP profile name %r", s[:i])
        comp_method = s[i + 1]
        logger.debug("Compression method %s", comp_method)
        if comp_method != 0:
            msg = f"Unknown compression method {comp_method} in iCCP chunk"
            raise SyntaxError(msg)
        try:
            icc_profile = _safe_zlib_decompress(s[i + 2 :])
        except ValueError:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                icc_profile = None
            else:
                raise
        except zlib.error:
            icc_profile = None  # FIXME
        self.im_info["icc_profile"] = icc_profile
        return s

    def chunk_IHDR(self, pos: int, length: int) -> bytes:
        # image header
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        if length < 13:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = "Truncated IHDR chunk"
            raise ValueError(msg)
        self.im_size = i32(s, 0), i32(s, 4)
        try:
            self.im_mode, self.im_rawmode = _MODES[(s[8], s[9])]
        except KeyError:
            pass
        if s[12]:
            self.im_info["interlace"] = 1
        if s[11]:
            msg = "unknown filter category"
            raise SyntaxError(msg)
        return s

    def chunk_IDAT(self, pos: int, length: int) -> NoReturn:
        # image data
        if "bbox" in self.im_info:
            tile = [ImageFile._Tile("zip", self.im_info["bbox"], pos, self.im_rawmode)]
        else:
            if self.im_n_frames is not None:
                self.im_info["default_image"] = True
            tile = [ImageFile._Tile("zip", (0, 0) + self.im_size, pos, self.im_rawmode)]
        self.im_tile = tile
        self.im_idat = length
        msg = "image data found"
        raise EOFError(msg)

    def chunk_IEND(self, pos: int, length: int) -> NoReturn:
        msg = "end of PNG image"
        raise EOFError(msg)

    def chunk_PLTE(self, pos: int, length: int) -> bytes:
        # palette
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        if self.im_mode == "P":
            self.im_palette = "RGB", s
        return s

    def chunk_tRNS(self, pos: int, length: int) -> bytes:
        # transparency
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        if self.im_mode == "P":
            if _simple_palette.match(s):
                # tRNS contains only one full-transparent entry,
                # other entries are full opaque
                i = s.find(b"\0")
                if i >= 0:
                    self.im_info["transparency"] = i
            else:
                # otherwise, we have a byte string with one alpha value
                # for each palette entry
                self.im_info["transparency"] = s
        elif self.im_mode == "1":
            self.im_info["transparency"] = 255 if i16(s) else 0
        elif self.im_mode in ("L", "I;16"):
            self.im_info["transparency"] = i16(s)
        elif self.im_mode == "RGB":
            self.im_info["transparency"] = i16(s), i16(s, 2), i16(s, 4)
        return s

    def chunk_gAMA(self, pos: int, length: int) -> bytes:
        # gamma setting
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        self.im_info["gamma"] = i32(s) / 100000.0
        return s

    def chunk_cHRM(self, pos: int, length: int) -> bytes:
        # chromaticity, 8 unsigned ints, actual value is scaled by 100,000
        # WP x,y, Red x,y, Green x,y Blue x,y

        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        raw_vals = struct.unpack(f">{len(s) // 4}I", s)
        self.im_info["chromaticity"] = tuple(elt / 100000.0 for elt in raw_vals)
        return s

    def chunk_sRGB(self, pos: int, length: int) -> bytes:
        # srgb rendering intent, 1 byte
        # 0 perceptual
        # 1 relative colorimetric
        # 2 saturation
        # 3 absolute colorimetric

        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        if length < 1:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = "Truncated sRGB chunk"
            raise ValueError(msg)
        self.im_info["srgb"] = s[0]
        return s

    def chunk_pHYs(self, pos: int, length: int) -> bytes:
        # pixels per unit
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        if length < 9:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = "Truncated pHYs chunk"
            raise ValueError(msg)
        px, py = i32(s, 0), i32(s, 4)
        unit = s[8]
        if unit == 1:  # meter
            dpi = px * 0.0254, py * 0.0254
            self.im_info["dpi"] = dpi
        elif unit == 0:
            self.im_info["aspect"] = px, py
        return s

    def chunk_tEXt(self, pos: int, length: int) -> bytes:
        # text
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        try:
            k, v = s.split(b"\0", 1)
        except ValueError:
            # fallback for broken tEXt tags
            k = s
            v = b""
        if k:
            k_str = k.decode("latin-1", "strict")
            v_str = v.decode("latin-1", "replace")

            self.im_info[k_str] = v if k == b"exif" else v_str
            self.im_text[k_str] = v_str
            self.check_text_memory(len(v_str))

        return s

    def chunk_zTXt(self, pos: int, length: int) -> bytes:
        # compressed text
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        try:
            k, v = s.split(b"\0", 1)
        except ValueError:
            k = s
            v = b""
        if v:
            comp_method = v[0]
        else:
            comp_method = 0
        if comp_method != 0:
            msg = f"Unknown compression method {comp_method} in zTXt chunk"
            raise SyntaxError(msg)
        try:
            v = _safe_zlib_decompress(v[1:])
        except ValueError:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                v = b""
            else:
                raise
        except zlib.error:
            v = b""

        if k:
            k_str = k.decode("latin-1", "strict")
            v_str = v.decode("latin-1", "replace")

            self.im_info[k_str] = self.im_text[k_str] = v_str
            self.check_text_memory(len(v_str))

        return s

    def chunk_iTXt(self, pos: int, length: int) -> bytes:
        # international text
        assert self.fp is not None
        r = s = ImageFile._safe_read(self.fp, length)
        try:
            k, r = r.split(b"\0", 1)
        except ValueError:
            return s
        if len(r) < 2:
            return s
        cf, cm, r = r[0], r[1], r[2:]
        try:
            lang, tk, v = r.split(b"\0", 2)
        except ValueError:
            return s
        if cf != 0:
            if cm == 0:
                try:
                    v = _safe_zlib_decompress(v)
                except ValueError:
                    if ImageFile.LOAD_TRUNCATED_IMAGES:
                        return s
                    else:
                        raise
                except zlib.error:
                    return s
            else:
                return s
        if k == b"XML:com.adobe.xmp":
            self.im_info["xmp"] = v
        try:
            k_str = k.decode("latin-1", "strict")
            lang_str = lang.decode("utf-8", "strict")
            tk_str = tk.decode("utf-8", "strict")
            v_str = v.decode("utf-8", "strict")
        except UnicodeError:
            return s

        self.im_info[k_str] = self.im_text[k_str] = iTXt(v_str, lang_str, tk_str)
        self.check_text_memory(len(v_str))

        return s

    def chunk_eXIf(self, pos: int, length: int) -> bytes:
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        self.im_info["exif"] = b"Exif\x00\x00" + s
        return s

    # APNG chunks
    def chunk_acTL(self, pos: int, length: int) -> bytes:
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        if length < 8:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = "APNG contains truncated acTL chunk"
            raise ValueError(msg)
        if self.im_n_frames is not None:
            self.im_n_frames = None
            warnings.warn("Invalid APNG, will use default PNG image if possible")
            return s
        n_frames = i32(s)
        if n_frames == 0 or n_frames > 0x80000000:
            warnings.warn("Invalid APNG, will use default PNG image if possible")
            return s
        self.im_n_frames = n_frames
        self.im_info["loop"] = i32(s, 4)
        self.im_custom_mimetype = "image/apng"
        return s

    def chunk_fcTL(self, pos: int, length: int) -> bytes:
        assert self.fp is not None
        s = ImageFile._safe_read(self.fp, length)
        if length < 26:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                return s
            msg = "APNG contains truncated fcTL chunk"
            raise ValueError(msg)
        seq = i32(s)
        if (self._seq_num is None and seq != 0) or (
            self._seq_num is not None and self._seq_num != seq - 1
        ):
            msg = "APNG contains frame sequence errors"
            raise SyntaxError(msg)
        self._seq_num = seq
        width, height = i32(s, 4), i32(s, 8)
        px, py = i32(s, 12), i32(s, 16)
        im_w, im_h = self.im_size
        if px + width > im_w or py + height > im_h:
            msg = "APNG contains invalid frames"
            raise SyntaxError(msg)
        self.im_info["bbox"] = (px, py, px + width, py + height)
        delay_num, delay_den = i16(s, 20), i16(s, 22)
        if delay_den == 0:
            delay_den = 100
        self.im_info["duration"] = float(delay_num) / float(delay_den) * 1000
        self.im_info["disposal"] = s[24]
        self.im_info["blend"] = s[25]
        return s

    def chunk_fdAT(self, pos: int, length: int) -> bytes:
        assert self.fp is not None
        if length < 4:
            if ImageFile.LOAD_TRUNCATED_IMAGES:
                s = ImageFile._safe_read(self.fp, length)
                return s
            msg = "APNG contains truncated fDAT chunk"
            raise ValueError(msg)
        s = ImageFile._safe_read(self.fp, 4)
        seq = i32(s)
        if self._seq_num != seq - 1:
            msg = "APNG contains frame sequence errors"
            raise SyntaxError(msg)
        self._seq_num = seq
        return self.chunk_IDAT(pos + 4, length - 4)


# --------------------------------------------------------------------
# PNG reader


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(_MAGIC)


##
# Image plugin for PNG images.


class PngImageFile(ImageFile.ImageFile):
    format = "PNG"
    format_description = "Portable network graphics"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(8)):
            msg = "not a PNG file"
            raise SyntaxError(msg)
        self._fp = self.fp
        self.__frame = 0

        #
        # Parse headers up to the first IDAT or fDAT chunk

        self.private_chunks: list[tuple[bytes, bytes] | tuple[bytes, bytes, bool]] = []
        self.png: PngStream | None = PngStream(self.fp)

        while True:
            #
            # get next chunk

            cid, pos, length = self.png.read()

            try:
                s = self.png.call(cid, pos, length)
            except EOFError:
                break
            except AttributeError:
                logger.debug("%r %s %s (unknown)", cid, pos, length)
                s = ImageFile._safe_read(self.fp, length)
                if cid[1:2].islower():
                    self.private_chunks.append((cid, s))

            self.png.crc(cid, s)

        #
        # Copy relevant attributes from the PngStream.  An alternative
        # would be to let the PngStream class modify these attributes
        # directly, but that introduces circular references which are
        # difficult to break if things go wrong in the decoder...
        # (believe me, I've tried ;-)

        self._mode = self.png.im_mode
        self._size = self.png.im_size
        self.info = self.png.im_info
        self._text: dict[str, str | iTXt] | None = None
        self.tile = self.png.im_tile
        self.custom_mimetype = self.png.im_custom_mimetype
        self.n_frames = self.png.im_n_frames or 1
        self.default_image = self.info.get("default_image", False)

        if self.png.im_palette:
            rawmode, data = self.png.im_palette
            self.palette = ImagePalette.raw(rawmode, data)

        if cid == b"fdAT":
            self.__prepare_idat = length - 4
        else:
            self.__prepare_idat = length  # used by load_prepare()

        if self.png.im_n_frames is not None:
            self._close_exclusive_fp_after_loading = False
            self.png.save_rewind()
            self.__rewind_idat = self.__prepare_idat
            self.__rewind = self._fp.tell()
            if self.default_image:
                # IDAT chunk contains default image and not first animation frame
                self.n_frames += 1
            self._seek(0)
        self.is_animated = self.n_frames > 1

    @property
    def text(self) -> dict[str, str | iTXt]:
        # experimental
        if self._text is None:
            # iTxt, tEXt and zTXt chunks may appear at the end of the file
            # So load the file to ensure that they are read
            if self.is_animated:
                frame = self.__frame
                # for APNG, seek to the final frame before loading
                self.seek(self.n_frames - 1)
            self.load()
            if self.is_animated:
                self.seek(frame)
        assert self._text is not None
        return self._text

    def verify(self) -> None:
        """Verify PNG file"""

        if self.fp is None:
            msg = "verify must be called directly after open"
            raise RuntimeError(msg)

        # back up to beginning of IDAT block
        self.fp.seek(self.tile[0][2] - 8)

        assert self.png is not None
        self.png.verify()
        self.png.close()

        super().verify()

    def seek(self, frame: int) -> None:
        if not self._seek_check(frame):
            return
        if frame < self.__frame:
            self._seek(0, True)

        last_frame = self.__frame
        try:
            for f in range(self.__frame + 1, frame + 1):
                self._seek(f)
        except EOFError as e:
            self.seek(last_frame)
            msg = "no more images in APNG file"
            raise EOFError(msg) from e

    def _seek(self, frame: int, rewind: bool = False) -> None:
        assert self.png is not None
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex

        self.dispose: _imaging.ImagingCore | None
        dispose_extent = None
        if frame == 0:
            if rewind:
                self._fp.seek(self.__rewind)
                self.png.rewind()
                self.__prepare_idat = self.__rewind_idat
                self._im = None
                self.info = self.png.im_info
                self.tile = self.png.im_tile
                self.fp = self._fp
            self._prev_im = None
            self.dispose = None
            self.default_image = self.info.get("default_image", False)
            self.dispose_op = self.info.get("disposal")
            self.blend_op = self.info.get("blend")
            dispose_extent = self.info.get("bbox")
            self.__frame = 0
        else:
            if frame != self.__frame + 1:
                msg = f"cannot seek to frame {frame}"
                raise ValueError(msg)

            # ensure previous frame was loaded
            self.load()

            if self.dispose:
                self.im.paste(self.dispose, self.dispose_extent)
            self._prev_im = self.im.copy()

            self.fp = self._fp

            # advance to the next frame
            if self.__prepare_idat:
                ImageFile._safe_read(self.fp, self.__prepare_idat)
                self.__prepare_idat = 0
            frame_start = False
            while True:
                self.fp.read(4)  # CRC

                try:
                    cid, pos, length = self.png.read()
                except (struct.error, SyntaxError):
                    break

                if cid == b"IEND":
                    msg = "No more images in APNG file"
                    raise EOFError(msg)
                if cid == b"fcTL":
                    if frame_start:
                        # there must be at least one fdAT chunk between fcTL chunks
                        msg = "APNG missing frame data"
                        raise SyntaxError(msg)
                    frame_start = True

                try:
                    self.png.call(cid, pos, length)
                except UnicodeDecodeError:
                    break
                except EOFError:
                    if cid == b"fdAT":
                        length -= 4
                        if frame_start:
                            self.__prepare_idat = length
                            break
                    ImageFile._safe_read(self.fp, length)
                except AttributeError:
                    logger.debug("%r %s %s (unknown)", cid, pos, length)
                    ImageFile._safe_read(self.fp, length)

            self.__frame = frame
            self.tile = self.png.im_tile
            self.dispose_op = self.info.get("disposal")
            self.blend_op = self.info.get("blend")
            dispose_extent = self.info.get("bbox")

            if not self.tile:
                msg = "image not found in APNG frame"
                raise EOFError(msg)
        if dispose_extent:
            self.dispose_extent: tuple[float, float, float, float] = dispose_extent

        # setup frame disposal (actual disposal done when needed in the next _seek())
        if self._prev_im is None and self.dispose_op == Disposal.OP_PREVIOUS:
            self.dispose_op = Disposal.OP_BACKGROUND

        self.dispose = None
        if self.dispose_op == Disposal.OP_PREVIOUS:
            if self._prev_im:
                self.dispose = self._prev_im.copy()
                self.dispose = self._crop(self.dispose, self.dispose_extent)
        elif self.dispose_op == Disposal.OP_BACKGROUND:
            self.dispose = Image.core.fill(self.mode, self.size)
            self.dispose = self._crop(self.dispose, self.dispose_extent)

    def tell(self) -> int:
        return self.__frame

    def load_prepare(self) -> None:
        """internal: prepare to read PNG file"""

        if self.info.get("interlace"):
            self.decoderconfig = self.decoderconfig + (1,)

        self.__idat = self.__prepare_idat  # used by load_read()
        ImageFile.ImageFile.load_prepare(self)

    def load_read(self, read_bytes: int) -> bytes:
        """internal: read more image data"""

        assert self.png is not None
        assert self.fp is not None
        while self.__idat == 0:
            # end of chunk, skip forward to next one

            self.fp.read(4)  # CRC

            cid, pos, length = self.png.read()

            if cid not in [b"IDAT", b"DDAT", b"fdAT"]:
                self.png.push(cid, pos, length)
                return b""

            if cid == b"fdAT":
                try:
                    self.png.call(cid, pos, length)
                except EOFError:
                    pass
                self.__idat = length - 4  # sequence_num has already been read
            else:
                self.__idat = length  # empty chunks are allowed

        # read more data from this chunk
        if read_bytes <= 0:
            read_bytes = self.__idat
        else:
            read_bytes = min(read_bytes, self.__idat)

        self.__idat = self.__idat - read_bytes

        return self.fp.read(read_bytes)

    def load_end(self) -> None:
        """internal: finished reading image data"""
        assert self.png is not None
        assert self.fp is not None
        if self.__idat != 0:
            self.fp.read(self.__idat)
        while True:
            self.fp.read(4)  # CRC

            try:
                cid, pos, length = self.png.read()
            except (struct.error, SyntaxError):
                break

            if cid == b"IEND":
                break
            elif cid == b"fcTL" and self.is_animated:
                # start of the next frame, stop reading
                self.__prepare_idat = 0
                self.png.push(cid, pos, length)
                break

            try:
                self.png.call(cid, pos, length)
            except UnicodeDecodeError:
                break
            except EOFError:
                if cid == b"fdAT":
                    length -= 4
                try:
                    ImageFile._safe_read(self.fp, length)
                except OSError as e:
                    if ImageFile.LOAD_TRUNCATED_IMAGES:
                        break
                    else:
                        raise e
            except AttributeError:
                logger.debug("%r %s %s (unknown)", cid, pos, length)
                s = ImageFile._safe_read(self.fp, length)
                if cid[1:2].islower():
                    self.private_chunks.append((cid, s, True))
        self._text = self.png.im_text
        if not self.is_animated:
            self.png.close()
            self.png = None
        else:
            if self._prev_im and self.blend_op == Blend.OP_OVER:
                updated = self._crop(self.im, self.dispose_extent)
                if self.im.mode == "RGB" and "transparency" in self.info:
                    mask = updated.convert_transparent(
                        "RGBA", self.info["transparency"]
                    )
                else:
                    if self.im.mode == "P" and "transparency" in self.info:
                        t = self.info["transparency"]
                        if isinstance(t, bytes):
                            updated.putpalettealphas(t)
                        elif isinstance(t, int):
                            updated.putpalettealpha(t)
                    mask = updated.convert("RGBA")
                self._prev_im.paste(updated, self.dispose_extent, mask)
                self.im = self._prev_im

    def _getexif(self) -> dict[int, Any] | None:
        if "exif" not in self.info:
            self.load()
        if "exif" not in self.info and "Raw profile type exif" not in self.info:
            return None
        return self.getexif()._get_merged_dict()

    def getexif(self) -> Image.Exif:
        if "exif" not in self.info:
            self.load()

        return super().getexif()


# --------------------------------------------------------------------
# PNG writer

_OUTMODES = {
    # supported PIL modes, and corresponding rawmode, bit depth and color type
    "1": ("1", b"\x01", b"\x00"),
    "L;1": ("L;1", b"\x01", b"\x00"),
    "L;2": ("L;2", b"\x02", b"\x00"),
    "L;4": ("L;4", b"\x04", b"\x00"),
    "L": ("L", b"\x08", b"\x00"),
    "LA": ("LA", b"\x08", b"\x04"),
    "I": ("I;16B", b"\x10", b"\x00"),
    "I;16": ("I;16B", b"\x10", b"\x00"),
    "I;16B": ("I;16B", b"\x10", b"\x00"),
    "P;1": ("P;1", b"\x01", b"\x03"),
    "P;2": ("P;2", b"\x02", b"\x03"),
    "P;4": ("P;4", b"\x04", b"\x03"),
    "P": ("P", b"\x08", b"\x03"),
    "RGB": ("RGB", b"\x08", b"\x02"),
    "RGBA": ("RGBA", b"\x08", b"\x06"),
}


def putchunk(fp: IO[bytes], cid: bytes, *data: bytes) -> None:
    """Write a PNG chunk (including CRC field)"""

    byte_data = b"".join(data)

    fp.write(o32(len(byte_data)) + cid)
    fp.write(byte_data)
    crc = _crc32(byte_data, _crc32(cid))
    fp.write(o32(crc))


class _idat:
    # wrap output from the encoder in IDAT chunks

    def __init__(self, fp: IO[bytes], chunk: Callable[..., None]) -> None:
        self.fp = fp
        self.chunk = chunk

    def write(self, data: bytes) -> None:
        self.chunk(self.fp, b"IDAT", data)


class _fdat:
    # wrap encoder output in fdAT chunks

    def __init__(self, fp: IO[bytes], chunk: Callable[..., None], seq_num: int) -> None:
        self.fp = fp
        self.chunk = chunk
        self.seq_num = seq_num

    def write(self, data: bytes) -> None:
        self.chunk(self.fp, b"fdAT", o32(self.seq_num), data)
        self.seq_num += 1


def _apply_encoderinfo(im: Image.Image, encoderinfo: dict[str, Any]) -> None:
    im.encoderconfig = (
        encoderinfo.get("optimize", False),
        encoderinfo.get("compress_level", -1),
        encoderinfo.get("compress_type", -1),
        encoderinfo.get("dictionary", b""),
    )


class _Frame(NamedTuple):
    im: Image.Image
    bbox: tuple[int, int, int, int] | None
    encoderinfo: dict[str, Any]


def _write_multiple_frames(
    im: Image.Image,
    fp: IO[bytes],
    chunk: Callable[..., None],
    mode: str,
    rawmode: str,
    default_image: Image.Image | None,
    append_images: list[Image.Image],
) -> Image.Image | None:
    duration = im.encoderinfo.get("duration")
    loop = im.encoderinfo.get("loop", im.info.get("loop", 0))
    disposal = im.encoderinfo.get("disposal", im.info.get("disposal", Disposal.OP_NONE))
    blend = im.encoderinfo.get("blend", im.info.get("blend", Blend.OP_SOURCE))

    if default_image:
        chain = itertools.chain(append_images)
    else:
        chain = itertools.chain([im], append_images)

    im_frames: list[_Frame] = []
    frame_count = 0
    for im_seq in chain:
        for im_frame in ImageSequence.Iterator(im_seq):
            if im_frame.mode == mode:
                im_frame = im_frame.copy()
            else:
                im_frame = im_frame.convert(mode)
            encoderinfo = im.encoderinfo.copy()
            if isinstance(duration, (list, tuple)):
                encoderinfo["duration"] = duration[frame_count]
            elif duration is None and "duration" in im_frame.info:
                encoderinfo["duration"] = im_frame.info["duration"]
            if isinstance(disposal, (list, tuple)):
                encoderinfo["disposal"] = disposal[frame_count]
            if isinstance(blend, (list, tuple)):
                encoderinfo["blend"] = blend[frame_count]
            frame_count += 1

            if im_frames:
                previous = im_frames[-1]
                prev_disposal = previous.encoderinfo.get("disposal")
                prev_blend = previous.encoderinfo.get("blend")
                if prev_disposal == Disposal.OP_PREVIOUS and len(im_frames) < 2:
                    prev_disposal = Disposal.OP_BACKGROUND

                if prev_disposal == Disposal.OP_BACKGROUND:
                    base_im = previous.im.copy()
                    dispose = Image.core.fill("RGBA", im.size, (0, 0, 0, 0))
                    bbox = previous.bbox
                    if bbox:
                        dispose = dispose.crop(bbox)
                    else:
                        bbox = (0, 0) + im.size
                    base_im.paste(dispose, bbox)
                elif prev_disposal == Disposal.OP_PREVIOUS:
                    base_im = im_frames[-2].im
                else:
                    base_im = previous.im
                delta = ImageChops.subtract_modulo(
                    im_frame.convert("RGBA"), base_im.convert("RGBA")
                )
                bbox = delta.getbbox(alpha_only=False)
                if (
                    not bbox
                    and prev_disposal == encoderinfo.get("disposal")
                    and prev_blend == encoderinfo.get("blend")
                    and "duration" in encoderinfo
                ):
                    previous.encoderinfo["duration"] += encoderinfo["duration"]
                    continue
            else:
                bbox = None
            im_frames.append(_Frame(im_frame, bbox, encoderinfo))

    if len(im_frames) == 1 and not default_image:
        return im_frames[0].im

    # animation control
    chunk(
        fp,
        b"acTL",
        o32(len(im_frames)),  # 0: num_frames
        o32(loop),  # 4: num_plays
    )

    # default image IDAT (if it exists)
    if default_image:
        default_im = im if im.mode == mode else im.convert(mode)
        _apply_encoderinfo(default_im, im.encoderinfo)
        ImageFile._save(
            default_im,
            cast(IO[bytes], _idat(fp, chunk)),
            [ImageFile._Tile("zip", (0, 0) + im.size, 0, rawmode)],
        )

    seq_num = 0
    for frame, frame_data in enumerate(im_frames):
        im_frame = frame_data.im
        if not frame_data.bbox:
            bbox = (0, 0) + im_frame.size
        else:
            bbox = frame_data.bbox
            im_frame = im_frame.crop(bbox)
        size = im_frame.size
        encoderinfo = frame_data.encoderinfo
        frame_duration = encoderinfo.get("duration", 0)
        delay = Fraction(frame_duration / 1000).limit_denominator(65535)
        if delay.numerator > 65535:
            msg = "cannot write duration"
            raise ValueError(msg)
        frame_disposal = encoderinfo.get("disposal", disposal)
        frame_blend = encoderinfo.get("blend", blend)
        # frame control
        chunk(
            fp,
            b"fcTL",
            o32(seq_num),  # sequence_number
            o32(size[0]),  # width
            o32(size[1]),  # height
            o32(bbox[0]),  # x_offset
            o32(bbox[1]),  # y_offset
            o16(delay.numerator),  # delay_numerator
            o16(delay.denominator),  # delay_denominator
            o8(frame_disposal),  # dispose_op
            o8(frame_blend),  # blend_op
        )
        seq_num += 1
        # frame data
        _apply_encoderinfo(im_frame, im.encoderinfo)
        if frame == 0 and not default_image:
            # first frame must be in IDAT chunks for backwards compatibility
            ImageFile._save(
                im_frame,
                cast(IO[bytes], _idat(fp, chunk)),
                [ImageFile._Tile("zip", (0, 0) + im_frame.size, 0, rawmode)],
            )
        else:
            fdat_chunks = _fdat(fp, chunk, seq_num)
            ImageFile._save(
                im_frame,
                cast(IO[bytes], fdat_chunks),
                [ImageFile._Tile("zip", (0, 0) + im_frame.size, 0, rawmode)],
            )
            seq_num = fdat_chunks.seq_num
    return None


def _save_all(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    _save(im, fp, filename, save_all=True)


def _save(
    im: Image.Image,
    fp: IO[bytes],
    filename: str | bytes,
    chunk: Callable[..., None] = putchunk,
    save_all: bool = False,
) -> None:
    # save an image to disk (called by the save method)

    if save_all:
        default_image = im.encoderinfo.get(
            "default_image", im.info.get("default_image")
        )
        modes = set()
        sizes = set()
        append_images = im.encoderinfo.get("append_images", [])
        for im_seq in itertools.chain([im], append_images):
            for im_frame in ImageSequence.Iterator(im_seq):
                modes.add(im_frame.mode)
                sizes.add(im_frame.size)
        for mode in ("RGBA", "RGB", "P"):
            if mode in modes:
                break
        else:
            mode = modes.pop()
        size = tuple(max(frame_size[i] for frame_size in sizes) for i in range(2))
    else:
        size = im.size
        mode = im.mode

    outmode = mode
    palette = []
    if im.palette:
        palette = im.getpalette() or []
    if mode == "P":
        #
        # attempt to minimize storage requirements for palette images
        if "bits" in im.encoderinfo:
            # number of bits specified by user
            colors = min(1 << im.encoderinfo["bits"], 256)
        else:
            # check palette contents
            if im.palette:
                colors = max(min(len(palette) // 3, 256), 1)
            else:
                colors = 256

        if colors <= 16:
            if colors <= 2:
                bits = 1
            elif colors <= 4:
                bits = 2
            else:
                bits = 4
            outmode += f";{bits}"

    # get the corresponding PNG mode
    try:
        rawmode, bit_depth, color_type = _OUTMODES[outmode]
    except KeyError as e:
        msg = f"cannot write mode {mode} as PNG"
        raise OSError(msg) from e
    if outmode == "I":
        deprecate("Saving I mode images as PNG", 13, stacklevel=4)

    #
    # write minimal PNG file

    fp.write(_MAGIC)

    chunk(
        fp,
        b"IHDR",
        o32(size[0]),  # 0: size
        o32(size[1]),
        bit_depth,
        color_type,
        b"\0",  # 10: compression
        b"\0",  # 11: filter category
        b"\0",  # 12: interlace flag
    )

    chunks = [b"cHRM", b"cICP", b"gAMA", b"sBIT", b"sRGB", b"tIME"]

    if icc := im.encoderinfo.get("icc_profile", im.info.get("icc_profile")):
        # ICC profile
        # according to PNG spec, the iCCP chunk contains:
        # Profile name  1-79 bytes (character string)
        # Null separator        1 byte (null character)
        # Compression method    1 byte (0)
        # Compressed profile    n bytes (zlib with deflate compression)
        name = b"ICC Profile"
        data = name + b"\0\0" + zlib.compress(icc)
        chunk(fp, b"iCCP", data)

        # You must either have sRGB or iCCP.
        # Disallow sRGB chunks when an iCCP-chunk has been emitted.
        chunks.remove(b"sRGB")

    if info := im.encoderinfo.get("pnginfo"):
        chunks_multiple_allowed = [b"sPLT", b"iTXt", b"tEXt", b"zTXt"]
        for info_chunk in info.chunks:
            cid, data = info_chunk[:2]
            if cid in chunks:
                chunks.remove(cid)
                chunk(fp, cid, data)
            elif cid in chunks_multiple_allowed:
                chunk(fp, cid, data)
            elif cid[1:2].islower():
                # Private chunk
                after_idat = len(info_chunk) == 3 and info_chunk[2]
                if not after_idat:
                    chunk(fp, cid, data)

    if im.mode == "P":
        palette_byte_number = colors * 3
        palette_bytes = bytes(palette[:palette_byte_number])
        while len(palette_bytes) < palette_byte_number:
            palette_bytes += b"\0"
        chunk(fp, b"PLTE", palette_bytes)

    transparency = im.encoderinfo.get("transparency", im.info.get("transparency"))

    if transparency is not None:
        if im.mode == "P":
            # limit to actual palette size
            alpha_bytes = colors
            if isinstance(transparency, bytes):
                chunk(fp, b"tRNS", transparency[:alpha_bytes])
            elif isinstance(transparency, int):
                transparency = max(0, min(255, transparency))
                alpha = b"\xff" * transparency + b"\0"
                chunk(fp, b"tRNS", alpha[:alpha_bytes])
            else:
                msg = "transparency for P must be an integer or bytes"
                raise ValueError(msg)
        elif im.mode in ("1", "L", "I", "I;16"):
            if isinstance(transparency, int):
                transparency = max(0, min(65535, transparency))
                chunk(fp, b"tRNS", o16(transparency))
            else:
                msg = f"transparency for {im.mode} must be an integer"
                raise ValueError(msg)
        elif im.mode == "RGB":
            if not isinstance(transparency, (list, tuple)):
                msg = "transparency for RGB must be list or tuple"
                raise ValueError(msg)
            elif len(transparency) != 3:
                msg = "transparency for RGB must have length 3"
                raise ValueError(msg)
            else:
                red, green, blue = transparency
                chunk(fp, b"tRNS", o16(red) + o16(green) + o16(blue))
        elif im.encoderinfo.get("transparency") is not None:
            # don't bother with transparency if it's an RGBA
            # and it's in the info dict. It's probably just stale.
            msg = "cannot use transparency for this mode"
            raise OSError(msg)
    elif im.mode == "P" and im.im.getpalettemode() == "RGBA":
        alpha = im.im.getpalette("RGBA", "A")
        alpha_bytes = colors
        chunk(fp, b"tRNS", alpha[:alpha_bytes])

    if dpi := im.encoderinfo.get("dpi"):
        chunk(
            fp,
            b"pHYs",
            o32(int(dpi[0] / 0.0254 + 0.5)),
            o32(int(dpi[1] / 0.0254 + 0.5)),
            b"\x01",
        )

    if info:
        chunks = [b"bKGD", b"hIST"]
        for info_chunk in info.chunks:
            cid, data = info_chunk[:2]
            if cid in chunks:
                chunks.remove(cid)
                chunk(fp, cid, data)

    if exif := im.encoderinfo.get("exif"):
        if isinstance(exif, Image.Exif):
            exif = exif.tobytes(8)
        if exif.startswith(b"Exif\x00\x00"):
            exif = exif[6:]
        chunk(fp, b"eXIf", exif)

    single_im: Image.Image | None = im
    if save_all:
        single_im = _write_multiple_frames(
            im, fp, chunk, mode, rawmode, default_image, append_images
        )
    if single_im:
        _apply_encoderinfo(single_im, im.encoderinfo)
        ImageFile._save(
            single_im,
            cast(IO[bytes], _idat(fp, chunk)),
            [ImageFile._Tile("zip", (0, 0) + single_im.size, 0, rawmode)],
        )

    if info:
        for info_chunk in info.chunks:
            cid, data = info_chunk[:2]
            if cid[1:2].islower():
                # Private chunk
                after_idat = len(info_chunk) == 3 and info_chunk[2]
                if after_idat:
                    chunk(fp, cid, data)

    chunk(fp, b"IEND", b"")

    if hasattr(fp, "flush"):
        fp.flush()


# --------------------------------------------------------------------
# PNG chunk converter


def getchunks(im: Image.Image, **params: Any) -> list[tuple[bytes, bytes, bytes]]:
    """Return a list of PNG chunks representing this image."""
    from io import BytesIO

    chunks = []

    def append(fp: IO[bytes], cid: bytes, *data: bytes) -> None:
        byte_data = b"".join(data)
        crc = o32(_crc32(byte_data, _crc32(cid)))
        chunks.append((cid, byte_data, crc))

    fp = BytesIO()

    try:
        im.encoderinfo = params
        _save(im, fp, "", append)
    finally:
        del im.encoderinfo

    return chunks


# --------------------------------------------------------------------
# Registry

Image.register_open(PngImageFile.format, PngImageFile, _accept)
Image.register_save(PngImageFile.format, _save)
Image.register_save_all(PngImageFile.format, _save_all)

Image.register_extensions(PngImageFile.format, [".png", ".apng"])

Image.register_mime(PngImageFile.format, "image/png")
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# Simple PostScript graphics interface
#
# History:
# 1996-04-20 fl   Created
# 1999-01-10 fl   Added gsave/grestore to image method
# 2005-05-04 fl   Fixed floating point issue in image (from Eric Etheridge)
#
# Copyright (c) 1997-2005 by Secret Labs AB.  All rights reserved.
# Copyright (c) 1996 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import sys
from typing import IO

from . import EpsImagePlugin

TYPE_CHECKING = False


##
# Simple PostScript graphics interface.


class PSDraw:
    """
    Sets up printing to the given file. If ``fp`` is omitted,
    ``sys.stdout.buffer`` is assumed.
    """

    def __init__(self, fp: IO[bytes] | None = None) -> None:
        if not fp:
            fp = sys.stdout.buffer
        self.fp = fp

    def begin_document(self, id: str | None = None) -> None:
        """Set up printing of a document. (Write PostScript DSC header.)"""
        # FIXME: incomplete
        self.fp.write(
            b"%!PS-Adobe-3.0\n"
            b"save\n"
            b"/showpage { } def\n"
            b"%%EndComments\n"
            b"%%BeginDocument\n"
        )
        # self.fp.write(ERROR_PS)  # debugging!
        self.fp.write(EDROFF_PS)
        self.fp.write(VDI_PS)
        self.fp.write(b"%%EndProlog\n")
        self.isofont: dict[bytes, int] = {}

    def end_document(self) -> None:
        """Ends printing. (Write PostScript DSC footer.)"""
        self.fp.write(b"%%EndDocument\nrestore showpage\n%%End\n")
        if hasattr(self.fp, "flush"):
            self.fp.flush()

    def setfont(self, font: str, size: int) -> None:
        """
        Selects which font to use.

        :param font: A PostScript font name
        :param size: Size in points.
        """
        font_bytes = bytes(font, "UTF-8")
        if font_bytes not in self.isofont:
            # reencode font
            self.fp.write(
                b"/PSDraw-%s ISOLatin1Encoding /%s E\n" % (font_bytes, font_bytes)
            )
            self.isofont[font_bytes] = 1
        # rough
        self.fp.write(b"/F0 %d /PSDraw-%s F\n" % (size, font_bytes))

    def line(self, xy0: tuple[int, int], xy1: tuple[int, int]) -> None:
        """
        Draws a line between the two points. Coordinates are given in
        PostScript point coordinates (72 points per inch, (0, 0) is the lower
        left corner of the page).
        """
        self.fp.write(b"%d %d %d %d Vl\n" % (*xy0, *xy1))

    def rectangle(self, box: tuple[int, int, int, int]) -> None:
        """
        Draws a rectangle.

        :param box: A tuple of four integers, specifying left, bottom, width and
           height.
        """
        self.fp.write(b"%d %d M 0 %d %d Vr\n" % box)

    def text(self, xy: tuple[int, int], text: str) -> None:
        """
        Draws text at the given position. You must use
        :py:meth:`~PIL.PSDraw.PSDraw.setfont` before calling this method.
        """
        # The font is loaded as ISOLatin1Encoding, so use latin-1 here.
        text_bytes = bytes(text, "latin-1")
        text_bytes = b"\\(".join(text_bytes.split(b"("))
        text_bytes = b"\\)".join(text_bytes.split(b")"))
        self.fp.write(b"%d %d M (%s) S\n" % (xy + (text_bytes,)))

    if TYPE_CHECKING:
        from . import Image

    def image(
        self, box: tuple[int, int, int, int], im: Image.Image, dpi: int | None = None
    ) -> None:
        """Draw a PIL image, centered in the given box."""
        # default resolution depends on mode
        if not dpi:
            if im.mode == "1":
                dpi = 200  # fax
            else:
                dpi = 100  # grayscale
        # image size (on paper)
        x = im.size[0] * 72 / dpi
        y = im.size[1] * 72 / dpi
        # max allowed size
        xmax = float(box[2] - box[0])
        ymax = float(box[3] - box[1])
        if x > xmax:
            y = y * xmax / x
            x = xmax
        if y > ymax:
            x = x * ymax / y
            y = ymax
        dx = (xmax - x) / 2 + box[0]
        dy = (ymax - y) / 2 + box[1]
        self.fp.write(b"gsave\n%f %f translate\n" % (dx, dy))
        if (x, y) != im.size:
            # EpsImagePlugin._save prints the image at (0,0,xsize,ysize)
            sx = x / im.size[0]
            sy = y / im.size[1]
            self.fp.write(b"%f %f scale\n" % (sx, sy))
        EpsImagePlugin._save(im, self.fp, "", 0)
        self.fp.write(b"\ngrestore\n")


# --------------------------------------------------------------------
# PostScript driver

#
# EDROFF.PS -- PostScript driver for Edroff 2
#
# History:
# 94-01-25 fl: created (edroff 2.04)
#
# Copyright (c) Fredrik Lundh 1994.
#


EDROFF_PS = b"""\
/S { show } bind def
/P { moveto show } bind def
/M { moveto } bind def
/X { 0 rmoveto } bind def
/Y { 0 exch rmoveto } bind def
/E {    findfont
        dup maxlength dict begin
        {
                1 index /FID ne { def } { pop pop } ifelse
        } forall
        /Encoding exch def
        dup /FontName exch def
        currentdict end definefont pop
} bind def
/F {    findfont exch scalefont dup setfont
        [ exch /setfont cvx ] cvx bind def
} bind def
"""

#
# VDI.PS -- PostScript driver for VDI meta commands
#
# History:
# 94-01-25 fl: created (edroff 2.04)
#
# Copyright (c) Fredrik Lundh 1994.
#

VDI_PS = b"""\
/Vm { moveto } bind def
/Va { newpath arcn stroke } bind def
/Vl { moveto lineto stroke } bind def
/Vc { newpath 0 360 arc closepath } bind def
/Vr {   exch dup 0 rlineto
        exch dup 0 exch rlineto
        exch neg 0 rlineto
        0 exch neg rlineto
        setgray fill } bind def
/Tm matrix def
/Ve {   Tm currentmatrix pop
        translate scale newpath 0 0 .5 0 360 arc closepath
        Tm setmatrix
} bind def
/Vf { currentgray exch setgray fill setgray } bind def
"""

#
# ERROR.PS -- Error handler
#
# History:
# 89-11-21 fl: created (pslist 1.10)
#

ERROR_PS = b"""\
/landscape false def
/errorBUF 200 string def
/errorNL { currentpoint 10 sub exch pop 72 exch moveto } def
errordict begin /handleerror {
    initmatrix /Courier findfont 10 scalefont setfont
    newpath 72 720 moveto $error begin /newerror false def
    (PostScript Error) show errorNL errorNL
    (Error: ) show
        /errorname load errorBUF cvs show errorNL errorNL
    (Command: ) show
        /command load dup type /stringtype ne { errorBUF cvs } if show
        errorNL errorNL
    (VMstatus: ) show
        vmstatus errorBUF cvs show ( bytes available, ) show
        errorBUF cvs show ( bytes used at level ) show
        errorBUF cvs show errorNL errorNL
    (Operand stargck: ) show errorNL /ostargck load {
        dup type /stringtype ne { errorBUF cvs } if 72 0 rmoveto show errorNL
    } forall errorNL
    (Execution stargck: ) show errorNL /estargck load {
        dup type /stringtype ne { errorBUF cvs } if 72 0 rmoveto show errorNL
    } forall
    end showpage
} def end
"""
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# Sun image file handling
#
# History:
# 1995-09-10 fl   Created
# 1996-05-28 fl   Fixed 32-bit alignment
# 1998-12-29 fl   Import ImagePalette module
# 2001-12-18 fl   Fixed palette loading (from Jean-Claude Rimbault)
#
# Copyright (c) 1997-2001 by Secret Labs AB
# Copyright (c) 1995-1996 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import Image, ImageFile, ImagePalette
from ._binary import i32be as i32


def _accept(prefix: bytes) -> bool:
    return len(prefix) >= 4 and i32(prefix) == 0x59A66A95


##
# Image plugin for Sun raster files.


class SunImageFile(ImageFile.ImageFile):
    format = "SUN"
    format_description = "Sun Raster File"

    def _open(self) -> None:
        # The Sun Raster file header is 32 bytes in length
        # and has the following format:

        #     typedef struct _SunRaster
        #     {
        #         DWORD MagicNumber;      /* Magic (identification) number */
        #         DWORD Width;            /* Width of image in pixels */
        #         DWORD Height;           /* Height of image in pixels */
        #         DWORD Depth;            /* Number of bits per pixel */
        #         DWORD Length;           /* Size of image data in bytes */
        #         DWORD Type;             /* Type of raster file */
        #         DWORD ColorMapType;     /* Type of color map */
        #         DWORD ColorMapLength;   /* Size of the color map in bytes */
        #     } SUNRASTER;

        assert self.fp is not None

        # HEAD
        s = self.fp.read(32)
        if not _accept(s):
            msg = "not an SUN raster file"
            raise SyntaxError(msg)

        offset = 32

        self._size = i32(s, 4), i32(s, 8)

        depth = i32(s, 12)
        # data_length = i32(s, 16)   # unreliable, ignore.
        file_type = i32(s, 20)
        palette_type = i32(s, 24)  # 0: None, 1: RGB, 2: Raw/arbitrary
        palette_length = i32(s, 28)

        if depth == 1:
            self._mode, rawmode = "1", "1;I"
        elif depth == 4:
            self._mode, rawmode = "L", "L;4"
        elif depth == 8:
            self._mode = rawmode = "L"
        elif depth == 24:
            if file_type == 3:
                self._mode, rawmode = "RGB", "RGB"
            else:
                self._mode, rawmode = "RGB", "BGR"
        elif depth == 32:
            if file_type == 3:
                self._mode, rawmode = "RGB", "RGBX"
            else:
                self._mode, rawmode = "RGB", "BGRX"
        else:
            msg = "Unsupported Mode/Bit Depth"
            raise SyntaxError(msg)

        if palette_length:
            if palette_length > 1024:
                msg = "Unsupported Color Palette Length"
                raise SyntaxError(msg)

            if palette_type != 1:
                msg = "Unsupported Palette Type"
                raise SyntaxError(msg)

            offset = offset + palette_length
            self.palette = ImagePalette.raw("RGB;L", self.fp.read(palette_length))
            if self.mode == "L":
                self._mode = "P"
                rawmode = rawmode.replace("L", "P")

        # 16 bit boundaries on stride
        stride = ((self.size[0] * depth + 15) // 16) * 2

        # file type: Type is the version (or flavor) of the bitmap
        # file. The following values are typically found in the Type
        # field:
        # 0000h Old
        # 0001h Standard
        # 0002h Byte-encoded
        # 0003h RGB format
        # 0004h TIFF format
        # 0005h IFF format
        # FFFFh Experimental

        # Old and standard are the same, except for the length tag.
        # byte-encoded is run-length-encoded
        # RGB looks similar to standard, but RGB byte order
        # TIFF and IFF mean that they were converted from T/IFF
        # Experimental means that it's something else.
        # (https://www.fileformat.info/format/sunraster/egff.htm)

        if file_type in (0, 1, 3, 4, 5):
            self.tile = [
                ImageFile._Tile("raw", (0, 0) + self.size, offset, (rawmode, stride))
            ]
        elif file_type == 2:
            self.tile = [
                ImageFile._Tile("sun_rle", (0, 0) + self.size, offset, rawmode)
            ]
        else:
            msg = "Unsupported Sun Raster file type"
            raise SyntaxError(msg)


#
# registry


Image.register_open(SunImageFile.format, SunImageFile, _accept)

Image.register_extension(SunImageFile.format, ".ras")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# Windows Icon support for PIL
#
# History:
#       96-05-27 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#

# This plugin is a refactored version of Win32IconImagePlugin by Bryan Davis
# <casadebender@gmail.com>.
# https://code.google.com/archive/p/casadebender/wikis/Win32IconImagePlugin.wiki
#
# Copyright 2008 Bryan Davis
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Icon format references:
#   * https://en.wikipedia.org/wiki/ICO_(file_format)
#   * https://msdn.microsoft.com/en-us/library/ms997538.aspx
from __future__ import annotations

import warnings
from io import BytesIO
from math import ceil, log
from typing import IO, NamedTuple

from . import BmpImagePlugin, Image, ImageFile, PngImagePlugin
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32

#
# --------------------------------------------------------------------

_MAGIC = b"\0\0\1\0"


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    fp.write(_MAGIC)  # (2+2)
    bmp = im.encoderinfo.get("bitmap_format") == "bmp"
    sizes = im.encoderinfo.get(
        "sizes",
        [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    frames = []
    provided_ims = [im] + im.encoderinfo.get("append_images", [])
    width, height = im.size
    for size in sorted(set(sizes)):
        if size[0] > width or size[1] > height or size[0] > 256 or size[1] > 256:
            continue

        for provided_im in provided_ims:
            if provided_im.size != size:
                continue
            frames.append(provided_im)
            if bmp:
                bits = BmpImagePlugin.SAVE[provided_im.mode][1]
                bits_used = [bits]
                for other_im in provided_ims:
                    if other_im.size != size:
                        continue
                    bits = BmpImagePlugin.SAVE[other_im.mode][1]
                    if bits not in bits_used:
                        # Another image has been supplied for this size
                        # with a different bit depth
                        frames.append(other_im)
                        bits_used.append(bits)
            break
        else:
            # TODO: invent a more convenient method for proportional scalings
            frame = provided_im.copy()
            frame.thumbnail(size, Image.Resampling.LANCZOS, reducing_gap=None)
            frames.append(frame)
    fp.write(o16(len(frames)))  # idCount(2)
    offset = fp.tell() + len(frames) * 16
    for frame in frames:
        width, height = frame.size
        # 0 means 256
        fp.write(o8(width if width < 256 else 0))  # bWidth(1)
        fp.write(o8(height if height < 256 else 0))  # bHeight(1)

        bits, colors = BmpImagePlugin.SAVE[frame.mode][1:] if bmp else (32, 0)
        fp.write(o8(colors))  # bColorCount(1)
        fp.write(b"\0")  # bReserved(1)
        fp.write(b"\0\0")  # wPlanes(2)
        fp.write(o16(bits))  # wBitCount(2)

        image_io = BytesIO()
        if bmp:
            frame.save(image_io, "dib")

            if bits != 32:
                and_mask = Image.new("1", size)
                ImageFile._save(
                    and_mask,
                    image_io,
                    [ImageFile._Tile("raw", (0, 0) + size, 0, ("1", 0, -1))],
                )
        else:
            frame.save(image_io, "png")
        image_io.seek(0)
        image_bytes = image_io.read()
        if bmp:
            image_bytes = image_bytes[:8] + o32(height * 2) + image_bytes[12:]
        bytes_len = len(image_bytes)
        fp.write(o32(bytes_len))  # dwBytesInRes(4)
        fp.write(o32(offset))  # dwImageOffset(4)
        current = fp.tell()
        fp.seek(offset)
        fp.write(image_bytes)
        offset = offset + bytes_len
        fp.seek(current)


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(_MAGIC)


class IconHeader(NamedTuple):
    width: int
    height: int
    nb_color: int
    reserved: int
    planes: int
    bpp: int
    size: int
    offset: int
    dim: tuple[int, int]
    square: int
    color_depth: int


class IcoFile:
    def __init__(self, buf: IO[bytes]) -> None:
        """
        Parse image from file-like object containing ico file data
        """

        # check magic
        s = buf.read(6)
        if not _accept(s):
            msg = "not an ICO file"
            raise SyntaxError(msg)

        self.buf = buf
        self.entry = []

        # Number of items in file
        self.nb_items = i16(s, 4)

        # Get headers for each item
        for i in range(self.nb_items):
            s = buf.read(16)

            # See Wikipedia
            width = s[0] or 256
            height = s[1] or 256

            # No. of colors in image (0 if >=8bpp)
            nb_color = s[2]
            bpp = i16(s, 6)
            icon_header = IconHeader(
                width=width,
                height=height,
                nb_color=nb_color,
                reserved=s[3],
                planes=i16(s, 4),
                bpp=i16(s, 6),
                size=i32(s, 8),
                offset=i32(s, 12),
                dim=(width, height),
                square=width * height,
                # See Wikipedia notes about color depth.
                # We need this just to differ images with equal sizes
                color_depth=bpp or (nb_color != 0 and ceil(log(nb_color, 2))) or 256,
            )

            self.entry.append(icon_header)

        self.entry = sorted(self.entry, key=lambda x: x.color_depth)
        # ICO images are usually squares
        self.entry = sorted(self.entry, key=lambda x: x.square, reverse=True)

    def sizes(self) -> set[tuple[int, int]]:
        """
        Get a set of all available icon sizes and color depths.
        """
        return {(h.width, h.height) for h in self.entry}

    def getentryindex(self, size: tuple[int, int], bpp: int | bool = False) -> int:
        for i, h in enumerate(self.entry):
            if size == h.dim and (bpp is False or bpp == h.color_depth):
                return i
        return 0

    def getimage(self, size: tuple[int, int], bpp: int | bool = False) -> Image.Image:
        """
        Get an image from the icon
        """
        return self.frame(self.getentryindex(size, bpp))

    def frame(self, idx: int) -> Image.Image:
        """
        Get an image from frame idx
        """

        header = self.entry[idx]

        self.buf.seek(header.offset)
        data = self.buf.read(8)
        self.buf.seek(header.offset)

        im: Image.Image
        if data[:8] == PngImagePlugin._MAGIC:
            # png frame
            im = PngImagePlugin.PngImageFile(self.buf)
            Image._decompression_bomb_check(im.size)
        else:
            # XOR + AND mask bmp frame
            im = BmpImagePlugin.DibImageFile(self.buf)
            Image._decompression_bomb_check(im.size)

            # change tile dimension to only encompass XOR image
            im._size = (im.size[0], int(im.size[1] / 2))
            d, e, o, a = im.tile[0]
            im.tile[0] = ImageFile._Tile(d, (0, 0) + im.size, o, a)

            # figure out where AND mask image starts
            if header.bpp == 32:
                # 32-bit color depth icon image allows semitransparent areas
                # PIL's DIB format ignores transparency bits, recover them.
                # The DIB is packed in BGRX byte order where X is the alpha
                # channel.

                # Back up to start of bmp data
                self.buf.seek(o)
                # extract every 4th byte (eg. 3,7,11,15,...)
                alpha_bytes = self.buf.read(im.size[0] * im.size[1] * 4)[3::4]

                # convert to an 8bpp grayscale image
                try:
                    mask = Image.frombuffer(
                        "L",  # 8bpp
                        im.size,  # (w, h)
                        alpha_bytes,  # source chars
                        "raw",  # raw decoder
                        ("L", 0, -1),  # 8bpp inverted, unpadded, reversed
                    )
                except ValueError:
                    if ImageFile.LOAD_TRUNCATED_IMAGES:
                        mask = None
                    else:
                        raise
            else:
                # get AND image from end of bitmap
                w = im.size[0]
                if (w % 32) > 0:
                    # bitmap row data is aligned to word boundaries
                    w += 32 - (im.size[0] % 32)

                # the total mask data is
                # padded row size * height / bits per char

                total_bytes = int((w * im.size[1]) / 8)
                and_mask_offset = header.offset + header.size - total_bytes

                self.buf.seek(and_mask_offset)
                mask_data = self.buf.read(total_bytes)

                # convert raw data to image
                try:
                    mask = Image.frombuffer(
                        "1",  # 1 bpp
                        im.size,  # (w, h)
                        mask_data,  # source chars
                        "raw",  # raw decoder
                        ("1;I", int(w / 8), -1),  # 1bpp inverted, padded, reversed
                    )
                except ValueError:
                    if ImageFile.LOAD_TRUNCATED_IMAGES:
                        mask = None
                    else:
                        raise

                # now we have two images, im is XOR image and mask is AND image

            # apply mask image as alpha channel
            if mask:
                im = im.convert("RGBA")
                im.putalpha(mask)

        return im


##
# Image plugin for Windows Icon files.


class IcoImageFile(ImageFile.ImageFile):
    """
    PIL read-only image support for Microsoft Windows .ico files.

    By default the largest resolution image in the file will be loaded. This
    can be changed by altering the 'size' attribute before calling 'load'.

    The info dictionary has a key 'sizes' that is a list of the sizes available
    in the icon file.

    Handles classic, XP and Vista icon formats.

    When saving, PNG compression is used. Support for this was only added in
    Windows Vista. If you are unable to view the icon in Windows, convert the
    image to "RGBA" mode before saving.

    This plugin is a refactored version of Win32IconImagePlugin by Bryan Davis
    <casadebender@gmail.com>.
    https://code.google.com/archive/p/casadebender/wikis/Win32IconImagePlugin.wiki
    """

    format = "ICO"
    format_description = "Windows Icon"

    def _open(self) -> None:
        assert self.fp is not None
        self.ico = IcoFile(self.fp)
        self.info["sizes"] = self.ico.sizes()
        self.size = self.ico.entry[0].dim
        self.load()

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @size.setter
    def size(self, value: tuple[int, int]) -> None:
        if value not in self.info["sizes"]:
            msg = "This is not one of the allowed sizes of this image"
            raise ValueError(msg)
        self._size = value

    def load(self) -> Image.core.PixelAccess | None:
        if self._im is not None and self.im.size == self.size:
            # Already loaded
            return Image.Image.load(self)
        im = self.ico.getimage(self.size)
        # if tile is PNG, it won't really be loaded yet
        im.load()
        self.im = im.im
        self._mode = im.mode
        if im.palette:
            self.palette = im.palette
        if im.size != self.size:
            warnings.warn("Image was not the expected size")

            index = self.ico.getentryindex(self.size)
            sizes = list(self.info["sizes"])
            sizes[index] = im.size
            self.info["sizes"] = set(sizes)

            self.size = im.size
        return Image.Image.load(self)

    def load_seek(self, pos: int) -> None:
        # Flag the ImageFile.Parser so that it
        # just does all the decode at the end.
        pass


#
# --------------------------------------------------------------------


Image.register_open(IcoImageFile.format, IcoImageFile, _accept)
Image.register_save(IcoImageFile.format, _save)
Image.register_extension(IcoImageFile.format, ".ico")

Image.register_mime(IcoImageFile.format, "image/x-icon")
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# base class for raster font file parsers
#
# history:
# 1997-06-05 fl   created
# 1997-08-19 fl   restrict image width
#
# Copyright (c) 1997-1998 by Secret Labs AB
# Copyright (c) 1997-1998 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import os
from typing import BinaryIO

from . import Image, ImageFont, _binary

WIDTH = 800


def puti16(
    fp: BinaryIO, values: tuple[int, int, int, int, int, int, int, int, int, int]
) -> None:
    """Write network order (big-endian) 16-bit sequence"""
    for v in values:
        if v < 0:
            v += 65536
        fp.write(_binary.o16be(v))


class FontFile:
    """Base class for raster font file handlers."""

    bitmap: Image.Image | None = None

    def __init__(self) -> None:
        self.info: dict[bytes, bytes | int] = {}
        self.glyph: list[
            tuple[
                tuple[int, int],
                tuple[int, int, int, int],
                tuple[int, int, int, int],
                Image.Image,
            ]
            | None
        ] = [None] * 256

    def __getitem__(self, ix: int) -> (
        tuple[
            tuple[int, int],
            tuple[int, int, int, int],
            tuple[int, int, int, int],
            Image.Image,
        ]
        | None
    ):
        return self.glyph[ix]

    def compile(self) -> None:
        """Create metrics and bitmap"""

        if self.bitmap:
            return

        # create bitmap large enough to hold all data
        h = w = maxwidth = 0
        lines = 1
        for glyph in self.glyph:
            if glyph:
                d, dst, src, im = glyph
                h = max(h, src[3] - src[1])
                w += src[2] - src[0]
                if w > WIDTH:
                    lines += 1
                    w = src[2] - src[0]
                maxwidth = max(maxwidth, w)

        xsize = maxwidth
        ysize = lines * h

        if xsize == 0 and ysize == 0:
            return

        self.ysize = h

        # paste glyphs into bitmap
        Image._decompression_bomb_check((xsize, ysize))
        self.bitmap = Image.new("1", (xsize, ysize))
        self.metrics: list[
            tuple[tuple[int, int], tuple[int, int, int, int], tuple[int, int, int, int]]
            | None
        ] = [None] * 256
        x = y = 0
        for i in range(256):
            glyph = self[i]
            if glyph:
                d, dst, src, im = glyph
                xx = src[2] - src[0]
                x0, y0 = x, y
                x = x + xx
                if x > WIDTH:
                    x, y = 0, y + h
                    x0, y0 = x, y
                    x = xx
                s = src[0] + x0, src[1] + y0, src[2] + x0, src[3] + y0
                self.bitmap.paste(im.crop(src), s)
                self.metrics[i] = d, dst, s

    def _encode_metrics(self) -> bytes:
        values: list[int] = []
        for i in range(256):
            m = self.metrics[i]
            if m:
                values.extend(m[0] + m[1] + m[2])
            else:
                values.extend((0,) * 10)

        data = bytearray()
        for v in values:
            if v < 0:
                v += 65536
            data += _binary.o16be(v)
        return bytes(data)

    def save(self, filename: str) -> None:
        """Save font"""

        self.compile()

        # font data
        if not self.bitmap:
            msg = "No bitmap created"
            raise ValueError(msg)
        self.bitmap.save(os.path.splitext(filename)[0] + ".pbm", "PNG")

        # font metrics
        with open(os.path.splitext(filename)[0] + ".pil", "wb") as fp:
            fp.write(b"PILfont\n")
            fp.write(f";;;;;;{self.ysize};\n".encode("ascii"))  # HACK!!!
            fp.write(b"DATA\n")
            fp.write(self._encode_metrics())

    def to_imagefont(self) -> ImageFont.ImageFont:
        """Convert to ImageFont"""

        self.compile()

        # font data
        if not self.bitmap:
            msg = "No bitmap created"
            raise ValueError(msg)

        imagefont = ImageFont.ImageFont()
        imagefont._load(self.bitmap, self._encode_metrics())
        return imagefont
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# a simple math add-on for the Python Imaging Library
#
# History:
# 1999-02-15 fl   Original PIL Plus release
# 2005-05-05 fl   Simplified and cleaned up for PIL 1.1.6
# 2005-09-12 fl   Fixed int() and float() for Python 2.4.1
#
# Copyright (c) 1999-2005 by Secret Labs AB
# Copyright (c) 2005 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import builtins

from . import Image, _imagingmath

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from types import CodeType
    from typing import Any


class _Operand:
    """Wraps an image operand, providing standard operators"""

    def __init__(self, im: Image.Image):
        self.im = im

    def __fixup(self, im1: _Operand | float) -> Image.Image:
        # convert image to suitable mode
        if isinstance(im1, _Operand):
            # argument was an image.
            if im1.im.mode in ("1", "L"):
                return im1.im.convert("I")
            elif im1.im.mode in ("I", "F"):
                return im1.im
            else:
                msg = f"unsupported mode: {im1.im.mode}"
                raise ValueError(msg)
        else:
            # argument was a constant
            if isinstance(im1, (int, float)) and self.im.mode in ("1", "L", "I"):
                return Image.new("I", self.im.size, im1)
            else:
                return Image.new("F", self.im.size, im1)

    def apply(
        self,
        op: str,
        im1: _Operand | float,
        im2: _Operand | float | None = None,
        mode: str | None = None,
    ) -> _Operand:
        im_1 = self.__fixup(im1)
        if im2 is None:
            # unary operation
            out = Image.new(mode or im_1.mode, im_1.size, None)
            try:
                op = getattr(_imagingmath, f"{op}_{im_1.mode}")
            except AttributeError as e:
                msg = f"bad operand type for '{op}'"
                raise TypeError(msg) from e
            _imagingmath.unop(op, out.getim(), im_1.getim())
        else:
            # binary operation
            im_2 = self.__fixup(im2)
            if im_1.mode != im_2.mode:
                # convert both arguments to floating point
                if im_1.mode != "F":
                    im_1 = im_1.convert("F")
                if im_2.mode != "F":
                    im_2 = im_2.convert("F")
            if im_1.size != im_2.size:
                # crop both arguments to a common size
                size = (
                    min(im_1.size[0], im_2.size[0]),
                    min(im_1.size[1], im_2.size[1]),
                )
                if im_1.size != size:
                    im_1 = im_1.crop((0, 0) + size)
                if im_2.size != size:
                    im_2 = im_2.crop((0, 0) + size)
            out = Image.new(mode or im_1.mode, im_1.size, None)
            try:
                op = getattr(_imagingmath, f"{op}_{im_1.mode}")
            except AttributeError as e:
                msg = f"bad operand type for '{op}'"
                raise TypeError(msg) from e
            _imagingmath.binop(op, out.getim(), im_1.getim(), im_2.getim())
        return _Operand(out)

    # unary operators
    def __bool__(self) -> bool:
        # an image is "true" if it contains at least one non-zero pixel
        return self.im.getbbox() is not None

    def __abs__(self) -> _Operand:
        return self.apply("abs", self)

    def __pos__(self) -> _Operand:
        return self

    def __neg__(self) -> _Operand:
        return self.apply("neg", self)

    # binary operators
    def __add__(self, other: _Operand | float) -> _Operand:
        return self.apply("add", self, other)

    def __radd__(self, other: _Operand | float) -> _Operand:
        return self.apply("add", other, self)

    def __sub__(self, other: _Operand | float) -> _Operand:
        return self.apply("sub", self, other)

    def __rsub__(self, other: _Operand | float) -> _Operand:
        return self.apply("sub", other, self)

    def __mul__(self, other: _Operand | float) -> _Operand:
        return self.apply("mul", self, other)

    def __rmul__(self, other: _Operand | float) -> _Operand:
        return self.apply("mul", other, self)

    def __truediv__(self, other: _Operand | float) -> _Operand:
        return self.apply("div", self, other)

    def __rtruediv__(self, other: _Operand | float) -> _Operand:
        return self.apply("div", other, self)

    def __mod__(self, other: _Operand | float) -> _Operand:
        return self.apply("mod", self, other)

    def __rmod__(self, other: _Operand | float) -> _Operand:
        return self.apply("mod", other, self)

    def __pow__(self, other: _Operand | float) -> _Operand:
        return self.apply("pow", self, other)

    def __rpow__(self, other: _Operand | float) -> _Operand:
        return self.apply("pow", other, self)

    # bitwise
    def __invert__(self) -> _Operand:
        return self.apply("invert", self)

    def __and__(self, other: _Operand | float) -> _Operand:
        return self.apply("and", self, other)

    def __rand__(self, other: _Operand | float) -> _Operand:
        return self.apply("and", other, self)

    def __or__(self, other: _Operand | float) -> _Operand:
        return self.apply("or", self, other)

    def __ror__(self, other: _Operand | float) -> _Operand:
        return self.apply("or", other, self)

    def __xor__(self, other: _Operand | float) -> _Operand:
        return self.apply("xor", self, other)

    def __rxor__(self, other: _Operand | float) -> _Operand:
        return self.apply("xor", other, self)

    def __lshift__(self, other: _Operand | float) -> _Operand:
        return self.apply("lshift", self, other)

    def __rshift__(self, other: _Operand | float) -> _Operand:
        return self.apply("rshift", self, other)

    # logical
    def __eq__(self, other: _Operand | float) -> _Operand:  # type: ignore[override]
        return self.apply("eq", self, other)

    def __ne__(self, other: _Operand | float) -> _Operand:  # type: ignore[override]
        return self.apply("ne", self, other)

    def __lt__(self, other: _Operand | float) -> _Operand:
        return self.apply("lt", self, other)

    def __le__(self, other: _Operand | float) -> _Operand:
        return self.apply("le", self, other)

    def __gt__(self, other: _Operand | float) -> _Operand:
        return self.apply("gt", self, other)

    def __ge__(self, other: _Operand | float) -> _Operand:
        return self.apply("ge", self, other)


# conversions
def imagemath_int(self: _Operand) -> _Operand:
    return _Operand(self.im.convert("I"))


def imagemath_float(self: _Operand) -> _Operand:
    return _Operand(self.im.convert("F"))


# logical
def imagemath_equal(self: _Operand, other: _Operand | float | None) -> _Operand:
    return self.apply("eq", self, other, mode="I")


def imagemath_notequal(self: _Operand, other: _Operand | float | None) -> _Operand:
    return self.apply("ne", self, other, mode="I")


def imagemath_min(self: _Operand, other: _Operand | float | None) -> _Operand:
    return self.apply("min", self, other)


def imagemath_max(self: _Operand, other: _Operand | float | None) -> _Operand:
    return self.apply("max", self, other)


def imagemath_convert(self: _Operand, mode: str) -> _Operand:
    return _Operand(self.im.convert(mode))


ops = {
    "int": imagemath_int,
    "float": imagemath_float,
    "equal": imagemath_equal,
    "notequal": imagemath_notequal,
    "min": imagemath_min,
    "max": imagemath_max,
    "convert": imagemath_convert,
}


def lambda_eval(expression: Callable[[dict[str, Any]], Any], **kw: Any) -> Any:
    """
    Returns the result of an image function.

    :py:mod:`~PIL.ImageMath` only supports single-layer images. To process multi-band
    images, use the :py:meth:`~PIL.Image.Image.split` method or
    :py:func:`~PIL.Image.merge` function.

    :param expression: A function that receives a dictionary.
    :param **kw: Values to add to the function's dictionary.
    :return: The expression result. This is usually an image object, but can
             also be an integer, a floating point value, or a pixel tuple,
             depending on the expression.
    """

    args: dict[str, Any] = ops.copy()
    args.update(kw)
    for k, v in args.items():
        if isinstance(v, Image.Image):
            args[k] = _Operand(v)

    out = expression(args)
    try:
        return out.im
    except AttributeError:
        return out


def unsafe_eval(expression: str, **kw: Any) -> Any:
    """
    Evaluates an image expression. This uses Python's ``eval()`` function to process
    the expression string, and carries the security risks of doing so. It is not
    recommended to process expressions without considering this.
    :py:meth:`~lambda_eval` is a more secure alternative.

    :py:mod:`~PIL.ImageMath` only supports single-layer images. To process multi-band
    images, use the :py:meth:`~PIL.Image.Image.split` method or
    :py:func:`~PIL.Image.merge` function.

    :param expression: A string containing a Python-style expression.
    :param **kw: Values to add to the evaluation context.
    :return: The evaluated expression. This is usually an image object, but can
             also be an integer, a floating point value, or a pixel tuple,
             depending on the expression.
    """

    # build execution namespace
    args: dict[str, Any] = ops.copy()
    for k in kw:
        if "__" in k or hasattr(builtins, k):
            msg = f"'{k}' not allowed"
            raise ValueError(msg)

    args.update(kw)
    for k, v in args.items():
        if isinstance(v, Image.Image):
            args[k] = _Operand(v)

    compiled_code = compile(expression, "<string>", "eval")

    def scan(code: CodeType) -> None:
        for const in code.co_consts:
            if type(const) is type(compiled_code):
                scan(const)

        for name in code.co_names:
            if name not in args and name != "abs":
                msg = f"'{name}' not allowed"
                raise ValueError(msg)

    scan(compiled_code)
    out = builtins.eval(expression, None, args)
    try:
        return out.im
    except AttributeError:
        return out
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# JPEG2000 file handling
#
# History:
# 2014-03-12 ajh  Created
# 2021-06-30 rogermb  Extract dpi information from the 'resc' header box
#
# Copyright (c) 2014 Coriolis Systems Limited
# Copyright (c) 2014 Alastair Houghton
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io
import os
import struct
from typing import cast

from . import Image, ImageFile, ImagePalette, _binary

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO


class BoxReader:
    """
    A small helper class to read fields stored in JPEG2000 header boxes
    and to easily step into and read sub-boxes.
    """

    def __init__(self, fp: IO[bytes], length: int = -1) -> None:
        self.fp = fp
        self.has_length = length >= 0
        self.length = length
        self.remaining_in_box = -1

    def _can_read(self, num_bytes: int) -> bool:
        if self.has_length and self.fp.tell() + num_bytes > self.length:
            # Outside box: ensure we don't read past the known file length
            return False
        if self.remaining_in_box >= 0:
            # Inside box contents: ensure read does not go past box boundaries
            return num_bytes <= self.remaining_in_box
        else:
            return True  # No length known, just read

    def _read_bytes(self, num_bytes: int) -> bytes:
        if not self._can_read(num_bytes):
            msg = "Not enough data in header"
            raise SyntaxError(msg)

        data = self.fp.read(num_bytes)
        if len(data) < num_bytes:
            msg = f"Expected to read {num_bytes} bytes but only got {len(data)}."
            raise OSError(msg)

        if self.remaining_in_box > 0:
            self.remaining_in_box -= num_bytes
        return data

    def read_fields(self, field_format: str) -> tuple[int | bytes, ...]:
        size = struct.calcsize(field_format)
        data = self._read_bytes(size)
        return struct.unpack(field_format, data)

    def read_boxes(self) -> BoxReader:
        size = self.remaining_in_box
        data = self._read_bytes(size)
        return BoxReader(io.BytesIO(data), size)

    def has_next_box(self) -> bool:
        if self.has_length:
            return self.fp.tell() + self.remaining_in_box < self.length
        else:
            return True

    def next_box_type(self) -> bytes:
        # Skip the rest of the box if it has not been read
        if self.remaining_in_box > 0:
            self.fp.seek(self.remaining_in_box, os.SEEK_CUR)
        self.remaining_in_box = -1

        # Read the length and type of the next box
        lbox, tbox = cast(tuple[int, bytes], self.read_fields(">I4s"))
        if lbox == 1:
            lbox = cast(int, self.read_fields(">Q")[0])
            hlen = 16
        else:
            hlen = 8

        if lbox < hlen or not self._can_read(lbox - hlen):
            msg = "Invalid header length"
            raise SyntaxError(msg)

        self.remaining_in_box = lbox - hlen
        return tbox


