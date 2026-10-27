# Copyright (c) 2001-2002 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile

from . import Image

TYPE_CHECKING = False
if TYPE_CHECKING:
    from . import ImageWin


def grab(
    bbox: tuple[int, int, int, int] | None = None,
    include_layered_windows: bool = False,
    all_screens: bool = False,
    xdisplay: str | None = None,
    window: int | ImageWin.HWND | None = None,
    *,
    scale_down: bool = False,
) -> Image.Image:
    im: Image.Image
    if xdisplay is None:
        if sys.platform == "darwin":
            fh, filepath = tempfile.mkstemp(".png")
            os.close(fh)
            args = ["screencapture"]
            if window is not None:
                args += ["-l", str(window)]
            elif bbox:
                left, top, right, bottom = bbox
                args += ["-R", f"{left},{top},{right-left},{bottom-top}"]
            args += ["-x", filepath]
            retcode = subprocess.call(args)
            if retcode:
                raise subprocess.CalledProcessError(retcode, args)
            im = Image.open(filepath)
            im.load()
            os.unlink(filepath)
            if bbox:
                if window is not None:
                    # Determine if the window was in Retina mode or not
                    # by capturing it without the shadow,
                    # and checking how different the width is
                    fh, filepath = tempfile.mkstemp(".png")
                    os.close(fh)
                    args = ["screencapture", "-l", str(window), "-o", "-x", filepath]
                    retcode = subprocess.call(args)
                    if retcode:
                        raise subprocess.CalledProcessError(retcode, args)
                    with Image.open(filepath) as im_no_shadow:
                        retina = im.width - im_no_shadow.width > 100
                    os.unlink(filepath)

                    # Since screencapture's -R does not work with -l,
                    # crop the image manually
                    if retina:
                        left, top, right, bottom = bbox
                        scale = 1 if scale_down else 2
                        im_cropped = im.resize(
                            ((right - left) * scale, (bottom - top) * scale),
                            box=tuple(coord * 2 for coord in bbox),
                        )
                    else:
                        im_cropped = im.crop(bbox)
                    im.close()
                    return im_cropped
                elif scale_down:
                    im_resized = im.resize((right - left, bottom - top))
                    im.close()
                    return im_resized
            return im
        elif sys.platform == "win32":
            if window is not None:
                all_screens = -1
            offset, size, data = Image.core.grabscreen_win32(
                include_layered_windows,
                all_screens,
                int(window) if window is not None else 0,
            )
            im = Image.frombytes(
                "RGB",
                size,
                data,
                # RGB, 32-bit line padding, origin lower left corner
                "raw",
                "BGR",
                (size[0] * 3 + 3) & -4,
                -1,
            )
            if bbox:
                x0, y0 = offset
                left, top, right, bottom = bbox
                im = im.crop((left - x0, top - y0, right - x0, bottom - y0))
            return im
    # Cast to Optional[str] needed for Windows and macOS.
    display_name: str | None = xdisplay
    try:
        if not Image.core.HAVE_XCB:
            msg = "Pillow was built without XCB support"
            raise OSError(msg)
        size, data = Image.core.grabscreen_x11(display_name)
    except OSError:
        if display_name is None and sys.platform not in ("darwin", "win32"):
            if shutil.which("gnome-screenshot"):
                args = ["gnome-screenshot", "-f"]
            elif shutil.which("grim"):
                args = ["grim"]
            elif shutil.which("spectacle"):
                args = ["spectacle", "-n", "-b", "-f", "-o"]
            else:
                raise
            fh, filepath = tempfile.mkstemp(".png")
            os.close(fh)
            args.append(filepath)
            retcode = subprocess.call(args)
            if retcode:
                raise subprocess.CalledProcessError(retcode, args)
            im = Image.open(filepath)
            im.load()
            os.unlink(filepath)
            if bbox:
                im_cropped = im.crop(bbox)
                im.close()
                return im_cropped
            return im
        else:
            raise
    else:
        im = Image.frombytes("RGB", size, data, "raw", "BGRX", size[0] * 4, 1)
        if bbox:
            im = im.crop(bbox)
        return im


def grabclipboard() -> Image.Image | list[str] | None:
    if sys.platform == "darwin":
        p = subprocess.run(
            ["osascript", "-e", "get the clipboard as «class PNGf»"],
            capture_output=True,
        )
        if p.returncode != 0:
            return None

        import binascii

        data = io.BytesIO(binascii.unhexlify(p.stdout[11:-3]))
        return Image.open(data)
    elif sys.platform == "win32":
        fmt, data = Image.core.grabclipboard_win32()
        if fmt == "file":  # CF_HDROP
            import struct

            o = struct.unpack_from("I", data)[0]
            if data[16] == 0:
                files = data[o:].decode("mbcs").split("\0")
            else:
                files = data[o:].decode("utf-16le").split("\0")
            return files[: files.index("")]
        if isinstance(data, bytes):
            data = io.BytesIO(data)
            if fmt == "png":
                from . import PngImagePlugin

                return PngImagePlugin.PngImageFile(data)
            elif fmt == "DIB":
                from . import BmpImagePlugin

                return BmpImagePlugin.DibImageFile(data)
        return None
    else:
        if os.getenv("WAYLAND_DISPLAY"):
            session_type = "wayland"
        elif os.getenv("DISPLAY"):
            session_type = "x11"
        else:  # Session type check failed
            session_type = None

        if shutil.which("wl-paste") and session_type in ("wayland", None):
            args = ["wl-paste", "-t", "image"]
        elif shutil.which("xclip") and session_type in ("x11", None):
            args = ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]
        else:
            msg = "wl-paste or xclip is required for ImageGrab.grabclipboard() on Linux"
            raise NotImplementedError(msg)

        p = subprocess.run(args, capture_output=True)
        if p.returncode != 0:
            err = p.stderr
            for silent_error in [
                # wl-paste, when the clipboard is empty
                b"Nothing is copied",
                # Ubuntu/Debian wl-paste, when the clipboard is empty
                b"No selection",
                # Ubuntu/Debian wl-paste, when an image isn't available
                b"No suitable type of content copied",
                # wl-paste or Ubuntu/Debian xclip, when an image isn't available
                b" not available",
                # xclip, when an image isn't available
                b"cannot convert ",
                # xclip, when the clipboard isn't initialized
                b"xclip: Error: There is no owner for the ",
            ]:
                if silent_error in err:
                    return None
            msg = f"{args[0]} error"
            if err:
                msg += f": {err.strip().decode()}"
            raise ChildProcessError(msg)

        data = io.BytesIO(p.stdout)
        im = Image.open(data)
        im.load()
        return im
<|endoftext|>#
# Python Imaging Library
# $Id$
#
# stuff to read (and render) GIMP gradient files
#
# History:
#       97-08-23 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#

"""
Stuff to translate curve segments to palette values (derived from
the corresponding code in GIMP, written by Federico Mena Quintero.
See the GIMP distribution for more information.)
"""

from __future__ import annotations

from math import log, pi, sin, sqrt

from ._binary import o8

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO

EPSILON = 1e-10
""""""  # Enable auto-doc for data member


def linear(middle: float, pos: float) -> float:
    if pos <= middle:
        if middle < EPSILON:
            return 0.0
        else:
            return 0.5 * pos / middle
    else:
        pos = pos - middle
        middle = 1.0 - middle
        if middle < EPSILON:
            return 1.0
        else:
            return 0.5 + 0.5 * pos / middle


def curved(middle: float, pos: float) -> float:
    return pos ** (log(0.5) / log(max(middle, EPSILON)))


def sine(middle: float, pos: float) -> float:
    return (sin((-pi / 2.0) + pi * linear(middle, pos)) + 1.0) / 2.0


def sphere_increasing(middle: float, pos: float) -> float:
    return sqrt(1.0 - (linear(middle, pos) - 1.0) ** 2)


def sphere_decreasing(middle: float, pos: float) -> float:
    return 1.0 - sqrt(1.0 - linear(middle, pos) ** 2)


SEGMENTS = [linear, curved, sine, sphere_increasing, sphere_decreasing]
""""""  # Enable auto-doc for data member


class GradientFile:
    gradient: (
        list[
            tuple[
                float,
                float,
                float,
                list[float],
                list[float],
                Callable[[float, float], float],
            ]
        ]
        | None
    ) = None

    def getpalette(self, entries: int = 256) -> tuple[bytes, str]:
        assert self.gradient is not None
        palette = []

        ix = 0
        x0, x1, xm, rgb0, rgb1, segment = self.gradient[ix]

        for i in range(entries):
            x = i / (entries - 1)

            while x1 < x:
                ix += 1
                x0, x1, xm, rgb0, rgb1, segment = self.gradient[ix]

            w = x1 - x0

            if w < EPSILON:
                scale = segment(0.5, 0.5)
            else:
                scale = segment((xm - x0) / w, (x - x0) / w)

            # expand to RGBA
            r = o8(int(255 * ((rgb1[0] - rgb0[0]) * scale + rgb0[0]) + 0.5))
            g = o8(int(255 * ((rgb1[1] - rgb0[1]) * scale + rgb0[1]) + 0.5))
            b = o8(int(255 * ((rgb1[2] - rgb0[2]) * scale + rgb0[2]) + 0.5))
            a = o8(int(255 * ((rgb1[3] - rgb0[3]) * scale + rgb0[3]) + 0.5))

            # add to palette
            palette.append(r + g + b + a)

        return b"".join(palette), "RGBA"


class GimpGradientFile(GradientFile):
    """File handler for GIMP's gradient format."""

    def __init__(self, fp: IO[bytes]) -> None:
        if not fp.readline().startswith(b"GIMP Gradient"):
            msg = "not a GIMP gradient file"
            raise SyntaxError(msg)

        line = fp.readline()

        # GIMP 1.2 gradient files don't contain a name, but GIMP 1.3 files do
        if line.startswith(b"Name: "):
            line = fp.readline().strip()

        count = int(line)

        self.gradient = []

        for i in range(count):
            s = fp.readline().split()
            w = [float(x) for x in s[:11]]

            x0, x1 = w[0], w[2]
            xm = w[1]
            rgb0 = w[3:7]
            rgb1 = w[7:11]

            segment = SEGMENTS[int(s[11])]
            cspace = int(s[12])

            if cspace != 0:
                msg = "cannot handle HSV colour space"
                raise OSError(msg)

            self.gradient.append((x0, x1, xm, rgb0, rgb1, segment))
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# PDF (Acrobat) file handling
#
# History:
# 1996-07-16 fl   Created
# 1997-01-18 fl   Fixed header
# 2004-02-21 fl   Fixes for 1/L/CMYK images, etc.
# 2004-02-24 fl   Fixes for 1 and P images.
#
# Copyright (c) 1997-2004 by Secret Labs AB.  All rights reserved.
# Copyright (c) 1996-1997 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

##
# Image plugin for PDF images (output only).
##
from __future__ import annotations

import io
import math
import os
import time
from typing import IO, Any

from . import Image, ImageFile, ImageSequence, PdfParser, features

#
# --------------------------------------------------------------------

# object ids:
#  1. catalogue
#  2. pages
#  3. image
#  4. page
#  5. page contents


def _save_all(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    _save(im, fp, filename, save_all=True)


##
# (Internal) Image save plugin for the PDF format.


def _write_image(
    im: Image.Image,
    filename: str | bytes,
    existing_pdf: PdfParser.PdfParser,
    image_refs: list[PdfParser.IndirectReference],
) -> tuple[PdfParser.IndirectReference, str]:
    # FIXME: Should replace ASCIIHexDecode with RunLengthDecode
    # (packbits) or LZWDecode (tiff/lzw compression).  Note that
    # PDF 1.2 also supports Flatedecode (zip compression).

    params = None
    decode = None

    #
    # Get image characteristics

    width, height = im.size

    dict_obj: dict[str, Any] = {"BitsPerComponent": 8}
    if im.mode == "1":
        if features.check("libtiff"):
            decode_filter = "CCITTFaxDecode"
            dict_obj["BitsPerComponent"] = 1
            params = PdfParser.PdfArray(
                [
                    PdfParser.PdfDict(
                        {
                            "K": -1,
                            "BlackIs1": True,
                            "Columns": width,
                            "Rows": height,
                        }
                    )
                ]
            )
        else:
            decode_filter = "DCTDecode"
        dict_obj["ColorSpace"] = PdfParser.PdfName("DeviceGray")
        procset = "ImageB"  # grayscale
    elif im.mode == "L":
        decode_filter = "DCTDecode"
        # params = f"<< /Predictor 15 /Columns {width-2} >>"
        dict_obj["ColorSpace"] = PdfParser.PdfName("DeviceGray")
        procset = "ImageB"  # grayscale
    elif im.mode == "LA":
        decode_filter = "JPXDecode"
        # params = f"<< /Predictor 15 /Columns {width-2} >>"
        procset = "ImageB"  # grayscale
        dict_obj["SMaskInData"] = 1
    elif im.mode == "P":
        decode_filter = "ASCIIHexDecode"
        palette = im.getpalette()
        assert palette is not None
        dict_obj["ColorSpace"] = [
            PdfParser.PdfName("Indexed"),
            PdfParser.PdfName("DeviceRGB"),
            len(palette) // 3 - 1,
            PdfParser.PdfBinary(palette),
        ]
        procset = "ImageI"  # indexed color

        if "transparency" in im.info:
            smask = im.convert("LA").getchannel("A")
            smask.encoderinfo = {}

            image_ref = _write_image(smask, filename, existing_pdf, image_refs)[0]
            dict_obj["SMask"] = image_ref
    elif im.mode == "RGB":
        decode_filter = "DCTDecode"
        dict_obj["ColorSpace"] = PdfParser.PdfName("DeviceRGB")
        procset = "ImageC"  # color images
    elif im.mode == "RGBA":
        decode_filter = "JPXDecode"
        procset = "ImageC"  # color images
        dict_obj["SMaskInData"] = 1
    elif im.mode == "CMYK":
        decode_filter = "DCTDecode"
        dict_obj["ColorSpace"] = PdfParser.PdfName("DeviceCMYK")
        procset = "ImageC"  # color images
        decode = [1, 0, 1, 0, 1, 0, 1, 0]
    else:
        msg = f"cannot save mode {im.mode}"
        raise ValueError(msg)

    #
    # image

    op = io.BytesIO()

    if decode_filter == "ASCIIHexDecode":
        ImageFile._save(im, op, [ImageFile._Tile("hex", (0, 0) + im.size, 0, im.mode)])
    elif decode_filter == "CCITTFaxDecode":
        im.save(
            op,
            "TIFF",
            compression="group4",
            # use a single strip
            strip_size=math.ceil(width / 8) * height,
        )
    elif decode_filter == "DCTDecode":
        from . import JpegImagePlugin

        JpegImagePlugin._save(im, op, filename)
    elif decode_filter == "JPXDecode":
        from . import Jpeg2KImagePlugin

        del dict_obj["BitsPerComponent"]
        Jpeg2KImagePlugin._save(im, op, filename)
    else:
        msg = f"unsupported PDF filter ({decode_filter})"
        raise ValueError(msg)

    stream = op.getvalue()
    filter: PdfParser.PdfArray | PdfParser.PdfName
    if decode_filter == "CCITTFaxDecode":
        stream = stream[8:]
        filter = PdfParser.PdfArray([PdfParser.PdfName(decode_filter)])
    else:
        filter = PdfParser.PdfName(decode_filter)

    image_ref = image_refs.pop(0)
    existing_pdf.write_obj(
        image_ref,
        stream=stream,
        Type=PdfParser.PdfName("XObject"),
        Subtype=PdfParser.PdfName("Image"),
        Width=width,  # * 72.0 / x_resolution,
        Height=height,  # * 72.0 / y_resolution,
        Filter=filter,
        Decode=decode,
        DecodeParms=params,
        **dict_obj,
    )

    return image_ref, procset


def _save(
    im: Image.Image, fp: IO[bytes], filename: str | bytes, save_all: bool = False
) -> None:
    is_appending = im.encoderinfo.get("append", False)
    filename_str = filename.decode() if isinstance(filename, bytes) else filename
    if is_appending:
        existing_pdf = PdfParser.PdfParser(f=fp, filename=filename_str, mode="r+b")
    else:
        existing_pdf = PdfParser.PdfParser(f=fp, filename=filename_str, mode="w+b")

    dpi = im.encoderinfo.get("dpi")
    if dpi:
        x_resolution = dpi[0]
        y_resolution = dpi[1]
    else:
        x_resolution = y_resolution = im.encoderinfo.get("resolution", 72.0)

    info = {
        "title": (
            None if is_appending else os.path.splitext(os.path.basename(filename))[0]
        ),
        "author": None,
        "subject": None,
        "keywords": None,
        "creator": None,
        "producer": None,
        "creationDate": None if is_appending else time.gmtime(),
        "modDate": None if is_appending else time.gmtime(),
    }
    for k, default in info.items():
        v = im.encoderinfo.get(k) if k in im.encoderinfo else default
        if v:
            existing_pdf.info[k[0].upper() + k[1:]] = v

    #
    # make sure image data is available
    im.load()

    existing_pdf.start_writing()
    existing_pdf.write_header()
    existing_pdf.write_comment("created by Pillow PDF driver")

    #
    # pages
    ims = [im]
    if save_all:
        append_images = im.encoderinfo.get("append_images", [])
        for append_im in append_images:
            append_im.encoderinfo = im.encoderinfo.copy()
            ims.append(append_im)
    number_of_pages = 0
    image_refs = []
    page_refs = []
    contents_refs = []
    for im in ims:
        im_number_of_pages = 1
        if save_all:
            im_number_of_pages = getattr(im, "n_frames", 1)
        number_of_pages += im_number_of_pages
        for i in range(im_number_of_pages):
            image_refs.append(existing_pdf.next_object_id(0))
            if im.mode == "P" and "transparency" in im.info:
                image_refs.append(existing_pdf.next_object_id(0))

            page_refs.append(existing_pdf.next_object_id(0))
            contents_refs.append(existing_pdf.next_object_id(0))
            existing_pdf.pages.append(page_refs[-1])

    #
    # catalog and list of pages
    existing_pdf.write_catalog()

    page_number = 0
    for im_sequence in ims:
        im_pages: ImageSequence.Iterator | list[Image.Image] = (
            ImageSequence.Iterator(im_sequence) if save_all else [im_sequence]
        )
        for im in im_pages:
            image_ref, procset = _write_image(im, filename, existing_pdf, image_refs)

            #
            # page

            existing_pdf.write_page(
                page_refs[page_number],
                Resources=PdfParser.PdfDict(
                    ProcSet=[PdfParser.PdfName("PDF"), PdfParser.PdfName(procset)],
                    XObject=PdfParser.PdfDict(image=image_ref),
                ),
                MediaBox=[
                    0,
                    0,
                    im.width * 72.0 / x_resolution,
                    im.height * 72.0 / y_resolution,
                ],
                Contents=contents_refs[page_number],
            )

            #
            # page contents

            page_contents = b"q %f 0 0 %f 0 0 cm /image Do Q\n" % (
                im.width * 72.0 / x_resolution,
                im.height * 72.0 / y_resolution,
            )

            existing_pdf.write_obj(contents_refs[page_number], stream=page_contents)

            page_number += 1

    #
    # trailer
    existing_pdf.write_xref_and_trailer()
    if hasattr(fp, "flush"):
        fp.flush()
    existing_pdf.close()


#
# --------------------------------------------------------------------


Image.register_save("PDF", _save)
Image.register_save_all("PDF", _save_all)

Image.register_extension("PDF", ".pdf")

Image.register_mime("PDF", "application/pdf")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# standard image operations
#
# History:
# 2001-10-20 fl   Created
# 2001-10-23 fl   Added autocontrast operator
# 2001-12-18 fl   Added Kevin's fit operator
# 2004-03-14 fl   Fixed potential division by zero in equalize
# 2005-05-05 fl   Fixed equalize for low number of values
#
# Copyright (c) 2001-2004 by Secret Labs AB
# Copyright (c) 2001-2004 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import functools
import operator
import re
from collections.abc import Sequence
from typing import Literal, Protocol, cast, overload

from . import ExifTags, Image, ImagePalette

#
# helpers


def _border(border: int | tuple[int, ...]) -> tuple[int, int, int, int]:
    if isinstance(border, tuple):
        if len(border) == 2:
            left, top = right, bottom = border
        elif len(border) == 4:
            left, top, right, bottom = border
        else:
            msg = "border must be an integer, or a tuple of two or four elements"
            raise ValueError(msg)
    else:
        left = top = right = bottom = border
    return left, top, right, bottom


def _color(color: str | int | tuple[int, ...], mode: str) -> int | tuple[int, ...]:
    if isinstance(color, str):
        from . import ImageColor

        color = ImageColor.getcolor(color, mode)
    return color


def _lut(image: Image.Image, lut: list[int]) -> Image.Image:
    if image.mode == "P":
        # FIXME: apply to lookup table, not image data
        msg = "mode P support coming soon"
        raise NotImplementedError(msg)
    elif image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        msg = f"not supported for mode {image.mode}"
        raise OSError(msg)


#
# actions


def autocontrast(
    image: Image.Image,
    cutoff: float | tuple[float, float] = 0,
    ignore: int | Sequence[int] | None = None,
    mask: Image.Image | None = None,
    preserve_tone: bool = False,
) -> Image.Image:
    """
    Maximize (normalize) image contrast. This function calculates a
    histogram of the input image (or mask region), removes ``cutoff`` percent of the
    lightest and darkest pixels from the histogram, and remaps the image
    so that the darkest pixel becomes black (0), and the lightest
    becomes white (255).

    :param image: The image to process.
    :param cutoff: The percent to cut off from the histogram on the low and
                   high ends. Either a tuple of (low, high), or a single
                   number for both.
    :param ignore: The background pixel value (use None for no background).
    :param mask: Histogram used in contrast operation is computed using pixels
                 within the mask. If no mask is given the entire image is used
                 for histogram computation.
    :param preserve_tone: Preserve image tone in Photoshop-like style autocontrast.

                          .. versionadded:: 8.2.0

    :return: An image.
    """
    if preserve_tone:
        histogram = image.convert("L").histogram(mask)
    else:
        histogram = image.histogram(mask)

    lut = []
    for layer in range(0, len(histogram), 256):
        h = histogram[layer : layer + 256]
        if ignore is not None:
            # get rid of outliers
            if isinstance(ignore, int):
                h[ignore] = 0
            else:
                for ix in ignore:
                    h[ix] = 0
        if cutoff:
            # cut off pixels from both ends of the histogram
            if not isinstance(cutoff, tuple):
                cutoff = (cutoff, cutoff)
            # get number of pixels
            n = 0
            for ix in range(256):
                n = n + h[ix]
            # remove cutoff% pixels from the low end
            cut = int(n * cutoff[0] // 100)
            for lo in range(256):
                if cut > h[lo]:
                    cut = cut - h[lo]
                    h[lo] = 0
                else:
                    h[lo] -= cut
                    cut = 0
                if cut <= 0:
                    break
            # remove cutoff% samples from the high end
            cut = int(n * cutoff[1] // 100)
            for hi in range(255, -1, -1):
                if cut > h[hi]:
                    cut = cut - h[hi]
                    h[hi] = 0
                else:
                    h[hi] -= cut
                    cut = 0
                if cut <= 0:
                    break
        # find lowest/highest samples after preprocessing
        for lo in range(256):
            if h[lo]:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            lut.extend(list(range(256)))
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            for ix in range(256):
                ix = int(ix * scale + offset)
                if ix < 0:
                    ix = 0
                elif ix > 255:
                    ix = 255
                lut.append(ix)
    return _lut(image, lut)


def colorize(
    image: Image.Image,
    black: str | tuple[int, ...],
    white: str | tuple[int, ...],
    mid: str | int | tuple[int, ...] | None = None,
    blackpoint: int = 0,
    whitepoint: int = 255,
    midpoint: int = 127,
) -> Image.Image:
    """
    Colorize grayscale image.
    This function calculates a color wedge which maps all black pixels in
    the source image to the first color and all white pixels to the
    second color. If ``mid`` is specified, it uses three-color mapping.
    The ``black`` and ``white`` arguments should be RGB tuples or color names;
    optionally you can use three-color mapping by also specifying ``mid``.
    Mapping positions for any of the colors can be specified
    (e.g. ``blackpoint``), where these parameters are the integer
    value corresponding to where the corresponding color should be mapped.
    These parameters must have logical order, such that
    ``blackpoint <= midpoint <= whitepoint`` (if ``mid`` is specified).

    :param image: The image to colorize.
    :param black: The color to use for black input pixels.
    :param white: The color to use for white input pixels.
    :param mid: The color to use for midtone input pixels.
    :param blackpoint: an int value [0, 255] for the black mapping.
    :param whitepoint: an int value [0, 255] for the white mapping.
    :param midpoint: an int value [0, 255] for the midtone mapping.
    :return: An image.
    """

    # Initial asserts
    assert image.mode == "L"
    if mid is None:
        assert 0 <= blackpoint <= whitepoint <= 255
    else:
        assert 0 <= blackpoint <= midpoint <= whitepoint <= 255

    # Define colors from arguments
    rgb_black = cast(Sequence[int], _color(black, "RGB"))
    rgb_white = cast(Sequence[int], _color(white, "RGB"))
    rgb_mid = cast(Sequence[int], _color(mid, "RGB")) if mid is not None else None

    # Empty lists for the mapping
    red = []
    green = []
    blue = []

    # Create the low-end values
    for i in range(blackpoint):
        red.append(rgb_black[0])
        green.append(rgb_black[1])
        blue.append(rgb_black[2])

    # Create the mapping (2-color)
    if rgb_mid is None:
        range_map = range(whitepoint - blackpoint)

        for i in range_map:
            red.append(
                rgb_black[0] + i * (rgb_white[0] - rgb_black[0]) // len(range_map)
            )
            green.append(
                rgb_black[1] + i * (rgb_white[1] - rgb_black[1]) // len(range_map)
            )
            blue.append(
                rgb_black[2] + i * (rgb_white[2] - rgb_black[2]) // len(range_map)
            )

    # Create the mapping (3-color)
    else:
        range_map1 = range(midpoint - blackpoint)
        range_map2 = range(whitepoint - midpoint)

        for i in range_map1:
            red.append(
                rgb_black[0] + i * (rgb_mid[0] - rgb_black[0]) // len(range_map1)
            )
            green.append(
                rgb_black[1] + i * (rgb_mid[1] - rgb_black[1]) // len(range_map1)
            )
            blue.append(
                rgb_black[2] + i * (rgb_mid[2] - rgb_black[2]) // len(range_map1)
            )
        for i in range_map2:
            red.append(rgb_mid[0] + i * (rgb_white[0] - rgb_mid[0]) // len(range_map2))
            green.append(
                rgb_mid[1] + i * (rgb_white[1] - rgb_mid[1]) // len(range_map2)
            )
            blue.append(rgb_mid[2] + i * (rgb_white[2] - rgb_mid[2]) // len(range_map2))

    # Create the high-end values
    for i in range(256 - whitepoint):
        red.append(rgb_white[0])
        green.append(rgb_white[1])
        blue.append(rgb_white[2])

    # Return converted image
    image = image.convert("RGB")
    return _lut(image, red + green + blue)


def contain(
    image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC
) -> Image.Image:
    """
    Returns a resized version of the image, set to the maximum width and height
    within the requested size, while maintaining the original aspect ratio.

    :param image: The image to resize.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                   See :ref:`concept-filters`.
    :return: An image.
    """

    im_ratio = image.width / image.height
    dest_ratio = size[0] / size[1]

    if im_ratio != dest_ratio:
        if im_ratio > dest_ratio:
            new_height = round(image.height / image.width * size[0])
            if new_height != size[1]:
                size = (size[0], new_height)
        else:
            new_width = round(image.width / image.height * size[1])
            if new_width != size[0]:
                size = (new_width, size[1])
    return image.resize(size, resample=method)


def cover(
    image: Image.Image, size: tuple[int, int], method: int = Image.Resampling.BICUBIC
) -> Image.Image:
    """
    Returns a resized version of the image, so that the requested size is
    covered, while maintaining the original aspect ratio.

    :param image: The image to resize.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                   See :ref:`concept-filters`.
    :return: An image.
    """

    im_ratio = image.width / image.height
    dest_ratio = size[0] / size[1]

    if im_ratio != dest_ratio:
        if im_ratio < dest_ratio:
            new_height = round(image.height / image.width * size[0])
            if new_height != size[1]:
                size = (size[0], new_height)
        else:
            new_width = round(image.width / image.height * size[1])
            if new_width != size[0]:
                size = (new_width, size[1])
    return image.resize(size, resample=method)


def pad(
    image: Image.Image,
    size: tuple[int, int],
    method: int = Image.Resampling.BICUBIC,
    color: str | int | tuple[int, ...] | None = None,
    centering: tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    """
    Returns a resized and padded version of the image, expanded to fill the
    requested aspect ratio and size.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                   See :ref:`concept-filters`.
    :param color: The background color of the padded image.
    :param centering: Control the position of the original image within the
                      padded version.

                          (0.5, 0.5) will keep the image centered
                          (0, 0) will keep the image aligned to the top left
                          (1, 1) will keep the image aligned to the bottom
                          right
    :return: An image.
    """

    resized = contain(image, size, method)
    if resized.size == size:
        out = resized
    else:
        out = Image.new(image.mode, size, color)
        if resized.palette:
            palette = resized.getpalette()
            if palette is not None:
                out.putpalette(palette)
        if resized.width != size[0]:
            x = round((size[0] - resized.width) * max(0, min(centering[0], 1)))
            out.paste(resized, (x, 0))
        else:
            y = round((size[1] - resized.height) * max(0, min(centering[1], 1)))
            out.paste(resized, (0, y))
    return out


def crop(image: Image.Image, border: int = 0) -> Image.Image:
    """
    Remove border from image.  The same amount of pixels are removed
    from all four sides.  This function works on all image modes.

    .. seealso:: :py:meth:`~PIL.Image.Image.crop`

    :param image: The image to crop.
    :param border: The number of pixels to remove.
    :return: An image.
    """
    left, top, right, bottom = _border(border)
    return image.crop((left, top, image.size[0] - right, image.size[1] - bottom))


def scale(
    image: Image.Image, factor: float, resample: int = Image.Resampling.BICUBIC
) -> Image.Image:
    """
    Returns a rescaled image by a specific factor given in parameter.
    A factor greater than 1 expands the image, between 0 and 1 contracts the
    image.

    :param image: The image to rescale.
    :param factor: The expansion factor, as a float.
    :param resample: Resampling method to use. Default is
                     :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                     See :ref:`concept-filters`.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    if factor == 1:
        return image.copy()
    elif factor <= 0:
        msg = "the factor must be greater than 0"
        raise ValueError(msg)
    else:
        size = (round(factor * image.width), round(factor * image.height))
        return image.resize(size, resample)


class SupportsGetMesh(Protocol):
    """
    An object that supports the ``getmesh`` method, taking an image as an
    argument, and returning a list of tuples. Each tuple contains two tuples,
    the source box as a tuple of 4 integers, and a tuple of 8 integers for the
    final quadrilateral, in order of top left, bottom left, bottom right, top
    right.
    """

    def getmesh(
        self, image: Image.Image
    ) -> list[
        tuple[tuple[int, int, int, int], tuple[int, int, int, int, int, int, int, int]]
    ]: ...


def deform(
    image: Image.Image,
    deformer: SupportsGetMesh,
    resample: int = Image.Resampling.BILINEAR,
) -> Image.Image:
    """
    Deform the image.

    :param image: The image to deform.
    :param deformer: A deformer object.  Any object that implements a
                    ``getmesh`` method can be used.
    :param resample: An optional resampling filter. Same values possible as
       in the PIL.Image.transform function.
    :return: An image.
    """
    return image.transform(
        image.size, Image.Transform.MESH, deformer.getmesh(image), resample
    )


def equalize(image: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """
    Equalize the image histogram. This function applies a non-linear
    mapping to the input image, in order to create a uniform
    distribution of grayscale values in the output image.

    :param image: The image to equalize.
    :param mask: An optional mask.  If given, only the pixels selected by
                 the mask are included in the analysis.
    :return: An image.
    """
    if image.mode == "P":
        image = image.convert("RGB")
    h = image.histogram(mask)
    lut = []
    for b in range(0, len(h), 256):
        histo = [_f for _f in h[b : b + 256] if _f]
        if len(histo) <= 1:
            lut.extend(list(range(256)))
        else:
            step = (functools.reduce(operator.add, histo) - histo[-1]) // 255
            if not step:
                lut.extend(list(range(256)))
            else:
                n = step // 2
                for i in range(256):
                    lut.append(n // step)
                    n = n + h[i + b]
    return _lut(image, lut)


def expand(
    image: Image.Image,
    border: int | tuple[int, ...] = 0,
    fill: str | int | tuple[int, ...] = 0,
) -> Image.Image:
    """
    Add border to the image

    :param image: The image to expand.
    :param border: Border width, in pixels.
    :param fill: Pixel fill value (a color value).  Default is 0 (black).
    :return: An image.
    """
    left, top, right, bottom = _border(border)
    width = left + image.size[0] + right
    height = top + image.size[1] + bottom
    color = _color(fill, image.mode)
    if image.palette:
        mode = image.palette.mode
        palette = ImagePalette.ImagePalette(mode, image.getpalette(mode))
        if isinstance(color, tuple) and (len(color) == 3 or len(color) == 4):
            color = palette.getcolor(color)
    else:
        palette = None
    out = Image.new(image.mode, (width, height), color)
    if palette:
        out.putpalette(palette.palette, mode)
    out.paste(image, (left, top))
    return out


def fit(
    image: Image.Image,
    size: tuple[int, int],
    method: int = Image.Resampling.BICUBIC,
    bleed: float = 0.0,
    centering: tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    """
    Returns a resized and cropped version of the image, cropped to the
    requested aspect ratio and size.

    This function was contributed by Kevin Cazabon.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                   See :ref:`concept-filters`.
    :param bleed: Remove a border around the outside of the image from all
                  four edges. The value is a decimal percentage (use 0.01 for
                  one percent). The default value is 0 (no border).
                  Cannot be greater than or equal to 0.5.
    :param centering: Control the cropping position.  Use (0.5, 0.5) for
                      center cropping (e.g. if cropping the width, take 50% off
                      of the left side, and therefore 50% off the right side).
                      (0.0, 0.0) will crop from the top left corner (i.e. if
                      cropping the width, take all of the crop off of the right
                      side, and if cropping the height, take all of it off the
                      bottom).  (1.0, 0.0) will crop from the bottom left
                      corner, etc. (i.e. if cropping the width, take all of the
                      crop off the left side, and if cropping the height take
                      none from the top, and therefore all off the bottom).
    :return: An image.
    """

    # by Kevin Cazabon, Feb 17/2000
    # kevin@cazabon.com
    # https://www.cazabon.com

    centering_x, centering_y = centering

    if not 0.0 <= centering_x <= 1.0:
        centering_x = 0.5
    if not 0.0 <= centering_y <= 1.0:
        centering_y = 0.5

    if not 0.0 <= bleed < 0.5:
        bleed = 0.0

    # calculate the area to use for resizing and cropping, subtracting
    # the 'bleed' around the edges

    # number of pixels to trim off on Top and Bottom, Left and Right
    bleed_pixels = (bleed * image.size[0], bleed * image.size[1])

    live_size = (
        image.size[0] - bleed_pixels[0] * 2,
        image.size[1] - bleed_pixels[1] * 2,
    )

    # calculate the aspect ratio of the live_size
    live_size_ratio = live_size[0] / live_size[1]

    # calculate the aspect ratio of the output image
    output_ratio = size[0] / size[1]

    # figure out if the sides or top/bottom will be cropped off
    if live_size_ratio == output_ratio:
        # live_size is already the needed ratio
        crop_width = live_size[0]
        crop_height = live_size[1]
    elif live_size_ratio >= output_ratio:
        # live_size is wider than what's needed, crop the sides
        crop_width = output_ratio * live_size[1]
        crop_height = live_size[1]
    else:
        # live_size is taller than what's needed, crop the top and bottom
        crop_width = live_size[0]
        crop_height = live_size[0] / output_ratio

    # make the crop
    crop_left = bleed_pixels[0] + (live_size[0] - crop_width) * centering_x
    crop_top = bleed_pixels[1] + (live_size[1] - crop_height) * centering_y

    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)

    # resize the image and return it
    return image.resize(size, method, box=crop)


def flip(image: Image.Image) -> Image.Image:
    """
    Flip the image vertically (top to bottom).

    :param image: The image to flip.
    :return: An image.
    """
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def grayscale(image: Image.Image) -> Image.Image:
    """
    Convert the image to grayscale.

    :param image: The image to convert.
    :return: An image.
    """
    return image.convert("L")


def invert(image: Image.Image) -> Image.Image:
    """
    Invert (negate) the image.

    :param image: The image to invert.
    :return: An image.
    """
    lut = list(range(255, -1, -1))
    return image.point(lut) if image.mode == "1" else _lut(image, lut)


def mirror(image: Image.Image) -> Image.Image:
    """
    Flip image horizontally (left to right).

    :param image: The image to mirror.
    :return: An image.
    """
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def posterize(image: Image.Image, bits: int) -> Image.Image:
    """
    Reduce the number of bits for each color channel.

    :param image: The image to posterize.
    :param bits: The number of bits to keep for each channel (1-8).
    :return: An image.
    """
    mask = ~(2 ** (8 - bits) - 1)
    lut = [i & mask for i in range(256)]
    return _lut(image, lut)


def solarize(image: Image.Image, threshold: int = 128) -> Image.Image:
    """
    Invert all pixel values above a threshold.

    :param image: The image to solarize.
    :param threshold: All pixels above this grayscale level are inverted.
    :return: An image.
    """
    lut = []
    for i in range(256):
        if i < threshold:
            lut.append(i)
        else:
            lut.append(255 - i)
    return _lut(image, lut)


@overload
def exif_transpose(image: Image.Image, *, in_place: Literal[True]) -> None: ...


@overload
def exif_transpose(
    image: Image.Image, *, in_place: Literal[False] = False
) -> Image.Image: ...


def exif_transpose(image: Image.Image, *, in_place: bool = False) -> Image.Image | None:
    """
    If an image has an EXIF Orientation tag, other than 1, transpose the image
    accordingly, and remove the orientation data.

    :param image: The image to transpose.
    :param in_place: Boolean. Keyword-only argument.
        If ``True``, the original image is modified in-place, and ``None`` is returned.
        If ``False`` (default), a new :py:class:`~PIL.Image.Image` object is returned
        with the transposition applied. If there is no transposition, a copy of the
        image will be returned.
    """
    image.load()
    image_exif = image.getexif()
    orientation = image_exif.get(ExifTags.Base.Orientation, 1)
    method = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(orientation)
    if method is not None:
        if in_place:
            image.im = image.im.transpose(method)
            image._size = image.im.size
        else:
            transposed_image = image.transpose(method)
        exif_image = image if in_place else transposed_image

        exif = exif_image.getexif()
        if ExifTags.Base.Orientation in exif:
            del exif[ExifTags.Base.Orientation]
            if "exif" in exif_image.info:
                exif_image.info["exif"] = exif.tobytes()
            elif "Raw profile type exif" in exif_image.info:
                exif_image.info["Raw profile type exif"] = exif.tobytes().hex()
            for key in ("XML:com.adobe.xmp", "xmp"):
                if key in exif_image.info:
                    for pattern in (
                        r'tiff:Orientation="([0-9])"',
                        r"<tiff:Orientation>([0-9])</tiff:Orientation>",
                    ):
                        value = exif_image.info[key]
                        if isinstance(value, str):
                            value = re.sub(pattern, "", value)
                        elif isinstance(value, tuple):
                            value = tuple(
                                re.sub(pattern.encode(), b"", v) for v in value
                            )
                        else:
                            value = re.sub(pattern.encode(), b"", value)
                        exif_image.info[key] = value
        if not in_place:
            return transposed_image
    elif not in_place:
        return image.copy()
    return None
<|endoftext|>from __future__ import annotations

import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
from typing import Any, NamedTuple

from . import ImageFile

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import IO

    _DictBase = collections.UserDict[str | bytes, Any]
else:
    _DictBase = collections.UserDict


# see 7.9.2.2 Text String Type on page 86 and D.3 PDFDocEncoding Character Set
# on page 656
def encode_text(s: str) -> bytes:
    return codecs.BOM_UTF16_BE + s.encode("utf_16_be")


PDFDocEncoding = {
    0x16: "\u0017",
    0x18: "\u02d8",
    0x19: "\u02c7",
    0x1A: "\u02c6",
    0x1B: "\u02d9",
    0x1C: "\u02dd",
    0x1D: "\u02db",
    0x1E: "\u02da",
    0x1F: "\u02dc",
    0x80: "\u2022",
    0x81: "\u2020",
    0x82: "\u2021",
    0x83: "\u2026",
    0x84: "\u2014",
    0x85: "\u2013",
    0x86: "\u0192",
    0x87: "\u2044",
    0x88: "\u2039",
    0x89: "\u203a",
    0x8A: "\u2212",
    0x8B: "\u2030",
    0x8C: "\u201e",
    0x8D: "\u201c",
    0x8E: "\u201d",
    0x8F: "\u2018",
    0x90: "\u2019",
    0x91: "\u201a",
    0x92: "\u2122",
    0x93: "\ufb01",
    0x94: "\ufb02",
    0x95: "\u0141",
    0x96: "\u0152",
    0x97: "\u0160",
    0x98: "\u0178",
    0x99: "\u017d",
    0x9A: "\u0131",
    0x9B: "\u0142",
    0x9C: "\u0153",
    0x9D: "\u0161",
    0x9E: "\u017e",
    0xA0: "\u20ac",
}


def decode_text(b: bytes) -> str:
    if b[: len(codecs.BOM_UTF16_BE)] == codecs.BOM_UTF16_BE:
        return b[len(codecs.BOM_UTF16_BE) :].decode("utf_16_be")
    else:
        return "".join(PDFDocEncoding.get(byte, chr(byte)) for byte in b)


class PdfFormatError(RuntimeError):
    """An error that probably indicates a syntactic or semantic error in the
    PDF file structure"""

    pass


def check_format_condition(condition: bool, error_message: str) -> None:
    if not condition:
        raise PdfFormatError(error_message)


class IndirectReferenceTuple(NamedTuple):
    object_id: int
    generation: int


class IndirectReference(IndirectReferenceTuple):
    def __str__(self) -> str:
        return f"{self.object_id} {self.generation} R"

    def __bytes__(self) -> bytes:
        return self.__str__().encode("us-ascii")

    def __eq__(self, other: object) -> bool:
        if self.__class__ is not other.__class__:
            return False
        assert isinstance(other, IndirectReference)
        return other.object_id == self.object_id and other.generation == self.generation

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash((self.object_id, self.generation))


class IndirectObjectDef(IndirectReference):
    def __str__(self) -> str:
        return f"{self.object_id} {self.generation} obj"


class XrefTable:
    def __init__(self) -> None:
        self.existing_entries: dict[int, tuple[int, int]] = (
            {}
        )  # object ID => (offset, generation)
        self.new_entries: dict[int, tuple[int, int]] = (
            {}
        )  # object ID => (offset, generation)
        self.deleted_entries = {0: 65536}  # object ID => generation
        self.reading_finished = False

    def __setitem__(self, key: int, value: tuple[int, int]) -> None:
        if self.reading_finished:
            self.new_entries[key] = value
        else:
            self.existing_entries[key] = value
        if key in self.deleted_entries:
            del self.deleted_entries[key]

    def __getitem__(self, key: int) -> tuple[int, int]:
        try:
            return self.new_entries[key]
        except KeyError:
            return self.existing_entries[key]

    def __delitem__(self, key: int) -> None:
        if key in self.new_entries:
            generation = self.new_entries[key][1] + 1
            del self.new_entries[key]
            self.deleted_entries[key] = generation
        elif key in self.existing_entries:
            generation = self.existing_entries[key][1] + 1
            self.deleted_entries[key] = generation
        elif key in self.deleted_entries:
            generation = self.deleted_entries[key]
        else:
            msg = f"object ID {key} cannot be deleted because it doesn't exist"
            raise IndexError(msg)

    def __contains__(self, key: int) -> bool:
        return key in self.existing_entries or key in self.new_entries

    def __len__(self) -> int:
        return len(
            set(self.existing_entries.keys())
            | set(self.new_entries.keys())
            | set(self.deleted_entries.keys())
        )

    def keys(self) -> set[int]:
        return (
            set(self.existing_entries.keys()) - set(self.deleted_entries.keys())
        ) | set(self.new_entries.keys())

    def write(self, f: IO[bytes]) -> int:
        keys = sorted(set(self.new_entries.keys()) | set(self.deleted_entries.keys()))
        deleted_keys = sorted(set(self.deleted_entries.keys()))
        startxref = f.tell()
        f.write(b"xref\n")
        while keys:
            # find a contiguous sequence of object IDs
            prev: int | None = None
            for index, key in enumerate(keys):
                if prev is None or prev + 1 == key:
                    prev = key
                else:
                    contiguous_keys = keys[:index]
                    keys = keys[index:]
                    break
            else:
                contiguous_keys = keys
                keys = []
            f.write(b"%d %d\n" % (contiguous_keys[0], len(contiguous_keys)))
            for object_id in contiguous_keys:
                if object_id in self.new_entries:
                    f.write(b"%010d %05d n \n" % self.new_entries[object_id])
                else:
                    this_deleted_object_id = deleted_keys.pop(0)
                    check_format_condition(
                        object_id == this_deleted_object_id,
                        f"expected the next deleted object ID to be {object_id}, "
                        f"instead found {this_deleted_object_id}",
                    )
                    try:
                        next_in_linked_list = deleted_keys[0]
                    except IndexError:
                        next_in_linked_list = 0
                    f.write(
                        b"%010d %05d f \n"
                        % (next_in_linked_list, self.deleted_entries[object_id])
                    )
        return startxref


class PdfName:
    name: bytes

    def __init__(self, name: PdfName | bytes | str) -> None:
        if isinstance(name, PdfName):
            self.name = name.name
        elif isinstance(name, bytes):
            self.name = name
        else:
            self.name = name.encode("us-ascii")

    def name_as_str(self) -> str:
        return self.name.decode("us-ascii")

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PdfName) and other.name == self.name
        ) or other == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.name)})"

    @classmethod
    def from_pdf_stream(cls, data: bytes) -> PdfName:
        return cls(PdfParser.interpret_name(data))

    allowed_chars = set(range(33, 127)) - {ord(c) for c in "#%/()<>[]{}"}

    def __bytes__(self) -> bytes:
        result = bytearray(b"/")
        for b in self.name:
            if b in self.allowed_chars:
                result.append(b)
            else:
                result.extend(b"#%02X" % b)
        return bytes(result)


class PdfArray(list[Any]):
    def __bytes__(self) -> bytes:
        return b"[ " + b" ".join(pdf_repr(x) for x in self) + b" ]"


class PdfDict(_DictBase):
    def __setattr__(self, key: str, value: Any) -> None:
        if key == "data":
            collections.UserDict.__setattr__(self, key, value)
        else:
            self[key.encode("us-ascii")] = value

    def __getattr__(self, key: str) -> str | time.struct_time:
        try:
            value = self[key.encode("us-ascii")]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, bytes):
            value = decode_text(value)
        if key.endswith("Date"):
            if value.startswith("D:"):
                value = value[2:]

            relationship = "Z"
            if len(value) > 17:
                relationship = value[14]
                offset = int(value[15:17]) * 60
                if len(value) > 20:
                    offset += int(value[18:20])

            format = "%Y%m%d%H%M%S"[: len(value) - 2]
            value = time.strptime(value[: len(format) + 2], format)
            if relationship in ["+", "-"]:
                offset *= 60
                if relationship == "+":
                    offset *= -1
                value = time.gmtime(calendar.timegm(value) + offset)
        return value

    def __bytes__(self) -> bytes:
        out = bytearray(b"<<")
        for key, value in self.items():
            if value is None:
                continue
            value = pdf_repr(value)
            out.extend(b"\n")
            out.extend(bytes(PdfName(key)))
            out.extend(b" ")
            out.extend(value)
        out.extend(b"\n>>")
        return bytes(out)


class PdfBinary:
    def __init__(self, data: list[int] | bytes) -> None:
        self.data = data

    def __bytes__(self) -> bytes:
        return b"<%s>" % b"".join(b"%02X" % b for b in self.data)


class PdfStream:
    def __init__(self, dictionary: PdfDict, buf: bytes) -> None:
        self.dictionary = dictionary
        self.buf = buf

    def decode(self, max_length: int = ImageFile.SAFEBLOCK) -> bytes:
        try:
            filter = self.dictionary[b"Filter"]
        except KeyError:
            return self.buf
        if filter == b"FlateDecode":
            dobj = zlib.decompressobj()
            plaintext = dobj.decompress(self.buf, max_length)
            if dobj.unconsumed_tail:
                msg = "Decompressed data too large"
                raise ValueError(msg)
            return plaintext
        else:
            msg = f"stream filter {repr(filter)} unknown/unsupported"
            raise NotImplementedError(msg)


def pdf_repr(x: Any) -> bytes:
    if x is True:
        return b"true"
    elif x is False:
        return b"false"
    elif x is None:
        return b"null"
    elif isinstance(x, (PdfName, PdfDict, PdfArray, PdfBinary)):
        return bytes(x)
    elif isinstance(x, (int, float)):
        return str(x).encode("us-ascii")
    elif isinstance(x, time.struct_time):
        return b"(D:" + time.strftime("%Y%m%d%H%M%SZ", x).encode("us-ascii") + b")"
    elif isinstance(x, dict):
        return bytes(PdfDict(x))
    elif isinstance(x, list):
        return bytes(PdfArray(x))
    elif isinstance(x, str):
        return pdf_repr(encode_text(x))
    elif isinstance(x, bytes):
        # XXX escape more chars? handle binary garbage
        x = x.replace(b"\\", b"\\\\")
        x = x.replace(b"(", b"\\(")
        x = x.replace(b")", b"\\)")
        return b"(" + x + b")"
    else:
        return bytes(x)


class PdfParser:
    """Based on
    https://www.adobe.com/content/dam/acom/en/devnet/acrobat/pdfs/PDF32000_2008.pdf
    Supports PDF up to 1.4
    """

    def __init__(
        self,
        filename: str | None = None,
        f: IO[bytes] | None = None,
        buf: bytes | bytearray | None = None,
        start_offset: int = 0,
        mode: str = "rb",
    ) -> None:
        if buf and f:
            msg = "specify buf or f or filename, but not both buf and f"
            raise RuntimeError(msg)
        self.filename = filename
        self.buf: bytes | bytearray | memoryview | mmap.mmap | None = buf
        self.f = f
        self.start_offset = start_offset
        self.should_close_buf = False
        self.should_close_file = False
        if filename is not None and f is None:
            self.f = f = open(filename, mode)
            self.should_close_file = True
        if f is not None:
            self.buf = self.get_buf_from_file(f)
            self.should_close_buf = True
            if not filename and hasattr(f, "name"):
                self.filename = f.name
        self.cached_objects: dict[IndirectReference, Any] = {}
        self.root_ref: IndirectReference | None
        self.info_ref: IndirectReference | None
        self.pages_ref: IndirectReference | None
        self.last_xref_section_offset: int | None
        if self.buf:
            try:
                self.read_pdf_info()
            except PdfFormatError:
                self.close()
                raise
        else:
            self.file_size_total = self.file_size_this = 0
            self.root = PdfDict()
            self.root_ref = None
            self.info = PdfDict()
            self.info_ref = None
            self.page_tree_root = PdfDict()
            self.pages: list[IndirectReference] = []
            self.orig_pages: list[IndirectReference] = []
            self.pages_ref = None
            self.last_xref_section_offset = None
            self.trailer_dict: dict[bytes, Any] = {}
            self.xref_table = XrefTable()
        self.xref_table.reading_finished = True
        if f:
            self.seek_end()

    def __enter__(self) -> PdfParser:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def start_writing(self) -> None:
        self.close_buf()
        self.seek_end()

    def close_buf(self) -> None:
        if isinstance(self.buf, memoryview):
            self.buf.release()
        elif isinstance(self.buf, mmap.mmap):
            self.buf.close()
        self.buf = None

    def close(self) -> None:
        if self.should_close_buf:
            self.close_buf()
        if self.f is not None and self.should_close_file:
            self.f.close()
            self.f = None

    def seek_end(self) -> None:
        assert self.f is not None
        self.f.seek(0, os.SEEK_END)

    def write_header(self) -> None:
        assert self.f is not None
        self.f.write(b"%PDF-1.4\n")

    def write_comment(self, s: str) -> None:
        assert self.f is not None
        self.f.write(f"% {s}\n".encode())

    def write_catalog(self) -> IndirectReference:
        assert self.f is not None
        self.del_root()
        self.root_ref = self.next_object_id(self.f.tell())
        self.pages_ref = self.next_object_id(0)
        self.rewrite_pages()
        self.write_obj(self.root_ref, Type=PdfName(b"Catalog"), Pages=self.pages_ref)
        self.write_obj(
            self.pages_ref,
            Type=PdfName(b"Pages"),
            Count=len(self.pages),
            Kids=self.pages,
        )
        return self.root_ref

    def rewrite_pages(self) -> None:
        pages_tree_nodes_to_delete = []
        for i, page_ref in enumerate(self.orig_pages):
            page_info = self.cached_objects[page_ref]
            del self.xref_table[page_ref.object_id]
            pages_tree_nodes_to_delete.append(page_info[PdfName(b"Parent")])
            if page_ref not in self.pages:
                # the page has been deleted
                continue
            # make dict keys into strings for passing to write_page
            stringified_page_info = {}
            for key, value in page_info.items():
                # key should be a PdfName
                stringified_page_info[key.name_as_str()] = value
            stringified_page_info["Parent"] = self.pages_ref
            new_page_ref = self.write_page(None, **stringified_page_info)
            for j, cur_page_ref in enumerate(self.pages):
                if cur_page_ref == page_ref:
                    # replace the page reference with the new one
                    self.pages[j] = new_page_ref
        # delete redundant Pages tree nodes from xref table
        for pages_tree_node_ref in pages_tree_nodes_to_delete:
            while pages_tree_node_ref:
                pages_tree_node = self.cached_objects[pages_tree_node_ref]
                if pages_tree_node_ref.object_id in self.xref_table:
                    del self.xref_table[pages_tree_node_ref.object_id]
                pages_tree_node_ref = pages_tree_node.get(b"Parent", None)
        self.orig_pages = []

    def write_xref_and_trailer(
        self, new_root_ref: IndirectReference | None = None
    ) -> None:
        assert self.f is not None
        if new_root_ref:
            self.del_root()
            self.root_ref = new_root_ref
        if self.info:
            self.info_ref = self.write_obj(None, self.info)
        start_xref = self.xref_table.write(self.f)
        num_entries = len(self.xref_table)
        trailer_dict: dict[str | bytes, Any] = {
            b"Root": self.root_ref,
            b"Size": num_entries,
        }
        if self.last_xref_section_offset is not None:
            trailer_dict[b"Prev"] = self.last_xref_section_offset
        if self.info:
            trailer_dict[b"Info"] = self.info_ref
        self.last_xref_section_offset = start_xref
        self.f.write(
            b"trailer\n"
            + bytes(PdfDict(trailer_dict))
            + b"\nstartxref\n%d\n%%%%EOF" % start_xref
        )

    def write_page(
        self, ref: int | IndirectReference | None, *objs: Any, **dict_obj: Any
    ) -> IndirectReference:
        obj_ref = self.pages[ref] if isinstance(ref, int) else ref
        if "Type" not in dict_obj:
            dict_obj["Type"] = PdfName(b"Page")
        if "Parent" not in dict_obj:
            dict_obj["Parent"] = self.pages_ref
        return self.write_obj(obj_ref, *objs, **dict_obj)

    def write_obj(
        self, ref: IndirectReference | None, *objs: Any, **dict_obj: Any
    ) -> IndirectReference:
        assert self.f is not None
        f = self.f
        if ref is None:
            ref = self.next_object_id(f.tell())
        else:
            self.xref_table[ref.object_id] = (f.tell(), ref.generation)
        f.write(bytes(IndirectObjectDef(*ref)))
        stream = dict_obj.pop("stream", None)
        if stream is not None:
            dict_obj["Length"] = len(stream)
        if dict_obj:
            f.write(pdf_repr(dict_obj))
        for obj in objs:
            f.write(pdf_repr(obj))
        if stream is not None:
            f.write(b"stream\n")
            f.write(stream)
            f.write(b"\nendstream\n")
        f.write(b"endobj\n")
        return ref

    def del_root(self) -> None:
        if self.root_ref is None:
            return
        del self.xref_table[self.root_ref.object_id]
        del self.xref_table[self.root[b"Pages"].object_id]

    @staticmethod
    def get_buf_from_file(f: IO[bytes]) -> bytes | mmap.mmap:
        if hasattr(f, "getbuffer"):
            return f.getbuffer()
        elif hasattr(f, "getvalue"):
            return f.getvalue()
        else:
            try:
                return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError:  # cannot mmap an empty file
                return b""

    def read_pdf_info(self) -> None:
        assert self.buf is not None
        self.file_size_total = len(self.buf)
        self.file_size_this = self.file_size_total - self.start_offset
        self.read_trailer()
        check_format_condition(
            self.trailer_dict.get(b"Root") is not None, "Root is missing"
        )
        self.root_ref = self.trailer_dict[b"Root"]
        assert self.root_ref is not None
        self.info_ref = self.trailer_dict.get(b"Info", None)
        self.root = PdfDict(self.read_indirect(self.root_ref))
        if self.info_ref is None:
            self.info = PdfDict()
        else:
            self.info = PdfDict(self.read_indirect(self.info_ref))
        check_format_condition(b"Type" in self.root, "/Type missing in Root")
        check_format_condition(
            self.root[b"Type"] == b"Catalog", "/Type in Root is not /Catalog"
        )
        check_format_condition(
            self.root.get(b"Pages") is not None, "/Pages missing in Root"
        )
        check_format_condition(
            isinstance(self.root[b"Pages"], IndirectReference),
            "/Pages in Root is not an indirect reference",
        )
        self.pages_ref = self.root[b"Pages"]
        assert self.pages_ref is not None
        self.page_tree_root = self.read_indirect(self.pages_ref)
        self.pages = self.linearize_page_tree(self.page_tree_root)
        # save the original list of page references
        # in case the user modifies, adds or deletes some pages
        # and we need to rewrite the pages and their list
        self.orig_pages = self.pages[:]

    def next_object_id(self, offset: int | None = None) -> IndirectReference:
        try:
            # TODO: support reuse of deleted objects
            reference = IndirectReference(max(self.xref_table.keys()) + 1, 0)
        except ValueError:
            reference = IndirectReference(1, 0)
        if offset is not None:
            self.xref_table[reference.object_id] = (offset, 0)
        return reference

    delimiter = rb"[][()<>{}/%]"
    delimiter_or_ws = rb"[][()<>{}/%\000\011\012\014\015\040]"
    whitespace = rb"[\000\011\012\014\015\040]"
    whitespace_or_hex = rb"[\000\011\012\014\015\0400-9a-fA-F]"
    whitespace_optional = whitespace + b"*"
    whitespace_mandatory = whitespace + b"+"
    # No "\012" aka "\n" or "\015" aka "\r":
    whitespace_optional_no_nl = rb"[\000\011\014\040]*"
    newline_only = rb"[\r\n]+"
    newline = whitespace_optional_no_nl + newline_only + whitespace_optional_no_nl
    re_trailer_end = re.compile(
        whitespace_mandatory
        + rb"trailer"
        + whitespace_optional
        + rb"<<(.*>>)"
        + newline
        + rb"startxref"
        + newline
        + rb"([0-9]+)"
        + newline
        + rb"%%EOF"
        + whitespace_optional
        + rb"$",
        re.DOTALL,
    )
    re_trailer_prev = re.compile(
        whitespace_optional
        + rb"trailer"
        + whitespace_optional
        + rb"<<(.*?>>)"
        + newline
        + rb"startxref"
        + newline
        + rb"([0-9]+)"
        + newline
        + rb"%%EOF"
        + whitespace_optional,
        re.DOTALL,
    )

    def read_trailer(self) -> None:
        assert self.buf is not None
        search_start_offset = len(self.buf) - 16384
        if search_start_offset < self.start_offset:
            search_start_offset = self.start_offset
        m = self.re_trailer_end.search(self.buf, search_start_offset)
        check_format_condition(m is not None, "trailer end not found")
        # make sure we found the LAST trailer
        last_match = m
        while m:
            last_match = m
            m = self.re_trailer_end.search(self.buf, m.start() + 16)
        if not m:
            m = last_match
        assert m is not None
        trailer_data = m.group(1)
        self.last_xref_section_offset = int(m.group(2))
        self.trailer_dict = self.interpret_trailer(trailer_data)
        self.xref_table = XrefTable()
        self.read_xref_table(xref_section_offset=self.last_xref_section_offset)
        if b"Prev" in self.trailer_dict:
            self.read_prev_trailer(self.trailer_dict[b"Prev"])

    def read_prev_trailer(
        self, xref_section_offset: int, processed_offsets: list[int] | None = None
    ) -> None:
        assert self.buf is not None
        trailer_offset = self.read_xref_table(xref_section_offset=xref_section_offset)
        m = self.re_trailer_prev.search(
            self.buf[trailer_offset : trailer_offset + 16384]
        )
        check_format_condition(m is not None, "previous trailer not found")
        assert m is not None
        trailer_data = m.group(1)
        check_format_condition(
            int(m.group(2)) == xref_section_offset,
            "xref section offset in previous trailer doesn't match what was expected",
        )
        trailer_dict = self.interpret_trailer(trailer_data)
        if b"Prev" in trailer_dict:
            if processed_offsets is None:
                processed_offsets = []
            processed_offsets.append(xref_section_offset)
            check_format_condition(
                trailer_dict[b"Prev"] not in processed_offsets, "trailer loop found"
            )
            self.read_prev_trailer(trailer_dict[b"Prev"], processed_offsets)

    re_whitespace_optional = re.compile(whitespace_optional)
    re_name = re.compile(
        whitespace_optional
        + rb"/([!-$&'*-.0-;=?-Z\\^-z|~]+)(?="
        + delimiter_or_ws
        + rb")"
    )
    re_dict_start = re.compile(whitespace_optional + rb"<<")
    re_dict_end = re.compile(whitespace_optional + rb">>" + whitespace_optional)

    @classmethod
    def interpret_trailer(cls, trailer_data: bytes) -> dict[bytes, Any]:
        trailer = {}
        offset = 0
        while True:
            m = cls.re_name.match(trailer_data, offset)
            if not m:
                m = cls.re_dict_end.match(trailer_data, offset)
                check_format_condition(
                    m is not None and m.end() == len(trailer_data),
                    "name not found in trailer, remaining data: "
                    + repr(trailer_data[offset:]),
                )
                break
            key = cls.interpret_name(m.group(1))
            assert isinstance(key, bytes)
            value, value_offset = cls.get_value(trailer_data, m.end())
            trailer[key] = value
            if value_offset is None:
                break
            offset = value_offset
        check_format_condition(
            b"Size" in trailer and isinstance(trailer[b"Size"], int),
            "/Size not in trailer or not an integer",
        )
        check_format_condition(
            b"Root" in trailer and isinstance(trailer[b"Root"], IndirectReference),
            "/Root not in trailer or not an indirect reference",
        )
        return trailer

    re_hashes_in_name = re.compile(rb"([^#]*)(#([0-9a-fA-F]{2}))?")

    @classmethod
    def interpret_name(cls, raw: bytes, as_text: bool = False) -> str | bytes:
        name = b""
        for m in cls.re_hashes_in_name.finditer(raw):
            if m.group(3):
                name += m.group(1) + bytearray.fromhex(m.group(3).decode("us-ascii"))
            else:
                name += m.group(1)
        if as_text:
            return name.decode("utf-8")
        else:
            return bytes(name)

    re_null = re.compile(whitespace_optional + rb"null(?=" + delimiter_or_ws + rb")")
    re_true = re.compile(whitespace_optional + rb"true(?=" + delimiter_or_ws + rb")")
    re_false = re.compile(whitespace_optional + rb"false(?=" + delimiter_or_ws + rb")")
    re_int = re.compile(
        whitespace_optional + rb"([-+]?[0-9]+)(?=" + delimiter_or_ws + rb")"
    )
    re_real = re.compile(
        whitespace_optional
        + rb"([-+]?([0-9]+\.[0-9]*|[0-9]*\.[0-9]+))(?="
        + delimiter_or_ws
        + rb")"
    )
    re_array_start = re.compile(whitespace_optional + rb"\[")
    re_array_end = re.compile(whitespace_optional + rb"]")
    re_string_hex = re.compile(
        whitespace_optional + rb"<(" + whitespace_or_hex + rb"*)>"
    )
    re_string_lit = re.compile(whitespace_optional + rb"\(")
    re_indirect_reference = re.compile(
        whitespace_optional
        + rb"([-+]?[0-9]+)"
        + whitespace_mandatory
        + rb"([-+]?[0-9]+)"
        + whitespace_mandatory
        + rb"R(?="
        + delimiter_or_ws
        + rb")"
    )
    re_indirect_def_start = re.compile(
        whitespace_optional
        + rb"([-+]?[0-9]+)"
        + whitespace_mandatory
        + rb"([-+]?[0-9]+)"
        + whitespace_mandatory
        + rb"obj(?="
        + delimiter_or_ws
        + rb")"
    )
    re_indirect_def_end = re.compile(
        whitespace_optional + rb"endobj(?=" + delimiter_or_ws + rb")"
    )
    re_comment = re.compile(
        rb"(" + whitespace_optional + rb"%[^\r\n]*" + newline + rb")*"
    )
    re_stream_start = re.compile(whitespace_optional + rb"stream\r?\n")
    re_stream_end = re.compile(
        whitespace_optional + rb"endstream(?=" + delimiter_or_ws + rb")"
    )

    @classmethod
    def get_value(
        cls,
        data: bytes | bytearray | memoryview | mmap.mmap,
        offset: int,
        expect_indirect: IndirectReference | None = None,
        max_nesting: int = -1,
    ) -> tuple[Any, int | None]:
        if max_nesting == 0:
            return None, None
        m = cls.re_comment.match(data, offset)
        if m:
            offset = m.end()
        m = cls.re_indirect_def_start.match(data, offset)
        if m:
            check_format_condition(
                int(m.group(1)) > 0,
                "indirect object definition: object ID must be greater than 0",
            )
            check_format_condition(
                int(m.group(2)) >= 0,
                "indirect object definition: generation must be non-negative",
            )
            check_format_condition(
                expect_indirect is None
                or expect_indirect
                == IndirectReference(int(m.group(1)), int(m.group(2))),
                "indirect object definition different than expected",
            )
            object, object_offset = cls.get_value(
                data, m.end(), max_nesting=max_nesting - 1
            )
            if object_offset is None:
                return object, None
            m = cls.re_indirect_def_end.match(data, object_offset)
            check_format_condition(
                m is not None, "indirect object definition end not found"
            )
            assert m is not None
            return object, m.end()
        check_format_condition(
            not expect_indirect, "indirect object definition not found"
        )
        m = cls.re_indirect_reference.match(data, offset)
        if m:
            check_format_condition(
                int(m.group(1)) > 0,
                "indirect object reference: object ID must be greater than 0",
            )
            check_format_condition(
                int(m.group(2)) >= 0,
                "indirect object reference: generation must be non-negative",
            )
            return IndirectReference(int(m.group(1)), int(m.group(2))), m.end()
        m = cls.re_dict_start.match(data, offset)
        if m:
            offset = m.end()
            result: dict[Any, Any] = {}
            m = cls.re_dict_end.match(data, offset)
            current_offset: int | None = offset
            while not m:
                assert current_offset is not None
                key, current_offset = cls.get_value(
                    data, current_offset, max_nesting=max_nesting - 1
                )
                if current_offset is None:
                    return result, None
                value, current_offset = cls.get_value(
                    data, current_offset, max_nesting=max_nesting - 1
                )
                result[key] = value
                if current_offset is None:
                    return result, None
                m = cls.re_dict_end.match(data, current_offset)
            current_offset = m.end()
            m = cls.re_stream_start.match(data, current_offset)
            if m:
                stream_len = result.get(b"Length")
                if stream_len is None or not isinstance(stream_len, int):
                    msg = f"bad or missing Length in stream dict ({stream_len})"
                    raise PdfFormatError(msg)
                stream_data = bytes(data[m.end() : m.end() + stream_len])
                m = cls.re_stream_end.match(data, m.end() + stream_len)
                check_format_condition(m is not None, "stream end not found")
                assert m is not None
                current_offset = m.end()
                return PdfStream(PdfDict(result), stream_data), current_offset
            return PdfDict(result), current_offset
        m = cls.re_array_start.match(data, offset)
        if m:
            offset = m.end()
            results = []
            m = cls.re_array_end.match(data, offset)
            current_offset = offset
            while not m:
                assert current_offset is not None
                value, current_offset = cls.get_value(
                    data, current_offset, max_nesting=max_nesting - 1
                )
                results.append(value)
                if current_offset is None:
                    return results, None
                m = cls.re_array_end.match(data, current_offset)
            return results, m.end()
        m = cls.re_null.match(data, offset)
        if m:
            return None, m.end()
        m = cls.re_true.match(data, offset)
        if m:
            return True, m.end()
        m = cls.re_false.match(data, offset)
        if m:
            return False, m.end()
        m = cls.re_name.match(data, offset)
        if m:
            return PdfName(cls.interpret_name(m.group(1))), m.end()
        m = cls.re_int.match(data, offset)
        if m:
            return int(m.group(1)), m.end()
        m = cls.re_real.match(data, offset)
        if m:
            # XXX Decimal instead of float???
            return float(m.group(1)), m.end()
        m = cls.re_string_hex.match(data, offset)
        if m:
            # filter out whitespace
            hex_string = bytearray(
                b for b in m.group(1) if b in b"0123456789abcdefABCDEF"
            )
            if len(hex_string) % 2 == 1:
                # append a 0 if the length is not even - yes, at the end
                hex_string.append(ord(b"0"))
            return bytearray.fromhex(hex_string.decode("us-ascii")), m.end()
        m = cls.re_string_lit.match(data, offset)
        if m:
            return cls.get_literal_string(data, m.end())
        # return None, offset  # fallback (only for debugging)
        msg = f"unrecognized object: {repr(data[offset : offset + 32])}"
        raise PdfFormatError(msg)

    re_lit_str_token = re.compile(
        rb"(\\[nrtbf()\\])|(\\[0-9]{1,3})|(\\(\r\n|\r|\n))|(\r\n|\r|\n)|(\()|(\))"
    )
    escaped_chars = {
        b"n": b"\n",
        b"r": b"\r",
        b"t": b"\t",
        b"b": b"\b",
        b"f": b"\f",
        b"(": b"(",
        b")": b")",
        b"\\": b"\\",
        ord(b"n"): b"\n",
        ord(b"r"): b"\r",
        ord(b"t"): b"\t",
        ord(b"b"): b"\b",
        ord(b"f"): b"\f",
        ord(b"("): b"(",
        ord(b")"): b")",
        ord(b"\\"): b"\\",
    }

    @classmethod
    def get_literal_string(
        cls, data: bytes | bytearray | memoryview | mmap.mmap, offset: int
    ) -> tuple[bytes, int]:
        nesting_depth = 0
        result = bytearray()
        for m in cls.re_lit_str_token.finditer(data, offset):
            result.extend(data[offset : m.start()])
            if m.group(1):
                result.extend(cls.escaped_chars[m.group(1)[1]])
            elif m.group(2):
                result.append(int(m.group(2)[1:], 8))
            elif m.group(3):
                pass
            elif m.group(5):
                result.extend(b"\n")
            elif m.group(6):
                result.extend(b"(")
                nesting_depth += 1
            elif m.group(7):
                if nesting_depth == 0:
                    return bytes(result), m.end()
                result.extend(b")")
                nesting_depth -= 1
            offset = m.end()
        msg = "unfinished literal string"
        raise PdfFormatError(msg)

    re_xref_section_start = re.compile(whitespace_optional + rb"xref" + newline)
    re_xref_subsection_start = re.compile(
        whitespace_optional
        + rb"([0-9]+)"
        + whitespace_mandatory
        + rb"([0-9]+)"
        + whitespace_optional
        + newline_only
    )
    re_xref_entry = re.compile(rb"([0-9]{10}) ([0-9]{5}) ([fn])( \r| \n|\r\n)")

    def read_xref_table(self, xref_section_offset: int) -> int:
        assert self.buf is not None
        subsection_found = False
        m = self.re_xref_section_start.match(
            self.buf, xref_section_offset + self.start_offset
        )
        check_format_condition(m is not None, "xref section start not found")
        assert m is not None
        offset = m.end()
        while True:
            m = self.re_xref_subsection_start.match(self.buf, offset)
            if not m:
                check_format_condition(
                    subsection_found, "xref subsection start not found"
                )
                break
            subsection_found = True
            offset = m.end()
            first_object = int(m.group(1))
            num_objects = int(m.group(2))
            for i in range(first_object, first_object + num_objects):
                m = self.re_xref_entry.match(self.buf, offset)
                check_format_condition(m is not None, "xref entry not found")
                assert m is not None
                offset = m.end()
                is_free = m.group(3) == b"f"
                if not is_free:
                    generation = int(m.group(2))
                    new_entry = (int(m.group(1)), generation)
                    if i not in self.xref_table:
                        self.xref_table[i] = new_entry
        return offset

    def read_indirect(self, ref: IndirectReference, max_nesting: int = -1) -> Any:
        offset, generation = self.xref_table[ref[0]]
        check_format_condition(
            generation == ref[1],
            f"expected to find generation {ref[1]} for object ID {ref[0]} in xref "
            f"table, instead found generation {generation} at offset {offset}",
        )
        assert self.buf is not None
        value = self.get_value(
            self.buf,
            offset + self.start_offset,
            expect_indirect=IndirectReference(*ref),
            max_nesting=max_nesting,
        )[0]
        self.cached_objects[ref] = value
        return value

    def linearize_page_tree(
        self, node: PdfDict | None = None
    ) -> list[IndirectReference]:
        page_node = node if node is not None else self.page_tree_root
        check_format_condition(
            page_node[b"Type"] == b"Pages", "/Type of page tree node is not /Pages"
        )
        pages = []
        for kid in page_node[b"Kids"]:
            kid_object = self.read_indirect(kid)
            if kid_object[b"Type"] == b"Page":
                pages.append(kid)
            else:
                pages.extend(self.linearize_page_tree(node=kid_object))
        return pages
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# Binary input/output support routines.
#
# Copyright (c) 1997-2003 by Secret Labs AB
# Copyright (c) 1995-2003 by Fredrik Lundh
# Copyright (c) 2012 by Brian Crowell
#
# See the README file for information on usage and redistribution.
#


"""Binary input/output support routines."""

from __future__ import annotations

from struct import pack, unpack_from


def i8(c: bytes) -> int:
    return c[0]


def o8(i: int) -> bytes:
    return bytes((i & 255,))


# Input, le = little endian, be = big endian
def i16le(c: bytes, o: int = 0) -> int:
    """
    Converts a 2-bytes (16 bits) string to an unsigned integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<H", c, o)[0]


def si16le(c: bytes, o: int = 0) -> int:
    """
    Converts a 2-bytes (16 bits) string to a signed integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<h", c, o)[0]


def si16be(c: bytes, o: int = 0) -> int:
    """
    Converts a 2-bytes (16 bits) string to a signed integer, big endian.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from(">h", c, o)[0]


def i32le(c: bytes, o: int = 0) -> int:
    """
    Converts a 4-bytes (32 bits) string to an unsigned integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<I", c, o)[0]


def si32le(c: bytes, o: int = 0) -> int:
    """
    Converts a 4-bytes (32 bits) string to a signed integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<i", c, o)[0]


def si32be(c: bytes, o: int = 0) -> int:
    """
    Converts a 4-bytes (32 bits) string to a signed integer, big endian.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from(">i", c, o)[0]


def i16be(c: bytes, o: int = 0) -> int:
    return unpack_from(">H", c, o)[0]


def i32be(c: bytes, o: int = 0) -> int:
    return unpack_from(">I", c, o)[0]


# Output, le = little endian, be = big endian
def o16le(i: int) -> bytes:
    return pack("<H", i)


def o32le(i: int) -> bytes:
    return pack("<I", i)


def o16be(i: int) -> bytes:
    return pack(">H", i)


def o32be(i: int) -> bytes:
    return pack(">I", i)
<|endoftext|>"""
A Pillow loader for .ftc and .ftu files (FTEX)
Jerome Leclanche <jerome@leclan.ch>

The contents of this file are hereby released in the public domain (CC0)
Full text of the CC0 license:
  https://creativecommons.org/publicdomain/zero/1.0/

Independence War 2: Edge Of Chaos - Texture File Format - 16 October 2001

The textures used for 3D objects in Independence War 2: Edge Of Chaos are in a
packed custom format called FTEX. This file format uses file extensions FTC
and FTU.
* FTC files are compressed textures (using standard texture compression).
* FTU files are not compressed.
Texture File Format
The FTC and FTU texture files both use the same format. This
has the following structure:
{header}
{format_directory}
{data}
Where:
{header} = {
    u32:magic,
    u32:version,
    u32:width,
    u32:height,
    u32:mipmap_count,
    u32:format_count
}

* The "magic" number is "FTEX".
* "width" and "height" are the dimensions of the texture.
* "mipmap_count" is the number of mipmaps in the texture.
* "format_count" is the number of texture formats (different versions of the
same texture) in this file.

{format_directory} = format_count * { u32:format, u32:where }

The format value is 0 for DXT1 compressed textures and 1 for 24-bit RGB
uncompressed textures.
The texture data for a format starts at the position "where" in the file.

Each set of texture data in the file has the following structure:
{data} = format_count * { u32:mipmap_size, mipmap_size * { u8 } }
* "mipmap_size" is the number of bytes in that mip level. For compressed
textures this is the size of the texture data compressed with DXT1. For 24 bit
uncompressed textures, this is 3 * width * height. Following this are the image
bytes for that mipmap level.

Note: All data is stored in little-Endian (Intel) byte order.
"""

from __future__ import annotations

import struct
from enum import IntEnum
from io import BytesIO

from . import Image, ImageFile

MAGIC = b"FTEX"


class Format(IntEnum):
    DXT1 = 0
    UNCOMPRESSED = 1


class FtexImageFile(ImageFile.ImageFile):
    format = "FTEX"
    format_description = "Texture File Format (IW2:EOC)"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(4)):
            msg = "not an FTEX file"
            raise SyntaxError(msg)
        struct.unpack("<i", self.fp.read(4))  # version
        self._size = struct.unpack("<2i", self.fp.read(8))
        mipmap_count, format_count = struct.unpack("<2i", self.fp.read(8))

        # Only support single-format files.
        # I don't know of any multi-format file.
        assert format_count == 1

        format, where = struct.unpack("<2i", self.fp.read(8))
        self.fp.seek(where)
        (mipmap_size,) = struct.unpack("<i", self.fp.read(4))

        data = self.fp.read(mipmap_size)

        if format == Format.DXT1:
            self._mode = "RGBA"
            self.tile = [ImageFile._Tile("bcn", (0, 0) + self.size, 0, (1,))]
        elif format == Format.UNCOMPRESSED:
            self._mode = "RGB"
            self.tile = [ImageFile._Tile("raw", (0, 0) + self.size, 0, "RGB")]
        else:
            msg = f"Invalid texture compression format: {repr(format)}"
            raise ValueError(msg)

        self.fp.close()
        self.fp = BytesIO(data)

    def load_seek(self, pos: int) -> None:
        pass


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(MAGIC)


Image.register_open(FtexImageFile.format, FtexImageFile, _accept)
Image.register_extensions(FtexImageFile.format, [".ftc", ".ftu"])
<|endoftext|>"""
A Pillow plugin for .dds files (S3TC-compressed aka DXTC)
Jerome Leclanche <jerome@leclan.ch>

Documentation:
https://web.archive.org/web/20170802060935/http://oss.sgi.com/projects/ogl-sample/registry/EXT/texture_compression_s3tc.txt

The contents of this file are hereby released in the public domain (CC0)
Full text of the CC0 license:
https://creativecommons.org/publicdomain/zero/1.0/
"""

from __future__ import annotations

import struct
import sys
from enum import IntEnum, IntFlag
from typing import IO

from . import Image, ImageFile, ImagePalette
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o32le as o32

# Magic ("DDS ")
DDS_MAGIC = 0x20534444


# DDS flags
class DDSD(IntFlag):
    CAPS = 0x1
    HEIGHT = 0x2
    WIDTH = 0x4
    PITCH = 0x8
    PIXELFORMAT = 0x1000
    MIPMAPCOUNT = 0x20000
    LINEARSIZE = 0x80000
    DEPTH = 0x800000


# DDS caps
class DDSCAPS(IntFlag):
    COMPLEX = 0x8
    TEXTURE = 0x1000
    MIPMAP = 0x400000


class DDSCAPS2(IntFlag):
    CUBEMAP = 0x200
    CUBEMAP_POSITIVEX = 0x400
    CUBEMAP_NEGATIVEX = 0x800
    CUBEMAP_POSITIVEY = 0x1000
    CUBEMAP_NEGATIVEY = 0x2000
    CUBEMAP_POSITIVEZ = 0x4000
    CUBEMAP_NEGATIVEZ = 0x8000
    VOLUME = 0x200000


# Pixel Format
class DDPF(IntFlag):
    ALPHAPIXELS = 0x1
    ALPHA = 0x2
    FOURCC = 0x4
    PALETTEINDEXED8 = 0x20
    RGB = 0x40
    LUMINANCE = 0x20000


# dxgiformat.h
class DXGI_FORMAT(IntEnum):
    UNKNOWN = 0
    R32G32B32A32_TYPELESS = 1
    R32G32B32A32_FLOAT = 2
    R32G32B32A32_UINT = 3
    R32G32B32A32_SINT = 4
    R32G32B32_TYPELESS = 5
    R32G32B32_FLOAT = 6
    R32G32B32_UINT = 7
    R32G32B32_SINT = 8
    R16G16B16A16_TYPELESS = 9
    R16G16B16A16_FLOAT = 10
    R16G16B16A16_UNORM = 11
    R16G16B16A16_UINT = 12
    R16G16B16A16_SNORM = 13
    R16G16B16A16_SINT = 14
    R32G32_TYPELESS = 15
    R32G32_FLOAT = 16
    R32G32_UINT = 17
    R32G32_SINT = 18
    R32G8X24_TYPELESS = 19
    D32_FLOAT_S8X24_UINT = 20
    R32_FLOAT_X8X24_TYPELESS = 21
    X32_TYPELESS_G8X24_UINT = 22
    R10G10B10A2_TYPELESS = 23
    R10G10B10A2_UNORM = 24
    R10G10B10A2_UINT = 25
    R11G11B10_FLOAT = 26
    R8G8B8A8_TYPELESS = 27
    R8G8B8A8_UNORM = 28
    R8G8B8A8_UNORM_SRGB = 29
    R8G8B8A8_UINT = 30
    R8G8B8A8_SNORM = 31
    R8G8B8A8_SINT = 32
    R16G16_TYPELESS = 33
    R16G16_FLOAT = 34
    R16G16_UNORM = 35
    R16G16_UINT = 36
    R16G16_SNORM = 37
    R16G16_SINT = 38
    R32_TYPELESS = 39
    D32_FLOAT = 40
    R32_FLOAT = 41
    R32_UINT = 42
    R32_SINT = 43
    R24G8_TYPELESS = 44
    D24_UNORM_S8_UINT = 45
    R24_UNORM_X8_TYPELESS = 46
    X24_TYPELESS_G8_UINT = 47
    R8G8_TYPELESS = 48
    R8G8_UNORM = 49
    R8G8_UINT = 50
    R8G8_SNORM = 51
    R8G8_SINT = 52
    R16_TYPELESS = 53
    R16_FLOAT = 54
    D16_UNORM = 55
    R16_UNORM = 56
    R16_UINT = 57
    R16_SNORM = 58
    R16_SINT = 59
    R8_TYPELESS = 60
    R8_UNORM = 61
    R8_UINT = 62
    R8_SNORM = 63
    R8_SINT = 64
    A8_UNORM = 65
    R1_UNORM = 66
    R9G9B9E5_SHAREDEXP = 67
    R8G8_B8G8_UNORM = 68
    G8R8_G8B8_UNORM = 69
    BC1_TYPELESS = 70
    BC1_UNORM = 71
    BC1_UNORM_SRGB = 72
    BC2_TYPELESS = 73
    BC2_UNORM = 74
    BC2_UNORM_SRGB = 75
    BC3_TYPELESS = 76
    BC3_UNORM = 77
    BC3_UNORM_SRGB = 78
    BC4_TYPELESS = 79
    BC4_UNORM = 80
    BC4_SNORM = 81
    BC5_TYPELESS = 82
    BC5_UNORM = 83
    BC5_SNORM = 84
    B5G6R5_UNORM = 85
    B5G5R5A1_UNORM = 86
    B8G8R8A8_UNORM = 87
    B8G8R8X8_UNORM = 88
    R10G10B10_XR_BIAS_A2_UNORM = 89
    B8G8R8A8_TYPELESS = 90
    B8G8R8A8_UNORM_SRGB = 91
    B8G8R8X8_TYPELESS = 92
    B8G8R8X8_UNORM_SRGB = 93
    BC6H_TYPELESS = 94
    BC6H_UF16 = 95
    BC6H_SF16 = 96
    BC7_TYPELESS = 97
    BC7_UNORM = 98
    BC7_UNORM_SRGB = 99
    AYUV = 100
    Y410 = 101
    Y416 = 102
    NV12 = 103
    P010 = 104
    P016 = 105
    OPAQUE_420 = 106
    YUY2 = 107
    Y210 = 108
    Y216 = 109
    NV11 = 110
    AI44 = 111
    IA44 = 112
    P8 = 113
    A8P8 = 114
    B4G4R4A4_UNORM = 115
    P208 = 130
    V208 = 131
    V408 = 132
    SAMPLER_FEEDBACK_MIN_MIP_OPAQUE = 189
    SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE = 190


class D3DFMT(IntEnum):
    UNKNOWN = 0
    R8G8B8 = 20
    A8R8G8B8 = 21
    X8R8G8B8 = 22
    R5G6B5 = 23
    X1R5G5B5 = 24
    A1R5G5B5 = 25
    A4R4G4B4 = 26
    R3G3B2 = 27
    A8 = 28
    A8R3G3B2 = 29
    X4R4G4B4 = 30
    A2B10G10R10 = 31
    A8B8G8R8 = 32
    X8B8G8R8 = 33
    G16R16 = 34
    A2R10G10B10 = 35
    A16B16G16R16 = 36
    A8P8 = 40
    P8 = 41
    L8 = 50
    A8L8 = 51
    A4L4 = 52
    V8U8 = 60
    L6V5U5 = 61
    X8L8V8U8 = 62
    Q8W8V8U8 = 63
    V16U16 = 64
    A2W10V10U10 = 67
    D16_LOCKABLE = 70
    D32 = 71
    D15S1 = 73
    D24S8 = 75
    D24X8 = 77
    D24X4S4 = 79
    D16 = 80
    D32F_LOCKABLE = 82
    D24FS8 = 83
    D32_LOCKABLE = 84
    S8_LOCKABLE = 85
    L16 = 81
    VERTEXDATA = 100
    INDEX16 = 101
    INDEX32 = 102
    Q16W16V16U16 = 110
    R16F = 111
    G16R16F = 112
    A16B16G16R16F = 113
    R32F = 114
    G32R32F = 115
    A32B32G32R32F = 116
    CxV8U8 = 117
    A1 = 118
    A2B10G10R10_XR_BIAS = 119
    BINARYBUFFER = 199

    UYVY = i32(b"UYVY")
    R8G8_B8G8 = i32(b"RGBG")
    YUY2 = i32(b"YUY2")
    G8R8_G8B8 = i32(b"GRGB")
    DXT1 = i32(b"DXT1")
    DXT2 = i32(b"DXT2")
    DXT3 = i32(b"DXT3")
    DXT4 = i32(b"DXT4")
    DXT5 = i32(b"DXT5")
    DX10 = i32(b"DX10")
    BC4S = i32(b"BC4S")
    BC4U = i32(b"BC4U")
    BC5S = i32(b"BC5S")
    BC5U = i32(b"BC5U")
    ATI1 = i32(b"ATI1")
    ATI2 = i32(b"ATI2")
    MULTI2_ARGB8 = i32(b"MET1")


# Backward compatibility layer
module = sys.modules[__name__]
for item in DDSD:
    assert item.name is not None
    setattr(module, f"DDSD_{item.name}", item.value)
for item1 in DDSCAPS:
    assert item1.name is not None
    setattr(module, f"DDSCAPS_{item1.name}", item1.value)
for item2 in DDSCAPS2:
    assert item2.name is not None
    setattr(module, f"DDSCAPS2_{item2.name}", item2.value)
for item3 in DDPF:
    assert item3.name is not None
    setattr(module, f"DDPF_{item3.name}", item3.value)

DDS_FOURCC = DDPF.FOURCC
DDS_RGB = DDPF.RGB
DDS_RGBA = DDPF.RGB | DDPF.ALPHAPIXELS
DDS_LUMINANCE = DDPF.LUMINANCE
DDS_LUMINANCEA = DDPF.LUMINANCE | DDPF.ALPHAPIXELS
DDS_ALPHA = DDPF.ALPHA
DDS_PAL8 = DDPF.PALETTEINDEXED8

DDS_HEADER_FLAGS_TEXTURE = DDSD.CAPS | DDSD.HEIGHT | DDSD.WIDTH | DDSD.PIXELFORMAT
DDS_HEADER_FLAGS_MIPMAP = DDSD.MIPMAPCOUNT
DDS_HEADER_FLAGS_VOLUME = DDSD.DEPTH
DDS_HEADER_FLAGS_PITCH = DDSD.PITCH
DDS_HEADER_FLAGS_LINEARSIZE = DDSD.LINEARSIZE

DDS_HEIGHT = DDSD.HEIGHT
DDS_WIDTH = DDSD.WIDTH

DDS_SURFACE_FLAGS_TEXTURE = DDSCAPS.TEXTURE
DDS_SURFACE_FLAGS_MIPMAP = DDSCAPS.COMPLEX | DDSCAPS.MIPMAP
DDS_SURFACE_FLAGS_CUBEMAP = DDSCAPS.COMPLEX

DDS_CUBEMAP_POSITIVEX = DDSCAPS2.CUBEMAP | DDSCAPS2.CUBEMAP_POSITIVEX
DDS_CUBEMAP_NEGATIVEX = DDSCAPS2.CUBEMAP | DDSCAPS2.CUBEMAP_NEGATIVEX
DDS_CUBEMAP_POSITIVEY = DDSCAPS2.CUBEMAP | DDSCAPS2.CUBEMAP_POSITIVEY
DDS_CUBEMAP_NEGATIVEY = DDSCAPS2.CUBEMAP | DDSCAPS2.CUBEMAP_NEGATIVEY
DDS_CUBEMAP_POSITIVEZ = DDSCAPS2.CUBEMAP | DDSCAPS2.CUBEMAP_POSITIVEZ
DDS_CUBEMAP_NEGATIVEZ = DDSCAPS2.CUBEMAP | DDSCAPS2.CUBEMAP_NEGATIVEZ

DXT1_FOURCC = D3DFMT.DXT1
DXT3_FOURCC = D3DFMT.DXT3
DXT5_FOURCC = D3DFMT.DXT5

DXGI_FORMAT_R8G8B8A8_TYPELESS = DXGI_FORMAT.R8G8B8A8_TYPELESS
DXGI_FORMAT_R8G8B8A8_UNORM = DXGI_FORMAT.R8G8B8A8_UNORM
DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = DXGI_FORMAT.R8G8B8A8_UNORM_SRGB
DXGI_FORMAT_BC5_TYPELESS = DXGI_FORMAT.BC5_TYPELESS
DXGI_FORMAT_BC5_UNORM = DXGI_FORMAT.BC5_UNORM
DXGI_FORMAT_BC5_SNORM = DXGI_FORMAT.BC5_SNORM
DXGI_FORMAT_BC6H_UF16 = DXGI_FORMAT.BC6H_UF16
DXGI_FORMAT_BC6H_SF16 = DXGI_FORMAT.BC6H_SF16
DXGI_FORMAT_BC7_TYPELESS = DXGI_FORMAT.BC7_TYPELESS
DXGI_FORMAT_BC7_UNORM = DXGI_FORMAT.BC7_UNORM
DXGI_FORMAT_BC7_UNORM_SRGB = DXGI_FORMAT.BC7_UNORM_SRGB


class DdsImageFile(ImageFile.ImageFile):
    format = "DDS"
    format_description = "DirectDraw Surface"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(4)):
            msg = "not a DDS file"
            raise SyntaxError(msg)
        (header_size,) = struct.unpack("<I", self.fp.read(4))
        if header_size != 124:
            msg = f"Unsupported header size {repr(header_size)}"
            raise OSError(msg)
        header = self.fp.read(header_size - 4)
        if len(header) != 120:
            msg = f"Incomplete header: {len(header)} bytes"
            raise OSError(msg)

        flags, height, width = struct.unpack("<3I", header[:12])
        self._size = (width, height)
        extents = (0, 0) + self.size

        pitch, depth, mipmaps = struct.unpack("<3I", header[12:24])
        struct.unpack("<11I", header[24:68])  # reserved

        # pixel format
        pfsize, pfflags, fourcc, bitcount = struct.unpack("<4I", header[68:84])
        n = 0
        rawmode = None
        if pfflags & DDPF.RGB:
            # Texture contains uncompressed RGB data
            if pfflags & DDPF.ALPHAPIXELS:
                self._mode = "RGBA"
                mask_count = 4
            else:
                self._mode = "RGB"
                mask_count = 3

            masks = struct.unpack(f"<{mask_count}I", header[84 : 84 + mask_count * 4])
            self.tile = [ImageFile._Tile("dds_rgb", extents, 0, (bitcount, masks))]
            return
        elif pfflags & DDPF.LUMINANCE:
            if bitcount == 8:
                self._mode = "L"
            elif bitcount == 16 and pfflags & DDPF.ALPHAPIXELS:
                self._mode = "LA"
            else:
                msg = f"Unsupported bitcount {bitcount} for {pfflags}"
                raise OSError(msg)
        elif pfflags & DDPF.PALETTEINDEXED8:
            self._mode = "P"
            self.palette = ImagePalette.raw("RGBA", self.fp.read(1024))
            self.palette.mode = "RGBA"
        elif pfflags & DDPF.FOURCC:
            offset = header_size + 4
            if fourcc == D3DFMT.DXT1:
                self._mode = "RGBA"
                self.pixel_format = "DXT1"
                n = 1
            elif fourcc == D3DFMT.DXT3:
                self._mode = "RGBA"
                self.pixel_format = "DXT3"
                n = 2
            elif fourcc == D3DFMT.DXT5:
                self._mode = "RGBA"
                self.pixel_format = "DXT5"
                n = 3
            elif fourcc in (D3DFMT.BC4U, D3DFMT.ATI1):
                self._mode = "L"
                self.pixel_format = "BC4"
                n = 4
            elif fourcc == D3DFMT.BC5S:
                self._mode = "RGB"
                self.pixel_format = "BC5S"
                n = 5
            elif fourcc in (D3DFMT.BC5U, D3DFMT.ATI2):
                self._mode = "RGB"
                self.pixel_format = "BC5"
                n = 5
            elif fourcc == D3DFMT.DX10:
                offset += 20
                # ignoring flags which pertain to volume textures and cubemaps
                (dxgi_format,) = struct.unpack("<I", self.fp.read(4))
                self.fp.read(16)
                if dxgi_format in (
                    DXGI_FORMAT.BC1_UNORM,
                    DXGI_FORMAT.BC1_TYPELESS,
                ):
                    self._mode = "RGBA"
                    self.pixel_format = "BC1"
                    n = 1
                elif dxgi_format in (DXGI_FORMAT.BC2_TYPELESS, DXGI_FORMAT.BC2_UNORM):
                    self._mode = "RGBA"
                    self.pixel_format = "BC2"
                    n = 2
                elif dxgi_format in (DXGI_FORMAT.BC3_TYPELESS, DXGI_FORMAT.BC3_UNORM):
                    self._mode = "RGBA"
                    self.pixel_format = "BC3"
                    n = 3
                elif dxgi_format in (DXGI_FORMAT.BC4_TYPELESS, DXGI_FORMAT.BC4_UNORM):
                    self._mode = "L"
                    self.pixel_format = "BC4"
                    n = 4
                elif dxgi_format in (DXGI_FORMAT.BC5_TYPELESS, DXGI_FORMAT.BC5_UNORM):
                    self._mode = "RGB"
                    self.pixel_format = "BC5"
                    n = 5
                elif dxgi_format == DXGI_FORMAT.BC5_SNORM:
                    self._mode = "RGB"
                    self.pixel_format = "BC5S"
                    n = 5
                elif dxgi_format == DXGI_FORMAT.BC6H_UF16:
                    self._mode = "RGB"
                    self.pixel_format = "BC6H"
                    n = 6
                elif dxgi_format == DXGI_FORMAT.BC6H_SF16:
                    self._mode = "RGB"
                    self.pixel_format = "BC6HS"
                    n = 6
                elif dxgi_format in (
                    DXGI_FORMAT.BC7_TYPELESS,
                    DXGI_FORMAT.BC7_UNORM,
                    DXGI_FORMAT.BC7_UNORM_SRGB,
                ):
                    self._mode = "RGBA"
                    self.pixel_format = "BC7"
                    n = 7
                    if dxgi_format == DXGI_FORMAT.BC7_UNORM_SRGB:
                        self.info["gamma"] = 1 / 2.2
                elif dxgi_format in (
                    DXGI_FORMAT.R8G8B8A8_TYPELESS,
                    DXGI_FORMAT.R8G8B8A8_UNORM,
                    DXGI_FORMAT.R8G8B8A8_UNORM_SRGB,
                ):
                    self._mode = "RGBA"
                    if dxgi_format == DXGI_FORMAT.R8G8B8A8_UNORM_SRGB:
                        self.info["gamma"] = 1 / 2.2
                else:
                    msg = f"Unimplemented DXGI format {dxgi_format}"
                    raise NotImplementedError(msg)
            else:
                msg = f"Unimplemented pixel format {repr(fourcc)}"
                raise NotImplementedError(msg)
        else:
            msg = f"Unknown pixel format flags {pfflags}"
            raise NotImplementedError(msg)

        if n:
            self.tile = [
                ImageFile._Tile("bcn", extents, offset, (n, self.pixel_format))
            ]
        else:
            self.tile = [ImageFile._Tile("raw", extents, 0, rawmode or self.mode)]

    def load_seek(self, pos: int) -> None:
        pass


class DdsRgbDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        bitcount, masks = self.args

        data = bytearray()
        bytecount = bitcount // 8
        if not bytecount:
            self.set_as_raw(data)
            return -1, 0

        # Some masks will be padded with zeros, e.g. R 0b11 G 0b1100
        # Calculate how many zeros each mask is padded with
        mask_offsets = []
        # And the maximum value of each channel without the padding
        mask_totals = []
        for mask in masks:
            offset = 0
            if mask != 0:
                while mask >> (offset + 1) << (offset + 1) == mask:
                    offset += 1
            mask_offsets.append(offset)
            mask_totals.append(mask >> offset)

        assert self.fd is not None
        dest_length = self.state.xsize * self.state.ysize * len(masks)
        while len(data) < dest_length:
            bytes_read = self.fd.read(bytecount)
            if len(bytes_read) < bytecount:
                break
            value = int.from_bytes(bytes_read, "little")
            for i, mask in enumerate(masks):
                masked_value = value & mask
                # Remove the zero padding, and scale it to 8 bits
                data += o8(
                    int(((masked_value >> mask_offsets[i]) / mask_totals[i]) * 255)
                    if mask_totals[i]
                    else 0
                )
        self.set_as_raw(data)
        return -1, 0


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode not in ("RGB", "RGBA", "L", "LA"):
        msg = f"cannot write mode {im.mode} as DDS"
        raise OSError(msg)

    flags = DDSD.CAPS | DDSD.HEIGHT | DDSD.WIDTH | DDSD.PIXELFORMAT
    bitcount = len(im.getbands()) * 8
    pixel_format = im.encoderinfo.get("pixel_format")
    args: tuple[int] | str
    if pixel_format:
        codec_name = "bcn"
        flags |= DDSD.LINEARSIZE
        pitch = (im.width + 3) * 4
        rgba_mask = [0, 0, 0, 0]
        pixel_flags = DDPF.FOURCC
        if pixel_format == "DXT1":
            fourcc = D3DFMT.DXT1
            args = (1,)
        elif pixel_format == "DXT3":
            fourcc = D3DFMT.DXT3
            args = (2,)
        elif pixel_format == "DXT5":
            fourcc = D3DFMT.DXT5
            args = (3,)
        else:
            fourcc = D3DFMT.DX10
            if pixel_format == "BC2":
                args = (2,)
                dxgi_format = DXGI_FORMAT.BC2_TYPELESS
            elif pixel_format == "BC3":
                args = (3,)
                dxgi_format = DXGI_FORMAT.BC3_TYPELESS
            elif pixel_format == "BC5":
                args = (5,)
                dxgi_format = DXGI_FORMAT.BC5_TYPELESS
                if im.mode != "RGB":
                    msg = "only RGB mode can be written as BC5"
                    raise OSError(msg)
            else:
                msg = f"cannot write pixel format {pixel_format}"
                raise OSError(msg)
    else:
        codec_name = "raw"
        flags |= DDSD.PITCH
        pitch = (im.width * bitcount + 7) // 8

        alpha = im.mode[-1] == "A"
        if im.mode[0] == "L":
            pixel_flags = DDPF.LUMINANCE
            args = im.mode
            if alpha:
                rgba_mask = [0x000000FF, 0x000000FF, 0x000000FF]
            else:
                rgba_mask = [0xFF000000, 0xFF000000, 0xFF000000]
        else:
            pixel_flags = DDPF.RGB
            args = im.mode[::-1]
            rgba_mask = [0x00FF0000, 0x0000FF00, 0x000000FF]

            if alpha:
                r, g, b, a = im.split()
                im = Image.merge("RGBA", (a, r, g, b))
        if alpha:
            pixel_flags |= DDPF.ALPHAPIXELS
        rgba_mask.append(0xFF000000 if alpha else 0)

        fourcc = D3DFMT.UNKNOWN
    fp.write(
        o32(DDS_MAGIC)
        + struct.pack(
            "<7I",
            124,  # header size
            flags,  # flags
            im.height,
            im.width,
            pitch,
            0,  # depth
            0,  # mipmaps
        )
        + struct.pack("11I", *((0,) * 11))  # reserved
        # pfsize, pfflags, fourcc, bitcount
        + struct.pack("<4I", 32, pixel_flags, fourcc, bitcount)
        + struct.pack("<4I", *rgba_mask)  # dwRGBABitMask
        + struct.pack("<5I", DDSCAPS.TEXTURE, 0, 0, 0, 0)
    )
    if fourcc == D3DFMT.DX10:
        fp.write(
            # dxgi_format, 2D resource, misc, array size, straight alpha
            struct.pack("<5I", dxgi_format, 3, 0, 0, 1)
        )
    ImageFile._save(im, fp, [ImageFile._Tile(codec_name, (0, 0) + im.size, 0, args)])


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"DDS ")


Image.register_open(DdsImageFile.format, DdsImageFile, _accept)
Image.register_decoder("dds_rgb", DdsRgbDecoder)
Image.register_save(DdsImageFile.format, _save)
Image.register_extension(DdsImageFile.format, ".dds")
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# JPEG (JFIF) file handling
#
# See "Digital Compression and Coding of Continuous-Tone Still Images,
# Part 1, Requirements and Guidelines" (CCITT T.81 / ISO 10918-1)
#
# History:
# 1995-09-09 fl   Created
# 1995-09-13 fl   Added full parser
# 1996-03-25 fl   Added hack to use the IJG command line utilities
# 1996-05-05 fl   Workaround Photoshop 2.5 CMYK polarity bug
# 1996-05-28 fl   Added draft support, JFIF version (0.1)
# 1996-12-30 fl   Added encoder options, added progression property (0.2)
# 1997-08-27 fl   Save mode 1 images as BW (0.3)
# 1998-07-12 fl   Added YCbCr to draft and save methods (0.4)
# 1998-10-19 fl   Don't hang on files using 16-bit DQT's (0.4.1)
# 2001-04-16 fl   Extract DPI settings from JFIF files (0.4.2)
# 2002-07-01 fl   Skip pad bytes before markers; identify Exif files (0.4.3)
# 2003-04-25 fl   Added experimental EXIF decoder (0.5)
# 2003-06-06 fl   Added experimental EXIF GPSinfo decoder
# 2003-09-13 fl   Extract COM markers
# 2009-09-06 fl   Added icc_profile support (from Florian Hoech)
# 2009-03-06 fl   Changed CMYK handling; always use Adobe polarity (0.6)
# 2009-03-08 fl   Added subsampling support (from Justin Huff).
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-1996 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings

from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import IO, Any

    from .MpoImagePlugin import MpoImageFile

#
# Parser


def Skip(self: JpegImageFile, marker: int) -> None:
    assert self.fp is not None
    n = i16(self.fp.read(2)) - 2
    ImageFile._safe_read(self.fp, n)


def APP(self: JpegImageFile, marker: int) -> None:
    #
    # Application marker.  Store these in the APP dictionary.
    # Also look for well-known application markers.

    assert self.fp is not None
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)

    app = f"APP{marker & 15}"

    self.app[app] = s  # compatibility
    self.applist.append((app, s))

    if marker == 0xFFE0 and s.startswith(b"JFIF"):
        # extract JFIF information
        self.info["jfif"] = version = i16(s, 5)  # version
        self.info["jfif_version"] = divmod(version, 256)
        # extract JFIF properties
        try:
            jfif_unit = s[7]
            jfif_density = i16(s, 8), i16(s, 10)
        except Exception:
            pass
        else:
            if jfif_unit == 1:
                self.info["dpi"] = jfif_density
            elif jfif_unit == 2:  # cm
                # 1 dpcm = 2.54 dpi
                self.info["dpi"] = tuple(d * 2.54 for d in jfif_density)
            self.info["jfif_unit"] = jfif_unit
            self.info["jfif_density"] = jfif_density
    elif marker == 0xFFE1 and s.startswith(b"Exif\0\0"):
        # extract EXIF information
        if "exif" in self.info:
            self.info["exif"] += s[6:]
        else:
            self.info["exif"] = s
            self._exif_offset = self.fp.tell() - n + 6
    elif marker == 0xFFE1 and s.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
        self.info["xmp"] = s.split(b"\x00", 1)[1]
    elif marker == 0xFFE2 and s.startswith(b"FPXR\0"):
        # extract FlashPix information (incomplete)
        self.info["flashpix"] = s  # FIXME: value will change
    elif marker == 0xFFE2 and s.startswith(b"ICC_PROFILE\0"):
        # Since an ICC profile can be larger than the maximum size of
        # a JPEG marker (64K), we need provisions to split it into
        # multiple markers. The format defined by the ICC specifies
        # one or more APP2 markers containing the following data:
        #   Identifying string      ASCII "ICC_PROFILE\0"  (12 bytes)
        #   Marker sequence number  1, 2, etc (1 byte)
        #   Number of markers       Total of APP2's used (1 byte)
        #   Profile data            (remainder of APP2 data)
        # Decoders should use the marker sequence numbers to
        # reassemble the profile, rather than assuming that the APP2
        # markers appear in the correct sequence.
        self.icclist.append(s)
    elif marker == 0xFFED and s.startswith(b"Photoshop 3.0\x00"):
        # parse the image resource block
        offset = 14
        photoshop = self.info.setdefault("photoshop", {})
        try:
            while s[offset : offset + 4] == b"8BIM":
                offset += 4
                # resource code
                code = i16(s, offset)
                offset += 2
                # resource name (usually empty)
                name_len = s[offset]
                # name = s[offset+1:offset+1+name_len]
                offset += 1 + name_len
                offset += offset & 1  # align
                # resource data block
                size = i32(s, offset)
                offset += 4
                data = s[offset : offset + size]
                if code == 0x03ED:  # ResolutionInfo
                    photoshop[code] = {
                        "XResolution": i32(data, 0) / 65536,
                        "DisplayedUnitsX": i16(data, 4),
                        "YResolution": i32(data, 8) / 65536,
                        "DisplayedUnitsY": i16(data, 12),
                    }
                else:
                    photoshop[code] = data
                offset += size
                offset += offset & 1  # align
        except struct.error:
            pass  # insufficient data

    elif marker == 0xFFEE and s.startswith(b"Adobe"):
        self.info["adobe"] = i16(s, 5)
        # extract Adobe custom properties
        try:
            adobe_transform = s[11]
        except IndexError:
            pass
        else:
            self.info["adobe_transform"] = adobe_transform
    elif marker == 0xFFE2 and s.startswith(b"MPF\0"):
        # extract MPO information
        self.info["mp"] = s[4:]
        # offset is current location minus buffer size
        # plus constant header size
        self.info["mpoffset"] = self.fp.tell() - n + 4


def COM(self: JpegImageFile, marker: int) -> None:
    #
    # Comment marker.  Store these in the APP dictionary.
    assert self.fp is not None
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)

    self.info["comment"] = s
    self.app["COM"] = s  # compatibility
    self.applist.append(("COM", s))


def SOF(self: JpegImageFile, marker: int) -> None:
    #
    # Start of frame marker.  Defines the size and mode of the
    # image.  JPEG is colour blind, so we use some simple
    # heuristics to map the number of layers to an appropriate
    # mode.  Note that this could be made a bit brighter, by
    # looking for JFIF and Adobe APP markers.

    assert self.fp is not None
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    self._size = i16(s, 3), i16(s, 1)
    if self._im is not None and self.size != self.im.size:
        self._im = None

    self.bits = s[0]
    if self.bits != 8:
        msg = f"cannot handle {self.bits}-bit layers"
        raise SyntaxError(msg)

    self.layers = s[5]
    if self.layers == 1:
        self._mode = "L"
    elif self.layers == 3:
        self._mode = "RGB"
    elif self.layers == 4:
        self._mode = "CMYK"
    else:
        msg = f"cannot handle {self.layers}-layer images"
        raise SyntaxError(msg)

    if marker in [0xFFC2, 0xFFC6, 0xFFCA, 0xFFCE]:
        self.info["progressive"] = self.info["progression"] = 1

    if self.icclist:
        # fixup icc profile
        self.icclist.sort()  # sort by sequence number
        if self.icclist[0][13] == len(self.icclist):
            profile = [p[14:] for p in self.icclist]
            icc_profile = b"".join(profile)
        else:
            icc_profile = None  # wrong number of fragments
        self.info["icc_profile"] = icc_profile
        self.icclist = []

    for i in range(6, len(s), 3):
        t = s[i : i + 3]
        # 4-tuples: id, vsamp, hsamp, qtable
        self.layer.append((t[0], t[1] // 16, t[1] & 15, t[2]))


def DQT(self: JpegImageFile, marker: int) -> None:
    #
    # Define quantization table.  Note that there might be more
    # than one table in each marker.

    # FIXME: The quantization tables can be used to estimate the
    # compression quality.

    assert self.fp is not None
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    while len(s):
        v = s[0]
        precision = 1 if (v // 16 == 0) else 2  # in bytes
        qt_length = 1 + precision * 64
        if len(s) < qt_length:
            msg = "bad quantization table marker"
            raise SyntaxError(msg)
        data = array.array("B" if precision == 1 else "H", s[1:qt_length])
        if sys.byteorder == "little" and precision > 1:
            data.byteswap()  # the values are always big-endian
        self.quantization[v & 15] = [data[i] for i in zigzag_index]
        s = s[qt_length:]


#
# JPEG marker table

MARKER = {
    0xFFC0: ("SOF0", "Baseline DCT", SOF),
    0xFFC1: ("SOF1", "Extended Sequential DCT", SOF),
    0xFFC2: ("SOF2", "Progressive DCT", SOF),
    0xFFC3: ("SOF3", "Spatial lossless", SOF),
    0xFFC4: ("DHT", "Define Huffman table", Skip),
    0xFFC5: ("SOF5", "Differential sequential DCT", SOF),
    0xFFC6: ("SOF6", "Differential progressive DCT", SOF),
    0xFFC7: ("SOF7", "Differential spatial", SOF),
    0xFFC8: ("JPG", "Extension", None),
    0xFFC9: ("SOF9", "Extended sequential DCT (AC)", SOF),
    0xFFCA: ("SOF10", "Progressive DCT (AC)", SOF),
    0xFFCB: ("SOF11", "Spatial lossless DCT (AC)", SOF),
    0xFFCC: ("DAC", "Define arithmetic coding conditioning", Skip),
    0xFFCD: ("SOF13", "Differential sequential DCT (AC)", SOF),
    0xFFCE: ("SOF14", "Differential progressive DCT (AC)", SOF),
    0xFFCF: ("SOF15", "Differential spatial (AC)", SOF),
    0xFFD0: ("RST0", "Restart 0", None),
    0xFFD1: ("RST1", "Restart 1", None),
    0xFFD2: ("RST2", "Restart 2", None),
    0xFFD3: ("RST3", "Restart 3", None),
    0xFFD4: ("RST4", "Restart 4", None),
    0xFFD5: ("RST5", "Restart 5", None),
    0xFFD6: ("RST6", "Restart 6", None),
    0xFFD7: ("RST7", "Restart 7", None),
    0xFFD8: ("SOI", "Start of image", None),
    0xFFD9: ("EOI", "End of image", None),
    0xFFDA: ("SOS", "Start of scan", Skip),
    0xFFDB: ("DQT", "Define quantization table", DQT),
    0xFFDC: ("DNL", "Define number of lines", Skip),
    0xFFDD: ("DRI", "Define restart interval", Skip),
    0xFFDE: ("DHP", "Define hierarchical progression", SOF),
    0xFFDF: ("EXP", "Expand reference component", Skip),
    0xFFE0: ("APP0", "Application segment 0", APP),
    0xFFE1: ("APP1", "Application segment 1", APP),
    0xFFE2: ("APP2", "Application segment 2", APP),
    0xFFE3: ("APP3", "Application segment 3", APP),
    0xFFE4: ("APP4", "Application segment 4", APP),
    0xFFE5: ("APP5", "Application segment 5", APP),
    0xFFE6: ("APP6", "Application segment 6", APP),
    0xFFE7: ("APP7", "Application segment 7", APP),
    0xFFE8: ("APP8", "Application segment 8", APP),
    0xFFE9: ("APP9", "Application segment 9", APP),
    0xFFEA: ("APP10", "Application segment 10", APP),
    0xFFEB: ("APP11", "Application segment 11", APP),
    0xFFEC: ("APP12", "Application segment 12", APP),
    0xFFED: ("APP13", "Application segment 13", APP),
    0xFFEE: ("APP14", "Application segment 14", APP),
    0xFFEF: ("APP15", "Application segment 15", APP),
    0xFFF0: ("JPG0", "Extension 0", None),
    0xFFF1: ("JPG1", "Extension 1", None),
    0xFFF2: ("JPG2", "Extension 2", None),
    0xFFF3: ("JPG3", "Extension 3", None),
    0xFFF4: ("JPG4", "Extension 4", None),
    0xFFF5: ("JPG5", "Extension 5", None),
    0xFFF6: ("JPG6", "Extension 6", None),
    0xFFF7: ("JPG7", "Extension 7", None),
    0xFFF8: ("JPG8", "Extension 8", None),
    0xFFF9: ("JPG9", "Extension 9", None),
    0xFFFA: ("JPG10", "Extension 10", None),
    0xFFFB: ("JPG11", "Extension 11", None),
    0xFFFC: ("JPG12", "Extension 12", None),
    0xFFFD: ("JPG13", "Extension 13", None),
    0xFFFE: ("COM", "Comment", COM),
}


def _accept(prefix: bytes) -> bool:
    # Magic number was taken from https://en.wikipedia.org/wiki/JPEG
    return prefix.startswith(b"\xff\xd8\xff")


##
# Image plugin for JPEG and JFIF images.


class JpegImageFile(ImageFile.ImageFile):
    format = "JPEG"
    format_description = "JPEG (ISO 10918)"

    def _open(self) -> None:
        assert self.fp is not None
        s = self.fp.read(3)

        if not _accept(s):
            msg = "not a JPEG file"
            raise SyntaxError(msg)
        s = b"\xff"

        # Create attributes
        self.bits = self.layers = 0
        self._exif_offset = 0

        # JPEG specifics (internal)
        self.layer: list[tuple[int, int, int, int]] = []
        self._huffman_dc: dict[Any, Any] = {}
        self._huffman_ac: dict[Any, Any] = {}
        self.quantization: dict[int, list[int]] = {}
        self.app: dict[str, bytes] = {}  # compatibility
        self.applist: list[tuple[str, bytes]] = []
        self.icclist: list[bytes] = []

        while True:
            i = s[0]
            if i == 0xFF:
                s = s + self.fp.read(1)
                i = i16(s)
            else:
                # Skip non-0xFF junk
                s = self.fp.read(1)
                continue

            if i in MARKER:
                name, description, handler = MARKER[i]
                if handler is not None:
                    handler(self, i)
                if i == 0xFFDA:  # start of scan
                    rawmode = self.mode
                    if self.mode == "CMYK":
                        rawmode = "CMYK;I"  # assume adobe conventions
                    self.tile = [
                        ImageFile._Tile("jpeg", (0, 0) + self.size, 0, (rawmode, ""))
                    ]
                    # self.__offset = self.fp.tell()
                    break
                s = self.fp.read(1)
            elif i in {0, 0xFFFF}:
                # padded marker or junk; move on
                s = b"\xff"
            elif i == 0xFF00:  # Skip extraneous data (escaped 0xFF)
                s = self.fp.read(1)
            else:
                msg = "no marker found"
                raise SyntaxError(msg)

        self._read_dpi_from_exif()

    def __getstate__(self) -> list[Any]:
        return super().__getstate__() + [self.layers, self.layer]

    def __setstate__(self, state: list[Any]) -> None:
        self.layers, self.layer = state[6:]
        super().__setstate__(state)

    def load_read(self, read_bytes: int) -> bytes:
        """
        internal: read more image data
        For premature EOF and LOAD_TRUNCATED_IMAGES adds EOI marker
        so libjpeg can finish decoding
        """
        assert self.fp is not None
        s = self.fp.read(read_bytes)

        if not s and ImageFile.LOAD_TRUNCATED_IMAGES and not hasattr(self, "_ended"):
            # Premature EOF.
            # Pretend file is finished adding EOI marker
            self._ended = True
            return b"\xff\xd9"

        return s

    def draft(
        self, mode: str | None, size: tuple[int, int] | None
    ) -> tuple[str, tuple[int, int, float, float]] | None:
        if len(self.tile) != 1:
            return None

        # Protect from second call
        if self.decoderconfig:
            return None

        d, e, o, a = self.tile[0]
        scale = 1
        original_size = self.size

        assert isinstance(a, tuple)
        if a[0] == "RGB" and mode in ["L", "YCbCr"]:
            self._mode = mode
            a = mode, ""

        if size:
            scale = min(self.size[0] // size[0], self.size[1] // size[1])
            for s in [8, 4, 2, 1]:
                if scale >= s:
                    break
            assert e is not None
            e = (
                e[0],
                e[1],
                (e[2] - e[0] + s - 1) // s + e[0],
                (e[3] - e[1] + s - 1) // s + e[1],
            )
            self._size = ((self.size[0] + s - 1) // s, (self.size[1] + s - 1) // s)
            scale = s

        self.tile = [ImageFile._Tile(d, e, o, a)]
        self.decoderconfig = (scale, 0)

        box = (0, 0, original_size[0] / scale, original_size[1] / scale)
        return self.mode, box

    def load_djpeg(self) -> None:
        # ALTERNATIVE: handle JPEGs via the IJG command line utilities

        f, path = tempfile.mkstemp()
        os.close(f)
        if os.path.exists(self.filename):
            subprocess.check_call(["djpeg", "-outfile", path, self.filename])
        else:
            try:
                os.unlink(path)
            except OSError:
                pass

            msg = "Invalid Filename"
            raise ValueError(msg)

        try:
            with Image.open(path) as _im:
                _im.load()
                self.im = _im.im
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        self._mode = self.im.mode
        self._size = self.im.size

        self.tile = []

    def _getexif(self) -> dict[int, Any] | None:
        return _getexif(self)

    def _read_dpi_from_exif(self) -> None:
        # If DPI isn't in JPEG header, fetch from EXIF
        if "dpi" in self.info or "exif" not in self.info:
            return
        try:
            exif = self.getexif()
            resolution_unit = exif[0x0128]
            x_resolution = exif[0x011A]
            try:
                dpi = float(x_resolution[0]) / x_resolution[1]
            except TypeError:
                dpi = x_resolution
            if math.isnan(dpi):
                msg = "DPI is not a number"
                raise ValueError(msg)
            if resolution_unit == 3:  # cm
                # 1 dpcm = 2.54 dpi
                dpi *= 2.54
            self.info["dpi"] = dpi, dpi
        except (
            struct.error,  # truncated EXIF
            KeyError,  # dpi not included
            SyntaxError,  # invalid/unreadable EXIF
            TypeError,  # dpi is an invalid float
            ValueError,  # dpi is an invalid float
            ZeroDivisionError,  # invalid dpi rational value
        ):
            self.info["dpi"] = 72, 72

    def _getmp(self) -> dict[int, Any] | None:
        return _getmp(self)


def _getexif(self: JpegImageFile) -> dict[int, Any] | None:
    if "exif" not in self.info:
        return None
    return self.getexif()._get_merged_dict()


def _getmp(self: JpegImageFile) -> dict[int, Any] | None:
    # Extract MP information.  This method was inspired by the "highly
    # experimental" _getexif version that's been in use for years now,
    # itself based on the ImageFileDirectory class in the TIFF plugin.

    # The MP record essentially consists of a TIFF file embedded in a JPEG
    # application marker.
    try:
        data = self.info["mp"]
    except KeyError:
        return None
    file_contents = io.BytesIO(data)
    head = file_contents.read(8)
    endianness = ">" if head.startswith(b"\x4d\x4d\x00\x2a") else "<"
    # process dictionary
    from . import TiffImagePlugin

    try:
        info = TiffImagePlugin.ImageFileDirectory_v2(head)
        file_contents.seek(info.next)
        info.load(file_contents)
        mp = dict(info)
    except Exception as e:
        msg = "malformed MP Index (unreadable directory)"
        raise SyntaxError(msg) from e
    # it's an error not to have a number of images
    try:
        quant = mp[0xB001]
    except KeyError as e:
        msg = "malformed MP Index (no number of images)"
        raise SyntaxError(msg) from e
    # get MP entries
    mpentries = []
    try:
        rawmpentries = mp[0xB002]
        for entrynum in range(quant):
            unpackedentry = struct.unpack_from(
                f"{endianness}LLLHH", rawmpentries, entrynum * 16
            )
            labels = ("Attribute", "Size", "DataOffset", "EntryNo1", "EntryNo2")
            mpentry = dict(zip(labels, unpackedentry))
            mpentryattr = {
                "DependentParentImageFlag": bool(mpentry["Attribute"] & (1 << 31)),
                "DependentChildImageFlag": bool(mpentry["Attribute"] & (1 << 30)),
                "RepresentativeImageFlag": bool(mpentry["Attribute"] & (1 << 29)),
                "Reserved": (mpentry["Attribute"] & (3 << 27)) >> 27,
                "ImageDataFormat": (mpentry["Attribute"] & (7 << 24)) >> 24,
                "MPType": mpentry["Attribute"] & 0x00FFFFFF,
            }
            if mpentryattr["ImageDataFormat"] == 0:
                mpentryattr["ImageDataFormat"] = "JPEG"
            else:
                msg = "unsupported picture format in MPO"
                raise SyntaxError(msg)
            mptypemap = {
                0x000000: "Undefined",
                0x010001: "Large Thumbnail (VGA Equivalent)",
                0x010002: "Large Thumbnail (Full HD Equivalent)",
                0x020001: "Multi-Frame Image (Panorama)",
                0x020002: "Multi-Frame Image: (Disparity)",
                0x020003: "Multi-Frame Image: (Multi-Angle)",
                0x030000: "Baseline MP Primary Image",
            }
            mpentryattr["MPType"] = mptypemap.get(mpentryattr["MPType"], "Unknown")
            mpentry["Attribute"] = mpentryattr
            mpentries.append(mpentry)
        mp[0xB002] = mpentries
    except KeyError as e:
        msg = "malformed MP Index (bad MP Entry)"
        raise SyntaxError(msg) from e
    # Next we should try and parse the individual image unique ID list;
    # we don't because I've never seen this actually used in a real MPO
    # file and so can't test it.
    return mp


# --------------------------------------------------------------------
# stuff to save JPEG files

RAWMODE = {
    "1": "L",
    "L": "L",
    "RGB": "RGB",
    "RGBX": "RGB",
    "CMYK": "CMYK;I",  # assume adobe conventions
    "YCbCr": "YCbCr",
}

# fmt: off
zigzag_index = (
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
)

samplings = {
    (1, 1, 1, 1, 1, 1): 0,
    (2, 1, 1, 1, 1, 1): 1,
    (2, 2, 1, 1, 1, 1): 2,
}
# fmt: on


def get_sampling(im: Image.Image) -> int:
    # There's no subsampling when images have only 1 layer
    # (grayscale images) or when they are CMYK (4 layers),
    # so set subsampling to the default value.
    #
    # NOTE: currently Pillow can't encode JPEG to YCCK format.
    # If YCCK support is added in the future, subsampling code will have
    # to be updated (here and in JpegEncode.c) to deal with 4 layers.
    if not isinstance(im, JpegImageFile) or im.layers in (1, 4):
        return -1
    sampling = im.layer[0][1:3] + im.layer[1][1:3] + im.layer[2][1:3]
    return samplings.get(sampling, -1)


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    try:
        rawmode = RAWMODE[im.mode]
    except KeyError as e:
        msg = f"cannot write mode {im.mode} as JPEG"
        raise OSError(msg) from e

    info = im.encoderinfo

    dpi = [round(x) for x in info.get("dpi", (0, 0))]

    quality = info.get("quality", -1)
    subsampling = info.get("subsampling", -1)
    qtables = info.get("qtables")

    if quality == "keep":
        quality = -1
        subsampling = "keep"
        qtables = "keep"
    elif quality in presets:
        preset = presets[quality]
        quality = -1
        subsampling = preset.get("subsampling", -1)
        qtables = preset.get("quantization")
    elif not isinstance(quality, int):
        msg = "Invalid quality setting"
        raise ValueError(msg)
    else:
        if subsampling in presets:
            subsampling = presets[subsampling].get("subsampling", -1)
        if isinstance(qtables, str) and qtables in presets:
            qtables = presets[qtables].get("quantization")

    if subsampling == "4:4:4":
        subsampling = 0
    elif subsampling == "4:2:2":
        subsampling = 1
    elif subsampling == "4:2:0":
        subsampling = 2
    elif subsampling == "4:1:1":
        # For compatibility. Before Pillow 4.3, 4:1:1 actually meant 4:2:0.
        # Set 4:2:0 if someone is still using that value.
        subsampling = 2
    elif subsampling == "keep":
        if im.format != "JPEG":
            msg = "Cannot use 'keep' when original image is not a JPEG"
            raise ValueError(msg)
        subsampling = get_sampling(im)

    def validate_qtables(
        qtables: (
            str | tuple[list[int], ...] | list[list[int]] | dict[int, list[int]] | None
        ),
    ) -> list[list[int]] | None:
        if qtables is None:
            return qtables
        if isinstance(qtables, str):
            try:
                lines = [
                    int(num)
                    for line in qtables.splitlines()
                    for num in line.split("#", 1)[0].split()
                ]
            except ValueError as e:
                msg = "Invalid quantization table"
                raise ValueError(msg) from e
            else:
                qtables = [lines[s : s + 64] for s in range(0, len(lines), 64)]
        if isinstance(qtables, (tuple, list, dict)):
            if isinstance(qtables, dict):
                qtables = [
                    qtables[key] for key in range(len(qtables)) if key in qtables
                ]
            elif isinstance(qtables, tuple):
                qtables = list(qtables)
            if not (0 < len(qtables) < 5):
                msg = "None or too many quantization tables"
                raise ValueError(msg)
            try:
                for idx, table in enumerate(qtables):
                    if len(table) != 64:
                        msg = "Invalid quantization table"
                        raise TypeError(msg)
                    qtables[idx] = list(array.array("H", table))
            except TypeError as e:
                msg = "Invalid quantization table"
                raise ValueError(msg) from e
            return qtables

    if qtables == "keep":
        if im.format != "JPEG":
            msg = "Cannot use 'keep' when original image is not a JPEG"
            raise ValueError(msg)
        qtables = getattr(im, "quantization", None)
    qtables = validate_qtables(qtables)

    extra = info.get("extra", b"")

    MAX_BYTES_IN_MARKER = 65533
    if xmp := info.get("xmp"):
        overhead_len = 29  # b"http://ns.adobe.com/xap/1.0/\x00"
        max_data_bytes_in_marker = MAX_BYTES_IN_MARKER - overhead_len
        if len(xmp) > max_data_bytes_in_marker:
            msg = "XMP data is too long"
            raise ValueError(msg)
        size = o16(2 + overhead_len + len(xmp))
        extra += b"\xff\xe1" + size + b"http://ns.adobe.com/xap/1.0/\x00" + xmp

    if icc_profile := info.get("icc_profile"):
        overhead_len = 14  # b"ICC_PROFILE\0" + o8(i) + o8(len(markers))
        max_data_bytes_in_marker = MAX_BYTES_IN_MARKER - overhead_len
        markers = []
        while icc_profile:
            markers.append(icc_profile[:max_data_bytes_in_marker])
            icc_profile = icc_profile[max_data_bytes_in_marker:]
        i = 1
        for marker in markers:
            size = o16(2 + overhead_len + len(marker))
            extra += (
                b"\xff\xe2"
                + size
                + b"ICC_PROFILE\0"
                + o8(i)
                + o8(len(markers))
                + marker
            )
            i += 1

    comment = info.get("comment", im.info.get("comment"))

    # "progressive" is the official name, but older documentation
    # says "progression"
    # FIXME: issue a warning if the wrong form is used (post-1.1.7)
    progressive = info.get("progressive", False) or info.get("progression", False)

    optimize = info.get("optimize", False)

    exif = info.get("exif", b"")
    if isinstance(exif, Image.Exif):
        exif = exif.tobytes()
    if len(exif) > MAX_BYTES_IN_MARKER:
        msg = "EXIF data is too long"
        raise ValueError(msg)

    # get keyword arguments
    im.encoderconfig = (
        quality,
        progressive,
        info.get("smooth", 0),
        optimize,
        info.get("keep_rgb", False),
        info.get("streamtype", 0),
        dpi,
        subsampling,
        info.get("restart_marker_blocks", 0),
        info.get("restart_marker_rows", 0),
        qtables,
        comment,
        extra,
        exif,
    )

    # if we optimize, libjpeg needs a buffer big enough to hold the whole image
    # in a shot. Guessing on the size, at im.size bytes. (raw pixel size is
    # channels*size, this is a value that's been used in a django patch.
    # https://github.com/matthewwithanm/django-imagekit/issues/50
    if optimize or progressive:
        # CMYK can be bigger
        if im.mode == "CMYK":
            bufsize = 4 * im.size[0] * im.size[1]
        # keep sets quality to -1, but the actual value may be high.
        elif quality >= 95 or quality == -1:
            bufsize = 2 * im.size[0] * im.size[1]
        else:
            bufsize = im.size[0] * im.size[1]
        if exif:
            bufsize += len(exif) + 5
        if extra:
            bufsize += len(extra) + 1
    else:
        # The EXIF info needs to be written as one block, + APP1, + one spare byte.
        # Ensure that our buffer is big enough. Same with the icc_profile block.
        bufsize = max(len(exif) + 5, len(extra) + 1)

    ImageFile._save(
        im, fp, [ImageFile._Tile("jpeg", (0, 0) + im.size, 0, rawmode)], bufsize
    )


##
# Factory for making JPEG and MPO instances
def jpeg_factory(
    fp: IO[bytes], filename: str | bytes | None = None
) -> JpegImageFile | MpoImageFile:
    im = JpegImageFile(fp, filename)
    try:
        mpheader = im._getmp()
        if mpheader is not None and mpheader[45057] > 1:
            for segment, content in im.applist:
                if segment == "APP1" and b' hdrgm:Version="' in content:
                    # Ultra HDR images are not yet supported
                    return im
            # It's actually an MPO
            from .MpoImagePlugin import MpoImageFile

            # Don't reload everything, just convert it.
            im = MpoImageFile.adopt(im, mpheader)
    except (TypeError, IndexError):
        # It is really a JPEG
        pass
    except SyntaxError:
        warnings.warn(
            "Image appears to be a malformed MPO file, it will be "
            "interpreted as a base JPEG file"
        )
    return im


# ---------------------------------------------------------------------
# Registry stuff

Image.register_open(JpegImageFile.format, jpeg_factory, _accept)
Image.register_save(JpegImageFile.format, _save)

Image.register_extensions(JpegImageFile.format, [".jfif", ".jpe", ".jpg", ".jpeg"])

Image.register_mime(JpegImageFile.format, "image/jpeg")
<|endoftext|>#
# The Python Imaging Library.
#
# QOI support for PIL
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import os
from typing import IO

from . import Image, ImageFile
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o32be as o32


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"qoif")


class QoiImageFile(ImageFile.ImageFile):
    format = "QOI"
    format_description = "Quite OK Image"

    def _open(self) -> None:
        assert self.fp is not None
        if not _accept(self.fp.read(4)):
            msg = "not a QOI file"
            raise SyntaxError(msg)

        self._size = i32(self.fp.read(4)), i32(self.fp.read(4))

        channels = self.fp.read(1)[0]
        self._mode = "RGB" if channels == 3 else "RGBA"

        self.fp.seek(1, os.SEEK_CUR)  # colorspace
        self.tile = [ImageFile._Tile("qoi", (0, 0) + self._size, self.fp.tell())]


class QoiDecoder(ImageFile.PyDecoder):
    _pulls_fd = True
    _previous_pixel: bytes | bytearray | None = None
    _previously_seen_pixels: dict[int, bytes | bytearray] = {}

    def _add_to_previous_pixels(self, value: bytes | bytearray) -> None:
        self._previous_pixel = value

        r, g, b, a = value
        hash_value = (r * 3 + g * 5 + b * 7 + a * 11) % 64
        self._previously_seen_pixels[hash_value] = value

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        assert self.fd is not None

        self._previously_seen_pixels = {}
        self._previous_pixel = bytearray((0, 0, 0, 255))

        data = bytearray()
        bands = Image.getmodebands(self.mode)
        dest_length = self.state.xsize * self.state.ysize * bands
        while len(data) < dest_length:
            byte = self.fd.read(1)[0]
            value: bytes | bytearray
            if byte == 0b11111110 and self._previous_pixel:  # QOI_OP_RGB
                value = bytearray(self.fd.read(3)) + self._previous_pixel[3:]
            elif byte == 0b11111111:  # QOI_OP_RGBA
                value = self.fd.read(4)
            else:
                op = byte >> 6
                if op == 0:  # QOI_OP_INDEX
                    op_index = byte & 0b00111111
                    value = self._previously_seen_pixels.get(
                        op_index, bytearray((0, 0, 0, 0))
                    )
                elif op == 1 and self._previous_pixel:  # QOI_OP_DIFF
                    value = bytearray(
                        (
                            (self._previous_pixel[0] + ((byte & 0b00110000) >> 4) - 2)
                            % 256,
                            (self._previous_pixel[1] + ((byte & 0b00001100) >> 2) - 2)
                            % 256,
                            (self._previous_pixel[2] + (byte & 0b00000011) - 2) % 256,
                            self._previous_pixel[3],
                        )
                    )
                elif op == 2 and self._previous_pixel:  # QOI_OP_LUMA
                    second_byte = self.fd.read(1)[0]
                    diff_green = (byte & 0b00111111) - 32
                    diff_red = ((second_byte & 0b11110000) >> 4) - 8
                    diff_blue = (second_byte & 0b00001111) - 8

                    value = bytearray(
                        tuple(
                            (self._previous_pixel[i] + diff_green + diff) % 256
                            for i, diff in enumerate((diff_red, 0, diff_blue))
                        )
                    )
                    value += self._previous_pixel[3:]
                elif op == 3 and self._previous_pixel:  # QOI_OP_RUN
                    run_length = (byte & 0b00111111) + 1
                    value = self._previous_pixel
                    if bands == 3:
                        value = value[:3]
                    data += value * run_length
                    continue
            self._add_to_previous_pixels(value)

            if bands == 3:
                value = value[:3]
            data += value
        self.set_as_raw(data)
        return -1, 0


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode == "RGB":
        channels = 3
    elif im.mode == "RGBA":
        channels = 4
    else:
        msg = "Unsupported QOI image mode"
        raise ValueError(msg)

    colorspace = 0 if im.encoderinfo.get("colorspace") == "sRGB" else 1

    fp.write(b"qoif")
    fp.write(o32(im.size[0]))
    fp.write(o32(im.size[1]))
    fp.write(o8(channels))
    fp.write(o8(colorspace))

    ImageFile._save(im, fp, [ImageFile._Tile("qoi", (0, 0) + im.size)])


class QoiEncoder(ImageFile.PyEncoder):
    _pushes_fd = True
    _previous_pixel: tuple[int, int, int, int] | None = None
    _previously_seen_pixels: dict[int, tuple[int, int, int, int]] = {}
    _run = 0

    def _write_run(self) -> bytes:
        data = o8(0b11000000 | (self._run - 1))  # QOI_OP_RUN
        self._run = 0
        return data

    def _delta(self, left: int, right: int) -> int:
        result = (left - right) & 255
        if result >= 128:
            result -= 256
        return result

    def encode(self, bufsize: int) -> tuple[int, int, bytes]:
        assert self.im is not None

        self._previously_seen_pixels = {0: (0, 0, 0, 0)}
        self._previous_pixel = (0, 0, 0, 255)

        data = bytearray()
        w, h = self.im.size
        bands = Image.getmodebands(self.mode)

        for y in range(h):
            for x in range(w):
                pixel = self.im.getpixel((x, y))
                if bands == 3:
                    pixel = (*pixel, 255)

                if pixel == self._previous_pixel:
                    self._run += 1
                    if self._run == 62:
                        data += self._write_run()
                else:
                    if self._run:
                        data += self._write_run()

                    r, g, b, a = pixel
                    hash_value = (r * 3 + g * 5 + b * 7 + a * 11) % 64
                    if self._previously_seen_pixels.get(hash_value) == pixel:
                        data += o8(hash_value)  # QOI_OP_INDEX
                    elif self._previous_pixel:
                        self._previously_seen_pixels[hash_value] = pixel

                        prev_r, prev_g, prev_b, prev_a = self._previous_pixel
                        if prev_a == a:
                            delta_r = self._delta(r, prev_r)
                            delta_g = self._delta(g, prev_g)
                            delta_b = self._delta(b, prev_b)

                            if (
                                -2 <= delta_r < 2
                                and -2 <= delta_g < 2
                                and -2 <= delta_b < 2
                            ):
                                data += o8(
                                    0b01000000
                                    | (delta_r + 2) << 4
                                    | (delta_g + 2) << 2
                                    | (delta_b + 2)
                                )  # QOI_OP_DIFF
                            else:
                                delta_gr = self._delta(delta_r, delta_g)
                                delta_gb = self._delta(delta_b, delta_g)
                                if (
                                    -8 <= delta_gr < 8
                                    and -32 <= delta_g < 32
                                    and -8 <= delta_gb < 8
                                ):
                                    data += o8(
                                        0b10000000 | (delta_g + 32)
                                    )  # QOI_OP_LUMA
                                    data += o8((delta_gr + 8) << 4 | (delta_gb + 8))
                                else:
                                    data += o8(0b11111110)  # QOI_OP_RGB
                                    data += bytes(pixel[:3])
                        else:
                            data += o8(0b11111111)  # QOI_OP_RGBA
                            data += bytes(pixel)

                self._previous_pixel = pixel

        if self._run:
            data += self._write_run()
        data += bytes((0, 0, 0, 0, 0, 0, 0, 1))  # padding

        return len(data), 0, bytes(data)


Image.register_open(QoiImageFile.format, QoiImageFile, _accept)
Image.register_decoder("qoi", QoiDecoder)
Image.register_extension(QoiImageFile.format, ".qoi")

Image.register_save(QoiImageFile.format, _save)
Image.register_encoder("qoi", QoiEncoder)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# standard filters
#
# History:
# 1995-11-27 fl   Created
# 2002-06-08 fl   Added rank and mode filters
# 2003-09-15 fl   Fixed rank calculation in rank filter; added expand call
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2002 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import abc
import functools
from collections.abc import Sequence
from typing import cast

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Any

    from . import _imaging
    from ._typing import NumpyArray


class Filter(abc.ABC):
    @abc.abstractmethod
    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        pass


class MultibandFilter(Filter):
    pass


class BuiltinFilter(MultibandFilter):
    name: str
    filterargs: tuple[Any, ...]

    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        if image.mode == "P":
            msg = "cannot filter palette images"
            raise ValueError(msg)
        return image.filter(*self.filterargs)


class Kernel(BuiltinFilter):
    """
    Create a convolution kernel. This only supports 3x3 and 5x5 integer and floating
    point kernels.

    Kernels can only be applied to "L" and "RGB" images.

    :param size: Kernel size, given as (width, height). This must be (3,3) or (5,5).
    :param kernel: A sequence containing kernel weights. The kernel will be flipped
                   vertically before being applied to the image.
    :param scale: Scale factor. If given, the result for each pixel is divided by this
                  value. The default is the sum of the kernel weights.
    :param offset: Offset. If given, this value is added to the result, after it has
                   been divided by the scale factor.
    """

    name = "Kernel"

    def __init__(
        self,
        size: tuple[int, int],
        kernel: Sequence[float],
        scale: float | None = None,
        offset: float = 0,
    ) -> None:
        if scale is None:
            # default scale is sum of kernel
            scale = functools.reduce(lambda a, b: a + b, kernel)
        if size[0] * size[1] != len(kernel):
            msg = "not enough coefficients in kernel"
            raise ValueError(msg)
        self.filterargs = size, scale, offset, kernel


class RankFilter(Filter):
    """
    Create a rank filter.  The rank filter sorts all pixels in
    a window of the given size, and returns the ``rank``'th value.

    :param size: The kernel size, in pixels.
    :param rank: What pixel value to pick.  Use 0 for a min filter,
                 ``size * size / 2`` for a median filter, ``size * size - 1``
                 for a max filter, etc.
    """

    name = "Rank"

    def __init__(self, size: int, rank: int) -> None:
        if size % 2 == 0:
            msg = "bad filter size"
            raise ValueError(msg)
        if size * size * 4 > (2**31 - 1):
            msg = "filter size too large"
            raise ValueError(msg)
        if rank < 0 or rank >= size * size:
            msg = "bad rank value"
            raise ValueError(msg)
        self.size = size
        self.rank = rank

    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        if image.mode == "P":
            msg = "cannot filter palette images"
            raise ValueError(msg)
        image = image.expand(self.size // 2)
        return image.rankfilter(self.size, self.rank)


class MedianFilter(RankFilter):
    """
    Create a median filter. Picks the median pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """

    name = "Median"

    def __init__(self, size: int = 3) -> None:
        rank = size * size // 2
        super().__init__(size, rank)


class MinFilter(RankFilter):
    """
    Create a min filter.  Picks the lowest pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """

    name = "Min"

    def __init__(self, size: int = 3) -> None:
        rank = 0
        super().__init__(size, rank)


class MaxFilter(RankFilter):
    """
    Create a max filter.  Picks the largest pixel value in a window with the
    given size.

    :param size: The kernel size, in pixels.
    """

    name = "Max"

    def __init__(self, size: int = 3) -> None:
        rank = size * size - 1
        super().__init__(size, rank)


class ModeFilter(Filter):
    """
    Create a mode filter. Picks the most frequent pixel value in a box with the
    given size.  Pixel values that occur only once or twice are ignored; if no
    pixel value occurs more than twice, the original pixel value is preserved.

    :param size: The kernel size, in pixels.
    """

    name = "Mode"

    def __init__(self, size: int = 3) -> None:
        self.size = size

    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        return image.modefilter(self.size)


class GaussianBlur(MultibandFilter):
    """Blurs the image with a sequence of extended box filters, which
    approximates a Gaussian kernel. For details on accuracy see
    <https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf>

    :param radius: Standard deviation of the Gaussian kernel. Either a sequence of two
                   numbers for x and y, or a single number for both.
    """

    name = "GaussianBlur"

    def __init__(self, radius: float | Sequence[float] = 2) -> None:
        self.radius = radius

    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        xy = self.radius
        if xy == (0, 0) or xy == 0:
            return image.copy()
        if isinstance(xy, (int, float)):
            xy = (xy, xy)
        return image.gaussian_blur(xy)


class BoxBlur(MultibandFilter):
    """Blurs the image by setting each pixel to the average value of the pixels
    in a square box extending radius pixels in each direction.
    Supports float radius of arbitrary size. Uses an optimized implementation
    which runs in linear time relative to the size of the image
    for any radius value.

    :param radius: Size of the box in a direction. Either a sequence of two numbers for
                   x and y, or a single number for both.

                   Radius 0 does not blur, returns an identical image.
                   Radius 1 takes 1 pixel in each direction, i.e. 9 pixels in total.
    """

    name = "BoxBlur"

    def __init__(self, radius: float | Sequence[float]) -> None:
        xy = radius if isinstance(radius, (tuple, list)) else (radius, radius)
        if xy[0] < 0 or xy[1] < 0:
            msg = "radius must be >= 0"
            raise ValueError(msg)
        self.radius = radius

    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        xy = self.radius
        if xy == (0, 0) or xy == 0:
            return image.copy()
        if isinstance(xy, (int, float)):
            xy = (xy, xy)
        return image.box_blur(xy)


class UnsharpMask(MultibandFilter):
    """Unsharp mask filter.

    See Wikipedia's entry on `digital unsharp masking`_ for an explanation of
    the parameters.

    :param radius: Blur Radius
    :param percent: Unsharp strength, in percent
    :param threshold: Threshold controls the minimum brightness change that
      will be sharpened

    .. _digital unsharp masking: https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking

    """

    name = "UnsharpMask"

    def __init__(
        self, radius: float = 2, percent: int = 150, threshold: int = 3
    ) -> None:
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        return image.unsharp_mask(self.radius, self.percent, self.threshold)


class BLUR(BuiltinFilter):
    name = "Blur"
    # fmt: off
    filterargs = (5, 5), 16, 0, (
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
    )
    # fmt: on


class CONTOUR(BuiltinFilter):
    name = "Contour"
    # fmt: off
    filterargs = (3, 3), 1, 255, (
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1,
    )
    # fmt: on


class DETAIL(BuiltinFilter):
    name = "Detail"
    # fmt: off
    filterargs = (3, 3), 6, 0, (
        0,  -1,  0,
        -1, 10, -1,
        0,  -1,  0,
    )
    # fmt: on


class EDGE_ENHANCE(BuiltinFilter):
    name = "Edge-enhance"
    # fmt: off
    filterargs = (3, 3), 2, 0, (
        -1, -1, -1,
        -1, 10, -1,
        -1, -1, -1,
    )
    # fmt: on


class EDGE_ENHANCE_MORE(BuiltinFilter):
    name = "Edge-enhance More"
    # fmt: off
    filterargs = (3, 3), 1, 0, (
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1,
    )
    # fmt: on


class EMBOSS(BuiltinFilter):
    name = "Emboss"
    # fmt: off
    filterargs = (3, 3), 1, 128, (
        -1, 0, 0,
        0,  1, 0,
        0,  0, 0,
    )
    # fmt: on


class FIND_EDGES(BuiltinFilter):
    name = "Find Edges"
    # fmt: off
    filterargs = (3, 3), 1, 0, (
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1,
    )
    # fmt: on


class SHARPEN(BuiltinFilter):
    name = "Sharpen"
    # fmt: off
    filterargs = (3, 3), 16, 0, (
        -2, -2, -2,
        -2, 32, -2,
        -2, -2, -2,
    )
    # fmt: on


class SMOOTH(BuiltinFilter):
    name = "Smooth"
    # fmt: off
    filterargs = (3, 3), 13, 0, (
        1, 1, 1,
        1, 5, 1,
        1, 1, 1,
    )
    # fmt: on


class SMOOTH_MORE(BuiltinFilter):
    name = "Smooth More"
    # fmt: off
    filterargs = (5, 5), 100, 0, (
        1, 1,  1, 1, 1,
        1, 5,  5, 5, 1,
        1, 5, 44, 5, 1,
        1, 5,  5, 5, 1,
        1, 1,  1, 1, 1,
    )
    # fmt: on


class Color3DLUT(MultibandFilter):
    """Three-dimensional color lookup table.

    Transforms 3-channel pixels using the values of the channels as coordinates
    in the 3D lookup table and interpolating the nearest elements.

    This method allows you to apply almost any color transformation
    in constant time by using pre-calculated decimated tables.

    .. versionadded:: 5.2.0

    :param size: Size of the table. One int or tuple of (int, int, int).
                 Minimal size in any dimension is 2, maximum is 65.
    :param table: Flat lookup table. A list of ``channels * size**3``
                  float elements or a list of ``size**3`` channels-sized
                  tuples with floats. Channels are changed first,
                  then first dimension, then second, then third.
                  Value 0.0 corresponds lowest value of output, 1.0 highest.
    :param channels: Number of channels in the table. Could be 3 or 4.
                     Default is 3.
    :param target_mode: A mode for the result image. Should have not less
                        than ``channels`` channels. Default is ``None``,
                        which means that mode wouldn't be changed.
    """

    name = "Color 3D LUT"

    def __init__(
        self,
        size: int | tuple[int, int, int],
        table: Sequence[float] | Sequence[Sequence[int]] | NumpyArray,
        channels: int = 3,
        target_mode: str | None = None,
        **kwargs: bool,
    ) -> None:
        if channels not in (3, 4):
            msg = "Only 3 or 4 output channels are supported"
            raise ValueError(msg)
        self.size = size = self._check_size(size)
        self.channels = channels
        self.mode = target_mode

        # Hidden flag `_copy_table=False` could be used to avoid extra copying
        # of the table if the table is specially made for the constructor.
        copy_table = kwargs.get("_copy_table", True)
        items = size[0] * size[1] * size[2]
        wrong_size = False

        numpy: ModuleType | None = None
        if hasattr(table, "shape"):
            try:
                import numpy
            except ImportError:
                pass

        if numpy and isinstance(table, numpy.ndarray):
            numpy_table: NumpyArray = table
            if copy_table:
                numpy_table = numpy_table.copy()

            if numpy_table.shape in [
                (items * channels,),
                (items, channels),
                (size[2], size[1], size[0], channels),
            ]:
                table = numpy_table.reshape(items * channels)
            else:
                wrong_size = True

        else:
            if copy_table:
                table = list(table)

            # Convert to a flat list
            if table and isinstance(table[0], (list, tuple)):
                raw_table = cast(Sequence[Sequence[int]], table)
                flat_table: list[int] = []
                for pixel in raw_table:
                    if len(pixel) != channels:
                        msg = (
                            "The elements of the table should "
                            f"have a length of {channels}."
                        )
                        raise ValueError(msg)
                    flat_table.extend(pixel)
                table = flat_table

        if wrong_size or len(table) != items * channels:
            msg = (
                "The table should have either channels * size**3 float items "
                "or size**3 items of channels-sized tuples with floats. "
                f"Table should be: {channels}x{size[0]}x{size[1]}x{size[2]}. "
                f"Actual length: {len(table)}"
            )
            raise ValueError(msg)
        self.table = table

    @staticmethod
    def _check_size(size: Any) -> tuple[int, int, int]:
        try:
            _, _, _ = size
        except ValueError as e:
            msg = "Size should be either an integer or a tuple of three integers."
            raise ValueError(msg) from e
        except TypeError:
            size = (size, size, size)
        size = tuple(int(x) for x in size)
        for size_1d in size:
            if not 2 <= size_1d <= 65:
                msg = "Size should be in [2, 65] range."
                raise ValueError(msg)
        return size

    @classmethod
    def generate(
        cls,
        size: int | tuple[int, int, int],
        callback: Callable[[float, float, float], tuple[float, ...]],
        channels: int = 3,
        target_mode: str | None = None,
    ) -> Color3DLUT:
        """Generates new LUT using provided callback.

        :param size: Size of the table. Passed to the constructor.
        :param callback: Function with three parameters which correspond
                         three color channels. Will be called ``size**3``
                         times with values from 0.0 to 1.0 and should return
                         a tuple with ``channels`` elements.
        :param channels: The number of channels which should return callback.
        :param target_mode: Passed to the constructor of the resulting
                            lookup table.
        """
        size_1d, size_2d, size_3d = cls._check_size(size)
        if channels not in (3, 4):
            msg = "Only 3 or 4 output channels are supported"
            raise ValueError(msg)

        table: list[float] = [0] * (size_1d * size_2d * size_3d * channels)
        idx_out = 0
        for b in range(size_3d):
            for g in range(size_2d):
                for r in range(size_1d):
                    table[idx_out : idx_out + channels] = callback(
                        r / (size_1d - 1), g / (size_2d - 1), b / (size_3d - 1)
                    )
                    idx_out += channels

        return cls(
            (size_1d, size_2d, size_3d),
            table,
            channels=channels,
            target_mode=target_mode,
            _copy_table=False,
        )

    def transform(
        self,
        callback: Callable[..., tuple[float, ...]],
        with_normals: bool = False,
        channels: int | None = None,
        target_mode: str | None = None,
    ) -> Color3DLUT:
        """Transforms the table values using provided callback and returns
        a new LUT with altered values.

        :param callback: A function which takes old lookup table values
                         and returns a new set of values. The number
                         of arguments which function should take is
                         ``self.channels`` or ``3 + self.channels``
                         if ``with_normals`` flag is set.
                         Should return a tuple of ``self.channels`` or
                         ``channels`` elements if it is set.
        :param with_normals: If true, ``callback`` will be called with
                             coordinates in the color cube as the first
                             three arguments. Otherwise, ``callback``
                             will be called only with actual color values.
        :param channels: The number of channels in the resulting lookup table.
        :param target_mode: Passed to the constructor of the resulting
                            lookup table.
        """
        if channels not in (None, 3, 4):
            msg = "Only 3 or 4 output channels are supported"
            raise ValueError(msg)
        ch_in = self.channels
        ch_out = channels or ch_in
        size_1d, size_2d, size_3d = self.size

        table: list[float] = [0] * (size_1d * size_2d * size_3d * ch_out)
        idx_in = 0
        idx_out = 0
        for b in range(size_3d):
            for g in range(size_2d):
                for r in range(size_1d):
                    values = self.table[idx_in : idx_in + ch_in]
                    if with_normals:
                        values = callback(
                            r / (size_1d - 1),
                            g / (size_2d - 1),
                            b / (size_3d - 1),
                            *values,
                        )
                    else:
                        values = callback(*values)
                    table[idx_out : idx_out + ch_out] = values
                    idx_in += ch_in
                    idx_out += ch_out

        return type(self)(
            self.size,
            table,
            channels=ch_out,
            target_mode=target_mode or self.mode,
            _copy_table=False,
        )

    def __repr__(self) -> str:
        r = [
            f"{self.__class__.__name__} from {self.table.__class__.__name__}",
            "size={:d}x{:d}x{:d}".format(*self.size),
            f"channels={self.channels:d}",
        ]
        if self.mode:
            r.append(f"target_mode={self.mode}")
        return "<{}>".format(" ".join(r))

    def filter(self, image: _imaging.ImagingCore) -> _imaging.ImagingCore:
        from . import Image

        return image.color_lut_3d(
            self.mode or image.mode,
            Image.Resampling.BILINEAR,
            self.channels,
            self.size,
            self.table,
        )
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# a Windows DIB display interface
#
# History:
# 1996-05-20 fl   Created
# 1996-09-20 fl   Fixed subregion exposure
# 1997-09-21 fl   Added draw primitive (for tzPrint)
# 2003-05-21 fl   Added experimental Window/ImageWindow classes
# 2003-09-05 fl   Added fromstring/tostring methods
#
# Copyright (c) Secret Labs AB 1997-2003.
# Copyright (c) Fredrik Lundh 1996-2003.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

from . import Image


class HDC:
    """
    Wraps an HDC integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods.
    """

    def __init__(self, dc: int) -> None:
        self.dc = dc

    def __int__(self) -> int:
        return self.dc


class HWND:
    """
    Wraps an HWND integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods, instead of a DC.
    """

    def __init__(self, wnd: int) -> None:
        self.wnd = wnd

    def __int__(self) -> int:
        return self.wnd


class Dib:
    """
    A Windows bitmap with the given mode and size.  The mode can be one of "1",
    "L", "P", or "RGB".

    If the display requires a palette, this constructor creates a suitable
    palette and associates it with the image. For an "L" image, 128 graylevels
    are allocated. For an "RGB" image, a 6x6x6 colour cube is used, together
    with 20 graylevels.

    To make sure that palettes work properly under Windows, you must call the
    ``palette`` method upon certain events from Windows.

    :param image: Either a PIL image, or a mode string. If a mode string is
                  used, a size must also be given.  The mode can be one of "1",
                  "L", "P", or "RGB".
    :param size: If the first argument is a mode string, this
                 defines the size of the image.
    """

    def __init__(
        self, image: Image.Image | str, size: tuple[int, int] | None = None
    ) -> None:
        if isinstance(image, str):
            mode = image
            image = ""
            if size is None:
                msg = "If first argument is mode, size is required"
                raise ValueError(msg)
        else:
            mode = image.mode
            size = image.size
        if mode not in ["1", "L", "P", "RGB"]:
            mode = Image.getmodebase(mode)
        self.image = Image.core.display(mode, size)
        self.mode = mode
        self.size = size
        if image:
            assert not isinstance(image, str)
            self.paste(image)

    def expose(self, handle: int | HDC | HWND) -> None:
        """
        Copy the bitmap contents to a device context.

        :param handle: Device context (HDC), cast to a Python integer, or an
                       HDC or HWND instance.  In PythonWin, you can use
                       ``CDC.GetHandleAttrib()`` to get a suitable handle.
        """
        handle_int = int(handle)
        if isinstance(handle, HWND):
            dc = self.image.getdc(handle_int)
            try:
                self.image.expose(dc)
            finally:
                self.image.releasedc(handle_int, dc)
        else:
            self.image.expose(handle_int)

    def draw(
        self,
        handle: int | HDC | HWND,
        dst: tuple[int, int, int, int],
        src: tuple[int, int, int, int] | None = None,
    ) -> None:
        """
        Same as expose, but allows you to specify where to draw the image, and
        what part of it to draw.

        The destination and source areas are given as 4-tuple rectangles. If
        the source is omitted, the entire image is copied. If the source and
        the destination have different sizes, the image is resized as
        necessary.
        """
        if src is None:
            src = (0, 0) + self.size
        handle_int = int(handle)
        if isinstance(handle, HWND):
            dc = self.image.getdc(handle_int)
            try:
                self.image.draw(dc, dst, src)
            finally:
                self.image.releasedc(handle_int, dc)
        else:
            self.image.draw(handle_int, dst, src)

    def query_palette(self, handle: int | HDC | HWND) -> int:
        """
        Installs the palette associated with the image in the given device
        context.

        This method should be called upon **QUERYNEWPALETTE** and
        **PALETTECHANGED** events from Windows. If this method returns a
        non-zero value, one or more display palette entries were changed, and
        the image should be redrawn.

        :param handle: Device context (HDC), cast to a Python integer, or an
                       HDC or HWND instance.
        :return: The number of entries that were changed (if one or more entries,
                 this indicates that the image should be redrawn).
        """
        handle_int = int(handle)
        if isinstance(handle, HWND):
            handle = self.image.getdc(handle_int)
            try:
                result = self.image.query_palette(handle)
            finally:
                self.image.releasedc(handle, handle)
        else:
            result = self.image.query_palette(handle_int)
        return result

    def paste(
        self, im: Image.Image, box: tuple[int, int, int, int] | None = None
    ) -> None:
        """
        Paste a PIL image into the bitmap image.

        :param im: A PIL image.  The size must match the target region.
                   If the mode does not match, the image is converted to the
                   mode of the bitmap image.
        :param box: A 4-tuple defining the left, upper, right, and
                    lower pixel coordinate.  See :ref:`coordinate-system`. If
                    None is given instead of a tuple, all of the image is
                    assumed.
        """
        im.load()
        if self.mode != im.mode:
            im = im.convert(self.mode)
        if box:
            self.image.paste(im.im, box)
        else:
            self.image.paste(im.im)

    def frombytes(self, buffer: bytes | memoryview) -> None:
        """
        Load display memory contents from byte data.

        :param buffer: A buffer containing display data (usually
                       data returned from :py:func:`~PIL.ImageWin.Dib.tobytes`)
        """
        self.image.frombytes(buffer)

    def tobytes(self) -> bytes:
        """
        Copy display memory contents to bytes object.

        :return: A bytes object containing display data.
        """
        return self.image.tobytes()


class Window:
    """Create a Window with the given title size."""

    def __init__(
        self, title: str = "PIL", width: int | None = None, height: int | None = None
    ) -> None:
        self.hwnd = Image.core.createwindow(
            title, self.__dispatcher, width or 0, height or 0
        )

    def __dispatcher(self, action: str, *args: int) -> None:
        getattr(self, f"ui_handle_{action}")(*args)

    def ui_handle_clear(self, dc: int, x0: int, y0: int, x1: int, y1: int) -> None:
        pass

    def ui_handle_damage(self, x0: int, y0: int, x1: int, y1: int) -> None:
        pass

    def ui_handle_destroy(self) -> None:
        pass

    def ui_handle_repair(self, dc: int, x0: int, y0: int, x1: int, y1: int) -> None:
        pass

    def ui_handle_resize(self, width: int, height: int) -> None:
        pass

    def mainloop(self) -> None:
        Image.core.eventloop()


class ImageWindow(Window):
    """Create an image window which displays the given image."""

    def __init__(self, image: Image.Image | Dib, title: str = "PIL") -> None:
        if not isinstance(image, Dib):
            image = Dib(image)
        self.image = image
        width, height = image.size
        super().__init__(title, width=width, height=height)

    def ui_handle_repair(self, dc: int, x0: int, y0: int, x1: int, y1: int) -> None:
        self.image.draw(dc, (x0, y0, x1, y1))
<|endoftext|>"""
Blizzard Mipmap Format (.blp)
Jerome Leclanche <jerome@leclan.ch>

The contents of this file are hereby released in the public domain (CC0)
Full text of the CC0 license:
  https://creativecommons.org/publicdomain/zero/1.0/

BLP1 files, used mostly in Warcraft III, are not fully supported.
All types of BLP2 files used in World of Warcraft are supported.

The BLP file structure consists of a header, up to 16 mipmaps of the
texture

Texture sizes must be powers of two, though the two dimensions do
not have to be equal; 512x256 is valid, but 512x200 is not.
The first mipmap (mipmap #0) is the full size image; each subsequent
mipmap halves both dimensions. The final mipmap should be 1x1.

BLP files come in many different flavours:
* JPEG-compressed (type == 0) - only supported for BLP1.
* RAW images (type == 1, encoding == 1). Each mipmap is stored as an
  array of 8-bit values, one per pixel, left to right, top to bottom.
  Each value is an index to the palette.
* DXT-compressed (type == 1, encoding == 2):
- DXT1 compression is used if alpha_encoding == 0.
  - An additional alpha bit is used if alpha_depth == 1.
  - DXT3 compression is used if alpha_encoding == 1.
  - DXT5 compression is used if alpha_encoding == 7.
"""

from __future__ import annotations

import abc
import os
import struct
from enum import IntEnum
from io import BytesIO
from typing import IO

from . import Image, ImageFile


class Format(IntEnum):
    JPEG = 0


class Encoding(IntEnum):
    UNCOMPRESSED = 1
    DXT = 2
    UNCOMPRESSED_RAW_BGRA = 3


class AlphaEncoding(IntEnum):
    DXT1 = 0
    DXT3 = 1
    DXT5 = 7


def unpack_565(i: int) -> tuple[int, int, int]:
    return ((i >> 11) & 0x1F) << 3, ((i >> 5) & 0x3F) << 2, (i & 0x1F) << 3


def decode_dxt1(
    data: bytes, alpha: bool = False
) -> tuple[bytearray, bytearray, bytearray, bytearray]:
    """
    input: one "row" of data (i.e. will produce 4*width pixels)
    """

    blocks = len(data) // 8  # number of blocks in row
    ret = (bytearray(), bytearray(), bytearray(), bytearray())

    for block_index in range(blocks):
        # Decode next 8-byte block.
        idx = block_index * 8
        color0, color1, bits = struct.unpack_from("<HHI", data, idx)

        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)

        # Decode this block into 4x4 pixels
        # Accumulate the results onto our 4 row accumulators
        for j in range(4):
            for i in range(4):
                # get next control op and generate a pixel

                control = bits & 3
                bits = bits >> 2

                a = 0xFF
                if control == 0:
                    r, g, b = r0, g0, b0
                elif control == 1:
                    r, g, b = r1, g1, b1
                elif control == 2:
                    if color0 > color1:
                        r = (2 * r0 + r1) // 3
                        g = (2 * g0 + g1) // 3
                        b = (2 * b0 + b1) // 3
                    else:
                        r = (r0 + r1) // 2
                        g = (g0 + g1) // 2
                        b = (b0 + b1) // 2
                elif control == 3:
                    if color0 > color1:
                        r = (2 * r1 + r0) // 3
                        g = (2 * g1 + g0) // 3
                        b = (2 * b1 + b0) // 3
                    else:
                        r, g, b, a = 0, 0, 0, 0

                if alpha:
                    ret[j].extend([r, g, b, a])
                else:
                    ret[j].extend([r, g, b])

    return ret


def decode_dxt3(data: bytes) -> tuple[bytearray, bytearray, bytearray, bytearray]:
    """
    input: one "row" of data (i.e. will produce 4*width pixels)
    """

    blocks = len(data) // 16  # number of blocks in row
    ret = (bytearray(), bytearray(), bytearray(), bytearray())

    for block_index in range(blocks):
        idx = block_index * 16
        block = data[idx : idx + 16]
        # Decode next 16-byte block.
        bits = struct.unpack_from("<8B", block)
        color0, color1 = struct.unpack_from("<HH", block, 8)

        (code,) = struct.unpack_from("<I", block, 12)

        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)

        for j in range(4):
            high = False  # Do we want the higher bits?
            for i in range(4):
                alphacode_index = (4 * j + i) // 2
                a = bits[alphacode_index]
                if high:
                    high = False
                    a >>= 4
                else:
                    high = True
                    a &= 0xF
                a *= 17  # We get a value between 0 and 15

                color_code = (code >> 2 * (4 * j + i)) & 0x03

                if color_code == 0:
                    r, g, b = r0, g0, b0
                elif color_code == 1:
                    r, g, b = r1, g1, b1
                elif color_code == 2:
                    r = (2 * r0 + r1) // 3
                    g = (2 * g0 + g1) // 3
                    b = (2 * b0 + b1) // 3
                elif color_code == 3:
                    r = (2 * r1 + r0) // 3
                    g = (2 * g1 + g0) // 3
                    b = (2 * b1 + b0) // 3

                ret[j].extend([r, g, b, a])

    return ret


def decode_dxt5(data: bytes) -> tuple[bytearray, bytearray, bytearray, bytearray]:
    """
    input: one "row" of data (i.e. will produce 4 * width pixels)
    """

    blocks = len(data) // 16  # number of blocks in row
    ret = (bytearray(), bytearray(), bytearray(), bytearray())

    for block_index in range(blocks):
        idx = block_index * 16
        block = data[idx : idx + 16]
        # Decode next 16-byte block.
        a0, a1 = struct.unpack_from("<BB", block)

        bits = struct.unpack_from("<6B", block, 2)
        alphacode1 = bits[2] | (bits[3] << 8) | (bits[4] << 16) | (bits[5] << 24)
        alphacode2 = bits[0] | (bits[1] << 8)

        color0, color1 = struct.unpack_from("<HH", block, 8)

        (code,) = struct.unpack_from("<I", block, 12)

        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)

        for j in range(4):
            for i in range(4):
                # get next control op and generate a pixel
                alphacode_index = 3 * (4 * j + i)

                if alphacode_index <= 12:
                    alphacode = (alphacode2 >> alphacode_index) & 0x07
                elif alphacode_index == 15:
                    alphacode = (alphacode2 >> 15) | ((alphacode1 << 1) & 0x06)
                else:  # alphacode_index >= 18 and alphacode_index <= 45
                    alphacode = (alphacode1 >> (alphacode_index - 16)) & 0x07

                if alphacode == 0:
                    a = a0
                elif alphacode == 1:
                    a = a1
                elif a0 > a1:
                    a = ((8 - alphacode) * a0 + (alphacode - 1) * a1) // 7
                elif alphacode == 6:
                    a = 0
                elif alphacode == 7:
                    a = 255
                else:
                    a = ((6 - alphacode) * a0 + (alphacode - 1) * a1) // 5

                color_code = (code >> 2 * (4 * j + i)) & 0x03

                if color_code == 0:
                    r, g, b = r0, g0, b0
                elif color_code == 1:
                    r, g, b = r1, g1, b1
                elif color_code == 2:
                    r = (2 * r0 + r1) // 3
                    g = (2 * g0 + g1) // 3
                    b = (2 * b0 + b1) // 3
                elif color_code == 3:
                    r = (2 * r1 + r0) // 3
                    g = (2 * g1 + g0) // 3
                    b = (2 * b1 + b0) // 3

                ret[j].extend([r, g, b, a])

    return ret


class BLPFormatError(NotImplementedError):
    pass


def _accept(prefix: bytes) -> bool:
    return prefix.startswith((b"BLP1", b"BLP2"))


class BlpImageFile(ImageFile.ImageFile):
    """
    Blizzard Mipmap Format
    """

    format = "BLP"
    format_description = "Blizzard Mipmap Format"

    def _open(self) -> None:
        assert self.fp is not None
        self.magic = self.fp.read(4)
        if not _accept(self.magic):
            msg = f"Bad BLP magic {repr(self.magic)}"
            raise BLPFormatError(msg)

        compression = struct.unpack("<i", self.fp.read(4))[0]
        if self.magic == b"BLP1":
            alpha = struct.unpack("<I", self.fp.read(4))[0] != 0
        else:
            encoding = struct.unpack("<b", self.fp.read(1))[0]
            alpha = struct.unpack("<b", self.fp.read(1))[0] != 0
            alpha_encoding = struct.unpack("<b", self.fp.read(1))[0]
            self.fp.seek(1, os.SEEK_CUR)  # mips

        self._size = struct.unpack("<II", self.fp.read(8))

        args: tuple[int, int, bool] | tuple[int, int, bool, int]
        if self.magic == b"BLP1":
            encoding = struct.unpack("<i", self.fp.read(4))[0]
            self.fp.seek(4, os.SEEK_CUR)  # subtype

            args = (compression, encoding, alpha)
            offset = 28
        else:
            args = (compression, encoding, alpha, alpha_encoding)
            offset = 20

        decoder = self.magic.decode()

        self._mode = "RGBA" if alpha else "RGB"
        self.tile = [ImageFile._Tile(decoder, (0, 0) + self.size, offset, args)]


class _BLPBaseDecoder(abc.ABC, ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        try:
            self._read_header()
            self._load()
        except struct.error as e:
            msg = "Truncated BLP file"
            raise OSError(msg) from e
        return -1, 0

    @abc.abstractmethod
    def _load(self) -> None:
        pass

    def _read_header(self) -> None:
        self._offsets = struct.unpack("<16I", self._safe_read(16 * 4))
        self._lengths = struct.unpack("<16I", self._safe_read(16 * 4))

    def _safe_read(self, length: int) -> bytes:
        assert self.fd is not None
        return ImageFile._safe_read(self.fd, length)

    def _read_palette(self) -> list[tuple[int, int, int, int]]:
        ret = []
        for i in range(256):
            try:
                b, g, r, a = struct.unpack("<4B", self._safe_read(4))
            except struct.error:
                break
            ret.append((b, g, r, a))
        return ret

    def _read_bgra(
        self, palette: list[tuple[int, int, int, int]], alpha: bool
    ) -> bytearray:
        data = bytearray()
        _data = BytesIO(self._safe_read(self._lengths[0]))
        while True:
            try:
                (offset,) = struct.unpack("<B", _data.read(1))
            except struct.error:
                break
            b, g, r, a = palette[offset]
            d: tuple[int, ...] = (r, g, b)
            if alpha:
                d += (a,)
            data.extend(d)
        return data


class BLP1Decoder(_BLPBaseDecoder):
    def _load(self) -> None:
        self._compression, self._encoding, alpha = self.args

        if self._compression == Format.JPEG:
            self._decode_jpeg_stream()

        elif self._compression == 1:
            if self._encoding in (4, 5):
                palette = self._read_palette()
                data = self._read_bgra(palette, alpha)
                self.set_as_raw(data)
            else:
                msg = f"Unsupported BLP encoding {repr(self._encoding)}"
                raise BLPFormatError(msg)
        else:
            msg = f"Unsupported BLP compression {repr(self._encoding)}"
            raise BLPFormatError(msg)

    def _decode_jpeg_stream(self) -> None:
        from .JpegImagePlugin import JpegImageFile

        (jpeg_header_size,) = struct.unpack("<I", self._safe_read(4))
        jpeg_header = self._safe_read(jpeg_header_size)
        assert self.fd is not None
        self._safe_read(self._offsets[0] - self.fd.tell())  # What IS this?
        data = self._safe_read(self._lengths[0])
        data = jpeg_header + data
        image = JpegImageFile(BytesIO(data))
        Image._decompression_bomb_check(image.size)
        if image.mode == "CMYK":
            args = image.tile[0].args
            assert isinstance(args, tuple)
            image.tile = [image.tile[0]._replace(args=(args[0], "CMYK"))]
        self.set_as_raw(image.convert("RGB").tobytes(), "BGR")


class BLP2Decoder(_BLPBaseDecoder):
    def _load(self) -> None:
        self._compression, self._encoding, alpha, self._alpha_encoding = self.args

        palette = self._read_palette()

        assert self.fd is not None
        self.fd.seek(self._offsets[0])

        if self._compression == 1:
            # Uncompressed or DirectX compression

            if self._encoding == Encoding.UNCOMPRESSED:
                data = self._read_bgra(palette, alpha)

            elif self._encoding == Encoding.DXT:
                data = bytearray()
                if self._alpha_encoding == AlphaEncoding.DXT1:
                    linesize = (self.state.xsize + 3) // 4 * 8
                    for yb in range((self.state.ysize + 3) // 4):
                        for d in decode_dxt1(self._safe_read(linesize), alpha):
                            data += d

                elif self._alpha_encoding == AlphaEncoding.DXT3:
                    linesize = (self.state.xsize + 3) // 4 * 16
                    for yb in range((self.state.ysize + 3) // 4):
                        for d in decode_dxt3(self._safe_read(linesize)):
                            data += d

                elif self._alpha_encoding == AlphaEncoding.DXT5:
                    linesize = (self.state.xsize + 3) // 4 * 16
                    for yb in range((self.state.ysize + 3) // 4):
                        for d in decode_dxt5(self._safe_read(linesize)):
                            data += d
                else:
                    msg = f"Unsupported alpha encoding {repr(self._alpha_encoding)}"
                    raise BLPFormatError(msg)
            else:
                msg = f"Unknown BLP encoding {repr(self._encoding)}"
                raise BLPFormatError(msg)

        else:
            msg = f"Unknown BLP compression {repr(self._compression)}"
            raise BLPFormatError(msg)

        self.set_as_raw(data)


class BLPEncoder(ImageFile.PyEncoder):
    _pushes_fd = True

    def _write_palette(self) -> bytes:
        data = b""
        assert self.im is not None
        palette = self.im.getpalette("RGBA", "RGBA")
        for i in range(len(palette) // 4):
            r, g, b, a = palette[i * 4 : (i + 1) * 4]
            data += struct.pack("<4B", b, g, r, a)
        while len(data) < 256 * 4:
            data += b"\x00" * 4
        return data

    def encode(self, bufsize: int) -> tuple[int, int, bytes]:
        palette_data = self._write_palette()

        offset = 20 + 16 * 4 * 2 + len(palette_data)
        data = struct.pack("<16I", offset, *((0,) * 15))

        assert self.im is not None
        w, h = self.im.size
        data += struct.pack("<16I", w * h, *((0,) * 15))

        data += palette_data

        for y in range(h):
            for x in range(w):
                data += struct.pack("<B", self.im.getpixel((x, y)))

        return len(data), 0, data


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode != "P":
        msg = "Unsupported BLP image mode"
        raise ValueError(msg)

    magic = b"BLP1" if im.encoderinfo.get("blp_version") == "BLP1" else b"BLP2"
    fp.write(magic)

    assert im.palette is not None
    fp.write(struct.pack("<i", 1))  # Uncompressed or DirectX compression

    alpha_depth = 1 if im.palette.mode == "RGBA" else 0
    if magic == b"BLP1":
        fp.write(struct.pack("<L", alpha_depth))
    else:
        fp.write(struct.pack("<b", Encoding.UNCOMPRESSED))
        fp.write(struct.pack("<b", alpha_depth))
        fp.write(struct.pack("<b", 0))  # alpha encoding
        fp.write(struct.pack("<b", 0))  # mips
    fp.write(struct.pack("<II", *im.size))
    if magic == b"BLP1":
        fp.write(struct.pack("<i", 5))
        fp.write(struct.pack("<i", 0))

    ImageFile._save(im, fp, [ImageFile._Tile("BLP", (0, 0) + im.size, 0, im.mode)])


Image.register_open(BlpImageFile.format, BlpImageFile, _accept)
Image.register_extension(BlpImageFile.format, ".blp")
Image.register_decoder("BLP1", BLP1Decoder)
Image.register_decoder("BLP2", BLP2Decoder)

Image.register_save(BlpImageFile.format, _save)
Image.register_encoder("BLP", BLPEncoder)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# GIF file handling
#
# History:
# 1995-09-01 fl   Created
# 1996-12-14 fl   Added interlace support
# 1996-12-30 fl   Added animation support
# 1997-01-05 fl   Added write support, fixed local colour map bug
# 1997-02-23 fl   Make sure to load raster data in getdata()
# 1997-07-05 fl   Support external decoder (0.4)
# 1998-07-09 fl   Handle all modes when saving (0.5)
# 1998-07-15 fl   Renamed offset attribute to avoid name clash
# 2001-04-16 fl   Added rewind support (seek to frame 0) (0.6)
# 2001-04-17 fl   Added palette optimization (0.7)
# 2002-06-06 fl   Added transparency support for save (0.8)
# 2004-02-24 fl   Disable interlacing for small images
#
# Copyright (c) 1997-2004 by Secret Labs AB
# Copyright (c) 1995-2004 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import itertools
import math
import os
import subprocess
from enum import IntEnum
from functools import cached_property
from typing import Any, NamedTuple, cast

from . import (
    Image,
    ImageChops,
    ImageFile,
    ImageMath,
    ImageOps,
    ImagePalette,
    ImageSequence,
)
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
from ._util import DeferredError

TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import IO, Literal

    from . import _imaging
    from ._typing import Buffer


class LoadingStrategy(IntEnum):
    """.. versionadded:: 9.1.0"""

    RGB_AFTER_FIRST = 0
    RGB_AFTER_DIFFERENT_PALETTE_ONLY = 1
    RGB_ALWAYS = 2


#: .. versionadded:: 9.1.0
LOADING_STRATEGY = LoadingStrategy.RGB_AFTER_FIRST

# --------------------------------------------------------------------
# Identify/read GIF files


def _accept(prefix: bytes) -> bool:
    return prefix.startswith((b"GIF87a", b"GIF89a"))


##
# Image plugin for GIF images.  This plugin supports both GIF87 and
# GIF89 images.


class GifImageFile(ImageFile.ImageFile):
    format = "GIF"
    format_description = "Compuserve GIF"
    _close_exclusive_fp_after_loading = False

    global_palette = None

    def data(self) -> bytes | None:
        assert self.fp is not None
        s = self.fp.read(1)
        if s and s[0]:
            return self.fp.read(s[0])
        return None

    def _is_palette_needed(self, p: bytes) -> bool:
        for i in range(0, len(p), 3):
            if not (i // 3 == p[i] == p[i + 1] == p[i + 2]):
                return True
        return False

    def _open(self) -> None:
        # Screen
        assert self.fp is not None
        s = self.fp.read(13)
        if not _accept(s):
            msg = "not a GIF file"
            raise SyntaxError(msg)

        self.info["version"] = s[:6]
        self._size = i16(s, 6), i16(s, 8)
        flags = s[10]
        bits = (flags & 7) + 1

        if flags & 128:
            # get global palette
            self.info["background"] = s[11]
            # check if palette contains colour indices
            p = self.fp.read(3 << bits)
            if self._is_palette_needed(p):
                palette = ImagePalette.raw("RGB", p)
                self.global_palette = self.palette = palette

        self._fp = self.fp  # FIXME: hack
        self.__rewind = self.fp.tell()
        self._n_frames: int | None = None
        self._seek(0)  # get ready to read first frame

    @property
    def n_frames(self) -> int:
        if self._n_frames is None:
            current = self.tell()
            try:
                while True:
                    self._seek(self.tell() + 1, False)
            except EOFError:
                self._n_frames = self.tell() + 1
            self.seek(current)
        return self._n_frames

    @cached_property
    def is_animated(self) -> bool:
        if self._n_frames is not None:
            return self._n_frames != 1

        current = self.tell()
        if current:
            return True

        try:
            self._seek(1, False)
            is_animated = True
        except EOFError:
            is_animated = False

        self.seek(current)
        return is_animated

    def seek(self, frame: int) -> None:
        if not self._seek_check(frame):
            return
        if frame < self.__frame:
            self._im = None
            self._seek(0)

        last_frame = self.__frame
        try:
            for f in range(self.__frame + 1, frame + 1):
                self._seek(f)
        except EOFError as e:
            self.seek(last_frame)
            msg = "no more images in GIF file"
            raise EOFError(msg) from e

    def _seek(self, frame: int, update_image: bool = True) -> None:
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex
        if frame == 0:
            # rewind
            self.__offset = 0
            self.dispose: _imaging.ImagingCore | None = None
            self.__frame = -1
            self._fp.seek(self.__rewind)
            self.disposal_method = 0
            if "comment" in self.info:
                del self.info["comment"]
        else:
            # ensure that the previous frame was loaded
