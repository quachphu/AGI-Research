            if self.tile and update_image:
                self.load()

        if frame != self.__frame + 1:
            msg = f"cannot seek to frame {frame}"
            raise ValueError(msg)

        self.fp = self._fp
        if self.__offset:
            # backup to last frame
            self.fp.seek(self.__offset)
            while self.data():
                pass
            self.__offset = 0

        s = self.fp.read(1)
        if not s or s == b";":
            msg = "no more images in GIF file"
            raise EOFError(msg)

        palette: ImagePalette.ImagePalette | Literal[False] | None = None

        info: dict[str, Any] = {}
        frame_transparency = None
        interlace = None
        frame_dispose_extent = None
        while True:
            if not s:
                s = self.fp.read(1)
            if not s or s == b";":
                break

            elif s == b"!":
                #
                # extensions
                #
                s = self.fp.read(1)
                block = self.data()
                if s[0] == 249 and block is not None:
                    #
                    # graphic control extension
                    #
                    flags = block[0]
                    if flags & 1:
                        frame_transparency = block[3]
                    info["duration"] = i16(block, 1) * 10

                    # disposal method - find the value of bits 4 - 6
                    dispose_bits = 0b00011100 & flags
                    dispose_bits = dispose_bits >> 2
                    if dispose_bits:
                        # only set the dispose if it is not
                        # unspecified. I'm not sure if this is
                        # correct, but it seems to prevent the last
                        # frame from looking odd for some animations
                        self.disposal_method = dispose_bits
                elif s[0] == 254:
                    #
                    # comment extension
                    #
                    comment = b""

                    # Read this comment block
                    while block:
                        comment += block
                        block = self.data()

                    if "comment" in info:
                        # If multiple comment blocks in frame, separate with \n
                        info["comment"] += b"\n" + comment
                    else:
                        info["comment"] = comment
                    s = b""
                    continue
                elif s[0] == 255 and frame == 0 and block is not None:
                    #
                    # application extension
                    #
                    info["extension"] = block, self.fp.tell()
                    if block.startswith(b"NETSCAPE2.0"):
                        block = self.data()
                        if block and len(block) >= 3 and block[0] == 1:
                            self.info["loop"] = i16(block, 1)
                while self.data():
                    pass

            elif s == b",":
                #
                # local image
                #
                s = self.fp.read(9)

                # extent
                x0, y0 = i16(s, 0), i16(s, 2)
                x1, y1 = x0 + i16(s, 4), y0 + i16(s, 6)
                if (x1 > self.size[0] or y1 > self.size[1]) and update_image:
                    self._size = max(x1, self.size[0]), max(y1, self.size[1])
                    Image._decompression_bomb_check(self._size)
                frame_dispose_extent = x0, y0, x1, y1
                flags = s[8]

                interlace = (flags & 64) != 0

                if flags & 128:
                    bits = (flags & 7) + 1
                    p = self.fp.read(3 << bits)
                    if self._is_palette_needed(p):
                        palette = ImagePalette.raw("RGB", p)
                    else:
                        palette = False

                # image data
                bits = self.fp.read(1)[0]
                self.__offset = self.fp.tell()
                break
            s = b""

        if interlace is None:
            msg = "image not found in GIF frame"
            raise EOFError(msg)

        self.__frame = frame
        if not update_image:
            return

        self.tile = []

        if self.dispose:
            self.im.paste(self.dispose, self.dispose_extent)

        self._frame_palette = palette if palette is not None else self.global_palette
        self._frame_transparency = frame_transparency
        if frame == 0:
            if self._frame_palette:
                if LOADING_STRATEGY == LoadingStrategy.RGB_ALWAYS:
                    self._mode = "RGBA" if frame_transparency is not None else "RGB"
                else:
                    self._mode = "P"
            else:
                self._mode = "L"

            if palette:
                self.palette = palette
            elif self.global_palette:
                from copy import copy

                self.palette = copy(self.global_palette)
            else:
                self.palette = None
        else:
            if self.mode == "P":
                if (
                    LOADING_STRATEGY != LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY
                    or palette
                ):
                    if "transparency" in self.info:
                        self.im.putpalettealpha(self.info["transparency"], 0)
                        self.im = self.im.convert("RGBA", Image.Dither.FLOYDSTEINBERG)
                        self._mode = "RGBA"
                        del self.info["transparency"]
                    else:
                        self._mode = "RGB"
                        self.im = self.im.convert("RGB", Image.Dither.FLOYDSTEINBERG)

        def _rgb(color: int) -> tuple[int, int, int]:
            if self._frame_palette:
                if color * 3 + 3 > len(self._frame_palette.palette):
                    color = 0
                return cast(
                    tuple[int, int, int],
                    tuple(self._frame_palette.palette[color * 3 : color * 3 + 3]),
                )
            else:
                return (color, color, color)

        self.dispose = None
        self.dispose_extent: tuple[int, int, int, int] | None = frame_dispose_extent
        if self.dispose_extent and self.disposal_method >= 2:
            try:
                if self.disposal_method == 2:
                    # replace with background colour

                    # only dispose the extent in this frame
                    x0, y0, x1, y1 = self.dispose_extent
                    dispose_size = (x1 - x0, y1 - y0)

                    Image._decompression_bomb_check(dispose_size)

                    # by convention, attempt to use transparency first
                    dispose_mode = "P"
                    color = self.info.get("transparency", frame_transparency)
                    if color is not None:
                        if self.mode in ("RGB", "RGBA"):
                            dispose_mode = "RGBA"
                            color = _rgb(color) + (0,)
                    else:
                        color = self.info.get("background", 0)
                        if self.mode in ("RGB", "RGBA"):
                            dispose_mode = "RGB"
                            color = _rgb(color)
                    self.dispose = Image.core.fill(dispose_mode, dispose_size, color)
                else:
                    # replace with previous contents
                    if self._im is not None:
                        # only dispose the extent in this frame
                        self.dispose = self._crop(self.im, self.dispose_extent)
                    elif frame_transparency is not None:
                        x0, y0, x1, y1 = self.dispose_extent
                        dispose_size = (x1 - x0, y1 - y0)

                        Image._decompression_bomb_check(dispose_size)
                        dispose_mode = "P"
                        color = frame_transparency
                        if self.mode in ("RGB", "RGBA"):
                            dispose_mode = "RGBA"
                            color = _rgb(frame_transparency) + (0,)
                        self.dispose = Image.core.fill(
                            dispose_mode, dispose_size, color
                        )
            except AttributeError:
                pass

        if interlace is not None:
            transparency = -1
            if frame_transparency is not None:
                if frame == 0:
                    if LOADING_STRATEGY != LoadingStrategy.RGB_ALWAYS:
                        self.info["transparency"] = frame_transparency
                elif self.mode not in ("RGB", "RGBA"):
                    transparency = frame_transparency
            self.tile = [
                ImageFile._Tile(
                    "gif",
                    (x0, y0, x1, y1),
                    self.__offset,
                    (bits, interlace, transparency),
                )
            ]

        if info.get("comment"):
            self.info["comment"] = info["comment"]
        for k in ["duration", "extension"]:
            if k in info:
                self.info[k] = info[k]
            elif k in self.info:
                del self.info[k]

    def load_prepare(self) -> None:
        temp_mode = "P" if self._frame_palette else "L"
        self._prev_im = None
        if self.__frame == 0:
            if self._frame_transparency is not None:
                self.im = Image.core.fill(
                    temp_mode, self.size, self._frame_transparency
                )
        elif self.mode in ("RGB", "RGBA"):
            self._prev_im = self.im
            if self._frame_palette:
                self.im = Image.core.fill("P", self.size, self._frame_transparency or 0)
                self.im.putpalette("RGB", *self._frame_palette.getdata())
            else:
                self._im = None
        if not self._prev_im and self._im is not None and self.size != self.im.size:
            expanded_im = Image.core.fill(self.im.mode, self.size)
            if self._frame_palette:
                expanded_im.putpalette("RGB", *self._frame_palette.getdata())
            expanded_im.paste(self.im, (0, 0) + self.im.size)

            self.im = expanded_im
        self._mode = temp_mode
        self._frame_palette = None

        super().load_prepare()

    def load_end(self) -> None:
        if self.__frame == 0:
            if self.mode == "P" and LOADING_STRATEGY == LoadingStrategy.RGB_ALWAYS:
                if self._frame_transparency is not None:
                    self.im.putpalettealpha(self._frame_transparency, 0)
                    self._mode = "RGBA"
                else:
                    self._mode = "RGB"
                self.im = self.im.convert(self.mode, Image.Dither.FLOYDSTEINBERG)
            return
        if not self._prev_im:
            return
        if self.size != self._prev_im.size:
            if self._frame_transparency is not None:
                expanded_im = Image.core.fill("RGBA", self.size)
            else:
                expanded_im = Image.core.fill("P", self.size)
                expanded_im.putpalette("RGB", "RGB", self.im.getpalette())
                expanded_im = expanded_im.convert("RGB")
            expanded_im.paste(self._prev_im, (0, 0) + self._prev_im.size)

            self._prev_im = expanded_im
            assert self._prev_im is not None
        if self._frame_transparency is not None:
            if self.mode == "L":
                frame_im = self.im.convert_transparent("LA", self._frame_transparency)
            else:
                self.im.putpalettealpha(self._frame_transparency, 0)
                frame_im = self.im.convert("RGBA")
        else:
            frame_im = self.im.convert("RGB")

        assert self.dispose_extent is not None
        frame_im = self._crop(frame_im, self.dispose_extent)

        self.im = self._prev_im
        self._mode = self.im.mode
        if frame_im.mode in ("LA", "RGBA"):
            self.im.paste(frame_im, self.dispose_extent, frame_im)
        else:
            self.im.paste(frame_im, self.dispose_extent)

    def tell(self) -> int:
        return self.__frame


# --------------------------------------------------------------------
# Write GIF files


RAWMODE = {"1": "L", "L": "L", "P": "P"}


def _normalize_mode(im: Image.Image) -> Image.Image:
    """
    Takes an image (or frame), returns an image in a mode that is appropriate
    for saving in a Gif.

    It may return the original image, or it may return an image converted to
    palette or 'L' mode.

    :param im: Image object
    :returns: Image object
    """
    if im.mode in RAWMODE:
        im.load()
        return im
    if Image.getmodebase(im.mode) == "RGB":
        im = im.convert("P", palette=Image.Palette.ADAPTIVE)
        assert im.palette is not None
        if im.palette.mode == "RGBA":
            for rgba in im.palette.colors:
                if rgba[3] == 0:
                    im.info["transparency"] = im.palette.colors[rgba]
                    break
        return im
    return im.convert("L")


_Palette = bytes | bytearray | list[int] | ImagePalette.ImagePalette


def _normalize_palette(
    im: Image.Image, palette: _Palette | None, info: dict[str, Any]
) -> Image.Image:
    """
    Normalizes the palette for image.
      - Sets the palette to the incoming palette, if provided.
      - Ensures that there's a palette for L mode images
      - Optimizes the palette if necessary/desired.

    :param im: Image object
    :param palette: bytes object containing the source palette, or ....
    :param info: encoderinfo
    :returns: Image object
    """
    source_palette = None
    if palette:
        # a bytes palette
        if isinstance(palette, (bytes, bytearray, list)):
            source_palette = bytearray(palette[:768])
        if isinstance(palette, ImagePalette.ImagePalette):
            source_palette = bytearray(palette.palette)

    if im.mode == "P":
        if not source_palette:
            im_palette = im.getpalette(None)
            assert im_palette is not None
            source_palette = bytearray(im_palette)
    else:  # L-mode
        if not source_palette:
            source_palette = bytearray(i // 3 for i in range(768))
        im.palette = ImagePalette.ImagePalette("RGB", palette=source_palette)
    assert source_palette is not None

    if palette:
        used_palette_colors: list[int | None] = []
        assert im.palette is not None
        for i in range(0, len(source_palette), 3):
            source_color = tuple(source_palette[i : i + 3])
            index = im.palette.colors.get(source_color)
            if index in used_palette_colors:
                index = None
            used_palette_colors.append(index)
        for i, index in enumerate(used_palette_colors):
            if index is None:
                for j in range(len(used_palette_colors)):
                    if j not in used_palette_colors:
                        used_palette_colors[i] = j
                        break
        dest_map: list[int] = []
        for index in used_palette_colors:
            assert index is not None
            dest_map.append(index)
        im = im.remap_palette(dest_map)
    else:
        optimized_palette_colors = _get_optimize(im, info)
        if optimized_palette_colors is not None:
            im = im.remap_palette(optimized_palette_colors, source_palette)
            if "transparency" in info:
                try:
                    info["transparency"] = optimized_palette_colors.index(
                        info["transparency"]
                    )
                except ValueError:
                    del info["transparency"]
            return im

    assert im.palette is not None
    im.palette.palette = source_palette
    return im


def _write_single_frame(
    im: Image.Image,
    fp: IO[bytes],
    palette: _Palette | None,
) -> None:
    im_out = _normalize_mode(im)
    for k, v in im_out.info.items():
        if isinstance(k, str):
            im.encoderinfo.setdefault(k, v)
    im_out = _normalize_palette(im_out, palette, im.encoderinfo)

    for s in _get_global_header(im_out, im.encoderinfo):
        fp.write(s)

    # local image header
    flags = 0
    if get_interlace(im):
        flags = flags | 64
    _write_local_header(fp, im, (0, 0), flags)

    im_out.encoderconfig = (8, get_interlace(im))
    ImageFile._save(
        im_out, fp, [ImageFile._Tile("gif", (0, 0) + im.size, 0, RAWMODE[im_out.mode])]
    )

    fp.write(b"\0")  # end of image data


def _getbbox(
    base_im: Image.Image, im_frame: Image.Image
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    palette_bytes = [
        bytes(im.palette.palette) if im.palette else b"" for im in (base_im, im_frame)
    ]
    if palette_bytes[0] != palette_bytes[1]:
        im_frame = im_frame.convert("RGBA")
        base_im = base_im.convert("RGBA")
    delta = ImageChops.subtract_modulo(im_frame, base_im)
    return delta, delta.getbbox(alpha_only=False)


class _Frame(NamedTuple):
    im: Image.Image
    bbox: tuple[int, int, int, int] | None
    encoderinfo: dict[str, Any]


def _write_multiple_frames(
    im: Image.Image, fp: IO[bytes], palette: _Palette | None
) -> bool:
    duration = im.encoderinfo.get("duration")
    disposal = im.encoderinfo.get("disposal", im.info.get("disposal"))

    im_frames: list[_Frame] = []
    previous_im: Image.Image | None = None
    frame_count = 0
    background_im = None
    for imSequence in itertools.chain([im], im.encoderinfo.get("append_images", [])):
        for im_frame in ImageSequence.Iterator(imSequence):
            # a copy is required here since seek can still mutate the image
            im_frame = _normalize_mode(im_frame.copy())
            if frame_count == 0:
                for k, v in im_frame.info.items():
                    if k == "transparency":
                        continue
                    if isinstance(k, str):
                        im.encoderinfo.setdefault(k, v)

            encoderinfo = im.encoderinfo.copy()
            if "transparency" in im_frame.info:
                encoderinfo.setdefault("transparency", im_frame.info["transparency"])
            im_frame = _normalize_palette(im_frame, palette, encoderinfo)
            if isinstance(duration, (list, tuple)):
                encoderinfo["duration"] = duration[frame_count]
            elif duration is None and "duration" in im_frame.info:
                encoderinfo["duration"] = im_frame.info["duration"]
            if isinstance(disposal, (list, tuple)):
                encoderinfo["disposal"] = disposal[frame_count]
            frame_count += 1

            diff_frame = None
            if im_frames and previous_im:
                # delta frame
                delta, bbox = _getbbox(previous_im, im_frame)
                if not bbox:
                    # This frame is identical to the previous frame
                    if encoderinfo.get("duration"):
                        im_frames[-1].encoderinfo["duration"] += encoderinfo["duration"]
                    continue
                if im_frames[-1].encoderinfo.get("disposal") == 2:
                    # To appear correctly in viewers using a convention,
                    # only consider transparency, and not background color
                    color = im.encoderinfo.get(
                        "transparency", im.info.get("transparency")
                    )
                    if color is not None:
                        if background_im is None:
                            background = _get_background(im_frame, color)
                            background_im = Image.new("P", im_frame.size, background)
                            first_palette = im_frames[0].im.palette
                            assert first_palette is not None
                            background_im.putpalette(first_palette, first_palette.mode)
                        bbox = _getbbox(background_im, im_frame)[1]
                    else:
                        bbox = (0, 0) + im_frame.size
                elif encoderinfo.get("optimize") and im_frame.mode != "1":
                    if "transparency" not in encoderinfo:
                        assert im_frame.palette is not None
                        try:
                            encoderinfo["transparency"] = (
                                im_frame.palette._new_color_index(im_frame)
                            )
                        except ValueError:
                            pass
                    if "transparency" in encoderinfo:
                        # When the delta is zero, fill the image with transparency
                        diff_frame = im_frame.copy()
                        fill = Image.new("P", delta.size, encoderinfo["transparency"])
                        if delta.mode == "RGBA":
                            r, g, b, a = delta.split()
                            mask = ImageMath.lambda_eval(
                                lambda args: args["convert"](
                                    args["max"](
                                        args["max"](
                                            args["max"](args["r"], args["g"]), args["b"]
                                        ),
                                        args["a"],
                                    )
                                    * 255,
                                    "1",
                                ),
                                r=r,
                                g=g,
                                b=b,
                                a=a,
                            )
                        else:
                            if delta.mode == "P":
                                # Convert to L without considering palette
                                delta_l = Image.new("L", delta.size)
                                delta_l.putdata(delta.get_flattened_data())
                                delta = delta_l
                            mask = ImageMath.lambda_eval(
                                lambda args: args["convert"](args["im"] * 255, "1"),
                                im=delta,
                            )
                        diff_frame.paste(fill, mask=ImageOps.invert(mask))
            else:
                bbox = None
            previous_im = im_frame
            im_frames.append(_Frame(diff_frame or im_frame, bbox, encoderinfo))

    if len(im_frames) == 1:
        if "duration" in im.encoderinfo:
            # Since multiple frames will not be written, use the combined duration
            im.encoderinfo["duration"] = im_frames[0].encoderinfo["duration"]
        return False

    for frame_data in im_frames:
        im_frame = frame_data.im
        if not frame_data.bbox:
            # global header
            for s in _get_global_header(im_frame, frame_data.encoderinfo):
                fp.write(s)
            offset = (0, 0)
        else:
            # compress difference
            if not palette:
                frame_data.encoderinfo["include_color_table"] = True

            if frame_data.bbox != (0, 0) + im_frame.size:
                im_frame = im_frame.crop(frame_data.bbox)
            offset = frame_data.bbox[:2]
        _write_frame_data(fp, im_frame, offset, frame_data.encoderinfo)
    return True


def _save_all(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    _save(im, fp, filename, save_all=True)


def _save(
    im: Image.Image, fp: IO[bytes], filename: str | bytes, save_all: bool = False
) -> None:
    # header
    if "palette" in im.encoderinfo or "palette" in im.info:
        palette = im.encoderinfo.get("palette", im.info.get("palette"))
    else:
        palette = None
        im.encoderinfo.setdefault("optimize", True)

    if not save_all or not _write_multiple_frames(im, fp, palette):
        _write_single_frame(im, fp, palette)

    fp.write(b";")  # end of file

    if hasattr(fp, "flush"):
        fp.flush()


def get_interlace(im: Image.Image) -> int:
    interlace = im.encoderinfo.get("interlace", 1)

    # workaround for @PIL153
    if min(im.size) < 16:
        interlace = 0

    return interlace


def _write_local_header(
    fp: IO[bytes], im: Image.Image, offset: tuple[int, int], flags: int
) -> None:
    try:
        transparency = im.encoderinfo["transparency"]
    except KeyError:
        transparency = None

    if "duration" in im.encoderinfo:
        duration = int(im.encoderinfo["duration"] / 10)
    else:
        duration = 0

    disposal = int(im.encoderinfo.get("disposal", 0))

    if transparency is not None or duration != 0 or disposal:
        packed_flag = 1 if transparency is not None else 0
        packed_flag |= disposal << 2

        fp.write(
            b"!"
            + o8(249)  # extension intro
            + o8(4)  # length
            + o8(packed_flag)  # packed fields
            + o16(duration)  # duration
            + o8(transparency or 0)  # transparency index
            + o8(0)
        )

    include_color_table = im.encoderinfo.get("include_color_table")
    if include_color_table:
        palette_bytes = _get_palette_bytes(im)
        color_table_size = _get_color_table_size(palette_bytes)
        if color_table_size:
            flags = flags | 128  # local color table flag
            flags = flags | color_table_size

    fp.write(
        b","
        + o16(offset[0])  # offset
        + o16(offset[1])
        + o16(im.size[0])  # size
        + o16(im.size[1])
        + o8(flags)  # flags
    )
    if include_color_table and color_table_size:
        fp.write(_get_header_palette(palette_bytes))
    fp.write(o8(8))  # bits


def _save_netpbm(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    # Unused by default.
    # To use, uncomment the register_save call at the end of the file.
    #
    # If you need real GIF compression and/or RGB quantization, you
    # can use the external NETPBM/PBMPLUS utilities.  See comments
    # below for information on how to enable this.
    tempfile = im._dump()

    try:
        with open(filename, "wb") as f:
            if im.mode != "RGB":
                subprocess.check_call(
                    ["ppmtogif", tempfile], stdout=f, stderr=subprocess.DEVNULL
                )
            else:
                # Pipe ppmquant output into ppmtogif
                # "ppmquant 256 %s | ppmtogif > %s" % (tempfile, filename)
                quant_cmd = ["ppmquant", "256", tempfile]
                togif_cmd = ["ppmtogif"]
                quant_proc = subprocess.Popen(
                    quant_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                togif_proc = subprocess.Popen(
                    togif_cmd,
                    stdin=quant_proc.stdout,
                    stdout=f,
                    stderr=subprocess.DEVNULL,
                )

                # Allow ppmquant to receive SIGPIPE if ppmtogif exits
                assert quant_proc.stdout is not None
                quant_proc.stdout.close()

                retcode = quant_proc.wait()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, quant_cmd)

                retcode = togif_proc.wait()
                if retcode:
                    raise subprocess.CalledProcessError(retcode, togif_cmd)
    finally:
        try:
            os.unlink(tempfile)
        except OSError:
            pass


# Force optimization so that we can test performance against
# cases where it took lots of memory and time previously.
_FORCE_OPTIMIZE = False


def _get_optimize(im: Image.Image, info: dict[str, Any]) -> list[int] | None:
    """
    Palette optimization is a potentially expensive operation.

    This function determines if the palette should be optimized using
    some heuristics, then returns the list of palette entries in use.

    :param im: Image object
    :param info: encoderinfo
    :returns: list of indexes of palette entries in use, or None
    """
    if (
        im.mode in ("P", "L")
        and info
        and info.get("optimize")
        and im.width != 0
        and im.height != 0
    ):
        # Potentially expensive operation.

        # The palette saves 3 bytes per color not used, but palette
        # lengths are restricted to 3*(2**N) bytes. Max saving would
        # be 768 -> 6 bytes if we went all the way down to 2 colors.
        # * If we're over 128 colors, we can't save any space.
        # * If there aren't any holes, it's not worth collapsing.
        # * If we have a 'large' image, the palette is in the noise.

        # create the new palette if not every color is used
        optimise = _FORCE_OPTIMIZE or im.mode == "L"
        if optimise or im.width * im.height < 512 * 512:
            # check which colors are used
            used_palette_colors = []
            for i, count in enumerate(im.histogram()):
                if count:
                    used_palette_colors.append(i)

            if optimise or max(used_palette_colors) >= len(used_palette_colors):
                return used_palette_colors

            assert im.palette is not None
            num_palette_colors = len(im.palette.palette) // Image.getmodebands(
                im.palette.mode
            )
            current_palette_size = 1 << (num_palette_colors - 1).bit_length()
            if (
                # check that the palette would become smaller when saved
                len(used_palette_colors) <= current_palette_size // 2
                # check that the palette is not already the smallest possible size
                and current_palette_size > 2
            ):
                return used_palette_colors
    return None


def _get_color_table_size(palette_bytes: bytes) -> int:
    # calculate the palette size for the header
    if not palette_bytes:
        return 0
    elif len(palette_bytes) < 9:
        return 1
    else:
        return math.ceil(math.log(len(palette_bytes) // 3, 2)) - 1


def _get_header_palette(palette_bytes: bytes) -> bytes:
    """
    Returns the palette, null padded to the next power of 2 (*3) bytes
    suitable for direct inclusion in the GIF header

    :param palette_bytes: Unpadded palette bytes, in RGBRGB form
    :returns: Null padded palette
    """
    color_table_size = _get_color_table_size(palette_bytes)

    # add the missing amount of bytes
    # the palette has to be 2<<n in size
    actual_target_size_diff = (2 << color_table_size) - len(palette_bytes) // 3
    if actual_target_size_diff > 0:
        palette_bytes += o8(0) * 3 * actual_target_size_diff
    return palette_bytes


def _get_palette_bytes(im: Image.Image) -> bytes:
    """
    Gets the palette for inclusion in the gif header

    :param im: Image object
    :returns: Bytes, len<=768 suitable for inclusion in gif header
    """
    if not im.palette:
        return b""

    palette = bytes(im.palette.palette)
    if im.palette.mode == "RGBA":
        palette = b"".join(palette[i * 4 : i * 4 + 3] for i in range(len(palette) // 3))
    return palette


def _get_background(
    im: Image.Image,
    info_background: int | tuple[int, int, int] | tuple[int, int, int, int] | None,
) -> int:
    background = 0
    if info_background:
        if isinstance(info_background, tuple):
            # WebPImagePlugin stores an RGBA value in info["background"]
            # So it must be converted to the same format as GifImagePlugin's
            # info["background"] - a global color table index
            assert im.palette is not None
            try:
                background = im.palette.getcolor(info_background, im)
            except ValueError as e:
                if str(e) not in (
                    # If all 256 colors are in use,
                    # then there is no need for the background color
                    "cannot allocate more than 256 colors",
                    # Ignore non-opaque WebP background
                    "cannot add non-opaque RGBA color to RGB palette",
                ):
                    raise
        else:
            background = info_background
    return background


def _get_global_header(im: Image.Image, info: dict[str, Any]) -> list[bytes]:
    """Return a list of strings representing a GIF header"""

    # Header Block
    # https://www.matthewflickinger.com/lab/whatsinagif/bits_and_bytes.asp

    version = b"87a"
    if im.info.get("version") == b"89a" or (
        info
        and (
            "transparency" in info
            or info.get("loop") is not None
            or info.get("duration")
            or info.get("comment")
        )
    ):
        version = b"89a"

    background = _get_background(im, info.get("background"))

    palette_bytes = _get_palette_bytes(im)
    color_table_size = _get_color_table_size(palette_bytes)

    header = [
        b"GIF"  # signature
        + version  # version
        + o16(im.size[0])  # canvas width
        + o16(im.size[1]),  # canvas height
        # Logical Screen Descriptor
        # size of global color table + global color table flag
        o8(color_table_size + 128),  # packed fields
        # background + reserved/aspect
        o8(background) + o8(0),
        # Global Color Table
        _get_header_palette(palette_bytes),
    ]
    if info.get("loop") is not None:
        header.append(
            b"!"
            + o8(255)  # extension intro
            + o8(11)
            + b"NETSCAPE2.0"
            + o8(3)
            + o8(1)
            + o16(info["loop"])  # number of loops
            + o8(0)
        )
    if info.get("comment"):
        comment_block = b"!" + o8(254)  # extension intro

        comment = info["comment"]
        if isinstance(comment, str):
            comment = comment.encode()
        for i in range(0, len(comment), 255):
            subblock = comment[i : i + 255]
            comment_block += o8(len(subblock)) + subblock

        comment_block += o8(0)
        header.append(comment_block)
    return header


def _write_frame_data(
    fp: IO[bytes],
    im_frame: Image.Image,
    offset: tuple[int, int],
    params: dict[str, Any],
) -> None:
    try:
        im_frame.encoderinfo = params

        # local image header
        _write_local_header(fp, im_frame, offset, 0)

        ImageFile._save(
            im_frame,
            fp,
            [ImageFile._Tile("gif", (0, 0) + im_frame.size, 0, RAWMODE[im_frame.mode])],
        )

        fp.write(b"\0")  # end of image data
    finally:
        del im_frame.encoderinfo


# --------------------------------------------------------------------
# Legacy GIF utilities


def getheader(
    im: Image.Image, palette: _Palette | None = None, info: dict[str, Any] | None = None
) -> tuple[list[bytes], list[int] | None]:
    """
    Legacy Method to get Gif data from image.

    Warning:: May modify image data.

    :param im: Image object
    :param palette: bytes object containing the source palette, or ....
    :param info: encoderinfo
    :returns: tuple of(list of header items, optimized palette)

    """
    if info is None:
        info = {}

    used_palette_colors = _get_optimize(im, info)

    if "background" not in info and "background" in im.info:
        info["background"] = im.info["background"]

    im_mod = _normalize_palette(im, palette, info)
    im.palette = im_mod.palette
    im.im = im_mod.im
    header = _get_global_header(im, info)

    return header, used_palette_colors


def getdata(
    im: Image.Image, offset: tuple[int, int] = (0, 0), **params: Any
) -> list[bytes]:
    """
    Legacy Method

    Return a list of strings representing this image.
    The first string is a local image header, the rest contains
    encoded image data.

    To specify duration, add the time in milliseconds,
    e.g. ``getdata(im_frame, duration=1000)``

    :param im: Image object
    :param offset: Tuple of (x, y) pixels. Defaults to (0, 0)
    :param \\**params: e.g. duration or other encoder info parameters
    :returns: List of bytes containing GIF encoded frame data

    """
    from io import BytesIO

    class Collector(BytesIO):
        data = []

        def write(self, data: Buffer) -> int:
            self.data.append(data)
            return len(data)

    im.load()  # make sure raster data is available

    fp = Collector()

    _write_frame_data(fp, im, offset, params)

    return fp.data


# --------------------------------------------------------------------
# Registry

Image.register_open(GifImageFile.format, GifImageFile, _accept)
Image.register_save(GifImageFile.format, _save)
Image.register_save_all(GifImageFile.format, _save_all)
Image.register_extension(GifImageFile.format, ".gif")
Image.register_mime(GifImageFile.format, "image/gif")

#
# Uncomment the following line if you wish to use NETPBM/PBMPLUS
# instead of the built-in "uncompressed" GIF encoder

# Image.register_save(GifImageFile.format, _save_netpbm)
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# global image statistics
#
# History:
# 1996-04-05 fl   Created
# 1997-05-21 fl   Added mask; added rms, var, stddev attributes
# 1997-08-05 fl   Added median
# 1998-07-05 hk   Fixed integer overflow error
#
# Notes:
# This class shows how to implement delayed evaluation of attributes.
# To get a certain value, simply access the corresponding attribute.
# The __getattr__ dispatcher takes care of the rest.
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996-97.
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import math
from functools import cached_property

from . import Image


class Stat:
    def __init__(
        self, image_or_list: Image.Image | list[int], mask: Image.Image | None = None
    ) -> None:
        """
        Calculate statistics for the given image. If a mask is included,
        only the regions covered by that mask are included in the
        statistics. You can also pass in a previously calculated histogram.

        :param image: A PIL image, or a precalculated histogram.

            .. note::

                For a PIL image, calculations rely on the
                :py:meth:`~PIL.Image.Image.histogram` method. The pixel counts are
                grouped into 256 bins, even if the image has more than 8 bits per
                channel. So ``I`` and ``F`` mode images have a maximum ``mean``,
                ``median`` and ``rms`` of 255, and cannot have an ``extrema`` maximum
                of more than 255.

        :param mask: An optional mask.
        """
        if isinstance(image_or_list, Image.Image):
            self.h = image_or_list.histogram(mask)
        elif isinstance(image_or_list, list):
            self.h = image_or_list
        else:
            msg = "first argument must be image or list"  # type: ignore[unreachable]
            raise TypeError(msg)
        self.bands = list(range(len(self.h) // 256))

    @cached_property
    def extrema(self) -> list[tuple[int, int]]:
        """
        Min/max values for each band in the image.

        .. note::
            This relies on the :py:meth:`~PIL.Image.Image.histogram` method, and
            simply returns the low and high bins used. This is correct for
            images with 8 bits per channel, but fails for other modes such as
            ``I`` or ``F``. Instead, use :py:meth:`~PIL.Image.Image.getextrema` to
            return per-band extrema for the image. This is more correct and
            efficient because, for non-8-bit modes, the histogram method uses
            :py:meth:`~PIL.Image.Image.getextrema` to determine the bins used.
        """

        def minmax(histogram: list[int]) -> tuple[int, int]:
            res_min, res_max = 255, 0
            for i in range(256):
                if histogram[i]:
                    res_min = i
                    break
            for i in range(255, -1, -1):
                if histogram[i]:
                    res_max = i
                    break
            return res_min, res_max

        return [minmax(self.h[i:]) for i in range(0, len(self.h), 256)]

    @cached_property
    def count(self) -> list[int]:
        """Total number of pixels for each band in the image."""
        return [sum(self.h[i : i + 256]) for i in range(0, len(self.h), 256)]

    @cached_property
    def sum(self) -> list[float]:
        """Sum of all pixels for each band in the image."""

        v = []
        for i in range(0, len(self.h), 256):
            layer_sum = 0.0
            for j in range(256):
                layer_sum += j * self.h[i + j]
            v.append(layer_sum)
        return v

    @cached_property
    def sum2(self) -> list[float]:
        """Squared sum of all pixels for each band in the image."""

        v = []
        for i in range(0, len(self.h), 256):
            sum2 = 0.0
            for j in range(256):
                sum2 += (j**2) * float(self.h[i + j])
            v.append(sum2)
        return v

    @cached_property
    def mean(self) -> list[float]:
        """Average (arithmetic mean) pixel level for each band in the image."""
        return [self.sum[i] / self.count[i] if self.count[i] else 0 for i in self.bands]

    @cached_property
    def median(self) -> list[int]:
        """Median pixel level for each band in the image."""

        v = []
        for i in self.bands:
            s = 0
            half = self.count[i] // 2
            b = i * 256
            for j in range(256):
                s = s + self.h[b + j]
                if s > half:
                    break
            v.append(j)
        return v

    @cached_property
    def rms(self) -> list[float]:
        """RMS (root-mean-square) for each band in the image."""
        return [
            math.sqrt(self.sum2[i] / self.count[i]) if self.count[i] else 0
            for i in self.bands
        ]

    @cached_property
    def var(self) -> list[float]:
        """Variance for each band in the image."""
        return [
            (
                (self.sum2[i] - (self.sum[i] ** 2.0) / self.count[i]) / self.count[i]
                if self.count[i]
                else 0
            )
            for i in self.bands
        ]

    @cached_property
    def stddev(self) -> list[float]:
        """Standard deviation for each band in the image."""
        return [math.sqrt(self.var[i]) for i in self.bands]


Global = Stat  # compatibility
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# PIL raster font management
#
# History:
# 1996-08-07 fl   created (experimental)
# 1997-08-25 fl   minor adjustments to handle fonts from pilfont 0.3
# 1999-02-06 fl   rewrote most font management stuff in C
# 1999-03-17 fl   take pth files into account in load_path (from Richard Jones)
# 2001-02-17 fl   added freetype support
# 2001-05-09 fl   added TransposedFont wrapper class
# 2002-03-04 fl   make sure we have a "L" or "1" font
# 2002-12-04 fl   skip non-directory entries in the system path
# 2003-04-29 fl   add embedded default font
# 2003-09-27 fl   added support for truetype charmap encodings
#
# Todo:
# Adapt to PILFONT2 format (16-bit fonts, compressed, single file)
#
# Copyright (c) 1997-2003 by Secret Labs AB
# Copyright (c) 1996-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

from __future__ import annotations

import abc
import base64
import os
import sys
import warnings
from enum import IntEnum
from io import BytesIO
from types import ModuleType
from typing import IO, Any, BinaryIO, TypedDict, cast

from . import Image
from ._typing import StrOrBytesPath
from ._util import DeferredError, is_path

TYPE_CHECKING = False
if TYPE_CHECKING:
    from . import ImageFile
    from ._imaging import ImagingFont
    from ._imagingft import Font


class Axis(TypedDict):
    minimum: int | None
    default: int | None
    maximum: int | None
    name: bytes | None


class Layout(IntEnum):
    BASIC = 0
    RAQM = 1


MAX_STRING_LENGTH = 1_000_000


core: ModuleType | DeferredError
try:
    from . import _imagingft as core
except ImportError as ex:
    core = DeferredError.new(ex)


def _string_length_check(text: str | bytes | bytearray) -> None:
    if MAX_STRING_LENGTH is not None and len(text) > MAX_STRING_LENGTH:
        msg = "too many characters in string"
        raise ValueError(msg)


# FIXME: add support for pilfont2 format (see FontFile.py)

# --------------------------------------------------------------------
# Font metrics format:
#       "PILfont" LF
#       fontdescriptor LF
#       (optional) key=value... LF
#       "DATA" LF
#       binary data: 256*10*2 bytes (dx, dy, dstbox, srcbox)
#
# To place a character, cut out srcbox and paste at dstbox,
# relative to the character position.  Then move the character
# position according to dx, dy.
# --------------------------------------------------------------------


class BaseImageFont(abc.ABC):
    """Used by ImageDraw and ImageText"""

    @abc.abstractmethod
    def getbbox(
        self, text: str | bytes | bytearray, *args: Any, **kwargs: Any
    ) -> tuple[float, float, float, float]:
        pass

    @abc.abstractmethod
    def getmask(
        self, text: str | bytes, mode: str = "", *args: Any, **kwargs: Any
    ) -> Image.core.ImagingCore:
        pass

    @abc.abstractmethod
    def getlength(self, text: str | bytes, *args: Any, **kwargs: Any) -> float:
        pass


class ImageFont(BaseImageFont):
    """PIL font wrapper"""

    font: ImagingFont

    def _load_pilfont(self, filename: str) -> None:
        with open(filename, "rb") as fp:
            image: ImageFile.ImageFile | None = None
            root = os.path.splitext(filename)[0]

            for ext in (".png", ".gif", ".pbm"):
                if image:
                    image.close()
                try:
                    fullname = root + ext
                    image = Image.open(fullname)
                except Exception:
                    pass
                else:
                    if image.mode in ("1", "L"):
                        break
            else:
                if image:
                    image.close()

                msg = f"cannot find glyph data file {root}.{{gif|pbm|png}}"
                raise OSError(msg)

            self.file = fullname

            self._load_pilfont_data(fp, image)
            image.close()

    def _load_pilfont_data(self, file: IO[bytes], image: Image.Image) -> None:
        # check image
        if image.mode not in ("1", "L"):
            image.close()

            msg = "invalid font image mode"
            raise TypeError(msg)

        # read PILfont header
        if file.read(8) != b"PILfont\n":
            image.close()

            msg = "Not a PILfont file"
            raise SyntaxError(msg)
        file.readline()
        self.info = []  # FIXME: should be a dictionary
        while True:
            s = file.readline()
            if not s or s == b"DATA\n":
                break
            self.info.append(s)

        # read PILfont metrics
        data = file.read(256 * 20)

        self._load(image, data)

    def _load(self, image: Image.Image, data: bytes) -> None:
        image.load()

        self.font = Image.core.font(image.im, data)

    def getmask(
        self, text: str | bytes, mode: str = "", *args: Any, **kwargs: Any
    ) -> Image.core.ImagingCore:
        """
        Create a bitmap for the text.

        If the font uses antialiasing, the bitmap should have mode ``L`` and use a
        maximum value of 255. Otherwise, it should have mode ``1``.

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

                     .. versionadded:: 1.1.5

        :return: An internal PIL storage memory instance as defined by the
                 :py:mod:`PIL.Image.core` interface module.
        """
        _string_length_check(text)
        Image._decompression_bomb_check(self.font.getsize(text))
        return self.font.getmask(text, mode)

    def getbbox(
        self, text: str | bytes | bytearray, *args: Any, **kwargs: Any
    ) -> tuple[int, int, int, int]:
        """
        Returns bounding box (in pixels) of given text.

        .. versionadded:: 9.2.0

        :param text: Text to render.

        :return: ``(left, top, right, bottom)`` bounding box
        """
        _string_length_check(text)
        width, height = self.font.getsize(text)
        return 0, 0, width, height

    def getlength(
        self, text: str | bytes | bytearray, *args: Any, **kwargs: Any
    ) -> int:
        """
        Returns length (in pixels) of given text.
        This is the amount by which following text should be offset.

        .. versionadded:: 9.2.0
        """
        _string_length_check(text)
        width, height = self.font.getsize(text)
        return width


##
# Wrapper for FreeType fonts.  Application code should use the
# <b>truetype</b> factory function to create font objects.


class FreeTypeFont(BaseImageFont):
    """FreeType font wrapper (requires _imagingft service)"""

    font: Font
    font_bytes: bytes

    def __init__(
        self,
        font: StrOrBytesPath | BinaryIO,
        size: float = 10,
        index: int = 0,
        encoding: str = "",
        layout_engine: Layout | None = None,
    ) -> None:
        # FIXME: use service provider instead

        if isinstance(core, DeferredError):
            raise core.ex

        if size <= 0:
            msg = f"font size must be greater than 0, not {size}"
            raise ValueError(msg)

        self.path = font
        self.size = size
        self.index = index
        self.encoding = encoding

        if layout_engine not in (Layout.BASIC, Layout.RAQM):
            layout_engine = Layout.BASIC
            if core.HAVE_RAQM:
                layout_engine = Layout.RAQM
        elif layout_engine == Layout.RAQM and not core.HAVE_RAQM:
            warnings.warn(
                "Raqm layout was requested, but Raqm is not available. "
                "Falling back to basic layout."
            )
            layout_engine = Layout.BASIC

        self.layout_engine = layout_engine

        def load_from_bytes(f: IO[bytes]) -> None:
            self.font_bytes = f.read()
            self.font = core.getfont(
                "", size, index, encoding, self.font_bytes, layout_engine
            )

        if is_path(font):
            font = os.fspath(font)
            if sys.platform == "win32":
                font_bytes_path = font if isinstance(font, bytes) else font.encode()
                try:
                    font_bytes_path.decode("ascii")
                except UnicodeDecodeError:
                    # FreeType cannot load fonts with non-ASCII characters on Windows
                    # So load it into memory first
                    with open(font, "rb") as f:
                        load_from_bytes(f)
                    return
            self.font = core.getfont(
                font, size, index, encoding, layout_engine=layout_engine
            )
        else:
            load_from_bytes(cast(IO[bytes], font))

    def __getstate__(self) -> list[Any]:
        return [self.path, self.size, self.index, self.encoding, self.layout_engine]

    def __setstate__(self, state: list[Any]) -> None:
        path, size, index, encoding, layout_engine = state
        FreeTypeFont.__init__(self, path, size, index, encoding, layout_engine)

    def getname(self) -> tuple[str | None, str | None]:
        """
        :return: A tuple of the font family (e.g. Helvetica) and the font style
            (e.g. Bold)
        """
        return self.font.family, self.font.style

    def getmetrics(self) -> tuple[int, int]:
        """
        :return: A tuple of the font ascent (the distance from the baseline to
            the highest outline point) and descent (the distance from the
            baseline to the lowest outline point, a negative value)
        """
        return self.font.ascent, self.font.descent

    def getlength(
        self,
        text: str | bytes,
        mode: str = "",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
    ) -> float:
        """
        Returns length (in pixels with 1/64 precision) of given text when rendered
        in font with provided direction, features, and language.

        This is the amount by which following text should be offset.
        Text bounding box may extend past the length in some fonts,
        e.g. when using italics or accents.

        The result is returned as a float; it is a whole number if using basic layout.

        Note that the sum of two lengths may not equal the length of a concatenated
        string due to kerning. If you need to adjust for kerning, include the following
        character and subtract its length.

        For example, instead of ::

          hello = font.getlength("Hello")
          world = font.getlength("World")
          hello_world = hello + world  # not adjusted for kerning
          assert hello_world == font.getlength("HelloWorld")  # may fail

        use ::

          hello = font.getlength("HelloW") - font.getlength("W")  # adjusted for kerning
          world = font.getlength("World")
          hello_world = hello + world  # adjusted for kerning
          assert hello_world == font.getlength("HelloWorld")  # True

        or disable kerning with (requires libraqm) ::

          hello = draw.textlength("Hello", font, features=["-kern"])
          world = draw.textlength("World", font, features=["-kern"])
          hello_world = hello + world  # kerning is disabled, no need to adjust
          assert hello_world == draw.textlength("HelloWorld", font, features=["-kern"])

        .. versionadded:: 8.0.0

        :param text: Text to measure.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

        :return: Either width for horizontal text, or height for vertical text.
        """
        _string_length_check(text)
        return self.font.getlength(text, mode, direction, features, language) / 64

    def getbbox(
        self,
        text: str | bytes | bytearray,
        mode: str = "",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        anchor: str | None = None,
    ) -> tuple[float, float, float, float]:
        """
        Returns bounding box (in pixels) of given text relative to given anchor
        when rendered in font with provided direction, features, and language.

        Use :py:meth:`getlength()` to get the offset of following text with
        1/64 pixel precision. The bounding box includes extra margins for
        some fonts, e.g. italics or accents.

        .. versionadded:: 8.0.0

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

        :param stroke_width: The width of the text stroke.

        :param anchor:  The text anchor alignment. Determines the relative location of
                        the anchor to the text. The default alignment is top left,
                        specifically ``la`` for horizontal text and ``lt`` for
                        vertical text. See :ref:`text-anchors` for details.

        :return: ``(left, top, right, bottom)`` bounding box
        """
        _string_length_check(text)
        size, offset = self.font.getsize(
            text, mode, direction, features, language, anchor
        )
        left, top = offset[0] - stroke_width, offset[1] - stroke_width
        width, height = size[0] + 2 * stroke_width, size[1] + 2 * stroke_width
        return left, top, left + width, top + height

    def getmask(
        self,
        text: str | bytes,
        mode: str = "",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        anchor: str | None = None,
        ink: int = 0,
        start: tuple[float, float] | None = None,
    ) -> Image.core.ImagingCore:
        """
        Create a bitmap for the text.

        If the font uses antialiasing, the bitmap should have mode ``L`` and use a
        maximum value of 255. If the font has embedded color data, the bitmap
        should have mode ``RGBA``. Otherwise, it should have mode ``1``.

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

                     .. versionadded:: 1.1.5

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

                          .. versionadded:: 4.2.0

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

                         .. versionadded:: 4.2.0

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

                         .. versionadded:: 6.0.0

        :param stroke_width: The width of the text stroke.

                         .. versionadded:: 6.2.0

        :param anchor:  The text anchor alignment. Determines the relative location of
                        the anchor to the text. The default alignment is top left,
                        specifically ``la`` for horizontal text and ``lt`` for
                        vertical text. See :ref:`text-anchors` for details.

                         .. versionadded:: 8.0.0

        :param ink: Foreground ink for rendering in RGBA mode.

                         .. versionadded:: 8.0.0

        :param start: Tuple of horizontal and vertical offset, as text may render
                      differently when starting at fractional coordinates.

                         .. versionadded:: 9.4.0

        :return: An internal PIL storage memory instance as defined by the
                 :py:mod:`PIL.Image.core` interface module.
        """
        return self.getmask2(
            text,
            mode,
            direction=direction,
            features=features,
            language=language,
            stroke_width=stroke_width,
            anchor=anchor,
            ink=ink,
            start=start,
        )[0]

    def getmask2(
        self,
        text: str | bytes,
        mode: str = "",
        direction: str | None = None,
        features: list[str] | None = None,
        language: str | None = None,
        stroke_width: float = 0,
        anchor: str | None = None,
        ink: int = 0,
        start: tuple[float, float] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Image.core.ImagingCore, tuple[int, int]]:
        """
        Create a bitmap for the text.

        If the font uses antialiasing, the bitmap should have mode ``L`` and use a
        maximum value of 255. If the font has embedded color data, the bitmap
        should have mode ``RGBA``. Otherwise, it should have mode ``1``.

        :param text: Text to render.
        :param mode: Used by some graphics drivers to indicate what mode the
                     driver prefers; if empty, the renderer may return either
                     mode. Note that the mode is always a string, to simplify
                     C-level implementations.

                     .. versionadded:: 1.1.5

        :param direction: Direction of the text. It can be 'rtl' (right to
                          left), 'ltr' (left to right) or 'ttb' (top to bottom).
                          Requires libraqm.

                          .. versionadded:: 4.2.0

        :param features: A list of OpenType font features to be used during text
                         layout. This is usually used to turn on optional
                         font features that are not enabled by default,
                         for example 'dlig' or 'ss01', but can be also
                         used to turn off default font features for
                         example '-liga' to disable ligatures or '-kern'
                         to disable kerning.  To get all supported
                         features, see
                         https://learn.microsoft.com/en-us/typography/opentype/spec/featurelist
                         Requires libraqm.

                         .. versionadded:: 4.2.0

        :param language: Language of the text. Different languages may use
                         different glyph shapes or ligatures. This parameter tells
                         the font which language the text is in, and to apply the
                         correct substitutions as appropriate, if available.
                         It should be a `BCP 47 language code
                         <https://www.w3.org/International/articles/language-tags/>`_
                         Requires libraqm.

                         .. versionadded:: 6.0.0

        :param stroke_width: The width of the text stroke.

                         .. versionadded:: 6.2.0

        :param anchor:  The text anchor alignment. Determines the relative location of
                        the anchor to the text. The default alignment is top left,
                        specifically ``la`` for horizontal text and ``lt`` for
                        vertical text. See :ref:`text-anchors` for details.

                         .. versionadded:: 8.0.0

        :param ink: Foreground ink for rendering in RGBA mode.

                         .. versionadded:: 8.0.0

        :param start: Tuple of horizontal and vertical offset, as text may render
                      differently when starting at fractional coordinates.

                         .. versionadded:: 9.4.0

        :return: A tuple of an internal PIL storage memory instance as defined by the
                 :py:mod:`PIL.Image.core` interface module, and the text offset, the
                 gap between the starting coordinate and the first marking
        """
        _string_length_check(text)
        if start is None:
            start = (0, 0)

        def fill(width: int, height: int) -> Image.core.ImagingCore:
            size = (width, height)
            Image._decompression_bomb_check(size)
            return Image.core.fill("RGBA" if mode == "RGBA" else "L", size)

        return self.font.render(
            text,
            fill,
            mode,
            direction,
            features,
            language,
            stroke_width,
            kwargs.get("stroke_filled", False),
            anchor,
            ink,
            start,
        )

    def font_variant(
        self,
        font: StrOrBytesPath | BinaryIO | None = None,
        size: float | None = None,
        index: int | None = None,
        encoding: str | None = None,
        layout_engine: Layout | None = None,
    ) -> FreeTypeFont:
        """
        Create a copy of this FreeTypeFont object,
        using any specified arguments to override the settings.

        Parameters are identical to the parameters used to initialize this
        object.

        :return: A FreeTypeFont object.
        """
        if font is None:
            try:
                font = BytesIO(self.font_bytes)
            except AttributeError:
                font = self.path
        return FreeTypeFont(
            font=font,
            size=self.size if size is None else size,
            index=self.index if index is None else index,
            encoding=self.encoding if encoding is None else encoding,
            layout_engine=layout_engine or self.layout_engine,
        )

    def get_variation_names(self) -> list[bytes]:
        """
        :returns: A list of the named styles in a variation font.
        :exception OSError: If the font is not a variation font.
        """
        names = []
        for name in self.font.getvarnames():
            name = name.replace(b"\x00", b"")
            if name not in names:
                names.append(name)
        return names

    def set_variation_by_name(self, name: str | bytes) -> None:
        """
        :param name: The name of the style.
        :exception OSError: If the font is not a variation font.
        """
        names = self.get_variation_names()
        if not isinstance(name, bytes):
            name = name.encode()
        index = names.index(name) + 1

        if index == getattr(self, "_last_variation_index", None):
            # When the same name is set twice in a row,
            # there is an 'unknown freetype error'
            # https://savannah.nongnu.org/bugs/?56186
            return
        self._last_variation_index = index

        self.font.setvarname(index)

    def get_variation_axes(self) -> list[Axis]:
        """
        :returns: A list of the axes in a variation font.
        :exception OSError: If the font is not a variation font.
        """
        axes = self.font.getvaraxes()
        for axis in axes:
            if axis["name"]:
                axis["name"] = axis["name"].replace(b"\x00", b"")
        return axes

    def set_variation_by_axes(self, axes: list[float]) -> None:
        """
        :param axes: A list of values for each axis.
        :exception OSError: If the font is not a variation font.
        """
        self.font.setvaraxes(axes)


class TransposedFont(BaseImageFont):
    """Wrapper for writing rotated or mirrored text"""

    def __init__(
        self, font: ImageFont | FreeTypeFont, orientation: Image.Transpose | None = None
    ):
        """
        Wrapper that creates a transposed font from any existing font
        object.

        :param font: A font object.
        :param orientation: An optional orientation.  If given, this should
            be one of Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, or
            Image.Transpose.ROTATE_270.
        """
        self.font = font
        self.orientation = orientation  # any 'transpose' argument, or None

    def getmask(
        self, text: str | bytes, mode: str = "", *args: Any, **kwargs: Any
    ) -> Image.core.ImagingCore:
        im = self.font.getmask(text, mode, *args, **kwargs)
        if self.orientation is not None:
            return im.transpose(self.orientation)
        return im

    def getbbox(
        self, text: str | bytes | bytearray, *args: Any, **kwargs: Any
    ) -> tuple[int, int, float, float]:
        # TransposedFont doesn't support getmask2, move top-left point to (0, 0)
        # this has no effect on ImageFont and simulates anchor="lt" for FreeTypeFont
        left, top, right, bottom = self.font.getbbox(text, *args, **kwargs)
        width = right - left
        height = bottom - top
        if self.orientation in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270):
            return 0, 0, height, width
        return 0, 0, width, height

    def getlength(self, text: str | bytes, *args: Any, **kwargs: Any) -> float:
        if self.orientation in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270):
            msg = "text length is undefined for text rotated by 90 or 270 degrees"
            raise ValueError(msg)
        return self.font.getlength(text, *args, **kwargs)


def load(filename: str) -> ImageFont:
    """
    Load a font file. This function loads a font object from the given
    bitmap font file, and returns the corresponding font object. For loading TrueType
    or OpenType fonts instead, see :py:func:`~PIL.ImageFont.truetype`.

    :param filename: Name of font file.
    :return: A font object.
    :exception OSError: If the file could not be read.
    """
    f = ImageFont()
    f._load_pilfont(filename)
    return f


def truetype(
    font: StrOrBytesPath | BinaryIO,
    size: float = 10,
    index: int = 0,
    encoding: str = "",
    layout_engine: Layout | None = None,
) -> FreeTypeFont:
    """
    Load a TrueType or OpenType font from a file or file-like object,
    and create a font object. This function loads a font object from the given
    file or file-like object, and creates a font object for a font of the given
    size. For loading bitmap fonts instead, see :py:func:`~PIL.ImageFont.load`
    and :py:func:`~PIL.ImageFont.load_path`.

    Pillow uses FreeType to open font files. On Windows, be aware that FreeType
    will keep the file open as long as the FreeTypeFont object exists. Windows
    limits the number of files that can be open in C at once to 512, so if many
    fonts are opened simultaneously and that limit is approached, an
    ``OSError`` may be thrown, reporting that FreeType "cannot open resource".
    A workaround would be to copy the file(s) into memory, and open that instead.

    This function requires the _imagingft service.

    :param font: A filename or file-like object containing a TrueType font.
                 If the file is not found in this filename, the loader may also
                 search in other directories, such as:

                 * The :file:`fonts/` directory on Windows,
                 * :file:`/Library/Fonts/`, :file:`/System/Library/Fonts/`
                   and :file:`~/Library/Fonts/` on macOS.
                 * :file:`~/.local/share/fonts`, :file:`/usr/local/share/fonts`,
                   and :file:`/usr/share/fonts` on Linux; or those specified by
                   the ``XDG_DATA_HOME`` and ``XDG_DATA_DIRS`` environment variables
                   for user-installed and system-wide fonts, respectively.

    :param size: The requested size, in pixels.
    :param index: Which font face to load (default is first available face).
    :param encoding: Which font encoding to use (default is Unicode). Possible
                     encodings include (see the FreeType documentation for more
                     information):

                     * "unic" (Unicode)
                     * "symb" (Microsoft Symbol)
                     * "ADOB" (Adobe Standard)
                     * "ADBE" (Adobe Expert)
                     * "ADBC" (Adobe Custom)
                     * "armn" (Apple Roman)
                     * "sjis" (Shift JIS)
                     * "gb  " (PRC)
                     * "big5"
                     * "wans" (Extended Wansung)
                     * "joha" (Johab)
                     * "lat1" (Latin-1)

                     This specifies the character set to use. It does not alter the
                     encoding of any text provided in subsequent operations.
    :param layout_engine: Which layout engine to use, if available:
                     :attr:`.ImageFont.Layout.BASIC` or :attr:`.ImageFont.Layout.RAQM`.
                     If it is available, Raqm layout will be used by default.
                     Otherwise, basic layout will be used.

                     Raqm layout is recommended for all non-English text. If Raqm layout
                     is not required, basic layout will have better performance.

                     You can check support for Raqm layout using
                     :py:func:`PIL.features.check_feature` with ``feature="raqm"``.

                     .. versionadded:: 4.2.0
    :return: A font object.
    :exception OSError: If the file could not be read.
    :exception ValueError: If the font size is not greater than zero.
    """

    def freetype(font: StrOrBytesPath | BinaryIO) -> FreeTypeFont:
        return FreeTypeFont(font, size, index, encoding, layout_engine)

    try:
        return freetype(font)
    except OSError:
        if not is_path(font):
            raise
        ttf_filename = os.path.basename(font)

        dirs = []
        if sys.platform == "win32":
            # check the windows font repository
            # NOTE: must use uppercase WINDIR, to work around bugs in
            # 1.5.2's os.environ.get()
            windir = os.environ.get("WINDIR")
            if windir:
                dirs.append(os.path.join(windir, "fonts"))
        elif sys.platform in ("linux", "linux2"):
            data_home = os.environ.get("XDG_DATA_HOME")
            if not data_home:
                # The freedesktop spec defines the following default directory for
                # when XDG_DATA_HOME is unset or empty. This user-level directory
                # takes precedence over system-level directories.
                data_home = os.path.expanduser("~/.local/share")
            xdg_dirs = [data_home]

            data_dirs = os.environ.get("XDG_DATA_DIRS")
            if not data_dirs:
                # Similarly, defaults are defined for the system-level directories
                data_dirs = "/usr/local/share:/usr/share"
            xdg_dirs += data_dirs.split(":")

            dirs += [os.path.join(xdg_dir, "fonts") for xdg_dir in xdg_dirs]
        elif sys.platform == "darwin":
            dirs += [
                "/Library/Fonts",
                "/System/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
            ]

        ext = os.path.splitext(ttf_filename)[1]
        first_font_with_a_different_extension = None
        for directory in dirs:
            for walkroot, walkdir, walkfilenames in os.walk(directory):
                for walkfilename in walkfilenames:
                    if ext and walkfilename == ttf_filename:
                        return freetype(os.path.join(walkroot, walkfilename))
                    elif not ext and os.path.splitext(walkfilename)[0] == ttf_filename:
                        fontpath = os.path.join(walkroot, walkfilename)
                        if os.path.splitext(fontpath)[1] == ".ttf":
                            return freetype(fontpath)
                        if not ext and first_font_with_a_different_extension is None:
                            first_font_with_a_different_extension = fontpath
        if first_font_with_a_different_extension:
            return freetype(first_font_with_a_different_extension)
        raise


def load_path(filename: str | bytes) -> ImageFont:
    """
    Load font file. Same as :py:func:`~PIL.ImageFont.load`, but searches for a
    bitmap font along the Python path.

    :param filename: Name of font file.
    :return: A font object.
    :exception OSError: If the file could not be read.
    """
    if not isinstance(filename, str):
        filename = filename.decode("utf-8")
    for directory in sys.path:
        try:
            return load(os.path.join(directory, filename))
        except OSError:  # noqa: PERF203
            pass
    msg = f'cannot find font file "{filename}" in sys.path'
    if os.path.exists(filename):
        msg += f', did you mean ImageFont.load("{filename}") instead?'

    raise OSError(msg)


def load_default_imagefont() -> ImageFont:
    f = ImageFont()
    f._load_pilfont_data(
        # courB08
        BytesIO(base64.b64decode(b"""
UElMZm9udAo7Ozs7OzsxMDsKREFUQQoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYAAAAA//8AAQAAAAAAAAABAAEA
BgAAAAH/+gADAAAAAQAAAAMABgAGAAAAAf/6AAT//QADAAAABgADAAYAAAAA//kABQABAAYAAAAL
AAgABgAAAAD/+AAFAAEACwAAABAACQAGAAAAAP/5AAUAAAAQAAAAFQAHAAYAAP////oABQAAABUA
AAAbAAYABgAAAAH/+QAE//wAGwAAAB4AAwAGAAAAAf/5AAQAAQAeAAAAIQAIAAYAAAAB//kABAAB
ACEAAAAkAAgABgAAAAD/+QAE//0AJAAAACgABAAGAAAAAP/6AAX//wAoAAAALQAFAAYAAAAB//8A
BAACAC0AAAAwAAMABgAAAAD//AAF//0AMAAAADUAAQAGAAAAAf//AAMAAAA1AAAANwABAAYAAAAB
//kABQABADcAAAA7AAgABgAAAAD/+QAFAAAAOwAAAEAABwAGAAAAAP/5AAYAAABAAAAARgAHAAYA
AAAA//kABQAAAEYAAABLAAcABgAAAAD/+QAFAAAASwAAAFAABwAGAAAAAP/5AAYAAABQAAAAVgAH
AAYAAAAA//kABQAAAFYAAABbAAcABgAAAAD/+QAFAAAAWwAAAGAABwAGAAAAAP/5AAUAAABgAAAA
ZQAHAAYAAAAA//kABQAAAGUAAABqAAcABgAAAAD/+QAFAAAAagAAAG8ABwAGAAAAAf/8AAMAAABv
AAAAcQAEAAYAAAAA//wAAwACAHEAAAB0AAYABgAAAAD/+gAE//8AdAAAAHgABQAGAAAAAP/7AAT/
/gB4AAAAfAADAAYAAAAB//oABf//AHwAAACAAAUABgAAAAD/+gAFAAAAgAAAAIUABgAGAAAAAP/5
AAYAAQCFAAAAiwAIAAYAAP////oABgAAAIsAAACSAAYABgAA////+gAFAAAAkgAAAJgABgAGAAAA
AP/6AAUAAACYAAAAnQAGAAYAAP////oABQAAAJ0AAACjAAYABgAA////+gAFAAAAowAAAKkABgAG
AAD////6AAUAAACpAAAArwAGAAYAAAAA//oABQAAAK8AAAC0AAYABgAA////+gAGAAAAtAAAALsA
BgAGAAAAAP/6AAQAAAC7AAAAvwAGAAYAAP////oABQAAAL8AAADFAAYABgAA////+gAGAAAAxQAA
AMwABgAGAAD////6AAUAAADMAAAA0gAGAAYAAP////oABQAAANIAAADYAAYABgAA////+gAGAAAA
2AAAAN8ABgAGAAAAAP/6AAUAAADfAAAA5AAGAAYAAP////oABQAAAOQAAADqAAYABgAAAAD/+gAF
AAEA6gAAAO8ABwAGAAD////6AAYAAADvAAAA9gAGAAYAAAAA//oABQAAAPYAAAD7AAYABgAA////
+gAFAAAA+wAAAQEABgAGAAD////6AAYAAAEBAAABCAAGAAYAAP////oABgAAAQgAAAEPAAYABgAA
////+gAGAAABDwAAARYABgAGAAAAAP/6AAYAAAEWAAABHAAGAAYAAP////oABgAAARwAAAEjAAYA
BgAAAAD/+gAFAAABIwAAASgABgAGAAAAAf/5AAQAAQEoAAABKwAIAAYAAAAA//kABAABASsAAAEv
AAgABgAAAAH/+QAEAAEBLwAAATIACAAGAAAAAP/5AAX//AEyAAABNwADAAYAAAAAAAEABgACATcA
AAE9AAEABgAAAAH/+QAE//wBPQAAAUAAAwAGAAAAAP/7AAYAAAFAAAABRgAFAAYAAP////kABQAA
AUYAAAFMAAcABgAAAAD/+wAFAAABTAAAAVEABQAGAAAAAP/5AAYAAAFRAAABVwAHAAYAAAAA//sA
BQAAAVcAAAFcAAUABgAAAAD/+QAFAAABXAAAAWEABwAGAAAAAP/7AAYAAgFhAAABZwAHAAYAAP//
//kABQAAAWcAAAFtAAcABgAAAAD/+QAGAAABbQAAAXMABwAGAAAAAP/5AAQAAgFzAAABdwAJAAYA
AP////kABgAAAXcAAAF+AAcABgAAAAD/+QAGAAABfgAAAYQABwAGAAD////7AAUAAAGEAAABigAF
AAYAAP////sABQAAAYoAAAGQAAUABgAAAAD/+wAFAAABkAAAAZUABQAGAAD////7AAUAAgGVAAAB
mwAHAAYAAAAA//sABgACAZsAAAGhAAcABgAAAAD/+wAGAAABoQAAAacABQAGAAAAAP/7AAYAAAGn
AAABrQAFAAYAAAAA//kABgAAAa0AAAGzAAcABgAA////+wAGAAABswAAAboABQAGAAD////7AAUA
AAG6AAABwAAFAAYAAP////sABgAAAcAAAAHHAAUABgAAAAD/+wAGAAABxwAAAc0ABQAGAAD////7
AAYAAgHNAAAB1AAHAAYAAAAA//sABQAAAdQAAAHZAAUABgAAAAH/+QAFAAEB2QAAAd0ACAAGAAAA
Av/6AAMAAQHdAAAB3gAHAAYAAAAA//kABAABAd4AAAHiAAgABgAAAAD/+wAF//0B4gAAAecAAgAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYAAAAB
//sAAwACAecAAAHpAAcABgAAAAD/+QAFAAEB6QAAAe4ACAAGAAAAAP/5AAYAAAHuAAAB9AAHAAYA
AAAA//oABf//AfQAAAH5AAUABgAAAAD/+QAGAAAB+QAAAf8ABwAGAAAAAv/5AAMAAgH/AAACAAAJ
AAYAAAAA//kABQABAgAAAAIFAAgABgAAAAH/+gAE//sCBQAAAggAAQAGAAAAAP/5AAYAAAIIAAAC
DgAHAAYAAAAB//kABf/+Ag4AAAISAAUABgAA////+wAGAAACEgAAAhkABQAGAAAAAP/7AAX//gIZ
AAACHgADAAYAAAAA//wABf/9Ah4AAAIjAAEABgAAAAD/+QAHAAACIwAAAioABwAGAAAAAP/6AAT/
+wIqAAACLgABAAYAAAAA//kABP/8Ai4AAAIyAAMABgAAAAD/+gAFAAACMgAAAjcABgAGAAAAAf/5
AAT//QI3AAACOgAEAAYAAAAB//kABP/9AjoAAAI9AAQABgAAAAL/+QAE//sCPQAAAj8AAgAGAAD/
///7AAYAAgI/AAACRgAHAAYAAAAA//kABgABAkYAAAJMAAgABgAAAAH//AAD//0CTAAAAk4AAQAG
AAAAAf//AAQAAgJOAAACUQADAAYAAAAB//kABP/9AlEAAAJUAAQABgAAAAH/+QAF//4CVAAAAlgA
BQAGAAD////7AAYAAAJYAAACXwAFAAYAAP////kABgAAAl8AAAJmAAcABgAA////+QAGAAACZgAA
Am0ABwAGAAD////5AAYAAAJtAAACdAAHAAYAAAAA//sABQACAnQAAAJ5AAcABgAA////9wAGAAAC
eQAAAoAACQAGAAD////3AAYAAAKAAAAChwAJAAYAAP////cABgAAAocAAAKOAAkABgAA////9wAG
AAACjgAAApUACQAGAAD////4AAYAAAKVAAACnAAIAAYAAP////cABgAAApwAAAKjAAkABgAA////
+gAGAAACowAAAqoABgAGAAAAAP/6AAUAAgKqAAACrwAIAAYAAP////cABQAAAq8AAAK1AAkABgAA
////9wAFAAACtQAAArsACQAGAAD////3AAUAAAK7AAACwQAJAAYAAP////gABQAAAsEAAALHAAgA
BgAAAAD/9wAEAAACxwAAAssACQAGAAAAAP/3AAQAAALLAAACzwAJAAYAAAAA//cABAAAAs8AAALT
AAkABgAAAAD/+AAEAAAC0wAAAtcACAAGAAD////6AAUAAALXAAAC3QAGAAYAAP////cABgAAAt0A
AALkAAkABgAAAAD/9wAFAAAC5AAAAukACQAGAAAAAP/3AAUAAALpAAAC7gAJAAYAAAAA//cABQAA
Au4AAALzAAkABgAAAAD/9wAFAAAC8wAAAvgACQAGAAAAAP/4AAUAAAL4AAAC/QAIAAYAAAAA//oA
Bf//Av0AAAMCAAUABgAA////+gAGAAADAgAAAwkABgAGAAD////3AAYAAAMJAAADEAAJAAYAAP//
//cABgAAAxAAAAMXAAkABgAA////9wAGAAADFwAAAx4ACQAGAAD////4AAYAAAAAAAoABwASAAYA
AP////cABgAAAAcACgAOABMABgAA////+gAFAAAADgAKABQAEAAGAAD////6AAYAAAAUAAoAGwAQ
AAYAAAAA//gABgAAABsACgAhABIABgAAAAD/+AAGAAAAIQAKACcAEgAGAAAAAP/4AAYAAAAnAAoA
LQASAAYAAAAA//gABgAAAC0ACgAzABIABgAAAAD/+QAGAAAAMwAKADkAEQAGAAAAAP/3AAYAAAA5
AAoAPwATAAYAAP////sABQAAAD8ACgBFAA8ABgAAAAD/+wAFAAIARQAKAEoAEQAGAAAAAP/4AAUA
AABKAAoATwASAAYAAAAA//gABQAAAE8ACgBUABIABgAAAAD/+AAFAAAAVAAKAFkAEgAGAAAAAP/5
AAUAAABZAAoAXgARAAYAAAAA//gABgAAAF4ACgBkABIABgAAAAD/+AAGAAAAZAAKAGoAEgAGAAAA
AP/4AAYAAABqAAoAcAASAAYAAAAA//kABgAAAHAACgB2ABEABgAAAAD/+AAFAAAAdgAKAHsAEgAG
AAD////4AAYAAAB7AAoAggASAAYAAAAA//gABQAAAIIACgCHABIABgAAAAD/+AAFAAAAhwAKAIwA
EgAGAAAAAP/4AAUAAACMAAoAkQASAAYAAAAA//gABQAAAJEACgCWABIABgAAAAD/+QAFAAAAlgAK
AJsAEQAGAAAAAP/6AAX//wCbAAoAoAAPAAYAAAAA//oABQABAKAACgClABEABgAA////+AAGAAAA
pQAKAKwAEgAGAAD////4AAYAAACsAAoAswASAAYAAP////gABgAAALMACgC6ABIABgAA////+QAG
AAAAugAKAMEAEQAGAAD////4AAYAAgDBAAoAyAAUAAYAAP////kABQACAMgACgDOABMABgAA////
+QAGAAIAzgAKANUAEw==
""")),
        Image.open(BytesIO(base64.b64decode(b"""
iVBORw0KGgoAAAANSUhEUgAAAx4AAAAUAQAAAAArMtZoAAAEwElEQVR4nABlAJr/AHVE4czCI/4u
Mc4b7vuds/xzjz5/3/7u/n9vMe7vnfH/9++vPn/xyf5zhxzjt8GHw8+2d83u8x27199/nxuQ6Od9
M43/5z2I+9n9ZtmDBwMQECDRQw/eQIQohJXxpBCNVE6QCCAAAAD//wBlAJr/AgALyj1t/wINwq0g
LeNZUworuN1cjTPIzrTX6ofHWeo3v336qPzfEwRmBnHTtf95/fglZK5N0PDgfRTslpGBvz7LFc4F
IUXBWQGjQ5MGCx34EDFPwXiY4YbYxavpnhHFrk14CDAAAAD//wBlAJr/AgKqRooH2gAgPeggvUAA
Bu2WfgPoAwzRAABAAAAAAACQgLz/3Uv4Gv+gX7BJgDeeGP6AAAD1NMDzKHD7ANWr3loYbxsAD791
NAADfcoIDyP44K/jv4Y63/Z+t98Ovt+ub4T48LAAAAD//wBlAJr/AuplMlADJAAAAGuAphWpqhMx
in0A/fRvAYBABPgBwBUgABBQ/sYAyv9g0bCHgOLoGAAAAAAAREAAwI7nr0ArYpow7aX8//9LaP/9
SjdavWA8ePHeBIKB//81/83ndznOaXx379wAAAD//wBlAJr/AqDxW+D3AABAAbUh/QMnbQag/gAY
AYDAAACgtgD/gOqAAAB5IA/8AAAk+n9w0AAA8AAAmFRJuPo27ciC0cD5oeW4E7KA/wD3ECMAn2tt
y8PgwH8AfAxFzC0JzeAMtratAsC/ffwAAAD//wBlAJr/BGKAyCAA4AAAAvgeYTAwHd1kmQF5chkG
ABoMIHcL5xVpTfQbUqzlAAAErwAQBgAAEOClA5D9il08AEh/tUzdCBsXkbgACED+woQg8Si9VeqY
lODCn7lmF6NhnAEYgAAA/NMIAAAAAAD//2JgjLZgVGBg5Pv/Tvpc8hwGBjYGJADjHDrAwPzAjv/H
/Wf3PzCwtzcwHmBgYGcwbZz8wHaCAQMDOwMDQ8MCBgYOC3W7mp+f0w+wHOYxO3OG+e376hsMZjk3
AAAAAP//YmCMY2A4wMAIN5e5gQETPD6AZisDAwMDgzSDAAPjByiHcQMDAwMDg1nOze1lByRu5/47
c4859311AYNZzg0AAAAA//9iYGDBYihOIIMuwIjGL39/fwffA8b//xv/P2BPtzzHwCBjUQAAAAD/
/yLFBrIBAAAA//9i1HhcwdhizX7u8NZNzyLbvT97bfrMf/QHI8evOwcSqGUJAAAA//9iYBB81iSw
pEE170Qrg5MIYydHqwdDQRMrAwcVrQAAAAD//2J4x7j9AAMDn8Q/BgYLBoaiAwwMjPdvMDBYM1Tv
oJodAAAAAP//Yqo/83+dxePWlxl3npsel9lvLfPcqlE9725C+acfVLMEAAAA//9i+s9gwCoaaGMR
evta/58PTEWzr21hufPjA8N+qlnBwAAAAAD//2JiWLci5v1+HmFXDqcnULE/MxgYGBj+f6CaJQAA
AAD//2Ji2FrkY3iYpYC5qDeGgeEMAwPDvwQBBoYvcTwOVLMEAAAA//9isDBgkP///0EOg9z35v//
Gc/eeW7BwPj5+QGZhANUswMAAAD//2JgqGBgYGBgqEMXlvhMPUsAAAAA//8iYDd1AAAAAP//AwDR
w7IkEbzhVQAAAABJRU5ErkJggg==
"""))),
    )
    return f


def load_default(size: float | None = None) -> FreeTypeFont | ImageFont:
    """If FreeType support is available, load a version of Aileron Regular,
    https://dotcolon.net/fonts/aileron, with a more limited character set.

    Otherwise, load a "better than nothing" font.

    .. versionadded:: 1.1.4

    :param size: The font size of Aileron Regular.

        .. versionadded:: 10.1.0

    :return: A font object.
    """
    if isinstance(core, ModuleType) or size is not None:
        return truetype(
            BytesIO(base64.b64decode(b"""
AAEAAAAPAIAAAwBwRkZUTYwDlUAAADFoAAAAHEdERUYAqADnAAAo8AAAACRHUE9ThhmITwAAKfgAA
AduR1NVQnHxefoAACkUAAAA4k9TLzJovoHLAAABeAAAAGBjbWFw5lFQMQAAA6gAAAGqZ2FzcP//AA
MAACjoAAAACGdseWYmRXoPAAAGQAAAHfhoZWFkE18ayQAAAPwAAAA2aGhlYQboArEAAAE0AAAAJGh
tdHjjERZ8AAAB2AAAAdBsb2NhuOexrgAABVQAAADqbWF4cAC7AEYAAAFYAAAAIG5hbWUr+h5lAAAk
OAAAA6Jwb3N0D3oPTQAAJ9wAAAEKAAEAAAABGhxJDqIhXw889QALA+gAAAAA0Bqf2QAAAADhCh2h/
2r/LgOxAyAAAAAIAAIAAAAAAAAAAQAAA8r/GgAAA7j/av9qA7EAAQAAAAAAAAAAAAAAAAAAAHQAAQ
AAAHQAQwAFAAAAAAACAAAAAQABAAAAQAAAAAAAAAADAfoBkAAFAAgCigJYAAAASwKKAlgAAAFeADI
BPgAAAAAFAAAAAAAAAAAAAAcAAAAAAAAAAAAAAABVS1dOAEAAIPsCAwL/GgDIA8oA5iAAAJMAAAAA
AhICsgAAACAAAwH0AAAAAAAAAU0AAADYAAAA8gA5AVMAVgJEAEYCRAA1AuQAKQKOAEAAsAArATsAZ
AE7AB4CMABVAkQAUADc/+EBEgAgANwAJQEv//sCRAApAkQAggJEADwCRAAtAkQAIQJEADkCRAArAk
QAMgJEACwCRAAxANwAJQDc/+ECRABnAkQAUAJEAEQB8wAjA1QANgJ/AB0CcwBkArsALwLFAGQCSwB
kAjcAZALGAC8C2gBkAQgAZAIgADcCYQBkAj8AZANiAGQCzgBkAuEALwJWAGQC3QAvAmsAZAJJADQC
ZAAiAqoAXgJuACADuAAaAnEAGQJFABMCTwAuATMAYgEv//sBJwAiAkQAUAH0ADIBLAApAhMAJAJjA
EoCEQAeAmcAHgIlAB4BIgAVAmcAHgJRAEoA7gA+AOn/8wIKAEoA9wBGA1cASgJRAEoCSgAeAmMASg
JnAB4BSgBKAcsAGAE5ABQCUABCAgIAAQMRAAEB4v/6AgEAAQHOABQBLwBAAPoAYAEvACECRABNA0Y
AJAItAHgBKgAcAkQAUAEsAHQAygAgAi0AOQD3ADYA9wAWAaEANgGhABYCbAAlAYMAeAGDADkA6/9q
AhsAFAIKABUB/QAVAAAAAwAAAAMAAAAcAAEAAAAAAKQAAwABAAAAHAAEAIgAAAAeABAAAwAOAH4Aq
QCrALEAtAC3ALsgGSAdICYgOiBEISL7Av//AAAAIACpAKsAsAC0ALcAuyAYIBwgJiA5IEQhIvsB//
//4/+5/7j/tP+y/7D/reBR4E/gR+A14CzfTwVxAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAEGAAABAAAAAAAAAAECAAAAAgAAAAAAAAAAAAAAAAAAAAEAAAMEBQYHCAkKCwwNDg8QERIT
FBUWFxgZGhscHR4fICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj9AQUJDREVGR0hJSktMT
U5PUFFSU1RVVldYWVpbXF1eX2BhAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGQAAA
AAAAAAYnFmAAAAAABlAAAAAAAAAAAAAAAAAAAAAAAAAAAAY2htAAAAAAAAAABrbGlqAAAAAHAAbm9
ycwBnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAmACYAJgAmAD4AUgCCAMoBCgFO
AVwBcgGIAaYBvAHKAdYB6AH2AgwCIAJKAogCpgLWAw4DIgNkA5wDugPUA+gD/AQQBEYEogS8BPoFJ
gVSBWoFgAWwBcoF1gX6BhQGJAZMBmgGiga0BuIHGgdUB2YHkAeiB8AH3AfyCAoIHAgqCDoITghcCG
oIogjSCPoJKglYCXwJwgnqCgIKKApACl4Klgq8CtwLDAs8C1YLjAuyC9oL7gwMDCYMSAxgDKAMrAz
qDQoNTA1mDYQNoA2uDcAN2g3oDfYODA4iDkoOXA5sDnoOnA7EDvwAAAAFAAAAAAH0ArwAAwAGAAkA
DAAPAAAxESERAxMhExcRASELARETAfT6qv6syKr+jgFUqsiqArz9RAGLAP/+1P8B/v3VAP8BLP4CA
P8AAgA5//IAuQKyAAMACwAANyMDMwIyFhQGIiY0oE4MZk84JCQ4JLQB/v3AJDgkJDgAAgBWAeUBPA
LfAAMABwAAEyMnMxcjJzOmRgpagkYKWgHl+vr6AAAAAAIARgAAAf4CsgAbAB8AAAEHMxUjByM3Iwc
jNyM1MzcjNTM3MwczNzMHMxUrAQczAZgdZXEvOi9bLzovWmYdZXEvOi9bLzovWp9bHlsBn4w429vb
2ziMONvb29s4jAAAAAMANf+mAg4DDAAfACYALAAAJRQGBxUjNS4BJzMeARcRLgE0Njc1MxUeARcjJ
icVHgEBFBYXNQ4BExU+ATU0Ag5xWDpgcgRcBz41Xl9oVTpVYwpcC1ttXP6cLTQuM5szOrVRZwlOTQ
ZqVzZECAEAGlukZAlOTQdrUG8O7iNlAQgxNhDlCDj+8/YGOjReAAAAAAUAKf/yArsCvAAHAAsAFQA
dACcAABIyFhQGIiY0EyMBMwQiBhUUFjI2NTQSMhYUBiImNDYiBhUUFjI2NTR5iFBQiFCVVwHAV/5c
OiMjOiPmiFBQiFCxOiMjOiMCvFaSVlaS/ZoCsjIzMC80NC8w/uNWklZWkhozMC80NC8wAAAAAgBA/
/ICbgLAACIALgAAARUjEQYjIiY1NDY3LgE1NDYzMhcVJiMiBhUUFhcWOwE1MxUFFBYzMjc1IyIHDg
ECbmBcYYOOVkg7R4hsQjY4Q0RNRD4SLDxW/pJUXzksPCkUUk0BgUb+zBVUZ0BkDw5RO1huCkULQzp
COAMBcHDHRz0J/AIHRQAAAAEAKwHlAIUC3wADAAATIycze0YKWgHl+gAAAAABAGT/sAEXAwwACQAA
EzMGEBcjLgE0Nt06dXU6OUBAAwzG/jDGVePs4wAAAAEAHv+wANEDDAAJAAATMx4BFAYHIzYQHjo5Q
EA5OnUDDFXj7ONVxgHQAAAAAQBVAFIB2wHbAA4AAAE3FwcXBycHJzcnNxcnMwEtmxOfcTJjYzJxnx
ObCj4BKD07KYolmZkliik7PbMAAQBQAFUB9AIlAAsAAAEjFSM1IzUzNTMVMwH0tTq1tTq1AR/Kyjj
OzgAAAAAB/+H/iACMAGQABAAANwcjNzOMWlFOXVrS3AAAAQAgAP8A8gE3AAMAABMjNTPy0tIA/zgA
AQAl//IApQByAAcAADYyFhQGIiY0STgkJDgkciQ4JCQ4AAAAAf/7/+IBNALQAAMAABcjEzM5Pvs+H
gLuAAAAAAIAKf/yAhsCwAADAAcAABIgECA2IBAgKQHy/g5gATL+zgLA/TJEAkYAAAAAAQCCAAABlg
KyAAgAAAERIxEHNTc2MwGWVr6SIygCsv1OAldxW1sWAAEAPAAAAg4CwAAZAAA3IRUhNRM+ATU0JiM
iDwEjNz4BMzIWFRQGB7kBUv4x+kI2QTt+EAFWAQp8aGVtSl5GRjEA/0RVLzlLmAoKa3FsUkNxXQAA
AAEALf/yAhYCwAAqAAABHgEVFAYjIi8BMxceATMyNjU0KwE1MzI2NTQmIyIGDwEjNz4BMzIWFRQGA
YxBSZJo2RUBVgEHV0JBUaQREUBUQzc5TQcBVgEKfGhfcEMBbxJbQl1x0AoKRkZHPn9GSD80QUVCCg
pfbGBPOlgAAAACACEAAAIkArIACgAPAAAlIxUjNSE1ATMRMyMRBg8BAiRXVv6qAVZWV60dHLCurq4
rAdn+QgFLMibzAAABADn/8gIZArIAHQAAATIWFRQGIyIvATMXFjMyNjU0JiMiByMTIRUhBzc2ATNv
d5Fl1RQBVgIad0VSTkVhL1IwAYj+vh8rMAHHgGdtgcUKCoFXTU5bYgGRRvAuHQAAAAACACv/8gITA
sAAFwAjAAABMhYVFAYjIhE0NjMyFh8BIycmIyIDNzYTMjY1NCYjIgYVFBYBLmp7imr0l3RZdAgBXA
IYZ5wKJzU6QVNJSz5SUAHSgWltiQFGxcNlVQoKdv7sPiz+ZF1LTmJbU0lhAAAAAQAyAAACGgKyAAY
AAAEVASMBITUCGv6oXAFL/oECsij9dgJsRgAAAAMALP/xAhgCwAAWACAALAAAAR4BFRQGIyImNTQ2
Ny4BNTQ2MhYVFAYmIgYVFBYyNjU0AzI2NTQmIyIGFRQWAZQ5S5BmbIpPOjA7ecp5P2F8Q0J8RIVJS
0pLTEtOAW0TXTxpZ2ZqPF0SE1A3VWVlVTdQ/UU0N0RENzT9/ko+Ok1NOj1LAAIAMf/yAhkCwAAXAC
MAAAEyERQGIyImLwEzFxYzMhMHBiMiJjU0NhMyNjU0JiMiBhUUFgEl9Jd0WXQIAVwCGGecCic1SWp
7imo+UlBAQVNJAsD+usXDZVUKCnYBFD4sgWltif5kW1NJYV1LTmIAAAACACX/8gClAiAABwAPAAAS
MhYUBiImNBIyFhQGIiY0STgkJDgkJDgkJDgkAiAkOCQkOP52JDgkJDgAAAAC/+H/iAClAiAABwAMA
AASMhYUBiImNBMHIzczSTgkJDgkaFpSTl4CICQ4JCQ4/mba5gAAAQBnAB4B+AH0AAYAAAENARUlNS
UB+P6qAVb+bwGRAbCmpkbJRMkAAAIAUAC7AfQBuwADAAcAAAEhNSERITUhAfT+XAGk/lwBpAGDOP8
AOAABAEQAHgHVAfQABgAAARUFNS0BNQHV/m8BVv6qAStEyUSmpkYAAAAAAgAj//IB1ALAABgAIAAA
ATIWFRQHDgEHIz4BNz4BNTQmIyIGByM+ARIyFhQGIiY0AQRibmktIAJWBSEqNig+NTlHBFoDezQ4J
CQ4JALAZ1BjaS03JS1DMD5LLDQ/SUVgcv2yJDgkJDgAAAAAAgA2/5gDFgKYADYAQgAAAQMGFRQzMj
Y1NCYjIg4CFRQWMzI2NxcGIyImNTQ+AjMyFhUUBiMiJwcGIyImNTQ2MzIfATcHNzYmIyIGFRQzMjY
Cej8EJjJJlnBAfGQ+oHtAhjUYg5OPx0h2k06Os3xRWQsVLjY5VHtdPBwJETcJDyUoOkZEJz8B0f74
EQ8kZl6EkTFZjVOLlyknMVm1pmCiaTq4lX6CSCknTVRmmR8wPdYnQzxuSWVGAAIAHQAAAncCsgAHA
AoAACUjByMTMxMjATMDAcj+UVz4dO5d/sjPZPT0ArL9TgE6ATQAAAADAGQAAAJMArIAEAAbACcAAA
EeARUUBgcGKwERMzIXFhUUJRUzMjc2NTQnJiMTPgE1NCcmKwEVMzIBvkdHZkwiNt7LOSGq/oeFHBt
hahIlSTM+cB8Yj5UWAW8QT0VYYgwFArIEF5Fv1eMED2NfDAL93AU+N24PBP0AAAAAAQAv//ICjwLA
ABsAAAEyFh8BIycmIyIGFRQWMzI/ATMHDgEjIiY1NDYBdX+PCwFWAiKiaHx5ZaIiAlYBCpWBk6a0A
sCAagoKpqN/gaOmCgplhcicn8sAAAIAZAAAAp8CsgAMABkAAAEeARUUBgcGKwERMzITPgE1NCYnJi
sBETMyAY59lJp8IzXN0jUVWmdjWRs5d3I4Aq4QqJWUug8EArL9mQ+PeHGHDgX92gAAAAABAGQAAAI
vArIACwAAJRUhESEVIRUhFSEVAi/+NQHB/pUBTf6zRkYCskbwRvAAAAABAGQAAAIlArIACQAAExUh
FSERIxEhFboBQ/69VgHBAmzwRv7KArJGAAAAAAEAL//yAo8CwAAfAAABMxEjNQcGIyImNTQ2MzIWH
wEjJyYjIgYVFBYzMjY1IwGP90wfPnWTprSSf48LAVYCIqJofHllVG+hAU3+s3hARsicn8uAagoKpq
N/gaN1XAAAAAEAZAAAAowCsgALAAABESMRIREjETMRIRECjFb+hFZWAXwCsv1OAS7+0gKy/sQBPAA
AAAABAGQAAAC6ArIAAwAAMyMRM7pWVgKyAAABADf/8gHoArIAEwAAAREUBw4BIyImLwEzFxYzMjc2
NREB6AIFcGpgbQIBVgIHfXQKAQKy/lYxIltob2EpKYyEFD0BpwAAAAABAGQAAAJ0ArIACwAACQEjA
wcVIxEzEQEzATsBJ3ntQlZWAVVlAWH+nwEnR+ACsv6RAW8AAQBkAAACLwKyAAUAACUVIREzEQIv/j
VWRkYCsv2UAAABAGQAAAMUArIAFAAAAREjETQ3BgcDIwMmJxYVESMRMxsBAxRWAiMxemx8NxsCVo7
MywKy/U4BY7ZLco7+nAFmoFxLtP6dArL9lwJpAAAAAAEAZAAAAoACsgANAAAhIwEWFREjETMBJjUR
MwKAhP67A1aEAUUDVAJeeov+pwKy/aJ5jAFZAAAAAgAv//ICuwLAAAkAEwAAEiAWFRQGICY1NBIyN
jU0JiIGFRTbATSsrP7MrNrYenrYegLAxaKhxsahov47nIeIm5uIhwACAGQAAAJHArIADgAYAAABHg
EVFAYHBisBESMRMzITNjQnJisBETMyAZRUX2VOHzuAVtY7GlxcGDWIiDUCrgtnVlVpCgT+5gKy/rU
V1BUF/vgAAAACAC//zAK9AsAAEgAcAAAlFhcHJiMiBwYjIiY1NDYgFhUUJRQWMjY1NCYiBgI9PUMx
UDcfKh8omqysATSs/dR62Hp62HpICTg7NgkHxqGixcWitbWHnJyHiJubAAIAZAAAAlgCsgAXACMAA
CUWFyMmJyYnJisBESMRMzIXHgEVFAYHFiUzMjc+ATU0JyYrAQIqDCJfGQwNWhAhglbiOx9QXEY1Tv
6bhDATMj1lGSyMtYgtOXR0BwH+1wKyBApbU0BSESRAAgVAOGoQBAABADT/8gIoAsAAJQAAATIWFyM
uASMiBhUUFhceARUUBiMiJiczHgEzMjY1NCYnLgE1NDYBOmd2ClwGS0E6SUNRdW+HZnKKC1wPWkQ9
Uk1cZGuEAsBwXUJHNjQ3OhIbZVZZbm5kREo+NT5DFRdYUFdrAAAAAAEAIgAAAmQCsgAHAAABIxEjE
SM1IQJk9lb2AkICbP2UAmxGAAEAXv/yAmQCsgAXAAABERQHDgEiJicmNREzERQXHgEyNjc2NRECZA
IIgfCBCAJWAgZYmlgGAgKy/k0qFFxzc1wUKgGz/lUrEkRQUEQSKwGrAAAAAAEAIAAAAnoCsgAGAAA
hIwMzGwEzAYJ07l3N1FwCsv2PAnEAAAEAGgAAA7ECsgAMAAABAyMLASMDMxsBMxsBA7HAcZyicrZi
kaB0nJkCsv1OAlP9rQKy/ZsCW/2kAmYAAAEAGQAAAm8CsgALAAAhCwEjEwMzGwEzAxMCCsrEY/bkY
re+Y/D6AST+3AFcAVb+5gEa/q3+oQAAAQATAAACUQKyAAgAAAERIxEDMxsBMwFdVvRjwLphARD+8A
EQAaL+sQFPAAABAC4AAAI5ArIACQAAJRUhNQEhNSEVAQI5/fUBof57Aen+YUZGQgIqRkX92QAAAAA
BAGL/sAEFAwwABwAAARUjETMVIxEBBWlpowMMOP0UOANcAAAB//v/4gE0AtAAAwAABSMDMwE0Pvs+
HgLuAAAAAQAi/7AAxQMMAAcAABcjNTMRIzUzxaNpaaNQOALsOAABAFAA1wH0AmgABgAAJQsBIxMzE
wGwjY1GsESw1wFZ/qcBkf5vAAAAAQAy/6oBwv/iAAMAAAUhNSEBwv5wAZBWOAAAAAEAKQJEALYCsg
ADAAATIycztjhVUAJEbgAAAAACACT/8gHQAiAAHQAlAAAhJwcGIyImNTQ2OwE1NCcmIyIHIz4BMzI
XFh0BFBcnMjY9ASYVFAF6CR0wVUtgkJoiAgdgaQlaBm1Zrg4DCuQ9R+5MOSFQR1tbDiwUUXBUXowf
J8c9SjRORzYSgVwAAAAAAgBK//ICRQLfABEAHgAAATIWFRQGIyImLwEVIxEzETc2EzI2NTQmIyIGH
QEUFgFUcYCVbiNJEyNWVigySElcU01JXmECIJd4i5QTEDRJAt/+3jkq/hRuZV55ZWsdX14AAQAe//
IB9wIgABgAAAEyFhcjJiMiBhUUFjMyNjczDgEjIiY1NDYBF152DFocbEJXU0A1Rw1aE3pbaoKQAiB
oWH5qZm1tPDlaXYuLgZcAAAACAB7/8gIZAt8AEQAeAAABESM1BwYjIiY1NDYzMhYfAREDMjY9ATQm
IyIGFRQWAhlWKDJacYCVbiNJEyOnSV5hQUlcUwLf/SFVOSqXeIuUExA0ARb9VWVrHV9ebmVeeQACA
B7/8gH9AiAAFQAbAAABFAchHgEzMjY3Mw4BIyImNTQ2MzIWJyIGByEmAf0C/oAGUkA1SwlaD4FXbI
WObmt45UBVBwEqDQEYFhNjWD84W16Oh3+akU9aU60AAAEAFQAAARoC8gAWAAATBh0BMxUjESMRIzU
zNTQ3PgEzMhcVJqcDbW1WOTkDB0k8Hx5oAngVITRC/jQBzEIsJRs5PwVHEwAAAAIAHv8uAhkCIAAi
AC8AAAERFAcOASMiLwEzFx4BMzI2NzY9AQcGIyImNTQ2MzIWHwE1AzI2PQE0JiMiBhUUFgIZAQSEd
NwRAVcBBU5DTlUDASgyWnGAlW4jSRMjp0leYUFJXFMCEv5wSh1zeq8KCTI8VU0ZIQk5Kpd4i5QTED
RJ/iJlax1fXm5lXnkAAQBKAAACCgLkABcAAAEWFREjETQnLgEHDgEdASMRMxE3NjMyFgIIAlYCBDs
6RVRWViE5UVViAYUbQP7WASQxGzI7AQJyf+kC5P7TPSxUAAACAD4AAACsAsAABwALAAASMhYUBiIm
NBMjETNeLiAgLiBiVlYCwCAuICAu/WACEgAC//P/LgCnAsAABwAVAAASMhYUBiImNBcRFAcGIyInN
RY3NjURWS4gIC4gYgMLcRwNSgYCAsAgLiAgLo79wCUbZAJGBzMOHgJEAAAAAQBKAAACCALfAAsAAC
EnBxUjETMREzMHEwGTwTJWVvdu9/rgN6kC3/4oAQv6/ugAAQBG//wA3gLfAA8AABMRFBceATcVBiM
iJicmNRGcAQIcIxkkKi4CAQLf/bkhERoSBD4EJC8SNAJKAAAAAQBKAAADEAIgACQAAAEWFREjETQn
JiMiFREjETQnJiMiFREjETMVNzYzMhYXNzYzMhYDCwVWBAxedFYEDF50VlYiJko7ThAvJkpEVAGfI
jn+vAEcQyRZ1v76ARxDJFnW/voCEk08HzYtRB9HAAAAAAEASgAAAgoCIAAWAAABFhURIxE0JyYjIg
YdASMRMxU3NjMyFgIIAlYCCXBEVVZWITlRVWIBhRtA/tYBJDEbbHR/6QISWz0sVAAAAAACAB7/8gI
sAiAABwARAAASIBYUBiAmNBIyNjU0JiIGFRSlAQCHh/8Ah7ieWlqeWgIgn/Cfn/D+s3ZfYHV1YF8A
AgBK/zwCRQIgABEAHgAAATIWFRQGIyImLwERIxEzFTc2EzI2NTQmIyIGHQEUFgFUcYCVbiNJEyNWV
igySElcU01JXmECIJd4i5QTEDT+8wLWVTkq/hRuZV55ZWsdX14AAgAe/zwCGQIgABEAHgAAAREjEQ
cGIyImNTQ2MzIWHwE1AzI2PQE0JiMiBhUUFgIZVigyWnGAlW4jSRMjp0leYUFJXFMCEv0qARk5Kpd
4i5QTEDRJ/iJlax1fXm5lXnkAAQBKAAABPgIeAA0AAAEyFxUmBhURIxEzFTc2ARoWDkdXVlYwIwIe
B0EFVlf+0gISU0cYAAEAGP/yAa0CIAAjAAATMhYXIyYjIgYVFBYXHgEVFAYjIiYnMxYzMjY1NCYnL
gE1NDbkV2MJWhNdKy04PF1XbVhWbgxaE2ktOjlEUllkAiBaS2MrJCUoEBlPQkhOVFZoKCUmLhIWSE
BIUwAAAAEAFP/4ARQCiQAXAAATERQXHgE3FQYjIiYnJjURIzUzNTMVMxWxAQMmMx8qMjMEAUdHVmM
BzP7PGw4mFgY/BSwxDjQBNUJ7e0IAAAABAEL/8gICAhIAFwAAAREjNQcGIyImJyY1ETMRFBceATMy
Nj0BAgJWITlRT2EKBVYEBkA1RFECEv3uWj4qTToiOQE+/tIlJC43c4DpAAAAAAEAAQAAAfwCEgAGA
AABAyMDMxsBAfzJaclfop8CEv3uAhL+LQHTAAABAAEAAAMLAhIADAAAAQMjCwEjAzMbATMbAQMLqW
Z2dmapY3t0a3Z7AhL97gG+/kICEv5AAcD+QwG9AAAB//oAAAHWAhIACwAAARMjJwcjEwMzFzczARq
8ZIuKY763ZoWFYwEO/vLV1QEMAQbNzQAAAQAB/y4B+wISABEAAAEDDgEjIic1FjMyNj8BAzMbAQH7
2iFZQB8NDRIpNhQH02GenQIS/cFVUAJGASozEwIt/i4B0gABABQAAAGxAg4ACQAAJRUhNQEhNSEVA
QGx/mMBNP7iAYL+zkREQgGIREX+ewAAAAABAED/sAEOAwwALAAAASMiBhUUFxYVFAYHHgEVFAcGFR
QWOwEVIyImNTQ3NjU0JzU2NTQnJjU0NjsBAQ4MKiMLDS4pKS4NCyMqDAtERAwLUlILDERECwLUGBk
WTlsgKzUFBTcrIFtOFhkYOC87GFVMIkUIOAhFIkxVGDsvAAAAAAEAYP84AJoDIAADAAAXIxEzmjo6
yAPoAAEAIf+wAO8DDAAsAAATFQYVFBcWFRQGKwE1MzI2NTQnJjU0NjcuATU0NzY1NCYrATUzMhYVF
AcGFRTvUgsMREQLDCojCw0uKSkuDQsjKgwLREQMCwF6OAhFIkxVGDsvOBgZFk5bICs1BQU3KyBbTh
YZGDgvOxhVTCJFAAABAE0A3wH2AWQAEwAAATMUIyImJyYjIhUjNDMyFhcWMzIBvjhuGywtQR0xOG4
bLC1BHTEBZIURGCNMhREYIwAAAwAk/94DIgLoAAcAEQApAAAAIBYQBiAmECQgBhUUFiA2NTQlMhYX
IyYjIgYUFjMyNjczDgEjIiY1NDYBAQFE3d3+vN0CB/7wubkBELn+xVBnD1wSWDo+QTcqOQZcEmZWX
HN2Aujg/rbg4AFKpr+Mjb6+jYxbWEldV5ZZNShLVn5na34AAgB4AFIB9AGeAAUACwAAAQcXIyc3Mw
cXIyc3AUqJiUmJifOJiUmJiQGepqampqampqYAAAIAHAHSAQ4CwAAHAA8AABIyFhQGIiY0NiIGFBY
yNjRgakREakSTNCEhNCECwEJqQkJqCiM4IyM4AAAAAAIAUAAAAfQCCwALAA8AAAEzFSMVIzUjNTM1
MxMhNSEBP7W1OrW1OrX+XAGkAVs4tLQ4sP31OAAAAQB0AkQBAQKyAAMAABMjNzOsOD1QAkRuAAAAA
AEAIADsAKoBdgAHAAASMhYUBiImNEg6KCg6KAF2KDooKDoAAAIAOQBSAbUBngAFAAsAACUHIzcnMw
UHIzcnMwELiUmJiUkBM4lJiYlJ+KampqampqYAAAABADYB5QDhAt8ABAAAEzczByM2Xk1OXQHv8Po
AAQAWAeUAwQLfAAQAABMHIzczwV5NTl0C1fD6AAIANgHlAYsC3wAEAAkAABM3MwcjPwEzByM2Xk1O
XapeTU5dAe/w+grw+gAAAgAWAeUBawLfAAQACQAAEwcjNzMXByM3M8FeTU5dql5NTl0C1fD6CvD6A
AADACX/8gI1AHIABwAPABcAADYyFhQGIiY0NjIWFAYiJjQ2MhYUBiImNEk4JCQ4JOw4JCQ4JOw4JC
Q4JHIkOCQkOCQkOCQkOCQkOCQkOAAAAAEAeABSAUoBngAFAAABBxcjJzcBSomJSYmJAZ6mpqamAAA
AAAEAOQBSAQsBngAFAAAlByM3JzMBC4lJiYlJ+KampgAAAf9qAAABgQKyAAMAACsBATM/VwHAVwKy
AAAAAAIAFAHIAdwClAAHABQAABMVIxUjNSM1BRUjNwcjJxcjNTMXN9pKMkoByDICKzQqATJLKysCl
CmjoykBy46KiY3Lm5sAAQAVAAABvALyABgAAAERIxEjESMRIzUzNTQ3NjMyFxUmBgcGHQEBvFbCVj
k5AxHHHx5iVgcDAg798gHM/jQBzEIOJRuWBUcIJDAVIRYAAAABABX//AHkAvIAJQAAJR4BNxUGIyI
mJyY1ESYjIgcGHQEzFSMRIxEjNTM1NDc2MzIXERQBowIcIxkkKi4CAR4nXgwDbW1WLy8DEbNdOmYa
EQQ/BCQvEjQCFQZWFSEWQv40AcxCDiUblhP9uSEAAAAAAAAWAQ4AAQAAAAAAAAATACgAAQAAAAAAA
QAHAEwAAQAAAAAAAgAHAGQAAQAAAAAAAwAaAKIAAQAAAAAABAAHAM0AAQAAAAAABQA8AU8AAQAAAA
AABgAPAawAAQAAAAAACAALAdQAAQAAAAAACQALAfgAAQAAAAAACwAXAjQAAQAAAAAADAAXAnwAAwA
BBAkAAAAmAAAAAwABBAkAAQAOADwAAwABBAkAAgAOAFQAAwABBAkAAwA0AGwAAwABBAkABAAOAL0A
AwABBAkABQB4ANUAAwABBAkABgAeAYwAAwABBAkACAAWAbwAAwABBAkACQAWAeAAAwABBAkACwAuA
gQAAwABBAkADAAuAkwATgBvACAAUgBpAGcAaAB0AHMAIABSAGUAcwBlAHIAdgBlAGQALgAATm8gUm
lnaHRzIFJlc2VydmVkLgAAQQBpAGwAZQByAG8AbgAAQWlsZXJvbgAAUgBlAGcAdQBsAGEAcgAAUmV
ndWxhcgAAMQAuADEAMAAyADsAVQBLAFcATgA7AEEAaQBsAGUAcgBvAG4ALQBSAGUAZwB1AGwAYQBy
AAAxLjEwMjtVS1dOO0FpbGVyb24tUmVndWxhcgAAQQBpAGwAZQByAG8AbgAAQWlsZXJvbgAAVgBlA
HIAcwBpAG8AbgAgADEALgAxADAAMgA7AFAAUwAgADAAMAAxAC4AMQAwADIAOwBoAG8AdABjAG8Abg
B2ACAAMQAuADAALgA3ADAAOwBtAGEAawBlAG8AdABmAC4AbABpAGIAMgAuADUALgA1ADgAMwAyADk
AAFZlcnNpb24gMS4xMDI7UFMgMDAxLjEwMjtob3Rjb252IDEuMC43MDttYWtlb3RmLmxpYjIuNS41
ODMyOQAAQQBpAGwAZQByAG8AbgAtAFIAZQBnAHUAbABhAHIAAEFpbGVyb24tUmVndWxhcgAAUwBvA
HIAYQAgAFMAYQBnAGEAbgBvAABTb3JhIFNhZ2FubwAAUwBvAHIAYQAgAFMAYQBnAGEAbgBvAABTb3
JhIFNhZ2FubwAAaAB0AHQAcAA6AC8ALwB3AHcAdwAuAGQAbwB0AGMAbwBsAG8AbgAuAG4AZQB0AAB
odHRwOi8vd3d3LmRvdGNvbG9uLm5ldAAAaAB0AHQAcAA6AC8ALwB3AHcAdwAuAGQAbwB0AGMAbwBs
AG8AbgAuAG4AZQB0AABodHRwOi8vd3d3LmRvdGNvbG9uLm5ldAAAAAACAAAAAAAA/4MAMgAAAAAAA
AAAAAAAAAAAAAAAAAAAAHQAAAABAAIAAwAEAAUABgAHAAgACQAKAAsADAANAA4ADwAQABEAEgATAB
QAFQAWABcAGAAZABoAGwAcAB0AHgAfACAAIQAiACMAJAAlACYAJwAoACkAKgArACwALQAuAC8AMAA
xADIAMwA0ADUANgA3ADgAOQA6ADsAPAA9AD4APwBAAEEAQgBDAEQARQBGAEcASABJAEoASwBMAE0A
TgBPAFAAUQBSAFMAVABVAFYAVwBYAFkAWgBbAFwAXQBeAF8AYABhAIsAqQCDAJMAjQDDAKoAtgC3A
LQAtQCrAL4AvwC8AIwAwADBAAAAAAAB//8AAgABAAAADAAAABwAAAACAAIAAwBxAAEAcgBzAAIABA
AAAAIAAAABAAAACgBMAGYAAkRGTFQADmxhdG4AGgAEAAAAAP//AAEAAAAWAANDQVQgAB5NT0wgABZ
ST00gABYAAP//AAEAAAAA//8AAgAAAAEAAmxpZ2EADmxvY2wAFAAAAAEAAQAAAAEAAAACAAYAEAAG
AAAAAgASADQABAAAAAEATAADAAAAAgAQABYAAQAcAAAAAQABAE8AAQABAGcAAQABAE8AAwAAAAIAE
AAWAAEAHAAAAAEAAQAvAAEAAQBnAAEAAQAvAAEAGgABAAgAAgAGAAwAcwACAE8AcgACAEwAAQABAE
kAAAABAAAACgBGAGAAAkRGTFQADmxhdG4AHAAEAAAAAP//AAIAAAABABYAA0NBVCAAFk1PTCAAFlJ
PTSAAFgAA//8AAgAAAAEAAmNwc3AADmtlcm4AFAAAAAEAAAAAAAEAAQACAAYADgABAAAAAQASAAIA
AAACAB4ANgABAAoABQAFAAoAAgABACQAPQAAAAEAEgAEAAAAAQAMAAEAOP/nAAEAAQAkAAIGigAEA
AAFJAXKABoAGQAA//gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAD/sv+4/+z/7v/MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAD/xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/9T/6AAAAAD/8QAA
ABD/vQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/7gAAAAAAAAAAAAAAAAAA//MAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABIAAAAAAAAAAP/5AAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/gAAD/4AAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//L/9AAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAA/+gAAAAAAAkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/zAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/mAAAAAAAAAAAAAAAAAAD
/4gAA//AAAAAA//YAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/+AAAAAAAAP/OAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/zv/qAAAAAP/0AAAACAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP/ZAAD/egAA/1kAAAAA/5D/rgAAAAAAAAAAAA
AAAAAAAAAAAAAAAAD/9AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAD/8AAA/7b/8P+wAAD/8P/E/98AAAAA/8P/+P/0//oAAAAAAAAAAAAA//gA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/+AAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/w//C/9MAAP/SAAD/9wAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAD/yAAA/+kAAAAA//QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/9wAAAAD//QAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAP/2AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAP/cAAAAAAAAAAAAAAAA/7YAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAP/8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD/6AAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAkAFAAEAAAAAQACwAAABcA
BgAAAAAAAAAIAA4AAAAAAAsAEgAAAAAAAAATABkAAwANAAAAAQAJAAAAAAAAAAAAAAAAAAAAGAAAA
AAABwAAAAAAAAAAAAAAFQAFAAAAAAAYABgAAAAUAAAACgAAAAwAAgAPABEAFgAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFAAEAEQBdAAYAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAcAAAAAAAAABwAAAAAACAAAAAAAAAAAAAcAAAAHAAAAEwAJ
ABUADgAPAAAACwAQAAAAAAAAAAAAAAAAAAUAGAACAAIAAgAAAAIAGAAXAAAAGAAAABYAFgACABYAA
gAWAAAAEQADAAoAFAAMAA0ABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASAAAAEgAGAAEAHgAkAC
YAJwApACoALQAuAC8AMgAzADcAOAA5ADoAPAA9AEUASABOAE8AUgBTAFUAVwBZAFoAWwBcAF0AcwA
AAAAAAQAAAADa3tfFAAAAANAan9kAAAAA4QodoQ==
""")),
            10 if size is None else size,
            layout_engine=Layout.BASIC,
        )
    return load_default_imagefont()
<|endoftext|>#
# The Python Imaging Library.
#
# SPIDER image file handling
#
# History:
# 2004-08-02    Created BB
# 2006-03-02    added save method
# 2006-03-13    added support for stack images
#
# Copyright (c) 2004 by Health Research Inc. (HRI) RENSSELAER, NY 12144.
# Copyright (c) 2004 by William Baxter.
# Copyright (c) 2004 by Secret Labs AB.
# Copyright (c) 2004 by Fredrik Lundh.
#

##
# Image plugin for the Spider image format. This format is used
# by the SPIDER software, in processing image data from electron
# microscopy and tomography.
##

#
# SpiderImagePlugin.py
#
# The Spider image format is used by SPIDER software, in processing
# image data from electron microscopy and tomography.
#
# Spider home page:
# https://spider.wadsworth.org/spider_doc/spider/docs/spider.html
#
# Details about the Spider image format:
# https://spider.wadsworth.org/spider_doc/spider/docs/image_doc.html
#
from __future__ import annotations

import os
import struct
import sys
from typing import IO, Any

from . import Image, ImageFile
from ._util import DeferredError

TYPE_CHECKING = False


def isInt(f: Any) -> int:
    try:
        i = int(f)
        if f - i == 0:
            return 1
        else:
            return 0
    except (ValueError, OverflowError):
        return 0


iforms = [1, 3, -11, -12, -21, -22]


# There is no magic number to identify Spider files, so just check a
# series of header locations to see if they have reasonable values.
# Returns no. of bytes in the header, if it is a valid Spider header,
# otherwise returns 0


def isSpiderHeader(t: tuple[float, ...]) -> int:
    h = (99,) + t  # add 1 value so can use spider header index start=1
    # header values 1,2,5,12,13,22,23 should be integers
    for i in [1, 2, 5, 12, 13, 22, 23]:
        if not isInt(h[i]):
            return 0
    # check iform
    iform = int(h[5])
    if iform not in iforms:
        return 0
    # check other header values
    labrec = int(h[13])  # no. records in file header
    labbyt = int(h[22])  # total no. of bytes in header
    lenbyt = int(h[23])  # record length in bytes
    if labbyt != (labrec * lenbyt):
        return 0
    # looks like a valid header
    return labbyt


def isSpiderImage(filename: str) -> int:
    with open(filename, "rb") as fp:
        f = fp.read(92)  # read 23 * 4 bytes
    t = struct.unpack(">23f", f)  # try big-endian first
    hdrlen = isSpiderHeader(t)
    if hdrlen == 0:
        t = struct.unpack("<23f", f)  # little-endian
        hdrlen = isSpiderHeader(t)
    return hdrlen


class SpiderImageFile(ImageFile.ImageFile):
    format = "SPIDER"
    format_description = "Spider 2D image"
    _close_exclusive_fp_after_loading = False

    def _open(self) -> None:
        # check header
        n = 27 * 4  # read 27 float values
        assert self.fp is not None
        f = self.fp.read(n)

        try:
            self.bigendian = 1
            t = struct.unpack(">27f", f)  # try big-endian first
            hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                self.bigendian = 0
                t = struct.unpack("<27f", f)  # little-endian
                hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                msg = "not a valid Spider file"
                raise SyntaxError(msg)
        except struct.error as e:
            msg = "not a valid Spider file"
            raise SyntaxError(msg) from e

        h = (99,) + t  # add 1 value : spider header index starts at 1
        iform = int(h[5])
        if iform != 1:
            msg = "not a Spider 2D image"
            raise SyntaxError(msg)

        self._size = int(h[12]), int(h[2])  # size in pixels (width, height)
        self.istack = int(h[24])
        self.imgnumber = int(h[27])

        if self.istack == 0 and self.imgnumber == 0:
            # stk=0, img=0: a regular 2D image
            offset = hdrlen
            self._nimages = 1
        elif self.istack > 0 and self.imgnumber == 0:
            # stk>0, img=0: Opening the stack for the first time
            self.imgbytes = int(h[12]) * int(h[2]) * 4
            self.hdrlen = hdrlen
            self._nimages = int(h[26])
            # Point to the first image in the stack
            offset = hdrlen * 2
            self.imgnumber = 1
        elif self.istack == 0 and self.imgnumber > 0:
            # stk=0, img>0: an image within the stack
            offset = hdrlen + self.stkoffset
            self.istack = 2  # So Image knows it's still a stack
        else:
            msg = "inconsistent stack header values"
            raise SyntaxError(msg)

        if self.bigendian:
            self.rawmode = "F;32BF"
        else:
            self.rawmode = "F;32F"
        self._mode = "F"

        self.tile = [ImageFile._Tile("raw", (0, 0) + self.size, offset, self.rawmode)]
        self._fp = self.fp  # FIXME: hack

    @property
    def n_frames(self) -> int:
        return self._nimages

    @property
    def is_animated(self) -> bool:
        return self._nimages > 1

    # 1st image index is zero (although SPIDER imgnumber starts at 1)
    def tell(self) -> int:
        if self.imgnumber < 1:
            return 0
        else:
            return self.imgnumber - 1

    def seek(self, frame: int) -> None:
        if self.istack == 0:
            msg = "attempt to seek in a non-stack file"
            raise EOFError(msg)
        if not self._seek_check(frame):
            return
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex
        self.stkoffset = self.hdrlen + frame * (self.hdrlen + self.imgbytes)
        self.fp = self._fp
        self.fp.seek(self.stkoffset)
        self._open()

    # returns a byte image after rescaling to 0..255
    def convert2byte(self, depth: int = 255) -> Image.Image:
        extrema = self.getextrema()
        assert isinstance(extrema[0], float)
        minimum, maximum = extrema
        m: float = 1
        if maximum != minimum:
            m = depth / (maximum - minimum)
        b = -m * minimum
        return self.point(lambda i: i * m + b).convert("L")

    if TYPE_CHECKING:
        from . import ImageTk

    # returns a ImageTk.PhotoImage object, after rescaling to 0..255
    def tkPhotoImage(self) -> ImageTk.PhotoImage:
        from . import ImageTk

        return ImageTk.PhotoImage(self.convert2byte(), palette=256)


# --------------------------------------------------------------------
# Image series


# given a list of filenames, return a list of images
def loadImageSeries(filelist: list[str] | None = None) -> list[Image.Image] | None:
    """create a list of :py:class:`~PIL.Image.Image` objects for use in a montage"""
    if filelist is None or len(filelist) < 1:
        return None

    byte_imgs = []
    for img in filelist:
        if not os.path.exists(img):
            print(f"unable to find {img}")
            continue
        try:
            with Image.open(img) as im:
                assert isinstance(im, SpiderImageFile)
                byte_im = im.convert2byte()
        except Exception:
            if not isSpiderImage(img):
                print(f"{img} is not a Spider image file")
            continue
        byte_im.info["filename"] = img
        byte_imgs.append(byte_im)
    return byte_imgs


# --------------------------------------------------------------------
# For saving images in Spider format


def makeSpiderHeader(im: Image.Image) -> list[bytes]:
    nsam, nrow = im.size
    lenbyt = max(1, nsam) * 4  # There are labrec records in the header
    labrec = int(1024 / lenbyt)
    if 1024 % lenbyt != 0:
        labrec += 1
    labbyt = labrec * lenbyt
    nvalues = int(labbyt / 4)
    if nvalues < 23:
        return []

    hdr = [0.0] * nvalues

    # NB these are Fortran indices
    hdr[1] = 1.0  # nslice (=1 for an image)
    hdr[2] = float(nrow)  # number of rows per slice
    hdr[3] = float(nrow)  # number of records in the image
    hdr[5] = 1.0  # iform for 2D image
    hdr[12] = float(nsam)  # number of pixels per line
    hdr[13] = float(labrec)  # number of records in file header
    hdr[22] = float(labbyt)  # total number of bytes in header
    hdr[23] = float(lenbyt)  # record length in bytes

    # adjust for Fortran indexing
    hdr = hdr[1:]
    hdr.append(0.0)
    # pack binary data into a string
    return [struct.pack("f", v) for v in hdr]


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    if im.mode != "F":
        im = im.convert("F")

    hdr = makeSpiderHeader(im)
    if len(hdr) < 256:
        msg = "Error creating Spider header"
        raise OSError(msg)

    # write the SPIDER header
    fp.writelines(hdr)

    rawmode = "F;32NF"  # 32-bit native floating point
    ImageFile._save(im, fp, [ImageFile._Tile("raw", (0, 0) + im.size, 0, rawmode)])


def _save_spider(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    # get the filename extension and register it with Image
    if filename_ext := os.path.splitext(filename)[1]:
        ext = filename_ext.decode() if isinstance(filename_ext, bytes) else filename_ext
        Image.register_extension(SpiderImageFile.format, ext)
    _save(im, fp, filename)


# --------------------------------------------------------------------


Image.register_open(SpiderImageFile.format, SpiderImageFile)
Image.register_save(SpiderImageFile.format, _save_spider)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Syntax: python3 SpiderImagePlugin.py [infile] [outfile]")
        sys.exit()

    filename = sys.argv[1]
    if not isSpiderImage(filename):
        print("input image must be in Spider format")
        sys.exit()

    with Image.open(filename) as im:
        print(f"image: {im}")
        print(f"format: {im.format}")
        print(f"size: {im.size}")
        print(f"mode: {im.mode}")
        print("max, min: ", end=" ")
        print(im.getextrema())

        if len(sys.argv) > 2:
            outfile = sys.argv[2]

            # perform some image operation
            transposed_im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            print(
                f"saving a flipped version of {os.path.basename(filename)} "
                f"as {outfile} "
            )
            transposed_im.save(outfile, SpiderImageFile.format)
<|endoftext|>#
# The Python Imaging Library
# $Id$
#
# map CSS3-style colour description strings to RGB
#
# History:
# 2002-10-24 fl   Added support for CSS-style color strings
# 2002-12-15 fl   Added RGBA support
# 2004-03-27 fl   Fixed remaining int() problems for Python 1.5.2
# 2004-07-19 fl   Fixed gray/grey spelling issues
# 2009-03-05 fl   Fixed rounding error in grayscale calculation
#
# Copyright (c) 2002-2004 by Secret Labs AB
# Copyright (c) 2002-2004 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import re
from functools import lru_cache

from . import Image


@lru_cache
def getrgb(color: str) -> tuple[int, int, int] | tuple[int, int, int, int]:
    """
     Convert a color string to an RGB or RGBA tuple. If the string cannot be
     parsed, this function raises a :py:exc:`ValueError` exception.

    .. versionadded:: 1.1.4

    :param color: A color string
    :return: ``(red, green, blue[, alpha])``
    """
    if len(color) > 100:
        msg = "color specifier is too long"
        raise ValueError(msg)
    color = color.lower()

    rgb = colormap.get(color, None)
    if rgb:
        if isinstance(rgb, tuple):
            return rgb
        rgb_tuple = getrgb(rgb)
        assert len(rgb_tuple) == 3
        colormap[color] = rgb_tuple
        return rgb_tuple

    # check for known string formats
    if re.match("#[a-f0-9]{3}$", color):
        return int(color[1] * 2, 16), int(color[2] * 2, 16), int(color[3] * 2, 16)

    if re.match("#[a-f0-9]{4}$", color):
        return (
            int(color[1] * 2, 16),
            int(color[2] * 2, 16),
            int(color[3] * 2, 16),
            int(color[4] * 2, 16),
        )

    if re.match("#[a-f0-9]{6}$", color):
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    if re.match("#[a-f0-9]{8}$", color):
        return (
            int(color[1:3], 16),
            int(color[3:5], 16),
            int(color[5:7], 16),
            int(color[7:9], 16),
        )

    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", color)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.match(r"rgb\(\s*(\d+)%\s*,\s*(\d+)%\s*,\s*(\d+)%\s*\)$", color)
    if m:
        return (
            int((int(m.group(1)) * 255) / 100.0 + 0.5),
            int((int(m.group(2)) * 255) / 100.0 + 0.5),
            int((int(m.group(3)) * 255) / 100.0 + 0.5),
        )

    m = re.match(
        r"hsl\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)%\s*,\s*(\d+\.?\d*)%\s*\)$", color
    )
    if m:
        from colorsys import hls_to_rgb

        rgb_floats = hls_to_rgb(
            float(m.group(1)) / 360.0,
            float(m.group(3)) / 100.0,
            float(m.group(2)) / 100.0,
        )
        return (
            int(rgb_floats[0] * 255 + 0.5),
            int(rgb_floats[1] * 255 + 0.5),
            int(rgb_floats[2] * 255 + 0.5),
        )

    m = re.match(
        r"hs[bv]\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)%\s*,\s*(\d+\.?\d*)%\s*\)$", color
    )
    if m:
        from colorsys import hsv_to_rgb

        rgb_floats = hsv_to_rgb(
            float(m.group(1)) / 360.0,
            float(m.group(2)) / 100.0,
            float(m.group(3)) / 100.0,
        )
        return (
            int(rgb_floats[0] * 255 + 0.5),
            int(rgb_floats[1] * 255 + 0.5),
            int(rgb_floats[2] * 255 + 0.5),
        )

    m = re.match(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", color)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    msg = f"unknown color specifier: {repr(color)}"
    raise ValueError(msg)


@lru_cache
def getcolor(color: str, mode: str) -> int | tuple[int, ...]:
    """
    Same as :py:func:`~PIL.ImageColor.getrgb` for most modes. However, if
    ``mode`` is HSV, converts the RGB value to a HSV value, or if ``mode`` is
    not color or a palette image, converts the RGB value to a grayscale value.
    If the string cannot be parsed, this function raises a :py:exc:`ValueError`
    exception.

    .. versionadded:: 1.1.4

    :param color: A color string
    :param mode: Convert result to this mode
    :return: ``graylevel, (graylevel, alpha) or (red, green, blue[, alpha])``
    """
    # same as getrgb, but converts the result to the given mode
    rgb, alpha = getrgb(color), 255
    if len(rgb) == 4:
        alpha = rgb[3]
        rgb = rgb[:3]

    if mode == "HSV":
        from colorsys import rgb_to_hsv

        r, g, b = rgb
        h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)
        return int(h * 255), int(s * 255), int(v * 255)
    elif Image.getmodebase(mode) == "L":
        r, g, b = rgb
        # ITU-R Recommendation 601-2 for nonlinear RGB
        # scaled to 24 bits to match the convert's implementation.
        graylevel = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16
        if mode[-1] == "A":
            return graylevel, alpha
        return graylevel
    elif mode[-1] == "A":
        return rgb + (alpha,)
    return rgb


colormap: dict[str, str | tuple[int, int, int]] = {
    # X11 colour table from https://drafts.csswg.org/css-color-4/, with
    # gray/grey spelling issues fixed.  This is a superset of HTML 4.0
    # colour names used in CSS 1.
    "aliceblue": "#f0f8ff",
    "antiquewhite": "#faebd7",
    "aqua": "#00ffff",
    "aquamarine": "#7fffd4",
    "azure": "#f0ffff",
    "beige": "#f5f5dc",
    "bisque": "#ffe4c4",
    "black": "#000000",
    "blanchedalmond": "#ffebcd",
    "blue": "#0000ff",
    "blueviolet": "#8a2be2",
    "brown": "#a52a2a",
    "burlywood": "#deb887",
    "cadetblue": "#5f9ea0",
    "chartreuse": "#7fff00",
    "chocolate": "#d2691e",
    "coral": "#ff7f50",
    "cornflowerblue": "#6495ed",
    "cornsilk": "#fff8dc",
    "crimson": "#dc143c",
    "cyan": "#00ffff",
    "darkblue": "#00008b",
    "darkcyan": "#008b8b",
    "darkgoldenrod": "#b8860b",
    "darkgray": "#a9a9a9",
    "darkgrey": "#a9a9a9",
    "darkgreen": "#006400",
    "darkkhaki": "#bdb76b",
    "darkmagenta": "#8b008b",
    "darkolivegreen": "#556b2f",
    "darkorange": "#ff8c00",
    "darkorchid": "#9932cc",
    "darkred": "#8b0000",
    "darksalmon": "#e9967a",
    "darkseagreen": "#8fbc8f",
    "darkslateblue": "#483d8b",
    "darkslategray": "#2f4f4f",
    "darkslategrey": "#2f4f4f",
    "darkturquoise": "#00ced1",
    "darkviolet": "#9400d3",
    "deeppink": "#ff1493",
    "deepskyblue": "#00bfff",
    "dimgray": "#696969",
    "dimgrey": "#696969",
    "dodgerblue": "#1e90ff",
    "firebrick": "#b22222",
    "floralwhite": "#fffaf0",
    "forestgreen": "#228b22",
    "fuchsia": "#ff00ff",
    "gainsboro": "#dcdcdc",
    "ghostwhite": "#f8f8ff",
    "gold": "#ffd700",
    "goldenrod": "#daa520",
    "gray": "#808080",
    "grey": "#808080",
    "green": "#008000",
    "greenyellow": "#adff2f",
    "honeydew": "#f0fff0",
    "hotpink": "#ff69b4",
    "indianred": "#cd5c5c",
    "indigo": "#4b0082",
    "ivory": "#fffff0",
    "khaki": "#f0e68c",
    "lavender": "#e6e6fa",
    "lavenderblush": "#fff0f5",
    "lawngreen": "#7cfc00",
    "lemonchiffon": "#fffacd",
    "lightblue": "#add8e6",
    "lightcoral": "#f08080",
    "lightcyan": "#e0ffff",
    "lightgoldenrodyellow": "#fafad2",
    "lightgreen": "#90ee90",
    "lightgray": "#d3d3d3",
    "lightgrey": "#d3d3d3",
    "lightpink": "#ffb6c1",
    "lightsalmon": "#ffa07a",
    "lightseagreen": "#20b2aa",
    "lightskyblue": "#87cefa",
    "lightslategray": "#778899",
    "lightslategrey": "#778899",
    "lightsteelblue": "#b0c4de",
    "lightyellow": "#ffffe0",
    "lime": "#00ff00",
    "limegreen": "#32cd32",
    "linen": "#faf0e6",
    "magenta": "#ff00ff",
    "maroon": "#800000",
    "mediumaquamarine": "#66cdaa",
    "mediumblue": "#0000cd",
    "mediumorchid": "#ba55d3",
    "mediumpurple": "#9370db",
    "mediumseagreen": "#3cb371",
    "mediumslateblue": "#7b68ee",
    "mediumspringgreen": "#00fa9a",
    "mediumturquoise": "#48d1cc",
    "mediumvioletred": "#c71585",
    "midnightblue": "#191970",
    "mintcream": "#f5fffa",
    "mistyrose": "#ffe4e1",
    "moccasin": "#ffe4b5",
    "navajowhite": "#ffdead",
    "navy": "#000080",
    "oldlace": "#fdf5e6",
    "olive": "#808000",
    "olivedrab": "#6b8e23",
    "orange": "#ffa500",
    "orangered": "#ff4500",
    "orchid": "#da70d6",
    "palegoldenrod": "#eee8aa",
    "palegreen": "#98fb98",
    "paleturquoise": "#afeeee",
    "palevioletred": "#db7093",
    "papayawhip": "#ffefd5",
    "peachpuff": "#ffdab9",
    "peru": "#cd853f",
    "pink": "#ffc0cb",
    "plum": "#dda0dd",
    "powderblue": "#b0e0e6",
    "purple": "#800080",
    "rebeccapurple": "#663399",
    "red": "#ff0000",
    "rosybrown": "#bc8f8f",
    "royalblue": "#4169e1",
    "saddlebrown": "#8b4513",
    "salmon": "#fa8072",
    "sandybrown": "#f4a460",
    "seagreen": "#2e8b57",
    "seashell": "#fff5ee",
    "sienna": "#a0522d",
    "silver": "#c0c0c0",
    "skyblue": "#87ceeb",
    "slateblue": "#6a5acd",
    "slategray": "#708090",
    "slategrey": "#708090",
    "snow": "#fffafa",
    "springgreen": "#00ff7f",
    "steelblue": "#4682b4",
    "tan": "#d2b48c",
    "teal": "#008080",
    "thistle": "#d8bfd8",
    "tomato": "#ff6347",
    "turquoise": "#40e0d0",
    "violet": "#ee82ee",
    "wheat": "#f5deb3",
    "white": "#ffffff",
    "whitesmoke": "#f5f5f5",
    "yellow": "#ffff00",
    "yellowgreen": "#9acd32",
}
<|endoftext|>#
# The Python Imaging Library.
# $Id$
#
# TIFF file handling
#
# TIFF is a flexible, if somewhat aged, image file format originally
# defined by Aldus.  Although TIFF supports a wide variety of pixel
# layouts and compression methods, the name doesn't really stand for
# "thousands of incompatible file formats," it just feels that way.
#
# To read TIFF data from a stream, the stream must be seekable.  For
# progressive decoding, make sure to use TIFF files where the tag
# directory is placed first in the file.
#
# History:
# 1995-09-01 fl   Created
# 1996-05-04 fl   Handle JPEGTABLES tag
# 1996-05-18 fl   Fixed COLORMAP support
# 1997-01-05 fl   Fixed PREDICTOR support
# 1997-08-27 fl   Added support for rational tags (from Perry Stoll)
# 1998-01-10 fl   Fixed seek/tell (from Jan Blom)
# 1998-07-15 fl   Use private names for internal variables
# 1999-06-13 fl   Rewritten for PIL 1.0 (1.0)
# 2000-10-11 fl   Additional fixes for Python 2.0 (1.1)
# 2001-04-17 fl   Fixed rewind support (seek to frame 0) (1.2)
# 2001-05-12 fl   Added write support for more tags (from Greg Couch) (1.3)
# 2001-12-18 fl   Added workaround for broken Matrox library
# 2002-01-18 fl   Don't mess up if photometric tag is missing (D. Alan Stewart)
# 2003-05-19 fl   Check FILLORDER tag
# 2003-09-26 fl   Added RGBa support
# 2004-02-24 fl   Added DPI support; fixed rational write support
# 2005-02-07 fl   Added workaround for broken Corel Draw 10 files
# 2006-01-09 fl   Added support for float/double tags (from Russell Nelson)
#
# Copyright (c) 1997-2006 by Secret Labs AB.  All rights reserved.
# Copyright (c) 1995-1997 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#
from __future__ import annotations

import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import Callable, MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from typing import IO, Any, cast

from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._util import DeferredError, is_path
from .TiffTags import TYPES

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import NoReturn

    from ._typing import Buffer, IntegralLike, StrOrBytesPath

logger = logging.getLogger(__name__)

# Set these to true to force use of libtiff for reading or writing.
READ_LIBTIFF = False
WRITE_LIBTIFF = False
STRIP_SIZE = 65536

II = b"II"  # little-endian (Intel style)
MM = b"MM"  # big-endian (Motorola style)

#
# --------------------------------------------------------------------
# Read TIFF files

# a few tag names, just to make the code below a bit more readable
OSUBFILETYPE = 255
IMAGEWIDTH = 256
IMAGELENGTH = 257
BITSPERSAMPLE = 258
COMPRESSION = 259
PHOTOMETRIC_INTERPRETATION = 262
FILLORDER = 266
IMAGEDESCRIPTION = 270
STRIPOFFSETS = 273
SAMPLESPERPIXEL = 277
ROWSPERSTRIP = 278
STRIPBYTECOUNTS = 279
X_RESOLUTION = 282
Y_RESOLUTION = 283
PLANAR_CONFIGURATION = 284
RESOLUTION_UNIT = 296
TRANSFERFUNCTION = 301
SOFTWARE = 305
DATE_TIME = 306
ARTIST = 315
PREDICTOR = 317
COLORMAP = 320
TILEWIDTH = 322
TILELENGTH = 323
TILEOFFSETS = 324
TILEBYTECOUNTS = 325
SUBIFD = 330
EXTRASAMPLES = 338
SAMPLEFORMAT = 339
JPEGTABLES = 347
YCBCRSUBSAMPLING = 530
REFERENCEBLACKWHITE = 532
COPYRIGHT = 33432
IPTC_NAA_CHUNK = 33723  # newsphoto properties
PHOTOSHOP_CHUNK = 34377  # photoshop properties
ICCPROFILE = 34675
EXIFIFD = 34665
XMP = 700
JPEGQUALITY = 65537  # pseudo-tag by libtiff

# https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/io/TiffDecoder.java
IMAGEJ_META_DATA_BYTE_COUNTS = 50838
IMAGEJ_META_DATA = 50839

COMPRESSION_INFO = {
    # Compression => pil compression name
    1: "raw",
    2: "tiff_ccitt",
    3: "group3",
    4: "group4",
    5: "tiff_lzw",
    6: "tiff_jpeg",  # obsolete
    7: "jpeg",
    8: "tiff_adobe_deflate",
    32771: "tiff_raw_16",  # 16-bit padding
    32773: "packbits",
    32809: "tiff_thunderscan",
    32946: "tiff_deflate",
    34676: "tiff_sgilog",
    34677: "tiff_sgilog24",
    34925: "lzma",
    50000: "zstd",
    50001: "webp",
}

COMPRESSION_INFO_REV = {v: k for k, v in COMPRESSION_INFO.items()}

OPEN_INFO = {
    # (ByteOrder, PhotoInterpretation, SampleFormat, FillOrder, BitsPerSample,
    #  ExtraSamples) => mode, rawmode
    (II, 0, (1,), 1, (1,), ()): ("1", "1;I"),
    (MM, 0, (1,), 1, (1,), ()): ("1", "1;I"),
    (II, 0, (1,), 2, (1,), ()): ("1", "1;IR"),
    (MM, 0, (1,), 2, (1,), ()): ("1", "1;IR"),
    (II, 1, (1,), 1, (1,), ()): ("1", "1"),
    (MM, 1, (1,), 1, (1,), ()): ("1", "1"),
    (II, 1, (1,), 2, (1,), ()): ("1", "1;R"),
    (MM, 1, (1,), 2, (1,), ()): ("1", "1;R"),
    (II, 0, (1,), 1, (2,), ()): ("L", "L;2I"),
    (MM, 0, (1,), 1, (2,), ()): ("L", "L;2I"),
    (II, 0, (1,), 2, (2,), ()): ("L", "L;2IR"),
    (MM, 0, (1,), 2, (2,), ()): ("L", "L;2IR"),
    (II, 1, (1,), 1, (2,), ()): ("L", "L;2"),
    (MM, 1, (1,), 1, (2,), ()): ("L", "L;2"),
    (II, 1, (1,), 2, (2,), ()): ("L", "L;2R"),
    (MM, 1, (1,), 2, (2,), ()): ("L", "L;2R"),
    (II, 0, (1,), 1, (4,), ()): ("L", "L;4I"),
    (MM, 0, (1,), 1, (4,), ()): ("L", "L;4I"),
    (II, 0, (1,), 2, (4,), ()): ("L", "L;4IR"),
    (MM, 0, (1,), 2, (4,), ()): ("L", "L;4IR"),
    (II, 1, (1,), 1, (4,), ()): ("L", "L;4"),
    (MM, 1, (1,), 1, (4,), ()): ("L", "L;4"),
    (II, 1, (1,), 2, (4,), ()): ("L", "L;4R"),
    (MM, 1, (1,), 2, (4,), ()): ("L", "L;4R"),
    (II, 0, (1,), 1, (8,), ()): ("L", "L;I"),
    (MM, 0, (1,), 1, (8,), ()): ("L", "L;I"),
    (II, 0, (1,), 2, (8,), ()): ("L", "L;IR"),
    (MM, 0, (1,), 2, (8,), ()): ("L", "L;IR"),
    (II, 1, (1,), 1, (8,), ()): ("L", "L"),
    (MM, 1, (1,), 1, (8,), ()): ("L", "L"),
    (II, 1, (2,), 1, (8,), ()): ("L", "L"),
    (MM, 1, (2,), 1, (8,), ()): ("L", "L"),
    (II, 1, (1,), 2, (8,), ()): ("L", "L;R"),
    (MM, 1, (1,), 2, (8,), ()): ("L", "L;R"),
    (II, 1, (1,), 1, (12,), ()): ("I;16", "I;12"),
    (II, 0, (1,), 1, (16,), ()): ("I;16", "I;16"),
    (II, 1, (1,), 1, (16,), ()): ("I;16", "I;16"),
    (MM, 1, (1,), 1, (16,), ()): ("I;16B", "I;16B"),
    (II, 1, (1,), 2, (16,), ()): ("I;16", "I;16R"),
    (II, 1, (2,), 1, (16,), ()): ("I", "I;16S"),
    (MM, 1, (2,), 1, (16,), ()): ("I", "I;16BS"),
    (II, 0, (3,), 1, (32,), ()): ("F", "F;32F"),
    (MM, 0, (3,), 1, (32,), ()): ("F", "F;32BF"),
    (II, 1, (1,), 1, (32,), ()): ("I", "I;32N"),
    (II, 1, (2,), 1, (32,), ()): ("I", "I;32S"),
    (MM, 1, (2,), 1, (32,), ()): ("I", "I;32BS"),
    (II, 1, (3,), 1, (32,), ()): ("F", "F;32F"),
    (MM, 1, (3,), 1, (32,), ()): ("F", "F;32BF"),
    (II, 1, (1,), 1, (8, 8), (2,)): ("LA", "LA"),
    (MM, 1, (1,), 1, (8, 8), (2,)): ("LA", "LA"),
    (II, 2, (1,), 1, (8, 8, 8), ()): ("RGB", "RGB"),
    (MM, 2, (1,), 1, (8, 8, 8), ()): ("RGB", "RGB"),
    (II, 2, (1,), 2, (8, 8, 8), ()): ("RGB", "RGB;R"),
    (MM, 2, (1,), 2, (8, 8, 8), ()): ("RGB", "RGB;R"),
    (II, 2, (1,), 1, (8, 8, 8, 8), ()): ("RGBA", "RGBA"),  # missing ExtraSamples
    (MM, 2, (1,), 1, (8, 8, 8, 8), ()): ("RGBA", "RGBA"),  # missing ExtraSamples
    (II, 2, (1,), 1, (8, 8, 8, 8), (0,)): ("RGB", "RGBX"),
    (MM, 2, (1,), 1, (8, 8, 8, 8), (0,)): ("RGB", "RGBX"),
    (II, 2, (1,), 1, (8, 8, 8, 8, 8), (0, 0)): ("RGB", "RGBXX"),
    (MM, 2, (1,), 1, (8, 8, 8, 8, 8), (0, 0)): ("RGB", "RGBXX"),
    (II, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0, 0)): ("RGB", "RGBXXX"),
    (MM, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0, 0)): ("RGB", "RGBXXX"),
    (II, 2, (1,), 1, (8, 8, 8, 8), (1,)): ("RGBA", "RGBa"),
    (MM, 2, (1,), 1, (8, 8, 8, 8), (1,)): ("RGBA", "RGBa"),
    (II, 2, (1,), 1, (8, 8, 8, 8, 8), (1, 0)): ("RGBA", "RGBaX"),
    (MM, 2, (1,), 1, (8, 8, 8, 8, 8), (1, 0)): ("RGBA", "RGBaX"),
    (II, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (1, 0, 0)): ("RGBA", "RGBaXX"),
    (MM, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (1, 0, 0)): ("RGBA", "RGBaXX"),
    (II, 2, (1,), 1, (8, 8, 8, 8), (2,)): ("RGBA", "RGBA"),
    (MM, 2, (1,), 1, (8, 8, 8, 8), (2,)): ("RGBA", "RGBA"),
    (II, 2, (1,), 1, (8, 8, 8, 8, 8), (2, 0)): ("RGBA", "RGBAX"),
    (MM, 2, (1,), 1, (8, 8, 8, 8, 8), (2, 0)): ("RGBA", "RGBAX"),
    (II, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (2, 0, 0)): ("RGBA", "RGBAXX"),
    (MM, 2, (1,), 1, (8, 8, 8, 8, 8, 8), (2, 0, 0)): ("RGBA", "RGBAXX"),
    (II, 2, (1,), 1, (8, 8, 8, 8), (999,)): ("RGBA", "RGBA"),  # Corel Draw 10
    (MM, 2, (1,), 1, (8, 8, 8, 8), (999,)): ("RGBA", "RGBA"),  # Corel Draw 10
    (II, 2, (1,), 1, (16, 16, 16), ()): ("RGB", "RGB;16L"),
    (MM, 2, (1,), 1, (16, 16, 16), ()): ("RGB", "RGB;16B"),
    (II, 2, (1,), 1, (16, 16, 16, 16), ()): ("RGBA", "RGBA;16L"),
    (MM, 2, (1,), 1, (16, 16, 16, 16), ()): ("RGBA", "RGBA;16B"),
    (II, 2, (1,), 1, (16, 16, 16, 16), (0,)): ("RGB", "RGBX;16L"),
    (MM, 2, (1,), 1, (16, 16, 16, 16), (0,)): ("RGB", "RGBX;16B"),
    (II, 2, (1,), 1, (16, 16, 16, 16), (1,)): ("RGBA", "RGBa;16L"),
    (MM, 2, (1,), 1, (16, 16, 16, 16), (1,)): ("RGBA", "RGBa;16B"),
    (II, 2, (1,), 1, (16, 16, 16, 16), (2,)): ("RGBA", "RGBA;16L"),
    (MM, 2, (1,), 1, (16, 16, 16, 16), (2,)): ("RGBA", "RGBA;16B"),
    (II, 3, (1,), 1, (1,), ()): ("P", "P;1"),
    (MM, 3, (1,), 1, (1,), ()): ("P", "P;1"),
    (II, 3, (1,), 2, (1,), ()): ("P", "P;1R"),
    (MM, 3, (1,), 2, (1,), ()): ("P", "P;1R"),
    (II, 3, (1,), 1, (2,), ()): ("P", "P;2"),
    (MM, 3, (1,), 1, (2,), ()): ("P", "P;2"),
    (II, 3, (1,), 2, (2,), ()): ("P", "P;2R"),
    (MM, 3, (1,), 2, (2,), ()): ("P", "P;2R"),
    (II, 3, (1,), 1, (4,), ()): ("P", "P;4"),
    (MM, 3, (1,), 1, (4,), ()): ("P", "P;4"),
    (II, 3, (1,), 2, (4,), ()): ("P", "P;4R"),
    (MM, 3, (1,), 2, (4,), ()): ("P", "P;4R"),
    (II, 3, (1,), 1, (8,), ()): ("P", "P"),
    (MM, 3, (1,), 1, (8,), ()): ("P", "P"),
    (II, 3, (1,), 1, (8, 8), (0,)): ("P", "PX"),
    (MM, 3, (1,), 1, (8, 8), (0,)): ("P", "PX"),
    (II, 3, (1,), 1, (8, 8), (2,)): ("PA", "PA"),
    (MM, 3, (1,), 1, (8, 8), (2,)): ("PA", "PA"),
    (II, 3, (1,), 2, (8,), ()): ("P", "P;R"),
    (MM, 3, (1,), 2, (8,), ()): ("P", "P;R"),
    (II, 5, (1,), 1, (8, 8, 8, 8), ()): ("CMYK", "CMYK"),
    (MM, 5, (1,), 1, (8, 8, 8, 8), ()): ("CMYK", "CMYK"),
    (II, 5, (1,), 1, (8, 8, 8, 8, 8), (0,)): ("CMYK", "CMYKX"),
    (MM, 5, (1,), 1, (8, 8, 8, 8, 8), (0,)): ("CMYK", "CMYKX"),
    (II, 5, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0)): ("CMYK", "CMYKXX"),
    (MM, 5, (1,), 1, (8, 8, 8, 8, 8, 8), (0, 0)): ("CMYK", "CMYKXX"),
    (II, 5, (1,), 1, (16, 16, 16, 16), ()): ("CMYK", "CMYK;16L"),
    (MM, 5, (1,), 1, (16, 16, 16, 16), ()): ("CMYK", "CMYK;16B"),
    (II, 6, (1,), 1, (8,), ()): ("L", "L"),
    (MM, 6, (1,), 1, (8,), ()): ("L", "L"),
    # JPEG compressed images handled by LibTiff and auto-converted to RGBX
    # Minimal Baseline TIFF requires YCbCr images to have 3 SamplesPerPixel
    (II, 6, (1,), 1, (8, 8, 8), ()): ("RGB", "RGBX"),
    (MM, 6, (1,), 1, (8, 8, 8), ()): ("RGB", "RGBX"),
    (II, 8, (1,), 1, (8, 8, 8), ()): ("LAB", "LAB"),
    (MM, 8, (1,), 1, (8, 8, 8), ()): ("LAB", "LAB"),
}

MAX_SAMPLESPERPIXEL = max(len(key_tp[4]) for key_tp in OPEN_INFO)

PREFIXES = [
    b"MM\x00\x2a",  # Valid TIFF header with big-endian byte order
    b"II\x2a\x00",  # Valid TIFF header with little-endian byte order
    b"MM\x2a\x00",  # Invalid TIFF header, assume big-endian
    b"II\x00\x2a",  # Invalid TIFF header, assume little-endian
    b"MM\x00\x2b",  # BigTIFF with big-endian byte order
    b"II\x2b\x00",  # BigTIFF with little-endian byte order
]


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(tuple(PREFIXES))


def _limit_rational(
    val: float | Fraction | IFDRational, max_val: int
) -> tuple[IntegralLike, IntegralLike]:
    inv = abs(val) > 1
    n_d = IFDRational(1 / val if inv else val).limit_rational(max_val)
    return n_d[::-1] if inv else n_d


def _limit_signed_rational(
    val: IFDRational, max_val: int, min_val: int
) -> tuple[IntegralLike, IntegralLike]:
    frac = Fraction(val)
    n_d: tuple[IntegralLike, IntegralLike] = frac.numerator, frac.denominator

    if min(float(i) for i in n_d) < min_val:
        n_d = _limit_rational(val, abs(min_val))

    n_d_float = tuple(float(i) for i in n_d)
    if max(n_d_float) > max_val:
        n_d = _limit_rational(n_d_float[0] / n_d_float[1], max_val)

    return n_d


##
# Wrapper for TIFF IFDs.

_load_dispatch: dict[int, tuple[int, _LoaderFunc]] = {}
_write_dispatch: dict[int, Callable[..., Any]] = {}


def _delegate(op: str) -> Any:
    def delegate(
        self: IFDRational, *args: tuple[float, ...]
    ) -> bool | float | Fraction:
        return getattr(self._val, op)(*args)

    return delegate


class IFDRational(Rational):
    """Implements a rational class where 0/0 is a legal value to match
    the in the wild use of exif rationals.

    e.g., DigitalZoomRatio - 0.00/0.00  indicates that no digital zoom was used
    """

    """ If the denominator is 0, store this as a float('nan'), otherwise store
    as a fractions.Fraction(). Delegate as appropriate

    """

    __slots__ = ("_numerator", "_denominator", "_val")

    def __init__(
        self, value: float | Fraction | IFDRational, denominator: int = 1
    ) -> None:
        """
        :param value: either an integer numerator, a
        float/rational/other number, or an IFDRational
        :param denominator: Optional integer denominator
        """
        self._val: Fraction | float
        if isinstance(value, IFDRational):
            self._numerator = value.numerator
            self._denominator = value.denominator
            self._val = value._val
            return

        if isinstance(value, Fraction):
            self._numerator = value.numerator
            self._denominator = value.denominator
        else:
            if TYPE_CHECKING:
                self._numerator = cast(IntegralLike, value)
            else:
                self._numerator = value
            self._denominator = denominator

        if denominator == 0:
            self._val = float("nan")
        elif denominator == 1:
            self._val = Fraction(value)
        elif int(value) == value:
            self._val = Fraction(int(value), denominator)
        else:
            self._val = Fraction(value / denominator)

    @property
    def numerator(self) -> IntegralLike:
        return self._numerator

    @property
    def denominator(self) -> int:
        return self._denominator

    def limit_rational(self, max_denominator: int) -> tuple[IntegralLike, int]:
        """

        :param max_denominator: Integer, the maximum denominator value
        :returns: Tuple of (numerator, denominator)
        """

        if self.denominator == 0:
            return self.numerator, self.denominator

        assert isinstance(self._val, Fraction)
        f = self._val.limit_denominator(max_denominator)
        return f.numerator, f.denominator

    def __repr__(self) -> str:
        return str(float(self._val))

    def __hash__(self) -> int:  # type: ignore[override]
        return self._val.__hash__()

    def __eq__(self, other: object) -> bool:
        val = self._val
        if isinstance(other, IFDRational):
            other = other._val
        if isinstance(other, float):
            val = float(val)
        return val == other

    def __getstate__(self) -> list[float | Fraction | IntegralLike]:
        return [self._val, self._numerator, self._denominator]

    def __setstate__(self, state: list[float | Fraction | IntegralLike]) -> None:
        IFDRational.__init__(self, 0)
        _val, _numerator, _denominator = state
        assert isinstance(_val, (float, Fraction))
        self._val = _val
        if TYPE_CHECKING:
            self._numerator = cast(IntegralLike, _numerator)
        else:
            self._numerator = _numerator
        assert isinstance(_denominator, int)
        self._denominator = _denominator

    """ a = ['add','radd', 'sub', 'rsub', 'mul', 'rmul',
             'truediv', 'rtruediv', 'floordiv', 'rfloordiv',
             'mod','rmod', 'pow','rpow', 'pos', 'neg',
             'abs', 'trunc', 'lt', 'gt', 'le', 'ge', 'bool',
             'ceil', 'floor', 'round']
        print("\n".join("__%s__ = _delegate('__%s__')" % (s,s) for s in a))
        """

    __add__ = _delegate("__add__")
    __radd__ = _delegate("__radd__")
    __sub__ = _delegate("__sub__")
    __rsub__ = _delegate("__rsub__")
    __mul__ = _delegate("__mul__")
    __rmul__ = _delegate("__rmul__")
    __truediv__ = _delegate("__truediv__")
    __rtruediv__ = _delegate("__rtruediv__")
    __floordiv__ = _delegate("__floordiv__")
    __rfloordiv__ = _delegate("__rfloordiv__")
    __mod__ = _delegate("__mod__")
    __rmod__ = _delegate("__rmod__")
    __pow__ = _delegate("__pow__")
    __rpow__ = _delegate("__rpow__")
    __pos__ = _delegate("__pos__")
    __neg__ = _delegate("__neg__")
    __abs__ = _delegate("__abs__")
    __trunc__ = _delegate("__trunc__")
    __lt__ = _delegate("__lt__")
    __gt__ = _delegate("__gt__")
    __le__ = _delegate("__le__")
    __ge__ = _delegate("__ge__")
    __bool__ = _delegate("__bool__")
    __ceil__ = _delegate("__ceil__")
    __floor__ = _delegate("__floor__")
    __round__ = _delegate("__round__")
    __float__ = _delegate("__float__")
    # Python >= 3.11
    if hasattr(Fraction, "__int__"):
        __int__ = _delegate("__int__")


_LoaderFunc = Callable[["ImageFileDirectory_v2", bytes, bool], Any]


def _register_loader(idx: int, size: int) -> Callable[[_LoaderFunc], _LoaderFunc]:
    def decorator(func: _LoaderFunc) -> _LoaderFunc:
        from .TiffTags import TYPES

        if func.__name__.startswith("load_"):
            TYPES[idx] = func.__name__[5:].replace("_", " ")
        _load_dispatch[idx] = size, func  # noqa: F821
        return func

    return decorator


def _register_writer(idx: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _write_dispatch[idx] = func  # noqa: F821
        return func

    return decorator


def _register_basic(idx_fmt_name: tuple[int, str, str]) -> None:
    from .TiffTags import TYPES

    idx, fmt, name = idx_fmt_name
    TYPES[idx] = name
    size = struct.calcsize(f"={fmt}")

    def basic_handler(
        self: ImageFileDirectory_v2, data: bytes, legacy_api: bool = True
    ) -> tuple[Any, ...]:
        return self._unpack(f"{len(data) // size}{fmt}", data)

    _load_dispatch[idx] = size, basic_handler  # noqa: F821
    _write_dispatch[idx] = lambda self, *values: (  # noqa: F821
        b"".join(self._pack(fmt, value) for value in values)
    )


if TYPE_CHECKING:
    _IFDv2Base = MutableMapping[int, Any]
else:
    _IFDv2Base = MutableMapping


class ImageFileDirectory_v2(_IFDv2Base):
    """This class represents a TIFF tag directory.  To speed things up, we
    don't decode tags unless they're asked for.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v2()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        'Some Data'

    Individual values are returned as the strings or numbers, sequences are
    returned as tuples of the values.

    The tiff metadata type of each item is stored in a dictionary of
    tag types in
    :attr:`~PIL.TiffImagePlugin.ImageFileDirectory_v2.tagtype`. The types
    are read from a tiff file, guessed from the type added, or added
    manually.

    Data Structures:

        * ``self.tagtype = {}``

          * Key: numerical TIFF tag number
          * Value: integer corresponding to the data type from
            :py:data:`.TiffTags.TYPES`

          .. versionadded:: 3.0.0

    'Internal' data structures:

        * ``self._tags_v2 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data, as tuple for multiple values

        * ``self._tagdata = {}``

          * Key: numerical TIFF tag number
          * Value: undecoded byte string from file

        * ``self._tags_v1 = {}``

          * Key: numerical TIFF tag number
          * Value: decoded data in the v1 format

    Tags will be found in the private attributes ``self._tagdata``, and in
    ``self._tags_v2`` once decoded.

    ``self.legacy_api`` is a value for internal use, and shouldn't be changed
    from outside code. In cooperation with
    :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`, if ``legacy_api``
    is true, then decoded tags will be populated into both ``_tags_v1`` and
    ``_tags_v2``. ``_tags_v2`` will be used if this IFD is used in the TIFF
    save routine. Tags should be read from ``_tags_v1`` if
    ``legacy_api == true``.

    """

    _load_dispatch: dict[int, tuple[int, _LoaderFunc]] = {}
    _write_dispatch: dict[int, Callable[..., Any]] = {}

    def __init__(
        self,
        ifh: bytes = b"II\x2a\x00\x00\x00\x00\x00",
        prefix: bytes | None = None,
        group: int | None = None,
    ) -> None:
        """Initialize an ImageFileDirectory.

        To construct an ImageFileDirectory from a real file, pass the 8-byte
        magic header to the constructor.  To only set the endianness, pass it
        as the 'prefix' keyword argument.

        :param ifh: One of the accepted magic headers (cf. PREFIXES); also sets
              endianness.
        :param prefix: Override the endianness of the file.
        """
        if not _accept(ifh):
            msg = f"not a TIFF file (header {repr(ifh)} not valid)"
            raise SyntaxError(msg)
        self._prefix = prefix if prefix is not None else ifh[:2]
        if self._prefix == MM:
            self._endian = ">"
        elif self._prefix == II:
            self._endian = "<"
        else:
            msg = "not a TIFF IFD"
            raise SyntaxError(msg)
        self._bigtiff = ifh[2] == 43
        self.group = group
        self.tagtype: dict[int, int] = {}
        """ Dictionary of tag types """
        self.reset()
        self.next = (
            self._unpack("Q", ifh[8:])[0]
            if self._bigtiff
            else self._unpack("L", ifh[4:])[0]
        )
        self._legacy_api = False

    prefix = property(lambda self: self._prefix)
    offset = property(lambda self: self._offset)

    @property
    def legacy_api(self) -> bool:
        return self._legacy_api

    @legacy_api.setter
    def legacy_api(self, value: bool) -> NoReturn:
        msg = "Not allowing setting of legacy api"
        raise Exception(msg)

    def reset(self) -> None:
        self._tags_v1: dict[int, Any] = {}  # will remain empty if legacy_api is false
        self._tags_v2: dict[int, Any] = {}  # main tag storage
        self._tagdata: dict[int, bytes] = {}
        self.tagtype = {}  # added 2008-06-05 by Florian Hoech
        self._next = None
        self._offset: int | None = None

    def __str__(self) -> str:
        return str(dict(self))

    def named(self) -> dict[str, Any]:
        """
        :returns: dict of name|key: value

        Returns the complete tag dictionary, with named tags where possible.
        """
        return {
            TiffTags.lookup(code, self.group).name: value
            for code, value in self.items()
        }

    def __len__(self) -> int:
        return len(set(self._tagdata) | set(self._tags_v2))

    def __getitem__(self, tag: int) -> Any:
        if tag not in self._tags_v2:  # unpack on the fly
            data = self._tagdata[tag]
            typ = self.tagtype[tag]
            size, handler = self._load_dispatch[typ]
            self[tag] = handler(self, data, self.legacy_api)  # check type
        val = self._tags_v2[tag]
        if self.legacy_api and not isinstance(val, (tuple, bytes)):
            val = (val,)
        return val

    def __contains__(self, tag: object) -> bool:
        return tag in self._tags_v2 or tag in self._tagdata

    def __setitem__(self, tag: int, value: Any) -> None:
        self._setitem(tag, value, self.legacy_api)

    def _setitem(self, tag: int, value: Any, legacy_api: bool) -> None:
        basetypes = (Number, bytes, str)

        info = TiffTags.lookup(tag, self.group)
        values = [value] if isinstance(value, basetypes) else value

        if tag not in self.tagtype:
            if info.type:
                self.tagtype[tag] = info.type
            else:
                self.tagtype[tag] = TiffTags.UNDEFINED
                if all(isinstance(v, IFDRational) for v in values):
                    for v in values:
                        assert isinstance(v, IFDRational)
                        if v < 0:
                            self.tagtype[tag] = TiffTags.SIGNED_RATIONAL
                            break
                    else:
                        self.tagtype[tag] = TiffTags.RATIONAL
                elif all(isinstance(v, int) for v in values):
                    short = True
                    signed_short = True
                    long = True
                    for v in values:
                        assert isinstance(v, int)
                        if short and not (0 <= v < 2**16):
                            short = False
                        if signed_short and not (-(2**15) < v < 2**15):
                            signed_short = False
                        if long and v < 0:
                            long = False
                    if short:
                        self.tagtype[tag] = TiffTags.SHORT
                    elif signed_short:
                        self.tagtype[tag] = TiffTags.SIGNED_SHORT
                    elif long:
                        self.tagtype[tag] = TiffTags.LONG
                    else:
                        self.tagtype[tag] = TiffTags.SIGNED_LONG
                elif all(isinstance(v, float) for v in values):
                    self.tagtype[tag] = TiffTags.DOUBLE
                elif all(isinstance(v, str) for v in values):
                    self.tagtype[tag] = TiffTags.ASCII
                elif all(isinstance(v, bytes) for v in values):
                    self.tagtype[tag] = TiffTags.BYTE

        if self.tagtype[tag] == TiffTags.UNDEFINED:
            values = [
                v.encode("ascii", "replace") if isinstance(v, str) else v
                for v in values
            ]
        elif self.tagtype[tag] == TiffTags.RATIONAL:
            values = [float(v) if isinstance(v, int) else v for v in values]

        is_ifd = self.tagtype[tag] == TiffTags.LONG and isinstance(values, dict)
        if not is_ifd:
            values = tuple(
                info.cvt_enum(value) if isinstance(value, str) else value
                for value in values
            )

        dest = self._tags_v1 if legacy_api else self._tags_v2

        # Three branches:
        # Spec'd length == 1, Actual length 1, store as element
        # Spec'd length == 1, Actual > 1, Warn and truncate. Formerly barfed.
        # No Spec, Actual length 1, Formerly (<4.2) returned a 1 element tuple.
        # Don't mess with the legacy api, since it's frozen.
        if not is_ifd and (
            (info.length == 1)
            or self.tagtype[tag] == TiffTags.BYTE
            or (info.length is None and len(values) == 1 and not legacy_api)
        ):
            # Don't mess with the legacy api, since it's frozen.
            if legacy_api and self.tagtype[tag] in [
                TiffTags.RATIONAL,
                TiffTags.SIGNED_RATIONAL,
            ]:  # rationals
                values = (values,)
            try:
                (dest[tag],) = values
            except ValueError:
                # We've got a builtin tag with 1 expected entry
                warnings.warn(
                    f"Metadata Warning, tag {tag} had too many entries: "
                    f"{len(values)}, expected 1"
                )
                dest[tag] = values[0]

        else:
            # Spec'd length > 1 or undefined
            # Unspec'd, and length > 1
            dest[tag] = values

    def __delitem__(self, tag: int) -> None:
        self._tags_v2.pop(tag, None)
        self._tags_v1.pop(tag, None)
        self._tagdata.pop(tag, None)

    def __iter__(self) -> Iterator[int]:
        return iter(set(self._tagdata) | set(self._tags_v2))

    def _unpack(self, fmt: str, data: bytes) -> tuple[Any, ...]:
        return struct.unpack(self._endian + fmt, data)

    def _pack(self, fmt: str, *values: Any) -> bytes:
        return struct.pack(self._endian + fmt, *values)

    list(
        map(
            _register_basic,
            [
                (TiffTags.SHORT, "H", "short"),
                (TiffTags.LONG, "L", "long"),
                (TiffTags.SIGNED_BYTE, "b", "signed byte"),
                (TiffTags.SIGNED_SHORT, "h", "signed short"),
                (TiffTags.SIGNED_LONG, "l", "signed long"),
                (TiffTags.FLOAT, "f", "float"),
                (TiffTags.DOUBLE, "d", "double"),
                (TiffTags.IFD, "L", "long"),
                (TiffTags.LONG8, "Q", "long8"),
            ],
        )
    )

    @_register_loader(1, 1)  # Basic type, except for the legacy API.
    def load_byte(self, data: bytes, legacy_api: bool = True) -> bytes:
        return data

    @_register_writer(1)  # Basic type, except for the legacy API.
    def write_byte(self, data: bytes | int | IFDRational) -> bytes:
        if isinstance(data, IFDRational):
            data = int(data)
        if isinstance(data, int):
            data = bytes((data,))
        return data

    @_register_loader(2, 1)
    def load_string(self, data: bytes, legacy_api: bool = True) -> str:
        if data.endswith(b"\0"):
            data = data[:-1]
        return data.decode("latin-1", "replace")

    @_register_writer(2)
    def write_string(self, value: str | bytes | int) -> bytes:
        # remerge of https://github.com/python-pillow/Pillow/pull/1416
        if isinstance(value, int):
            value = str(value)
        if not isinstance(value, bytes):
            value = value.encode("ascii", "replace")
        return value + b"\0"

    @_register_loader(5, 8)
    def load_rational(
        self, data: bytes, legacy_api: bool = True
    ) -> tuple[tuple[int, int] | IFDRational, ...]:
        vals = self._unpack(f"{len(data) // 4}L", data)

        def combine(a: int, b: int) -> tuple[int, int] | IFDRational:
            return (a, b) if legacy_api else IFDRational(a, b)

        return tuple(combine(num, denom) for num, denom in zip(vals[::2], vals[1::2]))

    @_register_writer(5)
    def write_rational(self, *values: IFDRational) -> bytes:
        return b"".join(
            self._pack("2L", *_limit_rational(frac, 2**32 - 1)) for frac in values
        )

    @_register_loader(7, 1)
    def load_undefined(self, data: bytes, legacy_api: bool = True) -> bytes:
        return data

    @_register_writer(7)
    def write_undefined(self, value: bytes | int | IFDRational) -> bytes:
        if isinstance(value, IFDRational):
            value = int(value)
        if isinstance(value, int):
            value = str(value).encode("ascii", "replace")
        return value

    @_register_loader(10, 8)
    def load_signed_rational(
        self, data: bytes, legacy_api: bool = True
    ) -> tuple[tuple[int, int] | IFDRational, ...]:
        vals = self._unpack(f"{len(data) // 4}l", data)

        def combine(a: int, b: int) -> tuple[int, int] | IFDRational:
            return (a, b) if legacy_api else IFDRational(a, b)

        return tuple(combine(num, denom) for num, denom in zip(vals[::2], vals[1::2]))

    @_register_writer(10)
    def write_signed_rational(self, *values: IFDRational) -> bytes:
        return b"".join(
            self._pack("2l", *_limit_signed_rational(frac, 2**31 - 1, -(2**31)))
            for frac in values
        )

    def _ensure_read(self, fp: IO[bytes], size: int) -> bytes:
        ret = fp.read(size)
        if len(ret) != size:
            msg = (
                "Corrupt EXIF data.  "
                f"Expecting to read {size} bytes but only got {len(ret)}. "
            )
            raise OSError(msg)
        return ret

    def load(self, fp: IO[bytes]) -> None:
        self.reset()
        self._offset = fp.tell()

        try:
            tag_count = (
                self._unpack("Q", self._ensure_read(fp, 8))
                if self._bigtiff
                else self._unpack("H", self._ensure_read(fp, 2))
            )[0]
            for i in range(tag_count):
                tag, typ, count, data = (
                    self._unpack("HHQ8s", self._ensure_read(fp, 20))
                    if self._bigtiff
                    else self._unpack("HHL4s", self._ensure_read(fp, 12))
                )

                tagname = TiffTags.lookup(tag, self.group).name
                typname = TYPES.get(typ, "unknown")
                msg = f"tag: {tagname} ({tag}) - type: {typname} ({typ})"

                try:
                    unit_size, handler = self._load_dispatch[typ]
                except KeyError:
                    logger.debug("%s - unsupported type %s", msg, typ)
                    continue  # ignore unsupported type
                size = count * unit_size
                if size > (8 if self._bigtiff else 4):
                    here = fp.tell()
                    (offset,) = self._unpack("Q" if self._bigtiff else "L", data)
                    msg += f" Tag Location: {here} - Data Location: {offset}"
                    fp.seek(offset)
                    data = ImageFile._safe_read(fp, size)
                    fp.seek(here)
                else:
                    data = data[:size]

                if len(data) != size:
                    warnings.warn(
                        "Possibly corrupt EXIF data.  "
                        f"Expecting to read {size} bytes but only got {len(data)}."
                        f" Skipping tag {tag}"
                    )
                    logger.debug(msg)
                    continue

                if not data:
                    logger.debug(msg)
                    continue

                self._tagdata[tag] = data
                self.tagtype[tag] = typ

                msg += " - value: "
                msg += f"<table: {size} bytes>" if size > 32 else repr(data)

                logger.debug(msg)

            (self.next,) = (
                self._unpack("Q", self._ensure_read(fp, 8))
                if self._bigtiff
                else self._unpack("L", self._ensure_read(fp, 4))
            )
        except OSError as msg:
            warnings.warn(str(msg))
            return

    def _get_ifh(self) -> bytes:
        ifh = self._prefix + self._pack("H", 43 if self._bigtiff else 42)
        if self._bigtiff:
            ifh += self._pack("HH", 8, 0)
        ifh += self._pack("Q", 16) if self._bigtiff else self._pack("L", 8)

        return ifh

    def tobytes(self, offset: int = 0) -> bytes:
        # FIXME What about tagdata?
        result = self._pack("Q" if self._bigtiff else "H", len(self._tags_v2))

        entries: list[tuple[int, int, int, bytes, bytes]] = []

        fmt = "Q" if self._bigtiff else "L"
        fmt_size = 8 if self._bigtiff else 4
        offset += (
            len(result) + len(self._tags_v2) * (20 if self._bigtiff else 12) + fmt_size
        )
        stripoffsets = None

        # pass 1: convert tags to binary format
        # always write tags in ascending order
        for tag, value in sorted(self._tags_v2.items()):
            if tag == STRIPOFFSETS:
                stripoffsets = len(entries)
            typ = self.tagtype[tag]
            logger.debug("Tag %s, Type: %s, Value: %s", tag, typ, repr(value))
            is_ifd = typ == TiffTags.LONG and isinstance(value, dict)
            if is_ifd:
                ifd = ImageFileDirectory_v2(self._get_ifh(), group=tag)
                values = self._tags_v2[tag]
                for ifd_tag, ifd_value in values.items():
                    ifd[ifd_tag] = ifd_value
                data = ifd.tobytes(offset)
            else:
                values = value if isinstance(value, tuple) else (value,)
                data = self._write_dispatch[typ](self, *values)

            tagname = TiffTags.lookup(tag, self.group).name
            typname = "ifd" if is_ifd else TYPES.get(typ, "unknown")
            msg = f"save: {tagname} ({tag}) - type: {typname} ({typ}) - value: "
            msg += f"<table: {len(data)} bytes>" if len(data) >= 16 else str(values)
            logger.debug(msg)

            # count is sum of lengths for string and arbitrary data
            if is_ifd:
                count = 1
            elif typ in [TiffTags.BYTE, TiffTags.ASCII, TiffTags.UNDEFINED]:
                count = len(data)
            else:
                count = len(values)
            # figure out if data fits into the entry
            if len(data) <= fmt_size:
                entries.append((tag, typ, count, data.ljust(fmt_size, b"\0"), b""))
            else:
                entries.append((tag, typ, count, self._pack(fmt, offset), data))
                offset += (len(data) + 1) // 2 * 2  # pad to word

        # update strip offset data to point beyond auxiliary data
        if stripoffsets is not None:
            tag, typ, count, value, data = entries[stripoffsets]
            if data:
                size, handler = self._load_dispatch[typ]
                values = [val + offset for val in handler(self, data, self.legacy_api)]
                data = self._write_dispatch[typ](self, *values)
            else:
                value = self._pack(fmt, self._unpack(fmt, value)[0] + offset)
            entries[stripoffsets] = tag, typ, count, value, data

        # pass 2: write entries to file
        for tag, typ, count, value, data in entries:
            logger.debug("%s %s %s %s %s", tag, typ, count, repr(value), repr(data))
            result += self._pack(
                "HHQ8s" if self._bigtiff else "HHL4s", tag, typ, count, value
            )

        # -- overwrite here for multi-page --
        result += self._pack(fmt, 0)  # end of entries

        # pass 3: write auxiliary data to file
        for tag, typ, count, value, data in entries:
            result += data
            if len(data) & 1:
                result += b"\0"

        return result

    def save(self, fp: IO[bytes]) -> int:
        if fp.tell() == 0:  # skip TIFF header on subsequent pages
            fp.write(self._get_ifh())

        offset = fp.tell()
        result = self.tobytes(offset)
        fp.write(result)
        return offset + len(result)


ImageFileDirectory_v2._load_dispatch = _load_dispatch
ImageFileDirectory_v2._write_dispatch = _write_dispatch
for idx, name in TYPES.items():
    name = name.replace(" ", "_")
    setattr(ImageFileDirectory_v2, f"load_{name}", _load_dispatch[idx][1])
    setattr(ImageFileDirectory_v2, f"write_{name}", _write_dispatch[idx])
del _load_dispatch, _write_dispatch, idx, name


# Legacy ImageFileDirectory support.
class ImageFileDirectory_v1(ImageFileDirectory_v2):
    """This class represents the **legacy** interface to a TIFF tag directory.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v1()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        ('Some Data',)

    Also contains a dictionary of tag types as read from the tiff image file,
    :attr:`~PIL.TiffImagePlugin.ImageFileDirectory_v1.tagtype`.

    Values are returned as a tuple.

    ..  deprecated:: 3.0.0
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._legacy_api = True

    tags = property(lambda self: self._tags_v1)
    tagdata = property(lambda self: self._tagdata)

    # defined in ImageFileDirectory_v2
    tagtype: dict[int, int]
    """Dictionary of tag types"""

    @classmethod
    def from_v2(cls, original: ImageFileDirectory_v2) -> ImageFileDirectory_v1:
        """Returns an
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`
        instance with the same data as is contained in the original
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`
        instance.

        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`

        """

        ifd = cls(prefix=original.prefix)
        ifd._tagdata = original._tagdata
        ifd.tagtype = original.tagtype
        ifd.next = original.next  # an indicator for multipage tiffs
        return ifd

    def to_v2(self) -> ImageFileDirectory_v2:
        """Returns an
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`
        instance with the same data as is contained in the original
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`
        instance.

        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`

        """

        ifd = ImageFileDirectory_v2(prefix=self.prefix)
        ifd._tagdata = dict(self._tagdata)
        ifd.tagtype = dict(self.tagtype)
        ifd._tags_v2 = dict(self._tags_v2)
        return ifd

    def __contains__(self, tag: object) -> bool:
        return tag in self._tags_v1 or tag in self._tagdata

    def __len__(self) -> int:
        return len(set(self._tagdata) | set(self._tags_v1))

    def __iter__(self) -> Iterator[int]:
        return iter(set(self._tagdata) | set(self._tags_v1))

    def __setitem__(self, tag: int, value: Any) -> None:
        for legacy_api in (False, True):
            self._setitem(tag, value, legacy_api)

    def __getitem__(self, tag: int) -> Any:
        if tag not in self._tags_v1:  # unpack on the fly
            data = self._tagdata[tag]
            typ = self.tagtype[tag]
            size, handler = self._load_dispatch[typ]
            for legacy in (False, True):
                self._setitem(tag, handler(self, data, legacy), legacy)
        val = self._tags_v1[tag]
        if not isinstance(val, (tuple, bytes)):
            val = (val,)
        return val


# undone -- switch this pointer
ImageFileDirectory = ImageFileDirectory_v1


##
# Image plugin for TIFF files.


class TiffImageFile(ImageFile.ImageFile):
    format = "TIFF"
    format_description = "Adobe TIFF"
    _close_exclusive_fp_after_loading = False

    def __init__(
        self,
        fp: StrOrBytesPath | IO[bytes],
        filename: str | bytes | None = None,
    ) -> None:
        self.tag_v2: ImageFileDirectory_v2
        """ Image file directory (tag dictionary) """

        self.tag: ImageFileDirectory_v1
        """ Legacy tag entries """

        super().__init__(fp, filename)

    def _open(self) -> None:
        """Open the first image in a TIFF file"""

        # Header
        assert self.fp is not None
        ifh = self.fp.read(8)
        if ifh[2] == 43:
            ifh += self.fp.read(8)

        self.tag_v2 = ImageFileDirectory_v2(ifh)

        # setup frame pointers
        self.__first = self.__next = self.tag_v2.next
        self.__frame = -1
        self._fp = self.fp
        self._frame_pos: list[int] = []
        self._n_frames: int | None = None

        logger.debug("*** TiffImageFile._open ***")
        logger.debug("- __first: %s", self.__first)
        logger.debug("- ifh: %s", repr(ifh))  # Use repr to avoid str(bytes)

        # and load the first frame
        self._seek(0)

    @property
    def n_frames(self) -> int:
        current_n_frames = self._n_frames
        if current_n_frames is None:
            current = self.tell()
            self._seek(len(self._frame_pos))
            while self._n_frames is None:
                self._seek(self.tell() + 1)
            self.seek(current)
        assert self._n_frames is not None
        return self._n_frames

    def seek(self, frame: int) -> None:
        """Select a given frame as current image"""
        if not self._seek_check(frame):
            return
        self._seek(frame)
        if self._im is not None and (
            self.im.size != self._tile_size
            or self.im.mode != self.mode
            or self.readonly
        ):
            self._im = None

    def _seek(self, frame: int) -> None:
        if isinstance(self._fp, DeferredError):
            raise self._fp.ex
        self.fp = self._fp

        while len(self._frame_pos) <= frame:
            if not self.__next:
                msg = "no more images in TIFF file"
                raise EOFError(msg)
            logger.debug(
                "Seeking to frame %s, on frame %s, __next %s, location: %s",
                frame,
                self.__frame,
                self.__next,
                self.fp.tell(),
            )
            if self.__next >= 2**63:
                msg = "Unable to seek to frame"
                raise ValueError(msg)
            self.fp.seek(self.__next)
            self._frame_pos.append(self.__next)
            logger.debug("Loading tags, location: %s", self.fp.tell())
            self.tag_v2.load(self.fp)
            if self.tag_v2.next in self._frame_pos:
                # This IFD has already been processed
                # Declare this to be the end of the image
                self.__next = 0
            else:
                self.__next = self.tag_v2.next
            if self.__next == 0:
                self._n_frames = frame + 1
            if len(self._frame_pos) == 1:
                self.is_animated = self.__next != 0
            self.__frame += 1
        self.fp.seek(self._frame_pos[frame])
        self.tag_v2.load(self.fp)
        if XMP in self.tag_v2:
            xmp = self.tag_v2[XMP]
            if isinstance(xmp, tuple) and len(xmp) == 1:
                xmp = xmp[0]
            self.info["xmp"] = xmp
        elif "xmp" in self.info:
            del self.info["xmp"]
        self._reload_exif()
        # fill the legacy tag/ifd entries
        self.tag = self.ifd = ImageFileDirectory_v1.from_v2(self.tag_v2)
        self.__frame = frame
        self._setup()

    def tell(self) -> int:
        """Return the current frame number"""
        return self.__frame

    def get_photoshop_blocks(self) -> dict[int, dict[str, bytes]]:
        """
        Returns a dictionary of Photoshop "Image Resource Blocks".
        The keys are the image resource ID. For more information, see
        https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/#50577409_pgfId-1037727

        :returns: Photoshop "Image Resource Blocks" in a dictionary.
        """
        blocks = {}
        val = self.tag_v2.get(ExifTags.Base.ImageResources)
        if val:
            while val.startswith(b"8BIM") and len(val) >= 12:
                id = i16(val[4:6])
                n = math.ceil((val[6] + 1) / 2) * 2
                try:
                    size = i32(val[6 + n : 10 + n])
                except struct.error:
                    break
                data = val[10 + n : 10 + n + size]
                blocks[id] = {"data": data}

                val = val[math.ceil((10 + n + size) / 2) * 2 :]
        return blocks

    def load(self) -> Image.core.PixelAccess | None:
        if self.tile and self.use_load_libtiff:
            return self._load_libtiff()
        return super().load()

    def load_prepare(self) -> None:
        if self._im is None:
            Image._decompression_bomb_check(self._tile_size)
            self.im = Image.core.new(self.mode, self._tile_size)
        ImageFile.ImageFile.load_prepare(self)

    def load_end(self) -> None:
        # allow closing if we're on the first frame, there's no next
        # This is the ImageFile.load path only, libtiff specific below.
        if not self.is_animated:
            self._close_exclusive_fp_after_loading = True

            # load IFD data from fp before it is closed
            exif = self.getexif()
            for key in TiffTags.TAGS_V2_GROUPS:
                if key not in exif:
                    continue
                exif.get_ifd(key)

        ImageOps.exif_transpose(self, in_place=True)
        if ExifTags.Base.Orientation in self.tag_v2:
            del self.tag_v2[ExifTags.Base.Orientation]

    def _load_libtiff(self) -> Image.core.PixelAccess | None:
        """Overload method triggered when we detect a compressed tiff
        Calls out to libtiff"""

        Image.Image.load(self)

        self.load_prepare()

        if not len(self.tile) == 1:
            msg = "Not exactly one tile"
            raise OSError(msg)

        # (self._compression, (extents tuple),
        #   0, (rawmode, self._compression, fp))
        extents = self.tile[0][1]
        args = self.tile[0][3]

        # To be nice on memory footprint, if there's a
        # file descriptor, use that instead of reading
        # into a string in python.
        assert self.fp is not None
        try:
            fp = hasattr(self.fp, "fileno") and self.fp.fileno()
            # flush the file descriptor, prevents error on pypy 2.4+
            # should also eliminate the need for fp.tell
            # in _seek
            if hasattr(self.fp, "flush"):
                self.fp.flush()
        except OSError:
            # io.BytesIO have a fileno, but returns an OSError if
            # it doesn't use a file descriptor.
            fp = False

        if fp:
            assert isinstance(args, tuple)
            args_list = list(args)
            args_list[2] = fp
            args = tuple(args_list)

        decoder = Image._getdecoder(self.mode, "libtiff", args, self.decoderconfig)
        try:
            decoder.setimage(self.im, extents)
        except ValueError as e:
            msg = "Couldn't set the image"
            raise OSError(msg) from e

        close_self_fp = self._exclusive_fp and not self.is_animated
        if hasattr(self.fp, "getvalue"):
            # We've got a stringio like thing passed in. Yay for all in memory.
            # The decoder needs the entire file in one shot, so there's not
            # a lot we can do here other than give it the entire file.
            # unless we could do something like get the address of the
            # underlying string for stringio.
            #
            # Rearranging for supporting byteio items, since they have a fileno
            # that returns an OSError if there's no underlying fp. Easier to
            # deal with here by reordering.
            logger.debug("have getvalue. just sending in a string from getvalue")
            n, err = decoder.decode(self.fp.getvalue())
        elif fp:
            # we've got a actual file on disk, pass in the fp.
            logger.debug("have fileno, calling fileno version of the decoder.")
            if not close_self_fp:
                self.fp.seek(0)
            # Save and restore the file position, because libtiff will move it
            # outside of the Python runtime, and that will confuse
            # io.BufferedReader and possible others.
            # NOTE: This must use os.lseek(), and not fp.tell()/fp.seek(),
            # because the buffer read head already may not equal the actual
            # file position, and fp.seek() may just adjust it's internal
            # pointer and not actually seek the OS file handle.
            pos = os.lseek(fp, 0, os.SEEK_CUR)
            # 4 bytes, otherwise the trace might error out
            n, err = decoder.decode(b"fpfp")
            os.lseek(fp, pos, os.SEEK_SET)
        else:
            # we have something else.
            logger.debug("don't have fileno or getvalue. just reading")
            self.fp.seek(0)
            # UNDONE -- so much for that buffer size thing.
            n, err = decoder.decode(self.fp.read())

        self.tile = []
        self.readonly = 0

        self.load_end()

        if close_self_fp:
            self.fp.close()
            self.fp = None  # might be shared

        if err < 0:
            msg = f"decoder error {err}"
            raise OSError(msg)

        return Image.Image.load(self)

    def _setup(self) -> None:
        """Setup this image object based on current tags"""

        if 0xBC01 in self.tag_v2:
            msg = "Windows Media Photo files not yet supported"
            raise OSError(msg)

        # extract relevant tags
        self._compression = COMPRESSION_INFO[self.tag_v2.get(COMPRESSION, 1)]
        self._planar_configuration = self.tag_v2.get(PLANAR_CONFIGURATION, 1)

        # photometric is a required tag, but not everyone is reading
        # the specification
        photo = self.tag_v2.get(PHOTOMETRIC_INTERPRETATION, 0)

        # old style jpeg compression images most certainly are YCbCr
        if self._compression == "tiff_jpeg":
            photo = 6

        fillorder = self.tag_v2.get(FILLORDER, 1)

        logger.debug("*** Summary ***")
        logger.debug("- compression: %s", self._compression)
        logger.debug("- photometric_interpretation: %s", photo)
        logger.debug("- planar_configuration: %s", self._planar_configuration)
        logger.debug("- fill_order: %s", fillorder)
        logger.debug("- YCbCr subsampling: %s", self.tag_v2.get(YCBCRSUBSAMPLING))

        # size
        try:
            xsize = self.tag_v2[IMAGEWIDTH]
            ysize = self.tag_v2[IMAGELENGTH]
        except KeyError as e:
            msg = "Missing dimensions"
            raise TypeError(msg) from e
        if not isinstance(xsize, int) or not isinstance(ysize, int):
            msg = "Invalid dimensions"
            raise ValueError(msg)
        self._tile_size = xsize, ysize
        orientation = self.tag_v2.get(ExifTags.Base.Orientation)
        if orientation in (5, 6, 7, 8):
            self._size = ysize, xsize
        else:
            self._size = xsize, ysize

        logger.debug("- size: %s", self.size)

        sample_format = self.tag_v2.get(SAMPLEFORMAT, (1,))
        if len(sample_format) > 1 and max(sample_format) == min(sample_format):
            # SAMPLEFORMAT is properly per band, so an RGB image will
            # be (1,1,1).  But, we don't support per band pixel types,
            # and anything more than one band is a uint8. So, just
            # take the first element. Revisit this if adding support
            # for more exotic images.
            sample_format = (sample_format[0],)

        bps_tuple = self.tag_v2.get(BITSPERSAMPLE, (1,))
        extra_tuple = self.tag_v2.get(EXTRASAMPLES, ())
        samples_per_pixel = self.tag_v2.get(
            SAMPLESPERPIXEL,
            3 if self._compression == "tiff_jpeg" and photo in (2, 6) else 1,
        )
        if photo in (2, 6, 8):  # RGB, YCbCr, LAB
            bps_count = 3
        elif photo == 5:  # CMYK
            bps_count = 4
        else:
            bps_count = 1
        if self._planar_configuration == 2 and extra_tuple and max(extra_tuple) == 0:
            # If components are stored separately,
            # then unspecified extra components at the end can be ignored
            bps_tuple = bps_tuple[: -len(extra_tuple)]
            samples_per_pixel -= len(extra_tuple)
            extra_tuple = ()
        bps_count += len(extra_tuple)
        bps_actual_count = len(bps_tuple)

        if samples_per_pixel > MAX_SAMPLESPERPIXEL:
            # DOS check, samples_per_pixel can be a Long, and we extend the tuple below
            logger.error(
                "More samples per pixel than can be decoded: %s", samples_per_pixel
            )
            msg = "Invalid value for samples per pixel"
            raise SyntaxError(msg)

        if samples_per_pixel < bps_actual_count:
            # If a file has more values in bps_tuple than expected,
            # remove the excess.
            bps_tuple = bps_tuple[:samples_per_pixel]
        elif samples_per_pixel > bps_actual_count and bps_actual_count == 1:
            # If a file has only one value in bps_tuple, when it should have more,
            # presume it is the same number of bits for all of the samples.
            bps_tuple = bps_tuple * samples_per_pixel

        if len(bps_tuple) != samples_per_pixel:
            msg = "unknown data organization"
            raise SyntaxError(msg)

        # mode: check photometric interpretation and bits per pixel
        key = (
            self.tag_v2.prefix,
            photo,
            sample_format,
            fillorder,
            bps_tuple,
            extra_tuple,
        )
        logger.debug("format key: %s", key)
        try:
            self._mode, rawmode = OPEN_INFO[key]
        except KeyError as e:
            logger.debug("- unsupported format")
            msg = "unknown pixel mode"
            raise SyntaxError(msg) from e

        logger.debug("- raw mode: %s", rawmode)
        logger.debug("- pil mode: %s", self.mode)

        self.info["compression"] = self._compression

        xres = self.tag_v2.get(X_RESOLUTION, 1)
        yres = self.tag_v2.get(Y_RESOLUTION, 1)

        if xres and yres:
            resunit = self.tag_v2.get(RESOLUTION_UNIT)
            if resunit == 2:  # dots per inch
                self.info["dpi"] = (xres, yres)
            elif resunit == 3:  # dots per centimeter. convert to dpi
                self.info["dpi"] = (xres * 2.54, yres * 2.54)
            elif resunit is None:  # used to default to 1, but now 2)
                self.info["dpi"] = (xres, yres)
                # For backward compatibility,
                # we also preserve the old behavior
                self.info["resolution"] = xres, yres
            else:  # No absolute unit of measurement
                self.info["resolution"] = xres, yres

        # build tile descriptors
        x = y = layer = 0
        self.tile = []
        self.use_load_libtiff = READ_LIBTIFF or self._compression != "raw"
        if self.use_load_libtiff:
            # Decoder expects entire file as one tile.
            # There's a buffer size limit in load (64k)
            # so large g4 images will fail if we use that
            # function.
            #
            # Setup the one tile for the whole image, then
            # use the _load_libtiff function.

            # libtiff handles the fillmode for us, so 1;IR should
            # actually be 1;I. Including the R double reverses the
            # bits, so stripes of the image are reversed.  See
            # https://github.com/python-pillow/Pillow/issues/279
            if fillorder == 2:
                # Replace fillorder with fillorder=1
                key = key[:3] + (1,) + key[4:]
                logger.debug("format key: %s", key)
                # this should always work, since all the
                # fillorder==2 modes have a corresponding
                # fillorder=1 mode
                self._mode, rawmode = OPEN_INFO[key]
            # YCbCr images with new jpeg compression with pixels in one plane
            # unpacked straight into RGB values
            if (
                photo == 6
                and self._compression == "jpeg"
                and self._planar_configuration == 1
            ):
                rawmode = "RGB"
            # libtiff always returns the bytes in native order.
            # we're expecting image byte order. So, if the rawmode
            # contains I;16, we need to convert from native to image
            # byte order.
            elif rawmode == "I;16":
                rawmode = "I;16N"
            elif rawmode.endswith((";16B", ";16L")):
                rawmode = rawmode[:-1] + "N"

            # Offset in the tile tuple is 0, we go from 0,0 to
            # w,h, and we only do this once -- eds
            a = (rawmode, self._compression, False, self.tag_v2.offset)
            self.tile.append(ImageFile._Tile("libtiff", (0, 0, xsize, ysize), 0, a))

        elif STRIPOFFSETS in self.tag_v2 or TILEOFFSETS in self.tag_v2:
            # striped image
            if STRIPOFFSETS in self.tag_v2:
                offsets = self.tag_v2[STRIPOFFSETS]
                h = self.tag_v2.get(ROWSPERSTRIP, ysize)
                w = xsize
            else:
                # tiled image
                offsets = self.tag_v2[TILEOFFSETS]
                tilewidth = self.tag_v2.get(TILEWIDTH)
                h = self.tag_v2.get(TILELENGTH)
                if not isinstance(tilewidth, int) or not isinstance(h, int):
                    msg = "Invalid tile dimensions"
                    raise ValueError(msg)
                w = tilewidth

            if w == xsize and h == ysize and self._planar_configuration != 2:
                # Every tile covers the image. Only use the last offset
                offsets = offsets[-1:]

            for offset in offsets:
                if x + w > xsize:
                    stride = w * sum(bps_tuple) / 8  # bytes per line
                else:
                    stride = 0

                tile_rawmode = rawmode
                if self._planar_configuration == 2:
                    # each band on it's own layer
                    tile_rawmode = rawmode[layer]
                    # adjust stride width accordingly
                    stride /= bps_count

                args = (tile_rawmode, int(stride), 1)
                self.tile.append(
                    ImageFile._Tile(
                        self._compression,
                        (x, y, min(x + w, xsize), min(y + h, ysize)),
                        offset,
                        args,
                    )
                )
                x += w
                if x >= xsize:
                    x, y = 0, y + h
                    if y >= ysize:
                        y = 0
                        layer += 1
        else:
            logger.debug("- unsupported data organization")
            msg = "unknown data organization"
            raise SyntaxError(msg)

        # Fix up info.
        if ICCPROFILE in self.tag_v2:
            self.info["icc_profile"] = self.tag_v2[ICCPROFILE]

        # fixup palette descriptor

        if self.mode in ["P", "PA"]:
            palette = [o8(b // 256) for b in self.tag_v2[COLORMAP]]
            self.palette = ImagePalette.raw("RGB;L", b"".join(palette))


#
# --------------------------------------------------------------------
# Write TIFF files

# little endian is default except for image modes with
# explicit big endian byte-order

SAVE_INFO = {
    # mode => rawmode, byteorder, photometrics,
    #           sampleformat, bitspersample, extra
    "1": ("1", II, 1, 1, (1,), None),
    "L": ("L", II, 1, 1, (8,), None),
    "LA": ("LA", II, 1, 1, (8, 8), 2),
    "P": ("P", II, 3, 1, (8,), None),
    "PA": ("PA", II, 3, 1, (8, 8), 2),
    "I": ("I;32S", II, 1, 2, (32,), None),
    "I;16": ("I;16", II, 1, 1, (16,), None),
    "I;16L": ("I;16L", II, 1, 1, (16,), None),
    "F": ("F;32F", II, 1, 3, (32,), None),
    "RGB": ("RGB", II, 2, 1, (8, 8, 8), None),
    "RGBX": ("RGBX", II, 2, 1, (8, 8, 8, 8), 0),
    "RGBA": ("RGBA", II, 2, 1, (8, 8, 8, 8), 2),
    "CMYK": ("CMYK", II, 5, 1, (8, 8, 8, 8), None),
    "YCbCr": ("YCbCr", II, 6, 1, (8, 8, 8), None),
    "LAB": ("LAB", II, 8, 1, (8, 8, 8), None),
    "I;16B": ("I;16B", MM, 1, 1, (16,), None),
}


def _save(im: Image.Image, fp: IO[bytes], filename: str | bytes) -> None:
    try:
        rawmode, prefix, photo, format, bits, extra = SAVE_INFO[im.mode]
    except KeyError as e:
        msg = f"cannot write mode {im.mode} as TIFF"
        raise OSError(msg) from e

    encoderinfo = im.encoderinfo
    encoderconfig = im.encoderconfig

    ifd = ImageFileDirectory_v2(prefix=prefix)
    if encoderinfo.get("big_tiff"):
        ifd._bigtiff = True

    try:
        compression = encoderinfo["compression"]
    except KeyError:
        compression = im.info.get("compression")
        if isinstance(compression, int):
            # compression value may be from BMP. Ignore it
            compression = None
    if compression is None:
        compression = "raw"
    elif compression == "tiff_jpeg":
        # OJPEG is obsolete, so use new-style JPEG compression instead
        compression = "jpeg"
    elif compression == "tiff_deflate":
        compression = "tiff_adobe_deflate"

    libtiff = WRITE_LIBTIFF or compression != "raw"

    # required for color libtiff images
    ifd[PLANAR_CONFIGURATION] = 1

    ifd[IMAGEWIDTH] = im.size[0]
    ifd[IMAGELENGTH] = im.size[1]

    # write any arbitrary tags passed in as an ImageFileDirectory
    if "tiffinfo" in encoderinfo:
        info = encoderinfo["tiffinfo"]
    elif "exif" in encoderinfo:
        info = encoderinfo["exif"]
        if isinstance(info, bytes):
            exif = Image.Exif()
            exif.load(info)
            info = exif
    else:
        info = {}
    logger.debug("Tiffinfo Keys: %s", list(info))
    if isinstance(info, ImageFileDirectory_v1):
        info = info.to_v2()
    for key in info:
        if isinstance(info, Image.Exif) and key in TiffTags.TAGS_V2_GROUPS:
            ifd[key] = info.get_ifd(key)
        else:
            ifd[key] = info.get(key)
        try:
            ifd.tagtype[key] = info.tagtype[key]
        except Exception:
            pass  # might not be an IFD. Might not have populated type

    legacy_ifd = {}
    if hasattr(im, "tag"):
        legacy_ifd = im.tag.to_v2()

    supplied_tags = {**legacy_ifd, **getattr(im, "tag_v2", {})}
    if supplied_tags.get(PLANAR_CONFIGURATION) == 2 and EXTRASAMPLES in supplied_tags:
        # If the image used separate component planes,
        # then EXTRASAMPLES should be ignored when saving contiguously
        if SAMPLESPERPIXEL in supplied_tags:
            supplied_tags[SAMPLESPERPIXEL] -= len(supplied_tags[EXTRASAMPLES])
        del supplied_tags[EXTRASAMPLES]
    for tag in (
        # IFD offset that may not be correct in the saved image
        EXIFIFD,
        # Determined by the image format and should not be copied from legacy_ifd.
        SAMPLEFORMAT,
    ):
        if tag in supplied_tags:
            del supplied_tags[tag]

    # additions written by Greg Couch, gregc@cgl.ucsf.edu
    # inspired by image-sig posting from Kevin Cazabon, kcazabon@home.com
    if hasattr(im, "tag_v2"):
        # preserve tags from original TIFF image file
        for key in (
            RESOLUTION_UNIT,
            X_RESOLUTION,
            Y_RESOLUTION,
            IPTC_NAA_CHUNK,
            PHOTOSHOP_CHUNK,
            XMP,
        ):
            if key in im.tag_v2:
                if key == IPTC_NAA_CHUNK and im.tag_v2.tagtype[key] not in (
                    TiffTags.BYTE,
                    TiffTags.UNDEFINED,
                ):
                    del supplied_tags[key]
                else:
                    ifd[key] = im.tag_v2[key]
                    ifd.tagtype[key] = im.tag_v2.tagtype[key]

    # preserve ICC profile (should also work when saving other formats
    # which support profiles as TIFF) -- 2008-06-06 Florian Hoech
    icc = encoderinfo.get("icc_profile", im.info.get("icc_profile"))
    if icc:
        ifd[ICCPROFILE] = icc

    for key, name in [
        (IMAGEDESCRIPTION, "description"),
        (X_RESOLUTION, "resolution"),
        (Y_RESOLUTION, "resolution"),
        (X_RESOLUTION, "x_resolution"),
        (Y_RESOLUTION, "y_resolution"),
        (RESOLUTION_UNIT, "resolution_unit"),
        (SOFTWARE, "software"),
        (DATE_TIME, "date_time"),
        (ARTIST, "artist"),
        (COPYRIGHT, "copyright"),
    ]:
        if name in encoderinfo:
            ifd[key] = encoderinfo[name]

    dpi = encoderinfo.get("dpi")
    if dpi:
        ifd[RESOLUTION_UNIT] = 2
        ifd[X_RESOLUTION] = dpi[0]
        ifd[Y_RESOLUTION] = dpi[1]

    if bits != (1,):
        ifd[BITSPERSAMPLE] = bits
        if len(bits) != 1:
            ifd[SAMPLESPERPIXEL] = len(bits)
    if extra is not None:
        ifd[EXTRASAMPLES] = extra
    if format != 1:
        ifd[SAMPLEFORMAT] = format

    if PHOTOMETRIC_INTERPRETATION not in ifd:
        ifd[PHOTOMETRIC_INTERPRETATION] = photo
    elif im.mode in ("1", "L") and ifd[PHOTOMETRIC_INTERPRETATION] == 0:
        if im.mode == "1":
            inverted_im = im.copy()
            px = inverted_im.load()
            if px is not None:
                for y in range(inverted_im.height):
                    for x in range(inverted_im.width):
                        px[x, y] = 0 if px[x, y] == 255 else 255
                im = inverted_im
        else:
            im = ImageOps.invert(im)

    if im.mode in ["P", "PA"]:
        lut = im.im.getpalette("RGB", "RGB;L")
        colormap = []
        colors = len(lut) // 3
        for i in range(3):
            colormap += [v * 256 for v in lut[colors * i : colors * (i + 1)]]
            colormap += [0] * (256 - colors)
        ifd[COLORMAP] = colormap
    # data orientation
    w, h = ifd[IMAGEWIDTH], ifd[IMAGELENGTH]
    stride = len(bits) * ((w * bits[0] + 7) // 8)
    if ROWSPERSTRIP not in ifd:
        # aim for given strip size (64 KB by default) when using libtiff writer
        if libtiff:
            im_strip_size = encoderinfo.get("strip_size", STRIP_SIZE)
            rows_per_strip = 1 if stride == 0 else min(im_strip_size // stride, h)
            # JPEG encoder expects multiple of 8 rows
            if compression == "jpeg":
                rows_per_strip = min(((rows_per_strip + 7) // 8) * 8, h)
        else:
            rows_per_strip = h
        if rows_per_strip == 0:
            rows_per_strip = 1
        ifd[ROWSPERSTRIP] = rows_per_strip
    strip_byte_counts = 1 if stride == 0 else stride * ifd[ROWSPERSTRIP]
    strips_per_image = (h + ifd[ROWSPERSTRIP] - 1) // ifd[ROWSPERSTRIP]
    if strip_byte_counts >= 2**16:
        ifd.tagtype[STRIPBYTECOUNTS] = TiffTags.LONG
    ifd[STRIPBYTECOUNTS] = (strip_byte_counts,) * (strips_per_image - 1) + (
        stride * h - strip_byte_counts * (strips_per_image - 1),
    )
    ifd[STRIPOFFSETS] = tuple(
        range(0, strip_byte_counts * strips_per_image, strip_byte_counts)
    )  # this is adjusted by IFD writer
    # no compression by default:
    ifd[COMPRESSION] = COMPRESSION_INFO_REV.get(compression, 1)

    if im.mode == "YCbCr":
        for tag, default_value in {
            YCBCRSUBSAMPLING: (1, 1),
            REFERENCEBLACKWHITE: (0, 255, 128, 255, 128, 255),
        }.items():
            ifd.setdefault(tag, default_value)

    blocklist = [TILEWIDTH, TILELENGTH, TILEOFFSETS, TILEBYTECOUNTS]
    if libtiff:
        if "quality" in encoderinfo:
            quality = encoderinfo["quality"]
            if not isinstance(quality, int) or quality < 0 or quality > 100:
                msg = "Invalid quality setting"
                raise ValueError(msg)
            if compression != "jpeg":
                msg = "quality setting only supported for 'jpeg' compression"
                raise ValueError(msg)
            ifd[JPEGQUALITY] = quality

        logger.debug("Saving using libtiff encoder")
        logger.debug("Items: %s", sorted(ifd.items()))
        _fp = 0
        if hasattr(fp, "fileno"):
            try:
                fp.seek(0)
                _fp = fp.fileno()
            except io.UnsupportedOperation:
                pass

        # optional types for non core tags
        types = {}
        # STRIPOFFSETS and STRIPBYTECOUNTS are added by the library
        # based on the data in the strip.
        # OSUBFILETYPE is deprecated.
        # The other tags expect arrays with a certain length (fixed or depending on
        # BITSPERSAMPLE, etc), passing arrays with a different length will result in
        # segfaults. Block these tags until we add extra validation.
        # SUBIFD may also cause a segfault.
        blocklist += [
            OSUBFILETYPE,
            REFERENCEBLACKWHITE,
            STRIPBYTECOUNTS,
            STRIPOFFSETS,
            TRANSFERFUNCTION,
            SUBIFD,
        ]

        # bits per sample is a single short in the tiff directory, not a list.
        atts: dict[int, Any] = {BITSPERSAMPLE: bits[0]}
        # Merge the ones that we have with (optional) more bits from
        # the original file, e.g x,y resolution so that we can
        # save(load('')) == original file.
        for tag, value in itertools.chain(ifd.items(), supplied_tags.items()):
            # Libtiff can only process certain core items without adding
            # them to the custom dictionary.
            # Custom items are supported for int, float, unicode, string and byte
            # values. Other types and tuples require a tagtype.
            if tag not in TiffTags.LIBTIFF_CORE:
                if tag in TiffTags.TAGS_V2_GROUPS:
                    types[tag] = TiffTags.LONG8
                elif tag in ifd.tagtype:
                    types[tag] = ifd.tagtype[tag]
                elif isinstance(value, (int, float, str, bytes)) or (
                    isinstance(value, tuple)
                    and all(isinstance(v, (int, float, IFDRational)) for v in value)
                ):
                    type = TiffTags.lookup(tag).type
                    if type:
                        types[tag] = type
            if tag not in atts and tag not in blocklist:
                if isinstance(value, str):
                    atts[tag] = value.encode("ascii", "replace") + b"\0"
                elif isinstance(value, IFDRational):
                    atts[tag] = float(value)
                else:
                    atts[tag] = value

        if SAMPLEFORMAT in atts and len(atts[SAMPLEFORMAT]) == 1:
            atts[SAMPLEFORMAT] = atts[SAMPLEFORMAT][0]

        logger.debug("Converted items: %s", sorted(atts.items()))

        # libtiff always expects the bytes in native order.
        # we're storing image byte order. So, if the rawmode
