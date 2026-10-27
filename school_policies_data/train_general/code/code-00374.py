            offsets=offsets, offset_transform=offset_transform,
        )

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = self.get_ydata(legend, xdescent, ydescent,
                               width, height, fontsize)

        sizes = self.get_sizes(legend, orig_handle, xdescent, ydescent,
                               width, height, fontsize)

        p = self.create_collection(
            orig_handle, sizes,
            offsets=list(zip(xdata_marker, ydata)), offset_transform=trans)

        self.update_prop(p, orig_handle, legend)
        p.set_offset_transform(trans)
        return [p]


class HandlerPathCollection(HandlerRegularPolyCollection):
    r"""Handler for `.PathCollection`\s, which are used by `~.Axes.scatter`."""

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        return type(orig_handle)(
            [orig_handle.get_paths()[0]], sizes=sizes,
            offsets=offsets, offset_transform=offset_transform,
        )


class HandlerCircleCollection(HandlerRegularPolyCollection):
    r"""Handler for `.CircleCollection`\s."""

    def create_collection(self, orig_handle, sizes, offsets, offset_transform):
        return type(orig_handle)(
            sizes, offsets=offsets, offset_transform=offset_transform)


class HandlerErrorbar(HandlerLine2D):
    """Handler for Errorbars."""

    def __init__(self, xerr_size=0.5, yerr_size=None,
                 marker_pad=0.3, numpoints=None, **kwargs):

        self._xerr_size = xerr_size
        self._yerr_size = yerr_size

        super().__init__(marker_pad=marker_pad, numpoints=numpoints, **kwargs)

    def get_err_size(self, legend, xdescent, ydescent,
                     width, height, fontsize):
        xerr_size = self._xerr_size * fontsize

        if self._yerr_size is None:
            yerr_size = xerr_size
        else:
            yerr_size = self._yerr_size * fontsize

        return xerr_size, yerr_size

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        plotlines, caplines, barlinecols = orig_handle

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = np.full_like(xdata, (height - ydescent) / 2)
        legline = Line2D(xdata, ydata)

        xdata_marker = np.asarray(xdata_marker)
        ydata_marker = np.asarray(ydata[:len(xdata_marker)])

        xerr_size, yerr_size = self.get_err_size(legend, xdescent, ydescent,
                                                 width, height, fontsize)

        legline_marker = Line2D(xdata_marker, ydata_marker)

        # when plotlines are None (only errorbars are drawn), we just
        # make legline invisible.
        if plotlines is None:
            legline.set_visible(False)
            legline_marker.set_visible(False)
        else:
            self.update_prop(legline, plotlines, legend)

            legline.set_drawstyle('default')
            legline.set_marker('none')

            self.update_prop(legline_marker, plotlines, legend)
            legline_marker.set_linestyle('None')

            if legend.markerscale != 1:
                newsz = legline_marker.get_markersize() * legend.markerscale
                legline_marker.set_markersize(newsz)

        handle_barlinecols = []
        handle_caplines = []

        if orig_handle.has_xerr:
            verts = [((x - xerr_size, y), (x + xerr_size, y))
                     for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)

            if caplines:
                capline_left = Line2D(xdata_marker - xerr_size, ydata_marker)
                capline_right = Line2D(xdata_marker + xerr_size, ydata_marker)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker("|")
                capline_right.set_marker("|")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        if orig_handle.has_yerr:
            verts = [((x, y - yerr_size), (x, y + yerr_size))
                     for x, y in zip(xdata_marker, ydata_marker)]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, barlinecols[0], legend)
            handle_barlinecols.append(coll)

            if caplines:
                capline_left = Line2D(xdata_marker, ydata_marker - yerr_size)
                capline_right = Line2D(xdata_marker, ydata_marker + yerr_size)
                self.update_prop(capline_left, caplines[0], legend)
                self.update_prop(capline_right, caplines[0], legend)
                capline_left.set_marker("_")
                capline_right.set_marker("_")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        artists = [
            *handle_barlinecols, *handle_caplines, legline, legline_marker,
        ]
        for artist in artists:
            artist.set_transform(trans)
        return artists


class HandlerStem(HandlerNpointsYoffsets):
    """
    Handler for plots produced by `~.Axes.stem`.
    """

    def __init__(self, marker_pad=0.3, numpoints=None,
                 bottom=None, yoffsets=None, **kwargs):
        """
        Parameters
        ----------
        marker_pad : float, default: 0.3
            Padding between points in legend entry.
        numpoints : int, optional
            Number of points to show in legend entry.
        bottom : float, optional

        yoffsets : array of floats, optional
            Length *numpoints* list of y offsets for each point in
            legend entry.
        **kwargs
            Keyword arguments forwarded to `.HandlerNpointsYoffsets`.
        """
        super().__init__(marker_pad=marker_pad, numpoints=numpoints,
                         yoffsets=yoffsets, **kwargs)
        self._bottom = bottom

    def get_ydata(self, legend, xdescent, ydescent, width, height, fontsize):
        if self._yoffsets is None:
            ydata = height * (0.5 * legend._scatteryoffsets + 0.5)
        else:
            ydata = height * np.asarray(self._yoffsets)

        return ydata

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        markerline, stemlines, baseline = orig_handle
        # Check to see if the stemcontainer is storing lines as a list or a
        # LineCollection. Eventually using a list will be removed, and this
        # logic can also be removed.
        using_linecoll = isinstance(stemlines, mcoll.LineCollection)

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)

        ydata = self.get_ydata(legend, xdescent, ydescent,
                               width, height, fontsize)

        if self._bottom is None:
            bottom = 0.
        else:
            bottom = self._bottom

        leg_markerline = Line2D(xdata_marker, ydata[:len(xdata_marker)])
        self.update_prop(leg_markerline, markerline, legend)

        leg_stemlines = [Line2D([x, x], [bottom, y])
                         for x, y in zip(xdata_marker, ydata)]

        if using_linecoll:
            # change the function used by update_prop() from the default
            # to one that handles LineCollection
            with cbook._setattr_cm(
                    self, _update_prop_func=self._copy_collection_props):
                for line in leg_stemlines:
                    self.update_prop(line, stemlines, legend)

        else:
            for lm, m in zip(leg_stemlines, stemlines):
                self.update_prop(lm, m, legend)

        leg_baseline = Line2D([np.min(xdata), np.max(xdata)],
                              [bottom, bottom])
        self.update_prop(leg_baseline, baseline, legend)

        artists = [*leg_stemlines, leg_baseline, leg_markerline]
        for artist in artists:
            artist.set_transform(trans)
        return artists

    def _copy_collection_props(self, legend_handle, orig_handle):
        """
        Copy properties from the `.LineCollection` *orig_handle* to the
        `.Line2D` *legend_handle*.
        """
        legend_handle.set_color(orig_handle.get_color()[0])
        legend_handle.set_linestyle(orig_handle.get_linestyle()[0])


class HandlerTuple(HandlerBase):
    """
    Handler for Tuple.
    """

    def __init__(self, ndivide=1, pad=None, **kwargs):
        """
        Parameters
        ----------
        ndivide : int or None, default: 1
            The number of sections to divide the legend area into.  If None,
            use the length of the input tuple.
        pad : float, default: :rc:`legend.borderpad`
            Padding in units of fraction of font size.
        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        self._ndivide = ndivide
        self._pad = pad
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        # docstring inherited
        handler_map = legend.get_legend_handler_map()

        if self._ndivide is None:
            ndivide = len(orig_handle)
        else:
            ndivide = self._ndivide

        if self._pad is None:
            pad = legend.borderpad * fontsize
        else:
            pad = self._pad * fontsize

        if ndivide > 1:
            width = (width - pad * (ndivide - 1)) / ndivide

        xds_cycle = cycle(xdescent - (width + pad) * np.arange(ndivide))

        a_list = []
        for handle1 in orig_handle:
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(
                legend, handle1,
                next(xds_cycle), ydescent, width, height, fontsize, trans)
            a_list.extend(_a_list)

        return a_list


class HandlerPolyCollection(HandlerBase):
    """
    Handler for `.PolyCollection` used in `~.Axes.fill_between` and
    `~.Axes.stackplot`.
    """
    def _update_prop(self, legend_handle, orig_handle):
        def first_color(colors):
            if colors.size == 0:
                return (0, 0, 0, 0)
            return tuple(colors[0])

        def get_first(prop_array):
            if len(prop_array):
                return prop_array[0]
            else:
                return None

        # orig_handle is a PolyCollection and legend_handle is a Patch.
        # Directly set Patch color attributes (must be RGBA tuples).
        legend_handle._facecolor = first_color(orig_handle.get_facecolor())
        legend_handle._edgecolor = first_color(orig_handle.get_edgecolor())
        legend_handle._original_facecolor = orig_handle._original_facecolor
        legend_handle._original_edgecolor = orig_handle._original_edgecolor
        legend_handle._fill = orig_handle.get_fill()
        legend_handle._hatch = orig_handle.get_hatch()
        # Hatch color is anomalous in having no getters and setters.
        legend_handle._hatch_color = orig_handle._hatch_color
        # Setters are fine for the remaining attributes.
        legend_handle.set_linewidth(get_first(orig_handle.get_linewidths()))
        legend_handle.set_linestyle(get_first(orig_handle.get_linestyles()))
        legend_handle.set_transform(get_first(orig_handle.get_transforms()))
        # Alpha is already taken into account by the color attributes.

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # docstring inherited
        p = Rectangle(xy=(-xdescent, -ydescent),
                      width=width, height=height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
<|endoftext|>"""
A module for finding, managing, and using fonts across platforms.

This module provides a single `FontManager` instance, ``fontManager``, that can
be shared across backends and platforms.  The `findfont`
function returns the best TrueType (TTF) font file in the local or
system font path that matches the specified `FontProperties`
instance.  The `FontManager` also handles Adobe Font Metrics
(AFM) font files for use by the PostScript backend.
The `FontManager.addfont` function adds a custom font from a file without
installing it into your operating system.

The design is based on the `W3C Cascading Style Sheet, Level 1 (CSS1)
font specification <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_.
Future versions may implement the Level 2 or 2.1 specifications.
"""

# KNOWN ISSUES
#
#   - documentation
#   - font variant is untested
#   - font stretch is incomplete
#   - font size is incomplete
#   - default font algorithm needs improvement and testing
#   - setWeights function needs improvement
#   - 'light' is an invalid weight value, remove it.

from __future__ import annotations

from base64 import b64encode
import copy
import dataclasses
from functools import lru_cache
import functools
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
import threading

import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
    parse_fontconfig_pattern, generate_fontconfig_pattern)
from matplotlib.rcsetup import _validators

_log = logging.getLogger(__name__)

font_scalings = {
    'xx-small': 0.579,
    'x-small':  0.694,
    'small':    0.833,
    'medium':   1.0,
    'large':    1.200,
    'x-large':  1.440,
    'xx-large': 1.728,
    'larger':   1.2,
    'smaller':  0.833,
    None:       1.0,
}
stretch_dict = {
    'ultra-condensed': 100,
    'extra-condensed': 200,
    'condensed':       300,
    'semi-condensed':  400,
    'normal':          500,
    'semi-expanded':   600,
    'semi-extended':   600,
    'expanded':        700,
    'extended':        700,
    'extra-expanded':  800,
    'extra-extended':  800,
    'ultra-expanded':  900,
    'ultra-extended':  900,
}
weight_dict = {
    'ultralight': 100,
    'light':      200,
    'normal':     400,
    'regular':    400,
    'book':       400,
    'medium':     500,
    'roman':      500,
    'semibold':   600,
    'demibold':   600,
    'demi':       600,
    'bold':       700,
    'heavy':      800,
    'extra bold': 800,
    'black':      900,
}
_weight_regexes = [
    # From fontconfig's FcFreeTypeQueryFaceInternal; not the same as
    # weight_dict!
    ("thin", 100),
    ("extralight", 200),
    ("ultralight", 200),
    ("demilight", 350),
    ("semilight", 350),
    ("light", 300),  # Needs to come *after* demi/semilight!
    ("book", 380),
    ("regular", 400),
    ("normal", 400),
    ("medium", 500),
    ("demibold", 600),
    ("demi", 600),
    ("semibold", 600),
    ("extrabold", 800),
    ("superbold", 800),
    ("ultrabold", 800),
    ("bold", 700),  # Needs to come *after* extra/super/ultrabold!
    ("ultrablack", 1000),
    ("superblack", 1000),
    ("extrablack", 1000),
    (r"\bultra", 1000),
    ("black", 900),  # Needs to come *after* ultra/super/extrablack!
    ("heavy", 900),
]
font_family_aliases = {
    'serif',
    'sans-serif',
    'sans serif',
    'cursive',
    'fantasy',
    'monospace',
    'sans',
}

# OS Font paths
try:
    _HOME = Path.home()
except Exception:  # Exceptions thrown by home() are not specified...
    _HOME = Path(os.devnull)  # Just an arbitrary path with no children.
MSFolders = \
    r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
MSFontDirectories = [
    r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts',
    r'SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts']
MSUserFontDirectories = [
    str(_HOME / 'AppData/Local/Microsoft/Windows/Fonts'),
    str(_HOME / 'AppData/Roaming/Microsoft/Windows/Fonts'),
]
X11FontDirectories = [
    # an old standard installation point
    "/usr/X11R6/lib/X11/fonts/TTF/",
    "/usr/X11/lib/X11/fonts",
    # here is the new standard location for fonts
    "/usr/share/fonts/",
    # documented as a good place to install new fonts
    "/usr/local/share/fonts/",
    # common application, not really useful
    "/usr/lib/openoffice/share/fonts/truetype/",
    # user fonts
    str((Path(os.environ.get('XDG_DATA_HOME') or _HOME / ".local/share"))
        / "fonts"),
    str(_HOME / ".fonts"),
]
OSXFontDirectories = [
    "/Library/Fonts/",
    "/Network/Library/Fonts/",
    "/System/Library/Fonts/",
    # fonts installed via MacPorts
    "/opt/local/share/fonts",
    # user fonts
    str(_HOME / "Library/Fonts"),
]


def get_fontext_synonyms(fontext):
    """
    Return a list of file extensions that are synonyms for
    the given file extension *fileext*.
    """
    return {
        'afm': ['afm'],
        'otf': ['otf', 'ttc', 'ttf'],
        'ttc': ['otf', 'ttc', 'ttf'],
        'ttf': ['otf', 'ttc', 'ttf'],
    }[fontext]


def list_fonts(directory, extensions):
    """
    Return a list of all fonts matching any of the extensions, found
    recursively under the directory.
    """
    extensions = ["." + ext for ext in extensions]
    return [os.path.join(dirpath, filename)
            # os.walk ignores access errors, unlike Path.glob.
            for dirpath, _, filenames in os.walk(directory)
            for filename in filenames
            if Path(filename).suffix.lower() in extensions]


def win32FontDirectory():
    r"""
    Return the user-specified font directory for Win32.  This is
    looked up from the registry key ::

      \\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders\Fonts

    If the key is not found, ``%WINDIR%\Fonts`` will be returned.
    """  # noqa: E501
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MSFolders) as user:
            return winreg.QueryValueEx(user, 'Fonts')[0]
    except OSError:
        return os.path.join(os.environ['WINDIR'], 'Fonts')


def _get_win32_installed_fonts():
    """List the font paths known to the Windows registry."""
    import winreg
    items = set()
    # Search and resolve fonts listed in the registry.
    for domain, base_dirs in [
            (winreg.HKEY_LOCAL_MACHINE, [win32FontDirectory()]),  # System.
            (winreg.HKEY_CURRENT_USER, MSUserFontDirectories),  # User.
    ]:
        for base_dir in base_dirs:
            for reg_path in MSFontDirectories:
                try:
                    with winreg.OpenKey(domain, reg_path) as local:
                        for j in range(winreg.QueryInfoKey(local)[1]):
                            # value may contain the filename of the font or its
                            # absolute path.
                            key, value, tp = winreg.EnumValue(local, j)
                            if not isinstance(value, str):
                                continue
                            try:
                                # If value contains already an absolute path,
                                # then it is not changed further.
                                path = Path(base_dir, value).resolve()
                            except RuntimeError:
                                # Don't fail with invalid entries.
                                continue
                            items.add(path)
                except (OSError, MemoryError):
                    continue
    return items


@lru_cache
def _get_fontconfig_fonts():
    """Cache and list the font paths known to ``fc-list``."""
    try:
        if b'--format' not in subprocess.check_output(['fc-list', '--help']):
            _log.warning(  # fontconfig 2.7 implemented --format.
                'Matplotlib needs fontconfig>=2.7 to query system fonts.')
            return []
        out = subprocess.check_output(['fc-list', '--format=%{file}\\n'])
    except (OSError, subprocess.CalledProcessError):
        return []
    return [Path(os.fsdecode(fname)) for fname in out.split(b'\n')]


@lru_cache
def _get_macos_fonts():
    """Cache and list the font paths known to ``system_profiler SPFontsDataType``."""
    try:
        d, = plistlib.loads(
            subprocess.check_output(["system_profiler", "-xml", "SPFontsDataType"]))
    except (OSError, subprocess.CalledProcessError, plistlib.InvalidFileException):
        return []
    return [Path(entry["path"]) for entry in d["_items"]]


def findSystemFonts(fontpaths=None, fontext='ttf'):
    """
    Search for fonts in the specified font paths.  If no paths are
    given, will use a standard set of system paths, as well as the
    list of fonts tracked by fontconfig if fontconfig is installed and
    available.  A list of TrueType fonts are returned by default with
    AFM fonts as an option.
    """
    fontfiles = set()
    fontexts = get_fontext_synonyms(fontext)

    if fontpaths is None:
        if sys.platform == 'win32':
            installed_fonts = _get_win32_installed_fonts()
            fontpaths = []
        else:
            installed_fonts = _get_fontconfig_fonts()
            if sys.platform == 'darwin':
                installed_fonts += _get_macos_fonts()
                fontpaths = [*X11FontDirectories, *OSXFontDirectories]
            else:
                fontpaths = X11FontDirectories
        fontfiles.update(str(path) for path in installed_fonts
                         if path.suffix.lower()[1:] in fontexts)

    elif isinstance(fontpaths, str):
        fontpaths = [fontpaths]

    for path in fontpaths:
        fontfiles.update(map(os.path.abspath, list_fonts(path, fontexts)))

    return [fname for fname in fontfiles if os.path.exists(fname)]


@dataclasses.dataclass(frozen=True)
class FontEntry:
    """
    A class for storing Font properties.

    It is used when populating the font lookup dictionary.
    """

    fname: str = ''
    name: str = ''
    style: str = 'normal'
    variant: str = 'normal'
    weight: str | int = 'normal'
    stretch: str = 'normal'
    size: str = 'medium'

    def _repr_html_(self) -> str:
        png_stream = self._repr_png_()
        png_b64 = b64encode(png_stream).decode()
        return f"<img src=\"data:image/png;base64, {png_b64}\" />"

    def _repr_png_(self) -> bytes:
        from matplotlib.figure import Figure  # Circular import.
        fig = Figure()
        font_path = Path(self.fname) if self.fname != '' else None
        fig.text(0, 0, self.name, font=font_path)
        with BytesIO() as buf:
            fig.savefig(buf, bbox_inches='tight', transparent=True)
            return buf.getvalue()


def ttfFontProperty(font):
    """
    Extract information from a TrueType font file.

    Parameters
    ----------
    font : `.FT2Font`
        The TrueType font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.

    """
    name = font.family_name

    #  Styles are: italic, oblique, and normal (default)

    sfnt = font.get_sfnt()
    mac_key = (1,  # platform: macintosh
               0,  # id: roman
               0)  # langid: english
    ms_key = (3,  # platform: microsoft
              1,  # id: unicode_cs
              0x0409)  # langid: english_united_states

    # These tables are actually mac_roman-encoded, but mac_roman support may be
    # missing in some alternative Python implementations and we are only going
    # to look for ASCII substrings, where any ASCII-compatible encoding works
    # - or big-endian UTF-16, since important Microsoft fonts use that.
    sfnt2 = (sfnt.get((*mac_key, 2), b'').decode('latin-1').lower() or
             sfnt.get((*ms_key, 2), b'').decode('utf_16_be').lower())
    sfnt4 = (sfnt.get((*mac_key, 4), b'').decode('latin-1').lower() or
             sfnt.get((*ms_key, 4), b'').decode('utf_16_be').lower())

    if sfnt4.find('oblique') >= 0:
        style = 'oblique'
    elif sfnt4.find('italic') >= 0:
        style = 'italic'
    elif sfnt2.find('regular') >= 0:
        style = 'normal'
    elif ft2font.StyleFlags.ITALIC in font.style_flags:
        style = 'italic'
    else:
        style = 'normal'

    #  Variants are: small-caps and normal (default)

    #  !!!!  Untested
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'

    # The weight-guessing algorithm is directly translated from fontconfig
    # 2.13.1's FcFreeTypeQueryFaceInternal (fcfreetype.c).
    wws_subfamily = 22
    typographic_subfamily = 16
    font_subfamily = 2
    styles = [
        sfnt.get((*mac_key, wws_subfamily), b'').decode('latin-1'),
        sfnt.get((*mac_key, typographic_subfamily), b'').decode('latin-1'),
        sfnt.get((*mac_key, font_subfamily), b'').decode('latin-1'),
        sfnt.get((*ms_key, wws_subfamily), b'').decode('utf-16-be'),
        sfnt.get((*ms_key, typographic_subfamily), b'').decode('utf-16-be'),
        sfnt.get((*ms_key, font_subfamily), b'').decode('utf-16-be'),
    ]
    styles = [*filter(None, styles)] or [font.style_name]

    def get_weight():  # From fontconfig's FcFreeTypeQueryFaceInternal.
        # OS/2 table weight.
        os2 = font.get_sfnt_table("OS/2")
        if os2 and os2["version"] != 0xffff:
            return os2["usWeightClass"]
        # PostScript font info weight.
        try:
            ps_font_info_weight = (
                font.get_ps_font_info()["weight"].replace(" ", "") or "")
        except ValueError:
            pass
        else:
            for regex, weight in _weight_regexes:
                if re.fullmatch(regex, ps_font_info_weight, re.I):
                    return weight
        # Style name weight.
        for style in styles:
            style = style.replace(" ", "")
            for regex, weight in _weight_regexes:
                if re.search(regex, style, re.I):
                    return weight
        if ft2font.StyleFlags.BOLD in font.style_flags:
            return 700  # "bold"
        return 500  # "medium", not "regular"!

    weight = int(get_weight())

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit

    if any(word in sfnt4 for word in ['narrow', 'condensed', 'cond']):
        stretch = 'condensed'
    elif 'demi cond' in sfnt4:
        stretch = 'semi-condensed'
    elif any(word in sfnt4 for word in ['wide', 'expanded', 'extended']):
        stretch = 'expanded'
    else:
        stretch = 'normal'

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g., 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    if not font.scalable:
        raise NotImplementedError("Non-scalable fonts are not supported")
    size = 'scalable'

    return FontEntry(font.fname, name, style, variant, weight, stretch, size)


def afmFontProperty(fontpath, font):
    """
    Extract information from an AFM font file.

    Parameters
    ----------
    fontpath : str
        The filename corresponding to *font*.
    font : AFM
        The AFM font file from which information will be extracted.

    Returns
    -------
    `FontEntry`
        The extracted font properties.
    """

    name = font.get_familyname()
    fontname = font.get_fontname().lower()

    #  Styles are: italic, oblique, and normal (default)

    if font.get_angle() != 0 or 'italic' in name.lower():
        style = 'italic'
    elif 'oblique' in name.lower():
        style = 'oblique'
    else:
        style = 'normal'

    #  Variants are: small-caps and normal (default)

    # !!!!  Untested
    if name.lower() in ['capitals', 'small-caps']:
        variant = 'small-caps'
    else:
        variant = 'normal'

    weight = font.get_weight().lower()
    if weight not in weight_dict:
        weight = 'normal'

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit
    if 'demi cond' in fontname:
        stretch = 'semi-condensed'
    elif any(word in fontname for word in ['narrow', 'cond']):
        stretch = 'condensed'
    elif any(word in fontname for word in ['wide', 'expanded', 'extended']):
        stretch = 'expanded'
    else:
        stretch = 'normal'

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g., 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    #  All AFM fonts are apparently scalable.

    size = 'scalable'

    return FontEntry(fontpath, name, style, variant, weight, stretch, size)


def _cleanup_fontproperties_init(init_method):
    """
    A decorator to limit the call signature to single a positional argument
    or alternatively only keyword arguments.

    We still accept but deprecate all other call signatures.

    When the deprecation expires we can switch the signature to::

        __init__(self, pattern=None, /, *, family=None, style=None, ...)

    plus a runtime check that pattern is not used alongside with the
    keyword arguments. This results eventually in the two possible
    call signatures::

        FontProperties(pattern)
        FontProperties(family=..., size=..., ...)

    """
    @functools.wraps(init_method)
    def wrapper(self, *args, **kwargs):
        # multiple args with at least some positional ones
        if len(args) > 1 or len(args) == 1 and kwargs:
            # Note: Both cases were previously handled as individual properties.
            # Therefore, we do not mention the case of font properties here.
            _api.warn_deprecated(
                "3.10",
                message="Passing individual properties to FontProperties() "
                        "positionally was deprecated in Matplotlib %(since)s and "
                        "will be removed in %(removal)s. Please pass all properties "
                        "via keyword arguments."
            )
        # single non-string arg -> clearly a family not a pattern
        if len(args) == 1 and not kwargs and not cbook.is_scalar_or_string(args[0]):
            # Case font-family list passed as single argument
            _api.warn_deprecated(
                "3.10",
                message="Passing family as positional argument to FontProperties() "
                        "was deprecated in Matplotlib %(since)s and will be removed "
                        "in %(removal)s. Please pass family names as keyword"
                        "argument."
            )
        # Note on single string arg:
        # This has been interpreted as pattern so far. We are already raising if a
        # non-pattern compatible family string was given. Therefore, we do not need
        # to warn for this case.
        return init_method(self, *args, **kwargs)

    return wrapper


class FontProperties:
    """
    A class for storing and manipulating font properties.

    The font properties are the six properties described in the
    `W3C Cascading Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification and *math_fontfamily* for math fonts:

    - family: A list of font names in decreasing order of priority.
      The items may include a generic font family name, either 'sans-serif',
      'serif', 'cursive', 'fantasy', or 'monospace'.  In that case, the actual
      font to be used will be looked up from the associated rcParam during the
      search process in `.findfont`. Default: :rc:`font.family`

    - style: Either 'normal', 'italic' or 'oblique'.
      Default: :rc:`font.style`

    - variant: Either 'normal' or 'small-caps'.
      Default: :rc:`font.variant`

    - stretch: A numeric value in the range 0-1000 or one of
      'ultra-condensed', 'extra-condensed', 'condensed',
      'semi-condensed', 'normal', 'semi-expanded', 'expanded',
      'extra-expanded' or 'ultra-expanded'. Default: :rc:`font.stretch`

    - weight: A numeric value in the range 0-1000 or one of
      'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
      'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
      'extra bold', 'black'. Default: :rc:`font.weight`

    - size: Either a relative value of 'xx-small', 'x-small',
      'small', 'medium', 'large', 'x-large', 'xx-large' or an
      absolute font size, e.g., 10. Default: :rc:`font.size`

    - math_fontfamily: The family of fonts used to render math text.
      Supported values are: 'dejavusans', 'dejavuserif', 'cm',
      'stix', 'stixsans' and 'custom'. Default: :rc:`mathtext.fontset`

    Alternatively, a font may be specified using the absolute path to a font
    file, by using the *fname* kwarg.  However, in this case, it is typically
    simpler to just pass the path (as a `pathlib.Path`, not a `str`) to the
    *font* kwarg of the `.Text` object.

    The preferred usage of font sizes is to use the relative values,
    e.g.,  'large', instead of absolute font sizes, e.g., 12.  This
    approach allows all text sizes to be made larger or smaller based
    on the font manager's default font size.

    This class accepts a single positional string as fontconfig_ pattern_,
    or alternatively individual properties as keyword arguments::

        FontProperties(pattern)
        FontProperties(*, family=None, style=None, variant=None, ...)

    This support does not depend on fontconfig; we are merely borrowing its
    pattern syntax for use here.

    .. _fontconfig: https://www.freedesktop.org/wiki/Software/fontconfig/
    .. _pattern:
       https://www.freedesktop.org/software/fontconfig/fontconfig-user.html

    Note that Matplotlib's internal font manager and fontconfig use a
    different algorithm to lookup fonts, so the results of the same pattern
    may be different in Matplotlib than in other applications that use
    fontconfig.
    """

    @_cleanup_fontproperties_init
    def __init__(self, family=None, style=None, variant=None, weight=None,
                 stretch=None, size=None,
                 fname=None,  # if set, it's a hardcoded filename to use
                 math_fontfamily=None):
        self.set_family(family)
        self.set_style(style)
        self.set_variant(variant)
        self.set_weight(weight)
        self.set_stretch(stretch)
        self.set_file(fname)
        self.set_size(size)
        self.set_math_fontfamily(math_fontfamily)
        # Treat family as a fontconfig pattern if it is the only parameter
        # provided.  Even in that case, call the other setters first to set
        # attributes not specified by the pattern to the rcParams defaults.
        if (isinstance(family, str)
                and style is None and variant is None and weight is None
                and stretch is None and size is None and fname is None):
            self.set_fontconfig_pattern(family)

    @classmethod
    def _from_any(cls, arg):
        """
        Generic constructor which can build a `.FontProperties` from any of the
        following:

        - a `.FontProperties`: it is passed through as is;
        - `None`: a `.FontProperties` using rc values is used;
        - an `os.PathLike`: it is used as path to the font file;
        - a `str`: it is parsed as a fontconfig pattern;
        - a `dict`: it is passed as ``**kwargs`` to `.FontProperties`.
        """
        if arg is None:
            return cls()
        elif isinstance(arg, cls):
            return arg
        elif isinstance(arg, os.PathLike):
            return cls(fname=arg)
        elif isinstance(arg, str):
            return cls(arg)
        else:
            return cls(**arg)

    def __hash__(self):
        l = (tuple(self.get_family()),
             self.get_slant(),
             self.get_variant(),
             self.get_weight(),
             self.get_stretch(),
             self.get_size(),
             self.get_file(),
             self.get_math_fontfamily())
        return hash(l)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return self.get_fontconfig_pattern()

    def get_family(self):
        """
        Return a list of individual font family names or generic family names.

        The font families or generic font families (which will be resolved
        from their respective rcParams when searching for a matching font) in
        the order of preference.
        """
        return self._family

    def get_name(self):
        """
        Return the name of the font that best matches the font properties.
        """
        return get_font(findfont(self)).family_name

    def get_style(self):
        """
        Return the font style.  Values are: 'normal', 'italic' or 'oblique'.
        """
        return self._slant

    def get_variant(self):
        """
        Return the font variant.  Values are: 'normal' or 'small-caps'.
        """
        return self._variant

    def get_weight(self):
        """
        Get the font weight.  Options are: A numeric value in the
        range 0-1000 or one of 'light', 'normal', 'regular', 'book',
        'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',
        'heavy', 'extra bold', 'black'
        """
        return self._weight

    def get_stretch(self):
        """
        Return the font stretch or width.  Options are: 'ultra-condensed',
        'extra-condensed', 'condensed', 'semi-condensed', 'normal',
        'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.
        """
        return self._stretch

    def get_size(self):
        """
        Return the font size.
        """
        return self._size

    def get_file(self):
        """
        Return the filename of the associated font.
        """
        return self._file

    def get_fontconfig_pattern(self):
        """
        Get a fontconfig_ pattern_ suitable for looking up the font as
        specified with fontconfig's ``fc-match`` utility.

        This support does not depend on fontconfig; we are merely borrowing its
        pattern syntax for use here.
        """
        return generate_fontconfig_pattern(self)

    def set_family(self, family):
        """
        Change the font family.  Can be either an alias (generic name
        is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',
        'fantasy', or 'monospace', a real font name or a list of real
        font names.  Real font names are not supported when
        :rc:`text.usetex` is `True`. Default: :rc:`font.family`
        """
        if family is None:
            family = mpl.rcParams['font.family']
        if isinstance(family, str):
            family = [family]
        self._family = family

    def set_style(self, style):
        """
        Set the font style.

        Parameters
        ----------
        style : {'normal', 'italic', 'oblique'}, default: :rc:`font.style`
        """
        if style is None:
            style = mpl.rcParams['font.style']
        _api.check_in_list(['normal', 'italic', 'oblique'], style=style)
        self._slant = style

    def set_variant(self, variant):
        """
        Set the font variant.

        Parameters
        ----------
        variant : {'normal', 'small-caps'}, default: :rc:`font.variant`
        """
        if variant is None:
            variant = mpl.rcParams['font.variant']
        _api.check_in_list(['normal', 'small-caps'], variant=variant)
        self._variant = variant

    def set_weight(self, weight):
        """
        Set the font weight.

        Parameters
        ----------
        weight : int or {'ultralight', 'light', 'normal', 'regular', 'book', \
'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', \
'extra bold', 'black'}, default: :rc:`font.weight`
            If int, must be in the range  0-1000.
        """
        if weight is None:
            weight = mpl.rcParams['font.weight']
        if weight in weight_dict:
            self._weight = weight
            return
        try:
            weight = int(weight)
        except ValueError:
            pass
        else:
            if 0 <= weight <= 1000:
                self._weight = weight
                return
        raise ValueError(f"{weight=} is invalid")

    def set_stretch(self, stretch):
        """
        Set the font stretch or width.

        Parameters
        ----------
        stretch : int or {'ultra-condensed', 'extra-condensed', 'condensed', \
'semi-condensed', 'normal', 'semi-expanded', 'expanded', 'extra-expanded', \
'ultra-expanded'}, default: :rc:`font.stretch`
            If int, must be in the range  0-1000.
        """
        if stretch is None:
            stretch = mpl.rcParams['font.stretch']
        if stretch in stretch_dict:
            self._stretch = stretch
            return
        try:
            stretch = int(stretch)
        except ValueError:
            pass
        else:
            if 0 <= stretch <= 1000:
                self._stretch = stretch
                return
        raise ValueError(f"{stretch=} is invalid")

    def set_size(self, size):
        """
        Set the font size.

        Parameters
        ----------
        size : float or {'xx-small', 'x-small', 'small', 'medium', \
'large', 'x-large', 'xx-large'}, default: :rc:`font.size`
            If a float, the font size in points. The string values denote
            sizes relative to the default font size.
        """
        if size is None:
            size = mpl.rcParams['font.size']
        try:
            size = float(size)
        except ValueError:
            try:
                scale = font_scalings[size]
            except KeyError as err:
                raise ValueError(
                    "Size is invalid. Valid font size are "
                    + ", ".join(map(str, font_scalings))) from err
            else:
                size = scale * FontManager.get_default_size()
        if size < 1.0:
            _log.info('Fontsize %1.2f < 1.0 pt not allowed by FreeType. '
                      'Setting fontsize = 1 pt', size)
            size = 1.0
        self._size = size

    def set_file(self, file):
        """
        Set the filename of the fontfile to use.  In this case, all
        other properties will be ignored.
        """
        self._file = os.fspath(file) if file is not None else None

    def set_fontconfig_pattern(self, pattern):
        """
        Set the properties by parsing a fontconfig_ *pattern*.

        This support does not depend on fontconfig; we are merely borrowing its
        pattern syntax for use here.
        """
        for key, val in parse_fontconfig_pattern(pattern).items():
            if type(val) is list:
                getattr(self, "set_" + key)(val[0])
            else:
                getattr(self, "set_" + key)(val)

    def get_math_fontfamily(self):
        """
        Return the name of the font family used for math text.

        The default font is :rc:`mathtext.fontset`.
        """
        return self._math_fontfamily

    def set_math_fontfamily(self, fontfamily):
        """
        Set the font family for text in math mode.

        If not set explicitly, :rc:`mathtext.fontset` will be used.

        Parameters
        ----------
        fontfamily : str
            The name of the font family.

            Available font families are defined in the
            :ref:`default matplotlibrc file <customizing-with-matplotlibrc-files>`.

        See Also
        --------
        .text.Text.get_math_fontfamily
        """
        if fontfamily is None:
            fontfamily = mpl.rcParams['mathtext.fontset']
        else:
            valid_fonts = _validators['mathtext.fontset'].valid.values()
            # _check_in_list() Validates the parameter math_fontfamily as
            # if it were passed to rcParams['mathtext.fontset']
            _api.check_in_list(valid_fonts, math_fontfamily=fontfamily)
        self._math_fontfamily = fontfamily

    def copy(self):
        """Return a copy of self."""
        return copy.copy(self)

    # Aliases
    set_name = set_family
    get_slant = get_style
    set_slant = set_style
    get_size_in_points = get_size


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, FontManager):
            return dict(o.__dict__, __class__='FontManager')
        elif isinstance(o, FontEntry):
            d = dict(o.__dict__, __class__='FontEntry')
            try:
                # Cache paths of fonts shipped with Matplotlib relative to the
                # Matplotlib data path, which helps in the presence of venvs.
                d["fname"] = str(Path(d["fname"]).relative_to(mpl.get_data_path()))
            except ValueError:
                pass
            return d
        else:
            return super().default(o)


def _json_decode(o):
    cls = o.pop('__class__', None)
    if cls is None:
        return o
    elif cls == 'FontManager':
        r = FontManager.__new__(FontManager)
        r.__dict__.update(o)
        return r
    elif cls == 'FontEntry':
        if not os.path.isabs(o['fname']):
            o['fname'] = os.path.join(mpl.get_data_path(), o['fname'])
        r = FontEntry(**o)
        return r
    else:
        raise ValueError("Don't know how to deserialize __class__=%s" % cls)


def json_dump(data, filename):
    """
    Dump `FontManager` *data* as JSON to the file named *filename*.

    See Also
    --------
    json_load

    Notes
    -----
    File paths that are children of the Matplotlib data path (typically, fonts
    shipped with Matplotlib) are stored relative to that data path (to remain
    valid across virtualenvs).

    This function temporarily locks the output file to prevent multiple
    processes from overwriting one another's output.
    """
    try:
        with cbook._lock_path(filename), open(filename, 'w') as fh:
            json.dump(data, fh, cls=_JSONEncoder, indent=2)
    except OSError as e:
        _log.warning('Could not save font_manager cache %s', e)


def json_load(filename):
    """
    Load a `FontManager` from the JSON file named *filename*.

    See Also
    --------
    json_dump
    """
    with open(filename) as fh:
        return json.load(fh, object_hook=_json_decode)


class FontManager:
    """
    On import, the `FontManager` singleton instance creates a list of ttf and
    afm fonts and caches their `FontProperties`.  The `FontManager.findfont`
    method does a nearest neighbor search to find the font that most closely
    matches the specification.  If no good enough match is found, the default
    font is returned.

    Fonts added with the `FontManager.addfont` method will not persist in the
    cache; therefore, `addfont` will need to be called every time Matplotlib is
    imported. This method should only be used if and when a font cannot be
    installed on your operating system by other means.

    Notes
    -----
    The `FontManager.addfont` method must be called on the global `FontManager`
    instance.

    Example usage::

        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        font_dirs = ["/resources/fonts"]  # The path to the custom font file.
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
    """
    # Increment this version number whenever the font cache data
    # format or behavior has changed and requires an existing font
    # cache files to be rebuilt.
    __version__ = 390

    def __init__(self, size=None, weight='normal'):
        self._version = self.__version__

        self.__default_weight = weight
        self.default_size = size

        # Create list of font paths.
        paths = [cbook._get_data_path('fonts', subdir)
                 for subdir in ['ttf', 'afm', 'pdfcorefonts']]
        _log.debug('font search path %s', paths)

        self.defaultFamily = {
            'ttf': 'DejaVu Sans',
            'afm': 'Helvetica'}

        self.afmlist = []
        self.ttflist = []

        # Delay the warning by 5s.
        timer = threading.Timer(5, lambda: _log.warning(
            'Matplotlib is building the font cache; this may take a moment.'))
        timer.start()
        try:
            for fontext in ["afm", "ttf"]:
                for path in [*findSystemFonts(paths, fontext=fontext),
                             *findSystemFonts(fontext=fontext)]:
                    try:
                        self.addfont(path)
                    except OSError as exc:
                        _log.info("Failed to open font file %s: %s", path, exc)
                    except Exception as exc:
                        _log.info("Failed to extract font properties from %s: "
                                  "%s", path, exc)
        finally:
            timer.cancel()

    def addfont(self, path):
        """
        Cache the properties of the font at *path* to make it available to the
        `FontManager`.  The type of font is inferred from the path suffix.

        Parameters
        ----------
        path : str or path-like

        Notes
        -----
        This method is useful for adding a custom font without installing it in
        your operating system. See the `FontManager` singleton instance for
        usage and caveats about this function.
        """
        # Convert to string in case of a path as
        # afmFontProperty and FT2Font expect this
        path = os.fsdecode(path)
        if Path(path).suffix.lower() == ".afm":
            with open(path, "rb") as fh:
                font = _afm.AFM(fh)
            prop = afmFontProperty(path, font)
            self.afmlist.append(prop)
        else:
            font = ft2font.FT2Font(path)
            prop = ttfFontProperty(font)
            self.ttflist.append(prop)
        self._findfont_cached.cache_clear()

    @property
    def defaultFont(self):
        # Lazily evaluated (findfont then caches the result) to avoid including
        # the venv path in the json serialization.
        return {ext: self.findfont(family, fontext=ext)
                for ext, family in self.defaultFamily.items()}

    def get_default_weight(self):
        """
        Return the default font weight.
        """
        return self.__default_weight

    @staticmethod
    def get_default_size():
        """
        Return the default font size.
        """
        return mpl.rcParams['font.size']

    def set_default_weight(self, weight):
        """
        Set the default font weight.  The initial value is 'normal'.
        """
        self.__default_weight = weight

    @staticmethod
    def _expand_aliases(family):
        if family in ('sans', 'sans serif'):
            family = 'sans-serif'
        return mpl.rcParams['font.' + family]

    # Each of the scoring functions below should return a value between
    # 0.0 (perfect match) and 1.0 (terrible match)
    def score_family(self, families, family2):
        """
        Return a match score between the list of font families in
        *families* and the font family name *family2*.

        An exact match at the head of the list returns 0.0.

        A match further down the list will return between 0 and 1.

        No match will return 1.0.
        """
        if not isinstance(families, (list, tuple)):
            families = [families]
        elif len(families) == 0:
            return 1.0
        family2 = family2.lower()
        step = 1 / len(families)
        for i, family1 in enumerate(families):
            family1 = family1.lower()
            if family1 in font_family_aliases:
                options = [*map(str.lower, self._expand_aliases(family1))]
                if family2 in options:
                    idx = options.index(family2)
                    return (i + (idx / len(options))) * step
            elif family1 == family2:
                # The score should be weighted by where in the
                # list the font was found.
                return i * step
        return 1.0

    def score_style(self, style1, style2):
        """
        Return a match score between *style1* and *style2*.

        An exact match returns 0.0.

        A match between 'italic' and 'oblique' returns 0.1.

        No match returns 1.0.
        """
        if style1 == style2:
            return 0.0
        elif (style1 in ('italic', 'oblique')
              and style2 in ('italic', 'oblique')):
            return 0.1
        return 1.0

    def score_variant(self, variant1, variant2):
        """
        Return a match score between *variant1* and *variant2*.

        An exact match returns 0.0, otherwise 1.0.
        """
        if variant1 == variant2:
            return 0.0
        else:
            return 1.0

    def score_stretch(self, stretch1, stretch2):
        """
        Return a match score between *stretch1* and *stretch2*.

        The result is the absolute value of the difference between the
        CSS numeric values of *stretch1* and *stretch2*, normalized
        between 0.0 and 1.0.
        """
        try:
            stretchval1 = int(stretch1)
        except ValueError:
            stretchval1 = stretch_dict.get(stretch1, 500)
        try:
            stretchval2 = int(stretch2)
        except ValueError:
            stretchval2 = stretch_dict.get(stretch2, 500)
        return abs(stretchval1 - stretchval2) / 1000.0

    def score_weight(self, weight1, weight2):
        """
        Return a match score between *weight1* and *weight2*.

        The result is 0.0 if both weight1 and weight 2 are given as strings
        and have the same value.

        Otherwise, the result is the absolute value of the difference between
        the CSS numeric values of *weight1* and *weight2*, normalized between
        0.05 and 1.0.
        """
        # exact match of the weight names, e.g. weight1 == weight2 == "regular"
        if cbook._str_equal(weight1, weight2):
            return 0.0
        w1 = weight1 if isinstance(weight1, Number) else weight_dict[weight1]
        w2 = weight2 if isinstance(weight2, Number) else weight_dict[weight2]
        return 0.95 * (abs(w1 - w2) / 1000) + 0.05

    def score_size(self, size1, size2):
        """
        Return a match score between *size1* and *size2*.

        If *size2* (the size specified in the font file) is 'scalable', this
        function always returns 0.0, since any font size can be generated.

        Otherwise, the result is the absolute distance between *size1* and
        *size2*, normalized so that the usual range of font sizes (6pt -
        72pt) will lie between 0.0 and 1.0.
        """
        if size2 == 'scalable':
            return 0.0
        # Size value should have already been
        try:
            sizeval1 = float(size1)
        except ValueError:
            sizeval1 = self.default_size * font_scalings[size1]
        try:
            sizeval2 = float(size2)
        except ValueError:
            return 1.0
        return abs(sizeval1 - sizeval2) / 72

    def findfont(self, prop, fontext='ttf', directory=None,
                 fallback_to_default=True, rebuild_if_missing=True):
        """
        Find the path to the font file most closely matching the given font properties.

        Parameters
        ----------
        prop : str or `~matplotlib.font_manager.FontProperties`
            The font properties to search for. This can be either a
            `.FontProperties` object or a string defining a
            `fontconfig patterns`_.

        fontext : {'ttf', 'afm'}, default: 'ttf'
            The extension of the font file:

            - 'ttf': TrueType and OpenType fonts (.ttf, .ttc, .otf)
            - 'afm': Adobe Font Metrics (.afm)

        directory : str, optional
            If given, only search this directory and its subdirectories.

        fallback_to_default : bool
            If True, will fall back to the default font family (usually
            "DejaVu Sans" or "Helvetica") if the first lookup hard-fails.

        rebuild_if_missing : bool
            Whether to rebuild the font cache and search again if the first
            match appears to point to a nonexisting font (i.e., the font cache
            contains outdated entries).

        Returns
        -------
        str
            The filename of the best matching font.

        Notes
        -----
        This performs a nearest neighbor search.  Each font is given a
        similarity score to the target font properties.  The first font with
        the highest score is returned.  If no matches below a certain
        threshold are found, the default font (usually DejaVu Sans) is
        returned.

        The result is cached, so subsequent lookups don't have to
        perform the O(n) nearest neighbor search.

        See the `W3C Cascading Style Sheet, Level 1
        <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation
        for a description of the font finding algorithm.

        .. _fontconfig patterns:
           https://www.freedesktop.org/software/fontconfig/fontconfig-user.html
        """
        # Pass the relevant rcParams (and the font manager, as `self`) to
        # _findfont_cached so to prevent using a stale cache entry after an
        # rcParam was changed.
        rc_params = tuple(tuple(mpl.rcParams[key]) for key in [
            "font.serif", "font.sans-serif", "font.cursive", "font.fantasy",
            "font.monospace"])
        ret = self._findfont_cached(
            prop, fontext, directory, fallback_to_default, rebuild_if_missing,
            rc_params)
        if isinstance(ret, cbook._ExceptionInfo):
            raise ret.to_exception()
        return ret

    def get_font_names(self):
        """Return the list of available fonts."""
        return list({font.name for font in self.ttflist})

    def _find_fonts_by_props(self, prop, fontext='ttf', directory=None,
                             fallback_to_default=True, rebuild_if_missing=True):
        """
        Find the paths to the font files most closely matching the given properties.

        Parameters
        ----------
        prop : str or `~matplotlib.font_manager.FontProperties`
            The font properties to search for. This can be either a
            `.FontProperties` object or a string defining a
            `fontconfig patterns`_.

        fontext : {'ttf', 'afm'}, default: 'ttf'
            The extension of the font file:

            - 'ttf': TrueType and OpenType fonts (.ttf, .ttc, .otf)
            - 'afm': Adobe Font Metrics (.afm)

        directory : str, optional
            If given, only search this directory and its subdirectories.

        fallback_to_default : bool
            If True, will fall back to the default font family (usually
            "DejaVu Sans" or "Helvetica") if none of the families were found.

        rebuild_if_missing : bool
            Whether to rebuild the font cache and search again if the first
            match appears to point to a nonexisting font (i.e., the font cache
            contains outdated entries).

        Returns
        -------
        list[str]
            The paths of the fonts found.

        Notes
        -----
        This is an extension/wrapper of the original findfont API, which only
        returns a single font for given font properties. Instead, this API
        returns a list of filepaths of multiple fonts which closely match the
        given font properties.  Since this internally uses the original API,
        there's no change to the logic of performing the nearest neighbor
        search.  See `findfont` for more details.
        """

        prop = FontProperties._from_any(prop)

        fpaths = []
        for family in prop.get_family():
            cprop = prop.copy()
            cprop.set_family(family)  # set current prop's family

            try:
                fpaths.append(
                    self.findfont(
                        cprop, fontext, directory,
                        fallback_to_default=False,  # don't fallback to default
                        rebuild_if_missing=rebuild_if_missing,
                    )
                )
            except ValueError:
                if family in font_family_aliases:
                    _log.warning(
                        "findfont: Generic family %r not found because "
                        "none of the following families were found: %s",
                        family, ", ".join(self._expand_aliases(family))
                    )
                else:
                    _log.warning("findfont: Font family %r not found.", family)

        # only add default family if no other font was found and
        # fallback_to_default is enabled
        if not fpaths:
            if fallback_to_default:
                dfamily = self.defaultFamily[fontext]
                cprop = prop.copy()
                cprop.set_family(dfamily)
                fpaths.append(
                    self.findfont(
                        cprop, fontext, directory,
                        fallback_to_default=True,
                        rebuild_if_missing=rebuild_if_missing,
                    )
                )
            else:
                raise ValueError("Failed to find any font, and fallback "
                                 "to the default font was disabled")

        return fpaths

    @lru_cache(1024)
    def _findfont_cached(self, prop, fontext, directory, fallback_to_default,
                         rebuild_if_missing, rc_params):

        prop = FontProperties._from_any(prop)

        fname = prop.get_file()
        if fname is not None:
            return fname

        if fontext == 'afm':
            fontlist = self.afmlist
        else:
            fontlist = self.ttflist

        best_score = 1e64
        best_font = None

        _log.debug('findfont: Matching %s.', prop)
        for font in fontlist:
            if (directory is not None and
                    Path(directory) not in Path(font.fname).parents):
                continue
            # Matching family should have top priority, so multiply it by 10.
            score = (self.score_family(prop.get_family(), font.name) * 10
                     + self.score_style(prop.get_style(), font.style)
                     + self.score_variant(prop.get_variant(), font.variant)
                     + self.score_weight(prop.get_weight(), font.weight)
                     + self.score_stretch(prop.get_stretch(), font.stretch)
                     + self.score_size(prop.get_size(), font.size))
            _log.debug('findfont: score(%s) = %s', font, score)
            if score < best_score:
                best_score = score
                best_font = font
            if score == 0:
                break

        if best_font is None or best_score >= 10.0:
            if fallback_to_default:
                _log.warning(
                    'findfont: Font family %s not found. Falling back to %s.',
                    prop.get_family(), self.defaultFamily[fontext])
                for family in map(str.lower, prop.get_family()):
                    if family in font_family_aliases:
                        _log.warning(
                            "findfont: Generic family %r not found because "
                            "none of the following families were found: %s",
                            family, ", ".join(self._expand_aliases(family)))
                default_prop = prop.copy()
                default_prop.set_family(self.defaultFamily[fontext])
                return self.findfont(default_prop, fontext, directory,
                                     fallback_to_default=False)
            else:
                # This return instead of raise is intentional, as we wish to
                # cache that it was not found, which will not occur if it was
                # actually raised.
                return cbook._ExceptionInfo(
                    ValueError,
                    f"Failed to find font {prop}, and fallback to the default font was "
                    f"disabled"
                )
        else:
            _log.debug('findfont: Matching %s to %s (%r) with score of %f.',
                       prop, best_font.name, best_font.fname, best_score)
            result = best_font.fname

        if not os.path.isfile(result):
            if rebuild_if_missing:
                _log.info(
                    'findfont: Found a missing font file.  Rebuilding cache.')
                new_fm = _load_fontmanager(try_read_cache=False)
                # Replace self by the new fontmanager, because users may have
                # a reference to this specific instance.
                # TODO: _load_fontmanager should really be (used by) a method
                # modifying the instance in place.
                vars(self).update(vars(new_fm))
                return self.findfont(
                    prop, fontext, directory, rebuild_if_missing=False)
            else:
                # This return instead of raise is intentional, as we wish to
                # cache that it was not found, which will not occur if it was
                # actually raised.
                return cbook._ExceptionInfo(ValueError, "No valid font could be found")

        return _cached_realpath(result)


@lru_cache
def is_opentype_cff_font(filename):
    """
    Return whether the given font is a Postscript Compact Font Format Font
    embedded in an OpenType wrapper.  Used by the PostScript and PDF backends
    that cannot subset these fonts.
    """
    if os.path.splitext(filename)[1].lower() == '.otf':
        with open(filename, 'rb') as fd:
            return fd.read(4) == b"OTTO"
    else:
        return False


@lru_cache(64)
def _get_font(font_filepaths, hinting_factor, *, _kerning_factor, thread_id):
    first_fontpath, *rest = font_filepaths
    return ft2font.FT2Font(
        first_fontpath, hinting_factor,
        _fallback_list=[
            ft2font.FT2Font(
                fpath, hinting_factor,
                _kerning_factor=_kerning_factor
            )
            for fpath in rest
        ],
        _kerning_factor=_kerning_factor
    )


# FT2Font objects cannot be used across fork()s because they reference the same
# FT_Library object.  While invalidating *all* existing FT2Fonts after a fork
# would be too complicated to be worth it, the main way FT2Fonts get reused is
# via the cache of _get_font, which we can empty upon forking (not on Windows,
# which has no fork() or register_at_fork()).
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_get_font.cache_clear)


@lru_cache(64)
def _cached_realpath(path):
    # Resolving the path avoids embedding the font twice in pdf/ps output if a
    # single font is selected using two different relative paths.
    return os.path.realpath(path)


def get_font(font_filepaths, hinting_factor=None):
    """
    Get an `.ft2font.FT2Font` object given a list of file paths.

    Parameters
    ----------
    font_filepaths : Iterable[str, Path, bytes], str, Path, bytes
        Relative or absolute paths to the font files to be used.

        If a single string, bytes, or `pathlib.Path`, then it will be treated
        as a list with that entry only.

        If more than one filepath is passed, then the returned FT2Font object
        will fall back through the fonts, in the order given, to find a needed
        glyph.

    Returns
    -------
    `.ft2font.FT2Font`

    """
    if isinstance(font_filepaths, (str, Path, bytes)):
        paths = (_cached_realpath(font_filepaths),)
    else:
        paths = tuple(_cached_realpath(fname) for fname in font_filepaths)

    if hinting_factor is None:
        hinting_factor = mpl.rcParams['text.hinting_factor']

    return _get_font(
        # must be a tuple to be cached
        paths,
        hinting_factor,
        _kerning_factor=mpl.rcParams['text.kerning_factor'],
        # also key on the thread ID to prevent segfaults with multi-threading
        thread_id=threading.get_ident()
    )


def _load_fontmanager(*, try_read_cache=True):
    fm_path = Path(
        mpl.get_cachedir(), f"fontlist-v{FontManager.__version__}.json")
    if try_read_cache:
        try:
            fm = json_load(fm_path)
        except Exception:
            pass
        else:
            if getattr(fm, "_version", object()) == FontManager.__version__:
                _log.debug("Using fontManager instance from %s", fm_path)
                return fm
    fm = FontManager()
    json_dump(fm, fm_path)
    _log.info("generated new fontManager")
    return fm


fontManager = _load_fontmanager()
findfont = fontManager.findfont
get_font_names = fontManager.get_font_names
<|endoftext|>"""
Routines to adjust subplot params so that subplots are
nicely fit in the figure. In doing so, only axis labels, tick labels, Axes
titles and offsetboxes that are anchored to Axes are currently considered.

Internally, this module assumes that the margins (left margin, etc.) which are
differences between ``Axes.get_tightbbox`` and ``Axes.bbox`` are independent of
Axes position. This may fail if ``Axes.adjustable`` is ``datalim`` as well as
such cases as when left or right margin are affected by xlabel.
"""

import numpy as np

import matplotlib as mpl
from matplotlib import _api, artist as martist
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox


def _auto_adjust_subplotpars(
        fig, renderer, shape, span_pairs, subplot_list,
        ax_bbox_list=None, pad=1.08, h_pad=None, w_pad=None, rect=None):
    """
    Return a dict of subplot parameters to adjust spacing between subplots
    or ``None`` if resulting Axes would have zero height or width.

    Note that this function ignores geometry information of subplot itself, but
    uses what is given by the *shape* and *subplot_list* parameters.  Also, the
    results could be incorrect if some subplots have ``adjustable=datalim``.

    Parameters
    ----------
    shape : tuple[int, int]
        Number of rows and columns of the grid.
    span_pairs : list[tuple[slice, slice]]
        List of rowspans and colspans occupied by each subplot.
    subplot_list : list of subplots
        List of subplots that will be used to calculate optimal subplot_params.
    pad : float
        Padding between the figure edge and the edges of subplots, as a
        fraction of the font size.
    h_pad, w_pad : float
        Padding (height/width) between edges of adjacent subplots, as a
        fraction of the font size.  Defaults to *pad*.
    rect : tuple
        (left, bottom, right, top), default: None.
    """
    rows, cols = shape

    font_size_inch = (FontProperties(
        size=mpl.rcParams["font.size"]).get_size_in_points() / 72)
    pad_inch = pad * font_size_inch
    vpad_inch = h_pad * font_size_inch if h_pad is not None else pad_inch
    hpad_inch = w_pad * font_size_inch if w_pad is not None else pad_inch

    if len(span_pairs) != len(subplot_list) or len(subplot_list) == 0:
        raise ValueError

    if rect is None:
        margin_left = margin_bottom = margin_right = margin_top = None
    else:
        margin_left, margin_bottom, _right, _top = rect
        margin_right = 1 - _right if _right else None
        margin_top = 1 - _top if _top else None

    vspaces = np.zeros((rows + 1, cols))
    hspaces = np.zeros((rows, cols + 1))

    if ax_bbox_list is None:
        ax_bbox_list = [
            Bbox.union([ax.get_position(original=True) for ax in subplots])
            for subplots in subplot_list]

    for subplots, ax_bbox, (rowspan, colspan) in zip(
            subplot_list, ax_bbox_list, span_pairs):
        if all(not ax.get_visible() for ax in subplots):
            continue

        bb = []
        for ax in subplots:
            if ax.get_visible():
                bb += [martist._get_tightbbox_for_layout_only(ax, renderer)]

        tight_bbox_raw = Bbox.union(bb)
        tight_bbox = fig.transFigure.inverted().transform_bbox(tight_bbox_raw)

        hspaces[rowspan, colspan.start] += ax_bbox.xmin - tight_bbox.xmin  # l
        hspaces[rowspan, colspan.stop] += tight_bbox.xmax - ax_bbox.xmax  # r
        vspaces[rowspan.start, colspan] += tight_bbox.ymax - ax_bbox.ymax  # t
        vspaces[rowspan.stop, colspan] += ax_bbox.ymin - tight_bbox.ymin  # b

    fig_width_inch, fig_height_inch = fig.get_size_inches()

    # margins can be negative for Axes with aspect applied, so use max(, 0) to
    # make them nonnegative.
    if not margin_left:
        margin_left = max(hspaces[:, 0].max(), 0) + pad_inch/fig_width_inch
        suplabel = fig._supylabel
        if suplabel and suplabel.get_in_layout():
            rel_width = fig.transFigure.inverted().transform_bbox(
                suplabel.get_window_extent(renderer)).width
            margin_left += rel_width + pad_inch/fig_width_inch
    if not margin_right:
        margin_right = max(hspaces[:, -1].max(), 0) + pad_inch/fig_width_inch
    if not margin_top:
        margin_top = max(vspaces[0, :].max(), 0) + pad_inch/fig_height_inch
        if fig._suptitle and fig._suptitle.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(
                fig._suptitle.get_window_extent(renderer)).height
            margin_top += rel_height + pad_inch/fig_height_inch
    if not margin_bottom:
        margin_bottom = max(vspaces[-1, :].max(), 0) + pad_inch/fig_height_inch
        suplabel = fig._supxlabel
        if suplabel and suplabel.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(
                suplabel.get_window_extent(renderer)).height
            margin_bottom += rel_height + pad_inch/fig_height_inch

    if margin_left + margin_right >= 1:
        _api.warn_external('Tight layout not applied. The left and right '
                           'margins cannot be made large enough to '
                           'accommodate all Axes decorations.')
        return None
    if margin_bottom + margin_top >= 1:
        _api.warn_external('Tight layout not applied. The bottom and top '
                           'margins cannot be made large enough to '
                           'accommodate all Axes decorations.')
        return None

    kwargs = dict(left=margin_left,
                  right=1 - margin_right,
                  bottom=margin_bottom,
                  top=1 - margin_top)

    if cols > 1:
        hspace = hspaces[:, 1:-1].max() + hpad_inch / fig_width_inch
        # axes widths:
        h_axes = (1 - margin_right - margin_left - hspace * (cols - 1)) / cols
        if h_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout '
                               'cannot make Axes width small enough to '
                               'accommodate all Axes decorations')
            return None
        else:
            kwargs["wspace"] = hspace / h_axes
    if rows > 1:
        vspace = vspaces[1:-1, :].max() + vpad_inch / fig_height_inch
        v_axes = (1 - margin_top - margin_bottom - vspace * (rows - 1)) / rows
        if v_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout '
                               'cannot make Axes height small enough to '
                               'accommodate all Axes decorations.')
            return None
        else:
            kwargs["hspace"] = vspace / v_axes

    return kwargs


def get_subplotspec_list(axes_list, grid_spec=None):
    """
    Return a list of subplotspec from the given list of Axes.

    For an instance of Axes that does not support subplotspec, None is inserted
    in the list.

    If grid_spec is given, None is inserted for those not from the given
    grid_spec.
    """
    subplotspec_list = []
    for ax in axes_list:
        axes_or_locator = ax.get_axes_locator()
        if axes_or_locator is None:
            axes_or_locator = ax

        if hasattr(axes_or_locator, "get_subplotspec"):
            subplotspec = axes_or_locator.get_subplotspec()
            if subplotspec is not None:
                subplotspec = subplotspec.get_topmost_subplotspec()
                gs = subplotspec.get_gridspec()
                if grid_spec is not None:
                    if gs != grid_spec:
                        subplotspec = None
                elif gs.locally_modified_subplot_params():
                    subplotspec = None
        else:
            subplotspec = None

        subplotspec_list.append(subplotspec)

    return subplotspec_list


def get_tight_layout_figure(fig, axes_list, subplotspec_list, renderer,
                            pad=1.08, h_pad=None, w_pad=None, rect=None):
    """
    Return subplot parameters for tight-layouted-figure with specified padding.

    Parameters
    ----------
    fig : Figure
    axes_list : list of Axes
    subplotspec_list : list of `.SubplotSpec`
        The subplotspecs of each Axes.
    renderer : renderer
    pad : float
        Padding between the figure edge and the edges of subplots, as a
        fraction of the font size.
    h_pad, w_pad : float
        Padding (height/width) between edges of adjacent subplots.  Defaults to
        *pad*.
    rect : tuple (left, bottom, right, top), default: None.
        rectangle in normalized figure coordinates
        that the whole subplots area (including labels) will fit into.
        Defaults to using the entire figure.

    Returns
    -------
    subplotspec or None
        subplotspec kwargs to be passed to `.Figure.subplots_adjust` or
        None if tight_layout could not be accomplished.
    """

    # Multiple Axes can share same subplotspec (e.g., if using axes_grid1);
    # we need to group them together.
    ss_to_subplots = {ss: [] for ss in subplotspec_list}
    for ax, ss in zip(axes_list, subplotspec_list):
        ss_to_subplots[ss].append(ax)
    if ss_to_subplots.pop(None, None):
        _api.warn_external(
            "This figure includes Axes that are not compatible with "
            "tight_layout, so results might be incorrect.")
    if not ss_to_subplots:
        return {}
    subplot_list = list(ss_to_subplots.values())
    ax_bbox_list = [ss.get_position(fig) for ss in ss_to_subplots]

    max_nrows = max(ss.get_gridspec().nrows for ss in ss_to_subplots)
    max_ncols = max(ss.get_gridspec().ncols for ss in ss_to_subplots)

    span_pairs = []
    for ss in ss_to_subplots:
        # The intent here is to support Axes from different gridspecs where
        # one's nrows (or ncols) is a multiple of the other (e.g. 2 and 4),
        # but this doesn't actually work because the computed wspace, in
        # relative-axes-height, corresponds to different physical spacings for
        # the 2-row grid and the 4-row grid.  Still, this code is left, mostly
        # for backcompat.
        rows, cols = ss.get_gridspec().get_geometry()
        div_row, mod_row = divmod(max_nrows, rows)
        div_col, mod_col = divmod(max_ncols, cols)
        if mod_row != 0:
            _api.warn_external('tight_layout not applied: number of rows '
                               'in subplot specifications must be '
                               'multiples of one another.')
            return {}
        if mod_col != 0:
            _api.warn_external('tight_layout not applied: number of '
                               'columns in subplot specifications must be '
                               'multiples of one another.')
            return {}
        span_pairs.append((
            slice(ss.rowspan.start * div_row, ss.rowspan.stop * div_row),
            slice(ss.colspan.start * div_col, ss.colspan.stop * div_col)))

    kwargs = _auto_adjust_subplotpars(fig, renderer,
                                      shape=(max_nrows, max_ncols),
                                      span_pairs=span_pairs,
                                      subplot_list=subplot_list,
                                      ax_bbox_list=ax_bbox_list,
                                      pad=pad, h_pad=h_pad, w_pad=w_pad)

    # kwargs can be none if tight_layout fails...
    if rect is not None and kwargs is not None:
        # if rect is given, the whole subplots area (including
        # labels) will fit into the rect instead of the
        # figure. Note that the rect argument of
        # *auto_adjust_subplotpars* specify the area that will be
        # covered by the total area of axes.bbox. Thus we call
        # auto_adjust_subplotpars twice, where the second run
        # with adjusted rect parameters.

        left, bottom, right, top = rect
        if left is not None:
            left += kwargs["left"]
        if bottom is not None:
            bottom += kwargs["bottom"]
        if right is not None:
            right -= (1 - kwargs["right"])
        if top is not None:
            top -= (1 - kwargs["top"])

        kwargs = _auto_adjust_subplotpars(fig, renderer,
                                          shape=(max_nrows, max_ncols),
                                          span_pairs=span_pairs,
                                          subplot_list=subplot_list,
                                          ax_bbox_list=ax_bbox_list,
                                          pad=pad, h_pad=h_pad, w_pad=w_pad,
                                          rect=(left, bottom, right, top))

    return kwargs
<|endoftext|>"""
The Colorizer class which handles the data to color pipeline via a
normalization and a colormap.

.. admonition:: Provisional status of colorizer

    The ``colorizer`` module and classes in this file are considered
    provisional and may change at any time without a deprecation period.

.. seealso::

  :doc:`/gallery/color/colormap_reference` for a list of builtin colormaps.

  :ref:`colormap-manipulation` for examples of how to make colormaps.

  :ref:`colormaps` for an in-depth discussion of choosing colormaps.

  :ref:`colormapnorms` for more details about data normalization.

"""

import functools

import numpy as np
from numpy import ma

from matplotlib import _api, colors, cbook, scale, artist
import matplotlib as mpl

mpl._docstring.interpd.register(
    colorizer_doc="""\
colorizer : `~matplotlib.colorizer.Colorizer` or None, default: None
    The Colorizer object used to map color to data. If None, a Colorizer
    object is created from a *norm* and *cmap*.""",
    )


class Colorizer:
    """
    Data to color pipeline.

    This pipeline is accessible via `.Colorizer.to_rgba` and executed via
    the `.Colorizer.norm` and `.Colorizer.cmap` attributes.

    Parameters
    ----------
    cmap: colorbar.Colorbar or str or None, default: None
        The colormap used to color data.

    norm: colors.Normalize or str or None, default: None
        The normalization used to normalize the data
    """
    def __init__(self, cmap=None, norm=None):

        self._cmap = None
        self._set_cmap(cmap)

        self._id_norm = None
        self._norm = None
        self.norm = norm

        self.callbacks = cbook.CallbackRegistry(signals=["changed"])
        self.colorbar = None

    def _scale_norm(self, norm, vmin, vmax, A):
        """
        Helper for initial scaling.

        Used by public functions that create a ScalarMappable and support
        parameters *vmin*, *vmax* and *norm*. This makes sure that a *norm*
        will take precedence over *vmin*, *vmax*.

        Note that this method does not set the norm.
        """
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)
            if isinstance(norm, colors.Normalize):
                raise ValueError(
                    "Passing a Normalize instance simultaneously with "
                    "vmin/vmax is not supported.  Please pass vmin/vmax "
                    "directly to the norm when creating it.")

        # always resolve the autoscaling so we have concrete limits
        # rather than deferring to draw time.
        self.autoscale_None(A)

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm):
        _api.check_isinstance((colors.Normalize, str, None), norm=norm)
        if norm is None:
            norm = colors.Normalize()
        elif isinstance(norm, str):
            try:
                scale_cls = scale._scale_mapping[norm]
            except KeyError:
                raise ValueError(
                    "Invalid norm str name; the following values are "
                    f"supported: {', '.join(scale._scale_mapping)}"
                ) from None
            norm = _auto_norm_from_scale(scale_cls)()

        if norm is self.norm:
            # We aren't updating anything
            return

        in_init = self.norm is None
        # Remove the current callback and connect to the new one
        if not in_init:
            self.norm.callbacks.disconnect(self._id_norm)
        self._norm = norm
        self._id_norm = self.norm.callbacks.connect('changed',
                                                    self.changed)
        if not in_init:
            self.changed()

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this Colorizer.

        There is one special case, for handling images that are already
        RGB or RGBA, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an RGB or RGBA array, and no mapping will be done.
        The array can be `~numpy.uint8`, or it can be floats with
        values in the 0-1 range; otherwise a ValueError will be raised.
        Any NaNs or masked elements will be set to 0 alpha.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the RGBA
        array will be floats in the 0-1 range; if it is *True*,
        the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        # First check for special case, image input:
        if isinstance(x, np.ndarray) and x.ndim == 3:
            return self._pass_image_data(x, alpha, bytes, norm)

        # Otherwise run norm -> colormap pipeline
        x = ma.asarray(x)
        if norm:
            x = self.norm(x)
        rgba = self.cmap(x, alpha=alpha, bytes=bytes)
        return rgba

    @staticmethod
    def _pass_image_data(x, alpha=None, bytes=False, norm=True):
        """
        Helper function to pass ndarray of shape (...,3) or (..., 4)
        through `to_rgba()`, see `to_rgba()` for docstring.
        """
        if x.shape[2] == 3:
            if alpha is None:
                alpha = 1
            if x.dtype == np.uint8:
                alpha = np.uint8(alpha * 255)
            m, n = x.shape[:2]
            xx = np.empty(shape=(m, n, 4), dtype=x.dtype)
            xx[:, :, :3] = x
            xx[:, :, 3] = alpha
        elif x.shape[2] == 4:
            xx = x
        else:
            raise ValueError("Third dimension must be 3 or 4")
        if xx.dtype.kind == 'f':
            # If any of R, G, B, or A is nan, set to 0
            if np.any(nans := np.isnan(x)):
                if x.shape[2] == 4:
                    xx = xx.copy()
                xx[np.any(nans, axis=2), :] = 0

            if norm and (xx.max() > 1 or xx.min() < 0):
                raise ValueError("Floating point image RGB values "
                                 "must be in the 0..1 range.")
            if bytes:
                xx = (xx * 255).astype(np.uint8)
        elif xx.dtype == np.uint8:
            if not bytes:
                xx = xx.astype(np.float32) / 255
        else:
            raise ValueError("Image RGB array must be uint8 or "
                             "floating point; found %s" % xx.dtype)
        # Account for any masked entries in the original array
        # If any of R, G, B, or A are masked for an entry, we set alpha to 0
        if np.ma.is_masked(x):
            xx[np.any(np.ma.getmaskarray(x), axis=2), 3] = 0
        return xx

    def autoscale(self, A):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        if A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale(A)

    def autoscale_None(self, A):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        if A is None:
            raise TypeError('You must first set_array for mappable')
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self.norm.autoscale_None(A)

    def _set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        # bury import to avoid circular imports
        from matplotlib import cm
        in_init = self._cmap is None
        self._cmap = cm._ensure_cmap(cmap)
        if not in_init:
            self.changed()  # Things are not set up properly yet.

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._set_cmap(cmap)

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             The limits may also be passed as a tuple (*vmin*, *vmax*) as a
             single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm, this causes an inconsistent
        # state, to prevent this blocked context manager is used
        if vmax is None:
            try:
                vmin, vmax = vmin
            except (TypeError, ValueError):
                pass

        orig_vmin_vmax = self.norm.vmin, self.norm.vmax

        # Blocked context manager prevents callbacks from being triggered
        # until both vmin and vmax are updated
        with self.norm.callbacks.blocked(signal='changed'):
            if vmin is not None:
                self.norm.vmin = colors._sanitize_extrema(vmin)
            if vmax is not None:
                self.norm.vmax = colors._sanitize_extrema(vmax)

        # emit a update signal if the limits are changed
        if orig_vmin_vmax != (self.norm.vmin, self.norm.vmax):
            self.norm.callbacks.process('changed')

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        return self.norm.vmin, self.norm.vmax

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        self.callbacks.process('changed')
        self.stale = True

    @property
    def vmin(self):
        return self.get_clim()[0]

    @vmin.setter
    def vmin(self, vmin):
        self.set_clim(vmin=vmin)

    @property
    def vmax(self):
        return self.get_clim()[1]

    @vmax.setter
    def vmax(self, vmax):
        self.set_clim(vmax=vmax)

    @property
    def clip(self):
        return self.norm.clip

    @clip.setter
    def clip(self, clip):
        self.norm.clip = clip


class _ColorizerInterface:
    """
    Base class that contains the interface to `Colorizer` objects from
    a `ColorizingArtist` or `.cm.ScalarMappable`.

    Note: This class only contain functions that interface the .colorizer
    attribute. Other functions that as shared between `.ColorizingArtist`
    and `.cm.ScalarMappable` are not included.
    """
    def _scale_norm(self, norm, vmin, vmax):
        self._colorizer._scale_norm(norm, vmin, vmax, self._A)

    def to_rgba(self, x, alpha=None, bytes=False, norm=True):
        """
        Return a normalized RGBA array corresponding to *x*.

        In the normal case, *x* is a 1D or 2D sequence of scalars, and
        the corresponding `~numpy.ndarray` of RGBA values will be returned,
        based on the norm and colormap set for this Colorizer.

        There is one special case, for handling images that are already
        RGB or RGBA, such as might have been read from an image file.
        If *x* is an `~numpy.ndarray` with 3 dimensions,
        and the last dimension is either 3 or 4, then it will be
        treated as an RGB or RGBA array, and no mapping will be done.
        The array can be `~numpy.uint8`, or it can be floats with
        values in the 0-1 range; otherwise a ValueError will be raised.
        Any NaNs or masked elements will be set to 0 alpha.
        If the last dimension is 3, the *alpha* kwarg (defaulting to 1)
        will be used to fill in the transparency.  If the last dimension
        is 4, the *alpha* kwarg is ignored; it does not
        replace the preexisting alpha.  A ValueError will be raised
        if the third dimension is other than 3 or 4.

        In either case, if *bytes* is *False* (default), the RGBA
        array will be floats in the 0-1 range; if it is *True*,
        the returned RGBA array will be `~numpy.uint8` in the 0 to 255 range.

        If norm is False, no normalization of the input data is
        performed, and it is assumed to be in the range (0-1).

        """
        return self._colorizer.to_rgba(x, alpha=alpha, bytes=bytes, norm=norm)

    def get_clim(self):
        """
        Return the values (min, max) that are mapped to the colormap limits.
        """
        return self._colorizer.get_clim()

    def set_clim(self, vmin=None, vmax=None):
        """
        Set the norm limits for image scaling.

        Parameters
        ----------
        vmin, vmax : float
             The limits.

             For scalar data, the limits may also be passed as a
             tuple (*vmin*, *vmax*) as a single positional argument.

             .. ACCEPTS: (vmin: float, vmax: float)
        """
        # If the norm's limits are updated self.changed() will be called
        # through the callbacks attached to the norm
        self._colorizer.set_clim(vmin, vmax)

    def get_alpha(self):
        try:
            return super().get_alpha()
        except AttributeError:
            return 1

    @property
    def cmap(self):
        return self._colorizer.cmap

    @cmap.setter
    def cmap(self, cmap):
        self._colorizer.cmap = cmap

    def get_cmap(self):
        """Return the `.Colormap` instance."""
        return self._colorizer.cmap

    def set_cmap(self, cmap):
        """
        Set the colormap for luminance data.

        Parameters
        ----------
        cmap : `.Colormap` or str or None
        """
        self.cmap = cmap

    @property
    def norm(self):
        return self._colorizer.norm

    @norm.setter
    def norm(self, norm):
        self._colorizer.norm = norm

    def set_norm(self, norm):
        """
        Set the normalization instance.

        Parameters
        ----------
        norm : `.Normalize` or str or None

        Notes
        -----
        If there are any colorbars using the mappable for this norm, setting
        the norm of the mappable will reset the norm, locator, and formatters
        on the colorbar to default.
        """
        self.norm = norm

    def autoscale(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array
        """
        self._colorizer.autoscale(self._A)

    def autoscale_None(self):
        """
        Autoscale the scalar limits on the norm instance using the
        current array, changing only limits that are None
        """
        self._colorizer.autoscale_None(self._A)

    @property
    def colorbar(self):
        """
        The last colorbar associated with this object. May be None
        """
        return self._colorizer.colorbar

    @colorbar.setter
    def colorbar(self, colorbar):
        self._colorizer.colorbar = colorbar

    def _format_cursor_data_override(self, data):
        # This function overwrites Artist.format_cursor_data(). We cannot
        # implement cm.ScalarMappable.format_cursor_data() directly, because
        # most cm.ScalarMappable subclasses inherit from Artist first and from
        # cm.ScalarMappable second, so Artist.format_cursor_data would always
        # have precedence over cm.ScalarMappable.format_cursor_data.

        # Note if cm.ScalarMappable is depreciated, this functionality should be
        # implemented as format_cursor_data() on ColorizingArtist.
        n = self.cmap.N
        if np.ma.getmask(data):
            return "[]"
        normed = self.norm(data)
        if np.isfinite(normed):
            if isinstance(self.norm, colors.BoundaryNorm):
                # not an invertible normalization mapping
                cur_idx = np.argmin(np.abs(self.norm.boundaries - data))
                neigh_idx = max(0, cur_idx - 1)
                # use max diff to prevent delta == 0
                delta = np.diff(
                    self.norm.boundaries[neigh_idx:cur_idx + 2]
                ).max()
            elif self.norm.vmin == self.norm.vmax:
                # singular norms, use delta of 10% of only value
                delta = np.abs(self.norm.vmin * .1)
            else:
                # Midpoints of neighboring color intervals.
                neighbors = self.norm.inverse(
                    (int(normed * n) + np.array([0, 1])) / n)
                delta = abs(neighbors - data).max()
            g_sig_digits = cbook._g_sig_digits(data, delta)
        else:
            g_sig_digits = 3  # Consistent with default below.
        return f"[{data:-#.{g_sig_digits}g}]"


class _ScalarMappable(_ColorizerInterface):
    """
    A mixin class to map one or multiple sets of scalar data to RGBA.

    The ScalarMappable applies data normalization before returning RGBA colors from
    the given `~matplotlib.colors.Colormap`.
    """

    # _ScalarMappable exists for compatibility with
    # code written before the introduction of the Colorizer
    # and ColorizingArtist classes.

    # _ScalarMappable can be depreciated so that ColorizingArtist
    # inherits directly from _ColorizerInterface.
    # in this case, the following changes should occur:
    # __init__() has its functionality moved to ColorizingArtist.
    # set_array(), get_array(), _get_colorizer() and
    # _check_exclusionary_keywords() are moved to ColorizingArtist.
    # changed() can be removed so long as colorbar.Colorbar
    # is changed to connect to the colorizer instead of the
    # ScalarMappable/ColorizingArtist,
    # otherwise changed() can be moved to ColorizingArtist.
    def __init__(self, norm=None, cmap=None, *, colorizer=None, **kwargs):
        """
        Parameters
        ----------
        norm : `.Normalize` (or subclass thereof) or str or None
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If a `str`, a `.Normalize` subclass is dynamically generated based
            on the scale with the corresponding name.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
        super().__init__(**kwargs)
        self._A = None
        self._colorizer = self._get_colorizer(colorizer=colorizer, norm=norm, cmap=cmap)

        self.colorbar = None
        self._id_colorizer = self._colorizer.callbacks.connect('changed', self.changed)
        self.callbacks = cbook.CallbackRegistry(signals=["changed"])

    def set_array(self, A):
        """
        Set the value array from array-like *A*.

        Parameters
        ----------
        A : array-like or None
            The values that are mapped to colors.

            The base class `.ScalarMappable` does not make any assumptions on
            the dimensionality and shape of the value array *A*.
        """
        if A is None:
            self._A = None
            return

        A = cbook.safe_masked_invalid(A, copy=True)
        if not np.can_cast(A.dtype, float, "same_kind"):
            raise TypeError(f"Image data of dtype {A.dtype} cannot be "
                            "converted to float")

        self._A = A
        if not self.norm.scaled():
            self._colorizer.autoscale_None(A)

    def get_array(self):
        """
        Return the array of values, that are mapped to colors.

        The base class `.ScalarMappable` does not make any assumptions on
        the dimensionality and shape of the array.
        """
        return self._A

    def changed(self):
        """
        Call this whenever the mappable is changed to notify all the
        callbackSM listeners to the 'changed' signal.
        """
        self.callbacks.process('changed', self)
        self.stale = True

    @staticmethod
    def _check_exclusionary_keywords(colorizer, **kwargs):
        """
        Raises a ValueError if any kwarg is not None while colorizer is not None
        """
        if colorizer is not None:
            if any([val is not None for val in kwargs.values()]):
                raise ValueError("The `colorizer` keyword cannot be used simultaneously"
                                 " with any of the following keywords: "
                                 + ", ".join(f'`{key}`' for key in kwargs.keys()))

    @staticmethod
    def _get_colorizer(cmap, norm, colorizer):
        if isinstance(colorizer, Colorizer):
            _ScalarMappable._check_exclusionary_keywords(
                Colorizer, cmap=cmap, norm=norm
            )
            return colorizer
        return Colorizer(cmap, norm)

# The docstrings here must be generic enough to apply to all relevant methods.
mpl._docstring.interpd.register(
    cmap_doc="""\
cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
    The Colormap instance or registered colormap name used to map scalar data
    to colors.""",
    norm_doc="""\
norm : str or `~matplotlib.colors.Normalize`, optional
    The normalization method used to scale scalar data to the [0, 1] range
    before mapping to colors using *cmap*. By default, a linear scaling is
    used, mapping the lowest value to 0 and the highest to 1.

    If given, this can be one of the following:

    - An instance of `.Normalize` or one of its subclasses
      (see :ref:`colormapnorms`).
    - A scale name, i.e. one of "linear", "log", "symlog", "logit", etc.  For a
      list of available scales, call `matplotlib.scale.get_scale_names()`.
      In that case, a suitable `.Normalize` subclass is dynamically generated
      and instantiated.""",
    vmin_vmax_doc="""\
vmin, vmax : float, optional
    When using scalar data and no explicit *norm*, *vmin* and *vmax* define
    the data range that the colormap covers. By default, the colormap covers
    the complete value range of the supplied data. It is an error to use
    *vmin*/*vmax* when a *norm* instance is given (but using a `str` *norm*
    name together with *vmin*/*vmax* is acceptable).""",
)


class ColorizingArtist(_ScalarMappable, artist.Artist):
    """
    Base class for artists that make map data to color using a `.colorizer.Colorizer`.

    The `.colorizer.Colorizer` applies data normalization before
    returning RGBA colors from a `~matplotlib.colors.Colormap`.

    """
    def __init__(self, colorizer, **kwargs):
        """
        Parameters
        ----------
        colorizer : `.colorizer.Colorizer`
        """
        _api.check_isinstance(Colorizer, colorizer=colorizer)
        super().__init__(colorizer=colorizer, **kwargs)

    @property
    def colorizer(self):
        return self._colorizer

    @colorizer.setter
    def colorizer(self, cl):
        _api.check_isinstance(Colorizer, colorizer=cl)
        self._colorizer.callbacks.disconnect(self._id_colorizer)
        self._colorizer = cl
        self._id_colorizer = cl.callbacks.connect('changed', self.changed)

    def _set_colorizer_check_keywords(self, colorizer, **kwargs):
        """
        Raises a ValueError if any kwarg is not None while colorizer is not None.
        """
        self._check_exclusionary_keywords(colorizer, **kwargs)
        self.colorizer = colorizer


def _auto_norm_from_scale(scale_cls):
    """
    Automatically generate a norm class from *scale_cls*.

    This differs from `.colors.make_norm_from_scale` in the following points:

    - This function is not a class decorator, but directly returns a norm class
      (as if decorating `.Normalize`).
    - The scale is automatically constructed with ``nonpositive="mask"``, if it
      supports such a parameter, to work around the difference in defaults
      between standard scales (which use "clip") and norms (which use "mask").

    Note that ``make_norm_from_scale`` caches the generated norm classes
    (not the instances) and reuses them for later calls.  For example,
    ``type(_auto_norm_from_scale("log")) == LogNorm``.
    """
    # Actually try to construct an instance, to verify whether
    # ``nonpositive="mask"`` is supported.
    try:
        norm = colors.make_norm_from_scale(
            functools.partial(scale_cls, nonpositive="mask"))(
            colors.Normalize)()
    except TypeError:
        norm = colors.make_norm_from_scale(scale_cls)(
            colors.Normalize)()
    return type(norm)
<|endoftext|>"""
`matplotlib.figure` implements the following classes:

`Figure`
    Top level `~matplotlib.artist.Artist`, which holds all plot elements.
    Many methods are implemented in `FigureBase`.

`SubFigure`
    A logical figure inside a figure, usually added to a figure (or parent `SubFigure`)
    with `Figure.add_subfigure` or `Figure.subfigures` methods.

Figures are typically created using pyplot methods `~.pyplot.figure`,
`~.pyplot.subplots`, and `~.pyplot.subplot_mosaic`.

.. plot::
    :include-source:

    fig, ax = plt.subplots(figsize=(2, 2), facecolor='lightskyblue',
                           layout='constrained')
    fig.suptitle('Figure')
    ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')

Some situations call for directly instantiating a `~.figure.Figure` class,
usually inside an application of some sort (see :ref:`user_interfaces` for a
list of examples) .  More information about Figures can be found at
:ref:`figure-intro`.
"""

from contextlib import ExitStack
import inspect
import itertools
import functools
import logging
from numbers import Integral
import threading

import numpy as np

import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
    Artist, allow_rasterization, _finalize_rasterization)
from matplotlib.backend_bases import (
    DrawEvent, FigureCanvasBase, NonGuiException, MouseButton, _get_renderer)
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotParams
from matplotlib.layout_engine import (
    ConstrainedLayoutEngine, TightLayoutEngine, LayoutEngine,
    PlaceHolderLayoutEngine
)
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
                                   TransformedBbox)

_log = logging.getLogger(__name__)


def _stale_figure_callback(self, val):
    if (fig := self.get_figure(root=False)) is not None:
        fig.stale = val


class _AxesStack:
    """
    Helper class to track Axes in a figure.

    Axes are tracked both in the order in which they have been added
    (``self._axes`` insertion/iteration order) and in the separate "gca" stack
    (which is the index to which they map in the ``self._axes`` dict).
    """

    def __init__(self):
        self._axes = {}  # Mapping of Axes to "gca" order.
        self._counter = itertools.count()

    def as_list(self):
        """List the Axes that have been added to the figure."""
        return [*self._axes]  # This relies on dict preserving order.

    def remove(self, a):
        """Remove the Axes from the stack."""
        self._axes.pop(a)

    def bubble(self, a):
        """Move an Axes, which must already exist in the stack, to the top."""
        if a not in self._axes:
            raise ValueError("Axes has not been added yet")
        self._axes[a] = next(self._counter)

    def add(self, a):
        """Add an Axes to the stack, ignoring it if already present."""
        if a not in self._axes:
            self._axes[a] = next(self._counter)

    def current(self):
        """Return the active Axes, or None if the stack is empty."""
        return max(self._axes, key=self._axes.__getitem__, default=None)

    def __getstate__(self):
        return {
            **vars(self),
            "_counter": max(self._axes.values(), default=0)
        }

    def __setstate__(self, state):
        next_counter = state.pop('_counter')
        vars(self).update(state)
        self._counter = itertools.count(next_counter)


class FigureBase(Artist):
    """
    Base class for `.Figure` and `.SubFigure` containing the methods that add
    artists to the figure or subfigure, create Axes, etc.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # remove the non-figure artist _axes property
        # as it makes no sense for a figure to be _in_ an Axes
        # this is used by the property methods in the artist base class
        # which are over-ridden in this class
        del self._axes

        self._suptitle = None
        self._supxlabel = None
        self._supylabel = None

        # groupers to keep track of x, y labels and title we want to align.
        # see self.align_xlabels, self.align_ylabels,
        # self.align_titles, and axis._get_tick_boxes_siblings
        self._align_label_groups = {
            "x": cbook.Grouper(),
            "y": cbook.Grouper(),
            "title": cbook.Grouper()
        }

        self._localaxes = []  # track all Axes
        self.artists = []
        self.lines = []
        self.patches = []
        self.texts = []
        self.images = []
        self.legends = []
        self.subfigs = []
        self.stale = True
        self.suppressComposite = None
        self.set(**kwargs)

    def _get_draw_artists(self, renderer):
        """Also runs apply_aspect"""
        artists = self.get_children()

        artists.remove(self.patch)
        artists = sorted(
            (artist for artist in artists if not artist.get_animated()),
            key=lambda artist: artist.get_zorder())
        for ax in self._localaxes:
            locator = ax.get_axes_locator()
            ax.apply_aspect(locator(ax, renderer) if locator else None)

            for child in ax.get_children():
                if hasattr(child, 'apply_aspect'):
                    locator = child.get_axes_locator()
                    child.apply_aspect(
                        locator(child, renderer) if locator else None)
        return artists

    def autofmt_xdate(
            self, bottom=0.2, rotation=30, ha='right', which='major'):
        """
        Date ticklabels often overlap, so it is useful to rotate them
        and right align them.  Also, a common use case is a number of
        subplots with shared x-axis where the x-axis is date data.  The
        ticklabels are often long, and it helps to rotate them on the
        bottom subplot and turn them off on other subplots, as well as
        turn off xlabels.

        Parameters
        ----------
        bottom : float, default: 0.2
            The bottom of the subplots for `subplots_adjust`.
        rotation : float, default: 30 degrees
            The rotation angle of the xtick labels in degrees.
        ha : {'left', 'center', 'right'}, default: 'right'
            The horizontal alignment of the xticklabels.
        which : {'major', 'minor', 'both'}, default: 'major'
            Selects which ticklabels to rotate.
        """
        _api.check_in_list(['major', 'minor', 'both'], which=which)
        axes = [ax for ax in self.axes if ax._label != '<colorbar>']
        allsubplots = all(ax.get_subplotspec() for ax in axes)
        if len(axes) == 1:
            for label in self.axes[0].get_xticklabels(which=which):
                label.set_ha(ha)
                label.set_rotation(rotation)
        else:
            if allsubplots:
                for ax in axes:
                    if ax.get_subplotspec().is_last_row():
                        for label in ax.get_xticklabels(which=which):
                            label.set_ha(ha)
                            label.set_rotation(rotation)
                    else:
                        for label in ax.get_xticklabels(which=which):
                            label.set_visible(False)
                        ax.set_xlabel('')

        engine = self.get_layout_engine()
        if allsubplots and (engine is None or engine.adjust_compatible):
            self.subplots_adjust(bottom=bottom)
        self.stale = True

    def get_children(self):
        """Get a list of artists contained in the figure."""
        return [self.patch,
                *self.artists,
                *self._localaxes,
                *self.lines,
                *self.patches,
                *self.texts,
                *self.images,
                *self.legends,
                *self.subfigs]

    def get_figure(self, root=None):
        """
        Return the `.Figure` or `.SubFigure` instance the (Sub)Figure belongs to.

        Parameters
        ----------
        root : bool, default=True
            If False, return the (Sub)Figure this artist is on.  If True,
            return the root Figure for a nested tree of SubFigures.

            .. deprecated:: 3.10

                From version 3.12 *root* will default to False.
        """
        if self._root_figure is self:
            # Top level Figure
            return self

        if self._parent is self._root_figure:
            # Return early to prevent the deprecation warning when *root* does not
            # matter
            return self._parent

        if root is None:
            # When deprecation expires, consider removing the docstring and just
            # inheriting the one from Artist.
            message = ('From Matplotlib 3.12 SubFigure.get_figure will by default '
                       'return the direct parent figure, which may be a SubFigure. '
                       'To suppress this warning, pass the root parameter.  Pass '
                       '`True` to maintain the old behavior and `False` to opt-in to '
                       'the future behavior.')
            _api.warn_deprecated('3.10', message=message)
            root = True

        if root:
            return self._root_figure

        return self._parent

    def set_figure(self, fig):
        """
        .. deprecated:: 3.10
            Currently this method will raise an exception if *fig* is anything other
            than the root `.Figure` this (Sub)Figure is on.  In future it will always
            raise an exception.
        """
        no_switch = ("The parent and root figures of a (Sub)Figure are set at "
                     "instantiation and cannot be changed.")
        if fig is self._root_figure:
            _api.warn_deprecated(
                "3.10",
                message=(f"{no_switch} From Matplotlib 3.12 this operation will raise "
                         "an exception."))
            return

        raise ValueError(no_switch)

    figure = property(functools.partial(get_figure, root=True), set_figure,
                      doc=("The root `Figure`.  To get the parent of a `SubFigure`, "
                           "use the `get_figure` method."))

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred on the figure.

        Returns
        -------
            bool, {}
        """
        if self._different_canvas(mouseevent):
            return False, {}
        inside = self.bbox.contains(mouseevent.x, mouseevent.y)
        return inside, {}

    def get_window_extent(self, renderer=None):
        # docstring inherited
        return self.bbox

    def _suplabels(self, t, info, **kwargs):
        """
        Add a centered %(name)s to the figure.

        Parameters
        ----------
        t : str
            The %(name)s text.
        x : float, default: %(x0)s
            The x location of the text in figure coordinates.
        y : float, default: %(y0)s
            The y location of the text in figure coordinates.
        horizontalalignment, ha : {'center', 'left', 'right'}, default: %(ha)s
            The horizontal alignment of the text relative to (*x*, *y*).
        verticalalignment, va : {'top', 'center', 'bottom', 'baseline'}, \
default: %(va)s
            The vertical alignment of the text relative to (*x*, *y*).
        fontsize, size : default: :rc:`figure.%(rc)ssize`
            The font size of the text. See `.Text.set_size` for possible
            values.
        fontweight, weight : default: :rc:`figure.%(rc)sweight`
            The font weight of the text. See `.Text.set_weight` for possible
            values.

        Returns
        -------
        text
            The `.Text` instance of the %(name)s.

        Other Parameters
        ----------------
        fontproperties : None or dict, optional
            A dict of font properties. If *fontproperties* is given the
            default values for font size and weight are taken from the
            `.FontProperties` defaults. :rc:`figure.%(rc)ssize` and
            :rc:`figure.%(rc)sweight` are ignored in this case.

        **kwargs
            Additional kwargs are `matplotlib.text.Text` properties.
        """

        x = kwargs.pop('x', None)
        y = kwargs.pop('y', None)
        if info['name'] in ['_supxlabel', '_suptitle']:
            autopos = y is None
        elif info['name'] == '_supylabel':
            autopos = x is None
        if x is None:
            x = info['x0']
        if y is None:
            y = info['y0']

        kwargs = cbook.normalize_kwargs(kwargs, Text)
        kwargs.setdefault('horizontalalignment', info['ha'])
        kwargs.setdefault('verticalalignment', info['va'])
        kwargs.setdefault('rotation', info['rotation'])

        if 'fontproperties' not in kwargs:
            kwargs.setdefault('fontsize', mpl.rcParams[info['size']])
            kwargs.setdefault('fontweight', mpl.rcParams[info['weight']])

        suplab = getattr(self, info['name'])
        if suplab is not None:
            suplab.set_text(t)
            suplab.set_position((x, y))
            suplab.set(**kwargs)
        else:
            suplab = self.text(x, y, t, **kwargs)
            setattr(self, info['name'], suplab)
            suplab._remove_method = functools.partial(self._remove_suplabel,
                                                      name=info['name'])

        suplab._autopos = autopos
        self.stale = True
        return suplab

    def _remove_suplabel(self, label, name):
        self.texts.remove(label)
        setattr(self, name, None)

    @_docstring.Substitution(x0=0.5, y0=0.98, name='super title', ha='center',
                             va='top', rc='title')
    @_docstring.copy(_suplabels)
    def suptitle(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_suptitle', 'x0': 0.5, 'y0': 0.98,
                'ha': 'center', 'va': 'top', 'rotation': 0,
                'size': 'figure.titlesize', 'weight': 'figure.titleweight'}
        return self._suplabels(t, info, **kwargs)

    def get_suptitle(self):
        """Return the suptitle as string or an empty string if not set."""
        text_obj = self._suptitle
        return "" if text_obj is None else text_obj.get_text()

    @_docstring.Substitution(x0=0.5, y0=0.01, name='super xlabel', ha='center',
                             va='bottom', rc='label')
    @_docstring.copy(_suplabels)
    def supxlabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supxlabel', 'x0': 0.5, 'y0': 0.01,
                'ha': 'center', 'va': 'bottom', 'rotation': 0,
                'size': 'figure.labelsize', 'weight': 'figure.labelweight'}
        return self._suplabels(t, info, **kwargs)

    def get_supxlabel(self):
        """Return the supxlabel as string or an empty string if not set."""
        text_obj = self._supxlabel
        return "" if text_obj is None else text_obj.get_text()

    @_docstring.Substitution(x0=0.02, y0=0.5, name='super ylabel', ha='left',
                             va='center', rc='label')
    @_docstring.copy(_suplabels)
    def supylabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supylabel', 'x0': 0.02, 'y0': 0.5,
                'ha': 'left', 'va': 'center', 'rotation': 'vertical',
                'rotation_mode': 'anchor', 'size': 'figure.labelsize',
                'weight': 'figure.labelweight'}
        return self._suplabels(t, info, **kwargs)

    def get_supylabel(self):
        """Return the supylabel as string or an empty string if not set."""
        text_obj = self._supylabel
        return "" if text_obj is None else text_obj.get_text()

    def get_edgecolor(self):
        """Get the edge color of the Figure rectangle."""
        return self.patch.get_edgecolor()

    def get_facecolor(self):
        """Get the face color of the Figure rectangle."""
        return self.patch.get_facecolor()

    def get_frameon(self):
        """
        Return the figure's background patch visibility, i.e.
        whether the figure background will be drawn. Equivalent to
        ``Figure.patch.get_visible()``.
        """
        return self.patch.get_visible()

    def set_linewidth(self, linewidth):
        """
        Set the line width of the Figure rectangle.

        Parameters
        ----------
        linewidth : number
        """
        self.patch.set_linewidth(linewidth)

    def get_linewidth(self):
        """
        Get the line width of the Figure rectangle.
        """
        return self.patch.get_linewidth()

    def set_edgecolor(self, color):
        """
        Set the edge color of the Figure rectangle.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        self.patch.set_edgecolor(color)

    def set_facecolor(self, color):
        """
        Set the face color of the Figure rectangle.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        self.patch.set_facecolor(color)

    def set_frameon(self, b):
        """
        Set the figure's background patch visibility, i.e.
        whether the figure background will be drawn. Equivalent to
        ``Figure.patch.set_visible()``.

        Parameters
        ----------
        b : bool
        """
        self.patch.set_visible(b)
        self.stale = True

    frameon = property(get_frameon, set_frameon)

    def add_artist(self, artist, clip=False):
        """
        Add an `.Artist` to the figure.

        Usually artists are added to `~.axes.Axes` objects using
        `.Axes.add_artist`; this method can be used in the rare cases where
        one needs to add artists directly to the figure instead.

        Parameters
        ----------
        artist : `~matplotlib.artist.Artist`
            The artist to add to the figure. If the added artist has no
            transform previously set, its transform will be set to
            ``figure.transSubfigure``.
        clip : bool, default: False
            Whether the added artist should be clipped by the figure patch.

        Returns
        -------
        `~matplotlib.artist.Artist`
            The added artist.
        """
        artist.set_figure(self)
        self.artists.append(artist)
        artist._remove_method = self.artists.remove

        if not artist.is_transform_set():
            artist.set_transform(self.transSubfigure)

        if clip and artist.get_clip_path() is None:
            artist.set_clip_path(self.patch)

        self.stale = True
        return artist

    @_docstring.interpd
    def add_axes(self, *args, **kwargs):
        """
        Add an `~.axes.Axes` to the figure.

        Call signatures::

            add_axes(rect, projection=None, polar=False, **kwargs)
            add_axes(ax)

        Parameters
        ----------
        rect : tuple (left, bottom, width, height)
            The dimensions (left, bottom, width, height) of the new
            `~.axes.Axes`. All quantities are in fractions of figure width and
            height.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
'polar', 'rectilinear', str}, optional
            The projection type of the `~.axes.Axes`. *str* is the name of
            a custom projection, see `~matplotlib.projections`. The default
            None results in a 'rectilinear' projection.

        polar : bool, default: False
            If True, equivalent to projection='polar'.

        axes_class : subclass type of `~.axes.Axes`, optional
            The `.axes.Axes` subclass that is instantiated.  This parameter
            is incompatible with *projection* and *polar*.  See
            :ref:`axisartist_users-guide-index` for examples.

        sharex, sharey : `~matplotlib.axes.Axes`, optional
            Share the x or y `~matplotlib.axis` with sharex and/or sharey.
            The axis will have the same limits, ticks, and scale as the axis
            of the shared Axes.

        label : str
            A label for the returned Axes.

        Returns
        -------
        `~.axes.Axes`, or a subclass of `~.axes.Axes`
            The returned Axes class depends on the projection used. It is
            `~.axes.Axes` if rectilinear projection is used and
            `.projections.polar.PolarAxes` if polar projection is used.

        Other Parameters
        ----------------
        **kwargs
            This method also takes the keyword arguments for
            the returned Axes class. The keyword arguments for the
            rectilinear Axes class `~.axes.Axes` can be found in
            the following table but there might also be other keyword
            arguments if another projection is used, see the actual Axes
            class.

            %(Axes:kwdoc)s

        Notes
        -----
        In rare circumstances, `.add_axes` may be called with a single
        argument, an Axes instance already created in the present figure but
        not in the figure's list of Axes.

        See Also
        --------
        .Figure.add_subplot
        .pyplot.subplot
        .pyplot.axes
        .Figure.subplots
        .pyplot.subplots

        Examples
        --------
        Some simple examples::

            rect = l, b, w, h
            fig = plt.figure()
            fig.add_axes(rect)
            fig.add_axes(rect, frameon=False, facecolor='g')
            fig.add_axes(rect, polar=True)
            ax = fig.add_axes(rect, projection='polar')
            fig.delaxes(ax)
            fig.add_axes(ax)
        """

        if not len(args) and 'rect' not in kwargs:
            raise TypeError("add_axes() missing 1 required positional argument: 'rect'")
        elif 'rect' in kwargs:
            if len(args):
                raise TypeError("add_axes() got multiple values for argument 'rect'")
            args = (kwargs.pop('rect'), )
        if len(args) != 1:
            raise _api.nargs_error("add_axes", 1, len(args))

        if isinstance(args[0], Axes):
            a, = args
            key = a._projection_init
            if a.get_figure(root=False) is not self:
                raise ValueError(
                    "The Axes must have been created in the present figure")
        else:
            rect, = args
            if not np.isfinite(rect).all():
                raise ValueError(f'all entries in rect must be finite not {rect}')
            projection_class, pkw = self._process_projection_requirements(**kwargs)

            # create the new Axes using the Axes class given
            a = projection_class(self, rect, **pkw)
            key = (projection_class, pkw)

        return self._add_axes_internal(a, key)

    @_docstring.interpd
    def add_subplot(self, *args, **kwargs):
        """
        Add an `~.axes.Axes` to the figure as part of a subplot arrangement.

        Call signatures::

           add_subplot(nrows, ncols, index, **kwargs)
           add_subplot(pos, **kwargs)
           add_subplot(ax)
           add_subplot()

        Parameters
        ----------
        *args : int, (int, int, *index*), or `.SubplotSpec`, default: (1, 1, 1)
            The position of the subplot described by one of

            - Three integers (*nrows*, *ncols*, *index*). The subplot will
              take the *index* position on a grid with *nrows* rows and
              *ncols* columns. *index* starts at 1 in the upper left corner
              and increases to the right.  *index* can also be a two-tuple
              specifying the (*first*, *last*) indices (1-based, and including
              *last*) of the subplot, e.g., ``fig.add_subplot(3, 1, (1, 2))``
              makes a subplot that spans the upper 2/3 of the figure.
            - A 3-digit integer. The digits are interpreted as if given
              separately as three single-digit integers, i.e.
              ``fig.add_subplot(235)`` is the same as
              ``fig.add_subplot(2, 3, 5)``. Note that this can only be used
              if there are no more than 9 subplots.
            - A `.SubplotSpec`.

            In rare circumstances, `.add_subplot` may be called with a single
            argument, a subplot Axes instance already created in the
            present figure but not in the figure's list of Axes.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
'polar', 'rectilinear', str}, optional
            The projection type of the subplot (`~.axes.Axes`). *str* is the
            name of a custom projection, see `~matplotlib.projections`. The
            default None results in a 'rectilinear' projection.

        polar : bool, default: False
            If True, equivalent to projection='polar'.

        axes_class : subclass type of `~.axes.Axes`, optional
            The `.axes.Axes` subclass that is instantiated.  This parameter
            is incompatible with *projection* and *polar*.  See
            :ref:`axisartist_users-guide-index` for examples.

        sharex, sharey : `~matplotlib.axes.Axes`, optional
            Share the x or y `~matplotlib.axis` with sharex and/or sharey.
            The axis will have the same limits, ticks, and scale as the axis
            of the shared Axes.

        label : str
            A label for the returned Axes.

        Returns
        -------
        `~.axes.Axes`

            The Axes of the subplot. The returned Axes can actually be an
            instance of a subclass, such as `.projections.polar.PolarAxes` for
            polar projections.

        Other Parameters
        ----------------
        **kwargs
            This method also takes the keyword arguments for the returned Axes
            base class; except for the *figure* argument. The keyword arguments
            for the rectilinear base class `~.axes.Axes` can be found in
            the following table but there might also be other keyword
            arguments if another projection is used.

            %(Axes:kwdoc)s

        See Also
        --------
        .Figure.add_axes
        .pyplot.subplot
        .pyplot.axes
        .Figure.subplots
        .pyplot.subplots

        Examples
        --------
        ::

            fig = plt.figure()

            fig.add_subplot(231)
            ax1 = fig.add_subplot(2, 3, 1)  # equivalent but more general

            fig.add_subplot(232, frameon=False)  # subplot with no frame
            fig.add_subplot(233, projection='polar')  # polar subplot
            fig.add_subplot(234, sharex=ax1)  # subplot sharing x-axis with ax1
            fig.add_subplot(235, facecolor="red")  # red subplot

            ax1.remove()  # delete ax1 from the figure
            fig.add_subplot(ax1)  # add ax1 back to the figure
        """
        if 'figure' in kwargs:
            # Axes itself allows for a 'figure' kwarg, but since we want to
            # bind the created Axes to self, it is not allowed here.
            raise _api.kwarg_error("add_subplot", "figure")

        if (len(args) == 1
                and isinstance(args[0], mpl.axes._base._AxesBase)
                and args[0].get_subplotspec()):
            ax = args[0]
            key = ax._projection_init
            if ax.get_figure(root=False) is not self:
                raise ValueError("The Axes must have been created in "
                                 "the present figure")
        else:
            if not args:
                args = (1, 1, 1)
            # Normalize correct ijk values to (i, j, k) here so that
            # add_subplot(211) == add_subplot(2, 1, 1).  Invalid values will
            # trigger errors later (via SubplotSpec._from_subplot_args).
            if (len(args) == 1 and isinstance(args[0], Integral)
                    and 100 <= args[0] <= 999):
                args = tuple(map(int, str(args[0])))
            projection_class, pkw = self._process_projection_requirements(**kwargs)
            ax = projection_class(self, *args, **pkw)
            key = (projection_class, pkw)
        return self._add_axes_internal(ax, key)

    def _add_axes_internal(self, ax, key):
        """Private helper for `add_axes` and `add_subplot`."""
        self._axstack.add(ax)
        if ax not in self._localaxes:
            self._localaxes.append(ax)
        self.sca(ax)
        ax._remove_method = self.delaxes
        # this is to support plt.subplot's re-selection logic
        ax._projection_init = key
        self.stale = True
        ax.stale_callback = _stale_figure_callback
        return ax

    def subplots(self, nrows=1, ncols=1, *, sharex=False, sharey=False,
                 squeeze=True, width_ratios=None, height_ratios=None,
                 subplot_kw=None, gridspec_kw=None):
        """
        Add a set of subplots to this figure.

        This utility wrapper makes it convenient to create common layouts of
        subplots in a single call.

        Parameters
        ----------
        nrows, ncols : int, default: 1
            Number of rows/columns of the subplot grid.

        sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
            Controls sharing of x-axis (*sharex*) or y-axis (*sharey*):

            - True or 'all': x- or y-axis will be shared among all subplots.
            - False or 'none': each subplot x- or y-axis will be independent.
            - 'row': each subplot row will share an x- or y-axis.
            - 'col': each subplot column will share an x- or y-axis.

            When subplots have a shared x-axis along a column, only the x tick
            labels of the bottom subplot are created. Similarly, when subplots
            have a shared y-axis along a row, only the y tick labels of the
            first column subplot are created. To later turn other subplots'
            ticklabels on, use `~matplotlib.axes.Axes.tick_params`.

            When subplots have a shared axis that has units, calling
            `.Axis.set_units` will update each axis with the new units.

            Note that it is not possible to unshare axes.

        squeeze : bool, default: True
            - If True, extra dimensions are squeezed out from the returned
              array of Axes:

              - if only one subplot is constructed (nrows=ncols=1), the
                resulting single Axes object is returned as a scalar.
              - for Nx1 or 1xM subplots, the returned object is a 1D numpy
                object array of Axes objects.
              - for NxM, subplots with N>1 and M>1 are returned as a 2D array.

            - If False, no squeezing at all is done: the returned Axes object
              is always a 2D array containing Axes instances, even if it ends
              up being 1x1.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.  Equivalent
            to ``gridspec_kw={'width_ratios': [...]}``.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height. Equivalent
            to ``gridspec_kw={'height_ratios': [...]}``.

        subplot_kw : dict, optional
            Dict with keywords passed to the `.Figure.add_subplot` call used to
            create each subplot.

        gridspec_kw : dict, optional
            Dict with keywords passed to the
            `~matplotlib.gridspec.GridSpec` constructor used to create
            the grid the subplots are placed on.

        Returns
        -------
        `~.axes.Axes` or array of Axes
            Either a single `~matplotlib.axes.Axes` object or an array of Axes
            objects if more than one subplot was created. The dimensions of the
            resulting array can be controlled with the *squeeze* keyword, see
            above.

        See Also
        --------
        .pyplot.subplots
        .Figure.add_subplot
        .pyplot.subplot

        Examples
        --------
        ::

            # First create some toy data:
            x = np.linspace(0, 2*np.pi, 400)
            y = np.sin(x**2)

            # Create a figure
            fig = plt.figure()

            # Create a subplot
            ax = fig.subplots()
            ax.plot(x, y)
            ax.set_title('Simple plot')

            # Create two subplots and unpack the output array immediately
            ax1, ax2 = fig.subplots(1, 2, sharey=True)
            ax1.plot(x, y)
            ax1.set_title('Sharing Y axis')
            ax2.scatter(x, y)

            # Create four polar Axes and access them through the returned array
            axes = fig.subplots(2, 2, subplot_kw=dict(projection='polar'))
            axes[0, 0].plot(x, y)
            axes[1, 1].scatter(x, y)

            # Share an X-axis with each column of subplots
            fig.subplots(2, 2, sharex='col')

            # Share a Y-axis with each row of subplots
            fig.subplots(2, 2, sharey='row')

            # Share both X- and Y-axes with all subplots
            fig.subplots(2, 2, sharex='all', sharey='all')

            # Note that this is the same as
            fig.subplots(2, 2, sharex=True, sharey=True)
        """
        gridspec_kw = dict(gridspec_kw or {})
        if height_ratios is not None:
            if 'height_ratios' in gridspec_kw:
                raise ValueError("'height_ratios' must not be defined both as "
                                 "parameter and as key in 'gridspec_kw'")
            gridspec_kw['height_ratios'] = height_ratios
        if width_ratios is not None:
            if 'width_ratios' in gridspec_kw:
                raise ValueError("'width_ratios' must not be defined both as "
                                 "parameter and as key in 'gridspec_kw'")
            gridspec_kw['width_ratios'] = width_ratios

        gs = self.add_gridspec(nrows, ncols, figure=self, **gridspec_kw)
        axs = gs.subplots(sharex=sharex, sharey=sharey, squeeze=squeeze,
                          subplot_kw=subplot_kw)
        return axs

    def delaxes(self, ax):
        """
        Remove the `~.axes.Axes` *ax* from the figure; update the current Axes.
        """
        self._remove_axes(ax, owners=[self._axstack, self._localaxes])

    def _remove_axes(self, ax, owners):
        """
        Common helper for removal of standard Axes (via delaxes) and of child Axes.

        Parameters
        ----------
        ax : `~.AxesBase`
            The Axes to remove.
        owners
            List of objects (list or _AxesStack) "owning" the Axes, from which the Axes
            will be remove()d.
        """
        for owner in owners:
            owner.remove(ax)

        self._axobservers.process("_axes_change_event", self)
        self.stale = True
        self._root_figure.canvas.release_mouse(ax)

        for name in ax._axis_names:  # Break link between any shared Axes
            grouper = ax._shared_axes[name]
            siblings = [other for other in grouper.get_siblings(ax) if other is not ax]
            if not siblings:  # Axes was not shared along this axis; we're done.
                continue
            grouper.remove(ax)
            # Formatters and locators may previously have been associated with the now
            # removed axis.  Update them to point to an axis still there (we can pick
            # any of them, and use the first sibling).
            remaining_axis = siblings[0]._axis_map[name]
            remaining_axis.get_major_formatter().set_axis(remaining_axis)
            remaining_axis.get_major_locator().set_axis(remaining_axis)
            remaining_axis.get_minor_formatter().set_axis(remaining_axis)
            remaining_axis.get_minor_locator().set_axis(remaining_axis)

        ax._twinned_axes.remove(ax)  # Break link between any twinned Axes.

    def clear(self, keep_observers=False):
        """
        Clear the figure.

        Parameters
        ----------
        keep_observers : bool, default: False
            Set *keep_observers* to True if, for example,
            a gui widget is tracking the Axes in the figure.
        """
        self.suppressComposite = None

        # first clear the Axes in any subfigures
        for subfig in self.subfigs:
            subfig.clear(keep_observers=keep_observers)
        self.subfigs = []

        for ax in tuple(self.axes):  # Iterate over the copy.
            ax.clear()
            self.delaxes(ax)  # Remove ax from self._axstack.

        self.artists = []
        self.lines = []
        self.patches = []
        self.texts = []
        self.images = []
        self.legends = []
        if not keep_observers:
            self._axobservers = cbook.CallbackRegistry()
        self._suptitle = None
        self._supxlabel = None
        self._supylabel = None

        self.stale = True

    # synonym for `clear`.
    def clf(self, keep_observers=False):
        """
        [*Discouraged*] Alias for the `clear()` method.

        .. admonition:: Discouraged

            The use of ``clf()`` is discouraged. Use ``clear()`` instead.

        Parameters
        ----------
        keep_observers : bool, default: False
            Set *keep_observers* to True if, for example,
            a gui widget is tracking the Axes in the figure.
        """
        return self.clear(keep_observers=keep_observers)

    # Note: the docstring below is modified with replace for the pyplot
    # version of this function because the method name differs (plt.figlegend)
    # the replacements are:
    #    " legend(" -> " figlegend(" for the signatures
    #    "fig.legend(" -> "plt.figlegend" for the code examples
    #    "ax.plot" -> "plt.plot" for consistency in using pyplot when able
    @_docstring.interpd
    def legend(self, *args, **kwargs):
        """
        Place a legend on the figure.

        Call signatures::

            legend()
            legend(handles, labels)
            legend(handles=handles)
            legend(labels)

        The call signatures correspond to the following different ways to use
        this method:

        **1. Automatic detection of elements to be shown in the legend**

        The elements to be added to the legend are automatically determined,
        when you do not pass in any extra arguments.

        In this case, the labels are taken from the artist. You can specify
        them either at artist creation or by calling the
        :meth:`~.Artist.set_label` method on the artist::

            ax.plot([1, 2, 3], label='Inline label')
            fig.legend()

        or::

            line, = ax.plot([1, 2, 3])
            line.set_label('Label via method')
            fig.legend()

        Specific lines can be excluded from the automatic legend element
        selection by defining a label starting with an underscore.
        This is default for all artists, so calling `.Figure.legend` without
        any arguments and without setting the labels manually will result in
        no legend being drawn.


        **2. Explicitly listing the artists and labels in the legend**

        For full control of which artists have a legend entry, it is possible
        to pass an iterable of legend artists followed by an iterable of
        legend labels respectively::

            fig.legend([line1, line2, line3], ['label1', 'label2', 'label3'])


        **3. Explicitly listing the artists in the legend**

        This is similar to 2, but the labels are taken from the artists'
        label properties. Example::

            line1, = ax1.plot([1, 2, 3], label='label1')
            line2, = ax2.plot([1, 2, 3], label='label2')
            fig.legend(handles=[line1, line2])


        **4. Labeling existing plot elements**

        .. admonition:: Discouraged

            This call signature is discouraged, because the relation between
            plot elements and labels is only implicit by their order and can
            easily be mixed up.

        To make a legend for all artists on all Axes, call this function with
        an iterable of strings, one for each legend item. For example::

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot([1, 3, 5], color='blue')
            ax2.plot([2, 4, 6], color='red')
            fig.legend(['the blues', 'the reds'])


        Parameters
        ----------
        handles : list of `.Artist`, optional
            A list of Artists (lines, patches) to be added to the legend.
            Use this together with *labels*, if you need full control on what
            is shown in the legend and the automatic mechanism described above
            is not sufficient.

            The length of handles and labels should be the same in this
            case. If they are not, they are truncated to the smaller length.

        labels : list of str, optional
            A list of labels to show next to the artists.
            Use this together with *handles*, if you need full control on what
            is shown in the legend and the automatic mechanism described above
            is not sufficient.

        Returns
        -------
        `~matplotlib.legend.Legend`

        Other Parameters
        ----------------
        %(_legend_kw_figure)s

        See Also
        --------
        .Axes.legend

        Notes
        -----
        Some artists are not supported by this function.  See
        :ref:`legend_guide` for details.
        """

        handles, labels, kwargs = mlegend._parse_legend_args(self.axes, *args, **kwargs)
        # explicitly set the bbox transform if the user hasn't.
        kwargs.setdefault("bbox_transform", self.transSubfigure)
        l = mlegend.Legend(self, handles, labels, **kwargs)
        self.legends.append(l)
        l._remove_method = self.legends.remove
        self.stale = True
        return l

    @_docstring.interpd
    def text(self, x, y, s, fontdict=None, **kwargs):
        """
        Add text to figure.

        Parameters
        ----------
        x, y : float
            The position to place the text. By default, this is in figure
            coordinates, floats in [0, 1]. The coordinate system can be changed
            using the *transform* keyword.

        s : str
            The text string.

        fontdict : dict, optional
            A dictionary to override the default text properties. If not given,
            the defaults are determined by :rc:`font.*`. Properties passed as
            *kwargs* override the corresponding ones given in *fontdict*.

        Returns
        -------
        `~.text.Text`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            Other miscellaneous text parameters.

            %(Text:kwdoc)s

        See Also
        --------
        .Axes.text
        .pyplot.text
        """
        effective_kwargs = {
            'transform': self.transSubfigure,
            **(fontdict if fontdict is not None else {}),
            **kwargs,
        }
        text = Text(x=x, y=y, text=s, **effective_kwargs)
        text.set_figure(self)
        text.stale_callback = _stale_figure_callback

        self.texts.append(text)
        text._remove_method = self.texts.remove
        self.stale = True
        return text

    @_docstring.interpd
    def colorbar(
            self, mappable, cax=None, ax=None, use_gridspec=True, **kwargs):
        """
        Add a colorbar to a plot.

        Parameters
        ----------
        mappable
            The `matplotlib.cm.ScalarMappable` (i.e., `.AxesImage`,
            `.ContourSet`, etc.) described by this colorbar.  This argument is
            mandatory for the `.Figure.colorbar` method but optional for the
            `.pyplot.colorbar` function, which sets the default to the current
            image.

            Note that one can create a `.ScalarMappable` "on-the-fly" to
            generate colorbars not attached to a previously drawn artist, e.g.
            ::

                fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

        cax : `~matplotlib.axes.Axes`, optional
            Axes into which the colorbar will be drawn.  If `None`, then a new
            Axes is created and the space for it will be stolen from the Axes(s)
            specified in *ax*.

        ax : `~matplotlib.axes.Axes` or iterable or `numpy.ndarray` of Axes, optional
            The one or more parent Axes from which space for a new colorbar Axes
            will be stolen. This parameter is only used if *cax* is not set.

            Defaults to the Axes that contains the mappable used to create the
            colorbar.

        use_gridspec : bool, optional
            If *cax* is ``None``, a new *cax* is created as an instance of
            Axes.  If *ax* is positioned with a subplotspec and *use_gridspec*
            is ``True``, then *cax* is also positioned with a subplotspec.

        Returns
        -------
        colorbar : `~matplotlib.colorbar.Colorbar`

        Other Parameters
        ----------------
        %(_make_axes_kw_doc)s
        %(_colormap_kw_doc)s

        Notes
        -----
        If *mappable* is a `~.contour.ContourSet`, its *extend* kwarg is
        included automatically.

        The *shrink* kwarg provides a simple way to scale the colorbar with
        respect to the Axes. Note that if *cax* is specified, it determines the
        size of the colorbar, and *shrink* and *aspect* are ignored.

        For more precise control, you can manually specify the positions of the
        axes objects in which the mappable and the colorbar are drawn.  In this
        case, do not use any of the Axes properties kwargs.

        It is known that some vector graphics viewers (svg and pdf) render
        white gaps between segments of the colorbar.  This is due to bugs in
        the viewers, not Matplotlib.  As a workaround, the colorbar can be
        rendered with overlapping segments::

            cbar = colorbar()
            cbar.solids.set_edgecolor("face")
            draw()

        However, this has negative consequences in other circumstances, e.g.
        with semi-transparent images (alpha < 1) and colorbar extensions;
        therefore, this workaround is not used by default (see issue #1188).

        """

        if ax is None:
            ax = getattr(mappable, "axes", None)

        if cax is None:
            if ax is None:
                raise ValueError(
                    'Unable to determine Axes to steal space for Colorbar. '
                    'Either provide the *cax* argument to use as the Axes for '
                    'the Colorbar, provide the *ax* argument to steal space '
                    'from it, or add *mappable* to an Axes.')
            fig = (  # Figure of first Axes; logic copied from make_axes.
                [*ax.flat] if isinstance(ax, np.ndarray)
                else [*ax] if np.iterable(ax)
                else [ax])[0].get_figure(root=False)
            current_ax = fig.gca()
            if (fig.get_layout_engine() is not None and
                    not fig.get_layout_engine().colorbar_gridspec):
                use_gridspec = False
            if (use_gridspec
                    and isinstance(ax, mpl.axes._base._AxesBase)
                    and ax.get_subplotspec()):
                cax, kwargs = cbar.make_axes_gridspec(ax, **kwargs)
            else:
                cax, kwargs = cbar.make_axes(ax, **kwargs)
            # make_axes calls add_{axes,subplot} which changes gca; undo that.
            fig.sca(current_ax)
            cax.grid(visible=False, which='both', axis='both')

        if (hasattr(mappable, "get_figure") and
                (mappable_host_fig := mappable.get_figure(root=True)) is not None):
            # Warn in case of mismatch
            if mappable_host_fig is not self._root_figure:
                _api.warn_external(
                        f'Adding colorbar to a different Figure '
                        f'{repr(mappable_host_fig)} than '
                        f'{repr(self._root_figure)} which '
                        f'fig.colorbar is called on.')

        NON_COLORBAR_KEYS = [  # remove kws that cannot be passed to Colorbar
            'fraction', 'pad', 'shrink', 'aspect', 'anchor', 'panchor']
        cb = cbar.Colorbar(cax, mappable, **{
            k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS})
        cax.get_figure(root=False).stale = True
        return cb

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=None):
        """
        Adjust the subplot layout parameters.

        Unset parameters are left unmodified; initial values are given by
        :rc:`figure.subplot.[name]`.

        .. plot:: _embedded_plots/figure_subplots_adjust.py

        Parameters
        ----------
        left : float, optional
            The position of the left edge of the subplots,
            as a fraction of the figure width.
        right : float, optional
            The position of the right edge of the subplots,
            as a fraction of the figure width.
        bottom : float, optional
            The position of the bottom edge of the subplots,
            as a fraction of the figure height.
        top : float, optional
            The position of the top edge of the subplots,
            as a fraction of the figure height.
        wspace : float, optional
            The width of the padding between subplots,
            as a fraction of the average Axes width.
        hspace : float, optional
            The height of the padding between subplots,
            as a fraction of the average Axes height.
        """
        if (self.get_layout_engine() is not None and
                not self.get_layout_engine().adjust_compatible):
            _api.warn_external(
                "This figure was using a layout engine that is "
                "incompatible with subplots_adjust and/or tight_layout; "
                "not calling subplots_adjust.")
            return
        self.subplotpars.update(left, bottom, right, top, wspace, hspace)
        for ax in self.axes:
            if ax.get_subplotspec() is not None:
                ax._set_position(ax.get_subplotspec().get_position(self))
        self.stale = True

    def align_xlabels(self, axs=None):
        """
        Align the xlabels of subplots in the same subplot row if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the bottom, it is aligned with labels on Axes that
        also have their label on the bottom and that have the same
        bottom-most subplot row.  If the label is on the top,
        it is aligned with labels on Axes with the same top-most row.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list of (or `~numpy.ndarray`) `~matplotlib.axes.Axes`
            to align the xlabels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_ylabels
        matplotlib.figure.Figure.align_titles
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that all Axes in ``axs`` are from the same `.GridSpec`,
        so that their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with rotated xtick labels::

            fig, axs = plt.subplots(1, 2)
            for tick in axs[0].get_xticklabels():
                tick.set_rotation(55)
            axs[0].set_xlabel('XLabel 0')
            axs[1].set_xlabel('XLabel 1')
            fig.align_xlabels()
        """
        if axs is None:
            axs = self.axes
        axs = [ax for ax in np.ravel(axs) if ax.get_subplotspec() is not None]
        for ax in axs:
            _log.debug(' Working on: %s', ax.get_xlabel())
            rowspan = ax.get_subplotspec().rowspan
            pos = ax.xaxis.get_label_position()  # top or bottom
            # Search through other Axes for label positions that are same as
            # this one and that share the appropriate row number.
            # Add to a grouper associated with each Axes of siblings.
            # This list is inspected in `axis.draw` by
            # `axis._update_label_position`.
            for axc in axs:
                if axc.xaxis.get_label_position() == pos:
                    rowspanc = axc.get_subplotspec().rowspan
                    if (pos == 'top' and rowspan.start == rowspanc.start or
                            pos == 'bottom' and rowspan.stop == rowspanc.stop):
                        # grouper for groups of xlabels to align
                        self._align_label_groups['x'].join(ax, axc)

    def align_ylabels(self, axs=None):
        """
        Align the ylabels of subplots in the same subplot column if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the left, it is aligned with labels on Axes that
        also have their label on the left and that have the same
        left-most subplot column.  If the label is on the right,
        it is aligned with labels on Axes with the same right-most column.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or `~numpy.ndarray`) of `~matplotlib.axes.Axes`
            to align the ylabels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels
        matplotlib.figure.Figure.align_titles
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that all Axes in ``axs`` are from the same `.GridSpec`,
        so that their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with large yticks labels::

            fig, axs = plt.subplots(2, 1)
            axs[0].plot(np.arange(0, 1000, 50))
            axs[0].set_ylabel('YLabel 0')
            axs[1].set_ylabel('YLabel 1')
            fig.align_ylabels()
        """
        if axs is None:
            axs = self.axes
        axs = [ax for ax in np.ravel(axs) if ax.get_subplotspec() is not None]
        for ax in axs:
            _log.debug(' Working on: %s', ax.get_ylabel())
            colspan = ax.get_subplotspec().colspan
            pos = ax.yaxis.get_label_position()  # left or right
            # Search through other Axes for label positions that are same as
            # this one and that share the appropriate column number.
            # Add to a list associated with each Axes of siblings.
            # This list is inspected in `axis.draw` by
            # `axis._update_label_position`.
            for axc in axs:
                if axc.yaxis.get_label_position() == pos:
                    colspanc = axc.get_subplotspec().colspan
                    if (pos == 'left' and colspan.start == colspanc.start or
                            pos == 'right' and colspan.stop == colspanc.stop):
                        # grouper for groups of ylabels to align
                        self._align_label_groups['y'].join(ax, axc)

    def align_titles(self, axs=None):
        """
        Align the titles of subplots in the same subplot row if title
        alignment is being done automatically (i.e. the title position is
        not manually set).

        Alignment persists for draw events after this is called.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list of (or ndarray) `~matplotlib.axes.Axes`
            to align the titles.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels
        matplotlib.figure.Figure.align_ylabels
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that all Axes in ``axs`` are from the same `.GridSpec`,
        so that their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with titles::

            fig, axs = plt.subplots(1, 2)
            axs[0].set_aspect('equal')
            axs[0].set_title('Title 0')
            axs[1].set_title('Title 1')
            fig.align_titles()
        """
        if axs is None:
            axs = self.axes
        axs = [ax for ax in np.ravel(axs) if ax.get_subplotspec() is not None]
        for ax in axs:
            _log.debug(' Working on: %s', ax.get_title())
            rowspan = ax.get_subplotspec().rowspan
            for axc in axs:
                rowspanc = axc.get_subplotspec().rowspan
                if (rowspan.start == rowspanc.start):
                    self._align_label_groups['title'].join(ax, axc)

    def align_labels(self, axs=None):
        """
        Align the xlabels and ylabels of subplots with the same subplots
        row or column (respectively) if label alignment is being
        done automatically (i.e. the label position is not manually set).

        Alignment persists for draw events after this is called.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or `~numpy.ndarray`) of `~matplotlib.axes.Axes`
            to align the labels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels
        matplotlib.figure.Figure.align_ylabels
        matplotlib.figure.Figure.align_titles

        Notes
        -----
        This assumes that all Axes in ``axs`` are from the same `.GridSpec`,
        so that their `.SubplotSpec` positions correspond to figure positions.
        """
        self.align_xlabels(axs=axs)
        self.align_ylabels(axs=axs)

    def add_gridspec(self, nrows=1, ncols=1, **kwargs):
        """
        Low-level API for creating a `.GridSpec` that has this figure as a parent.

        This is a low-level API, allowing you to create a gridspec and
        subsequently add subplots based on the gridspec. Most users do
        not need that freedom and should use the higher-level methods
        `~.Figure.subplots` or `~.Figure.subplot_mosaic`.

        Parameters
        ----------
        nrows : int, default: 1
            Number of rows in grid.

        ncols : int, default: 1
            Number of columns in grid.

        Returns
        -------
        `.GridSpec`

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments are passed to `.GridSpec`.

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        Adding a subplot that spans two rows::

            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            # spans two rows:
            ax3 = fig.add_subplot(gs[:, 1])

        """

        _ = kwargs.pop('figure', None)  # pop in case user has added this...
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=self, **kwargs)
        return gs

    def subfigures(self, nrows=1, ncols=1, squeeze=True,
                   wspace=None, hspace=None,
                   width_ratios=None, height_ratios=None,
                   **kwargs):
        """
        Add a set of subfigures to this figure or subfigure.

        A subfigure has the same artist methods as a figure, and is logically
        the same as a figure, but cannot print itself.
        See :doc:`/gallery/subplots_axes_and_figures/subfigures`.

        .. versionchanged:: 3.10
            subfigures are now added in row-major order.

        Parameters
        ----------
        nrows, ncols : int, default: 1
            Number of rows/columns of the subfigure grid.

        squeeze : bool, default: True
            If True, extra dimensions are squeezed out from the returned
            array of subfigures.

        wspace, hspace : float, default: None
            The amount of width/height reserved for space between subfigures,
            expressed as a fraction of the average subfigure width/height.
            If not given, the values will be inferred from rcParams if using
            constrained layout (see `~.ConstrainedLayoutEngine`), or zero if
            not using a layout engine.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=self,
                      wspace=wspace, hspace=hspace,
                      width_ratios=width_ratios,
                      height_ratios=height_ratios,
                      left=0, right=1, bottom=0, top=1)

        sfarr = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                sfarr[i, j] = self.add_subfigure(gs[i, j], **kwargs)

        if self.get_layout_engine() is None and (wspace is not None or
                                                 hspace is not None):
            # Gridspec wspace and hspace is ignored on subfigure instantiation,
            # and no space is left.  So need to account for it here if required.
            bottoms, tops, lefts, rights = gs.get_grid_positions(self)
            for sfrow, bottom, top in zip(sfarr, bottoms, tops):
                for sf, left, right in zip(sfrow, lefts, rights):
                    bbox = Bbox.from_extents(left, bottom, right, top)
                    sf._redo_transform_rel_fig(bbox=bbox)

        if squeeze:
            # Discarding unneeded dimensions that equal 1.  If we only have one
            # subfigure, just return it instead of a 1-element array.
            return sfarr.item() if sfarr.size == 1 else sfarr.squeeze()
        else:
            # Returned axis array will be always 2-d, even if nrows=ncols=1.
            return sfarr

    def add_subfigure(self, subplotspec, **kwargs):
        """
        Add a `.SubFigure` to the figure as part of a subplot arrangement.

        Parameters
        ----------
        subplotspec : `.gridspec.SubplotSpec`
            Defines the region in a parent gridspec where the subfigure will
            be placed.

        Returns
        -------
        `.SubFigure`

        Other Parameters
        ----------------
        **kwargs
            Are passed to the `.SubFigure` object.

        See Also
        --------
        .Figure.subfigures
        """
        sf = SubFigure(self, subplotspec, **kwargs)
        self.subfigs += [sf]
        sf._remove_method = self.subfigs.remove
        sf.stale_callback = _stale_figure_callback
        self.stale = True
        return sf

    def sca(self, a):
        """Set the current Axes to be *a* and return *a*."""
        self._axstack.bubble(a)
        self._axobservers.process("_axes_change_event", self)
        return a

    def gca(self):
        """
        Get the current Axes.

        If there is currently no Axes on this Figure, a new one is created
        using `.Figure.add_subplot`.  (To test whether there is currently an
        Axes on a Figure, check whether ``figure.axes`` is empty.  To test
        whether there is currently a Figure on the pyplot figure stack, check
        whether `.pyplot.get_fignums()` is empty.)
        """
        ax = self._axstack.current()
        return ax if ax is not None else self.add_subplot()

    def _gci(self):
        # Helper for `~matplotlib.pyplot.gci`.  Do not use elsewhere.
        """
        Get the current colorable artist.

        Specifically, returns the current `.ScalarMappable` instance (`.Image`
        created by `imshow` or `figimage`, `.Collection` created by `pcolor` or
        `scatter`, etc.), or *None* if no such instance has been defined.

        The current image is an attribute of the current Axes, or the nearest
        earlier Axes in the current figure that contains an image.

        Notes
        -----
        Historically, the only colorable artists were images; hence the name
        ``gci`` (get current image).
        """
        # Look first for an image in the current Axes.
        ax = self._axstack.current()
        if ax is None:
            return None
        im = ax._gci()
        if im is not None:
            return im
        # If there is no image in the current Axes, search for
        # one in a previously created Axes.  Whether this makes
        # sense is debatable, but it is the documented behavior.
        for ax in reversed(self.axes):
            im = ax._gci()
            if im is not None:
                return im
        return None

    def _process_projection_requirements(self, *, axes_class=None, polar=False,
                                         projection=None, **kwargs):
        """
        Handle the args/kwargs to add_axes/add_subplot/gca, returning::

            (axes_proj_class, proj_class_kwargs)

        which can be used for new Axes initialization/identification.
        """
        if axes_class is not None:
            if polar or projection is not None:
                raise ValueError(
                    "Cannot combine 'axes_class' and 'projection' or 'polar'")
            projection_class = axes_class
        else:

            if polar:
                if projection is not None and projection != 'polar':
                    raise ValueError(
                        f"polar={polar}, yet projection={projection!r}. "
                        "Only one of these arguments should be supplied."
                    )
                projection = 'polar'

            if isinstance(projection, str) or projection is None:
                projection_class = projections.get_projection_class(projection)
            elif hasattr(projection, '_as_mpl_axes'):
                projection_class, extra_kwargs = projection._as_mpl_axes()
                kwargs.update(**extra_kwargs)
            else:
                raise TypeError(
                    f"projection must be a string, None or implement a "
                    f"_as_mpl_axes method, not {projection!r}")
        return projection_class, kwargs

    def get_default_bbox_extra_artists(self):
        """
        Return a list of Artists typically used in `.Figure.get_tightbbox`.
        """
        bbox_artists = [artist for artist in self.get_children()
                        if (artist.get_visible() and artist.get_in_layout())]
        for ax in self.axes:
            if ax.get_visible():
                bbox_artists.extend(ax.get_default_bbox_extra_artists())
        return bbox_artists

    def get_tightbbox(self, renderer=None, *, bbox_extra_artists=None):
        """
        Return a (tight) bounding box of the figure *in inches*.

        Note that `.FigureBase` differs from all other artists, which return
        their `.Bbox` in pixels.

        Artists that have ``artist.set_in_layout(False)`` are not included
        in the bbox.

        Parameters
        ----------
        renderer : `.RendererBase` subclass
            Renderer that will be used to draw the figures (i.e.
            ``fig.canvas.get_renderer()``)

        bbox_extra_artists : list of `.Artist` or ``None``
            List of artists to include in the tight bounding box.  If
            ``None`` (default), then all artist children of each Axes are
            included in the tight bounding box.

        Returns
        -------
        `.BboxBase`
            containing the bounding box (in figure inches).
        """

        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()

        bb = []
        if bbox_extra_artists is None:
            artists = [artist for artist in self.get_children()
                       if (artist not in self.axes and artist.get_visible()
                           and artist.get_in_layout())]
        else:
            artists = bbox_extra_artists

        for a in artists:
            bbox = a.get_tightbbox(renderer)
            if bbox is not None:
                bb.append(bbox)

        for ax in self.axes:
            if ax.get_visible():
                # some Axes don't take the bbox_extra_artists kwarg so we
                # need this conditional....
                try:
                    bbox = ax.get_tightbbox(
                        renderer, bbox_extra_artists=bbox_extra_artists)
                except TypeError:
                    bbox = ax.get_tightbbox(renderer)
                bb.append(bbox)
        bb = [b for b in bb
              if (np.isfinite(b.width) and np.isfinite(b.height)
                  and (b.width != 0 or b.height != 0))]

        isfigure = hasattr(self, 'bbox_inches')
        if len(bb) == 0:
            if isfigure:
                return self.bbox_inches
            else:
                # subfigures do not have bbox_inches, but do have a bbox
                bb = [self.bbox]

        _bbox = Bbox.union(bb)

        if isfigure:
            # transform from pixels to inches...
            _bbox = TransformedBbox(_bbox, self.dpi_scale_trans.inverted())

        return _bbox

    @staticmethod
    def _norm_per_subplot_kw(per_subplot_kw):
        expanded = {}
        for k, v in per_subplot_kw.items():
            if isinstance(k, tuple):
                for sub_key in k:
                    if sub_key in expanded:
                        raise ValueError(f'The key {sub_key!r} appears multiple times.')
                    expanded[sub_key] = v
            else:
                if k in expanded:
                    raise ValueError(f'The key {k!r} appears multiple times.')
                expanded[k] = v
        return expanded

    @staticmethod
    def _normalize_grid_string(layout):
        if '\n' not in layout:
            # single-line string
            return [list(ln) for ln in layout.split(';')]
        else:
            # multi-line string
            layout = inspect.cleandoc(layout)
            return [list(ln) for ln in layout.strip('\n').split('\n')]

    def subplot_mosaic(self, mosaic, *, sharex=False, sharey=False,
                       width_ratios=None, height_ratios=None,
                       empty_sentinel='.',
                       subplot_kw=None, per_subplot_kw=None, gridspec_kw=None):
        """
        Build a layout of Axes based on ASCII art or nested lists.

        This is a helper function to build complex GridSpec layouts visually.

        See :ref:`mosaic`
        for an example and full API documentation

        Parameters
        ----------
        mosaic : list of list of {hashable or nested} or str

            A visual layout of how you want your Axes to be arranged
            labeled as strings.  For example ::

               x = [['A panel', 'A panel', 'edge'],
                    ['C panel', '.',       'edge']]

            produces 4 Axes:

            - 'A panel' which is 1 row high and spans the first two columns
            - 'edge' which is 2 rows high and is on the right edge
            - 'C panel' which in 1 row and 1 column wide in the bottom left
            - a blank space 1 row and 1 column wide in the bottom center

            Any of the entries in the layout can be a list of lists
            of the same form to create nested layouts.

            If input is a str, then it can either be a multi-line string of
            the form ::

              '''
              AAE
              C.E
              '''

            where each character is a column and each line is a row. Or it
            can be a single-line string where rows are separated by ``;``::

              'AB;CC'

            The string notation allows only single character Axes labels and
            does not support nesting but is very terse.

            The Axes identifiers may be `str` or a non-iterable hashable
            object (e.g. `tuple` s may not be used).

        sharex, sharey : bool, default: False
            If True, the x-axis (*sharex*) or y-axis (*sharey*) will be shared
            among all subplots.  In that case, tick label visibility and axis
            units behave as for `subplots`.  If False, each subplot's x- or
            y-axis will be independent.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.  Equivalent
            to ``gridspec_kw={'width_ratios': [...]}``. In the case of nested
            layouts, this argument applies only to the outer layout.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height. Equivalent
            to ``gridspec_kw={'height_ratios': [...]}``. In the case of nested
            layouts, this argument applies only to the outer layout.

        subplot_kw : dict, optional
            Dictionary with keywords passed to the `.Figure.add_subplot` call
            used to create each subplot.  These values may be overridden by
            values in *per_subplot_kw*.

        per_subplot_kw : dict, optional
            A dictionary mapping the Axes identifiers or tuples of identifiers
            to a dictionary of keyword arguments to be passed to the
            `.Figure.add_subplot` call used to create each subplot.  The values
            in these dictionaries have precedence over the values in
            *subplot_kw*.

            If *mosaic* is a string, and thus all keys are single characters,
            it is possible to use a single string instead of a tuple as keys;
            i.e. ``"AB"`` is equivalent to ``("A", "B")``.

            .. versionadded:: 3.7

        gridspec_kw : dict, optional
            Dictionary with keywords passed to the `.GridSpec` constructor used
            to create the grid the subplots are placed on. In the case of
            nested layouts, this argument applies only to the outer layout.
            For more complex layouts, users should use `.Figure.subfigures`
            to create the nesting.

        empty_sentinel : object, optional
            Entry in the layout to mean "leave this space empty".  Defaults
            to ``'.'``. Note, if *layout* is a string, it is processed via
            `inspect.cleandoc` to remove leading white space, which may
            interfere with using white-space as the empty sentinel.

        Returns
        -------
        dict[label, Axes]
           A dictionary mapping the labels to the Axes objects.  The order of
           the Axes is left-to-right and top-to-bottom of their position in the
           total layout.

        """
        subplot_kw = subplot_kw or {}
        gridspec_kw = dict(gridspec_kw or {})
        per_subplot_kw = per_subplot_kw or {}

        if height_ratios is not None:
            if 'height_ratios' in gridspec_kw:
                raise ValueError("'height_ratios' must not be defined both as "
                                 "parameter and as key in 'gridspec_kw'")
            gridspec_kw['height_ratios'] = height_ratios
        if width_ratios is not None:
            if 'width_ratios' in gridspec_kw:
                raise ValueError("'width_ratios' must not be defined both as "
                                 "parameter and as key in 'gridspec_kw'")
            gridspec_kw['width_ratios'] = width_ratios

        # special-case string input
        if isinstance(mosaic, str):
            mosaic = self._normalize_grid_string(mosaic)
            per_subplot_kw = {
                tuple(k): v for k, v in per_subplot_kw.items()
            }

        per_subplot_kw = self._norm_per_subplot_kw(per_subplot_kw)

        # Only accept strict bools to allow a possible future API expansion.
        _api.check_isinstance(bool, sharex=sharex, sharey=sharey)

        def _make_array(inp):
            """
            Convert input into 2D array

            We need to have this internal function rather than
            ``np.asarray(..., dtype=object)`` so that a list of lists
            of lists does not get converted to an array of dimension > 2.

            Returns
            -------
            2D object array
            """
            r0, *rest = inp
            if isinstance(r0, str):
                raise ValueError('List mosaic specification must be 2D')
            for j, r in enumerate(rest, start=1):
                if isinstance(r, str):
                    raise ValueError('List mosaic specification must be 2D')
                if len(r0) != len(r):
                    raise ValueError(
                        "All of the rows must be the same length, however "
                        f"the first row ({r0!r}) has length {len(r0)} "
                        f"and row {j} ({r!r}) has length {len(r)}."
                    )
            out = np.zeros((len(inp), len(r0)), dtype=object)
            for j, r in enumerate(inp):
                for k, v in enumerate(r):
                    out[j, k] = v
            return out

        def _identify_keys_and_nested(mosaic):
            """
            Given a 2D object array, identify unique IDs and nested mosaics

            Parameters
            ----------
            mosaic : 2D object array

            Returns
            -------
            unique_ids : tuple
                The unique non-sub mosaic entries in this mosaic
            nested : dict[tuple[int, int], 2D object array]
            """
            # make sure we preserve the user supplied order
            unique_ids = cbook._OrderedSet()
            nested = {}
            for j, row in enumerate(mosaic):
                for k, v in enumerate(row):
                    if v == empty_sentinel:
                        continue
                    elif not cbook.is_scalar_or_string(v):
                        nested[(j, k)] = _make_array(v)
                    else:
                        unique_ids.add(v)

            return tuple(unique_ids), nested

        def _do_layout(gs, mosaic, unique_ids, nested):
            """
            Recursively do the mosaic.

            Parameters
            ----------
            gs : GridSpec
            mosaic : 2D object array
                The input converted to a 2D array for this level.
            unique_ids : tuple
                The identified scalar labels at this level of nesting.
            nested : dict[tuple[int, int]], 2D object array
                The identified nested mosaics, if any.

            Returns
            -------
            dict[label, Axes]
                A flat dict of all of the Axes created.
            """
            output = dict()

            # we need to merge together the Axes at this level and the Axes
            # in the (recursively) nested sub-mosaics so that we can add
            # them to the figure in the "natural" order if you were to
            # ravel in c-order all of the Axes that will be created
            #
            # This will stash the upper left index of each object (axes or
            # nested mosaic) at this level
            this_level = dict()

            # go through the unique keys,
            for name in unique_ids:
                # sort out where each axes starts/ends
                indx = np.argwhere(mosaic == name)
                start_row, start_col = np.min(indx, axis=0)
                end_row, end_col = np.max(indx, axis=0) + 1
                # and construct the slice object
                slc = (slice(start_row, end_row), slice(start_col, end_col))
                # some light error checking
                if (mosaic[slc] != name).any():
                    raise ValueError(
                        f"While trying to layout\n{mosaic!r}\n"
                        f"we found that the label {name!r} specifies a "
                        "non-rectangular or non-contiguous area.")
                # and stash this slice for later
                this_level[(start_row, start_col)] = (name, slc, 'axes')

            # do the same thing for the nested mosaics (simpler because these
            # cannot be spans yet!)
            for (j, k), nested_mosaic in nested.items():
                this_level[(j, k)] = (None, nested_mosaic, 'nested')

            # now go through the things in this level and add them
            # in order left-to-right top-to-bottom
            for key in sorted(this_level):
                name, arg, method = this_level[key]
                # we are doing some hokey function dispatch here based
                # on the 'method' string stashed above to sort out if this
                # element is an Axes or a nested mosaic.
                if method == 'axes':
                    slc = arg
                    # add a single Axes
                    if name in output:
                        raise ValueError(f"There are duplicate keys {name} "
                                         f"in the layout\n{mosaic!r}")
                    ax = self.add_subplot(
                        gs[slc], **{
                            'label': str(name),
                            **subplot_kw,
                            **per_subplot_kw.get(name, {})
                        }
                    )
                    output[name] = ax
                elif method == 'nested':
                    nested_mosaic = arg
                    j, k = key
                    # recursively add the nested mosaic
                    rows, cols = nested_mosaic.shape
                    nested_output = _do_layout(
                        gs[j, k].subgridspec(rows, cols),
                        nested_mosaic,
                        *_identify_keys_and_nested(nested_mosaic)
                    )
                    overlap = set(output) & set(nested_output)
                    if overlap:
                        raise ValueError(
                            f"There are duplicate keys {overlap} "
                            f"between the outer layout\n{mosaic!r}\n"
                            f"and the nested layout\n{nested_mosaic}"
                        )
                    output.update(nested_output)
                else:
                    raise RuntimeError("This should never happen")
            return output

        mosaic = _make_array(mosaic)
        rows, cols = mosaic.shape
        gs = self.add_gridspec(rows, cols, **gridspec_kw)
        ret = _do_layout(gs, mosaic, *_identify_keys_and_nested(mosaic))
        ax0 = next(iter(ret.values()))
        for ax in ret.values():
            if sharex:
                ax.sharex(ax0)
                ax._label_outer_xaxis(skip_non_rectangular_axes=True)
            if sharey:
                ax.sharey(ax0)
                ax._label_outer_yaxis(skip_non_rectangular_axes=True)
        if extra := set(per_subplot_kw) - set(ret):
            raise ValueError(
                f"The keys {extra} are in *per_subplot_kw* "
                "but not in the mosaic."
            )
        return ret

    def _set_artist_props(self, a):
        if a != self:
            a.set_figure(self)
        a.stale_callback = _stale_figure_callback
        a.set_transform(self.transSubfigure)


@_docstring.interpd
class SubFigure(FigureBase):
    """
    Logical figure that can be placed inside a figure.

    See :ref:`figure-api-subfigure` for an index of methods on this class.
    Typically instantiated using `.Figure.add_subfigure` or
    `.SubFigure.add_subfigure`, or `.SubFigure.subfigures`.  A subfigure has
    the same methods as a figure except for those particularly tied to the size
    or dpi of the figure, and is confined to a prescribed region of the figure.
    For example the following puts two subfigures side-by-side::

        fig = plt.figure()
        sfigs = fig.subfigures(1, 2)
        axsL = sfigs[0].subplots(1, 2)
        axsR = sfigs[1].subplots(2, 1)

    See :doc:`/gallery/subplots_axes_and_figures/subfigures`
    """

    def __init__(self, parent, subplotspec, *,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 **kwargs):
        """
        Parameters
        ----------
        parent : `.Figure` or `.SubFigure`
            Figure or subfigure that contains the SubFigure.  SubFigures
            can be nested.

        subplotspec : `.gridspec.SubplotSpec`
            Defines the region in a parent gridspec where the subfigure will
            be placed.

        facecolor : default: ``"none"``
            The figure patch face color; transparent by default.

        edgecolor : default: :rc:`figure.edgecolor`
            The figure patch edge color.

        linewidth : float
            The linewidth of the frame (i.e. the edge linewidth of the figure
            patch).

        frameon : bool, default: :rc:`figure.frameon`
            If ``False``, suppress drawing the figure background patch.

        Other Parameters
        ----------------
        **kwargs : `.SubFigure` properties, optional

            %(SubFigure:kwdoc)s
        """
        super().__init__(**kwargs)
        if facecolor is None:
            facecolor = "none"
        if edgecolor is None:
            edgecolor = mpl.rcParams['figure.edgecolor']
        if frameon is None:
            frameon = mpl.rcParams['figure.frameon']

        self._subplotspec = subplotspec
        self._parent = parent
        self._root_figure = parent._root_figure

        # subfigures use the parent axstack
        self._axstack = parent._axstack
        self.subplotpars = parent.subplotpars
        self.dpi_scale_trans = parent.dpi_scale_trans
        self._axobservers = parent._axobservers
        self.transFigure = parent.transFigure
        self.bbox_relative = Bbox.null()
        self._redo_transform_rel_fig()
        self.figbbox = self._parent.figbbox
        self.bbox = TransformedBbox(self.bbox_relative,
                                    self._parent.transSubfigure)
        self.transSubfigure = BboxTransformTo(self.bbox)

        self.patch = Rectangle(
            xy=(0, 0), width=1, height=1, visible=frameon,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            # Don't let the figure patch influence bbox calculation.
            in_layout=False, transform=self.transSubfigure)
        self._set_artist_props(self.patch)
        self.patch.set_antialiased(False)

    @property
    def canvas(self):
        return self._parent.canvas

    @property
    def dpi(self):
        return self._parent.dpi

    @dpi.setter
    def dpi(self, value):
        self._parent.dpi = value

    def get_dpi(self):
        """
        Return the resolution of the parent figure in dots-per-inch as a float.
        """
        return self._parent.dpi

    def set_dpi(self, val):
        """
        Set the resolution of parent figure in dots-per-inch.

        Parameters
        ----------
        val : float
        """
        self._parent.dpi = val
        self.stale = True

    def _get_renderer(self):
        return self._parent._get_renderer()

    def _redo_transform_rel_fig(self, bbox=None):
        """
        Make the transSubfigure bbox relative to Figure transform.

        Parameters
        ----------
        bbox : bbox or None
            If not None, then the bbox is used for relative bounding box.
            Otherwise, it is calculated from the subplotspec.
        """
        if bbox is not None:
            self.bbox_relative.p0 = bbox.p0
            self.bbox_relative.p1 = bbox.p1
            return
        # need to figure out *where* this subplotspec is.
        gs = self._subplotspec.get_gridspec()
        wr = np.asarray(gs.get_width_ratios())
        hr = np.asarray(gs.get_height_ratios())
        dx = wr[self._subplotspec.colspan].sum() / wr.sum()
        dy = hr[self._subplotspec.rowspan].sum() / hr.sum()
        x0 = wr[:self._subplotspec.colspan.start].sum() / wr.sum()
        y0 = 1 - hr[:self._subplotspec.rowspan.stop].sum() / hr.sum()
        self.bbox_relative.p0 = (x0, y0)
        self.bbox_relative.p1 = (x0 + dx, y0 + dy)

    def get_constrained_layout(self):
        """
        Return whether constrained layout is being used.

        See :ref:`constrainedlayout_guide`.
        """
        return self._parent.get_constrained_layout()

    def get_constrained_layout_pads(self, relative=False):
        """
        Get padding for ``constrained_layout``.

        Returns a list of ``w_pad, h_pad`` in inches and
        ``wspace`` and ``hspace`` as fractions of the subplot.

        See :ref:`constrainedlayout_guide`.

        Parameters
        ----------
        relative : bool
            If `True`, then convert from inches to figure relative.
        """
        return self._parent.get_constrained_layout_pads(relative=relative)

    def get_layout_engine(self):
        return self._parent.get_layout_engine()

    @property
    def axes(self):
        """
        List of Axes in the SubFigure.  You can access and modify the Axes
        in the SubFigure through this list.

        Modifying this list has no effect. Instead, use `~.SubFigure.add_axes`,
        `~.SubFigure.add_subplot` or `~.SubFigure.delaxes` to add or remove an
        Axes.

        Note: The `.SubFigure.axes` property and `~.SubFigure.get_axes` method
        are equivalent.
        """
        return self._localaxes[:]

    get_axes = axes.fget

    def draw(self, renderer):
        # docstring inherited

        # draw the figure bounding box, perhaps none for white figure
        if not self.get_visible():
            return

        artists = self._get_draw_artists(renderer)

        try:
            renderer.open_group('subfigure', gid=self.get_gid())
            self.patch.draw(renderer)
            mimage._draw_list_compositing_images(
                renderer, self, artists, self.get_figure(root=True).suppressComposite)
            renderer.close_group('subfigure')

        finally:
            self.stale = False


@_docstring.interpd
class Figure(FigureBase):
    """
    The top level container for all the plot elements.

