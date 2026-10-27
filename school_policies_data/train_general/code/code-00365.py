        ax : `~matplotlib.axes.Axes`
            The parent Axes for the widget.
        labels : list of str
            The labels of the check buttons.
        actives : list of bool, optional
            The initial check states of the buttons. The list must have the
            same length as *labels*. If not given, all buttons are unchecked.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :ref:`blitting` for details.

            .. versionadded:: 3.7

        label_props : dict of lists, optional
            Dictionary of `.Text` properties to be used for the labels. Each
            dictionary value should be a list of at least a single element. If
            the list is of length M, its values are cycled such that the Nth
            label gets the (N mod M) property.

            .. versionadded:: 3.7
        frame_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            check button frame. Defaults (label font size / 2)**2 size, black
            edgecolor, no facecolor, and 1.0 linewidth.

            .. versionadded:: 3.7
        check_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            check button check. Defaults to (label font size / 2)**2 size,
            black color, and 1.0 linewidth.

            .. versionadded:: 3.7
        """
        super().__init__(ax)

        _api.check_isinstance((dict, None), label_props=label_props,
                              frame_props=frame_props, check_props=check_props)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        if actives is None:
            actives = [False] * len(labels)

        self._useblit = useblit and self.canvas.supports_blit
        self._background = None

        ys = np.linspace(1, 0, len(labels)+2)[1:-1]

        label_props = _expand_text_props(label_props)
        self.labels = [
            ax.text(0.25, y, label, transform=ax.transAxes,
                    horizontalalignment="left", verticalalignment="center",
                    **props)
            for y, label, props in zip(ys, labels, label_props)]
        text_size = np.array([text.get_fontsize() for text in self.labels]) / 2

        frame_props = {
            's': text_size**2,
            'linewidth': 1,
            **cbook.normalize_kwargs(frame_props, collections.PathCollection),
            'marker': 's',
            'transform': ax.transAxes,
        }
        frame_props.setdefault('facecolor', frame_props.get('color', 'none'))
        frame_props.setdefault('edgecolor', frame_props.pop('color', 'black'))
        self._frames = ax.scatter([0.15] * len(ys), ys, **frame_props)
        check_props = {
            'linewidth': 1,
            's': text_size**2,
            **cbook.normalize_kwargs(check_props, collections.PathCollection),
            'marker': 'x',
            'transform': ax.transAxes,
            'animated': self._useblit,
        }
        check_props.setdefault('facecolor', check_props.pop('color', 'black'))
        self._checks = ax.scatter([0.15] * len(ys), ys, **check_props)
        # The user may have passed custom colours in check_props, so we need to
        # create the checks (above), and modify the visibility after getting
        # whatever the user set.
        self._init_status(actives)

        self.connect_event('button_press_event', self._clicked)
        if self._useblit:
            self.connect_event('draw_event', self._clear)

        self._observers = cbook.CallbackRegistry(signals=["clicked"])

    def _clear(self, event):
        """Internal event handler to clear the buttons."""
        if self.ignore(event) or self.canvas.is_saving():
            return
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self._checks)

    def _clicked(self, event):
        if self.ignore(event) or event.button != 1 or not self.ax.contains(event)[0]:
            return
        idxs = [  # Indices of frames and of texts that contain the event.
            *self._frames.contains(event)[1]["ind"],
            *[i for i, text in enumerate(self.labels) if text.contains(event)[0]]]
        if idxs:
            coords = self._frames.get_offset_transform().transform(
                self._frames.get_offsets())
            self.set_active(  # Closest index, only looking in idxs.
                idxs[(((event.x, event.y) - coords[idxs]) ** 2).sum(-1).argmin()])

    def set_label_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Text` properties to be used for the labels. Same
            format as label_props argument of :class:`CheckButtons`.
        """
        _api.check_isinstance(dict, props=props)
        props = _expand_text_props(props)
        for text, prop in zip(self.labels, props):
            text.update(prop)

    def set_frame_props(self, props):
        """
        Set properties of the check button frames.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button frames.
        """
        _api.check_isinstance(dict, props=props)
        if 's' in props:  # Keep API consistent with constructor.
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        self._frames.update(props)

    def set_check_props(self, props):
        """
        Set properties of the check button checks.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the check
            button check.
        """
        _api.check_isinstance(dict, props=props)
        if 's' in props:  # Keep API consistent with constructor.
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        actives = self.get_status()
        self._checks.update(props)
        # If new colours are supplied, then we must re-apply the status.
        self._init_status(actives)

    def set_active(self, index, state=None):
        """
        Modify the state of a check button by index.

        Callbacks will be triggered if :attr:`eventson` is True.

        Parameters
        ----------
        index : int
            Index of the check button to toggle.

        state : bool, optional
            If a boolean value, set the state explicitly. If no value is
            provided, the state is toggled.

        Raises
        ------
        ValueError
            If *index* is invalid.
        TypeError
            If *state* is not boolean.
        """
        if index not in range(len(self.labels)):
            raise ValueError(f'Invalid CheckButton index: {index}')
        _api.check_isinstance((bool, None), state=state)

        invisible = colors.to_rgba('none')

        facecolors = self._checks.get_facecolor()
        if state is None:
            state = colors.same_color(facecolors[index], invisible)
        facecolors[index] = self._active_check_colors[index] if state else invisible
        self._checks.set_facecolor(facecolors)

        if self.drawon:
            if self._useblit:
                if self._background is not None:
                    self.canvas.restore_region(self._background)
                self.ax.draw_artist(self._checks)
                self.canvas.blit(self.ax.bbox)
            else:
                self.canvas.draw()

        if self.eventson:
            self._observers.process('clicked', self.labels[index].get_text())

    def _init_status(self, actives):
        """
        Initialize properties to match active status.

        The user may have passed custom colours in *check_props* to the
        constructor, or to `.set_check_props`, so we need to modify the
        visibility after getting whatever the user set.
        """
        self._active_check_colors = self._checks.get_facecolor()
        if len(self._active_check_colors) == 1:
            self._active_check_colors = np.repeat(self._active_check_colors,
                                                  len(actives), axis=0)
        self._checks.set_facecolor(
            [ec if active else "none"
             for ec, active in zip(self._active_check_colors, actives)])

    def clear(self):
        """Uncheck all checkboxes."""

        self._checks.set_facecolor(['none'] * len(self._active_check_colors))

        if hasattr(self, '_lines'):
            for l1, l2 in self._lines:
                l1.set_visible(False)
                l2.set_visible(False)

        if self.drawon:
            self.canvas.draw()

        if self.eventson:
            # Call with no label, as all checkboxes are being cleared.
            self._observers.process('clicked', None)

    def get_status(self):
        """
        Return a list of the status (True/False) of all of the check buttons.
        """
        return [not colors.same_color(color, colors.to_rgba("none"))
                for color in self._checks.get_facecolors()]

    def get_checked_labels(self):
        """Return a list of labels currently checked by user."""

        return [l.get_text() for l, box_checked in
                zip(self.labels, self.get_status())
                if box_checked]

    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Parameters
        ----------
        func : callable
            When the button is clicked, call *func* with button label.
            When all buttons are cleared, call *func* with None.
            The callback func must have the signature::

                def func(label: str | None) -> Any

            Return values may exist, but are ignored.

        Returns
        -------
        A connection id, which can be used to disconnect the callback.
        """
        return self._observers.connect('clicked', lambda text: func(text))

    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)


class TextBox(AxesWidget):
    """
    A GUI neutral text input box.

    For the text box to remain responsive you must keep a reference to it.

    Call `.on_text_change` to be updated whenever the text changes.

    Call `.on_submit` to be updated whenever the user hits enter or
    leaves the text entry field.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    label : `~matplotlib.text.Text`

    color : :mpltype:`color`
        The color of the text box when not hovering.
    hovercolor : :mpltype:`color`
        The color of the text box when hovering.
    """

    def __init__(self, ax, label, initial='', *,
                 color='.95', hovercolor='1', label_pad=.01,
                 textalignment="left"):
        """
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The `~.axes.Axes` instance the button will be placed into.
        label : str
            Label for this text box.
        initial : str
            Initial value in the text box.
        color : :mpltype:`color`
            The color of the box.
        hovercolor : :mpltype:`color`
            The color of the box when the mouse is over it.
        label_pad : float
            The distance between the label and the right side of the textbox.
        textalignment : {'left', 'center', 'right'}
            The horizontal location of the text.
        """
        super().__init__(ax)

        self._text_position = _api.check_getitem(
            {"left": 0.05, "center": 0.5, "right": 0.95},
            textalignment=textalignment)

        self.label = ax.text(
            -label_pad, 0.5, label, transform=ax.transAxes,
            verticalalignment='center', horizontalalignment='right')

        # TextBox's text object should not parse mathtext at all.
        self.text_disp = self.ax.text(
            self._text_position, 0.5, initial, transform=self.ax.transAxes,
            verticalalignment='center', horizontalalignment=textalignment,
            parse_math=False)

        self._observers = cbook.CallbackRegistry(signals=["change", "submit"])

        ax.set(
            xlim=(0, 1), ylim=(0, 1),  # s.t. cursor appears from first click.
            navigate=False, facecolor=color,
            xticks=[], yticks=[])

        self.cursor_index = 0

        self.cursor = ax.vlines(0, 0, 0, visible=False, color="k", lw=1,
                                transform=mpl.transforms.IdentityTransform())

        self.connect_event('button_press_event', self._click)
        self.connect_event('button_release_event', self._release)
        self.connect_event('motion_notify_event', self._motion)
        self.connect_event('key_press_event', self._keypress)
        self.connect_event('resize_event', self._resize)

        self.color = color
        self.hovercolor = hovercolor

        self.capturekeystrokes = False

    @property
    def text(self):
        return self.text_disp.get_text()

    def _rendercursor(self):
        # this is a hack to figure out where the cursor should go.
        # we draw the text up to where the cursor should go, measure
        # and save its dimensions, draw the real text, then put the cursor
        # at the saved dimensions

        # This causes a single extra draw if the figure has never been rendered
        # yet, which should be fine as we're going to repeatedly re-render the
        # figure later anyways.
        fig = self.ax.get_figure(root=True)
        if fig._get_renderer() is None:
            fig.canvas.draw()

        text = self.text_disp.get_text()  # Save value before overwriting it.
        widthtext = text[:self.cursor_index]

        bb_text = self.text_disp.get_window_extent()
        self.text_disp.set_text(widthtext or ",")
        bb_widthtext = self.text_disp.get_window_extent()

        if bb_text.y0 == bb_text.y1:  # Restoring the height if no text.
            bb_text.y0 -= bb_widthtext.height / 2
            bb_text.y1 += bb_widthtext.height / 2
        elif not widthtext:  # Keep width to 0.
            bb_text.x1 = bb_text.x0
        else:  # Move the cursor using width of bb_widthtext.
            bb_text.x1 = bb_text.x0 + bb_widthtext.width

        self.cursor.set(
            segments=[[(bb_text.x1, bb_text.y0), (bb_text.x1, bb_text.y1)]],
            visible=True)
        self.text_disp.set_text(text)

        fig.canvas.draw()

    def _release(self, event):
        if self.ignore(event):
            return
        if event.canvas.mouse_grabber != self.ax:
            return
        event.canvas.release_mouse(self.ax)

    def _keypress(self, event):
        if self.ignore(event):
            return
        if self.capturekeystrokes:
            key = event.key
            text = self.text
            if len(key) == 1:
                text = (text[:self.cursor_index] + key +
                        text[self.cursor_index:])
                self.cursor_index += 1
            elif key == "right":
                if self.cursor_index != len(text):
                    self.cursor_index += 1
            elif key == "left":
                if self.cursor_index != 0:
                    self.cursor_index -= 1
            elif key == "home":
                self.cursor_index = 0
            elif key == "end":
                self.cursor_index = len(text)
            elif key == "backspace":
                if self.cursor_index != 0:
                    text = (text[:self.cursor_index - 1] +
                            text[self.cursor_index:])
                    self.cursor_index -= 1
            elif key == "delete":
                if self.cursor_index != len(self.text):
                    text = (text[:self.cursor_index] +
                            text[self.cursor_index + 1:])
            self.text_disp.set_text(text)
            self._rendercursor()
            if self.eventson:
                self._observers.process('change', self.text)
                if key in ["enter", "return"]:
                    self._observers.process('submit', self.text)

    def set_val(self, val):
        newval = str(val)
        if self.text == newval:
            return
        self.text_disp.set_text(newval)
        self._rendercursor()
        if self.eventson:
            self._observers.process('change', self.text)
            self._observers.process('submit', self.text)

    def begin_typing(self):
        self.capturekeystrokes = True
        # Disable keypress shortcuts, which may otherwise cause the figure to
        # be saved, closed, etc., until the user stops typing.  The way to
        # achieve this depends on whether toolmanager is in use.
        stack = ExitStack()  # Register cleanup actions when user stops typing.
        self._on_stop_typing = stack.close
        toolmanager = getattr(
            self.ax.get_figure(root=True).canvas.manager, "toolmanager", None)
        if toolmanager is not None:
            # If using toolmanager, lock keypresses, and plan to release the
            # lock when typing stops.
            toolmanager.keypresslock(self)
            stack.callback(toolmanager.keypresslock.release, self)
        else:
            # If not using toolmanager, disable all keypress-related rcParams.
            # Avoid spurious warnings if keymaps are getting deprecated.
            with _api.suppress_matplotlib_deprecation_warning():
                stack.enter_context(mpl.rc_context(
                    {k: [] for k in mpl.rcParams if k.startswith("keymap.")}))

    def stop_typing(self):
        if self.capturekeystrokes:
            self._on_stop_typing()
            self._on_stop_typing = None
            notifysubmit = True
        else:
            notifysubmit = False
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        self.ax.get_figure(root=True).canvas.draw()
        if notifysubmit and self.eventson:
            # Because process() might throw an error in the user's code, only
            # call it once we've already done our cleanup.
            self._observers.process('submit', self.text)

    def _click(self, event):
        if self.ignore(event):
            return
        if not self.ax.contains(event)[0]:
            self.stop_typing()
            return
        if not self.eventson:
            return
        if event.canvas.mouse_grabber != self.ax:
            event.canvas.grab_mouse(self.ax)
        if not self.capturekeystrokes:
            self.begin_typing()
        self.cursor_index = self.text_disp._char_index_at(event.x)
        self._rendercursor()

    def _resize(self, event):
        self.stop_typing()

    def _motion(self, event):
        if self.ignore(event):
            return
        c = self.hovercolor if self.ax.contains(event)[0] else self.color
        if not colors.same_color(c, self.ax.get_facecolor()):
            self.ax.set_facecolor(c)
            if self.drawon:
                self.ax.get_figure(root=True).canvas.draw()

    def on_text_change(self, func):
        """
        When the text changes, call this *func* with event.

        A connection id is returned which can be used to disconnect.
        """
        return self._observers.connect('change', lambda text: func(text))

    def on_submit(self, func):
        """
        When the user hits enter or leaves the submission box, call this
        *func* with event.

        A connection id is returned which can be used to disconnect.
        """
        return self._observers.connect('submit', lambda text: func(text))

    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)


class RadioButtons(AxesWidget):
    """
    A GUI neutral radio button.

    For the buttons to remain responsive you must keep a reference to this
    object.

    Connect to the RadioButtons with the `~._Buttons.on_clicked` method.

    Attributes
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    activecolor : :mpltype:`color`
        The color of the selected button.
    labels : list of `.Text`
        The button labels.
    value_selected : str
        The label text of the currently selected button.
    index_selected : int
        The index of the selected button.
    """

    def __init__(self, ax, labels, active=0, activecolor=None, *,
                 useblit=True, label_props=None, radio_props=None):
        """
        Add radio buttons to an `~.axes.Axes`.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The Axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : :mpltype:`color`
            The color of the selected button. The default is ``'blue'`` if not
            specified here or in *radio_props*.
        useblit : bool, default: True
            Use blitting for faster drawing if supported by the backend.
            See the tutorial :ref:`blitting` for details.

            .. versionadded:: 3.7

        label_props : dict of lists, optional
            Dictionary of `.Text` properties to be used for the labels. Each
            dictionary value should be a list of at least a single element. If
            the list is of length M, its values are cycled such that the Nth
            label gets the (N mod M) property.

            .. versionadded:: 3.7
        radio_props : dict, optional
            Dictionary of scatter `.Collection` properties to be used for the
            radio buttons. Defaults to (label font size / 2)**2 size, black
            edgecolor, and *activecolor* facecolor (when active).

            .. note::
                If a facecolor is supplied in *radio_props*, it will override
                *activecolor*. This may be used to provide an active color per
                button.

            .. versionadded:: 3.7
        """
        super().__init__(ax)

        _api.check_isinstance((dict, None), label_props=label_props,
                              radio_props=radio_props)

        radio_props = cbook.normalize_kwargs(radio_props,
                                             collections.PathCollection)
        if activecolor is not None:
            if 'facecolor' in radio_props:
                _api.warn_external(
                    'Both the *activecolor* parameter and the *facecolor* '
                    'key in the *radio_props* parameter has been specified. '
                    '*activecolor* will be ignored.')
        else:
            activecolor = 'blue'  # Default.

        self._activecolor = activecolor
        self._initial_active = active
        self.value_selected = labels[active]
        self.index_selected = active

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        ys = np.linspace(1, 0, len(labels) + 2)[1:-1]

        self._useblit = useblit and self.canvas.supports_blit
        self._background = None

        label_props = _expand_text_props(label_props)
        self.labels = [
            ax.text(0.25, y, label, transform=ax.transAxes,
                    horizontalalignment="left", verticalalignment="center",
                    **props)
            for y, label, props in zip(ys, labels, label_props)]
        text_size = np.array([text.get_fontsize() for text in self.labels]) / 2

        radio_props = {
            's': text_size**2,
            **radio_props,
            'marker': 'o',
            'transform': ax.transAxes,
            'animated': self._useblit,
        }
        radio_props.setdefault('edgecolor', radio_props.get('color', 'black'))
        radio_props.setdefault('facecolor',
                               radio_props.pop('color', activecolor))
        self._buttons = ax.scatter([.15] * len(ys), ys, **radio_props)
        # The user may have passed custom colours in radio_props, so we need to
        # create the radios, and modify the visibility after getting whatever
        # the user set.
        self._active_colors = self._buttons.get_facecolor()
        if len(self._active_colors) == 1:
            self._active_colors = np.repeat(self._active_colors, len(labels),
                                            axis=0)
        self._buttons.set_facecolor(
            [activecolor if i == active else "none"
             for i, activecolor in enumerate(self._active_colors)])

        self.connect_event('button_press_event', self._clicked)
        if self._useblit:
            self.connect_event('draw_event', self._clear)

        self._observers = cbook.CallbackRegistry(signals=["clicked"])

    def _clear(self, event):
        """Internal event handler to clear the buttons."""
        if self.ignore(event) or self.canvas.is_saving():
            return
        self._background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self._buttons)

    def _clicked(self, event):
        if self.ignore(event) or event.button != 1 or not self.ax.contains(event)[0]:
            return
        idxs = [  # Indices of buttons and of texts that contain the event.
            *self._buttons.contains(event)[1]["ind"],
            *[i for i, text in enumerate(self.labels) if text.contains(event)[0]]]
        if idxs:
            coords = self._buttons.get_offset_transform().transform(
                self._buttons.get_offsets())
            self.set_active(  # Closest index, only looking in idxs.
                idxs[(((event.x, event.y) - coords[idxs]) ** 2).sum(-1).argmin()])

    def set_label_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Text` properties to be used for the labels. Same
            format as label_props argument of :class:`RadioButtons`.
        """
        _api.check_isinstance(dict, props=props)
        props = _expand_text_props(props)
        for text, prop in zip(self.labels, props):
            text.update(prop)

    def set_radio_props(self, props):
        """
        Set properties of the `.Text` labels.

        .. versionadded:: 3.7

        Parameters
        ----------
        props : dict
            Dictionary of `.Collection` properties to be used for the radio
            buttons.
        """
        _api.check_isinstance(dict, props=props)
        if 's' in props:  # Keep API consistent with constructor.
            props['sizes'] = np.broadcast_to(props.pop('s'), len(self.labels))
        self._buttons.update(props)
        self._active_colors = self._buttons.get_facecolor()
        if len(self._active_colors) == 1:
            self._active_colors = np.repeat(self._active_colors,
                                            len(self.labels), axis=0)
        self._buttons.set_facecolor(
            [activecolor if text.get_text() == self.value_selected else "none"
             for text, activecolor in zip(self.labels, self._active_colors)])

    @property
    def activecolor(self):
        return self._activecolor

    @activecolor.setter
    def activecolor(self, activecolor):
        colors._check_color_like(activecolor=activecolor)
        self._activecolor = activecolor
        self.set_radio_props({'facecolor': activecolor})

    def set_active(self, index):
        """
        Select button with number *index*.

        Callbacks will be triggered if :attr:`eventson` is True.

        Parameters
        ----------
        index : int
            The index of the button to activate.

        Raises
        ------
        ValueError
            If the index is invalid.
        """
        if index not in range(len(self.labels)):
            raise ValueError(f'Invalid RadioButton index: {index}')
        self.value_selected = self.labels[index].get_text()
        self.index_selected = index
        button_facecolors = self._buttons.get_facecolor()
        button_facecolors[:] = colors.to_rgba("none")
        button_facecolors[index] = colors.to_rgba(self._active_colors[index])
        self._buttons.set_facecolor(button_facecolors)

        if self.drawon:
            if self._useblit:
                if self._background is not None:
                    self.canvas.restore_region(self._background)
                self.ax.draw_artist(self._buttons)
                self.canvas.blit(self.ax.bbox)
            else:
                self.canvas.draw()

        if self.eventson:
            self._observers.process('clicked', self.labels[index].get_text())

    def clear(self):
        """Reset the active button to the initially active one."""
        self.set_active(self._initial_active)

    def on_clicked(self, func):
        """
        Connect the callback function *func* to button click events.

        Parameters
        ----------
        func : callable
            When the button is clicked, call *func* with button label.
            When all buttons are cleared, call *func* with None.
            The callback func must have the signature::

                def func(label: str | None) -> Any

            Return values may exist, but are ignored.

        Returns
        -------
        A connection id, which can be used to disconnect the callback.
        """
        return self._observers.connect('clicked', func)

    def disconnect(self, cid):
        """Remove the observer with connection id *cid*."""
        self._observers.disconnect(cid)


class SubplotTool(Widget):
    """
    A tool to adjust the subplot params of a `.Figure`.
    """

    def __init__(self, targetfig, toolfig):
        """
        Parameters
        ----------
        targetfig : `~matplotlib.figure.Figure`
            The figure instance to adjust.
        toolfig : `~matplotlib.figure.Figure`
            The figure instance to embed the subplot tool into.
        """

        self.figure = toolfig
        self.targetfig = targetfig
        toolfig.subplots_adjust(left=0.2, right=0.9)
        toolfig.suptitle("Click on slider to adjust subplot param")

        self._sliders = []
        names = ["left", "bottom", "right", "top", "wspace", "hspace"]
        # The last subplot, removed below, keeps space for the "Reset" button.
        for name, ax in zip(names, toolfig.subplots(len(names) + 1)):
            ax.set_navigate(False)
            slider = Slider(ax, name, 0, 1,
                            valinit=getattr(targetfig.subplotpars, name))
            slider.on_changed(self._on_slider_changed)
            self._sliders.append(slider)
        toolfig.axes[-1].remove()
        (self.sliderleft, self.sliderbottom, self.sliderright, self.slidertop,
         self.sliderwspace, self.sliderhspace) = self._sliders
        for slider in [self.sliderleft, self.sliderbottom,
                       self.sliderwspace, self.sliderhspace]:
            slider.closedmax = False
        for slider in [self.sliderright, self.slidertop]:
            slider.closedmin = False

        # constraints
        self.sliderleft.slidermax = self.sliderright
        self.sliderright.slidermin = self.sliderleft
        self.sliderbottom.slidermax = self.slidertop
        self.slidertop.slidermin = self.sliderbottom

        bax = toolfig.add_axes([0.8, 0.05, 0.15, 0.075])
        self.buttonreset = Button(bax, 'Reset')
        self.buttonreset.on_clicked(self._on_reset)

    def _on_slider_changed(self, _):
        self.targetfig.subplots_adjust(
            **{slider.label.get_text(): slider.val
               for slider in self._sliders})
        if self.drawon:
            self.targetfig.canvas.draw()

    def _on_reset(self, event):
        with ExitStack() as stack:
            # Temporarily disable drawing on self and self's sliders, and
            # disconnect slider events (as the subplotparams can be temporarily
            # invalid, depending on the order in which they are restored).
            stack.enter_context(cbook._setattr_cm(self, drawon=False))
            for slider in self._sliders:
                stack.enter_context(
                    cbook._setattr_cm(slider, drawon=False, eventson=False))
            # Reset the slider to the initial position.
            for slider in self._sliders:
                slider.reset()
        if self.drawon:
            event.canvas.draw()  # Redraw the subplottool canvas.
        self._on_slider_changed(None)  # Apply changes to the target window.


class Cursor(AxesWidget):
    """
    A crosshair cursor that spans the Axes and moves with mouse cursor.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.
    horizOn : bool, default: True
        Whether to draw the horizontal line.
    vertOn : bool, default: True
        Whether to draw the vertical line.
    useblit : bool, default: False
        Use blitting for faster drawing if supported by the backend.
        See the tutorial :ref:`blitting` for details.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    Examples
    --------
    See :doc:`/gallery/widgets/cursor`.
    """
    def __init__(self, ax, *, horizOn=True, vertOn=True, useblit=False,
                 **lineprops):
        super().__init__(ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)

        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit

        if self.useblit:
            lineprops['animated'] = True
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=False, **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], visible=False, **lineprops)

        self.background = None
        self.needclear = False

    def clear(self, event):
        """Internal event handler to clear the cursor."""
        if self.ignore(event) or self.canvas.is_saving():
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def onmove(self, event):
        """Internal event handler to draw the cursor when the mouse moves."""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if not self.ax.contains(event)[0]:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)
            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        xdata, ydata = self._get_data_coords(event)
        self.linev.set_xdata((xdata, xdata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_ydata((ydata, ydata))
        self.lineh.set_visible(self.visible and self.horizOn)
        if not (self.visible and (self.vertOn or self.horizOn)):
            return
        # Redraw.
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()


class MultiCursor(Widget):
    """
    Provide a vertical (default) and/or horizontal line cursor shared between
    multiple Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Parameters
    ----------
    canvas : object
        This parameter is entirely unused and only kept for back-compatibility.

    axes : list of `~matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.

    useblit : bool, default: True
        Use blitting for faster drawing if supported by the backend.
        See the tutorial :ref:`blitting`
        for details.

    horizOn : bool, default: False
        Whether to draw the horizontal line.

    vertOn : bool, default: True
        Whether to draw the vertical line.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.

    Examples
    --------
    See :doc:`/gallery/widgets/multicursor`.
    """

    def __init__(self, canvas, axes, *, useblit=True, horizOn=False, vertOn=True,
                 **lineprops):
        # canvas is stored only to provide the deprecated .canvas attribute;
        # once it goes away the unused argument won't need to be stored at all.
        self._canvas = canvas

        self.axes = axes
        self.horizOn = horizOn
        self.vertOn = vertOn

        self._canvas_infos = {
            ax.get_figure(root=True).canvas:
                {"cids": [], "background": None} for ax in axes}

        xmin, xmax = axes[-1].get_xlim()
        ymin, ymax = axes[-1].get_ylim()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        self.visible = True
        self.useblit = (
            useblit
            and all(canvas.supports_blit for canvas in self._canvas_infos))

        if self.useblit:
            lineprops['animated'] = True

        self.vlines = [ax.axvline(xmid, visible=False, **lineprops)
                       for ax in axes]
        self.hlines = [ax.axhline(ymid, visible=False, **lineprops)
                       for ax in axes]

        self.connect()

    def connect(self):
        """Connect events."""
        for canvas, info in self._canvas_infos.items():
            info["cids"] = [
                canvas.mpl_connect('motion_notify_event', self.onmove),
                canvas.mpl_connect('draw_event', self.clear),
            ]

    def disconnect(self):
        """Disconnect events."""
        for canvas, info in self._canvas_infos.items():
            for cid in info["cids"]:
                canvas.mpl_disconnect(cid)
            info["cids"].clear()

    def clear(self, event):
        """Clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            for canvas, info in self._canvas_infos.items():
                # someone has switched the canvas on us!  This happens if
                # `savefig` needs to save to a format the previous backend did
                # not support (e.g. saving a figure using an Agg based backend
                # saved to a vector format).
                if canvas is not canvas.figure.canvas:
                    continue
                info["background"] = canvas.copy_from_bbox(canvas.figure.bbox)

    def onmove(self, event):
        axs = [ax for ax in self.axes if ax.contains(event)[0]]
        if self.ignore(event) or not axs or not event.canvas.widgetlock.available(self):
            return
        ax = cbook._topmost_artist(axs)
        xdata, ydata = ((event.xdata, event.ydata) if event.inaxes is ax
                        else ax.transData.inverted().transform((event.x, event.y)))
        for line in self.vlines:
            line.set_xdata((xdata, xdata))
            line.set_visible(self.visible and self.vertOn)
        for line in self.hlines:
            line.set_ydata((ydata, ydata))
            line.set_visible(self.visible and self.horizOn)
        if not (self.visible and (self.vertOn or self.horizOn)):
            return
        # Redraw.
        if self.useblit:
            for canvas, info in self._canvas_infos.items():
                if info["background"]:
                    canvas.restore_region(info["background"])
            if self.vertOn:
                for ax, line in zip(self.axes, self.vlines):
                    ax.draw_artist(line)
            if self.horizOn:
                for ax, line in zip(self.axes, self.hlines):
                    ax.draw_artist(line)
            for canvas in self._canvas_infos:
                canvas.blit()
        else:
            for canvas in self._canvas_infos:
                canvas.draw_idle()


class _SelectorWidget(AxesWidget):

    def __init__(self, ax, onselect=None, useblit=False, button=None,
                 state_modifier_keys=None, use_data_coordinates=False):
        super().__init__(ax)

        self._visible = True
        if onselect is None:
            self.onselect = lambda *args: None
        else:
            self.onselect = onselect
        self.useblit = useblit and self.canvas.supports_blit
        self.connect_default_events()

        self._state_modifier_keys = dict(move=' ', clear='escape',
                                         square='shift', center='control',
                                         rotate='r')
        self._state_modifier_keys.update(state_modifier_keys or {})
        self._use_data_coordinates = use_data_coordinates

        self.background = None

        if isinstance(button, Integral):
            self.validButtons = [button]
        else:
            self.validButtons = button

        # Set to True when a selection is completed, otherwise is False
        self._selection_completed = False

        # will save the data (position at mouseclick)
        self._eventpress = None
        # will save the data (pos. at mouserelease)
        self._eventrelease = None
        self._prev_event = None
        self._state = set()

    def set_active(self, active):
        super().set_active(active)
        if active:
            self.update_background(None)

    def _get_animated_artists(self):
        """
        Convenience method to get all animated artists of the figure containing
        this widget, excluding those already present in self.artists.
        The returned tuple is not sorted by 'z_order': z_order sorting is
        valid only when considering all artists and not only a subset of all
        artists.
        """
        return tuple(a for ax_ in self.ax.get_figure().get_axes()
                     for a in ax_.get_children()
                     if a.get_animated() and a not in self.artists)

    def update_background(self, event):
        """Force an update of the background."""
        # If you add a call to `ignore` here, you'll want to check edge case:
        # `release` can call a draw event even when `ignore` is True.
        if not self.useblit:
            return
        if self.canvas.is_saving():
            return  # saving does not use blitting
        # Make sure that widget artists don't get accidentally included in the
        # background, by re-rendering the background if needed (and then
        # re-re-rendering the canvas with the visible widget artists).
        # We need to remove all artists which will be drawn when updating
        # the selector: if we have animated artists in the figure, it is safer
        # to redrawn by default, in case they have updated by the callback
        # zorder needs to be respected when redrawing
        artists = sorted(self.artists + self._get_animated_artists(),
                         key=lambda a: a.get_zorder())
        needs_redraw = any(artist.get_visible() for artist in artists)
        with ExitStack() as stack:
            if needs_redraw:
                for artist in artists:
                    stack.enter_context(artist._cm_set(visible=False))
                self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        if needs_redraw:
            for artist in artists:
                self.ax.draw_artist(artist)

    def connect_default_events(self):
        """Connect the major canvas events to methods."""
        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('button_press_event', self.press)
        self.connect_event('button_release_event', self.release)
        self.connect_event('draw_event', self.update_background)
        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('key_release_event', self.on_key_release)
        self.connect_event('scroll_event', self.on_scroll)

    def ignore(self, event):
        # docstring inherited
        if not self.active or not self.ax.get_visible():
            return True
        # If canvas was locked
        if not self.canvas.widgetlock.available(self):
            return True
        if not hasattr(event, 'button'):
            event.button = None
        # Only do rectangle selection if event was triggered
        # with a desired button
        if (self.validButtons is not None
                and event.button not in self.validButtons):
            return True
        # If no button was pressed yet ignore the event if it was out of the Axes.
        if self._eventpress is None:
            return not self.ax.contains(event)[0]
        # If a button was pressed, check if the release-button is the same.
        if event.button == self._eventpress.button:
            return False
        # If a button was pressed, check if the release-button is the same.
        return (not self.ax.contains(event)[0] or
                event.button != self._eventpress.button)

    def update(self):
        """Draw using blit() or draw_idle(), depending on ``self.useblit``."""
        if (not self.ax.get_visible() or
                self.ax.get_figure(root=True)._get_renderer() is None):
            return
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            else:
                self.update_background(None)
            # We need to draw all artists, which are not included in the
            # background, therefore we also draw self._get_animated_artists()
            # and we make sure that we respect z_order
            artists = sorted(self.artists + self._get_animated_artists(),
                             key=lambda a: a.get_zorder())
            for artist in artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _get_data(self, event):
        """Get the xdata and ydata for event, with limits."""
        if event.xdata is None:
            return None, None
        xdata, ydata = self._get_data_coords(event)
        xdata = np.clip(xdata, *self.ax.get_xbound())
        ydata = np.clip(ydata, *self.ax.get_ybound())
        return xdata, ydata

    def _clean_event(self, event):
        """
        Preprocess an event:

        - Replace *event* by the previous event if *event* has no ``xdata``.
        - Get ``xdata`` and ``ydata`` from this widget's Axes, and clip them to the axes
          limits.
        - Update the previous event.
        """
        if event.xdata is None:
            event = self._prev_event
        else:
            event = copy.copy(event)
        event.xdata, event.ydata = self._get_data(event)
        self._prev_event = event
        return event

    def press(self, event):
        """Button press handler and validator."""
        if not self.ignore(event):
            event = self._clean_event(event)
            self._eventpress = event
            self._prev_event = event
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            # move state is locked in on a button press
            if key == self._state_modifier_keys['move']:
                self._state.add('move')
            self._press(event)
            return True
        return False

    def _press(self, event):
        """Button press event handler."""

    def release(self, event):
        """Button release event handler and validator."""
        if not self.ignore(event) and self._eventpress:
            event = self._clean_event(event)
            self._eventrelease = event
            self._release(event)
            self._eventpress = None
            self._eventrelease = None
            self._state.discard('move')
            return True
        return False

    def _release(self, event):
        """Button release event handler."""

    def onmove(self, event):
        """Cursor move event handler and validator."""
        if not self.ignore(event) and self._eventpress:
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler."""

    def on_scroll(self, event):
        """Mouse scroll event handler and validator."""
        if not self.ignore(event):
            self._on_scroll(event)

    def _on_scroll(self, event):
        """Mouse scroll event handler."""

    def on_key_press(self, event):
        """Key press event handler and validator for all selection widgets."""
        if self.active:
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            if key == self._state_modifier_keys['clear']:
                self.clear()
                return
            for (state, modifier) in self._state_modifier_keys.items():
                if modifier in key.split('+'):
                    # 'rotate' is changing _state on press and is not removed
                    # from _state when releasing
                    if state == 'rotate':
                        if state in self._state:
                            self._state.discard(state)
                        else:
                            self._state.add(state)
                    else:
                        self._state.add(state)
            self._on_key_press(event)

    def _on_key_press(self, event):
        """Key press event handler - for widget-specific key press actions."""

    def on_key_release(self, event):
        """Key release event handler and validator."""
        if self.active:
            key = event.key or ''
            for (state, modifier) in self._state_modifier_keys.items():
                # 'rotate' is changing _state on press and is not removed
                # from _state when releasing
                if modifier in key.split('+') and state != 'rotate':
                    self._state.discard(state)
            self._on_key_release(event)

    def _on_key_release(self, event):
        """Key release event handler."""

    def set_visible(self, visible):
        """Set the visibility of the selector artists."""
        self._visible = visible
        for artist in self.artists:
            artist.set_visible(visible)

    def get_visible(self):
        """Get the visibility of the selector artists."""
        return self._visible

    def clear(self):
        """Clear the selection and set the selector ready to make a new one."""
        self._clear_without_update()
        self.update()

    def _clear_without_update(self):
        self._selection_completed = False
        self.set_visible(False)

    @property
    def artists(self):
        """Tuple of the artists of the selector."""
        handles_artists = getattr(self, '_handles_artists', ())
        return (self._selection_artist,) + handles_artists

    def set_props(self, **props):
        """
        Set the properties of the selector artist.

        See the *props* argument in the selector docstring to know which properties are
        supported.
        """
        artist = self._selection_artist
        props = cbook.normalize_kwargs(props, artist)
        artist.set(**props)
        if self.useblit:
            self.update()

    def set_handle_props(self, **handle_props):
        """
        Set the properties of the handles selector artist. See the
        `handle_props` argument in the selector docstring to know which
        properties are supported.
        """
        if not hasattr(self, '_handles_artists'):
            raise NotImplementedError("This selector doesn't have handles.")

        artist = self._handles_artists[0]
        handle_props = cbook.normalize_kwargs(handle_props, artist)
        for handle in self._handles_artists:
            handle.set(**handle_props)
        if self.useblit:
            self.update()
        self._handle_props.update(handle_props)

    def _validate_state(self, state):
        supported_state = [
            key for key, value in self._state_modifier_keys.items()
            if key != 'clear' and value != 'not-applicable'
            ]
        _api.check_in_list(supported_state, state=state)

    def add_state(self, state):
        """
        Add a state to define the widget's behavior. See the
        `state_modifier_keys` parameters for details.

        Parameters
        ----------
        state : str
            Must be a supported state of the selector. See the
            `state_modifier_keys` parameters for details.

        Raises
        ------
        ValueError
            When the state is not supported by the selector.

        """
        self._validate_state(state)
        self._state.add(state)

    def remove_state(self, state):
        """
        Remove a state to define the widget's behavior. See the
        `state_modifier_keys` parameters for details.

        Parameters
        ----------
        state : str
            Must be a supported state of the selector. See the
            `state_modifier_keys` parameters for details.

        Raises
        ------
        ValueError
            When the state is not supported by the selector.

        """
        self._validate_state(state)
        self._state.remove(state)


class SpanSelector(_SelectorWidget):
    """
    Visually select a min/max range on a single axis and call a function with
    those values.

    To guarantee that the selector remains responsive, keep a reference to it.

    In order to turn off the SpanSelector, set ``span_selector.active`` to
    False.  To turn it back on, set it to True.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`

    onselect : callable with signature ``func(min: float, max: float)``
        A callback function that is called after a release event and the
        selection is created, changed or removed.

    direction : {"horizontal", "vertical"}
        The direction along which to draw the span selector.

    minspan : float, default: 0
        If selection is less than or equal to *minspan*, the selection is
        removed (when already existing) or cancelled.

    useblit : bool, default: False
        If True, use the backend-dependent blitting features for faster
        canvas updates. See the tutorial :ref:`blitting` for details.

    props : dict, default: {'facecolor': 'red', 'alpha': 0.5}
        Dictionary of `.Patch` properties.

    onmove_callback : callable with signature ``func(min: float, max: float)``, optional
        Called on mouse move while the span is being selected.

    interactive : bool, default: False
        Whether to draw a set of handles that allow interaction with the
        widget after it is drawn.

    button : `.MouseButton` or list of `.MouseButton`, default: all buttons
        The mouse buttons which activate the span selector.

    handle_props : dict, default: None
        Properties of the handle lines at the edges of the span. Only used
        when *interactive* is True. See `.Line2D` for valid properties.

    grab_range : float, default: 10
        Distance in pixels within which the interactive tool handles can be activated.

    state_modifier_keys : dict, optional
        Keyboard modifiers which affect the widget's behavior.  Values
        amend the defaults, which are:

        - "clear": Clear the current shape, default: "escape".

    drag_from_anywhere : bool, default: False
        If `True`, the widget can be moved by clicking anywhere within its bounds.

    ignore_event_outside : bool, default: False
        If `True`, the event triggered outside the span selector will be ignored.

    snap_values : 1D array-like, optional
        Snap the selector edges to the given values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(vmin, vmax):
    ...     print(vmin, vmax)
    >>> span = mwidgets.SpanSelector(ax, onselect, 'horizontal',
    ...                              props=dict(facecolor='blue', alpha=0.5))
    >>> fig.show()

    See also: :doc:`/gallery/widgets/span_selector`
    """

    def __init__(self, ax, onselect, direction, *, minspan=0, useblit=False,
                 props=None, onmove_callback=None, interactive=False,
                 button=None, handle_props=None, grab_range=10,
                 state_modifier_keys=None, drag_from_anywhere=False,
                 ignore_event_outside=False, snap_values=None):

        if state_modifier_keys is None:
            state_modifier_keys = dict(clear='escape',
                                       square='not-applicable',
                                       center='not-applicable',
                                       rotate='not-applicable')
        super().__init__(ax, onselect, useblit=useblit, button=button,
                         state_modifier_keys=state_modifier_keys)

        if props is None:
            props = dict(facecolor='red', alpha=0.5)

        props['animated'] = self.useblit

        self.direction = direction
        self._extents_on_press = None
        self.snap_values = snap_values

        self.onmove_callback = onmove_callback
        self.minspan = minspan

        self.grab_range = grab_range
        self._interactive = interactive
        self._edge_handles = None
        self.drag_from_anywhere = drag_from_anywhere
        self.ignore_event_outside = ignore_event_outside

        self.new_axes(ax, _props=props, _init=True)

        # Setup handles
        self._handle_props = {
            'color': props.get('facecolor', 'r'),
            **cbook.normalize_kwargs(handle_props, Line2D)}

        if self._interactive:
            self._edge_order = ['min', 'max']
            self._setup_edge_handles(self._handle_props)

        self._active_handle = None

    def new_axes(self, ax, *, _props=None, _init=False):
        """Set SpanSelector to operate on a new Axes."""
        reconnect = False
        if _init or self.canvas is not ax.get_figure(root=True).canvas:
            if self.canvas is not None:
                self.disconnect_events()
            reconnect = True
        self.ax = ax
        if reconnect:
            self.connect_default_events()

        # Reset
        self._selection_completed = False

        if self.direction == 'horizontal':
            trans = ax.get_xaxis_transform()
            w, h = 0, 1
        else:
            trans = ax.get_yaxis_transform()
            w, h = 1, 0
        rect_artist = Rectangle((0, 0), w, h, transform=trans, visible=False)
        if _props is not None:
            rect_artist.update(_props)
        elif self._selection_artist is not None:
            rect_artist.update_from(self._selection_artist)

        self.ax.add_patch(rect_artist)
        self._selection_artist = rect_artist

    def _setup_edge_handles(self, props):
        # Define initial position using the axis bounds to keep the same bounds
        if self.direction == 'horizontal':
            positions = self.ax.get_xbound()
        else:
            positions = self.ax.get_ybound()
        self._edge_handles = ToolLineHandles(self.ax, positions,
                                             direction=self.direction,
                                             line_props=props,
                                             useblit=self.useblit)

    @property
    def _handles_artists(self):
        if self._edge_handles is not None:
            return self._edge_handles.artists
        else:
            return ()

    def _set_cursor(self, enabled):
        """Update the canvas cursor based on direction of the selector."""
        if enabled:
            cursor = (backend_tools.Cursors.RESIZE_HORIZONTAL
                      if self.direction == 'horizontal' else
                      backend_tools.Cursors.RESIZE_VERTICAL)
        else:
            cursor = backend_tools.Cursors.POINTER

        self.ax.get_figure(root=True).canvas.set_cursor(cursor)

    def connect_default_events(self):
        # docstring inherited
        super().connect_default_events()
        if getattr(self, '_interactive', False):
            self.connect_event('motion_notify_event', self._hover)

    def _press(self, event):
        """Button press event handler."""
        self._set_cursor(True)
        if self._interactive and self._selection_artist.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None

        if self._active_handle is None or not self._interactive:
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        xdata, ydata = self._get_data_coords(event)
        v = xdata if self.direction == 'horizontal' else ydata

        if self._active_handle is None and not self.ignore_event_outside:
            # when the press event outside the span, we initially set the
            # visibility to False and extents to (v, v)
            # update will be called when setting the extents
            self._visible = False
            self._set_extents((v, v))
            # We need to set the visibility back, so the span selector will be
            # drawn when necessary (span width > 0)
            self._visible = True
        else:
            self.set_visible(True)

        return False

    @property
    def direction(self):
        """Direction of the span selector: 'vertical' or 'horizontal'."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set the direction of the span selector."""
        _api.check_in_list(['horizontal', 'vertical'], direction=direction)
        if hasattr(self, '_direction') and direction != self._direction:
            # remove previous artists
            self._selection_artist.remove()
            if self._interactive:
                self._edge_handles.remove()
            self._direction = direction
            self.new_axes(self.ax)
            if self._interactive:
                self._setup_edge_handles(self._handle_props)
        else:
            self._direction = direction

    def _release(self, event):
        """Button release event handler."""
        self._set_cursor(False)

        if not self._interactive:
            self._selection_artist.set_visible(False)

        if (self._active_handle is None and self._selection_completed and
                self.ignore_event_outside):
            return

        vmin, vmax = self.extents
        span = vmax - vmin

        if span <= self.minspan:
            # Remove span and set self._selection_completed = False
            self.set_visible(False)
            if self._selection_completed:
                # Call onselect, only when the span is already existing
                self.onselect(vmin, vmax)
            self._selection_completed = False
        else:
            self.onselect(vmin, vmax)
            self._selection_completed = True

        self.update()

        self._active_handle = None

        return False

    def _hover(self, event):
        """Update the canvas cursor if it's over a handle."""
        if self.ignore(event):
            return

        if self._active_handle is not None or not self._selection_completed:
            # Do nothing if button is pressed and a handle is active, which may
            # occur with drag_from_anywhere=True.
            # Do nothing if selection is not completed, which occurs when
            # a selector has been cleared
            return

        _, e_dist = self._edge_handles.closest(event.x, event.y)
        self._set_cursor(e_dist <= self.grab_range)

    def _onmove(self, event):
        """Motion notify event handler."""

        xdata, ydata = self._get_data_coords(event)
        if self.direction == 'horizontal':
            v = xdata
            vpress = self._eventpress.xdata
        else:
            v = ydata
            vpress = self._eventpress.ydata

        # move existing span
        # When "dragging from anywhere", `self._active_handle` is set to 'C'
        # (match notation used in the RectangleSelector)
        if self._active_handle == 'C' and self._extents_on_press is not None:
            vmin, vmax = self._extents_on_press
            dv = v - vpress
            vmin += dv
            vmax += dv

        # resize an existing shape
        elif self._active_handle and self._active_handle != 'C':
            vmin, vmax = self._extents_on_press
            if self._active_handle == 'min':
                vmin = v
            else:
                vmax = v
        # new shape
        else:
            # Don't create a new span if there is already one when
            # ignore_event_outside=True
            if self.ignore_event_outside and self._selection_completed:
                return
            vmin, vmax = vpress, v
            if vmin > vmax:
                vmin, vmax = vmax, vmin

        self._set_extents((vmin, vmax))

        if self.onmove_callback is not None:
            self.onmove_callback(vmin, vmax)

        return False

    def _draw_shape(self, vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if self.direction == 'horizontal':
            self._selection_artist.set_x(vmin)
            self._selection_artist.set_width(vmax - vmin)
        else:
            self._selection_artist.set_y(vmin)
            self._selection_artist.set_height(vmax - vmin)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)

        # Prioritise center handle over other handles
        # Use 'C' to match the notation used in the RectangleSelector
        if 'move' in self._state:
            self._active_handle = 'C'
        elif e_dist > self.grab_range:
            # Not close to any handles
            self._active_handle = None
            if self.drag_from_anywhere and self._contains(event):
                # Check if we've clicked inside the region
                self._active_handle = 'C'
                self._extents_on_press = self.extents
            else:
                self._active_handle = None
                return
        else:
            # Closest to an edge handle
            self._active_handle = self._edge_order[e_idx]

        # Save coordinates of rectangle at the start of handle movement.
        self._extents_on_press = self.extents

    def _contains(self, event):
        """Return True if event is within the patch."""
        return self._selection_artist.contains(event, radius=0)[0]

    @staticmethod
    def _snap(values, snap_values):
        """Snap values to a given array values (snap_values)."""
        # take into account machine precision
        eps = np.min(np.abs(np.diff(snap_values))) * 1e-12
        return tuple(
            snap_values[np.abs(snap_values - v + np.sign(v) * eps).argmin()]
            for v in values)

    @property
    def extents(self):
        """
        (float, float)
            The values, in data coordinates, for the start and end points of the current
            selection. If there is no selection then the start and end values will be
            the same.
        """
        if self.direction == 'horizontal':
            vmin = self._selection_artist.get_x()
            vmax = vmin + self._selection_artist.get_width()
        else:
            vmin = self._selection_artist.get_y()
            vmax = vmin + self._selection_artist.get_height()
        return vmin, vmax

    @extents.setter
    def extents(self, extents):
        self._set_extents(extents)
        self._selection_completed = True

    def _set_extents(self, extents):
        # Update displayed shape
        if self.snap_values is not None:
            extents = tuple(self._snap(extents, self.snap_values))
        self._draw_shape(*extents)
        if self._interactive:
            # Update displayed handles
            self._edge_handles.set_data(self.extents)
        self.set_visible(self._visible)
        self.update()


class ToolLineHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    positions : 1D array
        Positions of handles in data coordinates.
    direction : {"horizontal", "vertical"}
        Direction of handles, either 'vertical' or 'horizontal'
    line_props : dict, optional
        Additional line properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    """

    def __init__(self, ax, positions, direction, *, line_props=None,
                 useblit=True):
        self.ax = ax

        _api.check_in_list(['horizontal', 'vertical'], direction=direction)
        self._direction = direction

        line_props = {
            **(line_props if line_props is not None else {}),
            'visible': False,
            'animated': useblit,
        }

        line_fun = ax.axvline if self.direction == 'horizontal' else ax.axhline

        self._artists = [line_fun(p, **line_props) for p in positions]

    @property
    def artists(self):
        return tuple(self._artists)

    @property
    def positions(self):
        """Positions of the handle in data coordinates."""
        method = 'get_xdata' if self.direction == 'horizontal' else 'get_ydata'
        return [getattr(line, method)()[0] for line in self.artists]

    @property
    def direction(self):
        """Direction of the handle: 'vertical' or 'horizontal'."""
        return self._direction

    def set_data(self, positions):
        """
        Set x- or y-positions of handles, depending on if the lines are
        vertical or horizontal.

        Parameters
        ----------
        positions : tuple of length 2
            Set the positions of the handle in data coordinates
        """
        method = 'set_xdata' if self.direction == 'horizontal' else 'set_ydata'
        for line, p in zip(self.artists, positions):
            getattr(line, method)([p, p])

    def set_visible(self, value):
        """Set the visibility state of the handles artist."""
        for artist in self.artists:
            artist.set_visible(value)

    def set_animated(self, value):
        """Set the animated state of the handles artist."""
        for artist in self.artists:
            artist.set_animated(value)

    def remove(self):
        """Remove the handles artist from the figure."""
        for artist in self._artists:
            artist.remove()

    def closest(self, x, y):
        """
        Return index and pixel distance to closest handle.

        Parameters
        ----------
        x, y : float
            x, y position from which the distance will be calculated to
            determinate the closest handle

        Returns
        -------
        index, distance : index of the handle and its distance from
            position x, y
        """
        if self.direction == 'horizontal':
            p_pts = np.array([
                self.ax.transData.transform((p, 0))[0] for p in self.positions
                ])
            dist = abs(p_pts - x)
        else:
            p_pts = np.array([
                self.ax.transData.transform((0, p))[1] for p in self.positions
                ])
            dist = abs(p_pts - y)
        index = np.argmin(dist)
        return index, dist[index]


class ToolHandles:
    """
    Control handles for canvas tools.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Matplotlib Axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str, default: 'o'
        Shape of marker used to display handle. See `~.pyplot.plot`.
    marker_props : dict, optional
        Additional marker properties. See `.Line2D`.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    """

    def __init__(self, ax, x, y, *, marker='o', marker_props=None, useblit=True):
        self.ax = ax
        props = {'marker': marker, 'markersize': 7, 'markerfacecolor': 'w',
                 'linestyle': 'none', 'alpha': 0.5, 'visible': False,
                 'label': '_nolegend_',
                 **cbook.normalize_kwargs(marker_props, Line2D._alias_map)}
        self._markers = Line2D(x, y, animated=useblit, **props)
        self.ax.add_line(self._markers)

    @property
    def x(self):
        return self._markers.get_xdata()

    @property
    def y(self):
        return self._markers.get_ydata()

    @property
    def artists(self):
        return (self._markers, )

    def set_data(self, pts, y=None):
        """Set x and y positions of handles."""
        if y is not None:
            x = pts
            pts = np.array([x, y])
        self._markers.set_data(pts)

    def set_visible(self, val):
        self._markers.set_visible(val)

    def set_animated(self, val):
        self._markers.set_animated(val)

    def closest(self, x, y):
        """Return index and pixel distance to closest index."""
        pts = np.column_stack([self.x, self.y])
        # Transform data coordinates to pixel coordinates.
        pts = self.ax.transData.transform(pts)
        diff = pts - [x, y]
        dist = np.hypot(*diff.T)
        min_index = np.argmin(dist)
        return min_index, dist[min_index]


_RECTANGLESELECTOR_PARAMETERS_DOCSTRING = \
    r"""
    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.

    onselect : function, optional
        A callback function that is called after a release event and the
        selection is created, changed or removed.
        It must have the signature::

            def onselect(eclick: MouseEvent, erelease: MouseEvent)

        where *eclick* and *erelease* are the mouse click and release
        `.MouseEvent`\s that start and complete the selection.

    minspanx : float, default: 0
        Selections with an x-span less than or equal to *minspanx* are removed
        (when already existing) or cancelled.

    minspany : float, default: 0
        Selections with an y-span less than or equal to *minspanx* are removed
        (when already existing) or cancelled.

    useblit : bool, default: False
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.

    props : dict, optional
        Properties with which the __ARTIST_NAME__ is drawn. See
        `.Patch` for valid properties.
        Default:

        ``dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)``

    spancoords : {"data", "pixels"}, default: "data"
        Whether to interpret *minspanx* and *minspany* in data or in pixel
        coordinates.

    button : `.MouseButton`, list of `.MouseButton`, default: all buttons
        Button(s) that trigger rectangle selection.

    grab_range : float, default: 10
        Distance in pixels within which the interactive tool handles can be
        activated.

    handle_props : dict, optional
        Properties with which the interactive handles (marker artists) are
        drawn. See the marker arguments in `.Line2D` for valid
        properties.  Default values are defined in ``mpl.rcParams`` except for
        the default value of ``markeredgecolor`` which will be the same as the
        ``edgecolor`` property in *props*.

    interactive : bool, default: False
        Whether to draw a set of handles that allow interaction with the
        widget after it is drawn.

    state_modifier_keys : dict, optional
        Keyboard modifiers which affect the widget's behavior.  Values
        amend the defaults, which are:

        - "move": Move the existing shape, default: no modifier.
        - "clear": Clear the current shape, default: "escape".
        - "square": Make the shape square, default: "shift".
        - "center": change the shape around its center, default: "ctrl".
        - "rotate": Rotate the shape around its center between -45° and 45°,
          default: "r".

        "square" and "center" can be combined. The square shape can be defined
        in data or display coordinates as determined by the
        ``use_data_coordinates`` argument specified when creating the selector.

    drag_from_anywhere : bool, default: False
        If `True`, the widget can be moved by clicking anywhere within
        its bounds.

    ignore_event_outside : bool, default: False
        If `True`, the event triggered outside the span selector will be
        ignored.

    use_data_coordinates : bool, default: False
        If `True`, the "square" shape of the selector is defined in
        data coordinates instead of display coordinates.
    """


@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace(
    '__ARTIST_NAME__', 'rectangle'))
class RectangleSelector(_SelectorWidget):
    """
    Select a rectangular region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.widgets as mwidgets
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [10, 50, 100])
    >>> def onselect(eclick, erelease):
    ...     print(eclick.xdata, eclick.ydata)
    ...     print(erelease.xdata, erelease.ydata)
    >>> props = dict(facecolor='blue', alpha=0.5)
    >>> rect = mwidgets.RectangleSelector(ax, onselect, interactive=True,
    ...                                   props=props)
    >>> fig.show()
    >>> rect.add_state('square')

    See also: :doc:`/gallery/widgets/rectangle_selector`
    """

    def __init__(self, ax, onselect=None, *, minspanx=0,
                 minspany=0, useblit=False,
                 props=None, spancoords='data', button=None, grab_range=10,
                 handle_props=None, interactive=False,
                 state_modifier_keys=None, drag_from_anywhere=False,
                 ignore_event_outside=False, use_data_coordinates=False):
        super().__init__(ax, onselect, useblit=useblit, button=button,
                         state_modifier_keys=state_modifier_keys,
                         use_data_coordinates=use_data_coordinates)

        self._interactive = interactive
        self.drag_from_anywhere = drag_from_anywhere
        self.ignore_event_outside = ignore_event_outside
        self._rotation = 0.0
        self._aspect_ratio_correction = 1.0

        # State to allow the option of an interactive selector that can't be
        # interactively drawn. This is used in PolygonSelector as an
        # interactive bounding box to allow the polygon to be easily resized
        self._allow_creation = True

        if props is None:
            props = dict(facecolor='red', edgecolor='black',
                         alpha=0.2, fill=True)
        props = {**props, 'animated': self.useblit}
        self._visible = props.pop('visible', self._visible)
        to_draw = self._init_shape(**props)
        self.ax.add_patch(to_draw)

        self._selection_artist = to_draw
        self._set_aspect_ratio_correction()

        self.minspanx = minspanx
        self.minspany = minspany

        _api.check_in_list(['data', 'pixels'], spancoords=spancoords)
        self.spancoords = spancoords

        self.grab_range = grab_range

        if self._interactive:
            self._handle_props = {
                'markeredgecolor': (props or {}).get('edgecolor', 'black'),
                **cbook.normalize_kwargs(handle_props, Line2D)}

            self._corner_order = ['SW', 'SE', 'NE', 'NW']
            xc, yc = self.corners
            self._corner_handles = ToolHandles(self.ax, xc, yc,
                                               marker_props=self._handle_props,
                                               useblit=self.useblit)

            self._edge_order = ['W', 'S', 'E', 'N']
            xe, ye = self.edge_centers
            self._edge_handles = ToolHandles(self.ax, xe, ye, marker='s',
                                             marker_props=self._handle_props,
                                             useblit=self.useblit)

            xc, yc = self.center
            self._center_handle = ToolHandles(self.ax, [xc], [yc], marker='s',
                                              marker_props=self._handle_props,
                                              useblit=self.useblit)

            self._active_handle = None

        self._extents_on_press = None

    @property
    def _handles_artists(self):
        return (*self._center_handle.artists, *self._corner_handles.artists,
                *self._edge_handles.artists)

    def _init_shape(self, **props):
        return Rectangle((0, 0), 0, 1, visible=False,
                         rotation_point='center', **props)

    def _press(self, event):
        """Button press event handler."""
        # make the drawn box/line visible get the click-coordinates, button, ...
        if self._interactive and self._selection_artist.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None

        if ((self._active_handle is None or not self._interactive) and
                self._allow_creation):
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        if (self._active_handle is None and not self.ignore_event_outside and
                self._allow_creation):
            x, y = self._get_data_coords(event)
            self._visible = False
            self.extents = x, x, y, y
            self._visible = True
        else:
            self.set_visible(True)

        self._extents_on_press = self.extents
        self._rotation_on_press = self._rotation
        self._set_aspect_ratio_correction()

        return False

    def _release(self, event):
        """Button release event handler."""
        if not self._interactive:
            self._selection_artist.set_visible(False)

        if (self._active_handle is None and self._selection_completed and
                self.ignore_event_outside):
            return

        # update the eventpress and eventrelease with the resulting extents
        x0, x1, y0, y1 = self.extents
        self._eventpress.xdata = x0
        self._eventpress.ydata = y0
        xy0 = self.ax.transData.transform([x0, y0])
        self._eventpress.x, self._eventpress.y = xy0

        self._eventrelease.xdata = x1
        self._eventrelease.ydata = y1
        xy1 = self.ax.transData.transform([x1, y1])
        self._eventrelease.x, self._eventrelease.y = xy1

        # calculate dimensions of box or line
        if self.spancoords == 'data':
            spanx = abs(self._eventpress.xdata - self._eventrelease.xdata)
            spany = abs(self._eventpress.ydata - self._eventrelease.ydata)
        elif self.spancoords == 'pixels':
            spanx = abs(self._eventpress.x - self._eventrelease.x)
            spany = abs(self._eventpress.y - self._eventrelease.y)
        else:
            _api.check_in_list(['data', 'pixels'],
                               spancoords=self.spancoords)
        # check if drawn distance (if it exists) is not too small in
        # either x or y-direction
        if spanx <= self.minspanx or spany <= self.minspany:
            if self._selection_completed:
                # Call onselect, only when the selection is already existing
                self.onselect(self._eventpress, self._eventrelease)
            self._clear_without_update()
        else:
            self.onselect(self._eventpress, self._eventrelease)
            self._selection_completed = True

        self.update()
        self._active_handle = None
        self._extents_on_press = None

        return False

    def _onmove(self, event):
        """
        Motion notify event handler.

        This can do one of four things:
        - Translate
        - Rotate
        - Re-size
        - Continue the creation of a new shape
        """
        eventpress = self._eventpress
        # The calculations are done for rotation at zero: we apply inverse
        # transformation to events except when we rotate and move
        state = self._state
        rotate = 'rotate' in state and self._active_handle in self._corner_order
        move = self._active_handle == 'C'
        resize = self._active_handle and not move

        xdata, ydata = self._get_data_coords(event)
        if resize:
            inv_tr = self._get_rotation_transform().inverted()
            xdata, ydata = inv_tr.transform([xdata, ydata])
            eventpress.xdata, eventpress.ydata = inv_tr.transform(
                (eventpress.xdata, eventpress.ydata))

        dx = xdata - eventpress.xdata
        dy = ydata - eventpress.ydata
        # refmax is used when moving the corner handle with the square state
        # and is the maximum between refx and refy
        refmax = None
        if self._use_data_coordinates:
            refx, refy = dx, dy
        else:
            # Get dx/dy in display coordinates
            refx = event.x - eventpress.x
            refy = event.y - eventpress.y

        x0, x1, y0, y1 = self._extents_on_press
        # rotate an existing shape
        if rotate:
            # calculate angle abc
            a = (eventpress.xdata, eventpress.ydata)
            b = self.center
            c = (xdata, ydata)
            angle = (np.arctan2(c[1]-b[1], c[0]-b[0]) -
                     np.arctan2(a[1]-b[1], a[0]-b[0]))
            self.rotation = np.rad2deg(self._rotation_on_press + angle)

        elif resize:
            size_on_press = [x1 - x0, y1 - y0]
            center = (x0 + size_on_press[0] / 2, y0 + size_on_press[1] / 2)

            # Keeping the center fixed
            if 'center' in state:
                # hh, hw are half-height and half-width
                if 'square' in state:
                    # when using a corner, find which reference to use
                    if self._active_handle in self._corner_order:
                        refmax = max(refx, refy, key=abs)
                    if self._active_handle in ['E', 'W'] or refmax == refx:
                        hw = xdata - center[0]
                        hh = hw / self._aspect_ratio_correction
                    else:
                        hh = ydata - center[1]
                        hw = hh * self._aspect_ratio_correction
                else:
                    hw = size_on_press[0] / 2
                    hh = size_on_press[1] / 2
                    # cancel changes in perpendicular direction
                    if self._active_handle in ['E', 'W'] + self._corner_order:
                        hw = abs(xdata - center[0])
                    if self._active_handle in ['N', 'S'] + self._corner_order:
                        hh = abs(ydata - center[1])

                x0, x1, y0, y1 = (center[0] - hw, center[0] + hw,
                                  center[1] - hh, center[1] + hh)

            else:
                # change sign of relative changes to simplify calculation
                # Switch variables so that x1 and/or y1 are updated on move
                if 'W' in self._active_handle:
                    x0 = x1
                if 'S' in self._active_handle:
                    y0 = y1
                if self._active_handle in ['E', 'W'] + self._corner_order:
                    x1 = xdata
                if self._active_handle in ['N', 'S'] + self._corner_order:
                    y1 = ydata
                if 'square' in state:
                    # when using a corner, find which reference to use
                    if self._active_handle in self._corner_order:
                        refmax = max(refx, refy, key=abs)
                    if self._active_handle in ['E', 'W'] or refmax == refx:
                        sign = np.sign(ydata - y0)
                        y1 = y0 + sign * abs(x1 - x0) / self._aspect_ratio_correction
                    else:
                        sign = np.sign(xdata - x0)
                        x1 = x0 + sign * abs(y1 - y0) * self._aspect_ratio_correction

        elif move:
            x0, x1, y0, y1 = self._extents_on_press
            dx = xdata - eventpress.xdata
            dy = ydata - eventpress.ydata
            x0 += dx
            x1 += dx
            y0 += dy
            y1 += dy

        else:
            # Create a new shape
            self._rotation = 0
            # Don't create a new rectangle if there is already one when
            # ignore_event_outside=True
            if ((self.ignore_event_outside and self._selection_completed) or
                    not self._allow_creation):
                return
            center = [eventpress.xdata, eventpress.ydata]
            dx = (xdata - center[0]) / 2
            dy = (ydata - center[1]) / 2

            # square shape
            if 'square' in state:
                refmax = max(refx, refy, key=abs)
                if refmax == refx:
                    dy = np.sign(dy) * abs(dx) / self._aspect_ratio_correction
                else:
                    dx = np.sign(dx) * abs(dy) * self._aspect_ratio_correction

            # from center
            if 'center' in state:
                dx *= 2
                dy *= 2

            # from corner
            else:
                center[0] += dx
                center[1] += dy

            x0, x1, y0, y1 = (center[0] - dx, center[0] + dx,
                              center[1] - dy, center[1] + dy)

        self.extents = x0, x1, y0, y1

    @property
    def _rect_bbox(self):
        return self._selection_artist.get_bbox().bounds

    def _set_aspect_ratio_correction(self):
        aspect_ratio = self.ax._get_aspect_ratio()
        self._selection_artist._aspect_ratio_correction = aspect_ratio
        if self._use_data_coordinates:
            self._aspect_ratio_correction = 1
        else:
            self._aspect_ratio_correction = aspect_ratio

    def _get_rotation_transform(self):
        aspect_ratio = self.ax._get_aspect_ratio()
        return Affine2D().translate(-self.center[0], -self.center[1]) \
                .scale(1, aspect_ratio) \
                .rotate(self._rotation) \
                .scale(1, 1 / aspect_ratio) \
                .translate(*self.center)

    @property
    def corners(self):
        """
        Corners of rectangle in data coordinates from lower left,
        moving clockwise.
        """
        x0, y0, width, height = self._rect_bbox
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        transform = self._get_rotation_transform()
        coords = transform.transform(np.array([xc, yc]).T).T
        return coords[0], coords[1]

    @property
    def edge_centers(self):
        """
        Midpoint of rectangle edges in data coordinates from left,
        moving anti-clockwise.
        """
        x0, y0, width, height = self._rect_bbox
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        transform = self._get_rotation_transform()
        coords = transform.transform(np.array([xe, ye]).T).T
        return coords[0], coords[1]

    @property
    def center(self):
        """Center of rectangle in data coordinates."""
        x0, y0, width, height = self._rect_bbox
        return x0 + width / 2., y0 + height / 2.

    @property
    def extents(self):
        """
        Return (xmin, xmax, ymin, ymax) in data coordinates as defined by the
        bounding box before rotation.
        """
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        # Update displayed shape
        self._draw_shape(extents)
        if self._interactive:
            # Update displayed handles
            self._corner_handles.set_data(*self.corners)
            self._edge_handles.set_data(*self.edge_centers)
            x, y = self.center
            self._center_handle.set_data([x], [y])
        self.set_visible(self._visible)
        self.update()

    @property
    def rotation(self):
        """
        Rotation in degree in interval [-45°, 45°]. The rotation is limited in
        range to keep the implementation simple.
        """
        return np.rad2deg(self._rotation)

    @rotation.setter
    def rotation(self, value):
        # Restrict to a limited range of rotation [-45°, 45°] to avoid changing
        # order of handles
        if -45 <= value and value <= 45:
            self._rotation = np.deg2rad(value)
            # call extents setter to draw shape and update handles positions
            self.extents = self.extents

    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        xlim = sorted(self.ax.get_xlim())
        ylim = sorted(self.ax.get_ylim())

        xmin = max(xlim[0], xmin)
        ymin = max(ylim[0], ymin)
        xmax = min(xmax, xlim[1])
        ymax = min(ymax, ylim[1])

        self._selection_artist.set_x(xmin)
        self._selection_artist.set_y(ymin)
        self._selection_artist.set_width(xmax - xmin)
        self._selection_artist.set_height(ymax - ymin)
        self._selection_artist.set_angle(self.rotation)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        c_idx, c_dist = self._corner_handles.closest(event.x, event.y)
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        m_idx, m_dist = self._center_handle.closest(event.x, event.y)

        if 'move' in self._state:
            self._active_handle = 'C'
        # Set active handle as closest handle, if mouse click is close enough.
        elif m_dist < self.grab_range * 2:
            # Prioritise center handle over other handles
            self._active_handle = 'C'
        elif c_dist > self.grab_range and e_dist > self.grab_range:
            # Not close to any handles
            if self.drag_from_anywhere and self._contains(event):
                # Check if we've clicked inside the region
                self._active_handle = 'C'
            else:
                self._active_handle = None
                return
        elif c_dist < e_dist:
            # Closest to a corner handle
            self._active_handle = self._corner_order[c_idx]
        else:
            # Closest to an edge handle
            self._active_handle = self._edge_order[e_idx]

    def _contains(self, event):
        """Return True if event is within the patch."""
        return self._selection_artist.contains(event, radius=0)[0]

    @property
    def geometry(self):
        """
        Return an array of shape (2, 5) containing the
        x (``RectangleSelector.geometry[1, :]``) and
        y (``RectangleSelector.geometry[0, :]``) data coordinates of the four
        corners of the rectangle starting and ending in the top left corner.
        """
        if hasattr(self._selection_artist, 'get_verts'):
            xfm = self.ax.transData.inverted()
            y, x = xfm.transform(self._selection_artist.get_verts()).T
            return np.array([x, y])
        else:
            return np.array(self._selection_artist.get_data())


@_docstring.Substitution(_RECTANGLESELECTOR_PARAMETERS_DOCSTRING.replace(
    '__ARTIST_NAME__', 'ellipse'))
class EllipseSelector(RectangleSelector):
    """
    Select an elliptical region of an Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Press and release events triggered at the same coordinates outside the
    selection will clear the selector, except when
    ``ignore_event_outside=True``.

    %s

    Examples
    --------
    :doc:`/gallery/widgets/rectangle_selector`
    """
    def _init_shape(self, **props):
        return Ellipse((0, 0), 0, 1, visible=False, **props)

    def _draw_shape(self, extents):
        x0, x1, y0, y1 = extents
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        center = [x0 + (x1 - x0) / 2., y0 + (y1 - y0) / 2.]
        a = (xmax - xmin) / 2.
        b = (ymax - ymin) / 2.

        self._selection_artist.center = center
        self._selection_artist.width = 2 * a
        self._selection_artist.height = 2 * b
        self._selection_artist.angle = self.rotation

    @property
    def _rect_bbox(self):
        x, y = self._selection_artist.center
        width = self._selection_artist.width
        height = self._selection_artist.height
        return x - width / 2., y - height / 2., width, height


class LassoSelector(_SelectorWidget):
    """
    Selection curve of an arbitrary shape.

    For the selector to remain responsive you must keep a reference to it.

    The selected path can be used in conjunction with `~.Path.contains_point`
    to select data points from an image.

    In contrast to `Lasso`, `LassoSelector` is written with an interface
    similar to `RectangleSelector` and `SpanSelector`, and will continue to
    interact with the Axes until disconnected.

    Example usage::

        ax = plt.subplot()
        ax.plot(x, y)

        def onselect(verts):
            print(verts)
        lasso = LassoSelector(ax, onselect)

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    onselect : function, optional
        Whenever the lasso is released, the *onselect* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    props : dict, optional
        Properties with which the line is drawn, see `.Line2D`
        for valid properties. Default values are defined in ``mpl.rcParams``.
    button : `.MouseButton` or list of `.MouseButton`, optional
        The mouse buttons used for rectangle selection.  Default is ``None``,
        which corresponds to all buttons.
    """

    def __init__(self, ax, onselect=None, *, useblit=True, props=None, button=None):
        super().__init__(ax, onselect, useblit=useblit, button=button)
        self.verts = None
        props = {
            **(props if props is not None else {}),
            # Note that self.useblit may be != useblit, if the canvas doesn't
            # support blitting.
            'animated': self.useblit, 'visible': False,
        }
        line = Line2D([], [], **props)
        self.ax.add_line(line)
        self._selection_artist = line

    def _press(self, event):
        self.verts = [self._get_data(event)]
        self._selection_artist.set_visible(True)

    def _release(self, event):
        if self.verts is not None:
            self.verts.append(self._get_data(event))
            self.onselect(self.verts)
        self._selection_artist.set_data([[], []])
        self._selection_artist.set_visible(False)
        self.verts = None

    def _onmove(self, event):
        if self.verts is None:
            return
        self.verts.append(self._get_data(event))
        self._selection_artist.set_data(list(zip(*self.verts)))

        self.update()


class PolygonSelector(_SelectorWidget):
    """
    Select a polygon region of an Axes.

    Place vertices with each mouse click, and make the selection by completing
    the polygon (clicking on the first vertex). Once drawn individual vertices
    can be moved by clicking and dragging with the left mouse button, or
    removed by clicking the right mouse button.

    In addition, the following modifier keys can be used:

    - Hold *ctrl* and click and drag a vertex to reposition it before the
      polygon has been completed.
    - Hold the *shift* key and click and drag anywhere in the Axes to move
      all vertices.
    - Press the *esc* key to start a new polygon.

    For the selector to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.

    onselect : function, optional
        When a polygon is completed or modified after completion,
        the *onselect* function is called and passed a list of the vertices as
        ``(xdata, ydata)`` tuples.

    useblit : bool, default: False
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.

    props : dict, optional
        Properties with which the line is drawn, see `.Line2D` for valid properties.
        Default::

            dict(color='k', linestyle='-', linewidth=2, alpha=0.5)

    handle_props : dict, optional
        Artist properties for the markers drawn at the vertices of the polygon.
        See the marker arguments in `.Line2D` for valid
        properties.  Default values are defined in ``mpl.rcParams`` except for
        the default value of ``markeredgecolor`` which will be the same as the
        ``color`` property in *props*.

    grab_range : float, default: 10
        A vertex is selected (to complete the polygon or to move a vertex) if
        the mouse click is within *grab_range* pixels of the vertex.

    draw_bounding_box : bool, optional
        If `True`, a bounding box will be drawn around the polygon selector
        once it is complete. This box can be used to move and resize the
        selector.

    box_handle_props : dict, optional
        Properties to set for the box handles. See the documentation for the
        *handle_props* argument to `RectangleSelector` for more info.

    box_props : dict, optional
        Properties to set for the box. See the documentation for the *props*
        argument to `RectangleSelector` for more info.

    Examples
    --------
    :doc:`/gallery/widgets/polygon_selector_simple`
    :doc:`/gallery/widgets/polygon_selector_demo`

    Notes
    -----
    If only one point remains after removing points, the selector reverts to an
    incomplete state and you can start drawing a new polygon from the existing
    point.
    """

    def __init__(self, ax, onselect=None, *, useblit=False,
                 props=None, handle_props=None, grab_range=10,
                 draw_bounding_box=False, box_handle_props=None,
                 box_props=None):
        # The state modifiers 'move', 'square', and 'center' are expected by
        # _SelectorWidget but are not supported by PolygonSelector
        # Note: could not use the existing 'move' state modifier in-place of
        # 'move_all' because _SelectorWidget automatically discards 'move'
        # from the state on button release.
        state_modifier_keys = dict(clear='escape', move_vertex='control',
                                   move_all='shift', move='not-applicable',
                                   square='not-applicable',
                                   center='not-applicable',
                                   rotate='not-applicable')
        super().__init__(ax, onselect, useblit=useblit,
                         state_modifier_keys=state_modifier_keys)

        self._xys = [(0, 0)]

        if props is None:
            props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
        props = {**props, 'animated': self.useblit}
        self._selection_artist = line = Line2D([], [], **props)
        self.ax.add_line(line)

        if handle_props is None:
            handle_props = dict(markeredgecolor='k',
                                markerfacecolor=props.get('color', 'k'))
        self._handle_props = handle_props
        self._polygon_handles = ToolHandles(self.ax, [], [],
                                            useblit=self.useblit,
                                            marker_props=self._handle_props)

        self._active_handle_idx = -1
        self.grab_range = grab_range

        self.set_visible(True)
        self._draw_box = draw_bounding_box
        self._box = None

        if box_handle_props is None:
            box_handle_props = {}
        self._box_handle_props = self._handle_props.update(box_handle_props)
        self._box_props = box_props

    def _get_bbox(self):
        return self._selection_artist.get_bbox()

    def _add_box(self):
        self._box = RectangleSelector(self.ax,
                                      useblit=self.useblit,
                                      grab_range=self.grab_range,
                                      handle_props=self._box_handle_props,
                                      props=self._box_props,
                                      interactive=True)
        self._box._state_modifier_keys.pop('rotate')
        self._box.connect_event('motion_notify_event', self._scale_polygon)
        self._update_box()
        # Set state that prevents the RectangleSelector from being created
        # by the user
        self._box._allow_creation = False
        self._box._selection_completed = True
        self._draw_polygon()

    def _remove_box(self):
        if self._box is not None:
            self._box.set_visible(False)
            self._box = None

    def _update_box(self):
        # Update selection box extents to the extents of the polygon
        if self._box is not None:
            bbox = self._get_bbox()
            self._box.extents = [bbox.x0, bbox.x1, bbox.y0, bbox.y1]
            # Save a copy
            self._old_box_extents = self._box.extents

    def _scale_polygon(self, event):
        """
        Scale the polygon selector points when the bounding box is moved or
        scaled.

        This is set as a callback on the bounding box RectangleSelector.
        """
        if not self._selection_completed:
            return

        if self._old_box_extents == self._box.extents:
            return

        # Create transform from old box to new box
        x1, y1, w1, h1 = self._box._rect_bbox
        old_bbox = self._get_bbox()
        t = (transforms.Affine2D()
             .translate(-old_bbox.x0, -old_bbox.y0)
             .scale(1 / old_bbox.width, 1 / old_bbox.height)
             .scale(w1, h1)
             .translate(x1, y1))

        # Update polygon verts.  Must be a list of tuples for consistency.
        new_verts = [(x, y) for x, y in t.transform(np.array(self.verts))]
        self._xys = [*new_verts, new_verts[0]]
        self._draw_polygon()
        self._old_box_extents = self._box.extents

    @property
    def _handles_artists(self):
        return self._polygon_handles.artists

    def _remove_vertex(self, i):
        """Remove vertex with index i."""
        if (len(self._xys) > 2 and
                self._selection_completed and
                i in (0, len(self._xys) - 1)):
            # If selecting the first or final vertex, remove both first and
            # last vertex as they are the same for a closed polygon
            self._xys.pop(0)
            self._xys.pop(-1)
            # Close the polygon again by appending the new first vertex to the
            # end
            self._xys.append(self._xys[0])
        else:
            self._xys.pop(i)
        if len(self._xys) <= 2:
            # If only one point left, return to incomplete state to let user
            # start drawing again
            self._selection_completed = False
            self._remove_box()

    def _press(self, event):
        """Button press event handler."""
        # Check for selection of a tool handle.
        if ((self._selection_completed or 'move_vertex' in self._state)
                and len(self._xys) > 0):
            h_idx, h_dist = self._polygon_handles.closest(event.x, event.y)
            if h_dist < self.grab_range:
                self._active_handle_idx = h_idx
        # Save the vertex positions at the time of the press event (needed to
        # support the 'move_all' state modifier).
        self._xys_at_press = self._xys.copy()

    def _release(self, event):
        """Button release event handler."""
        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self._draw_polygon()
            self._active_handle_idx = -1

        # Complete the polygon.
        elif len(self._xys) > 3 and self._xys[-1] == self._xys[0]:
            self._selection_completed = True
            if self._draw_box and self._box is None:
                self._add_box()

        # Place new vertex.
        elif (not self._selection_completed
              and 'move_all' not in self._state
              and 'move_vertex' not in self._state):
            self._xys.insert(-1, self._get_data_coords(event))

        if self._selection_completed:
            self.onselect(self.verts)

    def onmove(self, event):
        """Cursor move event handler and validator."""
        # Method overrides _SelectorWidget.onmove because the polygon selector
        # needs to process the move callback even if there is no button press.
        # _SelectorWidget.onmove include logic to ignore move event if
        # _eventpress is None.
        if self.ignore(event):
            # Hide the cursor when interactive zoom/pan is active
            if not self.canvas.widgetlock.available(self) and self._xys:
                self._xys[-1] = (np.nan, np.nan)
                self._draw_polygon()
            return False

        else:
            event = self._clean_event(event)
            self._onmove(event)
            return True

    def _onmove(self, event):
        """Cursor move event handler."""
        # Move the active vertex (ToolHandle).
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            self._xys[idx] = self._get_data_coords(event)
            # Also update the end of the polygon line if the first vertex is
            # the active handle and the polygon is completed.
            if idx == 0 and self._selection_completed:
                self._xys[-1] = self._get_data_coords(event)

        # Move all vertices.
        elif 'move_all' in self._state and self._eventpress:
            xdata, ydata = self._get_data_coords(event)
            dx = xdata - self._eventpress.xdata
            dy = ydata - self._eventpress.ydata
            for k in range(len(self._xys)):
                x_at_press, y_at_press = self._xys_at_press[k]
                self._xys[k] = x_at_press + dx, y_at_press + dy

        # Do nothing if completed or waiting for a move.
        elif (self._selection_completed
              or 'move_vertex' in self._state or 'move_all' in self._state):
            return

        # Position pending vertex.
        else:
            # Calculate distance to the start vertex.
            x0, y0 = \
                self._selection_artist.get_transform().transform(self._xys[0])
            v0_dist = np.hypot(x0 - event.x, y0 - event.y)
            # Lock on to the start vertex if near it and ready to complete.
            if len(self._xys) > 3 and v0_dist < self.grab_range:
                self._xys[-1] = self._xys[0]
            else:
                self._xys[-1] = self._get_data_coords(event)

        self._draw_polygon()

    def _on_key_press(self, event):
        """Key press event handler."""
        # Remove the pending vertex if entering the 'move_vertex' or
        # 'move_all' mode
        if (not self._selection_completed
                and ('move_vertex' in self._state or
                     'move_all' in self._state)):
            self._xys.pop()
            self._draw_polygon()

    def _on_key_release(self, event):
        """Key release event handler."""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if (not self._selection_completed
                and
                (event.key == self._state_modifier_keys.get('move_vertex')
                 or event.key == self._state_modifier_keys.get('move_all'))):
            self._xys.append(self._get_data_coords(event))
            self._draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self._state_modifier_keys.get('clear'):
            event = self._clean_event(event)
            self._xys = [self._get_data_coords(event)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)

    def _draw_polygon_without_update(self):
        """Redraw the polygon based on new vertex positions, no update()."""
        xs, ys = zip(*self._xys) if self._xys else ([], [])
        self._selection_artist.set_data(xs, ys)
        self._update_box()
        # Only show one tool handle at the start and end vertex of the polygon
        # if the polygon is completed or the user is locked on to the start
        # vertex.
        if (self._selection_completed
                or (len(self._xys) > 3
                    and self._xys[-1] == self._xys[0])):
            self._polygon_handles.set_data(xs[:-1], ys[:-1])
        else:
            self._polygon_handles.set_data(xs, ys)

    def _draw_polygon(self):
        """Redraw the polygon based on the new vertex positions."""
        self._draw_polygon_without_update()
        self.update()

    @property
    def verts(self):
        """The polygon vertices, as a list of ``(x, y)`` pairs."""
        return self._xys[:-1]

    @verts.setter
    def verts(self, xys):
        """
        Set the polygon vertices.

        This will remove any preexisting vertices, creating a complete polygon
        with the new vertices.
        """
        self._xys = [*xys, xys[0]]
        self._selection_completed = True
        self.set_visible(True)
        if self._draw_box and self._box is None:
            self._add_box()
        self._draw_polygon()

    def _clear_without_update(self):
        self._selection_completed = False
        self._xys = [(0, 0)]
        self._draw_polygon_without_update()


class Lasso(AxesWidget):
    """
    Selection curve of an arbitrary shape.

    The selected path can be used in conjunction with
    `~matplotlib.path.Path.contains_point` to select data points from an image.

    Unlike `LassoSelector`, this must be initialized with a starting
    point *xy*, and the `Lasso` events are destroyed upon release.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The parent Axes for the widget.
    xy : (float, float)
        Coordinates of the start of the lasso.
    callback : callable
        Whenever the lasso is released, the *callback* function is called and
        passed the vertices of the selected path.
    useblit : bool, default: True
        Whether to use blitting for faster drawing (if supported by the
        backend). See the tutorial :ref:`blitting`
        for details.
    props: dict, optional
        Lasso line properties. See `.Line2D` for valid properties.
        Default *props* are::

            {'linestyle' : '-', 'color' : 'black', 'lw' : 2}

        .. versionadded:: 3.9
    """
    def __init__(self, ax, xy, callback, *, useblit=True, props=None):
        super().__init__(ax)

        self.useblit = useblit and self.canvas.supports_blit
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        style = {'linestyle': '-', 'color': 'black', 'lw': 2}

        if props is not None:
            style.update(props)

        x, y = xy
        self.verts = [(x, y)]
        self.line = Line2D([x], [y], **style)
        self.ax.add_line(self.line)
        self.callback = callback
        self.connect_event('button_release_event', self.onrelease)
        self.connect_event('motion_notify_event', self.onmove)

    def onrelease(self, event):
        if self.ignore(event):
            return
        if self.verts is not None:
            self.verts.append(self._get_data_coords(event))
            if len(self.verts) > 2:
                self.callback(self.verts)
            self.line.remove()
        self.verts = None
        self.disconnect_events()

    def onmove(self, event):
        if (self.ignore(event)
                or self.verts is None
                or event.button != 1
                or not self.ax.contains(event)[0]):
            return
        self.verts.append(self._get_data_coords(event))
        self.line.set_data(list(zip(*self.verts)))

        if self.useblit:
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()
<|endoftext|>"""
Classes for including text in a figure.
"""

import functools
import logging
import math
from numbers import Real
import weakref

import numpy as np

import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
    Affine2D, Bbox, BboxBase, BboxTransformTo, IdentityTransform, Transform)


_log = logging.getLogger(__name__)


def _get_textbox(text, renderer):
    """
    Calculate the bounding box of the text.

    The bbox position takes text rotation into account, but the width and
    height are those of the unrotated box (unlike `.Text.get_window_extent`).
    """
    # TODO : This function may move into the Text class as a method. As a
    # matter of fact, the information from the _get_textbox function
    # should be available during the Text._get_layout() call, which is
    # called within the _get_textbox. So, it would better to move this
    # function as a method with some refactoring of _get_layout method.

    projected_xs = []
    projected_ys = []

    theta = np.deg2rad(text.get_rotation())
    tr = Affine2D().rotate(-theta)

    _, parts, d = text._get_layout(renderer)

    for t, wh, x, y in parts:
        w, h = wh

        xt1, yt1 = tr.transform((x, y))
        yt1 -= d
        xt2, yt2 = xt1 + w, yt1 + h

        projected_xs.extend([xt1, xt2])
        projected_ys.extend([yt1, yt2])

    xt_box, yt_box = min(projected_xs), min(projected_ys)
    w_box, h_box = max(projected_xs) - xt_box, max(projected_ys) - yt_box

    x_box, y_box = Affine2D().rotate(theta).transform((xt_box, yt_box))

    return x_box, y_box, w_box, h_box


def _get_text_metrics_with_cache(renderer, text, fontprop, ismath, dpi):
    """Call ``renderer.get_text_width_height_descent``, caching the results."""
    # Cached based on a copy of fontprop so that later in-place mutations of
    # the passed-in argument do not mess up the cache.
    return _get_text_metrics_with_cache_impl(
        weakref.ref(renderer), text, fontprop.copy(), ismath, dpi)


@functools.lru_cache(4096)
def _get_text_metrics_with_cache_impl(
        renderer_ref, text, fontprop, ismath, dpi):
    # dpi is unused, but participates in cache invalidation (via the renderer).
    return renderer_ref().get_text_width_height_descent(text, fontprop, ismath)


@_docstring.interpd
@_api.define_aliases({
    "color": ["c"],
    "fontproperties": ["font", "font_properties"],
    "fontfamily": ["family"],
    "fontname": ["name"],
    "fontsize": ["size"],
    "fontstretch": ["stretch"],
    "fontstyle": ["style"],
    "fontvariant": ["variant"],
    "fontweight": ["weight"],
    "horizontalalignment": ["ha"],
    "verticalalignment": ["va"],
    "multialignment": ["ma"],
})
class Text(Artist):
    """Handle storing and drawing of text in window or data coordinates."""

    zorder = 3
    _charsize_cache = dict()

    def __repr__(self):
        return f"Text({self._x}, {self._y}, {self._text!r})"

    def __init__(self,
                 x=0, y=0, text='', *,
                 color=None,           # defaults to rc params
                 verticalalignment='baseline',
                 horizontalalignment='left',
                 multialignment=None,
                 fontproperties=None,  # defaults to FontProperties()
                 rotation=None,
                 linespacing=None,
                 rotation_mode=None,
                 usetex=None,          # defaults to rcParams['text.usetex']
                 wrap=False,
                 transform_rotates_text=False,
                 parse_math=None,    # defaults to rcParams['text.parse_math']
                 antialiased=None,  # defaults to rcParams['text.antialiased']
                 **kwargs
                 ):
        """
        Create a `.Text` instance at *x*, *y* with string *text*.

        The text is aligned relative to the anchor point (*x*, *y*) according
        to ``horizontalalignment`` (default: 'left') and ``verticalalignment``
        (default: 'baseline'). See also
        :doc:`/gallery/text_labels_and_annotations/text_alignment`.

        While Text accepts the 'label' keyword argument, by default it is not
        added to the handles of a legend.

        Valid keyword arguments are:

        %(Text:kwdoc)s
        """
        super().__init__()
        self._x, self._y = x, y
        self._text = ''
        self._reset_visual_defaults(
            text=text,
            color=color,
            fontproperties=fontproperties,
            usetex=usetex,
            parse_math=parse_math,
            wrap=wrap,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            multialignment=multialignment,
            rotation=rotation,
            transform_rotates_text=transform_rotates_text,
            linespacing=linespacing,
            rotation_mode=rotation_mode,
            antialiased=antialiased
        )
        self.update(kwargs)

    def _reset_visual_defaults(
        self,
        text='',
        color=None,
        fontproperties=None,
        usetex=None,
        parse_math=None,
        wrap=False,
        verticalalignment='baseline',
        horizontalalignment='left',
        multialignment=None,
        rotation=None,
        transform_rotates_text=False,
        linespacing=None,
        rotation_mode=None,
        antialiased=None
    ):
        self.set_text(text)
        self.set_color(mpl._val_or_rc(color, "text.color"))
        self.set_fontproperties(fontproperties)
        self.set_usetex(usetex)
        self.set_parse_math(mpl._val_or_rc(parse_math, 'text.parse_math'))
        self.set_wrap(wrap)
        self.set_verticalalignment(verticalalignment)
        self.set_horizontalalignment(horizontalalignment)
        self._multialignment = multialignment
        self.set_rotation(rotation)
        self._transform_rotates_text = transform_rotates_text
        self._bbox_patch = None  # a FancyBboxPatch instance
        self._renderer = None
        if linespacing is None:
            linespacing = 1.2  # Maybe use rcParam later.
        self.set_linespacing(linespacing)
        self.set_rotation_mode(rotation_mode)
        self.set_antialiased(antialiased if antialiased is not None else
                             mpl.rcParams['text.antialiased'])

    def update(self, kwargs):
        # docstring inherited
        ret = []
        kwargs = cbook.normalize_kwargs(kwargs, Text)
        sentinel = object()  # bbox can be None, so use another sentinel.
        # Update fontproperties first, as it has lowest priority.
        fontproperties = kwargs.pop("fontproperties", sentinel)
        if fontproperties is not sentinel:
            ret.append(self.set_fontproperties(fontproperties))
        # Update bbox last, as it depends on font properties.
        bbox = kwargs.pop("bbox", sentinel)
        ret.extend(super().update(kwargs))
        if bbox is not sentinel:
            ret.append(self.set_bbox(bbox))
        return ret

    def __getstate__(self):
        d = super().__getstate__()
        # remove the cached _renderer (if it exists)
        d['_renderer'] = None
        return d

    def contains(self, mouseevent):
        """
        Return whether the mouse event occurred inside the axis-aligned
        bounding-box of the text.
        """
        if (self._different_canvas(mouseevent) or not self.get_visible()
                or self._renderer is None):
            return False, {}
        # Explicitly use Text.get_window_extent(self) and not
        # self.get_window_extent() so that Annotation.contains does not
        # accidentally cover the entire annotation bounding box.
        bbox = Text.get_window_extent(self)
        inside = (bbox.x0 <= mouseevent.x <= bbox.x1
                  and bbox.y0 <= mouseevent.y <= bbox.y1)
        cattr = {}
        # if the text has a surrounding patch, also check containment for it,
        # and merge the results with the results for the text.
        if self._bbox_patch:
            patch_inside, patch_cattr = self._bbox_patch.contains(mouseevent)
            inside = inside or patch_inside
            cattr["bbox_patch"] = patch_cattr
        return inside, cattr

    def _get_xy_display(self):
        """
        Get the (possibly unit converted) transformed x, y in display coords.
        """
        x, y = self.get_unitless_position()
        return self.get_transform().transform((x, y))

    def _get_multialignment(self):
        if self._multialignment is not None:
            return self._multialignment
        else:
            return self._horizontalalignment

    def _char_index_at(self, x):
        """
        Calculate the index closest to the coordinate x in display space.

        The position of text[index] is assumed to be the sum of the widths
        of all preceding characters text[:index].

        This works only on single line texts.
        """
        if not self._text:
            return 0

        text = self._text

        fontproperties = str(self._fontproperties)
        if fontproperties not in Text._charsize_cache:
            Text._charsize_cache[fontproperties] = dict()

        charsize_cache = Text._charsize_cache[fontproperties]
        for char in set(text):
            if char not in charsize_cache:
                self.set_text(char)
                bb = self.get_window_extent()
                charsize_cache[char] = bb.x1 - bb.x0

        self.set_text(text)
        bb = self.get_window_extent()

        size_accum = np.cumsum([0] + [charsize_cache[x] for x in text])
        std_x = x - bb.x0
        return (np.abs(size_accum - std_x)).argmin()

    def get_rotation(self):
        """Return the text angle in degrees between 0 and 360."""
        if self.get_transform_rotates_text():
            return self.get_transform().transform_angles(
                [self._rotation], [self.get_unitless_position()]).item(0)
        else:
            return self._rotation

    def get_transform_rotates_text(self):
        """
        Return whether rotations of the transform affect the text direction.
        """
        return self._transform_rotates_text

    def set_rotation_mode(self, m):
        """
        Set text rotation mode.

        Parameters
        ----------
        m : {None, 'default', 'anchor'}
            If ``"default"``, the text will be first rotated, then aligned according
            to their horizontal and vertical alignments.  If ``"anchor"``, then
            alignment occurs before rotation. Passing ``None`` will set the rotation
            mode to ``"default"``.
        """
        if m is None:
            m = "default"
        else:
            _api.check_in_list(("anchor", "default"), rotation_mode=m)
        self._rotation_mode = m
        self.stale = True

    def get_rotation_mode(self):
        """Return the text rotation mode."""
        return self._rotation_mode

    def set_antialiased(self, antialiased):
        """
        Set whether to use antialiased rendering.

        Parameters
        ----------
        antialiased : bool

        Notes
        -----
        Antialiasing will be determined by :rc:`text.antialiased`
        and the parameter *antialiased* will have no effect if the text contains
        math expressions.
        """
        self._antialiased = antialiased
        self.stale = True

    def get_antialiased(self):
        """Return whether antialiased rendering is used."""
        return self._antialiased

    def update_from(self, other):
        # docstring inherited
        super().update_from(other)
        self._color = other._color
        self._multialignment = other._multialignment
        self._verticalalignment = other._verticalalignment
        self._horizontalalignment = other._horizontalalignment
        self._fontproperties = other._fontproperties.copy()
        self._usetex = other._usetex
        self._rotation = other._rotation
        self._transform_rotates_text = other._transform_rotates_text
        self._picker = other._picker
        self._linespacing = other._linespacing
        self._antialiased = other._antialiased
        self.stale = True

    def _get_layout(self, renderer):
        """
        Return the extent (bbox) of the text together with
        multiple-alignment information. Note that it returns an extent
        of a rotated text when necessary.
        """
        thisx, thisy = 0.0, 0.0
        lines = self._get_wrapped_text().split("\n")  # Ensures lines is not empty.

        ws = []
        hs = []
        xs = []
        ys = []

        # Full vertical extent of font, including ascenders and descenders:
        _, lp_h, lp_d = _get_text_metrics_with_cache(
            renderer, "lp", self._fontproperties,
            ismath="TeX" if self.get_usetex() else False,
            dpi=self.get_figure(root=True).dpi)
        min_dy = (lp_h - lp_d) * self._linespacing

        for i, line in enumerate(lines):
            clean_line, ismath = self._preprocess_math(line)
            if clean_line:
                w, h, d = _get_text_metrics_with_cache(
                    renderer, clean_line, self._fontproperties,
                    ismath=ismath, dpi=self.get_figure(root=True).dpi)
            else:
                w = h = d = 0

            # For multiline text, increase the line spacing when the text
            # net-height (excluding baseline) is larger than that of a "l"
            # (e.g., use of superscripts), which seems what TeX does.
            h = max(h, lp_h)
            d = max(d, lp_d)

            ws.append(w)
            hs.append(h)

            # Metrics of the last line that are needed later:
            baseline = (h - d) - thisy

            if i == 0:
                # position at baseline
                thisy = -(h - d)
            else:
                # put baseline a good distance from bottom of previous line
                thisy -= max(min_dy, (h - d) * self._linespacing)

            xs.append(thisx)  # == 0.
            ys.append(thisy)

            thisy -= d

        # Metrics of the last line that are needed later:
        descent = d

        # Bounding box definition:
        width = max(ws)
        xmin = 0
        xmax = width
        ymax = 0
        ymin = ys[-1] - descent  # baseline of last line minus its descent

        # get the rotation matrix
        M = Affine2D().rotate_deg(self.get_rotation())

        # now offset the individual text lines within the box
        malign = self._get_multialignment()
        if malign == 'left':
            offset_layout = [(x, y) for x, y in zip(xs, ys)]
        elif malign == 'center':
            offset_layout = [(x + width / 2 - w / 2, y)
                             for x, y, w in zip(xs, ys, ws)]
        elif malign == 'right':
            offset_layout = [(x + width - w, y)
                             for x, y, w in zip(xs, ys, ws)]

        # the corners of the unrotated bounding box
        corners_horiz = np.array(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])

        # now rotate the bbox
        corners_rotated = M.transform(corners_horiz)
        # compute the bounds of the rotated box
        xmin = corners_rotated[:, 0].min()
        xmax = corners_rotated[:, 0].max()
        ymin = corners_rotated[:, 1].min()
        ymax = corners_rotated[:, 1].max()
        width = xmax - xmin
        height = ymax - ymin

        # Now move the box to the target position offset the display
        # bbox by alignment
        halign = self._horizontalalignment
        valign = self._verticalalignment

        rotation_mode = self.get_rotation_mode()
        if rotation_mode != "anchor":
            # compute the text location in display coords and the offsets
            # necessary to align the bbox with that location
            if halign == 'center':
                offsetx = (xmin + xmax) / 2
            elif halign == 'right':
                offsetx = xmax
            else:
                offsetx = xmin

            if valign == 'center':
                offsety = (ymin + ymax) / 2
            elif valign == 'top':
                offsety = ymax
            elif valign == 'baseline':
                offsety = ymin + descent
            elif valign == 'center_baseline':
                offsety = ymin + height - baseline / 2.0
            else:
                offsety = ymin
        else:
            xmin1, ymin1 = corners_horiz[0]
            xmax1, ymax1 = corners_horiz[2]

            if halign == 'center':
                offsetx = (xmin1 + xmax1) / 2.0
            elif halign == 'right':
                offsetx = xmax1
            else:
                offsetx = xmin1

            if valign == 'center':
                offsety = (ymin1 + ymax1) / 2.0
            elif valign == 'top':
                offsety = ymax1
            elif valign == 'baseline':
                offsety = ymax1 - baseline
            elif valign == 'center_baseline':
                offsety = ymax1 - baseline / 2.0
            else:
                offsety = ymin1

            offsetx, offsety = M.transform((offsetx, offsety))

        xmin -= offsetx
        ymin -= offsety

        bbox = Bbox.from_bounds(xmin, ymin, width, height)

        # now rotate the positions around the first (x, y) position
        xys = M.transform(offset_layout) - (offsetx, offsety)

        return bbox, list(zip(lines, zip(ws, hs), *xys.T)), descent

    def set_bbox(self, rectprops):
        """
        Draw a bounding box around self.

        Parameters
        ----------
        rectprops : dict with properties for `.patches.FancyBboxPatch`
             The default boxstyle is 'square'. The mutation
             scale of the `.patches.FancyBboxPatch` is set to the fontsize.

        Examples
        --------
        ::

            t.set_bbox(dict(facecolor='red', alpha=0.5))
        """

        if rectprops is not None:
            props = rectprops.copy()
            boxstyle = props.pop("boxstyle", None)
            pad = props.pop("pad", None)
            if boxstyle is None:
                boxstyle = "square"
                if pad is None:
                    pad = 4  # points
                pad /= self.get_size()  # to fraction of font size
            else:
                if pad is None:
                    pad = 0.3
            # boxstyle could be a callable or a string
            if isinstance(boxstyle, str) and "pad" not in boxstyle:
                boxstyle += ",pad=%0.2f" % pad
            self._bbox_patch = FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle=boxstyle, transform=IdentityTransform(), **props)
        else:
            self._bbox_patch = None

        self._update_clip_properties()

    def get_bbox_patch(self):
        """
        Return the bbox Patch, or None if the `.patches.FancyBboxPatch`
        is not made.
        """
        return self._bbox_patch

    def update_bbox_position_size(self, renderer):
        """
        Update the location and the size of the bbox.

        This method should be used when the position and size of the bbox needs
        to be updated before actually drawing the bbox.
        """
        if self._bbox_patch:
            # don't use self.get_unitless_position here, which refers to text
            # position in Text:
            posx = float(self.convert_xunits(self._x))
            posy = float(self.convert_yunits(self._y))
            posx, posy = self.get_transform().transform((posx, posy))

            x_box, y_box, w_box, h_box = _get_textbox(self, renderer)
            self._bbox_patch.set_bounds(0., 0., w_box, h_box)
            self._bbox_patch.set_transform(
                Affine2D()
                .rotate_deg(self.get_rotation())
                .translate(posx + x_box, posy + y_box))
            fontsize_in_pixel = renderer.points_to_pixels(self.get_size())
            self._bbox_patch.set_mutation_scale(fontsize_in_pixel)

    def _update_clip_properties(self):
        if self._bbox_patch:
            clipprops = dict(clip_box=self.clipbox,
                             clip_path=self._clippath,
                             clip_on=self._clipon)
            self._bbox_patch.update(clipprops)

    def set_clip_box(self, clipbox):
        # docstring inherited.
        super().set_clip_box(clipbox)
        self._update_clip_properties()

    def set_clip_path(self, path, transform=None):
        # docstring inherited.
        super().set_clip_path(path, transform)
        self._update_clip_properties()

    def set_clip_on(self, b):
        # docstring inherited.
        super().set_clip_on(b)
        self._update_clip_properties()

    def get_wrap(self):
        """Return whether the text can be wrapped."""
        return self._wrap

    def set_wrap(self, wrap):
        """
        Set whether the text can be wrapped.

        Wrapping makes sure the text is confined to the (sub)figure box. It
        does not take into account any other artists.

        Parameters
        ----------
        wrap : bool

        Notes
        -----
        Wrapping does not work together with
        ``savefig(..., bbox_inches='tight')`` (which is also used internally
        by ``%matplotlib inline`` in IPython/Jupyter). The 'tight' setting
        rescales the canvas to accommodate all content and happens before
        wrapping.
        """
        self._wrap = wrap

    def _get_wrap_line_width(self):
        """
        Return the maximum line width for wrapping text based on the current
        orientation.
        """
        x0, y0 = self.get_transform().transform(self.get_position())
        figure_box = self.get_figure().get_window_extent()

        # Calculate available width based on text alignment
        alignment = self.get_horizontalalignment()
        self.set_rotation_mode('anchor')
        rotation = self.get_rotation()

        left = self._get_dist_to_box(rotation, x0, y0, figure_box)
        right = self._get_dist_to_box(
            (180 + rotation) % 360, x0, y0, figure_box)

        if alignment == 'left':
            line_width = left
        elif alignment == 'right':
            line_width = right
        else:
            line_width = 2 * min(left, right)

        return line_width

    def _get_dist_to_box(self, rotation, x0, y0, figure_box):
        """
        Return the distance from the given points to the boundaries of a
        rotated box, in pixels.
        """
        if rotation > 270:
            quad = rotation - 270
            h1 = (y0 - figure_box.y0) / math.cos(math.radians(quad))
            h2 = (figure_box.x1 - x0) / math.cos(math.radians(90 - quad))
        elif rotation > 180:
            quad = rotation - 180
            h1 = (x0 - figure_box.x0) / math.cos(math.radians(quad))
            h2 = (y0 - figure_box.y0) / math.cos(math.radians(90 - quad))
        elif rotation > 90:
            quad = rotation - 90
            h1 = (figure_box.y1 - y0) / math.cos(math.radians(quad))
            h2 = (x0 - figure_box.x0) / math.cos(math.radians(90 - quad))
        else:
            h1 = (figure_box.x1 - x0) / math.cos(math.radians(rotation))
            h2 = (figure_box.y1 - y0) / math.cos(math.radians(90 - rotation))

        return min(h1, h2)

    def _get_rendered_text_width(self, text):
        """
        Return the width of a given text string, in pixels.
        """

        w, h, d = _get_text_metrics_with_cache(
            self._renderer, text, self.get_fontproperties(),
            cbook.is_math_text(text),
            self.get_figure(root=True).dpi)
        return math.ceil(w)

    def _get_wrapped_text(self):
        """
        Return a copy of the text string with new lines added so that the text
        is wrapped relative to the parent figure (if `get_wrap` is True).
        """
        if not self.get_wrap():
            return self.get_text()

        # Not fit to handle breaking up latex syntax correctly, so
        # ignore latex for now.
        if self.get_usetex():
            return self.get_text()

        # Build the line incrementally, for a more accurate measure of length
        line_width = self._get_wrap_line_width()
        wrapped_lines = []

        # New lines in the user's text force a split
        unwrapped_lines = self.get_text().split('\n')

        # Now wrap each individual unwrapped line
        for unwrapped_line in unwrapped_lines:

            sub_words = unwrapped_line.split(' ')
            # Remove items from sub_words as we go, so stop when empty
            while len(sub_words) > 0:
                if len(sub_words) == 1:
                    # Only one word, so just add it to the end
                    wrapped_lines.append(sub_words.pop(0))
                    continue

                for i in range(2, len(sub_words) + 1):
                    # Get width of all words up to and including here
                    line = ' '.join(sub_words[:i])
                    current_width = self._get_rendered_text_width(line)

                    # If all these words are too wide, append all not including
                    # last word
                    if current_width > line_width:
                        wrapped_lines.append(' '.join(sub_words[:i - 1]))
                        sub_words = sub_words[i - 1:]
                        break

                    # Otherwise if all words fit in the width, append them all
                    elif i == len(sub_words):
                        wrapped_lines.append(' '.join(sub_words[:i]))
                        sub_words = []
                        break

        return '\n'.join(wrapped_lines)

    @artist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited

        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible():
            return
        if self.get_text() == '':
            return

        renderer.open_group('text', self.get_gid())

        with self._cm_set(text=self._get_wrapped_text()):
            bbox, info, descent = self._get_layout(renderer)
            trans = self.get_transform()

            # don't use self.get_position here, which refers to text
            # position in Text:
            x, y = self._x, self._y
            if np.ma.is_masked(x):
                x = np.nan
            if np.ma.is_masked(y):
                y = np.nan
            posx = float(self.convert_xunits(x))
            posy = float(self.convert_yunits(y))
            posx, posy = trans.transform((posx, posy))
            if np.isnan(posx) or np.isnan(posy):
                return  # don't throw a warning here
            if not np.isfinite(posx) or not np.isfinite(posy):
                _log.warning("posx and posy should be finite values")
                return
            canvasw, canvash = renderer.get_canvas_width_height()

            # Update the location and size of the bbox
            # (`.patches.FancyBboxPatch`), and draw it.
            if self._bbox_patch:
                self.update_bbox_position_size(renderer)
                self._bbox_patch.draw(renderer)

            gc = renderer.new_gc()
            gc.set_foreground(self.get_color())
            gc.set_alpha(self.get_alpha())
            gc.set_url(self._url)
            gc.set_antialiased(self._antialiased)
            self._set_gc_clip(gc)

            angle = self.get_rotation()

            for line, wh, x, y in info:

                mtext = self if len(info) == 1 else None
                x = x + posx
                y = y + posy
                if renderer.flipy():
                    y = canvash - y
                clean_line, ismath = self._preprocess_math(line)

                if self.get_path_effects():
                    from matplotlib.patheffects import PathEffectRenderer
                    textrenderer = PathEffectRenderer(
                        self.get_path_effects(), renderer)
                else:
                    textrenderer = renderer

                if self.get_usetex():
                    textrenderer.draw_tex(gc, x, y, clean_line,
                                          self._fontproperties, angle,
                                          mtext=mtext)
                else:
                    textrenderer.draw_text(gc, x, y, clean_line,
                                           self._fontproperties, angle,
                                           ismath=ismath, mtext=mtext)

        gc.restore()
        renderer.close_group('text')
        self.stale = False

    def get_color(self):
        """Return the color of the text."""
        return self._color

    def get_fontproperties(self):
        """Return the `.font_manager.FontProperties`."""
        return self._fontproperties

    def get_fontfamily(self):
        """
        Return the list of font families used for font lookup.

        See Also
        --------
        .font_manager.FontProperties.get_family
        """
        return self._fontproperties.get_family()

    def get_fontname(self):
        """
        Return the font name as a string.

        See Also
        --------
        .font_manager.FontProperties.get_name
        """
        return self._fontproperties.get_name()

    def get_fontstyle(self):
        """
        Return the font style as a string.

        See Also
        --------
        .font_manager.FontProperties.get_style
        """
        return self._fontproperties.get_style()

    def get_fontsize(self):
        """
        Return the font size as an integer.

        See Also
        --------
        .font_manager.FontProperties.get_size_in_points
        """
        return self._fontproperties.get_size_in_points()

    def get_fontvariant(self):
        """
        Return the font variant as a string.

        See Also
        --------
        .font_manager.FontProperties.get_variant
        """
        return self._fontproperties.get_variant()

    def get_fontweight(self):
        """
        Return the font weight as a string or a number.

        See Also
        --------
        .font_manager.FontProperties.get_weight
        """
        return self._fontproperties.get_weight()

    def get_stretch(self):
        """
        Return the font stretch as a string or a number.

        See Also
        --------
        .font_manager.FontProperties.get_stretch
        """
        return self._fontproperties.get_stretch()

    def get_horizontalalignment(self):
        """
        Return the horizontal alignment as a string.  Will be one of
        'left', 'center' or 'right'.
        """
        return self._horizontalalignment

    def get_unitless_position(self):
        """Return the (x, y) unitless position of the text."""
        # This will get the position with all unit information stripped away.
        # This is here for convenience since it is done in several locations.
        x = float(self.convert_xunits(self._x))
        y = float(self.convert_yunits(self._y))
        return x, y

    def get_position(self):
        """Return the (x, y) position of the text."""
        # This should return the same data (possible unitized) as was
        # specified with 'set_x' and 'set_y'.
        return self._x, self._y

    def get_text(self):
        """Return the text string."""
        return self._text

    def get_verticalalignment(self):
        """
        Return the vertical alignment as a string.  Will be one of
        'top', 'center', 'bottom', 'baseline' or 'center_baseline'.
        """
        return self._verticalalignment

    def get_window_extent(self, renderer=None, dpi=None):
        """
        Return the `.Bbox` bounding the text, in display units.

        In addition to being used internally, this is useful for specifying
        clickable regions in a png file on a web page.

        Parameters
        ----------
        renderer : Renderer, optional
            A renderer is needed to compute the bounding box.  If the artist
            has already been drawn, the renderer is cached; thus, it is only
            necessary to pass this argument when calling `get_window_extent`
            before the first draw.  In practice, it is usually easier to
            trigger a draw first, e.g. by calling
            `~.Figure.draw_without_rendering` or ``plt.show()``.

        dpi : float, optional
            The dpi value for computing the bbox, defaults to
            ``self.get_figure(root=True).dpi`` (*not* the renderer dpi); should be set
            e.g. if to match regions with a figure saved with a custom dpi value.
        """
        if not self.get_visible():
            return Bbox.unit()

        fig = self.get_figure(root=True)
        if dpi is None:
            dpi = fig.dpi
        if self.get_text() == '':
            with cbook._setattr_cm(fig, dpi=dpi):
                tx, ty = self._get_xy_display()
                return Bbox.from_bounds(tx, ty, 0, 0)

        if renderer is not None:
            self._renderer = renderer
        if self._renderer is None:
            self._renderer = fig._get_renderer()
        if self._renderer is None:
            raise RuntimeError(
                "Cannot get window extent of text w/o renderer. You likely "
                "want to call 'figure.draw_without_rendering()' first.")

        with cbook._setattr_cm(fig, dpi=dpi):
            bbox, info, descent = self._get_layout(self._renderer)
            x, y = self.get_unitless_position()
            x, y = self.get_transform().transform((x, y))
            bbox = bbox.translated(x, y)
            return bbox

    def set_backgroundcolor(self, color):
        """
        Set the background color of the text by updating the bbox.

        Parameters
        ----------
        color : :mpltype:`color`

        See Also
        --------
        .set_bbox : To change the position of the bounding box
        """
        if self._bbox_patch is None:
            self.set_bbox(dict(facecolor=color, edgecolor=color))
        else:
            self._bbox_patch.update(dict(facecolor=color))

        self._update_clip_properties()
        self.stale = True

    def set_color(self, color):
        """
        Set the foreground color of the text

        Parameters
        ----------
        color : :mpltype:`color`
        """
        # "auto" is only supported by axisartist, but we can just let it error
        # out at draw time for simplicity.
        if not cbook._str_equal(color, "auto"):
            mpl.colors._check_color_like(color=color)
        self._color = color
        self.stale = True

    def set_horizontalalignment(self, align):
        """
        Set the horizontal alignment relative to the anchor point.

        See also :doc:`/gallery/text_labels_and_annotations/text_alignment`.

        Parameters
        ----------
        align : {'left', 'center', 'right'}
        """
        _api.check_in_list(['center', 'right', 'left'], align=align)
        self._horizontalalignment = align
        self.stale = True

    def set_multialignment(self, align):
        """
        Set the text alignment for multiline texts.

        The layout of the bounding box of all the lines is determined by the
        horizontalalignment and verticalalignment properties. This property
        controls the alignment of the text lines within that box.

        Parameters
        ----------
        align : {'left', 'right', 'center'}
        """
        _api.check_in_list(['center', 'right', 'left'], align=align)
        self._multialignment = align
        self.stale = True

    def set_linespacing(self, spacing):
        """
        Set the line spacing as a multiple of the font size.

        The default line spacing is 1.2.

        Parameters
        ----------
        spacing : float (multiple of font size)
        """
        _api.check_isinstance(Real, spacing=spacing)
        self._linespacing = spacing
        self.stale = True

    def set_fontfamily(self, fontname):
        """
        Set the font family.  Can be either a single string, or a list of
        strings in decreasing priority.  Each string may be either a real font
        name or a generic font class name.  If the latter, the specific font
        names will be looked up in the corresponding rcParams.

        If a `Text` instance is constructed with ``fontfamily=None``, then the
        font is set to :rc:`font.family`, and the
        same is done when `set_fontfamily()` is called on an existing
        `Text` instance.

        Parameters
        ----------
        fontname : {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', \
'monospace'}

        See Also
        --------
        .font_manager.FontProperties.set_family
        """
        self._fontproperties.set_family(fontname)
        self.stale = True

    def set_fontvariant(self, variant):
        """
        Set the font variant.

        Parameters
        ----------
        variant : {'normal', 'small-caps'}

        See Also
        --------
        .font_manager.FontProperties.set_variant
        """
        self._fontproperties.set_variant(variant)
        self.stale = True

    def set_fontstyle(self, fontstyle):
        """
        Set the font style.

        Parameters
        ----------
        fontstyle : {'normal', 'italic', 'oblique'}

        See Also
        --------
        .font_manager.FontProperties.set_style
        """
        self._fontproperties.set_style(fontstyle)
        self.stale = True

    def set_fontsize(self, fontsize):
        """
        Set the font size.

        Parameters
        ----------
        fontsize : float or {'xx-small', 'x-small', 'small', 'medium', \
'large', 'x-large', 'xx-large'}
            If a float, the fontsize in points. The string values denote sizes
            relative to the default font size.

        See Also
        --------
        .font_manager.FontProperties.set_size
        """
        self._fontproperties.set_size(fontsize)
        self.stale = True

    def get_math_fontfamily(self):
        """
        Return the font family name for math text rendered by Matplotlib.

        The default value is :rc:`mathtext.fontset`.

        See Also
        --------
        set_math_fontfamily
        """
        return self._fontproperties.get_math_fontfamily()

    def set_math_fontfamily(self, fontfamily):
        """
        Set the font family for math text rendered by Matplotlib.

        This does only affect Matplotlib's own math renderer. It has no effect
        when rendering with TeX (``usetex=True``).

        Parameters
        ----------
        fontfamily : str
            The name of the font family.

            Available font families are defined in the
            :ref:`default matplotlibrc file
            <customizing-with-matplotlibrc-files>`.

        See Also
        --------
        get_math_fontfamily
        """
        self._fontproperties.set_math_fontfamily(fontfamily)

    def set_fontweight(self, weight):
        """
        Set the font weight.

        Parameters
        ----------
        weight : {a numeric value in range 0-1000, 'ultralight', 'light', \
'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', \
'demi', 'bold', 'heavy', 'extra bold', 'black'}

        See Also
        --------
        .font_manager.FontProperties.set_weight
        """
        self._fontproperties.set_weight(weight)
        self.stale = True

    def set_fontstretch(self, stretch):
        """
        Set the font stretch (horizontal condensation or expansion).

        Parameters
        ----------
        stretch : {a numeric value in range 0-1000, 'ultra-condensed', \
'extra-condensed', 'condensed', 'semi-condensed', 'normal', 'semi-expanded', \
'expanded', 'extra-expanded', 'ultra-expanded'}

        See Also
        --------
        .font_manager.FontProperties.set_stretch
        """
        self._fontproperties.set_stretch(stretch)
        self.stale = True

    def set_position(self, xy):
        """
        Set the (*x*, *y*) position of the text.

        Parameters
        ----------
        xy : (float, float)
        """
        self.set_x(xy[0])
        self.set_y(xy[1])

    def set_x(self, x):
        """
        Set the *x* position of the text.

        Parameters
        ----------
        x : float
        """
        self._x = x
        self.stale = True

    def set_y(self, y):
        """
        Set the *y* position of the text.

        Parameters
        ----------
        y : float
        """
        self._y = y
        self.stale = True

    def set_rotation(self, s):
        """
        Set the rotation of the text.

        Parameters
        ----------
        s : float or {'vertical', 'horizontal'}
            The rotation angle in degrees in mathematically positive direction
            (counterclockwise). 'horizontal' equals 0, 'vertical' equals 90.
        """
        if isinstance(s, Real):
            self._rotation = float(s) % 360
        elif cbook._str_equal(s, 'horizontal') or s is None:
            self._rotation = 0.
        elif cbook._str_equal(s, 'vertical'):
            self._rotation = 90.
        else:
            raise ValueError("rotation must be 'vertical', 'horizontal' or "
                             f"a number, not {s}")
        self.stale = True

    def set_transform_rotates_text(self, t):
        """
        Whether rotations of the transform affect the text direction.

        Parameters
        ----------
        t : bool
        """
        self._transform_rotates_text = t
        self.stale = True

    def set_verticalalignment(self, align):
        """
        Set the vertical alignment relative to the anchor point.

        See also :doc:`/gallery/text_labels_and_annotations/text_alignment`.

        Parameters
        ----------
        align : {'baseline', 'bottom', 'center', 'center_baseline', 'top'}
        """
        _api.check_in_list(
            ['top', 'bottom', 'center', 'baseline', 'center_baseline'],
            align=align)
        self._verticalalignment = align
        self.stale = True

    def set_text(self, s):
        r"""
        Set the text string *s*.

        It may contain newlines (``\n``) or math in LaTeX syntax.

        Parameters
        ----------
        s : object
            Any object gets converted to its `str` representation, except for
            ``None`` which is converted to an empty string.
        """
        s = '' if s is None else str(s)
        if s != self._text:
            self._text = s
            self.stale = True

    def _preprocess_math(self, s):
        """
        Return the string *s* after mathtext preprocessing, and the kind of
        mathtext support needed.

        - If *self* is configured to use TeX, return *s* unchanged except that
          a single space gets escaped, and the flag "TeX".
        - Otherwise, if *s* is mathtext (has an even number of unescaped dollar
          signs) and ``parse_math`` is not set to False, return *s* and the
          flag True.
        - Otherwise, return *s* with dollar signs unescaped, and the flag
          False.
        """
        if self.get_usetex():
            if s == " ":
                s = r"\ "
            return s, "TeX"
        elif not self.get_parse_math():
            return s, False
        elif cbook.is_math_text(s):
            return s, True
        else:
            return s.replace(r"\$", "$"), False

    def set_fontproperties(self, fp):
        """
        Set the font properties that control the text.

        Parameters
        ----------
        fp : `.font_manager.FontProperties` or `str` or `pathlib.Path`
            If a `str`, it is interpreted as a fontconfig pattern parsed by
            `.FontProperties`.  If a `pathlib.Path`, it is interpreted as the
            absolute path to a font file.
        """
        self._fontproperties = FontProperties._from_any(fp).copy()
        self.stale = True

    @_docstring.kwarg_doc("bool, default: :rc:`text.usetex`")
    def set_usetex(self, usetex):
        """
        Parameters
        ----------
        usetex : bool or None
            Whether to render using TeX, ``None`` means to use
            :rc:`text.usetex`.
        """
        if usetex is None:
            self._usetex = mpl.rcParams['text.usetex']
        else:
            self._usetex = bool(usetex)
        self.stale = True

    def get_usetex(self):
        """Return whether this `Text` object uses TeX for rendering."""
        return self._usetex

    def set_parse_math(self, parse_math):
        """
        Override switch to disable any mathtext parsing for this `Text`.

        Parameters
        ----------
        parse_math : bool
            If False, this `Text` will never use mathtext.  If True, mathtext
            will be used if there is an even number of unescaped dollar signs.
        """
        self._parse_math = bool(parse_math)

    def get_parse_math(self):
        """Return whether mathtext parsing is considered for this `Text`."""
        return self._parse_math

    def set_fontname(self, fontname):
        """
        Alias for `set_fontfamily`.

        One-way alias only: the getter differs.

        Parameters
        ----------
        fontname : {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', \
'monospace'}

        See Also
        --------
        .font_manager.FontProperties.set_family

        """
        self.set_fontfamily(fontname)


class OffsetFrom:
    """Callable helper class for working with `Annotation`."""

    def __init__(self, artist, ref_coord, unit="points"):
        """
        Parameters
        ----------
        artist : `~matplotlib.artist.Artist` or `.BboxBase` or `.Transform`
            The object to compute the offset from.

        ref_coord : (float, float)
            If *artist* is an `.Artist` or `.BboxBase`, this values is
            the location to of the offset origin in fractions of the
            *artist* bounding box.

            If *artist* is a transform, the offset origin is the
            transform applied to this value.

        unit : {'points, 'pixels'}, default: 'points'
            The screen units to use (pixels or points) for the offset input.
        """
        self._artist = artist
        x, y = ref_coord  # Make copy when ref_coord is an array (and check the shape).
        self._ref_coord = x, y
        self.set_unit(unit)

    def set_unit(self, unit):
        """
        Set the unit for input to the transform used by ``__call__``.

        Parameters
        ----------
        unit : {'points', 'pixels'}
        """
        _api.check_in_list(["points", "pixels"], unit=unit)
        self._unit = unit

    def get_unit(self):
        """Return the unit for input to the transform used by ``__call__``."""
        return self._unit

    def __call__(self, renderer):
        """
        Return the offset transform.

        Parameters
        ----------
        renderer : `RendererBase`
            The renderer to use to compute the offset

        Returns
        -------
        `Transform`
            Maps (x, y) in pixel or point units to screen units
            relative to the given artist.
        """
        if isinstance(self._artist, Artist):
            bbox = self._artist.get_window_extent(renderer)
            xf, yf = self._ref_coord
            x = bbox.x0 + bbox.width * xf
            y = bbox.y0 + bbox.height * yf
        elif isinstance(self._artist, BboxBase):
            bbox = self._artist
            xf, yf = self._ref_coord
            x = bbox.x0 + bbox.width * xf
            y = bbox.y0 + bbox.height * yf
        elif isinstance(self._artist, Transform):
            x, y = self._artist.transform(self._ref_coord)
        else:
            _api.check_isinstance((Artist, BboxBase, Transform), artist=self._artist)
        scale = 1 if self._unit == "pixels" else renderer.points_to_pixels(1)
        return Affine2D().scale(scale).translate(x, y)


class _AnnotationBase:
    def __init__(self,
                 xy,
                 xycoords='data',
                 annotation_clip=None):

        x, y = xy  # Make copy when xy is an array (and check the shape).
        self.xy = x, y
        self.xycoords = xycoords
        self.set_annotation_clip(annotation_clip)

        self._draggable = None

    def _get_xy(self, renderer, xy, coords):
        x, y = xy
        xcoord, ycoord = coords if isinstance(coords, tuple) else (coords, coords)
        if xcoord == 'data':
            x = float(self.convert_xunits(x))
        if ycoord == 'data':
            y = float(self.convert_yunits(y))
        return self._get_xy_transform(renderer, coords).transform((x, y))

    def _get_xy_transform(self, renderer, coords):

        if isinstance(coords, tuple):
            xcoord, ycoord = coords
            from matplotlib.transforms import blended_transform_factory
            tr1 = self._get_xy_transform(renderer, xcoord)
            tr2 = self._get_xy_transform(renderer, ycoord)
            return blended_transform_factory(tr1, tr2)
        elif callable(coords):
            tr = coords(renderer)
            if isinstance(tr, BboxBase):
                return BboxTransformTo(tr)
            elif isinstance(tr, Transform):
                return tr
            else:
                raise TypeError(
                    f"xycoords callable must return a BboxBase or Transform, not a "
                    f"{type(tr).__name__}")
        elif isinstance(coords, Artist):
            bbox = coords.get_window_extent(renderer)
            return BboxTransformTo(bbox)
        elif isinstance(coords, BboxBase):
            return BboxTransformTo(coords)
        elif isinstance(coords, Transform):
            return coords
        elif not isinstance(coords, str):
            raise TypeError(
                f"'xycoords' must be an instance of str, tuple[str, str], Artist, "
                f"Transform, or Callable, not a {type(coords).__name__}")

        if coords == 'data':
            return self.axes.transData
        elif coords == 'polar':
            from matplotlib.projections import PolarAxes
            tr = PolarAxes.PolarTransform(apply_theta_transforms=False)
            trans = tr + self.axes.transData
            return trans

        try:
            bbox_name, unit = coords.split()
        except ValueError:  # i.e. len(coords.split()) != 2.
            raise ValueError(f"{coords!r} is not a valid coordinate") from None

        bbox0, xy0 = None, None

        # if unit is offset-like
        if bbox_name == "figure":
            bbox0 = self.get_figure(root=False).figbbox
        elif bbox_name == "subfigure":
            bbox0 = self.get_figure(root=False).bbox
        elif bbox_name == "axes":
            bbox0 = self.axes.bbox

        # reference x, y in display coordinate
        if bbox0 is not None:
            xy0 = bbox0.p0
        elif bbox_name == "offset":
            xy0 = self._get_position_xy(renderer)
        else:
            raise ValueError(f"{coords!r} is not a valid coordinate")

        if unit == "points":
            tr = Affine2D().scale(
                self.get_figure(root=True).dpi / 72)  # dpi/72 dots per point
        elif unit == "pixels":
            tr = Affine2D()
        elif unit == "fontsize":
            tr = Affine2D().scale(
                self.get_size() * self.get_figure(root=True).dpi / 72)
        elif unit == "fraction":
            tr = Affine2D().scale(*bbox0.size)
        else:
            raise ValueError(f"{unit!r} is not a recognized unit")

        return tr.translate(*xy0)

    def set_annotation_clip(self, b):
        """
        Set the annotation's clipping behavior.

        Parameters
        ----------
        b : bool or None
            - True: The annotation will be clipped when ``self.xy`` is
              outside the Axes.
            - False: The annotation will always be drawn.
            - None: The annotation will be clipped when ``self.xy`` is
              outside the Axes and ``self.xycoords == "data"``.
        """
        self._annotation_clip = b

    def get_annotation_clip(self):
        """
        Return the annotation's clipping behavior.

        See `set_annotation_clip` for the meaning of return values.
        """
        return self._annotation_clip

    def _get_position_xy(self, renderer):
        """Return the pixel position of the annotated point."""
        return self._get_xy(renderer, self.xy, self.xycoords)

    def _check_xy(self, renderer=None):
        """Check whether the annotation at *xy_pixel* should be drawn."""
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        b = self.get_annotation_clip()
        if b or (b is None and self.xycoords == "data"):
            # check if self.xy is inside the Axes.
            xy_pixel = self._get_position_xy(renderer)
            return self.axes.contains_point(xy_pixel)
        return True

    def draggable(self, state=None, use_blit=False):
        """
        Set whether the annotation is draggable with the mouse.

        Parameters
        ----------
        state : bool or None
            - True or False: set the draggability.
            - None: toggle the draggability.
        use_blit : bool, default: False
            Use blitting for faster image composition. For details see
            :ref:`func-animation`.

        Returns
        -------
        DraggableAnnotation or None
            If the annotation is draggable, the corresponding
            `.DraggableAnnotation` helper is returned.
        """
        from matplotlib.offsetbox import DraggableAnnotation
        is_draggable = self._draggable is not None

        # if state is None we'll toggle
        if state is None:
            state = not is_draggable

        if state:
            if self._draggable is None:
                self._draggable = DraggableAnnotation(self, use_blit)
        else:
            if self._draggable is not None:
                self._draggable.disconnect()
            self._draggable = None

        return self._draggable


class Annotation(Text, _AnnotationBase):
    """
    An `.Annotation` is a `.Text` that can refer to a specific position *xy*.
    Optionally an arrow pointing from the text to *xy* can be drawn.

    Attributes
    ----------
    xy
        The annotated position.
    xycoords
        The coordinate system for *xy*.
    arrow_patch
        A `.FancyArrowPatch` to point from *xytext* to *xy*.
    """

    def __str__(self):
        return f"Annotation({self.xy[0]:g}, {self.xy[1]:g}, {self._text!r})"

    def __init__(self, text, xy,
                 xytext=None,
                 xycoords='data',
                 textcoords=None,
                 arrowprops=None,
                 annotation_clip=None,
                 **kwargs):
        """
        Annotate the point *xy* with text *text*.

        In the simplest form, the text is placed at *xy*.

        Optionally, the text can be displayed in another position *xytext*.
        An arrow pointing from the text to the annotated point *xy* can then
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.

        xy : (float, float)
            The point *(x, y)* to annotate. The coordinate system is determined
            by *xycoords*.

        xytext : (float, float), default: *xy*
            The position *(x, y)* to place the text at. The coordinate system
            is determined by *textcoords*.

        xycoords : single or two-tuple of str or `.Artist` or `.Transform` or \
callable, default: 'data'

            The coordinate system that *xy* is given in. The following types
            of values are supported:

            - One of the following strings:

              ==================== ============================================
              Value                Description
              ==================== ============================================
              'figure points'      Points from the lower left of the figure
              'figure pixels'      Pixels from the lower left of the figure
              'figure fraction'    Fraction of figure from lower left
              'subfigure points'   Points from the lower left of the subfigure
              'subfigure pixels'   Pixels from the lower left of the subfigure
              'subfigure fraction' Fraction of subfigure from lower left
              'axes points'        Points from lower left corner of the Axes
              'axes pixels'        Pixels from lower left corner of the Axes
              'axes fraction'      Fraction of Axes from lower left
              'data'               Use the coordinate system of the object
                                   being annotated (default)
              'polar'              *(theta, r)* if not native 'data'
                                   coordinates
              ==================== ============================================

              Note that 'subfigure pixels' and 'figure pixels' are the same
              for the parent figure, so users who want code that is usable in
              a subfigure can use 'subfigure pixels'.

            - An `.Artist`: *xy* is interpreted as a fraction of the artist's
              `~matplotlib.transforms.Bbox`. E.g. *(0, 0)* would be the lower
              left corner of the bounding box and *(0.5, 1)* would be the
              center top of the bounding box.

            - A `.Transform` to transform *xy* to screen coordinates.

            - A function with one of the following signatures::

                def transform(renderer) -> Bbox
                def transform(renderer) -> Transform

              where *renderer* is a `.RendererBase` subclass.

              The result of the function is interpreted like the `.Artist` and
              `.Transform` cases above.

            - A tuple *(xcoords, ycoords)* specifying separate coordinate
              systems for *x* and *y*. *xcoords* and *ycoords* must each be
              of one of the above described types.

            See :ref:`plotting-guide-annotation` for more details.

        textcoords : single or two-tuple of str or `.Artist` or `.Transform` \
or callable, default: value of *xycoords*
            The coordinate system that *xytext* is given in.

            All *xycoords* values are valid as well as the following strings:

            =================   =================================================
            Value               Description
            =================   =================================================
            'offset points'     Offset, in points, from the *xy* value
            'offset pixels'     Offset, in pixels, from the *xy* value
            'offset fontsize'   Offset, relative to fontsize, from the *xy* value
            =================   =================================================

        arrowprops : dict, optional
            The properties used to draw a `.FancyArrowPatch` arrow between the
            positions *xy* and *xytext*.  Defaults to None, i.e. no arrow is
            drawn.

            For historical reasons there are two different ways to specify
            arrows, "simple" and "fancy":

            **Simple arrow:**

            If *arrowprops* does not contain the key 'arrowstyle' the
            allowed keys are:

            ==========  =================================================
            Key         Description
            ==========  =================================================
            width       The width of the arrow in points
            headwidth   The width of the base of the arrow head in points
            headlength  The length of the arrow head in points
            shrink      Fraction of total length to shrink from both ends
            ?           Any `.FancyArrowPatch` property
            ==========  =================================================

            The arrow is attached to the edge of the text box, the exact
            position (corners or centers) depending on where it's pointing to.

            **Fancy arrow:**

            This is used if 'arrowstyle' is provided in the *arrowprops*.

            Valid keys are the following `.FancyArrowPatch` parameters:

            ===============  ===================================
            Key              Description
            ===============  ===================================
            arrowstyle       The arrow style
            connectionstyle  The connection style
            relpos           See below; default is (0.5, 0.5)
            patchA           Default is bounding box of the text
            patchB           Default is None
            shrinkA          In points. Default is 2 points
            shrinkB          In points. Default is 2 points
            mutation_scale   Default is text size (in points)
            mutation_aspect  Default is 1
            ?                Any `.FancyArrowPatch` property
            ===============  ===================================

            The exact starting point position of the arrow is defined by
            *relpos*. It's a tuple of relative coordinates of the text box,
            where (0, 0) is the lower left corner and (1, 1) is the upper
            right corner. Values <0 and >1 are supported and specify points
            outside the text box. By default (0.5, 0.5), so the starting point
            is centered in the text box.

        annotation_clip : bool or None, default: None
            Whether to clip (i.e. not draw) the annotation when the annotation
            point *xy* is outside the Axes area.

            - If *True*, the annotation will be clipped when *xy* is outside
              the Axes.
            - If *False*, the annotation will always be drawn.
            - If *None*, the annotation will be clipped when *xy* is outside
              the Axes and *xycoords* is 'data'.

        **kwargs
            Additional kwargs are passed to `.Text`.

        Returns
        -------
        `.Annotation`

        See Also
        --------
        :ref:`annotations`

        """
        _AnnotationBase.__init__(self,
                                 xy,
                                 xycoords=xycoords,
                                 annotation_clip=annotation_clip)
        # warn about wonky input data
        if (xytext is None and
                textcoords is not None and
                textcoords != xycoords):
            _api.warn_external("You have used the `textcoords` kwarg, but "
                               "not the `xytext` kwarg.  This can lead to "
                               "surprising results.")

        # clean up textcoords and assign default
        if textcoords is None:
            textcoords = self.xycoords
        self._textcoords = textcoords

        # cleanup xytext defaults
        if xytext is None:
            xytext = self.xy
        x, y = xytext

        self.arrowprops = arrowprops
        if arrowprops is not None:
            arrowprops = arrowprops.copy()
            if "arrowstyle" in arrowprops:
                self._arrow_relpos = arrowprops.pop("relpos", (0.5, 0.5))
            else:
                # modified YAArrow API to be used with FancyArrowPatch
                for key in ['width', 'headwidth', 'headlength', 'shrink']:
                    arrowprops.pop(key, None)
            self.arrow_patch = FancyArrowPatch((0, 0), (1, 1), **arrowprops)
        else:
            self.arrow_patch = None

        # Must come last, as some kwargs may be propagated to arrow_patch.
        Text.__init__(self, x, y, text, **kwargs)

    def contains(self, mouseevent):
        if self._different_canvas(mouseevent):
            return False, {}
        contains, tinfo = Text.contains(self, mouseevent)
        if self.arrow_patch is not None:
            in_patch, _ = self.arrow_patch.contains(mouseevent)
            contains = contains or in_patch
        return contains, tinfo

    @property
    def xycoords(self):
        return self._xycoords

    @xycoords.setter
    def xycoords(self, xycoords):
        def is_offset(s):
            return isinstance(s, str) and s.startswith("offset")

        if (isinstance(xycoords, tuple) and any(map(is_offset, xycoords))
                or is_offset(xycoords)):
            raise ValueError("xycoords cannot be an offset coordinate")
        self._xycoords = xycoords

    @property
    def xyann(self):
        """
        The text position.

        See also *xytext* in `.Annotation`.
        """
        return self.get_position()

    @xyann.setter
    def xyann(self, xytext):
        self.set_position(xytext)

    def get_anncoords(self):
        """
        Return the coordinate system to use for `.Annotation.xyann`.

        See also *xycoords* in `.Annotation`.
        """
        return self._textcoords

    def set_anncoords(self, coords):
        """
        Set the coordinate system to use for `.Annotation.xyann`.

        See also *xycoords* in `.Annotation`.
        """
        self._textcoords = coords

    anncoords = property(get_anncoords, set_anncoords, doc="""
        The coordinate system to use for `.Annotation.xyann`.""")

    def set_figure(self, fig):
        # docstring inherited
        if self.arrow_patch is not None:
            self.arrow_patch.set_figure(fig)
        Artist.set_figure(self, fig)

    def update_positions(self, renderer):
        """
        Update the pixel positions of the annotation text and the arrow patch.
        """
        # generate transformation
        self.set_transform(self._get_xy_transform(renderer, self.anncoords))

        arrowprops = self.arrowprops
        if arrowprops is None:
            return

        bbox = Text.get_window_extent(self, renderer)

        arrow_end = x1, y1 = self._get_position_xy(renderer)  # Annotated pos.

        ms = arrowprops.get("mutation_scale", self.get_size())
        self.arrow_patch.set_mutation_scale(ms)

        if "arrowstyle" not in arrowprops:
            # Approximately simulate the YAArrow.
            shrink = arrowprops.get('shrink', 0.0)
            width = arrowprops.get('width', 4)
            headwidth = arrowprops.get('headwidth', 12)
            headlength = arrowprops.get('headlength', 12)

            # NB: ms is in pts
            stylekw = dict(head_length=headlength / ms,
                           head_width=headwidth / ms,
                           tail_width=width / ms)

            self.arrow_patch.set_arrowstyle('simple', **stylekw)

            # using YAArrow style:
            # pick the corner of the text bbox closest to annotated point.
            xpos = [(bbox.x0, 0), ((bbox.x0 + bbox.x1) / 2, 0.5), (bbox.x1, 1)]
            ypos = [(bbox.y0, 0), ((bbox.y0 + bbox.y1) / 2, 0.5), (bbox.y1, 1)]
            x, relposx = min(xpos, key=lambda v: abs(v[0] - x1))
            y, relposy = min(ypos, key=lambda v: abs(v[0] - y1))
            self._arrow_relpos = (relposx, relposy)
            r = np.hypot(y - y1, x - x1)
            shrink_pts = shrink * r / renderer.points_to_pixels(1)
            self.arrow_patch.shrinkA = self.arrow_patch.shrinkB = shrink_pts

        # adjust the starting point of the arrow relative to the textbox.
        # TODO : Rotation needs to be accounted.
        arrow_begin = bbox.p0 + bbox.size * self._arrow_relpos
        # The arrow is drawn from arrow_begin to arrow_end.  It will be first
        # clipped by patchA and patchB.  Then it will be shrunk by shrinkA and
        # shrinkB (in points).  If patchA is not set, self.bbox_patch is used.
        self.arrow_patch.set_positions(arrow_begin, arrow_end)

        if "patchA" in arrowprops:
            patchA = arrowprops["patchA"]
        elif self._bbox_patch:
            patchA = self._bbox_patch
        elif self.get_text() == "":
            patchA = None
        else:
            pad = renderer.points_to_pixels(4)
            patchA = Rectangle(
                xy=(bbox.x0 - pad / 2, bbox.y0 - pad / 2),
                width=bbox.width + pad, height=bbox.height + pad,
                transform=IdentityTransform(), clip_on=False)
        self.arrow_patch.set_patchA(patchA)

    @artist.allow_rasterization
    def draw(self, renderer):
        # docstring inherited
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible() or not self._check_xy(renderer):
            return
        # Update text positions before `Text.draw` would, so that the
        # FancyArrowPatch is correctly positioned.
        self.update_positions(renderer)
        self.update_bbox_position_size(renderer)
        if self.arrow_patch is not None:  # FancyArrowPatch
            if (self.arrow_patch.get_figure(root=False) is None and
                    (fig := self.get_figure(root=False)) is not None):
                self.arrow_patch.set_figure(fig)
            self.arrow_patch.draw(renderer)
        # Draw text, including FancyBboxPatch, after FancyArrowPatch.
        # Otherwise, a wedge arrowstyle can land partly on top of the Bbox.
        Text.draw(self, renderer)

    def get_window_extent(self, renderer=None):
        # docstring inherited
        # This block is the same as in Text.get_window_extent, but we need to
        # set the renderer before calling update_positions().
        if not self.get_visible() or not self._check_xy(renderer):
            return Bbox.unit()
        if renderer is not None:
            self._renderer = renderer
        if self._renderer is None:
            self._renderer = self.get_figure(root=True)._get_renderer()
        if self._renderer is None:
            raise RuntimeError('Cannot get window extent without renderer')

        self.update_positions(self._renderer)

        text_bbox = Text.get_window_extent(self)
        bboxes = [text_bbox]

        if self.arrow_patch is not None:
            bboxes.append(self.arrow_patch.get_window_extent())

        return Bbox.union(bboxes)

    def get_tightbbox(self, renderer=None):
        # docstring inherited
        if not self._check_xy(renderer):
            return Bbox.null()
        return super().get_tightbbox(renderer)


_docstring.interpd.register(Annotation=Annotation.__init__.__doc__)
<|endoftext|>"""
The classes here provide support for using custom classes with
Matplotlib, e.g., those that do not expose the array interface but know
how to convert themselves to arrays.  It also supports classes with
units and units conversion.  Use cases include converters for custom
objects, e.g., a list of datetime objects, as well as for objects that
are unit aware.  We don't assume any particular units implementation;
rather a units implementation must register with the Registry converter
dictionary and provide a `ConversionInterface`.  For example,
here is a complete implementation which supports plotting with native
datetime objects::

    import matplotlib.units as units
    import matplotlib.dates as dates
    import matplotlib.ticker as ticker
    import datetime

    class DateConverter(units.ConversionInterface):

        @staticmethod
        def convert(value, unit, axis):
            "Convert a datetime value to a scalar or array."
            return dates.date2num(value)

        @staticmethod
        def axisinfo(unit, axis):
            "Return major and minor tick locators and formatters."
            if unit != 'date':
                return None
            majloc = dates.AutoDateLocator()
            majfmt = dates.AutoDateFormatter(majloc)
            return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='date')

        @staticmethod
        def default_units(x, axis):
            "Return the default unit for x or None."
            return 'date'

    # Finally we register our object type with the Matplotlib units registry.
    units.registry[datetime.date] = DateConverter()
"""

from decimal import Decimal
from numbers import Number

import numpy as np
from numpy import ma

from matplotlib import cbook


class ConversionError(TypeError):
    pass


def _is_natively_supported(x):
    """
    Return whether *x* is of a type that Matplotlib natively supports or an
    array of objects of such types.
    """
    # Matplotlib natively supports all number types except Decimal.
    if np.iterable(x):
        # Assume lists are homogeneous as other functions in unit system.
        for thisx in x:
            if thisx is ma.masked:
                continue
            return isinstance(thisx, Number) and not isinstance(thisx, Decimal)
    else:
        return isinstance(x, Number) and not isinstance(x, Decimal)


class AxisInfo:
    """
    Information to support default axis labeling, tick labeling, and limits.

    An instance of this class must be returned by
    `ConversionInterface.axisinfo`.
    """
    def __init__(self, majloc=None, minloc=None,
                 majfmt=None, minfmt=None, label=None,
                 default_limits=None):
        """
        Parameters
        ----------
        majloc, minloc : Locator, optional
            Tick locators for the major and minor ticks.
        majfmt, minfmt : Formatter, optional
            Tick formatters for the major and minor ticks.
        label : str, optional
            The default axis label.
        default_limits : optional
            The default min and max limits of the axis if no data has
            been plotted.

        Notes
        -----
        If any of the above are ``None``, the axis will simply use the
        default value.
        """
        self.majloc = majloc
        self.minloc = minloc
        self.majfmt = majfmt
        self.minfmt = minfmt
        self.label = label
        self.default_limits = default_limits


class ConversionInterface:
    """
    The minimal interface for a converter to take custom data types (or
    sequences) and convert them to values Matplotlib can use.
    """

    @staticmethod
    def axisinfo(unit, axis):
        """Return an `.AxisInfo` for the axis with the specified units."""
        return None

    @staticmethod
    def default_units(x, axis):
        """Return the default unit for *x* or ``None`` for the given axis."""
        return None

    @staticmethod
    def convert(obj, unit, axis):
        """
        Convert *obj* using *unit* for the specified *axis*.

        If *obj* is a sequence, return the converted sequence.  The output must
        be a sequence of scalars that can be used by the numpy array layer.
        """
        return obj


class DecimalConverter(ConversionInterface):
    """Converter for decimal.Decimal data to float."""

    @staticmethod
    def convert(value, unit, axis):
        """
        Convert Decimals to floats.

        The *unit* and *axis* arguments are not used.

        Parameters
        ----------
        value : decimal.Decimal or iterable
            Decimal or list of Decimal need to be converted
        """
        if isinstance(value, Decimal):
            return float(value)
        # value is Iterable[Decimal]
        elif isinstance(value, ma.MaskedArray):
            return ma.asarray(value, dtype=float)
        else:
            return np.asarray(value, dtype=float)

    # axisinfo and default_units can be inherited as Decimals are Numbers.


class Registry(dict):
    """Register types with conversion interface."""

    def get_converter(self, x):
        """Get the converter interface instance for *x*, or None."""
        # Unpack in case of e.g. Pandas or xarray object
        x = cbook._unpack_to_numpy(x)

        if isinstance(x, np.ndarray):
            # In case x in a masked array, access the underlying data (only its
            # type matters).  If x is a regular ndarray, getdata() just returns
            # the array itself.
            x = np.ma.getdata(x).ravel()
            # If there are no elements in x, infer the units from its dtype
            if not x.size:
                return self.get_converter(np.array([0], dtype=x.dtype))
        for cls in type(x).__mro__:  # Look up in the cache.
            try:
                return self[cls]
            except KeyError:
                pass
        try:  # If cache lookup fails, look up based on first element...
            first = cbook._safe_first_finite(x)
        except (TypeError, StopIteration):
            pass
        else:
            # ... and avoid infinite recursion for pathological iterables for
            # which indexing returns instances of the same iterable class.
            if type(first) is not type(x):
                return self.get_converter(first)
        return None


registry = Registry()
registry[Decimal] = DecimalConverter()
<|endoftext|>"""
Defines classes for path effects. The path effects are supported in `.Text`,
`.Line2D` and `.Patch`.

.. seealso::
   :ref:`patheffects_guide`
"""

from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np


class AbstractPathEffect:
    """
    A base class for path effects.

    Subclasses should override the ``draw_path`` method to add effect
    functionality.
    """

    def __init__(self, offset=(0., 0.)):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, measured in points.
        """
        self._offset = offset

    def _offset_transform(self, renderer):
        """Apply the offset to the given transform."""
        return mtransforms.Affine2D().translate(
            *map(renderer.points_to_pixels, self._offset))

    def _update_gc(self, gc, new_gc_dict):
        """
        Update the given GraphicsContext with the given dict of properties.

        The keys in the dictionary are used to identify the appropriate
        ``set_`` method on the *gc*.
        """
        new_gc_dict = new_gc_dict.copy()

        dashes = new_gc_dict.pop("dashes", None)
        if dashes:
            gc.set_dashes(**dashes)

        for k, v in new_gc_dict.items():
            set_method = getattr(gc, 'set_' + k, None)
            if not callable(set_method):
                raise AttributeError(f'Unknown property {k}')
            set_method(v)
        return gc

    def draw_path(self, renderer, gc, tpath, affine, rgbFace=None):
        """
        Derived should override this method. The arguments are the same
        as :meth:`matplotlib.backend_bases.RendererBase.draw_path`
        except the first argument is a renderer.
        """
        # Get the real renderer, not a PathEffectRenderer.
        if isinstance(renderer, PathEffectRenderer):
            renderer = renderer._renderer
        return renderer.draw_path(gc, tpath, affine, rgbFace)


class PathEffectRenderer(RendererBase):
    """
    Implements a Renderer which contains another renderer.

    This proxy then intercepts draw calls, calling the appropriate
    :class:`AbstractPathEffect` draw method.

    .. note::
        Not all methods have been overridden on this RendererBase subclass.
        It may be necessary to add further methods to extend the PathEffects
        capabilities further.
    """

    def __init__(self, path_effects, renderer):
        """
        Parameters
        ----------
        path_effects : iterable of :class:`AbstractPathEffect`
            The path effects which this renderer represents.
        renderer : `~matplotlib.backend_bases.RendererBase` subclass

        """
        self._path_effects = path_effects
        self._renderer = renderer

    def copy_with_path_effect(self, path_effects):
        return self.__class__(path_effects, self._renderer)

    def __getattribute__(self, name):
        if name in ['flipy', 'get_canvas_width_height', 'new_gc',
                    'points_to_pixels', '_text2path', 'height', 'width']:
            return getattr(self._renderer, name)
        else:
            return object.__getattribute__(self, name)

    def draw_path(self, gc, tpath, affine, rgbFace=None):
        for path_effect in self._path_effects:
            path_effect.draw_path(self._renderer, gc, tpath, affine,
                                  rgbFace)

    def draw_markers(
            self, gc, marker_path, marker_trans, path, *args, **kwargs):
        # We do a little shimmy so that all markers are drawn for each path
        # effect in turn. Essentially, we induce recursion (depth 1) which is
        # terminated once we have just a single path effect to work with.
        if len(self._path_effects) == 1:
            # Call the base path effect function - this uses the unoptimised
            # approach of calling "draw_path" multiple times.
            return super().draw_markers(gc, marker_path, marker_trans, path,
                                        *args, **kwargs)

        for path_effect in self._path_effects:
