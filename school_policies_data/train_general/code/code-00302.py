        return "DynamicQuantizedRNN"

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def __repr__(self):
        # We don't want to show `ModuleList` children, hence custom
        # `__repr__`. This is the same as nn.Module.__repr__, except the check
        # for the `PackedParameter` and `nn.ModuleList`.
        # You should still override `extra_repr` to add more info.
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            if isinstance(module, (PackedParameter, nn.ModuleList)):
                continue
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f"input must have {expected_input_dim} dimensions, got {input.dim()}"
            )
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}"
            )

    def get_expected_hidden_size(
        self, input: Tensor, batch_sizes: Optional[Tensor]
    ) -> tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.hidden_size,
        )
        return expected_hidden_size

    def check_hidden_size(
        self,
        hx: Tensor,
        expected_hidden_size: tuple[int, int, int],
        msg: str = "Expected hidden size {}, got {}",
    ) -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(
        self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]
    ) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        self.check_hidden_size(
            hidden, expected_hidden_size, msg="Expected hidden size {}, got {}"
        )

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        self.version = version
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_weight_bias(self, weight_bias_dict):
        def weight_bias_name(ihhh, layer, suffix):
            weight_name = f"weight_{ihhh}_l{layer}{suffix}"
            bias_name = f"bias_{ihhh}_l{layer}{suffix}"
            return weight_name, bias_name

        num_directions = 2 if self.bidirectional else 1
        # TODO: dedup with __init__ of RNNBase
        _all_weight_values = []
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                w_ih_name, b_ih_name = weight_bias_name("ih", layer, suffix)
                w_hh_name, b_hh_name = weight_bias_name("hh", layer, suffix)
                w_ih = weight_bias_dict[w_ih_name]
                b_ih = weight_bias_dict[b_ih_name]
                w_hh = weight_bias_dict[w_hh_name]
                b_hh = weight_bias_dict[b_hh_name]
                if w_ih.dtype == torch.qint8:
                    packed_ih = torch.ops.quantized.linear_prepack(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack(w_hh, b_hh)
                    if self.version is None or self.version < 2:
                        cell_params = (
                            torch.ops.quantized.make_quantized_cell_params_dynamic(
                                packed_ih, packed_hh, b_ih, b_hh
                            )
                        )
                    else:
                        cell_params = (
                            torch.ops.quantized.make_quantized_cell_params_dynamic(
                                packed_ih, packed_hh, b_ih, b_hh, True
                            )
                        )
                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(w_hh, b_hh)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh
                    )

                _all_weight_values.append(PackedParameter(cell_params))
        self._all_weight_values = torch.nn.ModuleList(_all_weight_values)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) not in {torch.nn.LSTM, torch.nn.GRU}:
            raise AssertionError(
                "nn.quantized.dynamic.RNNBase.from_float only works for nn.LSTM and nn.GRU"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.ao.quantization.qconfig import default_dynamic_qconfig

            weight_observer_method = default_dynamic_qconfig.weight

        dtype = weight_observer_method().dtype
        supported_scalar_types = [torch.qint8, torch.float16]
        if dtype not in supported_scalar_types:
            raise RuntimeError(
                f"Unsupported dtype for dynamic RNN quantization: {dtype}"
            )
        # RNNBase can be either LSTM or GRU
        qRNNBase: Union[LSTM, GRU]
        if mod.mode == "LSTM":
            qRNNBase = LSTM(
                mod.input_size,
                mod.hidden_size,
                mod.num_layers,
                mod.bias,
                mod.batch_first,
                mod.dropout,
                mod.bidirectional,
                dtype,
            )
        elif mod.mode == "GRU":
            qRNNBase = GRU(
                mod.input_size,
                mod.hidden_size,
                mod.num_layers,
                mod.bias,
                mod.batch_first,
                mod.dropout,
                mod.bidirectional,
                dtype,
            )
        else:
            raise NotImplementedError(
                "Only LSTM/GRU is supported for QuantizedRNN for now"
            )

        num_directions = 2 if mod.bidirectional else 1

        if not mod.bias:
            raise AssertionError("mod.bias must be True")

        _all_weight_values = []
        for layer in range(qRNNBase.num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""

                def retrieve_weight_bias(ihhh):
                    weight_name = f"weight_{ihhh}_l{layer}{suffix}"
                    bias_name = f"bias_{ihhh}_l{layer}{suffix}"
                    weight = getattr(mod, weight_name)
                    bias = getattr(mod, bias_name)
                    return weight, bias

                weight_ih, bias_ih = retrieve_weight_bias("ih")
                weight_hh, bias_hh = retrieve_weight_bias("hh")

                if dtype == torch.qint8:

                    def quantize_and_pack(w, b):
                        weight_observer = weight_observer_method()
                        weight_observer(w)
                        qweight = _quantize_weight(w.float(), weight_observer)
                        packed_weight = torch.ops.quantized.linear_prepack(qweight, b)
                        return packed_weight

                    packed_ih = quantize_and_pack(weight_ih, bias_ih)
                    packed_hh = quantize_and_pack(weight_hh, bias_hh)
                    if qRNNBase.version is None or qRNNBase.version < 2:
                        cell_params = (
                            torch.ops.quantized.make_quantized_cell_params_dynamic(
                                packed_ih, packed_hh, bias_ih, bias_hh
                            )
                        )
                    else:
                        cell_params = (
                            torch.ops.quantized.make_quantized_cell_params_dynamic(
                                packed_ih, packed_hh, bias_ih, bias_hh, True
                            )
                        )

                elif dtype == torch.float16:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(
                        weight_ih.float(), bias_ih
                    )
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(
                        weight_hh.float(), bias_hh
                    )

                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh
                    )
                else:
                    raise RuntimeError(
                        "Unsupported dtype specified for dynamic quantized LSTM!"
                    )

                _all_weight_values.append(PackedParameter(cell_params))
        qRNNBase._all_weight_values = torch.nn.ModuleList(_all_weight_values)

        return qRNNBase

    def _weight_bias(self):
        # Returns a dict of weights and biases
        weight_bias_dict: Dict[str, Dict] = {"weight": {}, "bias": {}}
        count = 0
        num_directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                key_name1 = f"weight_ih_l{layer}{suffix}"
                key_name2 = f"weight_hh_l{layer}{suffix}"
                # packed weights are part of torchbind class, CellParamsSerializationType
                # Within the packed weight class, the weight and bias are accessible as Tensors
                packed_weight_bias = self._all_weight_values[  # type: ignore[index]
                    count
                ].param.__getstate__()[0][4]
                weight_bias_dict["weight"][key_name1] = packed_weight_bias[
                    0
                ].__getstate__()[0][0]
                weight_bias_dict["weight"][key_name2] = packed_weight_bias[
                    1
                ].__getstate__()[0][0]
                key_name1 = f"bias_ih_l{layer}{suffix}"
                key_name2 = f"bias_hh_l{layer}{suffix}"
                weight_bias_dict["bias"][key_name1] = packed_weight_bias[
                    0
                ].__getstate__()[0][1]
                weight_bias_dict["bias"][key_name2] = packed_weight_bias[
                    1
                ].__getstate__()[0][1]
                count = count + 1
        return weight_bias_dict

    def get_weight(self):
        return self._weight_bias()["weight"]

    def get_bias(self):
        return self._weight_bias()["bias"]


class LSTM(RNNBase):
    r"""
    A dynamic quantized LSTM module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.LSTM`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    _FLOAT_MODULE = nn.LSTM

    __overloads__ = {"forward": ["forward_packed", "forward_tensor"]}

    def __init__(self, *args, **kwargs):
        super().__init__("LSTM", *args, **kwargs)

    def _get_name(self):
        return "DynamicQuantizedLSTM"

    def forward_impl(
        self,
        input: Tensor,
        hx: Optional[tuple[Tensor, Tensor]],
        batch_sizes: Optional[Tensor],
        max_batch_size: int,
        sorted_indices: Optional[Tensor],
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)

        _all_params = [m.param for m in self._all_weight_values]
        if batch_sizes is None:
            result = torch.quantized_lstm(
                input,
                hx,
                _all_params,
                self.bias,
                self.num_layers,
                float(self.dropout),
                self.training,
                self.bidirectional,
                self.batch_first,
                dtype=self.dtype,
                use_dynamic=True,
            )
        else:
            result = torch.quantized_lstm(
                input,
                batch_sizes,
                hx,
                _all_params,
                self.bias,
                self.num_layers,
                float(self.dropout),
                self.training,
                self.bidirectional,
                dtype=self.dtype,
                use_dynamic=True,
            )
        output = result[0]
        hidden = result[1:]

        return output, hidden

    @torch.jit.export
    def forward_tensor(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        output, hidden = self.forward_impl(
            input, hx, batch_sizes, max_batch_size, sorted_indices
        )

        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch.jit.export
    def forward_packed(
        self, input: PackedSequence, hx: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[PackedSequence, tuple[Tensor, Tensor]]:
        input_, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = int(batch_sizes[0])

        output_, hidden = self.forward_impl(
            input_, hx, batch_sizes, max_batch_size, sorted_indices
        )

        output = PackedSequence(output_, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    # "type: ignore" is required due to issue #43072
    def permute_hidden(  # type: ignore[override]
        self,
        hx: tuple[Tensor, Tensor],
        permutation: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(
            hx[1], permutation
        )

    # "type: ignore" is required due to issue #43072
    def check_forward_args(  # type: ignore[override]
        self,
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        batch_sizes: Optional[Tensor],
    ) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(
            hidden[0], expected_hidden_size, "Expected hidden[0] size {}, got {}"
        )
        self.check_hidden_size(
            hidden[1], expected_hidden_size, "Expected hidden[1] size {}, got {}"
        )

    @torch.jit.ignore
    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_mod):
        if not hasattr(ref_mod, "weight_ih_l0_dtype"):
            raise AssertionError(
                "We are assuming weight_ih_l0 exists in LSTM, "
                "may need to relax the assumption to support the use case"
            )
        qmod = cls(
            ref_mod.input_size,
            ref_mod.hidden_size,
            ref_mod.num_layers,
            ref_mod.bias,
            ref_mod.batch_first,
            ref_mod.dropout,
            ref_mod.bidirectional,
            # assuming there is layer 0, which should be OK
            ref_mod.weight_ih_l0_dtype,
        )
        qmod.set_weight_bias(ref_mod.get_quantized_weight_bias_dict())
        return qmod


class GRU(RNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features h_t from the last layer of the GRU,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.

          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

    Shape:
        - Input1: :math:`(L, N, H_{in})` tensor containing input features where
          :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
        - Input2: :math:`(S, N, H_{out})` tensor
          containing the initial hidden state for each element in the batch.
          :math:`H_{out}=\text{hidden\_size}`
          Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
        - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        The calculation of new gate :math:`n_t` subtly differs from the original paper and other frameworks.
        In the original implementation, the Hadamard product :math:`(\odot)` between :math:`r_t` and the
        previous hidden state :math:`h_{(t-1)}` is done before the multiplication with the weight matrix
        `W` and addition of bias:

        .. math::
            \begin{aligned}
                n_t = \tanh(W_{in} x_t + b_{in} + W_{hn} ( r_t \odot h_{(t-1)} ) + b_{hn})
            \end{aligned}

        This is in contrast to PyTorch implementation, which is done after :math:`W_{hn} h_{(t-1)}`

        .. math::
            \begin{aligned}
                n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn}))
            \end{aligned}

        This implementation differs on purpose for efficiency.

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    _FLOAT_MODULE = nn.GRU

    __overloads__ = {"forward": ["forward_packed", "forward_tensor"]}

    def __init__(self, *args, **kwargs):
        super().__init__("GRU", *args, **kwargs)

    def _get_name(self):
        return "DynamicQuantizedGRU"

    def check_forward_args(
        self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]
    ) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(
            hidden, expected_hidden_size, "Expected hidden size {}, got {}"
        )

    def forward_impl(
        self,
        input: Tensor,
        hx: Optional[Tensor],
        batch_sizes: Optional[Tensor],
        max_batch_size: int,
        sorted_indices: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = zeros
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)

        _all_params = [m.param for m in self._all_weight_values]
        if batch_sizes is None:
            result = torch.quantized_gru(
                input,
                hx,
                _all_params,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = torch.quantized_gru(
                input,
                batch_sizes,
                hx,
                _all_params,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1]

        return output, hidden

    @torch.jit.export
    def forward_tensor(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        output, hidden = self.forward_impl(
            input, hx, batch_sizes, max_batch_size, sorted_indices
        )

        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch.jit.export
    def forward_packed(
        self, input: PackedSequence, hx: Optional[Tensor] = None
    ) -> tuple[PackedSequence, Tensor]:
        input_, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = int(batch_sizes[0])
        output_, hidden = self.forward_impl(
            input_, hx, batch_sizes, max_batch_size, sorted_indices
        )

        output = PackedSequence(output_, batch_sizes, sorted_indices, unsorted_indices)
        return output, self.permute_hidden(hidden, unsorted_indices)

    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)

    @torch.jit.ignore
    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            return self.forward_tensor(input, hx)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_mod):
        if not hasattr(ref_mod, "weight_ih_l0_dtype"):
            raise AssertionError(
                "We are assuming weight_ih_l0 exists in GRU, "
                "may need to relax the assumption to support the use case"
            )
        qmod = cls(
            ref_mod.input_size,
            ref_mod.hidden_size,
            ref_mod.num_layers,
            ref_mod.bias,
            ref_mod.batch_first,
            ref_mod.dropout,
            ref_mod.bidirectional,
            # assuming there is layer 0, which should be OK
            ref_mod.weight_ih_l0_dtype,
        )
        qmod.set_weight_bias(ref_mod.get_quantized_weight_bias_dict())
        return qmod


class RNNCellBase(torch.nn.Module):
    # _FLOAT_MODULE = nn.CellRNNBase
    __constants__ = ["input_size", "hidden_size", "bias"]

    def __init__(
        self, input_size, hidden_size, bias=True, num_chunks=4, dtype=torch.qint8
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_dtype = dtype
        if bias:
            self.bias_ih = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
            self.bias_hh = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        weight_ih = torch.randn(num_chunks * hidden_size, input_size).to(torch.float)
        weight_hh = torch.randn(num_chunks * hidden_size, hidden_size).to(torch.float)
        if dtype == torch.qint8:
            weight_ih = torch.quantize_per_tensor(
                weight_ih, scale=1, zero_point=0, dtype=torch.qint8
            )
            weight_hh = torch.quantize_per_tensor(
                weight_hh, scale=1, zero_point=0, dtype=torch.qint8
            )

        if dtype == torch.qint8:
            # for each layer, for each direction we need to quantize and pack
            # weights and pack parameters in this order:
            #
            #   w_ih, w_hh
            packed_weight_ih = torch.ops.quantized.linear_prepack(
                weight_ih, self.bias_ih
            )
            packed_weight_hh = torch.ops.quantized.linear_prepack(
                weight_hh, self.bias_hh
            )
        else:
            # for each layer, for each direction we need to quantize and pack
            # weights and pack parameters in this order:
            #
            #   packed_ih, packed_hh, b_ih, b_hh
            packed_weight_ih = torch.ops.quantized.linear_prepack_fp16(
                weight_ih, self.bias_ih
            )
            packed_weight_hh = torch.ops.quantized.linear_prepack_fp16(
                weight_hh, self.bias_hh
            )

        self._packed_weight_ih = packed_weight_ih
        self._packed_weight_hh = packed_weight_hh

    def _get_name(self):
        return "DynamicQuantizedRNNBase"

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                f"input has inconsistent input_size: got {input.size(1)}, expected {self.input_size}"
            )

    def check_forward_hidden(
        self, input: Tensor, hx: Tensor, hidden_label: str = ""
    ) -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                f"Input batch size {input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}"
            )

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                f"hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}"
            )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) not in {torch.nn.LSTMCell, torch.nn.GRUCell, torch.nn.RNNCell}:
            raise AssertionError(
                "nn.quantized.dynamic.RNNCellBase.from_float "
                "only works for nn.LSTMCell, nn.GRUCell and nn.RNNCell"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.ao.quantization.qconfig import default_dynamic_qconfig

            weight_observer_method = default_dynamic_qconfig.weight

        dtype = weight_observer_method().dtype
        supported_scalar_types = [torch.qint8, torch.float16]
        if dtype not in supported_scalar_types:
            raise RuntimeError(
                f"Unsupported dtype for dynamic RNN quantization: {dtype}"
            )

        qRNNCellBase: Union[LSTMCell, GRUCell, RNNCell]

        if type(mod) is torch.nn.LSTMCell:
            qRNNCellBase = LSTMCell(
                mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype
            )
        elif type(mod) is torch.nn.GRUCell:
            qRNNCellBase = GRUCell(
                mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype
            )
        elif type(mod) is torch.nn.RNNCell:
            qRNNCellBase = RNNCell(
                mod.input_size,
                mod.hidden_size,
                bias=mod.bias,
                nonlinearity=mod.nonlinearity,
                dtype=dtype,
            )
        else:
            raise NotImplementedError(
                "Only LSTMCell, GRUCell and RNNCell \
            are supported for QuantizedRNN for now"
            )

        if not mod.bias:
            raise AssertionError("mod.bias must be True")

        def _observe_and_quantize_weight(weight):
            if dtype == torch.qint8:
                weight_observer = weight_observer_method()
                weight_observer(weight)
                qweight = _quantize_weight(weight.float(), weight_observer)
                return qweight
            else:
                return weight.float()

        qRNNCellBase._packed_weight_ih = pack_weight_bias(
            _observe_and_quantize_weight(mod.weight_ih), mod.bias_ih, dtype
        )
        qRNNCellBase._packed_weight_hh = pack_weight_bias(
            _observe_and_quantize_weight(mod.weight_hh), mod.bias_hh, dtype
        )
        return qRNNCellBase

    @classmethod
    def from_reference(cls, ref_mod):
        if not hasattr(ref_mod, "weight_ih_dtype"):
            raise AssertionError(
                "We are assuming weight_ih exists in reference module, "
                "may need to relax the assumption to support the use case"
            )
        if hasattr(ref_mod, "nonlinearity"):
            qmod = cls(
                ref_mod.input_size,
                ref_mod.hidden_size,
                ref_mod.bias,
                ref_mod.nonlinearity,
                dtype=ref_mod.weight_ih_dtype,
            )
        else:
            qmod = cls(
                ref_mod.input_size,
                ref_mod.hidden_size,
                ref_mod.bias,
                dtype=ref_mod.weight_ih_dtype,
            )
        weight_bias_dict = {
            "weight": {
                "weight_ih": ref_mod.get_quantized_weight_ih(),
                "weight_hh": ref_mod.get_quantized_weight_hh(),
            },
            "bias": {
                "bias_ih": ref_mod.bias_ih,
                "bias_hh": ref_mod.bias_hh,
            },
        }
        qmod.set_weight_bias(weight_bias_dict)
        return qmod

    def _weight_bias(self):
        # Returns a dict of weights and biases
        weight_bias_dict: Dict[str, Dict] = {"weight": {}, "bias": {}}
        w1, b1 = self._packed_weight_ih.__getstate__()[0]
        w2, b2 = self._packed_weight_hh.__getstate__()[0]
        # TODO: these can be simplified to one level? e.g. using weight_ih as key
        # directly
        weight_bias_dict["weight"]["weight_ih"] = w1
        weight_bias_dict["weight"]["weight_hh"] = w2
        weight_bias_dict["bias"]["bias_ih"] = b1
        weight_bias_dict["bias"]["bias_hh"] = b2
        return weight_bias_dict

    def get_weight(self):
        return self._weight_bias()["weight"]

    def get_bias(self):
        return self._weight_bias()["bias"]

    def set_weight_bias(self, weight_bias_dict):
        # TODO: these can be simplified to one level? e.g. using weight_ih as key
        # directly
        self._packed_weight_ih = pack_weight_bias(
            weight_bias_dict["weight"]["weight_ih"],
            weight_bias_dict["bias"]["bias_ih"],
            self.weight_dtype,
        )
        self._packed_weight_hh = pack_weight_bias(
            weight_bias_dict["weight"]["weight_hh"],
            weight_bias_dict["bias"]["bias_hh"],
            self.weight_dtype,
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "_packed_weight_ih"] = self._packed_weight_ih
        destination[prefix + "_packed_weight_hh"] = self._packed_weight_hh

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._packed_weight_ih = state_dict.pop(prefix + "_packed_weight_ih")
        self._packed_weight_hh = state_dict.pop(prefix + "_packed_weight_hh")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.
    A dynamic quantized RNNCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.RNNCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.RNNCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "nonlinearity"]

    def __init__(
        self, input_size, hidden_size, bias=True, nonlinearity="tanh", dtype=torch.qint8
    ):
        super().__init__(input_size, hidden_size, bias, num_chunks=1, dtype=dtype)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        return "DynamicQuantizedRNNCell"

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        self.check_forward_hidden(input, hx, "")
        if self.nonlinearity == "tanh":
            ret = torch.ops.quantized.quantized_rnn_tanh_cell_dynamic(
                input,
                hx,
                self._packed_weight_ih,
                self._packed_weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = torch.ops.quantized.quantized_rnn_relu_cell_dynamic(
                input,
                hx,
                self._packed_weight_ih,
                self._packed_weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")
        return ret

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    A dynamic quantized LSTMCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.LSTMCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, num_chunks=4, **kwargs)  # type: ignore[misc]

    def _get_name(self):
        return "DynamicQuantizedLSTMCell"

    def forward(
        self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None
    ) -> tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], "[0]")
        self.check_forward_hidden(input, hx[1], "[1]")
        return torch.ops.quantized.quantized_lstm_cell_dynamic(
            input,
            hx,
            self._packed_weight_ih,
            self._packed_weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell

    A dynamic quantized GRUCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.GRUCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.GRUCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, dtype=torch.qint8):
        super().__init__(input_size, hidden_size, bias, num_chunks=3, dtype=dtype)

    def _get_name(self):
        return "DynamicQuantizedGRUCell"

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        self.check_forward_hidden(input, hx, "")
        return torch.ops.quantized.quantized_gru_cell_dynamic(
            input,
            hx,
            self._packed_weight_ih,
            self._packed_weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized as nnq
from torch.ao.nn.quantized.modules.utils import _quantize_weight


__all__ = [
    "Linear",
]


class Linear(nnq.Linear):
    r"""
    A dynamic quantized linear module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module which are of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable floating point bias of the module of shape
                       :math:`(\text{out\_features})`. If :attr:`bias` is ``True``,
                       the values are initialized to zero.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # version used in this class is different from the parent class nnq.Linear
    _version = 4

    def __init__(self, in_features, out_features, bias_=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias_, dtype=dtype)
        # We don't muck around with buffers or attributes or anything here
        # to keep the module simple. *everything* is simply a Python attribute.
        # Serialization logic is explicitly handled in the below serialization and
        # deserialization modules
        self.version = 4

    def forward(self, x):
        # Note that we can handle self.bias == None case.
        if self._packed_params.dtype == torch.qint8:
            if self.version is None or self.version < 4:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params
                )
            else:
                Y = torch.ops.quantized.linear_dynamic(
                    x, self._packed_params._packed_params, reduce_range=True
                )
        elif self._packed_params.dtype == torch.float16:
            Y = torch.ops.quantized.linear_dynamic_fp16(
                x, self._packed_params._packed_params
            )
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear!")
        return Y.to(x.dtype)

    def _get_name(self):
        return "DynamicQuantizedLinear"

    def extra_repr(self):
        extra_repr_str = f"in_features={self.in_features}, out_features={self.out_features}, dtype={self._packed_params.dtype}"
        if self._packed_params.dtype == torch.qint8:
            extra_repr_str += f", qscheme={self.weight().qscheme()}"
        return extra_repr_str

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        self.version = version
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a dynamic quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
        """
        float_modules = [
            torch.nn.Linear,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
            torch.ao.nn.intrinsic.modules.fused.LinearReLU,
            torch.ao.nn.qat.dynamic.Linear,
        ]

        if type(mod) not in float_modules:
            raise AssertionError(
                "nn.quantized.dynamic.Linear.from_float only works for one of"
                + str([float_mod.__name__ for float_mod in float_modules])
                + f", got {type(mod)}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if type(mod) is nni.LinearReLU:
            mod = mod[0]

        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer = mod.qconfig.weight()
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.ao.quantization.qconfig import default_dynamic_qconfig

            weight_observer = default_dynamic_qconfig.weight()
        dtype = weight_observer.dtype
        if dtype not in [torch.qint8, torch.float16]:
            raise AssertionError(
                f"The only supported dtypes for dynamic quantized linear are qint8 and float16, got: {dtype}"
            )
        weight_observer(mod.weight)
        if dtype == torch.qint8:
            qweight = _quantize_weight(mod.weight.float(), weight_observer)
        elif dtype == torch.float16:
            qweight = mod.weight.float()
        else:
            raise RuntimeError(
                "Unsupported dtype specified for dynamic quantized Linear!"
            )
        qlinear = cls(mod.in_features, mod.out_features, dtype=dtype)

        qlinear.set_weight_bias(qweight, mod.bias)
        return qlinear

    @classmethod
    def from_reference(cls, ref_qlinear):  # type: ignore[override]
        """Create a (fbgemm/qnnpack) dynamic quantized module from a reference quantized
        module
        Args:
            ref_qlinear (Module): a reference quantized  module, either produced by
            torch.ao.quantization functions or provided by the user
        """
        qlinear = cls(
            ref_qlinear.in_features,
            ref_qlinear.out_features,
            dtype=ref_qlinear.weight_dtype,
        )
        qweight = ref_qlinear.get_quantized_weight()
        bias = ref_qlinear.bias
        qlinear.set_weight_bias(qweight, bias)
        return qlinear
<|endoftext|>import types

from .modules import *  # noqa: F403
from .modules.fused import _FusedModule


# # Subpackages
# from . import qat
# from . import quantized

__all__ = [
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]


# We are exposing all subpackages to the end-user.
# Because of possible inter-dependency, we want to avoid
# the cyclic imports, thus implementing lazy version
# as per https://peps.python.org/pep-0562/
def __getattr__(name: str) -> types.ModuleType:
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
<|endoftext|>from .bn_relu import BNReLU2d, BNReLU3d
from .conv_add import ConvAdd2d, ConvAddReLU2d
from .conv_relu import ConvReLU1d, ConvReLU2d, ConvReLU3d
from .linear_relu import LinearLeakyReLU, LinearReLU, LinearTanh


__all__ = [
    "LinearReLU",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "BNReLU2d",
    "BNReLU3d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq
import torch.nn.functional as F


_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding


class ConvAdd2d(nnq.Conv2d):
    r"""
    A ConvAdd2d module is a fused module of Conv2d and Add

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAdd2d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input, extra_input):  # type: ignore[override]
        r"""Applies fused quantized Conv2d and addition."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_add(
            input, extra_input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvAdd2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module."""
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Creates a quantized module from a reference module."""
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class ConvAddReLU2d(nnq.Conv2d):
    r"""
    A ConvAddReLU2d module is a fused module of Conv2d, Add and Relu

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvAddReLU2d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input, extra_input):  # type: ignore[override]
        r"""Applies fused quantized Conv2d, addition, and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_add_relu(
            input, extra_input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvAddReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module."""
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Creates a quantized module from a reference module."""
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized as nnq
from torch.ao.nn.quantized.modules.utils import _quantize_weight


__all__ = [
    "LinearReLU",
    "LinearLeakyReLU",
    "LinearTanh",
]


class LinearReLU(nnq.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearReLU  # type: ignore[assignment]

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_relu(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedLinearReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant)

    @classmethod
    def from_reference(cls, ref_linear_relu, output_scale, output_zero_point):
        return super().from_reference(
            ref_linear_relu[0], output_scale, output_zero_point
        )


class LinearLeakyReLU(nnq.Linear):
    r"""
    For onednn backend only
    A LinearLeakyReLU module fused from Linear and LeakyReLU modules
    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.
    Attributes:
        Same as torch.ao.nn.quantized.Linear
        + negative_slope
    Examples::
        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearLeakyReLU(20, 30, 0.01)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearLeakyReLU  # type: ignore[assignment]

    def __init__(
        self, in_features, out_features, negative_slope, bias=True, dtype=torch.qint8
    ):
        super().__init__(in_features, out_features, bias, dtype)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_leaky_relu(
            x,
            self._packed_params._packed_params,
            self.scale,
            self.zero_point,
            self.negative_slope,
        )

    def _get_name(self):
        return "QuantizedLinearLeakyReLU"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not nni.LinearLeakyReLU:
            raise AssertionError("Input float module should be LinearLeakyReLU")
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        activation_post_process = mod.activation_post_process
        leaky_relu = mod[1]
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear_leaky_relu = cls(
            mod.in_features, mod.out_features, leaky_relu.negative_slope, dtype=dtype
        )
        qlinear_leaky_relu.set_weight_bias(qweight, mod.bias)  # type: ignore[arg-type]
        qlinear_leaky_relu.scale = float(act_scale)
        qlinear_leaky_relu.zero_point = int(act_zp)
        return qlinear_leaky_relu

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        linear = ref_mod[0]
        leaky_relu = ref_mod[1]
        qlinear_leaky_relu = cls(
            linear.in_features, linear.out_features, leaky_relu.negative_slope
        )
        qweight = linear.get_quantized_weight()
        qlinear_leaky_relu.set_weight_bias(qweight, linear.bias)
        qlinear_leaky_relu.scale = float(output_scale)
        qlinear_leaky_relu.zero_point = int(output_zero_point)
        return qlinear_leaky_relu


class LinearTanh(nnq.Linear):
    r"""
    A LinearTanh module fused from Linear and Tanh modules

    We adopt the same interface as :class:`torch.ao.nn.quantized.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.LinearTanh(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    _FLOAT_MODULE = nni.LinearTanh  # type: ignore[assignment]

    def __init__(self, in_features, out_features, bias=True, dtype=torch.qint8):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear_tanh(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedLinearTanh"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not nni.LinearTanh:
            raise AssertionError("Input float module should be LinearTanh")
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        activation_post_process = mod.activation_post_process
        mod = mod[0]
        weight_post_process = mod.qconfig.weight()  # type: ignore[union-attr,operator]
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[union-attr,operator]
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        qlinear_tanh = cls(mod.in_features, mod.out_features, dtype=dtype)
        qlinear_tanh.set_weight_bias(qweight, mod.bias)  # type: ignore[arg-type]
        qlinear_tanh.scale = float(act_scale)
        qlinear_tanh.zero_point = int(act_zp)
        return qlinear_tanh

    @classmethod
    def from_reference(cls, ref_mod, output_scale, output_zero_point):
        linear = ref_mod[0]
        qlinear_tanh = cls(linear.in_features, linear.out_features)
        qweight = linear.get_quantized_weight()
        qlinear_tanh.set_weight_bias(qweight, linear.bias)
        qlinear_tanh.scale = float(output_scale)
        qlinear_tanh.zero_point = int(output_zero_point)
        return qlinear_tanh
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq
import torch.nn.functional as F
from torch.nn.utils import fuse_conv_bn_weights


__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
]

_reverse_repeat_padding = nnq.modules.conv._reverse_repeat_padding


# TODO: factor out the common parts to ConvNd
class ConvReLU1d(nnq.Conv1d):
    r"""
    A ConvReLU1d module is a fused module of Conv1d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv1d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv1d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU1d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        r"""Applies fused quantized Conv1d and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 3:
            raise ValueError("Input shape must be `(N, C, L)`!")
        if self.padding_mode != "zeros":
            # Padding in Conv1d is stored as (p, p), need to get (p,)
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding[:1])
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv1d_relu(
            input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvReLU1d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module."""
        if type(mod) is torch.ao.nn.intrinsic.qat.ConvBnReLU1d:
            if mod.bn.running_var is None or mod.bn.running_mean is None:
                raise AssertionError(
                    "mod.bn.running_var and mod.bn.running_mean must not be None"
                )
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        return super().from_float(mod, use_precomputed_fake_quant)

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Creates a quantized module from a reference module."""
        if type(ref_qconv) is torch.ao.nn.intrinsic.ConvBnReLU1d:
            raise AssertionError(
                "BatchNorm1d should be fused into Conv1d before converting to reference module"
            )
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class ConvReLU2d(nnq.Conv2d):
    r"""
    A ConvReLU2d module is a fused module of Conv2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv2d`.

    Attributes:
        Same as torch.ao.nn.quantized.Conv2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU2d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        r"""Applies fused quantized Conv2d and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv2d_relu(
            input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module."""
        if type(mod) is torch.ao.nn.intrinsic.qat.ConvBnReLU2d:
            if mod.bn.running_var is None or mod.bn.running_mean is None:
                raise AssertionError(
                    "mod.bn.running_var and mod.bn.running_mean must not be None"
                )
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Creates a quantized module from a reference module."""
        if type(ref_qconv) is torch.ao.nn.intrinsic.ConvBnReLU2d:
            raise AssertionError(
                "BatchNorm2d should be fused into Conv2d before converting to reference module"
            )
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)


class ConvReLU3d(nnq.Conv3d):
    r"""
    A ConvReLU3d module is a fused module of Conv3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.Conv3d`.

    Attributes: Same as torch.ao.nn.quantized.Conv3d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvReLU3d  # type: ignore[assignment]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        if padding_mode == "reflect":
            raise AssertionError("Conv3d does not support reflection padding")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, input):
        r"""Applies fused quantized Conv3d and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        if self.padding_mode != "zeros":
            _reversed_padding_repeated_twice = _reverse_repeat_padding(self.padding)
            input = F.pad(
                input, _reversed_padding_repeated_twice, mode=self.padding_mode
            )
        return torch.ops.quantized.conv3d_relu(
            input, self._packed_params, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedConvReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module."""
        if type(mod) is torch.ao.nn.intrinsic.qat.ConvBnReLU3d:
            if mod.bn.running_var is None or mod.bn.running_mean is None:
                raise AssertionError(
                    "mod.bn.running_var and mod.bn.running_mean must not be None"
                )
            mod.weight, mod.bias = fuse_conv_bn_weights(
                mod.weight,
                mod.bias,
                mod.bn.running_mean,
                mod.bn.running_var,
                mod.bn.eps,
                mod.bn.weight,
                mod.bn.bias,
            )
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qconv, output_scale, output_zero_point):
        r"""Creates a quantized module from a reference module."""
        if type(ref_qconv) is torch.ao.nn.intrinsic.ConvBnReLU3d:
            raise AssertionError(
                "BatchNorm3d should be fused into Conv3d before converting to reference module"
            )
        return super().from_reference(ref_qconv[0], output_scale, output_zero_point)
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.qat
import torch.ao.nn.quantized as nnq


__all__ = ["BNReLU2d", "BNReLU3d"]


class BNReLU2d(nnq.BatchNorm2d):
    r"""
    A BNReLU2d module is a fused module of BatchNorm2d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm2d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm2d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU2d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__(
            num_features, eps=eps, momentum=momentum, device=device, dtype=dtype
        )

    def forward(self, input):
        r"""Applies fused BatchNorm2d and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return torch.ops.quantized.batch_norm2d_relu(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        return "QuantizedBNReLU2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module."""
        # TODO: Add qat support for BNReLU2d
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        r"""Creates a quantized module from a reference module."""
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)


class BNReLU3d(nnq.BatchNorm3d):
    r"""
    A BNReLU3d module is a fused module of BatchNorm3d and ReLU

    We adopt the same interface as :class:`torch.ao.nn.quantized.BatchNorm3d`.

    Attributes:
        Same as torch.ao.nn.quantized.BatchNorm3d

    """

    _FLOAT_MODULE = torch.ao.nn.intrinsic.BNReLU3d

    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None, dtype=None):
        super().__init__(
            num_features, eps=eps, momentum=momentum, device=device, dtype=dtype
        )

    def forward(self, input):
        r"""Applies fused BatchNorm3d and ReLU."""
        # Temporarily using len(shape) instead of ndim due to JIT issue
        # https://github.com/pytorch/pytorch/issues/23890
        if len(input.shape) != 5:
            raise ValueError("Input shape must be `(N, C, D, H, W)`!")
        return torch.ops.quantized.batch_norm3d_relu(
            input,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        return "QuantizedBNReLU3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a quantized module from a float module."""
        # TODO: Add qat support for BNReLU3d
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, bn_relu, output_scale, output_zero_point):
        r"""Creates a quantized module from a reference module."""
        return super().from_reference(bn_relu[0], output_scale, output_zero_point)
<|endoftext|>from typing import Any
from typing_extensions import Self

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.quantized.dynamic as nnqd


__all__ = ["LinearReLU"]


class LinearReLU(nnqd.Linear):
    r"""
    A LinearReLU module fused from Linear and ReLU modules that can be used
    for dynamic quantization.
    Supports both, FP16 and INT8 quantization.

    We adopt the same interface as :class:`torch.ao.nn.quantized.dynamic.Linear`.

    Attributes:
        Same as torch.ao.nn.quantized.dynamic.Linear

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.intrinsic.quantized.dynamic.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # pyrefly: ignore [bad-override]
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.qint8,
    ) -> None:
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._packed_params.dtype == torch.qint8:
            # TODO check if we should set reduce_range = True by default here
            Y = torch.ops.quantized.linear_relu_dynamic(
                x, self._packed_params._packed_params, reduce_range=True
            )
        elif self._packed_params.dtype == torch.float16:
            Y = torch.ops.quantized.linear_relu_dynamic_fp16(
                x, self._packed_params._packed_params
            )
        else:
            raise RuntimeError("Unsupported dtype on dynamic quantized linear relu!")
        return Y.to(x.dtype)

    def _get_name(self) -> str:
        return "DynamicQuantizedLinearReLU"

    @classmethod
    def from_float(
        cls, mod: torch.nn.Module, use_precomputed_fake_quant: bool = False
    ) -> Self:
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )

    @classmethod
    def from_reference(cls, ref_qlinear_relu: Any) -> Self:  # type: ignore[override]
        return super().from_reference(ref_qlinear_relu[0])
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    Linear,
    ReLU,
)
from torch.nn.utils.parametrize import type_before_parametrizations


__all__ = [
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "LinearReLU",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBn3d",
    "ConvBnReLU3d",
    "BNReLU2d",
    "BNReLU3d",
    "LinearBn1d",
    "LinearLeakyReLU",
    "LinearTanh",
    "ConvAdd2d",
    "ConvAddReLU2d",
]


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class ConvReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class ConvReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class ConvReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, relu):
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, relu)


class LinearReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, relu):
        if not (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(linear).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(linear, relu)


class ConvBn1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBn2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d and Batch Norm 2d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBnReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv 1d, Batch Norm 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        if not (
            type_before_parametrizations(conv) == Conv1d
            and type_before_parametrizations(bn) == BatchNorm1d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class ConvBnReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv 2d, Batch Norm 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        if not (
            type_before_parametrizations(conv) == Conv2d
            and type_before_parametrizations(bn) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class ConvBn3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d and Batch Norm 3d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn):
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(conv, bn)


class ConvBnReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv 3d, Batch Norm 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, bn, relu):
        if not (
            type_before_parametrizations(conv) == Conv3d
            and type_before_parametrizations(bn) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(conv).__name__}, "
                f"{type_before_parametrizations(bn).__name__}, and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(conv, bn, relu)


class BNReLU2d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        if not (
            type_before_parametrizations(batch_norm) == BatchNorm2d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(batch_norm).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(batch_norm, relu)


class BNReLU3d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm 3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, batch_norm, relu):
        if not (
            type_before_parametrizations(batch_norm) == BatchNorm3d
            and type_before_parametrizations(relu) == ReLU
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(batch_norm).__name__} and "
                f"{type_before_parametrizations(relu).__name__}"
            )
        super().__init__(batch_norm, relu)


class LinearBn1d(_FusedModule):
    r"""This is a sequential container which calls the Linear and BatchNorm1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, bn):
        if not (
            type_before_parametrizations(linear) == Linear
            and type_before_parametrizations(bn) == BatchNorm1d
        ):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type_before_parametrizations(linear).__name__} and "
                f"{type_before_parametrizations(bn).__name__}"
            )
        super().__init__(linear, bn)


class LinearLeakyReLU(_FusedModule):
    r"""This is a sequential container which calls the Linear and LeakyReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, leaky_relu):
        if not (type(linear) is Linear and type(leaky_relu) is torch.nn.LeakyReLU):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(leaky_relu).__name__}"
            )
        super().__init__(linear, leaky_relu)


class LinearTanh(_FusedModule):
    r"""This is a sequential container which calls the Linear and Tanh modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, linear, tanh):
        if not (type(linear) is Linear and type(tanh) is torch.nn.Tanh):
            raise AssertionError(
                f"Incorrect types for input modules: "
                f"{type(linear).__name__} and {type(tanh).__name__}"
            )
        super().__init__(linear, tanh)


class ConvAdd2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d modules with extra Add.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add):
        super().__init__(conv)
        self.add = add

    def forward(self, x1, x2):  # type: ignore[override]
        r"""Applies convolution to x1 and adds the result to x2."""
        return self.add(self[0](x1), x2)


class ConvAddReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d, add, Relu.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, conv, add, relu):
        super().__init__(conv)
        self.add = add
        self.relu = relu

    def forward(self, x1, x2):  # type: ignore[override]
        r"""Applies convolution to x1, adds the result to x2, and applies ReLU."""
        return self.relu(self.add(self[0](x1), x2))
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights


__all__ = [
    "LinearBn1d",
]


class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    r"""
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    def __init__(
        self,
        # Linear args
        in_features,
        out_features,
        bias=True,
        # BatchNorm1d args
        # num_features: out_features
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        nn.modules.linear.Linear.__init__(self, in_features, out_features, bias)
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def forward(self, input):
        if self.bn.running_var is None:
            raise AssertionError("self.bn.running_var must not be None")

        # Scale the linear weights by BN's running statistics to reduce
        # weight jitter, see https://arxiv.org/pdf/1806.08342.pdf, page 18
        # for motivation.
        #
        # Instead of
        #
        #   x1 = F.linear(x0, fq(w), b)
        #   x2 = self.bn(x1)
        #
        # We have
        #
        #   # scale the weight by previous batch's running statistics
        #   scale_factor = bn.w / bn.running_std_from_prev_batch
        #   # do the linear transformation without bias
        #   x1_scaled = F.linear(x0, fq(w * scale_factor), 0)
        #   # reverse the scaling and add original bias
        #   x1_orig = x1_scaled / scale_factor + b
        #   x2 = self.bn(x1_orig)

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(
            self.weight * scale_factor.reshape(weight_shape)
        )
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_features, device=scaled_weight.device)
        linear_out = F.linear(input, scaled_weight, zero_bias)
        linear_out_orig = linear_out / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            linear_out_orig = linear_out_orig + self.bias.reshape(bias_shape)
        bn_out = self.bn(linear_out_orig)
        return bn_out

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict

        Args:
            mod: A float module, either produced by torch.ao.quantization
                utilities or directly from the user.
        """
        if type(mod) is not nni.LinearBn1d:
            raise AssertionError(
                "qat."
                + cls.__name__
                + ".from_float only works for "
                + nni.LinearBn1d.__name__
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid config")
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            bn.eps,
            bn.momentum,
            False,
            qconfig,
        )
        qat_linearbn.weight = linear.weight  # type: ignore[assignment]
        qat_linearbn.bias = linear.bias  # type: ignore[assignment]
        qat_linearbn.bn.weight = bn.weight  # type: ignore[assignment]
        qat_linearbn.bn.bias = bn.bias  # type: ignore[assignment]
        qat_linearbn.bn.running_mean = bn.running_mean  # type: ignore[assignment]
        qat_linearbn.bn.running_var = bn.running_var  # type: ignore[assignment]
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[assignment]
        return qat_linearbn

    def to_float(self):
        linear = torch.nn.Linear(self.in_features, self.out_features)
        if self.bn.running_var is None or self.bn.running_mean is None:
            raise AssertionError(
                "self.bn.running_var and self.bn.running_mean must not be None"
            )
        linear.weight, linear.bias = fuse_linear_bn_weights(
            self.weight,
            self.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias,
        )
        return linear
<|endoftext|>from .conv_fused import (
    ConvBn1d,
    ConvBn2d,
    ConvBn3d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBnReLU3d,
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
    freeze_bn_stats,
    update_bn_stats,
)
from .linear_fused import LinearBn1d
from .linear_relu import LinearReLU


__all__ = [
    "LinearReLU",
    "LinearBn1d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "update_bn_stats",
    "freeze_bn_stats",
]
<|endoftext|># mypy: allow-untyped-defs
import math
from typing import ClassVar

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.parameter import Parameter
from torch.nn.utils import fuse_conv_bn_weights


__all__ = [
    "ConvBn1d",
    "ConvBnReLU1d",
    "ConvReLU1d",
    "ConvBn2d",
    "ConvBnReLU2d",
    "ConvReLU2d",
    "ConvBn3d",
    "ConvBnReLU3d",
    "ConvReLU3d",
    "update_bn_stats",
    "freeze_bn_stats",
]
_BN_CLASS_MAP = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


class _ConvBnNd(nn.modules.conv._ConvNd, nni._FusedModule):
    _version = 2
    _FLOAT_MODULE: ClassVar[type[nn.modules.conv._ConvNd]]

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode,
        # BatchNormNd args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
        dim=2,
    ):
        nn.modules.conv._ConvNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            False,
            padding_mode,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = _BN_CLASS_MAP[dim](out_channels, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

        self._enable_slow_path_for_better_numerical_stability = False

    def reset_running_stats(self):
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)
        # note: below is actually for conv, not BN
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def update_bn_stats(self):
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        self.bn.training = False
        return self

    def _forward(self, input):
        if self._enable_slow_path_for_better_numerical_stability:
            return self._forward_slow(input)
        return self._forward_approximate(input)

    def _forward_approximate(self, input):
        """Approximated method to fuse conv and bn. It requires only one forward pass.
        conv_orig = conv / scale_factor where scale_factor = bn.weight / running_std
        """
        if self.bn.running_var is None:
            raise AssertionError("self.bn.running_var must not be None")
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(
            self.weight * scale_factor.reshape(weight_shape)
        )
        # using zero bias here since the bias for original conv
        # will be added later
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(
                self.out_channels, device=scaled_weight.device, dtype=input.dtype
            )
        conv = self._conv_forward(input, scaled_weight, zero_bias)
        conv_orig = conv / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            conv_orig = conv_orig + self.bias.reshape(bias_shape)
        conv = self.bn(conv_orig)
        return conv

    def _forward_slow(self, input):
        """
        A more accurate but slow method to compute conv bn fusion, following https://arxiv.org/pdf/1806.08342.pdf
        It requires two forward passes but handles the case bn.weight == 0

        Conv: Y = WX + B_c
        Conv without bias: Y0 = WX = Y - B_c, Y = Y0 + B_c

        Batch statistics:
          mean_Y = Y.mean()
                 = Y0.mean() + B_c
          var_Y = (Y - mean_Y)^2.mean()
                = (Y0 - Y0.mean())^2.mean()
        BN (r: bn.weight, beta: bn.bias):
          Z = r * (Y - mean_Y) / sqrt(var_Y + eps) + beta
            = r * (Y0 - Y0.mean()) / sqrt(var_Y + eps) + beta

        Fused Conv BN training (std_Y = sqrt(var_Y + eps)):
          Z = (r * W / std_Y) * X + r * (B_c - mean_Y) / std_Y + beta
            = (r * W / std_Y) * X - r * Y0.mean() / std_Y + beta

        Fused Conv BN inference (running_std = sqrt(running_var + eps)):
          Z = (r * W / running_std) * X - r * (running_mean - B_c) / running_std + beta

        QAT with fused conv bn:
          Z_train = fake_quant(r * W / running_std) * X * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
                  = conv(X, fake_quant(r * W / running_std)) * (running_std / std_Y) - r * Y0.mean() / std_Y + beta
          Z_inference = conv(X, fake_quant(r * W / running_std)) - r * (running_mean - B_c) / running_std + beta
        """

        if self.bn.running_var is None:
            raise AssertionError("self.bn.running_var must not be None")
        if self.bn.running_mean is None:
            raise AssertionError("self.bn.running_mean must not be None")

        # using zero bias here since the bias for original conv
        # will be added later
        zero_bias = torch.zeros(
            self.out_channels, device=self.weight.device, dtype=input.dtype
        )

        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1

        if self.bn.training:
            # needed to compute batch mean/std
            conv_out = self._conv_forward(input, self.weight, zero_bias)
            # update bn statistics
            with torch.no_grad():
                conv_out_bias = (
                    conv_out
                    if self.bias is None
                    else conv_out + self.bias.reshape(bias_shape)
                )
                self.bn(conv_out_bias)

            # fused conv + bn without bias using bn running statistics
            running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            scaled_weight = self.weight_fake_quant(
                self.weight * scale_factor.reshape(weight_shape)
            )
            # fused conv without bias for inference: (r * W / running_std) * X
            conv_bn = self._conv_forward(input, scaled_weight, zero_bias)

            avg_dims = [0] + list(range(2, len(self.weight.shape)))
            batch_mean = conv_out.mean(avg_dims)
            batch_var = torch.square(conv_out - batch_mean.reshape(bias_shape)).mean(
                avg_dims
            )
            batch_std = torch.sqrt(batch_var + self.bn.eps)

            # scale to use batch std in training mode
            # conv(X, r * W / std_Y) = conv(X, r * W / running_std) * (running_std / std_Y)
            unscale_factor = running_std / batch_std
            conv_bn *= unscale_factor.reshape(bias_shape)

            fused_mean = batch_mean
            fused_std = batch_std
        else:
            # fused conv + bn without bias using bn running statistics
            running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            scaled_weight = self.weight_fake_quant(
                self.weight * scale_factor.reshape(weight_shape)
            )
            # fused conv without bias for inference: (r * W / running_std) * X
            conv_bn = self._conv_forward(input, scaled_weight, zero_bias)

            fused_mean = self.bn.running_mean - (
                self.bias if self.bias is not None else 0
            )
            fused_std = running_std

        # fused bias = beta - r * mean / std
        fused_bias = self.bn.bias - self.bn.weight * fused_mean / fused_std
        conv_bn += fused_bias.reshape(bias_shape)

        # HACK to let conv bias participate in loss to avoid DDP error (parameters
        #   were not used in producing loss)
        if self.bias is not None:
            conv_bn += (self.bias - self.bias).reshape(bias_shape)

        return conv_bn

    def forward(self, input):
        return self._forward(input)

    def train(self, mode=True):
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    # ===== Serialization version history =====
    #
    # Version 1/None
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- gamma : Tensor
    #   |--- beta : Tensor
    #   |--- running_mean : Tensor
    #   |--- running_var : Tensor
    #   |--- num_batches_tracked : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- bn : Module
    #        |--- weight : Tensor (moved from v1.self.gamma)
    #        |--- bias : Tensor (moved from v1.self.beta)
    #        |--- running_mean : Tensor (moved from v1.self.running_mean)
    #        |--- running_var : Tensor (moved from v1.self.running_var)
    #        |--- num_batches_tracked : Tensor (moved from v1.self.num_batches_tracked)
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version == 1:
            # BN related parameters and buffers were moved into the BN module for v2
            v2_to_v1_names = {
                "bn.weight": "gamma",
                "bn.bias": "beta",
                "bn.running_mean": "running_mean",
                "bn.running_var": "running_var",
                "bn.num_batches_tracked": "num_batches_tracked",
            }
            for v2_name, v1_name in v2_to_v1_names.items():
                if prefix + v1_name in state_dict:
                    state_dict[prefix + v2_name] = state_dict[prefix + v1_name]
                    state_dict.pop(prefix + v1_name)
                elif prefix + v2_name in state_dict:
                    # there was a brief period where forward compatibility
                    # for this module was broken (between
                    # https://github.com/pytorch/pytorch/pull/38478
                    # and https://github.com/pytorch/pytorch/pull/38820)
                    # and modules emitted the v2 state_dict format while
                    # specifying that version == 1. This patches the forward
                    # compatibility issue by allowing the v2 style entries to
                    # be used.
                    pass
                elif strict:
                    missing_keys.append(prefix + v2_name)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
        # The ignore is because _FLOAT_MODULE is a TypeVar here where the bound
        # has no __name__ (code is fine though)
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                "qat."
                + cls.__name__
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid qconfig")
        qconfig = mod.qconfig
        conv, bn = mod[0], mod[1]  # type: ignore[index]
        qat_convbn = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            bn.eps,
            bn.momentum,
            False,
            qconfig,
        )
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked
        return qat_convbn

    def to_float(self):
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined]
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
        )
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())

        if cls._FLOAT_BN_MODULE:  # type: ignore[attr-defined]
            # fuse bn into conv
            if self.bn.running_var is None or self.bn.running_mean is None:
                raise AssertionError(
                    "self.bn.running_var and self.bn.running_mean must not be None"
                )
            conv.weight, conv.bias = fuse_conv_bn_weights(
                conv.weight,
                conv.bias,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.eps,
                self.bn.weight,
                self.bn.bias,
            )

        if cls._FLOAT_RELU_MODULE:  # type: ignore[attr-defined]
            modules = []
            modules.append(conv)
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            conv_relu = cls._FUSED_FLOAT_MODULE(*modules)  # type: ignore[attr-defined]
            conv_relu.train(self.training)
            return conv_relu
        else:
            conv.train(self.training)
            return conv


class ConvBn1d(_ConvBnNd, nn.Conv1d):
    r"""
    A ConvBn1d module is a module fused from Conv1d and BatchNorm1d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Conv1d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]] = nn.BatchNorm1d
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = None
    _FLOAT_MODULE: ClassVar[type[nn.Module]] = nni.ConvBn1d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d

    def __init__(
        self,
        # Conv1d args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm1d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=1,
        )


class ConvBnReLU1d(ConvBn1d):
    r"""
    A ConvBnReLU1d module is a module fused from Conv1d, BatchNorm1d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv1d` and
    :class:`torch.nn.BatchNorm1d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv1d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    # base class defines _FLOAT_MODULE as "ConvBn1d"
    _FLOAT_MODULE: ClassVar[type[nn.Module]] = nni.ConvBnReLU1d
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm1d]] = nn.BatchNorm1d
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = nn.ReLU
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE: ClassVar[type[nn.Module] | None] = nni.ConvReLU1d

    def forward(self, input):
        r"""Performs forward pass through fused Conv1d, BatchNorm1d, and ReLU."""
        return F.relu(self._forward(input))

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Creates a QAT module from a floating point module."""
        return super().from_float(mod, use_precomputed_fake_quant)


class ConvReLU1d(nnqat.Conv1d, nni._FusedModule):
    r"""A ConvReLU1d module is a fused module of Conv1d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv1d` and
    :class:`~torch.nn.BatchNorm1d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU1d]] = nni.ConvReLU1d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv1d]] = nn.Conv1d
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = None
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = nn.ReLU

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        r"""Performs forward pass through fused Conv1d and ReLU."""
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a QAT module from a floating point module."""
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class ConvBn2d(_ConvBnNd, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvBn2d]] = nni.ConvBn2d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = nn.BatchNorm2d
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm2d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=2,
        )


class ConvBnReLU2d(ConvBn2d):
    r"""
    A ConvBnReLU2d module is a module fused from Conv2d, BatchNorm2d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    # base class defines _FLOAT_MODULE as "ConvBn2d"
    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU2d]] = nni.ConvBnReLU2d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm2d]] = nn.BatchNorm2d
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = nn.ReLU
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU2d] | None] = nni.ConvReLU2d

    def forward(self, input):
        r"""Performs forward pass through fused Conv2d, BatchNorm2d, and ReLU."""
        return F.relu(self._forward(input))

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Creates a QAT module from a floating point module."""
        return super().from_float(mod, use_precomputed_fake_quant)


class ConvReLU2d(nnqat.Conv2d, nni._FusedModule):
    r"""A ConvReLU2d module is a fused module of Conv2d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv2d` and
    :class:`~torch.nn.BatchNorm2d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_MODULE: ClassVar[type[nn.Module]] = nni.ConvReLU2d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv2d]] = nn.Conv2d
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = None
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = nn.ReLU

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        r"""Performs forward pass through fused Conv2d and ReLU."""
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a QAT module from a floating point module."""
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class ConvBn3d(_ConvBnNd, nn.Conv3d):
    r"""
    A ConvBn3d module is a module fused from Conv3d and BatchNorm3d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d`.

    Similar to :class:`torch.nn.Conv3d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvBn3d]] = nni.ConvBn3d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = nn.BatchNorm3d
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = None

    def __init__(
        self,
        # ConvNd args
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=None,
        padding_mode="zeros",
        # BatchNorm3d args
        # num_features: out_channels
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        _ConvBnNd.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            eps,
            momentum,
            freeze_bn,
            qconfig,
            dim=3,
        )


class ConvBnReLU3d(ConvBn3d):
    r"""
    A ConvBnReLU3d module is a module fused from Conv3d, BatchNorm3d and ReLU,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv3d` and
    :class:`torch.nn.BatchNorm3d` and :class:`torch.nn.ReLU`.

    Similar to `torch.nn.Conv3d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvBnReLU3d]] = nni.ConvBnReLU3d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _FLOAT_BN_MODULE: ClassVar[type[nn.BatchNorm3d]] = nn.BatchNorm3d
    _FLOAT_RELU_MODULE: ClassVar[type[nn.ReLU] | None] = nn.ReLU
    # module class after fusing bn into conv
    _FUSED_FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d] | None] = nni.ConvReLU3d

    def forward(self, input):
        r"""Performs forward pass through fused Conv3d, BatchNorm3d, and ReLU."""
        return F.relu(ConvBn3d._forward(self, input))

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Creates a QAT module from a floating point module."""
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


class ConvReLU3d(nnqat.Conv3d, nni._FusedModule):
    r"""A ConvReLU3d module is a fused module of Conv3d and ReLU, attached with
    FakeQuantize modules for weight for
    quantization aware training.

    We combined the interface of :class:`~torch.nn.Conv3d` and
    :class:`~torch.nn.BatchNorm3d`.

    Attributes:
        weight_fake_quant: fake quant module for weight

    """

    _FLOAT_MODULE: ClassVar[type[nni.ConvReLU3d]] = nni.ConvReLU3d  # type: ignore[assignment]
    _FLOAT_CONV_MODULE: ClassVar[type[nn.Conv3d]] = nn.Conv3d
    _FLOAT_BN_MODULE: ClassVar[type[nn.Module] | None] = None
    _FLOAT_RELU_MODULE: ClassVar[type[nn.Module] | None] = nn.ReLU

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconfig=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            # pyrefly: ignore [bad-argument-type]
            padding_mode=padding_mode,
            qconfig=qconfig,
        )
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input):
        r"""Performs forward pass through fused Conv3d and ReLU."""
        return F.relu(
            self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        )

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):  # type: ignore[override]
        r"""Creates a QAT module from a floating point module."""
        return super().from_float(
            mod, use_precomputed_fake_quant=use_precomputed_fake_quant
        )


def update_bn_stats(mod):
    if type(mod) in {
        ConvBnReLU1d,
        ConvBnReLU2d,
        ConvBnReLU3d,
        ConvBn1d,
        ConvBn2d,
        ConvBn3d,
    }:
        mod.update_bn_stats()


def freeze_bn_stats(mod):
    if type(mod) in {
        ConvBnReLU1d,
        ConvBnReLU2d,
        ConvBnReLU3d,
        ConvBn1d,
        ConvBn2d,
        ConvBn3d,
    }:
        mod.freeze_bn_stats()
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.ao.nn.intrinsic.modules.fused import _FusedModule


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import QConfigAny


__all__ = ["LinearReLU"]


class LinearReLU(nnqat.Linear, _FusedModule):
    r"""
    A LinearReLU module fused from Linear and ReLU modules, attached with
    FakeQuantize modules for weight, used in
    quantization aware training.

    We adopt the same interface as :class:`torch.nn.Linear`.

    Similar to `torch.ao.nn.intrinsic.LinearReLU`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.qat.LinearReLU(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    # pyrefly: ignore [bad-override]
    _FLOAT_MODULE = nni.LinearReLU

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        qconfig: QConfigAny = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, qconfig)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(input, self.weight_fake_quant(self.weight), self.bias))

    @classmethod
    def from_float(
        cls,
        mod: torch.nn.Module,
        use_precomputed_fake_quant: bool = False,
    ) -> LinearReLU:
        return super().from_float(mod, use_precomputed_fake_quant)  # type: ignore[no-untyped-call,no-any-return]

    def to_float(self) -> nni.LinearReLU:
        linear = torch.nn.Linear(
            self.in_features, self.out_features, self.bias is not None
        )
        linear.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            linear.bias = torch.nn.Parameter(self.bias.detach())
        relu = torch.nn.ReLU()
        return torch.ao.nn.intrinsic.LinearReLU(linear, relu)  # type: ignore[no-untyped-call]
<|endoftext|>import threading


__all__ = ["LinearBlockSparsePattern"]


def _is_valid_linear_block_sparse_pattern(
    row_block_size: int, col_block_size: int
) -> bool:
    return (row_block_size == 1 and col_block_size == 4) or (
        row_block_size == 8 and col_block_size == 1
    )


# This is a stop-gap measure as current flow does not allow module
# specific block sparse pattern.
# In fact there is no way to convey sparse pattern via module config
# of quantization flow. Thus using the global context to convey
# sparsity pattern.
# Once the flow supports it, this should be removed.
class LinearBlockSparsePattern:
    rlock = threading.RLock()
    row_block_size: int = 1
    col_block_size: int = 4
    prev_row_block_size: int = 1
    prev_col_block_size: int = 4

    def __init__(self, row_block_size: int = 1, col_block_size: int = 4):
        if not _is_valid_linear_block_sparse_pattern(row_block_size, col_block_size):
            raise AssertionError(
                f"Invalid linear block sparse pattern: "
                f"row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        LinearBlockSparsePattern.rlock.acquire()
        LinearBlockSparsePattern.prev_row_block_size = (
            LinearBlockSparsePattern.row_block_size
        )
        LinearBlockSparsePattern.prev_col_block_size = (
            LinearBlockSparsePattern.col_block_size
        )
        LinearBlockSparsePattern.row_block_size = row_block_size
        LinearBlockSparsePattern.col_block_size = col_block_size

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        backtrace: object | None,
    ) -> None:
        LinearBlockSparsePattern.row_block_size = (
            LinearBlockSparsePattern.prev_row_block_size
        )
        LinearBlockSparsePattern.col_block_size = (
            LinearBlockSparsePattern.prev_col_block_size
        )
        LinearBlockSparsePattern.rlock.release()

    @staticmethod
    def block_size() -> tuple[int, int]:
        return (
            LinearBlockSparsePattern.row_block_size,
            LinearBlockSparsePattern.col_block_size,
        )
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch.ao.nn.quantized.modules.utils import (
    _hide_packed_params_repr,
    _quantize_weight,
)


__all__ = ["LinearPackedParams", "Linear"]


# TODO (zaf): Inherit from `quantized.LinearPackedParams` (T83294430)
class LinearPackedParams(torch.nn.Module):
    _version = 1

    def __init__(self, row_block_size=1, col_block_size=4, dtype=torch.qint8):
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError("Linear prepacking only supports QINT8")
        self.dtype = dtype
        wq = torch._empty_affine_quantized(
            [1, 1], scale=1.0, zero_point=0, dtype=torch.qint8
        )
        self.set_weight_bias(wq, None, row_block_size, col_block_size)

    def _get_name(self):
        return "SparseQuantizedLinearPackedParams"

    @torch.jit.export
    def set_weight_bias(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None:
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                "row_block_size and col_block_size must not be None, got "
                f"row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        self._packed_params = torch.ops.sparse.qlinear_prepack(
            weight, bias, row_block_size, col_block_size
        )

    @torch.jit.export
    def _weight_bias(self):
        (weight, bias, block_sizes) = torch.ops.sparse.qlinear_unpack(
            self._packed_params
        )
        return (weight, bias, block_sizes[0], block_sizes[1])

    def forward(self, x):
        return x

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "dtype"] = self.dtype
        destination[prefix + "_packed_params"] = self._weight_bias()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is not None and version > self._version:
            raise AssertionError(f"version {version} > self._version {self._version}")

        self.dtype = state_dict.pop(prefix + "dtype")
        weight, bias, row_block_size, col_block_size = state_dict.pop(
            prefix + "_packed_params"
        )
        self.set_weight_bias(weight, bias, row_block_size, col_block_size)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.jit.export
    def __getstate__(self):
        return self._packed_params, self.training, self.dtype

    @torch.jit.export
    def __setstate__(self, state):
        (self._packed_params, self.training, self.dtype) = state

    def __repr__(self):
        return self._weight_bias().__repr__()


# TODO (zaf): Inherit from `quantized.Linear` (T83294430)
class Linear(torch.nn.Module):
    r"""
    A quantized sparse linear module with quantized tensor as inputs and outputs.
    """

    _version = 1
    _FLOAT_MODULE = torch.nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        row_block_size,
        col_block_size,
        bias=True,
        dtype=torch.qint8,
    ):
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError(
                "Only QINT8 is supported for Sparse Quantized Linear"
            )

        self.in_features = in_features
        self.out_features = out_features

        if bias:
            bias = torch.zeros(self.out_features, dtype=torch.float)
        else:
            bias = None

        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8
        )
        self._packed_params = LinearPackedParams(
            row_block_size=row_block_size, col_block_size=col_block_size, dtype=dtype
        )
        self._packed_params.set_weight_bias(
            qweight, bias, row_block_size, col_block_size
        )
        self.scale = 1.0
        self.zero_point = 0

    @classmethod
    def _get_name(cls):
        return "SparseQuantizedLinear"

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}, "
            f"zero_point={self.zero_point}, qscheme={self.weight().qscheme()}"
        )

    def __repr__(self):
        return _hide_packed_params_repr(self, LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sparse.qlinear(
            x, self._packed_params._packed_params, self.scale, self.zero_point
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "scale"] = torch.tensor(self.scale)
        destination[prefix + "zero_point"] = torch.tensor(self.zero_point)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.scale = float(state_dict[prefix + "scale"])
        state_dict.pop(prefix + "scale")

        self.zero_point = int(state_dict[prefix + "zero_point"])
        state_dict.pop(prefix + "zero_point")

        state_dict.pop(prefix + "op_type")

        version = local_metadata.get("version", None)
        if version is not None and version > self._version:
            raise AssertionError(f"version {version} > self._version {self._version}")

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _weight_bias(self):
        return self._packed_params._weight_bias()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def set_weight_bias(
        self,
        w: torch.Tensor,
        b: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None:
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                "row_block_size and col_block_size must not be None, "
                f"got row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized sparse module from a float module.

        We only care about the convert at this stage, no need for observers just yet.

        TODO(zaf): Need to add the sparse params to the qconfig
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                cls._get_name()
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        if not hasattr(mod, "sparse_params"):
            raise AssertionError(
                "Expecting the Linear to have `sparse_params`. Make sure you have provided arguments "
                'in the `sparsifier.squash_mask(params_to_save=("sparse_block_shape",))` method.'
            )
        sparse_block_shape = mod.sparse_params.get("sparse_block_shape", None)  # type: ignore[operator, union-attr]
        if not isinstance(sparse_block_shape, (tuple, list)):
            raise AssertionError(
                f"sparse_block_shape must be tuple or list, got {type(sparse_block_shape)}"
            )
        if len(sparse_block_shape) != 2:
            raise AssertionError(
                f"sparse_block_shape must have length 2, got {len(sparse_block_shape)}"
            )
        # TODO: Need to add options to qconfig to avoid the calibration.
        # TODO: Add calibration for the sparsity
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        activation_post_process = mod.activation_post_process
        weight_post_process = mod.qconfig.weight()  # type: ignore[operator, union-attr]

        # Assumption is that the weight is already sparsified by the
        # `sparsifier.convert`
        weight = mod.weight

        weight_post_process(weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()  # type: ignore[operator, union-attr]
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        w_sc, w_zp = weight_post_process.calculate_qparams()
        if isinstance(w_zp, torch.Tensor):
            if torch.any(w_zp.bool()):
                raise AssertionError("All weight zero points must map to 0")
        else:
            if w_zp != 0:
                raise AssertionError(f"Weight zero point must map to 0, got {w_zp}")
        qweight = _quantize_weight(weight.float(), weight_post_process)

        row_block_size = mod.sparse_params["sparse_block_shape"][0]  # type: ignore[index]
        col_block_size = mod.sparse_params["sparse_block_shape"][1]  # type: ignore[index]
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            row_block_size,
            col_block_size,
            dtype=dtype,
        )
        qlinear.set_weight_bias(
            qweight,
            mod.bias,
            row_block_size,  # type: ignore[arg-type]
            col_block_size,  # type: ignore[arg-type]
        )
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        return qlinear
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.ao.nn.intrinsic as nni
from torch.ao.nn.quantized.modules.utils import (
    _hide_packed_params_repr,
    _quantize_weight,
)
from torch.ao.nn.sparse.quantized import linear
from torch.ao.nn.sparse.quantized.utils import LinearBlockSparsePattern


__all__ = ["Linear"]


class Linear(torch.nn.Module):
    r"""
    A dynamically quantized sparse linear module with float tensor as inputs and outputs.
    """

    _version = 1
    _op_type = "sparse_dynamic"
    _FLOAT_MODULE = torch.nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        row_block_size,
        col_block_size,
        bias=True,
        dtype=torch.qint8,
    ):
        super().__init__()

        if dtype != torch.qint8:
            raise NotImplementedError(
                "Only QINT8 is supported for Sparse Quantized Linear Dynamic"
            )

        self.in_features = in_features
        self.out_features = out_features

        if bias:
            bias = torch.zeros(self.out_features, dtype=torch.float)
        else:
            bias = None

        qweight = torch._empty_affine_quantized(
            [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8
        )
        self._packed_params = linear.LinearPackedParams(
            row_block_size=row_block_size, col_block_size=col_block_size, dtype=dtype
        )
        self._packed_params.set_weight_bias(
            qweight, bias, row_block_size, col_block_size
        )

    def _get_name(self):
        return "SparseQuantizedDynamicLinear"

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, qscheme={self.weight().qscheme()}"

    def __repr__(self):
        return _hide_packed_params_repr(self, linear.LinearPackedParams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sparse.qlinear_dynamic(x, self._packed_params._packed_params)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "op_type"] = self._op_type

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        op_type = int(state_dict[prefix + "op_type"])
        if op_type != "sparse":
            raise AssertionError(
                f"Cannot load from op_type [{op_type}], expecting [{self._op_type}]"
            )
        state_dict.pop(prefix + "op_type")

        version = local_metadata.get("version", None)
        if version is not None and version > self._version:
            raise AssertionError(f"version {version} > self._version {self._version}")

        # Is this code valid? In old quantization it seemed to be used to load
        # older model
        weight = state_dict.pop(prefix + "weight")
        bias = state_dict.pop(prefix + "bias")
        state_dict.update(
            {
                prefix + "_packed_params.weight": weight,
                prefix + "_packed_params.bias": bias,
            }
        )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _weight_bias(self):
        return self._packed_params._weight_bias()

    def weight(self):
        return self._weight_bias()[0]

    def bias(self):
        return self._weight_bias()[1]

    def set_weight_bias(
        self,
        w: torch.Tensor,
        b: torch.Tensor | None,
        row_block_size: int | None,
        col_block_size: int | None,
    ) -> None:
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                f"row_block_size and col_block_size must not be None, got {row_block_size=}, {col_block_size=}"
            )
        self.out_features = w.shape[0]
        self.in_features = w.shape[1]
        self._packed_params.set_weight_bias(w, b, row_block_size, col_block_size)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        r"""Create a quantized sparse dynamic module from a float module.

        We only care about the convert at this stage, no need for observers just yet.
        """
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                " nnq."
                + cls.__name__
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        # TODO: Need to add options to qconfig to avoid the calibration.
        # TODO: Add calibration for the sparsity
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if type(mod) is nni.LinearReLU:
            mod = mod[0]
        # pyrefly: ignore [missing-attribute]
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            # pyrefly: ignore [not-callable]
            weight_observer = mod.qconfig.weight()
        else:
            # We have the circular import issues if we import the qconfig in the beginning of this file:
            # https://github.com/pytorch/pytorch/pull/24231. The current workaround is to postpone the
            # import until we need it.
            from torch.ao.quantization.qconfig import default_dynamic_qconfig

            weight_observer = default_dynamic_qconfig.weight()

        # It is important to multiply by the mask BEFORE calling the `weight_observer`
        # TODO (zaf): Mask might not be part of the qconfig (T83295194)
        weight = mod.weight
        if getattr(mod.qconfig, "mask", False):
            weight = mod.qconfig.mask * mod.weight

        weight_observer(weight)
        dtype = weight_observer.dtype
        if dtype != torch.qint8:
            raise AssertionError(
                f"Weight observer must have dtype torch.qint8, got {dtype}"
            )
        _w_sc, w_zp = weight_observer.calculate_qparams()
        if isinstance(w_zp, torch.Tensor):
            if torch.any(w_zp.bool()):
                raise AssertionError("All weight zero points must map to 0")
        else:
            if w_zp != 0:
                raise AssertionError(f"Weight zero point must map to 0, got {w_zp}")
        qweight = _quantize_weight(weight.float(), weight_observer)

        row_block_size, col_block_size = LinearBlockSparsePattern.block_size()
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            row_block_size,
            col_block_size,
            dtype=dtype,
        )
        # pyrefly: ignore [bad-argument-type]
        qlinear.set_weight_bias(qweight, mod.bias, row_block_size, col_block_size)
        return qlinear
<|endoftext|># mypy: allow-untyped-defs
import warnings

import torch
import torch.jit  # this is needed to avoid a circular import
import torch.nn.functional as F
from torch import nn, Tensor


__all__ = ["MultiheadAttention"]


class MultiheadAttention(nn.MultiheadAttention):
    _FLOAT_MODULE = nn.MultiheadAttention

    r"""Quantizable implementation of the MultiheadAttention.

    Note::
        Please, refer to :class:`~torch.nn.MultiheadAttention` for more
        information

    Allows the model to jointly attend to information from different
    representation subspaces.
    See reference: Attention Is All You Need

    The original MHA module is not quantizable.
    This reimplements it by explicitly instantiating the linear layers.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> multihead_attn = nnqa.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    Note::
        Please, follow the quantization flow to convert the quantizable MHA.
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            **factory_kwargs,
        )
        self.linear_Q = nn.Linear(
            self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs
        )
        self.linear_K = nn.Linear(
            self.kdim, self.embed_dim, bias=bias, **factory_kwargs
        )
        self.linear_V = nn.Linear(
            self.vdim, self.embed_dim, bias=bias, **factory_kwargs
        )
        # for the type: ignore, see https://github.com/pytorch/pytorch/issues/58969
        # pyrefly: ignore [bad-assignment]
        self.out_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs
        )  # type: ignore[assignment]

        # Functionals
        self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()
        # note: importing torch.ao.nn.quantized at top creates a circular import

        # Quant/Dequant
        self.quant_attn_output = torch.ao.quantization.QuantStub()
        self.quant_attn_output_weights = torch.ao.quantization.QuantStub()
        self.dequant_q = torch.ao.quantization.DeQuantStub()
        self.dequant_k = torch.ao.quantization.DeQuantStub()
        self.dequant_v = torch.ao.quantization.DeQuantStub()

    def _get_name(self):
        return "QuantizableMultiheadAttention"

    @classmethod
    def from_float(cls, other):
        if type(other) is not cls._FLOAT_MODULE:
            raise AssertionError(
                f"Expected type {cls._FLOAT_MODULE}, got {type(other)}"
            )
        if not hasattr(other, "qconfig"):
            raise AssertionError("The float module must have 'qconfig'")
        # Setting the dropout to 0.0!
        observed = cls(
            other.embed_dim,
            other.num_heads,
            other.dropout,
            (other.in_proj_bias is not None),
            (other.bias_k is not None),
            other.add_zero_attn,
            other.kdim,
            other.vdim,
            other.batch_first,
        )
        observed.bias_k = other.bias_k
        observed.bias_v = other.bias_v
        observed.qconfig = other.qconfig

        # Set the linear weights
        # for the type: ignores, see https://github.com/pytorch/pytorch/issues/58969
        observed.out_proj.weight = other.out_proj.weight
        observed.out_proj.bias = other.out_proj.bias
        if other._qkv_same_embed_dim:
            # Use separate params
            bias = other.in_proj_bias
            _start = 0
            _end = _start + other.embed_dim
            weight = other.in_proj_weight[_start:_end, :]
            if bias is not None:
                bias = torch.nn.Parameter(bias[_start:_end], bias.requires_grad)
            observed.linear_Q.weight = torch.nn.Parameter(weight, weight.requires_grad)
            observed.linear_Q.bias = bias

            bias = other.in_proj_bias
            _start = _end
            _end = _start + other.embed_dim
            weight = other.in_proj_weight[_start:_end, :]
            if bias is not None:
                bias = torch.nn.Parameter(bias[_start:_end], bias.requires_grad)
            observed.linear_K.weight = torch.nn.Parameter(weight, weight.requires_grad)
            observed.linear_K.bias = bias

            bias = other.in_proj_bias
            _start = _end
            weight = other.in_proj_weight[_start:, :]
            if bias is not None:
                bias = torch.nn.Parameter(bias[_start:], bias.requires_grad)
            observed.linear_V.weight = torch.nn.Parameter(weight, weight.requires_grad)
            observed.linear_V.bias = bias
        else:
            observed.linear_Q.weight = nn.Parameter(other.q_proj_weight)
            observed.linear_K.weight = nn.Parameter(other.k_proj_weight)
            observed.linear_V.weight = nn.Parameter(other.v_proj_weight)
            if other.in_proj_bias is None:
                # pyrefly: ignore [bad-assignment]
                observed.linear_Q.bias = None
                # pyrefly: ignore [bad-assignment]
                observed.linear_K.bias = None
                # pyrefly: ignore [bad-assignment]
                observed.linear_V.bias = None
            else:
                observed.linear_Q.bias = nn.Parameter(
                    other.in_proj_bias[0 : other.embed_dim]
                )
                observed.linear_K.bias = nn.Parameter(
                    other.in_proj_bias[other.embed_dim : (other.embed_dim * 2)]
                )
                observed.linear_V.bias = nn.Parameter(
                    other.in_proj_bias[(other.embed_dim * 2) :]
                )
        observed.eval()
        # Explicit prepare
        observed = torch.ao.quantization.prepare(observed, inplace=True)
        return observed

    @torch.jit.unused
    def dequantize(self):
        r"""Utility to convert the quantized MHA back to float.

        The motivation for this is that it is not trivial to convert the weights
        from the format that is used in the quantized version back to the
        float.
        """
        fp = self._FLOAT_MODULE(
            self.embed_dim,
            self.num_heads,
            self.dropout,
            (self.linear_Q._weight_bias()[1] is not None),  # type: ignore[operator]
            (self.bias_k is not None),
            self.add_zero_attn,
            self.kdim,
            self.vdim,
            self.batch_first,
        )
        if fp._qkv_same_embed_dim != self._qkv_same_embed_dim:
            raise AssertionError(
                f"_qkv_same_embed_dim mismatch: {fp._qkv_same_embed_dim} != {self._qkv_same_embed_dim}"
            )
        if self.bias_k is not None:
            fp.bias_k = nn.Parameter(self.bias_k.dequantize())
        if self.bias_v is not None:
            fp.bias_v = nn.Parameter(self.bias_v.dequantize())

        # Set the linear weights
        # Note: Because the linear layers are quantized, mypy does not know how
        # to deal with them -- might need to ignore the typing checks.
        # for the type: ignore[has-type], see https://github.com/pytorch/pytorch/issues/58969
        w, b = self.out_proj._weight_bias()  # type: ignore[operator, has-type]
        fp.out_proj.weight = nn.Parameter(w.dequantize())
        if b is not None:
            fp.out_proj.bias = nn.Parameter(b)

        wQ, bQ = self.linear_Q._weight_bias()  # type: ignore[operator]
        wQ = wQ.dequantize()
        wK, bK = self.linear_K._weight_bias()  # type: ignore[operator]
        wK = wK.dequantize()
        wV, bV = self.linear_V._weight_bias()  # type: ignore[operator]
        wV = wV.dequantize()
        if fp._qkv_same_embed_dim:
            # Use separate params
            _start = 0
            _end = _start + fp.embed_dim
            fp.in_proj_weight[_start:_end, :] = wQ
            if fp.in_proj_bias is not None:
                # pyrefly: ignore [bad-argument-type]
                if not all(bQ == 0):
                    raise AssertionError("Expected all bQ elements to be 0")
                fp.in_proj_bias[_start:_end] = bQ

            _start = _end
            _end = _start + fp.embed_dim
            fp.in_proj_weight[_start:_end, :] = wK
            if fp.in_proj_bias is not None:
                # pyrefly: ignore [bad-argument-type]
                if not all(bK == 0):
                    raise AssertionError("Expected all bK elements to be 0")
                fp.in_proj_bias[_start:_end] = bK

            _start = _end
            fp.in_proj_weight[_start:, :] = wV
            if fp.in_proj_bias is not None:
                # pyrefly: ignore [bad-argument-type]
                if not all(bV == 0):
                    raise AssertionError("Expected all bV elements to be 0")
                fp.in_proj_bias[_start:] = bV
        else:
            fp.q_proj_weight = nn.Parameter(wQ)
            fp.k_proj_weight = nn.Parameter(wK)
            fp.v_proj_weight = nn.Parameter(wV)
            if fp.in_proj_bias is None:
                # pyrefly: ignore [bad-assignment]
                self.linear_Q.bias = None
                # pyrefly: ignore [bad-assignment]
                self.linear_K.bias = None
                # pyrefly: ignore [bad-assignment]
                self.linear_V.bias = None
            else:
                fp.in_proj_bias[0 : fp.embed_dim] = bQ
                fp.in_proj_bias[fp.embed_dim : (fp.embed_dim * 2)] = bK
                fp.in_proj_bias[(fp.embed_dim * 2) :] = bV

        return fp

    @classmethod
    def from_observed(cls, other):
        # The whole flow is float -> observed -> quantized
        # This class does float -> observed only
        # See nn.quantized.MultiheadAttention
        raise NotImplementedError(
            "It looks like you are trying to prepare an "
            "MHA module. Please, see "
            "the examples on quantizable MHAs."
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        r"""
        Note::
            Please, refer to :func:`~torch.nn.MultiheadAttention.forward` for more
            information

        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
              Default: ``False``.
            - average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
              heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
              effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: If ``average_attn_weights=True``, returns attention weights averaged
              across heads of shape :math:`(N, L, S)`, where N is the batch size, L is the target sequence length,
              S is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(N, num_heads, L, S)`.
        """
        return self._forward_impl(
            query,
            key,
            value,
            key_padding_mask,
            need_weights,
            attn_mask,
            average_attn_weights,
            is_causal,
        )

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        # This version will not deal with the static key/value pairs.
        # Keeping it here for future changes.
        #
        # TODO: This method has some duplicate lines with the
        # `torch.nn.functional.multi_head_attention`. Will need to refactor.
        static_k = None
        static_v = None

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        if is_causal:
            raise AssertionError("causal mask not supported by AO MHA module")

        if self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))

        tgt_len, bsz, embed_dim_to_check = query.size()
        if self.embed_dim != embed_dim_to_check:
            raise AssertionError(
                f"embed_dim mismatch: {self.embed_dim} != {embed_dim_to_check}"
            )
        # allow MHA to have different sizes for the feature dimension
        if key.size(0) != value.size(0) or key.size(1) != value.size(1):
            raise AssertionError(
                f"key and value size mismatch: key.size()={key.size()}, value.size()={value.size()}"
            )

        head_dim = self.embed_dim // self.num_heads
        if head_dim * self.num_heads != self.embed_dim:
            raise AssertionError("embed_dim must be divisible by num_heads")
        scaling = float(head_dim) ** -0.5

        q = self.linear_Q(query)
        k = self.linear_K(key)
        v = self.linear_V(value)

        q = self.q_scaling_product.mul_scalar(q, scaling)

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for `attn_mask` in `nn.MultiheadAttention` is deprecated. "
                    "Use bool tensor instead.",
                    stacklevel=3,
                )
                attn_mask = attn_mask.to(torch.bool)
            if not attn_mask.is_floating_point() and attn_mask.dtype != torch.bool:
                raise AssertionError(
                    f"Only float and bool types are supported for attn_mask, not {attn_mask.dtype}"
                )

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for `key_padding_mask` in `nn.MultiheadAttention` is deprecated. "
                "Use bool tensor instead.",
                stacklevel=3,
            )
            key_padding_mask = key_padding_mask.to(torch.bool)
        if self.bias_k is not None and self.bias_v is not None:
            if static_k is None and static_v is None:
                # Explicitly check that bias_k and bias_v are not None
                # in a way that TorchScript can understand.
                bias_k = self.bias_k
                if bias_k is None:
                    raise AssertionError("bias_k must not be None")
                bias_v = self.bias_v
                if bias_v is None:
                    raise AssertionError("bias_v must not be None")

                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))
            else:
                if static_k is not None:
                    raise AssertionError("bias cannot be added to static key.")
                if static_v is not None:
                    raise AssertionError("bias cannot be added to static value.")
        else:
            if self.bias_k is not None:
                raise AssertionError(
                    "self.bias_k must be None when self.bias_v is None"
                )
            if self.bias_v is not None:
                raise AssertionError(
                    "self.bias_v must be None when self.bias_k is None"
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            if static_k.size(0) != bsz * self.num_heads:
                raise AssertionError(
                    f"static_k.size(0) must be {bsz * self.num_heads}, got {static_k.size(0)}"
                )
            if static_k.size(2) != head_dim:
                raise AssertionError(
                    f"static_k.size(2) must be {head_dim}, got {static_k.size(2)}"
                )
            k = static_k

        if static_v is not None:
            if static_v.size(0) != bsz * self.num_heads:
                raise AssertionError(
                    f"static_v.size(0) must be {bsz * self.num_heads}, got {static_v.size(0)}"
                )
            if static_v.size(2) != head_dim:
                raise AssertionError(
                    f"static_v.size(2) must be {head_dim}, got {static_v.size(2)}"
                )
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz:
                raise AssertionError(
                    f"key_padding_mask.size(0) must be {bsz}, got {key_padding_mask.size(0)}"
                )
            if key_padding_mask.size(1) != src_len:
                raise AssertionError(
                    f"key_padding_mask.size(1) must be {src_len}, got {key_padding_mask.size(1)}"
                )

        if self.add_zero_attn:
            src_len += 1

            k_zeros = torch.zeros((k.size(0), 1) + k.size()[2:])

            if k.is_quantized:
                k_zeros = torch.quantize_per_tensor(
                    k_zeros,
                    k.q_scale(),
                    k.q_zero_point(),
                    k.dtype,
                )

            k = torch.cat([k, k_zeros], dim=1)

            v_zeros = torch.zeros((v.size(0), 1) + k.size()[2:])

            if v.is_quantized:
                v_zeros = torch.quantize_per_tensor(
                    v_zeros,
                    v.q_scale(),
                    v.q_zero_point(),
                    v.dtype,
                )

            v = torch.cat([v, v_zeros], dim=1)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Leaving the quantized zone here
        q = self.dequant_q(q)
        k = self.dequant_k(k)
        v = self.dequant_v(v)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        expected_size = [bsz * self.num_heads, tgt_len, src_len]
        if list(attn_output_weights.size()) != expected_size:
            raise AssertionError(
                f"attn_output_weights size mismatch: expected {expected_size}, "
                f"got {list(attn_output_weights.size())}"
            )

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(
            attn_output_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        expected_output_size = [bsz * self.num_heads, tgt_len, head_dim]
        if list(attn_output.size()) != expected_output_size:
            raise AssertionError(
                f"attn_output size mismatch: expected {expected_output_size}, "
                f"got {list(attn_output.size())}"
            )
        if self.batch_first:
            attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        else:
            attn_output = (
                attn_output.transpose(0, 1)
                .contiguous()
                .view(tgt_len, bsz, self.embed_dim)
            )

        # Reentering the quantized zone
        attn_output = self.quant_attn_output(attn_output)
        # for the type: ignore[has-type], see https://github.com/pytorch/pytorch/issues/58969
        attn_output = self.out_proj(attn_output)  # type: ignore[has-type]
        attn_output_weights = self.quant_attn_output_weights(attn_output_weights)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None
<|endoftext|>"""
We will recreate all the RNN modules as we require the modules to be decomposed
into its building blocks to be able to observe.
"""

# mypy: allow-untyped-defs

import numbers
import warnings

import torch
from torch import Tensor


__all__ = ["LSTMCell", "LSTM"]


class LSTMCell(torch.nn.Module):
    r"""A quantizable long short-term memory (LSTM) cell.

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTMCell`

    `split_gates`: specify True to compute the input/forget/cell/output gates separately
    to avoid an intermediate tensor which is subsequently chunk'd. This optimization can
    be beneficial for on-device inference latency. This flag is cascaded down from the
    parent classes.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTMCell(10, 20)
        >>> input = torch.randn(6, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    _FLOAT_MODULE = torch.nn.LSTMCell
    __constants__ = ["split_gates"]  # for jit.script

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        split_gates=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.bias = bias
        self.split_gates = split_gates

        if not split_gates:
            self.igates: torch.nn.Module = torch.nn.Linear(
                input_dim, 4 * hidden_dim, bias=bias, **factory_kwargs
            )
            self.hgates: torch.nn.Module = torch.nn.Linear(
                hidden_dim, 4 * hidden_dim, bias=bias, **factory_kwargs
            )
            self.gates: torch.nn.Module = torch.ao.nn.quantized.FloatFunctional()
        else:
            # keep separate Linear layers for each gate
            self.igates = torch.nn.ModuleDict()
            self.hgates = torch.nn.ModuleDict()
            self.gates = torch.nn.ModuleDict()
            for g in ["input", "forget", "cell", "output"]:
                # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
                self.igates[g] = torch.nn.Linear(
                    input_dim, hidden_dim, bias=bias, **factory_kwargs
                )
                # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
                self.hgates[g] = torch.nn.Linear(
                    hidden_dim, hidden_dim, bias=bias, **factory_kwargs
                )
                # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
                self.gates[g] = torch.ao.nn.quantized.FloatFunctional()

        self.input_gate = torch.nn.Sigmoid()
        self.forget_gate = torch.nn.Sigmoid()
        self.cell_gate = torch.nn.Tanh()
        self.output_gate = torch.nn.Sigmoid()

        self.fgate_cx = torch.ao.nn.quantized.FloatFunctional()
        self.igate_cgate = torch.ao.nn.quantized.FloatFunctional()
        self.fgate_cx_igate_cgate = torch.ao.nn.quantized.FloatFunctional()

        self.ogate_cy = torch.ao.nn.quantized.FloatFunctional()

        self.initial_hidden_state_qparams: tuple[float, int] = (1.0, 0)
        self.initial_cell_state_qparams: tuple[float, int] = (1.0, 0)
        self.hidden_state_dtype: torch.dtype = torch.quint8
        self.cell_state_dtype: torch.dtype = torch.quint8

    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        if hidden is None or hidden[0] is None or hidden[1] is None:
            hidden = self.initialize_hidden(x.shape[0], x.is_quantized)
        hx, cx = hidden

        if not self.split_gates:
            igates = self.igates(x)
            hgates = self.hgates(hx)
            gates = self.gates.add(igates, hgates)  # type: ignore[operator]

            input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

            input_gate = self.input_gate(input_gate)
            forget_gate = self.forget_gate(forget_gate)
            cell_gate = self.cell_gate(cell_gate)
            out_gate = self.output_gate(out_gate)
        else:
            # apply each input + hidden projection and add together
            gate = {}
            for (key, gates), igates, hgates in zip(
                self.gates.items(),  # type: ignore[operator]
                self.igates.values(),  # type: ignore[operator]
                self.hgates.values(),  # type: ignore[operator]
            ):
                gate[key] = gates.add(igates(x), hgates(hx))

            input_gate = self.input_gate(gate["input"])
            forget_gate = self.forget_gate(gate["forget"])
            cell_gate = self.cell_gate(gate["cell"])
            out_gate = self.output_gate(gate["output"])

        fgate_cx = self.fgate_cx.mul(forget_gate, cx)
        igate_cgate = self.igate_cgate.mul(input_gate, cell_gate)
        fgate_cx_igate_cgate = self.fgate_cx_igate_cgate.add(fgate_cx, igate_cgate)
        cy = fgate_cx_igate_cgate

        # TODO: make this tanh a member of the module so its qparams can be configured
        tanh_cy = torch.tanh(cy)
        hy = self.ogate_cy.mul(out_gate, tanh_cy)
        return hy, cy

    def initialize_hidden(
        self, batch_size: int, is_quantized: bool = False
    ) -> tuple[Tensor, Tensor]:
        h, c = (
            torch.zeros((batch_size, self.hidden_size)),
            torch.zeros((batch_size, self.hidden_size)),
        )
        if is_quantized:
            (h_scale, h_zp) = self.initial_hidden_state_qparams
            (c_scale, c_zp) = self.initial_cell_state_qparams
            h = torch.quantize_per_tensor(
                h, scale=h_scale, zero_point=h_zp, dtype=self.hidden_state_dtype
            )
            c = torch.quantize_per_tensor(
                c, scale=c_scale, zero_point=c_zp, dtype=self.cell_state_dtype
            )
        return h, c

    def _get_name(self):
        return "QuantizableLSTMCell"

    @classmethod
    def from_params(cls, wi, wh, bi=None, bh=None, split_gates=False):
        """Uses the weights and biases to create a new LSTM cell.

        Args:
            wi, wh: Weights for the input and hidden layers
            bi, bh: Biases for the input and hidden layers
        """
        if (bi is None) != (bh is None):
            raise AssertionError("bi and bh must both be None or both have values")
        input_size = wi.shape[1]
        hidden_size = wh.shape[1]
        cell = cls(
            input_dim=input_size,
            hidden_dim=hidden_size,
            bias=(bi is not None),
            split_gates=split_gates,
        )

        if not split_gates:
            cell.igates.weight = torch.nn.Parameter(wi)
            if bi is not None:
                cell.igates.bias = torch.nn.Parameter(bi)
            cell.hgates.weight = torch.nn.Parameter(wh)
            if bh is not None:
                cell.hgates.bias = torch.nn.Parameter(bh)
        else:
            # split weight/bias
            for w, b, gates in zip([wi, wh], [bi, bh], [cell.igates, cell.hgates]):
                for w_chunk, gate in zip(w.chunk(4, dim=0), gates.values()):  # type: ignore[operator]
                    gate.weight = torch.nn.Parameter(w_chunk)

                if b is not None:
                    for b_chunk, gate in zip(b.chunk(4, dim=0), gates.values()):  # type: ignore[operator]
                        gate.bias = torch.nn.Parameter(b_chunk)

        return cell

    @classmethod
    def from_float(cls, other, use_precomputed_fake_quant=False, split_gates=False):
        if type(other) is not cls._FLOAT_MODULE:
            raise AssertionError(
                f"Expected module type {cls._FLOAT_MODULE}, got {type(other)}"
            )
        if not hasattr(other, "qconfig"):
            raise AssertionError("The float module must have 'qconfig'")
        observed = cls.from_params(
            other.weight_ih,
            other.weight_hh,
            other.bias_ih,
            other.bias_hh,
            split_gates=split_gates,
        )
        observed.qconfig = other.qconfig
        observed.igates.qconfig = other.qconfig
        observed.hgates.qconfig = other.qconfig
        if split_gates:
            # also apply qconfig directly to Linear modules
            for g in observed.igates.values():
                g.qconfig = other.qconfig
            for g in observed.hgates.values():
                g.qconfig = other.qconfig
        return observed


class _LSTMSingleLayer(torch.nn.Module):
    r"""A single one-directional LSTM layer.

    The difference between a layer and a cell is that the layer can process a
    sequence, while the cell only expects an instantaneous value.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        split_gates=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.cell = LSTMCell(
            input_dim, hidden_dim, bias=bias, split_gates=split_gates, **factory_kwargs
        )

    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None):
        result = []
        seq_len = x.shape[0]
        for i in range(seq_len):
            hidden = self.cell(x[i], hidden)
            result.append(hidden[0])  # type: ignore[index]
        result_tensor = torch.stack(result, 0)
        return result_tensor, hidden

    @classmethod
    def from_params(cls, *args, **kwargs):
        cell = LSTMCell.from_params(*args, **kwargs)
        layer = cls(
            cell.input_size, cell.hidden_size, cell.bias, split_gates=cell.split_gates
        )
        layer.cell = cell
        return layer


class _LSTMLayer(torch.nn.Module):
    r"""A single bi-directional LSTM layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bias: bool = True,
        batch_first: bool = False,
        bidirectional: bool = False,
        device=None,
        dtype=None,
        *,
        split_gates=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.layer_fw = _LSTMSingleLayer(
            input_dim, hidden_dim, bias=bias, split_gates=split_gates, **factory_kwargs
        )
        if self.bidirectional:
            self.layer_bw = _LSTMSingleLayer(
                input_dim,
                hidden_dim,
                bias=bias,
                split_gates=split_gates,
                **factory_kwargs,
            )

    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None):
        if self.batch_first:
            x = x.transpose(0, 1)
        if hidden is None:
            hx_fw, cx_fw = (None, None)
        else:
            hx_fw, cx_fw = hidden
        hidden_bw: tuple[Tensor, Tensor] | None = None
        if self.bidirectional:
            if hx_fw is None:
                hx_bw = None
            else:
                hx_bw = hx_fw[1]
                hx_fw = hx_fw[0]
            if cx_fw is None:
                cx_bw = None
            else:
                cx_bw = cx_fw[1]
                cx_fw = cx_fw[0]
            if hx_bw is not None and cx_bw is not None:
                hidden_bw = hx_bw, cx_bw
        if hx_fw is None and cx_fw is None:
            hidden_fw = None
        else:
            hidden_fw = (
                torch.jit._unwrap_optional(hx_fw),
                torch.jit._unwrap_optional(cx_fw),
            )
        result_fw, hidden_fw = self.layer_fw(x, hidden_fw)

        if hasattr(self, "layer_bw") and self.bidirectional:
            x_reversed = x.flip(0)
            result_bw, hidden_bw = self.layer_bw(x_reversed, hidden_bw)
            result_bw = result_bw.flip(0)

            result = torch.cat([result_fw, result_bw], result_fw.dim() - 1)
            if hidden_fw is None and hidden_bw is None:
                h = None
                c = None
            elif hidden_fw is None:
                (h, c) = torch.jit._unwrap_optional(hidden_bw)
            elif hidden_bw is None:
                (h, c) = torch.jit._unwrap_optional(hidden_fw)
            else:
                h = torch.stack([hidden_fw[0], hidden_bw[0]], 0)  # type: ignore[list-item]
                c = torch.stack([hidden_fw[1], hidden_bw[1]], 0)  # type: ignore[list-item]
        else:
            result = result_fw
            h, c = torch.jit._unwrap_optional(hidden_fw)  # type: ignore[assignment]

        if self.batch_first:
            result.transpose_(0, 1)

        return result, (h, c)

    @classmethod
    def from_float(cls, other, layer_idx=0, qconfig=None, **kwargs):
        r"""
        There is no FP equivalent of this class. This function is here just to
        mimic the behavior of the `prepare` within the `torch.ao.quantization`
        flow.
        """
        if not hasattr(other, "qconfig") and qconfig is None:
            raise AssertionError("other must have qconfig or qconfig must be provided")

        input_size = kwargs.get("input_size", other.input_size)
        hidden_size = kwargs.get("hidden_size", other.hidden_size)
        bias = kwargs.get("bias", other.bias)
        batch_first = kwargs.get("batch_first", other.batch_first)
        bidirectional = kwargs.get("bidirectional", other.bidirectional)
        split_gates = kwargs.get("split_gates", False)

        layer = cls(
            # pyrefly: ignore [bad-argument-type]
            input_size,
            # pyrefly: ignore [bad-argument-type]
            hidden_size,
            # pyrefly: ignore [bad-argument-type]
            bias,
            # pyrefly: ignore [bad-argument-type]
            batch_first,
            # pyrefly: ignore [bad-argument-type]
            bidirectional,
            split_gates=split_gates,
        )
        # pyrefly: ignore [bad-argument-type]
        layer.qconfig = getattr(other, "qconfig", qconfig)
        wi = getattr(other, f"weight_ih_l{layer_idx}")
        wh = getattr(other, f"weight_hh_l{layer_idx}")
        bi = getattr(other, f"bias_ih_l{layer_idx}", None)
        bh = getattr(other, f"bias_hh_l{layer_idx}", None)

        layer.layer_fw = _LSTMSingleLayer.from_params(
            wi, wh, bi, bh, split_gates=split_gates
        )

        if other.bidirectional:
            wi = getattr(other, f"weight_ih_l{layer_idx}_reverse")
            wh = getattr(other, f"weight_hh_l{layer_idx}_reverse")
            bi = getattr(other, f"bias_ih_l{layer_idx}_reverse", None)
            bh = getattr(other, f"bias_hh_l{layer_idx}_reverse", None)
            layer.layer_bw = _LSTMSingleLayer.from_params(
                wi, wh, bi, bh, split_gates=split_gates
            )
        return layer


class LSTM(torch.nn.Module):
    r"""A quantizable long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples below.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
        >>> # To get the weights:
        >>> # xdoctest: +SKIP
        >>> print(rnn.layers[0].weight_ih)
        tensor([[...]])
        >>> print(rnn.layers[0].weight_hh)
        AssertionError: There is no reverse path in the non-bidirectional layer
    """

    _FLOAT_MODULE = torch.nn.LSTM

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
        *,
        split_gates: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.training = False  # Default to eval mode. If we want to train, we will explicitly set to training.

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )

        if dropout > 0:
            warnings.warn(
                "dropout option for quantizable LSTM is ignored. "
                "If you are training, please, use nn.LSTM version "
                "followed by `prepare` step.",
                stacklevel=2,
            )
            if num_layers == 1:
                warnings.warn(
                    "dropout option adds dropout after all but last "
                    "recurrent layer, so non-zero dropout expects "
                    f"num_layers greater than 1, but got dropout={dropout} "
                    f"and num_layers={num_layers}",
                    stacklevel=2,
                )

        layers = [
            _LSTMLayer(
                self.input_size,
                self.hidden_size,
                self.bias,
                batch_first=False,
                bidirectional=self.bidirectional,
                split_gates=split_gates,
                **factory_kwargs,
            )
        ]
        layers.extend(
            _LSTMLayer(
                self.hidden_size,
                self.hidden_size,
                self.bias,
                batch_first=False,
                bidirectional=self.bidirectional,
                split_gates=split_gates,
                **factory_kwargs,
            )
            for _ in range(1, num_layers)
        )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        max_batch_size = x.size(1)
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            zeros = torch.zeros(
                num_directions,
                max_batch_size,
                self.hidden_size,
                dtype=torch.float,
                device=x.device,
            )
            zeros.squeeze_(0)
            if x.is_quantized:
                zeros = torch.quantize_per_tensor(
                    zeros, scale=1.0, zero_point=0, dtype=x.dtype
                )
            hxcx = [(zeros, zeros) for _ in range(self.num_layers)]
        else:
            hidden_non_opt = torch.jit._unwrap_optional(hidden)
            if isinstance(hidden_non_opt[0], Tensor):
                hx = hidden_non_opt[0].reshape(
                    self.num_layers, num_directions, max_batch_size, self.hidden_size
                )
                cx = hidden_non_opt[1].reshape(
                    self.num_layers, num_directions, max_batch_size, self.hidden_size
                )
                hxcx = [
                    (hx[idx].squeeze(0), cx[idx].squeeze(0))
                    for idx in range(self.num_layers)
                ]
            else:
                hxcx = hidden_non_opt

        hx_list = []
        cx_list = []
        for idx, layer in enumerate(self.layers):
            x, (h, c) = layer(x, hxcx[idx])
            hx_list.append(torch.jit._unwrap_optional(h))
