        is_outputs_tuple, outputs = output_info
        # Step 3: for each of the output tangents, split along dim 0
        jacobian_input_output = []
        for jac_output_i, output_i in zip(outputs_before_split, outputs):
            jacobian_output_i_output = []
            for jac, input_j in zip(jac_output_i.split(input_numels, dim=0), inputs):
                # We need to transpose the Jacobian because in forward AD, the
                # batch dimension represents that of the inputs
                jacobian_input_i_output_j = jac.permute(*range(1, jac.ndim), 0).reshape(
                    (*output_i.shape, *input_j.shape)
                )

                jacobian_output_i_output.append(jacobian_input_i_output_j)
            jacobian_input_output.append(jacobian_output_i_output)

        # Omit [Step 4] because everything is already transposed w/ forward AD
        return _tuple_postprocess(
            jacobian_input_output, (is_outputs_tuple, is_inputs_tuple)
        )
    else:
        raise NotImplementedError(
            "Computing Jacobian using forward-AD or forward-over-reverse Hessian is"
            "only implemented for `vectorize=True`."
        )


def jacobian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    strategy="reverse-mode",
):
    r"""Compute the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): This feature is experimental.
            Please consider using :func:`torch.func.jacrev` or
            :func:`torch.func.jacfwd` instead if you are looking for something
            less experimental and more performant.
            When computing the jacobian, usually we invoke
            ``autograd.grad`` once per row of the jacobian. If this flag is
            ``True``, we perform only a single ``autograd.grad`` call with
            ``batched_grad=True`` which uses the vmap prototype feature.
            Though this should lead to performance improvements in many cases,
            because this feature is still experimental, there may be performance
            cliffs. See :func:`torch.autograd.grad`'s ``batched_grad`` parameter for
            more information.
        strategy (str, optional): Set to ``"forward-mode"`` or ``"reverse-mode"`` to
            determine whether the Jacobian will be computed with forward or reverse
            mode AD. Currently, ``"forward-mode"`` requires ``vectorized=True``.
            Defaults to ``"reverse-mode"``. If ``func`` has more outputs than
            inputs, ``"forward-mode"`` tends to be more performant. Otherwise,
            prefer to use ``"reverse-mode"``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors): if there is a single
        input and output, this will be a single Tensor containing the
        Jacobian for the linearized inputs and output. If one of the two is
        a tuple, then the Jacobian will be a tuple of Tensors. If both of
        them are tuples, then the Jacobian will be a tuple of tuple of
        Tensors where ``Jacobian[i][j]`` will contain the Jacobian of the
        ``i``\th output and ``j``\th input and will have as size the
        concatenation of the sizes of the corresponding output and the
        corresponding input and will have same dtype and device as the
        corresponding input. If strategy is ``forward-mode``, the dtype will be
        that of the output; otherwise, the input.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> jacobian(exp_reducer, inputs)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]])

        >>> jacobian(exp_reducer, inputs, create_graph=True)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],
                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]], grad_fn=<ViewBackward>)

        >>> def exp_adder(x, y):
        ...     return 2 * x.exp() + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> jacobian(exp_adder, inputs)
        (tensor([[2.8052, 0.0000],
                [0.0000, 3.3963]]),
         tensor([[3., 0.],
                 [0., 3.]]))

        >>> def linear_model(x):
        ...     W = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
        ...     b = torch.tensor([1.0, 0.5])
        ...     return x @ W.T + b

        >>> x = torch.randn(4, 2, requires_grad=True)
        >>> jac = jacobian(linear_model, x, vectorize=True)
        >>> jac.shape
        torch.Size([4, 2, 4, 2])
    """
    if strategy not in ("forward-mode", "reverse-mode"):
        raise AssertionError(
            'Expected strategy to be either "forward-mode" or "reverse-mode". Hint: If your '
            'function has more outputs than inputs, "forward-mode" tends to be more performant. '
            'Otherwise, prefer to use "reverse-mode".'
        )
    if strategy == "forward-mode":
        if create_graph:
            raise NotImplementedError(
                "torch.autograd.functional.jacobian: `create_graph=True` "
                'and `strategy="forward-mode"` are not supported together (yet). '
                "Please either set `create_graph=False` or "
                '`strategy="reverse-mode"`.'
            )
        return _jacfwd(func, inputs, strict, vectorize)

    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jacobian"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if vectorize:
            if strict:
                raise RuntimeError(
                    "torch.autograd.functional.jacobian: `strict=True` "
                    "and `vectorized=True` are not supported together. "
                    "Please either set `strict=False` or "
                    "`vectorize=False`."
                )
            # NOTE: [Computing jacobian with vmap and grad for multiple outputs]
            #
            # Let's consider f(x) = (x**2, x.sum()) and let x = torch.randn(3).
            # It turns out we can compute the jacobian of this function with a single
            # call to autograd.grad by using vmap over the correct grad_outputs.
            #
            # Firstly, one way to compute the jacobian is to stack x**2 and x.sum()
            # into a 4D vector. E.g., use g(x) = torch.stack([x**2, x.sum()])
            #
            # To get the first row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([1, 0, 0, 0]))
            # To get the 2nd row of the jacobian, we call
            # >>> autograd.grad(g(x), x, grad_outputs=torch.tensor([0, 1, 0, 0]))
            # and so on.
            #
            # Using vmap, we can vectorize all 4 of these computations into one by
            # passing the standard basis for R^4 as the grad_output.
            # vmap(partial(autograd.grad, g(x), x))(torch.eye(4)).
            #
            # Now, how do we compute the jacobian *without stacking the output*?
            # We can just split the standard basis across the outputs. So to
            # compute the jacobian of f(x), we'd use
            # >>> autograd.grad(f(x), x, grad_outputs=_construct_standard_basis_for(...))
            # The grad_outputs looks like the following:
            # ( torch.tensor([[1, 0, 0],
            #                 [0, 1, 0],
            #                 [0, 0, 1],
            #                 [0, 0, 0]]),
            #   torch.tensor([[0],
            #                 [0],
            #                 [0],
            #                 [1]]) )
            #
            # But we're not done yet!
            # >>> vmap(partial(autograd.grad(f(x), x, grad_outputs=...)))
            # returns a Tensor of shape [4, 3]. We have to remember to split the
            # jacobian of shape [4, 3] into two:
            # - one of shape [3, 3] for the first output
            # - one of shape [   3] for the second output

            # Step 1: Construct grad_outputs by splitting the standard basis
            output_numels = tuple(output.numel() for output in outputs)
            grad_outputs = _construct_standard_basis_for(outputs, output_numels)
            flat_outputs = tuple(output.reshape(-1) for output in outputs)

            # Step 2: Call vmap + autograd.grad
            def vjp(grad_output):
                vj = list(
                    _autograd_grad(
                        flat_outputs,
                        inputs,
                        grad_output,
                        create_graph=create_graph,
                        is_grads_batched=True,
                    )
                )
                for el_idx, vj_el in enumerate(vj):
                    if vj_el is not None:
                        continue
                    vj[el_idx] = torch.zeros_like(inputs[el_idx]).expand(
                        (sum(output_numels),) + inputs[el_idx].shape
                    )
                return tuple(vj)

            jacobians_of_flat_output = vjp(grad_outputs)

            # Step 3: The returned jacobian is one big tensor per input. In this step,
            # we split each Tensor by output.
            jacobian_input_output = []
            for jac_input_i, input_i in zip(jacobians_of_flat_output, inputs):
                jacobian_input_i_output = []
                for jac, output_j in zip(
                    jac_input_i.split(output_numels, dim=0), outputs
                ):
                    jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
                    jacobian_input_i_output.append(jacobian_input_i_output_j)
                jacobian_input_output.append(jacobian_input_i_output)

            # Step 4: Right now, `jacobian` is a List[List[Tensor]].
            # The outer List corresponds to the number of inputs,
            # the inner List corresponds to the number of outputs.
            # We need to exchange the order of these and convert to tuples
            # before returning.
            jacobian_output_input = tuple(zip(*jacobian_input_output))

            jacobian_output_input = _grad_postprocess(
                jacobian_output_input, create_graph
            )
            return _tuple_postprocess(
                jacobian_output_input, (is_outputs_tuple, is_inputs_tuple)
            )

        jacobian: tuple[torch.Tensor, ...] = ()

        for i, out in enumerate(outputs):
            # mypy complains that expression and variable have different types due to the empty list
            jac_i: tuple[list[torch.Tensor]] = tuple([] for _ in range(len(inputs)))  # type: ignore[assignment]
            for j in range(out.nelement()):
                vj = _autograd_grad(
                    (out.reshape(-1)[j],),
                    inputs,
                    retain_graph=True,
                    create_graph=create_graph,
                )

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(
                    zip(jac_i, vj, inputs)
                ):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = (
                                "The jacobian of the user-provided function is "
                                f"independent of input {i}. This is not allowed in "
                                "strict mode when create_graph=True."
                            )
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = (
                                f"Output {i} of the user-provided function is "
                                f"independent of input {el_idx}. This is not allowed in "
                                "strict mode."
                            )
                            raise RuntimeError(msg)
                        jac_i_el.append(torch.zeros_like(inp_el))

            # pyrefly: ignore [bad-assignment]
            jacobian += (
                tuple(
                    torch.stack(jac_i_el, dim=0).view(
                        out.size() + inputs[el_idx].size()  # type: ignore[operator]
                    )
                    for (el_idx, jac_i_el) in enumerate(jac_i)
                ),
            )

        jacobian = _grad_postprocess(jacobian, create_graph)

        return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple))


def hessian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
):
    r"""Compute the Hessian of a given scalar function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Hessian will be computed in
            a differentiable manner. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            hessian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
        vectorize (bool, optional): This feature is experimental.
            Please consider using :func:`torch.func.hessian`
            instead if you are looking for something less experimental and more performant.
            When computing the hessian, usually we invoke
            ``autograd.grad`` once per row of the hessian. If this flag is
            ``True``, we use the vmap prototype feature as the backend to
            vectorize calls to ``autograd.grad`` so we only invoke it once
            instead of once per row. This should lead to performance
            improvements in many use cases, however, due to this feature
            being incomplete, there may be performance cliffs. Please
            use `torch._C._debug_only_display_vmap_fallback_warnings(True)`
            to show any performance warnings and file us issues if
            warnings exist for your use case. Defaults to ``False``.
        outer_jacobian_strategy (str, optional): The Hessian is computed by
            computing the Jacobian of a Jacobian. The inner Jacobian is always
            computed in reverse-mode AD. Setting strategy to ``"forward-mode"``
            or ``"reverse-mode"`` determines whether the outer Jacobian will be
            computed with forward or reverse mode AD. Currently, computing the outer
            Jacobian in ``"forward-mode"`` requires ``vectorized=True``. Defaults
            to ``"reverse-mode"``.

    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors): if there is a single input,
        this will be a single Tensor containing the Hessian for the input.
        If it is a tuple, then the Hessian will be a tuple of tuples where
        ``Hessian[i][j]`` will contain the Hessian of the ``i``\th input
        and ``j``\th input with size the sum of the size of the ``i``\th input plus
        the size of the ``j``\th input. ``Hessian[i][j]`` will have the same
        dtype and device as the corresponding ``i``\th input.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> hessian(pow_reducer, inputs)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],
                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],
                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],
                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]])

        >>> hessian(pow_reducer, inputs, create_graph=True)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],
                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],
                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],
                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]], grad_fn=<ViewBackward>)


        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> hessian(pow_adder_reducer, inputs)
        ((tensor([[4., 0.],
                  [0., 4.]]),
          tensor([[0., 0.],
                  [0., 0.]])),
         (tensor([[0., 0.],
                  [0., 0.]]),
          tensor([[6., 0.],
                  [0., 6.]])))
    """
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hessian")
    if outer_jacobian_strategy not in (
        "forward-mode",
        "reverse-mode",
    ):
        raise AssertionError(
            'Expected strategy to be either "forward-mode" or "reverse-mode".'
        )

    def ensure_single_output_function(*inp):
        out = func(*inp)
        is_out_tuple, t_out = _as_tuple(
            out, "outputs of the user-provided function", "hessian"
        )
        _check_requires_grad(t_out, "outputs", strict=strict)

        if is_out_tuple or not isinstance(out, torch.Tensor):
            raise RuntimeError(
                "The function given to hessian should return a single Tensor"
            )

        if out.nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hessian should contain a single element"
            )

        return out.squeeze()

    def jac_func(*inp):
        if outer_jacobian_strategy == "forward-mode":
            # _grad_preprocess requires create_graph=True and input to require_grad
            # or else the input will be detached
            inp = tuple(t.requires_grad_(True) for t in inp)
        jac = jacobian(ensure_single_output_function, inp, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
        return jac

    res = jacobian(
        jac_func,
        inputs,
        create_graph=create_graph,
        strict=strict,
        vectorize=vectorize,
        strategy=outer_jacobian_strategy,
    )
    return _tuple_postprocess(res, (is_inputs_tuple, is_inputs_tuple))


def vhp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between vector ``v`` and Hessian of a  given scalar function at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Hessian
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vhp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vhp (tuple of Tensors or Tensor): result of the dot product with the
            same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> vhp(pow_reducer, inputs, v)
        (tensor(0.5591),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]]))
        >>> vhp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.5591, grad_fn=<SumBackward0>),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]], grad_fn=<MulBackward0>))
        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> vhp(pow_adder_reducer, inputs, v)
        (tensor(4.8053),
         (tensor([0., 0.]),
          tensor([6., 6.])))
    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vhp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "vhp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vhp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError(
                "The function given to vhp should return a single Tensor"
            )

        if outputs[0].nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to vhp should contain a single element"
            )

        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)
        vhp = _fill_in_zeros(grad_res, inputs, strict, create_graph, "double_back")

    outputs = _grad_postprocess(outputs, create_graph)
    vhp = _grad_postprocess(vhp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vhp, is_inputs_tuple
    )


def hvp(func, inputs, v=None, create_graph=False, strict=False):
    r"""Compute the dot product between the scalar function's Hessian and a vector ``v`` at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Hessian vector
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            hvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.
    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            hvp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> hvp(pow_reducer, inputs, v)
        (tensor(0.1448),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]]))

        >>> hvp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.1448, grad_fn=<SumBackward0>),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> hvp(pow_adder_reducer, inputs, v)
        (tensor(2.3030),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    Note:

        This function is significantly slower than `vhp` due to backward mode AD constraints.
        If your function is twice continuously differentiable, then hvp = vhp.t(). So if you
        know that your function satisfies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "hvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        else:
            if len(inputs) != 1 or inputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the input to the user-provided function "
                    "is a single Tensor with a single element."
                )
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "hvp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError(
                "The function given to hvp should return a single Tensor"
            )

        if outputs[0].nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hvp should contain a single element"
            )

        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)

        grad_jac = tuple(torch.zeros_like(inp, requires_grad=True) for inp in inputs)

        double_back = _autograd_grad(jac, inputs, grad_jac, create_graph=True)
        _check_requires_grad(jac, "hessian", strict=strict)

    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(double_back, grad_jac, v, create_graph=create_graph)
        hvp = _fill_in_zeros(
            grad_res, inputs, strict, create_graph, "double_back_trick"
        )

    outputs = _grad_postprocess(outputs, create_graph)
    hvp = _grad_postprocess(hvp, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        hvp, is_inputs_tuple
    )
<|endoftext|># mypy: allow-untyped-defs
from typing import Any

import torch
from torch.utils._contextlib import (
    _DecoratorContextManager,
    _NoParamDecoratorContextManager,
    F,
)


__all__ = [
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "inference_mode",
    "set_multithreading_enabled",
    "enforce_grad_layout_policy",
]


class no_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Tensor.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Tensor and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        No-grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.
        If you want to disable forward AD for a computation, you can unpack
        your dual tensors.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> @torch.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> z.requires_grad
        False
        >>> @torch.no_grad()
        ... def tripler(x):
        ...     return x * 3
        >>> z = tripler(x)
        >>> z.requires_grad
        False
        >>> # factory function exception
        >>> with torch.no_grad():
        ...     a = torch.nn.Parameter(torch.rand(10))
        >>> a.requires_grad
        True
    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_grad_enabled(self.prev)


class enable_grad(_NoParamDecoratorContextManager):
    r"""Context-manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        enable_grad is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> with torch.no_grad():
        ...     with torch.enable_grad():
        ...         y = x * 2
        >>> y.requires_grad
        True
        >>> y.backward()
        >>> x.grad
        tensor([2.])
        >>> @torch.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with torch.no_grad():
        ...     z = doubler(x)
        >>> z.requires_grad
        True
        >>> @torch.enable_grad()
        ... def tripler(x):
        ...     return x * 3
        >>> with torch.no_grad():
        ...     z = tripler(x)
        >>> z.requires_grad
        True

    """

    def __enter__(self) -> None:
        self.prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)


class set_grad_enabled(_DecoratorContextManager):
    r"""Context-manager that sets gradient calculation on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    .. note::
        set_grad_enabled is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    Example::
        >>> # xdoctest: +SKIP
        >>> x = torch.tensor([1.], requires_grad=True)
        >>> is_train = False
        >>> with torch.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> y.requires_grad
        False
        >>> _ = torch.set_grad_enabled(True)
        >>> y = x * 2
        >>> y.requires_grad
        True
        >>> _ = torch.set_grad_enabled(False)
        >>> y = x * 2
        >>> y.requires_grad
        False

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch.is_grad_enabled()
        self.mode = mode
        torch._C._set_grad_enabled(mode)

    def __call__(self, orig_func: F) -> F:
        torch._C._set_grad_enabled(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        torch._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_enabled(self.prev)

    def __str__(self) -> str:
        return f"{torch.typename(self)}(mode={self.mode})"

    def __repr__(self) -> str:
        return str(self)

    def clone(self) -> "set_grad_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class inference_mode(_DecoratorContextManager):
    r"""Context manager that enables or disables inference mode.

    InferenceMode is analogous to :class:`~no_grad` and should be used
    when you are certain your operations will not interact with autograd
    (e.g., during data loading or model evaluation). Compared to
    :class:`~no_grad`, it removes additional overhead by disabling view
    tracking and version counter bumps. It is also more restrictive, in
    that tensors created in this mode cannot be used in computations
    recorded by autograd.

    This context manager is thread-local; it does not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        Inference mode is one of several mechanisms that can locally enable
        or disable gradients. See :ref:`locally-disable-grad-doc` for a
        comparison. If avoiding the use of tensors created in inference mode
        in autograd-tracked regions is difficult, consider benchmarking your
        code with and without inference mode to weigh the performance benefits
        against the trade-offs. You can always use :class:`~no_grad` instead.

    .. note::
       Unlike some other mechanisms that locally enable or disable grad,
       entering inference_mode also disables :ref:`forward-mode AD <forward-mode-ad>`.

    .. warning::
        `inference_mode` does NOT automatically set the model to evaluation mode.
        For proper inference behavior (e.g., disabling dropout, using running statistics
        in batch normalization), you must explicitly set your model to evaluation mode using
        `model.eval()` in addition to using this context manager.

    Args:
        mode (bool or function): Either a boolean flag to enable or disable
            inference mode, or a Python function to decorate with inference
            mode enabled.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> import torch
        >>> x = torch.ones(1, 2, 3, requires_grad=True)
        >>> with torch.inference_mode():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> # xdoctest: +SKIP("want string isn't quite right")
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @torch.inference_mode()
        ... def func(x):
        ...     return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False
        >>> @torch.inference_mode()
        ... def doubler(x):
        ...     return x * 2
        >>> out = doubler(x)
        >>> out.requires_grad
        False

    """

    def __init__(self, mode: bool = True) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.mode = mode

    def __new__(cls, mode=True):
        if isinstance(mode, bool):
            return super().__new__(cls)
        return cls()(mode)

    def __enter__(self) -> None:
        self._inference_mode_context = torch._C._InferenceMode(self.mode)
        self._inference_mode_context.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._inference_mode_context.__exit__(exc_type, exc_value, traceback)

    def clone(self) -> "inference_mode":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


def _enter_inference_mode(mode):
    mode_context = torch._C._InferenceMode(mode)
    mode_context.__enter__()
    return mode_context


def _exit_inference_mode(mode):
    mode.__exit__(None, None, None)


class set_multithreading_enabled(_DecoratorContextManager):
    r"""Context-manager that enables or disables multithreaded backward.

    Ordinarily, when :ref:`accelerator<accelerators>` devices are in use,
    the backward pass runs on device-specific worker threads. The engine
    creates these threads based on the number of available devices and
    reuses them across iterations.

    When ``mode=False``, the backward pass runs on the calling thread
    instead. ``mode=True`` restores the default behavior.

    This can be used as a context-manager or as a function. It is
    thread-local and will not affect computation in other threads.

    Args:
        mode (bool): Whether to enable multithreaded backward (``True``,
                    default) or disable (``False``).

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`,
        which never uses multithreading.

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._is_multithreading_enabled()
        torch._C._set_multithreading_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_multithreading_enabled(self.prev)

    def clone(self) -> "set_multithreading_enabled":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class enforce_grad_layout_policy(_DecoratorContextManager):
    r"""Context-manager that controls the gradient layout policy enforcement.

    The gradient layout contract ensures that accumulated gradients have
    strides matching their corresponding parameters (for non-overlapping
    dense parameters) or are row-major contiguous (otherwise). This aids
    performance in optimizers and distributed reducers.

    When ``enable=False``, the autograd engine relaxes this enforcement:

    - Stealable gradients whose layout does *not* match the parameter can
      still be stolen directly (avoiding an extra copy).
    - The "grad and param do not obey the gradient layout contract" warning
      is suppressed.

    The logic that *creates* a brand-new gradient (e.g., cloning into the
    correct layout when the gradient is not stealable) is **not** affected
    by this flag.

    This can be used as a context-manager or as a function. It is
    thread-local and will not affect computation in other threads.

    Args:
        enable (bool): Whether to enforce the gradient layout contract
            (``True``, default) or relax enforcement (``False``).

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch
        >>> p = torch.empty(2, 3, 4).permute(2, 0, 1).requires_grad_()
        >>> with torch.autograd.enforce_grad_layout_policy(False):
        ...     (p * 2).sum().backward()
        ...     # p.grad may now have the same strides as the incoming
        ...     # gradient rather than being forced to match p's strides.
    """

    def __init__(self, enable: bool = True) -> None:
        self.prev = torch._C._is_grad_layout_enforcement_enabled()
        torch._C._set_grad_layout_enforcement_enabled(enable)
        self.mode = enable

    def __call__(self, orig_func: F) -> F:
        torch._C._set_grad_layout_enforcement_enabled(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_grad_layout_enforcement_enabled(self.prev)

    def clone(self) -> "enforce_grad_layout_policy":
        r"""
        Create a copy of this class
        """
        return self.__class__(self.mode)


class _force_original_view_tracking(_DecoratorContextManager):
    r"""Context-manager that sets whether or not to always enable view-replay in autograd.

    ``set_view_replay_enabled`` will enable or disable view-replay based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    When a tensor view is mutated, the autograd engine needs to decide whether or not
    to regenerate the "updated view" by either replaying the chain of views from the updated base,
    or with a single call to as_strided.

    If set_view_replay_enabled is set to True, then autograd will always use view replay.
    Otherwise, it will fall back to its existing logic.

    Args:
        mode (bool): Flag whether to enable view-replay (``True``), or disable
                     (``False``).

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._is_view_replay_enabled()
        self.mode = mode
        torch._C._set_view_replay_enabled(mode)

    def __call__(self, orig_func: F) -> F:
        torch._C._set_view_replay_enabled(self.prev)
        return super().__call__(orig_func)

    def __enter__(self) -> None:
        torch._C._set_view_replay_enabled(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_view_replay_enabled(self.prev)

    def clone(self):
        return self.__class__(self.mode)


class _unsafe_preserve_version_counter(_DecoratorContextManager):
    r"""DO NOT USE THIS UNLESS YOU KNOW EXACTLY WHAT YOU'RE DOING.

    This context manager can lead to arbitrary silent-correctness issues in any other part of your code
    (even the ones not touched directly by the context manager)!

    Ordinarily, autograd will track mutations to tensors by incrementing its `._version` attribute.
    This is generally important for correctness, as for example, mutating a tensor that autograd has saved
    for the backwards pass can result in incorrect gradients, and autograd uses the version counter to detect
    and error out in this situation.

    However, there are rare instances where it might be useful to hide mutations from autograd. For example:
    if a tensor is very large, and you'd like to free its memory by storing it elsewhere, and re-populate
    the tensor right before it is needed by autograd.

    Args:
        tensor (torch.Tensor): the tensor in question, that you would like to preserve the version counter of.

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, tensors: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        self.tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tensors
        if not isinstance(self.tensors, tuple):
            raise AssertionError("Expected tensors to be a tuple")
        self.prev_versions = tuple(t._version for t in self.tensors)

    def __enter__(self) -> None:
        pass

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args) -> None:
        torch._C._autograd._unsafe_set_version_counter(self.tensors, self.prev_versions)
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch._C import _ImperativeEngine as ImperativeEngine


__all__ = ["VariableMeta", "Variable"]


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    _execution_engine = ImperativeEngine()
<|endoftext|># mypy: allow-untyped-defs
import functools
import inspect
import itertools
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Concatenate, TypeVar
from typing_extensions import deprecated, ParamSpec

import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call


__all__ = [
    "FunctionCtx",
    "BackwardCFunction",
    "FunctionMeta",
    "Function",
    "once_differentiable",
    "InplaceFunction",
    "NestedIOFunction",
]

# Unique id provider for each class inheriting from Function
# This is incremented in FunctionMeta during class definition
AUTOGRAD_FUNCTION_COUNTER = itertools.count()

_T = TypeVar("_T")
_R = TypeVar("_R")
_P = ParamSpec("_P")


# Formerly known as: _ContextMethodMixin
class FunctionCtx:
    output_grad_dtypes: "tuple[torch.dtype | None, ...] | None"

    def save_for_backward(self, *tensors: torch.Tensor):
        r"""Save given tensors for a future call to :func:`~Function.backward`.

        ``save_for_backward`` should be called at most once, in either the
        :func:`setup_context` or :func:`forward` methods, and only with tensors.

        All tensors intended to be used in the backward pass should be saved
        with ``save_for_backward`` (as opposed to directly on ``ctx``) to prevent
        incorrect gradients and memory leaks, and enable the application of saved
        tensor hooks. See :class:`torch.autograd.graph.saved_tensors_hooks`.
        See :ref:`extending-autograd` for more details.

        Note that if intermediary tensors, tensors that are neither inputs
        nor outputs of :func:`forward`, are saved for backward, your custom Function
        may not support double backward.
        Custom Functions that do not support double backward should decorate their
        :func:`backward` method with ``@once_differentiable`` so that performing
        double backward raises an error. If you'd like to support double backward,
        you can either recompute intermediaries based on the inputs during backward
        or return the intermediaries as the outputs of the custom Function. See the
        `double backward tutorial <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_
        for more details.

        In :func:`backward`, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``. This is a no-op.

        See :ref:`extending-autograd` for more details on how to use this method.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
            >>>         w = x * z
            >>>         out = x * y + y * z + w * y
            >>>         ctx.save_for_backward(x, y, w, out)
            >>>         ctx.z = z  # z is not a tensor
            >>>         return out
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, grad_out):
            >>>         x, y, w, out = ctx.saved_tensors
            >>>         z = ctx.z
            >>>         gx = grad_out * (y + y * z)
            >>>         gy = grad_out * (x + z + w)
            >>>         gz = None
            >>>         return gx, gy, gz
            >>>
            >>> a = torch.tensor(1., requires_grad=True, dtype=torch.double)
            >>> b = torch.tensor(2., requires_grad=True, dtype=torch.double)
            >>> c = 4
            >>> d = Func.apply(a, b, c)

        """
        self.to_save = tensors

    def save_for_forward(self, *tensors: torch.Tensor):
        r"""Save given tensors for a future call to :func:`~Function.jvp`.

        ``save_for_forward`` should be called at most once, in either the
        :func:`setup_context` or :func:`forward` methods, and all arguments
        should be tensors.

        In :func:`jvp`, saved objects can be accessed through the :attr:`saved_tensors`
        attribute.

        Arguments can also be ``None``. This is a no-op.

        See :ref:`extending-autograd` for more details on how to use this method.

        Example::

            >>> # xdoctest: +SKIP
            >>> class Func(torch.autograd.Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
            >>>         ctx.save_for_backward(x, y)
            >>>         ctx.save_for_forward(x, y)
            >>>         ctx.z = z
            >>>         return x * y * z
            >>>
            >>>     @staticmethod
            >>>     def jvp(ctx, x_t, y_t, _):
            >>>         x, y = ctx.saved_tensors
            >>>         z = ctx.z
            >>>         return z * (y * x_t + x * y_t)
            >>>
            >>>     @staticmethod
            >>>     def vjp(ctx, grad_out):
            >>>         x, y = ctx.saved_tensors
            >>>         z = ctx.z
            >>>         return z * grad_out * y, z * grad_out * x, None
            >>>
            >>>     a = torch.tensor(1., requires_grad=True, dtype=torch.double)
            >>>     t = torch.tensor(1., dtype=torch.double)
            >>>     b = torch.tensor(2., requires_grad=True, dtype=torch.double)
            >>>     c = 4
            >>>
            >>>     with fwAD.dual_level():
            >>>         a_dual = fwAD.make_dual(a, t)
            >>>         d = Func.apply(a_dual, b, c)

        """
        for tensor in tensors:
            if not (isinstance(tensor, torch.Tensor) or tensor is None):
                raise AssertionError(
                    "save_for_forward expects all arguments to be tensors; you should "
                    "save non-tensors as attributes on ctx."
                )

        self.saved_for_forward = tensors

    def mark_dirty(self, *args: torch.Tensor):
        r"""Mark given tensors as modified in an in-place operation.

        This should be called at most once, in either the :func:`setup_context`
        or :func:`forward` methods, and all arguments should be inputs.

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class Inplace(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         x_npy = x.numpy() # x_npy shares storage with x
            >>>         x_npy += 1
            >>>         ctx.mark_dirty(x)
            >>>         return x
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, grad_output):
            >>>         return grad_output
            >>>
            >>> a = torch.tensor(1., requires_grad=True, dtype=torch.double).clone()
            >>> b = a * a
            >>> Inplace.apply(a)  # This would lead to wrong gradients!
            >>>                   # but the engine would not know unless we mark_dirty
            >>> # xdoctest: +SKIP
            >>> b.backward() # RuntimeError: one of the variables needed for gradient
            >>>              # computation has been modified by an inplace operation

        """
        self.dirty_tensors = args

    @deprecated(
        "`mark_shared_storage` is deprecated. "
        "Tensors with shared storages are automatically tracked. "
        "Note that calls to `set_()` are not tracked",
        category=FutureWarning,
    )
    def mark_shared_storage(self, *pairs):
        pass

    def mark_non_differentiable(self, *args: torch.Tensor):
        r"""Mark outputs as non-differentiable.

        This should be called at most once, in either the :func:`setup_context`
        or :func:`forward` methods, and all arguments should be tensor outputs.

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.

        This is used e.g. for indices returned from a sort. See example::
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         sorted, idx = x.sort()
            >>>         ctx.mark_non_differentiable(idx)
            >>>         ctx.save_for_backward(x, idx)
            >>>         return sorted, idx
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, g1, g2):  # still need to accept g2
            >>>         x, idx = ctx.saved_tensors
            >>>         grad_input = torch.zeros_like(x)
            >>>         grad_input.index_add_(0, idx, g1)
            >>>         return grad_input

        """
        self.non_differentiable = args

    def set_output_grad_dtype(self, *dtypes: "torch.dtype | None") -> None:
        r"""Declare the gradient dtype for each of this Function's outputs.

        This should be called at most once, from either the
        :func:`setup_context` or :func:`forward` methods. The number of
        declarations must match the number of returned values, and each argument
        corresponds positionally to the output at the same index.

        For each output, pass the dtype your backward should receive its
        gradient in:

        - Pass a :class:`torch.dtype` and the engine guarantees the gradient
          handed to backward has that dtype. This is only valid for a
          differentiable Tensor output.
        - Pass ``None`` and the gradient is handed to backward with whatever
          dtype it already has. This is also the only valid choice for a
          non-Tensor or non-differentiable output, which has no gradient.
        - Omit this call (or pass the output's own dtype) and the gradient is
          handed to backward in the output's dtype, which is the default.

        For example::

            >>> # xdoctest: +SKIP
            >>> @staticmethod
            >>> def forward(ctx, x):
            >>>     t1 = x.sin()
            >>>     t2 = x.cos()
            >>>     t3 = x.tan()
            >>>     ctx.set_output_grad_dtype(torch.float32, t2.dtype, None, None)
            >>>     return t1, t2, t3, "not a tensor"

        This ensures that backward receives ``t1``'s gradient in ``float32``,
        keeps the default behavior for ``t2``'s gradient via ``t2.dtype``,
        passes ``t3``'s gradient through uncast with ``None``, and uses ``None``
        as the placeholder for the trailing non-Tensor output.
        """
        if self.output_grad_dtypes is not None:
            raise RuntimeError("set_output_grad_dtype can only be called once")
        self.output_grad_dtypes = dtypes

    def set_materialize_grads(self, value: bool):
        r"""Set whether to materialize grad tensors. Default is ``True``.

        This should be called only from either the :func:`setup_context` or
        :func:`forward` methods.

        If ``True``, undefined grad tensors will be expanded to tensors full of zeros
        prior to calling the :func:`backward` and :func:`jvp` methods.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class SimpleFunc(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         return x.clone(), x.clone()
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, g1, g2):
            >>>         return g1 + g2  # No check for None necessary
            >>>
            >>> # We modify SimpleFunc to handle non-materialized grad outputs
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         ctx.set_materialize_grads(False)
            >>>         ctx.save_for_backward(x)
            >>>         return x.clone(), x.clone()
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, g1, g2):
            >>>         x, = ctx.saved_tensors
            >>>         grad_input = torch.zeros_like(x)
            >>>         if g1 is not None:  # We must check for None now
            >>>             grad_input += g1
            >>>         if g2 is not None:
            >>>             grad_input += g2
            >>>         return grad_input
            >>>
            >>> a = torch.tensor(1., requires_grad=True)
            >>> b, _ = Func.apply(a)  # induces g2 to be undefined

        """
        self.materialize_grads = value


# DO NOT USE: This is only defined to be able to load old serialized models
_ContextMethodMixin = FunctionCtx


class _HookMixin:
    @staticmethod
    def _register_hook(backward_hooks, hook):
        if backward_hooks is None:
            backward_hooks = OrderedDict()
        handle = hooks.RemovableHandle(backward_hooks)
        backward_hooks[handle.id] = hook
        return backward_hooks, handle


class BackwardCFunction(_C._FunctionBase, FunctionCtx, _HookMixin):
    r"""
    This class is used for internal autograd work. Do not use.
    """

    def _get_user_fn(self):
        backward_fn = self._forward_cls.backward  # type: ignore[attr-defined]
        vjp_fn = self._forward_cls.vjp  # type: ignore[attr-defined]
        if backward_fn is not Function.backward and vjp_fn is not Function.vjp:
            raise RuntimeError(
                "Implementing both 'backward' and 'vjp' for a custom "
                "Function is not allowed. You should only implement one "
                "of them."
            )
        return vjp_fn if vjp_fn is not Function.vjp else backward_fn

    def apply(self, *args):
        r"""
        Apply method used when executing this Node during the backward.

        Called by the autograd engine (non-boxed path) and by direct
        grad_fn.apply() calls. When boxed_grads_call is True, boxes
        grads into a mutable list before calling user's backward.
        """
        user_fn = self._get_user_fn()
        fwd_cls = self._forward_cls  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]
        if getattr(fwd_cls, "boxed_grads_call", False):
            args = (list(args),)
        return user_fn(self, *args)

    def apply_boxed(self, *args):
        r"""
        Apply method called by the autograd engine when boxed_grads_call
        is True. Grads arrive as a single mutable list argument, allowing
        backward to free individual grads mid-execution.
        """
        return self._get_user_fn()(self, *args)

    def apply_jvp(self, *args):
        r"""
        Apply method used when executing forward mode AD during the forward
        """
        # _forward_cls is defined by derived class
        return self._forward_cls.jvp(self, *args)  # type: ignore[attr-defined]

    def _compiled_autograd_key(self):
        return self._forward_cls._compiled_autograd_key(self)  # type: ignore[attr-defined]


class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """

    def __init__(cls, name, bases, attrs):
        backward_fn = type(
            name + "Backward", (BackwardCFunction,), {"_forward_cls": cls}
        )
        backward_fn._autograd_function_id = next(AUTOGRAD_FUNCTION_COUNTER)  # type: ignore[attr-defined]
        cls._backward_cls = backward_fn

        super().__init__(name, bases, attrs)


class _SingleLevelFunction(
    _C._FunctionBase, FunctionCtx, _HookMixin, metaclass=FunctionMeta
):
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        r"""Define the forward of the custom autograd Function.

        This function is to be overridden by all subclasses.
        There are two ways to define forward:

        Usage 1 (Combined forward and ctx)::

            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass

        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details

        Usage 2 (Separate forward and ctx)::

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass


            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass

        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details

        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.
        """
        raise NotImplementedError(
            "You must implement the forward function for custom autograd.Function."
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        r"""There are two ways to define the forward pass of an autograd.Function.

        Either:

        1. Override forward with the signature ``forward(ctx, *args, **kwargs)``.
           ``setup_context`` is not overridden. Setting up the ctx for backward
           happens inside the ``forward``.
        2. Override forward with the signature ``forward(*args, **kwargs)`` and
           override ``setup_context``. Setting up the ctx for backward happens
           inside ``setup_context`` (as opposed to inside the ``forward``)

        See :meth:`torch.autograd.Function.forward` and :ref:`extending-autograd` for more details.
        """
        raise NotImplementedError("setup_context is not implemented.")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        r"""Define a formula for differentiating the operation with backward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.
        """
        raise NotImplementedError(
            "You must implement either the backward or vjp method for "
            "your custom autograd.Function to use it with backward "
            "mode AD."
        )

    # vjp and backward are alias of each other
    vjp = backward

    """
    Bool that specifies if PyTorch should clear saved tensors after the first
    access to ``ctx.saved_tensors``. When set to True, accessing saved_tensors
    clears the internal references, allowing the tensors to be cleared as soon
    as the Tensor returned by saved_tensors is deleted.

    This is useful for reducing memory pressure in backward passes when you
    only need to access saved tensors once.

    Default is False.
    """
    clear_saved_tensors_on_access = False

    """
    Bool that specifies if backward should receive grads as a single mutable
    list argument instead of individual args in an immutable tuple. This allows
    backward to free individual grads mid-execution by removing them from the
    list, reducing peak memory.

    When True, ``backward(ctx, grads)`` receives a single list instead of
    ``backward(ctx, *grads)``.

    Default is False.
    """
    boxed_grads_call = False

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        r"""Define a formula for differentiating the operation with forward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many inputs as the :func:`forward` got (None will be passed in
        for non tensor inputs of the forward function),
        and it should return as many tensors as there were outputs to
        :func:`forward`. Each argument is the gradient w.r.t the given input,
        and each returned value should be the gradient w.r.t. the
        corresponding output. If an output is not a Tensor or the function is not
        differentiable with respect to that output, you can just pass None as a
        gradient for that input.

        You can use the :attr:`ctx` object to pass any value from the forward to this
        functions.
        """
        raise NotImplementedError(
            "You must implement the jvp function for custom "
            "autograd.Function to use it with forward mode AD."
        )


class Function(_SingleLevelFunction):
    r"""Base class to create custom `autograd.Function`.

    To create a custom `autograd.Function`, subclass this class and implement
    the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
    op in the forward pass, call the class method ``apply``. Do not call
    :meth:`forward` directly.

    To ensure correctness and best performance, make sure you are calling the
    correct methods on ``ctx`` and validating your backward function using
    :func:`torch.autograd.gradcheck`.

    See :ref:`extending-autograd` for more details on how to use this class.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> class Exp(Function):
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> # Use it by calling the apply method:
        >>> # xdoctest: +SKIP
        >>> output = Exp.apply(input)
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            f"{self.__class__} should not be instantiated. Methods on autograd functions "
            "are all static, so you should invoke them on the class itself. "
            "Instantiating an autograd function will raise an "
            "error in a future version of PyTorch.",
            DeprecationWarning,
            stacklevel=2,
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Legacy autograd function with non-static forward method is deprecated. "
            "Please use new-style autograd function with static forward method. "
            "(Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)"
        )

    """
    Bool that specifies if PyTorch should attempt to autogenerate
    :func:`torch.vmap` support for this autograd.Function. You may set this to
    True only if this autograd.Function's forward, backward, and jvp (if they
    exist) are written using PyTorch operations; otherwise, please override
    :meth:`torch.autograd.Function.vmap` to add support for :func:`torch.vmap`.

    Please see :ref:`func-autograd-function` for more details.
    """
    generate_vmap_rule = False

    @staticmethod
    def vmap(info, in_dims, *args):
        r"""Define the behavior for this autograd.Function underneath :func:`torch.vmap`.

        For a :func:`torch.autograd.Function` to support
        :func:`torch.vmap`, you must either override this static method, or set
        ``generate_vmap_rule`` to ``True`` (you may not do both).

        If you choose to override this staticmethod: it must accept

        - an ``info`` object as the first argument. ``info.batch_size``
          specifies the size of the dimension being vmapped over,
          while ``info.randomness`` is the randomness option passed to
          :func:`torch.vmap`.
        - an ``in_dims`` tuple as the second argument.
          For each arg in ``args``, ``in_dims`` has a corresponding
          ``Optional[int]``. It is ``None`` if the arg is not a Tensor or if
          the arg is not being vmapped over, otherwise, it is an integer
          specifying what dimension of the Tensor is being vmapped over.
        - ``*args``, which is the same as the args to :meth:`~Function.forward`.

        The return of the vmap staticmethod is a tuple of ``(output, out_dims)``.
        Similar to ``in_dims``, ``out_dims`` should be of the same structure as
        ``output`` and contain one ``out_dim`` per output that specifies if the
        output has the vmapped dimension and what index it is in.

        Please see :ref:`func-autograd-function` for more details.
        """
        raise NotImplementedError(
            "To use autograd.Function with vmap, you must either override the "
            "vmap staticmethod or set generate_vmap_rule=True."
        )

    apply = _C._FunctionBase.__dict__["apply"]

    @staticmethod
    def _compiled_autograd_key(ctx):
        return (ctx._autograd_function_id,)


def _bind_default_args(
    func: Callable[_P, Any],
    args: _P.args,  # type: ignore[valid-type]
    kwargs: _P.kwargs,  # type: ignore[valid-type]
) -> tuple[_P.args, _P.kwargs]:  # type: ignore[valid-type]
    signature = inspect.signature(func)
    bound_args = signature.bind(*args, **({} if kwargs is None else kwargs))
    bound_args.apply_defaults()
    return bound_args.args, bound_args.kwargs


def _call_custom_function_call(cls, args, kwargs):
    if kwargs is None:
        return custom_function_call(cls, *args)
    return custom_function_call(cls, *args, **kwargs)


def _is_setup_context_defined(fn):
    return fn != _SingleLevelFunction.setup_context


def once_differentiable(
    fn: Callable[Concatenate[_T, _P], _R],
) -> Callable[Concatenate[_T, _P], _R]:
    @functools.wraps(fn)
    def wrapper(ctx: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with torch.no_grad():
            outputs = fn(ctx, *args, **kwargs)

        if not torch.is_grad_enabled():
            return outputs

        # If any of the inputs have requires_grad=True, we force the outputs
        # to have requires_grad=True but point to a grad_fn which throws an
        # error message during (double) back-propagation.
        # XXX: this is only an approximation of requires_grad - there's no way
        # to figure out if fn didn't use ctx.saved_tensors and as a result
        # some Tensors might require grad, even if no args do.
        # Unfortunately, this leads to unexpected error messages ("no nodes
        # require computing gradients"), but I don't have a better idea.
        # These functions would raise an error in backward anyway.
        requires_grad = any(
            isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args
        )
        if not requires_grad:
            return outputs

        if not isinstance(outputs, tuple):
            outputs_ = (outputs,)
        else:
            outputs_ = outputs

        err_fn = _functions.DelayedError(
            b"trying to differentiate twice a function that was marked "
            b"with @once_differentiable",
            len(outputs_),
        )

        # Create aliases of each output that has requires_grad=True. We need
        # at least one of the inputs to err_fn to require grad so that the
        # output will have a grad_fn.
        def fake_requires_grad(var):
            if var is not None:
                var = var.detach()
                var.requires_grad = True
            return var

        return err_fn(*[fake_requires_grad(v) for v in outputs_])  # type: ignore[return-value]

    return wrapper


class InplaceFunction(Function):
    r"""
    This class is here only for backward compatibility reasons.
    Use :class:`Function` instead of this for any new use case.
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace


def _nested_map(condition, fn, condition_msg=None):
    def _map(obj):
        if condition(obj):
            return fn(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            mapped = (_map(x) for x in obj)
            if hasattr(obj, "_fields"):
                # obj is namedtuple
                return type(obj)(*mapped)
            return type(obj)(mapped)
        elif isinstance(obj, dict):
            return {x: _map(obj[x]) for x in obj}
        else:
            raise ValueError(
                "Auto nesting doesn't know how to process "
                "an input object of type "
                + torch.typename(obj)
                + (
                    ". Accepted types: " + condition_msg + ", or lists/tuples of them"
                    if condition_msg
                    else ""
                )
            )

    return _map


def _jit_unwrap_structured(obj):
    if hasattr(obj, "_jit_unwrap"):
        return obj._jit_unwrap()
    return obj


def _iter_filter(condition, allow_unknown=False, condition_msg=None, conversion=None):
    def _iter(obj):
        if conversion is not None:
            obj = conversion(obj)
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                yield from _iter(o)
        elif isinstance(obj, dict):
            # We only accept primitive key types, so we needn't inspect them
            for o in obj.values():
                yield from _iter(o)
        elif allow_unknown:
            yield obj
        else:
            raise ValueError(
                "Auto nesting doesn't know how to process "
                "an input object of type "
                + torch.typename(obj)
                + (
                    ". Accepted types: " + condition_msg + ", or lists/tuples of them"
                    if condition_msg
                    else ""
                )
            )

    return _iter


def _unflatten(input, proto):
    # unflatten a list or tuple input into a nested list/tuple structure
    # specified by proto
    def unflatten_helper(input, proto):
        res: list[torch.Tensor | None] = []
        if hasattr(proto, "_jit_wrap"):
            return proto._jit_wrap(input)
        if not isinstance(proto, (list, tuple)):
            return input[0], input[1:]
        for e in proto:
            if e is None:
                res.append(e)
            else:
                res_e, input = unflatten_helper(input, e)
                res.append(res_e)
        return type(proto)(res), input

    return unflatten_helper(input, proto)[0]


_iter_jit_values = _iter_filter(
    lambda o: o is None or isinstance(o, torch._C.Value),
    condition_msg="jit's Values or None",
)
_iter_tensors = _iter_filter(
    lambda x: isinstance(x, torch.Tensor),
    condition_msg="Tensors",
    conversion=_jit_unwrap_structured,
)
_iter_tensors_permissive = _iter_filter(
    lambda x: isinstance(x, torch.Tensor),
    allow_unknown=True,
    condition_msg="Tensors (permissive)",
)
_iter_None_tensors = _iter_filter(
    lambda o: o is None or isinstance(o, torch.Tensor), condition_msg="Tensors or None"
)
_map_tensor_data = _nested_map(
    lambda x: isinstance(x, torch.Tensor), lambda o: o.data, condition_msg="Tensors"
)


class NestedIOFunction(Function):
    r"""
    This class is here only for backward compatibility reasons.
    Use :class:`Function` instead of this for any new use case.
    """

    # The 'type: ignore' statements are needed here because these functions are declared as '@staticmethod' in the
    # superclass (Function) but are instance methods here, which mypy reports as incompatible.

    def _do_forward(self, *input):
        self._nested_input = input
        flat_input = tuple(_iter_tensors(input))
        flat_output = super()._do_forward(*flat_input)  # type: ignore[misc]
        nested_tensors = _unflatten(flat_output, self._nested_output)
        return nested_tensors

    def _do_backward(self, gradients, retain_variables):
        self.retain_variables = retain_variables
        result = super()._do_backward(gradients, retain_variables)  # type: ignore[misc]
        if not retain_variables:
            del self._nested_output
            del self._to_save_nested
        return result

    def backward(self, *gradients: Any) -> Any:  # type: ignore[override]
        r"""
        Shared backward utility.
        """
        nested_gradients = _unflatten(gradients, self._nested_output)
        result = self.backward_extended(*nested_gradients)  # type: ignore[func-returns-value]
        return tuple(_iter_None_tensors(result))

    __call__ = _do_forward

    def forward(self, *args: Any) -> Any:  # type: ignore[override]
        r"""
        Shared forward utility.
        """
        nested_tensors = _map_tensor_data(self._nested_input)
        result = self.forward_extended(*nested_tensors)  # type: ignore[func-returns-value]
        del self._nested_input
        self._nested_output = result
        return tuple(_iter_tensors(result))

    def save_for_backward(self, *args: Any) -> None:
        r"""
        See :meth:`Function.save_for_backward`.
        """
        self.to_save = tuple(_iter_tensors(args))
        self._to_save_nested = args

    @property
    def saved_tensors(self):  # type: ignore[override]
        r"""
        See :meth:`Function.saved_tensors`.
        """
        flat_tensors = super().saved_tensors  # type: ignore[misc]
        return _unflatten(flat_tensors, self._to_save_nested)

    def mark_dirty(self, *args: Any, **kwargs: Any) -> None:
        r"""
        See :meth:`Function.mark_dirty`.
        """
        self.dirty_tensors = tuple(_iter_tensors((args, kwargs)))

    def mark_non_differentiable(self, *args: Any, **kwargs: Any) -> None:
        r"""
        See :meth:`Function.mark_non_differentiable`.
        """
        self.non_differentiable = tuple(_iter_tensors((args, kwargs)))

    def forward_extended(self, *input: Any) -> None:
        r"""
        User defined forward.
        """
        raise NotImplementedError

    def backward_extended(self, *grad_output: Any) -> None:
        r"""
        User defined backward.
        """
        raise NotImplementedError
<|endoftext|># mypy: allow-untyped-defs
import os
from typing import Any, NamedTuple

import torch

from .grad_mode import _DecoratorContextManager


__all__ = [
    "UnpackedDualTensor",
    "enter_dual_level",
    "exit_dual_level",
    "make_dual",
    "unpack_dual",
    "dual_level",
]

# Global variable used to make the python API simpler to use
_current_level = -1


def enter_dual_level():
    r"""Enter a new forward grad level.

    This level can be used to make and unpack dual Tensors to compute
    forward gradients.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    from torch._functorch.predispatch import (
        _enter_dual_level as _predispatch_enter_dual_level,
    )

    global _current_level
    new_level = _predispatch_enter_dual_level()
    if new_level != _current_level + 1:
        raise RuntimeError(
            "Entering a new forward AD level but the current level "
            "is not valid. Make sure you did not modify it directly."
        )
    _current_level = new_level
    return new_level


def exit_dual_level(*, level=None):
    r"""Exit a forward grad level.

    This function deletes all the gradients associated with this
    level. Only deleting the latest entered level is allowed.

    This function also updates the current level that is used by default
    by the other functions in this API.
    """
    global _current_level
    if level is None:
        level = _current_level
    if level != _current_level:
        raise RuntimeError(
            "Trying to exit a forward AD level that was not the last one "
            "that was created. This is not supported."
        )
    from torch._functorch.predispatch import (
        _exit_dual_level as _predispatch_exit_dual_level,
    )

    _predispatch_exit_dual_level(level=level)
    _current_level = level - 1


def _maybe_load_decompositions():
    if os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__:
        from torch._decomp import decompositions_for_jvp  # noqa: F401


def make_dual(tensor, tangent, *, level=None):
    r"""Associate a tensor value with its tangent to create a "dual tensor" for forward AD gradient computation.

    The result is a new tensor aliased to :attr:`tensor` with :attr:`tangent` embedded
    as an attribute as-is if it has the same storage layout or copied otherwise.
    The tangent attribute can be recovered with :func:`unpack_dual`.

    This function is backward differentiable.

    Given a function `f` whose jacobian is `J`, it allows one to compute the Jacobian-vector product (`jvp`)
    between `J` and a given vector `v` as follows.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, v)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.

    """
    # See NOTE: [forward-mode AD decompositions mechanism]
    #
    # Import from torch._decomp import decompositions_for_jvp to register
    # decompositions for jvp to the jit registry
    #
    # FIXME: We specify that __debug__ must be True because
    # if python is run with -OO or -O flags (i.e., __debug__ is False), we encounter the
    # following error:
    #
    # Return value was annotated as having type Tuple[NoneType, NoneType] but is actually of
    # type Tuple[Tensor, Tensor]:
    #   File ".../torch/_decomp/__init__.py", line 1585
    #     else:
    #         buffer = z
    #     return min - torch.log1p(z), buffer
    #     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
    _maybe_load_decompositions()

    if level is None:
        level = _current_level

    if level < 0:
        raise RuntimeError(
            "Trying to create a dual Tensor for forward AD but no level "
            "exists, make sure to enter_dual_level() first."
        )
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(
            f"Expected primal to be floating point or complex, but got: {tensor.dtype}"
        )
    if not (tangent.is_floating_point() or tangent.is_complex()):
        raise ValueError(
            f"Expected tangent to be floating point or complex, but got: {tangent.dtype}"
        )

    from torch._functorch.predispatch import _make_dual as _predispatch_make_dual

    return _predispatch_make_dual(tensor, tangent, level=level)


class UnpackedDualTensor(NamedTuple):
    r"""Namedtuple returned by :func:`unpack_dual` containing the primal and tangent components of the dual tensor.

    See :func:`unpack_dual` for more details.
    """

    primal: torch.Tensor
    tangent: torch.Tensor | None


def unpack_dual(tensor, *, level=None):
    r"""Unpack a "dual tensor" to get both its Tensor value and its forward AD gradient.

    The result is a namedtuple ``(primal, tangent)`` where ``primal`` is a view of
    :attr:`tensor`'s primal and ``tangent`` is :attr:`tensor`'s tangent as-is.
    Neither of these tensors can be dual tensor of level :attr:`level`.

    This function is backward differentiable.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     out = f(inp)
        ...     y, jvp = unpack_dual(out)
        ...     jvp = unpack_dual(out).tangent

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """
    if level is None:
        level = _current_level

    if level < 0:
        return UnpackedDualTensor(tensor, None)

    from torch._functorch.predispatch import _unpack_dual as _predispatch_unpack_dual

    primal, dual = _predispatch_unpack_dual(tensor, level=level)

    return UnpackedDualTensor(primal, dual)


class dual_level(_DecoratorContextManager):
    r"""Context-manager for forward AD, where all forward AD computation must occur within the ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exits the dual level to
        control the current forward AD level, which is used by default by the other
        functions in this API.

        We currently don't plan to support nested ``dual_level`` contexts, however, so
        only a single forward AD level is supported. To compute higher-order
        forward grads, one can use :func:`torch.func.jvp`.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     # Do computations with inp
        ...     out = your_fn(inp)
        ...     _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """

    def __enter__(self):
        return enter_dual_level()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        exit_dual_level()


# Private helper functions
_is_fwd_grad_enabled = torch._C._is_fwd_grad_enabled


# Private helper function to enable or disable fwd grad.
# If you're a user and want to use this, please file an issue to discuss the use case.
class _set_fwd_grad_enabled(_DecoratorContextManager):
    def __init__(self, mode: bool) -> None:
        self.prev = _is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_fwd_grad_enabled(self.prev)
<|endoftext|># mypy: allow-untyped-defs
import collections
import functools
import warnings
from collections.abc import Callable, Iterable
from itertools import product
from typing_extensions import deprecated

import torch
import torch.testing

# pyrefly: ignore [deprecated]
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrOptionalTensors, _TensorOrTensors


# Note: `get_*_jacobian` functions are added here even though we didn't intend to make them public
# since they have been exposed from before we added `__all__`  and we already maintain BC for them
# We should eventually deprecate them and remove them from `__all__`
__all__ = [
    "gradcheck",
    "gradgradcheck",
    "GradcheckError",
    "get_numerical_jacobian",
    "get_analytical_jacobian",
    "get_numerical_jacobian_wrt_specific_input",
]


class GradcheckError(RuntimeError):
    r"""Error raised by :func:`gradcheck` and :func:`gradgradcheck`."""


def _is_sparse_compressed_tensor(obj: torch.Tensor):
    return obj.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


def _is_sparse_any_tensor(obj: torch.Tensor):
    return _is_sparse_compressed_tensor(obj) or obj.layout is torch.sparse_coo


def _is_float_or_complex_tensor(obj):
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())


def _allocate_jacobians_with_inputs(
    input_tensors: tuple, numel_output
) -> tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from inputs. If `numel_output` is not None, for
    # each tensor in `input_tensors`, returns a new zero-filled tensor with height
    # of `t.numel` and width of `numel_output`. Otherwise, for each tensor, returns
    # a 1-d tensor with size `(t.numel,)`. Each new tensor will be strided and have
    # the same dtype and device as those of the corresponding input.
    out: list[torch.Tensor] = [
        t.new_zeros((t.numel(), numel_output), layout=torch.strided)
        for t in input_tensors
        if _is_float_or_complex_tensor(t) and t.requires_grad
    ]
    return tuple(out)


def _allocate_jacobians_with_outputs(
    output_tensors: tuple, numel_input, dtype=None, device=None
) -> tuple[torch.Tensor, ...]:
    # Makes zero-filled tensors from outputs. If `dim` is not None, for each tensor
    # in `output_tensors`, returns a new zero-filled tensor with height of `dim` and
    # width of `t.numel`. Otherwise, for each tensor, returns a 1-d tensor with size
    # (t.numel,).
    options = {"dtype": dtype, "device": device, "layout": torch.strided}
    out: list[torch.Tensor] = [
        t.new_zeros((numel_input, t.numel()), **options)
        for t in output_tensors
        if _is_float_or_complex_tensor(t)
    ]
    return tuple(out)


def _iter_tensors(
    x: torch.Tensor | Iterable[torch.Tensor], only_requiring_grad: bool = False
) -> Iterable[torch.Tensor]:
    if is_tensor_like(x):
        # mypy doesn't narrow type of `x` to torch.Tensor
        if x.requires_grad or not only_requiring_grad:  # type: ignore[union-attr]
            yield x  # type: ignore[misc]
    elif isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        for elem in x:
            yield from _iter_tensors(elem, only_requiring_grad)


def _densify(x):
    # return a copy of sparse x with all unspecified elements
    # "replaced" with zero-valued elements
    if isinstance(x, (list, tuple)):
        return type(x)(map(_densify, x))
    elif not is_tensor_like(x) or x.layout in {torch.strided, torch._mkldnn}:  # type: ignore[attr-defined] # no attr _mkldnn
        return x
    elif x.layout is torch.sparse_coo:
        device = x.device
        indices_dtype = x._indices().dtype
        tmp = torch.ones(x.shape[: x.sparse_dim()], dtype=torch.int8, device=device)
        indices = tmp.nonzero().t().to(dtype=indices_dtype)
        values = torch.zeros(
            (tmp.numel(), *x.shape[x.sparse_dim() :]), dtype=x.dtype, device=device
        )
        x_coalesced = x.detach().coalesce()
        if x_coalesced.numel() > 0:
            stride = tmp.stride()
            flat_indices = (
                x_coalesced.indices()
                .mul(
                    torch.tensor(stride, dtype=indices_dtype, device=device).unsqueeze(
                        1
                    )
                )
                .sum(0)
            )
            values[flat_indices] = x_coalesced.values()
        return (
            torch.sparse_coo_tensor(indices, values, x.shape)
            ._coalesced_(True)
            .requires_grad_(x.requires_grad)
        )
    elif _is_sparse_compressed_tensor(x):
        blocksize = (
            x.values().shape[1:3]
            if x.layout in {torch.sparse_bsr, torch.sparse_bsc}
            else None
        )
        compressed_indices = (
            x.crow_indices()
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}
            else x.ccol_indices()
        )
        # We'll use intermediate sparse COO for simplicity
        r = _densify(x.detach().to_sparse(layout=torch.sparse_coo)).to_sparse(
            layout=x.layout, blocksize=blocksize
        )
        # Check that all elements are specified also after `to_sparse` op:
        dense_numel = r.values().numel() // max(1, r.values().shape[0])
        batch_numel = compressed_indices.numel() // compressed_indices.shape[-1]
        sparse_numel = r.numel() // max(1, dense_numel * batch_numel)
        if sparse_numel != r._nnz():
            raise AssertionError(
                f"{x.layout} densify failed: expected nnz={sparse_numel} but got {r._nnz()}"
            )
        return r.requires_grad_(x.requires_grad)
    elif _is_sparse_any_tensor(x):
        raise NotImplementedError(x.layout)
    return x


def _iter_tensor(x_tensor):
    # (Only used for slow gradcheck) Returns a generator that yields the following
    # elements at each iteration:
    #  1) a tensor: the same tensor is returned across all iterations. The tensor
    #     is not the same as the original x_tensor as given as input - it is
    #     prepared so that it can be modified in-place. Depending on whether the
    #     input tensor is strided, sparse, or dense, the returned tensor may or may
    #     not share storage with x_tensor.
    #  2) a tuple of indices that can be used with advanced indexing (yielded in
    #     dictionary order)
    #  3) flattened index that will be used to index into the Jacobian tensor
    #
    # For a tensor t with size (2, 2), _iter_tensor yields:
    #     `x, (0, 0), 0`, `x, (0, 1), 1`, `x, (1, 0), 2`, `x, (1, 1), 3`
    #
    # where x is the t.data of the original tensor. Perturbing the entry of x
    # at index (1, 1) yields the 3rd column of the overall Jacobian matrix.
    if _is_sparse_any_tensor(x_tensor):

        def get_stride(size):
            dim = len(size)
            tmp = 1
            stride = [0] * dim
            for i in reversed(range(dim)):
                stride[i] = tmp
                tmp *= size[i]
            return stride

        x_nnz = x_tensor._nnz()
        x_size = list(x_tensor.size())
        if x_tensor.layout is torch.sparse_coo:
            x_indices = x_tensor._indices().t()
            x_values = x_tensor._values()
        elif x_tensor.layout is torch.sparse_csr:
            x_indices = torch._convert_indices_from_csr_to_coo(
                x_tensor.crow_indices(), x_tensor.col_indices()
            ).t()
            x_values = x_tensor.values()
        elif x_tensor.layout is torch.sparse_csc:
            x_indices = torch._convert_indices_from_csr_to_coo(
                x_tensor.ccol_indices(), x_tensor.row_indices(), transpose=True
            ).t()
            x_values = x_tensor.values()
        elif x_tensor.layout is torch.sparse_bsr:
            x_block_values = x_tensor.values()
            x_blocksize = x_block_values.size()[1:3]
            x_indices = (
                torch._convert_indices_from_csr_to_coo(
                    x_tensor.crow_indices(), x_tensor.col_indices()
                )
                .repeat_interleave(x_blocksize[0] * x_blocksize[1], 1)
                .mul_(torch.tensor(x_blocksize, device=x_tensor.device).reshape(2, 1))
                .add_(
                    torch.stack(
                        torch.where(torch.ones(x_blocksize, device=x_tensor.device))
                    ).repeat(1, x_nnz)
                )
                .t()
            )
            x_values = x_block_values.flatten(0, 2)
            x_nnz = x_values.size(0)
        elif x_tensor.layout is torch.sparse_bsc:
            x_block_values = x_tensor.values()
            x_blocksize = x_block_values.size()[1:3]
            x_indices = (
                torch._convert_indices_from_csr_to_coo(
                    x_tensor.ccol_indices(), x_tensor.row_indices(), transpose=True
                )
                .repeat_interleave(x_blocksize[0] * x_blocksize[1], 1)
                .mul_(torch.tensor(x_blocksize, device=x_tensor.device).reshape(2, 1))
                .add_(
                    torch.stack(
                        torch.where(torch.ones(x_blocksize, device=x_tensor.device))
                    ).repeat(1, x_nnz)
                )
                .t()
            )
            x_values = x_block_values.flatten(0, 2)
            x_nnz = x_values.size(0)
        else:
            raise NotImplementedError(f"_iter_tensor for {x_tensor.layout} input")
        x_stride = get_stride(x_size)
        # Use .data here to get around the version check
        x_values = x_values.data
        for i in range(x_nnz):
            x_value = x_values[i]
            for x_idx in product(*[range(m) for m in x_values.size()[1:]]):
                indices = x_indices[i].tolist() + list(x_idx)
                d_idx = sum(indices[k] * x_stride[k] for k in range(len(x_size)))
                yield x_value, x_idx, d_idx
    elif x_tensor.layout == torch._mkldnn:  # type: ignore[attr-defined]
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            # this is really inefficient, but without indexing implemented, there's
            # not really a better way than converting back and forth
            x_tensor_dense = x_tensor.to_dense()
            yield x_tensor_dense, x_idx, d_idx
    else:
        # Use .data here to get around the version check
        x_tensor = x_tensor.data
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.size()])):
            yield x_tensor, x_idx, d_idx


def _get_numerical_jacobian(
    fn, inputs, outputs=None, target=None, eps=1e-3, is_forward_ad=False
) -> list[tuple[torch.Tensor, ...]]:
    """Compute the numerical Jacobian of `fn(inputs)` with respect to `target`.

    If not specified, targets are the input. Returns M * N Jacobians where N is the
    number of tensors in target that require grad and M is the number of non-integral
    outputs.

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        is_forward_ad: if this numerical jacobian is computed to be checked wrt
                       forward AD gradients (this is used for error checking only)

    Returns:
        A list of M N-tuples of tensors

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    jacobians: list[tuple[torch.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if not is_forward_ad and any(o.is_complex() for o in outputs):
        raise ValueError(
            "Expected output to be non-complex. get_numerical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    if target is None:
        target = inputs
    inp_indices = [
        i for i, a in enumerate(target) if is_tensor_like(a) and a.requires_grad
    ]
    for inp, inp_idx in zip(_iter_tensors(target, True), inp_indices):
        jacobians += [
            get_numerical_jacobian_wrt_specific_input(
                fn,
                inp_idx,
                inputs,
                outputs,
                eps,
                input=inp,
                is_forward_ad=is_forward_ad,
            )
        ]
    return jacobians


@deprecated(
    "`get_numerical_jacobian` was part of PyTorch's private API and not "
    "meant to be exposed. We are deprecating it and it will be removed "
    "in a future version of PyTorch. If you have a specific use for "
    "this or feature request for this to be a stable API, please file "
    "us an issue at https://github.com/pytorch/pytorch/issues/new",
    category=FutureWarning,
)
def get_numerical_jacobian(fn, inputs, target=None, eps=1e-3, grad_out=1.0):
    """Compute the numerical Jacobian for a given fn and its inputs.

    This is a Deprecated API.

    Args:
        fn: the function to compute the Jacobian for (must take inputs as a tuple)
        inputs: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        grad_out: defaults to 1.0.

    Returns:
        A list of Jacobians of `fn` (restricted to its first output) with respect to
        each input or target, if provided.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    if (
        grad_out != 1.0
    ):  # grad_out param is only kept for backward compatibility reasons
        raise ValueError(
            "Expected grad_out to be 1.0. get_numerical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )

    def fn_pack_inps(*inps):
        return fn(inps)

    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)

    return tuple(jacobian_for_each_output[0] for jacobian_for_each_output in jacobians)


def _compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    # Computes numerical directional derivative as finite difference
    # of function `fn` at input `entry`, perturbed by vector `v`.
    if _is_sparse_compressed_tensor(entry):
        # sparse compressed tensors don't implement sub/add/copy_
        # yet. However, in non-masked semantics context entry and v
        # have the same sparse indices ...
        if entry.layout != v.layout:
            raise AssertionError(
                f"Expected entry and v to have the same layout, but got {entry.layout} and {v.layout}"
            )
        if entry._nnz() != v._nnz():
            raise AssertionError(
                f"Expected entry and v to have the same nnz, but got {entry._nnz()} and {v._nnz()} "
                f"with entry shape {entry.shape}"
            )
        # ... the finite differencing can be performed on values only:
        entry = entry.values()
        v = v.values()
        # we'll detach to avoid backward computations that sparse
        # tensors have limited support for.
        entry = entry.detach()

    orig = entry.clone()
    entry.copy_(orig - v)
    outa = fn()
    entry.copy_(orig + v)
    outb = fn()
    entry.copy_(orig)

    def compute(a, b):
        nbhd_checks_fn(a, b)
        ret = (b - a) / (2 * norm_v)  # use central difference approx
        return ret.detach().reshape(-1)

    return tuple(compute(a, b) for (a, b) in zip(outa, outb))


def _compute_numerical_jvps_wrt_specific_input(
    jvp_fn, delta, input_is_complex, is_forward_ad=False
) -> list[torch.Tensor]:
    # Computing the jacobian only works for real delta
    # For details on the algorithm used here, refer:
    # Section 3.5.3 https://arxiv.org/pdf/1701.00392.pdf
    # s = fn(z) where z = x for real valued input
    # and z = x + yj for complex valued input
    jvps: list[torch.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)

    if input_is_complex:  # C -> R
        ds_dy_tup = (
            jvp_fn(delta[1] * 1j) if isinstance(delta, tuple) else jvp_fn(delta * 1j)
        )
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            if ds_dx.is_complex():
                raise AssertionError("Expected ds_dx to be real-valued, not complex")
            # conjugate wirtinger derivative
            conj_w_d = ds_dx + ds_dy * 1j
            jvps.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:  # R -> R or (R -> C for the forward AD case)
            if not is_forward_ad and ds_dx.is_complex():
                raise AssertionError("Expected ds_dx to be real-valued, not complex.")
            jvps.append(ds_dx)
    return jvps


def _combine_jacobian_cols(
    jacobians_cols: dict[int, list[torch.Tensor]], outputs, input, numel
) -> tuple[torch.Tensor, ...]:
    # jacobian_cols maps column_idx -> output_idx -> single column of jacobian Tensor
    # we return a list that maps output_idx -> full jacobian Tensor
    jacobians = _allocate_jacobians_with_outputs(
        outputs, numel, dtype=input.dtype if input.dtype.is_complex else None
    )
    for i, jacobian in enumerate(jacobians):
        for k, v in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians


def _prepare_input(
    input: torch.Tensor, maybe_perturbed_input: torch.Tensor | None, fast_mode=False
) -> torch.Tensor:
    # Prepares the inputs to be passed into the function while including the new
    # modified input.
    if input.layout == torch._mkldnn:  # type: ignore[attr-defined] # no attr _mkldnn
        # Convert back to mkldnn
        if maybe_perturbed_input is not None:
            return maybe_perturbed_input.to_mkldnn()
        else:
            return input
    elif _is_sparse_any_tensor(input):
        if fast_mode and maybe_perturbed_input is not None:
            # entry is already a "cloned" version of the original tensor
            # thus changes to entry are not reflected in the input
            return maybe_perturbed_input
        else:
            return input
    else:
        # We cannot use entry (input.data) if we want gradgrad to work because
        # fn (in the gradgrad case) needs to compute grad wrt input
        return input


def _check_outputs_same_dtype_and_shape(output1, output2, eps, idx=None) -> None:
    # Check that the returned outputs don't have different dtype or shape when you
    # perturb the input
    on_index = f"on index {idx} " if idx is not None else ""
    if output1.shape != output2.shape:
        raise AssertionError(
            f"Expected `func` to return outputs with the same shape"
            f" when inputs are perturbed {on_index}by {eps}, but got:"
            f" shapes {output1.shape} and {output2.shape}."
        )
    if output1.dtype != output2.dtype:
        raise AssertionError(
            f"Expected `func` to return outputs with the same dtype"
            f" when inputs are perturbed {on_index}by {eps}, but got:"
            f" dtypes {output1.dtype} and {output2.dtype}."
        )


def get_numerical_jacobian_wrt_specific_input(
    fn, input_idx, inputs, outputs, eps, input=None, is_forward_ad=False
) -> tuple[torch.Tensor, ...]:
    # Computes the numerical jacobians wrt to a single input. Returns N jacobian
    # tensors, where N is the number of outputs. We use a dictionary for
    # jacobian_cols because indices aren't necessarily consecutive for sparse inputs
    # When we perturb only a single element of the input tensor at a time, the jvp
    # is equivalent to a single col of the Jacobian matrix of fn.
    jacobian_cols: dict[int, list[torch.Tensor]] = {}
    input = inputs[input_idx] if input is None else input
    if not input.requires_grad:
        raise AssertionError("Expected input to have requires_grad=True")
    for x, idx, d_idx in _iter_tensor(input):
        wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, x)
        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(
            _check_outputs_same_dtype_and_shape, idx=idx, eps=eps
        )
        jvp_fn = _get_numerical_jvp_fn(
            wrapped_fn, input_to_perturb, eps, nbhd_checks_fn
        )
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(
            jvp_fn, eps, x.is_complex(), is_forward_ad
        )
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())


def _get_analytical_jacobian_forward_ad(
    fn, inputs, outputs, *, check_grad_dtypes=False, all_u=None
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Compute the analytical Jacobian using forward mode AD of `fn(inputs)` using forward mode AD with respect to `target`.

    Return N * M Jacobians where N is the number of tensors in target that require grad and
    M is the number of non-integral outputs.
    Contrary to other functions here, this function requires "inputs" to actually be used by the function.
    The computed value is expected to be wrong if the function captures the inputs by side effect instead of
    using the passed ones (many torch.nn tests do this).

    Args:
        fn: the function to compute the jacobian for
        inputs: inputs to `fn`
        outputs: provide precomputed outputs to avoid one extra invocation of fn
        check_grad_dtypes: if True, will check that the gradient dtype are valid
        all_u (optional): if provided, the Jacobian will be right multiplied with this vector

    Returns:
        A tuple of M N-tuples of tensors
    """
    # To avoid early import issues
    fwAD = torch.autograd.forward_ad

    tensor_inputs = tuple(i for i in inputs if is_tensor_like(i) and i.requires_grad)

    if any(i.is_complex() for i in tensor_inputs):
        raise ValueError(
            "Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad."
        )

    if all_u:
        jacobians = tuple(
            _allocate_jacobians_with_outputs(outputs, 1) for i in tensor_inputs
        )
    else:
        jacobians = tuple(
            _allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs
        )

    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        for inp in inputs:
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:  # type: ignore[attr-defined]
                    raise ValueError(
                        "MKLDNN inputs are not supported for forward AD gradcheck."
                    )

                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                # If inp is a differentiable view, the dual might not be the tangent given to
                # make_dual, so read it explicitly from the dual tensor
                fw_grads.append(fwAD.unpack_dual(inp)[1])
            dual_inputs.append(inp)

        if all_u:
            # Do the full reduction in one pass
            # To be consistent with numerical evaluation, we actually compute one reduction per input
            for i, (fw_grad, u) in enumerate(zip(fw_grads, all_u)):
                fw_grad.copy_(u.view_as(fw_grad))
                raw_outputs = _as_tuple(fn(*dual_inputs))
                dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                for index_o, d_o in enumerate(dual_outputs):
                    val, res = fwAD.unpack_dual(d_o)
                    if (
                        check_grad_dtypes
                        and res is not None
                        and val.is_complex() != res.is_complex()
                    ):
                        raise GradcheckError("Forward AD gradient has dtype mismatch.")

                    # Remove extra dimension of size 1 corresponding to the reduced input
                    jacobians[i][index_o].squeeze_(0)
                    if res is None:
                        jacobians[i][index_o].zero_()
                    else:
                        jacobians[i][index_o].copy_(res.reshape(-1))
                fw_grad.zero_()
        else:
            # Reconstruct the full Jacobian column by column
            for i, fw_grad in enumerate(fw_grads):
                for lin_idx, grad_idx in enumerate(
                    product(*[range(m) for m in fw_grad.size()])
                ):
                    fw_grad[grad_idx] = 1.0
                    raw_outputs = _as_tuple(fn(*dual_inputs))
                    dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                    for index_o, d_o in enumerate(dual_outputs):
                        val, res = fwAD.unpack_dual(d_o)
                        if (
                            check_grad_dtypes
                            and res is not None
                            and val.is_complex() != res.is_complex()
                        ):
                            raise GradcheckError(
                                "Forward AD gradient has dtype mismatch."
                            )

                        if res is None:
                            jacobians[i][index_o][lin_idx].zero_()
                        else:
                            jacobians[i][index_o][lin_idx].copy_(res.reshape(-1))
                    fw_grad[grad_idx] = 0.0

    return jacobians


def _get_input_to_perturb(input):
    # Prepare the input so that it can be modified in-place and do certain
    # operations that require the tensor to have strides. If fast_mode=False,
    # _iter_tensor would handle the below cases:
    if input.layout == torch._mkldnn:  # type: ignore[attr-defined] # no attr _mkldnn
        # Convert to dense so we can perform operations that require strided tensors
        input_to_perturb = input.to_dense()
    elif _is_sparse_any_tensor(input):
        # Clone because input may require grad, and copy_ calls resize_,
        # which is not allowed for .data
        input_to_perturb = input.clone()
    else:
        input_to_perturb = input.data
    return input_to_perturb


def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    # Wraps `fn` so that its inputs are already supplied
    def wrapped_fn():
        inp = tuple(
            _prepare_input(a, input_to_perturb if i == input_idx else None, fast_mode)
            if is_tensor_like(a)
            else a
            for i, a in enumerate(_as_tuple(inputs))
        )
        return tuple(a.clone() for a in _as_tuple(fn(*inp)))

    return wrapped_fn


def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    # Wraps jvp_fn so that certain arguments are already supplied
    def jvp_fn(delta):
        return _compute_numerical_gradient(
            wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn
        )

    return jvp_fn


def _reshape_tensor_or_tuple(u, shape):
    # We don't need to reshape when input corresponding to u is sparse
    if isinstance(u, tuple):
        if not _is_sparse_any_tensor(u[0]):
            return (u[0].reshape(shape), u[1].reshape(shape))
    else:
        if not _is_sparse_any_tensor(u):
            return u.reshape(shape)
    return u


def _mul_tensor_or_tuple(u, k):
    if isinstance(u, tuple):
        return (k * u[0], k * u[1])
    else:
        return k * u


def _get_numerical_jvp_wrt_specific_input(
    fn, input_idx, inputs, u, eps, is_forward_ad=False
) -> list[torch.Tensor]:
    input = inputs[input_idx]
    input_to_perturb = _get_input_to_perturb(input)
    wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, True)
    nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, eps=eps)
    jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
    u = _reshape_tensor_or_tuple(u, input_to_perturb.shape)
    u = _mul_tensor_or_tuple(u, eps)
    return _compute_numerical_jvps_wrt_specific_input(
        jvp_fn, u, input.is_complex(), is_forward_ad
    )


def _get_numerical_vJu(
    fn, inputs, inp_indices, func_out, all_u, all_v, eps, is_forward_ad
):
    # Note that all_v can also be None, in that case, this function only computes Ju.
    reduced_jacobians: list[list[torch.Tensor]] = []
    for inp_idx, u in zip(inp_indices, all_u):
        all_Ju = _get_numerical_jvp_wrt_specific_input(
            fn, inp_idx, inputs, u, eps, is_forward_ad
        )
        # Filter out the Ju for non floating point outputs
        filtered_Ju = []
        func_out = _as_tuple(func_out)
        if len(all_Ju) != len(func_out):
            raise AssertionError(
                f"Expected all_Ju and func_out to have the same length, "
                f"but got {len(all_Ju)} and {len(func_out)}"
            )
        for Ju, output in zip(all_Ju, func_out):
            if _is_float_or_complex_tensor(output):
                filtered_Ju.append(Ju)
            else:
                # TODO: handle the other Ju
                pass
        if all_v is not None:
            jacobian_scalars: list[torch.Tensor] = []
            for v, Ju in zip(all_v, filtered_Ju):
                jacobian_scalars.append(_dot_with_type_promotion(v, Ju))
            reduced_jacobians.append(jacobian_scalars)
        else:
            reduced_jacobians.append(filtered_Ju)
    return reduced_jacobians


def _check_jacobians_equal(j1, j2, atol):
    # Check whether the max difference between two Jacobian tensors are within some
    # tolerance `atol`.
    for j1_x, j2_x in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True


def _stack_and_check_tensors(
    list_of_list_of_tensors, inputs, numel_outputs
) -> tuple[tuple[torch.Tensor, ...], bool, bool]:
    # For the ith tensor in the inner list checks whether it has the same size and
    # dtype as the ith differentiable input.
    out_jacobians = _allocate_jacobians_with_inputs(inputs, numel_outputs)
    diff_input_list = list(_iter_tensors(inputs, True))
    correct_grad_sizes = True
    correct_grad_types = True
    for i, tensor_list in enumerate(list_of_list_of_tensors):
        inp = diff_input_list[i]
        out_jacobian = out_jacobians[i]
        for j, tensor in enumerate(tensor_list):
            if tensor is not None and tensor.size() != inp.size():
                correct_grad_sizes = False
            elif tensor is not None and tensor.dtype != inp.dtype:
                correct_grad_types = False
            if tensor is None:
                out_jacobian[:, j].zero_()
            else:
                dense = tensor.to_dense() if tensor.layout != torch.strided else tensor
                if out_jacobian[:, j].numel() != dense.numel():
                    raise AssertionError(
                        f"Expected out_jacobian column to have {dense.numel()} elements, "
                        f"but got {out_jacobian[:, j].numel()}"
                    )
                out_jacobian[:, j] = dense.reshape(-1)
    return out_jacobians, correct_grad_sizes, correct_grad_types


FAILED_NONDET_MSG = """\n
NOTE: If your op relies on non-deterministic operations i.e., it is listed here:
https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
this failure might be expected.

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `nondet_tol=<tol>` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `gradcheck_nondet_tol=<tol>`.
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_nondet_tol=<tol>`
"""


def _check_analytical_jacobian_attributes(
    inputs, output, nondet_tol, check_grad_dtypes, fast_mode=False, v=None
) -> tuple[torch.Tensor, ...]:
    # This is used by both fast and slow mode:
    #  - For slow mode, vjps[i][j] is the jth row of the Jacobian wrt the ith
    #    input.
    #  - For fast mode, vjps[i][0] is a linear combination of the rows
    #    of the Jacobian wrt the ith input
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    if fast_mode:
        vjps1 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
        vjps2 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
    else:
        vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
        vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel() if not fast_mode else 1
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(
        vjps1, inputs, output_numel
    )
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    if not types_ok and check_grad_dtypes:
        raise GradcheckError("Gradient has dtype mismatch")
    if not sizes_ok:
        raise GradcheckError("Analytical gradient has incorrect size")
    if not reentrant:
        raise GradcheckError(
            "Backward is not reentrant, i.e., running backward with "
            "same input and grad_output multiple times gives different values, "
            "although analytical gradient matches numerical gradient."
            f"The tolerance for nondeterminism was {nondet_tol}." + FAILED_NONDET_MSG
        )
    return jacobians1


def _get_analytical_vJu_backward_mode(
    inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u
):
    reduced_jacobians: list[list[torch.Tensor]] = []
    for output, v in zip(outputs, all_v):
        all_vJ = _check_analytical_jacobian_attributes(
            inputs, output, nondet_tol, check_grad_dtypes, fast_mode=True, v=v
        )
        jacobian_scalars: list[torch.Tensor] = []
        for vJ, u in zip(all_vJ, all_u):
            # Why do we need squeeze here? vJ is a 2-d tensor so that we can reuse
            # the error checking logic from slow mode
            vJ = vJ.T.squeeze(0)
            if vJ.is_complex():  # C -> R
                tv = torch.view_as_real(vJ.resolve_conj())
                tr = tv.select(-1, 0)
                ti = tv.select(-1, 1)
                jacobian_scalars.append(tr.dot(u[0]) + 1j * ti.dot(u[1]))
            else:  # R -> R
                jacobian_scalars.append(vJ.dot(u))
        reduced_jacobians.append(jacobian_scalars)
    return reduced_jacobians


@deprecated(
    "`get_analytical_jacobian` was part of PyTorch's private API and not "
    "meant to be exposed. We are deprecating it and it will be removed "
    "in a future version of PyTorch. If you have a specific use for "
    "this or feature request for this to be a stable API, please file "
    "us an issue at https://github.com/pytorch/pytorch/issues/new",
    category=FutureWarning,
)
def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    # Replicates the behavior of the old get_analytical_jacobian before the refactor
    # This shares much of its code with _check_analytical_jacobian_attributes
    if (
        grad_out != 1.0
    ):  # grad_out param is only kept for backward compatibility reasons
        raise ValueError(
            "Expected grad_out to be 1.0. get_analytical_jacobian no longer "
            "supports values of grad_out != 1.0."
        )
    if output.is_complex():
        raise ValueError(
            "Expected output to be non-complex. get_analytical_jacobian no "
            "longer supports functions that return complex outputs."
        )
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        return torch.autograd.grad(
            output, diff_input_list, grad_output, retain_graph=True, allow_unused=True
        )

    # Compute everything twice to check for nondeterminism (which we call reentrancy)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())

    output_numel = output.numel()
    jacobians1, types_ok, sizes_ok = _stack_and_check_tensors(
        vjps1, inputs, output_numel
    )
    jacobians2, _, _ = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)

    return jacobians1, reentrant, sizes_ok, types_ok


def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    # Computes the analytical Jacobian in slow mode for a single input-output pair.
    # Forgoes performing checks on dtype, shape, and reentrancy.
    jacobians = _check_analytical_jacobian_attributes(
        inputs, outputs[output_idx], nondet_tol=float("inf"), check_grad_dtypes=False
    )
    return jacobians[input_idx]


def _compute_analytical_jacobian_rows(
    vjp_fn, sample_output
) -> list[list[torch.Tensor | None]]:
    # Computes Jacobian row-by-row by projecting `vjp_fn` = v^T J on standard basis
    # vectors: vjp_fn(e) = e^T J is a corresponding row of the Jacobian.
    # NB: this function does not assume vjp_fn(v) to return tensors with the same
    # number of elements for different v. This is checked when we later combine the
    # rows into a single tensor.
    grad_out_base = torch.zeros_like(
        sample_output, memory_format=torch.legacy_contiguous_format
    )
    flat_grad_out = grad_out_base.view(-1)
    # jacobians_rows[i][j] is the Jacobian jth row for the ith input
    jacobians_rows: list[list[torch.Tensor | None]] = []
    for j in range(flat_grad_out.numel()):
        flat_grad_out.zero_()
        flat_grad_out[j] = 1.0  # projection for jth row of Jacobian
        grad_inputs = vjp_fn(grad_out_base)
        for i, d_x in enumerate(grad_inputs):
            if j == 0:
                jacobians_rows.append([])
            jacobians_rows[i] += [
                d_x.clone() if isinstance(d_x, torch.Tensor) else None
            ]
    return jacobians_rows


def _get_analytical_vjps_wrt_specific_output(
    vjp_fn, sample_output, v
) -> list[list[torch.Tensor | None]]:
    grad_inputs = vjp_fn(v.reshape(sample_output.shape))
    vjps: list[list[torch.Tensor | None]] = [
        [vjp.clone() if isinstance(vjp, torch.Tensor) else None] for vjp in grad_inputs
    ]
    return vjps


def _check_inputs(tupled_inputs) -> bool:
    # Make sure that gradients are saved for at least one input
    any_input_requiring_grad = False
    for idx, inp in enumerate(tupled_inputs):
        if is_tensor_like(inp) and inp.requires_grad:
            if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
                warnings.warn(
                    f"Input #{idx} requires gradient and "
                    "is not a double precision floating point or complex. "
                    "This check will likely fail if all the inputs are "
                    "not of double precision floating point or complex. ",
                    stacklevel=2,
                )
            if inp.is_sparse:
                content = inp._values()
            elif _is_sparse_compressed_tensor(inp):
                content = inp.values()
            else:
                content = inp
            # TODO: To cover more problematic cases, replace stride = 0 check with
            # "any overlap in memory" once we have a proper function to check it.
            if content.layout is not torch._mkldnn:  # type: ignore[attr-defined]
                if not all(
                    st > 0 or sz <= 1
                    for st, sz in zip(content.stride(), content.size())
                ):
                    raise RuntimeError(
                        f"The {idx}th input has a dimension with stride 0. gradcheck only "
                        "supports inputs that are non-overlapping to be able to "
                        "compute the numerical gradients correctly. You should call "
                        ".contiguous on the input before passing it to gradcheck."
                    )
            any_input_requiring_grad = True

    if not any_input_requiring_grad:
        raise ValueError(
            "gradcheck expects at least one input tensor to require gradient, "
            "but none of them have requires_grad=True."
        )
    return True


def _check_outputs(outputs) -> None:
    if any(_is_sparse_any_tensor(t) for t in outputs if isinstance(t, torch.Tensor)):
        # it is easier to call to_dense() on the sparse output than
        # to modify analytical jacobian
        raise ValueError(
            "Sparse output is not supported at gradcheck yet. "
            "Please call to_dense(masked_grad=...) on the output of fn for gradcheck."
        )
    if any(t.layout == torch._mkldnn for t in outputs if isinstance(t, torch.Tensor)):  # type: ignore[attr-defined]
        raise ValueError(
            "MKLDNN output is not supported at gradcheck yet. "
            "Please call to_dense(masked_grad=...) on the output of fn for gradcheck."
        )


def _check_no_differentiable_outputs(
    func, inputs, func_out, eps, *, is_forward_ad
) -> bool:
    # When there are no differentiable outputs, numerical gradient for a function is
    # expected to be zero.
    jacobians_all_inputs_outputs = _get_numerical_jacobian(
        func, inputs, func_out, eps=eps, is_forward_ad=is_forward_ad
    )
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if torch.ne(jacobian, 0).sum() > 0:
                raise GradcheckError(
                    "Numerical gradient for function expected to be zero"
                )
    return True


def _check_no_differentiable_outputs_fast(
    func, func_out, all_inputs, inputs_indices, all_u, eps, nondet_tol
):
    for inp_idx, u in zip(inputs_indices, all_u):
        jvps = _get_numerical_jvp_wrt_specific_input(func, inp_idx, all_inputs, u, eps)
        for jvp in jvps:
            if jvp.numel() == 0:
                continue
            if (jvp - torch.zeros_like(jvp)).abs().max() > nondet_tol:
                raise GradcheckError(
                    "Numerical gradient for function expected to be zero"
                )
    return True


FAILED_BATCHED_GRAD_MSG = """
gradcheck or gradgradcheck failed while testing batched gradient computation.
This could have been invoked in a number of ways (via a test that calls
gradcheck/gradgradcheck directly or via an autogenerated test).

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `check_batched_grad=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `check_batched_grad=False` and/or `check_batched_gradgrad=False`.

If you're modifying an existing operator that supports batched grad computation,
or wish to make a new operator work with batched grad computation, please read
the following.

To compute batched grads (e.g., jacobians, hessians), we vmap over the backward
computation. The most common failure case is if there is a 'vmap-incompatible
operation' in the backward pass. Please see
NOTE: [How to write vmap-compatible backward formulas]
in the codebase for an explanation of how to fix this.
""".strip()

FAILED_BATCHED_GRAD_MSG_FWD_AD = """
gradcheck failed while testing batched gradient computation with forward-mode AD.
This test is enabled automatically when both `check_batched_grad=True`
and `check_forward_ad=True`, but can be disabled in the following ways
depending on how the test was invoked (via a test that calls gradcheck
directly or via an autogenerated test).

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.
If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `check_batched_forward_grad=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `check_batched_forward_grad=False`
"""


def _get_failed_batched_grad_test_msg(
    output_idx, input_idx, res, exp, is_forward_ad=False
):
    return f"""
For output {output_idx} and input {input_idx}:

{FAILED_BATCHED_GRAD_MSG_FWD_AD if is_forward_ad else FAILED_BATCHED_GRAD_MSG}

Got:
{res}

Expected:
{exp}
""".strip()


def _test_batched_grad_forward_ad(func, inputs) -> bool:
    fwAD = torch.autograd.forward_ad  # To avoid early import issues (do we need this?)
    if not isinstance(inputs, tuple):
        raise AssertionError("Expected inputs to be a tuple")

    for input_idx, current_input in enumerate(inputs):
        if not (is_tensor_like(current_input) and current_input.requires_grad):
            continue

        def jvp(tangent: torch.Tensor):
            with fwAD.dual_level():
                dual = fwAD.make_dual(current_input.detach(), tangent)
                inputs_with_dual = tuple(
                    dual
                    if idx == input_idx
                    else (inp.detach() if is_tensor_like(inp) else inp)
                    for idx, inp in enumerate(inputs)
                )
                dual_outputs = _as_tuple(func(*inputs_with_dual))
                ret = []
                for dual_output in dual_outputs:
                    if dual_output is None:
                        continue
                    primal_out, tangent_out = fwAD.unpack_dual(dual_output)
                    if tangent_out is not None:
                        ret.append(tangent_out)
                    else:
                        ret.append(
                            torch.zeros(
                                [], dtype=primal_out.dtype, device=primal_out.device
                            ).expand(primal_out.shape)
                        )
                return tuple(ret)

        if not _is_float_or_complex_tensor(current_input):
            continue

        tangents = [torch.randn_like(current_input) for _ in range(2)]
        expected = [jvp(t) for t in tangents]
        expected = [torch.stack(shards) for shards in zip(*expected)]

        try:
            result = _vmap(jvp)(torch.stack(tangents))
        except RuntimeError as ex:
            # Rethrow to provide a better error message
            raise GradcheckError(
                f"While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG_FWD_AD}"
            ) from ex

        for input_idx, (res, exp) in enumerate(zip(result, expected)):
            if torch.allclose(res, exp):
                continue
            raise GradcheckError(
                _get_failed_batched_grad_test_msg(
                    input_idx, input_idx, res, exp, is_forward_ad=True
                )
            )
    return True


def _test_batched_grad(input, output, output_idx) -> bool:
    # NB: _test_batched_grad compares two autograd.grad invocations with a single
    # vmap(autograd.grad) invocation. It's not exactly a "gradcheck" in the
    # sense that we're not comparing an analytical jacobian with a numeric one,
    # but it is morally similar (we could have computed a full analytic jac
    # via vmap, but that is potentially slow)
    diff_input_list = list(_iter_tensors(input, True))
    grad = functools.partial(
        torch.autograd.grad,
        output,
        diff_input_list,
        retain_graph=True,
        allow_unused=True,
    )

    def vjp(v):
        results = grad(v)
        results = tuple(
            grad
            if grad is not None
            else torch.zeros([], dtype=inp.dtype, device=inp.device).expand(inp.shape)
            for grad, inp in zip(results, diff_input_list)
        )
        return results

    grad_outputs = [torch.randn_like(output) for _ in range(2)]

    expected = [vjp(gO) for gO in grad_outputs]
    expected = [torch.stack(shards) for shards in zip(*expected)]

    # Squash warnings since these are expected to happen in most cases
    # NB: this doesn't work for CUDA tests: https://github.com/pytorch/pytorch/issues/50209
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="There is a performance drop")
        warnings.filterwarnings("ignore", message="Please use `torch.vmap`")
        try:
            result = vmap(vjp)(torch.stack(grad_outputs))
        except RuntimeError as ex:
            # It's OK that we're not raising the error at the correct callsite.
            # That's because the callsite is always going to inside the Python
            # autograd.grad instead of the C++ traceback of what line in the
            # backward formula
            raise GradcheckError(
                f"While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG}"
            ) from ex

    for input_idx, (res, exp) in enumerate(zip(result, expected)):
        if torch.allclose(res, exp):
            continue
        raise GradcheckError(
            _get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp)
        )
    return True


def _test_backward_mul_by_grad_output(outputs, inputs, masked) -> bool:
    # Tests that backward is multiplied by grad_output
    diff_input_list: list[torch.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")
    grads_input = torch.autograd.grad(
        outputs,
        diff_input_list,
        [
            torch.zeros_like(o, memory_format=torch.legacy_contiguous_format)
            for o in outputs
        ],
        allow_unused=True,
    )
    for gi, di in zip(grads_input, diff_input_list):
        if gi is None:
            continue
        if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
            if gi.layout != di.layout:
                raise GradcheckError(
                    "grad is incorrect layout ("
                    + str(gi.layout)
                    + " is not "
                    + str(di.layout)
                    + ")"
                )
            if _is_sparse_any_tensor(gi):
                sparse_kind = str(gi.layout).replace("torch.", "").replace("_coo", "")
                if gi.sparse_dim() != di.sparse_dim():
                    raise GradcheckError(
                        f"grad is {sparse_kind} tensor, but has incorrect sparse_dim"
                        f" {gi.sparse_dim()}, expected {di.sparse_dim()}"
                    )
                if gi.dense_dim() != di.dense_dim():
                    raise GradcheckError(
                        f"grad is {sparse_kind} tensor, but has incorrect dense_dim"
                        f" {gi.dense_dim()}, expected {di.dense_dim()}"
                    )
            gi = gi.to_dense()
            di = di.to_dense()
        if masked:
            if not torch.allclose(gi, torch.zeros_like(gi)):
                raise GradcheckError("backward not multiplied by grad_output")
        elif not gi.eq(0).all():
            raise GradcheckError("backward not multiplied by grad_output")
        if gi.dtype != di.dtype:
            raise GradcheckError("grad is incorrect type")
        if gi.device != di.device:
            raise GradcheckError("grad is incorrect device")
        if gi.size() != di.size():
            raise GradcheckError("grad is incorrect size")
    return True


def _test_undefined_forward_mode(func, outputs, inputs):
    fwAD = torch.autograd.forward_ad

    _inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    _all_v, all_u, _all_u_dense = _make_vectors(
        inp_tensors, outputs, use_forward_ad=True
    )

    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        tensor_indices = set()
        for i, inp in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:  # type: ignore[attr-defined]
                    raise ValueError(
                        "MKLDNN inputs are not supported for forward AD gradcheck."
                    )

                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                # If inp is a differentiable view, the dual might not be the tangent given to
                # make_dual, so read it explicitly from the dual tensor
                fw_grads.append(fwAD.unpack_dual(inp)[1])
                tensor_indices.add(i)
            dual_inputs.append(inp)

        for fw_grad, u in zip(fw_grads, all_u):
            fw_grad.copy_(u.view_as(fw_grad))

        for idx, inp in enumerate(inputs):
            if idx not in tensor_indices:
                continue
            dual_inp_obj = dual_inputs[idx]

            # case 1 (Materialized Zero Tensor Tangent)
            dual_inputs[idx] = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs1 = filter(_is_float_or_complex_tensor, raw_outputs)

            # case 2 (Efficient Zero Tensor Tangent since we don't make a dual object and pass a regular tensor)
            dual_inputs[idx] = inp.detach()
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs2 = filter(_is_float_or_complex_tensor, raw_outputs)

            # reset
            dual_inputs[idx] = dual_inp_obj

            for index_o, (d_o1, d_o2) in enumerate(zip(dual_outputs1, dual_outputs2)):
                _val1, res1 = fwAD.unpack_dual(d_o1)
                _val2, res2 = fwAD.unpack_dual(d_o2)

                if not (res1 is None or res2 is None):
                    if not torch.allclose(res1, res2):
                        raise GradcheckError(
                            "Mismatch in tangent values for output with index: ",
                            index_o,
                            " when input: ",
                            inp,
                            " has an undefined tangent value. ",
                            " Got: ",
                            res1,
                            " but expected: ",
                            res2,
                        )
    return True


def _test_undefined_backward_mode(func, outputs, inputs) -> bool:
    diff_input_list: list[torch.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError("no Tensors requiring grad found in input")

    def warn_bc_breaking():
        warnings.warn(
            "Backwards compatibility: New undefined gradient support checking "
            "feature is enabled by default, but it may break existing callers "
            "of this function. If this is true for you, you can call this "
            'function with "check_undefined_grad=False" to disable the feature',
            stacklevel=2,
        )

    def check_undefined_grad_support(output_to_check):
        grads_output = [
            torch.zeros_like(o, memory_format=torch.legacy_contiguous_format)
            for o in output_to_check
        ]
        try:
            grads_input = torch.autograd.grad(
                output_to_check, diff_input_list, grads_output, allow_unused=True
            )
        except RuntimeError as e:
            warn_bc_breaking()
            raise GradcheckError(
                "Expected backward function to handle undefined output grads. "
                'Please look at "Notes about undefined output gradients" in '
                '"tools/autograd/derivatives.yaml"'
            ) from e

        for gi in grads_input:
            if (gi is not None) and (not gi.eq(0).all()):
                warn_bc_breaking()
                raise GradcheckError(
                    "Expected all input grads to be undefined or zero when all output grads are undefined "
                    'or zero. Please look at "Notes about undefined output gradients" in '
                    '"tools/autograd/derivatives.yaml"'
                )
        return True

    # All backward functions must work properly if all output grads are undefined
    outputs_to_check = [
        [
            torch._C._functions.UndefinedGrad()(o)
            for o in _differentiable_outputs(func(*inputs))
            # This check filters out Tensor-likes that aren't instances of Tensor.
            if isinstance(o, torch.Tensor)
        ]
    ]

    # If there are multiple output grads, we should be able to undef one at a time without error
    if len(outputs_to_check[0]) > 1:
        for undef_grad_idx in range(len(outputs)):
            output_to_check = _differentiable_outputs(func(*inputs))
            outputs_to_check.append(
                [
                    torch._C._functions.UndefinedGrad()(o)
                    if idx == undef_grad_idx
                    else o
                    for idx, o in enumerate(output_to_check)
                ]
            )

    return all(check_undefined_grad_support(output) for output in outputs_to_check)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def _get_notallclose_msg(
    analytical,
    numerical,
    output_idx,
    input_idx,
    complex_indices,
    test_imag=False,
    is_forward_ad=False,
) -> str:
    out_is_complex = (
        (not is_forward_ad) and complex_indices and output_idx in complex_indices
    )
    inp_is_complex = is_forward_ad and complex_indices and input_idx in complex_indices
    part = "imaginary" if test_imag else "real"
    element = "inputs" if is_forward_ad else "outputs"
    prefix = (
        ""
        if not (out_is_complex or inp_is_complex)
        else f"While considering the {part} part of complex {element} only, "
    )
    mode = "computed with forward mode " if is_forward_ad else ""
    return (
        prefix
        + f"Jacobian {mode}mismatch for output {output_idx:d} with respect to input {input_idx:d},\n"
        f"numerical:{numerical}\nanalytical:{analytical}\n"
    )


def _transpose(matrix_of_tensors):
    # returns list of tuples
    return list(zip(*matrix_of_tensors))


def _real_and_imag_output(fn):
    # returns new functions real(fn), and imag(fn) where real(fn) and imag(fn) behave the same as
    # the original fn, except torch.real or torch.imag are applied to the complex outputs
    def apply_to_c_outs(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            outs = _as_tuple(fn(*inputs))
            return tuple(fn_to_apply(o) if o.is_complex() else o for o in outs)

        return wrapped_fn

    return apply_to_c_outs(fn, torch.real), apply_to_c_outs(fn, torch.imag)


def _real_and_imag_input(fn, complex_inp_indices, tupled_inputs):
    # returns new functions that take real inputs instead of complex inputs as
    # (x, y) -> fn(x + y * 1j). And it computes: inp -> fn(inp + y * 1j) and inp -> fn(x + inp * 1j).
    # In each case, the other part is considered constant.
    # We do not use 0 for the constant here to make sure we always call the user function with a valid input.
    def apply_to_c_inps(fn, fn_to_apply):
        def wrapped_fn(*inputs):
            new_inputs = list(inputs)
            for should_be_complex in complex_inp_indices:
                new_inputs[should_be_complex] = fn_to_apply(
                    new_inputs[should_be_complex], tupled_inputs[should_be_complex]
                )
            return _as_tuple(fn(*new_inputs))

        return wrapped_fn

    real_fn = apply_to_c_inps(fn, lambda inp, orig: inp + orig.imag * 1j)
    imag_fn = apply_to_c_inps(fn, lambda inp, orig: orig.real + inp * 1j)
    return real_fn, imag_fn


def _gradcheck_real_imag(
    gradcheck_fn,
    func,
    func_out,
    tupled_inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    check_forward_ad,
    check_backward_ad,
    nondet_tol,
    check_undefined_grad,
):
    complex_out_indices = [i for i, o in enumerate(outputs) if o.is_complex()]
    has_any_complex_output = any(o.is_complex() for o in _as_tuple(func_out))
    if check_backward_ad:
        if has_any_complex_output:
            real_fn, imag_fn = _real_and_imag_output(func)

            imag_func_out = imag_fn(*tupled_inputs)
            imag_outputs = _differentiable_outputs(imag_func_out)
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                tupled_inputs,
                imag_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
                test_imag=True,
            )

            real_func_out = real_fn(*tupled_inputs)
            real_outputs = _differentiable_outputs(real_func_out)
            gradcheck_fn(
                real_fn,
                real_func_out,
                tupled_inputs,
                real_outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_out_indices,
            )
        else:
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
            )

    if check_forward_ad:
        complex_inp_indices = [
            i
            for i, inp in enumerate(tupled_inputs)
            if is_tensor_like(inp) and inp.is_complex()
        ]
        if complex_inp_indices:
            real_fn, imag_fn = _real_and_imag_input(
                func, complex_inp_indices, tupled_inputs
            )

            imag_inputs = [
                inp.imag if is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            imag_func_out = imag_fn(*imag_inputs)
            diff_imag_func_out = _differentiable_outputs(imag_func_out)
            gradcheck_fn(
                imag_fn,
                imag_func_out,
                imag_inputs,
                diff_imag_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                test_imag=True,
                use_forward_ad=True,
            )

            real_inputs = [
                inp.real if is_tensor_like(inp) and inp.is_complex() else inp
                for inp in tupled_inputs
            ]
            real_func_out = real_fn(*real_inputs)
            diff_real_func_out = _differentiable_outputs(real_func_out)
            gradcheck_fn(
                real_fn,
                real_func_out,
                real_inputs,
                diff_real_func_out,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                complex_indices=complex_inp_indices,
                use_forward_ad=True,
            )
            if check_undefined_grad:
                _test_undefined_forward_mode(imag_fn, imag_func_out, imag_inputs)
                _test_undefined_forward_mode(real_fn, real_func_out, real_inputs)
        else:
            gradcheck_fn(
                func,
                func_out,
                tupled_inputs,
                outputs,
                eps,
                rtol,
                atol,
                check_grad_dtypes,
                nondet_tol,
                use_forward_ad=True,
            )
            if check_undefined_grad:
                _test_undefined_forward_mode(func, outputs, tupled_inputs)


def _slow_gradcheck(
    func,
    func_out,
    tupled_inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    nondet_tol,
    *,
    use_forward_ad=False,
    complex_indices=None,
    test_imag=False,
    masked=False,
):
    func_out = _as_tuple(func_out)
    if not outputs:
        return _check_no_differentiable_outputs(
            func, tupled_inputs, func_out, eps=eps, is_forward_ad=use_forward_ad
        )
    tupled_inputs_numerical = tupled_inputs if masked else _densify(tupled_inputs)

    numerical = _transpose(
        _get_numerical_jacobian(
            func,
            tupled_inputs_numerical,
            func_out,
            eps=eps,
            is_forward_ad=use_forward_ad,
        )
    )
    # Note: [numerical vs analytical output length]
    # The numerical path returns jacobian quantity for all outputs, even if requires_grad of that
    # output is False. This behavior is necessary for _check_no_differentiable_outputs to work.
    numerical = [nj for o, nj in zip(func_out, numerical) if o.requires_grad]
    if use_forward_ad:
        analytical_forward = _get_analytical_jacobian_forward_ad(
            func, tupled_inputs, func_out, check_grad_dtypes=check_grad_dtypes
        )

        for i, n_per_out in enumerate(numerical):
            for j, n in enumerate(n_per_out):
                a = analytical_forward[j][i]
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(
                        _get_notallclose_msg(
                            a, n, i, j, complex_indices, test_imag, is_forward_ad=True
                        )
                    )
    else:
        for i, o in enumerate(outputs):
            analytical = _check_analytical_jacobian_attributes(
                tupled_inputs, o, nondet_tol, check_grad_dtypes
            )

            for j, (a, n) in enumerate(zip(analytical, numerical[i])):
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(
                        _get_notallclose_msg(a, n, i, j, complex_indices, test_imag)
                    )

    return True


def _dot_with_type_promotion(u, v):
    if u.dim() != 1 or v.dim() != 1:
        raise AssertionError(
            f"Expected u and v to be 1D tensors, but got dims {u.dim()} and {v.dim()}"
        )
    return (u * v).sum()


def _allclose_with_type_promotion(a, b, rtol, atol):
    promoted_type = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return torch.allclose(a, b, rtol, atol)


def _to_real_dtype(dtype):
    if dtype == torch.complex128:
        return torch.float64
    elif dtype == torch.complex64:
        return torch.float32
    else:
        return dtype


def _vec_from_tensor(x, generator, downcast_complex=False):
    # Create a random vector with the same number of elements as x and the same
    # dtype/device. If x is complex and downcast_complex is False, we create a
    # complex tensor with only real component.
    if x.layout == torch.sparse_coo:
        # For sparse, create a random sparse vec with random values in the same
        # indices. Make sure size is set so that it isn't inferred to be smaller.
        x_values = x._values()
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        values = (
            torch.rand(x_values.numel(), generator=generator)
            .to(dtype=dtype, device=x.device)
            .view(x_values.shape)
        )
        values /= values.norm()
        vec = torch.sparse_coo_tensor(x._indices(), values, x.size(), device=x.device)
    elif _is_sparse_compressed_tensor(x):
        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices, plain_indices = x.crow_indices(), x.col_indices()
        else:
            compressed_indices, plain_indices = x.ccol_indices(), x.row_indices()
        x_values = x.values()
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        values = (
            torch.rand(x_values.numel(), generator=generator)
            .to(dtype=dtype, device=x.device)
            .view(x_values.shape)
        )
        values /= values.norm()
        vec = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            x.size(),
            layout=x.layout,
            device=x.device,
        )
    else:
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        vec = torch.rand(x.numel(), generator=generator).to(
            dtype=dtype, device=x.device
        )
        vec /= vec.norm()
    return vec


def _get_inp_tensors(tupled_inputs):
    inp_idx_tup = [
        (i, t)
        for i, t in enumerate(tupled_inputs)
        if is_tensor_like(t) and t.requires_grad
    ]
    return [tup[0] for tup in inp_idx_tup], [tup[1] for tup in inp_idx_tup]


def _adjusted_atol(atol, u, v):
    # In slow gradcheck, we compare A and B element-wise, i.e., for some a, b we
    # allow: |a - b| < atol + rtol * b. But since we now compare q1 = v^T A u and
    # q2 = v^T B u, we must allow |q1 - q2| < v^T E u + rtol * v^T B u, where E is
    # the correctly sized matrix in which each entry is atol.
    #
    # We see that atol needs to be scaled by v^T M u (where M is an all-ones M x N
    # matrix): v^T M u = \sum_{i} \sum_{j} u[i] * v[j] = sum(u) * sum(v).

    # For the case of complex inputs, u has re and im. components: u = (ur, ui).
    # Let q = a - b = (qr, qi) in the above notation, eg. q is the difference between analytic and numerical
    # Jacobians. Then the transformed tolerance checks being done in the torch.allclose ops
    # are of form abs(Re{q_r u_r - q_i u_i}) < atol * (abs(u_r) + abs(u_i)), and equivalently
    # for imaginary components abs(Im{q_r u_r - q_i ui}) < atol * (abs(u_r) + abs(u_i)).
    # Since u is drawn randomly non-negative, the end effect is a factor
    # (sum(u_r) + sum(u_i)) * sum(v) increase in atol for complex inputs, eg.
    # a statistical factor of 2 as compared to the real case.

    sum_v = 1.0 if v is None else v.sum()

    if isinstance(u, tuple):
        # case of complex input
        ur, ui = u[0], u[1]
        sum_ur = ur.sum()
        sum_ui = ui.sum()
        complex_modified_atol = atol * (float(sum_ur) + float(sum_ui)) * float(sum_v)
        return complex_modified_atol

    # case of real input
    sum_u = u.sum()
    modified_atol = atol * float(sum_u) * float(sum_v)
    return modified_atol


FAST_FAIL_SLOW_OK_MSG = """
Fast gradcheck failed but element-wise differences are small. This means that the
test might've passed in slow_mode!

If you are adding a new operator, please file an issue and then use one of the
workarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck:

If the test
- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck
  with `fast_mode=False` as a keyword argument.
- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test
  to have `gradcheck_fast_mode=False`
- is a Module test (e.g., in common_nn.py), then modify the corresponding
  module_test entry to have `gradcheck_fast_mode=False`
""".strip()


def _run_slow_mode_and_get_error(
    func, tupled_inputs, outputs, input_idx, output_idx, rtol, atol, eps, is_forward_ad
):
    # Compute jacobians in slow mode for better error message
    slow_numerical = _get_numerical_jacobian(
        func, tupled_inputs, outputs, eps=eps, is_forward_ad=is_forward_ad
    )[input_idx][output_idx]
    if is_forward_ad:

        def new_fn(inp):
            new_inputs = list(tupled_inputs)
            new_inputs[input_idx] = inp
            return _as_tuple(func(*new_inputs))[output_idx]

        slow_analytical = _get_analytical_jacobian_forward_ad(
            new_fn, (tupled_inputs[input_idx],), (outputs[output_idx],)
        )[0][0]
    else:
        slow_analytical = _get_analytical_jacobian(
            tupled_inputs, outputs, input_idx, output_idx
        )

    # Assume jacobians are non-empty and have the same shape
    slow_max_diff = (slow_numerical - slow_analytical).abs().max()

    slow_allclose = torch.allclose(slow_analytical, slow_numerical, rtol, atol)
    msg = (
        "\nThe above quantities relating the numerical and analytical jacobians are computed \n"
        "in fast mode. See: https://github.com/pytorch/pytorch/issues/53876 for more background \n"
        "about fast mode. Below, we recompute numerical and analytical jacobians in slow mode:\n\n"
        f"Numerical:\n {slow_numerical}\n"
        f"Analytical:\n{slow_analytical}\n\n"
        f"The max per-element difference (slow mode) is: {slow_max_diff}.\n"
    )
    if slow_allclose:
        # Slow gradcheck would've passed!
        msg += FAST_FAIL_SLOW_OK_MSG
    return msg


def _to_flat_dense_if_sparse(tensor):
    if _is_sparse_any_tensor(tensor):
        return tensor.to_dense().reshape(-1)
    else:
        return tensor


def _make_vectors(inp_tensors, outputs, *, use_forward_ad):
    # Use our own generator to avoid messing with the user's RNG state
    g_cpu = torch.Generator()

    def _vec_from_tensor_cpu(*args):
        # Default allocate all tensors on CPU, so they are on the same device as the generator
        # even if the user specified a default device
        with torch.device("cpu"):
            return _vec_from_tensor(*args)

    all_u = []
    all_u_dense = []
    for inp in inp_tensors:
        ur = _vec_from_tensor_cpu(inp, g_cpu, True)
        ur_dense = _to_flat_dense_if_sparse(ur)
        if inp.is_complex():
            ui = _vec_from_tensor_cpu(inp, g_cpu, True)
            all_u.append((ur, ui))
            ui_dense = _to_flat_dense_if_sparse(ui)
            all_u_dense.append((ur_dense, ui_dense))
        else:
            all_u.append(ur)
            all_u_dense.append(ur_dense)
    all_v = (
        None
        if use_forward_ad
        else [_vec_from_tensor_cpu(out, g_cpu) for out in outputs]
    )
    return all_v, all_u, all_u_dense


def _check_analytical_numerical_equal(
    all_analytical,
    all_numerical,
    complex_indices,
    tupled_inputs,
    outputs,
    func,
    all_v,
    all_u,
    rtol,
    atol,
    eps,
    test_imag,
    *,
    is_forward_ad=False,
):
    for i, all_numerical_for_input_i in enumerate(all_numerical):
        for j, n in enumerate(all_numerical_for_input_i):
            # Forward AD generates the transpose of what this function expects
            if is_forward_ad:
                a = all_analytical[i][j]
            else:
                a = all_analytical[j][i]
            n = n.to(device=a.device)
            updated_atol = _adjusted_atol(atol, all_u[i], all_v[j] if all_v else None)
            if not _allclose_with_type_promotion(a, n.to(a.device), rtol, updated_atol):
                jacobians_str = _run_slow_mode_and_get_error(
                    func, tupled_inputs, outputs, i, j, rtol, atol, eps, is_forward_ad
                )
                raise GradcheckError(
                    _get_notallclose_msg(
                        a, n, j, i, complex_indices, test_imag, is_forward_ad
                    )
                    + jacobians_str
                )


def _fast_gradcheck(
    func,
    func_out,
    inputs,
    outputs,
    eps,
    rtol,
    atol,
    check_grad_dtypes,
    nondet_tol,
    *,
    use_forward_ad=False,
    complex_indices=None,
    test_imag=False,
    masked=False,
):
    # See https://github.com/pytorch/pytorch/issues/53876 for details
    inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    # Backward mode computes v^T * J (VJP)
    # Since we computed J * u (JVP) through finite difference method, we perform an equality check
    # between VJP * u, v * JVP
    # ----
    # Forward mode computes J * u (JVP)
    # Since we already compute JVP through finite difference method,
    # we don't need v for correctness check here as asserted below
    all_v, all_u, all_u_dense = _make_vectors(
        inp_tensors, outputs, use_forward_ad=use_forward_ad
    )

    inputs_numerical, all_u_numerical, all_v_numerical = (
        (inputs, all_u, all_v) if masked else _densify((inputs, all_u, all_v))
    )

    numerical_vJu = _get_numerical_vJu(
        func,
        inputs_numerical,
        inp_tensors_idx,
        func_out,
        all_u_numerical,
        all_v_numerical,
        eps,
        is_forward_ad=use_forward_ad,
    )
    # TODO: replicate https://github.com/pytorch/pytorch/pull/77743 for fast gradcheck as well
    if use_forward_ad:
        if all_v is not None:
            raise AssertionError("Expected all_v to be None.")
        analytical_vJu = _get_analytical_jacobian_forward_ad(
            func,
            inputs,
            _as_tuple(func_out),
            all_u=all_u,
            check_grad_dtypes=check_grad_dtypes,
        )
    else:
        if not outputs:
            _check_no_differentiable_outputs_fast(
                func, func_out, inputs, inp_tensors_idx, all_u, eps, nondet_tol
            )

        analytical_vJu = _get_analytical_vJu_backward_mode(
            inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u_dense
        )

    _check_analytical_numerical_equal(
        analytical_vJu,
        numerical_vJu,
        complex_indices,
        inputs,
        outputs,
        func,
        all_v,
        all_u,
        rtol,
        atol,
        eps,
        test_imag,
        is_forward_ad=use_forward_ad,
    )

    return True


# Note [VarArg of Tensors]
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 'func' accepts a vararg of tensors, which isn't expressible in the type system at the moment.
# If https://mypy.readthedocs.io/en/latest/additional_features.html?highlight=callable#extended-callable-types is accepted,
# the '...' first argument of Callable can be replaced with VarArg(Tensor).
# For now, we permit any input.
def gradcheck(
    func: Callable[..., _TensorOrTensors],  # See Note [VarArg of Tensors]
    inputs: _TensorOrTensors,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_batched_forward_grad: bool = False,
    check_forward_ad: bool = False,
    check_backward_ad: bool = True,
    fast_mode: bool = False,
    masked: bool | None = None,
) -> bool:
    r"""Check gradients computed via small finite differences against analytical
    gradients wrt tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    For most of the complex functions we consider for optimization purposes, no notion of
    Jacobian exists. Instead, gradcheck verifies if the numerical and analytical values of
    the Wirtinger and Conjugate Wirtinger derivatives are consistent. Because the gradient
    computation is done under the assumption that the overall function has a real-valued
    output, we treat functions with complex output in a special way. For these functions,
    gradcheck is applied to two real-valued functions corresponding to taking the real
    components of the complex outputs for the first, and taking the imaginary components
    of the complex outputs for the second. For more details, check out
    :ref:`complex_autograd-doc`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. note::
        Gradcheck may fail when evaluated on non-differentiable points
        because the numerically computed gradients via finite differencing may differ
        those computed analytically (not necessarily because either is incorrect).
        For more context, see :ref:`non-differentiable-func-grad`.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.Tensor.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.
        check_undefined_grad (bool, optional): if ``True``, check if undefined output grads
            are supported and treated as zeros, for ``Tensor`` outputs.
        check_batched_grad (bool, optional): if ``True``, check if we can compute
            batched gradients using prototype vmap support. Defaults to False.
        check_batched_forward_grad (bool, optional): if ``True``, checks if we can compute
            batched forward gradients using forward ad and prototype vmap support. Defaults to ``False``.
        check_forward_ad (bool, optional): if ``True``, check that the gradients computed with forward
            mode AD match the numerical ones. Defaults to ``False``.
        check_backward_ad (bool, optional): if ``False``, do not perform any checks that rely on
            backward mode AD to be implemented. Defaults to ``True``.
        fast_mode (bool, optional): Fast mode for gradcheck and gradgradcheck is currently only
            implemented for R to R functions. If none of the inputs and outputs are complex
            a faster implementation of gradcheck that no longer computes the entire jacobian
            is run; otherwise, we fall back to the slow implementation.
        masked (bool, optional): if ``True``, the gradients of unspecified elements of
            sparse tensors are ignored. Defaults to ``False``.
    Returns:
        ``True`` if all differences satisfy allclose condition

    """
    if not (check_forward_ad or check_backward_ad):
        raise AssertionError(
            "Expected at least one of check_forward_ad or check_backward_ad to be True"
        )
    if check_batched_grad and not check_backward_ad:
        raise AssertionError(
            "Setting check_batched_grad=True requires check_backward_ad to be True"
        )
    if check_batched_forward_grad and not check_forward_ad:
        raise AssertionError(
            "Setting check_batched_forward_grad=True requires check_forward_ad to be True"
        )
    args = locals().copy()
    args.pop("raise_exception")
    if not raise_exception:
        try:
            return _gradcheck_helper(**args)
        except GradcheckError:
            return False
    else:
        return _gradcheck_helper(**args)


def _gradcheck_helper(
    func,
    inputs,
    eps,
    atol,
    rtol,
    nondet_tol,
    check_undefined_grad,
    check_grad_dtypes,
    check_batched_grad,
    check_batched_forward_grad,
    check_forward_ad,
    check_backward_ad,
    fast_mode,
    masked,
):
    tupled_inputs = _as_tuple(inputs)
    _check_inputs(tupled_inputs)

    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)
    _check_outputs(outputs)

    gradcheck_fn = functools.partial(
        _fast_gradcheck if fast_mode else _slow_gradcheck, masked=masked
    )
    _gradcheck_real_imag(
        gradcheck_fn,
        func,
        func_out,
        tupled_inputs,
        outputs,
        eps,
        rtol,
        atol,
        check_grad_dtypes,
        check_forward_ad=check_forward_ad,
        check_backward_ad=check_backward_ad,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
    )

    if check_batched_forward_grad:
        _test_batched_grad_forward_ad(func, tupled_inputs)

    # Short circuit because remaining tests rely on backward AD to be implemented
    if not check_backward_ad:
        return True

    for i, o in enumerate(outputs):
        if check_batched_grad:
            _test_batched_grad(tupled_inputs, o, i)

    _test_backward_mul_by_grad_output(outputs, tupled_inputs, masked)

    if check_undefined_grad and check_backward_ad:
        _test_undefined_backward_mode(func, outputs, tupled_inputs)
    return True


def gradgradcheck(
    func: Callable[..., _TensorOrTensors],  # See Note [VarArg of Tensors]
    inputs: _TensorOrTensors,
    grad_outputs: _TensorOrOptionalTensors | None = None,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    gen_non_contig_grad_outputs: bool = False,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
    check_batched_grad: bool = False,
    check_fwd_over_rev: bool = False,
    check_rev_over_rev: bool = True,
    fast_mode: bool = False,
    masked: bool = False,
) -> bool:
    r"""Check gradients of gradients computed via small finite differences
    against analytical gradients wrt tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point or complex type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :func:`torch.Tensor.expand`), this check will likely fail
       because the numerical gradients computed by point perturbation at such
       indices will change values at all other indices that share the same
       memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of [Tensor or None] or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance. Note that a small amount
            of nondeterminism in the gradient will lead to larger inaccuracies in
            the second derivative.
        check_undefined_grad (bool, optional): if True, check if undefined output grads
            are supported and treated as zeros
        check_batched_grad (bool, optional): if True, check if we can compute
            batched gradients using prototype vmap support. Defaults to False.
        fast_mode (bool, optional): if True, run a faster implementation of gradgradcheck that
            no longer computes the entire jacobian.
        masked (bool, optional): if True, the gradients of unspecified elements of
            sparse tensors are ignored (default, False).
    Returns:
        True if all differences satisfy allclose condition
    """
    if not (check_fwd_over_rev or check_rev_over_rev):
        raise AssertionError(
            "Expected at least one of check_fwd_over_rev or check_rev_over_rev to be True"
        )
    if check_undefined_grad and not check_rev_over_rev:
        raise AssertionError(
            "Setting check_undefined_grad=True requires check_rev_over_rev to be True"
        )
    if check_batched_grad and not check_rev_over_rev:
        raise AssertionError(
            "Setting check_batched_grad=True requires check_rev_over_rev to be True"
        )
    # TODO: do we want to test this too?
    # assert not (check_batched_forward_grad and not check_fwd_over_rev), (
    #     "Setting check_batched_forward_grad=True requires check_fwd_over_rev to be True")
    tupled_inputs = _as_tuple(inputs)

    if grad_outputs is None:
        # If grad_outputs is not specified, create random Tensors of the same shape, type, and device as the outputs

        outputs = _differentiable_outputs(func(*tupled_inputs))
        tupled_grad_outputs = tuple(
            torch.testing.make_tensor(
                x.shape,
                dtype=x.dtype
                if x.is_floating_point() or x.is_complex()
                else torch.double,
                device=x.device,
                low=-1,
                high=1,
                requires_grad=True,
                noncontiguous=gen_non_contig_grad_outputs,
            )
            for x in outputs
        )
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)

    num_outputs = len(tupled_grad_outputs)

    # NB: We need to save the requires_grad information about the inputs here because gradcheck detaches inputs
    #     before running forward mode AD
    diff_input_args_indices = {
        i for i, x in enumerate(tupled_inputs) if is_tensor_like(x) and x.requires_grad
    }
    diff_grad_output_indices = {
        i for i, x in enumerate(tupled_grad_outputs) if x.requires_grad
    }

    def new_func(*args):
        # Restore the requires_grad information
        input_args = tuple(
            x.requires_grad_() if i in diff_input_args_indices else x
            for i, x in enumerate(args[:-num_outputs])
        )
        outputs = _differentiable_outputs(func(*input_args))
        grad_outputs = tuple(
            x.requires_grad_() if i in diff_grad_output_indices else x
            for i, x in enumerate(args[-num_outputs:])
        )
        diff_input_args = tuple(
            x for i, x in enumerate(input_args) if i in diff_input_args_indices
        )
        grad_inputs = torch.autograd.grad(
            outputs, diff_input_args, grad_outputs, create_graph=True, allow_unused=True
        )
        grad_inputs = tuple(g for g in grad_inputs if g is not None)
        return grad_inputs

    return gradcheck(
        new_func,
        tupled_inputs + tupled_grad_outputs,
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=raise_exception,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
        check_grad_dtypes=check_grad_dtypes,
        check_batched_grad=check_batched_grad,
        fast_mode=fast_mode,
        check_forward_ad=check_fwd_over_rev,
        check_backward_ad=check_rev_over_rev,
        masked=masked,
    )
<|endoftext|># mypy: allow-untyped-defs


def maybe_view(tensor, size, check_same_size=True):
    if check_same_size and tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)


def maybe_unexpand(tensor, old_size, check_same_size=True):
    if check_same_size and tensor.size() == old_size:
        return tensor
    num_unsqueezed = tensor.dim() - len(old_size)
    expanded_dims = [
        dim
        for dim, (expanded, original) in enumerate(
            zip(tensor.size()[num_unsqueezed:], old_size)
        )
        if expanded != original
    ]

    for _ in range(num_unsqueezed):
        tensor = tensor.sum(0, keepdim=False)
    for dim in expanded_dims:
        tensor = tensor.sum(dim, keepdim=True)
    return tensor
<|endoftext|># mypy: allow-untyped-defs
import operator
from functools import reduce
from typing_extensions import deprecated

import torch
import torch._utils
from torch.autograd.function import Function


class Type(Function):
    @staticmethod
    @deprecated(
        "`torch.autograd._functions.Type` is deprecated as of PyTorch 2.1, "
        "please use `torch.tensor.to(dtype=dtype)` instead.",
        category=FutureWarning,
    )
    # pyrefly: ignore [bad-override]
    def forward(ctx, i, dest_type):
        ctx.input_type = type(i)
        ctx.input_device = -1 if not i.is_cuda else i.get_device()
        return i.type(dest_type)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if ctx.input_device == -1:
            return grad_output.type(ctx.input_type), None
        else:
            with torch.accelerator.device_index(ctx.input_device):
                return grad_output.type(ctx.input_type), None


# TODO: deprecate this
class Resize(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, tensor, sizes):
        ctx.sizes = sizes
        ctx.numel = reduce(operator.mul, sizes, 1)
        if tensor.numel() != ctx.numel:
            raise RuntimeError(
                (
                    "requested resize to {} ({} elements in total), "
                    "but the given tensor has a size of {} ({} elements). "
                    "autograd's resize can only change the shape of a given "
                    "tensor, while preserving the number of elements. "
                ).format(
                    "x".join(map(str, sizes)),
                    ctx.numel,
                    "x".join(map(str, tensor.size())),
                    tensor.numel(),
                )
            )
        ctx.input_sizes = tensor.size()
        if tensor.is_quantized:
            tensor.copy_(tensor)
            return tensor.contiguous().view(*sizes)
        if tensor.is_contiguous():
            result = tensor.new(tensor).contiguous().view(*sizes)
            return result
        else:
            return tensor.contiguous().view(*sizes)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if grad_output.numel() != ctx.numel:
            raise AssertionError(
                f"Expected grad_output to have {ctx.numel} elements, but got {grad_output.numel()}"
            )
        return grad_output.contiguous().view(ctx.input_sizes), None
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch._prims_common as utils

# Utilities should come BEFORE this import
from torch._decomp import register_decomposition
from torch._prims_common import TensorLikeType
from torch._prims_common.wrappers import out_wrapper
from torch._refs import _broadcast_shapes


# Data conversion references.
#
# Note: this module breaks the usual _refs to torch naming scheme where
# _refs.foo.bar is a ref for torch.foo.bar.  The following definitions are not
# part of _refs/__init__.py to avoid name clashes with Python builtin types
# (like int).

__all__ = [
    # dtypes
    "bfloat16",
    "bool",
    "byte",
    "cdouble",
    "cfloat",
    "chalf",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
    # misc
    "complex",
    "polar",
]


def _make_conversion_method(name: str, dtype: torch.dtype):
    def fn(
        self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
    ) -> TensorLikeType:
        return self.to(dtype, memory_format=memory_format)  # type: ignore[call-overload]

    fn.__name__ = name
    return fn


bfloat16 = _make_conversion_method("bfloat16", torch.bfloat16)

bool = _make_conversion_method("bool", torch.bool)

byte = _make_conversion_method("byte", torch.uint8)

cdouble = _make_conversion_method("cdouble", torch.cdouble)

cfloat = _make_conversion_method("cfloat", torch.cfloat)

chalf = _make_conversion_method("chalf", torch.complex32)

char = _make_conversion_method("char", torch.int8)

double = _make_conversion_method("double", torch.double)

float = _make_conversion_method("float", torch.float)

half = _make_conversion_method("half", torch.half)

int = _make_conversion_method("int", torch.int)

long = _make_conversion_method("long", torch.long)

short = _make_conversion_method("short", torch.short)


@register_decomposition(torch._ops.ops.aten.complex)
# Note: complex has type promotion tests disabled due to different semantics.
# exact_dtype is for compat with complex_check_dtype from core.
@out_wrapper(exact_dtype=True)
def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType:
    allowed_dtypes = (torch.float32, torch.float64, torch.float16)
    torch._check(
        real.dtype in allowed_dtypes and imag.dtype in allowed_dtypes,
        lambda: (
            f"Expected both inputs to be Half, Float or Double tensors but got "
            f"{real.dtype} and {imag.dtype}"
        ),
    )
    torch._check(
        real.dtype == imag.dtype,
        lambda: (
            f"Expected object of scalar type {real.dtype} but got "
            f"scalar type {imag.dtype} for second argument"
        ),
    )
    result_dtype = utils.corresponding_complex_dtype(real.dtype)  # type: ignore[arg-type]
    common_shape = _broadcast_shapes(real.shape, imag.shape)
    result = real.new_empty(
        common_shape,
        dtype=result_dtype,
        layout=real.layout,
        device=real.device,
        # pin_memory=real.is_pinned(),  # NYI
    )
    result.real = real
    result.imag = imag
    return result


@register_decomposition(torch._ops.ops.aten.polar)
# Note: polar has type promotion tests disabled due to different semantics.
# exact_dtype is for compat with complex_check_dtype from core.
@out_wrapper(exact_dtype=True)
def polar(abs: TensorLikeType, angle: TensorLikeType) -> TensorLikeType:
    result = torch.complex(abs, angle)
    result.real = abs * torch.cos(angle)
    result.imag = abs * torch.sin(angle)
    return result
<|endoftext|>import math
from collections.abc import Iterable, Sequence
from typing import Literal, NamedTuple

import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper


__all__ = [
    # Transforms
    "fft",
    "fft2",
    "fftn",
    "hfft",
    "hfft2",
    "hfftn",
    "rfft",
    "rfft2",
    "rfftn",
    "ifft",
    "ifft2",
    "ifftn",
    "ihfft",
    "ihfft2",
    "ihfftn",
    "irfft",
    "irfft2",
    "irfftn",
    # Helpers
    "fftshift",
    "ifftshift",
]

NormType = None | Literal["forward", "backward", "ortho"]
_NORM_VALUES = {None, "forward", "backward", "ortho"}
aten = torch._ops.ops.aten


def _apply_norm(
    x: TensorLikeType, norm: NormType, signal_numel: int, forward: bool
) -> TensorLikeType:
    """Apply normalization to the un-normalized FFT result"""
    torch._check(norm in _NORM_VALUES, lambda: f"Invalid normalization mode: {norm}")

    if norm == "ortho":
        return x * (1 / math.sqrt(signal_numel))

    normalize = (not forward and (norm is None or norm == "backward")) or (
        forward and norm == "forward"
    )
    return x * (1 / signal_numel) if normalize else x


def _promote_type_fft(
    dtype: torch.dtype, require_complex: bool, device: torch.device
) -> torch.dtype:
    """Helper to promote a dtype to one supported by the FFT primitives"""
    if dtype.is_complex:
        return dtype

    # Promote integral to default float type
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()

    allowed_types = [torch.float32, torch.float64]
    maybe_support_half = device.type in ["cuda", "meta", "xpu"]

    if maybe_support_half:
        allowed_types.append(torch.float16)
        # bfloat16 is supported for FFT on CUDA/XPU devices, but there is no
        # corresponding "complex bfloat16" dtype (corresponding_complex_dtype
        # maps it to complex64). Promote bfloat16 -> float32 here so that:
        #   * the decomposition output dtype (complex64) matches aten, and
        #   * a real dtype conversion happens, so the result never aliases the
        #     input. Otherwise, for size-1 inputs the prim would return a view
        #     while aten returns a fresh tensor, breaking view-consistency
        #     (test_python_ref__refs_fft_*_cuda_bfloat16).
        if dtype == torch.bfloat16:
            dtype = torch.float32
    torch._check_not_implemented(
        dtype in allowed_types, lambda: f"Unsupported dtype {dtype}"
    )

    if require_complex:
        dtype = utils.corresponding_complex_dtype(dtype)

    return dtype


def _maybe_promote_tensor_fft(
    t: TensorLikeType, require_complex: bool = False
) -> TensorLikeType:
    """Helper to promote a tensor to a dtype supported by the FFT primitives"""
    cur_type = t.dtype
    new_type = _promote_type_fft(cur_type, require_complex, t.device)
    return _maybe_convert_to_dtype(t, new_type)  # type: ignore[return-value]


def _resize_fft_input(
    x: TensorLikeType, dims: tuple[int, ...], sizes: tuple[int, ...]
) -> TensorLikeType:
    """
    Fixes the shape of x such that x.size(dims[i]) == sizes[i],
    either by zero-padding, or by slicing x starting from 0.
    """
    if len(dims) != len(sizes):
        raise AssertionError(
            f"dims and sizes must have the same length, got {len(dims)} and {len(sizes)}"
        )
    must_copy = False
    x_sizes = x.shape
    pad_amount = [0] * len(x_sizes) * 2
    for i in range(len(dims)):
        if sizes[i] == -1:
            continue

        if x_sizes[dims[i]] < sizes[i]:
            must_copy = True
            pad_idx = len(pad_amount) - 2 * dims[i] - 1

            pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]]

        if x_sizes[dims[i]] > sizes[i]:
            x = x.narrow(dims[i], 0, sizes[i])

    return torch.constant_pad_nd(x, pad_amount) if must_copy else x


def _fft_c2r(
    func_name: str,
    input: TensorLikeType,
    n: int | None,
    dim: int,
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    """Common code for performing any complex to real FFT (irfft or hfft)"""
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    last_dim_size = n if n is not None else 2 * (input.shape[dim] - 1)
    torch._check(
        last_dim_size >= 1,
        lambda: f"Invalid number of data points ({last_dim_size}) specified",
    )

    if n is not None:
        input = _resize_fft_input(input, dims=dims, sizes=(last_dim_size // 2 + 1,))

    if forward:
        input = torch.conj(input)

    output = prims.fft_c2r(input, dim=dims, last_dim_size=last_dim_size)
    return _apply_norm(output, norm=norm, signal_numel=last_dim_size, forward=forward)


def _fft_r2c(
    func_name: str,
    input: TensorLikeType,
    n: int | None,
    dim: int,
    norm: NormType,
    forward: bool,
    onesided: bool,
) -> TensorLikeType:
    """Common code for performing any real to complex FFT (rfft or ihfft)"""
    torch._check_type(
        not input.dtype.is_complex,
        lambda: f"{func_name} expects a floating point input tensor, but got {input.dtype}",
    )
    input = _maybe_promote_tensor_fft(input)
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    dim_size = n if n is not None else input.shape[dim]
    torch._check(
        dim_size >= 1, lambda: f"Invalid number of data points ({dim_size}) specified"
    )

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_r2c(input, dim=dims, onesided=onesided)
    ret = _apply_norm(ret, norm, dim_size, forward)
    return ret if forward else torch.conj(ret)


def _fft_c2c(
    func_name: str,
    input: TensorLikeType,
    n: int | None,
    dim: int,
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    """Common code for performing any complex to complex FFT (fft or ifft)"""
    torch._check(
        input.dtype.is_complex,
        lambda: f"{func_name} expects a complex input tensor, but got {input.dtype}",
    )
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    dim_size = n if n is not None else input.shape[dim]
    torch._check(
        dim_size >= 1, lambda: f"Invalid number of data points ({dim_size}) specified"
    )

    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    ret = prims.fft_c2c(input, dim=dims, forward=forward)
    return _apply_norm(ret, norm, dim_size, forward)


@register_decomposition(aten.fft_fft)
@out_wrapper()
def fft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    if input.dtype.is_complex:
        return _fft_c2c("fft", input, n, dim, norm, forward=True)
    else:
        return _fft_r2c("fft", input, n, dim, norm, forward=True, onesided=False)


@register_decomposition(aten.fft_ifft)
@out_wrapper()
def ifft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    if input.dtype.is_complex:
        return _fft_c2c("ifft", input, n, dim, norm, forward=False)
    else:
        return _fft_r2c("ifft", input, n, dim, norm, forward=False, onesided=False)


@register_decomposition(aten.fft_rfft)
@out_wrapper()
def rfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_r2c("rfft", input, n, dim, norm, forward=True, onesided=True)


@register_decomposition(aten.fft_irfft)
@out_wrapper()
def irfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_c2r("irfft", input, n, dim, norm, forward=False)


@register_decomposition(aten.fft_hfft)
@out_wrapper()
def hfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_c2r("hfft", input, n, dim, norm, forward=True)


@register_decomposition(aten.fft_ihfft)
@out_wrapper()
def ihfft(
    input: TensorLikeType,
    n: int | None = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    return _fft_r2c("ihfft", input, n, dim, norm, forward=False, onesided=True)


class _ShapeAndDims(NamedTuple):
    shape: tuple[int, ...]
    dims: tuple[int, ...]


def _canonicalize_fft_shape_and_dim_args(
    input: TensorLikeType, shape: ShapeType | None, dim: DimsType | None
) -> _ShapeAndDims:
    """Convert the shape and dim arguments into a canonical form where neither are optional"""
    input_dim = input.ndim
    input_sizes = input.shape

    if dim is not None:
        if not isinstance(dim, Sequence):
            dim = (dim,)
