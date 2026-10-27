        Instead of taking individual arguments, the generated function takes
        a single 'args_list' parameter, extracts placeholder values from it,
        and clears the list.
        """
        # Generate the function signature with args_list parameter
        fn_def = f"def {self._func_name}(self, args_list){maybe_return_annotation}:"

        if free_vars:
            # This is horribly manual but we don't get the "raw" free vars
            # without a bigger refactor.
            placeholder_vars = [
                v.split(":")[0].split("=")[0].strip() for v in free_vars if v != "self"
            ]

            if placeholder_vars:
                fn_def += "\n    args_iter = iter(args_list)"
                for var in placeholder_vars:
                    fn_def += f"\n    {var} = next(args_iter)"
                fn_def += "\n    args_list.clear()"

        return fn_def


class _PyTreeCodeGen(CodeGen):
    def __init__(self, pytree_info: _PyTreeInfo) -> None:
        super().__init__()
        self.pytree_info: _PyTreeInfo = pytree_info

    def process_inputs(self, *inputs: Any) -> Any:
        flat_args = pytree.arg_tree_leaves(*inputs)
        return flat_args

    def process_outputs(self, out: Any) -> Any:
        if self.pytree_info is None or self.pytree_info.out_spec is None:
            return out
        if not isinstance(out, (list, tuple)):
            out = [out]
        if self.pytree_info.out_spec is None:
            raise AssertionError("pytree_info.out_spec is None")
        return pytree.tree_unflatten(out, self.pytree_info.out_spec)

    def _format_annotations(self, free_vars: list[str], expanded_def: bool) -> str:
        """Helper to format annotations for variables in pytree codegen."""
        if not free_vars:
            return ""

        has_annotation = [x for x in free_vars if ":" in x]
        if not has_annotation:
            return ""

        if expanded_def:
            return "\n    " + "\n    ".join(has_annotation)
        else:
            # Use join() to avoid trailing whitespace (breaks expecttest snapshots).
            return "\n    " + "; ".join(has_annotation) + ";\n"

    def gen_var_bindings(
        self, fn_args: list[str], free_vars: list[str], expanded_def: bool
    ) -> str:
        in_spec = self.pytree_info.in_spec
        # when kwargs is present, in_spec is tuple(args, kwargs)
        has_args_kwargs_tuple = (
            in_spec.type is tuple
            and in_spec.num_children == 2
            and in_spec.child(0).type is tuple
            and in_spec.child(1).type is dict
        )
        fn_kwargs = "{}"
        fn_signature = f"[{', '.join(fn_args)}], self._in_spec"
        if has_args_kwargs_tuple:
            count_args = in_spec.child(0).num_children
            fn_args = self.pytree_info.orig_args[:count_args]
            fn_kwargs = (
                "{"
                + ", ".join(
                    f"'{k}':{v}"
                    for k, v in zip(
                        in_spec.child(1).context,
                        self.pytree_info.orig_args[count_args:],
                    )
                )
                + "}"
            )
            fn_signature = f"([{', '.join(fn_args)}], {fn_kwargs}), self._in_spec"

        # in Python, `var1: annotation1, var2: annotation2 = function_call()` is invalid.
        # we need to split it to two lines:
        # one for annotation: `var1: annotation1; var2: annotation2;` (note the semicolon)
        # one for code: `var1, var2, = function_call()`
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        bindings = self._format_annotations(free_vars, expanded_def)
        bindings += f"""
    {", ".join(without_annotation)}, = fx_pytree.tree_flatten_spec({fn_signature})"""
        return bindings

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        # Given a user function/model:
        #   myargs = [myargs0, myargs1]
        #   mykwargs = {'mykwargs0': ..., 'mykwargs1': ...}
        #   def forward(self, mypos, *myargs, mykey=None, **mykwargs):
        #
        # The generated code flattens all keywords into positional arguments for `forward()`
        #   e.g forward(self, mypos, myargs0, myargs1, mykey, mykwargs0, mykwargs1):
        #
        # Within `forward`, `tree_flatten_spec``still parses args and kwargs separately
        #   e.g. tree_flatten_spec(([mypos, myargs0, myargs1],
        #                           {'mykey':mykey, 'mykwargs0':mykwargs0, 'mykwargs1':mykwargs1}),
        #                          self._in_spec)
        #
        # If the user function/model does not have keywords, the dict is suppressed from tree_flatten_spec
        #   e.g. tree_flatten_spec([mypos, myargs0, myargs1]), self._in_spec)
        if self.pytree_info is None:
            return super().gen_fn_def(
                free_vars, maybe_return_annotation, expanded_def=expanded_def
            )

        fn_args = self.pytree_info.orig_args
        has_orig_self = (fn_args[0] == "self") if len(fn_args) > 0 else False
        if has_orig_self:
            free_vars.insert(0, "self")
        fn_definition = super().gen_fn_def(
            fn_args[:], maybe_return_annotation, expanded_def=expanded_def
        )

        if len(free_vars) > 0:  # pytree has placeholders in it
            fn_definition += self.gen_var_bindings(fn_args, free_vars, expanded_def)
        return fn_definition

    def generate_output(
        self,
        output_args: Argument,
        *,
        descs: Sequence[str] | None = None,
        repr_fn: Callable[[object], str] | None = None,
    ) -> str:
        if repr_fn is None:
            repr_fn = repr
        if self.pytree_info and self.pytree_info.out_spec:
            if descs is not None and isinstance(output_args, (list, tuple)):
                return (
                    self._format_multiline_container(
                        output_args,
                        descs,
                        "return pytree.tree_unflatten(",
                        repr_fn=repr_fn,
                    )
                    + ", self._out_spec)"
                )
            else:
                return f"return pytree.tree_unflatten({repr_fn(output_args)}, self._out_spec)"
        else:
            return super().generate_output(output_args, descs=descs, repr_fn=repr_fn)


class _ExportCodeGen(_PyTreeCodeGen):
    def __init__(
        self,
        pytree_info: _PyTreeInfo,
        in_shuffle_graph: GraphModule,
        out_shuffle_graph: GraphModule,
        tree_leaf_names: list[str],
        root: torch.nn.Module | None,
    ) -> None:
        super().__init__(pytree_info)
        self.in_shuffle_graph = in_shuffle_graph
        self.out_shuffle_graph = out_shuffle_graph
        self.tree_leaf_names = tree_leaf_names
        self.root = root

    def process_inputs(self, *inputs: Any) -> Any:
        flat_args = super().process_inputs(*inputs)
        if self.root is not None:
            flat_args = (self.root, *flat_args)
        self.flat_args = flat_args
        return self.in_shuffle_graph(*flat_args)

    def process_outputs(self, out: Any) -> Any:
        flat_outs = self.out_shuffle_graph(*self.flat_args, *out)
        del self.flat_args
        ret = super().process_outputs(flat_outs)
        return ret

    def gen_fn_def(self, *args: Any, **kwargs: Any) -> str:
        fn_def = super().gen_fn_def(*args, **kwargs)
        return fn_def

    def gen_var_bindings(
        self, fn_args: list[str], free_vars: list[str], expanded_def: bool
    ) -> str:
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        fn_signature: str = f"{', '.join(fn_args)}"
        if self.root is not None:
            fn_signature = f"self, {fn_signature}"
        return f"""
    {", ".join(self.tree_leaf_names)}, = pytree.tree_leaves(({fn_signature},))
    {", ".join(without_annotation)}, = self._in_shuffle_graph({", ".join(self.tree_leaf_names)})"""

    def generate_output(self, output_args: Argument, *args: Any, **kwargs: Any) -> str:
        if not isinstance(output_args, (list, tuple)):
            raise TypeError(
                f"Expected list or tuple for output_args, got {type(output_args)}"
            )
        output = f"self._out_shuffle_graph({', '.join(self.tree_leaf_names)}, {', '.join([str(a) for a in output_args])})"
        return f"return pytree.tree_unflatten({output}, self._out_spec)"


class _FindNodesLookupTable:
    """
    Side table for the graph for the purpose of doing fast queries
    """

    def __init__(self) -> None:
        self.table: dict[tuple[str, Target | None], dict[Node, None]] = defaultdict(
            dict
        )

    def _key(self, node: Node) -> tuple[str, Target | None]:
        return (node.op, node.target if node.op == "call_function" else None)

    def __contains__(self, node: Node) -> bool:
        return node in self.table[self._key(node)]

    def insert(self, node: Node) -> None:
        self.table[self._key(node)][node] = None

    def remove(self, node: Node) -> None:
        self.table[self._key(node)].pop(node)

    # TODO: should return list[Node], see _node_list.__iter__ comment
    def find_nodes(self, *, op: str, target: Target | None = None) -> list[Any]:
        if op == "call_function":
            if target is None:
                raise AssertionError("target must not be None for call_function op")
            return [*self.table[(op, target)].keys()]

        if target is None:
            return [*self.table[(op, None)].keys()]

        # op is call_method, get_attr, call_module
        return [node for node in self.table[(op, None)] if node.target == target]


@compatibility(is_backward_compatible=True)
class Graph:
    """
    ``Graph`` is the main data structure used in the FX Intermediate Representation.
    It consists of a series of ``Node`` s, each representing callsites (or other
    syntactic constructs). The list of ``Node`` s, taken together, constitute a
    valid Python function.

    For example, the following code

    .. code-block:: python

        import torch
        import torch.fx


        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return torch.topk(
                    torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
                )


        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

    Will produce the following Graph::

        print(gm.graph)

    .. code-block:: text

        graph(x):
            %linear_weight : [num_users=1] = self.linear.weight
            %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
            %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
            %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
            %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
            %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
            return topk_1

    For the semantics of operations represented in the ``Graph``, please see :class:`Node`.
    """

    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        owning_module: GraphModule | None = None,
        tracer_cls: type[Tracer] | None = None,
        tracer_extras: dict[str, Any] | None = None,
    ) -> None:
        """
        Construct an empty Graph.
        """
        self._root: Node = Node(self, "", "root", "", (), {})
        self._used_names: dict[str, int] = {}  # base name -> number
        self._insert = self._root.prepend
        self._len = 0
        self._graph_namespace = _Namespace()
        self._owning_module = owning_module
        self._tracer_cls = tracer_cls
        self._tracer_extras = tracer_extras
        self._codegen = CodeGen()
        self._co_fields: dict[str, Any] = {}
        self._find_nodes_lookup_table = _FindNodesLookupTable()

    @property
    def owning_module(self) -> GraphModule | None:
        return self._owning_module

    @owning_module.setter
    def owning_module(self, mod: GraphModule | None) -> None:
        self._owning_module = mod

    @property
    def nodes(self) -> _node_list:
        """
        Get the list of Nodes that constitute this Graph.

        Note that this ``Node`` list representation is a doubly-linked list. Mutations
        during iteration (e.g. delete a Node, add a Node) are safe.

        Returns:

            A doubly-linked list of Nodes. Note that ``reversed`` can be called on
            this list to switch iteration order.
        """
        return _node_list(self)

    @compatibility(is_backward_compatible=False)
    def output_node(self) -> Node:
        output_node = next(iter(reversed(self.nodes)))
        if output_node.op != "output":
            raise AssertionError(f"Expected output node, got op={output_node.op}")
        return output_node

    @compatibility(is_backward_compatible=False)
    # TODO: should return list[Node], see _node_list.__iter__ comment
    def find_nodes(
        self, *, op: str, target: Target | None = None, sort: bool = True
    ) -> list[Any]:
        """
        Allows for fast query of nodes

        Args:

            op (str): the name of the operation

            target (Optional[Target]): the target of the node. For call_function,
                the target is required. For other ops, the target is optional.

            sort (bool): whether to return nodes in the order they appear on
                         on the graph.

        Returns:

            Iterable of nodes with the requested op and target.
        """
        node_list = self._find_nodes_lookup_table.find_nodes(op=op, target=target)
        if sort:
            return sorted(node_list)
        return node_list

    @compatibility(is_backward_compatible=True)
    def graph_copy(
        self, g: Graph, val_map: dict[Node, Node], return_output_node: bool = False
    ) -> Argument | None:
        """
        Copy all nodes from a given graph into ``self``.

        Args:

            g (Graph): The source graph from which to copy Nodes.

            val_map (Dict[Node, Node]): a dictionary that will be populated with a mapping
                from nodes in ``g`` to nodes in ``self``. Note that ``val_map`` can be passed
                in with values in it already to override copying of certain values.

        Returns:

            The value in ``self`` that is now equivalent to the output value in ``g``,
            if ``g`` had an ``output`` node. ``None`` otherwise.
        """
        for node in g.nodes:
            if node in val_map:
                continue
            if node.op == "output":
                rv = map_arg(node.args[0], lambda n: val_map[n])
                return rv if not return_output_node else (rv, node)
            val_map[node] = self.node_copy(node, lambda n: val_map[n])
        return None

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Graph:
        """
        Explicitly implement __deepcopy__ to prevent excessive recursion depth
        from the default implementation. This uses graph_copy to copy the nodes
        in an iterative way, rather than recursive. It also populates the
        memoization table to prevent unnecessary copies (e.g. references to
        nodes or other parts of the Graph from a custom GraphModule implementation.
        """
        memo = memo if memo else {}
        g = Graph(tracer_cls=self._tracer_cls)
        output_vals = g.graph_copy(
            self,
            val_map=memo,  # pyrefly: ignore[bad-argument-type]
            return_output_node=True,
        )
        g._codegen = copy.deepcopy(self._codegen)
        if output_vals is not None:
            if not isinstance(output_vals, tuple):
                raise AssertionError(
                    f"Expected output_vals to be tuple, got {type(output_vals)}"
                )
            output_val, old_output_node = output_vals
            new_output_node = g.output(
                # pyrefly: ignore [bad-argument-type]
                output_val,
                type_expr=getattr(old_output_node, "type", None),
            )
            # pyrefly: ignore [missing-attribute]
            new_output_node.meta = copy.copy(old_output_node.meta)
        return g

    @compatibility(is_backward_compatible=True)
    def create_node(
        self,
        op: str,
        target: Target,
        args: tuple[Argument, ...] | None = None,
        kwargs: dict[str, Argument] | None = None,
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> Node:
        """
        Create a ``Node`` and add it to the ``Graph`` at the current insert-point.
        Note that the current insert-point can be set via :meth:`Graph.inserting_before`
        and :meth:`Graph.inserting_after`.

        Args:
            op (str): the opcode for this Node. One of 'call_function', 'call_method', 'get_attr',
                'call_module', 'placeholder', or 'output'. The semantics of these opcodes are
                described in the ``Graph`` docstring.

            args (Optional[Tuple[Argument, ...]]): is a tuple of arguments to this node.

            kwargs (Optional[Dict[str, Argument]]): the kwargs of this Node

            name (Optional[str]): an optional string name for the ``Node``.
                This will influence the name of the value assigned to in the
                Python generated code.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted node.
        """
        # `target in _legal_ops` is checked in Node.__init__
        if not args:
            args = ()
        else:
            if not isinstance(args, tuple):
                raise AssertionError(f"args must be a tuple, got {type(args)}")
        if not kwargs:
            kwargs = immutable_dict()
        else:
            if not isinstance(kwargs, dict):
                raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")

        # TODO: Generalize this invariant to all FX node args once the broader
        # FX argument contract no longer permits raw symbolic leaves.
        if op in ("call_function", "call_method", "call_module") and (args or kwargs):
            for val in pytree.tree_iter((args, kwargs)):
                if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                    warnings.warn(
                        f"Raw {type(val).__name__} value ({val}) passed as argument to "
                        f"Graph.create_node(op='{op}', target={target}). "
                        f"Use create_*_node() helpers for tensor metadata queries "
                        f"or materialize_symints() for general symbolic expressions.",
                        stacklevel=2,
                    )
                    break

        candidate = name if name is not None else self._target_to_str(target)
        name = self._graph_namespace.create_name(candidate, None)
        n = Node(self, name, op, target, args, kwargs, type_expr)

        if (
            self.owning_module is not None
            and getattr(self.owning_module, "_create_node_hooks", None) is not None
        ):
            for f in self.owning_module._create_node_hooks:
                f(n)

        self._graph_namespace.associate_name_with_obj(name, n)

        self._insert(n)
        self._find_nodes_lookup_table.insert(n)
        self._len += 1
        return n

    @compatibility(is_backward_compatible=False)
    def process_inputs(self, *args: Any) -> Any:
        """
        Processes args so that they can be passed to the FX graph.
        """
        return self._codegen.process_inputs(*args)

    @compatibility(is_backward_compatible=False)
    def process_outputs(self, out: Any) -> Any:
        return self._codegen.process_outputs(out)

    @compatibility(is_backward_compatible=True)
    def erase_node(self, to_erase: Node) -> None:
        """
        Erases a ``Node`` from the ``Graph``. Throws an exception if
        there are still users of that node in the ``Graph``.

        Args:

            to_erase (Node): The ``Node`` to erase from the ``Graph``.
        """
        if len(to_erase.users) > 0:
            raise RuntimeError(
                f"Tried to erase Node {to_erase} but it still had {len(to_erase.users)} "
                f"users in the graph: {to_erase.users}!"
            )
        if to_erase.graph != self:
            raise RuntimeError(f"Attempting to remove {to_erase} from wrong graph!")
        if to_erase._erased:
            warnings.warn(f"erase_node({to_erase}) on an already erased node")
            return

        if (
            self.owning_module is not None
            and getattr(self.owning_module, "_erase_node_hooks", None) is not None
        ):
            for f in self.owning_module._erase_node_hooks:
                f(to_erase)

        self._find_nodes_lookup_table.remove(to_erase)

        to_erase._remove_from_list()
        to_erase._erased = True  # iterators may retain handles to erased nodes
        self._len -= 1

        # Null out this Node's argument nodes so that the Nodes referred to
        # can update their ``users`` accordingly
        to_erase._update_args_kwargs(
            map_arg(to_erase._args, lambda n: None),
            map_arg(to_erase._kwargs, lambda n: None),
        )

    @compatibility(is_backward_compatible=True)
    def inserting_before(self, n: Node | None = None) -> _InsertPoint:
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporarily set the insert point and
        then restore it when the with statement exits::

            with g.inserting_before(n):
                ...  # inserting before node n
            ...  # insert point restored to what it was previously
            g.inserting_before(n)  #  set the insert point permanently

        Args:

            n (Optional[Node]): The node before which to insert. If None this will insert before
                the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        if n is None:
            return self.inserting_after(self._root)
        if n.graph != self:
            raise AssertionError("Node to insert before is not in graph.")
        return _InsertPoint(self, n.prepend)

    @compatibility(is_backward_compatible=True)
    def inserting_after(self, n: Node | None = None) -> _InsertPoint:
        """Set the point at which create_node and companion methods will insert into the graph.
        When used within a 'with' statement, this will temporarily set the insert point and
        then restore it when the with statement exits::

            with g.inserting_after(n):
                ...  # inserting after node n
            ...  # insert point restored to what it was previously
            g.inserting_after(n)  #  set the insert point permanently

        Args:

            n (Optional[Node]): The node before which to insert. If None this will insert after
                the beginning of the entire graph.

        Returns:
            A resource manager that will restore the insert point on ``__exit__``.
        """
        if n is None:
            return self.inserting_before(self._root)
        if n.graph != self:
            raise AssertionError("Node to insert after is not in graph.")
        return _InsertPoint(self, n.append)

    @compatibility(is_backward_compatible=True)
    def placeholder(
        self,
        name: str,
        type_expr: Any | None = None,
        default_value: Any = inspect.Signature.empty,
    ) -> Node:
        """
        Insert a ``placeholder`` node into the Graph. A ``placeholder`` represents
        a function input.

        Args:

            name (str): A name for the input value. This corresponds to the name
                of the positional argument to the function this ``Graph`` represents.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have. This is needed in some
                cases for proper code generation (e.g. when the function is used
                subsequently in TorchScript compilation).

            default_value (Any): The default value this function argument should take
                on. NOTE: to allow for `None` as a default value, `inspect.Signature.empty`
                should be passed as this argument to specify that the parameter does _not_
                have a default value.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        args = () if default_value is inspect.Signature.empty else (default_value,)
        return self.create_node("placeholder", name, args=args, type_expr=type_expr)

    @compatibility(is_backward_compatible=True)
    def get_attr(self, qualified_name: str, type_expr: Any | None = None) -> Node:
        """
        Insert a ``get_attr`` node into the Graph. A ``get_attr`` ``Node`` represents the
        fetch of an attribute from the ``Module`` hierarchy.

        Args:

            qualified_name (str): the fully-qualified name of the attribute to be retrieved.
                For example, if the traced Module has a submodule named ``foo``, which has a
                submodule named ``bar``, which has an attribute named ``baz``, the qualified
                name ``foo.bar.baz`` should be passed as ``qualified_name``.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.


        Returns:

            The newly-created and inserted ``get_attr`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """

        def _get_attr_reference_exists(
            mod: torch.nn.Module, qualified_name: str
        ) -> bool:
            module_path, _, name = qualified_name.rpartition(".")

            try:
                submod: torch.nn.Module = mod.get_submodule(module_path)
            except AttributeError:
                warnings.warn(f"Failed to fetch module {module_path}!")
                return False

            if not hasattr(submod, name):
                return False

            res = getattr(submod, name)

            if (
                not isinstance(res, torch.nn.Module)
                and not isinstance(res, torch.nn.Parameter)
                and name not in submod._buffers
            ):
                return False

            return True

        if self.owning_module is not None and not _get_attr_reference_exists(
            self.owning_module, qualified_name
        ):
            warnings.warn(
                "Attempted to insert a get_attr Node with no "
                "underlying reference in the owning "
                "GraphModule! Call "
                "GraphModule.add_submodule to add the "
                "necessary submodule, "
                "GraphModule.add_parameter to add the "
                "necessary Parameter, or "
                "nn.Module.register_buffer to add the "
                "necessary buffer",
                stacklevel=2,
            )
        return self.create_node("get_attr", qualified_name, type_expr=type_expr)

    @compatibility(is_backward_compatible=True)
    def call_module(
        self,
        module_name: str,
        args: tuple[Argument, ...] | None = None,
        kwargs: dict[str, Argument] | None = None,
        type_expr: Any | None = None,
    ) -> Node:
        """
        Insert a ``call_module`` ``Node`` into the ``Graph``. A ``call_module`` node
        represents a call to the forward() function of a ``Module`` in the ``Module``
        hierarchy.

        Args:

            module_name (str): The qualified name of the ``Module`` in the ``Module``
                hierarchy to be called. For example, if the traced ``Module`` has a
                submodule named ``foo``, which has a submodule named ``bar``, the
                qualified name ``foo.bar`` should be passed as ``module_name`` to
                call that module.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this should *not* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly-created and inserted ``call_module`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        if (
            self.owning_module is not None
            and self.owning_module.get_submodule(module_name) is None
        ):
            warnings.warn(
                "Attempted to insert a call_module Node with "
                "no underlying reference in the owning "
                "GraphModule! Call "
                "GraphModule.add_submodule to add the "
                "necessary submodule"
            )
        return self.create_node(
            "call_module", module_name, args, kwargs, type_expr=type_expr
        )

    @compatibility(is_backward_compatible=True)
    def call_method(
        self,
        method_name: str,
        args: tuple[Argument, ...] | None = None,
        kwargs: dict[str, Argument] | None = None,
        type_expr: Any | None = None,
    ) -> Node:
        """
        Insert a ``call_method`` ``Node`` into the ``Graph``. A ``call_method`` node
        represents a call to a given method on the 0th element of ``args``.

        Args:

            method_name (str): The name of the method to apply to the self argument.
                For example, if args[0] is a ``Node`` representing a ``Tensor``,
                then to call ``relu()`` on that ``Tensor``, pass ``relu`` to ``method_name``.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called method. Note that this *should* include a ``self`` argument.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called method

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        Returns:

            The newly created and inserted ``call_method`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node(
            "call_method", method_name, args, kwargs, type_expr=type_expr
        )

    @compatibility(is_backward_compatible=True)
    def call_function(
        self,
        the_function: Callable[..., Any],
        args: tuple[Argument, ...] | None = None,
        kwargs: dict[str, Argument] | None = None,
        type_expr: Any | None = None,
        name: str | None = None,
    ) -> Node:
        """
        Insert a ``call_function`` ``Node`` into the ``Graph``. A ``call_function`` node
        represents a call to a Python callable, specified by ``the_function``.

        Args:

            the_function (Callable[..., Any]): The function to be called. Can be any PyTorch
                operator, Python function, or member of the ``builtins`` or ``operator``
                namespaces.

            args (Optional[Tuple[Argument, ...]]): The positional arguments to be passed
                to the called function.

            kwargs (Optional[Dict[str, Argument]]): The keyword arguments to be passed
                to the called function

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

            name (Optional[str]): The name of the node. If not specified, set to None

        Returns:

            The newly created and inserted ``call_function`` node.

        .. note::
            The same insertion point and type expression rules apply for this method
            as :meth:`Graph.create_node`.
        """
        return self.create_node(
            "call_function", the_function, args, kwargs, name=name, type_expr=type_expr
        )

    @staticmethod
    def _get_tensor_meta_val(tensor_node: Node) -> tuple[Any, str]:
        """Read the example tensor stored on ``tensor_node`` under either
        ``meta['val']`` (export / proxy_tensor convention) or
        ``meta['example_value']`` (dynamo convention).

        Returns ``(value, key)`` where ``key`` is the meta key the value was
        found under (defaulting to ``"val"`` if no meta is present). Callers
        that emit a new node should mirror ``key`` so the new node matches
        the surrounding graph's convention.
        """
        if "val" in tensor_node.meta:
            return tensor_node.meta["val"], "val"
        if "example_value" in tensor_node.meta:
            return tensor_node.meta["example_value"], "example_value"
        return None, "val"

    @compatibility(is_backward_compatible=False)
    def create_size_node(self, tensor_node: Node, dim: int) -> Node:
        """Create an FX node for ``tensor_node.size(dim)``."""
        val, key = self._get_tensor_meta_val(tensor_node)
        node = self.call_function(torch.ops.aten.sym_size.int, (tensor_node, dim))
        if val is not None:
            node.meta[key] = val.size(dim)
        return node

    @compatibility(is_backward_compatible=False)
    def create_stride_node(self, tensor_node: Node, dim: int) -> Node:
        """Create an FX node for ``tensor_node.stride(dim)``."""
        val, key = self._get_tensor_meta_val(tensor_node)
        node = self.call_function(torch.ops.aten.sym_stride.int, (tensor_node, dim))
        if val is not None:
            node.meta[key] = val.stride(dim)
        return node

    @compatibility(is_backward_compatible=False)
    def create_storage_offset_node(self, tensor_node: Node) -> Node:
        """Create an FX node for ``tensor_node.storage_offset()``."""
        val, key = self._get_tensor_meta_val(tensor_node)
        node = self.call_function(
            torch.ops.aten.sym_storage_offset.default, (tensor_node,)
        )
        if val is not None:
            node.meta[key] = val.storage_offset()
        return node

    @compatibility(is_backward_compatible=False)
    def _resolve_unbacked_binding(
        self,
        producer: Node,
        keypath: tuple[object, ...],
        lower_symint: Callable[[torch.SymInt], Node | int],
    ) -> Node:
        """Walk an ``unbacked_bindings`` keypath, emitting FX ops on this
        graph to recover the bound SymInt from ``producer``'s result.

        ``lower_symint`` is invoked for the only keypath component that may
        carry a non-literal value: ``DivideByKey`` with a SymInt divisor.
        Callers must provide one (e.g. a wrapper around their sympy interp
        cache).
        """
        import operator

        from torch.fx.experimental.symbolic_shapes import (
            CallMethodKey,
            ConvertIntKey,
            DivideByKey,
            InnerTensorKey,
        )

        node = producer
        i = 0
        while i < len(keypath):
            k = keypath[i]
            nxt = keypath[i + 1] if i + 1 < len(keypath) else None
            if isinstance(k, CallMethodKey) and isinstance(nxt, pytree.SequenceKey):
                idx = nxt.idx
                if k.name == "size":
                    node = self.create_size_node(node, idx)
                elif k.name == "stride":
                    node = self.create_stride_node(node, idx)
                else:
                    node = self.call_method(k.name, (node, idx))
                i += 2
            elif isinstance(k, CallMethodKey):
                if k.name == "storage_offset":
                    node = self.create_storage_offset_node(node)
                else:
                    node = self.call_method(k.name, (node,))
                i += 1
            elif isinstance(k, pytree.SequenceKey):
                node = self.call_function(operator.getitem, (node, k.idx))
                i += 1
            elif isinstance(k, ConvertIntKey):
                node = self.call_function(torch.sym_ite, (node, 1, 0))
                i += 1
            elif isinstance(k, DivideByKey):
                divisor = k.divisor
                if isinstance(divisor, torch.SymInt):
                    divisor = lower_symint(divisor)
                node = self.call_function(operator.floordiv, (node, divisor))
                i += 1
            elif isinstance(k, InnerTensorKey):
                node = self.call_function(getattr, (node, k.inner_name))
                i += 1
            else:
                raise AssertionError(f"unrecognized keypath component {k}")
        return node

    @compatibility(is_backward_compatible=False)
    def materialize_symints(
        self, values: Sequence[torch.SymInt | int]
    ) -> list[Node | int]:
        """Materialize a list of ``SymInt``/``int`` values as FX subgraphs rooted
        at existing nodes in this graph whose meta produces the referenced
        symbols (typically SymInt placeholders or other sym ops).

        consider a graph with a tensor placeholder
        ``%x`` of shape ``(s32, s32)`` and we want to record this stride
        as an FX value to pass to a later op, two ways to do it.

        * ``g.create_stride_node(%x, 0)`` emits ``%t = aten.sym_stride.int(%x, 0)``.
          The semantics of this op are "ask ``%x`` for its current stride at
          dim 0". If a later pass (e.g. mkldnn channels-last conversion) mutates
          ``%x``'s layout, re-running ``FakeTensorProp`` will overwrite
          ``%t.meta["val"]`` with the NEW stride. This is the right behavior
          when you want a *live* query on the producer.

        * ``g.materialize_symints([%x.meta["val"].stride(0)])`` walks the sympy
          expression ``s32`` and emits an FX subgraph that recomputes it from
          the existing producer of ``s32`` (here the placeholder ``%x`` itself
          via ``aten.sym_size.int(%x, 0)``, or a SymInt placeholder if one
          exists). The resulting node's value is "what ``s32`` is at runtime"
          -- which is determined by the input's shape and is INDEPENDENT of any
          layout change to ``%x``. This is the right behavior when you want to
          *freeze* the trace-time stride into the graph.

        Note: like other ``Graph`` node-creation APIs (``call_function``,
        ``create_size_node``, etc.), nodes are emitted at the graph's current
        insertion point. The default insertion point (``Graph._root.prepend``)
        appends new nodes to the end of the graph, which on a graph that
        already has an ``output`` node means they land *after* ``return``
        (orphaned). Callers typically scope this in
        ``with graph.inserting_before(graph.output_node()):`` so the new
        nodes land in the body.

        Performance: each call performs an O(graph_size) producer-discovery
        scan and builds a per-call ``expr_to_proxy`` hash-cons cache. Prefer
        batching every SymInt you need to lift into a single call (or as few
        calls as possible) over calling it once per value -- this amortises
        the graph scan and lets symints with shared sub-expressions get
        hash-consed into a single subgraph.
        """
        # Local imports keep sympy off the ``import torch`` path; the helper
        # is only invoked from passes that already depend on symbolic shapes.
        import operator

        import sympy

        from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
        from torch.utils._sympy.reference import PythonReferenceAnalysis

        # TODO: consider caching `expr_to_proxy` / `sym_size_sources` across
        # calls.
        # Read tensor/SymInt metadata with fallback. Edge graphs can have
        # a mixed convention: lifted parameters carry both ``"val"`` and
        # ``"example_value"`` (dynamo-style) while real user inputs carry
        # only ``"val"`` (export-style). Probing the graph for a single
        # key would lock onto one convention and silently miss symbols on
        # the others; reading per-node with fallback handles both.
        def _node_val(n: Node) -> Any:
            val = n.meta.get("val")
            if val is not None:
                return val
            return n.meta.get("example_value")

        def _set_node_val(n: Node, val: Any) -> None:
            """Set ``n.meta['val']`` and, if any of ``n``'s input nodes carry
            ``"example_value"``, mirror it there too so the new node matches
            the surrounding graph's meta-key convention."""
            n.meta["val"] = val
            if any(isinstance(a, Node) and "example_value" in a.meta for a in n.args):
                n.meta["example_value"] = val

        # Build the symbol -> Proxy map once; shared across all inputs so common
        # sub-expressions across symints get hash-consed into single subgraphs.
        tracer = torch.fx.proxy.GraphAppendingTracer(self)
        expr_to_proxy: dict[sympy.Expr, torch.fx.Proxy] = {}

        # Pass 1: walk the graph to discover producers. All backed symbols
        # must be found here -- their only safe producer is an input.
        # Populates:
        #   - ``expr_to_proxy``: SymInt placeholders -> Proxy (direct).
        #   - ``sym_size_sources``: tensor placeholder shape symbol
        #     -> (placeholder, dim, divisor) (lazy, used by Pass 2).
        #     divisor=1 when the placeholder dim IS the bare ``Symbol``;
        #     divisor>1 when the dim is ``Symbol * c`` (the ``Dim("h")*N``
        #     derived-shape pattern). Recovery emits
        #     ``sym_size.int(ph, dim) // divisor``.
        #   - ``sym_to_binding``: unbacked symbol -> (producer node, keypath)
        #     from ``node.meta["unbacked_bindings"]`` (lazy, used by Pass 2).
        sym_size_sources: dict[sympy.Symbol, tuple[Node, int, int]] = {}
        sym_to_binding: dict[sympy.Symbol, tuple[Node, tuple[object, ...]]] = {}

        def _record_shape_source(expr: sympy.Expr, node: Node, dim: int) -> None:
            """Register ``(node, dim)`` as a recoverable source for ``expr``.

            - bare ``Symbol``: divisor = 1.
            - ``Symbol * c`` (positive integer constant): divisor = c.
              Recovery is ``sym_size.int(ph, dim) // c``.
            - anything else: skipped (general inverse needs sympy.solve).
            """
            if isinstance(expr, sympy.Symbol):
                sym_size_sources.setdefault(expr, (node, dim, 1))
                return
            if isinstance(expr, sympy.Mul) and len(expr.args) == 2:
                a, b = expr.args
                if isinstance(b, sympy.Symbol) and isinstance(a, sympy.Integer):
                    a, b = b, a  # normalize to (Symbol, Integer) order
                if (
                    isinstance(a, sympy.Symbol)
                    and isinstance(b, sympy.Integer)
                    and int(b) > 0
                ):
                    sym_size_sources.setdefault(a, (node, dim, int(b)))

        for node in self.nodes:
            if node.op == "placeholder":
                val = _node_val(node)
                if isinstance(val, py_sym_types):
                    expr = val.node.expr
                    if isinstance(expr, sympy.Symbol) and expr not in expr_to_proxy:
                        expr_to_proxy[expr] = torch.fx.Proxy(node, tracer=tracer)
                elif isinstance(val, torch.Tensor):
                    for dim, s in enumerate(val.shape):
                        if isinstance(s, torch.SymInt):
                            _record_shape_source(s.node.expr, node, dim)
            else:
                bindings = node.meta.get("unbacked_bindings")
                if bindings:
                    for sym, keypath in bindings.items():
                        sym_to_binding.setdefault(sym, (node, tuple(keypath)))

        from sympy.logic.boolalg import BooleanAtom

        from torch.fx.experimental.symbolic_shapes import (
            DivideByKey,
            free_unbacked_symbols,
        )

        def _arg_meta_val(a: Any) -> Any:
            if isinstance(a, torch.fx.Node):
                return _node_val(a)
            return a

        def _sympy_interp_cached(expr: sympy.Expr) -> Any:
            # Common expressions cached in expr_to_proxy
            if expr in expr_to_proxy:
                return expr_to_proxy[expr]
            if isinstance(
                expr, (sympy.Integer, sympy.Number, sympy.Symbol, BooleanAtom)
            ):
                # handle leaf terms.
                return sympy_interp(PythonReferenceAnalysis, expr_to_proxy, expr)
            # handle composite expressions.
            result = _run_sympy_handler(
                PythonReferenceAnalysis,
                [_sympy_interp_cached(arg) for arg in expr.args],
                expr,
            )
            if not isinstance(result, torch.fx.Proxy):
                return result
            expr_to_proxy[expr] = result
            node = result.node
            target = node.target
            if not callable(target):
                return result
            try:
                fake_args = tuple(_arg_meta_val(a) for a in node.args)
                fake_kwargs = {k: _arg_meta_val(v) for k, v in node.kwargs.items()}
                _set_node_val(node, target(*fake_args, **fake_kwargs))
            except (TypeError, ValueError, AttributeError) as e:
                log.debug(
                    "materialize_symints: skipping meta annotation for %s: %s",
                    expr,
                    e,
                )
            return result

        def _lower_symint_divisor(d: torch.SymInt) -> Node | int:
            r = _sympy_interp_cached(d.node.expr)
            return r.node if isinstance(r, torch.fx.Proxy) else r

        # Pass 2 (lazy, recursive): materialize a producer for each needed
        # symbol. Three cases:
        #   - already in expr_to_proxy: nothing to do.
        #   - in sym_size_sources: emit sym_size.int for shape symbols
        #     recoverable from a tensor placeholder shape.
        #   - in sym_to_binding: fallback for missing unbacked symbols. Use
        #     the node.meta["unbacked_bindings"] map -- the authoritative
        #     source for "this node produces these unbacked symbols". The
        #     keypath tells us how to recover the SymInt from the node's
        #     result (e.g. .size(i), .stride(i), storage_offset()).
        def _ensure_produced(sym: sympy.Symbol) -> None:
            if sym in expr_to_proxy:
                return
            if sym in sym_size_sources:
                ph, dim, divisor = sym_size_sources[sym]
                sym_node = self.call_function(torch.ops.aten.sym_size.int, (ph, dim))
                tensor = _node_val(ph)
                if isinstance(tensor, torch.Tensor):
                    _set_node_val(sym_node, tensor.shape[dim])
                if divisor != 1:
                    # Placeholder dim was ``sym * divisor`` (e.g. from
                    # ``Dim("h") * N``); recover the underlying symbol as
                    # ``sym_size.int(ph, dim) // divisor``.
                    div_node = self.call_function(
                        operator.floordiv, (sym_node, divisor)
                    )
                    if isinstance(tensor, torch.Tensor):
                        _set_node_val(div_node, tensor.shape[dim] // divisor)
                    expr_to_proxy[sym] = torch.fx.Proxy(div_node, tracer=tracer)
                else:
                    expr_to_proxy[sym] = torch.fx.Proxy(sym_node, tracer=tracer)
                return
            if sym in sym_to_binding:
                if not free_unbacked_symbols(sym):
                    raise AssertionError(
                        f"materialize_symints: backed symbol {sym} has no "
                        f"input producer (backed symbols cannot be recovered "
                        f"from non-placeholder nodes)"
                    )
                producer_node, keypath = sym_to_binding[sym]
                for k in keypath:
                    if isinstance(k, DivideByKey) and isinstance(
                        k.divisor, torch.SymInt
                    ):
                        for s2 in k.divisor.node.expr.free_symbols:
                            _ensure_produced(s2)
                producer = self._resolve_unbacked_binding(
                    producer_node, keypath, lower_symint=_lower_symint_divisor
                )
                expr_to_proxy[sym] = torch.fx.Proxy(producer, tracer=tracer)
                return
            raise AssertionError(f"materialize_symints: no producer for {sym}")

        out: list[Node | int] = []
        for s in values:
            if isinstance(s, torch.SymInt):
                pass
            elif isinstance(s, int):  # also covers bool (subclass of int)
                out.append(s)
                continue
            else:
                raise TypeError(
                    f"materialize_symints: expected SymInt or int, got "
                    f"{type(s).__name__} ({s!r})"
                )
            target_expr = s.node.expr
            if target_expr.is_number:
                out.append(int(s))
                continue
            for sym in target_expr.free_symbols:
                _ensure_produced(sym)
            result = _sympy_interp_cached(target_expr)
            if not isinstance(result, torch.fx.Proxy):
                # ``target_expr`` was non-constant per sympy but our interpreter
                # still folded it to a Python int/bool. Unwrap to that constant.
                if not isinstance(result, (int, bool)):
                    raise AssertionError(
                        f"materialize_symints: non-Proxy result {result!r} for "
                        f"non-constant target {target_expr!r}"
                    )
                out.append(int(result))
                continue
            out_node = result.node
            _set_node_val(out_node, s)
            out.append(out_node)
        return out

    @compatibility(is_backward_compatible=False)
    def materialize_symint(self, value: torch.SymInt | int) -> Node | int:
        """Single-value convenience wrapper around :meth:`materialize_symints`."""
        return self.materialize_symints([value])[0]

    @compatibility(is_backward_compatible=True)
    def node_copy(
        self, node: Node, arg_transform: Callable[[Node], Argument] = lambda x: x
    ) -> Node:
        """
        Copy a node from one graph into another. ``arg_transform`` needs to transform arguments from
        the graph of node to the graph of self. Example::

            # Copying all the nodes in `g` into `new_graph`
            g: torch.fx.Graph = ...
            new_graph = torch.fx.graph()
            value_remap = {}
            for node in g.nodes:
                value_remap[node] = new_graph.node_copy(node, lambda n: value_remap[n])

        Args:

            node (Node): The node to copy into ``self``.

            arg_transform (Callable[[Node], Argument]): A function that transforms
                ``Node`` arguments in node's ``args`` and ``kwargs`` into the
                equivalent argument in ``self``. In the simplest case, this should
                retrieve a value out of a table mapping Nodes in the original
                graph to ``self``.
        """
        args = map_arg(node.args, arg_transform)
        kwargs = map_arg(node.kwargs, arg_transform)
        if not isinstance(args, tuple):
            raise AssertionError(f"Expected args to be tuple, got {type(args)}")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"Expected kwargs to be dict, got {type(kwargs)}")
        result_node = self.create_node(
            node.op, node.target, args, kwargs, node.name, node.type
        )
        result_node.meta = copy.copy(node.meta)
        return result_node

    @compatibility(is_backward_compatible=True)
    # TODO: should return Node, see _node_list.__iter__ comment
    def output(  # pyrefly: ignore[unannotated-return]
        self, result: Argument, type_expr: Any | None = None
    ):
        """
        Insert an ``output`` ``Node`` into the ``Graph``. An ``output`` node represents
        a ``return`` statement in Python code. ``result`` is the value that should
        be returned.

        Args:

            result (Argument): The value to be returned.

            type_expr (Optional[Any]): an optional type annotation representing the
                Python type the output of this node will have.

        .. note::

            The same insertion point and type expression rules apply for this method
            as ``Graph.create_node``.
        """
        return self.create_node(
            op="output", target="output", args=(result,), type_expr=type_expr
        )

    def _target_to_str(self, target: Target | None) -> str:
        if callable(target):
            op = target.__name__
        else:
            if not isinstance(target, str):
                raise AssertionError(f"Expected target to be str, got {type(target)}")
            op = target
            if _is_magic(op):
                op = op[2:-2]
        op = _snake_case(op)
        return op

    @compatibility(is_backward_compatible=True)
    def python_code(
        self,
        root_module: str,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        expanded_def: bool = False,
        record_func: bool = False,
        additional_meta: list[str] | None = None,
    ) -> PythonCode:
        """
        Turn this ``Graph`` into valid Python code.

        Args:

            root_module (str): The name of the root module on which to look-up
                qualified name targets. This is usually 'self'.

        Returns:

            A PythonCode object, consisting of two fields:
                src: the Python source code representing the object
                globals: a dictionary of global names in `src` -> the objects that they reference.
        """
        # NOTE: [Graph Namespaces]
        #
        # There are two types of symbols in generated Python source code:
        # locals and globals.
        #   Locals are locally defined by the output of a node in the Graph.
        #   Globals are references to external objects, like functions or types.
        #
        # When generating Python code, we need to make sure to name things
        # appropriately. In particular:
        # - All names should be unique, to avoid weird shadowing bugs.
        # - These names need to be consistent, e.g. an object should always be
        #   referenced by the same name.
        #
        # To do this, we create a new namespace just for this source. All names
        # that get printed must come from this namespace.
        #
        # Why can't we reuse node.name? Because it was generated within the
        # namespace `self._graph_namespace`. In order to provide uniqueness
        # over both locals (node.name) *and* globals, we create a completely
        # new namespace to put all identifiers in.
        namespace = _Namespace()

        # Override Node's repr to generate a valid name within our namespace.
        # Since repr() is designed to produce a valid Python expression, it
        # makes sense to reuse it. This way, it's easy to print something like
        # Tuple[Node, Node] by simply calling repr() on it. Node's __repr__ is
        # implemented cooperatively to allow this.
        def node_repr(n: Node) -> str:
            return namespace.create_name(n.name, n)

        @contextmanager
        def override_node_repr(graph: Graph) -> Generator[None, None, None]:
            orig_repr_fns = {}
            for node in graph.nodes:
                orig_repr_fns[node] = node._repr_fn
                node._repr_fn = node_repr
            try:
                yield None
            finally:
                # restore the original repr functions
                for node in graph.nodes:
                    node._repr_fn = orig_repr_fns[node]

        with override_node_repr(self):
            return self._python_code(
                root_module,
                namespace,
                verbose=verbose,
                include_stride=include_stride,
                include_device=include_device,
                colored=colored,
                expanded_def=expanded_def,
                record_func=record_func,
                additional_meta=additional_meta,
            )

    def _python_code(
        self,
        root_module: str,
        namespace: _Namespace,
        *,
        verbose: bool = False,
        include_stride: bool = False,
        include_device: bool = False,
        colored: bool = False,
        expanded_def: bool = False,
        record_func: bool = False,
        additional_meta: list[str] | None = None,
    ) -> PythonCode:
        return self._codegen._gen_python_code(
            self.nodes,
            root_module,
            namespace,
            verbose=verbose,
            include_stride=include_stride,
            include_device=include_device,
            colored=colored,
            expanded_def=expanded_def,
            record_func=record_func,
            additional_meta=additional_meta,
        )

    def __str__(self) -> str:
        """
        Return a human-readable (not machine-readable) string representation
        of this Graph
        """
        placeholder_names: list[str] = []
        # This is a one-element array just so ``format_node`` can modify the closed
        # over value
        maybe_return_typename: list[str] = [""]

        node_strs = [node.format_node(placeholder_names) for node in self.nodes]
        param_str = ", ".join(placeholder_names)
        s = f"graph({param_str}){maybe_return_typename[0]}:"
        for node_str in node_strs:
            if node_str:
                s += "\n    " + node_str
        return s

    @compatibility(is_backward_compatible=True)
    def print_tabular(self) -> None:
        """
        Prints the intermediate representation of the graph in tabular
        format. Note that this API requires the ``tabulate`` module to be
        installed.
        """
        try:
            from tabulate import tabulate  # pyrefly: ignore[missing-import]
        except ImportError:
            print(
                "`print_tabular` relies on the library `tabulate`, "
                "which could not be found on this machine. Run `pip "
                "install tabulate` to install the library."
            )
            raise

        node_specs = [[n.op, n.name, n.target, n.args, n.kwargs] for n in self.nodes]
        print(
            tabulate(node_specs, headers=["opcode", "name", "target", "args", "kwargs"])
        )

    @compatibility(is_backward_compatible=True)
    def lint(self) -> None:
        """
        Runs various checks on this Graph to make sure it is well-formed. In
        particular:
        - Checks Nodes have correct ownership (owned by this graph)
        - Checks Nodes appear in topological order
        - If this Graph has an owning GraphModule, checks that targets
        exist in that GraphModule
        """

        # Check topo order
        def check_arg(arg: Node, n: Node | None = None) -> None:
            context_str = f" of Node '{n}' " if n else " "
            if arg.graph is not self:
                raise RuntimeError(
                    f"Argument '{arg}'{context_str}does not belong to this Graph, "
                    f"but was used as an argument! If you are copying nodes from another graph, make "
                    f"sure to use ``arg_transform`` on node_copy() to remap values\n{self}"
                )
            if arg not in seen_values:
                raise RuntimeError(
                    f"Argument '{arg}'{context_str}was used before it has been "
                    f"defined! Please check that Nodes in the graph are topologically ordered\n{self}"
                )

        seen_names: set[str] = set()
        seen_values: set[Node] = set()
        for node in self.nodes:
            if node.op not in _legal_ops:
                raise RuntimeError(f"Node {node} had unknown opcode {node.op}!")
            if node.graph is not self:
                raise RuntimeError(f"Node '{node}' does not belong to this Graph!")
            if node not in self._find_nodes_lookup_table:
                raise RuntimeError(f"Node '{node}' is not added to the side table")
            for arg in node._input_nodes:
                check_arg(arg, node)
            seen_values.add(node)

            if node.name in seen_names:
                raise RuntimeError(f"Node redefined name {node.name}!")
            seen_names.add(node.name)

        # Check targets are legit
        if self.owning_module:
            for node in self.nodes:
                if node.op == "call_function":
                    if not callable(node.target):
                        raise ValueError(
                            f"Node {node} target {node.target} has type {torch.typename(node.target)} but "
                            "a Callable is expected"
                        )
                else:
                    if not isinstance(node.target, str):
                        raise ValueError(
                            f"Node {node} target {node.target} has type {torch.typename(node.target)} but "
                            "a str is expected"
                        )
                if node.op in ["get_attr", "call_module"]:
                    # pyrefly: ignore [missing-attribute]
                    target_atoms = node.target.split(".")
                    m_itr = self.owning_module
                    for i, atom in enumerate(target_atoms):
                        new_m_itr = getattr(m_itr, atom, None)
                        seen_qualname = ".".join(target_atoms[:i])
                        if new_m_itr is None:
                            raise RuntimeError(
                                f"Node {node} target {node.target} references nonexistent attribute "
                                f"{atom} of {seen_qualname}"
                            )
                        if node.op == "call_module" and not isinstance(
                            new_m_itr, torch.nn.Module
                        ):
                            raise RuntimeError(
                                f"Node {node} target {node.target} {atom} of {seen_qualname} does "
                                "not reference an nn.Module"
                            )

                        m_itr = new_m_itr

    @compatibility(is_backward_compatible=True)
    def eliminate_dead_code(
        self, is_impure_node: Callable[[Node], bool] | None = None
    ) -> bool:
        """
        Remove all dead code from the graph, based on each node's number of
        users, and whether the nodes have any side effects. The graph must be
        topologically sorted before calling.

        Args:
            is_impure_node (Optional[Callable[[Node], bool]]): A function that returns
            whether a node is impure. If this is None, then the default behavior is to
            use Node.is_impure.

        Returns:
          bool: Whether the graph was changed as a result of the pass.

        Example:

        Before dead code is eliminated, `a` from `a = x + 1` below has no users
        and thus can be eliminated from the graph without having an effect.

        .. code-block:: python

            def forward(self, x):
                a = x + 1
                return x + self.attr_1

        After dead code is eliminated, `a = x + 1` has been removed, and the rest
        of `forward` remains.

        .. code-block:: python

            def forward(self, x):
                return x + self.attr_1

        .. warning::

            Dead code elimination has some heuristics to avoid removing
            side-effectful nodes (see Node.is_impure) but in general coverage
            is very bad, so you should assume that this method is not sound
            to call unless you know that your FX graph consists entirely
            of functional operations or you supply your own custom
            function for detecting side-effectful nodes.
        """
        from torch.utils._ordered_set import OrderedSet

        # Lint the graph first to make sure its topologically sorted, otherwise
        # DCE below will not behave as expected.
        self.lint()

        impure_random = True
        if torch._guards.TracingContext.try_get():
            impure_random = torch._inductor.config.fallback_random

        def has_side_effect(node: Node) -> bool:
            if is_impure_node is not None:
                return is_impure_node(node)
            return node.is_impure(impure_random)

        # Reverse iterate so that when we remove a node, any nodes used as an
        # input to that node have an updated user count that no longer reflects
        # the removed node.
        removed_nodes = set()
        for node in reversed(self.nodes):
            if not has_side_effect(node) and len(node.users) == 0:
                self.erase_node(node)
                removed_nodes.add(node.name)

        changed = len(removed_nodes) > 0
        if changed:
            log.info("The following nodes were dead code eliminated: %s", removed_nodes)

        # Call DCE on the subgraphs
        if self.owning_module is not None:
            subgraph_names = OrderedSet(
                x.target for x in self.find_nodes(op="get_attr")
            )
            for child_name, child_module in self.owning_module.named_children():
                # Sometimes an owning_module can have unused children. Skip them
                # by checking them from get_attr node targets.
                if child_name in subgraph_names and isinstance(
                    child_module, torch.fx.GraphModule
                ):
                    changed |= child_module.graph.eliminate_dead_code()
                    child_module.recompile()

        return changed

    @compatibility(is_backward_compatible=False)
    def set_codegen(self, codegen: CodeGen) -> None:
        self._codegen = codegen

    @compatibility(is_backward_compatible=False)
    def on_generate_code(
        self,
        make_transformer: Callable[[TransformCodeFunc | None], TransformCodeFunc],
    ) -> contextlib.AbstractContextManager[None]:
        """Register a transformer function when python code is generated

        Args:
            make_transformer (Callable[[Optional[TransformCodeFunc]], TransformCodeFunc]):
                a function that returns a code transformer to be registered.
                This function is called by `on_generate_code` to obtain the
                code transformer.

                This function is also given as its input the currently
                registered code transformer (or None if nothing is registered),
                in case it is not desirable to overwrite it. This is useful to
                chain code transformers together.

        Returns:
            a context manager that when used in a `with` statement, to automatically
            restore the previously registered code transformer.

        Example:

        .. code-block:: python


            gm: fx.GraphModule = ...


            # This is a code transformer we want to register. This code
            # transformer prepends a pdb import and trace statement at the very
            # beginning of the generated torch.fx code to allow for manual
            # debugging with the PDB library.
            def insert_pdb(body):
                return ["import pdb; pdb.set_trace()\\n", *body]


            # Registers `insert_pdb`, and overwrites the current registered
            # code transformer (given by `_` to the lambda):
            gm.graph.on_generate_code(lambda _: insert_pdb)

            # Or alternatively, registers a code transformer which first
            # runs `body` through existing registered transformer, then
            # through `insert_pdb`:
            gm.graph.on_generate_code(
                lambda current_trans: (
                    lambda body: insert_pdb(
                        current_trans(body) if current_trans else body
                    )
                )
            )

            gm.recompile()
            gm(*inputs)  # drops into pdb


        This function can also be used as a context manager, with the benefit to
        automatically restores the previously registered code transformer:

        .. code-block:: python

            # ... continue from previous example

            with gm.graph.on_generate_code(lambda _: insert_pdb):
                # do more stuff with `gm`...
                gm.recompile()
                gm(*inputs)  # drops into pdb

            # now previous code transformer is restored (but `gm`'s code with pdb
            # remains - that means you can run `gm` with pdb here too, until you
            # run next `recompile()`).
        """
        on_gen_code_old = self._codegen._body_transformer
        self._codegen._body_transformer = make_transformer(on_gen_code_old)

        @contextlib.contextmanager
        def on_generate_code_context_manager() -> Generator[None, None, None]:
            try:
                yield
            finally:
                self._codegen._body_transformer = on_gen_code_old

        return on_generate_code_context_manager()

    def _clear_nodes(self) -> None:
        for node in reversed(self.nodes):
            node.meta.clear()
            self.erase_node(node)


@contextmanager
def _override_sym_repr(
    override: Callable[[torch.types.PySymType], str],
) -> Generator[None, None, None]:
    tmp = CodeGen._sym_repr
    try:
        CodeGen._sym_repr = override
        yield
    finally:
        CodeGen._sym_repr = tmp


def _identity(x: str) -> str:
    return x


def _make_color_fn(code: str) -> Callable[[str], str]:
    def f(s: str) -> str:
        reset = "\033[0m"
        return f"{code}{s}{reset}"

    return f


_color_codes = {
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "red": "\033[31m",
    "dim": "\033[2m",
    "dim_blue": "\033[2m\033[34m",
    "dim_green": "\033[2m\033[32m",
}
_color_fns = {k: _make_color_fn(v) for k, v in _color_codes.items()}
_counter_regexp = re.compile(r"# COUNTER: (\d+)")


reflectable_magic_methods = {
    "add": "{} + {}",
    "sub": "{} - {}",
    "mul": "{} * {}",
    "floordiv": "{} // {}",
    "truediv": "{} / {}",
    "div": "{} / {}",
    "mod": "{} % {}",
    "pow": "{} ** {}",
    "lshift": "{} << {}",
    "rshift": "{} >> {}",
    "and_": "{} & {}",
    "or_": "{} | {}",
    "xor": "{} ^ {}",
    "getitem": "{}[{}]",
    "matmul": "{} @ {}",
}

magic_methods = {
    "eq": "{} == {}",
    "ne": "{} != {}",
    "lt": "{} < {}",
    "gt": "{} > {}",
    "le": "{} <= {}",
    "ge": "{} >= {}",
    "pos": "+{}",
    "neg": "-{}",
    "invert": "~{}",
    **reflectable_magic_methods,
}

inplace_methods = {
    "iadd": "{} += {}",
    "iand": "{} &= {}",
    "ifloordiv": "{} //= {}",
    "ilshift": "{} <<= {}",
    "imod": "{} %= {}",
    "imul": "{} *= {}",
    "imatmul": "{} @= {}",
    "ior": "{} |= {}",
    "ipow": "{} **= {}",
    "irshift": "{} >>= {}",
    "isub": "{} -= {}",
    "itruediv": "{} /= {}",
    "ixor": "{} ^= {}",
    "setitem": "{}[{}] = {}",
}
<|endoftext|>import enum
import inspect
import numbers
import types
import typing
import warnings
from collections.abc import Callable
from typing import Any, cast, Literal, NamedTuple, overload, TYPE_CHECKING

import torch
from torch._jit_internal import boolean_dispatched
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._inspect import _fast_bind

from ._compatibility import compatibility


if TYPE_CHECKING:
    from .node import Argument

__all__ = [
    "ArgsKwargsPair",
    "check_for_mutable_operation",
    "get_signature_for_torch_op",
    "create_type_hint",
    "type_matches",
    "normalize_function",
    "normalize_module",
]


@compatibility(is_backward_compatible=False)
class ArgsKwargsPair(NamedTuple):
    """
    Simple named tuple for wrapping args/kwargs pairs.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


_manual_overrides: dict[Callable[..., Any], list[inspect.Signature]] = {}


def _nonzero_schemas() -> list[inspect.Signature]:
    signatures = []

    def nonzero(self: torch.Tensor) -> None:
        pass

    signatures.append(inspect.signature(nonzero))

    def nonzero(self: torch.Tensor, *, as_tuple: bool) -> None:  # type: ignore[no-redef]
        pass

    signatures.append(inspect.signature(nonzero))

    return signatures


_manual_overrides[torch.nonzero] = _nonzero_schemas()


class _FakeGlobalNamespace:
    def __getattr__(self, name: str) -> types.ModuleType:
        if name == "torch":
            return torch
        raise RuntimeError("Expected a torch namespace lookup")


_type_eval_globals = {
    "Tensor": torch.Tensor,
    "Device": torch.device,
    "Layout": torch.layout,
    "number": numbers.Number,
    "Future": torch.jit.Future,
    "AnyEnumType": enum.Enum,
    "QScheme": torch.qscheme,
    "__torch__": _FakeGlobalNamespace(),
    "NoneType": type(None),
    "Storage": torch.UntypedStorage,
    "t": typing.TypeVar("t"),
    "PyObject": Any,
}
for k in dir(typing):
    _type_eval_globals[k] = getattr(typing, k)


def _torchscript_type_to_python_type(ts_type: "torch._C.JitType") -> Any:
    """
    Convert a TorchScript type to a Python type (including subtypes) via
    eval'ing the annotation_str. _type_eval_globals sets up expressions
    like "List" and "Future" to map to actual types (typing.List and jit.Future)
    """
    return eval(ts_type.annotation_str, _type_eval_globals)


def _torchscript_schema_to_signature_impl(
    ts_schema: torch._C.FunctionSchema,
) -> inspect.Signature:
    from inspect import Parameter

    parameters: list[Parameter] = []
    for arg in ts_schema.arguments:
        arg_type = _torchscript_type_to_python_type(arg.type)
        default = arg.default_value if arg.has_default_value() else Parameter.empty
        # TODO: Figure out if this is safe. It seems like when generating the type signatures for
        # PythonArgParser, we emit signatures with `input` instead of `self` as the first tensor
        # argument name. Downstream, if someone converts that positional argument to a keyword
        # argument, the name mismatch will break things, so here we're going to normalize the
        # name to "input"
        name = arg.name if arg.name != "self" else "input"
        kind = (
            Parameter.KEYWORD_ONLY
            if arg.kwarg_only
            else Parameter.POSITIONAL_OR_KEYWORD
        )
        # "from" is a keyword therefore it must be a POSITIONAL_ONLY argument
        if name == "from":
            if kind != Parameter.POSITIONAL_OR_KEYWORD:
                raise AssertionError(f"Expected POSITIONAL_OR_KEYWORD, got {kind}")
            # ParameterKind type is internal implementation detail to inspect package
            # which makes it hard to do type annotation
            kind = Parameter.POSITIONAL_ONLY  # type: ignore[assignment]
            # This renders all previous arguments to positional only

            for idx, p in enumerate(parameters):
                if p.kind != Parameter.POSITIONAL_OR_KEYWORD:
                    raise AssertionError(
                        f"Expected POSITIONAL_OR_KEYWORD for param {p.name}, got {p.kind}"
                    )
                parameters[idx] = Parameter(
                    name=p.name,
                    kind=Parameter.POSITIONAL_ONLY,
                    default=p.default,
                    annotation=p.annotation,
                )

        parameters.append(
            Parameter(name=name, kind=kind, default=default, annotation=arg_type)
        )
    return_types = [
        _torchscript_type_to_python_type(ret.type) for ret in ts_schema.returns
    ]
    if len(return_types) == 0:
        return_type = None
    elif len(return_types) == 1:
        return_type = return_types[0]
    else:
        return_type = tuple.__class_getitem__(tuple(return_types))

    return inspect.Signature(parameters, return_annotation=return_type)


_SCHEMA_TO_SIGNATURE_CACHE: dict[tuple[str, str], inspect.Signature] = {}


def _torchscript_schema_to_signature(
    ts_schema: torch._C.FunctionSchema,
) -> inspect.Signature:
    # Cached as it's called in the hot path of FakeTensor dispatch
    cache_key = ts_schema.name, ts_schema.overload_name
    cache_val = _SCHEMA_TO_SIGNATURE_CACHE.get(cache_key)
    if cache_val is not None:
        return cache_val

    res = _torchscript_schema_to_signature_impl(ts_schema)
    _SCHEMA_TO_SIGNATURE_CACHE[cache_key] = res
    return res


@compatibility(is_backward_compatible=False)
def check_for_mutable_operation(
    target: Callable[..., Any],
    args: tuple["Argument", ...],
    kwargs: dict[str, "Argument"],
) -> None:
    signatures, schemas = get_signature_for_torch_op(target, return_schemas=True)

    if signatures and schemas:
        matched_schemas: list[tuple[inspect.Signature, torch._C.FunctionSchema]] = []

        # Iterate through all of the schema until we find one that matches
        # If one matches, populate `new_args_and_kwargs` with the new args/kwargs
        # values. If none matches, `new_args_and_kwargs` will be None
        for candidate_signature, schema in zip(signatures, schemas):
            try:
                _fast_bind(candidate_signature, *args, **kwargs)
                matched_schemas.append((candidate_signature, schema))
            except TypeError:
                continue

        def throw_if_mutable(schema: torch._C.FunctionSchema) -> None:
            if schema.is_mutable:
                raise RuntimeError(
                    f"Tried to trace mutable operation {schema}. FX only supports functional "
                    f"code, so operations that mutate operands in-place (e.g. via `out` arguments) "
                    f"are not supported"
                )

        if len(matched_schemas) == 0:
            # Did not match any schema. Cannot check for mutation
            pass
        elif len(matched_schemas) == 1:
            # Matched exactly one schema, unambiguous
            _, schema_to_check = matched_schemas[0]
            throw_if_mutable(schema_to_check)
        else:
            # Ambiguous schema match. Since mutability checking is best effort,
            # do nothing.
            pass


@overload
def get_signature_for_torch_op(
    op: Callable[..., Any], return_schemas: Literal[True]
) -> tuple[list[inspect.Signature] | None, list[torch._C.FunctionSchema] | None]: ...


@overload
def get_signature_for_torch_op(
    op: Callable[..., Any], return_schemas: Literal[False] = ...
) -> list[inspect.Signature] | None: ...


@compatibility(is_backward_compatible=False)
def get_signature_for_torch_op(
    op: Callable[..., Any], return_schemas: bool = False
) -> (
    list[inspect.Signature]
    | tuple[list[inspect.Signature] | None, list[torch._C.FunctionSchema] | None]
    | None
):
    """
    Given an operator on the `torch` namespace, return a list of `inspect.Signature`
    objects corresponding to the overloads of that op. May return `None` if a signature
    could not be retrieved.

    Args:
        op (Callable): An operator on the `torch` namespace to look up a signature for

    Returns:
        Optional[List[inspect.Signature]]: A list of signatures for the overloads of this
            operator, or None if the operator signatures could not be retrieved. If
            return_schemas=True, returns a tuple containing the optional Python signatures
            and the optional TorchScript Function signature
    """
    if isinstance(op, OpOverload):
        schemas = [op._schema]
    elif isinstance(op, OpOverloadPacket):
        schemas = [overload._schema for overload in op.op_overloads()]
    else:
        override = _manual_overrides.get(op)
        if override:
            return (override, None) if return_schemas else None

        aten_fn = torch.jit._builtins._find_builtin(op)

        if aten_fn is None:
            return (None, None) if return_schemas else None
        schemas = torch._C._jit_get_schemas_for_operator(aten_fn)

    signatures = [_torchscript_schema_to_signature(schema) for schema in schemas]
    return (signatures, schemas) if return_schemas else signatures


@compatibility(is_backward_compatible=False)
def create_type_hint(x: object) -> object:
    """
    Produces a type hint for the given argument.

    The :func:`create_type_hint` looks for a type hint compatible with the input argument `x`.

    If `x` is a `list` or `tuple`, it looks for an object in the list whose type is a superclass
    of the rest, and uses that as `base_type` for the `List` or `Tuple` to be returned.
    If no such object is found, it defaults to `List[Any]`.

    If `x` is neither a `list` nor a `tuple`, it returns `x`.
    """
    try:
        if isinstance(x, (list, tuple)):
            # todo(chilli): Figure out the right way for mypy to handle this
            if isinstance(x, list):

                def ret_type(x: Any) -> Any:
                    return list[x]  # type: ignore[valid-type]

            else:

                def ret_type(x: Any) -> Any:
                    return tuple[x, ...]  # type: ignore[valid-type]

            if len(x) == 0:
                return ret_type(Any)
            base_type = x[0]
            for t in x:
                if issubclass(t, base_type):
                    continue
                elif issubclass(base_type, t):
                    base_type = t
                else:
                    return ret_type(Any)
            return ret_type(base_type)
    except Exception:
        # We tried to create a type hint for list but failed.
        warnings.warn(
            f"We were not able to successfully create type hint from the type {x}"
        )
    return x


@compatibility(is_backward_compatible=False)
def type_matches(signature_type: Any, argument_type: Any) -> bool:
    sig_origin_type = getattr(signature_type, "__origin__", signature_type)

    if signature_type is argument_type:
        return True

    # Union types in signature. Given type needs to match one of the
    # contained types in the Union
    if sig_origin_type is typing.Union and signature_type != argument_type:
        sig_contained = typing.get_args(signature_type)
        return any(type_matches(c, argument_type) for c in sig_contained)

    if getattr(signature_type, "__origin__", None) is list:
        sig_el_type = typing.get_args(signature_type)[0]

        # int can be promoted to list[int]
        if argument_type is int and sig_el_type is int:
            return True

        if not inspect.isclass(sig_el_type):
            warnings.warn(
                f"Does not support nested parametric types, got {signature_type}. Please file a bug."
            )
            return False
        if getattr(argument_type, "__origin__", None) is list:
            return issubclass(typing.get_args(argument_type)[0], sig_el_type)

        def is_homogeneous_tuple(t: object) -> bool:
            if typing.get_origin(t) is not tuple:
                return False
            contained = typing.get_args(t)
            if contained == ((),):  # Tuple[()].__args__ == ((),) for some reason
                return True
            return all((c is Ellipsis) or issubclass(c, sig_el_type) for c in contained)

        # Tuple[T] is accepted for List[T] parameters
        return is_homogeneous_tuple(argument_type)

    # Dtype is an int in schemas
    if signature_type is int and argument_type is torch.dtype:
        return True

    if signature_type is numbers.Number and argument_type in {int, float}:
        return True
    if inspect.isclass(argument_type) and inspect.isclass(signature_type):
        return issubclass(argument_type, signature_type)

    return False


@compatibility(is_backward_compatible=False)
def _normalize_function_or_error(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    arg_types: tuple[Any] | None = None,
    kwarg_types: dict[str, Any] | None = None,
    normalize_to_only_use_kwargs: bool = False,
) -> ArgsKwargsPair:
    """
    Wrapper around normalize_function that never returns None, but
    loudly errors instead
    """
    res = normalize_function(
        target, args, kwargs, arg_types, kwarg_types, normalize_to_only_use_kwargs
    )
    if res is None:
        raise RuntimeError(
            f"Failed to normalize function {target} with args {args} and kwargs {kwargs}"
        )
    else:
        return res


@compatibility(is_backward_compatible=False)
def normalize_function(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    arg_types: tuple[Any] | None = None,
    kwarg_types: dict[str, Any] | None = None,
    normalize_to_only_use_kwargs: bool = False,
) -> ArgsKwargsPair | None:
    """
    Returns normalized arguments to PyTorch functions. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (``*args``, ``**kwargs``). Does not support modules.

    May require `arg_types` and `kwarg_types` in order to disambiguate overloads.

    Args:
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        arg_types (Optional[Tuple[Any]]): Tuple of arg types for the args
        kwarg_types (Optional[Dict[str, Any]]): Dict of arg types for the kwargs
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    if kwargs is None:
        kwargs = {}
    new_args_and_kwargs = None
    if (
        not isinstance(target, types.BuiltinFunctionType)
        and not (isinstance(target, (OpOverloadPacket, OpOverload)))
        and hasattr(target, "_op")
    ):
        # ExecuTorch's EdgeOpOverload are a wrapper around PyTorch's OpOverload,
        # so we can unwrap it here to get its schema
        # Can't import EdgeOpOverload directly because of a circular dependency,
        # so checking for "_op" existing is the next best thing.
        target = target._op

    # Repeat the condition after checking for the inner _op field.
    if not isinstance(target, types.BuiltinFunctionType) and not (
        isinstance(target, (OpOverloadPacket, OpOverload))
    ):
        target_for_analysis = target
        if target in boolean_dispatched:
            # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
            # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
            # branches of the dispatch have exactly the same signature. If they do, use the `true`
            # branch signature for analysis. Otherwise, leave this un-normalized
            if isinstance(target, str):
                raise AssertionError("target should not be a string here")
            dispatched = boolean_dispatched[target]
            if_true, if_false = dispatched["if_true"], dispatched["if_false"]
            if (
                inspect.signature(if_true).parameters
                != inspect.signature(if_false).parameters
            ):
                return None
            target_for_analysis = if_true

        if not callable(target_for_analysis):
            raise AssertionError(
                f"target_for_analysis must be callable, got {type(target_for_analysis)}"
            )
        sig = inspect.signature(inspect.unwrap(target_for_analysis))
        new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
            sig, args, kwargs, normalize_to_only_use_kwargs
        )
    else:
        if not callable(target):
            raise AssertionError(f"target must be callable, got {type(target)}")
        torch_op_schemas = get_signature_for_torch_op(target)
        matched_schemas: list[inspect.Signature] = []
        if torch_op_schemas:
            # Iterate through all of the schema until we find one that matches
            # If one matches, populate `new_args_and_kwargs` with the new args/kwargs
            # values. If none matches, `new_args_and_kwargs` will be None
            for candidate_signature in torch_op_schemas:
                try:
                    _fast_bind(candidate_signature, *args, **kwargs)
                    matched_schemas.append(candidate_signature)
                except TypeError:
                    continue

            if len(matched_schemas) == 0:
                # Did not match any schema. Cannot normalize
                pass
            elif len(matched_schemas) == 1:
                # Matched exactly one schema, unambiguous
                new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
                    matched_schemas[0], args, kwargs, normalize_to_only_use_kwargs
                )
            else:
                if arg_types is not None or kwarg_types is not None:
                    arg_types = arg_types if arg_types else cast(tuple[Any], ())
                    kwarg_types = kwarg_types if kwarg_types else {}
                    for candidate_signature in torch_op_schemas:
                        sig_matches = True
                        try:
                            bound_types = _fast_bind(
                                candidate_signature, *arg_types, **kwarg_types
                            )
                            for arg_name, arg_type in bound_types.arguments.items():
                                param = candidate_signature.parameters[arg_name]
                                sig_matches = sig_matches and type_matches(
                                    param.annotation, arg_type
                                )
                        except TypeError:
                            sig_matches = False
                        if sig_matches:
                            new_args_and_kwargs = (
                                _args_kwargs_to_normalized_args_kwargs(
                                    candidate_signature,
                                    args,
                                    kwargs,
                                    normalize_to_only_use_kwargs,
                                )
                            )
                            break
                else:
                    # Matched more than one schema. In this situation, the caller must provide the types of
                    # the arguments of the overload they expect.
                    schema_printouts = "\n".join(
                        str(schema) for schema in matched_schemas
                    )
                    raise RuntimeError(
                        f"Tried to normalize arguments to {torch.typename(target)} but "
                        f"the schema match was ambiguous! Please provide argument types to "
                        f"the normalize_arguments() call. Available schemas:\n{schema_printouts}"
                    )

    return new_args_and_kwargs


@compatibility(is_backward_compatible=False)
def normalize_module(
    root: torch.nn.Module,
    target: str,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = None,
    normalize_to_only_use_kwargs: bool = False,
) -> ArgsKwargsPair | None:
    """
    Returns normalized arguments to PyTorch modules. This means that
    `args/kwargs` will be matched up to the functional's
    signature and return exclusively kwargs in positional order if
    `normalize_to_only_use_kwargs` is True.
    Also populates default values. Does not support positional-only
    parameters or varargs parameters (``*args``, ``**kwargs``).

    Args:
        root (nn.Module): root module upon which we query modules
        target (Callable): Function that we are normalizing
        args (Tuple[Any]): Tuple of args to the function
        kwargs (Optional[Dict[str, Any]]): Dict of kwargs to the function
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Returns normalized_args_and_kwargs, or `None` if not successful.
    """
    try:
        submod = root.get_submodule(target)
    except AttributeError as e:
        raise RuntimeError(
            f"Tried to normalize node with target {target} but root did not "
            f"have that target!"
        ) from e
    if hasattr(submod.__class__, "__name__"):
        classname = submod.__class__.__name__
        if getattr(torch.nn, classname, None) == submod.__class__:
            sig = inspect.signature(inspect.unwrap(submod.forward))
            if kwargs is None:
                kwargs = {}
            new_args_and_kwargs = _args_kwargs_to_normalized_args_kwargs(
                sig, args, kwargs, normalize_to_only_use_kwargs
            )
            return new_args_and_kwargs
    return None


def _args_kwargs_to_normalized_args_kwargs(
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    normalize_to_only_use_kwargs: bool,
) -> ArgsKwargsPair | None:
    """
    Given a call target, args, and kwargs, return the arguments normalized into
    an ArgsKwargsPair, or None if the type signature is not supported by
    this normalization.

    Args:

        sig (inspect.Signature): Signature object for the target
        args (Tuple): Arguments that appear at the callsite for `target`
        kwargs (Dict): Keyword arguments that appear at the callsite for `target`
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Optional[ArgsKwargsPair]: Normalized args and kwargs for `target`, or `None` if
            this target is not supported.
    """

    # Don't currently support positional-only
    # or varargs (*args, **kwargs) signatures
    supported_parameter_types = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    if any(p.kind not in supported_parameter_types for p in sig.parameters.values()):
        # Add an exception for one signature, which is common for random/uniform, i.e.:
        # Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None
        # `from` is Python keyword and as such functions with that signature should have
        # positional-only args, but at the same time they could be dispatched as kwargs
        if list(sig.parameters.keys()) != ["input", "from", "to", "generator"]:
            return None

    bound_args = _fast_bind(sig, *args, **kwargs)
    bound_args.apply_defaults()

    new_kwargs: dict[str, Any] = {}
    new_args: list[Any] = []
    for i, param in enumerate(sig.parameters):
        if not normalize_to_only_use_kwargs and i < len(args):
            new_args.append(bound_args.arguments[param])
        else:
            new_kwargs[param] = bound_args.arguments[param]

    return ArgsKwargsPair(tuple(new_args), new_kwargs)
<|endoftext|>import copy
import logging
import threading
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional, ParamSpec, TypeVar, Union

from torch._utils_internal import signpost_event

from ._compatibility import compatibility
from .graph import Graph
from .graph_module import GraphModule
from .node import Node


_P = ParamSpec("_P")
_R = TypeVar("_R")

log = logging.getLogger(__name__)

__all__ = [
    "annotate",
    "annotate_fn",
    "preserve_node_meta",
    "has_preserved_node_meta",
    "set_stack_trace",
    "set_grad_fn_seq_nr",
    "reset_grad_fn_seq_nr",
    "format_stack",
    "set_current_meta",
    "get_current_meta",
    "NodeSource",
    "NodeSourceAction",
    "get_graph_provenance_json",
    "set_current_replay_node",
    "get_current_replay_node",
]

current_meta: dict[str, Any] = {}
current_replay_node: Node | None = None
# Thread-local holder for the optional name of the subgraph being compiled by a
# nested/regional inductor compile (e.g. a regional inductor scooped-out region
# or an invoke_subgraph submodule). Used to disambiguate trace_structured
# records emitted from sub-compiles within a single frame. Used for debugging
# only!
_regional_inductor_subgraph_name_tls = threading.local()
# Preserve the node meta fields in torch.fx.proxy._COPY_META_FIELDS
should_preserve_node_meta = False
# Preserve the "seq_nr" node meta field
_should_preserve_node_meta = False

GRADIENT_ACC_SPECIAL_STACK = (
    "Gradient addition node due to multiple use of tensor around:"
)
# =============================================================================
# FX Metadata Registry for Memory Profiler
# =============================================================================
# Global in-memory registry for FX metadata
# Maps module_name -> metadata dict containing lineno_map and node_metadata
_FX_METADATA_REGISTRY: dict[str, dict[str, Any]] = {}


def _register_fx_metadata(module_name: str, metadata: dict[str, Any]) -> None:
    """
    Register FX metadata in the global in-memory registry.

    This is called automatically during graph module compilation to store metadata
    for later use by memory profiler augmentation.

    Args:
        module_name: The module identifier (content-addressed filename)
        metadata: Metadata dict containing lineno_map, node_metadata, and source_code
    """
    # TODO: add logging to tlparse
    _FX_METADATA_REGISTRY[module_name] = metadata


@compatibility(is_backward_compatible=False)
class NodeSourceAction(Enum):
    """Enum representing the action taken to produce a node in provenance tracking."""

    CREATE = "create"
    REPLACE = "replace"


@compatibility(is_backward_compatible=False)
class NodeSource:
    """
    NodeSource is a data structure that contains the provenance information of a node.
    If node `a` is created from node `b`, then `a.meta["from_node"]` may contain NodeSource(b).
    """

    class NodeInfo:
        def __init__(self, name: str, target: str, graph_id: int) -> None:
            self.name = name
            self.target = target
            self.graph_id = graph_id

    pass_name: str
    action: list["NodeSourceAction"]
    from_node: list["NodeSource"]
    node_info: Optional["NodeInfo"]
    _dict: dict[str, Any] | None
    _action_string: str | None

    def __init__(
        self,
        node: Node | None,
        pass_name: str = "",
        action: Union["NodeSourceAction", list["NodeSourceAction"]] | None = None,
    ) -> None:
        self.pass_name = pass_name

        if action is None:
            action = []
        elif not isinstance(action, list):
            action = [action]
        for a in action:
            if not isinstance(a, NodeSourceAction):
                raise AssertionError(f"Expected NodeSourceAction, got {type(a)}")
        self.action = action
        if node:
            self.node_info = self.NodeInfo(
                name=node.name, target=str(node.target), graph_id=id(node.graph)
            )
            self.from_node = (
                copy.deepcopy(node.meta["from_node"])
                if "from_node" in node.meta
                else []
            )
        else:
            self.node_info = None
            self.from_node = []

        # cache the action string and dict representation for performance.
        self._action_string: str | None = None
        self._dict: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.node_info.name if self.node_info else ""

    @property
    def target(self) -> str:
        return self.node_info.target if self.node_info else ""

    @property
    def graph_id(self) -> int:
        return self.node_info.graph_id if self.node_info else -1

    def __repr__(self) -> str:
        return self.print_readable()

    def _get_action_string(self) -> str:
        if self._action_string is None:
            self._action_string = "+".join([a.name.lower() for a in self.action])
        return self._action_string

    def print_readable(self, indent: int = 0) -> str:
        if indent > 9:
            return ""
        result = ""
        action_string = self._get_action_string()
        result += (
            " " * indent * 4
            + f"(name={self.name}, pass_name={self.pass_name}, action={action_string}, graph_id={self.graph_id})\n"
        )
        for item in self.from_node:
            result += item.print_readable(indent + 1)
        return result

    def to_dict(self) -> dict[str, Any]:
        if self._dict is None:
            # Convert the object to a dictionary
            action_string = self._get_action_string()
            self._dict = {
                "name": self.name,
                "target": self.target,
                "graph_id": self.graph_id,
                "pass_name": self.pass_name,
                "action": action_string,
                "from_node": [node.to_dict() for node in self.from_node],
            }

        if self._dict is None:
            raise AssertionError("_dict is None after initialization")
        return self._dict

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NodeSource):
            return False
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        # Create a hash based on the dictionary representation
        # We need to convert the dict to a hashable form
        def _make_hashable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(_make_hashable(item) for item in obj)
            else:
                return obj

        return hash(_make_hashable(self.to_dict()))

    @classmethod
    def _from_dict(cls, d: dict[str, Any] | None) -> Optional["NodeSource"]:
        """
        Recursively deserialize from_node metadata from dictionary data.
        It is used to deserialize the from_node field from serialized metadata.
        Please use constructor NodeSource(node, ...) to create a NodeSource object.
        """
        if d is None:
            return None

        if not isinstance(d, dict):
            raise AssertionError(f"Expected a dict, got {type(d)}")

        # Create a NodeSource object directly without going through the constructor
        # to avoid issues with graph ID and node creation
        node_source = NodeSource.__new__(NodeSource)

        # Reset the cached properties
        node_source._action_string = None
        node_source._dict = None

        # Set the basic attributes
        node_source.pass_name = d.get("pass_name", "")

        # Parse action string back to NodeSourceAction enum list
        action_str = d.get("action", "")
        actions = []
        if action_str:
            for action_name in action_str.split("+"):
                if action_name.upper() == "CREATE":
                    actions.append(NodeSourceAction.CREATE)
                elif action_name.upper() == "REPLACE":
                    actions.append(NodeSourceAction.REPLACE)
        node_source.action = actions

        # Create the NodeInfo object directly
        if "name" in d and "target" in d and "graph_id" in d:
            node_info = NodeSource.NodeInfo(
                d.get("name", ""), d.get("target", ""), d.get("graph_id", -1)
            )
            node_source.node_info = node_info
        else:
            node_source.node_info = None

        # Recursively deserialize nested from_node
        if d.get("from_node", None) is not None:
            node_source.from_node = [
                result
                for fn in d.get("from_node", [])
                if (result := cls._from_dict(fn)) is not None
            ]
        else:
            node_source.from_node = []
        return node_source


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta(enable: bool = True) -> Iterator[None]:
    global should_preserve_node_meta
    global current_meta
    saved_should_preserve_node_meta = should_preserve_node_meta
    # Shallow copy is OK since fields of current_meta are not mutated
    saved_current_meta = current_meta.copy()
    try:
        should_preserve_node_meta = enable
        yield
    finally:
        should_preserve_node_meta = saved_should_preserve_node_meta
        current_meta = saved_current_meta


@contextmanager
def _preserve_node_seq_nr(preserve_seq_nr: bool = True) -> Iterator[None]:
    """
    Temporarily enables or disables the preservation of node.meta["seq_nr"] in the
    tracing context.
    """
    global _should_preserve_node_meta
    saved = _should_preserve_node_meta

    try:
        _should_preserve_node_meta = preserve_seq_nr
        yield
    finally:
        _should_preserve_node_meta = saved


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: list[str]) -> None:
    global current_meta

    if should_preserve_node_meta:
        if stack:
            current_meta["stack_trace"] = "".join(stack)
        else:
            # when the stack is empty, we explicitly clear the stack_trace to avoid
            # propagating it to future node.
            current_meta.pop("stack_trace", None)


@compatibility(is_backward_compatible=False)
@contextmanager
def annotate(annotation_dict: dict[str, Any]) -> Iterator[None]:
    """
    Temporarily adds custom annotations to the current tracing context.
    The fx_node produced from this tracing context will have the
    custom annotations in node.metadata["custom"] field.

    This context manager allows you to insert arbitrary metadata into the PT2
    tracing system by updating the global `current_meta["custom"]` dictionary.
    The annotations are automatically reverted after the context exits.

    Gradient accumulation nodes will not be annotated.

    This is intended for advanced users who need to attach additional metadata to the fx nodes
    (e.g., for debugging, analysis, or external tooling) during export tracing.

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        annotation_dict (dict): A dictionary of custom key-value pairs to inject
            into the FX trace metadata.

    Example:
        After exiting the context, custom annotations are removed.

        >>> with annotate({"source": "custom_pass", "tag": 42}):
        ...     pass  # Your computation here
    """

    global current_meta

    has_custom = "custom" in current_meta
    old_custom = copy.copy(current_meta.get("custom", {}))

    try:
        if not has_custom:
            current_meta["custom"] = dict[str, Any]()

        # Update with all key-value pairs from the input dict
        current_meta["custom"].update(annotation_dict)
        yield
    finally:
        if has_custom:
            # Restore the original custom dict
            current_meta["custom"] = old_custom
        else:
            del current_meta["custom"]


# Key under node.meta["custom"] used to carry the activation-checkpointing budget
# set by ``torch.autograd.graph.region_activation_memory_budget``. Shared by that
# writer and the reader below so the partitioner / AOTAutograd cache read it the
# same way.
MEMORY_BUDGET_ANNOTATION_KEY = "_region_activation_memory_budget"


def _get_memory_budget_annotation(node: Node) -> float | None:
    """
    Read the ``region_activation_memory_budget`` annotation off an FX node,
    returning ``None`` if absent. The value is written only by
    ``torch.autograd.graph.region_activation_memory_budget``, which validates it
    and stores a ``float``.
    """
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return None
    return custom.get(MEMORY_BUDGET_ANNOTATION_KEY)


@compatibility(is_backward_compatible=False)
def annotate_fn(
    annotation_dict: dict[str, Any],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    A decorator that wraps a function with the annotate context manager.
    Use this when you want to annotate an entire function instead of a specific code block.

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        annotation_dict (dict): A dictionary of custom key-value pairs to inject
            into the FX trace metadata for all operations in the function.

    Example:
        All operations in my_function will have {"pp_stage": 1} in their metadata.

        >>> @annotate_fn({"pp_stage": 1})
        ... def my_function(x):
        ...     return x + 1
    """
    from functools import wraps

    from torch.fx.proxy import _register_stack_trace_anchor

    def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
        # Register the wrapped function as a stack-trace anchor so make_fx
        # preserves its frame in node.meta["stack_trace"]. The user
        # explicitly opted into annotating this function, so its frame is a
        # meaningful attribution point even though it isn't an
        # ``nn.Module.forward``.
        _register_stack_trace_anchor(func)

        @wraps(func)
        # NB: Do not annotate with _P.args/_P.kwargs here. Dynamo guards on
        # the identity of ParamSpec annotation objects, causing guard failures.
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with annotate(annotation_dict):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def _set_autograd_backward(enable: bool = True) -> Iterator[None]:
    global current_meta

    had_autograd_backward = "autograd_backward" in current_meta
    old_autograd_backward = current_meta.get("autograd_backward", False)

    if enable:
        _mark_autograd_backward()
    try:
        yield
    finally:
        if had_autograd_backward:
            current_meta["autograd_backward"] = old_autograd_backward
        else:
            _reset_autograd_backward()


@compatibility(is_backward_compatible=False)
def _mark_autograd_backward() -> None:
    global current_meta

    current_meta["autograd_backward"] = True


@compatibility(is_backward_compatible=False)
def _reset_autograd_backward() -> None:
    global current_meta

    current_meta.pop("autograd_backward", None)


@compatibility(is_backward_compatible=False)
def set_grad_fn_seq_nr(seq_nr: int) -> None:
    global current_meta

    if should_preserve_node_meta:
        # The seq_nr is captured by eager mode in the grad_fn during forward
        current_meta["grad_fn_seq_nr"] = current_meta.get("grad_fn_seq_nr", []) + [
            seq_nr
        ]
        current_meta["in_grad_fn"] = current_meta.get("in_grad_fn", 0) + 1


@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr() -> None:
    # NB: reset state properly, this would be helpful towards supporting
    #     reentrant autograd if we actually wanted to do that.
    global current_meta
    if should_preserve_node_meta:
        current_level = current_meta.get("in_grad_fn", 0)
        if current_level <= 0:
            raise AssertionError(f"Expected current_level > 0, got {current_level}")
        if current_level == 1:
            del current_meta["in_grad_fn"]
            del current_meta["grad_fn_seq_nr"]
        else:
            current_meta["in_grad_fn"] = current_level - 1
            current_meta["grad_fn_seq_nr"] = current_meta["grad_fn_seq_nr"][:-1]


@compatibility(is_backward_compatible=False)
def format_stack() -> list[str]:
    if should_preserve_node_meta:
        return [current_meta.get("stack_trace", "")]
    else:
        # fallback to traceback.format_stack()
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    return should_preserve_node_meta


def _is_preserving_node_seq_nr() -> bool:
    return _should_preserve_node_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(node: Node, pass_name: str = "") -> Iterator[None]:
    global current_meta
    if should_preserve_node_meta and node.meta:
        saved_meta = current_meta
        try:
            current_meta = node.meta.copy()

            # Update the "from_node" field in current_meta for provenance tracking.
            # Instead of appending, overwrite the "from_node" field because current_meta
            # will be assigned to the new node. The new NodeSource(node, ...) will
            # include the information from the previous current_meta["from_node"].
            current_meta["from_node"] = [
                NodeSource(node, pass_name, NodeSourceAction.CREATE)
            ]
            yield
        finally:
            current_meta = saved_meta
    else:
        yield


@compatibility(is_backward_compatible=False)
def get_current_meta() -> dict[str, Any]:
    return current_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_replay_node(node: Node | None) -> Iterator[None]:
    """
    Set the current replay node. If `current_replay_node` is not None,
    then we're re-generating the `current_replay_node` in FunctionalTensorMode.
    """
    # See [Note] annotation for more details.
    global current_replay_node
    saved_current_replay_node = current_replay_node
    try:
        current_replay_node = node
        yield
    finally:
        current_replay_node = saved_current_replay_node


@compatibility(is_backward_compatible=False)
def get_current_replay_node() -> Node | None:
    """
    Get the current replay node
    """
    return current_replay_node


@compatibility(is_backward_compatible=False)
@contextmanager
def _set_regional_inductor_subgraph_name(name: str | None) -> Iterator[None]:
    """
    Set the name of the subgraph currently being sub-compiled by regional
    inductor (a scooped-out region or an invoke_subgraph submodule). Used to
    tag trace_structured records so artifacts emitted from different
    sub-compiles within the same frame can be distinguished.
    """
    saved = getattr(_regional_inductor_subgraph_name_tls, "value", None)
    try:
        _regional_inductor_subgraph_name_tls.value = name
        yield
    finally:
        _regional_inductor_subgraph_name_tls.value = saved


@compatibility(is_backward_compatible=False)
def _get_regional_inductor_subgraph_name() -> str | None:
    """
    Get the name of the subgraph currently being sub-compiled by regional
    inductor, or None.
    """
    return getattr(_regional_inductor_subgraph_name_tls, "value", None)


@compatibility(is_backward_compatible=False)
def get_graph_provenance_json(graph: Graph) -> dict[str, Any]:
    """
    Given an fx.Graph, return a json that contains the provenance information of each node.
    """
    try:
        provenance_tracking_json = {}
        for node in graph.nodes:
            if node.op == "call_function":
                provenance_tracking_json[node.name] = (
                    [source.to_dict() for source in node.meta["from_node"]]
                    if "from_node" in node.meta
                    else []
                )
        return provenance_tracking_json
    except Exception as e:
        # Since this is just debugging, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "get_graph_provenance_json",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        return {}


def _get_custom_metadata(gm: GraphModule) -> str:
    if not isinstance(gm, GraphModule):
        raise AssertionError(f"Expected GraphModule, got {type(gm)}")

    def helper(gm: GraphModule) -> list[Any]:
        custom_metadata = []
        for node in gm.graph.nodes:
            if hasattr(node, "meta") and node.meta.get("custom", None):
                custom_metadata.append((node.op, node.name, node.meta["custom"]))
            if node.op == "get_attr" and isinstance(
                getattr(gm, node.target), GraphModule
            ):
                custom_metadata.append(
                    # pyrefly: ignore[bad-argument-type]
                    helper(getattr(gm, node.target))
                )
        return custom_metadata

    return "\n".join(str(x) for x in helper(gm))


def _get_ordered_seq_nr_groups(
    gm: GraphModule | list[GraphModule],
) -> list[list[str]]:
    """
    Group call_function nodes by seq_nr, order by seq_nr value,
    and return a list of lists of node names (sorted alphabetically).

    Args:
        gm: A single GraphModule or a list of GraphModules to process.
            When a list is provided, nodes from all graphs are grouped together.

    Returns:
        A list of lists, where each inner list contains node names that share the same seq_nr,
        sorted alphabetically. The outer list is ordered by seq_nr value.
    """
    # Normalize input to a list
    if isinstance(gm, GraphModule):
        gms = [gm]
    else:
        gms = gm

    seq_nr_dict: dict[int, list[str]] = defaultdict(list)
    for graph_module in gms:
        for node in graph_module.graph.nodes:
            if node.op == "call_function":
                seq_nr = node.meta.get("seq_nr")
                if seq_nr is not None:
                    seq_nr_dict[seq_nr].append(node.name)
    # Sort by seq_nr and return list of sorted lists
    return [sorted(seq_nr_dict[k]) for k in sorted(seq_nr_dict.keys())]
<|endoftext|>r'''
FX is a toolkit for developers to use to transform ``nn.Module``
instances. FX consists of three main components: a **symbolic tracer,**
an **intermediate representation**, and **Python code generation**. A
demonstration of these components in action:

::

    import torch


    # Simple module for demonstration
    class MyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return self.linear(x + self.param).clamp(min=0.0, max=1.0)


    module = MyModule()

    from torch.fx import symbolic_trace

    # Symbolic tracing frontend - captures the semantics of the module
    symbolic_traced: torch.fx.GraphModule = symbolic_trace(module)

    # High-level intermediate representation (IR) - Graph representation
    print(symbolic_traced.graph)
    """
    graph():
        %x : [num_users=1] = placeholder[target=x]
        %param : [num_users=1] = get_attr[target=param]
        %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
        %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
        %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
        return clamp
    """

    # Code generation - valid Python code
    print(symbolic_traced.code)
    """
    def forward(self, x):
        param = self.param
        add = x + param;  x = param = None
        linear = self.linear(add);  add = None
        clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
        return clamp
    """

The **symbolic tracer** performs "symbolic execution" of the Python
code. It feeds fake values, called Proxies, through the code. Operations
on these Proxies are recorded. More information about symbolic tracing
can be found in the :func:`symbolic_trace` and :class:`Tracer`
documentation.

The **intermediate representation** is the container for the operations
that were recorded during symbolic tracing. It consists of a list of
Nodes that represent function inputs, callsites (to functions, methods,
or :class:`torch.nn.Module` instances), and return values. More information
about the IR can be found in the documentation for :class:`Graph`. The
IR is the format on which transformations are applied.

**Python code generation** is what makes FX a Python-to-Python (or
Module-to-Module) transformation toolkit. For each Graph IR, we can
create valid Python code matching the Graph's semantics. This
functionality is wrapped up in :class:`GraphModule`, which is a
:class:`torch.nn.Module` instance that holds a :class:`Graph` as well as a
``forward`` method generated from the Graph.

Taken together, this pipeline of components (symbolic tracing ->
intermediate representation -> transforms -> Python code generation)
constitutes the Python-to-Python transformation pipeline of FX. In
addition, these components can be used separately. For example,
symbolic tracing can be used in isolation to capture a form of
the code for analysis (and not transformation) purposes. Code
generation can be used for programmatically generating models, for
example from a config file. There are many uses for FX!

Several example transformations can be found at the
`examples <https://github.com/pytorch/examples/tree/master/fx>`__
repository.
'''

from torch.fx import immutable_collections
from torch.fx._symbolic_trace import (
    PH,
    ProxyableClassMeta,
    symbolic_trace,
    Tracer,
    wrap,
)
from torch.fx.graph import CodeGen, Graph
from torch.fx.graph_module import GraphModule
from torch.fx.interpreter import Interpreter, Transformer
from torch.fx.node import has_side_effect, map_arg, Node
from torch.fx.proxy import Proxy
from torch.fx.subgraph_rewriter import replace_pattern


__all__ = [
    "symbolic_trace",
    "Tracer",
    "wrap",
    "Graph",
    "GraphModule",
    "Interpreter",
    "Transformer",
    "Node",
    "Proxy",
    "replace_pattern",
    "has_side_effect",
    "map_arg",
]
<|endoftext|>from collections.abc import Iterable
from typing import Any, NoReturn, TypeVar
from typing_extensions import Self

from torch.utils._pytree import (
    _dict_flatten,
    _dict_flatten_with_keys,
    _dict_unflatten,
    _list_flatten,
    _list_flatten_with_keys,
    _list_unflatten,
    Context,
    register_pytree_node,
)

from ._compatibility import compatibility


__all__ = ["immutable_list", "immutable_dict"]


_help_mutation = """
If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:

    new_args = ...  # copy and mutate args
    node.args = new_args
""".strip()


_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


def _no_mutation(self: Any, *args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError(
        f"{type(self).__name__!r} object does not support mutation. {_help_mutation}",
    )


@compatibility(is_backward_compatible=True)
class immutable_list(list[_T]):
    """An immutable version of :class:`list`."""

    __delitem__ = _no_mutation
    __iadd__ = _no_mutation
    __imul__ = _no_mutation
    __setitem__ = _no_mutation
    append = _no_mutation
    clear = _no_mutation
    extend = _no_mutation
    insert = _no_mutation
    pop = _no_mutation
    remove = _no_mutation
    reverse = _no_mutation
    sort = _no_mutation

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))

    def __reduce__(self) -> tuple[type[Self], tuple[tuple[_T, ...]]]:
        return (type(self), (tuple(self),))


@compatibility(is_backward_compatible=True)
class immutable_dict(dict[_KT, _VT]):
    """An immutable version of :class:`dict`."""

    __delitem__ = _no_mutation
    __ior__ = _no_mutation
    __setitem__ = _no_mutation
    clear = _no_mutation
    pop = _no_mutation
    popitem = _no_mutation
    setdefault = _no_mutation
    update = _no_mutation  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type[Self], tuple[tuple[tuple[_KT, _VT], ...]]]:
        return (type(self), (tuple(self.items()),))


# Register immutable collections for PyTree operations
def _immutable_list_flatten(d: immutable_list[_T]) -> tuple[list[_T], Context]:
    return _list_flatten(d)


def _immutable_list_unflatten(
    values: Iterable[_T],
    context: Context,
) -> immutable_list[_T]:
    return immutable_list(_list_unflatten(values, context))


def _immutable_dict_flatten(d: immutable_dict[Any, _VT]) -> tuple[list[_VT], Context]:
    return _dict_flatten(d)


def _immutable_dict_unflatten(
    values: Iterable[_VT],
    context: Context,
) -> immutable_dict[Any, _VT]:
    return immutable_dict(_dict_unflatten(values, context))


register_pytree_node(
    immutable_list,
    _immutable_list_flatten,
    _immutable_list_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)
register_pytree_node(
    immutable_dict,
    _immutable_dict_flatten,
    _immutable_dict_unflatten,
    serialized_type_name="torch.fx.immutable_collections.immutable_dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)
<|endoftext|>from collections import namedtuple
from collections.abc import Callable
from typing import Any, TypeVar
from typing_extensions import NamedTuple

import torch.return_types
from torch.utils._pytree import PyTree, tree_flatten, TreeSpec


FlattenFnSpec = Callable[[PyTree, TreeSpec], list[Any]]
FlattenFnExactMatchSpec = Callable[[PyTree, TreeSpec], bool]

# Keep deprecated alias for backward compatibility
FlattenFuncSpec = FlattenFnSpec  # deprecated
FlattenFuncExactMatchSpec = FlattenFnExactMatchSpec  # deprecated

SUPPORTED_NODES: dict[type[Any], FlattenFnSpec] = {}
SUPPORTED_NODES_EXACT_MATCH: dict[type[Any], FlattenFnExactMatchSpec | None] = {}

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


def register_pytree_flatten_spec(
    cls: type[Any],
    flatten_fn_spec: FlattenFnSpec,
    flatten_fn_exact_match_spec: FlattenFnExactMatchSpec | None = None,
) -> None:
    SUPPORTED_NODES[cls] = flatten_fn_spec
    SUPPORTED_NODES_EXACT_MATCH[cls] = flatten_fn_exact_match_spec


def _deregister_pytree_flatten_spec(
    cls: type[Any],
) -> None:
    del SUPPORTED_NODES[cls]
    del SUPPORTED_NODES_EXACT_MATCH[cls]


def tree_flatten_spec(
    pytree: PyTree,
    spec: TreeSpec,
) -> list[Any]:
    if spec.is_leaf():
        return [pytree]
    # I guess these exist for BC, FC reasons.
    # In general, we should be able to directly
    # use pytree tree flattener to flatten them,
    # as export serializes the pytree separately.
    # Will remove it in follow up PR.
    if spec.type in SUPPORTED_NODES:
        flatten_fn_spec = SUPPORTED_NODES[spec.type]
        child_pytrees = flatten_fn_spec(pytree, spec)
        result: list[Any] = []
        for child, child_spec in zip(child_pytrees, spec.children()):
            flat = tree_flatten_spec(child, child_spec)
            result += flat
        return result
    flat_result, real_spec = tree_flatten(pytree)
    if spec != real_spec:
        raise RuntimeError(
            f"Real spec {real_spec} of object {pytree} is different from expected spec {spec}. "
            f"Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml"
        )
    return flat_result


def _dict_flatten_spec(d: dict[_K, _V], spec: TreeSpec) -> list[_V]:
    return [d[k] for k in spec.context]


def _list_flatten_spec(d: list[_T], spec: TreeSpec) -> list[_T]:
    return [d[i] for i in range(spec.num_children)]


def _tuple_flatten_spec(d: tuple[_T, ...], spec: TreeSpec) -> list[_T]:
    return [d[i] for i in range(spec.num_children)]


def _namedtuple_flatten_spec(d: NamedTuple, spec: TreeSpec) -> list[Any]:
    return [d[i] for i in range(spec.num_children)]


def _dict_flatten_spec_exact_match(d: dict[_K, _V], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _list_flatten_spec_exact_match(d: list[_T], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _tuple_flatten_spec_exact_match(d: tuple[_T, ...], spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


def _namedtuple_flatten_spec_exact_match(d: NamedTuple, spec: TreeSpec) -> bool:
    return len(d) == spec.num_children


register_pytree_flatten_spec(dict, _dict_flatten_spec, _dict_flatten_spec_exact_match)
register_pytree_flatten_spec(list, _list_flatten_spec, _list_flatten_spec_exact_match)
register_pytree_flatten_spec(
    tuple,
    _tuple_flatten_spec,
    _tuple_flatten_spec_exact_match,
)
for return_type in torch.return_types.all_return_types:
    register_pytree_flatten_spec(
        return_type,
        _tuple_flatten_spec,
        _tuple_flatten_spec_exact_match,
    )
register_pytree_flatten_spec(
    namedtuple,  # type: ignore[arg-type]
    _namedtuple_flatten_spec,
    _namedtuple_flatten_spec_exact_match,
)
<|endoftext|>import contextlib
import dataclasses
import importlib
import io
import itertools
import pickle
import weakref
from abc import abstractmethod
from collections.abc import Callable, Generator
from typing import Any, NewType, TypeVar
from typing_extensions import override, Self

from torch.utils._import_utils import import_dill


dill = import_dill()
if dill is not None:
    pickle = dill

import torch
import torch.utils._pytree as pytree
from torch._guards import TracingContext
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    is_fake_tensor,
    Tensor,
)
from torch._subclasses.meta_utils import (
    MetaConverter,
    MetaTensorDesc,
    MetaTensorDescriber,
)
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._mode_utils import no_dispatch


_SymNodeT = TypeVar("_SymNodeT", torch.SymInt, torch.SymFloat)


def _ops_filter_safe(name: str) -> bool:
    """
    An ops filter which allows pickle-safe ops. Pickle-safe ops are built-in
    ones where it will be possible to unpickle on any machine which has PyTorch.
    """
    # TODO: This list is pretty pessimistic right now. What's the full list?
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
        )
    )


def _node_metadata_key_filter_safe(key: str) -> bool:
    """
    A metadata filter which allows pickle-safe node metadata. These often times contain
    stacks with pointers to unserializable objects, so we clear them out.
    """
    return key not in ["source_fn_stack", "nn_module_stack", "fwd_source_fn_stack"]


@dataclasses.dataclass
class Options:
    # A filter for which ops will cause the pickler to raise a
    # BypassFxGraphCache exception. If None then all ops are allowed.
    ops_filter: Callable[[str], bool] | None = _ops_filter_safe
    node_metadata_key_filter: Callable[[str], bool] | None = (
        _node_metadata_key_filter_safe
    )
    # If True, raw torch.fx.Node objects encountered during pickling will be
    # silently replaced with None instead of raising an AssertionError.
    ignore_raw_node: bool = False


def _unpickle_as_none() -> None:
    return None


def _unpickle_as_weakref(referent: object) -> weakref.ref[object]:
    return weakref.ref(referent)


def _unpickle_as_dead_weakref() -> Callable[[], None]:
    return lambda: None


@contextlib.contextmanager
def patch_pytree_map_over_slice() -> Generator[None]:
    if slice in pytree.SUPPORTED_NODES:
        yield
        return

    pytree._private_register_pytree_node(
        slice, lambda x: ([x.start, x.stop, x.step], None), lambda x, c: slice(*x)
    )

    try:
        yield
    finally:
        pytree._deregister_pytree_node(slice)


# pyrefly: ignore [invalid-inheritance]
class GraphPickler(pickle.Pickler):
    """
    GraphPickler is a Pickler which helps pickling fx graph - in particular
    GraphModule.
    """

    def __init__(self, file: io.BytesIO, options: Options | None = None) -> None:
        if dill is not None:
            super().__init__(file, byref=True)
        else:
            super().__init__(file)
        self.options = options or Options()

        # This abomination is so we can pass external decoding state to the
        # unpickler functions. We serialize _unpickle_state as a persistent
        # external item and when we deserialize it we return the common state
        # object.
        self._unpickle_state = _UnpickleStateToken(object())

        # This is used to describe tensors. It needs to be common across the
        # pickle so that duplicates and views are properly handled.
        self._meta_tensor_describer = MetaTensorDescriber(copy_data=False)

    _PASSTHROUGH_TYPES = frozenset({int, float, str, bytes, bool, type(None)})

    @override
    # pyrefly: ignore [bad-override]
    def reducer_override(
        self, obj: object
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        # This function is supposed to return either NotImplemented (meaning to
        # do the default pickle behavior) or a pair of (unpickle callable, data
        # to pass to unpickle).

        # We could instead teach individual classes how to pickle themselves but
        # that has a few problems:
        #
        #   1. If we have some special needs (maybe for this use-case we don't
        #      want to fully serialize every field) then we're adding private
        #      details to a public interface.
        #
        #   2. If we need to have some common shared data (such as a
        #      FakeTensorMode) which is passed to each value it's harder to
        #      support.

        # These are the types that need special handling. See the individual
        # *PickleData classes for details on pickling that particular type.
        if type(obj) in self._PASSTHROUGH_TYPES:
            return NotImplemented

        if is_fake_tensor(obj):
            return _TensorPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch.fx.GraphModule):
            return _GraphModulePickleData.reduce_helper(self, obj)
        elif isinstance(obj, (torch._ops.OperatorBase, torch._ops.OpOverloadPacket)):
            return _OpPickleData.reduce_helper(self, obj)
        elif isinstance(obj, ShapeEnv):
            return _ShapeEnvPickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch.SymInt):
            return _SymNodePickleData.reduce_helper(self, obj)
        elif isinstance(obj, torch._guards.TracingContext):
            return _TracingContextPickleData.reduce_helper(self, obj)
        elif isinstance(obj, FakeScriptObject):
            from torch._library.opaque_object import is_opaque_constant_type

            real_obj = object.__getattribute__(obj, "real_obj")
            if real_obj is not None and is_opaque_constant_type(type(real_obj)):
                # Use default pickling; value-type opaques are picklable.
                return NotImplemented
            # Reference-type FakeScriptObjects can't be default-pickled.
            return (_unpickle_as_none, ())
        elif isinstance(obj, weakref.ref):
            # Serialize weakrefs properly: if the referent is alive,
            # serialize it and reconstruct the weakref on unpickle.
            # If the referent is dead, unpickle as a dead-weakref-like callable.
            referent = obj()
            if referent is not None:
                return (_unpickle_as_weakref, (referent,))
            else:
                return (_unpickle_as_dead_weakref, ())
        else:
            # We should never get a raw Node!
            if isinstance(obj, torch.fx.Node):
                if self.options.ignore_raw_node:
                    return (_unpickle_as_none, ())
                raise AssertionError("Unexpected raw Node during pickling")
            if reduce := _TorchNumpyPickleData.reduce_helper(self, obj):
                return reduce

            # returning `NotImplemented` causes pickle to revert to the default
            # behavior for this object.
            return NotImplemented

    @override
    # pyrefly: ignore [bad-override]
    def persistent_id(self, obj: object) -> str | None:
        if obj is self._unpickle_state:
            return "unpickle_state"
        else:
            return None

    @classmethod
    def dumps(cls, obj: object, options: Options | None = None) -> bytes:
        """
        Pickle an object.
        """
        with patch_pytree_map_over_slice(), io.BytesIO() as stream:
            pickler = cls(stream, options)
            pickler.dump(obj)
            return stream.getvalue()

    @staticmethod
    def loads(data: bytes, fake_mode: FakeTensorMode) -> object:
        """
        Unpickle an object.
        """
        from torch._dynamo.utils import dynamo_timed

        with patch_pytree_map_over_slice(), dynamo_timed("GraphPickler.loads"):
            state = _UnpickleState(fake_mode)
            with io.BytesIO(data) as stream:
                unpickler = _GraphUnpickler(stream, state)
                return unpickler.load()

    @classmethod
    def debug_dumps(
        cls,
        obj: object,
        options: "Options | None" = None,
        *,
        max_depth: int = 80,
        max_iter_items: int = 50,
        verbose: bool = True,
    ) -> str | None:
        """
        Find the first leaf that GraphPickler.dumps cannot serialize and return its path.

        This is GraphPickler-aware and avoids infinite loops by:
          - Traversing builtin containers directly (dict/list/tuple/set) instead of
            exploring their __reduce_ex__ tuples.
          - Only using __reduce_ex__ / __reduce__ for "opaque" objects.
          - Bounding recursion depth and iterator expansion.

        Args:
            obj: The object to attempt to pickle and debug.
            options: Optional Options instance for the GraphPickler.
            max_depth: Maximum recursion depth before stopping traversal.
            max_iter_items: Maximum number of items to materialize from iterators.
            verbose: If True, prints detailed traversal information.

        Returns:
            A string representing the path to the first unpicklable leaf,
            or None if the object is fully picklable.
        """
        options = options or Options()
        pickler = cls(io.BytesIO(), options)

        visited: set[int] = set()

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        def fail_exc(o: Any) -> BaseException | None:
            try:
                cls.dumps(o, options)
                return None
            except Exception as e:
                return e

        def walk(o: Any, path: str, depth: int) -> str | None:
            if depth > max_depth:
                log(f"{'  ' * depth}Depth limit at {path} ({type(o)})")
                return path + " (depth_limit)"

            key = id(o)
            if key in visited:
                return None
            visited.add(key)

            indent = "  " * depth
            log(f"{indent}Walking: {path} ({type(o)})")

            e = fail_exc(o)
            if e is None:
                log(f"{indent}✓ Pickles fine alone")
                return None
            log(f"{indent}[FAIL pickle] {type(o)} -> {e}")

            # 1) Builtin containers: walk contents directly (do NOT call __reduce_ex__)
            if isinstance(o, dict):
                for k, v in o.items():
                    bad = walk(v, f"{path}[{k!r}]", depth + 1)
                    if bad:
                        return bad
                return path

            if isinstance(o, (list, tuple)):
                for i, v in enumerate(o):
                    bad = walk(v, f"{path}[{i}]", depth + 1)
                    if bad:
                        return bad
                return path

            if isinstance(o, (set, frozenset)):
                for i, v in enumerate(o):
                    bad = walk(v, f"{path}[{i}]", depth + 1)
                    if bad:
                        return bad
                return path

            # 2) Iterator types: materialize a bounded prefix
            if hasattr(o, "__iter__") and type(o).__name__.endswith("iterator"):
                try:
                    prefix = list(itertools.islice(iter(o), max_iter_items + 1))
                except Exception:
                    prefix = None
                if prefix is not None:
                    if len(prefix) > max_iter_items:
                        log(
                            f"{indent}⚠ Iterator has more than {max_iter_items} items, "
                            f"only checking first {max_iter_items}"
                        )
                        prefix = prefix[:max_iter_items]
                    for i, v in enumerate(prefix):
                        bad = walk(v, f"{path}[{i}]", depth + 1)
                        if bad:
                            return bad
                    return path

            # 3) GraphPickler reducer_override
            try:
                red = pickler.reducer_override(o)
                log(f"{indent}reducer_override -> {type(red)}")
            except Exception as e2:
                log(f"{indent}💥 reducer_override crashed: {e2}")
                return path

            if red is not NotImplemented:
                _, args = red
                log(f"{indent}Using custom reduce, args={len(args)}")
                for i, a in enumerate(args):
                    bad = walk(a, f"{path}.reduce_args[{i}]", depth + 1)
                    if bad:
                        return bad

            # 4) Dataclasses
            if dataclasses.is_dataclass(o):
                for f in dataclasses.fields(o):
                    try:
                        v = getattr(o, f.name)
                    except Exception:
                        return f"{path}.{f.name}"
                    bad = walk(v, f"{path}.{f.name}", depth + 1)
                    if bad:
                        return bad
                return path

            # 5) __getstate__ and __dict__/__slots__
            getstate = getattr(o, "__getstate__", None)
            if callable(getstate):
                try:
                    state = getstate()
                    log(f"{indent}__getstate__ -> {type(state)}")
                except Exception as e3:
                    log(f"{indent}💥 __getstate__ failed: {e3}")
                    return path + ".__getstate__()"
                bad = walk(state, path + ".__getstate__()", depth + 1)
                if bad:
                    return bad

            if hasattr(o, "__dict__"):
                for name, v in vars(o).items():
                    bad = walk(v, f"{path}.{name}", depth + 1)
                    if bad:
                        return bad
                return path

            if hasattr(o, "__slots__"):
                for slot in o.__slots__:
                    if hasattr(o, slot):
                        bad = walk(getattr(o, slot), f"{path}.{slot}", depth + 1)
                        if bad:
                            return bad
                return path

            # 6) Last resort: reduce protocol for non-container / opaque objects
            reduce_tuple = None
            try:
                if hasattr(o, "__reduce_ex__"):
                    reduce_tuple = o.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
                    log(f"{indent}__reduce_ex__ -> {type(reduce_tuple)}")
                elif hasattr(o, "__reduce__"):
                    reduce_tuple = o.__reduce__()
                    log(f"{indent}__reduce__ -> {type(reduce_tuple)}")
            except Exception as e4:
                log(f"{indent}💥 reduce protocol failed: {e4}")
                return path

            if isinstance(reduce_tuple, tuple):
                for i, part in enumerate(reduce_tuple):
                    if part is None:
                        continue
                    bad = walk(part, f"{path}.__reduce__[{i}]", depth + 1)
                    if bad:
                        return bad

            return path

        bad = walk(obj, "root", 0)
        return bad


class _UnpickleState:
    def __init__(self, fake_mode: FakeTensorMode) -> None:
        self.fake_mode = fake_mode
        self.meta_converter: MetaConverter[FakeTensor] = MetaConverter()


# This token is passed when pickling to indicate that we want to use the
# unpickler's _UnpickleState as a parameter in that position.
_UnpickleStateToken = NewType("_UnpickleStateToken", object)


# pyrefly: ignore [invalid-inheritance]
class _GraphUnpickler(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, unpickle_state: _UnpickleState) -> None:
        super().__init__(stream)
        self._unpickle_state = unpickle_state

    @override
    # pyrefly: ignore [bad-override]
    def persistent_load(self, pid: object) -> object:
        if pid == "unpickle_state":
            return self._unpickle_state
        else:
            raise pickle.UnpicklingError("Invalid persistent ID")


class _ShapeEnvPickleData:
    data: dict[str, object]

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: ShapeEnv
    ) -> tuple[
        Callable[[Self, _UnpickleState], ShapeEnv], tuple[Self, _UnpickleStateToken]
    ]:
        return cls.unpickle, (cls(obj), pickler._unpickle_state)

    def __init__(self, env: ShapeEnv) -> None:
        # In theory pickle should recognize that a given ShapeEnv was already
        # pickled and reuse the resulting _ShapeEnvPickleData (so two objects
        # pointing at the same ShapeEnv get the same ShapeEnv out).
        if env._translation_validation_enabled:
            raise AssertionError("Translation validation must be disabled for pickling")
        self.data = env.__dict__.copy()
        del self.data["tracked_fakes"]
        del self.data["fake_tensor_cache"]

    def unpickle(self, unpickle_state: _UnpickleState) -> ShapeEnv:
        # Fill in the existing ShapeEnv rather than creating a new one
        if not unpickle_state.fake_mode:
            raise AssertionError("unpickle_state.fake_mode is not set")
        if not unpickle_state.fake_mode.shape_env:
            raise AssertionError("unpickle_state.fake_mode.shape_env is not set")

        for k, v in self.data.items():
            setattr(unpickle_state.fake_mode.shape_env, k, v)

        return unpickle_state.fake_mode.shape_env


class _SymNodePickleData:
    @classmethod
    def reduce_helper(
        cls,
        pickler: GraphPickler,
        obj: _SymNodeT,
    ) -> tuple[
        Callable[[Self, _UnpickleState], _SymNodeT], tuple[Self, _UnpickleStateToken]
    ]:
        args = (cls(obj.node), pickler._unpickle_state)
        if isinstance(obj, torch.SymInt):
            # pyrefly: ignore [bad-return]
            return _SymNodePickleData.unpickle_sym_int, args
        else:
            raise NotImplementedError(f"Unhandled SymNode type {type(obj)}")

    def __init__(self, node: SymNode) -> None:
        self.expr = node._expr
        self.shape_env = node.shape_env
        self.pytype = node.pytype
        self.hint = node._hint

    def _to_sym_node(self) -> SymNode:
        if self.shape_env is None:
            raise AssertionError("shape_env is None")
        return SymNode(self.expr, self.shape_env, self.pytype, self.hint)

    def unpickle_sym_int(self, unpickle_state: _UnpickleState) -> torch.SymInt:
        return torch.SymInt(self._to_sym_node())


class _TensorPickleData:
    metadata: MetaTensorDesc[FakeTensor]

    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: Tensor
    ) -> tuple[
        Callable[[Self, _UnpickleState], FakeTensor], tuple[Self, _UnpickleStateToken]
    ]:
        return cls.unpickle, (
            cls(pickler._meta_tensor_describer, obj),
            pickler._unpickle_state,
        )

    def __init__(self, describer: MetaTensorDescriber, t: Tensor) -> None:
        # THINGS TO WORRY ABOUT:
        # 1. Need to make sure that two tensors with the same id end up with the
        #    same id on the other side of the wire.

        metadata = describer.describe_tensor(t)

        # view_func is fine if it's either None or a _FakeTensorViewFunc. A
        # custom one (which is basically a lambda) can't be serialized.
        if metadata.view_func and not isinstance(
            metadata.view_func, torch._subclasses.meta_utils._FakeTensorViewFunc
        ):
            raise AssertionError(
                f"view_func must be None or _FakeTensorViewFunc, got "
                f"{type(metadata.view_func)}"
            )
        self.metadata = dataclasses.replace(metadata, fake_mode=None)

        # Some debugging/verification
        for k in MetaTensorDesc._UNSERIALIZABLE:
            if k in ("fake_mode", "view_func"):
                continue
            if getattr(self.metadata, k) is not None:
                raise AssertionError(f"not None: {k}: {getattr(self.metadata, k)}")

    def unpickle(self, unpickle_state: _UnpickleState) -> FakeTensor:
        # TODO: make common w/ _output_from_cache_entry() in fake_tensor.py?
        metadata = dataclasses.replace(
            self.metadata,
            fake_mode=unpickle_state.fake_mode,
        )

        # also need to set the fake_mode on the base of a tensor if it's a view
        if metadata.is_view and metadata.base is not None:
            new_base = dataclasses.replace(
                metadata.base,
                fake_mode=unpickle_state.fake_mode,
            )
            metadata = dataclasses.replace(metadata, base=new_base)

        def with_fake(
            make_meta_t: Callable[[], torch.Tensor], device: torch.device | str
        ) -> FakeTensor:
            with no_dispatch():
                return FakeTensor(
                    unpickle_state.fake_mode,
                    make_meta_t(),
                    # pyrefly: ignore [bad-argument-type]
                    device,
                )

        return unpickle_state.meta_converter.meta_tensor(
            metadata,
            unpickle_state.fake_mode.shape_env,
            with_fake,
            None,
            None,
        )


class _TorchNumpyPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: object
    ) -> (
        tuple[
            Callable[[Self, _UnpickleState], object], tuple[Self, _UnpickleStateToken]
        ]
        | None
    ):
        if data := cls.from_object(obj):
            return (cls.unpickle, (data, pickler._unpickle_state))
        else:
            return None

    def __init__(self, mod: str, name: str) -> None:
        self.mod = mod
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> Callable[..., object]:
        np = getattr(importlib.import_module(self.mod), self.name)
        return torch._dynamo.variables.misc.get_np_to_tnp_map()[np]

    @classmethod
    def from_object(cls, tnp: object) -> Self | None:
        if not callable(tnp):
            return None

        tnp_to_np = torch._dynamo.variables.misc.get_tnp_to_np_map()
        try:
            if not (np := tnp_to_np.get(tnp)):
                return None
        except TypeError:
            return None

        if not (mod := getattr(np, "__module__", None)):
            mod = "numpy"

        if not (name := getattr(np, "__name__", None)):
            return None

        # pyrefly: ignore [unbound-name]
        if np != getattr(importlib.import_module(mod), name):
            raise AssertionError(
                f"Numpy object mismatch for {mod}.{name}"  # pyrefly: ignore [unbound-name]
            )
        # pyrefly: ignore [unbound-name]
        return cls(mod, name)


class _GraphModulePickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch.fx.GraphModule
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch.fx.GraphModule],
        tuple[Self, _UnpickleStateToken],
    ]:
        return cls.unpickle, (
            cls(obj, pickler.options),
            pickler._unpickle_state,
        )

    def __init__(self, gm: torch.fx.GraphModule, options: Options) -> None:
        # Need to do this to ensure the code is created for later pickling.
        if isinstance(gm, torch.fx._lazy_graph_module._LazyGraphModule):
            _python_code = gm._real_recompile()
        else:
            _python_code = gm.recompile()
        if hasattr(gm, "__getstate__"):
            self.gm_dict = gm.__getstate__()
        else:
            self.gm_dict = gm.__dict__.copy()
        del self.gm_dict["_graph"]
        self.graph = _GraphPickleData(gm._graph, options)

    def unpickle(self, unpickle_state: _UnpickleState) -> torch.fx.GraphModule:
        gm = torch.fx.GraphModule.__new__(torch.fx.GraphModule)
        gm.__dict__ = self.gm_dict
        gm._graph = self.graph.unpickle(gm, unpickle_state)
        return gm


class _NodePickleData:
    def __init__(
        self,
        node: torch.fx.Node,
        mapping: dict[torch.fx.Node, "_NodePickleData"],
        options: Options,
    ) -> None:
        self.args = pytree.tree_map_only(torch.fx.Node, lambda n: mapping[n], node.args)
        self.kwargs = pytree.tree_map_only(
            torch.fx.Node, lambda n: mapping[n], node.kwargs
        )
        # -- self.graph = node.graph
        self.name = node.name
        self.op = node.op
        self.target = _OpPickleData.pickle(node.target, options)
        # self.input_nodes = node._input_nodes
        # self.users = node.users
        self.type = node.type
        # self.sort_key = node._sort_key
        # self.repr_fn = node._repr_fn
        # self.meta = node.meta
        self.meta = {
            k: v
            for k, v in node.meta.items()
            if (
                not options.node_metadata_key_filter
                or options.node_metadata_key_filter(k)
            )
        }

    def unpickle(
        self,
        graph: torch.fx.Graph,
        mapping: dict["_NodePickleData", torch.fx.Node],
        unpickle_state: _UnpickleState,
    ) -> torch.fx.Node:
        args = pytree.tree_map_only(_NodePickleData, lambda n: mapping[n], self.args)
        kwargs = pytree.tree_map_only(
            _NodePickleData, lambda n: mapping[n], self.kwargs
        )
        target = self.target.unpickle(unpickle_state)
        if not (callable(target) or isinstance(target, str)):
            raise AssertionError(f"target must be callable or str, got {type(target)}")
        node = graph.create_node(self.op, target, args, kwargs, self.name, self.type)
        node.meta = self.meta
        return node


class _OpPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, op: object
    ) -> tuple[Callable[[_UnpickleState], object], tuple[_UnpickleStateToken]]:
        result = cls.pickle(op, pickler.options)
        return (result.unpickle, (pickler._unpickle_state,))

    @classmethod
    def pickle(cls, op: object, options: Options) -> "_OpPickleData":
        if isinstance(op, str):
            return _OpStrPickleData(op)

        if isinstance(getattr(op, "__wrapped__", None), AOTCompiledArtifact):
            if not hasattr(op, "__wrapped__"):
                raise AssertionError("op missing __wrapped__ attribute")
            artifact = op.__wrapped__
            if not isinstance(artifact, AOTCompiledArtifact):
                raise AssertionError(
                    f"Expected AOTCompiledArtifact, got {type(artifact)}"
                )
            return _OpPrecompiledPickleData(artifact)

        name = torch.fx.Node._pretty_print_target(op)

        if isinstance(op, torch._ops.OpOverload):
            return cls._pickle_op(name, _OpOverloadPickleData, options)
        elif isinstance(op, torch._ops.OpOverloadPacket):
            return cls._pickle_op(name, _OpOverloadPacketPickleData, options)
        elif name.startswith(_OpFunctionPickleData.SUPPORTED_ROOTS):
            root, detail = name.split(".", 1)
            return _OpFunctionPickleData(root, detail)
        else:
            # TODO: raise a BypassFxGraphCache so we will just bypass this one...
            raise NotImplementedError(f"TARGET: {type(op)} {op} {name}")

    @staticmethod
    def _pickle_op(
        name: str,
        datacls: type["_OpOverloadPickleData"] | type["_OpOverloadPacketPickleData"],
        options: Options,
    ) -> "_OpPickleData":
        if (ops_filter := options.ops_filter) and not ops_filter(name):
            from torch._inductor.codecache import CacheabilityValidator

            CacheabilityValidator.bypass(f"Unable to pickle non-standard op: {name}")
        return datacls(name)

    @abstractmethod
    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        pass

    @classmethod
    def _lookup_global_by_name(cls, name: str) -> object:
        """
        Like `globals()[name]` but supports dotted names.
        """
        if "." in name:
            mod, rest = name.split(".", 1)
            root = globals()[mod]
            return cls._getattr_by_name(root, rest)
        else:
            return globals()[name]

    @staticmethod
    def _getattr_by_name(root: object, name: str) -> object:
        """
        Like `getattr(root, name)` but supports dotted names.
        """
        while "." in name:
            mod, name = name.split(".", 1)
            root = getattr(root, mod)
        return getattr(root, name)


class _OpStrPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> str:
        return self.name


class _OpOverloadPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverload:
        obj = self._lookup_global_by_name(self.name)
        if not isinstance(obj, torch._ops.OpOverload):
            raise AssertionError(f"Expected OpOverload, got {type(obj)}")
        return obj


class _OpOverloadPacketPickleData(_OpPickleData):
    def __init__(self, name: str) -> None:
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> torch._ops.OpOverloadPacket:
        obj = self._lookup_global_by_name(self.name)
        if not isinstance(obj, torch._ops.OpOverloadPacket):
            raise AssertionError(f"Expected OpOverloadPacket, got {type(obj)}")
        return obj


class _OpPrecompiledPickleData(_OpPickleData):
    def __init__(self, artifact: AOTCompiledArtifact) -> None:
        self.contents = artifact.serialize()

    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        precompiled_artifact = AOTCompiledArtifact.deserialize(self.contents)
        import functools

        @functools.wraps(precompiled_artifact)
        def wrapped(*args: Any) -> Any:
            return precompiled_artifact(*args)

        return wrapped


class _OpFunctionPickleData(_OpPickleData):
    """
    Supports pickling a set of standard/common functions
    These must be prefixed with the full namespace in order to properly
    be pickled (i.e `einops.rearrange` and not `from einops import rearrange`)
    """

    # Static variable listing supported root names
    SUPPORTED_ROOTS = ("builtins.", "math.", "torch.", "operator.", "einops.")

    def __init__(self, root: str, name: str) -> None:
        self.root = root
        self.name = name

    def unpickle(self, unpickle_state: _UnpickleState) -> object:
        if self.root == "builtins":
            return __builtins__.get(self.name)  # type: ignore[attr-defined]
        elif self.root == "math":
            import math

            return self._getattr_by_name(math, self.name)
        elif self.root == "torch":
            return self._getattr_by_name(torch, self.name)
        elif self.root == "operator":
            import operator

            return self._getattr_by_name(operator, self.name)
        elif self.root == "einops":
            import einops

            return self._getattr_by_name(einops, self.name)
        else:
            raise NotImplementedError


class _GraphPickleData:
    def __init__(self, graph: torch.fx.Graph, options: Options) -> None:
        self.tracer_cls = graph._tracer_cls
        self.tracer_extras = graph._tracer_extras

        nodes: dict[torch.fx.Node, _NodePickleData] = {}
        for node in graph.nodes:
            nodes[node] = _NodePickleData(node, nodes, options)
        self.nodes = tuple(nodes.values())
        self._codegen = graph._codegen

        # Unpickled variables:
        # self._used_names = graph._used_names
        # -- self._insert = self._root.prepend
        # self._len = graph._len
        # self._graph_namespace = graph._graph_namespace
        # self._owning_module = graph._owning_module
        # self._co_fields: Dict[str, Any] = graph._co_fields
        # -- self._find_nodes_lookup_table = _FindNodesLookupTable()

    def unpickle(
        self, gm: torch.fx.GraphModule, unpickle_state: _UnpickleState
    ) -> torch.fx.Graph:
        graph = torch.fx.Graph(gm, self.tracer_cls, self.tracer_extras)

        nodes: dict[_NodePickleData, torch.fx.Node] = {}
        for nd in self.nodes:
            nodes[nd] = nd.unpickle(graph, nodes, unpickle_state)
        if hasattr(self, "_codegen"):
            graph._codegen = self._codegen

        return graph


class _TracingContextPickleData:
    @classmethod
    def reduce_helper(
        cls, pickler: GraphPickler, obj: torch._guards.TracingContext
    ) -> tuple[
        Callable[[Self, _UnpickleState], torch._guards.TracingContext],
        tuple[Self, _UnpickleStateToken],
    ]:
        return (
            cls.unpickle,
            (
                cls(obj),
                pickler._unpickle_state,
            ),
        )

    def __init__(self, context: TracingContext) -> None:
        # TODO: Do we really need all of this?
        self.module_context = context.module_context
        self.frame_summary_stack = context.frame_summary_stack
        self.loc_in_frame = context.loc_in_frame
        self.aot_graph_name = context.aot_graph_name
        self.params_flat = context.params_flat
        self.params_flat_unwrap_subclasses = context.params_flat_unwrap_subclasses
        self.params_unwrapped_to_flat_index = context.params_unwrapped_to_flat_index
        self.output_strides = context.output_strides
        self.force_unspec_int_unbacked_size_like = (
            context.force_unspec_int_unbacked_size_like
        )
        # Not saved (because it's difficult and maybe not needed?):
        #   self.fw_metadata = context.fw_metadata
        #   self.guards_context = None
        #   self.global_context = None
        #   self.fake_mode = None
        #   self.fakify_first_call = None
        #   self.hop_dispatch_set_cache = None
        #   self.tensor_to_context = context.tensor_to_context

    def unpickle(self, unpickle_state: _UnpickleState) -> TracingContext:
        context = TracingContext(unpickle_state.fake_mode)
        context.module_context = self.module_context
        context.frame_summary_stack = self.frame_summary_stack
        context.loc_in_frame = self.loc_in_frame
        context.aot_graph_name = self.aot_graph_name
        context.params_flat = self.params_flat
        context.params_flat_unwrap_subclasses = self.params_flat_unwrap_subclasses
        context.params_unwrapped_to_flat_index = self.params_unwrapped_to_flat_index
        context.output_strides = self.output_strides
        context.force_unspec_int_unbacked_size_like = (
            self.force_unspec_int_unbacked_size_like
        )
        return context
<|endoftext|>import sys
from typing import Any

import torch
from torch._logging import LazyString


def lazy_format_graph_code(
    name: str, gm: torch.fx.GraphModule, maybe_id: int | None = None, **kwargs: Any
) -> LazyString:  # pyrefly: ignore[implicit-any]
    """
    Returns a LazyString that formats the graph code.
    """

    def format_name() -> str:
        if maybe_id is not None:
            return f"{name} {maybe_id}"
        else:
            return name

    if "print_output" not in kwargs:
        kwargs["print_output"] = False

    if "colored" in kwargs:
        try:
            if not sys.stdout.isatty():
                kwargs["colored"] = False
        except AttributeError:
            kwargs["colored"] = False

    return LazyString(
        lambda: _format_graph_code(
            f"===== {format_name()} =====\n",
            gm.forward.__code__.co_filename,
            gm.print_readable(**kwargs),
        )
    )


def _format_graph_code(name: str, filename: str, graph_str: str) -> str:
    """
    Returns a string that formats the graph code.
    """
    return f"TRACED GRAPH\n {name} {filename} {graph_str}\n"


def first_call_function_nn_module_stack(graph: torch.fx.Graph) -> dict[str, Any] | None:
    """
    Returns the nn_module_stack of the first call_function node.
    """
    for node in graph.nodes:
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            return node.meta["nn_module_stack"]
    return None


def get_node_context(node: torch.fx.Node, num_nodes: int = 2) -> str:
    """
    Returns a string of the last num_nodes nodes in the graph.
    """
    node_contexts = []
    cur = node
    for _ in range(num_nodes):
        # cast to str to handle None return value
        node_contexts.append(str(cur.format_node()))
        if cur.op == "root":
            break
        cur = cur.prev
    return "\n".join(node_contexts[::-1])
<|endoftext|>import builtins
import collections
import contextlib
import copy
import functools
import inspect
import logging
import math
import os
import threading
import warnings
from collections.abc import Callable, Iterable, Iterator
from itertools import chain
from types import CodeType, FunctionType, ModuleType, TracebackType
from typing import Any, get_args, NamedTuple, overload, ParamSpec, TypeAlias, TypeVar

import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class, is_opaque_symbolic_type

from ._compatibility import compatibility
from ._lazy_graph_module import _make_graph_module
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, Scope, ScopeContextManager, TracerBase


log = logging.getLogger(__name__)

HAS_VARSTUFF = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS

_F = TypeVar("_F", bound=Callable[..., Any])
_P = ParamSpec("_P")
_T = TypeVar("_T")

# These need to run in global scope to handle nested calls correctly
_orig_module_call: Callable[..., Any] = torch.nn.Module.__call__
_orig_module_getattr: Callable[..., Any] = torch.nn.Module.__getattr__

_proxyable_classes: dict[type, None] = {}

_is_fx_tracing_tls = threading.local()

_ConstantAttributeType: TypeAlias = (
    torch.Tensor | torch.ScriptObject | FakeScriptObject | pytree.TreeSpec
)

_constant_attribute_types = get_args(_ConstantAttributeType)


# We only want to print this once to avoid flooding logs
@functools.lru_cache
def is_fx_tracing_warning() -> None:
    log.warning(
        "is_fx_tracing will return true for both fx.symbolic_trace and "
        "torch.export. Please use "
        "is_fx_tracing_symbolic_tracing() for specifically fx.symbolic_trace "
        "or torch.compiler.is_compiling() for specifically torch.export/compile."
    )


def _set_is_fx_tracing(value: bool) -> None:
    _is_fx_tracing_tls.flag = value


def _get_is_fx_tracing() -> bool:
    return getattr(_is_fx_tracing_tls, "flag", False)


@contextlib.contextmanager
def _is_fx_tracing_context(value: bool) -> Iterator[None]:
    previous = _get_is_fx_tracing()
    _set_is_fx_tracing(value)
    try:
        yield
    finally:
        _set_is_fx_tracing(previous)


def is_fx_tracing() -> bool:
    is_fx_tracing_warning()
    return _get_is_fx_tracing()


def is_fx_symbolic_tracing() -> bool:
    return _get_is_fx_tracing() and not torch.compiler.is_compiling()


@compatibility(is_backward_compatible=True)
class ProxyableClassMeta(type):
    """
    ProxyableClassMeta allows you to make construction of a given Python class
    symbolically traceable. For example::

        import torch
        import torch.fx


        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)


        def use_tensor_pair_ctor(x: TensorPair, y: torch.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)


        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = torch.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : torch.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    From this example, we can see that construction of a class (``TensorPair``)
    defined with ``ProxyableClassMeta`` as metaclass can be recorded in symbolic
    tracing.
    """

    def __init__(
        cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> None:
        _proxyable_classes.setdefault(cls)
        super().__init__(name, bases, attrs)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        instance = cls.__new__(cls)  # type: ignore[call-overload]

        if not is_fx_tracing():
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance

        found_proxies: list[Proxy] = []

        def check_proxy(a: object) -> None:
            if isinstance(a, Proxy):
                found_proxies.append(a)

        map_aggregate(args, check_proxy)
        map_aggregate(kwargs, check_proxy)

        if len(found_proxies) != 0:
            tracer = found_proxies[0].tracer
            return tracer.create_proxy("call_function", cls, args, kwargs)
        else:
            cls.__init__(instance, *args, **kwargs)  # type: ignore[misc]
            return instance


def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args: tuple[Any, ...]
    if hasattr(co, "co_qualname"):
        # Python-3.11+ code signature
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_qualname,  # type: ignore[attr-defined]
            co.co_firstlineno,
            co.co_linetable,
            co.co_exceptiontable,  # type: ignore[attr-defined]
            co.co_freevars,
            co.co_cellvars,
        )
    elif hasattr(co, "co_posonlyargcount"):
        co_args = (
            nargs,
            0,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    else:
        co_args = (
            nargs,
            0,
            co.co_nlocals,
            co.co_stacksize,
            co_flags,
            co.co_code,
            co.co_consts,
            co.co_names,
            co.co_varnames,
            co.co_filename,
            co.co_name,
            co.co_firstlineno,
            co.co_lnotab,
            co.co_freevars,
            co.co_cellvars,
        )
    new_code = CodeType(*co_args)  # type: ignore[arg-type]
    return FunctionType(
        new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )

    # we need to insert placeholder nodes for *args and **kwargs
    # we can't call this function normally, otherwise it would try to unpack them
    # instead, let's make python think that args and kwargs are normal variables


@compatibility(is_backward_compatible=False)
class PHBase:
    """
    Object representing an input placeholder to `concrete_args`
    """

    def __repr__(self) -> str:
        return "PH"


PH = PHBase()


@compatibility(is_backward_compatible=False)
class PHWithMeta(PHBase):
    """
    Object representing an input placeholder to `concrete_args`
    """

    def __init__(self, ph_key: str | None = None) -> None:
        super().__init__()

        # Provide a key for user to identify placeholder node during analysis
        self.ph_key = ph_key


def _transfer_attrs(fr: object, to: object) -> None:
    for attr_name in dir(fr):
        attr_val = getattr(fr, attr_name)
        if (
            not callable(attr_val)
            and not attr_name.startswith("__")
            and not hasattr(to, attr_name)
        ):
            setattr(to, attr_name, attr_val)


@compatibility(is_backward_compatible=True)
class Tracer(TracerBase):
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `math`s path from the
    # build environment (e.g. `<module 'math' from '/leaked/path').

    """Tracer(autowrap_modules=(math,), autowrap_functions=())

    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.

    Tracer can be subclassed to override various behaviors of the tracing
    process. The different behaviors that can be overridden are described
    in the docstrings of the methods on this class.
    """

    # Not checking BC on this API because the default value for `autowrap_modules`
    # includes the local filepath to the `math` module, which would jitter
    # across machines.
    @compatibility(is_backward_compatible=True)
    def __init__(
        self,
        autowrap_modules: tuple[ModuleType] = (math,),
        autowrap_functions: tuple[Callable[..., Any], ...] = (),
        param_shapes_constant: bool = False,
    ) -> None:
        # This method's signature is overridden by the first line of this class'
        # docstring. If this method's signature is modified, the signature that
        # overrides it also should be modified accordingly.

        """
        Construct a Tracer object.

        Args:

            autowrap_modules (Tuple[ModuleType]): defaults to `(math, )`,
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap(). Backward-compatibility for
                this parameter is guaranteed.

            autowrap_functions (Tuple[Callable, ...]): defaults to `()`,
                Python functions that should be wrapped automatically without
                needing to use fx.wrap(). Backward compatibility for this
                parameter is guaranteed.

            param_shapes_constant (bool): When this flag is set,  calls to shape,
                size and a few other shape like attributes of a module's parameter
                will be evaluated directly, rather than returning a new Proxy value
                for an attribute access. Backward compatibility for this parameter
                is guaranteed.
        """

        super().__init__()

        # Functions we will eagerly wrap when we see them while tracing
        # this captures both `math.sqrt()` and `from math import sqrt` automatically
        self._autowrap_function_ids: set[int] = {
            id(value)
            for name, value in chain.from_iterable(
                m.__dict__.items() for m in autowrap_modules
            )
            if not name.startswith("_") and callable(value)
        }
        self._autowrap_function_ids.update({id(f) for f in autowrap_functions})

        # Python modules to apply autowrap to at the start, in addition to
        # modules we see while tracing
        self._autowrap_search: list[ModuleType] = list(autowrap_modules)
        self.param_shapes_constant = param_shapes_constant

        self.submodule_paths: dict[torch.nn.Module, str] | None = None
        self.root_module_name: str = ""
        # Maps the containing module's name to the operator name
        self.scope = Scope("", None)
        # Records the module call stack
        self.module_stack = collections.OrderedDict()
        self.num_calls: dict[str, int] = {}
        # Mapping of node name to module scope
        self.node_name_to_scope: dict[str, tuple[str, type]] = {}

    _qualname_counter: dict[str, int] = collections.defaultdict(int)

    @compatibility(is_backward_compatible=True)
    def get_fresh_qualname(self, prefix: str) -> str:
        """
        Gets a fresh name for a prefix and returns it. This function ensures
        that it will not clash with an existing attribute on the graph.
        """
        # The idea here is that if the module doesn't have this prefix at all we
        # should reset the counter to start from the beginning
        # It's a ... little bit hacky (doesn't cover all cases) but the precise
        # naming of the prefixes isn't a correctness issue, just a niceness
        # issue
        qualname = f"{prefix}0"
        if not hasattr(self.root, qualname):
            self._qualname_counter[prefix] = 0
            return qualname

        i = self._qualname_counter[prefix]
        while True:
            qualname = f"{prefix}{i}"
            i += 1
            if not hasattr(self.root, qualname):
                break
        self._qualname_counter[prefix] = i

        return qualname

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> "Argument":
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        By default, the behavior includes:

        #. Iterate through collection types (e.g. tuple, list, dict) and recursively
           call ``create_args`` on the elements.
        #. Given a Proxy object, return a reference to the underlying IR ``Node``
        #. Given a non-Proxy Tensor object, emit IR for various cases:

            * For a Parameter, emit a ``get_attr`` node referring to that Parameter
            * For a non-Parameter Tensor, store the Tensor away in a special
              attribute referring to that attribute.

        This method can be overridden to support more types.

        Args:

            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.


        Returns:

            The value ``a`` converted into the appropriate ``Argument``
        """
        # The base tracer is used to construct Graphs when there is no associated
        # module hierarchy, so it can never create parameter references.
        # The default tracer adds the ability to refer to parameters when
        # tracing modules.
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})
            raise NameError("parameter is not a member of this module")
        elif isinstance(a, torch.Tensor):
            for n_, p_ in self.root.named_buffers():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        elif isinstance(a, torch.nn.Module):
            for n_, p_ in self.root.named_modules():
                if a is p_:
                    return self.create_node("get_attr", n_, (), {})
        # For NamedTuple instances that appear literally as args, we emit
        # a node to construct the NamedTuple and use that Node as the argument.
        if isinstance(a, tuple) and hasattr(a, "_fields"):
            args = tuple(self.create_arg(elem) for elem in a)
            return self.create_node("call_function", a.__class__, args, {})

        # Tensors do not have a reliable string repr() from which they can be
        # constructed (and we probably don't want to rely on that, either), so
        # for any constant Tensor values we encounter, first search for if they
        # are an attribute of some module in the module hierarchy. If so, emit
        # a get_attr to retrieve that tensor. Otherwise, we'll store away the
        # tensor value into a special attribute on the Module s.t. we can
        # retrieve it with a get_attr.
        if isinstance(a, _constant_attribute_types) or (
            is_opaque_symbolic_type(type(a))
        ):
            qualname: str | None = self.tensor_attrs.get(a)

            # Tensor was not found in the Module hierarchy, stow it away in a
            # special attribute and set the qualname to refer to that
            if not qualname:
                if isinstance(a, torch.Tensor):
                    base_name = "_tensor_constant"
                elif isinstance(a, (FakeScriptObject, ScriptObject)):
                    base_name = "_torchbind_obj"
                elif isinstance(a, pytree.TreeSpec):
                    base_name = "_tree_spec_constant"
                elif is_custom_class(type(a)):
                    base_name = "_opaque_obj"
                else:
                    raise RuntimeError(
                        f"cannot create constant arg for {a} of type {type(a)}."
                    )
                qualname = self.get_fresh_qualname(base_name)
                if not isinstance(qualname, str):
                    raise AssertionError(
                        f"Expected qualname to be str, got {type(qualname)}"
                    )
                self.tensor_attrs[a] = qualname
                setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})

        if type(a) in _proxyable_classes:
            # This is an instance of a proxyable class for which we did not
            # witness its construction. Intern this as a constant attribute

            # TODO: binary search
            qualname = self.get_fresh_qualname(f"_{a.__class__.__name__}_constant_")
            if not isinstance(qualname, str):
                raise AssertionError(
                    f"Expected qualname to be str, got {type(qualname)}"
                )
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})
        return super().create_arg(a)

    @compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        return (
            m.__module__.startswith("torch.nn")
            or m.__module__.startswith("torch.ao.nn")
        ) and not isinstance(m, torch.nn.Sequential)

    @compatibility(is_backward_compatible=True)
    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Helper method to find the qualified name of ``mod`` in the Module hierarchy
        of ``root``. For example, if ``root`` has a submodule named ``foo``, which has
        a submodule named ``bar``, passing ``bar`` into this function will return
        the string "foo.bar".

        Args:

            mod (str): The ``Module`` to retrieve the qualified name for.
        """
        # Prefer the O(1) algorithm
        if self.submodule_paths:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError("module is not installed as a submodule")
            if not isinstance(path, str):
                raise AssertionError(f"Expected path to be str, got {type(path)}")
            return path
        # O(N^2) fallback in the case that we didn't store the submodule
        # paths.
        else:
            for n, p in self.root.named_modules():
                if mod is p:
                    return n
            raise NameError("module is not installed as a submodule")

    @compatibility(is_backward_compatible=True)
    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(
            self.scope, Scope(module_qualified_name, type(m))
        ) as _scope:
            # module_stack is an ordered dict so writing then deleting the
            # entry is equivalent to push/pop on a list
            num_calls = self.num_calls.get(module_qualified_name, 0)
            module_key = (
                f"{_scope.module_path}@{num_calls}"
                if num_calls > 0
                else _scope.module_path
            )
            self.module_stack[module_key] = (module_qualified_name, _scope.module_type)
            self.num_calls[module_qualified_name] = num_calls + 1
            if not self.is_leaf_module(m, module_qualified_name):
                ret_val = forward(*args, **kwargs)
            else:
                ret_val = self.create_proxy(
                    "call_module", module_qualified_name, args, kwargs
                )
            key, _ = self.module_stack.popitem(last=True)
            if key != module_key:
                raise AssertionError(f"Unexpected key {key}, expected {module_key}")

        return ret_val

    @compatibility(is_backward_compatible=False)
    def getattr(
        self, attr: str, attr_val: Any, parameter_proxy_cache: dict[str, Proxy]
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when we call getattr
        on a call to an ``nn.Module`` instance.

        By default, the behavior is to return a proxy value for the attribute. It
        also stores the proxy value in the ``parameter_proxy_cache``, so that future
        calls will reuse the proxy rather than creating a new one.

        This method can be overridden to --for example-- not return proxies when
        querying parameters.

        Args:

            attr (str): The name of the attribute being queried
            attr_val (Any): The value of the attribute
            parameter_proxy_cache (Dict[str, Any]): A cache of attr names to proxies

        Return:

            The return value from the getattr call.
        """

        def maybe_get_proxy_for_attr(
            attr_val: Any,
            collection_to_search: Iterable[tuple[str, Any]],
            parameter_proxy_cache: dict[str, Proxy],
        ) -> Proxy | None:
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if (
                            "proxy_factory_fn"
                            in inspect.signature(self.create_proxy).parameters
                        ):
                            kwargs["proxy_factory_fn"] = (
                                # pyrefly: ignore [unsupported-operation]
                                None
                                if not self.param_shapes_constant
                                else lambda node: ParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache
            )
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        return attr_val

    # This method will be refactored
    @compatibility(is_backward_compatible=False)
    def create_args_for_root(
        self,
        root_fn: Callable[..., Any],
        is_module: bool,
        concrete_args: dict[str, Any] | tuple[Any, ...] | None = None,
    ) -> tuple[Any, list[Any]]:
        """
        Create ``placeholder`` nodes corresponding to the signature of the ``root``
        Module. This method introspects root's signature and emits those
        nodes accordingly, also supporting ``*args`` and ``**kwargs``.
        """
        # In some cases, a function or method has been decorated with a wrapper
        # defined via ``functools.wraps``. In this case, the outer code object
        # will likely not contain the actual parameters we care about, so unwrap
        # the function to get to the innermost callable.
        fn_for_analysis = inspect.unwrap(root_fn)
        co = fn_for_analysis.__code__
        total_args = co.co_argcount + co.co_kwonlyargcount
        orig_args = list(co.co_varnames)
        names_iter = iter(co.co_varnames)
        args: list[Any] = []
        skip_arg_idx = 0
        if is_module:
            if total_args == 0:
                raise RuntimeError(
                    "``self`` argument cannot be part of *args expansion!"
                )
            skip_arg_idx = 1
            next(names_iter)  # skip self
            args.append(self.root)

        sig = inspect.signature(fn_for_analysis)

        # This covers the very specific case where we are passing in flat
        # concrete_args as a tuple, but our traced fn takes (*args, **kwargs).
        # In this case, just take the concrete_args and pass them through.
        name_idx = 0
        if (
            isinstance(concrete_args, tuple)
            and len(concrete_args) > 0
            and (co.co_flags & HAS_VARSTUFF)
            and total_args == 1
        ):
            for concrete_arg in concrete_args:
                out = self.create_proxy("placeholder", f"input_{name_idx}", (), {})
                if isinstance(concrete_arg, PHBase):
                    if concrete_arg != PH:
                        # Transfer attrs in the case where you're using a placeholder other
                        # than the singleton PH (PH has no attributes to transfer).
                        # Proxies were created out of the placeholders.
                        # Transfer any metadata (put on the placeholders in the form of
                        # attributes set by the user) from the placeholder to the
                        # underlying nodes (the proxy is unwrapped by the user, but
                        # the metadata should hold).
                        _transfer_attrs(fr=concrete_arg, to=out.node)
                args.append(out)
                name_idx += 1
            return root_fn, args

        arg_names = [next(names_iter) for idx in range(skip_arg_idx, total_args)]
        if isinstance(concrete_args, tuple):
            if len(arg_names) != len(concrete_args):
                raise RuntimeError(
                    f"Tracing expected {len(arg_names)} arguments but got {len(concrete_args)} concrete arguments"
                )
            concrete_args = dict(zip(arg_names, concrete_args))

        def proxy_placeholder(name: str) -> Any:
            return self._proxy_placeholder(name, concrete_args, sig, fn_for_analysis)

        args.extend(proxy_placeholder(names) for names in arg_names)

        if co.co_kwonlyargcount > 0 or co.co_flags & HAS_VARSTUFF:
            # TODO: type annotations for *args and **kwargs
            if co.co_flags & inspect.CO_VARARGS:
                args.append(proxy_placeholder("*" + next(names_iter)))
            if co.co_flags & inspect.CO_VARKEYWORDS:
                args.append(proxy_placeholder("**" + next(names_iter)))
            root_fn = _patch_function(root_fn, len(args))

        flat_args, in_spec = pytree.tree_flatten(tuple(args))
        if not all(child.is_leaf() for child in in_spec.children()):
            # In the case that we have pytree-flattened inputs in
            # `concrete_args`, generate a flattening wrapper around the
            # original root function and return that.
            self.graph._codegen = _PyTreeCodeGen(  # type: ignore[has-type]
                _PyTreeInfo(orig_args[:total_args], in_spec, None)
            )

            # TODO: annotate return type. inspect.get_annotations(flatten_fn)
            # leaks the return annotation into the generated forward() code.
            def flatten_fn(  # pyrefly: ignore[unannotated-parameter, unannotated-return]
                *args: Any,
            ):
                tree_args = pytree.tree_unflatten(list(args), in_spec)
                tree_out = root_fn(*tree_args)
                out_args, out_spec = pytree.tree_flatten(tree_out)
                if not isinstance(self.graph._codegen, _PyTreeCodeGen):  # type: ignore[has-type]
                    raise AssertionError(
                        f"Expected _codegen to be _PyTreeCodeGen, got "
                        f"{type(self.graph._codegen)}"
                    )
                self.graph._codegen.pytree_info = (
                    self.graph._codegen.pytree_info._replace(out_spec=out_spec)
                )
                return out_args

            return flatten_fn, flat_args
        return root_fn, args

    @compatibility(is_backward_compatible=True)
    def trace(
        self,
        root: torch.nn.Module | Callable[..., Any],
        concrete_args: dict[str, Any] | None = None,
    ) -> Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants
        to.


        Args:

            root (Union[Module, Callable]): Either a ``Module`` or a function to be
