                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            length = self.range_length()
            start = self.start()
            step = self.step()
            new_start = start + (length - 1) * step
            new_step = -step
            return RangeIteratorVariable(new_start, 0, new_step, length)
        elif name == "index":
            x = args[0].as_python_constant()
            start, stop, step = self.start(), self.stop(), self.step()
            in_range = (start <= x < stop) if step > 0 else (stop < x <= start)
            if in_range and ((x - start) % step) == 0:
                return VariableTracker.build(tx, (x - start) // step)
            raise_observed_exception(
                ValueError,
                tx,
                args=[f"{x} is not in range"],
            )
        return super().call_method(tx, name, args, kwargs)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        fields = ["start", "stop", "step"]
        if name in fields:
            return self.items[fields.index(name)]

        return super().tp_getattro_impl(tx, name)

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # CPython range_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/rangeobject.c#L572
        return hash(self.as_python_constant()), False


class CommonListMethodsVariable(BaseListVariable):
    """
    Implement methods common to List and other List-like things
    """

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .tensor import SymNodeVariable

        if name == "append" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            (arg,) = args
            tx.output.side_effects.mutation(self)
            self.items.append(arg)
            return ConstantVariable.create(None)
        elif name == "extend" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            # CPython has a series of checks to optimize list.extend for different data types
            # ref: https://github.com/python/cpython/blob/0fd4fd4496c557b68477a99c1c231a5870c91daf/Objects/listobject.c#L1389-L1444
            from .dicts import ConstDictVariable
            from .sets import SetVariable
            from .user_defined import UserDefinedObjectVariable

            sz = len(self.items)
            if isinstance(args[0], (ListVariable, TupleVariable)):
                self.items.extend(args[0].items)
            elif isinstance(args[0], UserDefinedObjectVariable):
                self.items.extend(unpack_iterable(tx, args[0]))
            elif isinstance(args[0], (ConstDictVariable, SetVariable)):
                items = [item.vt for item in args[0].items]
                self.items.extend(items)
            elif isinstance(args[0], ConstantVariable):
                items = unpack_iterable(tx, args[0])
                self.items.extend(items)
            else:
                unpack_and_apply_fn(
                    tx, args[0], lambda item: self.call_method(tx, "append", [item], {})
                )

            if len(self.items) > sz:
                tx.output.side_effects.mutation(self)
            return ConstantVariable.create(None)
        elif name == "insert" and self.is_mutable():
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            idx, value = args
            if isinstance(idx, SymNodeVariable):
                const_idx = idx.evaluate_expr()
            else:
                const_idx = idx.as_python_constant()
            tx.output.side_effects.mutation(self)
            # type: ignore[arg-type]
            self.items.insert(const_idx, value)
            return ConstantVariable.create(None)
        elif name == "pop" and self.is_mutable():
            if kwargs or len(args) > 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "at most 1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            if len(self.items) == 0:
                raise_observed_exception(IndexError, tx, args=["pop from empty list"])

            if len(args):
                idx = args[0].as_python_constant()
                if idx >= len(self.items):
                    raise_observed_exception(
                        IndexError, tx, args=["pop index out of range"]
                    )
            tx.output.side_effects.mutation(self)
            return self.items.pop(*[a.as_python_constant() for a in args])
        elif name == "clear" and self.is_mutable():
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return ConstantVariable.create(None)
        elif name == "copy":
            # List copy() doesn't have args and kwargs
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            items_lst: list[VariableTracker] = list(self.items)
            return self.modified(items_lst, mutation_type=ValueMutationNew())
        elif name == "reverse" and self.is_mutable():
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            self.items.reverse()
            tx.output.side_effects.mutation(self)
            return ConstantVariable.create(None)
        elif name == "remove" and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            idx = self.call_method(tx, "index", args, kwargs)
            self.call_method(tx, "pop", [idx], {})
            return ConstantVariable.create(None)
        else:
            return super().call_method(tx, name, args, kwargs)


class ListVariable(CommonListMethodsVariable):
    # PyList_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L3776
    _cpython_type = list

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        return self._seq_richcompare(tx, other, op, list)

    def python_type(self) -> type:
        return list

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)})"

    def debug_repr(self) -> str:
        return self.debug_repr_helper("[", "]")

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        items = ", ".join(tracked_repr(tx, item) for item in self.items)
        return VariableTracker.build(tx, f"[{items}]")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self._contains_self_reference():
            # Self-referential list: create empty, cache, then extend
            codegen.extend_output(
                [
                    create_instruction("BUILD_LIST", arg=0),
                    create_dup_top(),
                ]
            )
            codegen.add_cache(self)

            codegen.foreach(self.items)
            codegen.extend_output(
                [
                    create_instruction("BUILD_LIST", arg=len(self.items)),
                    create_instruction("LIST_EXTEND", arg=1),
                ]
            )
        else:
            # Non-self-referential: use simple codegen
            codegen.foreach(self.items)
            codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Include/internal/pycore_list.h#L55-L59
        return ListIteratorVariable(self.items, mutation_type=ValueMutationNew())

    def sq_inplace_repeat_impl(
        self,
        tx: "InstructionTranslatorBase",
        count: VariableTracker,
    ) -> VariableTracker:
        # list_inplace_repeat: https://github.com/python/cpython/blob/v3.13.13/Objects/listobject.c#L1071
        if not self.is_mutable():
            raise AssertionError(
                f"sq_inplace_repeat_impl reached an immutable {type(self).__name__}; "
                "every construction site should set mutation_type."
            )
        n = count.as_python_constant()
        try:
            new_items = self.items * n
        except (MemoryError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        tx.output.side_effects.mutation(self)
        self.items[:] = new_items
        return self

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "sort" and self.is_mutable():
            if len(args) != 0:
                raise_args_mismatch(tx, name, "0 args", f"{len(args)} args")
            key_fn_var = kwargs.pop("key", ConstantVariable.create(None))
            reverse = kwargs.pop(
                "reverse", ConstantVariable.create(False)
            ).as_python_constant()
            if len(kwargs) != 0:
                raise_args_mismatch(tx, name, "0 kwargs", f"{len(kwargs)} kwargs")

            tx.output.side_effects.mutation(self)
            # CPython's list.sort detaches the list while sorting: reads
            # during the sort observe an empty list, items added during the
            # sort are discarded, and the mutation raises ValueError after
            # the sorted result is installed.
            saved = list(self.items)
            self.items.clear()

            class _TracedKey:
                # Compares through Dynamo so user-defined __lt__ (e.g. from
                # functools.cmp_to_key) is traced like CPython's timsort,
                # which only ever uses "<".
                def __init__(self, key: VariableTracker) -> None:
                    self.key = key

                def __lt__(self, other: "_TracedKey") -> bool:
                    result = variables.BuiltinVariable(operator.lt).call_function(
                        tx, [self.key, other.key], {}
                    )
                    if not result.is_python_constant():
                        unimplemented(
                            gb_type="sort with non-constant keys",
                            context=str(self.key),
                            explanation=(
                                f"Cannot perform sort whose key comparison is not "
                                f"a compile-time constant. "
                                f"Key type: {self.key.python_type()}. "
                                f"Most notably, we cannot sort with Tensor or SymInt "
                                f"keys, but we can sort ints."
                            ),
                            hints=[
                                "Use something else as the key.",
                                *graph_break_hints.SUPPORTABLE,
                            ],
                        )
                    return bool(result.as_python_constant())

            try:
                if key_fn_var.is_constant_none():
                    keys = saved
                else:
                    keys = [key_fn_var.call_function(tx, [x], {}) for x in saved]

                if all(k.is_python_constant() for k in keys):
                    order = sorted(
                        range(len(saved)),
                        key=lambda i: keys[i].as_python_constant(),
                        reverse=reverse,
                    )
                else:
                    order = sorted(
                        range(len(saved)),
                        key=lambda i: _TracedKey(keys[i]),
                        reverse=reverse,
                    )
                new_items = [saved[i] for i in order]
            except Exception as e:
                self.items[:] = saved
                if isinstance(e, (ObservedException, Unsupported)):
                    raise
                raise_observed_exception(type(e), tx, args=list(e.args))
            modified_during_sort = bool(self.items)
            self.items[:] = new_items
            if modified_during_sort:
                raise_observed_exception(
                    ValueError, tx, args=["list modified during sort"]
                )
            return ConstantVariable.create(None)
        return super().call_method(tx, name, args, kwargs)

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # list___init___impl: clear the list, then extend with the optional
        # iterable arg.
        # https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L2966-L2986
        if kwargs or len(args) > 1:
            raise_args_mismatch(
                tx,
                "__init__",
                "at most 1 args and 0 kwargs",
                f"{len(args)} args and {len(kwargs)} kwargs",
            )
        tx.output.side_effects.mutation(self)
        self.items.clear()
        if len(args) == 1:
            self.call_method(tx, "extend", args, {})
        return ConstantVariable.create(None)

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if name == "__class__":
            source = AttrSource(self.source, name) if self.source else None
            class_type = self.python_type()
            if class_type is list:
                return VariableTracker.build(tx, class_type, source=source)
            else:
                return VariableTracker.build(tx, class_type, source)
        return super().tp_getattro_impl(tx, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.python_type() is not list:
            return super().call_obj_hasattr(tx, name)
        return VariableTracker.build(tx, hasattr([], name))

    def sq_ass_item_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L865-L890 (list_ass_item)
        if not self.is_mutable():
            return super().sq_ass_item_impl(tx, key, value)
        # value=None signals delete (CPython NULL sentinel).
        idx = key.nb_index_impl(tx).as_python_constant()
        if not (0 <= idx < len(self.items)):
            raise_observed_exception(
                IndexError, tx, args=["list assignment index out of range"]
            )
        try:
            if value is None:
                self.items.__delitem__(idx)
            else:
                self.items[idx] = value
        except (IndexError, TypeError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        tx.output.side_effects.mutation(self)
        return ConstantVariable.create(None)

    def mp_ass_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L3119-L3150 (list_ass_subscript)
        if not self.is_mutable():
            return super().mp_ass_subscript_impl(tx, key, value)
        # value=None signals delete (CPython NULL sentinel).
        if pyindex_check(key.python_type()):
            i = key.nb_index_impl(tx).as_python_constant()
            if i < 0:
                i += len(self.items)
            return self.sq_ass_item_impl(tx, ConstantVariable(i), value)
        elif pyslice_check(key):
            # CPython runs PySlice_Unpack first, which raises ValueError on
            # step==0 before the iterable check.
            key_as_const = key.as_python_constant()
            if key_as_const.step == 0:
                raise_observed_exception(
                    ValueError, tx, args=["slice step cannot be zero"]
                )
            if value is None:
                # delete slice
                try:
                    self.items.__delitem__(key_as_const)
                except ValueError as exc:
                    raise_observed_exception(
                        ValueError,
                        tx,
                        args=list(exc.args),
                    )
                tx.output.side_effects.mutation(self)
            else:
                # assign slice
                # CPython list_ass_subscript runs PySequence_Fast on value
                # before the step==1 branch, so this message applies to all
                # slice forms.
                if not vt_is_iterable(value):
                    msg = "must assign iterable to extended slice"
                    step = key_as_const.step
                    if step in (None, 1) and _cpython_has_simple_slice_bug():
                        msg = "can only assign an iterable"
                    raise_type_error(tx, msg)

                value_unpack = unpack_iterable(tx, value)
                try:
                    self.items[key_as_const] = value_unpack
                except ValueError as exc:
                    raise_observed_exception(
                        ValueError,
                        tx,
                        args=list(exc.args),
                    )
                tx.output.side_effects.mutation(self)
        else:
            raise_type_error(
                tx,
                f"list indices must be integers or slices, not {key.python_type_name()}",
            )
        return ConstantVariable.create(None)

    def sq_concat_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
    ) -> VariableTracker:
        # Implements PySequence_Concat for sequences
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L725-L773 (list_concat)
        if not pylist_check(other):
            raise_type_error(
                tx,
                f"can only concatenate list (not '{other.python_type_name()}') to list",
            )

        items = self.items + unpack_iterable(tx, other)
        return ListVariable(items, mutation_type=ValueMutationNew())

    def sq_inplace_concat_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
    ) -> VariableTracker:
        # Implements PySequence_InPlaceConcat for sequences
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L1464-L1472
        self.call_method(tx, "extend", [other], {})
        return self

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        from ..exc import raise_type_error

        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")


_deque_state_mutating_methods = frozenset(
    {
        "append",
        "appendleft",
        "extend",
        "extendleft",
        "pop",
        "popleft",
        "insert",
        "remove",
        "clear",
    }
)


# TODO(dynamo-team): Split deque from CommonListMethodsVariable / ListVariable
class DequeVariable(CommonListMethodsVariable):
    # deque_spec: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L1866
    # tp_hash = PyObject_HashNotImplemented (unhashable)
    _cpython_type = collections.deque

    _nonvar_fields = {
        "state",
        *CommonListMethodsVariable._nonvar_fields,
    }

    # deque exposes a single read-only getset descriptor, maxlen.
    tp_getset = {
        "maxlen": GetSet(lambda self, tx: self.maxlen, None),
    }

    def _rotate(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        # deque_rotate: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L927
        # rotate(n=1, /) shifts n steps to the right (left if n < 0). n is
        # resolved through nb_index. A deque of length <= 1 is a no-op. A
        # non-mutable deque declines (returns None) so dispatch falls through
        # to the object protocol, matching the old is_mutable() guard.
        if not self.is_mutable():
            return None
        # deque.rotate is METH_FASTCALL, so the flags derived from ml_flags
        # reject kwargs but cannot bound the positional count; deque_rotate
        # does that itself with _PyArg_CheckPositional.
        check_positional(tx, "rotate", len(args), 0, 1)
        n_vt = args[0] if args else ConstantVariable.create(1)
        n = n_vt.nb_index_impl(tx).as_python_constant()
        length = len(self.items)
        if length > 1:
            tx.output.side_effects.mutation(self)
            k = n % length
            if k != 0:
                self.items[:] = self.items[-k:] + self.items[:-k]
            self.state += 1
        return ConstantVariable.create(None)

    tp_methods = {"rotate": Method(_rotate)}

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        return self._seq_richcompare(tx, other, op, collections.deque)

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        from ..exc import raise_type_error

        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    @staticmethod
    def validate_maxlen(
        tx: "InstructionTranslatorBase", maxlen: VariableTracker
    ) -> None:
        # deque_init: maxlenobj != Py_None is run through PyLong_AsSsize_t
        # https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L1729-L1736
        if isinstance(maxlen, ConstantVariable) and maxlen.value is None:
            return
        if pylong_as_ssize_t(tx, maxlen) < 0:
            raise_observed_exception(
                ValueError, tx, args=["maxlen must be non-negative"]
            )

    def __init__(
        self,
        items: list[VariableTracker],
        maxlen: VariableTracker | None = None,
        **kwargs: Any,
    ) -> None:
        if maxlen is None:
            maxlen = ConstantVariable.create(None)
        if not maxlen.is_python_constant():
            raise AssertionError(
                f"maxlen must be a constant, got: {maxlen.debug_repr()}"
            )
        self.maxlen = maxlen
        items = list(items)
        if self.maxlen.as_python_constant() is not None:
            items = items[-maxlen.as_python_constant() :]
        super().__init__(items, **kwargs)
        # Mirrors CPython deque->state: bumped on every structural mutation so
        # deque iterators can detect mutation during iteration.
        self.state = 0

    def python_type(self) -> type:
        return collections.deque

    def sq_item_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # deque_item: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L1888
        # CPython's sq_item takes Py_ssize_t (already int from generic_getitem's
        # nb_index_impl).  deque has no mp_subscript, so this is the real path.
        index = key.as_python_constant()
        try:
            return self.items[index]
        except IndexError:
            raise_observed_exception(IndexError, tx, args=["deque index out of range"])

    def sq_ass_item_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
        value: VariableTracker | None,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c (deque_ass_item)
        if not self.is_mutable():
            return super().sq_ass_item_impl(tx, key, value)
        # value=None signals delete (CPython NULL sentinel).
        idx = key.nb_index_impl(tx).as_python_constant()
        length = len(self.items)
        if idx < 0:
            idx += length
        if not (0 <= idx < length):
            raise_observed_exception(IndexError, tx, args=["deque index out of range"])
        tx.output.side_effects.mutation(self)
        if value is None:
            # deque_ass_item_lock_held bumps deque->state only on the delete
            # path (deque_del_item); the set path (Py_SETREF) does not bump.
            self.state += 1
            self.items.__delitem__(idx)
        else:
            self.items[idx] = value
        return ConstantVariable.create(None)

    def sq_concat_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
    ) -> VariableTracker:
        # Implements PySequence_Concat for deque
        # ref: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L691-L699

        if not issubclass(other.python_type(), collections.deque):
            raise_type_error(
                tx,
                f"can only concatenate deque (not '{other.python_type_name()}') to deque",
            )

        return DequeVariable(
            self.items + unpack_iterable(tx, other),
            maxlen=self.maxlen,
            mutation_type=ValueMutationNew(),
        )

    def sq_inplace_concat_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
    ) -> VariableTracker:
        # Implements PySequence_InPlaceConcat for deque
        # ref: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L572-L584
        self.call_method(tx, "extend", [other], {})
        return self

    def sq_inplace_repeat_impl(
        self,
        tx: "InstructionTranslatorBase",
        count: VariableTracker,
    ) -> VariableTracker:
        # deque_inplace_repeat: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L811
        if not self.is_mutable():
            raise AssertionError(
                f"sq_inplace_repeat_impl reached an immutable {type(self).__name__}; "
                "every construction site should set mutation_type."
            )
        n = count.as_python_constant()
        try:
            new_items = self.items * n
        except (MemoryError, OverflowError) as e:
            raise_observed_exception(type(e), tx, args=list(e.args))
        # A bounded deque drops from the left, keeping the last maxlen items
        # (maxlen == 0 yields an empty deque).
        maxlen = self.maxlen.as_python_constant()
        if maxlen is not None:
            new_items = new_items[-maxlen:] if maxlen else new_items[:0]
        tx.output.side_effects.mutation(self)
        self.items[:] = new_items
        return self

    def debug_repr(self) -> str:
        if self.maxlen.as_python_constant() is None:
            return self.debug_repr_helper("deque([", "])")
        return self.debug_repr_helper(
            "deque([", "], maxlen=" + self.maxlen.debug_repr() + ")"
        )

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        items = ", ".join(tracked_repr(tx, item) for item in self.items)
        if self.maxlen.as_python_constant() is None:
            return VariableTracker.build(tx, f"{self.python_type_name()}([{items}])")
        return VariableTracker.build(
            tx,
            f"{self.python_type_name()}([{items}], maxlen={tracked_repr(tx, self.maxlen)})",
        )

    def as_python_constant(self) -> collections.deque[Any]:
        return self.python_type()(
            [x.as_python_constant() for x in self.items],
            maxlen=self.maxlen.as_python_constant(),
        )

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # To deal with self-referential sets
        codegen.add_push_null(
            lambda: codegen.append_output(
                codegen.create_load_python_module(collections.deque)  # type: ignore[arg-type]
            )
        )
        codegen.append_output(create_instruction("BUILD_LIST", arg=0))
        codegen(self.maxlen)
        codegen.extend_output(
            [
                *codegen.create_call_function_kw(2, ("maxlen",), False),
                create_dup_top(),
            ]
        )
        codegen.add_cache(self)

        codegen.append_output(create_dup_top())
        codegen.load_method("extend")

        codegen.foreach(self.items)
        codegen.extend_output(
            [
                create_instruction("BUILD_LIST", arg=len(self.items)),
                *create_call_method(1),
                create_instruction("POP_TOP"),
            ]
        )

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # __init__ resets maxlen, so it must bypass the trailing maxlen clamp
        # below (which uses the pre-call maxlen captured here).
        if name == "__init__":
            return self.tp_init_impl(tx, args, kwargs)

        if name == "__setattr__":
            # deque has no __dict__, so every attribute write raises. A name
            # backed by a getset descriptor (maxlen) is read-only; anything else
            # is simply absent.
            attr_name = args[0].as_python_constant()
            if self.lookup_tp_getset_member(attr_name) is not None:
                msg = (
                    f"attribute '{attr_name}' of 'collections.deque' objects "
                    "is not writable"
                )
            else:
                msg = (
                    f"'collections.deque' object has no attribute '{attr_name}' "
                    "and no __dict__ for setting new attributes"
                )
            raise_observed_exception(AttributeError, tx, args=[msg])

        if name == "__reversed__":
            # deque.__reversed__ returns a _deque_reverse_iterator that snapshots
            # the current state and detects mutation during iteration.
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return DequeReverseIteratorVariable(
                list(reversed(self.items)),
                self,
                self.state,
                mutation_type=ValueMutationNew(),
            )

        maxlen = self.maxlen.as_python_constant()
        if maxlen is not None:
            slice_within_maxlen = slice(-maxlen, None)
        else:
            slice_within_maxlen = None

        if name in ("copy", "__copy__"):
            # deque_copy preserves maxlen: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L890
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return DequeVariable(
                list(self.items),
                maxlen=self.maxlen,
                mutation_type=ValueMutationNew(),
            )

        pre_len = len(self.items)

        if name == "extendleft" and self.is_mutable() and len(args) > 0:
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            # NOTE this is inefficient, but the alternative is to represent self.items
            # as a deque, which is a more intrusive change.
            unpack_and_apply_fn(
                tx, args[0], lambda item: self.call_method(tx, "appendleft", [item], {})
            )
            slice_within_maxlen = slice(None, maxlen)
            result = ConstantVariable.create(None)
        elif name == "popleft" and self.is_mutable():
            if kwargs or len(args) > 0:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            tx.output.side_effects.mutation(self)
            result, *self.items[:] = self.items
        elif name == "appendleft" and len(args) > 0 and self.is_mutable():
            if kwargs or len(args) != 1:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            tx.output.side_effects.mutation(self)
            self.items[:] = [args[0], *self.items]
            slice_within_maxlen = slice(None, maxlen)
            result = ConstantVariable.create(None)
        elif name == "insert" and len(args) > 0 and self.is_mutable():
            if kwargs or len(args) != 2:
                raise_args_mismatch(
                    tx,
                    name,
                    "2 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            if maxlen is not None and len(self.items) == maxlen:
                raise_observed_exception(
                    IndexError, tx, args=["deque already at its maximum size"]
                )
            result = super().call_method(tx, name, args, kwargs)
        else:
            result = super().call_method(tx, name, args, kwargs)

        # Capture growth before the maxlen clamp: extend/extendleft appended an
        # item iff the length grew here (the clamp may shrink it back).
        extend_appended = len(self.items) > pre_len
        if (
            slice_within_maxlen is not None
            and maxlen is not None
            and len(self.items) > maxlen
        ):
            self.items[:] = self.items[slice_within_maxlen]
        if name in _deque_state_mutating_methods and self.is_mutable():
            # extend/extendleft with an empty iterable append nothing -> no bump.
            if name not in ("extend", "extendleft") or extend_appended:
                self.state += 1
        return result

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.python_type() is collections.deque:
            return VariableTracker.build(tx, name in collections.deque.__dict__)
        return super().call_obj_hasattr(tx, name)

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # deque_init: https://github.com/python/cpython/blob/v3.13.0/Modules/_collectionsmodule.c#L1810
        # Re-init resets maxlen (to None unless given), clears the deque,
        # then extends with the iterable.
        if not self.is_mutable():
            return super().tp_init_impl(tx, args, kwargs)
        iterable = args[0] if len(args) >= 1 else kwargs.pop("iterable", None)
        new_maxlen = (
            args[1]
            if len(args) >= 2
            else kwargs.pop("maxlen", ConstantVariable.create(None))
        )
        if len(args) > 2 or kwargs:
            raise_args_mismatch(tx, "__init__")
        self.validate_maxlen(tx, new_maxlen)
        tx.output.side_effects.mutation(self)
        self.state += 1
        self.maxlen = new_maxlen
        self.items.clear()
        if iterable is not None:
            self.call_method(tx, "extend", [iterable], {})
        return ConstantVariable.create(None)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Modules/_collectionsmodule.c#L1886-L1904
        return DequeIteratorVariable(
            list(self.items),
            self,
            self.state,
            mutation_type=ValueMutationNew(),
        )


class TupleVariable(BaseListVariable):
    # PyTuple_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/tupleobject.c#L846
    _cpython_type = tuple

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        return self._seq_richcompare(tx, other, op, tuple)

    def python_type(self) -> type[tuple]:  # type: ignore[type-arg]
        return tuple

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)})"

    def debug_repr(self) -> str:
        if len(self.items) == 1:
            return self.debug_repr_helper("(", ",)")
        return self.debug_repr_helper("(", ")")

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        items = ", ".join(tracked_repr(tx, item) for item in self.items)
        if len(self.items) == 1:
            items += ","
        return VariableTracker.build(tx, f"({items})")

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/tupleobject.c#L1101-L1117
        return TupleIteratorVariable(self.items, mutation_type=ValueMutationNew())

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_build_tuple(len(self.items)))

    def reconstruct_pycode(self, codegen):
        if len(self.items) == 0:
            return "()"
        else:
            return f"({', '.join([item.reconstruct_pycode(codegen) for item in self.items])},)"

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if name == "__class__":
            source = AttrSource(self.source, name) if self.source else None
            class_type = self.python_type()
            return VariableTracker.build(tx, class_type, source=source)
        return super().tp_getattro_impl(tx, name)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.python_type() is not tuple:
            return super().call_obj_hasattr(tx, name)
        return VariableTracker.build(tx, hasattr((), name))

    def sq_concat_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
    ):
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/tupleobject.c#L441-L486 (tuple_concat)
        if len(self.items) == 0 and pytuple_checkexact(other):
            return other

        if not pytuple_check(other):
            raise_type_error(
                tx,
                f"can only concatenate tuple (not '{other.python_type_name()}') to tuple",
            )

        return TupleVariable(
            self.items + unpack_iterable(tx, other), mutation_type=ValueMutationNew()
        )

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # CPython tuplehash: https://github.com/python/cpython/blob/e76aa128fe/Objects/tupleobject.c#L321
        # Collect per-item hashes and feed them through tuplehash.  We wrap
        # each hash in RawHash to bypass int.__hash__'s modular reduction,
        # so tuplehash sees the original hash values — matching CPython exactly.
        from .hashable import RawHash
        from .object_protocol import generic_hash_impl

        is_fake = False
        raw_hashes = []
        for item in self.items:
            h, fake = generic_hash_impl(tx, item)
            is_fake = is_fake or fake
            raw_hashes.append(RawHash(h))
        return hash(tuple(raw_hashes)), is_fake


class SizeVariable(TupleVariable):
    """torch.Size(...)"""

    _cpython_type = torch.Size

    _nonvar_fields = {
        "proxy",
        *TupleVariable._nonvar_fields,
    }

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # torch.Size items may be TensorVariables wrapping scalar tensors with
        # known constant values (node.meta["example_value"].constant).  Extract
        # the actual int values so the hash matches eager torch.Size hashing.
        from .tensor import TensorVariable

        int_items: list[int] = []
        for item in self.items:
            if item.is_python_constant():
                int_items.append(item.as_python_constant())
                continue
            if isinstance(item, TensorVariable):
                proxy = getattr(item, "proxy", None)
                node = getattr(proxy, "node", None)
                meta = getattr(node, "meta", None) if node is not None else None
                example_value = (
                    meta.get("example_value") if isinstance(meta, dict) else None
                )
                constant = getattr(example_value, "constant", None)
                if isinstance(constant, torch.Tensor) and constant.numel() == 1:
                    int_items.append(int(constant.item()))
                    continue
            # Can't extract a concrete int — fall back to TupleVariable
            return super().hash_impl(tx)
        return hash(torch.Size(int_items)), False

    def __init__(
        self,
        items: list[VariableTracker],
        proxy: torch.fx.Proxy | None = None,
        **kwargs: Any,
    ) -> None:
        self.proxy = proxy
        super().__init__(items, **kwargs)

    def debug_repr(self) -> str:
        return self.debug_repr_helper("torch.Size([", "])")

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        items = ", ".join(tracked_repr(tx, item) for item in self.items)
        return VariableTracker.build(tx, f"torch.Size([{items}])")

    def python_type(self) -> type:
        return torch.Size

    def as_proxy(self) -> Any:
        if self.proxy is not None:
            return self.proxy

        # torch.Size needs special handling.  Normally, we pun a list-like
        # container to directly contain Proxy/Node objects from FX, and FX
        # knows to look inside containers (via map_aggregate).  But torch.Size
        # is weird; although it subclasses from tuple, it doesn't allow
        # members which aren't int-like (rejecting Proxy and Node).  This
        # means we can't use the normal representation trick
        # torch.Size([proxy0, proxy1]).  I looked into seeing if I could
        # relax torch.Size in PyTorch proper, but if torch.Size constructor
        # sees a type that it doesn't recognize, it will try to call
        # __index__() on it, so there is no BC way to actually change this
        # behavior (though it occurs to me that I could have just added a
        # YOLO no checking alternate constructor.)
        #
        # To work around this problem, I represent a torch.Size proxy as
        # a straight up proxy, that would have been constructed by taking
        # the constituent proxies as arguments.  This trick can be generally
        # used for any construct that we need a proxy for but we can't
        # directly represent as an aggregate; I don't see very many examples
        # of this in torchdynamo though!

        # Look for a proxy.  If there are none, do the legacy behavior
        tracer = None
        proxies = self._as_proxy()
        for proxy in proxies:
            if isinstance(proxy, torch.fx.Proxy):
                tracer = proxy.tracer
                break

        if tracer is None:
            return torch.Size(proxies)

        proxy = tracer.create_proxy("call_function", torch.Size, (proxies,), {})
        set_example_value(
            proxy.node,
            torch.Size(
                [
                    p.node.meta["example_value"] if not isinstance(p, int) else p
                    for p in proxies
                ]
            ),
        )
        return proxy

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(lambda: codegen.load_import_from("torch", "Size"))
        codegen.foreach(self.items)
        build_torch_size = [
            create_build_tuple(len(self.items)),
        ] + create_call_function(1, False)
        codegen.extend_output(build_torch_size)

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        return list(self.items)

    def numel(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .builder import SourcelessBuilder
        from .tensor import SymNodeVariable

        const_result = 1
        # pyrefly: ignore [implicit-any]
        sym_sizes = []

        for v in self.items:
            if v.is_python_constant():
                const_result *= v.as_python_constant()
            else:
                if not isinstance(v, SymNodeVariable):
                    raise AssertionError(
                        f"expected SymNodeVariable, got {type(v).__name__}"
                    )
                # Delay proxy calls  until we know it will be necessary
                sym_sizes.append(v)

        result = VariableTracker.build(tx, const_result)
        if sym_sizes and const_result == 1:
            # Skip multiplying by 1
            result, *sym_sizes = sym_sizes

        if not sym_sizes or const_result == 0:
            return result

        mul = SourcelessBuilder.create(tx, operator.mul)
        for v in sym_sizes:
            result = mul.call_function(tx, [result, v], {})
        return result

    def mp_subscript_impl(
        self,
        tx: "InstructionTranslatorBase",
        key: VariableTracker,
    ) -> VariableTracker:
        # tuple_subscript: https://github.com/python/cpython/blob/62a6e898e01/Objects/tupleobject.c#L877-L930
        try:
            key_type = key.python_type()
        except NotImplementedError:
            key_type = None
        if key_type not in (int, bool, slice):
            if key_type is not None and not type_implements_nb_index(key_type):
                raise_observed_exception(
                    TypeError,
                    tx,
                    args=[
                        f"tuple indices must be integers or slices, not {key.python_type_name()}"
                    ],
                )
            key = key.nb_index_impl(tx)
        return self.get_item_dyn(tx, key)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "numel":
            if args or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "0 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            return self.numel(tx)

        return super().call_method(tx, name, args, kwargs)

    def get_item_dyn(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        from .tensor import SymNodeVariable, TensorVariable

        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        elif isinstance(arg, TensorVariable):
            value = get_fake_value(arg.as_proxy().node, tx)
            if value.constant is None or value.constant.numel() != 1:
                unimplemented(
                    gb_type="Indexing torch.Size with non-scalar tensor",
                    context=f"get_item_dyn {self} {arg}",
                    explanation=(
                        "Attempted to index torch.Size with a tensor that is not a scalar constant."
                    ),
                    hints=[*graph_break_hints.USER_ERROR],
                )
            index = value.constant.item()
        else:
            index = arg.as_python_constant()

        if isinstance(index, slice):
            return SizeVariable(self.items[index])
        else:
            if not isinstance(index, (int, torch.SymInt)):
                raise AssertionError(
                    f"index must be int or SymInt, got {type(index).__name__}"
                )
            return self.items[index]

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        return VariableTracker.build(tx, hasattr(torch.Size, name))

    def sq_concat_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        """
        Implements torch.Size concatenation via sq_concat protocol.
        torch.Size accepts concatenation with any tuple-like object.
        Ref: torch/csrc/Size.cpp::THPSize_concat
        """
        if not pytuple_check(other):
            raise_type_error(
                tx,
                f"can only concatenate tuple (not '{other.python_type_name()}') to torch.Size",
            )
        # Size nb_add uses sq_concat. We do the opposite here because of the reverse keyword
        return self.nb_add_impl(tx, other)

    def nb_add_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # NOTE: The python interpreter tries, in order:
        #    1. right.nb_add(left, right)  (only if right is a subclass of left)
        #    2. left.nb_add(left, right)
        #    3. right.nb_add(left, right)
        #    4. left.sq_concat(right)
        #  Hence, to support tuple + size -> size, we need to implement nb_add
        if not pytuple_check(other):
            return ConstantVariable(NotImplemented)
        self_, other_ = (other, self) if reverse else (self, other)
        a, b = unpack_iterable(tx, self_), unpack_iterable(tx, other_)
        return SizeVariable(list(a) + list(b), mutation_type=ValueMutationNew())


class SliceVariable(VariableTracker):
    # PySlice_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/sliceobject.c#L689
    _cpython_type = slice

    def __init__(
        self,
        items: list[VariableTracker],
        tx: Optional["InstructionTranslatorBase"] = None,
        **kwargs: Any,
    ) -> None:
        items_to_map = items
        start, stop, step = [variables.ConstantVariable.create(None)] * 3

        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError

        # Convert TensorVariable to SymIntVariable by calling .item()
        # This decomposes a[:t] to u=t.item(); a[:u] at the dynamo level
        if start.is_tensor():
            if tx is None:
                raise AssertionError(
                    "tx is required when slice indices are TensorVariables"
                )
            start = start.call_method(tx, "item", [], {})
        if stop.is_tensor():
            if tx is None:
                raise AssertionError(
                    "tx is required when slice indices are TensorVariables"
                )
            stop = stop.call_method(tx, "item", [], {})
        if step.is_tensor():
            if tx is None:
                raise AssertionError(
                    "tx is required when slice indices are TensorVariables"
                )
            step = step.call_method(tx, "item", [], {})

        self.items = (start, stop, step)

        super().__init__(**kwargs)

    def debug_repr(self) -> str:
        return "slice(" + ", ".join(i.debug_repr() for i in self.items) + ")"

    def as_proxy(self) -> slice:
        return slice(*[x.as_proxy() for x in self.items])

    def python_type(self) -> type:
        return slice

    def as_python_constant(self) -> slice:
        return slice(*[guard_if_dyn(x) for x in self.items])

    def as_index_slice(self, tx: "InstructionTranslatorBase") -> slice:
        # PySlice_Unpack -> evaluate_slice_index: each non-None member is
        # coerced through __index__ (nb_index), so user objects with an
        # __index__ method are valid slice bounds.
        # https://github.com/python/cpython/blob/62a6e898e01/Objects/sliceobject.c#L196-L242
        members = []
        for member in self.items:
            if isinstance(member, ConstantVariable) and member.value is None:
                members.append(None)
                continue
            if not pyindex_check(maybe_get_python_type(member)):
                raise_type_error(
                    tx,
                    "slice indices must be integers or None or have an __index__ method",
                )
            members.append(pynumber_index(tx, member).as_python_constant())
        return slice(*members)

    def is_hashable(self) -> bool:
        # Slices became hashable in Python 3.12 (CPython slicehash).
        return sys.version_info >= (3, 12)

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        if sys.version_info < (3, 12):
            raise_type_error(tx, "unhashable type: 'slice'")
        # CPython slicehash: https://github.com/python/cpython/blob/e76aa128fe/Objects/sliceobject.c#L667
        s = self.as_python_constant()
        try:
            h = hash(s)
        except TypeError as e:
            raise_observed_exception(TypeError, tx, args=[str(e)])
        return h, False

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """slice_richcompare: delegates all 6 ops to (start, stop, step) tuple comparison.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/sliceobject.c#L667-L686
        """
        from .object_protocol import generic_richcompare

        if not isinstance(other, SliceVariable):
            return ConstantVariable.create(NotImplemented)
        self_tuple = TupleVariable(list(self.items))
        other_tuple = TupleVariable(list(other.items))
        return generic_richcompare(tx, self_tuple, other_tuple, op)

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_SLICE", arg=len(self.items)))

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if name in cmp_name_to_op_mapping or name in ("__hash__", "indices"):
            return variables.GetAttrVariable(
                self, name, py_type=type(getattr(slice, name))
            )
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(
                gb_type="Unsupported attribute for slice() object",
                context=f"tp_getattro_impl {self} {name}",
                explanation=f"Expected attribute to be one of {','.join(fields)} "
                f"but got {name}",
                hints=[*graph_break_hints.USER_ERROR],
            )
        return self.items[fields.index(name)]

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "indices":
            if len(args) != 1 or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            length_var = args[0].nb_index_impl(tx)
            length = length_var.as_python_constant()
            items: list[int | None] = []
            for item in self.items:
                if item.is_constant_none():
                    items.append(None)
                else:
                    idx_var = item.nb_index_impl(tx)
                    items.append(idx_var.as_python_constant())

            try:
                result = slice(*items).indices(length)
            except (ValueError, TypeError) as e:
                raise_observed_exception(type(e), tx, args=list(e.args))
            return VariableTracker.build(tx, result)

        return super().call_method(tx, name, args, kwargs)


class ListIteratorVariable(IteratorVariable):
    # PyListIter_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/listobject.c#L3842
    _cpython_type = type(iter([]))

    _nonvar_fields = {
        "index",
        *IteratorVariable._nonvar_fields,
    }

    def __init__(
        self, items: list[VariableTracker], index: int = 0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(items, list):
            raise AssertionError(f"items must be a list, got {type(items).__name__}")
        # Removing this check as it slows things down too much
        # https://github.com/pytorch/pytorch/pull/87533#issuecomment-1287574492

        # assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index
        self.is_exhausted = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={len(self.items)}, index={repr(self.index)})"

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/6280bb547840b609feedb78887c6491af75548e8/Objects/listobject.c#L4110-L4133
        if not self.is_mutable():
            raise AssertionError("ListIteratorVariable must be mutable to iterate")
        old_index = self.index
        if old_index >= len(self.items) or self.is_exhausted:
            self.is_exhausted = True
            raise_observed_exception(StopIteration, tx)

        tx.output.side_effects.mutation(self)
        self.index += 1
        return self.items[old_index]

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        return VariableTracker.build(tx, hasattr(iter([]), name))

    def python_type(self) -> type:
        return type(iter([]))

    def as_python_constant(self) -> Any:
        if self.index > 0:
            raise NotImplementedError
        return iter([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        if self.is_exhausted:
            return []
        self.is_exhausted = True
        return list(self.items[self.index :])

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # starting in 3.15 GET_ITER creates virtual iterators (see https://github.com/python/cpython/issues/145668), so use builtin iter instead
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(iter))  # type: ignore[arg-type]
        )
        if not self.is_exhausted:
            remaining_items = self.items[self.index :]
        else:
            # pyrefly: ignore [implicit-any]
            remaining_items = []
        codegen.foreach(remaining_items)
        codegen.append_output(create_build_tuple(len(remaining_items)))
        codegen.extend_output(create_call_function(1, False))


class TupleIteratorVariable(ListIteratorVariable):
    # PyTupleIter_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/tupleobject.c#L1067
    _cpython_type = type(iter(()))


class DequeIteratorVariable(ListIteratorVariable):
    _cpython_type = type(iter(collections.deque()))

    _nonvar_fields = {
        "saved_state",
        *ListIteratorVariable._nonvar_fields,
    }

    def __init__(
        self,
        items: list[VariableTracker],
        source_deque: "DequeVariable",
        saved_state: int,
        index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(items, index=index, **kwargs)
        self.source_deque = source_deque
        self.saved_state = saved_state

    def _check_mutation(self, tx: "InstructionTranslatorBase") -> None:
        if self.source_deque.state != self.saved_state:
            raise_observed_exception(
                RuntimeError, tx, args=["deque mutated during iteration"]
            )

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        self._check_mutation(tx)
        return super().tp_iternext_impl(tx)

    def python_type(self) -> type:
        return type(iter(collections.deque()))


class DequeReverseIteratorVariable(DequeIteratorVariable):
    _cpython_type = type(reversed(collections.deque()))

    def python_type(self) -> type:
        return type(reversed(collections.deque()))


class RangeIteratorVariable(IteratorVariable):
    # PyRangeIter_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/rangeobject.c#L896
    _cpython_type = type(iter(range(0)))

    def __init__(
        self, start: int, stop: int, step: int, len_: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.step = step
        self.len = len_

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        """Range iterators are their own iterator."""
        return self

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        if self.python_type() is range_iterator:
            ri = iter(range(0))
            return VariableTracker.build(tx, hasattr(ri, name))
        return super().call_obj_hasattr(tx, name)

    def tp_iternext_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/rangeobject.c#L1072-L1091
        if self.len <= 0:
            raise_observed_exception(StopIteration, tx)

        self.len -= 1
        current = self.start
        self.start += self.step
        return VariableTracker.build(tx, current)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__setstate__":
            # rangeiter_setstate clamps the arg against the current remaining
            # length, then advances: r->start += arg*step; r->len -= arg.
            # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/rangeobject.c#L1093-L1107
            if len(args) != 1 or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "1 args and 0 kwargs",
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )
            arg = args[0].as_python_constant()
            index = min(max(arg, 0), self.len)
            self.start += index * self.step
            self.len -= index
            return ConstantVariable.create(None)
        elif name == "__length_hint__":
            # rangeiter_len: remaining items.
            # ref: https://github.com/python/cpython/blob/v3.13.3/Objects/rangeobject.c#L1109-L1115
            return ConstantVariable.create(self.len)
        return super().call_method(tx, name, args, kwargs)

    def python_type(self) -> type:
        return range_iterator

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # starting in 3.15 GET_ITER creates virtual iterators (see https://github.com/python/cpython/issues/145668), so use builtin iter instead
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(iter))  # type: ignore[arg-type]
        )
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(range))  # type: ignore[arg-type]
        )
        codegen.append_output(codegen.create_load_const(self.start))
        codegen.append_output(codegen.create_load_const(self.stop))
        codegen.append_output(codegen.create_load_const(self.step))
        codegen.extend_output(create_call_function(3, False))
        codegen.extend_output(create_call_function(1, False))
<|endoftext|>"""
This module implements variable tracking for torch functions and operations during Dynamo tracing.

It provides classes to handle different types of torch operations:

TorchInGraphFunctionVariable: Handles torch.* functions that should be captured in the FX graph.
Provides special handling for constant folding, tensor methods, and torch function overrides.
Manages complex cases like out= variants and parameter construction.

TorchCtxManagerClassVariable: Handles torch context managers like torch.no_grad(), autocast, etc.
Provides implementations for entering/exiting these contexts during tracing.

DispatchKeySetVariable: Represents torch.DispatchKeySet for managing dispatch keys and
device-specific operations during tracing.

The module includes special handling for:
- Constant folding of pure functions
- Tensor method calls
- torch.nn.Parameter construction
- __torch_function__ overrides
- Context manager state tracking
- Device and dtype management

This is a core part of Dynamo's tracing system, translating torch operations into
traceable graph nodes while preserving correct semantics and handling edge cases.
"""

import enum
import functools
import inspect
import logging
import math
import re
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import Any, cast, NoReturn, TYPE_CHECKING, TypeVar, Union
from typing_extensions import TypeIs

import torch._C
import torch._refs
import torch.fx
import torch.nn
import torch.utils._pytree as _pytree
from torch._C import DispatchKeySet
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.streams import StreamVariable
from torch._dynamo.variables.torch_function import TorchFunctionModeVariable
from torch._guards import Guard, Source, TracingContext
from torch._logging import warning_once
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.autograd.graph import GradientEdge
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import config, graph_break_hints, polyfills, variables
from ..codegen import PyCodegen
from ..create_parameter_op import (
    can_convert_to_tracable_parameter,
    new_parameter_placeholder,
    tracable_create_parameter,
)
from ..device_interface import get_registered_device_interfaces
from ..exc import (
    raise_observed_exception,
    raise_type_error,
    unimplemented,
    UserError,
    UserErrorType,
)
from ..guards import (
    _COW_TENSOR_UNSUPPORTED,
    _try_is_cow_tensor,
    GuardBuilder,
    install_guard,
)
from ..source import (
    AttrSource,
    CallFunctionNoArgsSource,
    GlobalStateSource,
    ImportSource,
    SyntheticLocalSource,
)
from ..utils import (
    _is_tensorify_enabled,
    check_unspec_or_constant_args,
    guard_if_dyn,
    has_torch_function,
    hashable,
    is_wrapper_or_member_descriptor,
    product,
    proxy_args_kwargs,
    unpack_iterable,
    unwrap_if_wrapper,
)
from .base import typestr, VariableTracker
from .ctx_manager import (
    AutocastModeVariable,
    ProfilerContextVariable,
    ProfilerRecordFunctionContextVariable,
    TorchFunctionDisableVariable,
)
from .distributed import DistributedVariable
from .functions import (
    bind_args_cached,
    ClosureConversionError,
    NestedUserFunctionVariable,
    UserFunctionVariable,
)
from .lists import ListVariable, TupleVariable
from .object_protocol import vt_is_iterable
from .script_object import CustomClassObjectVariable
from .torch_function import (
    can_dispatch_torch_function,
    dispatch_torch_function,
    TensorWithTFOverrideVariable,
    TorchFunctionModeStackVariable,
)
from .user_defined import UserDefinedTupleVariable


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    from torch.distributed.fsdp._fully_shard import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from torch._custom_class_base import CustomClassBase
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch.utils._pytree import TreeSpec


V = TypeVar("V")
T = TypeVar("T")

log = logging.getLogger(__name__)


def _is_supported_out_tensor_layout(
    fake_out: torch.Tensor, *, false_if_dde: bool
) -> bool:
    # Functionalization can preserve standard dense memory formats, but not
    # arbitrary non-contiguous view strides.
    return (
        torch._prims_common.is_contiguous(fake_out, false_if_dde=false_if_dde)
        or torch._prims_common.is_contiguous_for_memory_format(
            fake_out,
            memory_format=torch.channels_last,
            false_if_dde=false_if_dde,
        )
        or torch._prims_common.is_contiguous_for_memory_format(
            fake_out,
            memory_format=torch.channels_last_3d,
            false_if_dde=false_if_dde,
        )
    )


supported_ctx_manager_classes = dict.fromkeys(
    [
        torch.profiler.profiler.profile,
        torch.autograd.forward_ad._set_fwd_grad_enabled,
        torch.autograd.forward_ad.dual_level,
        torch.autograd.profiler.profile,
        torch.autograd.profiler.record_function,
        torch._C.DisableTorchFunctionSubclass,
        torch._C.DisableTorchFunction,
        torch._functorch.vmap.vmap_increment_nesting,
        torch._functorch.eager_transforms.grad_increment_nesting,
        torch._functorch.eager_transforms.jvp_increment_nesting,
        torch._functorch.eager_transforms.enable_inplace_requires_grad,
        torch.amp.autocast_mode.autocast,
        torch.autograd.grad_mode.enable_grad,
        torch.autograd.grad_mode.inference_mode,
        torch.autograd.grad_mode.no_grad,
        torch.autograd.grad_mode.set_grad_enabled,
        torch.autograd.graph.disable_saved_tensors_hooks,
        torch.cpu.amp.autocast_mode.autocast,
        torch.cuda.amp.autocast_mode.autocast,
        torch.cuda.use_mem_pool,
        torch.cuda.use_mem_pool.__wrapped__,  # type: ignore[attr-defined]
        torch.fx.traceback.annotate,
        torch.fx.traceback.annotate.__wrapped__,  # type: ignore[attr-defined]
        # We'll let Dynamo inline into the contextlib part of these context
        # manager instances, all the way till it invokes the wrapped function
        # itself (at which point we wrap it back to special context manager
        # VTs).
        #
        # This allows us to support calling functions decorated with these
        # context managers, without much extra effort or code dup.
        torch.nn.attention.sdpa_kernel.__wrapped__,  # type: ignore[attr-defined]
    ]
)


REWRITE_OPS_TO_TENSOR_SIZE_METHOD = dict.fromkeys(
    [
        torch._shape_as_tensor,
    ]
)

constant_fold_functions_need_guards = [
    torch._C._functorch.get_dynamic_layer_stack_depth,
    torch.accelerator.current_device_index,
    torch.accelerator.current_accelerator,
    torch.cuda.current_device,
    torch.cuda.is_initialized,
    torch.xpu.current_device,
    torch.xpu.is_initialized,
    torch.__future__.get_overwrite_module_params_on_conversion,
]

constant_fold_functions = [
    torch._assert,
    torch._utils._get_device_index,
    torch._C._get_cublas_allow_tf32,
    torch._C._is_any_autocast_enabled,
    torch.accelerator.is_available,
    torch.backends.mps.is_available.__wrapped__,  # type: ignore[attr-defined]
    torch.backends.mps.is_built,
    torch.cuda.get_device_properties,
    torch.cuda.is_available,
    torch.distributed.is_available,
    torch.get_autocast_dtype,
    torch.get_autocast_gpu_dtype,
    torch.get_default_dtype,
    torch.is_autocast_cache_enabled,
    torch.is_autocast_cpu_enabled,
    torch.is_autocast_enabled,
    torch.is_complex,
    torch.is_floating_point,
    torch.nn.functional._Reduction.get_enum,  # type: ignore[attr-defined]
    torch.promote_types,
    torch._C._get_privateuse1_backend_name,
    torch.autograd._is_checkpoint_valid,
    torch.mps.is_available,
    torch.mtia.is_available,
    torch.xpu.get_device_properties,
    torch.xpu.is_available,
] + constant_fold_functions_need_guards
if torch.distributed.is_available():
    constant_fold_functions.extend(
        [
            torch.distributed.is_initialized,
            torch.distributed.get_rank,
            torch.distributed.get_world_size,
        ]
    )
# Convert to dict for O(1) access times
constant_fold_functions_need_guards = dict.fromkeys(constant_fold_functions_need_guards)
constant_fold_functions = dict.fromkeys(constant_fold_functions)

# Ops that consume scalar values from 0-d tensors (via .item()) for computation
# only, not for output shapes. When capture_scalar_outputs is enabled, these ops
# create unbacked symbols that don't affect the outputs, causing
# PendingUnbackedSymbolNotFound errors.
#
# We use an explicit allowlist rather than generically suppressing for any 0-d
# tensor argument because an op could legitimately create unbacked symbols for
# data-dependent output shapes. Suppressing those would silently drop binding
# information.
ops_consuming_unbacked_scalars: frozenset[Callable[..., Any]] = frozenset(
    {
        # ops with Scalar alpha/beta/value arguments
        torch.add,
        torch.sub,
        torch.addmm,
        torch.addmv,
        torch.addr,
        torch.addbmm,
        torch.baddbmm,
        torch.addcmul,
        torch.addcdiv,
        # foreach ops with scalar/alpha arguments
        torch._foreach_add,
        torch._foreach_add_,
        torch._foreach_sub,
        torch._foreach_sub_,
        torch._foreach_mul,
        torch._foreach_mul_,
        torch._foreach_div,
        torch._foreach_div_,
        torch._foreach_clamp_max,
        torch._foreach_clamp_max_,
        torch._foreach_clamp_min,
        torch._foreach_clamp_min_,
        torch._foreach_maximum,
        torch._foreach_maximum_,
        torch._foreach_minimum,
        torch._foreach_minimum_,
        torch._foreach_pow,
        torch._foreach_pow_,
        torch._foreach_lerp,
        torch._foreach_lerp_,
        torch._foreach_addcmul,
        torch._foreach_addcmul_,
        torch._foreach_addcdiv,
        torch._foreach_addcdiv_,
    }
)

# Derived set of method names for TensorVariable.call_method. Includes inplace
# variants (e.g., "add_") since those are only callable as tensor methods.
# This is a set of strings (not function objects) because the call_method path
# only has the method name, whereas call_function has the actual function object.
methods_consuming_unbacked_scalars: frozenset[str] = frozenset(
    name
    for fn in ops_consuming_unbacked_scalars
    for name in (fn.__name__, fn.__name__ + "_")
    if hasattr(torch.Tensor, name)
)


@functools.cache
def tracing_state_functions() -> dict[Callable[[], Any], bool | None]:
    # Defined as a function to avoid circular import like torch.onnx
    return {
        torch.jit.is_scripting: False,
        torch.jit.is_tracing: False,
        torch._C._get_tracing_state: None,
        torch.fx._symbolic_trace.is_fx_tracing: False,
        torch.fx._symbolic_trace.is_fx_symbolic_tracing: False,
        torch.onnx.is_in_onnx_export: False,
        # pyrefly: ignore [deprecated]
        torch._dynamo.external_utils.is_compiling: True,
        # pyrefly: ignore [deprecated]
        torch._utils.is_compiling: True,
        torch.compiler.is_compiling: True,
        torch.compiler.is_dynamo_compiling: True,
        torch.compiler.is_exporting: True,
        torch._dynamo.eval_frame._is_in_optimized_module: True,
        # Look into https://github.com/pytorch/pytorch/pull/164721 why this is
        # turned to True for Dynamo.
        torch.nn.modules.activation._is_make_fx_tracing: True,
    }


bin_ops = dict.fromkeys(["add", "sub", "mul", "div", "sqrt"])

dispatch_key_set_functions = {
    torch._C._dispatch_keys,
    torch._C._dispatch_tls_local_include_set,
    torch._C._dispatch_tls_local_exclude_set,
}


def _check_for_gradient_edge(var: VariableTracker, arg_name: str) -> None:
    """Check if var contains a GradientEdge from outside the compiled region.

    Used by handle_autograd_grad to reject external GradientEdge objects that
    cannot be traced through.
    """
    from .dicts import ConstDictVariable
    from .lists import BaseListVariable

    if isinstance(var, UserDefinedTupleVariable) and type(var.value) is GradientEdge:
        # Try to get source info for context
        source_info = var.source.name if var.source else None
        context = f"GradientEdge in {arg_name}"
        if source_info:
            context += f": {source_info}"

        unimplemented(
            gb_type="autograd.grad with external GradientEdge",
            context=context,
            explanation=(
                "torch.autograd.grad() cannot be used with GradientEdge inputs "
                "passed from outside the compiled region. The GradientEdge contains "
                "a reference to an autograd node that was created before torch.compile "
                "started tracing, so Dynamo cannot trace through its computation."
            ),
            hints=[
                "Move the autograd.grad() call outside the torch.compile region.",
                "Or use tensor inputs directly instead of GradientEdge objects.",
                *graph_break_hints.SUPPORTABLE,
            ],
        )
    elif isinstance(var, BaseListVariable):
        for i, item in enumerate(var.items):
            _check_for_gradient_edge(item, f"{arg_name}[{i}]")
    elif isinstance(var, ConstDictVariable):
        for hash_key, item in var.items.items():
            if not hash_key.vt.is_python_constant():
                unimplemented(
                    gb_type="autograd.grad with non-constant dict key",
                    context=f"non-constant key in {arg_name}",
                    explanation="autograd.grad/backward dict inputs must have constant keys.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )
            key_repr = hash_key.vt.as_python_constant()
            _check_for_gradient_edge(item, f"{arg_name}[{key_repr!r}]")


def _collect_all_grad_fns(tensor: torch.Tensor) -> set[torch.autograd.graph.Node]:
    from torch._subclasses.fake_tensor import get_plain_tensors
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    grad_fns: set[torch.autograd.graph.Node] = set()

    plain_tensors: list[torch.SymInt | torch.Tensor | int | CustomClassBase] = []
    # Get all plain tensors (handles nested subclasses)
    if is_traceable_wrapper_subclass(tensor):
        get_plain_tensors(tensor, out=plain_tensors)
    else:
        plain_tensors.append(tensor)

    for t in plain_tensors:
        if not isinstance(t, torch.Tensor):
            continue

        if t.grad_fn is not None:
            grad_fns.add(t.grad_fn)

        # For views, also include the base tensor's grad_fn
        if t._base is not None and t._base.grad_fn is not None:
            grad_fns.add(t._base.grad_fn)

    return grad_fns


def _collect_tensors_with_sources(
    var: VariableTracker,
) -> list[tuple[torch.Tensor, str | None]]:
    """Extract (fake_tensor, source_name) pairs from a VariableTracker.

    Used by handle_autograd_grad to collect tensors from the outputs and inputs
    arguments for grad_fn reachability analysis.
    """
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    from .dicts import ConstDictVariable
    from .lazy import LazyVariableTracker
    from .lists import BaseListVariable
    from .tensor import TensorVariable

    results: list[tuple[torch.Tensor, str | None]] = []
    if isinstance(var, TensorVariable):
        fake_tensor = var.as_proxy().node.meta.get("example_value")
        if not isinstance(fake_tensor, torch.Tensor):
            raise AssertionError(
                f"Expected fake_tensor to be a torch.Tensor, got {type(fake_tensor)}"
            )
        if is_fake_tensor(fake_tensor):
            pass
        elif is_traceable_wrapper_subclass(fake_tensor):
            # For tensor subclasses (e.g. DTensor), verify the inner tensors
            # are FakeTensors but keep the original subclass for grad_fn
            # reachability analysis.
            plain: list[object] = []
            torch._subclasses.fake_tensor.get_plain_tensors(
                fake_tensor,  # pyrefly: ignore[bad-argument-type]
                out=plain,  # pyrefly: ignore[bad-argument-type]
            )
            if not all(is_fake_tensor(t) for t in plain if isinstance(t, torch.Tensor)):
                raise AssertionError(
                    f"Expected all plain tensors to be FakeTensors, got {[type(t) for t in plain]}"
                )
        else:
            raise AssertionError(
                f"Expected FakeTensor or subclass, got {type(fake_tensor)}"
            )
        source_name = var.source.name if var.source else None
        results.append((fake_tensor, source_name))
    elif isinstance(var, LazyVariableTracker):
        # Realize the lazy var to get the actual TensorVariable
        results.extend(_collect_tensors_with_sources(var.realize()))
    elif isinstance(var, BaseListVariable):
        for item in var.items:
            results.extend(_collect_tensors_with_sources(item))
    elif isinstance(var, ConstDictVariable):
        for item in var.items.values():
            results.extend(_collect_tensors_with_sources(item))
    else:
        unimplemented(
            gb_type="autograd.grad with unsupported argument type",
            context=f"got {type(var).__name__}",
            explanation=(
                f"torch.autograd.grad() received an argument of type {type(var).__name__} "
                "which is not supported. Expected tensor or sequence of tensors."
            ),
            hints=[
                "Ensure outputs and inputs arguments are tensors or sequences of tensors.",
            ],
        )
    return results


def _collect_placeholder_nodes(var: "VariableTracker") -> list[torch.fx.Node]:
    """Recursively collect FX placeholder nodes from a VariableTracker.

    The returned placeholder nodes carry grapharg.example (real tensor) and
    example_value (FakeTensor) metadata — comparing these reveals lost
    autograd linkage (e.g., grad_fn dropped during tracing).
    See NOTE [Detecting lost autograd linkage in closure-captured tensors].
    """
    from .lazy import LazyVariableTracker
    from .lists import BaseListVariable
    from .tensor import TensorVariable

    result: list[torch.fx.Node] = []
    if isinstance(var, TensorVariable):
        node = var.as_proxy().node
        if node.op == "placeholder":
            result.append(node)
    elif isinstance(var, LazyVariableTracker):
        result.extend(_collect_placeholder_nodes(var.realize()))
    elif isinstance(var, BaseListVariable):
        for item in var.items:
            result.extend(_collect_placeholder_nodes(item))
    else:
        unimplemented(
            gb_type="_autograd_grad with unsupported argument type",
            context=f"got {type(var).__name__}",
            explanation=(
                f"_autograd_grad() received an argument of type {type(var).__name__} "
                "which is not supported. Expected tensor or sequence of tensors."
            ),
            hints=[
                "Ensure outputs and inputs arguments are tensors or sequences of tensors.",
            ],
        )
    return result


@functools.cache
def get_overridable_functions() -> set[Callable[..., Any]]:
    from itertools import chain

    from torch.overrides import get_overridable_functions as get_overridable_functions_
    from torch.utils._device import _device_constructors

    funcs = set(chain.from_iterable(get_overridable_functions_().values()))
    funcs.update(_device_constructors())
    more: set[Callable[..., Any]] = {
        torch.ones_like,
        torch.zeros_like,
        torch.empty_like,
        torch.full_like,
    }
    funcs.update(more)
    return funcs


class BaseTorchVariable(VariableTracker):
    """common base for all torch.* functions, classes, modules and other things"""

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BaseTorchVariable":
        if inspect.isclass(value):
            install_guard(source.make_guard(GuardBuilder.CLASS_MATCH))
        elif inspect.ismodule(value):
            install_guard(source.make_guard(GuardBuilder.MODULE_MATCH))
        elif inspect.isfunction(value):
            install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        elif inspect.isbuiltin(value) or isinstance(
            value, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
        ):
            install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        elif is_wrapper_or_member_descriptor(value) or isinstance(
            value, torch._dynamo.compiled_autograd.Op
        ):
            # Don't need to guard on wrappers
            pass
        else:
            install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    def __init__(self, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.value = value

    def reconstruct(self, codegen: "PyCodegen") -> None:
        try:
            name = f"{self.value.__module__}.{self.value.__name__}"
        except Exception:
            name = f"torch_obj_{id(self.value)}"
        unique_var_name = "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        codegen.extend_output(
            codegen.setup_globally_cached(unique_var_name, self.value)
        )

    def as_proxy(self) -> Any:
        return self.value

    def as_python_constant(self) -> Any:
        return self.value

    def get_real_python_backed_value(self) -> Any:
        return self.value

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._nb_binop_impl(tx, other, "__or__", "__ror__", reverse)

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        return self._nb_binop_impl(tx, other, "__sub__", "__rsub__", reverse)

    def _nb_binop_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        forward: str,
        reverse_dunder: str,
        reverse: bool,
    ) -> VariableTracker:
        dunder = reverse_dunder if reverse else forward
        method = getattr(type(self.value), dunder, None)
        if method is None:
            return VariableTracker.build(tx, NotImplemented)
        try:
            other_val = other.as_python_constant()
        except NotImplementedError:
            return VariableTracker.build(tx, NotImplemented)
        result = method(self.value, other_val)
        if result is NotImplemented:
            return VariableTracker.build(tx, NotImplemented)
        return VariableTracker.build(tx, result)

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        return hash(self.value), False

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        result = hasattr(self.value, name)
        return VariableTracker.build(tx, result)

    def can_constant_fold_through(self) -> bool:
        if self.value in constant_fold_functions:
            return True

        if (
            self.value is torch.autograd._profiler_enabled
            and config.constant_fold_autograd_profiler_enabled
        ):
            # The relevant flag is enabled only for export. One might wonder
            # why?
            #
            # Actually we would like to not graph break even in the case of
            # Dynamo. But there is a weird-unsolved bug with Kineto + Dynamo
            # when there are distributed jobs that lead to NCCL timeouts. This
            # bug is a rare edege case, but we have not been able to root cause
            # it yet. See https://www.internalfb.com/sevmanager/view/560336 for
            # more details.
            #
            # So is this safe for export? Yes, for export, we do not anticipate
            # JIT tracing in distributed job training, and the weird edge-case
            # interaction with Kineto is not a valid usecase. So, this is ok.
            return True

        return getattr(self.value, "__module__", None) == "math"


class TorchCtxManagerClassVariable(BaseTorchVariable):
    """Points to a context manager class in torch.* that dynamo has implementations"""

    def __repr__(self) -> str:
        return f"TorchCtxManagerClassVariable({self.value})"

    @staticmethod
    def is_matching_cls(value: Any) -> bool:
        # Unwrap if it's a functools.lru_cache wrapper
        value = unwrap_if_wrapper(value)
        # We can't do isinstance(value, type) check because some ctx managers
        # are implemented as a function decorated by contextlib.contextmanager,
        # E.g., torch._functorch.vmap.vmap_increment_nesting.
        return (
            # Context manager type or function with @contextmanager is callable
            callable(value)
            and (
                hashable(value)  # accesses value.__hash__()
                and value in supported_ctx_manager_classes
            )
        )

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import (
            CUDAMemPoolContextVariable,
            DisabledSavedTensorsHooksVariable,
            DualLevelContextManager,
            FSDPParamGroupUseTrainingStateVariable,
            FxTracebackAnnotateVariable,
            GradIncrementNestingCtxManagerVariable,
            GradInplaceRequiresGradCtxManagerVariable,
            GradModeVariable,
            InferenceModeVariable,
            JvpIncrementNestingCtxManagerVariable,
            SDPAKernelVariable,
            SetFwdGradEnabledContextManager,
            StreamVariable,
            VmapIncrementNestingCtxManagerVariable,
        )

        if self.value is torch.no_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, False)
                return ctx.call_function(tx, args, kwargs)
            else:
                return GradModeVariable.create(tx, False)
        elif self.value in (
            torch.cuda.use_mem_pool,
            torch.cuda.use_mem_pool.__wrapped__,  # type: ignore[attr-defined]
        ):
            unexpected_kwargs = [k for k in kwargs if k not in ("pool", "device")]
            if unexpected_kwargs:
                raise_type_error(
                    tx,
                    "use_mem_pool() got an unexpected keyword argument "
                    f"'{unexpected_kwargs[0]}'",
                )
            if args and "pool" in kwargs:
                raise_type_error(
                    tx, "use_mem_pool() got multiple values for argument 'pool'"
                )
            if len(args) > 1 and "device" in kwargs:
                raise_type_error(
                    tx, "use_mem_pool() got multiple values for argument 'device'"
                )
            if len(args) > 2:
                raise_type_error(
                    tx,
                    "use_mem_pool() takes from 1 to 2 positional arguments "
                    f"but {len(args)} were given",
                )
            if not args and "pool" not in kwargs:
                raise_type_error(
                    tx,
                    "use_mem_pool() missing 1 required positional argument: 'pool'",
                )
            mempool = args[0] if args else kwargs["pool"]
            device = args[1] if len(args) > 1 else kwargs.get("device")
            return CUDAMemPoolContextVariable.create(tx, mempool, device)
        elif self.value is torch.enable_grad:
            if len(args) == 1 and isinstance(
                args[0], variables.functions.BaseUserFunctionVariable
            ):
                ctx = GradModeVariable.create(tx, True)
                return ctx.call_function(tx, args, kwargs)
            return GradModeVariable.create(tx, True)
        elif self.value is torch.set_grad_enabled and len(args) == 1:
            return GradModeVariable.create(
                tx, args[0].as_python_constant(), initialized=True
            )
        elif self.value is torch.inference_mode:
            if len(args) > 1:
                raise AssertionError(
                    f"torch.inference_mode expects at most 1 arg, got {len(args)}"
                )
            if len(kwargs) != 0:
                raise AssertionError(
                    f"torch.inference_mode expects no kwargs, got {len(kwargs)}"
                )
            inf_mode = args[0].as_python_constant() if len(args) == 1 else True
            return InferenceModeVariable.create(tx, inf_mode)
        elif self.value in (
            torch.fx.traceback.annotate,
            torch.fx.traceback.annotate.__wrapped__,  # type: ignore[attr-defined]
        ):
            if len(args) > 1:
                raise AssertionError(
                    f"torch.fx.traceback.annotate expects at most 1 arg, got {len(args)}"
                )
            if len(kwargs) != 0:
                raise AssertionError(
                    f"torch.fx.traceback.annotate expects no kwargs, got {len(kwargs)}"
                )
            return FxTracebackAnnotateVariable(
                args[0].as_python_constant(), source=self.source
            )
        elif inspect.isclass(self.value) and issubclass(self.value, torch.Stream):
            from torch._dynamo.variables.builder import wrap_fx_proxy_cls

            return wrap_fx_proxy_cls(
                StreamVariable,
                tx,
                tx.output.create_proxy(
                    "call_function",
                    self.value,
                    (),
                    {},
                ),
            )
        elif self.value in (
            torch.amp.autocast_mode.autocast,
            torch.cuda.amp.autocast,
            torch.cpu.amp.autocast,
        ):
            # pyrefly: ignore [bad-argument-type]
            return AutocastModeVariable.create(self.value, args, kwargs)
        elif self.value in (
            torch.profiler.record_function,
            torch.autograd.profiler.record_function,
        ):
            return ProfilerRecordFunctionContextVariable.create(
                func=self.value, record_args=args, record_kwargs=kwargs
            )
        elif self.value in (
            torch.profiler.profile,
            torch.autograd.profiler.profile,
        ):
            warning_once(log, "Profiler function %s will be ignored", self.value)
            return ProfilerContextVariable()
        elif (
            self.value is torch._C.DisableTorchFunctionSubclass
            or self.value is torch._C.DisableTorchFunction
        ):
            if args or kwargs:
                raise AssertionError(
                    "DisableTorchFunctionSubclass/DisableTorchFunction expects no args or kwargs"
                )
            return TorchFunctionDisableVariable.create(
                tx, only_subclass=self.value is torch._C.DisableTorchFunctionSubclass
            )
        elif self.value is torch._functorch.vmap.vmap_increment_nesting:
            if len(args) != 2:
                raise AssertionError(
                    f"vmap_increment_nesting expects 2 args, got {len(args)}"
                )
            return VmapIncrementNestingCtxManagerVariable.create(
                tx,
                args,
            )
        elif self.value is torch._functorch.eager_transforms.jvp_increment_nesting:
            if len(args) != 0:
                raise AssertionError(
                    f"jvp_increment_nesting expects 0 args, got {len(args)}"
                )
            return JvpIncrementNestingCtxManagerVariable.create(tx)
        elif self.value is torch.autograd.forward_ad._set_fwd_grad_enabled:
            if len(args) != 1:
                raise AssertionError(
                    f"_set_fwd_grad_enabled expects 1 arg, got {len(args)}"
                )
            return SetFwdGradEnabledContextManager.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is torch.autograd.forward_ad.dual_level:
            if len(args) != 0:
                raise AssertionError(f"dual_level expects 0 args, got {len(args)}")
            return DualLevelContextManager.create(tx)
        elif self.value is torch._functorch.eager_transforms.grad_increment_nesting:
            if len(args) != 0:
                raise AssertionError(
                    f"grad_increment_nesting expects 0 args, got {len(args)}"
                )
            return GradIncrementNestingCtxManagerVariable.create(tx)
        elif (
            self.value is torch._functorch.eager_transforms.enable_inplace_requires_grad
        ):
            if len(args) != 1:
                raise AssertionError(
                    f"enable_inplace_requires_grad expects 1 arg, got {len(args)}"
                )
            return GradInplaceRequiresGradCtxManagerVariable.create(
                tx,
                [guard_if_dyn(x) for x in args],
            )
        elif self.value is torch.autograd.graph.disable_saved_tensors_hooks:
            if len(args) != 1:
                raise AssertionError(
                    f"disable_saved_tensors_hooks expects 1 arg, got {len(args)}"
                )
            return DisabledSavedTensorsHooksVariable.create(
                tx, args[0].as_python_constant()
            )
        elif (
            _fsdp_param_group is not None
            and self.value is _fsdp_param_group.FSDPParamGroup.use_training_state
        ):
            if len(args) != 2:
                raise AssertionError(
                    f"FSDPParamGroup.use_training_state expects 2 args, got {len(args)}"
                )
            return FSDPParamGroupUseTrainingStateVariable.create(
                tx, args[0], args[1].as_python_constant()
            )
        elif self.value is torch.nn.attention.sdpa_kernel.__wrapped__:  # type: ignore[attr-defined]
            name_to_arg_map = bind_args_cached(
                self.value,
                tx,
                self.source,
                args,
                kwargs,
            )
            backends = name_to_arg_map["backends"].as_python_constant()
            set_priority = name_to_arg_map["set_priority"].as_python_constant()
            return SDPAKernelVariable.create(tx, backends, set_priority)

        return super().call_function(tx, args, kwargs)


class AllowInGraphKind(enum.Enum):
    DEFAULT = "default"
    NONSTRICT_TRACE = "nonstrict_trace"
    LEAF_FUNCTION = "leaf_function"


class TorchInGraphFunctionVariable(BaseTorchVariable):
    """Points to a torch function/method that should be put in FX graph"""

    def __init__(
        self,
        value: Callable[..., Any],
        kind: AllowInGraphKind | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(value, **kwargs)
        from ..trace_rules import is_leaf_function, is_nonstrict_trace_callable

        if kind is None:
            if is_leaf_function(value):
                kind = AllowInGraphKind.LEAF_FUNCTION
            elif is_nonstrict_trace_callable(value):
                kind = AllowInGraphKind.NONSTRICT_TRACE
            else:
                kind = AllowInGraphKind.DEFAULT

        self.kind = kind

    def __repr__(self) -> str:
        return f"TorchInGraphFunctionVariable({self.value}, kind={self.kind})"

    def get_function(self) -> Callable[..., Any]:
        return self.value

    @staticmethod
    @functools.cache
    def _get_handlers() -> dict[Callable[..., Any], Callable[..., Any]]:
        """Build a dict from function -> method to handle it so that we are O(1)
        in terms of the number of function with special handling."""
        # pyrefly: ignore [implicit-any]
        handlers = {}

        def register(
            *fns: Callable[..., Any],
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def _register(handler: Callable[..., Any]) -> Callable[..., Any]:
                for fn in fns:
                    if fn in handlers:
                        raise AssertionError(f"Handler already registered for {fn}")
                    handlers[fn] = handler
                return handler

            if not callable(fns[0]):
                raise AssertionError(
                    f"Expected first argument to be callable, got {type(fns[0])}"
                )
            return _register

        from torch.backends.cuda import SDPAParams

        from . import (
            ConstantVariable,
            GradModeVariable,
            InferenceModeVariable,
            StreamContextVariable,
            SymNodeVariable,
            TensorVariable,
            UserDefinedObjectVariable,
        )
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        @register(*tracing_state_functions())
        def handle_tracing_state_functions(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if args:
                raise AssertionError(
                    f"Tracing state functions expect no args, got {len(args)}"
                )
            if kwargs:
                raise AssertionError(
                    f"Tracing state functions expect no kwargs, got {len(kwargs)}"
                )
            # See: https://github.com/pytorch/pytorch/issues/110765
            if self.value in (
                # pyrefly: ignore [deprecated]
                torch._utils.is_compiling,
                # pyrefly: ignore [deprecated]
                torch._dynamo.external_utils.is_compiling,
                torch.compiler.is_compiling,
                torch.compiler.is_dynamo_compiling,
                torch.compiler.is_exporting,
                torch._dynamo.eval_frame._is_in_optimized_module,
            ):
                tx.mark_inconsistent_side_effects()
            # Read dynamically: the cached dict always has True, but
            # is_exporting() should only be True during torch.export.
            if self.value is torch.compiler.is_exporting:
                return VariableTracker.build(tx, torch.compiler._is_exporting_flag)
            return VariableTracker.build(tx, tracing_state_functions()[self.value])

        @register(*dispatch_key_set_functions)
        def handle_dispatch_key_set_functions(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if kwargs:
                raise AssertionError(
                    f"Dispatch key set functions expect no kwargs, got {len(kwargs)}"
                )
            if self.value is torch._C._dispatch_keys:
                if len(args) != 1:
                    raise AssertionError(
                        f"_dispatch_keys expects 1 arg, got {len(args)}"
                    )
                if not args[0].is_tensor():
                    raise AssertionError(
                        "Expected first argument to _dispatch_keys to be a tensor"
                    )
                # pyrefly: ignore[missing-attribute]
                example_value = args[0].proxy.node.meta["example_value"]
                dks = self.value(example_value)
                # Remove Python and PythonTLSSnapshot from the dispatch key set,
                # as they originate from FakeTensor propagation.
                # This should only be done if the example_value is a FakeTensor.
                # However, if tensor subclasses are present,
                # it is reasonable for Python to remain in the dispatch key set.
                if torch._subclasses.fake_tensor.is_fake_tensor(example_value):
                    dks = (
                        dks
                        - torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
                        - torch._C.DispatchKeySet(
                            torch._C.DispatchKey.PythonTLSSnapshot
                        )
                    )
                return DispatchKeySetVariable.create(dks)
            else:
                if args:
                    raise AssertionError(
                        f"Dispatch key set function expects no args, got {len(args)}"
                    )
                return DispatchKeySetVariable.create(self.value())

        @register(torch.overrides.get_default_nowrap_functions.__wrapped__)
        def handle_get_default_nowrap_functions(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            # [Note: __torch_function__] we return empty here because we restrict
            # the set of functions that we trace __torch_function__ on to
            # functions outside of the actual set. Implementing this properly will require implementing
            # some variable types to track and compare tensor getset descriptors
            return VariableTracker.build(
                tx, torch.overrides.get_default_nowrap_functions()
            )

        @register(torch.ops.inductor.accumulate_grad_.default)
        def handle_accumulate_grad_(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) == 2 and not kwargs:
                variable, new_grad = args
                if isinstance(variable, variables.LazyVariableTracker):
                    variable = variable.realize()
                if not variable.is_tensor():
                    raise AssertionError(
                        "Expected first argument to accumulate_grad_ to be a tensor"
                    )
                variable_grad = variable.tp_getattro_impl(tx, "grad")
                updated_grad = tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.accumulate_grad),
                    [variable, variable_grad, new_grad],
                    kwargs,
                )
                updated_grad = updated_grad.clone(source=None, mutation_type=None)
                tx.output.side_effects.store_attr(variable, "grad", updated_grad)
                return ConstantVariable.create(None)
            if len(args) == 3 and not kwargs:
                variable, variable_grad, new_grad = args
                if isinstance(variable, variables.LazyVariableTracker):
                    variable = variable.realize()
                if not variable.is_tensor():
                    raise AssertionError(
                        "Expected first argument to accumulate_grad_ to be a tensor"
                    )
                updated_grad = tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.accumulate_grad_no_alias),
                    [variable, variable_grad, new_grad],
                    kwargs,
                )
                tx.output.side_effects.store_attr(
                    variable,
                    "grad",
                    updated_grad.clone(source=None, mutation_type=None),
                )
                return updated_grad
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.accumulate_grad_no_alias),
                list(args),
                kwargs,
            )

        @register(math.radians)
        def handle_radians(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if not check_unspec_or_constant_args(args, kwargs):
                # Use polyfill to convert math.radians(x) into math.pi * x / 180.0
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.radians),
                    list(args),
                    kwargs,
                )

        if hasattr(math, "fma"):  # Python 3.13+

            @register(math.fma)
            def handle_fma(
                self,
                tx: "InstructionTranslatorBase",
                *args: VariableTracker,
                **kwargs: VariableTracker,
            ) -> VariableTracker | None:
                if len(args) != 3 or kwargs:
                    return None

                if all(arg.is_tensor() for arg in args):
                    x, y, z = args
                    addcmul_fn = TorchInGraphFunctionVariable(torch.addcmul)
                    return addcmul_fn.call_function(tx, [z, x, y], {})

                # Use math.fma if constants
                return None

        @register(torch.is_inference_mode_enabled)
        def handle_is_inference_mode_enabled(
            self, tx: "InstructionTranslatorBase"
        ) -> NoReturn:
            unimplemented(
                gb_type="Encountered torch.is_inference_mode_enabled during tracing",
                context="",
                explanation="torch.is_inference_mode_enabled() is not supported",
                hints=[
                    *graph_break_hints.FUNDAMENTAL,
                    *graph_break_hints.INFERENCE_MODE,
                ],
            )

        @register(torch.is_tensor, torch.overrides.is_tensor_like)
        def handle_is_tensor(
            self, tx: "InstructionTranslatorBase", arg: VariableTracker
        ) -> VariableTracker:
            if arg.is_tensor() or (
                self.value is torch.overrides.is_tensor_like
                and isinstance(arg, UserDefinedObjectVariable)
                and hasattr(arg.value, "__torch_function__")
            ):
                return ConstantVariable.create(True)
            else:
                return ConstantVariable.create(False)

        @register(
            torch.is_floating_point,
            torch.is_complex,
        )
        def handle_is_floating_point(
            self, tx: "InstructionTranslatorBase", input: Any
        ) -> VariableTracker | None:
            input_arg = input
            if input_arg.is_tensor() and input_arg.dtype is not None:
                if self.value is torch.is_floating_point:
                    return VariableTracker.build(tx, input_arg.dtype.is_floating_point)
                elif self.value is torch.is_complex:
                    return VariableTracker.build(tx, input_arg.dtype.is_complex)
                else:
                    raise AssertionError(f"calling {self.value}")
            return None

        @register(torch.numel)
        def handle_numel(
            self, tx: "InstructionTranslatorBase", input: Any
        ) -> VariableTracker | None:
            if input.is_tensor() and input.valid_size():
                return VariableTracker.build(tx, product(input.size))
            elif input.is_tensor():
                # Workaround dynamic shapes issue
                return input.call_method(tx, "numel", [], {})
            return None

        @register(torch.compile)
        def handle_torch_compile(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) == 1:
                # torch.compile is a no-op in dynamo
                return args[0]

            unimplemented(
                gb_type="torch.compile call with > 1 args",
                context=f"args={args}, kwargs={kwargs}",
                explanation="Attempted to call `torch.compile` with > 1 args. Dynamo does not support this.",
                hints=[
                    "Remove the torch.compile call or its additional args.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        @register(torch.library.wrap_triton, torch._library.capture_triton)
        def handle_wrap_triton(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) == 1:
                # wrap_triton / capture_triton is a no-op in dynamo
                return args[0]

            unimplemented(
                gb_type="torch.library.wrap_triton call with > 1 args",
                context=f"args={args}, kwargs={kwargs}",
                explanation="Attempted to call `wrap_triton`/`capture_triton`"
                " with > 1 args. Dynamo does not support this.",
                hints=[
                    "Remove the wrap_triton/capture_triton call or its additional args.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        @register(*REWRITE_OPS_TO_TENSOR_SIZE_METHOD)
        def handle_tensor_size_rewrites(
            self, tx: "InstructionTranslatorBase", input: VariableTracker
        ) -> VariableTracker:
            if not input.is_tensor():
                raise AssertionError(
                    "Expected input to tensor size rewrite to be a tensor"
                )
            return input.call_method(tx, "size", [], {})

        @register(
            torch.nn.modules.utils._single,
            torch.nn.modules.utils._pair,
            torch.nn.modules.utils._triple,
            torch.nn.modules.utils._quadruple,
            torch.nn.modules.utils._ntuple,
        )
        def handle_ntuple(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return self._call_ntuple(tx, args, kwargs)

        @register(torch.is_grad_enabled)
        def handle_is_grad_enabled(
            self, tx: "InstructionTranslatorBase"
        ) -> ConstantVariable:
            install_guard(GradModeVariable._guards_singleton)
            return VariableTracker.build(tx, torch.is_grad_enabled())

        @register(torch.autograd.grad_mode._enter_inference_mode)
        def handle_enter_inference_mode(
            self, tx: "InstructionTranslatorBase", mode: VariableTracker
        ) -> InferenceModeVariable:
            ctx = InferenceModeVariable.create(tx, mode.as_python_constant())
            ctx.enter(tx)
            return ctx

        @register(torch.autograd.grad_mode._exit_inference_mode)
        def handle_exit_inference_mode(
            self, tx: "InstructionTranslatorBase", mode: InferenceModeVariable
        ) -> VariableTracker:
            return mode.exit(tx)

        @register(torch.use_deterministic_algorithms)
        def handle_use_deterministic_algorithms(
            self,
            tx: "InstructionTranslatorBase",
            mode: Any,
            warn_only: VariableTracker | bool = False,
        ) -> VariableTracker:
            value = mode.as_python_constant()
            warn_only_value = (
                warn_only.as_python_constant()
                if isinstance(warn_only, VariableTracker)
                else warn_only
            )
            tx.output.create_node(
                "call_function",
                torch._C._set_deterministic_algorithms,
                (value,),
                {"warn_only": warn_only_value},
            )
            torch._C._set_deterministic_algorithms(value, warn_only=warn_only_value)
            return ConstantVariable.create(None)

        @register(torch.autocast_increment_nesting)
        def handle_autocast_increment_nesting(
            self, tx: "InstructionTranslatorBase"
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch.autocast_increment_nesting, (), {}
            )
            prev = torch.autocast_increment_nesting()
            tx.output.add_cleanup_hook(lambda: torch.autocast_decrement_nesting())
            return VariableTracker.build(tx, prev)

        @register(torch.autocast_decrement_nesting)
        def handle_autocast_decrement_nesting(
            self, tx: "InstructionTranslatorBase"
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch.autocast_decrement_nesting, (), {}
            )
            prev = torch.autocast_decrement_nesting()
            tx.output.add_cleanup_hook(lambda: torch.autocast_increment_nesting())
            return VariableTracker.build(tx, prev)

        @register(torch.set_autocast_enabled)
        def handle_set_autocast_enabled(
            self,
            tx: "InstructionTranslatorBase",
            device_type: VariableTracker,
            enabled: VariableTracker,
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function",
                torch.set_autocast_enabled,
                (device_type.as_proxy(), enabled.as_proxy()),
            )
            dev_py_const = device_type.as_python_constant()
            prev = torch.is_autocast_enabled(dev_py_const)
            torch.set_autocast_enabled(dev_py_const, enabled.as_python_constant())
            tx.output.add_cleanup_hook(
                lambda: torch.set_autocast_enabled(dev_py_const, prev)
            )
            return ConstantVariable.create(None)

        @register(torch.set_autocast_dtype)
        def handle_set_autocast_dtype(
            self,
            tx: "InstructionTranslatorBase",
            device_type: VariableTracker,
            dtype: VariableTracker,
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function",
                torch.set_autocast_dtype,
                (device_type.as_proxy(), dtype.as_proxy()),
            )
            dev_py_const = device_type.as_python_constant()
            prev = torch.get_autocast_dtype(dev_py_const)
            torch.set_autocast_dtype(dev_py_const, dtype.as_python_constant())
            tx.output.add_cleanup_hook(
                lambda: torch.set_autocast_dtype(dev_py_const, prev)
            )
            return ConstantVariable.create(None)

        @register(torch.set_autocast_cache_enabled)
        def handle_set_autocast_cache_enabled(
            self, tx: "InstructionTranslatorBase", enabled: VariableTracker
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch.set_autocast_cache_enabled, (enabled.as_proxy(),)
            )
            prev = torch.is_autocast_cache_enabled()
            torch.set_autocast_cache_enabled(enabled.as_python_constant())
            tx.output.add_cleanup_hook(lambda: torch.set_autocast_cache_enabled(prev))
            return ConstantVariable.create(None)

        @register(torch._functorch.predispatch._jvp_increment_nesting)
        def handle_jvp_increment_nesting(
            self, tx: "InstructionTranslatorBase"
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch._functorch.predispatch._jvp_increment_nesting
            )
            level = torch._functorch.predispatch._jvp_increment_nesting()
            tx.output.add_cleanup_hook(
                torch._functorch.predispatch._jvp_decrement_nesting
            )
            return VariableTracker.build(tx, level)

        @register(torch._functorch.predispatch._jvp_decrement_nesting)
        def handle_jvp_decrement_nesting(
            self, tx: "InstructionTranslatorBase"
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch._functorch.predispatch._jvp_decrement_nesting
            )
            level = torch._functorch.predispatch._jvp_decrement_nesting()
            tx.output.add_cleanup_hook(
                torch._functorch.predispatch._jvp_increment_nesting
            )
            return VariableTracker.build(tx, level)

        @register(torch._functorch.predispatch._enter_dual_level)
        def handle_enter_dual_level(
            self, tx: "InstructionTranslatorBase"
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch._functorch.predispatch._enter_dual_level
            )
            level = torch._functorch.predispatch._enter_dual_level()
            tx.output.add_cleanup_hook(
                lambda: torch._functorch.predispatch._exit_dual_level(level=level)
            )
            return VariableTracker.build(tx, level)

        @register(torch._functorch.predispatch._exit_dual_level)
        def handle_exit_dual_level(
            self, tx: "InstructionTranslatorBase", level: VariableTracker
        ) -> VariableTracker:
            level_const = level.as_python_constant()
            tx.output.create_node(
                "call_function",
                torch._functorch.predispatch._exit_dual_level,
                (),
                {
                    "level": level_const,
                },
            )
            torch._functorch.predispatch._exit_dual_level(level=level_const)

            def cleanup():
                new_level = torch._functorch.predispatch._enter_dual_level()
                if new_level != level_const:
                    raise AssertionError("Invalid _exit_dual_level")

            tx.output.add_cleanup_hook(cleanup)
            return ConstantVariable.create(None)

        @register(torch._C._functorch._grad_increment_nesting)
        def handle_grad_increment_nesting(
            self, tx: "InstructionTranslatorBase"
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch._C._functorch._grad_increment_nesting
            )
            level = torch._C._functorch._grad_increment_nesting()
            tx.output.add_cleanup_hook(torch._C._functorch._grad_decrement_nesting)
            return VariableTracker.build(tx, level)

        @register(torch._C._functorch._grad_decrement_nesting)
        def handle_grad_decrement_nesting(
            self, tx: "InstructionTranslatorBase"
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function", torch._C._functorch._grad_decrement_nesting
            )
            level = torch._C._functorch._grad_decrement_nesting()
            tx.output.add_cleanup_hook(torch._C._functorch._grad_increment_nesting)
            return VariableTracker.build(tx, level)

        @register(torch._C._functorch.set_inplace_requires_grad_allowed)
        def handle_set_inplace_requires_grad_allowed(
            self, tx: "InstructionTranslatorBase", allowed: VariableTracker
        ) -> VariableTracker:
            tx.output.create_node(
                "call_function",
                torch._C._functorch.set_inplace_requires_grad_allowed,
                (allowed.as_proxy(),),
            )
            prev = torch._C._functorch.get_inplace_requires_grad_allowed()
            torch._C._functorch.set_inplace_requires_grad_allowed(
                allowed.as_python_constant()
            )
            tx.output.add_cleanup_hook(
                lambda: torch._C._functorch.set_inplace_requires_grad_allowed(prev)
            )
            return ConstantVariable.create(None)

        @register(torch.are_deterministic_algorithms_enabled)
        def handle_are_deterministic_algorithms_enabled(
            self, tx: "InstructionTranslatorBase"
        ) -> ConstantVariable:
            guard = Guard(
                GlobalStateSource(),
                GuardBuilder.DETERMINISTIC_ALGORITHMS,  # type: ignore[arg-type]
            )
            install_guard(guard)
            return VariableTracker.build(
                tx, torch.are_deterministic_algorithms_enabled()
            )

        @register(torch._C._is_torch_function_enabled)
        def handle_is_torch_function_enabled(
            self, tx: "InstructionTranslatorBase"
        ) -> ConstantVariable:
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            # see comment on SymbolicTorchFunctionState class as to why
            # this is not a bug
            return VariableTracker.build(
                tx, tx.symbolic_torch_function_state.torch_function_subclass_enabled
            )

        @register(torch._C._is_cow_tensor)  # pyrefly: ignore[missing-attribute]
        def handle_is_cow_tensor(
            self, tx: "InstructionTranslatorBase", arg: VariableTracker
        ) -> ConstantVariable:
            if not arg.is_tensor():
                raise AssertionError(
                    f"_is_cow_tensor expects a tensor, got {arg.python_type_name()}"
                )

            def has_prior_cow_state_changing_op() -> bool:
                if not isinstance(arg, TensorVariable):
                    return False
                graph = arg.as_proxy().node.graph
                return any(
                    (node.op == "call_method" and node.target == "_lazy_clone")
                    or (node.op == "call_function" and node.target is torch._lazy_clone)
                    for node in graph.nodes
                )

            if arg.source is None:
                unimplemented(
                    gb_type="source-less COW tensor check",
                    context="torch._C._is_cow_tensor on source-less tensor",
                    explanation=(
                        "Dynamo cannot safely guard COW state for an intermediate "
                        "tensor without a source."
                    ),
                    hints=[
                        "Avoid checking COW state on intermediate tensors inside "
                        "torch.compile regions.",
                    ],
                )
            if tx.output.current_tracer.is_export or torch.compiler._is_exporting_flag:
                unimplemented(
                    gb_type="COW tensor check during export",
                    context="torch._C._is_cow_tensor during export",
                    explanation=(
                        "Dynamo cannot safely export COW-state-dependent "
                        "control flow because COW state is not represented in "
                        "the exported graph."
                    ),
                    hints=[
                        "Avoid checking COW state inside torch.export regions.",
                    ],
                )
            fake_version = arg._get_fake_version()  # pyrefly: ignore[missing-attribute]
            if fake_version is not None and fake_version > 0:
                unimplemented(
                    gb_type="COW tensor check after mutation",
                    context="torch._C._is_cow_tensor after tensor mutation",
                    explanation=(
                        "Dynamo cannot safely fold a COW state check after "
                        "the tensor's state may have changed inside the "
                        "compiled frame."
                    ),
                    hints=[
                        "Move the COW state check before tensor mutations or "
                        "outside the torch.compile region.",
                    ],
                )
            if has_prior_cow_state_changing_op():
                unimplemented(
                    gb_type="COW tensor check after COW-state-changing op",
                    context="torch._C._is_cow_tensor after _lazy_clone",
                    explanation=(
                        "Dynamo cannot safely fold a COW state check after "
                        "an op in the current graph may have changed that "
                        "tensor's COW state."
                    ),
                    hints=[
                        "Move the COW state check before _lazy_clone or outside "
                        "the torch.compile region.",
                    ],
                )
            real_value = arg.get_real_value()  # pyrefly: ignore[missing-attribute]
            if is_fake_tensor(real_value):
                unimplemented(
                    gb_type="COW tensor check on FakeTensor",
                    context="torch._C._is_cow_tensor on FakeTensor",
                    explanation=(
                        "Dynamo cannot safely evaluate COW state from a "
                        "FakeTensor because COW state is not represented in "
                        "FakeTensor metadata."
                    ),
                    hints=[
                        "Avoid checking COW state on FakeTensors inside "
                        "torch.compile regions.",
                    ],
                )
            cow_state = _try_is_cow_tensor(real_value)
            if cow_state is _COW_TENSOR_UNSUPPORTED:
                unimplemented(
                    gb_type="COW tensor check on Python tensor subclass",
                    context="torch._C._is_cow_tensor on Python tensor subclass",
                    explanation=(
                        "Dynamo cannot safely evaluate COW state for Python "
                        "tensor subclasses because their storage semantics are "
                        "controlled by __torch_dispatch__."
                    ),
                    hints=[
                        "Avoid checking COW state on tensor subclasses inside "
                        "torch.compile regions.",
                    ],
                )
            install_guard(arg.source.make_guard(GuardBuilder.COW_TENSOR_MATCH))
            return VariableTracker.build(tx, cast(bool, cow_state))

        @register(torch._C._is_torch_function_all_disabled)
        def handle_is_torch_function_all_disabled(
            self, tx: "InstructionTranslatorBase"
        ) -> ConstantVariable:
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            return VariableTracker.build(
                tx, not tx.symbolic_torch_function_state.torch_function_mode_enabled
            )

        @register(torch._C._is_torch_function_mode_enabled)
        def handle_is_torch_function_mode_enabled(
            self, tx: "InstructionTranslatorBase"
        ) -> ConstantVariable:
            install_guard(TorchFunctionDisableVariable._guards_singleton)
            # _is_torch_function_mode_enabled returns True only if:
            # 1. Torch function modes are not disabled (DisableTorchFunction not entered)
            # 2. There are actually modes on the stack
            return VariableTracker.build(
                tx,
                tx.symbolic_torch_function_state.torch_function_mode_enabled
                and tx.symbolic_torch_function_state.in_torch_function_mode(),
            )

        @register(
            torch.overrides.has_torch_function,
            torch.overrides.has_torch_function_variadic,
            torch.overrides.has_torch_function_unary,
        )
        def handle_has_torch_function(
            self, tx: "InstructionTranslatorBase", *args: VariableTracker
        ) -> ConstantVariable:
            tf_state = tx.symbolic_torch_function_state
            if tf_state.skip_next:
                tf_state.skip_next = False
                return VariableTracker.build(tx, False)
            elems = (
                unpack_iterable(tx, args[0])
                if len(args) == 1 and isinstance(args[0], TupleVariable)
                else args
            )
            return VariableTracker.build(tx, any(has_torch_function(x) for x in elems))

        @register(torch._C._skip_one_hop_torch_function)
        def handle_skip_one_hop_torch_function(
            self,
            tx: "InstructionTranslatorBase",
            func: VariableTracker,
            types: VariableTracker,
            args: VariableTracker,
            kwargs: VariableTracker,
        ) -> VariableTracker:
            tf_state = tx.symbolic_torch_function_state
            if tf_state.skip_next:
                from torch._dynamo.exc import raise_observed_exception

                raise_observed_exception(
                    RuntimeError,
                    tx,
                    args=[
                        "you cannot skip two levels of __torch_function__, "
                        "you need to run one level of __torch_function__ "
                        "before being able to skip again."
                    ],
                )
            tf_state.skip_next = True
            tx.skip_one_hop_torch_function_depth += 1
            try:
                return func.call_function(
                    tx,
                    unpack_iterable(tx, args),
                    kwargs.keys_as_python_constant(),  # type: ignore[attr-defined]
                )
            finally:
                tx.skip_one_hop_torch_function_depth -= 1
                tf_state.skip_next = False

        @register(
            *dict.fromkeys(  # remove duplicates
                device_interface.stream
                for _, device_interface in get_registered_device_interfaces()
            )
        )
        def handle_device_interface_stream(
            self, tx: "InstructionTranslatorBase", stream: "StreamVariable"
        ) -> StreamContextVariable:
            return StreamContextVariable.create(tx, stream)

        @register(torch.from_numpy)
        def handle_from_numpy(
            self, tx: "InstructionTranslatorBase", *args: VariableTracker
        ) -> TensorVariable:
            if not config.trace_numpy:
                unimplemented(
                    gb_type="call `torch.from_numpy` with `torch._dynamo.config.trace_numpy=False`",
                    context=f"trace_numpy={config.trace_numpy}",
                    explanation=(
                        "Attempted to call `torch.from_numpy` with config "
                        "`torch._dynamo.config.trace_numpy` set to `False`."
                    ),
                    hints=[
                        "Change `torch._dynamo.config.trace_numpy` to `True`.",
                    ],
                )
            if not np:
                unimplemented(
                    gb_type="`torch.from_numpy` with NumPy unavailable",
                    context="",
                    explanation="Attempted to call `torch.numpy` but NumPy could not be imported.",
                    hints=[
                        "Check NumPy version and installation in your environment.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            return wrap_fx_proxy_cls(
                target_cls=TensorVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.as_tensor,
                    *proxy_args_kwargs(args, {}),
                ),
                example_value=None,
            )

        @register(torch.jit.annotate)
        def handle_jit_annotate(
            self, tx: "InstructionTranslatorBase", the_type: Any, the_value: V
        ) -> V:
            return the_value

        @register(torch.backends.cudnn.is_acceptable)
        def handle_cudnn_is_acceptable(
            self, tx: "InstructionTranslatorBase", tensor: Any, *extra: Any
        ) -> ConstantVariable:
            # is_acceptable(tensor) returns true if
            #   (a) tensor dtype/device are supported by cudnn
            #   (b) cudnn is available
            #   (c) some initialization has completed
            # technically, it depends on some global state from (c) (torch.backends.cudnn.__cudnn_version)
            if extra:
                raise AssertionError("Expect 1 input to cudnn.is_acceptable")
            if not tensor.is_tensor():
                raise AssertionError(
                    "Expect input to cudnn.is_acceptable to be a tensor"
                )
            tensor_inp = torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
            return VariableTracker.build(
                tx, torch.backends.cudnn.is_acceptable(tensor_inp)
            )

        @register(torch.utils.hooks.BackwardHook)
        def handle_backward_hook(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return variables.BackwardHookVariable.create(tx, *args, **kwargs)

        @register(torch.nn.Parameter)
        def handle_parameter(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return self.call_nn_parameter(tx, *args, **kwargs)

        @register(torch.ops.aten.sym_size, torch.ops.aten.sym_size.int)
        def handle_sym_size(
            self_: Any, tx: "InstructionTranslatorBase", self, dim: Any | None = None
        ) -> VariableTracker | None:
            # we see this when retracing already traced code
            if dim is not None:
                return self.call_method(tx, "size", [dim], {})
            return None

        @register(torch.ops.aten.sym_stride, torch.ops.aten.sym_stride.int)
        def handle_sym_stride(
            self_: Any, tx: "InstructionTranslatorBase", self, dim: Any | None = None
        ) -> VariableTracker | None:
            if dim is not None:
                return self.call_method(tx, "stride", [dim], {})
            return None

        @register(torch.addcdiv)
        def handle_addcdiv(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if len(args) == 3 and "value" in kwargs and len(kwargs) == 1:
                # decompose addcdiv into constituent ops, prevents a graph break due to converting
                # value to a scalar
                result = TorchInGraphFunctionVariable(torch.div).call_function(
                    tx, [*args[1:]], {}
                )
                result = TorchInGraphFunctionVariable(torch.mul).call_function(
                    tx, [result, kwargs["value"]], {}
                )
                return TorchInGraphFunctionVariable(torch.add).call_function(
                    tx, [args[0], result], {}
                )
            return None

        @register(torch.full)
        def handle_full(
            self,
            tx: "InstructionTranslatorBase",
            size: VariableTracker,
            fill_value: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if fill_value.is_tensor() and not issubclass(
                fill_value.python_type(), torch.nn.Parameter
            ):
                # Decompose: create empty tensor and fill it
                # This avoids the scalar extraction at compile time
                empty_result = TorchInGraphFunctionVariable(torch.empty).call_function(
                    tx, [size], kwargs
                )
                # Call fill_ method on the empty tensor
                return empty_result.call_method(tx, "fill_", [fill_value], {})
            return None

        @register(torch.tensor_split)
        def handle_tensor_split(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            indices_or_sections = (
                args[1] if len(args) >= 2 else kwargs.get("indices_or_sections")
            )
            if not isinstance(indices_or_sections, TensorVariable):
                return None

            example_value = indices_or_sections.as_proxy().node.meta.get(
                "example_value"
            )
            if (
                example_value is None
                or example_value.dim() != 0
                or example_value.device.type != "cpu"
                or example_value.dtype != torch.int64
                or (item_memo := example_value.item_memo) is None
            ):
                return None

            sections = torch.fx.experimental.symbolic_shapes.guard_scalar(item_memo)
            if not isinstance(sections, int):
                return None

            # tensor_split's scalar-tensor overload needs a concrete output
            # arity. Guard the memoized value here so downstream AOT tracing
            # does not re-read the 0-D tensor as a fresh unbacked scalar.
            sections_vt = ConstantVariable.create(sections)
            if len(args) >= 2:
                args = (args[0], sections_vt, *args[2:])
            else:
                kwargs["indices_or_sections"] = sections_vt
            return TorchInGraphFunctionVariable(torch.tensor_split).call_function(
                tx, list(args), kwargs
            )

        @register(torch._foreach_lerp_)
        def handle_inplace_foreach_lerp_scalar(
            _: Any,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            # Decompose via addcmul_ only when the weight is a tensor, so
            # tensor weights (e.g. 0-dim tensor from tensor betas in Adam) stay
            # in tensor arguments instead of hitting float() in the native
            # lerp_scalar lowering.  Python scalar weights can use the native
            # foreach op directly, avoiding extra full-size weight tensors.
            if (
                config.enable_dynamo_decompositions
                and len(args) == 3
                and args[2].is_tensor()
                and not kwargs
            ):
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_lerp_inplace),
                    list(args),
                    kwargs,
                )
            return None

        @register(torch._foreach_pow)
        def handle_foreach_pow_scalar(
            _: Any,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            if not config.enable_dynamo_decompositions:
                return None

            # In eager it's more performant to call item() from within the C op implementation
            # in compile, it's more performant to not graph break.
            if len(args) == 2 and args[0].is_tensor() and not kwargs:
                return tx.inline_user_function_return(
                    VariableTracker.build(tx, polyfills.foreach_pow_scalar),
                    list(args),
                    kwargs,
                )
            return None

        @register(torch._C._group_tensors_by_device_and_dtype)
        def handle_group_tensors_by_device_and_dtype(
            _: Any,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return tx.inline_user_function_return(
                VariableTracker.build(tx, polyfills.group_tensors_by_device_and_dtype),
                list(args),
                kwargs,
            )

        @register(torch._assert)
        def handle_assert(
            self,
            tx: "InstructionTranslatorBase",
            condition: VariableTracker,
            message: VariableTracker,
        ) -> VariableTracker | None:
            if (condition.is_python_constant() and condition.as_python_constant()) or (
                isinstance(condition, variables.SymNodeVariable)
                and condition.evaluate_expr()
            ):
                return ConstantVariable.create(None)
            return None

        @register(SDPAParams)
        def handle_sdpa_params(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return wrap_fx_proxy(
                tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch._C._SDPAParams,
                    *proxy_args_kwargs(args, kwargs),
                ),
                param_vars=args,
            )

        if DistributedVariable.is_available():
            from torch.distributed.distributed_c10d import (
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                _resolve_group_name_by_ranks_and_tag,
                get_process_group_ranks,
                get_rank,
                get_world_size,
            )

            @register(
                _get_group_size_by_name,
                _get_group_tag,
                _rank_not_in_group,
                get_process_group_ranks,
                _resolve_group_name_by_ranks_and_tag,
                get_rank,
                get_world_size,
            )
            def handle_constant_processgroup_functions(
                self,
                tx: "InstructionTranslatorBase",
                *args: VariableTracker,
                **kwargs: VariableTracker,
            ) -> VariableTracker:
                # We desugar it at trace-time into ranks by directly calling util
                # bake the result into the trace
                if len(args) == 0 and len(kwargs) == 0:
                    # get_rank() or get_world_size() with no args (uses default group)
                    pass
                elif len(args) == 1 and len(kwargs) == 0:
                    # group or group name
                    if not (
                        args[0].is_python_constant()
                        or (
                            isinstance(args[0], CustomClassObjectVariable)
                            and args[  # pyrefly: ignore[missing-attribute]
                                0
                            ].value.script_class_name  # pyrefly: ignore[missing-attribute]
                            == "torch.distributed.distributed_c10d.ProcessGroup"
                        )
                    ):
                        raise AssertionError(
                            f"Expected group arg to be a python constant or ProcessGroup, got {args[0]}"
                        )
                elif len(args) == 2 and len(kwargs) == 0:
                    # ranks + tag
                    if not isinstance(args[0], ListVariable):
                        raise AssertionError(
                            f"Expected first arg to be a ListVariable (ranks), got {type(args[0])}"
                        )
                    if not args[1].is_python_constant():
                        raise AssertionError(
                            "Expected second arg (tag) to be a python constant"
                        )
                elif len(args) == 0 and len(kwargs) > 0:
                    # All keyword arguments (e.g., get_world_size(group=...))
                    pass
                else:
                    raise AssertionError(
                        f"Invalid group value ({args}, {kwargs}) for constant pg "
                        f"function {self.value}"
                    )

                def get_arg_value(arg: VariableTracker) -> Any:
                    # CustomClassObjectVariable for ProcessGroup doesn't support
                    # as_python_constant(), so extract real_obj directly
                    if isinstance(arg, CustomClassObjectVariable):
                        return arg.value.real_obj  # pyrefly: ignore[missing-attribute]
                    return arg.as_python_constant()

                args_as_value = [get_arg_value(arg) for arg in args]
                kwargs_as_value = {k: get_arg_value(v) for k, v in kwargs.items()}
                invocation_result = self.value(*args_as_value, **kwargs_as_value)

                # Note - while we *could* cook up sources around invocations, like a FunctionSource
                # the space of invoking functions in the middle of the guard chain is very iffy. As such,
                # guard propagation via options is the best we can do.
                return VariableTracker.build(tx, invocation_result)

        @register(torch.nested.nested_tensor)
        def handle_nested_tensor(
            self,
            tx: "InstructionTranslatorBase",
            tensor_list: VariableTracker | None = None,
            *args: VariableTracker,
            layout: Any | None = None,
            **kwargs: VariableTracker,
        ) -> None:
            from .lists import BaseListVariable

            if layout and layout.is_constant_match(torch.strided):
                unimplemented(
                    gb_type="Attempted to use strided NestedTensor",
                    context=f"layout={layout}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change layout=torch.jagged.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            if not isinstance(tensor_list, BaseListVariable):
                unimplemented(
                    gb_type="Attempted to use `nested_tensor` with non-list input",
                    context=f"tensor_list={tensor_list}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change `nested_tensor` with list input.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            return None

        @register(torch.nn.functional.one_hot)
        def handle_one_hot(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> None:
            if len(args) + len(kwargs) == 1 or (
                len(args) == 2 and args[1].is_constant_match(-1)
            ):
                unimplemented(
                    gb_type="Attempted to use `torch.nn.functional.one_hot` with data-dependent output shape",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Explicitly set the `num_classes` param of the function call "
                        "`torch.nn.functional.one_hot` to something other than -1.",
                    ],
                )
            return None

        @register(torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw)
        def handle_guarding_hint_or_throw(
            self,
            tx: "InstructionTranslatorBase",
            expr: VariableTracker,
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                return VariableTracker.build(
                    tx,
                    torch.fx.experimental.symbolic_shapes.guarding_hint_or_throw(
                        expr.sym_num
                    ),
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(torch.fx.experimental.symbolic_shapes.optimization_hint)
        def handle_optimization_hint(
            self,
            tx: "InstructionTranslatorBase",
            expr: VariableTracker,
            fallback: VariableTracker | None = None,
        ) -> VariableTracker | None:
            fallback_int = fallback.as_python_constant() if fallback else None
            if isinstance(expr, SymNodeVariable):
                return VariableTracker.build(
                    tx,
                    torch.fx.experimental.symbolic_shapes.optimization_hint(
                        expr.sym_num, fallback_int
                    ),
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(torch.fx.experimental.symbolic_shapes.guard_size_oblivious)  # type: ignore[deprecated]
        def handle_guard_size_oblivious(
            self, tx: "InstructionTranslatorBase", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return VariableTracker.build(
                    tx,
                    torch.fx.experimental.symbolic_shapes.guard_size_oblivious(  # type: ignore[deprecated]
                        expr.sym_num
                    ),
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(torch.fx.experimental.symbolic_shapes.guard_or_true)
        def handle_guard_or_true(
            self, tx: "InstructionTranslatorBase", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return VariableTracker.build(
                    tx,
                    torch.fx.experimental.symbolic_shapes.guard_or_true(expr.sym_num),
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(torch.fx.experimental.symbolic_shapes.guard_or_false)
        def handle_guard_or_false(
            self, tx: "InstructionTranslatorBase", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                # TODO: this probably should be folded somewhere else but I'm not sure where
                # TODO: some of the other symbolic_shapes special tools can also get this treatment too
                return VariableTracker.build(
                    tx,
                    torch.fx.experimental.symbolic_shapes.guard_or_false(expr.sym_num),
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(torch.fx.experimental.symbolic_shapes.statically_known_false)
        def handle_statically_known_false(
            self, tx: "InstructionTranslatorBase", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                return VariableTracker.build(
                    tx,
                    torch.fx.experimental.symbolic_shapes.statically_known_false(
                        expr.sym_num
                    ),
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(torch.fx.experimental.symbolic_shapes.has_free_unbacked_symbols)
        def handle_has_free_unbacked_symbols(
            self, tx: "InstructionTranslatorBase", x: VariableTracker
        ) -> VariableTracker | None:
            from .tensor import TensorVariable

            if isinstance(x, TensorVariable):
                example_value = x.as_proxy().node.meta.get("example_value")
                if example_value is not None:
                    return variables.ConstantVariable.create(
                        torch.fx.experimental.symbolic_shapes.has_free_unbacked_symbols(
                            example_value
                        )
                    )
            return None

        @register(torch.fx.experimental.symbolic_shapes.guard_scalar)
        def guard_scalar(
            self, tx: "InstructionTranslatorBase", expr: VariableTracker
        ) -> VariableTracker:
            if isinstance(expr, SymNodeVariable):
                val = expr.sym_num
            elif expr.is_python_constant():
                val = expr.as_python_constant()
            else:
                unimplemented(
                    gb_type="torch.fx.experimental.symbolic_shapes.guard_scalar branch not supported",
                    context=f"expr: {expr}",
                    explanation="Expected `expr` to be a symbolic variable or constant.",
                    hints=[],
                )
            return VariableTracker.build(
                tx,
                # pyrefly: ignore [bad-argument-type, unbound-name]
                torch.fx.experimental.symbolic_shapes.guard_scalar(val),
            )

        @register(torch.fx.experimental.symbolic_shapes.statically_known_true)
        def handle_statically_known_true(
            self, tx: "InstructionTranslatorBase", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                return VariableTracker.build(
                    tx,
                    torch.fx.experimental.symbolic_shapes.statically_known_true(
                        expr.sym_num
                    ),
                )
            elif expr.is_python_constant():
                return expr
            else:
                return None

        @register(torch.fx.experimental.symbolic_shapes.sym_and)
        def handle_sym_and(
            self, tx: "InstructionTranslatorBase", *terms: VariableTracker
        ) -> VariableTracker | None:
            if all(isinstance(x, SymNodeVariable) for x in terms):
                return SymNodeVariable.create(
                    tx,
                    torch.fx.experimental.symbolic_shapes.sym_and(
                        *(x.as_proxy() for x in terms)
                    ),
                    sym_num=None,
                )
            return None

        @register(torch.fx.experimental.symbolic_shapes.sym_or)
        def handle_sym_or(
            self, tx: "InstructionTranslatorBase", *terms: VariableTracker
        ) -> VariableTracker | None:
            if all(isinstance(x, SymNodeVariable) for x in terms):
                return SymNodeVariable.create(
                    tx,
                    torch.fx.experimental.symbolic_shapes.sym_or(
                        *(x.as_proxy() for x in terms)
                    ),
                    sym_num=None,
                )
            return None

        @register(torch.fx.experimental.symbolic_shapes.has_static_value)
        def handle_has_static_value(
            self, tx: "InstructionTranslatorBase", expr: VariableTracker
        ) -> VariableTracker | None:
            if isinstance(expr, SymNodeVariable):
                val = expr.sym_num
            elif expr.is_python_constant():
                val = expr.as_python_constant()
            else:
                return None

            return VariableTracker.build(
                tx, torch.fx.experimental.symbolic_shapes.has_static_value(val)
            )

        @register(torch._C._autograd._unsafe_set_version_counter)
        def handle_unsafe_set_version_counter(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            from ..tensor_version_op import _unsafe_set_version_counter

            return TorchInGraphFunctionVariable(
                _unsafe_set_version_counter
            ).call_function(tx, [*args], kwargs)

        @register(torch._C._functorch.peek_interpreter_stack)
        def handle_functorch_peek_interpreter_stack(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> UserDefinedObjectVariable:
            # Wrap C++ interpreter (torch._C._functorch.CInterpreter) as UserDefinedObjectVariable,
            # but Python interpreter (torch._functorch.pyfunctorch.FuncTorchInterpreter) as FuncTorchInterpreterVariable.
            return UserDefinedObjectVariable(
                torch._C._functorch.peek_interpreter_stack()
            )

        @register(torch._functorch.pyfunctorch.coerce_cinterpreter)
        def handle_functorch_pyfunctorch_coerce_cinterpreter(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> FuncTorchInterpreterVariable:
            # pyrefly: ignore[missing-attribute]
            cinterpreter = args[0].value
            return FuncTorchInterpreterVariable(
                torch._functorch.pyfunctorch.coerce_cinterpreter(cinterpreter)
            )

        @register(torch.tensor)
        def handle_torch_tensor(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            def check_any_unspec(x: VariableTracker) -> bool:
                # NB: This includes UnspecializedPythonVariable
                if x.is_tensor() or isinstance(x, SymNodeVariable):
                    return True
                elif isinstance(x, (ListVariable, TupleVariable)):
                    return any(check_any_unspec(y) for y in x.items)
                # TODO: there maybe other recursive structures you need to
                # check
                else:
                    return False

            data_arg = None
            if args:
                data_arg = args[0]
            elif "data" in kwargs:
                data_arg = kwargs["data"]

            # NB: OK to pass torch.tensor(tensor), this will trace fine
            if (
                data_arg is not None
                and not data_arg.is_tensor()
                and check_any_unspec(data_arg)
            ):
                # This is slower and less canonical, so only use it if we
                # have to
                return TorchInGraphFunctionVariable(torch._refs.tensor).call_function(
                    tx, [*args], kwargs
                )
            else:
                return None

        @register(torch._C._pop_torch_function_stack)
        def handle_pop_torch_function(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> TorchFunctionModeVariable:
            if args:
                raise AssertionError(
                    f"_pop_torch_function_stack expects no args, got {len(args)}"
                )
            if kwargs:
                raise AssertionError(
                    f"_pop_torch_function_stack expects no kwargs, got {len(kwargs)}"
                )
            if not tx.symbolic_torch_function_state.mode_stack:
                unimplemented(
                    gb_type="Attempted to pop from empty torch function mode stack",
                    context="",
                    explanation="Called `torch._C._pop_torch_function_stack` when torch function mode stack is empty.",
                    hints=[
                        "Do not pop from empty torch function mode stack.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            TorchFunctionModeStackVariable.register_mutation(tx)
            return tx.symbolic_torch_function_state.pop_torch_function_mode()

        @register(torch._C._push_on_torch_function_stack)
        def handle_push_torch_function(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) != 1 or kwargs:
                raise_type_error(
                    tx,
                    f"push_torch_function takes exactly one argument ({len(args)} given)",
                )
            TorchFunctionModeStackVariable.register_mutation(tx)
            # type: ignore[arg-type]
            tx.symbolic_torch_function_state.push_torch_function_mode(args[0])
            return ConstantVariable.create(None)

        @register(torch._C._len_torch_function_stack)
        def handle_len_torch_function(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if args or kwargs:
                raise_type_error(tx, "len_torch_function_stack takes no arguments")
            return VariableTracker.build(
                tx, len(tx.symbolic_torch_function_state.mode_stack)
            )

        @register(torch._C._get_function_stack_at)
        def handle_get_stack_at(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> TorchFunctionModeVariable:
            if len(args) != 1 or kwargs:
                raise_type_error(
                    tx,
                    f"get_function_stack_at takes exactly one argument ({len(args)} given)",
                )
            ind = args[0].as_python_constant()
            if ind < 0:
                raise AssertionError(
                    f"Expected non-negative index for get_function_stack_at, got {ind}"
                )
            if ind >= len(tx.symbolic_torch_function_state.mode_stack):
                raise AssertionError(
                    f"Index {ind} out of range for torch function mode stack "
                    f"of length {len(tx.symbolic_torch_function_state.mode_stack)}"
                )
            return tx.symbolic_torch_function_state.mode_stack[ind]

        @register(torch.utils._python_dispatch._get_current_dispatch_mode_stack)
        def handle_get_current_dispatch_mode_stack(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if args or kwargs:
                raise_type_error(
                    tx,
                    "_get_current_dispatch_mode_stack() takes no arguments",
                )
            # During tracing, Dynamo runs with dispatch modes removed, so
            # the stack is always empty. Return [] as a constant instead
            # of graph-breaking.
            return VariableTracker.build(tx, [])

        @register(torch.get_device_module.__wrapped__)
        def handle_get_device_module(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if len(args) + len(kwargs) > 1 or (kwargs and "device" not in kwargs):
                unimplemented(
                    gb_type="improper torch.get_device_module arguments",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="torch.get_device_module accepts 1 optional argument `device`",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            try:
                if kwargs:
                    device = kwargs["device"].as_python_constant()
                elif args:
                    device = args[0].as_python_constant()
                else:
                    device = None
                module = torch.get_device_module(device)
            except Exception as e:
                unimplemented(
                    gb_type="bad device argument to torch.get_device_module",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Expected valid string/torch.device argument ('cpu', 'cuda', etc.)",
                    hints=[*graph_break_hints.USER_ERROR],
                    from_exc=e,
                )

            # need to guard only on no-arg get_device_module
            if device is None:
                source = CallFunctionNoArgsSource(self.source)
                install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            # assumes `module` is in the form `torch.xyz`
            torch_source = ImportSource("torch")
            install_guard(torch_source.make_guard(GuardBuilder.ID_MATCH))
            new_source = AttrSource(
                torch_source,
                module.__name__.rsplit(".", maxsplit=1)[-1],
            )
            return VariableTracker.build(tx, module, new_source)

        @register(
            torch.accelerator.current_stream,
            torch.cuda.current_stream,
            torch.xpu.current_stream,
        )
        def handle_current_stream(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> StreamVariable:
            from .streams import _get_stream_variable_cls

            if len(args) + len(kwargs) > 1 or (kwargs and "device" not in kwargs):
                unimplemented(
                    gb_type="unsupported arguments to torch.accelerator.current_stream",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="torch.accelerator.current_stream accepts one optional argument `device`",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            try:
                if kwargs:
                    device = torch.device(kwargs["device"].as_python_constant())
                elif args:
                    device = torch.device(args[0].as_python_constant())
                else:
                    device = None

                stream_var = tx.symbolic_stream_state.cur_stream(device)
                stream_variable_cls = _get_stream_variable_cls(self.value)
                if stream_variable_cls is not None and not isinstance(
                    stream_var, stream_variable_cls
                ):
                    stream_var = stream_variable_cls(
                        stream_var.proxy,
                        stream_var.value,
                        stream_var.user_object_index,
                        source=stream_var.source,
                    )
                return stream_var
            except Exception as e:
                unimplemented(
                    gb_type="bad device argument to torch.accelerator.current_stream",
                    context=f"args={args}, kwargs={kwargs}",
                    explanation="Expected valid string/torch.device argument ('cpu', 'cuda', etc.)",
                    hints=[*graph_break_hints.USER_ERROR],
                    from_exc=e,
                )

        _synchronize_fn_to_device_type = {
            torch.cuda.synchronize: "cuda",
            torch.xpu.synchronize: "xpu",
            torch.mps.synchronize: "mps",
            torch.cpu.synchronize: "cpu",
        }

        @register(
            torch.accelerator.synchronize,
            torch.cuda.synchronize,
            torch.xpu.synchronize,
            torch.mps.synchronize,
            torch.cpu.synchronize,
        )
        def handle_synchronize(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            device = None
            if kwargs and "device" in kwargs:
                device = torch.device(kwargs["device"].as_python_constant())
            elif args:
                device = torch.device(args[0].as_python_constant())

            if device is None:
                device_type = _synchronize_fn_to_device_type.get(self.value)
                if device_type is None:
                    # torch.accelerator.synchronize with no args
                    accelerator = torch.accelerator.current_accelerator()
                    if accelerator is None:
                        raise AssertionError(
                            "No accelerator available for torch.accelerator.synchronize"
                        )
                    device_type = accelerator.type
                device = torch.device(device_type)

            # CPU synchronize is a no-op, skip emitting the op
            if device.type == "cpu":
                return ConstantVariable.create(None)

            tx.output.create_proxy(
                "call_function",
                torch.ops.streams.synchronize_device,
                (device.type, device.index or 0),
                {},
            )
            return ConstantVariable.create(None)

        @register(torch.set_default_device)
        def handle_set_default_device(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            # Today this is inserted in the graph, once TF mode
            # handling is complete, we can trace the device context
            # like any other TF mode and remove this special handling
            # Insert the TF mode representing the device context at
            # the bottom of the stack to match the eager semantics
            # Running the graph will ensure that the DeviceContext mode is
            # at the correct position in the stack
            TorchFunctionModeStackVariable.register_mutation(tx)
            if args[0].is_constant_none():
                TorchFunctionModeStackVariable.clear_default_device(tx)
            else:
                TorchFunctionModeStackVariable.register_device_context_insertion(tx)

            return ConstantVariable.create(None)

        from torch._prims_common import elementwise_dtypes

        @register(elementwise_dtypes)
        def handle_elementwise_dtypes(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            from .builder import SourcelessBuilder

            type_promotion_kind = kwargs["type_promotion_kind"].as_python_constant()
            real_args = []
            for arg in args:
                if isinstance(arg, TensorVariable):
                    real_args.append(arg.as_proxy().node.meta["example_value"])
                elif arg.is_python_constant():
                    real_args.append(arg.as_python_constant())
                else:
                    unimplemented(
                        gb_type="elementwise_dtypes unsupported arg type",
                        context=str(arg),
                        explanation=(
                            "elementwise_dtypes requires tensor or constant arguments, "
                            f"got {type(arg).__name__}"
                        ),
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )
            result = elementwise_dtypes(
                *real_args, type_promotion_kind=type_promotion_kind
            )
            return SourcelessBuilder.create(tx, result)

        def check_impl(
            self,
            tx: "InstructionTranslatorBase",
            predicate_vt: VariableTracker | None,
            message_vt: VariableTracker | None,
            error_type: type[Exception] = RuntimeError,
        ) -> VariableTracker:
            if predicate_vt is None:
                raise_type_error(
                    tx,
                    f"{self.value.__name__}() missing 1 required positional argument: 'cond'",
                )

            if predicate_vt.is_tensor():
                raise_type_error(
                    tx,
                    "cond must be a bool, but got a Tensor. "
                    "For tensor conditions, use torch._check_tensor_all() "
                    "to assert that all elements are true.",
                )

            message_eager = None
            message_graph_arg = None
            # Realize a possible LazyVariableTracker so the exact-type checks
            # below see the underlying VariableTracker rather than the wrapper.
            if message_vt is not None:
                message_vt = message_vt.realize()
            match message_vt:
                case None:
                    pass
                case NestedUserFunctionVariable() | UserFunctionVariable() if type(
                    message_vt
                ) in (NestedUserFunctionVariable, UserFunctionVariable):
                    # These report is_python_constant() True (as_python_constant
                    # returns the underlying fn), so this arm must precede the
                    # constant arm below. Restrict to exact types: other
                    # function-VT subclasses (e.g. bound methods, whose
                    # get_function drops the receiver) cannot be faithfully
                    # reconstructed as a zero-arg message callable, so they fall
                    # through to the graph-break arm.
                    try:
                        message_eager = message_vt.get_function()
                    except ClosureConversionError:
                        unimplemented(
                            gb_type="Can't convert torch._check*() message closure",
                            context=str(message_vt),
                            explanation=(
                                "The message argument of torch._check*() must be a function "
                                "whose closure variables are Python constants."
                            ),
                            hints=[
                                "Remove closure variables that reference non-constant values, e.g. "
                                "remove references to tensor `x` in `lambda: f'{x} failed check'`",
                                *graph_break_hints.SUPPORTABLE,
                            ],
                        )

                    message_graph_arg = tx.output.register_static_attr_and_return_proxy(
                        "_check_message", message_eager
                    )
                case _ if message_vt.is_python_constant():
                    # Mirror eager torch._check, which accepts any constant and
                    # stringifies it; None means "use the default message".
                    message_eager = message_vt.as_python_constant()
                    if message_eager is not None:
                        message_graph_arg = str(message_eager)
                case _:
                    unimplemented(
                        gb_type="Can't extract message from torch._check*()",
                        context=str(message_vt),
                        explanation=(
                            "The message argument of torch._check*() must be a string, None, "
                            "or a function."
                        ),
                        hints=[
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )

            if predicate_vt.is_python_constant():
                if predicate_vt.as_python_constant():
                    return ConstantVariable.create(None)
                msg = (
                    str(message_eager() if callable(message_eager) else message_eager)
                    if message_eager is not None
                    else ("Expected cond to be True, but got False.")
                )
                raise_observed_exception(error_type, tx, args=[msg])

            predicate_proxy = predicate_vt.as_proxy()

            proxy_args: tuple[Any, ...]
            if message_graph_arg is None:
                proxy_args = (predicate_proxy,)
            else:
                proxy_args = (predicate_proxy, message_graph_arg)

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch._check,
                    proxy_args,
                    {},
                ),
            )

        @register(torch._check)
        def handle_check(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            predicate_vt = args[0] if args else kwargs.get("cond")
            rest_args = args[1:] if args else ()
            message_vt = rest_args[0] if rest_args else kwargs.get("message")
            return check_impl(self, tx, predicate_vt, message_vt)

        @register(
            torch._check_with,
            torch._check_index,
            torch._check_value,
            torch._check_type,
            torch._check_not_implemented,
        )
        def handle_check_with(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if self.value is torch._check_with:
                error_type_vt = args[0] if args else kwargs.get("error_type")
                rest_args = args[1:] if args else ()
            else:
                error_type_vt = None
                rest_args = args

            _CHECK_ERROR_TYPES: dict[object, type[Exception]] = {
                torch._check_index: IndexError,
                torch._check_value: ValueError,
                torch._check_type: TypeError,
                torch._check_not_implemented: NotImplementedError,
            }

            error_type = (
                error_type_vt.as_python_constant()
                if error_type_vt is not None
                else _CHECK_ERROR_TYPES.get(self.value, RuntimeError)
            )

            predicate_vt = rest_args[0] if rest_args else kwargs.get("cond")
            rest_args = rest_args[1:] if rest_args else ()
            message_vt = rest_args[0] if rest_args else kwargs.get("message")
            return check_impl(self, tx, predicate_vt, message_vt, error_type)

        @register(
            torch._check_tensor_all_with,
            torch._check_tensor_all,
        )
        def handle_check_tensor_all(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            if self.value is torch._check_tensor_all_with:
                error_type_vt = args[0] if args else kwargs.get("error_type")
                rest_args = args[1:] if args else ()
                error_type = (
                    error_type_vt.as_python_constant()
                    if error_type_vt is not None
                    else RuntimeError
                )
            else:
                rest_args = args
                error_type = RuntimeError

            cond_vt = rest_args[0] if rest_args else kwargs.get("cond")
            rest_args = rest_args[1:] if rest_args else ()
            message_vt = rest_args[0] if rest_args else kwargs.get("message")

            if cond_vt is None:
                return check_impl(self, tx, None, message_vt, error_type)

            scalar_vt = cond_vt.call_method(tx, "_is_all_true", [], {}).call_method(
                tx, "item", [], {}
            )

            return check_impl(self, tx, scalar_vt, message_vt, error_type)

        def exchange_device_helper(
            tx: "InstructionTranslatorBase",
            args: list[VariableTracker],
            kwargs: dict[str, VariableTracker],
            fn: Callable[[int], int | None],
        ) -> VariableTracker:
            if len(args) != 1 or kwargs:
                raise_type_error(
                    tx,
                    f"{fn.__name__} takes exactly one argument ({len(args)} given)",
                )
            torch_source = ImportSource("torch")
            install_guard(torch_source.make_guard(GuardBuilder.ID_MATCH))
            current_device_source = CallFunctionNoArgsSource(
                AttrSource(AttrSource(torch_source, "cuda"), "current_device")
            )
            install_guard(current_device_source.make_guard(GuardBuilder.EQUALS_MATCH))
            arg = args[0].as_python_constant()
            prev = fn(arg)
            tx.output.create_node(
                "call_function",
                fn,
                (arg,),
                {},
            )
            tx.output.add_cleanup_hook(lambda: torch.cuda.set_device(prev))
            return VariableTracker.build(tx, prev)

        @register(torch.cuda._exchange_device)
        def handle_exchange_device(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return exchange_device_helper(
                tx, list(args), kwargs, torch.cuda._exchange_device
            )

        @register(torch.cuda._maybe_exchange_device)
        def handle_maybe_exchange_device(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            return exchange_device_helper(
                tx, list(args), kwargs, torch.cuda._maybe_exchange_device
            )

        @register(torch._dynamo.decorators.override_optimization_hint)
        def handle_override_optimization_hint(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker:
            from .constant import ConstantVariable
            from .tensor import SymNodeVariable

            x_vt = args[0]
            val_vt = args[1]
            val = val_vt.as_python_constant()

            if x_vt.is_python_constant():
                torch._dynamo.decorators.override_optimization_hint(
                    x_vt.as_python_constant(), val
                )
                return ConstantVariable.create(None)

            if isinstance(x_vt, SymNodeVariable):
                torch._dynamo.decorators.override_optimization_hint(x_vt.sym_num, val)
                return ConstantVariable.create(None)

            raise UserError(
                UserErrorType.INVALID_INPUT,
                "override_optimization_hint expects a SymInt or int argument, "
                f"got {type(x_vt).__name__}",
            )

        @register(torch.autograd.grad)
        def handle_autograd_grad(
            self, tx: "InstructionTranslatorBase", *args, **kwargs
        ):
            """
            Handle torch.autograd.grad() calls within compiled regions.

            NOTE [Tracing autograd.grad in dynamo]

            We validate two things:

            1. External grad_fns cannot be consumed: The grad_fn on external inputs
               could change at runtime, so we would need to guard on it if we wanted
               to trace through it. For now, we reject this case.
               We compute "consumed" grad_fns (reachable from outputs, excluding
               autograd.grad inputs parameter) and verify no graph input's grad_fn is in this set.

            2. Returned tensors cannot have consumed grad_fns: If autograd.grad
               consumes a grad_fn and we return a tensor connected to it, the user
               would get "backward through graph a second time" error. We track
               consumed grad_fns and check at output time. If violated, we retry
               with a graph break at autograd.grad.

            Safe vs Unsafe Cases:

            Case 1 - Safe (external tensor is autograd.grad input):
                x = torch.randn(4, requires_grad=True)
                external = x * 2  # has external grad_fn

                @torch.compile
                def fn(external_input):
                    loss = external_input.sum()
                    return torch.autograd.grad(loss, external_input)

                Safe because autograd.grad stops at external_input, never consuming
                its external grad_fn.

            Case 2 - Unsafe (external grad_fn in path):
                @torch.compile
                def fn(external_input):
                    loss = mod(external_input).sum()
                    return torch.autograd.grad(loss, mod.weight)

                Unsafe because autograd.grad must traverse through external_input's
                grad_fn to reach mod.weight. The external grad_fn could change at
                runtime, so we would need to guard on it (like AOTAutograd does).
                For now, we reject this case.

            Case 3 - Unsafe (returning tensor with consumed grad_fn):
                @torch.compile
                def fn(x):
                    y = x * 2
                    grad = torch.autograd.grad(y.sum(), x)
                    return y, grad  # y's grad_fn was consumed!

                Unsafe because y's grad_fn was consumed by autograd.grad. Trying to
                backward through y later would error.
            """
            from .. import compiled_autograd, config
            from .builder import wrap_fx_proxy
            from .constant import ConstantVariable
            from .dicts import ConstDictVariable
            from .lists import BaseListVariable
            from .tensor import TensorVariable

            if not config.trace_autograd_ops:
                unimplemented(
                    gb_type="using `torch.autograd.grad` with `torch._dynamo.config.trace_autograd_ops=False`",
                    context=f"trace_autograd_ops={config.trace_autograd_ops}",
                    explanation=(
                        "Attempted to call `torch.autograd.grad` with config "
                        "`torch._dynamo.config.trace_autograd_ops` set to `False`."
                    ),
                    hints=[
                        "Change `torch._dynamo.config.trace_autograd_ops` to `True`.",
                    ],
                )

            # Graph break if we detected on a previous attempt that autograd.grad
            # consumed grad_fns of returned tensors. This gives better compile
            # coverage than failing the entire compile.
            if tx.speculation_log.graph_break_on_autograd_grad:
                leaked = tx.speculation_log.autograd_grad_leaked_tensors
                leaked_str = ", ".join(leaked) if leaked else "unknown"
                unimplemented(
                    gb_type="autograd.grad consumed returned tensor's grad_fn",
                    context=f"Leaked output tensors: {leaked_str}",
                    explanation=(
                        "torch.autograd.grad() consumes grad_fns that are needed by tensors "
                        "returned from this compiled function. This would cause 'backward "
                        "through graph a second time' errors.\n"
                        f"  The following returned tensors have consumed grad_fns: {leaked_str}"
                    ),
                    hints=[
                        f"Detach the problematic tensor(s) before returning: e.g. `{leaked[0]}.detach()`"
                        if leaked
                        else "Call .detach() on the tensor before returning.",
                        "If you need to backward through the returned tensor, use retain_graph=True in autograd.grad().",
                    ],
                )

            # Graph break if compiled_autograd is enabled.
            # Compiled autograd has limitations (e.g., view_fn in CopySlices)
            # that would cause errors during fake tensor execution.
            if compiled_autograd.compiled_autograd_enabled:
                unimplemented(
                    gb_type="autograd.grad with compiled autograd",
                    context="compiled_autograd is currently enabled",
                    explanation=(
                        "torch.autograd.grad() inside torch.compile is not supported when "
                        "compiled autograd is enabled. These two features have conflicting "
                        "requirements for how the autograd graph is traced."
                    ),
                    hints=[
                        "Disable compiled autograd by removing the compiled_autograd context manager.",
                        "Or move the autograd.grad() call outside the torch.compile region.",
                        "Or restructure your code so autograd.grad() and compiled_autograd don't overlap.",
                    ],
                )

            # Check for external GradientEdge objects in outputs and inputs args
            # if there is it will be a graph break
            if len(args) >= 1:
                _check_for_gradient_edge(args[0], "outputs")
            if len(args) >= 2:
                _check_for_gradient_edge(args[1], "inputs")

            # Collect external grad_fn objects from graph inputs, along with their sources.
            # We need to collect ALL grad_fns associated with each input tensor:
            # - Direct grad_fn, base tensor's grad_fn (for views)
            # - Inner tensors (for subclasses)
            external_grad_fns: set[torch.autograd.graph.Node] = set()
            # Map grad_fn -> source name for better error messages
            grad_fn_to_source: dict[torch.autograd.graph.Node, str] = {}
            for var in tx.output.input_source_to_var.values():
                if isinstance(var, TensorVariable):
                    fake_tensor = var.as_proxy().node.meta.get("example_value")
                    if not isinstance(fake_tensor, torch.Tensor):
                        raise AssertionError(
                            f"Expected fake_tensor to be a torch.Tensor, got {type(fake_tensor)}"
                        )
                    tensor_grad_fns = _collect_all_grad_fns(fake_tensor)
                    external_grad_fns.update(tensor_grad_fns)
                    # Track source name for error messages
                    if var.source is not None:
                        for gf in tensor_grad_fns:
                            grad_fn_to_source[gf] = var.source.name

            # Collect tensors from outputs and inputs args
            from ..output_graph import collect_reachable_grad_fns

            outputs_with_sources = (
                _collect_tensors_with_sources(args[0]) if len(args) >= 1 else []
            )
            inputs_with_sources = (
                _collect_tensors_with_sources(args[1]) if len(args) >= 2 else []
            )

            # Collect grad_fns from the autograd.grad inputs tensors to use as stop points.
            # For non-leaf tensors: we stop at their grad_fn
            # For leaf tensors (requires_grad=True, grad_fn=None): we don't add anything here,
            # but this is fine because their AccumulateGrad is created during fake tensor
            # tracing and is not in external_grad_fns, so it won't trigger a false positive.
            inputs_grad_fns: set[torch.autograd.graph.Node] = set()
            for tensor, _ in inputs_with_sources:
                if isinstance(tensor, torch.Tensor) and tensor.grad_fn is not None:
                    inputs_grad_fns.add(tensor.grad_fn)

            # Collect all consumed grad_fns that are reachable from outputs, stopping at inputs.
            #
            # Note: Do not try to "optimize" by only checking inputs in the `inputs` arg.
            # Without guarding on the autograd graph, we can't distinguish:
            #   Case 1: x, y are independent leaves -> OK, y's path not consumed
            #   Case 2: y = x * 2 (y.grad_fn external) -> BAD, we hit external grad_fn
            # Since the same compiled code could be called with either, we must check
            # ALL graph inputs for external grad_fns.
            consumed_grad_fns = collect_reachable_grad_fns(
                outputs_with_sources, stop_at=inputs_grad_fns
            )

            # Check if any graph input's grad_fn is in the consumed set.
            # If so, autograd.grad would need to traverse through external autograd nodes,
            # which we cannot trace. (If a graph input is also an autograd.grad input,
            # its grad_fn is already excluded from consumed_grad_fns via stop_at.)
            external_in_consumed = consumed_grad_fns & external_grad_fns

            if external_in_consumed:
                sources = [
                    grad_fn_to_source[gf]
                    for gf in external_in_consumed
                    if gf in grad_fn_to_source
                ]
                context = f"inputs with external grad_fn: {sources}" if sources else ""
                unimplemented(
                    gb_type="autograd.grad with external grad_fn",
                    context=context,
                    explanation=(
                        "torch.autograd.grad() cannot trace through the autograd graph because "
                        "it's output depends on a tensor that was created outside "
                        "the compiled region and has a grad_fn attached. The autograd graph "
                        "extends beyond the compiled region boundary, which Dynamo cannot trace."
                    ),
                    hints=[
                        "If you don't need gradients to flow back to the original tensor outside "
                        "the compiled region, detach the input: `tensor.detach().requires_grad_(True)`.",
                        "Otherwise, move the autograd.grad() call outside the compiled region.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            # Track consumed grad_fns for later validation
