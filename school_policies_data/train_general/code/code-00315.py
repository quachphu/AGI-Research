                _get_instruction_by_offset(offset_to_inst, entry.start),  # type: ignore[arg-type]
                _get_instruction_by_offset(offset_to_inst, end_offset),  # type: ignore[arg-type]
                _get_instruction_by_offset(offset_to_inst, entry.target),  # type: ignore[arg-type]
                entry.depth,
                entry.lasti,
            )
            return entry, inst_entry

        entry, inst_entry = step()
        for inst in instructions:
            if inst.offset is None:
                raise AssertionError(
                    f"instruction {inst.opname} has no offset during exception "
                    "table virtualization"
                )
            while inst.offset > entry.end:
                entry, inst_entry = step()
            if inst.offset >= entry.start:
                inst.exn_tab_entry = copy.copy(inst_entry)
    except StopIteration:
        pass


def compute_exception_table(
    instructions: list[Instruction],
) -> list[ExceptionTableEntry]:
    """Compute exception table in list format from instructions with exn_tab_entries"""
    exn_dict: dict[tuple[int, int], tuple[int, int, bool]] = {}
    indexof = get_indexof(instructions)

    for inst in instructions:
        if inst.exn_tab_entry:
            # account for prefixed EXTENDED_ARGS
            start = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.start]
            ).offset
            if start is None:
                raise AssertionError(
                    "exception table entry start instruction has no offset"
                )
            # point to the last 2 bytes of the end instruction
            end = (
                cast(int, inst.exn_tab_entry.end.offset)
                + instruction_size(inst.exn_tab_entry.end)
                - 2
            )
            if end is None:
                raise AssertionError(
                    "exception table entry end instruction has no offset"
                )
            target = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.target]
            ).offset
            if target is None:
                raise AssertionError(
                    "exception table entry target instruction has no offset"
                )
            key = (start, end)
            val = (target, inst.exn_tab_entry.depth, inst.exn_tab_entry.lasti)
            if key in exn_dict:
                if exn_dict[key] != val:
                    raise AssertionError(
                        f"conflicting exception table entries for range "
                        f"{key}: {exn_dict[key]} vs {val}"
                    )
            exn_dict[key] = val

    # Dynamo may construct nested exception table entries for convenience,
    # but Python expects exception table entries to not overlap.
    # NOTE: below, "keys" refer to old instruction entries' starts and ends,
    # and "entries" refer to the generated exception table entries.

    # Sort keys by increasing start, then decreasing end
    keys_sorted = sorted(exn_dict.keys(), key=lambda t: (t[0], -t[1]))
    # smallest byte that the next exception table entry can start at
    nexti = 0
    # stack of current nested keys
    key_stack: list[tuple[int, int]] = []
    exn_tab: list[ExceptionTableEntry] = []

    def pop() -> None:
        """
        Pop the key_stack and append an exception table entry if possible.
        """
        nonlocal nexti
        if key_stack:
            key = key_stack.pop()
            if nexti <= key[1]:
                exn_tab.append(
                    ExceptionTableEntry(max(key[0], nexti), key[1], *exn_dict[key])
                )
                nexti = key[1] + 2

    for key in keys_sorted:
        # pop keys that are no longer nested over the current key
        while key_stack and key_stack[-1][1] < key[0]:
            pop()
        if key_stack:
            # create an entry covering to the current key, if possible
            if not (key_stack[-1][0] <= key[0] <= key[1] <= key_stack[-1][1]):
                raise AssertionError(
                    f"exception table key {key} is not properly nested within "
                    f"parent {key_stack[-1]}"
                )
            left = max(nexti, key_stack[-1][0])
            if left < key[0]:
                exn_tab.append(
                    ExceptionTableEntry(left, key[0] - 2, *exn_dict[key_stack[-1]])
                )
            nexti = key[0]
        key_stack.append(key)
    while key_stack:
        pop()
    check_exception_table(exn_tab)
    return exn_tab


def check_inst_exn_tab_entries_nested(
    tab: list[InstructionExnTabEntry], indexof: dict[Instruction, int]
) -> None:
    """
    Checks `tab` is a properly sorted list of nested InstructionExnTabEntry's,
    i.e. no entries partially overlap.
    "Properly sorted" means entries are sorted by increasing starts, then
    decreasing ends.
    """
    entry_stack: list[tuple[int, int]] = []
    for entry in tab:
        key = (indexof[entry.start], indexof[entry.end])
        while entry_stack and entry_stack[-1][1] < key[0]:
            entry_stack.pop()
        if entry_stack:
            if not (entry_stack[-1][0] <= key[0] <= key[1] <= entry_stack[-1][1]):
                raise AssertionError(
                    f"exception table entry {key} is not properly nested "
                    f"within parent {entry_stack[-1]}"
                )
        entry_stack.append(key)


def propagate_inst_exn_table_entries(instructions: list[Instruction]) -> None:
    """
    Copies exception table entries to all instructions in an entry's range.
    Supports nested exception table entries.
    """
    indexof = get_indexof(instructions)
    entries: dict[tuple[int, int], InstructionExnTabEntry] = {}
    for inst in instructions:
        if inst.exn_tab_entry:
            key = (
                indexof[inst.exn_tab_entry.start],
                indexof[inst.exn_tab_entry.end],
            )
            if key in entries:
                if inst.exn_tab_entry != entries[key]:
                    raise AssertionError(
                        f"conflicting exn_tab_entry for instruction at key {key}"
                    )
            entries[key] = inst.exn_tab_entry
    sorted_entries = [
        entries[key] for key in sorted(entries.keys(), key=lambda t: (t[0], -t[1]))
    ]
    check_inst_exn_tab_entries_nested(sorted_entries, indexof)
    # Propagation of nested entries works since nested entries come later
    # in sorted order.
    for entry in sorted_entries:
        for i in range(indexof[entry.start], indexof[entry.end] + 1):
            instructions[i].exn_tab_entry = copy.copy(entry)


def check_inst_exn_tab_entries_valid(instructions: list[Instruction]) -> None:
    """
    Checks that exn_tab_entries of instructions are valid.
    An entry's start, end, and target must be in instructions.
    Instructions with an exn_tab_entry are located within
    the entry's start and end instructions.
    Instructions do not share exn_tab_entries.

    Implicitly checks for no duplicate instructions.
    """
    indexof = get_indexof(instructions)
    exn_tab_entry_set = set()
    for i, inst in enumerate(instructions):
        if inst.exn_tab_entry:
            if sys.version_info < (3, 11):
                raise AssertionError("exn_tab_entry is only supported in Python 3.11+")
            if id(inst.exn_tab_entry) in exn_tab_entry_set:
                raise AssertionError(
                    f"duplicate exn_tab_entry at instruction index {i}"
                )
            exn_tab_entry_set.add(id(inst.exn_tab_entry))
            entry = inst.exn_tab_entry
            if entry.start not in indexof:
                raise AssertionError(
                    "exn_tab_entry start instruction not found in instructions"
                )
            if entry.end not in indexof:
                raise AssertionError(
                    "exn_tab_entry end instruction not found in instructions"
                )
            if entry.target not in indexof:
                raise AssertionError(
                    "exn_tab_entry target instruction not found in instructions"
                )
            if not (indexof[entry.start] <= i <= indexof[entry.end]):
                raise AssertionError(
                    f"instruction at index {i} is outside its exn_tab_entry "
                    f"range [{indexof[entry.start]}, {indexof[entry.end]}]"
                )


def strip_extended_args(instructions: list[Instruction]) -> None:
    instructions[:] = [i for i in instructions if i.opcode != dis.EXTENDED_ARG]


# Overwrites old_inst with a sequence of new instructions.
# This is necessary in order to preserve jump targets to the old
# instruction, exception table entries, and positions.
# Returns the modified sequence of instructions (including the modified
# old instruction!) that can be manipulated elsewhere.
def overwrite_instruction(
    old_inst: Instruction, new_insts: list[Instruction]
) -> list[Instruction]:
    # update old_inst.exn_tab_entry.end if necessary
    if (
        old_inst.exn_tab_entry
        and old_inst.exn_tab_entry.end is old_inst
        and len(new_insts) > 1
    ):
        old_inst.exn_tab_entry.end = new_insts[-1]
    # preserve exception table entries and positions
    for inst in new_insts[1:]:
        inst.exn_tab_entry = copy.copy(old_inst.exn_tab_entry)
        inst.positions = old_inst.positions
    # modify old_inst in-place to preserve jump target
    old_inst.opcode = new_insts[0].opcode
    old_inst.opname = new_insts[0].opname
    old_inst.arg = new_insts[0].arg
    old_inst.argval = new_insts[0].argval
    old_inst.target = new_insts[0].target
    return [old_inst] + new_insts[1:]


def remove_load_call_method(instructions: list[Instruction]) -> list[Instruction]:
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
    if sys.version_info >= (3, 11):
        raise AssertionError(
            "remove_load_call_method should not be called on Python 3.11+"
        )
    rewrites = {"LOAD_METHOD": "LOAD_ATTR", "CALL_METHOD": "CALL_FUNCTION"}
    for inst in instructions:
        if inst.opname in rewrites:
            inst.opname = rewrites[inst.opname]
            inst.opcode = dis.opmap[inst.opname]
    return instructions


def remove_jump_if_none(instructions: list[Instruction]) -> None:
    new_insts = []
    for inst in instructions:
        if "_NONE" in inst.opname:
            is_op = create_instruction("IS_OP", arg=int("NOT" in inst.opname))
            # need both argval and arg set correctly now (not later)
            is_op.argval = is_op.arg

            if sys.version_info < (3, 12):
                jump_op = create_instruction(
                    (
                        "POP_JUMP_FORWARD_IF_TRUE"
                        if "FORWARD" in inst.opname
                        else "POP_JUMP_BACKWARD_IF_TRUE"
                    ),
                    target=inst.target,
                )
            else:
                jump_op = create_instruction("POP_JUMP_IF_TRUE", target=inst.target)

            replace_insts = [
                create_instruction("LOAD_CONST", argval=None),
                is_op,
                jump_op,
            ]
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


def remove_binary_store_slice(instructions: list[Instruction]) -> None:
    new_insts = []
    for inst in instructions:
        new_insts.append(inst)
        if inst.opname in ("BINARY_SLICE", "STORE_SLICE"):
            # new instruction
            if sys.version_info >= (3, 14) and inst.opname == "BINARY_SLICE":
                subscr_inst = create_binary_subscr()
            else:
                subscr_inst = create_instruction(inst.opname.replace("SLICE", "SUBSCR"))
            if inst.exn_tab_entry and inst.exn_tab_entry.end is inst:
                inst.exn_tab_entry.end = subscr_inst
            subscr_inst.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            subscr_inst.positions = inst.positions
            # modify inst in-place to preserve jump target
            inst.opcode = dis.opmap["BUILD_SLICE"]
            inst.opname = "BUILD_SLICE"
            inst.arg = 2
            inst.argval = 2
            new_insts.append(subscr_inst)
    instructions[:] = new_insts


def remove_load_attr_method_variant(instructions: list[Instruction]) -> None:
    """On 3.12+, LOAD_ATTR with arg%2==1 is the method variant that pushes 2
    values (NULL/self + method).  Normalize it to a non-method LOAD_ATTR
    (1 value) followed by PUSH_NULL (+ SWAP on 3.12 where NULL goes below
    the callable).  This lets break_graph_if_unsupported handle LOAD_ATTR
    uniformly as a single-output instruction."""
    new_insts: list[Instruction] = []
    for inst in instructions:
        if inst.opname == "LOAD_ATTR" and inst.arg is not None and inst.arg % 2:
            replace_insts = [
                create_instruction("LOAD_ATTR", arg=inst.arg & ~1, argval=inst.argval),
                create_instruction("PUSH_NULL"),
            ]
            if sys.version_info < (3, 13):
                replace_insts.append(create_instruction("SWAP", arg=2))
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


FUSED_INSTS = {
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST_BORROW", "LOAD_FAST_BORROW"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
}


def remove_fused_load_store(instructions: list[Instruction]) -> None:
    new_insts = []
    for inst in instructions:
        if inst.opname in FUSED_INSTS:
            inst0, inst1 = FUSED_INSTS[inst.opname]
            argval0, argval1 = inst.argval

            replace_insts = [
                create_instruction(inst0, argval=argval0),
                create_instruction(inst1, argval=argval1),
            ]
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


# adds GRAPH_BREAK_IF_LEAF (not a real instruction) before RETURN_* instructions
# for testing purposes
def add_graph_break_if_leaf_instructions(instructions: list[Instruction]) -> None:
    new_insts = []
    for inst in instructions:
        if "RETURN" in inst.opname:
            replace_insts = [
                create_instruction("NOP", argval="GRAPH_BREAK_IF_LEAF"),
                create_instruction(inst.opname, argval=inst.argval),
            ]
            new_insts.extend(overwrite_instruction(inst, replace_insts))
        else:
            new_insts.append(inst)
    instructions[:] = new_insts


def remove_graph_break_if_leaf_instructions(instructions: list[Instruction]) -> None:
    new_insts = []
    for inst, next_inst in itertools.pairwise(instructions):
        if (
            inst.opname == "NOP"
            and inst.argval == "GRAPH_BREAK_IF_LEAF"
            and next_inst.opname.startswith("RETURN")
        ):
            # remove this instruction and update all other instructions' jump targets
            for i in range(len(instructions)):
                if instructions[i].target is inst:
                    instructions[i].target = next_inst
                if instructions[i].exn_tab_entry:
                    # linter is mistakenly complaining that None has no attribute "..."
                    # but this codepath only runs if instructions[i] is not None
                    if instructions[i].exn_tab_entry.start is inst:  # type: ignore[union-attr]
                        instructions[i].exn_tab_entry.start = next_inst  # type: ignore[union-attr]
                    if instructions[i].exn_tab_entry.end is inst:  # type: ignore[union-attr]
                        instructions[i].exn_tab_entry.end = next_inst  # type: ignore[union-attr]
                    if instructions[i].exn_tab_entry.target is inst:  # type: ignore[union-attr]
                        instructions[i].exn_tab_entry.target = next_inst  # type: ignore[union-attr]
        else:
            new_insts.append(inst)
    new_insts.append(instructions[-1])
    instructions[:] = new_insts


def explicit_super(code: types.CodeType, instructions: list[Instruction]) -> None:
    """convert super() with no args into explicit arg form"""
    cell_and_free = (code.co_cellvars or ()) + (code.co_freevars or ())
    if not len(code.co_varnames):
        # A function with no argument cannot contain a valid "super()" call
        return
    output = []
    for idx, inst in enumerate(instructions):
        output.append(inst)
        if inst.opname == "LOAD_GLOBAL" and inst.argval == "super":
            nexti = instructions[idx + 1]
            if nexti.arg == 0 and (
                (sys.version_info >= (3, 12) and nexti.opname == "CALL")
                or (
                    sys.version_info >= (3, 11)
                    and sys.version_info < (3, 12)
                    and nexti.opname == "PRECALL"
                )
                or (sys.version_info < (3, 11) and nexti.opname == "CALL_FUNCTION")
            ):
                if "__class__" not in cell_and_free:
                    raise AssertionError(
                        "__class__ not found in cell_and_free for super() call"
                    )
                output.append(create_instruction("LOAD_DEREF", argval="__class__"))
                first_var = code.co_varnames[0]
                if first_var in cell_and_free:
                    output.append(create_instruction("LOAD_DEREF", argval=first_var))
                else:
                    output.append(create_instruction("LOAD_FAST", argval=first_var))
                nexti.arg = 2
                nexti.argval = 2
                if nexti.opname == "PRECALL":
                    # also update the following CALL instruction
                    call_inst = instructions[idx + 2]
                    call_inst.arg = 2
                    call_inst.argval = 2

    instructions[:] = output


def fix_extended_args(instructions: list[Instruction]) -> int:
    """Fill in correct argvals for EXTENDED_ARG ops"""
    output: list[Instruction] = []

    def maybe_pop_n(n: int) -> None:
        for _ in range(n):
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()

    for inst in instructions:
        if inst.opcode == dis.EXTENDED_ARG:
            # Leave this instruction alone for now so we never shrink code
            inst.arg = 0
        elif inst.arg and inst.arg > 0xFFFFFF:
            maybe_pop_n(3)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 24))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFFFF:
            maybe_pop_n(2)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFF:
            maybe_pop_n(1)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        output.append(inst)

    added = len(output) - len(instructions)
    if added < 0:
        raise AssertionError(
            f"fix_extended_args removed instructions (added={added}), "
            "expected non-negative"
        )
    instructions[:] = output
    return added


def instruction_size(inst: Instruction) -> int:
    import torch

    if sys.version_info >= (3, 11):
        return 2 * (torch._C._dynamo.eval_frame.py_opcode_caches[inst.opcode] + 1)
    return 2


def check_offsets(instructions: Sequence[Instruction]) -> None:
    offset = 0
    for inst in instructions:
        if inst.offset != offset:
            raise AssertionError(
                f"instruction {inst.opname} has offset {inst.offset}, expected {offset}"
            )
        offset += instruction_size(inst)


def update_offsets(instructions: Sequence[Instruction]) -> None:
    offset = 0
    for inst in instructions:
        inst.offset = offset

        offset += instruction_size(inst)


def debug_bytes(*args: bytes) -> str:
    index = range(max(map(len, args)))
    result = [
        " ".join(f"{x:03}" for x in arg)
        for arg in [index]
        + list(args)
        + [[int(a != b) for a, b in zip(args[-1], args[-2])]]
    ]

    return "bytes mismatch\n" + "\n".join(result)


def debug_checks(code: types.CodeType) -> None:
    """Make sure our assembler produces same bytes as we start with"""
    dode, _ = transform_code_object(code, lambda x, y: None, safe=True)
    if code.co_code != dode.co_code:
        raise AssertionError(debug_bytes(code.co_code, dode.co_code))
    # TODO: reenable after better understanding how to handle sourceless
    # (None) codes
    # assert list(code.co_lines()) == list(dode.co_lines()), (
    #     "line table mismatch via co_lines()"
    # )


HAS_LOCAL = set(dis.haslocal)
HAS_NAME = set(dis.hasname)
HAS_FREE = set(dis.hasfree)
HAS_CONST = set(dis.hasconst)


def get_const_index(code_options: dict[str, Any], val: Any) -> int:
    for i, v in enumerate(code_options["co_consts"]):
        # NOTE: stronger comparison is required, since we have
        # examples where two values compare equal but have
        # different semantic meaning in some cases, e.g.
        # 0.0 == -0.0 but have different effects in torch.copysign.
        if val is v:
            return i
    code_options["co_consts"] += (val,)
    return len(code_options["co_consts"]) - 1


def fix_vars(
    instructions: list[Instruction],
    code_options: dict[str, Any],
    varname_from_oparg: Callable[..., Any] | None = None,
) -> None:
    # compute instruction arg from argval if arg is not provided
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}

    def get_name_index(name: str) -> int:
        try:
            idx = names[name]
        except KeyError:
            # Add a missing item to co_names
            idx = names[name] = len(names)
            code_options["co_names"] = (*code_options["co_names"], name)
            if len(code_options["co_names"]) != len(names):
                raise AssertionError(
                    f"co_names length {len(code_options['co_names'])} does not "
                    f"match names dict length {len(names)}"
                ) from None
        return idx

    if sys.version_info < (3, 11):
        if varname_from_oparg is not None:
            raise AssertionError("varname_from_oparg should be None for Python < 3.11")
        varnames = {name: idx for idx, name in enumerate(code_options["co_varnames"])}
        freenames = {
            name: idx
            for idx, name in enumerate(
                code_options["co_cellvars"] + code_options["co_freevars"]
            )
        }
    else:
        if not callable(varname_from_oparg):
            raise AssertionError("varname_from_oparg must be callable for Python 3.11+")
        allnames = {}
        for idx in itertools.count():
            try:
                name = varname_from_oparg(idx)
                allnames[name] = idx
            except IndexError:
                break
        varnames = {name: allnames[name] for name in code_options["co_varnames"]}
        freenames = {
            name: allnames[name]
            for name in code_options["co_cellvars"] + code_options["co_freevars"]
        }
    for i in range(len(instructions)):

        def should_compute_arg() -> bool:
            # argval is prioritized over arg
            return instructions[i].argval is not _NotProvided

        if instructions[i].opname == "LOAD_GLOBAL":
            # 3.11 LOAD_GLOBAL requires both arg and argval - see create_instruction
            if instructions[i].argval is _NotProvided:
                raise AssertionError("LOAD_GLOBAL instruction must have argval set")
            if sys.version_info >= (3, 11):
                if instructions[i].arg is None:
                    raise AssertionError(
                        "LOAD_GLOBAL instruction must have arg set in Python 3.11+"
                    )
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_ATTR":
            # 3.12 LOAD_ATTR requires both arg and argval, like LOAD_GLOBAL
            if instructions[i].argval is _NotProvided:
                raise AssertionError("LOAD_ATTR instruction must have argval set")
            if sys.version_info >= (3, 12):
                if instructions[i].arg is None:
                    raise AssertionError(
                        "LOAD_ATTR instruction must have arg set in Python 3.12+"
                    )
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_SUPER_ATTR":
            if instructions[i].arg is None:
                raise AssertionError("LOAD_SUPER_ATTR instruction must have arg set")
            if instructions[i].argval is _NotProvided:
                raise AssertionError("LOAD_SUPER_ATTR instruction must have argval set")
            # Copy low bit, force second bit on for explicit super (the "+ 2")
            instructions[i].arg = (
                (get_name_index(instructions[i].argval) << 2)
                + (cast(int, instructions[i].arg) % 2)
                + 2
            )
        elif instructions[i].opname in FUSED_INSTS:
            if sys.version_info < (3, 13):
                raise AssertionError(
                    f"fused instruction {instructions[i].opname} requires Python 3.13+"
                )
            if not isinstance(instructions[i].argval, tuple):
                raise AssertionError(
                    f"fused instruction {instructions[i].opname} argval must be a tuple, "
                    f"got {type(instructions[i].argval)}"
                )
            if len(instructions[i].argval) != 2:
                raise AssertionError(
                    f"fused instruction {instructions[i].opname} argval must have "
                    f"length 2, got {len(instructions[i].argval)}"
                )
            arg_tuple = tuple(
                varnames[name] if name in varnames else freenames[name]
                for name in instructions[i].argval
            )
            instructions[i].arg = (arg_tuple[0] << 4) + (arg_tuple[1] & 15)
        elif instructions[i].opcode in HAS_LOCAL:
            if should_compute_arg():
                if (
                    sys.version_info >= (3, 13)
                    and instructions[i].argval not in varnames
                ):
                    # instructions like LOAD_FAST used for both local and free vars
                    instructions[i].arg = freenames[instructions[i].argval]
                else:
                    instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            if should_compute_arg():
                arg = get_name_index(instructions[i].argval)
                if (
                    sys.version_info >= (3, 15)
                    and instructions[i].opname == "IMPORT_NAME"
                ):
                    arg <<= 2
                    # Set the second bit to ensure eager imports, since lazy imports are only valid at module scope
                    arg |= 0x02
                instructions[i].arg = arg
        elif instructions[i].opcode in HAS_FREE:
            if should_compute_arg():
                instructions[i].arg = freenames[instructions[i].argval]
        elif instructions[i].opcode in HAS_CONST:
            # NOTE: only update argval if arg is not provided. This assumes
            # that any additions to co_consts are appended.
            if instructions[i].arg is None:
                # cannot use a dictionary since consts may not be hashable
                idx = get_const_index(code_options, instructions[i].argval)
                if idx < 0:
                    raise AssertionError(
                        f"failed to find constant index for {instructions[i].argval}"
                    )
                instructions[i].arg = idx


def clear_instruction_args(instructions: list[Instruction]) -> None:
    # Clear the instruction arg for instructions that have argvals.
    # Useful for using dis'd bytecode within generated bytecode.
    for inst in instructions:
        if (
            inst.argval is not _NotProvided
            and (
                inst.opcode in HAS_LOCAL
                or inst.opcode in HAS_NAME
                or inst.opcode in HAS_FREE
                or inst.opcode in HAS_CONST
            )
            and inst.opname not in ("LOAD_GLOBAL", "LOAD_ATTR", "LOAD_SUPER_ATTR")
        ):
            inst.arg = None


@functools.lru_cache
def get_code_keys() -> list[str]:
    # Python 3.11 changes to code keys are not fully documented.
    # See https://github.com/python/cpython/blob/3.11/Objects/clinic/codeobject.c.h#L24
    # for new format.
    keys = ["co_argcount"]
    keys.append("co_posonlyargcount")
    keys.extend(
        [
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_filename",
            "co_name",
        ]
    )
    if sys.version_info >= (3, 11):
        keys.append("co_qualname")
    keys.append("co_firstlineno")
    keys.append("co_linetable")
    if sys.version_info >= (3, 11):
        # not documented, but introduced in https://github.com/python/cpython/issues/84403
        keys.append("co_exceptiontable")
    keys.extend(
        [
            "co_freevars",
            "co_cellvars",
        ]
    )
    return keys


def transform_code_object(
    code: types.CodeType,
    transformations: Callable[
        [list[Instruction], dict[str, Any]], Optional["DynamoTracerOutput"]
    ],
    safe: bool = False,
) -> tuple[types.CodeType, Optional["DynamoTracerOutput"]]:
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    if len(code_options["co_varnames"]) != code_options["co_nlocals"]:
        raise AssertionError(
            f"co_varnames length {len(code_options['co_varnames'])} does not "
            f"match co_nlocals {code_options['co_nlocals']}"
        )

    instructions = cleaned_instructions(code, safe)
    # propagate line nums again for added instructions
    propagate_line_nums(instructions)

    tracer_output = transformations(instructions, code_options)
    _, bytecode = clean_and_assemble_instructions(instructions, keys, code_options)
    return bytecode, tracer_output


def clean_and_assemble_instructions(
    instructions: list[Instruction], keys: list[str], code_options: dict[str, Any]
) -> tuple[list[Instruction], types.CodeType]:
    remove_graph_break_if_leaf_instructions(instructions)
    # also implicitly checks for no duplicate instructions
    check_inst_exn_tab_entries_valid(instructions)

    code_options["co_nlocals"] = len(code_options["co_varnames"])
    varname_from_oparg = None
    if sys.version_info >= (3, 11):
        # temporary code object with updated names
        tmp_code = types.CodeType(*[code_options[k] for k in keys])
        varname_from_oparg = tmp_code._varname_from_oparg  # type: ignore[attr-defined]
    fix_vars(instructions, code_options, varname_from_oparg=varname_from_oparg)

    dirty = True
    while dirty:
        update_offsets(instructions)
        devirtualize_jumps(instructions)
        # this pass might change offsets, if so we need to try again
        dirty = bool(fix_extended_args(instructions))

    remove_extra_line_nums(instructions)
    bytecode, lnotab = assemble(instructions, code_options["co_firstlineno"])

    code_options["co_linetable"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_stacksize"] = stacksize_analysis(instructions)
    if set(keys) - {"co_posonlyargcount"} != set(code_options.keys()) - {
        "co_posonlyargcount"
    }:
        raise AssertionError(
            f"code_options keys mismatch: expected {set(keys) - {'co_posonlyargcount'}}, "
            f"got {set(code_options.keys()) - {'co_posonlyargcount'}}"
        )
    if sys.version_info >= (3, 11):
        code_options["co_exceptiontable"] = assemble_exception_table(
            compute_exception_table(instructions)
        )

    return instructions, types.CodeType(*[code_options[k] for k in keys])


def populate_kw_names_argval(instructions: Sequence[Instruction], consts: Any) -> None:
    for inst in instructions:
        if inst.opname == "KW_NAMES":
            inst.argval = consts[inst.arg]


# If safe=True, we do not make any bytecode modifications.
# Mainly used for debugging bytecode_transformation (see debug_checks)
def cleaned_instructions(code: types.CodeType, safe: bool = False) -> list[Instruction]:
    instructions = _cached_cleaned_instructions(code, safe)
    # We have a lot of code that implicitly mutates the instruction array. We
    # could do better here by making the copies explicit when necessary.
    return _clone_instructions(instructions)


# Copy an instructions array, making sure to remap the individual instruction targets.
def _clone_instructions(instructions: Sequence[Instruction]) -> list[Instruction]:
    # This is super hot and this is the fastest way to do this (tried copy.copy
    # and dataclasses.replace).
    copied = [
        Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.starts_line,
            i.is_jump_target,
            i.positions,
            i.target,
            i.exn_tab_entry,
            i.argrepr,
        )
        for i in instructions
    ]

    remap = dict(zip(instructions, copied))
    # Handle `None` in the remapper so we don't need an extra `if`.
    remap[None] = None  # type: ignore[index, assignment]

    for i in copied:
        i.target = remap[i.target]  # type: ignore[index]
        if entry := i.exn_tab_entry:
            i.exn_tab_entry = InstructionExnTabEntry(
                remap[entry.start],
                remap[entry.end],
                remap[entry.target],
                entry.depth,
                entry.lasti,
            )
    return copied


@functools.lru_cache
def _cached_cleaned_instructions(
    code: types.CodeType, safe: bool = False
) -> Sequence[Instruction]:
    instructions = list(map(convert_instruction, dis.get_instructions(code)))
    # propagate now in case we remove some instructions
    propagate_line_nums(instructions)
    check_offsets(instructions)
    if sys.version_info >= (3, 11):
        populate_kw_names_argval(instructions, code.co_consts)
        virtualize_exception_table(code.co_exceptiontable, instructions)
    virtualize_jumps(instructions)
    strip_extended_args(instructions)
    if not safe:
        if sys.version_info < (3, 11):
            remove_load_call_method(instructions)
        if sys.version_info < (3, 12):
            explicit_super(code, instructions)
        if sys.version_info >= (3, 11):
            remove_jump_if_none(instructions)
            if sys.version_info >= (3, 12):
                remove_binary_store_slice(instructions)
                remove_load_attr_method_variant(instructions)
            if sys.version_info >= (3, 13):
                remove_fused_load_store(instructions)
        if config.debug_force_graph_break_on_leaf_return:
            add_graph_break_if_leaf_instructions(instructions)
    if sys.version_info >= (3, 11):
        update_offsets(instructions)
        devirtualize_jumps(instructions)
    return instructions


_unique_id_counter = itertools.count()


def unique_id(name: str, with_uuid: bool = False) -> str:
    ret = f"{name}_{next(_unique_id_counter)}"
    if with_uuid:
        ret += f"_{uuid.uuid4()}".replace("-", "_")
    return ret


COMPILED_FN_PREFIX = "__compiled_fn"
_COMPILED_FN_NAME_RE = re.compile(
    rf"^{COMPILED_FN_PREFIX}_\d+_"
    r"[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}$"
)


def make_compiled_fn_name() -> str:
    return unique_id(COMPILED_FN_PREFIX, with_uuid=True)


def is_compiled_fn_name(name: str) -> bool:
    return _COMPILED_FN_NAME_RE.fullmatch(name) is not None


def is_generator(code: types.CodeType) -> bool:
    co_generator = 0x20
    return (code.co_flags & co_generator) > 0


def bytecode_from_template(
    fn: Callable[..., Any],
    varname_map: Mapping[Any, Any] | None = None,
    noreturn: bool = True,
    noprefix: bool = True,
) -> list[Instruction]:
    """Generates bytecode from a template function `fn` for use in
    dynamo bytecode generation.

    For example, we can generate Python-version-independent bytecode
    for looping through a dictionary and copying the values to a new dictionary.

    def template(d1, d2):
        for k, v in d1.items():
            d2[k] = v


    or a try block:

    def template():
        try:
            dummy1
        except:
            dummy2
            raise
        dummy3

    Args:
        fn: a function template to generate bytecode from
        varname_map: a mapping of `fn`'s varnames to new names. This
            map will be applied to the generated bytecode's varnames.
            For example, local variables in `fn` can be replaced with
            new names that are generated by `OutputGraph.new_var`.
        noreturn: remove all RETURN_* bytecodes and replace them with a jump
            to the end of the bytecode. NOTE: any items pushed to the stack
            for return WILL remain on the stack! Append a POP_TOP if you don't want
            that item to be present.
        noprefix: remove prefix bytecodes (all bytecode before the first RESUME, inclusive).
    """
    insts = cleaned_instructions(fn.__code__)
    clear_instruction_args(insts)

    if noprefix:
        for i, inst in enumerate(insts):
            if inst.opname == "RESUME":
                insts = insts[i + 1 :]
                break

    for inst in insts:
        # If we don't reset starts_line, then the generated
        # bytecode's line number will be based on fn's.
        inst.starts_line = None
        inst.positions = None
        if varname_map and inst.argval in varname_map:
            inst.argval = varname_map[inst.argval]

    if noreturn:
        if sys.version_info >= (3, 12):
            # replace RETURN_CONST with LOAD_CONST RETURN_VALUE
            new_insts = []
            for inst in insts:
                if inst.opname == "RETURN_CONST":
                    inst.opcode = dis.opmap["LOAD_CONST"]
                    inst.opname = "LOAD_CONST"
                    new_insts.append(inst)
                    # no need to propagate target/exn table
                    new_insts.append(create_instruction("RETURN_VALUE"))
                else:
                    new_insts.append(inst)
            insts = new_insts

        # pyrefly: ignore [implicit-any]
        returns = []
        for inst in insts:
            if inst.opname == "RETURN_VALUE":
                returns.append(inst)

        if len(returns) == 1 and returns[0] is insts[-1]:
            # only 1 return at the end - just pop it
            insts.pop(-1)
        elif len(returns) > 0:
            # create jump target - if the last inst is a return,
            # we can replace it with a NOP and make that the jump target.
            if insts[-1] is returns[-1]:
                insts[-1].opname = "NOP"
                insts[-1].opcode = dis.opmap["NOP"]
                insts[-1].arg = None
                insts[-1].argval = _NotProvided
                returns.pop(-1)
            else:
                insts.append(create_instruction("NOP"))

            # replace returns with jumps
            for inst in returns:
                # don't replace inst with new instruction
                # due to targeting/exn table/etc.
                jump_inst = create_jump_absolute(insts[-1])
                inst.opname = jump_inst.opname
                inst.opcode = jump_inst.opcode
                inst.arg = jump_inst.arg
                inst.argval = jump_inst.argval
                inst.target = jump_inst.target

    return insts
<|endoftext|>import weakref
from collections.abc import Callable
from typing import Any

from torch._dynamo.source import Source


PyCodegen = Any

# This file is to handle types that we don't want to support
# as explicit FX graph inputs. This uses a sidetable which
# we populate in bytecode and is loaded during graph execution

# We use a dynamo-generated index as a level of indirection
# this allows us to register objects externally in pre-graph bytecode that we want
# to pass to the graph, but not support their types as graph inputs
index_to_bytecode_constructor: dict[int, Callable[[PyCodegen], None]] = {}

index_to_external_object_weakref: dict[int, weakref.ReferenceType[Any]] = {}

keep_alive: list[Any] = []


def has_user_objects() -> bool:
    return bool(index_to_bytecode_constructor)


def stash_graph_created_object(obj: Any) -> Any:
    keep_alive.append(obj)
    return obj


CURRENT_STREAM_INDEX = 0


def set_external_object_by_index(index: int, value: Any) -> None:
    """Update an entry in the external object registry at runtime."""
    keep_alive.append(value)
    index_to_external_object_weakref[index] = weakref.ref(value)


def get_external_object_by_index(index: int) -> Any:
    if index not in index_to_external_object_weakref:
        raise AssertionError("Index not registered in index_to_user_object_weakref")
    obj = index_to_external_object_weakref[index]()
    if obj is None:
        raise AssertionError("User object is no longer alive")
    return index_to_external_object_weakref[index]()


def store_user_object_weakrefs(*args: Any) -> None:
    global index_to_external_object_weakref
    index_to_external_object_weakref.clear()
    index_to_external_object_weakref.update(
        {i: weakref.ref(arg) for i, arg in enumerate(args)}
    )


def reset_user_object_tracking() -> None:
    index_to_bytecode_constructor.clear()
    index_to_external_object_weakref.clear()
    keep_alive.clear()


def register_graph_created_object(
    example_value: Any, construct_fn: Callable[[int, PyCodegen], None]
) -> int:
    global index_to_bytecode_constructor
    global keep_alive
    keep_alive.append(example_value)
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: construct_fn(index, cg)
    try:
        index_to_external_object_weakref[index] = weakref.ref(example_value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to graph-created external object",
            context=f"user_object: {example_value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a user object to be used in the graph
def register_user_object(value: Any, source: Source) -> int:
    global index_to_bytecode_constructor
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: cg(source)
    try:
        index_to_external_object_weakref[index] = weakref.ref(value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a callback so invoke_leaf_function can retrieve nn.Module instances at runtime.
# We use a callback pattern instead of having invoke_leaf_function import get_external_object_by_index
# directly, because higher-order ops should not depend on dynamo (dynamo depends on them, not vice versa).
from torch._higher_order_ops.invoke_leaf_function import (
    set_leaf_function_module_retriever,
)


set_leaf_function_module_retriever(get_external_object_by_index)
<|endoftext|>"""
This module provides callback management functionality for TorchDynamo's compilation process.

It implements a thread-safe system for registering, managing and executing callbacks that run
at the start and end of TorchDynamo compilations. Key features include:

- Registration and deregistration of compilation callbacks
- Thread-safe callback handling with proper locking mechanisms
- Prevention of duplicate callback execution when configured
- Decorator utilities for easy callback registration
- Context manager for controlled callback lifecycle

The module centers around the CompilationCallbackHandler class which maintains separate
lists for start and end callbacks, manages their execution order, and ensures thread-safety.
Utility decorators @on_compile_start and @on_compile_end provide a convenient way to
register compilation hooks.

Example usage:
    @on_compile_start
    def my_start_callback():
        print("Starting compilation")

    @on_compile_end
    def my_end_callback():
        print("Compilation complete")
"""

import enum
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


class CallbackTrigger(enum.Enum):
    # most common case, dynamo attempts to trace a new frame
    DYNAMO = 1
    # backward compilation can be deferred to runtime
    LAZY_BACKWARD = 2
    # some backends autotune at runtime
    TRITON_AUTOTUNING = 3  # Temporarily disabled due to spam
    # cudagraphs record at runtime
    CUDAGRAPH_RECORDING = 4


@dataclass
class CallbackArgs:
    callback_trigger: CallbackTrigger
    compile_id: str


@dataclass
class CompilationCallbackHandler:
    start_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)
    end_callbacks: list[Callable[[CallbackArgs], None]] = field(default_factory=list)

    __pending_callbacks_counter: int = field(default=0, init=False, repr=False)
    __pending_callbacks_counter_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def register_start_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]:
        """
        Register a callback function to be called when the compilation starts.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.start_callbacks.append(callback)
        return callback

    def register_end_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]:
        """
        Register a callback function to be called when the compilation ends.

        Args:
        - callback (Callable): The callback function to register.
        """
        self.end_callbacks.append(callback)
        return callback

    def remove_start_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
        """
        Remove a registered start callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.start_callbacks.remove(callback)

    def remove_end_callback(self, callback: Callable[[CallbackArgs], None]) -> None:
        """
        Remove a registered end callback function.

        Args:
        - callback (Callable): The callback function to remove.
        """
        self.end_callbacks.remove(callback)

    def run_start_callbacks(self, args: CallbackArgs) -> None:
        """
        Execute all registered start callbacks.
        """
        for callback in self.start_callbacks:
            callback(args)

    def run_end_callbacks(self, args: CallbackArgs) -> None:
        """
        Execute all registered end callbacks.
        """
        for callback in self.end_callbacks:
            callback(args)

    @contextmanager
    def install_callbacks(
        self, trigger: CallbackTrigger, compile_id: str
    ) -> Generator[None, Any, Any]:
        """
        Context manager to install the callbacks and run them when the context is exited.
        """
        args = CallbackArgs(trigger, compile_id)
        try:
            with self.__pending_callbacks_counter_lock:
                self.__pending_callbacks_counter += 1
                if self.__pending_callbacks_counter == 1:
                    self.run_start_callbacks(args)
            yield
        finally:
            with self.__pending_callbacks_counter_lock:
                if self.__pending_callbacks_counter <= 0:
                    raise AssertionError(
                        "Pending callbacks counter cannot become negative."
                    )
                if self.__pending_callbacks_counter == 1:
                    self.run_end_callbacks(args)
                self.__pending_callbacks_counter -= 1

    def clear(self) -> None:
        """
        Clear all registered callbacks.
        """
        self.start_callbacks.clear()
        self.end_callbacks.clear()
        if self.__pending_callbacks_counter != 0:
            raise AssertionError(
                f"Cannot clear callbacks while {self.__pending_callbacks_counter} are pending"
            )


callback_handler = CompilationCallbackHandler()


def on_compile_start(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]:
    """
    Decorator to register a callback function for the start of the compilation.
    """
    callback_handler.register_start_callback(callback)
    return callback


def on_compile_end(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]:
    """
    Decorator to register a callback function for the end of the compilation.
    """
    callback_handler.register_end_callback(callback)
    return callback
<|endoftext|>"""
This module provides thread-safe code context management for TorchDynamo using weak references.

The CodeContextDict class maintains a mapping between Python code objects and their associated
context data, using weak references to automatically clean up entries when code objects are
garbage collected. This prevents memory leaks while allowing context data to be associated
with code objects throughout their lifecycle.

Key features:
- Thread-safe context storage and retrieval
- Automatic cleanup using weak references
- Safe context management for Python code objects
- Memory-leak prevention

Example usage:
    code_obj = compile('x = 1', '<string>', 'exec')

    # Store context
    context = code_context.get_context(code_obj)
    context['metadata'] = {'optimized': True}

    # Retrieve context
    if code_context.has_context(code_obj):
        ctx = code_context.get_context(code_obj)
        # Use context data...

    # Remove context
    ctx = code_context.pop_context(code_obj)
"""

import types
from typing import Any

from .utils import ExactWeakKeyDictionary


class CodeContextDict:
    def __init__(self) -> None:
        self.code_context: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    def has_context(self, code: types.CodeType) -> bool:
        return code in self.code_context

    def get_context(self, code: types.CodeType) -> dict[str, Any]:
        ctx = self.code_context.get(code)
        if ctx is None:
            # pyrefly: ignore [implicit-any]
            ctx = {}
            self.code_context[code] = ctx
        return ctx

    def pop_context(self, code: types.CodeType) -> dict[str, Any]:
        ctx = self.get_context(code)
        self.code_context._remove_id(id(code))
        return ctx

    def clear(self) -> None:
        self.code_context.clear()


code_context: CodeContextDict = CodeContextDict()
<|endoftext|>"""
This module provides functionality for resuming Python execution at specific points in code,
primarily used by PyTorch Dynamo for control flow handling and optimization. It implements
bytecode transformation and execution state management to enable:

- Resuming execution at arbitrary points in Python bytecode
- Managing context managers and their state across execution boundaries
- Transforming and generating new code objects with preserved execution state
- Supporting Python 3.11+ exception handling and block management
- Restoring torch function mode stacks and other execution context

The module is critical for PyTorch Dynamo's ability to optimize code while preserving
Python semantics and execution state.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from typing import Any, cast, TYPE_CHECKING

from .bytecode_transformation import (
    add_push_null,
    bytecode_from_template,
    create_binary_subscr,
    create_call_function,
    create_call_function_ex,
    create_instruction,
    create_jump_absolute,
    create_load_const,
    Instruction,
    overwrite_instruction,
    transform_code_object,
    unique_id,
)
from .utils import ExactWeakKeyDictionary


if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Iterable
    from contextlib import AbstractContextManager

    from .output_graph import CodeOptions


# taken from code.h in cpython
CO_OPTIMIZED = 0x0001
CO_NEWLOCALS = 0x0002
CO_VARARGS = 0x0004
CO_VARKEYWORDS = 0x0008
CO_NESTED = 0x0010
CO_GENERATOR = 0x0020
CO_NOFREE = 0x0040
CO_COROUTINE = 0x0080
CO_ITERABLE_COROUTINE = 0x0100
CO_ASYNC_GENERATOR = 0x0200

# trace_rules.py import this constant for consistency
TORCH_DYNAMO_RESUME_IN_PREFIX = "torch_dynamo_resume_in"
IS_TRACING_RESUME_PROLOGUE_VARNAME = "__is_tracing_resume_prologue"


# If is_resume - this codegen is for a resume function
def _initial_push_null(insts: list[Instruction]) -> None:
    if sys.version_info >= (3, 11):
        insts.append(create_instruction("PUSH_NULL"))
        if sys.version_info < (3, 13):
            insts.append(create_instruction("SWAP", arg=2))


# Generates bytecode from template and splits the code where LOAD_FAST dummy is present.
def _bytecode_from_template_with_split(
    template: Callable[..., Any],
    stack_index: int,
    varname_map: dict[str, Any] | None = None,
) -> tuple[list[Instruction], list[Instruction]]:
    template_code = bytecode_from_template(template, varname_map=varname_map)
    template_code.append(create_instruction("POP_TOP"))

    # adjust exception table entry depth
    for inst in template_code:
        if inst.exn_tab_entry:
            inst.exn_tab_entry.depth += stack_index

    # search for LOAD_FAST dummy and replace it with 2 NOPs (we can break up the bytecode between them)
    dummy_idx, dummy_inst = next(
        (
            (i, inst)
            for i, inst in enumerate(template_code)
            if inst.opname in ("LOAD_FAST", "LOAD_FAST_BORROW")
            and inst.argval == "dummy"
        ),
        (None, None),
    )
    if dummy_idx is None:
        raise AssertionError("LOAD_FAST dummy instruction not found in template code")
    if dummy_inst is None:
        raise AssertionError("LOAD_FAST dummy instruction instance is None")

    # replace LOAD_FAST dummy with first NOP marking exception area
    overwrite_instruction(dummy_inst, [create_instruction("NOP")])

    # POP_TOP follows LOAD_FAST dummy - replace with NOP marking end of exception area
    if template_code[dummy_idx + 1].opname != "POP_TOP":
        raise AssertionError(
            f"Expected POP_TOP after LOAD_FAST dummy, got {template_code[dummy_idx + 1].opname}"
        )
    overwrite_instruction(template_code[dummy_idx + 1], [create_instruction("NOP")])

    return template_code[: dummy_idx + 1], template_code[dummy_idx + 1 :]


def _try_except_tf_mode_template(dummy: Any) -> None:
    # NOTE: Make sure this name matches what is generated by symbolic_convert:import_source
    # on torch._dynamo.utils.
    # pyrefly: ignore [unknown-name]
    global __import_torch_dot__dynamo_dot_utils
    try:
        dummy
    except:
        __import_torch_dot__dynamo_dot_utils._pop_torch_function_stack()  # type: ignore[name-defined]
        raise


@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int
    target_values: tuple[Any, ...] | None = None

    def try_except_torch_function_mode(
        self, code_options: CodeOptions, cleanup: list[Instruction]
    ) -> list[Instruction]:
        """
        Codegen based off of:
        try:
            (rest)
        except:
            (pop the torch function mode that was pushed for this context)
            raise
        """
        setup_try_except, epilogue = _bytecode_from_template_with_split(
            _try_except_tf_mode_template,
            self.stack_index,
        )
        cleanup[:] = epilogue + cleanup

        return setup_try_except

    # If we do not want to destroy the stack, we can do the same thing as a
    # `SETUP_WITH` block, only that we store the context manager in a local_symbol
    def try_finally(
        self, code_options: CodeOptions, cleanup: list[Instruction]
    ) -> list[Instruction]:
        """
        Codegen based off of:
        load args
        enter context
        try:
            (rest)
        finally:
            exit context
        """
        # NOTE: we assume that TOS is a context manager CLASS!
        # pyrefly: ignore [implicit-any]
        load_args = []
        if self.target_values:
            load_args = [create_load_const(val) for val in self.target_values]
        ctx_name = unique_id(f"___context_manager_{self.stack_index}")
        if ctx_name not in code_options["co_varnames"]:
            code_options["co_varnames"] += (ctx_name,)
        for name in ["__enter__", "__exit__"]:
            if name not in code_options["co_names"]:
                code_options["co_names"] += (name,)

        create_ctx: list[Instruction] = []
        _initial_push_null(create_ctx)
        create_ctx.extend(
            [
                *load_args,
                *create_call_function(len(load_args), False),
                create_instruction("STORE_FAST", argval=ctx_name),
            ]
        )

        def _template(ctx: AbstractContextManager[Any], dummy: Any) -> None:
            ctx.__enter__()
            try:
                dummy
            finally:
                ctx.__exit__(None, None, None)

        setup_try_finally, epilogue = _bytecode_from_template_with_split(
            _template, self.stack_index, varname_map={"ctx": ctx_name}
        )
        cleanup[:] = epilogue + cleanup
        return create_ctx + setup_try_finally

    def __call__(
        self, code_options: dict[str, Any], cleanup: list[Instruction]
    ) -> tuple[list[Instruction], Instruction | None]:
        """
        Codegen based off of:
        with ctx(args):
            (rest)
        """
        # NOTE: we assume that TOS is a context manager CLASS!
        # pyrefly: ignore [implicit-any]
        load_args = []
        if self.target_values:
            load_args = [create_load_const(val) for val in self.target_values]

        create_ctx: list[Instruction] = []
        # Do not push NULL in Python 3.14+ since the NULL should be on the symbolic stack.
        if sys.version_info < (3, 14):
            _initial_push_null(create_ctx)
        create_ctx.extend(
            [
                *load_args,
                *create_call_function(len(load_args), False),
            ]
        )

        def _template(ctx: AbstractContextManager[Any], dummy: Any) -> None:
            with ctx:
                dummy

        setup_with, epilogue = _bytecode_from_template_with_split(
            _template, self.stack_index
        )
        cleanup[:] = epilogue + cleanup

        load_fast_ctx_inst = next(
            (
                inst
                for inst in setup_with
                if inst.opname in ("LOAD_FAST", "LOAD_FAST_BORROW")
                and inst.argval == "ctx"
            ),
            None,
        )
        if load_fast_ctx_inst is None:
            raise AssertionError("LOAD_FAST ctx instruction not found in setup_with")
        # ctx already loaded on stack before the template - no need to LOAD_FAST
        overwrite_instruction(load_fast_ctx_inst, [create_instruction("NOP")])

        # 3.11+ only
        push_exc_info_gen = (
            inst for inst in epilogue if inst.opname == "PUSH_EXC_INFO"
        )
        push_exc_info_inst = next(push_exc_info_gen, None)
        # expect only 1 PUSH_EXC_INFO in epilogue
        if next(push_exc_info_gen, None) is not None:
            raise AssertionError("Expected only 1 PUSH_EXC_INFO in epilogue")

        return create_ctx + setup_with, push_exc_info_inst


@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: list[Instruction] = dataclasses.field(default_factory=list)
    # Python 3.11+ fields
    # NOTE: Python 3.11 removed blocks, but for our purposes, a "block" consists
    # of instructions of all exception table entries that have the same target.

    # map from PUSH_EXC_INFO's in the prefix to original block target offset
    prefix_block_target_offset_remap: list[int] = dataclasses.field(
        default_factory=list
    )
    # per-offset map from new block target offsets to original block target offsets
    block_target_offset_remap: dict[tuple[int, int], dict[int, int]] = (
        dataclasses.field(default_factory=dict)
    )


def _filter_iter(
    l1: Iterable[Any],
    l2: Iterable[Any],
    cond: Callable[[Any, Any], bool],
) -> list[Any]:
    """
    Two-pointer conditional filter.
    e.g. _filter_iter(insts, sorted_offsets, lambda i, o: i.offset == o)
    returns the instructions with offsets in sorted_offsets
    """
    it = iter(l2)
    res: list[Instruction] = []
    try:
        cur = next(it)
        for val in l1:
            if cond(val, cur):
                res.append(val)
                cur = next(it)
    except StopIteration:
        pass
    return res


def _load_tuple_and_call(tup: tuple[Any, ...]) -> list[Instruction]:
    insts: list[Instruction] = []
    _initial_push_null(insts)
    insts.extend(create_load_const(val) for val in tup)
    insts.extend(create_call_function(len(tup), False))
    return insts


class ContinueExecutionCache:
    cache = ExactWeakKeyDictionary()
    generated_code_metadata = ExactWeakKeyDictionary()

    @classmethod
    def lookup(
        cls, code: types.CodeType, lineno: int, init_offset: int, *key: Any
    ) -> types.CodeType:
        if code not in cls.cache:
            cls.cache[code] = {}
        key = tuple(key)
        if key not in cls.cache[code]:
            cls.cache[code][key] = cls.generate(code, lineno, init_offset, *key)
        return cls.cache[code][key]

    @classmethod
    def generate(
        cls,
        code: types.CodeType,
        lineno: int,
        init_offset: int,
        resume_offset: int,
        setup_fn_target_offsets: tuple[int, ...],  # only used in Python 3.11+
        nstack: int,
        argnames: tuple[str, ...],
        argnames_null: tuple[str, ...],
        setup_fns: tuple[ReenterWith, ...],
        handle_inactive_ctx: bool,
        stack_ctx_vars: tuple[tuple[int, tuple[Any, ...]], ...],
        argnames_ctx_vars: tuple[tuple[str, tuple[Any, ...]], ...],
        null_idxes: tuple[int, ...],
        # mainly used to ensure distinct code objects per stack trace,
        # which prevents excessive recompilation of inner frames
        nested_code_objs: tuple[types.CodeType],
        # Are we currently graph breaking on an instruction that doesn't push
        # its result to the stack? If so, and we are not the leaf resume, then we need to pop
        # the result of calling the next resume function.
        pop_nested_resume_result: bool,
    ) -> types.CodeType:
        if resume_offset is None:
            raise AssertionError("resume_offset must not be None")
        if code.co_flags & (
            CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR
        ):
            raise AssertionError(
                "Cannot generate resume function for generator, coroutine, or async generator"
            )
        if not (code.co_flags & CO_OPTIMIZED):
            raise AssertionError("Code object must have CO_OPTIMIZED flag set")
        if code in ContinueExecutionCache.generated_code_metadata:
            return cls.generate_based_on_original_code_object(
                code,
                lineno,
                init_offset,
                resume_offset,
                setup_fn_target_offsets,
                nstack,
                argnames,
                argnames_null,
                setup_fns,
                handle_inactive_ctx,
                stack_ctx_vars,
                argnames_ctx_vars,
                null_idxes,
                nested_code_objs,
                pop_nested_resume_result,
            )

        is_py311_plus = sys.version_info >= (3, 11)
        meta = ResumeFunctionMetadata(code)

        def update(
            instructions: list[Instruction], code_options: dict[str, Any]
        ) -> None:
            meta.instructions = copy.deepcopy(instructions)

            args = ["__nested_resume_fns", "__nested_frame_values"]
            args += [f"___stack{i}" for i in range(nstack)]
            args.extend(v for v in argnames if v not in args)
            freevars = tuple(code_options["co_cellvars"] or []) + tuple(
                code_options["co_freevars"] or []
            )
            freevars = tuple(sorted(freevars))
            code_options["co_name"] = (
                f"{TORCH_DYNAMO_RESUME_IN_PREFIX}_{code_options['co_name']}_at_{lineno}"
            )
            if is_py311_plus:
                qualified_path = code_options["co_qualname"].rsplit(".", maxsplit=1)
                if len(qualified_path) == 1:
                    code_options["co_qualname"] = code_options["co_name"]
                else:
                    if len(qualified_path) != 2:
                        raise AssertionError(
                            f"Expected qualified path to have 2 parts, got {len(qualified_path)}"
                        )
                    module_name, co_name = qualified_path
                    code_options["co_qualname"] = (
                        f"{module_name}.{TORCH_DYNAMO_RESUME_IN_PREFIX}_{co_name}_at_{lineno}"
                    )
            code_options["co_firstlineno"] = lineno
            code_options["co_cellvars"] = ()
            code_options["co_freevars"] = freevars
            code_options["co_argcount"] = len(args)
            code_options["co_posonlyargcount"] = 0
            code_options["co_kwonlyargcount"] = 0
            code_options["co_varnames"] = tuple(
                args
                + [v for v in argnames_null if v not in args]
                + [v for v in code_options["co_varnames"] if v not in args]
                + [IS_TRACING_RESUME_PROLOGUE_VARNAME]
            )
            code_options["co_flags"] = code_options["co_flags"] & ~(
                CO_VARARGS | CO_VARKEYWORDS
            )
            target = next(i for i in instructions if i.offset == resume_offset)

            prefix = []
            if is_py311_plus:
                if freevars:
                    prefix.append(
                        create_instruction("COPY_FREE_VARS", arg=len(freevars))
                    )
                prefix.append(create_instruction("RESUME", arg=0))

            # Set is_tracing_resume_prologue to prevent graph breaks.
            # This doesn't really do anything at runtime, but dynamo will trace this
            # and will know that we're in a resume function prologue.
            prefix.extend(
                [
                    create_instruction("LOAD_CONST", argval=True),
                    create_instruction(
                        "STORE_FAST", argval=IS_TRACING_RESUME_PROLOGUE_VARNAME
                    ),
                ]
            )

            cleanup: list[Instruction] = []
            hooks = {fn.stack_index: fn for fn in setup_fns}
            hook_target_offsets = {
                fn.stack_index: setup_fn_target_offsets[i]
                for i, fn in enumerate(setup_fns)
            }
            offset_to_inst = {inst.offset: inst for inst in instructions}
            # map old hook targets to new targets generated by the hook
            # pyrefly: ignore [implicit-any]
            old_hook_target_remap = {}
            stack_i = 0
            null_i = 0
            stack_ctx_vars_d = dict(stack_ctx_vars)  # type: ignore[var-annotated,arg-type]
            for i in range(nstack + len(null_idxes)):
                if null_i < len(null_idxes) and null_idxes[null_i] == i:
                    prefix.append(create_instruction("PUSH_NULL"))
                    null_i += 1
                else:
                    prefix.append(
                        create_instruction("LOAD_FAST", argval=f"___stack{stack_i}")
                    )
                    if handle_inactive_ctx and stack_i in stack_ctx_vars_d:
                        # NOTE: we assume that current stack var is a context manager CLASS!
                        # Load args for context variable and construct it
                        prefix.extend(_load_tuple_and_call(stack_ctx_vars_d[stack_i]))
                    stack_i += 1

                if i in hooks:
                    hook = hooks.pop(i)
                    hook_insts, exn_target = hook(code_options, cleanup)
                    prefix.extend(hook_insts)
                    if is_py311_plus:
                        hook_target_offset = hook_target_offsets.pop(i)
                        old_hook_target = offset_to_inst[hook_target_offset]
                        meta.prefix_block_target_offset_remap.append(hook_target_offset)
                        old_hook_target_remap[old_hook_target] = exn_target

            if is_py311_plus:
                # reverse the mapping since targets of later/nested contexts are inserted
                # into the mapping later, but show up earlier in the prefix.
                meta.prefix_block_target_offset_remap = list(
                    reversed(meta.prefix_block_target_offset_remap)
                )

            if hooks:
                raise AssertionError(f"Unprocessed hooks remaining: {hooks}")

            # NOTE: we assume that local var is a context manager CLASS!
            # initialize inactive context vars in argnames
            if handle_inactive_ctx:
                for name, vals in argnames_ctx_vars:
                    prefix.append(create_instruction("LOAD_FAST", argval=name))
                    prefix.extend(_load_tuple_and_call(vals))
                    prefix.append(create_instruction("STORE_FAST", argval=name))

            # 3.12+: store NULL into variables that were NULL
            if argnames_null:
                if sys.version_info < (3, 12):
                    raise AssertionError(
                        f"argnames_null requires Python 3.12+, got {sys.version_info}"
                    )
                for v in argnames_null:
                    if v in args:
                        raise AssertionError(
                            f"argnames_null variable {v!r} should not be in args"
                        )
                    prefix.extend(
                        [
                            create_instruction("PUSH_NULL"),
                            create_instruction("STORE_FAST", argval=v),
                        ]
                    )

            # Call nested resume function
            if nested_code_objs:
                prefix.extend(
                    [
                        # set up __nested_resume_fns[-1] call
                        *add_push_null(
                            [
                                create_instruction(
                                    "LOAD_FAST", argval="__nested_resume_fns"
                                ),
                                create_instruction("LOAD_CONST", argval=-1),
                                create_binary_subscr(),
                            ]
                        ),
                        # del __nested_resume_fns[-1]
                        create_instruction("LOAD_FAST", argval="__nested_resume_fns"),
                        create_instruction("LOAD_CONST", argval=-1),
                        create_instruction("DELETE_SUBSCR"),
                        # load [__nested_resume_fns, __nested_frame_values]
                        create_instruction("LOAD_FAST", argval="__nested_resume_fns"),
                        create_instruction("LOAD_FAST", argval="__nested_frame_values"),
                        create_instruction("BUILD_LIST", arg=2),
                        # load __nested_frame_values[-1]
                        create_instruction("LOAD_FAST", argval="__nested_frame_values"),
                        create_instruction("LOAD_CONST", argval=-1),
                        create_binary_subscr(),
                        # create [
                        #     __nested_resume_fns,
                        #     __nested_frame_values,
                        #     *__nested_frame_values[-1],
                        # ]
                        create_instruction("LIST_EXTEND", arg=1),
                        # del __nested_frame_values[-1]
                        create_instruction("LOAD_FAST", argval="__nested_frame_values"),
                        create_instruction("LOAD_CONST", argval=-1),
                        create_instruction("DELETE_SUBSCR"),
                        # delete __nested values
                        create_instruction("DELETE_FAST", argval="__nested_resume_fns"),
                        create_instruction(
                            "DELETE_FAST", argval="__nested_frame_values"
                        ),
                        # Set is_tracing_resume_prologue back to allow graph breaks
                        # in the nested resume
                        create_instruction("LOAD_CONST", argval=False),
                        create_instruction(
                            "STORE_FAST", argval=IS_TRACING_RESUME_PROLOGUE_VARNAME
                        ),
                        # finish the call
                        *create_call_function_ex(False, False),
                    ]
                )
                if pop_nested_resume_result:
                    # pop the result of calling the nested resume function
                    prefix.append(create_instruction("POP_TOP"))
            else:
                # Set is_tracing_resume_prologue back to allow graph breaks after the jump
                prefix.extend(
                    [
                        create_instruction("LOAD_CONST", argval=False),
                        create_instruction(
                            "STORE_FAST", argval=IS_TRACING_RESUME_PROLOGUE_VARNAME
                        ),
                    ]
                )

            prefix.append(create_jump_absolute(target))

            # because the line number table monotonically increases from co_firstlineno
            # remove starts_line for any instructions before the graph break instruction
            # this will ensure the instructions after the break have the correct line numbers
            for inst in instructions:
                if inst.offset == target.offset:
                    break
                inst.starts_line = None
                if sys.version_info >= (3, 11):
                    inst.positions = None

            if cleanup:
                prefix.extend(cleanup)
                prefix.extend(cls.unreachable_codes(code_options))

            # remap original instructions' exception table entries
            if old_hook_target_remap:
                # pyrefly: ignore [unbound-name]
                if not is_py311_plus:
                    raise AssertionError("old_hook_target_remap requires Python 3.11+")
                for inst in instructions:
                    if (
                        inst.exn_tab_entry
                        and inst.exn_tab_entry.target in old_hook_target_remap
                    ):
                        inst.exn_tab_entry.target = old_hook_target_remap[  # type: ignore[assignment]
                            inst.exn_tab_entry.target
                        ]

            # TODO(jansel): add dead code elimination here
            instructions[:] = prefix + instructions

        new_code, _ = transform_code_object(code, update)
        ContinueExecutionCache.generated_code_metadata[new_code] = meta
        return new_code

    @staticmethod
    def unreachable_codes(code_options: dict[str, Any]) -> list[Instruction]:
        """Codegen a `raise None` to make analysis work for unreachable code"""
        return [
            create_load_const(None),
            create_instruction("RAISE_VARARGS", arg=1),
        ]

    @classmethod
    def generate_based_on_original_code_object(
        cls,
        code: types.CodeType,
        lineno: int,
        init_offset: int,
        resume_offset: int,
        setup_fn_target_offsets: tuple[int, ...],
        *args: Any,
    ) -> types.CodeType:
        """
        This handles the case of generating a resume into code generated
        to resume something else.  We want to always generate starting
        from the original code object so that if control flow paths
        converge we only generated 1 resume function (rather than 2^n
        resume functions).
        """

        meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[
            code
        ]

        def find_orig_offset(cur_offset: int) -> int:
            orig_offset = -1

            def find_orig_offset_transform(
                instructions: list[Instruction], code_options: dict[str, Any]
            ) -> None:
                nonlocal orig_offset
                (target,) = (i for i in instructions if i.offset == cur_offset)
                # match the functions starting at the last instruction as we have added a prefix
                new_target_tuple = tuple(
                    i2
                    for i1, i2 in zip(
                        reversed(instructions), reversed(meta.instructions)
                    )
                    if i1 is target
                )

                if not new_target_tuple:
                    # Instruction with cur_offset in instructions was not found
                    # in the original code - orig_offset left as -1.
                    # Caller expected to handle this case.
                    return

                if len(new_target_tuple) != 1:
                    raise AssertionError(
                        f"Expected exactly 1 matching target, got {len(new_target_tuple)}"
                    )
                new_target = new_target_tuple[0]

                if target.opcode != new_target.opcode:
                    raise AssertionError(
                        f"Opcode mismatch: target has {target.opcode}, "
                        f"new_target has {new_target.opcode}"
                    )
                if new_target.offset is None:
                    raise AssertionError("new_target.offset must not be None")
                orig_offset = new_target.offset

            transform_code_object(code, find_orig_offset_transform)
            return orig_offset

        orig_init_offset = find_orig_offset(init_offset)
        # It is fine if the initial instruction is not found in the original code;
        # this means we graph broke in the prefix, which only happens with nested graph breaks.
        # We should not be running into ambiguous graph break issues here.
        orig_resume_offset = find_orig_offset(resume_offset)
        if orig_resume_offset <= -1:
            raise AssertionError(
                "resume instruction not found in original code - this is a bug."
            )

        if sys.version_info >= (3, 11):
            # setup_fn_target_offsets currently contains the target offset of
            # each setup_fn, based on `code`. When we codegen the resume function
            # based on the original code object, `meta.code`, the offsets in
            # setup_fn_target_offsets must be based on `meta.code` instead.
            offset_key = (orig_init_offset, orig_resume_offset)
            # NOTE: we key by offset_key since the same resume function may graph
            # break in multiple places and we need different block_target_offset_remap's
            # for each graph break location. Keying by orig_resume_offset may not be enough
            # if 2 graph breaks on different initial offsets resume on the same instruction
            # (although this is rare and not tested anywhere).
            if offset_key not in meta.block_target_offset_remap:
                block_target_offset_remap = meta.block_target_offset_remap[
                    offset_key
                    # pyrefly: ignore [implicit-any]
                ] = {}

                def remap_block_offsets(
                    instructions: list[Instruction], code_options: dict[str, Any]
                ) -> None:
                    # NOTE: each prefix block generates exactly one PUSH_EXC_INFO,
                    # so we can tell which block a prefix PUSH_EXC_INFO belongs to,
                    # by counting. Then we can use meta.prefix_block_target_offset_remap
                    # to determine where in the original code the PUSH_EXC_INFO offset
                    # replaced.
                    prefix_blocks: list[Instruction] = []
                    for inst in instructions:
                        # NOTE meta.prefix_block_target_offset_remap is based off of how we codegen'd
                        # context managers at the prefix/prologue of the resume function. It is the same for
                        # every graph break in the same resume function, so we do not need to recompute
                        # for each graph break (unlike for meta.block_target_offset_remap)
                        if len(prefix_blocks) == len(
                            meta.prefix_block_target_offset_remap
                        ):
                            break
                        if inst.opname == "PUSH_EXC_INFO":
                            prefix_blocks.append(inst)

                    # remap block target offsets for blocks generated in the resume prefix
                    for inst, o in zip(
                        prefix_blocks, meta.prefix_block_target_offset_remap
                    ):
                        block_target_offset_remap[cast(int, inst.offset)] = o

                    # current bytecode targets are after the prefix PUSH_EXC_INFO's
                    cur_start_offset = (
                        cast(int, prefix_blocks[-1].offset) if prefix_blocks else -1
                    )
                    # get the remaining block target offsets of the current bytecode
                    cur_inst_offsets = sorted(
                        n for n in setup_fn_target_offsets if n > cur_start_offset
                    )
                    targets = _filter_iter(
                        instructions, cur_inst_offsets, lambda inst, o: inst.offset == o
                    )
                    # The original code and resume code should have matching suffixes.
                    # Match the post-prefix block target offsets of the current resume code
                    # and the original code.
                    orig_targets = reversed(
                        _filter_iter(
                            zip(reversed(instructions), reversed(meta.instructions)),
                            reversed(targets),
                            lambda v1, v2: v1[0] is v2,
                        )
                    )
                    for orig, cur in zip(orig_targets, targets):
                        block_target_offset_remap[cur.offset] = orig[1].offset

                transform_code_object(code, remap_block_offsets)

            # if offset_key or offset is not in setup_fn_target_offsets, it is an error
            # that needs to be fixed
            setup_fn_target_offsets = tuple(
                meta.block_target_offset_remap[offset_key][n]
                for n in setup_fn_target_offsets
            )
        return ContinueExecutionCache.lookup(
            meta.code,
            lineno,
            orig_init_offset,
            orig_resume_offset,
            setup_fn_target_offsets,
            *args,
        )
<|endoftext|>import copy
import json
import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch._dynamo.package import (
    _BackendId,
    _DynamoCacheEntry,
    DynamoCache,
    PrecompileCacheEntry,
)


"""
Classes and implementations related to precompile
"""

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class BackendCacheArtifact(Generic[T]):
    """
    Represents a single serializable backend artifact from a dynamo backend.
    Each BackendCacheArtifact has a key associated with it along with some
    serializable content.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """

    key: str
    content: Any

    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...

    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the contents of the artifact.
        """
        self.content = edit_fn(self.content)


class EagerCacheArtifact(BackendCacheArtifact[Any]):
    def after_deserialization(self) -> Any:
        return self.content


class BypassDynamoCacheEntry(Exception):
    pass


class PrecompileContext:
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    PrecompileContext has two main portions: dynamo_cache_entries and backend_cache_artifacts.
    When saving, PrecompileContext.serialize() will serialize all dynamo cache entries along with any PrecompileCacheArtifacts that
    are needed to save those dynamo cache entries.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact

    """

    # Protected by the compile_lock
    # _backend_artifacts_by_key organizes results by the key of each artifact.
    # Each object here must be serializable
    _backend_artifacts_by_key: dict[_BackendId, BackendCacheArtifact[Any]] = {}

    # On call to `serialize()`, all cache artifacts in _dynamo_cache_entries are converted
    # into DynamoCacheArtifacts and added to _new_cache_artifacts for serialization
    _dynamo_cache_entries: dict[str, _DynamoCacheEntry] = {}

    @classmethod
    def clear(cls) -> None:
        cls._backend_artifacts_by_key.clear()
        cls._dynamo_cache_entries.clear()

    @classmethod
    def record_artifact(
        cls,
        artifact: BackendCacheArtifact[Any],
    ) -> None:
        """
        Records a backend artifact to be used with dynamo cache entries
        """
        # Temporarily disable all dispatch modes (including FakeTensorMode) during
        # deepcopy to avoid issues with cloning fake tensors (e.g., device mesh
        # with meta tensors that fail when cloning due to device mismatches)
        from torch.utils._mode_utils import no_dispatch

        with no_dispatch():
            cls._backend_artifacts_by_key[_BackendId(artifact.key)] = copy.deepcopy(
                artifact
            )

    @classmethod
    def record_dynamo_cache_entry(
        cls, cache_entry: _DynamoCacheEntry, key: str
    ) -> None:
        cls._dynamo_cache_entries[key] = cache_entry

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        if key not in cls._backend_artifacts_by_key:
            raise AssertionError(f"Key {key} not found in artifacts")
        artifact = cls._backend_artifacts_by_key[_BackendId(key)]
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> BackendCacheArtifact[Any] | None:
        """
        Return the backend cache artifact with the associated key
        """
        return cls._backend_artifacts_by_key.get(_BackendId(key), None)

    @staticmethod
    def dump_debug_info(
        dynamo_entries: dict[str, _DynamoCacheEntry],
        backend_artifacts: dict[_BackendId, BackendCacheArtifact[Any]],
    ) -> dict[str, Any]:
        """
        Return a JSON serializable debug dump of all entries in the precompile context
        Called in serialize before serialization, and in populate_caches after deserialization
        """
        # Print debug information
        debug_info: defaultdict[str, list[Any]] = defaultdict(list)
        for key, cache_entry in dynamo_entries.items():
            info = cache_entry.debug_info()
            info["key"] = key
            debug_info["dynamo"].append(info)

        for artifact in backend_artifacts.values():
            debug_info["backends"].append(artifact.key)

        return debug_info

    @classmethod
    def save_to_dynamo_cache(cls) -> dict[str, Any]:
        precompile_cache_entries, debug_info = cls.create_cache_entries()
        for key, entry in precompile_cache_entries.items():
            DynamoCache.write(entry, key)
        return debug_info

    @classmethod
    def create_cache_entries(
        cls,
    ) -> tuple[dict[str, PrecompileCacheEntry], dict[str, Any]]:
        """
        Grabs all the cache entries in the precompile context and
        stitches them together into full PrecompileCacheEntries.
        """
        dynamo_entries = cls._dynamo_cache_entries
        backend_artifacts = cls._backend_artifacts_by_key

        num_artifacts = len(dynamo_entries)

        debug_info = PrecompileContext.dump_debug_info(
            dynamo_entries, backend_artifacts
        )
        debug_str = json.dumps(
            {
                "num_entries": num_artifacts,
                "artifacts": debug_info,
            },
        )
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_cache_entries",
                "encoding": "json",
            },
            payload_fn=lambda: debug_str,
            expect_trace_id=False,
        )

        precompile_cache_entries = {}

        for key, cache_entry in dynamo_entries.items():
            try:
                result = PrecompileCacheEntry.from_cache_entry(
                    cache_entry, backend_artifacts
                )
                if result is not None:
                    precompile_cache_entries[key] = result
            except Exception as e:
                logger.warning("Failed to create cache entry %s", key, exc_info=True)

                error = e
                data = json.dumps(
                    {
                        "key": key,
                        "error": str(error),
                    }
                )
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "dynamo_cache_exception",
                        "encoding": "json",
                    },
                    payload_fn=lambda: data,
                )
                continue
        return precompile_cache_entries, debug_info
<|endoftext|>"""trace_wrapped(*args, fn) is equivalent to fn(*args), but with a twist:
if you make_fx trace through this call, we will not actually trace into fn; instead,
we will directly insert it as a call_function to fn in the graph.
(Unlike make_fx, Dynamo WILL inline into fn.)
You can think of this as a one off allow_in_graph equivalent for proxy tensor tracing.

Because proxy tensor tracing does not actually run the function, there are
requirements on the behavior of fn. We are still figuring it out, but here is the current state:

1) fn SHOULD only take a single argument, which must be a tensor
2) fn MUST return a new tensor with the same metadata as the original tensor
   (e.g., zeros_like(input) is a permissible implementation of fn).
   This is verified via an extra assert that is inserted into the traced graph.
3) fn MAY have side effects, but it MAY NOT perform metadata mutation on other tensors
   participating in proxy tensor tracing (it MAY mutate other tensors, it MAY mutate Python state)
These requirements stem from the requirement that we need to continue performing proxy tensor tracing,
which assumes accurate fake tensor metadata, without actually running fn.
In the future, we may allow for a "meta" function associated with fn to allow for more interesting input-output patterns.

Note that tensors / Python state are allowed to be mutated.
This relaxed constraint is not always sound, but it is sound for backward tracing with fake
tensors as it takes place in AOTAutograd, as the backward pass is guaranteed not to depend on concrete
tensor values (via fake tensor) or Python state (because the autograd engine doesn't depend on Python).

The intended use case for this function is to allow AOTAutograd to defer complex
backward hooks to compiled autograd. AOTAutograd performs a make_fx trace which preserves
the function call as is in the graph, and only when we Dynamo through the backward graph in
compiled autograd do we inline into the function.
"""

from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses import FakeTensorMode
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import _get_current_dispatch_mode
from torch.utils._pytree import tree_map_only


Tensor = torch.Tensor


__all__ = ["trace_wrapped"]


@torch.library.custom_op("flex_lib::zeros_and_scatter", mutates_args=())  # type: ignore[misc]
def zeros_and_scatter(
    shape: list[int],
    indices: list[Tensor],
    vals: Tensor,
) -> Tensor:
    """Custom Op so that we can register a custom lowering for the new_output + scatter in the backwards pass"""
    if not indices:
        if shape:
            raise RuntimeError(
                "zeros_and_scatter with no indices only supports scalar outputs"
            )
        return vals.sum()
    grad = torch.zeros(shape, device=vals.device, dtype=vals.dtype)
    return torch.ops.aten.index_put(grad, indices, vals, accumulate=True)


@zeros_and_scatter.register_fake  # type: ignore[misc]
def _(
    shape: list[int],
    indices: list[Tensor],
    vals: Tensor,
) -> Tensor:
    return vals.new_empty(shape)


@zeros_and_scatter.register_vmap  # type: ignore[misc]
def _(info, indims, shape, indices, value):  # type: ignore[no-untyped-def]
    """The batching rule is special in that it returns a tensor that is not batched"""
    indices_indims: list[int | None] = indims[1] if indims[1] is not None else []
    expanded_indices = []
    for idx, idx_indim in zip(indices, indices_indims):
        # The index is not being batched, we should unsqueeze and expand to val
        if idx_indim is None:
            expanded_indices.append(idx.expand(value.shape))
        else:
            # the index is being part of the vmap batch, it should be the same size as val
            if idx.shape != value.shape:
                raise AssertionError(
                    f"idx shape {idx.shape} must match value shape {value.shape}"
                )
            expanded_indices.append(idx)

    out = torch.ops.flex_lib.zeros_and_scatter(
        shape,
        expanded_indices,
        value,
    )
    return out, None


class ModIndex(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(x: Tensor, indices: list[Tensor]) -> Tensor:
        if not indices:
            if x.ndim != 0:
                raise RuntimeError(
                    "mod_index with no indices only supports scalar tensors"
                )
            return x
        return torch.ops.aten.index(x, indices)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        x, indices = inputs
        ctx.save_for_backward(*indices)
        ctx.input_shape = x.shape

    @staticmethod
    def backward(ctx, gradOut):  # type: ignore[no-untyped-def]
        indices = ctx.saved_tensors
        return (
            torch.ops.flex_lib.zeros_and_scatter(
                ctx.input_shape,
                indices,
                gradOut,
            ),
            None,
        )

    @classmethod
    @torch._export.wrappers.allow_in_pre_dispatch_graph
    def apply(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        return super().apply(*args, **kwargs)


mod_index = ModIndex.apply


class TransformGetItemToIndex(TorchFunctionMode):
    # This is needed since we want to support calling
    # A[q_idx], where q_idx is a scalar tensor in score_mod.
    # Today, when q_idx is a scalar tensor, we implicitly convert it to a python
    # scalar and create a view. We do not want that behavior in this case, so we
    # use this torchfunctionmode to override that behavior for score_mod
    # wherever we're running it.
    #
    # We also convert integer indices to 0-D tensors so that temp[0] produces
    # the same backward graph as temp[0 * q_idx] (zeros_and_scatter with atomic_add).
    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func is torch.Tensor.__getitem__:
            tensor_to_index = args[0]
            if not isinstance(tensor_to_index, torch.Tensor):
                raise AssertionError(
                    f"expected torch.Tensor, got {type(tensor_to_index)}"
                )
            index_args = pytree.tree_leaves(args[1])
            if all(isinstance(x, (torch.Tensor, int)) for x in index_args):
                converted_indices = [
                    torch.tensor(x, dtype=torch.int64, device=tensor_to_index.device)
                    if isinstance(x, int)
                    else x
                    for x in index_args
                ]
                return mod_index(tensor_to_index, converted_indices)
        return func(*args, **(kwargs or {}))


def trace_wrapped(*args: Any, **kwargs: Any) -> Any:
    with torch.no_grad():
        return _trace_wrapped_op(*args, **kwargs)


class TraceWrapped(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("trace_wrapped")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(*args, **kwargs)


# TODO(jansel): need to ensure this does not get DCEed
_trace_wrapped_op = TraceWrapped()


def _assert_meta(
    grad: torch.Tensor,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if grad.size() != size:
        raise AssertionError("size mismatch")
    if grad.stride() != stride:
        raise AssertionError("stride mismatch")
    if grad.dtype != dtype:
        raise AssertionError("dtype mismatch")
    return grad


@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(
    mode: ProxyTorchDispatchMode,
    *args: Any,
    bw_state: BackwardState | None = None,
    **kwargs: Any,
) -> Any:
    def self_invoke(*args: Any, **dyn_kwargs: Any) -> Any:
        with torch.no_grad():
            return _trace_wrapped_op(*args, **dyn_kwargs, **kwargs)

    def unwrap_proxies(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return mode.tracer.unwrap_proxy(x)  # type: ignore[union-attr]
        if isinstance(x, (list, tuple)):
            return type(x)(map(unwrap_proxies, x))
        if x is None:
            return None
        raise AssertionError(f"unhandled type: {type(x)}")

    proxy_kwargs = {}
    if bw_state is not None:
        if not isinstance(bw_state, BackwardState):
            raise AssertionError(f"expected BackwardState, got {type(bw_state)}")
        if bw_state.proxy is None:
            raise AssertionError("bw_state.proxy must not be None")
        proxy_kwargs["bw_state"] = bw_state.proxy
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        self_invoke,
        unwrap_proxies(args),
        proxy_kwargs,
        name="trace_wrapped",
    )

    if args[0] is None:
        grad = args[1]  # module backward hooks
    else:
        grad = args[0]  # other backward hooks
    grad = tree_map_only(torch.Tensor, torch.empty_like, grad)
    track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return grad


@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("This op should never be invoked here")


@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args: Any, fn: Any, **kwargs: Any) -> Any:
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    return fn(*args, **kwargs)


_trace_wrapped_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_trace_wrapped_op, deferred_error=True)
)


@_trace_wrapped_op.py_functionalize_impl
def _trace_wrapped_functionalized(ctx: Any, *args: Any, **kwargs: Any) -> Any:
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(_trace_wrapped_op(*unwrapped_args, **kwargs))


def autograd_function_backward_rewritten(original_backward: Any) -> Any:
    def new_backward(ctx: Any, *grads: Any) -> Any:
        # pyrefly: ignore [bad-assignment]
        grads = [g.contiguous() for g in grads]
        return original_backward(ctx, *grads)

    return new_backward
<|endoftext|>import dataclasses
import importlib
import inspect
import io
import logging
import os
import pickle
import tempfile
import types
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.convert_frame import GraphRuntimeEnv
from torch._dynamo.graph_utils import _graph_device_type
from torch._dynamo.package import SystemInfo

from . import convert_frame
from .aot_compile_types import (
    BundledAOTAutogradSerializableCallable,
    SerializableCallable,
)
from .hooks import Hooks


if TYPE_CHECKING:
    from .guards import GuardManagerWrapper
    from .package import SerializedCode, SourceInfo


log = logging.getLogger(__name__)


def bind_locals(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return bound_arguments.arguments


@dataclass
class CompileArtifacts:
    signature: inspect.Signature
    guard_manager: Optional["GuardManagerWrapper"]
    guards_state: bytes
    backend_id: str
    compiled_fn: SerializableCallable
    original_code: types.CodeType
    runtime_env: GraphRuntimeEnv
    source_info: "SourceInfo"
    device_type: str
    backend_name: str
    system_info: SystemInfo = dataclasses.field(default_factory=SystemInfo.current)

    def check_compatibility(self) -> None:
        current_system = SystemInfo.current()
        current_system.check_compatibility(self.system_info, self.device_type)


class AOTCompilePickler(pickle.Pickler):
    def __init__(self, external_data: dict[str, object], buf: io.BytesIO) -> None:
        super().__init__(buf)
        self.external_data = external_data
        self.id_map: dict[int, str] = {
            id(value): key for key, value in external_data.items()
        }
        self.errors = {}

    def persistent_id(self, obj: object) -> int | str | None:
        if id(obj) in self.id_map:
            return self.id_map[id(obj)]
        elif isinstance(obj, torch.nn.Module):
            self.errors[id(obj)] = obj
            return id(obj)
        else:
            return None

    @classmethod
    def _unpickle_cell(cls, val: object) -> object:
        def _() -> object:
            return val

        if _.__closure__ is None:
            raise AssertionError("closure must not be None")
        return _.__closure__[0]

    @classmethod
    # pyrefly: ignore [implicit-any]
    def _unpickle_bound_method(cls, func: Callable, base: object) -> types.MethodType:
        return types.MethodType(func, base)

    @classmethod
    def _unpickle_module(cls, name: str) -> types.ModuleType:
        return importlib.import_module(name)

    @classmethod
    def _unpickle_code(cls, serialized_code: "SerializedCode") -> types.CodeType:
        from torch._dynamo.package import SerializedCode

        return SerializedCode.to_code_object(serialized_code)

    @classmethod
    def _unpickle_nested_function(
        cls,
        code: types.CodeType,
        module: str,
        qualname: str,
        argdefs: tuple[object, ...] | None,
        closure: tuple[types.CellType, ...] | None,
    ) -> types.FunctionType:
        f_globals = importlib.import_module(module).__dict__
        return types.FunctionType(code, f_globals, qualname, argdefs, closure)

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, type((lambda x: lambda: x)(0).__closure__[0])):  # type: ignore[index] # noqa: PLC3002
            return type(self)._unpickle_cell, (obj.cell_contents,)
        elif inspect.iscode(obj):
            from torch._dynamo.package import SerializedCode

            return type(self)._unpickle_code, (SerializedCode.from_code_object(obj),)

        elif inspect.ismodule(obj):
            return type(self)._unpickle_module, (obj.__name__,)
        elif inspect.ismethod(obj):
            """
            By default, pickle will call getattr() directly on the self object
            for pickling bounded methods, this is not what we want, instead we
            always want to serialize the original function and the self object
            in their original form.
            """
            func = obj.__func__
            method_self = obj.__self__
            inner_func = getattr(method_self, func.__name__)
            if inspect.ismethod(inner_func):
                inner_func = inner_func.__func__
            if func is not inner_func:
                return type(self)._unpickle_bound_method, (func, method_self)
        elif inspect.isfunction(obj):
            if "<locals>" in obj.__qualname__:
                return type(self)._unpickle_nested_function, (
                    obj.__code__,
                    obj.__module__,
                    obj.__qualname__,
                    obj.__defaults__,
                    obj.__closure__,
                )

        return NotImplemented


class AOTCompileUnpickler(pickle.Unpickler):
    def __init__(self, external_data: dict[str, object], file: io.BytesIO) -> object:
        super().__init__(file)
        self.external_data = external_data

    def persistent_load(self, key: str) -> object:
        if key not in self.external_data:
            raise RuntimeError(
                f"Missing required external reference to data: {key}. "
                "Please load AOT compiled function with "
                "`external_data=<external data dictionary>`"
                f"{self.external_data}"
            )
        return self.external_data[key]


@dataclass
class AOTCompileSaveResult:
    serialized_data: bytes


def atomic_write_binary(file_path: str, data: bytes):
    dir_name = os.path.dirname(file_path) or "."

    with tempfile.NamedTemporaryFile(
        dir=dir_name, delete=False, mode="wb"
    ) as temp_file:
        temp_path = temp_file.name
        temp_file.write(data)
        temp_file.flush()
        os.fsync(temp_file.fileno())

    os.replace(temp_path, file_path)


@dataclass
class AOTCompiledFunction:
    _artifacts: CompileArtifacts
    _guard_check_enabled: bool = True
    _extra_globals: dict[str, object] | None = None

    def prepare_f_locals(self, *args: object, **kwargs: object) -> dict[str, object]:
        f_locals: dict[str, object] = {}
        env = self._artifacts.runtime_env
        if env.closure:
            if not env.bytecode.co_freevars or len(env.closure) != len(
                env.bytecode.co_freevars
            ):
                raise AssertionError("closure length must match co_freevars length")
            f_locals = {
                name: cell.cell_contents
                for name, cell in zip(env.bytecode.co_freevars, env.closure)
            }
        f_locals.update(bind_locals(self._artifacts.signature, *args, **kwargs))
        return f_locals

    def guard_check(self, *args: Any, **kwargs: Any) -> bool:
        f_locals = self.prepare_f_locals(*args, **kwargs)
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        return self._artifacts.guard_manager.check(f_locals)

    def __post_init__(self) -> None:
        from .package import load_guard_manager, load_guards_state

        self._artifacts.check_compatibility()

        self.fn = self._artifacts.runtime_env.forward_callable(
            self._artifacts.backend_id,
            self._artifacts.compiled_fn,
            extra_globals=self._extra_globals,
        )

        if self._artifacts.guard_manager is None:
            guards_state = load_guards_state(self._artifacts.guards_state)
            self._artifacts.guard_manager = load_guard_manager(
                guards_state,
                self._artifacts.original_code,
                self.fn.__globals__,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._artifacts.guard_manager is None:
            raise AssertionError("guard_manager must not be None")
        if self._guard_check_enabled and not self.guard_check(*args, **kwargs):
            f_locals = self.prepare_f_locals(*args, **kwargs)
            reason = str(self._artifacts.guard_manager.check_verbose(f_locals))
            raise RuntimeError(f"GuardManager check failed, reason: {reason}")
        return self.fn(*args, **kwargs)

    def source_info(self) -> "SourceInfo":
        return self._artifacts.source_info

    def save_compiled_function(
        self, path: str, external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        result = type(self).serialize(self, external_data)
        atomic_write_binary(path, result.serialized_data)
        return result

    @classmethod
    def serialize(
        cls, fn: "AOTCompiledFunction", external_data: dict[str, Any] | None = None
    ) -> AOTCompileSaveResult:
        from torch._dynamo.package import SerializedCode

        state = fn._artifacts.__dict__.copy()
        state["guard_manager"] = None
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.from_code_object(state["runtime_env"].bytecode),
        )
        compiled_fn = state["compiled_fn"]
        state["compiled_fn"] = (
            type(compiled_fn).deserialize_compile_artifacts,
            type(compiled_fn).serialize_compile_artifacts(compiled_fn),
        )
        state["original_code"] = SerializedCode.from_code_object(state["original_code"])
        buf = io.BytesIO()
        pickler = AOTCompilePickler(external_data or {}, buf)
        pickler.dump(state)
        if pickler.errors:
            raise RuntimeError(
                f"Failed to serialize the following objects: {list(pickler.errors.values())}\n"
                "Please mark these as external data by using `external_data={'key': ...}`"
            )
        return AOTCompileSaveResult(serialized_data=buf.getvalue())

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        f_globals: dict[str, object] | None = None,
        external_closure_data: dict[str, Any] | None = None,
    ) -> "AOTCompiledFunction":
        from torch._dynamo.package import SerializedCode

        f = io.BytesIO(data)
        f.seek(0)
        unpickler = AOTCompileUnpickler(external_closure_data or {}, f)
        state = unpickler.load()
        f.close()
        state["runtime_env"] = dataclasses.replace(
            state["runtime_env"],
            bytecode=SerializedCode.to_code_object(state["runtime_env"].bytecode),
        )
        deserializer, compiled_fn_state = state["compiled_fn"]
        with torch._inductor.config.patch(enable_autograd_for_aot=True):
            state["compiled_fn"] = deserializer(compiled_fn_state)
        state["original_code"] = SerializedCode.to_code_object(state["original_code"])

        artifacts = CompileArtifacts(**state)
        return cls(artifacts, _extra_globals=f_globals)

    def disable_guard_check(self) -> None:
        self._guard_check_enabled = False


def aot_compile_fullgraph(
    model: Any,
    example_inputs: tuple[tuple[Any, ...], dict[str, Any]],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
    dynamic: bool | None = None,
) -> AOTCompiledFunction:
    from torch._dynamo.guards import CheckFunctionManager
    from torch._dynamo.package import SourceInfo
    from torch._dynamo.utils import dynamo_timed, get_metrics_context
    from torch._dynamo.variables.torch_function import (
        torch_function_mode_stack_state_mgr,
    )
    from torch._guards import TracingContext

    args, kwargs = example_inputs

    dynamic_ctx = nullcontext()
    if dynamic is not None:
        from torch._dynamo.eval_frame import set_enable_dynamic

        dynamic_ctx = set_enable_dynamic(dynamic)

    with (
        get_metrics_context(),
        dynamo_timed("fullgraph_capture"),
        torch._functorch.config.patch(strict_autograd_cache=True),
        dynamic_ctx,
        torch_function_mode_stack_state_mgr,
    ):
        capture_output = convert_frame.fullgraph_capture(model, args, kwargs)
        graph_capture_output = capture_output.graph_capture_output
        if graph_capture_output.output_graph is None:
            raise AssertionError("output_graph must not be None")

        if not hooks.guard_filter_fn:
            from torch._dynamo.types import GuardFilterEntry

            def new_guard_filter_fn(
                guard_entries: Sequence[GuardFilterEntry],
            ) -> Sequence[bool]:
                return [
                    (
                        not (
                            g.is_global
                            or g.guard_type
                            in CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
                        )
                    )
                    for g in guard_entries
                ]

            hooks.guard_filter_fn = new_guard_filter_fn

        fn, _ = convert_frame.get_traced_fn(model)

        backend_input = capture_output.backend_input
        if backend_input is None:
            raise AssertionError("backend_input must not be None")
        backend_input.graph_module._backend_id = backend_input.backend_id  # type: ignore[assignment]
        device_type = _graph_device_type(backend_input.graph_module.graph)
        if (
            backend_input.fake_mode.shape_env
            is not graph_capture_output.output_graph.shape_env
        ):
            raise AssertionError(
                "fake_mode.shape_env must be the same as output_graph.shape_env"
            )
        tracing_context = TracingContext(backend_input.fake_mode)
        tracing_context.tensor_to_context = backend_input.tensor_to_context
        with (
            torch._guards.tracing(tracing_context),
            torch._functorch.config.patch(
                {
                    "strict_autograd_cache": True,
                    "bypass_autograd_cache_key": True,
                    "bundled_autograd_cache": True,
                    "force_non_lazy_backward_lowering": True,
                    "force_autograd_cache": True,
                }
            ),
        ):
            compiled_fn = backend(
                backend_input.graph_module, backend_input.example_inputs
            )
            # If Inductor backend or AOTAutograd-based backend is used,
            # wrap the compiled_fn for serialization.
            # TODO: this should be replaced once we make the backend return the SerializableCallable directly.
            if (
                isinstance(backend, torch._TorchCompileInductorWrapper)
                or (
                    hasattr(backend, "compiler_fn")
                    and isinstance(
                        backend.compiler_fn, torch._dynamo.backends.common.AotAutograd
                    )
                )
                or (
                    hasattr(compiled_fn, "serialize")
                    and compiled_fn.serialize is not None
                )
            ):
                compiled_fn = BundledAOTAutogradSerializableCallable(compiled_fn)

        if not isinstance(compiled_fn, SerializableCallable):
            if hasattr(backend, "compiler_fn"):
                compiler_fn = backend.compiler_fn
            else:
                compiler_fn = backend
            raise RuntimeError(
                f"Compiled function type {type(compiled_fn)} (produced "
                + f"from backend {compiler_fn}) does not implement SerializableCallable."
            )

        # Temporarily restore the mode stack so guard expressions that
        # reference modes can evaluate, matching the compile_inner path.
        build_guards_ctx = ExitStack()
        if torch_function_mode_stack_state_mgr.stack:
            build_guards_ctx.enter_context(
                torch_function_mode_stack_state_mgr.temp_restore_stack()
            )
        with build_guards_ctx:
            check_fn = graph_capture_output.build_guards(
                fn.__code__, hooks=hooks, save=True, strict_error=True
            )

        if check_fn.guards_state is None:
            raise AssertionError("guards_state must not be None")

        source_info = SourceInfo(inlined_sources=set())
        for traced_code in graph_capture_output.traced_code:
            source_info.add_code(traced_code)

        artifacts = CompileArtifacts(
            signature=convert_frame._get_signature(fn),
            guard_manager=check_fn.guard_manager,
            guards_state=check_fn.guards_state,
            backend_id=backend_input.backend_id,
            compiled_fn=compiled_fn,
            original_code=fn.__code__,
            runtime_env=graph_capture_output.get_runtime_env(),
            source_info=source_info,
            device_type=device_type,
            backend_name=getattr(backend, "compiler_name", "unknown"),
        )
        aot_compiled_fn = AOTCompiledFunction(
            _artifacts=artifacts, _extra_globals=fn.__globals__
        )

    return aot_compiled_fn


@dataclass
class ModelInput:
    """
    WIP type: represents a single model input
    Which consists of a tuple of arguments and a set of contexts in which to run the model.

    For each ModelInput, we'll compile one full graph of the model, and then use the guards generated
    to dispatch between the compiled graphs.


    """

    args: tuple[Any]
    kwargs: dict[str, Any]
    contexts: list[AbstractContextManager[Any]]


@dataclass
class AOTCompiledModel:
    # Represents a single forward function of a model along with dispatch
    # compiled_results is serializable. We require the model to deserialize again.
    model: torch.nn.Module
    compiled_results: list[AOTCompiledFunction]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for result in self.compiled_results:
            if result.guard_check(self.model, *args, **kwargs):
                return result(self.model, *args, **kwargs)
        # All guards failed, just run one of them and throw the guard check error.
        return self.compiled_results[0](self.model, *args, **kwargs)

    def serialize(self) -> bytes:
        data: list[bytes] = []
        for result in self.compiled_results:
            data.append(AOTCompiledFunction.serialize(result).serialized_data)
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, model: torch.nn.Module, data: bytes) -> "AOTCompiledModel":
        from torch._dynamo.utils import get_metrics_context
        from torch._guards import compile_context, CompileContext

        results: list[bytes] = pickle.loads(data)
        compiled_results = []
        for result in results:
            with (
                compile_context(CompileContext(convert_frame.get_compile_id({}))),
                get_metrics_context(),
            ):
                compiled_results.append(AOTCompiledFunction.deserialize(result))
        return cls(model, compiled_results)


def aot_compile_module(
    model: torch.nn.Module,
    inputs: list[ModelInput],
    hooks: Hooks,
    backend: Callable[[torch.fx.GraphModule, list[torch.Tensor]], SerializableCallable],
) -> AOTCompiledModel:
    """
    Compiles a single nn.Module with any number of inputs, and returns a compiled forward function.
    """

    def compile_single_graph(model_input: ModelInput) -> AOTCompiledFunction:
        example_inputs = (model_input.args, model_input.kwargs)
        orig_forward = model.forward
        with ExitStack() as stack:
            for ctx in model_input.contexts:
                stack.enter_context(ctx)
            return aot_compile_fullgraph(
                orig_forward,
                example_inputs,
                hooks=hooks,
                backend=backend,
            )

    # pyrefly: ignore [implicit-any]
    compiled_results = []
    for model_input in inputs:
        log.info("Compiling input %s..", model_input)
        compiled_results.append(compile_single_graph(model_input))

    if len(compiled_results) == 0:
        raise AssertionError("Expected at least one compiled result")

    return AOTCompiledModel(model, compiled_results)
<|endoftext|>"""Common utilities for testing Dynamo's minifier functionality.

This module provides the base infrastructure for running minification tests in Dynamo.
It includes:
- MinifierTestResult: A dataclass for storing and processing minifier test results
- MinifierTestBase: A base test class with utilities for:
  - Running tests in isolated environments
  - Managing temporary directories and configurations
  - Executing minifier launcher scripts
  - Running and validating reproduction scripts
  - Supporting both compile-time and runtime error testing

The minifier helps reduce failing Dynamo compilations to minimal reproductions.
"""

import dataclasses
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.trace_rules import _as_posix_path
from torch.utils._traceback import report_compile_source_on_error


def _decode_subprocess_output(output: bytes) -> str:
    return output.decode("utf-8", errors="replace")


@dataclasses.dataclass
class MinifierTestResult:
    minifier_code: str
    repro_code: str

    def _get_module(self, t: str) -> str:
        match = re.search(r"class Repro\(torch\.nn\.Module\):\s+([ ].*\n| *\n)+", t)
        if match is None:
            raise AssertionError("failed to find module")
        r = match.group(0)
        r = re.sub(r"\s+$", "\n", r, flags=re.MULTILINE)
        r = re.sub(r"\n{3,}", "\n\n", r)
        return r.strip()

    def get_exported_program_path(self) -> str | None:
        # Extract the exported program file path from AOTI minifier's repro.py
        # Regular expression pattern to match the file path
        pattern = r'torch\.export\.load\(\s*["\'](.*?)["\']\s*\)'
        # Search for the pattern in the text
        match = re.search(pattern, self.repro_code)
        # Extract and print the file path if a match is found
        if match:
            file_path = match.group(1)
            return file_path
        return None

    def minifier_module(self) -> str:
        return self._get_module(self.minifier_code)

    def repro_module(self) -> str:
        return self._get_module(self.repro_code)


class MinifierTestBase(torch._dynamo.test_case.TestCase):
    DEBUG_DIR = tempfile.mkdtemp()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not os.path.exists(cls.DEBUG_DIR):
            cls.DEBUG_DIR = tempfile.mkdtemp()
        cls._exit_stack.enter_context(  # type: ignore[attr-defined]
            torch._dynamo.config.patch(debug_dir_root=cls.DEBUG_DIR)
        )
        # These configurations make new process startup slower.  Disable them
        # for the minification tests to speed them up.
        cls._exit_stack.enter_context(  # type: ignore[attr-defined]
            torch._inductor.config.patch(
                {
                    # https://github.com/pytorch/pytorch/issues/100376
                    "pattern_matcher": False,
                    # multiprocess compilation takes a long time to warmup
                    "compile_threads": 1,
                    # https://github.com/pytorch/pytorch/issues/100378
                    "cpp.vec_isa_ok": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if os.getenv("PYTORCH_KEEP_TMPDIR", "0") != "1":
            shutil.rmtree(cls.DEBUG_DIR)
        else:
            print(f"test_minifier_common tmpdir kept at: {cls.DEBUG_DIR}")
        cls._exit_stack.close()  # type: ignore[attr-defined]

    def _gen_codegen_fn_patch_code(self, device: str, bug_type: str) -> str:
        if bug_type not in ("compile_error", "runtime_error", "accuracy"):
            raise AssertionError(
                f"bug_type must be one of compile_error, runtime_error, accuracy, got {bug_type!r}"
            )
        return f"""\
{torch._dynamo.config.codegen_config()}
{torch._inductor.config.codegen_config()}
torch._inductor.config.{"cpp" if device == "cpu" else "triton"}.inject_relu_bug_TESTING_ONLY = {bug_type!r}
"""

    def _maybe_subprocess_run(
        self, args: Sequence[Any], *, isolate: bool, cwd: str | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        from torch._inductor.cpp_builder import normalize_path_separator

        if not isolate:
            if len(args) < 2:
                raise AssertionError(f"expected at least 2 args, got {args}")
            if args[0] != "python3":
                raise AssertionError(f"expected args[0] to be 'python3', got {args}")
            if args[1] == "-c":
                if len(args) != 3:
                    raise AssertionError(
                        f"expected exactly 3 args for -c mode, got {args}"
                    )
                code = args[2]
                args = ["-c"]
            else:
                if len(args) < 2:
                    raise AssertionError(f"expected at least 2 args, got {args}")
                with open(args[1]) as f:
                    # Need normalize path of the code.
                    code = normalize_path_separator(f.read())
                args = args[1:]

            # WARNING: This is not a perfect simulation of running
            # the program out of tree.  We only interpose on things we KNOW we
            # need to handle for tests.  If you need more stuff, you will
            # need to augment this appropriately.

            # NB: Can't use save_config because that will omit some fields,
            # but we must save and reset ALL fields
            dynamo_config = torch._dynamo.config.get_config_copy()
            inductor_config = torch._inductor.config.get_config_copy()
            try:
                stderr = io.StringIO()
                log_handler = logging.StreamHandler(stderr)
                log = logging.getLogger("torch._dynamo")
                log.addHandler(log_handler)
                try:
                    prev_cwd = _as_posix_path(os.getcwd())
                    if cwd is not None:
                        cwd = _as_posix_path(cwd)
                        os.chdir(cwd)
                    with patch("sys.argv", args), report_compile_source_on_error():
                        exec(code, {"__name__": "__main__", "__compile_source__": code})
                    rc = 0
                except Exception:
                    rc = 1
                    traceback.print_exc(file=stderr)
                finally:
                    log.removeHandler(log_handler)
                    if cwd is not None:
                        os.chdir(prev_cwd)  # type: ignore[possibly-undefined]
                    # Make sure we don't leave buggy compiled frames lying
                    # around
                    torch._dynamo.reset()
            finally:
                torch._dynamo.config.load_config(dynamo_config)
                torch._inductor.config.load_config(inductor_config)

            # TODO: return a more appropriate data structure here
            return subprocess.CompletedProcess(
                args,
                rc,
                b"",
                stderr.getvalue().encode("utf-8"),
            )
        else:
            if cwd is not None:
                cwd = _as_posix_path(cwd)
            return subprocess.run(args, capture_output=True, cwd=cwd, check=False)

    # Run `code` in a separate python process.
    # Returns the completed process state and the directory containing the
    # minifier launcher script, if `code` outputted it.
    def _run_test_code(
        self, code: str, *, isolate: bool
    ) -> tuple[subprocess.CompletedProcess[bytes], str | Any]:
        proc = self._maybe_subprocess_run(
            ["python3", "-c", code], isolate=isolate, cwd=self.DEBUG_DIR
        )

        print("test stdout:", _decode_subprocess_output(proc.stdout))
        stderr = _decode_subprocess_output(proc.stderr)
        print("test stderr:", stderr)
        repro_dir_match = re.search(r"(\S+)minifier_launcher.py", stderr)
        if repro_dir_match is not None:
            return proc, repro_dir_match.group(1)
        return proc, None

    # Runs the minifier launcher script in `repro_dir`
    def _run_minifier_launcher(
        self,
        repro_dir: str,
        isolate: bool,
        *,
        minifier_args: Sequence[Any] = (),
        repro_after: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[bytes], str]:
        self.assertIsNotNone(repro_dir)
        launch_file = _as_posix_path(os.path.join(repro_dir, "minifier_launcher.py"))
        with open(launch_file) as f:
            launch_code = f.read()

        self.assertTrue(os.path.exists(launch_file))

        args = ["python3", launch_file, "minify", *minifier_args]
        if not isolate and repro_after != "aot_inductor":
            # AOTI minifier doesn't have --no-isolate flag.
            # Everything in AOTI minifier is in no-isolate mode.
            args.append("--no-isolate")
        launch_proc = self._maybe_subprocess_run(args, isolate=isolate, cwd=repro_dir)
        print("minifier stdout:", _decode_subprocess_output(launch_proc.stdout))
        stderr = _decode_subprocess_output(launch_proc.stderr)
        print("minifier stderr:", stderr)

        self.assertNotIn("Input graph did not fail the tester", stderr)

        return launch_proc, launch_code

    # Runs the repro script in `repro_dir`
    def _run_repro(
        self, repro_dir: str, *, isolate: bool = True
    ) -> tuple[subprocess.CompletedProcess[bytes], str]:
        self.assertIsNotNone(repro_dir)
        repro_file = _as_posix_path(os.path.join(repro_dir, "repro.py"))
        with open(repro_file) as f:
            repro_code = f.read()

        self.assertTrue(os.path.exists(repro_file))

        repro_proc = self._maybe_subprocess_run(
            ["python3", repro_file], isolate=isolate, cwd=repro_dir
        )
        print("repro stdout:", _decode_subprocess_output(repro_proc.stdout))
        print("repro stderr:", _decode_subprocess_output(repro_proc.stderr))
        return repro_proc, repro_code

    # Template for testing code.
    # `run_code` is the code to run for the test case.
    # `patch_code` is the code to be patched in every generated file; usually
    # just use this to turn on bugs via the config
    def _gen_test_code(self, run_code: str, repro_after: str, repro_level: int) -> str:
        repro_after_line = ""
        if repro_after == "aot_inductor":
            repro_after_line = (
                "torch._inductor.config.aot_inductor.dump_aoti_minifier = True"
            )
        elif repro_after:
            repro_after_line = f"""\
torch._dynamo.config.repro_after = "{repro_after}"
        """
        return f"""\
import torch
import torch._dynamo
import torch._inductor
{_as_posix_path(torch._dynamo.config.codegen_config())}
{_as_posix_path(torch._inductor.config.codegen_config())}
{repro_after_line}
torch._dynamo.config.repro_level = {repro_level}
torch._inductor.config.aot_inductor.repro_level = {repro_level}
torch._dynamo.config.debug_dir_root = "{_as_posix_path(self.DEBUG_DIR)}"
{run_code}
"""

    # Runs a full minifier test.
    # Minifier tests generally consist of 3 stages:
    # 1. Run the problematic code
    # 2. Run the generated minifier launcher script
    # 3. Run the generated repro script
    #
    # If possible, you should run the test with isolate=False; use
    # isolate=True only if the bug you're testing would otherwise
    # crash the process
    def _run_full_test(
        self,
        run_code: str,
        repro_after: str,
        expected_error: str | None,
        *,
        isolate: bool,
        minifier_args: Sequence[Any] = (),
    ) -> MinifierTestResult | None:
        if isolate:
            repro_level = 3
        elif expected_error is None or expected_error == "AccuracyError":
            repro_level = 4
        else:
            repro_level = 2
        test_code = self._gen_test_code(run_code, repro_after, repro_level)
        print("running test", file=sys.stderr)
        test_proc, repro_dir = self._run_test_code(test_code, isolate=isolate)
        if expected_error is None:
            # Just check that there was no error
            self.assertEqual(test_proc.returncode, 0)

            self.assertIsNone(repro_dir)
            return None
        # NB: Intentionally do not test return code; we only care about
        # actually generating the repro, we don't have to crash

        self.assertIn(expected_error, _decode_subprocess_output(test_proc.stderr))

        self.assertIsNotNone(repro_dir)
        print("running minifier", file=sys.stderr)
        _minifier_proc, minifier_code = self._run_minifier_launcher(
            repro_dir,
            isolate=isolate,
            minifier_args=minifier_args,
            repro_after=repro_after,
        )
        print("running repro", file=sys.stderr)
        repro_proc, repro_code = self._run_repro(repro_dir, isolate=isolate)

        self.assertIn(expected_error, _decode_subprocess_output(repro_proc.stderr))
        self.assertNotEqual(repro_proc.returncode, 0)
        return MinifierTestResult(minifier_code=minifier_code, repro_code=repro_code)
<|endoftext|>"""
Hashability utilities for PyTorch Dynamo variable tracking.

This module provides the HashableTracker wrapper class and associated utilities
for making VariableTracker instances usable as dictionary keys and set elements
during symbolic execution. Used by both ConstDictVariable and SetVariable.
"""

from typing import TYPE_CHECKING

import torch

from .. import variables
from ..exc import raise_observed_exception
from ..utils import guard_if_dyn, specialize_symnode
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


def raise_unhashable(
    arg: VariableTracker, tx: "InstructionTranslatorBase | None" = None
) -> None:
    if tx is None:
        from torch._dynamo.symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()

    try:
        arg_type = arg.python_type()
    except Exception:
        arg_type = None

    # Safety check: if we know the real Python type and it IS hashable,
    # our is_hashable() disagrees with CPython. Graph-break rather than
    # raising a wrong TypeError.
    if arg_type is not None and arg_type.__hash__ is not None:
        from .. import graph_break_hints
        from ..exc import unimplemented

        unimplemented(
            gb_type="Hashability mismatch",
            context=f"raise_unhashable {arg}",
            explanation=f"Dynamo thinks {arg_type.__name__} is unhashable, "
            f"but its __hash__ is not None. This likely indicates a missing "
            f"or incorrect is_hashable() override.",
            hints=[*graph_break_hints.DYNAMO_BUG],
        )

    type_name = arg_type.__name__ if arg_type is not None else type(arg).__name__
    raise_observed_exception(
        TypeError,
        tx,
        args=[
            f"unhashable type: '{type_name}'",
        ],
    )


def is_hashable(x: VariableTracker) -> bool:
    # LazyVT optimization: check hashability without realizing the VT to avoid
    # accidentally inserting guards.
    if (
        isinstance(x, variables.LazyVariableTracker)
        and not x.is_realized()
        and x.is_hashable_lazy()
    ):
        return True

    return x.is_hashable()


class RawHash:
    """Wraps a pre-computed hash value to bypass int.__hash__'s modular reduction.

    When building a tuple/frozenset of per-item hashes, using bare ints would
    apply long_hash (mod sys.hash_info.modulus), corrupting the values.
    Wrapping in RawHash makes tuplehash/frozenset_hash see the original hash.
    """

    __slots__ = ("h",)

    def __init__(self, h: int) -> None:
        self.h = h

    def __hash__(self) -> int:
        return self.h

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RawHash) and self.h == other.h


class HashableTracker:
    """
    Class that wraps a VariableTracker and makes it hashable.
    Note that it's fine to put VTs into dictionaries and sets, but doing so
    does not take into account aliasing.
    """

    _MISSING = object()

    def __init__(self, vt: VariableTracker) -> None:
        # We specialize SymNodes
        vt = specialize_symnode(vt)

        # Fast path for unrealized LazyVariableTrackers: check and hash without
        # realizing, to avoid inserting guards.  If the fast-path check fails,
        # fall through to realize the VT and try the full is_hashable check.
        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and vt.is_hashable_lazy()
        ):
            self._hash = hash(vt.original_value())
            self._hash_is_identity = False
            self.vt = vt
            return

        # Compute hash via the tp_hash slot (generic_hash_impl).
        # For unhashable types, hash_impl raises ObservedTypeError.
        from torch._dynamo.symbolic_convert import InstructionTranslator

        from .object_protocol import generic_hash_impl

        tx = InstructionTranslator.current_tx()
        # is_fake marks an identity-based hash (e.g. id(fake_tensor)); such VTs
        # have no python-constant value, so their key equality is identity.
        self._hash, self._hash_is_identity = generic_hash_impl(tx, vt)
        self.vt = vt

    @classmethod
    def _maybe_constant_torch_size(cls, vt: VariableTracker) -> object:
        from .lists import SizeVariable
        from .tensor import TensorVariable

        if (
            isinstance(vt, variables.LazyVariableTracker)
            and not vt.is_realized()
            and isinstance(vt.original_value(), torch.Size)
        ):
            return vt.original_value()

        if not isinstance(vt, SizeVariable):
            return cls._MISSING

        items = []
        for item in vt.items:
            if item.is_python_constant():
                items.append(item.as_python_constant())
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
                    items.append(constant.item())
                    continue

            return cls._MISSING

        return torch.Size(items)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        """
        Checks equality between two HashableTracker instances.

        Mirrors CPython's PyObject_RichCompareBool: routes the comparison
        through generic_richcompare_bool so any user-defined __eq__ runs.

        Args:
            other: Another HashableTracker instance to compare with

        Returns:
            True if the underlying variable trackers are Python-equal, False otherwise
        """
        if not isinstance(other, HashableTracker):
            return False
        if self.vt is other.vt:
            return True

        self_constant = self._maybe_constant_torch_size(self.vt)
        other_constant = self._maybe_constant_torch_size(other.vt)
        if self_constant is not self._MISSING and other_constant is not self._MISSING:
            return self_constant == other_constant

        # Tensor keys hash by identity (Tensor.__hash__ is id(self)), so CPython
        # only ever compares identical tensors during a dict/set lookup; a
        # tensor's elementwise __eq__ is never consulted for membership. Mirror
        # that with an identity comparison instead of running __eq__, which would
        # otherwise emit a stray elementwise-eq node into the FX graph and
        # corrupt the surrounding trace.
        from .tensor import TensorVariable

        if isinstance(self.vt, TensorVariable) or isinstance(other.vt, TensorVariable):
            return (
                self._hash_is_identity
                and other._hash_is_identity
                and self._hash == other._hash
            )

        # All other keys: mirror PyObject_RichCompareBool and run the
        # comparison through generic_richcompare_bool so any user-defined __eq__
        # runs and its result (or any exception it raises) is observed by the
        # traced program.
        from ..symbolic_convert import InstructionTranslator
        from .object_protocol import generic_richcompare_bool

        tx = InstructionTranslator.current_tx()

        result = generic_richcompare_bool(tx, self.vt, other.vt, op="__eq__")
        if result.is_python_constant():
            return bool(result.as_python_constant())

        if result.is_symnode_like():
            return bool(guard_if_dyn(result))

        # Comparison did not resolve to a constant (e.g. keys whose __eq__ could
        # not be determined). Fall back to identity-based hash equality, which
        # mirrors CPython treating such keys as equal only when identical.
        return (
            self._hash_is_identity
            and other._hash_is_identity
            and self._hash == other._hash
        )
<|endoftext|>"""
This module contains the InvokeSubgraphHigherOrderVariable class and its
supporting helpers for subgraph reuse (auto-cache) in Dynamo's invoke_subgraph
higher-order operator.
"""

import collections
import enum
import logging
import traceback
import types
from dataclasses import dataclass
from typing import Any, cast, NamedTuple, TYPE_CHECKING, TypeGuard

import torch
import torch._higher_order_ops
from torch._dynamo import graph_break_hints
from torch._dynamo.exc import unimplemented
from torch._dynamo.guards import (
    extract_tensor_metadata,
    GUARD_VALUE_DISPATCH,
    GuardCheckSpec,
    SKIP_GUARD,
    UnsupportedGuardCheckSpec,
)
from torch._dynamo.source import SyntheticLocalSource
from torch._dynamo.utils import _make_inlined, unpack_iterable
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.higher_order_ops import WrapHigherOrderVariable
from torch._dynamo.variables.lists import ListVariable, TupleVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
from torch._dynamo.variables.tensor import SymNodeVariable, TensorVariable
from torch._dynamo.variables.user_defined import UserDefinedObjectVariable
from torch._guards import (
    Guard,
    InvokeSubgraphReuseCondition,
    InvokeSubgraphReuseEntry,
    Source,
)
from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions
from torch.fx.graph_module import GraphModule
from torch.fx.proxy import Proxy
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._dynamo.variables.higher_order_ops import SubgraphTracingInfo

log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")

# Note: [invoke_subgraph subgraph reuse]
#
# When mark_compile_region wraps a function called N times (e.g. 80 identical
# transformer layers), Dynamo traces the subgraph once and stamps out cached
# copies for subsequent calls. It does safety checks to ensure that a subgraph
# is reusable, if not (e.g. side-effect), it will fallback to tracing the
# next invocation.
#
# HIGH-LEVEL FLOW
# ===============
#   User code: model.layers[0](x), model.layers[1](x), ..., model.layers[79](x)
#                     |                     |                        |
#                     v                     v                        v
#              +--------------+     +--------------+        +--------------+
#              |  First Call  |     |  Second Call  |  ...   |  80th Call   |
#              +------+-------+     +------+-------+        +------+-------+
#                     |                    |                        |
#                     v                    v                        v
#              +--------------+     +--------------+        +--------------+
#              | Full subgraph|     | Cache lookup  |        | Cache lookup  |
#              |   trace      |     | (is_reusable) |        | (is_reusable) |
#              +------+-------+     +------+-------+        +------+-------+
#                     |                    |                        |
#                     v                    v                        v
#              +--------------+     +--------------+        +--------------+
#              | save_reuse_  |     | stamp_out_   |        | stamp_out_   |
#              | entry()      |     | subgraph()   |        | subgraph()   |
#              +--------------+     +--------------+        +--------------+
#
# WHAT GETS CACHED
# ================
# After the first trace, save_reuse_entry stores an InvokeSubgraphReuseEntry
# (in _guards.py) containing:
#   - body_name/body_gmod: the traced subgraph
#   - arg_sources: sources of the original call's arguments
#   - subgraph_input_mapping: how each lifted arg maps back to user inputs or captures
#   - output_metadata: shape/stride/dtype/device of outputs
#
# Paired with an InvokeSubgraphReuseCondition containing:
#   - input_checks: (tag, tensor_metadata) per input
#   - guards: (source, handler, expected, guard) tuples
#   - treespec: pytree structure of the args
#   - traced_sources: sources accessed during the trace
#
# CACHE LOOKUP (is_reusable)
# ==========================
# On subsequent calls:
#   1. Input structure match -- same treespec, tags, tensor metadata.
#   2. Source replacement -- clone each guard's source with a replacement map
#      (old: L['self'].layers[0].weight -> new: L['self'].layers[1].weight),
#      then evaluate against the new source's runtime value.
#   3. Mutation check -- reject if the subgraph mutated any captured var.
#
# A shared resolve_cache memoizes intermediate source resolution (e.g.
# L['self'].layers is evaluated once and reused across all guards).
#
# STAMP OUT (stamp_out_subgraph)
# ==============================
# On cache hit, reconstruct the argument list using the freevar mapping
# (list[LiftedArgOrigin]):
#
#   LiftedUserArg(index)
#       User arg (activation / explicit input).
#       Looked up from new call's flat proxies.
#
#   LiftedCapturedSource(source)
#       Sourceful captured var (weight, param, etc).
#       Source is cloned with replacement map, resolved via
#       VariableBuilder. Deduplicates via input_source_to_var.
#
#   LiftedSyntheticObject(ctor_fn, ctor_args, ctor_arg_sources)
#       Synthetic object (opaque type with SyntheticLocalSource).
#       Reconstructed via synthetic_graph_input with cached constructor info.
#
# SAFETY
# ======
# In normal Dynamo compilation, safety is enforced at runtime: guards are
# installed during tracing and re-evaluated on every subsequent call against
# real Python objects.  Subgraph reuse operates differently — we are in the
# middle of tracing, there are no real Python objects, only VariableTrackers.
# We must answer: what could cause the second invocation of a nested compile
# region to produce a different trace than the first?
#
# VariableTrackers fall into two categories:
#
# 1. Intermediates — values produced during tracing with no originating source
#    (e.g. the result of a prior FX op). These can reach a nested compile region
#    only via (a) the region's explicit function arguments, or (b) closure
#    capture. We do not support nested-function regions that close over tensors,
#    so only (a) applies. For explicit arguments, the set of types we support is
#    small and well-defined: TensorVariable, SymNodeVariable, and
#    ConstantVariable (enum members included). Each has a cheap structural
#    comparison (tensor metadata, symnode identity, constant value equality).
#    We also snapshot the pytree treespec of the argument list and verify it
#    matches on lookup, ensuring the flattened structure is identical.
#    A sourceless object -- an nn.Module built during tracing, say -- has
#    neither a structural comparison nor guards, so it is not eligible at all.
#
# 2. Sourceful variables — values with a known originating source (e.g. a
#    module attribute or a local variable visible in the outer frame). For these
#    we collect the guard delta from the first trace, parameterize the guard
#    sources by replacing the original arg sources with the new arg sources, and
#    re-evaluate the guards by resolving each source against the live f_locals /
#    f_globals. The one extra hazard here is mutation: if the outer trace
#    mutates a sourceful object between the first and second invocations, the
#    cached guards would evaluate against stale values. We therefore also check
#    that none of the sources read by the cached subgraph have been mutated in
#    the outer SideEffects tracker before accepting a reuse.
#
# - max_reuse_entries (default 8, configurable via nested_compile_region arg)
#   caps cache entries per function. Exceeding it raises RuntimeError.
# - Guard failures logged with guard type + user stack trace.
#   Enable: TORCH_LOGS='+hierarchical_compile'
# ---------------------------------------------------------------------------
# Auto-cache helpers for invoke_subgraph
# ---------------------------------------------------------------------------


class InputTag(enum.Enum):
    TENSOR = "tensor"
    SYMNODE = "symnode"
    CONSTANT = "constant"
    OBJECT = "object"


class InputFingerprint(NamedTuple):
    # (InputTag, VariableTracker) pairs for each leaf input.
    flat_vts: list[tuple[InputTag, VariableTracker]]
    # 1-1 mapping to flat_vts: source for each leaf, or None if the VT has no source.
    arg_sources: list[Source | None]
    # True if any leaf VT had an unsupported type for reuse.
    has_unknown: bool = False
    # TreeSpec from pytree.tree_flatten of the (args, kwargs) structure.
    treespec: pytree.TreeSpec | None = None


def is_constant_like(
    vt: Any,
) -> TypeGuard[ConstantVariable | UserDefinedObjectVariable]:
    """Whether a leaf VT is compared by value, via ``vt.value``.

    Enum members are UserDefinedObjectVariable (not ConstantVariable) in
    Dynamo, but they are immutable singletons, so value comparison is as
    sound as it is for ConstantVariable.
    """
    if isinstance(vt, ConstantVariable):
        return True
    return isinstance(vt, UserDefinedObjectVariable) and isinstance(vt.value, enum.Enum)


def classify_vt(vt: Any) -> InputTag | None:
    """Return the tag for a leaf VT, or None if unsupported."""
    if isinstance(vt, TensorVariable):
        return InputTag.TENSOR
    elif isinstance(vt, SymNodeVariable):
        return InputTag.SYMNODE
    elif is_constant_like(vt):
        return InputTag.CONSTANT
    elif isinstance(vt, UserDefinedObjectVariable) and vt.source is not None:
        # Covers nn.Modules too -- UnspecializedNNModuleVariable is a
        # UserDefinedObjectVariable subclass. No metadata is recorded; reuse
        # safety comes entirely from re-evaluating the guards installed on this
        # object's source and on the sources derived from it. A sourceless
        # object has no guards to re-evaluate, so it stays unsupported.
        return InputTag.OBJECT
    return None


def build_input_fingerprint(
    tx: "InstructionTranslatorBase",
    fn_args_vt: Any,
    kwargs: dict[str, Any],
) -> InputFingerprint:
    """Build an InputFingerprint by flattening (args, kwargs) via pytree.

    Flattens the argument structure into leaf VTs, classifying each leaf as
    tensor/symnode/constant/module. Also records the TreeSpec so that
    cache lookups can verify structural equivalence.

    Fast path: when kwargs is empty and all args are already leaf VTs
    (tensor/symnode/constant/module), skip the pytree flatten entirely.
    """
    # Fast path: flat args, no kwargs — skip pytree machinery.
    if not kwargs:
        all_leaf = True
        for vt in fn_args_vt:
            if classify_vt(vt) is None:
                all_leaf = False
                break
        if all_leaf:
            return build_fingerprint_fast(fn_args_vt)

    return build_fingerprint_with_pytree(tx, fn_args_vt, kwargs)


def build_fingerprint_fast(fn_args_vt: Any) -> InputFingerprint:
    """Build fingerprint for the common case of flat leaf args, no kwargs."""
    flat_vts: list[tuple[InputTag, VariableTracker]] = []
    arg_sources: list[Source | None] = []
    for vt in fn_args_vt:
        tag = classify_vt(vt)
        if tag is None:
            raise AssertionError(
                f"classify_vt returned None for {type(vt).__name__} in fast path"
            )
        flat_vts.append((tag, vt))
        # Always append (even None) to keep positional alignment with flat_vts
        # so that source_replacement zip pairing is correct across calls.
        arg_sources.append(getattr(vt, "source", None))
    return InputFingerprint(flat_vts, arg_sources)


def build_fingerprint_with_pytree(
    tx: "InstructionTranslatorBase",
    fn_args_vt: Any,
    kwargs: dict[str, Any],
) -> InputFingerprint:
    """Build fingerprint via pytree flatten for nested/kwargs cases.

    Recurses over the pytree structure natively (untraced), inlining/tracing
    only each container node's own ``flatten_fn`` rather than the full
    recursive tree_flatten dispatch around it. This is safe because node-type
    classification only depends on ``type()``, never on tensor values.

    Note: skipping the traced registry dispatch also means we no longer
    install guards on the pytree registry itself, for the plain-function
    flatten_fn case (the non-FunctionType fallback below still traces the
    full tree_flatten and so still installs them). That's fine either way:
    this fingerprint is built fresh from the live registry on every
    reuse-lookup (register_pytree_node isn't traceable, so the registry
    can't change mid-trace), and reuse is separately gated on treespec/tag
    equality. So a registry change between compiles can only make a cached
    subgraph ineligible for reuse (has_unknown / treespec mismatch), never
    silently wrong.
    """
    from torch._dynamo.variables.builder import SourcelessBuilder

    flat_vts: list[tuple[InputTag, VariableTracker]] = []
    arg_sources: list[Source | None] = []
    has_unknown = False

    def add_leaf(vt: VariableTracker) -> None:
        nonlocal has_unknown
        tag = classify_vt(vt)
        if tag is None:
            has_unknown = True
        else:
            flat_vts.append((tag, vt))
            # Always append (even None) to keep positional alignment with flat_vts.
            arg_sources.append(getattr(vt, "source", None))

    def flatten(node_vt: VariableTracker) -> pytree.TreeSpec:
        nonlocal has_unknown
        try:
            node_type = node_vt.python_type()
        except NotImplementedError:
            has_unknown = True
            return pytree.treespec_leaf()
        # Keep in sync with pytree._get_node_type.
        if pytree.is_namedtuple_class(node_type):
            node_type = collections.namedtuple

        if node_type not in pytree.SUPPORTED_NODES:
            add_leaf(node_vt)
            return pytree.treespec_leaf()

        flatten_fn = pytree.SUPPORTED_NODES[node_type].flatten_fn
        if not isinstance(flatten_fn, types.FunctionType):
            # _make_inlined only supports plain Python functions (it always
            # wraps its argument in a UserFunctionVariable). A flatten_fn
            # registered as e.g. a functools.partial, bound method, or
            # callable object can't go through it directly. Fall back to
            # tracing the full recursive tree_flatten for this subtree, which
            # dispatches calls generically and so handles any callable.
            leaves_vt, treespec_vt = unpack_iterable(
                tx, _make_inlined(tx, pytree.tree_flatten)(node_vt)
            )
            for leaf_vt in unpack_iterable(tx, leaves_vt):
                add_leaf(leaf_vt)
            return treespec_vt.as_python_constant()

        children_vt, context_vt = unpack_iterable(
            tx, _make_inlined(tx, flatten_fn)(node_vt)
        )
        context = context_vt.as_python_constant()
        child_specs = [flatten(child) for child in unpack_iterable(tx, children_vt)]
        return pytree.TreeSpec(node_type, context, child_specs)

    container_vt = SourcelessBuilder.create(tx, (list(fn_args_vt), kwargs))
    treespec = flatten(container_vt)

    return InputFingerprint(flat_vts, arg_sources, has_unknown, treespec)


def sym_num_key(sym_num: Any) -> Any:
    """Key for matching a symbolic input against a cached one.

    Compares the symbolic expression rather than the SymInt object. Each tensor
    holds its own SymInt objects, so two arguments carrying the same symbol
    (e.g. the atom count threaded through successive layers) are distinct
    objects. Equal expressions mean the same value, which is what reuse needs;
    distinct symbols still have distinct expressions.

    ``expr`` rather than ``_expr`` on purpose: it has the ShapeEnv's
    replacements applied, so a symbol that was specialized keys on the value it
    was specialized to.
    """
    return sym_num.node.expr


def get_flat_proxies(fingerprint: InputFingerprint) -> list[Proxy]:
    """Collect deduplicated proxies from tensor/symnode leaves."""
    seen: set[torch.fx.Node] = set()
    flat_proxies: list[Proxy] = []
    for tag, vt in fingerprint.flat_vts:
        if tag in (InputTag.TENSOR, InputTag.SYMNODE):
            proxy = vt.as_proxy()
            if proxy.node not in seen:
                seen.add(proxy.node)
                flat_proxies.append(proxy)
    return flat_proxies


@dataclass
class LiftedUserArg:
    """Lifted arg that came from a user argument (intermediate activation or explicit input)."""

    index: int


@dataclass
class LiftedCapturedSource:
    """Lifted arg that is a captured variable (e.g. a weight or parameter) with a Source."""

    source: Any  # Source


@dataclass
class LiftedSyntheticObject:
    """Lifted arg that is a TorchScriptObject with a SyntheticLocalSource."""

    ctor_fn: Any  # Callable
    ctor_args: tuple[Any, ...]
    ctor_arg_sources: tuple[Any, ...] | None


@dataclass
class LiftedBoundSymbol:
    """Lifted arg that is a SymInt already bound as a graph input.

    SymInt graph inputs are created during tensor wrapping (not through
    VariableBuilder.wrap_symint), so they aren't registered in
    unspec_variable_map or variable_tracker_cache. Using LiftedCapturedSource
    for these would resolve the source to a concrete Python int via
    source.get_value() instead of reusing the existing symbolic proxy.
    """

    expr: Any  # sympy.Expr


LiftedArgOrigin = (
    LiftedUserArg | LiftedCapturedSource | LiftedSyntheticObject | LiftedBoundSymbol
)


def get_fn_code(fn_var: Any) -> types.CodeType | None:
    if isinstance(fn_var, UserFunctionVariable):
        return fn_var.get_function().__code__
    elif isinstance(fn_var, UnspecializedNNModuleVariable):
        return (
            fn_var.value.forward.__func__.__code__  # pyrefly: ignore[missing-attribute]
        )
    return None


def has_mutated_vars(
    tx: "InstructionTranslatorBase",
    traced_sources: OrderedSet[Source],
) -> bool:
    """Check if any source accessed by the subgraph has been mutated.

    SideEffects.mutated_sources records the exact AttrSource for every
    store_attr call. A simple set intersection with traced_sources tells
    us whether any source the subgraph read was later written to.
    """
    overlap = tx.output.side_effects.mutated_sources & traced_sources
    if overlap:
        hc_log.debug(
            "subgraph_reuse: mutated sources detected -- %s",
            overlap,
        )
        return True
    return False


def is_reuse_eligible(
    tx: "InstructionTranslatorBase",
    body_r: Any,
    fingerprint: InputFingerprint,
    tracing_info: "SubgraphTracingInfo",
    traced_sources: OrderedSet[Source] | None = None,
    has_reuse_hash_fn: bool = False,
) -> bool:
    """Best-effort check for whether a traced subgraph result can be reused.

    It is possible that a subgraph is morally reusable but does not fall
    into the limited support that Dynamo has today. Current limitations:
      - The subgraph must not have side effects.
      - No sourceful variable accessed by the subgraph may have been
        mutated, because guards are snapshotted on source values at trace
        time — if the underlying object changed since then, the cached
        guards would silently evaluate against stale values.
      - Output must be a single tensor, or a tuple/list of plain tensors.
      - All flattened inputs must be one of: tensor, symnode, constant
        (including enum members), or a sourceful user-defined object
        (nn.Modules included) — for sourceless or other input types we
        rely on the treespec and tags for structural matching, so only
        types with well-defined comparison semantics are supported.

    When ``has_reuse_hash_fn`` is True, side-effect and mutation checks are
    skipped because the hash key replaces guards — there are no guards to
    go stale from mutations.
    """
    if not has_reuse_hash_fn:
        if tracing_info.side_effect_stack is not None:
            stack_msg = "\n" + "".join(
                traceback.format_list(tracing_info.side_effect_stack)
            )
            hc_log.debug(
                "subgraph_reuse: not eligible -- subgraph has side effects%s",
                stack_msg,
            )
            return False

        if traced_sources and has_mutated_vars(tx, traced_sources):
            return False

    if isinstance(body_r, TensorVariable):
        pass
    elif isinstance(body_r, (TupleVariable, ListVariable)):
        non_tensor = [
            type(item).__name__
            for item in body_r.items
            if not isinstance(item, TensorVariable)
        ]
        if non_tensor:
            hc_log.debug(
                "subgraph_reuse: not eligible -- output contains non-tensor types: %s",
                non_tensor,
            )
            return False
    else:
        hc_log.debug(
            "subgraph_reuse: not eligible -- output type %s is not tensor or tuple/list",
            type(body_r).__name__,
        )
        return False

    if fingerprint.has_unknown:
        hc_log.debug(
            "subgraph_reuse: not eligible -- unsupported input VT types",
        )
        return False

    return True


def build_reuse_condition(
    tx: "InstructionTranslatorBase",
    fingerprint: InputFingerprint,
    traced_sources: OrderedSet[Source],
) -> InvokeSubgraphReuseCondition | None:
    """Build an InvokeSubgraphReuseCondition from a traced subgraph.

    A reuse condition is a mix of two kinds of checks:

    1. **Input tag checks** (from flat_vts): For each flattened leaf VT,
       we record its tag (InputTag.TENSOR/SYMNODE/CONSTANT/OBJECT) and
       metadata (e.g. tensor shape/stride/dtype/device/requires_grad).
       At lookup time, the treespec ensures structural equivalence, and
       then we compare tags and metadata leaf-by-leaf.

    2. **Guard checks** (from traced_sources): During the subgraph trace,
       every source accessed via VariableBuilder is recorded. We look up
       all guards installed on those sources (and on the arg_sources) to
       build the set of guards that must be re-evaluated on cache hit.
       This is more robust than guard diffing because it catches guards
       that were already installed before the subgraph trace began.

    Raise if any guard type is unsupported, as a feedback for compiler
    developers to support that guard type.
    """
    from torch._guards import InvokeSubgraphReuseCondition

    input_checks: list[tuple[InputTag, object]] = []
    for tag, vt in fingerprint.flat_vts:
        if tag == InputTag.TENSOR:
            if not isinstance(vt, TensorVariable):
                raise AssertionError(
                    f"expected TensorVariable for TENSOR tag, got {type(vt).__name__}"
                )
            example = vt.proxy.node.meta.get("example_value", None)
            if example is None:
                hc_log.debug(
                    "subgraph_reuse: cannot build condition -- tensor input has no example_value"
                )
                return None
            input_checks.append((InputTag.TENSOR, extract_tensor_metadata(example)))
        elif tag == InputTag.SYMNODE:
            if not isinstance(vt, SymNodeVariable):
                raise AssertionError(
                    f"expected SymNodeVariable for SYMNODE tag, got {type(vt).__name__}"
                )
            input_checks.append((InputTag.SYMNODE, sym_num_key(vt.sym_num)))
        elif tag == InputTag.CONSTANT:
            if not is_constant_like(vt):
                raise AssertionError(
                    f"expected constant-like VT for CONSTANT tag, got {type(vt).__name__}"
                )
            # Type is part of the key: `Mode.ADD == 1` and `True == 1` compare
            # equal, but an isinstance() check inside the region traces
            # differently for each, so value equality alone is not enough.
            input_checks.append((InputTag.CONSTANT, (type(vt.value), vt.value)))
        elif tag == InputTag.OBJECT:
            input_checks.append((InputTag.OBJECT, None))
        else:
            raise AssertionError(
                f"Unexpected input tag '{tag}' for {type(vt).__name__} -- "
                f"is_reuse_eligible should have rejected this"
            )

    # Collect all guards for sources accessed during the subgraph trace
    # and for the flattened arg sources.
    all_sources = set(traced_sources)
    all_sources.update(s for s in fingerprint.arg_sources if s is not None)
    all_relevant_guards: set[Guard] = set()
    for source in all_sources:
        all_relevant_guards.update(tx.output.guards.get_guards_for_source(source))

    guard_tuples: list[tuple[Source, GuardCheckSpec, object, Guard]] = []
    for guard in all_relevant_guards:
        source = guard.originating_source
        type_str = guard.create_fn_name()
        handler = GUARD_VALUE_DISPATCH.get(type_str)

        if handler is SKIP_GUARD:
            continue

        if handler is None or isinstance(handler, UnsupportedGuardCheckSpec):
            raise RuntimeError(
                f"subgraph_reuse: unsupported guard type '{type_str}' on source '{source.name}'"
            )

        try:
            value = tx.output.resolve_source_value(source)
        except Exception:
            raise RuntimeError(
                f"subgraph_reuse: failed to resolve source '{source.name}' for {type_str} guard"
            ) from None

        # TODO(anijain2305): vLLM workaround -- skip CONSTANT_MATCH on
        # strings. Re-evaluate once vLLM migrates off this pattern.
        # if type_str == "CONSTANT_MATCH" and isinstance(value, str):
        #     continue

        handler = cast(GuardCheckSpec, handler)
        expected = handler.get_metadata_fn(guard, value)
        guard_tuples.append((source, handler, expected, guard))

    hc_log.debug("Number of guards %s", len(guard_tuples))

    return InvokeSubgraphReuseCondition(
        input_checks=input_checks,
        guards=guard_tuples,
        treespec=fingerprint.treespec,
        traced_sources=traced_sources,
    )


def build_source_replacement(
    old_arg_sources: list[Source | None],
    new_arg_sources: list[Source | None],
) -> dict[Source, Source]:
    """Map old arg sources to new arg sources for remapping captured variable sources."""
    return {
        old: new
        for old, new in zip(old_arg_sources, new_arg_sources)
        if old is not None and new is not None and old != new
    }


def is_reusable(
    tx: "InstructionTranslatorBase",
    condition: "InvokeSubgraphReuseCondition",
    fingerprint: InputFingerprint,
    cached_entry: InvokeSubgraphReuseEntry,
) -> bool:
    """Check if a cached subgraph can be reused for the current call.

    Three-phase check:
    (1) Verify that intermediates (tensor metadata, symnode types, constant
        values) match the cached input_checks — these are lightweight
        structural comparisons that don't require source resolution.
    (2) Check for mutations on the remapped traced_sources — if any source
        the subgraph read has been mutated since the original trace, the
        cached guards would evaluate against stale values.
    (3) Build a source replacement mapping (old sources → new sources) and
        re-evaluate the snapshotted guards under the new sources.
    """
    # Structural check: treespec must match first.
    if condition.treespec is not None and fingerprint.treespec != condition.treespec:
        hc_log.debug(
            "subgraph_reuse: reuse failed -- treespec mismatch",
        )
        return False

    # Input count, tags, and metadata must match.
    # Tensor metadata (shape, stride, dtype, device, requires_grad) is checked
    # here because TENSOR_MATCH guards for subgraph inputs typically already
    # exist in the outer graph before tracing and thus won't appear in the
    # guard delta.
    if len(condition.input_checks) != len(fingerprint.flat_vts):
        hc_log.debug(
            "subgraph_reuse: reuse failed -- input count mismatch: cached %d vs current %d",
            len(condition.input_checks),
            len(fingerprint.flat_vts),
        )
        return False

    for i, ((cached_tag, cached_val), (cur_tag, cur_vt)) in enumerate(
        zip(condition.input_checks, fingerprint.flat_vts)
    ):
        if cached_tag != cur_tag:
            hc_log.debug(
                "subgraph_reuse: reuse failed -- input %d tag mismatch: cached '%s' vs current '%s'",
                i,
                cached_tag,
                cur_tag,
            )
            return False
        if cached_tag == InputTag.TENSOR:
            if not isinstance(cur_vt, TensorVariable):
                raise AssertionError(
                    f"expected TensorVariable for TENSOR tag, got {type(cur_vt).__name__}"
                )
            example = cur_vt.proxy.node.meta.get("example_value", None)
            if example is None:
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d tensor has no example_value",
                    i,
                )
                return False
            cur_meta = extract_tensor_metadata(example)
            if cur_meta != cached_val:
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d tensor metadata mismatch",
                    i,
                )
                return False
        elif cached_tag == InputTag.SYMNODE:
            if not isinstance(cur_vt, SymNodeVariable):
                raise AssertionError(
                    f"expected SymNodeVariable for SYMNODE tag, got {type(cur_vt).__name__}"
                )
            if sym_num_key(cur_vt.sym_num) != cached_val:
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d symnode mismatch: cached '%s' vs current '%s'",
                    i,
                    cached_val,
                    cur_vt.sym_num,
                )
                return False
        elif cached_tag == InputTag.CONSTANT:
            if not is_constant_like(cur_vt):
                raise AssertionError(
                    f"expected constant-like VT for CONSTANT tag, got {type(cur_vt).__name__}"
                )
            cached_type, cached_value = cast(tuple[type, Any], cached_val)
            if type(cur_vt.value) is not cached_type:
                # Not deferred to the source check below: a value guard cannot
                # catch a type change that alters the trace (isinstance, etc).
                hc_log.debug(
                    "subgraph_reuse: reuse failed -- input %d constant type "
                    "mismatch: cached '%s' vs current '%s'",
                    i,
                    cached_type,
                    type(cur_vt.value),
                )
                return False
            if cur_vt.value != cached_value:
                # If both the cached and current arg have sources, source
                # replacement in stamp_out will resolve the correct value.
                cached_src = (
                    cached_entry.arg_sources[i]
                    if i < len(cached_entry.arg_sources)
                    else None
                )
                new_src = (
                    fingerprint.arg_sources[i]
                    if i < len(fingerprint.arg_sources)
                    else None
                )
                if cached_src is None or new_src is None:
                    hc_log.debug(
                        "subgraph_reuse: reuse failed -- input %d constant mismatch "
                        "with no source to replace: cached '%s' vs current '%s'",
                        i,
                        cached_val,
                        cur_vt.value,
                    )
                    return False

    source_replacement = build_source_replacement(
        cached_entry.arg_sources, fingerprint.arg_sources
    )

    # Parameterized source - this function gives you new sources parameterized
    # on the arg_sources. For example, if the input to the nested compile region
    # is a nn Module layer with source `layers[0]`, then old source
    # `layers[0].weight` gets remapped to `layers[1].weight`. This
    # parameterization is central in getting the new sources and then running
    # guards on them.
    def replacement_fn(s: Source) -> Source:
        return source_replacement.get(s, s)

    # Check for mutations on remapped traced_sources.
    if source_replacement:
        remapped = OrderedSet(s.clone(replacement_fn) for s in condition.traced_sources)
    else:
        remapped = condition.traced_sources
    if has_mutated_vars(tx, remapped):
        return False

    # If no sources changed, all guards were already checked during the
    # original trace and will trivially pass again.
    if not source_replacement:
        return True

    # Shared resolution context so source.get_value memoizes intermediate
    # results (e.g. common base sources) across all guards in this check.
    resolve_globals: dict[str, Any] = {
        "G": tx.output.root_tx.f_globals,
        "L": tx.output.root_tx.f_locals,
    }
    resolve_locals: dict[str, Any] = {}
    resolve_cache: dict[Source, Any] = {}

    for source, handler, expected, guard in condition.guards:
        new_source = source.clone(replacement_fn)
        # Source unchanged after replacement — guard already passed during
        # the original trace, skip re-evaluation.
        if new_source == source:
            continue

        try:
            value = new_source.get_value(resolve_globals, resolve_locals, resolve_cache)
        except Exception:
            hc_log.debug(
                "subgraph_reuse: reuse failed -- cannot resolve source\n"
                "  guard type: %s\n"
                "  guard source: %s\n"
                "  guard source name: %s\n"
                "  user stack:\n%s",
                guard.create_fn_name(),
                new_source,
                new_source.name,
                "".join(guard.user_stack.format())
                if guard.user_stack
                else "<no stack>",
            )
            return False

        if not handler.eval_fn(value, expected):
            hc_log.debug(
                "subgraph_reuse: reuse failed --\n"
                "  guard type: %s\n"
                "  guard source: %s\n"
                "  guard source name: %s\n"
                "  expected: %s\n"
                "  got: %s\n"
                "  user stack:\n%s",
                guard.create_fn_name(),
                new_source,
                new_source.name,
                expected,
                value,
                "".join(guard.user_stack.format())
                if guard.user_stack
                else "<no stack>",
            )
            return False

    return True


def has_reuse_entries(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
) -> bool:
    """Cheap check: does the cache have any entries for this function?"""
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return False
    fn_code = get_fn_code(fn_var)
    return fn_code is not None and fn_code in invoke_subgraph_cache.subgraph_reuse_cache


def find_reuse_match(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
    fingerprint: InputFingerprint,
) -> InvokeSubgraphReuseEntry | None:
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return None
    fn_code = get_fn_code(fn_var)
    if fn_code is None:
        return None

    # this evaluator function is called one by one for all the invoke subgraph
    # reuse entries - the one that evaluates to True is stamped out in the
    # graph.
    def evaluator(
        cond: "InvokeSubgraphReuseCondition", entry: InvokeSubgraphReuseEntry
    ) -> bool:
        return is_reusable(tx, cond, fingerprint, entry)

    return invoke_subgraph_cache.find_reuse_entry(fn_code, evaluator)


def save_reuse_entry(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
    fingerprint: InputFingerprint,
    body_name: str,
    body_gmod: torch.fx.GraphModule,
    config: NestedCompileRegionOptions | None,
    p_args: tuple[Any, ...],
    body_r: VariableTracker,
    example_value: Any,
    max_reuse_entries: int = 8,
    condition: "InvokeSubgraphReuseCondition | None" = None,
    hash_key: int | None = None,
) -> None:
    """Save a traced subgraph into the reuse cache for future cache hits.

    Builds an InvokeSubgraphReuseEntry with the freevar mapping (how each
    lifted arg maps back to user inputs or captured variables), output
    metadata, and arg sources. On a future cache hit, stamp_out_subgraph
    uses this entry to emit a new invoke_subgraph call without re-tracing.

    Exactly one of ``condition`` or ``hash_key`` must be provided.
    ``condition`` stores the entry in the guard-based cache (linear scan);
    ``hash_key`` stores it in the hash-key cache (O(1) lookup).
    """
    from torch._guards import InvokeSubgraphCache

    if not ((condition is None) != (hash_key is None)):
        raise AssertionError("Exactly one of condition or hash_key must be provided")

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return

    fn_code = get_fn_code(fn_var)
    if fn_code is None:
        return

    subgraph_input_mapping = build_subgraph_input_mapping(
        tx, p_args, fingerprint.flat_vts
    )
    single_tensor_output = isinstance(body_r, TensorVariable)

    # Count user-visible outputs from body_r. The graph may have additional
    # outputs from side-effect intermediates that stamp_out_subgraph must
    # not include when reconstructing the user-visible return value.
    user_output_vts: list[VariableTracker] = []
    VariableTracker.visit(
        lambda vt: user_output_vts.append(vt)
        if vt.is_tensor() or isinstance(vt, SymNodeVariable)
        else None,
        body_r,
    )
    num_user_outputs = len(user_output_vts)

    # Cache output tensor metadata so we can construct fresh FakeTensors on
    # cache hit without re-running the subgraph. This is safe because
    # invoke_subgraph does not support aliasing between inputs and outputs
    # (speculate_subgraph will fail if that happens).
    # example_value may contain SymInts (e.g. shape values for backward);
    # only record metadata for actual tensors.
    output_metadata = [
        (t.shape, t.stride(), t.dtype, t.device, t.requires_grad)
        for t in example_value
        if isinstance(t, torch.Tensor)
    ]

    entry = InvokeSubgraphReuseEntry(
        body_name=body_name,
        body_gmod=body_gmod,
        config=config,
        subgraph_input_mapping=subgraph_input_mapping,
        single_tensor_output=single_tensor_output,
        output_metadata=output_metadata,
        # Record arg sources so that on cache hit we can build a
        # source replacement mapping (old sources → new sources) to
        # rewrite captured variable sources for the current invocation.
        arg_sources=fingerprint.arg_sources,
        num_user_outputs=num_user_outputs,
    )
    if condition is not None:
        invoke_subgraph_cache.add_reuse_entry(
            fn_code, condition, entry, max_reuse_entries
        )
    else:
        if hash_key is None:
            raise AssertionError("hash_key must not be None when condition is None")
        invoke_subgraph_cache.add_reuse_entry_by_key(
            fn_code, hash_key, entry, max_reuse_entries
        )


def trace_reuse_hash_fn(
    tx: "InstructionTranslatorBase",
    reuse_hash_fn: Any,
    fn_args_vt: "list[VariableTracker]",
    kwargs: dict[str, VariableTracker],
) -> int:
    """Trace the user's reuse_hash_fn to get a constant integer hash key.

    Guards installed during the hash function tracing are skipped — the hash
    key itself is the reuse condition, not the guards.
    """
    from torch._dynamo.exc import Unsupported

    with tx.output.tracing_context.guards_context.skip_guard_install():
        try:
            result = _make_inlined(tx, reuse_hash_fn)(*fn_args_vt, **kwargs)
        except Unsupported as e:
            raise RuntimeError(
                f"reuse_hash_fn must be fully traceable without graph breaks. Got: {e}"
            ) from e

    if not isinstance(result, ConstantVariable) or not isinstance(result.value, int):
        raise RuntimeError(
            f"reuse_hash_fn must return a constant integer, got {result}"
        )

    return result.value


def find_reuse_entry_by_key(
    tx: "InstructionTranslatorBase",
    fn_var: Any,
    hash_key: int,
) -> InvokeSubgraphReuseEntry | None:
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return None
    fn_code = get_fn_code(fn_var)
    if fn_code is None:
        return None
    return invoke_subgraph_cache.find_reuse_entry_by_key(fn_code, hash_key)


def stamp_out_subgraph(
    tx: "InstructionTranslatorBase",
    fingerprint: InputFingerprint,
    cached: InvokeSubgraphReuseEntry,
) -> VariableTracker:
    """Emit a new invoke_subgraph call by stamping out a cached subgraph.

    Sources in the cached entry are parameterized: they refer to the original
    call's sources and must be rewritten to the current call's sources via
    source replacement before we can look up or create the corresponding
    graph placeholders.
    """
    from torch._dynamo.variables.builder import VariableBuilder
    from torch._dynamo.variables.higher_order_ops import add_call_function, make_attr

    flat_proxies = get_flat_proxies(fingerprint)
    new_arg_sources = fingerprint.arg_sources

    source_replacement = build_source_replacement(cached.arg_sources, new_arg_sources)

    new_lifted_args = []
    # Shared resolution context so get_value memoizes intermediate results
    # (e.g. L['self'].layers) across all freevars in this stamp-out.
    resolve_globals: dict[str, Any] = {
        "G": tx.output.root_tx.f_globals,
        "L": tx.output.root_tx.f_locals,
    }
    resolve_locals: dict[str, Any] = {}
    resolve_cache: dict[Source, Any] = {}

    # Find the args for the about-to-be-inserted invoke_subgraph call.
    for subgraph_input in cached.subgraph_input_mapping:
        if isinstance(subgraph_input, LiftedUserArg):
            new_lifted_args.append(flat_proxies[subgraph_input.index])
        elif isinstance(subgraph_input, LiftedBoundSymbol):
            from torch._dynamo.output_graph import LazyProxy

            proxy = tx.output.current_tracer.bound_symbols[subgraph_input.expr]
            if isinstance(proxy, LazyProxy):
                proxy = proxy()
                tx.output.current_tracer.bound_symbols[subgraph_input.expr] = proxy
            new_lifted_args.append(proxy)
        elif isinstance(subgraph_input, LiftedSyntheticObject):
            ctor_args = subgraph_input.ctor_args
            ctor_arg_sources = subgraph_input.ctor_arg_sources
            if ctor_arg_sources and source_replacement:
                new_ctor_args = []
                new_ctor_arg_sources = []
                for val, arg_src in zip(ctor_args, ctor_arg_sources):
                    if arg_src is not None:
                        new_src = arg_src.clone(lambda s: source_replacement.get(s, s))
                        val = new_src.get_value(
                            resolve_globals, resolve_locals, resolve_cache
                        )
                        arg_src = new_src
                    new_ctor_args.append(val)
                    new_ctor_arg_sources.append(arg_src)
                ctor_args = tuple(new_ctor_args)
                ctor_arg_sources = tuple(new_ctor_arg_sources)
            vt = tx.output.synthetic_graph_input(
                subgraph_input.ctor_fn, ctor_args, ctor_arg_sources
            )
            new_lifted_args.append(vt.as_proxy())
        elif isinstance(subgraph_input, LiftedCapturedSource):
            new_source = subgraph_input.source
            if source_replacement:
                new_source = new_source.clone(lambda s: source_replacement.get(s, s))
            # VariableBuilder deduplicates via input_source_to_var,
            # so this reuses existing graph placeholders automatically.
            value = new_source.get_value(resolve_globals, resolve_locals, resolve_cache)
            vt = VariableBuilder(tx, new_source)(value)
            new_lifted_args.append(vt.as_proxy())

    # Generate fake tensor outputs
    if tx.fake_mode is None:
        raise AssertionError("tx.fake_mode must not be None for stamp_out_subgraph")
    with tx.fake_mode:
        example_value = tuple(
            torch.empty_strided(
                shape,
                stride,
                dtype=dtype,
                device=device,
                requires_grad=req_grad,
            )
            for shape, stride, dtype, device, req_grad in cached.output_metadata
        )

    # Install the invoke_subgraph call
    body_node = make_attr(tx, cached.body_name)
    p_args = (body_node, cached.body_name, *new_lifted_args)
    flat_variable = add_call_function(
        tx,
        torch._higher_order_ops.invoke_subgraph,
        tuple(p_args),
        {},
        example_value,
        cached.config,
    )

    # Return only the user-visible outputs. The graph may have extra
    # intermediate outputs from side effects (allow_side_effects=True)
    # that should not be part of the user-facing return value.
    if cached.single_tensor_output:
        items = flat_variable.items  # pyrefly: ignore[missing-attribute]
        if not isinstance(items[0], TensorVariable):
            raise AssertionError(
                f"Expected tensor output but got {type(items[0]).__name__}"
            )
        return items[0]

    items = flat_variable.items  # pyrefly: ignore[missing-attribute]
    n = cached.num_user_outputs
    if n > 0 and n < len(items):
        from .builder import SourcelessBuilder

        return SourcelessBuilder.create(tx, tuple(items[:n]))
    return flat_variable


def build_subgraph_input_mapping(
    tx: "InstructionTranslatorBase",
    p_args: tuple[Any, ...],
    flat_vts: list[tuple[InputTag, VariableTracker]],
) -> list[LiftedArgOrigin]:
    """Build a mapping that records the origin of each lifted arg for a subgraph.

    On a cache hit, we stamp out a new invoke_subgraph call and need to
    reconstruct its argument list in the correct order. Each lifted arg
    (p_args[2:], skipping body_node and body_name) comes from one of:

    - LiftedUserArg: a user argument (intermediate activation or explicit input)
    - LiftedCapturedSource: a captured variable (e.g. a weight or parameter)
    - LiftedSyntheticObject: a TorchScriptObject with a SyntheticLocalSource
    - LiftedBoundSymbol: a SymInt already bound as a graph input
    """
    proxy_node_to_idx: dict[torch.fx.Node, int] = {}
    idx = 0
    for tag, vt in flat_vts:
        if tag in (InputTag.TENSOR, InputTag.SYMNODE):
            node = vt.as_proxy().node
            if node not in proxy_node_to_idx:
                proxy_node_to_idx[node] = idx
                idx += 1

    subgraph_input_mapping: list[LiftedArgOrigin] = []
    for outer_proxy in p_args[2:]:
        matched_idx = proxy_node_to_idx.get(outer_proxy.node, -1)
        if matched_idx >= 0:
            subgraph_input_mapping.append(LiftedUserArg(matched_idx))
        else:
            grapharg = outer_proxy.node.meta.get("grapharg", None)
            source = grapharg.source if grapharg is not None else None
            # SymInt freevars must reuse the existing symbolic proxy rather
            # than resolving via source.get_value() (which returns the
            # concrete int). They appear as either:
            # - placeholder nodes with grapharg.example being a SymInt
            # - call_function nodes (e.g. sym_size_int) with no grapharg
            # In both cases, store the sympy expression and look it up in
            # bound_symbols during stamp-out.
            example = (
                grapharg.example
                if grapharg is not None
                else outer_proxy.node.meta.get("example_value", None)
            )
            if isinstance(example, torch.SymInt):
                # _expr rather than expr: expr applies the ShapeEnv's
                # replacements, so a symbol the region lifted before it was
                # specialized comes back as a constant, which bound_symbols has
                # no entry for.
                subgraph_input_mapping.append(LiftedBoundSymbol(example.node._expr))
                continue
            if source is None:
                raise AssertionError(
                    f"Freevar has no source: node.op={outer_proxy.node.op} "
                    f"node.name={outer_proxy.node.name} -- this likely means a "
                    f"function argument was not included in the proxy matching"
                )
            if isinstance(source, SyntheticLocalSource):
                ctor_info = tx.output.synthetic_source_ctor_info.get(source)
                if ctor_info is not None:
                    ctor_fn, ctor_args, ctor_arg_sources = ctor_info
                    subgraph_input_mapping.append(
                        LiftedSyntheticObject(ctor_fn, ctor_args, ctor_arg_sources)
                    )
                    continue
            subgraph_input_mapping.append(LiftedCapturedSource(source))
    return subgraph_input_mapping


class InvokeSubgraphHigherOrderVariable(WrapHigherOrderVariable):
    _HOP_NAME = "torch.ops.higher_order.invoke_subgraph"
    _ALLOW_FALLBACK_TO_EAGER = False
    supports_input_mutation = True
    supports_aliasing = False
    allow_side_effects = True
    # invoke_subgraph is NOT desugared in AOTAutograd, so the HOP input/output
    # shouldn't alias. For checkpoint HOP, we inline it so we don't need
    # alias analysis as functionalization would just work on the flat graph.
    filter_aliased_intermediates = True

    # pyrefly: ignore[bad-override]
    def install_subgraph_in_output_graph(
        self,
        tx: "InstructionTranslatorBase",
        fn_vt: VariableTracker,
        fn_args_vt: "list[VariableTracker]",
        kwargs: dict[str, VariableTracker],
        body_gmod: GraphModule,
        attr_name: str,
    ) -> str:
        # Check if the subgraph from speculate_subgraph (body_gmod) and the fake
        # inputs have already been seen before. If yes, the subgraph is already
        # installed in the output graph and we can just access the subgraph
        # using the saved attr name.

        if not isinstance(fn_vt, (UnspecializedNNModuleVariable, UserFunctionVariable)):
            unimplemented(
                gb_type="Encountered non user function variable during invoke_subgraph HOP tracing",
                context=str(fn_vt),
                explanation="invoke_subgraph does not support non user function variable",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        invoke_subgraph_cache = (
            tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
                torch._higher_order_ops.invoke_subgraph
            )
        )

        if isinstance(fn_vt, UserFunctionVariable):
            fn_code = fn_vt.get_function().__code__
            fn_name = fn_vt.get_function().__name__
        else:
            if not isinstance(fn_vt, UnspecializedNNModuleVariable):
                raise AssertionError(
                    f"expected UnspecializedNNModuleVariable, got {type(fn_vt).__name__}"
                )
            fn_code = fn_vt.value.forward.__func__.__code__  # type: ignore[attr-defined]
            fn_name = fn_vt.value.forward.__name__  # type: ignore[attr-defined]
        # pyrefly: ignore [implicit-any]
        previously_installed_submodules = []
        if invoke_subgraph_cache:
            previously_installed_submodules = (
                invoke_subgraph_cache.get_dynamo_installed_submodules(fn_code)
            )
            current_mod = body_gmod
            # NB - reverse is more likely to cause a hit sooner because first
            # graph can have requires_grad=False for a few inputs
            for submodule_name in reversed(previously_installed_submodules):
                if submodule_name not in tx.output.nn_modules:
                    raise AssertionError(
                        f"submodule '{submodule_name}' not found in nn_modules"
                    )
                previous_mod = tx.output.nn_modules[submodule_name]
                if not tx.fake_mode:
                    raise AssertionError(
                        "tx.fake_mode must be set for subgraph comparison"
                    )
                from torch._dynamo.variables.higher_order_ops import (
