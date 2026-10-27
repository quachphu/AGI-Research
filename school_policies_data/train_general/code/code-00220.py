        :func:`~torch.utils.checkpoint.checkpoint` for
        the important considerations and limitations of this variant. It is
        recommended that you use ``use_reentrant=False``.

    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.5 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint.checkpoint_sequential: the use_reentrant "
            "parameter should be passed explicitly. "
            "In version 2.9 we will raise an exception if use_reentrant "
            "is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.", stacklevel=2
        )
        use_reentrant = True

    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        input = checkpoint(
            run_function(start, end, functions),
            input,
            use_reentrant=use_reentrant,
            preserve_rng_state=preserve,
        )
    return run_function(end + 1, len(functions) - 1, functions)(input)


def _internal_assert(cond) -> None:
    if not cond:
        raise AssertionError(
            "Something went unexpectedly wrong in activation checkpoint. "
            "Please report this bug by filing an issue to PyTorch."
        )


# NOTE [ Nestable Checkpoint ]
#
# The semantics of nested checkpoint can be defined by two basic rules.
# Following the two rules leads to an important implication that is central
# to motivating the design.
#
# Rule 1. Saved tensors are managed by inner-most checkpoint only and hidden
#         from any outer layers of checkpoint.
#
# Rule 2. The inputs of inner checkpoints are treated as tensors saved to its
#         parent checkpoint.
#
# Implication: To recompute any given saved tensor, we need to recompute all of
#              the checkpoints wrapping it.
#
# Why is this implied? To unpack a saved tensor X during backward we need to
# recompute the inner-most checkpoint (#1), and in order to recompute that
# checkpoint I need to have its inputs, which are managed by that checkpoint's
# parent (#2), which thus also needs to be recomputed first. Continue this line
# of reasoning and we realize that in order to unpack X, all checkpoints that
# were active at the time X was saved need to be recomputed. (unless we have
# already done so in that backward for some other saved tensor).
#
# In practice, we use a noop autograd Function to save inputs as saved tensors.
# During unpack calling ctx.saved_tensor triggers the parent checkpoint to
# recompute.
#
# Rule 3. We should start recomputation as if there are no checkpoints currently
#         active. Checkpoints encountered during recomputation are still
#         respected.
#
# When we start recomputation, we push the saved variable hook meant for
# recomputation on the stack. See examples in Rule 6 for more context.
#
#                                  * * * *
#
# Beyond the basic semantics specific to nested checkpoint, we impose several
# more constraints that may apply to checkpointing in general.
#
# Rule 4. Lifetime of recomputed tensors
#
#         Recomputed tensors are considered specific to particular invocations
#         of backward and are always cleared immediately as they are unpacked
#         Particularly, we require this to happen even if retain_graph=True.
#
# [ Implementation details of Rule 4 ]
#
# If we were okay with recomputed tensors staying alive after backward is run
# with retain_graph=True, we would store recomputed variables as the values of a
# WeakKeyDictionary and pack strong references to the keys, so that as we
# backward, those packed keys would be cleared as long as retain_graph=False.
# Clearing the packed key clears the corresponding entry in the WKD.
#
# If we wish recomputed variables to be immediately cleared as we unpack them in
# the retain_graph=True case, we cannot rely on the packed keys to be cleared by
# backward automatically. Instead of packing the strong reference to the key
# directly, we pack a container object, which we manually clear as we unpack.
#
# An important detail is that if a second backward happens, the second
# recomputation needs to reset the container with a newly created key.
#
# Rule 5. Stop recomputation as soon as we've recomputed the saved tensors we
#         know we need.
#
# [ Implementation details of Rule 5 ]
#
# During recomputation, raise an exception if the number of recomputed tensors
# matches the number of tensors that we expected to recompute. We wrap the
# recomputation call with a try-catch to catch this specific exception. See
# Rule #6 below for some examples.
#
# Rule 6. We support doing backward inside checkpoint context
#
# [ retain_graph is True]
#
# def fn(x):
#   y = x.sin()
#   z = y.cos()
#   gx, = torch.autograd.grad(z, x, retains_grad=True)
#   return gx, z
#
# out = checkpoint(fn)(inp)
# out.backward()
#
# Because z is saved by cos while checkpoint is enabled, it would not be
# actually saved, and so the .grad() call inside must trigger a recomputation.
#
# During recomputation the "inner pack hook" has two responsibilities:
#
# 1) As usual, populating the WeakKeyDictionary storing recomputed tensors
# 2) Pack the actual tensor (detached) so that one may perform backward on the
#    recomputed graph. The tensors saved to this graph will live until the end
#    of recomputation, or die earlier if someone performs backward with
#    retain_graph=False.
#
# More generally performing backward on the recomputed graph occurs in the
# following cases:
# - If backward is performed inside forward,
#   - During the original forward IF early-stop is disabled
#   - During the original backward
# - If there are multiple .grad()/.backward() calls, we would perform backward
#   on the recomputed graph even if early-stop is enabled (see the example below)
#
# [ retain_graph is False ]
#
# The example below shows what happens if during recomputation we find that some
# of the tensors we are trying to recompute have already been cleared.
#
# Spoiler: we don't do anything special, we just skip over them!
#
# def fn(x):
#   y = x.sin()                           # (1)
#   z = y.cos()                           # (2)
#   gx, = torch.autograd.grad(z, x)       # (3)
#   return x.cos() * gx                   # (4)
#
# out = checkpoint(fn)(inp)
# out.backward()                          # (5)
#
# 1, 2. Don't save x and y since we are inside a checkpoint.
# 3. Trigger a recompute of fn since x and y weren't saved.
#    And depending on whether early stop is enabled, either stop at (2) or
#    continue running the function.
#    Because we are running backward with retain_graph=False, we clear x and y's
#    holders.
# 4. Don't save x since we are inside a checkpoint.
# 5. Calling backward triggers another recompute of fn. During recompute, we see
#    that x and y have already been cleared in the original graph as indicated
#    by holder=None. We skip over them. We still save x at (4) (since its holder
#    is still alive.)

_enable_checkpoint_early_stop: bool | None = None


@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    """Controls whether checkpoint should stop recomputation early.

    By default, non-reentrant checkpoint stops recomputation as soon as it
    has computed all needed Tensors. This context manager can be used to disable
    that feature if it is problematic for your specific application.

    This context manager only needs to be active when forward is run. It does
    not need to be active during backward.

    Example::

        >>> # xdoctest: +SKIP(failing)
        >>> message = "saved tensors default hooks are disabled"
        >>> with set_checkpoint_early_stop(False):
        ...     # Any checkpoint under this context manager will respect this
        ...     # context manager, even if its backward is performed outside.
        ...     out = checkpoint(fn, inputs)
        ...
        >>> out.backward()
    """
    global _enable_checkpoint_early_stop
    try:
        prev = _enable_checkpoint_early_stop
        _enable_checkpoint_early_stop = enable
        yield
    finally:
        _enable_checkpoint_early_stop = prev


class _Handle:
    pass


class _Holder:
    def __init__(self) -> None:
        self.handles: dict[int, _Handle | None] = {}


class _CheckpointFrame:
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn) -> None:
        self.recompute_fn = recompute_fn
        self.saved_args: List[Any] = []
        self.weak_holders: List[ReferenceType] = []
        # We store this as a weakkeydictionary so that in the case of a partial
        # backward, the entries in the dict are cleared alongside the Holder
        # which will be removed when the SavedVariable is cleared.
        self.recomputed: DefaultDict[
            int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]
        ] = defaultdict(weakref.WeakKeyDictionary)
        # We need both recomp_counter and recomputed since they can diverge
        # https://github.com/pytorch/pytorch/pull/90105#discussion_r1135889885
        self.recomp_counter: DefaultDict[int, int] = defaultdict(int)
        self.is_recomputed: DefaultDict[int, bool] = defaultdict(bool)

        # See Rule 5
        self.early_stop = early_stop

        # Debugging
        self.metadata_fn = metadata_fn
        self.unpack_error_cb = unpack_error_cb
        self.x_metadatas = []
        self.forward_completed = False
        self.ignore_saved_mismatch = False

    def save_inputs(self, *args):
        self.saved_args = [
            _make_saved_tensor(arg, is_output=False)
            if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]

    def get_inputs(self):
        return [
            arg.unpack() if isinstance(arg, SavedTensor) else arg
            for arg in self.saved_args
        ]

    def check_recomputed_tensors_match(self, gid) -> None:
        if self.ignore_saved_mismatch:
            # TODO: we can probably make this check stricter by checking that
            #       the metadata of the first tensors still match.
            return
        # NOTE [ Error handling for checkpoint ]
        #
        # At a high level, we need to check that the tensors saved
        # during original forward matches tensors saved during recompute
        # This means handling 3 cases:
        #
        # 1. During recompute, more tensors were saved.
        #
        #    Usually this is hidden due to the StopRecomputationError
        #    but if early stop is not enabled, or we would have errored
        #    anyway because there aren't enough weak_holders. But we
        #    do want to have a nice error. See the _recomputation_hook
        #    for details.
        if not len(self.weak_holders) == self.recomp_counter[gid]:
            # 2. During recompute, fewer tensors were saved
            #
            # We know that every time we save something during original forward
            # we append to weak_holder, and every time we save a tensor
            # during recompute we increment recompute_counter.
            raise CheckpointError(
                "torch.utils.checkpoint: A different number of tensors was saved "
                "during the original forward and recomputation.\n"
                f"Number of tensors saved during forward: {len(self.weak_holders)}\n"
                f"Number of tensors saved during recomputation: {self.recomp_counter[gid]}.\n"
                f"{_debug_tip_msg}"
            )

        # 3. During recompute, the same tensors were saved, but they
        #    have different metadata
        nb_meta_different = []
        for idx, weak_holder in enumerate(self.weak_holders):
            holder = weak_holder()
            if holder is None:
                continue
            # We've seen all holders since we iterate over them in order
            # For every holder that is still alive now, it must've been
            # alive when we saw it during recompute, therefore, the
            # gid must be set.
            _internal_assert(gid in holder.handles)
            # We know this is the first unpack, so it couldn't have been set
            # to None yet.
            _internal_assert(holder.handles[gid] is not None)
            # We always set these together in the recomputation hook
            _internal_assert(holder.handles[gid] in self.recomputed[gid])
            # see pack hook, x_metadata is 1:1 with weak_holders.
            x_meta = self.x_metadatas[idx]
            recomputed_x = self.recomputed[gid][holder.handles[gid]]
            if x_meta != self.metadata_fn(recomputed_x):
                nb_meta_different.append((idx, x_meta, self.metadata_fn(recomputed_x)))

        if len(nb_meta_different) > 0:
            mismatched_tensors = ""
            for idx, x_meta, recomputed_meta in nb_meta_different:
                mismatched_tensors += (
                    f"tensor at position {idx}:\n"
                    f"saved metadata: {x_meta}\n"
                    f"recomputed metadata: {recomputed_meta}\n"
                )
            raise CheckpointError(
                "torch.utils.checkpoint: Recomputed values for the following tensors "
                "have different metadata than during the forward pass.\n"
                f"{mismatched_tensors}.\n"
                f"{_debug_tip_msg}"
            )


_debug_tip_msg = """
Tip: To see a more detailed error message, either pass `debug=True` to
`torch.utils.checkpoint.checkpoint(...)` or wrap the code block
with `with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):` to
enable checkpoint‑debug mode globally.

If this error occurs under torch.compile with automatic_dynamic_shapes enabled,
it may be because the recomputation selected a different compiled graph than the
forward pass (e.g., a dynamic graph instead of the original static graph).
To fix this, either:
  - Use torch._dynamo.mark_dynamic() to explicitly mark varying dimensions as
    dynamic upfront, avoiding the static-to-dynamic transition.
  - Call torch._C._dynamo.eval_frame._set_lru_cache(False) to disable LRU cache
    reordering, which can change which graph is checked first between forward
    and recompute.
See https://github.com/pytorch/pytorch/issues/166926 for more details and
workaround examples.
"""


_checkpoint_error_template = """ \
An error happened while unpacking tensors; dumping logs of latest computation
because you passed `debug=True` to `torch.utils.checkpoint.checkpoint()`.
Scroll all the way down for guidance on how to navigate these logs.

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        1. Stack traces of the operators that ran in the original forward     |
+------------------------------------------------------------------------------+

{forward_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|        2. Stack traces of the operators that ran during recomputation        |
+------------------------------------------------------------------------------+

{recompute_traces}
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
|       3. Log of operators in the original forward and recomputation          |
+------------------------------------------------------------------------------+
(Scroll up to correlate stack traces with each operation listed below. This
 helps identify their source in the code.)

IMPORTANT: Differences in "detach" calls between the original forward and the
           recomputation are expected. They are introduced by the checkpointing
           mechanism and can be ignored.

Operations executed during the original forward:

{forward_ops}

Operations executed during recomputation:

{recompute_ops}

+------------------------------------------------------------------------------+
 ERROR: Detected non-determinism while running activation checkpointing

 You are seeing this error because you passed `debug=True` to checkpoint and
 tensors to be saved during the original forward and differ between those saved
 during recomputation. This can happen if different operators were ran in the
 original forward and in the recomputation.

 To identify where the mismatch may be coming from, you can do the following:

 1) Compare the operators ran during original forward and recomputation to
    see where they differ. These operators are printed above in the order they
    were executed.

 2) Review the stack trace for each operator to locate its invocation source.
    Each operator's stack trace is printed in their execution order.

 Note that the logs can be quite long. Here's how they are structured:
 (Tip: you can Ctrl-f for these headers)

 1. Stack traces of the operators that ran in the original forward
 2. Stack traces of the operators that ran during recomputation
 3. Log of operators in the original forward and recomputation
 4. Error message                                             <--- You are here
--------------------------------------------------------------------------------
"""

class CheckpointError(RuntimeError):
    pass


def _get_debug_context_and_cb() -> Tuple[Callable[[], Any], Callable[[CheckpointError], None]]:
    # This function returns the context_fn and error_cb to be used by the
    # checkpointing mechanism. error_cb is invoked when an error is detected
    # during unpack.

    cpp_tb = platform.machine() in ('x86_64', 'aarch64', 'arm64') and platform.system() == 'Linux'

    class CaptureLogs:
        def __init__(self) -> None:
            self.logs = None
            self.tbs = None

        def get_context_manager(self):
            @contextlib.contextmanager
            def logging_mode():
                with LoggingTensorMode(), \
                     capture_logs(True, python_tb=True, script_tb=True, cpp_tb=cpp_tb) as logs_and_tb:
                    self.logs, self.tbs = logs_and_tb
                    yield logs_and_tb
            return logging_mode()

    capture_logs_fwd = CaptureLogs()
    capture_logs_recompute = CaptureLogs()

    def unpack_error_cb(e: CheckpointError) -> NoReturn:
        def get_str_tb(label, capture_logs):
            out = ""
            total_len = len(capture_logs.logs)
            for i, (log, tb) in enumerate(zip(capture_logs.logs, capture_logs.tbs, strict=False)):
                out += f"{log}   ({i + 1} of {total_len} in {label})\n\n"
                found_torch_dispatch = False
                for line in tb:
                    # Start printing stack trace only after __torch_dispatch__ is found
                    is_torch_dispatch = line['name'] == '__torch_dispatch__'
                    if not found_torch_dispatch and not is_torch_dispatch:
                        continue
                    elif is_torch_dispatch:
                        found_torch_dispatch = True
                        continue
                    out += f"{line['filename']}:{line['line']}:{line['name']}\n"
                out += "\n\n"
            return out
        if capture_logs_fwd.logs is None:
            raise AssertionError("capture_logs_fwd.logs is None")
        if capture_logs_recompute.logs is None:
            raise AssertionError("capture_logs_recompute.logs is None")
        raise CheckpointError(
            _checkpoint_error_template.format(
                forward_traces=get_str_tb("original", capture_logs_fwd),
                recompute_traces=get_str_tb("recompute", capture_logs_recompute),
                forward_ops="\n".join(capture_logs_fwd.logs),
                recompute_ops="\n".join(capture_logs_recompute.logs)
            )
        ) from e

    def context_fn():
        return capture_logs_fwd.get_context_manager(), capture_logs_recompute.get_context_manager()

    return context_fn, unpack_error_cb

def _default_meta_extractor(x: torch.Tensor) -> Dict[str, Any]:
    # These properties are fast to check, easy to understand
    return {
        "shape": x.shape,
        "dtype": x.dtype,
        "device": x.device
    }

_allowed_determinism_checks_to_fns: Dict[str, Callable[[torch.Tensor], Any]] = {
    _DEFAULT_DETERMINISM_MODE: _default_meta_extractor,
    "none": lambda _: None,
}

# See Rule 5
class _StopRecomputationError(Exception):
    pass


def _current_user_saved_tensors_hooks():
    # The (pack, unpack) pair a SavedVariable created right now would use,
    # skipping checkpoint-internal hooks, or None. Checkpoint-internal hooks
    # record the pair that was visible when they were pushed so that nested
    # checkpoints resolve to the user's hooks, not each other's.
    hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
    while hooks is not None and getattr(hooks[0], "_checkpoint_internal", False):
        hooks = hooks[0]._user_hooks  # pyrefly: ignore[missing-attribute]
    return hooks


class _checkpoint_internal_hook(torch.autograd.graph.saved_tensors_hooks):
    # _user_hooks must only exist while we are on the hooks TLS stack (the
    # only time _current_user_saved_tensors_hooks can reach it): the graph
    # retains pack_hook via SavedVariable in a way gc cannot traverse, so a
    # persistent attribute would give the graph a strong ref to user hooks,
    # leaking uncollectably whenever those hooks reach back to the graph.
    def __enter__(self):
        self.pack_hook._user_hooks = _current_user_saved_tensors_hooks()  # pyrefly: ignore[missing-attribute]
        return super().__enter__()

    def __exit__(self, *args):
        del self.pack_hook._user_hooks  # pyrefly: ignore[missing-attribute]
        return super().__exit__(*args)


class _recomputation_hook(_checkpoint_internal_hook):
    def __init__(self, target_frame_ref: ReferenceType, gid: GraphExecGroup | int) -> None:
        # Dynamo guards on WeakKeyDictionary internals are unstable here
        # (dict length/keys change every call), causing recompilation storms.
        # with `.compile()` so we disable
        @torch._dynamo.disable
        def pack_hook(x):
            x = x.detach() if x.requires_grad else x
            target_frame = target_frame_ref()
            if target_frame is None:
                raise AssertionError("Internal error: target_frame reference is None")
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1

            if recomp_idx >= len(target_frame.weak_holders):
                if target_frame.early_stop:
                    raise AssertionError("Unexpected state: target_frame.early_stop is set")
                if not target_frame.forward_completed:
                    # We run into this case when early stop is not enabled and do
                    # grad within checkpoint.
                    # We need to set this flag, so we don't error out later when
                    # we check if the number of tensors saved during forward and
                    # recomputation match.
                    target_frame.ignore_saved_mismatch = True
                    return x
                raise CheckpointError(
                    "torch.utils.checkpoint: trying to save more tensors during "
                    "recomputation than during the original forward pass.\n"
                    f"{_debug_tip_msg}"
                )

            holder = target_frame.weak_holders[recomp_idx]()

            # This holder may have been cleared because someone may have called
            # backward within forward. If so, we don't need to save.
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x

            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                target_frame.weak_holders
            ):
                raise _StopRecomputationError
            # See Rule 6: [ retain_graph is True ] above
            return x

        def unpack_hook(x):
            # See Rule 6: [ retain_graph is True ] above for an example of when
            # the graph created during recomputation could be backwarded.
            return x

        pack_hook._checkpoint_internal = True  # type: ignore[attr-defined]
        super().__init__(pack_hook, unpack_hook)


# torch._disable_dynamo creates a reference cycle with decorated function
# This function is used to ensure that the decorated function does not have
# a closure, so that other objects aren't also kept alive.
# https://github.com/pytorch/pytorch/issues/154642
# Note: does not work when fn is compiled
@torch._disable_dynamo
def _run_fn_with_dynamo_disabled(fn, *args, **kwargs):
    return fn(*args, **kwargs)


class _checkpoint_hook(_checkpoint_internal_hook):
    def __init__(self, frame) -> None:
        def pack_hook(x):
            # See Rule 4 above
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # Save metadata to detect non-determinism
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            # First check if we're inside a GraphExecGroup context
            gid: GraphExecGroup | None | int = GraphExecGroup._get_current_group()
            if gid is None:
                # Fallback to using the current graph task id
                gid = torch._C._current_graph_task_id()
                if gid == -1:
                    # generate a temporary id if we trigger unpack outside of a backward call
                    gid = int(uuid.uuid4())

            if not frame.is_recomputed[gid]:
                args = frame.get_inputs()

                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        # See Note: [compiled autograd and checkpoint unpack hook]
                        _run_fn_with_dynamo_disabled(frame.recompute_fn, *args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)

            _internal_assert(gid in holder.handles)

            if holder.handles[gid] is None:
                extra = ""
                if torch._C._get_graph_exec_group() is not None:
                    extra = (
                        "Performing two backward calls that overlap (i.e. require the same "
                        "saved activation in order to compute gradients) is not allowed while "
                        "under the torch.utils.checkpoint.GraphExecGroup context. "
                    )
                raise CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    f"unpacked once. {extra}If you are calling ctx.saved_tensors in backward, make sure "
                    "to do so only once. Otherwise please open an issue with details on your use case."
                )
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        pack_hook._checkpoint_internal = True  # type: ignore[attr-defined]
        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)


def _is_compiling(func, args, kwargs):
    # Check if we are under AOTAutograd tracing or export tracing
    # Checking that a proxy mode is active should always do what we want
    if torch.compiler._is_non_strict_tracing():
        return False
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY) is not None


class _VersionWrapper:
    # Check that cached tensors are not mutated.
    def __init__(self, val, unpack_hook=None) -> None:
        self.val: torch.Tensor | Any = val
        self.version: int | None = val._version if isinstance(val, torch.Tensor) else None
        # Set when val was produced by a user saved-tensors pack hook; applied
        # on retrieval so SAC-cached tensors honor hooks like save_on_cpu.
        self.unpack_hook = unpack_hook

    def get_val(self, allow_cache_entry_mutation):
        if self.version is not None and not allow_cache_entry_mutation:
            if self.val._version != self.version:
                # Can we give user a stack trace of where the mutation happened?
                raise RuntimeError(
                    "Tensor cached during selective activation checkpoint has been mutated"
                )
        return self.val if self.unpack_hook is None else self.unpack_hook(self.val)


def _detach_helper(x):
    # We detach for two separate reasons:
    # - For view ops, we need to ensure that when the tensor is returned from
    #   CachedDispatchMode, as_view sees that the AutogradMeta is nullptr
    # - Avoid reference cycles
    if isinstance(x, torch.Tensor):
        with torch._C._SetExcludeDispatchKeyGuard(torch._C.DispatchKey.ADInplaceOrView, False):
            # Ensure that view performed beneath autograd properly propagates
            # version counter. TODO: Use reentrant_dispatch instead of
            # manually manipulating dispatch keys. Using reentrant_dispatch
            # would respect inference_mode, though that is not relevant for
            # this case.
            x = x.detach()
    return x


class SelectiveCheckpointContext:
    """
    Context passed to policy function during selective checkpointing.

    This class is used to pass relevant metadata to the policy function during
    selective checkpointing.

    The policy function is only called during the forward pass. During
    recomputation, cached values are retrieved by index, so ``is_recompute``
    is deprecated and always ``False``.

    Example:
        >>> # xdoctest: +SKIP(stub)
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    print(ctx.op_output)
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    def __init__(self, *, is_recompute, op_output=None) -> None:
        self.is_recompute = is_recompute
        self.op_output = op_output


class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    The following policies are supported:

    - ``{MUST,PREFER}_SAVE``: The operation's output will be saved during the forward
      pass and will not be recomputed during the backward pass
    - ``{MUST,PREFER}_RECOMPUTE``: The operation's output will not be saved during the
      forward pass and will be recomputed during the backward pass
    - ``{MUST,PREFER}_CPU_OFFLOAD``: The operation's output will be saved during the
      forward pass, offloaded to CPU, and reloaded to GPU during the backward pass

    Use ``MUST_*`` over ``PREFER_*`` to indicate that the policy should not be overridden
    by other subsystems like `torch.compile`.

    .. note::
        A policy function that always returns ``PREFER_RECOMPUTE`` is
        equivalent to vanilla checkpointing.

        A policy function that returns ``PREFER_SAVE`` every op is
        NOT equivalent to not using checkpointing. Using such a policy would
        save additional tensors not limited to ones that are actually needed for
        gradient computation.
    """
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3
    MUST_CPU_OFFLOAD = 4
    PREFER_CPU_OFFLOAD = 5


def _policy_from_bool(b):
    # For backward compatibility
    return CheckpointPolicy.MUST_SAVE if b else CheckpointPolicy.PREFER_RECOMPUTE


SAC_IGNORED_OPS = {
    # AC inserts different number of detach during forward and recompute.
    torch.ops.aten.detach.default,
    # AC's determinism check invokes additional metadata ops during forward.
    # With subclasses involved, these metadata ops become dispatchable, this
    # can result in incorrectness if these ops are selected cached.
    torch.ops.prim.device.default,
} | set(torch._subclasses.functional_tensor.FunctionalTensor.metadata_fns)  # type: ignore[has-type]


def _sac_storage_key(func, args):
    """Compute the SAC storage key for a given op.

    For inductor_compiled_code, each compiled region gets its own FIFO queue
    keyed by the callable's unique idx.  Without this, all compiled regions
    share one queue and a cache-hit that skips a region during recompute
    causes the queue to return the wrong entry (see gh-175258).
    """
    from torch._higher_order_ops.wrap import (
        _resolve_inductor_callable,
        inductor_compiled_code as _inductor_compiled_code,
    )
    if func is _inductor_compiled_code and args:
        return (func, _resolve_inductor_callable(args[0]).idx)
    return func


class _CachingTorchDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls):
        return True

    # Used together with _CachedTorchDispatchMode to implement SAC.
    def __init__(self, policy_fn, storage, ac_graph_id=None) -> None:
        self.policy_fn = policy_fn
        self.storage = storage
        self.ac_graph_id = ac_graph_id
        # Tri-state opt-in for routing SAC-cached tensors through the
        # surrounding user saved_tensors_hooks, set by checkpoint() after this
        # mode is constructed (like ac_graph_id). None: legacy behavior (hooks
        # bypassed) plus a FutureWarning if user hooks are in scope; True/False:
        # opt into the new/old behavior explicitly and silence the warning.
        self.respect_saved_tensors_hooks: bool | None = None
        self.func_counter: Dict[Any, int] = defaultdict(int)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        is_compiling = _is_compiling(func, args, kwargs)

        if is_compiling:
            fx_traceback.current_meta["ac_graph_id"] = self.ac_graph_id
            fx_traceback.current_meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        proxy_mode = None
        graph_len_before = 0
        if is_compiling:
            from torch.fx.experimental.proxy_tensor import get_proxy_mode
            proxy_mode = get_proxy_mode()
            if proxy_mode is not None:
                graph_len_before = len(proxy_mode.tracer.graph.nodes)

        out = func(*args, **kwargs)

        key = _sac_storage_key(func, args)
        idx = self.func_counter[key]
        self.func_counter[key] += 1

        policy = self.policy_fn(SelectiveCheckpointContext(is_recompute=False, op_output=out),
                                func, *args, **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        if is_compiling:
            if proxy_mode is not None:
                graph = proxy_mode.tracer.graph
                num_new = len(graph.nodes) - graph_len_before
                for node in itertools.islice(reversed(graph.nodes), num_new):
                    node.meta["recompute"] = policy

        if policy in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE) or is_compiling:
            # SAC caches these tensors outside the autograd graph, bypassing
            # SavedVariable, so simulate pack/unpack with the user's
            # saved-tensors hooks (if any): hooks like save_on_cpu must see
            # every tensor retained until backward, whether autograd or SAC
            # holds it. Skipped under compile, where storage is a tracing
            # artifact rather than a real cache.
            #
            # Honoring the hooks is a BC break, so it is gated behind the
            # respect_saved_tensors_hooks opt-in on checkpoint(), which defaults
            # to the legacy behavior (hooks bypassed). When a user hook is
            # actually in scope we warn, since the default will flip.
            user_hooks = None if is_compiling else _current_user_saved_tensors_hooks()
            if user_hooks is not None:
                respects = self.respect_saved_tensors_hooks
                if respects is None:
                    warnings.warn(
                        "A saved_tensors_hooks context (e.g. save_on_cpu) is "
                        "active around a selective activation checkpoint region. "
                        "Tensors that SAC saves currently bypass these hooks, but "
                        "a future release will route them through the hooks like "
                        "any other saved tensor. Silence this warning and choose "
                        "the behavior explicitly by passing "
                        "respect_saved_tensors_hooks=True|False to "
                        "torch.utils.checkpoint.checkpoint.",
                        FutureWarning,
                        stacklevel=2,
                    )
                if not respects:
                    user_hooks = None

            def wrap(x):
                x = _detach_helper(x)
                if user_hooks is not None and isinstance(x, torch.Tensor):
                    return _VersionWrapper(user_hooks[0](x), unpack_hook=user_hooks[1])
                return _VersionWrapper(x)

            self.storage[key][idx] = tree_map(wrap, out)
        else:
            self.storage[key][idx] = _RECOMPUTE
        return out

_RECOMPUTE = object()
_CONSUMED = object()

_SAC_MISMATCH_MSG = (
    "This can happen if the operations in the checkpointed region are "
    "nondeterministic or depend on global state that changed between "
    "forward and backward."
)


class _CachedTorchDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls):
        return True

    # Used together with _CachingTorchDispatchMode to implement SAC.
    # policy_fn is accepted but ignored for BC (xformers subclasses this).
    def __init__(self, policy_fn, storage, allow_cache_entry_mutation=False) -> None:
        self.storage = storage
        self.allow_cache_entry_mutation = allow_cache_entry_mutation
        self.func_counter: Dict[Any, int] = defaultdict(int)

    def __enter__(self):
        # Reset so retain_graph=True hits "backward an extra time" not "index not found".
        self.func_counter.clear()
        return super().__enter__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        key = _sac_storage_key(func, args)
        idx = self.func_counter[key]
        self.func_counter[key] += 1

        func_storage = self.storage.get(key)
        if func_storage is None:
            raise RuntimeError(
                f"{func} encountered during backward but not found in "
                f"storage. {_SAC_MISMATCH_MSG}"
            )
        entry = func_storage.get(idx)
        if entry is None:
            raise RuntimeError(
                f"{func} invocation index {idx} encountered during backward "
                f"but not found in storage. {_SAC_MISMATCH_MSG}"
            )
        elif entry is _CONSUMED:
            raise RuntimeError(
                "Trying to backward an extra time. You are only allowed to backward once "
                "on any region computed under selective activation checkpoint."
            )
        elif entry is _RECOMPUTE:
            kwargs = {} if kwargs is None else kwargs
            return func(*args, **kwargs)
        else:
            func_storage[idx] = _CONSUMED
            return tree_map(lambda x: x.get_val(self.allow_cache_entry_mutation), entry)


def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    # NB: If grad_mode is disabled, checkpoint would not run forward under
    #     context_fn anyway, so proceed as usual.
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                _extra_msg = (
                    "Please update the OpOverloadPacket to a specific OpOverload."
                    "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                ) if isinstance(op, torch._ops.OpOverloadPacket) else ""
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op in policy_fn_or_list:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage: Dict[Any, Dict[int, Any]] = defaultdict(dict)
    return (
        _CachingTorchDispatchMode(policy_fn, storage),
        _CachedTorchDispatchMode(None, storage, allow_cache_entry_mutation),
    )

# NB: this helper wraps fn before calling checkpoint_impl. kwargs and
#     saving/restoring of global state is handled here.

def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    *args,
    respect_saved_tensors_hooks: bool | None = None,
    **kwargs,
):
    # BC shim: this retains the original positional-args signature (args/kwargs
    # trailing the config) for out-of-tree callers that invoke it positionally.
    # New in-tree callers should use _checkpoint_without_reentrant_generator_impl,
    # which takes args/kwargs as explicit tuple/dict parameters.
    return _checkpoint_without_reentrant_generator_impl(
        fn,
        args,
        kwargs,
        preserve_rng_state=preserve_rng_state,
        context_fn=context_fn,
        determinism_check=determinism_check,
        debug=debug,
        early_stop=early_stop,
        respect_saved_tensors_hooks=respect_saved_tensors_hooks,
    )


def _checkpoint_without_reentrant_generator_impl(
    fn,
    args,
    kwargs,
    *,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    respect_saved_tensors_hooks: bool | None = None,
):
    """Checkpointing without reentrant autograd.

    Args:
        fn: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        args: Tuple of arguments to pass in to the given ``function``.
        kwargs: Dict of keyword arguments to pass into the given ``function``.
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation.
        early_stop(bool, optional): If ``True``, non-reentrant checkpoint stops
            recomputation as soon as it has computed all needed Tensors. Can be
            overridden globally using :func:`set_checkpoint_early_stop` context
            manager. Default: ``True``.
    """
    unpack_error_cb = None

    if _checkpoint_debug_enabled if _checkpoint_debug_enabled is not None else debug:
        if context_fn is not noop_context_fn:
            raise ValueError(
                "debug=True is incompatible with non-default context_fn"
            )
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device_type = _infer_device_type(*args)
    device_module = _get_device_module(device_type)
    forward_context, recompute_context = context_fn()
    # SAC caches some tensors outside the autograd graph; this tells its caching
    # mode whether to route them through the user's saved_tensors_hooks. Set by
    # attribute after construction, like ac_graph_id in the compile path. No-op
    # for context_fns that are not SAC (e.g. the default), which cache nothing.
    if isinstance(forward_context, _CachingTorchDispatchMode):
        forward_context.respect_saved_tensors_hooks = respect_saved_tensors_hooks
    if _is_compiling(fn, args, kwargs) and context_fn is not noop_context_fn:
        if (
            not isinstance(forward_context, TorchDispatchMode)
            or not isinstance(recompute_context, TorchDispatchMode)
        ):
            raise AssertionError(
                "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` "
                "must generate a tuple of two `TorchDispatchMode`s."
            )
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device_type)

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args)

    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._device import DeviceContext

    # recompute_fn should respect the device context of the original forward.
    # Capture the device, not the mode instance: re-entering the captured
    # DeviceContext during recompute (possibly on the backward thread) corrupts
    # its exit state and leaks it. A fresh context avoids that.
    device_ctx = next(
        (
            DeviceContext(mode.device)
            for mode in reversed(_get_current_function_mode_stack())
            if isinstance(mode, DeviceContext)
        ),
        contextlib.nullcontext(),
    )
    error_on_nested_fx_trace = torch._dynamo.config.error_on_nested_fx_trace
    is_non_strict_tracing = torch.compiler._is_non_strict_tracing()
    is_fx_tracing = torch.fx._symbolic_trace._get_is_fx_tracing()

    def recompute_fn(*args) -> None:
        # This will be called later during recomputation. This wrapping enables
        # the necessary global state to be captured.
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=preserve_rng_state, device_type=device_type
        ):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states, device_type=device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=device_type, **device_autocast_kwargs
            ) if torch.amp.is_autocast_available(device_type) else contextlib.nullcontext()
            nested_fx_trace_ctx = (
                torch._dynamo.config.patch(
                    error_on_nested_fx_trace=error_on_nested_fx_trace
                )
                if is_non_strict_tracing
                else contextlib.nullcontext()
            )
            with (
                device_autocast_ctx,
                torch.amp.autocast("cpu", **cpu_autocast_kwargs),
                recompute_context,
                device_ctx,
                nested_fx_trace_ctx,
                torch.fx._symbolic_trace._is_fx_tracing_context(is_fx_tracing),
            ):  # type: ignore[attr-defined]
                fn(*args, **kwargs)

    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop if _enable_checkpoint_early_stop is not None else early_stop,
        unpack_error_cb,
        metadata_fn
    )

    if not torch.is_grad_enabled():
        yield
        return

    if (
        isinstance(forward_context, _CachingTorchDispatchMode)
        and torch._C._autograd._top_saved_tensors_default_hooks(False) is not None
    ):
        torch._C._log_api_usage_once(
            "torch.utils.checkpoint.sac_with_saved_tensors_hooks"
        )

    new_frame.save_inputs(*args)

    forward_context_suppressed_exc = False
    with _checkpoint_hook(new_frame), forward_context:
        try:
            yield
        except BaseException:
            forward_context_suppressed_exc = True
            raise
    if forward_context_suppressed_exc:
        raise RuntimeError(
            "torch.utils.checkpoint: the forward context provided by "
            "context_fn suppressed an exception raised during the "
            "checkpointed forward. This is not supported because checkpoint "
            "cannot return a value for a failed forward."
        )
    new_frame.forward_completed = True

    if getattr(device_module, "_initialized", False) and \
       preserve_rng_state and not had_device_in_fwd:  # type: ignore[possibly-undefined]
        # Device was not initialized before running the forward, so we didn't
        # stash the device state.
        raise RuntimeError(
            "PyTorch's device state was initialized in the forward pass "
            "of a Checkpoint, which is not allowed. Please open an issue "
            "if you need this feature."
        )

    return


class GraphExecGroup:
    """Any checkpointed regions encountered by backward under the same instance
    of this context manager will trigger recompute at most once, even if
    there are multiple calls to backward.

    Backward calls under the same instance of this context manager must execute
    over non-overlapping regions of the backward graph even if retain_graph=True.
    In particular, any two backward call cannot use the same saved activation for
    gradient computation.

    .. note::
        This context manager only affects checkpoint with use_reentrant=False, and
        is a no-op otherwise.
    """

    def __enter__(self) -> Self:
        if torch._C._get_graph_exec_group() is not None:
            raise RuntimeError(
                "GraphExecGroup contexts cannot be nested. "
                f"Already inside group {torch._C._get_graph_exec_group()}"
            )
        torch._C._set_graph_exec_group(self)
        return self

    def __exit__(self, *args: object) -> None:
        torch._C._set_graph_exec_group(None)

    @classmethod
    def _get_current_group(cls) -> GraphExecGroup | None:
        # Private API to be used by utils like AC
        return torch._C._get_graph_exec_group()


# Note: [compiled autograd and checkpoint unpack hook]
# When tracing via compiled autograd, this hook will be visible to the
# compiler if the forward of this checkpointed region ran in eager.
# If the forward had ran under compile, it would have been wrapped in a
# higher order op. See Note: [torch.compile and checkpoint].
#
# Since we run the recomputation hook under a enable_grad context,
# AOTDispatch will trace a joint graph for this hook, and may
# save different activations than in eager. This conflicts with the
# strict activation count checks in `frame.check_recomputed_tensors_match`.
# So, we disable this hook to force it to recompute eager checkpointed regions
# in eager. This could be removed if we can disable the partitioner for this
# graph segment.
<|endoftext|># mypy: allow-untyped-defs
import argparse
import glob
import os
from pathlib import Path
from zipfile import ZipFile


# Exclude some standard library modules to:
# 1. Slim down the final zipped file size
# 2. Remove functionality we don't want to support.
DENY_LIST = [
    # Interface to unix databases
    "dbm",
    # ncurses bindings (terminal interfaces)
    "curses",
    # Tcl/Tk GUI
    "tkinter",
    "tkinter",
    # Tests for the standard library
    "test",
    "tests",
    "idle_test",
    "__phello__.foo.py",
    # importlib frozen modules. These are already baked into CPython.
    "_bootstrap.py",
    "_bootstrap_external.py",
]

strip_file_dir = ""


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def write_to_zip(file_path, strip_file_path, zf, prepend_str="") -> None:
    stripped_file_path = prepend_str + remove_prefix(file_path, strip_file_dir + "/")
    path = Path(stripped_file_path)
    if path.name in DENY_LIST:
        return
    zf.write(file_path, stripped_file_path)


def main() -> None:
    global strip_file_dir
    parser = argparse.ArgumentParser(description="Zip py source")
    parser.add_argument("paths", nargs="*", help="Paths to zip.")
    parser.add_argument(
        "--install-dir", "--install_dir", help="Root directory for all output files"
    )
    parser.add_argument(
        "--strip-dir",
        "--strip_dir",
        help="The absolute directory we want to remove from zip",
    )
    parser.add_argument(
        "--prepend-str",
        "--prepend_str",
        help="A string to prepend onto all paths of a file in the zip",
        default="",
    )
    parser.add_argument("--zip-name", "--zip_name", help="Output zip name")

    args = parser.parse_args()

    zip_file_name = args.install_dir + "/" + args.zip_name
    strip_file_dir = args.strip_dir
    prepend_str = args.prepend_str
    with ZipFile(zip_file_name, mode="w") as zf:
        for p in sorted(args.paths):
            if os.path.isdir(p):
                files = glob.glob(p + "/**/*.py", recursive=True)
                for file_path in sorted(files):
                    # strip the absolute path
                    write_to_zip(
                        file_path, strip_file_dir + "/", zf, prepend_str=prepend_str
                    )
            else:
                write_to_zip(p, strip_file_dir + "/", zf, prepend_str=prepend_str)


if __name__ == "__main__":
    main()  # pragma: no cover
<|endoftext|># mypy: allow-untyped-defs
import sys
import types

import torch


class _Deterministic(types.ModuleType):
    @property
    def fill_uninitialized_memory(self):
        """
        Whether to fill uninitialized memory with a known value when
        :meth:`torch.use_deterministic_algorithms()` is set to ``True``.
        """
        return torch._C._get_deterministic_fill_uninitialized_memory()

    @fill_uninitialized_memory.setter
    def fill_uninitialized_memory(self, mode):
        return torch._C._set_deterministic_fill_uninitialized_memory(mode)


sys.modules[__name__].__class__ = _Deterministic
<|endoftext|># mypy: allow-untyped-defs

# Unlike the rest of the PyTorch this file must be python2 compliant.
# This script outputs relevant system environment info
# Run it with `python collect_env.py` or `python -m torch.utils.collect_env`
import datetime
import json
import locale
import os
import re
import subprocess
import sys
from collections import namedtuple
from typing import cast as _cast, Dict as _Dict


try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "torch_version",
        "is_debug_build",
        "cuda_compiled_version",
        "gcc_version",
        "clang_version",
        "cmake_version",
        "os",
        "libc_version",
        "python_version",
        "python_platform",
        "is_cuda_available",
        "cuda_runtime_version",
        "cuda_module_loading",
        "nvidia_driver_version",
        "nvidia_gpu_models",
        "cudnn_version",
        "is_xpu_available",
        "pip_version",  # 'pip' or 'pip3'
        "pip_packages",
        "conda_packages",
        "hip_compiled_version",
        "hip_runtime_version",
        "miopen_runtime_version",
        "caching_allocator_config",
        "is_xnnpack_available",
        "cpu_info",
    ],
)

COMMON_PATTERNS = [
    "torch",
    "numpy",
    "triton",
    "optree",
]

NVIDIA_PATTERNS = [
    "cuda-cudart",
    "cuda-cupti",
    "cuda-libraries",
    "cuda-opencl",
    "cuda-nvrtc",
    "cuda-runtime",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvjitlink",
    "nvtx",
]

ONEAPI_PATTERNS = [
    "dpcpp-cpp-rt",
    "intel-cmplr-lib-rt",
    "intel-cmplr-lib-ur",
    "intel-cmplr-lic-rt",
    "intel-opencl-rt",
    "intel-sycl-rt",
    "mkl",
    "onemkl-sycl-blas",
    "onemkl-sycl-dft",
    "onemkl-sycl-lapack",
    "onemkl-sycl-rng",
    "onemkl-sycl-sparse",
    "intel-openmp",
    "tbb",
    "impi-rt",
    "impi-devel",
    "oneccl",
    "oneccl-devel",
    "intel-pti",
    "umf",
    "tcmlib",
]

CONDA_PATTERNS = [
    "cudatoolkit",
    "soumith",
    "mkl",
    "magma",
]

PIP_PATTERNS = [
    "mypy",
    "flake8",
    "onnx",
]


def run(command):
    """Return (return-code, stdout, stderr)."""
    shell = type(command) is str
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    if get_platform() == "win32":
        enc = "oem"
    else:
        enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Run command using run_lambda; reads and returns entire output if rc is 0."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Run command using run_lambda, returns the first regex match if it exists."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def run_and_return_first_line(run_lambda, command):
    """Run command using run_lambda and returns first line if output is not empty."""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split("\n")[0]


def get_conda_packages(run_lambda, patterns=None):
    if patterns is None:
        patterns = CONDA_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS + ONEAPI_PATTERNS
    conda = os.environ.get("CONDA_EXE", "conda")
    out = run_and_read_all(run_lambda, "{} list".format(conda))
    if out is None:
        return out

    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#") and any(name in line for name in patterns)
    )


def get_gcc_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )


def get_cmake_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_nvidia_driver_version(run_lambda):
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(
            run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]"
        )
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")


def get_gpu_info(run_lambda):
    if get_platform() == "darwin" or (
        TORCH_AVAILABLE
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.version.hip is not None:
                prop = torch.cuda.get_device_properties(0)
                if hasattr(prop, "gcnArchName"):
                    gcnArch = " ({})".format(prop.gcnArchName)
                else:
                    gcnArch = "NoGCNArchNameOnOldPyTorch"
            else:
                gcnArch = ""
            return torch.cuda.get_device_name(None) + gcnArch
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)


def get_running_cuda_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_cudnn_version(run_lambda):
    """Return a list of libcudnn.so; it's hard to tell which one is being used."""
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        cuda_path = os.environ.get("CUDA_PATH", "%CUDA_PATH%")
        where_cmd = os.path.join(system_root, "System32", "where")
        cudnn_cmd = '{} /R "{}\\bin" cudnn*.dll'.format(where_cmd, cuda_path)
    elif get_platform() == "darwin":
        # CUDA libraries and drivers can be found in /usr/local/cuda/. See
        # https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-mac-os-x/index.html#installation
        # https://docs.nvidia.com/deeplearning/cudnn/installation/latest/
        # Use CUDNN_LIBRARY when cudnn library is installed elsewhere.
        cudnn_cmd = "ls /usr/local/cuda/lib/libcudnn*"
    else:
        cudnn_cmd = 'ldconfig -p | grep libcudnn | rev | cut -d" " -f1 | rev'
    rc, out, _ = run_lambda(cudnn_cmd)
    # find will return 1 if there are permission errors or if not found
    if len(out) == 0 or (rc != 1 and rc != 0):
        l = os.environ.get("CUDNN_LIBRARY")
        if l is not None and os.path.isfile(l):
            return os.path.realpath(l)
        return None
    files_set = set()
    for fn in out.split("\n"):
        fn = os.path.realpath(fn)  # eliminate symbolic links
        if os.path.isfile(fn):
            files_set.add(fn)
    if not files_set:
        return None
    # Alphabetize the result because the order is non-deterministic otherwise
    files = sorted(files_set)
    if len(files) == 1:
        return files[0]
    result = "\n".join(files)
    return "Probably one of the following:\n{}".format(result)


def get_nvidia_smi():
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        legacy_path = os.path.join(
            program_files_root, "NVIDIA Corporation", "NVSMI", smi
        )
        new_path = os.path.join(system_root, "System32", smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if os.path.exists(candidate_smi):
                smi = '"{}"'.format(candidate_smi)
                break
    return smi


def _detect_linux_pkg_manager():
    if get_platform() != "linux":
        return "N/A"
    for mgr_name in ["dpkg", "dnf", "yum", "zypper"]:
        rc, _, _ = run(f"which {mgr_name}")
        if rc == 0:
            return mgr_name
    return "N/A"


def get_linux_pkg_version(run_lambda, pkg_name):
    pkg_mgr = _detect_linux_pkg_manager()
    if pkg_mgr == "N/A":
        return "N/A"

    grep_version = {
        "dpkg": {
            "field_index": 2,
            "command": "dpkg -l | grep {}",
        },
        "dnf": {
            "field_index": 1,
            "command": "dnf list | grep {}",
        },
        "yum": {
            "field_index": 1,
            "command": "yum list | grep {}",
        },
        "zypper": {
            "field_index": 2,
            "command": "zypper info {} | grep Version",
        },
    }

    field_index: int = int(_cast(int, grep_version[pkg_mgr]["field_index"]))
    cmd: str = str(grep_version[pkg_mgr]["command"])
    cmd = cmd.format(pkg_name)
    ret = run_and_read_all(run_lambda, cmd)
    if ret is None or ret == "":
        return "N/A"
    lst = re.sub(" +", " ", ret).split(" ")
    if len(lst) <= field_index:
        return "N/A"
    return lst[field_index]


def get_intel_gpu_driver_version(run_lambda):
    lst = []
    platform = get_platform()
    if platform == "linux":
        pkgs = {  # type: ignore[var-annotated]
            "dpkg": {
                "intel-opencl-icd",
                "libze1",
                "level-zero",
            },
            "dnf": {
                "intel-opencl",
                "level-zero",
            },
            "yum": {
                "intel-opencl",
                "level-zero",
            },
            "zypper": {
                "intel-opencl",
                "level-zero",
            },
        }.get(_detect_linux_pkg_manager(), {})
        for pkg in pkgs:
            ver = get_linux_pkg_version(run_lambda, pkg)
            if ver != "N/A":
                lst.append(f"* {pkg}:\t{ver}")
    if platform in ["win32", "cygwin"]:
        txt = run_and_read_all(
            run_lambda,
            'powershell.exe "gwmi -Class Win32_PnpSignedDriver | where{$_.DeviceClass -eq \\"DISPLAY\\"\
            -and $_.Manufacturer -match \\"Intel\\"} | Select-Object -Property DeviceName,DriverVersion,DriverDate\
            | ConvertTo-Json"',
        )
        try:
            obj = json.loads(txt)
            if type(obj) is list:
                for o in obj:
                    lst.append(
                        f'* {o["DeviceName"]}: {o["DriverVersion"]} ({o["DriverDate"]})'
                    )
            else:
                lst.append(f'* {obj["DriverVersion"]} ({obj["DriverDate"]})')
        except ValueError as e:
            lst.append(txt)
            lst.append(str(e))
    return "\n".join(lst)


def get_intel_gpu_onboard(run_lambda):
    lst: list[str] = []
    platform = get_platform()
    if platform == "linux":
        txt = run_and_read_all(run_lambda, "xpu-smi discovery -j")
        if txt:
            try:
                obj = json.loads(txt)
                device_list = obj.get("device_list", [])
                if isinstance(device_list, list) and device_list:
                    lst.extend(f'* {device["device_name"]}' for device in device_list)
                else:
                    lst.append("N/A")
            except (ValueError, TypeError) as e:
                lst.append(txt)
                lst.append(str(e))
        else:
            lst.append("N/A")
    if platform in ["win32", "cygwin"]:
        txt = run_and_read_all(
            run_lambda,
            'powershell.exe "gwmi -Class Win32_PnpSignedDriver | where{$_.DeviceClass -eq \\"DISPLAY\\"\
            -and $_.Manufacturer -match \\"Intel\\"} | Select-Object -Property DeviceName | ConvertTo-Json"',
        )
        if txt:
            try:
                obj = json.loads(txt)
                if isinstance(obj, list) and obj:
                    lst.extend(f'* {device["DeviceName"]}' for device in obj)
                else:
                    lst.append(f'* {obj.get("DeviceName", "N/A")}')
            except ValueError as e:
                lst.append(txt)
                lst.append(str(e))
        else:
            lst.append("N/A")
    return "\n".join(lst)


def get_intel_gpu_detected(run_lambda):
    if not TORCH_AVAILABLE or not hasattr(torch, "xpu"):
        return "N/A"

    device_count = torch.xpu.device_count()
    if device_count == 0:
        return "N/A"

    devices = [
        f"* [{i}] {torch.xpu.get_device_properties(i)}" for i in range(device_count)
    ]
    return "\n".join(devices)


# example outputs of CPU infos
#  * linux
#    Architecture:            x86_64
#      CPU op-mode(s):        32-bit, 64-bit
#      Address sizes:         46 bits physical, 48 bits virtual
#      Byte Order:            Little Endian
#    CPU(s):                  128
#      On-line CPU(s) list:   0-127
#    Vendor ID:               GenuineIntel
#      Model name:            Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#        CPU family:          6
#        Model:               106
#        Thread(s) per core:  2
#        Core(s) per socket:  32
#        Socket(s):           2
#        Stepping:            6
#        BogoMIPS:            5799.78
#        Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr
#                             sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl
#                             xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq monitor ssse3 fma cx16
#                             pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand
#                             hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced
#                             fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap
#                             avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1
#                             xsaves wbnoinvd ida arat avx512vbmi pku ospke avx512_vbmi2 gfni vaes vpclmulqdq
#                             avx512_vnni avx512_bitalg tme avx512_vpopcntdq rdpid md_clear flush_l1d arch_capabilities
#    Virtualization features:
#      Hypervisor vendor:     KVM
#      Virtualization type:   full
#    Caches (sum of all):
#      L1d:                   3 MiB (64 instances)
#      L1i:                   2 MiB (64 instances)
#      L2:                    80 MiB (64 instances)
#      L3:                    108 MiB (2 instances)
#    NUMA:
#      NUMA node(s):          2
#      NUMA node0 CPU(s):     0-31,64-95
#      NUMA node1 CPU(s):     32-63,96-127
#    Vulnerabilities:
#      Itlb multihit:         Not affected
#      L1tf:                  Not affected
#      Mds:                   Not affected
#      Meltdown:              Not affected
#      Mmio stale data:       Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
#      Retbleed:              Not affected
#      Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl and seccomp
#      Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
#      Spectre v2:            Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
#      Srbds:                 Not affected
#      Tsx async abort:       Not affected
#  * win32
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU0
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142
#
#    Architecture=9
#    CurrentClockSpeed=2900
#    DeviceID=CPU1
#    Family=179
#    L2CacheSize=40960
#    L2CacheSpeed=
#    Manufacturer=GenuineIntel
#    MaxClockSpeed=2900
#    Name=Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz
#    ProcessorType=3
#    Revision=27142


def get_cpu_info(run_lambda):
    rc, out, err = 0, "", ""
    if get_platform() == "linux":
        rc, out, err = run_lambda("lscpu")
    elif get_platform() == "win32":
        rc, out, err = run_lambda(
            'powershell.exe "gwmi -Class Win32_Processor | Select-Object -Property Name,Manufacturer,Family,\
            Architecture,ProcessorType,DeviceID,CurrentClockSpeed,MaxClockSpeed,L2CacheSize,L2CacheSpeed,Revision\
            | ConvertTo-Json"'
        )
        if rc == 0:
            lst = []
            try:
                obj = json.loads(out)
                if type(obj) is list:
                    for o in obj:
                        lst.append("----------------------")
                        lst.extend([f"{k}: {v}" for (k, v) in o.items()])
                else:
                    lst.extend([f"{k}: {v}" for (k, v) in obj.items()])
            except ValueError as e:
                lst.append(out)
                lst.append(str(e))
            out = "\n".join(lst)
    elif get_platform() == "darwin":
        rc, out, err = run_lambda("sysctl -n machdep.cpu.brand_string")
    cpu_info = "None"
    if rc == 0:
        cpu_info = out
    else:
        cpu_info = err
    return cpu_info


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform


def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda):
    ret = run_and_read_all(
        run_lambda,
        'powershell.exe "gwmi -Class Win32_OperatingSystem | Select-Object -Property Caption,\
        OSArchitecture,Version | ConvertTo-Json"',
    )
    try:
        obj = json.loads(ret)
        ret = f'{obj["Caption"]} ({obj["Version"]} {obj["OSArchitecture"]})'
    except ValueError as e:
        ret += f"\n{str(e)}"
    return ret


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "lsb_release -a", r"Description:\t(.*)"
    )


def check_release_file(run_lambda):
    return run_and_parse_first_match(
        run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
    )


def get_os(run_lambda):
    from platform import machine

    platform = get_platform()

    if platform in ["win32", "cygwin"]:
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return "macOS {} ({})".format(version, machine())

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return "{} ({})".format(desc, machine())

        return "{} ({})".format(platform, machine())

    # Unknown platform
    return platform


def get_python_platform():
    import platform

    return platform.platform()


def get_libc_version():
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def get_pip_packages(run_lambda, patterns=None):
    """Return `pip list` output. Note: will also find conda-installed pytorch and numpy packages."""
    if patterns is None:
        patterns = PIP_PATTERNS + COMMON_PATTERNS + NVIDIA_PATTERNS + ONEAPI_PATTERNS

    pip_version = "pip3" if sys.version_info.major == 3 else "pip"

    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    # People generally have pip as `pip` or `pip3`
    # But here it is invoked as `python -mpip`
    out = run_and_read_all(
        run_lambda, [sys.executable, "-mpip", "list", "--format=freeze"]
    )
    if out is None:
        return pip_version, out

    filtered_out = "\n".join(
        line for line in out.splitlines() if any(name in line for name in patterns)
    )

    return pip_version, filtered_out


def get_cachingallocator_config() -> _Dict[str, str]:
    """Return the caching allocator configuration from environment variables.
    """
    # pyrefly: ignore [bad-return]
    return {
        var: os.environ.get(var)
        for var in (
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
        )
        if os.environ.get(var)
    }


def get_cuda_module_loading_config():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.init()
        config = os.environ.get("CUDA_MODULE_LOADING", "")
        return config
    else:
        return "N/A"


def is_xnnpack_available():
    if TORCH_AVAILABLE:
        import torch.backends.xnnpack

        return str(torch.backends.xnnpack.enabled)  # type: ignore[attr-defined]
    else:
        return "N/A"


def get_env_info():
    """
    Collects environment information to aid in debugging.

    The returned environment information contains details on torch version, is debug build
    or not, cuda compiled version, gcc version, clang version, cmake version, operating
    system, libc version, python version, python platform, CUDA availability, CUDA
    runtime version, CUDA module loading config, GPU model and configuration, Nvidia
    driver version, cuDNN version, pip version and versions of relevant pip and
    conda packages, HIP runtime version, MIOpen runtime version,
    Caching allocator config, XNNPACK availability and CPU information.

    Returns:
        SystemEnv (namedtuple): A tuple containing various environment details
            and system information.
    """
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        xpu_available_str = str(torch.xpu.is_available())
        if torch.xpu.is_available():
            xpu_available_str = (
                f"{xpu_available_str}\n"
                + f"XPU used to build PyTorch: {torch.version.xpu}\n"
                + f"Intel GPU driver version:\n{get_intel_gpu_driver_version(run_lambda)}\n"
                + f"Intel GPU models onboard:\n{get_intel_gpu_onboard(run_lambda)}\n"
                + f"Intel GPU models detected:\n{get_intel_gpu_detected(run_lambda)}"
            )
        if (
            not hasattr(torch.version, "hip") or torch.version.hip is None
        ):  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
        else:  # HIP version

            def get_version_or_na(cfg, prefix):
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
                return _lst[0] if _lst else "N/A"

            cfg = torch._C._show_config().split("\n")
            hip_runtime_version = get_version_or_na(cfg, "HIP Runtime")
            miopen_runtime_version = get_version_or_na(cfg, "MIOpen")
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = xpu_available_str = "N/A"  # type: ignore[assignment]
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"

    sys_version = sys.version.replace("\n", " ")

    conda_packages = get_conda_packages(run_lambda)

    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version="{} ({}-bit runtime)".format(
            sys_version, sys.maxsize.bit_length() + 1
        ),
        python_platform=get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        cuda_module_loading=get_cuda_module_loading_config(),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        is_xpu_available=xpu_available_str,
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=conda_packages,
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        caching_allocator_config=get_cachingallocator_config(),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run_lambda),
    )


env_info_fmt = """
PyTorch version: {torch_version}
Is debug build: {is_debug_build}
CUDA used to build PyTorch: {cuda_compiled_version}
ROCM used to build PyTorch: {hip_compiled_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
CUDA_MODULE_LOADING set to: {cuda_module_loading}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}
cuDNN version: {cudnn_version}
Is XPU available: {is_xpu_available}
HIP runtime version: {hip_runtime_version}
MIOpen runtime version: {miopen_runtime_version}
Is XNNPACK available: {is_xnnpack_available}
Caching allocator config: {caching_allocator_config}

CPU:
{cpu_info}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        for key in dct:
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        for key in dct:
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text, tag="[prepend]"):
        lines = text.split("\n")
        updated_lines = [tag + line for line in lines]
        return "\n".join(updated_lines)

    def replace_if_empty(text, replacement="No relevant packages"):
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string):
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return "\n{}\n".format(string)
        return string

    mutable_dict = envinfo._asdict()

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(
        envinfo.nvidia_gpu_models
    )

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
    ]
    all_cuda_fields = dynamic_cuda_fields + ["cudnn_version"]
    all_dynamic_cuda_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_cuda_fields
    )
    if (
        TORCH_AVAILABLE
        and not torch.cuda.is_available()
        and all_dynamic_cuda_fields_missing
    ):
        for field in all_cuda_fields:
            mutable_dict[field] = "No CUDA"
        if envinfo.cuda_compiled_version is None:
            mutable_dict["cuda_compiled_version"] = "None"

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])
    mutable_dict["conda_packages"] = replace_if_empty(mutable_dict["conda_packages"])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = prepend(
            mutable_dict["pip_packages"], "[{}] ".format(envinfo.pip_version)
        )
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = prepend(
            mutable_dict["conda_packages"], "[conda] "
        )
    mutable_dict["cpu_info"] = envinfo.cpu_info
    mutable_dict["caching_allocator_config"] = envinfo.caching_allocator_config
    if not envinfo.caching_allocator_config:
        mutable_dict["caching_allocator_config"] = "N/A"
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    """
    Returns a pretty string of environment information.

    This function retrieves environment information by calling the `get_env_info` function
    and then formats the information into a human-readable string. The retrieved environment
    information is listed in the document of `get_env_info`.
    This function is used in `python collect_env.py` that should be executed when reporting a bug.

    Returns:
        str: A pretty string of the environment information.
    """
    return pretty_str(get_env_info())


def main() -> None:
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if (
        TORCH_AVAILABLE
        and hasattr(torch, "utils")
        and hasattr(torch.utils, "_crash_handler")
    ):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [
                os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)
            ]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg = (
                "\n*** Detected a minidump at {} created on {}, ".format(
                    latest, creation_time
                )
                + "if this is related to your bug please include it when you file a report ***"
            )
            print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
<|endoftext|>from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    MutableSet,
    Reversible,
    Set as AbstractSet,
)
from typing import Any, cast, TypeVar


T = TypeVar("T", bound=Hashable)
T_co = TypeVar("T_co", bound=Hashable, covariant=True)

__all__ = ["OrderedSet"]


class OrderedSet(MutableSet[T], Reversible[T]):
    """
    Insertion ordered set, similar to OrderedDict.
    """

    __slots__ = ("_dict",)

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._dict = dict.fromkeys(iterable, None) if iterable is not None else {}

    @staticmethod
    def _from_dict(dict_inp: dict[T, None]) -> OrderedSet[T]:
        s: OrderedSet[T] = OrderedSet()
        s._dict = dict_inp
        return s

    #
    # Required overridden abstract methods
    #
    def __contains__(self, elem: object) -> bool:
        return elem in self._dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._dict)

    def add(self, elem: T) -> None:
        self._dict[elem] = None

    def discard(self, elem: T) -> None:
        self._dict.pop(elem, None)

    def clear(self) -> None:
        # overridden because MutableSet impl is slow
        self._dict.clear()

    # Unimplemented set() methods in _collections_abc.MutableSet

    @classmethod
    def _wrap_iter_in_set(cls, other: Any) -> Any:
        """
        Wrap non-Set Iterables in OrderedSets

        Some of the magic methods are more strict on input types than
        the public apis, so we need to wrap inputs in sets.
        """

        if not isinstance(other, AbstractSet) and isinstance(other, Iterable):
            return cls(other)
        else:
            return other

    def pop(self) -> T:
        if not self:
            raise KeyError("pop from an empty set")
        return self._dict.popitem()[0]

    def copy(self) -> OrderedSet[T]:
        return OrderedSet._from_dict(self._dict.copy())

    def difference(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self.copy()
        res.difference_update(*others)
        return res

    def difference_update(self, *others: Iterable[T]) -> None:
        for other in others:
            self -= other  # type: ignore[arg-type]

    def update(self, *others: Iterable[T]) -> None:
        for other in others:
            self |= other

    def intersection(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self.copy()
        for other in others:
            if other is not self:
                res &= other  # type: ignore[arg-type]
        return res

    def intersection_update(self, *others: Iterable[T]) -> None:
        for other in others:
            self &= other  # type: ignore[arg-type]

    def issubset(self, other: Iterable[T]) -> bool:
        return self <= self._wrap_iter_in_set(other)

    def issuperset(self, other: Iterable[T]) -> bool:
        return self >= self._wrap_iter_in_set(other)

    def symmetric_difference(self, other: Iterable[T]) -> OrderedSet[T]:
        return self ^ other  # type: ignore[operator]

    def symmetric_difference_update(self, other: Iterable[T]) -> None:
        self ^= other  # type: ignore[arg-type]

    def union(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self.copy()
        for other in others:
            if other is self:
                continue
            res |= other
        return res

    # Specify here for correct type inference, otherwise would
    # return AbstractSet[T]
    def __sub__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # following cpython set impl optimization
        if isinstance(other, OrderedSet) and (len(self) * 4) > len(other):
            out = self.copy()
            out -= other
            return out
        return cast(OrderedSet[T], super().__sub__(other))

    def __ior__(self, other: Iterable[T]) -> OrderedSet[T]:  # type: ignore[misc, override]   # noqa: PYI034
        if isinstance(other, OrderedSet):
            self._dict.update(other._dict)
            return self
        return super().__ior__(other)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self._dict == other._dict
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self._dict != other._dict
        return super().__ne__(other)

    def __or__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__or__(other))

    def __and__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # MutableSet impl will iterate over other, iter over smaller of two sets
        if isinstance(other, OrderedSet) and len(self) < len(other):
            # pyrefly: ignore [unsupported-operation, bad-return]
            return other & self
        return cast(OrderedSet[T], super().__and__(other))

    def __xor__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__xor__(other))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"

    def __getstate__(self) -> list[T]:
        return list(self._dict.keys())

    def __setstate__(self, state: list[T]) -> None:
        self._dict = dict.fromkeys(state, None)

    def __reduce__(self) -> tuple[type[OrderedSet[T]], tuple[list[T]]]:
        return (OrderedSet, (list(self),))
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs
from typing import Any, TypeVar, NamedTuple
from collections.abc import Callable, Sequence
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module


T = TypeVar("T")

MAX_RAW_TENSOR_SIZE = 16

class InflatableArg(NamedTuple):
    """Helper type for bundled inputs.

    'value' is the compressed/deflated input that is stored in the model. Value
    must be of the same type as the argument to the function that it is a deflated
    input for.

    'fmt' is a formattable code string that is executed to inflate the compressed data into
    the appropriate input. It can use 'value' as an input to the format str. It must result
    in a value of the same type as 'value'.

    'fmt_fn' is a formattable function code string that is executed to inflate the compressed
    data into the appropriate input. It must result in a value of the same type as 'value'.
    The function name should be the formattable part of the string.

    Note: Only top level InflatableArgs can be inflated. i.e. you cannot place
    an inflatable arg inside of some other structure. You should instead create
    an inflatable arg such that the fmt code string returns the full structure
    of your input.
    """

    value: Any
    fmt: str = "{}"
    fmt_fn: str = ""


def bundle_inputs(
        model: torch.jit.ScriptModule,
        inputs: Sequence[tuple[Any, ...]] | dict[Callable, Sequence[tuple[Any, ...]] | None] | None,
        info: list[str] | dict[Callable, list[str]] | None = None,
        *,
        _receive_inflate_expr: list[str] | None = None,
) -> torch.jit.ScriptModule:
    """Create and return a copy of the specified model with inputs attached.

    The original model is not mutated or changed in any way.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    If inputs is passed in as a list then the inputs will be bundled for 'forward'.
    If inputs is instead passed in as a map then all the methods specified in the map
    will have their corresponding inputs bundled. Info should match whatever type is
    chosen for the inputs.

    The returned model will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions will also be defined on the returned module:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        Alternatively if only bundling inputs for forward the map can be omitted and a singular list of inputs
        can be provided instead.

        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. Alternatively if only bundling inputs for forward the map can be omitted and
    a singular list of information can be provided instead. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")  # noqa: TRY002

    ignored_methods, ignored_attrs = _get_bundled_inputs_attributes_and_methods(model)
    clone = torch._C._hack_do_not_use_clone_module_with_class(  # type: ignore[attr-defined]
        model._c,
        ignored_methods,
        ignored_attrs,
    )

    # The above cloning function returns a torch._C.scriptmodule and we need a torch.jit.scriptmodule.
    # Fortunately there is a function in _recursive that does exactly that conversion.
    cloned_module = wrap_cpp_module(clone)
    if isinstance(inputs, dict):
        if not isinstance(info, dict) and info is not None:
            raise AssertionError("If inputs is a dict, info must be a dict or None")
        augment_many_model_functions_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    else:
        if not isinstance(info, list) and info is not None:
            raise AssertionError("If inputs is a list, info must be a list or None")
        augment_model_with_bundled_inputs(cloned_module, inputs, _receive_inflate_expr, info)
    return cloned_module

def augment_model_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: Sequence[tuple[Any, ...]] | None = None,
        _receive_inflate_expr: list[str] | None = None,  # For debugging.
        info: list[str] | None = None,  # Optional argument to provide info about forward or its inputs
        skip_size_check=False,
) -> None:
    """Add bundled sample inputs to a model for the forward function.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_forward`.
        If the user chooses this method inputs should be None

      - `inputs` is a list of inputs of form List[Tuple[Any, ...]]. A list of tuples where the elements
        of each tuple are the args that make up one input.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")  # noqa: TRY002

    forward: Callable = model.forward

    # Sometimes forward won't have a name attached so just in case
    if not hasattr(forward, "__name__"):
        forward.__name__ = 'forward'
    augment_many_model_functions_with_bundled_inputs(
        model,
        inputs={forward : inputs},
        _receive_inflate_expr=_receive_inflate_expr,
        info={forward : info} if info else None,
        skip_size_check=skip_size_check,
    )


def augment_many_model_functions_with_bundled_inputs(
        model: torch.jit.ScriptModule,
        inputs: dict[Callable, Sequence[tuple[Any, ...]] | None],
        _receive_inflate_expr: list[str] | None = None,  # For debugging.
        info: dict[Callable, list[str]] | None = None,  # Optional argument to provide info about the function or its inputs
        skip_size_check=False,
) -> None:
    """Add bundled sample inputs to a model for an arbitrary list of public functions.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions are also defined:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception("Only ScriptModule is supported.")  # noqa: TRY002

    if not inputs:
        raise Exception("Please provide inputs for at least 1 function")  # noqa: TRY002

    if hasattr(model, "get_all_bundled_inputs") or hasattr(model, "get_bundled_inputs_functions_and_info"):
        raise Exception(  # noqa: TRY002
            "Models can only be augmented with bundled inputs once. "
            "This Model seems to have already been augmented with "
            "bundled inputs. Please start afresh with one that "
            "doesn't have bundled inputs.",
        )

    get_bundled_inputs_functions_and_info_template = ""

    for function, input_list in inputs.items():
        if hasattr(function, "__name__"):
            function_name = function.__name__
        else:
            if hasattr(function, "name"):
                function_name = function.name  # type: ignore[attr-defined]
            else:
                raise Exception(  # noqa: TRY002
                    'At least one of your functions has no attribute name please ensure all have one. m.foo.name = "foo"')


        if input_list is not None and not isinstance(input_list, Sequence):
            raise TypeError(f"Error inputs for function {function_name} is not a Sequence")

        function_arg_types = [arg.type for arg in function.schema.arguments[1:]]  # type: ignore[attr-defined]
        deflated_inputs_type: ListType = ListType(TupleType(function_arg_types))
        model._c._register_attribute(f"_bundled_inputs_deflated_{function_name}", deflated_inputs_type, [])

        if hasattr(model, "_generate_bundled_inputs_for_" + function_name):
            if input_list is not None:
                raise Exception(  # noqa: TRY002
                    f"inputs[{function_name}] is not None, but _generate_bundled_inputs_for_{function_name} is already defined"
                )
            # Model author already defined _generate_bundled_inputs_for_<function_name>.
        elif input_list is None or len(input_list) == 0:
            raise Exception(  # noqa: TRY002
                f"inputs for {function_name} must be specified if "
                f"_generate_bundled_inputs_for_{function_name} is not already defined"
            )
        else:
            # Iterate over the inputs and args in each input.
            # Accumulate `deflated_inputs` as (possibly) compressed values
            # and `parts` to be joined into the expression that unpacks them.
            deflated_inputs = []
            parts = []
            for inp_idx, args in enumerate(input_list):
                if not isinstance(args, tuple) and not isinstance(args, list):  # type: ignore[arg-type]
                    raise TypeError(
                        f"Error bundled input for function {function_name} idx: {inp_idx} is not a Tuple or a List"
                    )
                deflated_args = []
                parts.append("(")
                for arg_idx, arg in enumerate(args):
                    inflate_helper_fn_name = _get_inflate_helper_fn_name(arg_idx, inp_idx, function_name)
                    deflated, inflater, helper_definition = _inflate_expr(
                        arg,
                        f"deflated[{inp_idx}][{arg_idx}]",
                        inflate_helper_fn_name,
                        skip_size_check=skip_size_check,
                    )
                    deflated_args.append(deflated)
                    parts.append(f"    {inflater},")
                    if helper_definition:
                        model.define(textwrap.dedent(helper_definition))
                deflated_inputs.append(tuple(deflated_args))
                parts.append("),")
            parts.append("")
            expr = "\n".join(parts)

            # Back-channel return this expr for debugging.
            if _receive_inflate_expr is not None:
                _receive_inflate_expr.append(expr)
            setattr(model, f"_bundled_inputs_deflated_{function_name}", deflated_inputs)
            definition = textwrap.dedent("""
                def _generate_bundled_inputs_for_{name}(self):
                    deflated = self._bundled_inputs_deflated_{name}
                    return [
                {expr}
                    ]
                """).format(expr=expr, name=function_name)
            model.define(definition)

        # Define get_all_bundled_inputs_for_<function_name> that caches the generated inputs.
        model.define(textwrap.dedent("""
            def get_all_bundled_inputs_for_{name}(self):
                all_inputs = self._generate_bundled_inputs_for_{name}()
                assert all_inputs is not None
                return all_inputs
            """).format(name=function_name))

        # Add to the high level helper methods
        inputs_info = repr(info[function]) if info and function in info else '[]'
        get_bundled_inputs_functions_and_info_template += f"""
            temp_dict : Dict[str,List[str]] = {{}}
            info: List[str] = {inputs_info}

            temp_dict['info'] = info
            temp_dict['get_inputs_function_name'] = ['get_all_bundled_inputs_for_{function_name}']
            all_inputs['{function_name}'] = temp_dict
            """

        # To ensure backwards compatibility and a streamlined api for forward these wrappers are provided
        if function_name == 'forward':
            model.define(textwrap.dedent("""
                def get_all_bundled_inputs(self):
                    return self.get_all_bundled_inputs_for_forward()
                """))
            model.define(textwrap.dedent("""
                def get_num_bundled_inputs(self):
                    return len(self.get_all_bundled_inputs_for_forward())
                """))

    # Define some high level helper methods that act on all bundled inputs
    model.define(textwrap.dedent(f"""
        def get_bundled_inputs_functions_and_info(self):
            all_inputs : Dict[str, Dict[str,List[str]]] = {{}}
            {get_bundled_inputs_functions_and_info_template}
            return all_inputs
        """))

def _inflate_expr(
    arg: T, ref: str, inflate_helper_fn_name: str, skip_size_check: bool = False
) -> tuple[T | torch.Tensor, str, str | None]:
    # Allow custom inflation expressions any object.
    # For example, calling custom image-decoding ops.
    # Or just use "{}" as the format string to ignore size limits.
    if isinstance(arg, InflatableArg):
        if arg.fmt_fn:
            if arg.fmt not in ["{}", ""]:
                raise Exception(  # noqa: TRY002
                    f"Bundled input argument at position '{ref}' has "
                    f"both arg.fmt_fn => \n{arg.fmt_fn} "
                    f"\n and arg.fmt  => {arg.fmt}. "
                    "Please choose `arg.fmt` if the deflater is straightforward or "
                    "`arg.fmt_fn` if you need a function."
                )

            helper_definition = arg.fmt_fn.format(inflate_helper_fn_name)
            expr = f"self.{inflate_helper_fn_name}({ref})"

            return arg.value, expr, helper_definition
        else:
            return arg.value, arg.fmt.format(ref), None

    if isinstance(arg, torch.Tensor):
        # Small-storage tensors can just be saved directly.
        if arg._typed_storage().size() <= MAX_RAW_TENSOR_SIZE or skip_size_check:
            return arg, ref, None
        # Small contiguous tensors can be cloned to have small storage.
        # TODO: Should we do this even for non-contiguous tensors?
        if arg.is_contiguous() and arg.numel() <= MAX_RAW_TENSOR_SIZE:
            return arg.clone(), ref, None
        # Example inputs commonly come from torch.zeros, torch.ones, or torch.full.
        # These can be represented compactly.
        for fmt in [torch.contiguous_format, torch.channels_last]:
            if arg.is_contiguous(memory_format=fmt) and (arg == arg.flatten()[0]).all().item():
                return (arg.flatten()[0].clone().expand(*arg.size()),
                        f"{ref}.contiguous(memory_format={fmt})", None)
        # Prevent big tensors from being bundled by default.
        # TODO: Provide more useful diagnostics.
        raise Exception(  # noqa: TRY002
            f"Bundled input argument at position '{ref}' is "
            f"a tensor with storage size {arg._typed_storage().size()}. "
            f"You probably don't want to bundle this as an input. "
        )
    else:
        return arg, ref, None

def _get_bundled_inputs_attributes_and_methods(script_module: torch.jit.ScriptModule) -> tuple[list[str], list[str]]:
    methods: list[str] = []
    attributes: list[str] = []

    # Has bundled inputs for forward
    if hasattr(script_module, 'get_all_bundled_inputs'):
        methods.append('get_all_bundled_inputs')
        methods.append('get_num_bundled_inputs')
        methods.append('run_on_bundled_input')

    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        methods.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            methods.append("get_all_bundled_inputs_for_" + function_name)
            methods.append("_generate_bundled_inputs_for_" + function_name)
            attributes.append("_bundled_inputs_deflated_" + function_name)

            bundled_inputs_fn = getattr(
                script_module,
                f"get_all_bundled_inputs_for_{function_name}"
            )
            num_bundled_inputs: int = len(bundled_inputs_fn())

            # Check inflate helper functions for each function, argument and bundled input
            func = getattr(script_module, function_name)
            for arg_idx in range(len(func.schema.arguments) - 1):
                for input_idx in range(num_bundled_inputs):
                    helper_fn_name = _get_inflate_helper_fn_name(
                        arg_idx=arg_idx,
                        input_idx=input_idx,
                        function_name=function_name
                    )
                    # if the arg has an InflatableArg with fmt_fn, add the helper function name
                    if hasattr(script_module, helper_fn_name):
                        methods.append(helper_fn_name)

    return (methods, attributes)


def _get_inflate_helper_fn_name(
    arg_idx: int,
    input_idx: int,
    function_name: str,
) -> str:
    return f"_inflate_helper_for_{function_name}_input_{input_idx}_arg_{arg_idx}"



def bundle_randn(*size, dtype=None):
    """Generate a tensor that will be inflated with torch.randn."""
    stub = torch.zeros(1, dtype=dtype).expand(*size)
    return InflatableArg(value=stub, fmt="torch.randn_like({})")


def bundle_large_tensor(t):
    """Wrap a tensor to allow bundling regardless of size."""
    return InflatableArg(value=t, fmt="{}")
<|endoftext|>import base64
import zlib
from collections.abc import Callable, Iterable
from typing import Generic, TypeVar


T = TypeVar("T")

_ENCODING_VERSION: int = 1

__all__ = ["AppendingByteSerializer"]


#######################################
# Helper classes
#######################################

CHECKSUM_DIGEST_SIZE = 4


class BytesWriter:
    def __init__(self) -> None:
        # Reserve CHECKSUM_DIGEST_SIZE bytes for checksum
        self._data = bytearray(CHECKSUM_DIGEST_SIZE)

    def write_uint64(self, i: int) -> None:
        self._data.extend(i.to_bytes(8, byteorder="big", signed=False))

    def write_str(self, s: str) -> None:
        payload = base64.b64encode(s.encode("utf-8"))
        self.write_bytes(payload)

    def write_bytes(self, b: bytes) -> None:
        self.write_uint64(len(b))
        self._data.extend(b)

    def to_bytes(self) -> bytes:
        digest = zlib.crc32(self._data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        if len(digest) != CHECKSUM_DIGEST_SIZE:
            raise AssertionError("Computed checksum digest has unexpected size")
        self._data[0:CHECKSUM_DIGEST_SIZE] = digest
        return bytes(self._data)


class BytesReader:
    def __init__(self, data: bytes) -> None:
        # Check for data corruption
        if len(data) < CHECKSUM_DIGEST_SIZE:
            raise AssertionError("Input data is too short to contain checksum")
        digest = zlib.crc32(data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        if len(digest) != CHECKSUM_DIGEST_SIZE:
            raise AssertionError("Computed checksum digest has unexpected size")
        if data[0:CHECKSUM_DIGEST_SIZE] != digest:
            raise RuntimeError(
                "Bytes object is corrupted, checksum does not match. "
                f"Expected: {data[0:CHECKSUM_DIGEST_SIZE]!r}, Got: {digest!r}"
            )

        self._data = data
        self._i = CHECKSUM_DIGEST_SIZE

    def is_finished(self) -> bool:
        return len(self._data) == self._i

    def read_uint64(self) -> int:
        result = int.from_bytes(
            self._data[self._i : self._i + 8], byteorder="big", signed=False
        )
        self._i += 8
        return result

    def read_str(self) -> str:
        return base64.b64decode(self.read_bytes()).decode("utf-8")

    def read_bytes(self) -> bytes:
        size = self.read_uint64()
        result = self._data[self._i : self._i + size]
        self._i += size
        return result


#######################################
# AppendingByteSerializer
#######################################


class AppendingByteSerializer(Generic[T]):
    """
    Provides efficient serialization and deserialization of list of bytes
    Note that this does not provide any guarantees around byte order
    """

    _serialize_fn: Callable[[BytesWriter, T], None]
    _writer: BytesWriter

    def __init__(
        self,
        *,
        serialize_fn: Callable[[BytesWriter, T], None],
    ) -> None:
        self._serialize_fn = serialize_fn
        self.clear()

    def clear(self) -> None:
        self._writer = BytesWriter()
        # First 8-bytes are for version
        self._writer.write_uint64(_ENCODING_VERSION)

    def append(self, data: T) -> None:
        self._serialize_fn(self._writer, data)

    def extend(self, elems: Iterable[T]) -> None:
        for elem in elems:
            self.append(elem)

    def to_bytes(self) -> bytes:
        return self._writer.to_bytes()

    @staticmethod
    def to_list(data: bytes, *, deserialize_fn: Callable[[BytesReader], T]) -> list[T]:
        reader = BytesReader(data)
        if reader.read_uint64() != _ENCODING_VERSION:
            raise AssertionError(
                f"Encoding version mismatch in AppendingByteSerializer.to_list, \
                    got {reader.read_uint64()}"
            )

        result: list[T] = []
        while not reader.is_finished():
            result.append(deserialize_fn(reader))
        return result
<|endoftext|>from typing import Any

import torch
import enum

from torch._C import _to_dlpack as to_dlpack
from torch.types import Device as _Device

__all__ = [
    "DLDeviceType",
    "from_dlpack",
    "ReadOnlyTensorWrapper",
]

class DLDeviceType(enum.IntEnum):
    # Enums as in DLPack specification (aten/src/ATen/dlpack.h)
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
    kDLMAIA = 17,


class ReadOnlyTensorWrapper(torch.Tensor):
    r"""A zero-copy, read-only view of a tensor for DLPack interop only.

    Wrapping a tensor with ``ReadOnlyTensorWrapper`` declares the intent that
    consumers must not mutate its data. It changes only the DLPack export
    behavior; the wrapper shares storage with the source tensor and does not
    copy.

    Both DLPack export paths are routed to read-only variants:

    * the fast ``__dlpack_c_exchange_api__`` C exchange protocol (used by
      tvm-ffi / CuteDSL) points at the const exchange API, which exports
      through ``const_data_ptr()`` and sets
      ``DLPACK_FLAG_BITMASK_READ_ONLY``;
    * the ``__dlpack__()`` capsule protocol forces ``read_only=True``.

    Because the export uses ``const_data_ptr()``, exporting a copy-on-write
    tensor does not materialize it.

    The wrapper is export-only: every torch operation other than the DLPack
    protocol methods raises ``RuntimeError``. Unwrap it (e.g. via the original
    tensor) to operate on the data.

    Example::

        x = torch.randn(8)
        ro = ReadOnlyTensorWrapper(x)
        cute.runtime.from_dlpack(ro, enable_tvm_ffi=True)  # read-only export
    """

    # Consumers discover the C exchange API on the type. Pointing it at the
    # const variant makes the fast path export read-only.
    # pyrefly: ignore [missing-attribute]
    __dlpack_c_exchange_api__: object = torch._C._const_dlpack_exchange_api()

    # The only torch functions the wrapper participates in. Everything else is
    # rejected so the wrapper cannot be used as a normal tensor.
    _DLPACK_ALLOWED = frozenset(
        {torch.Tensor.__dlpack__, torch.Tensor.__dlpack_device__}
    )

    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> "ReadOnlyTensorWrapper":
        return tensor.as_subclass(cls)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in cls._DLPACK_ALLOWED:
            # Run as a plain tensor so the export internals (property accesses,
            # is_conj(), etc.) are not re-intercepted and re-blocked.
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))
        raise RuntimeError(
            f"{cls.__name__} only supports DLPack export; "
            f"'{getattr(func, '__name__', func)}' is not allowed. "
            "Unwrap the tensor to operate on its data."
        )

    def __dlpack__(self, *, max_version=None, **kwargs):  # type: ignore[override]
        # The capsule path can only carry read-only on the versioned protocol.
        if max_version is None:
            max_version = (1, 0)
        return super().__dlpack__(max_version=max_version, read_only=True, **kwargs)


torch._C._add_docstr(to_dlpack, r"""to_dlpack(tensor) -> PyCapsule

Returns an opaque object (a "DLPack capsule") representing the tensor.

.. note::
  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns
  cannot be used for anything in Python other than use it as input to
  ``from_dlpack``. The more idiomatic use of DLPack is to call
  ``from_dlpack`` directly on the tensor object - this works when that
  object has a ``__dlpack__`` method, which PyTorch and most other
  libraries indeed have now.

.. warning::
  Only call ``from_dlpack`` once per capsule produced with ``to_dlpack``.
  Behavior when a capsule is consumed multiple times is undefined.

Args:
    tensor: a tensor to be exported

The DLPack capsule shares the tensor's memory.
""")


# TODO: add a typing.Protocol to be able to tell Mypy that only objects with
# __dlpack__ and __dlpack_device__ methods are accepted.
def from_dlpack(
    ext_tensor: Any,
    *,
    device: _Device | None = None,
    copy: bool | None = None
) -> 'torch.Tensor':
    """from_dlpack(ext_tensor) -> Tensor

    Converts a tensor from an external library into a ``torch.Tensor``.

    The returned PyTorch tensor will share the memory with the input tensor
    (which may have come from another library). Note that in-place operations
    will therefore also affect the data of the input tensor. This may lead to
    unexpected issues (e.g., other libraries may have read-only flags or
    immutable data structures), so the user should only do this if they know
    for sure that this is fine.

    Args:
        ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):
            The tensor or DLPack capsule to convert.

            If ``ext_tensor`` is a tensor (or ndarray) object, it must support
            the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``
            method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is
            an opaque ``PyCapsule`` instance, typically produced by a
            ``to_dlpack`` function or method.

        device (torch.device or str or None): An optional PyTorch device
            specifying where to place the new tensor. If None (default), the
            new tensor will be on the same device as ``ext_tensor``.

        copy (bool or None): An optional boolean indicating whether or not to copy
            ``self``. If None, PyTorch will copy only if necessary.

    Examples::

        >>> import torch.utils.dlpack
        >>> t = torch.arange(4)

        # Convert a tensor directly (supported in PyTorch >= 1.10)
        >>> t2 = torch.from_dlpack(t)
        >>> t2[:2] = -1  # show that memory is shared
        >>> t2
        tensor([-1, -1,  2,  3])
        >>> t
        tensor([-1, -1,  2,  3])

        # The old-style DLPack usage, with an intermediate capsule object
        >>> capsule = torch.utils.dlpack.to_dlpack(t)
        >>> capsule
        <capsule object "dltensor" at ...>
        >>> t3 = torch.from_dlpack(capsule)
        >>> t3
        tensor([-1, -1,  2,  3])
        >>> t3[0] = -9  # now we're sharing memory between 3 tensors
        >>> t3
        tensor([-9, -1,  2,  3])
        >>> t2
        tensor([-9, -1,  2,  3])
        >>> t
        tensor([-9, -1,  2,  3])

    """

    if hasattr(ext_tensor, '__dlpack__'):
        # Only populate kwargs if any of the optional arguments are, in fact, not None. Otherwise,
        # leave them out, since we might end up falling back to no-extra-kwargs __dlpack__ call.
        kwargs: dict[str, Any] = {}
        kwargs["max_version"] = (1, 0)

        # Track copy request for potential manual handling
        requested_copy = copy
        producer_handled_copy = True
        cross_device_transfer = False  # Will be set to True if device transfer is needed

        if copy is not None:
            kwargs["copy"] = copy

        # Parse the device parameter.
        # At this moment, it can either be a torch.device or a str representing
        # a torch.device, e.g. "cpu", "cuda", etc.
        # Get source device first (we need it to detect cross-device transfers)
        ext_device = ext_tensor.__dlpack_device__()

        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            if not isinstance(device, torch.device):
                raise AssertionError(f"from_dlpack: unsupported device type: {type(device)}")

            # Convert target device to DLPack format
            target_dl_device = torch._C._torchDeviceToDLDevice(device)

            # Detect cross-device transfer by comparing source and target devices
            # E.g. CPU->CUDA, cuda:0->cuda:1, etc.
            cross_device_transfer = (ext_device != target_dl_device)

            # Only pass dl_device to producer if NOT cross-device transfer
            if not cross_device_transfer:
                kwargs["dl_device"] = target_dl_device

            # Cross-device transfer always requires a copy
            if cross_device_transfer and copy is False:
                raise ValueError(
                    f"cannot move DLPack tensor from device {ext_device} to {target_dl_device} "
                    "without copying. Set copy=None or copy=True."
                )

        # ext_device is either CUDA or ROCm, we need to pass the current
        # stream
        if ext_device[0] in (DLDeviceType.kDLCUDA, DLDeviceType.kDLROCM):
            stream = torch.cuda.current_stream(f'cuda:{ext_device[1]}')
            # cuda_stream is the pointer to the stream and it is a public
            # attribute, but it is not documented
            # The array API specify that the default legacy stream must be passed
            # with a value of 1 for CUDA
            # https://data-apis.org/array-api/latest/API_specification/array_object.html?dlpack-self-stream-none#dlpack-self-stream-none
            is_cuda = ext_device[0] == DLDeviceType.kDLCUDA
            # Since pytorch is not using PTDS by default, lets directly pass
            # the legacy stream
            stream_ptr = 1 if is_cuda and stream.cuda_stream == 0 else stream.cuda_stream
            kwargs["stream"] = stream_ptr

        # Try different parameter combinations until one works
        dlpack = None

        # Attempt 1: Try with all the parameters
        try:
            dlpack = ext_tensor.__dlpack__(**kwargs)
        except TypeError:
            pass

        # Attempt 2: Remove max_version
        if dlpack is None:
            kwargs.pop("max_version", None)
            try:
                dlpack = ext_tensor.__dlpack__(**kwargs)
            except TypeError:
                pass

        # Attempt 3: Remove copy
        if dlpack is None:
            kwargs.pop("copy", None)
            producer_handled_copy = False
            try:
                dlpack = ext_tensor.__dlpack__(**kwargs)
            except TypeError:
                pass

        # Attempt 4: Remove dl_device
        if dlpack is None:
            kwargs.pop("dl_device", None)
            dlpack = ext_tensor.__dlpack__(**kwargs)

        tensor = torch._C._from_dlpack(dlpack)

        # Manual copy if producer didn't handle it (cross-device already copies via .to())
        if requested_copy is True and not producer_handled_copy and not cross_device_transfer:
            tensor = tensor.clone()

        # Handle cross-device transfer by moving tensor to target device
        if cross_device_transfer:
            tensor = tensor.to(device)

        return tensor

    else:
        if device is not None or copy is not None:
            raise AssertionError(
                "device and copy kwargs not supported when ext_tensor is already a DLPack capsule."
            )
        # Old versions just call the converter
        dlpack = ext_tensor
        return torch._C._from_dlpack(dlpack)
<|endoftext|># mypy: allow-untyped-defs
"""This module contains utility method for mobile model optimization and lint."""

import torch
from enum import Enum
from torch._C import _MobileOptimizerType as MobileOptimizerType
from typing import AnyStr

class LintCode(Enum):
    BUNDLED_INPUT = 1
    REQUIRES_GRAD = 2
    DROPOUT = 3
    BATCHNORM = 4

def optimize_for_mobile(
        script_module: torch.jit.ScriptModule,
        optimization_blocklist: set[MobileOptimizerType] | None = None,
        preserved_methods: list[AnyStr] | None = None,
        backend: str = 'CPU') -> torch.jit.RecursiveScriptModule:
    """
    Optimize a torch script module for mobile deployment.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.
        optimization_blocklist: A set with type of MobileOptimizerType. When set is not passed,
            optimization method will run all the optimizer pass; otherwise, optimizer
            method will run the optimization pass that is not included inside optimization_blocklist.
        preserved_methods: A list of methods that needed to be preserved when freeze_module pass is invoked
        backend: Device type to use for running the result model ('CPU'(default), 'Vulkan' or 'Metal').
    Returns:
        A new optimized torch script module
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            f'Got {type(script_module)}, but ScriptModule is expected.')

    if optimization_blocklist is None:
        optimization_blocklist = set()

    if preserved_methods is None:
        preserved_methods = []

    # Convert potential byte arrays into strings (if there is any) to pass type checking
    # Here we use a new name as assigning it back to preserved_methods will invoke
    # mypy errors (i.e. List[AnyStr] = List[str])
    preserved_methods_str: list[str] = [str(method) for method in preserved_methods]

    bundled_inputs_attributes = _get_bundled_inputs_preserved_attributes(script_module, preserved_methods_str)
    if all(hasattr(script_module, method) for method in bundled_inputs_attributes):
        preserved_methods_str = list(set(preserved_methods_str + bundled_inputs_attributes))

    non_exist_methods = [method for method in preserved_methods_str if not hasattr(script_module, method)]
    if non_exist_methods:
        raise AttributeError(
            f"The following methods to preserve do not exist in script_module: {', '.join(non_exist_methods)}")

    backend = backend.lower()
    if backend == 'cpu':
        optimized_cpp_module = torch._C._jit_pass_optimize_for_mobile(
            script_module._c,
            optimization_blocklist,
            preserved_methods_str)
    elif backend == 'vulkan':
        optimized_cpp_module = torch._C._jit_pass_vulkan_optimize_for_mobile(
            script_module._c,
            optimization_blocklist,
            preserved_methods_str)
    elif backend == 'metal':
        optimized_cpp_module = torch._C._jit_pass_metal_optimize_for_mobile(script_module._c, preserved_methods_str)
    else:
        raise TypeError("Unknown backend, must be one of 'CPU', 'Vulkan' or 'Metal'")

    return torch.jit._recursive.wrap_cpp_module(optimized_cpp_module)


def generate_mobile_module_lints(script_module: torch.jit.ScriptModule):
    """
    Generate a list of lints for a given torch script module.

    Args:
        script_module: An instance of torch script module with type of ScriptModule.

    Returns:
        lint_map: A list of dictionary that contains modules lints
    """
    if not isinstance(script_module, torch.jit.ScriptModule):
        raise TypeError(
            f'Got {type(script_module)}, but ScriptModule is expected.')

    lint_list = []

    if not hasattr(script_module, "_generate_bundled_inputs_for_forward"):
        lint_list.append({"name": LintCode.BUNDLED_INPUT.name, "message": "No bundled input for forward, please add bundled inputs "
                          "before saving the module using torch.utils.bundled_inputs.augment_model_with_bundled_inputs."})

    for name, param in script_module.named_parameters():
        if param.requires_grad:
            lint_list.append({"name": LintCode.REQUIRES_GRAD.name, "message": f"Param {name} requires grad, "
                             "please set torch.no_grad() to reduce memory usage and improve computation speed during "
                              "inference phase."})

    op_names = torch.jit.export_opnames(script_module)
    for op_name in op_names:
        if "dropout" in op_name:
            lint_list.append({"name": LintCode.DROPOUT.name,
                              "message": f"Operator {op_name} exists, remember to call eval() before "
                              "saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop dropout "
                              "operator."})
        if "batch_norm" in op_name:
            lint_list.append({"name": LintCode.BATCHNORM.name,
                              "message": f"Operator {op_name} exists, remember to call eval() before "
                              "saving the module and call torch.utils.mobile_optimizer.optimize_for_mobile to drop batch_norm "
                              "operator."})

    return lint_list

def _get_bundled_inputs_preserved_attributes(script_module: torch.jit.ScriptModule, preserved_methods: list[str]) -> list[str]:

    bundled_inputs_attributes = []
    # Has bundled inputs for forward
    if hasattr(script_module, 'get_all_bundled_inputs'):
        bundled_inputs_attributes.append('get_all_bundled_inputs')
        bundled_inputs_attributes.append('get_num_bundled_inputs')

    # Bundled inputs in module after the change that introduced bundled inputs for multiple functions
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        bundled_inputs_attributes.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            if function_name not in preserved_methods:
                bundled_inputs_attributes.append(function_name)
            bundled_inputs_attributes.append("get_all_bundled_inputs_for_" + function_name)
            bundled_inputs_attributes.append("_bundled_inputs_deflated_" + function_name)

    return bundled_inputs_attributes
<|endoftext|>from types import TracebackType
from typing_extensions import Self

from filelock import FileLock as base_FileLock

from torch.monitor import _WaitCounter


class FileLock(base_FileLock):
    """
    This behaves like a normal file lock.

    However, it adds waitcounters for acquiring and releasing the filelock
    as well as for the critical region within it.

    pytorch.filelock.enter - While we're acquiring the filelock.
    pytorch.filelock.region - While we're holding the filelock and doing work.
    pytorch.filelock.exit - While we're releasing the filelock.
    """

    def __enter__(self) -> Self:
        self.region_counter = _WaitCounter("pytorch.filelock.region").guard()
        with _WaitCounter("pytorch.filelock.enter").guard():
            result = super().__enter__()
        self.region_counter.__enter__()
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.region_counter.__exit__()
        with _WaitCounter("pytorch.filelock.exit").guard():
            # Returns nothing per
            # https://github.com/tox-dev/filelock/blob/57f488ff8fdc2193572efe102408fb63cfefe4e4/src/filelock/_api.py#L379
            super().__exit__(exc_type, exc_value, traceback)
        # Returns nothing per
        # https://github.com/pytorch/pytorch/blob/0f6bfc58a2cfb7a5c052bea618ab62becaf5c912/torch/csrc/monitor/python_init.cpp#L315
        return None
<|endoftext|># mypy: allow-untyped-defs

import copyreg
import os.path as _osp
import weakref

import torch
from torch.utils import (
    backcompat as backcompat,
    collect_env as collect_env,
    data as data,
    deterministic as deterministic,
    hooks as hooks,
)
from torch.utils.backend_registration import (
    generate_methods_for_privateuse1_backend,
    rename_privateuse1_backend,
)
from torch.utils.cpp_backtrace import get_cpp_backtrace
from torch.utils.throughput_benchmark import ThroughputBenchmark


def set_module(obj, mod):
    """
    Set the module attribute on a python object for a given object for nicer printing
    """
    if not isinstance(mod, str):
        raise TypeError("The mod argument should be a string")
    obj.__module__ = mod


cmake_prefix_path = _osp.join(_osp.dirname(_osp.dirname(__file__)), "share", "cmake")


def swap_tensors(t1, t2):
    """
    This function swaps the content of the two Tensor objects.
    At a high level, this will make t1 have the content of t2 while preserving
    its identity.

    This will not work if t1 and t2 have different slots.
    """
    # Ensure there are no weakrefs
    if weakref.getweakrefs(t1):
        raise RuntimeError("Cannot swap t1 because it has weakref associated with it")
    if weakref.getweakrefs(t2):
        raise RuntimeError("Cannot swap t2 because it has weakref associated with it")
    t1_slots = set(copyreg._slotnames(t1.__class__))  # type: ignore[attr-defined]
    t2_slots = set(copyreg._slotnames(t2.__class__))  # type: ignore[attr-defined]
    if t1_slots != t2_slots:
        raise RuntimeError("Cannot swap t1 and t2 if they have different slots")

    def swap_attr(name):
        tmp = getattr(t1, name)
        setattr(t1, name, (getattr(t2, name)))
        setattr(t2, name, tmp)

    def error_pre_hook(grad_outputs):
        raise RuntimeError(
            "Trying to execute AccumulateGrad node that was poisoned by swap_tensors "
            "this can happen when you try to run backward on a tensor that was swapped. "
            "For a module m with `torch.__future__.set_swap_module_params_on_conversion(True)` "
            "you should not change the device or dtype of the module (e.g. `m.cpu()` or `m.half()`) "
            "between running forward and backward. To resolve this, please only change the "
            "device/dtype before running forward (or after both forward and backward)."
        )

    def check_use_count(t, name="t1"):
        use_count = t._use_count()
        error_str = (
            f"Expected use_count of {name} to be 1 or 2 with an AccumulateGrad node but got {use_count} "
            f"make sure you are not holding references to the tensor in other places."
        )
        if use_count > 1:
            if use_count == 2 and t.is_leaf:
                accum_grad_node = torch.autograd.graph.get_gradient_edge(t).node
                # Make sure that the accumulate_grad node was not lazy_init-ed by get_gradient_edge
                if t._use_count() == 2:
                    accum_grad_node.register_prehook(error_pre_hook)
                else:
                    raise RuntimeError(error_str)
            else:
                raise RuntimeError(error_str)

    check_use_count(t1, "t1")
    check_use_count(t2, "t2")

    # Swap the types
    # Note that this will fail if there are mismatched slots
    swap_attr("__class__")

    # Swap the dynamic attributes
    swap_attr("__dict__")

    # Swap the slots
    for slot in t1_slots:
        if hasattr(t1, slot) and hasattr(t2, slot):
            swap_attr(slot)
        elif hasattr(t1, slot):
            setattr(t2, slot, (getattr(t1, slot)))
            delattr(t1, slot)
        elif hasattr(t2, slot):
            setattr(t1, slot, (getattr(t2, slot)))
            delattr(t2, slot)

    # Swap the at::Tensor they point to
    torch._C._swap_tensor_impl(t1, t2)
<|endoftext|>from typing import TypeAlias

import torch
from torch import Tensor
from torch.autograd.grad_mode import no_grad


def _get_foreach_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda", "xpu", "mtia", torch._C._get_privateuse1_backend_name()]


def _get_fused_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports fused kernels in optimizer."""
    return [
        "mps",
        "cuda",
        "xpu",
        "hpu",
        "cpu",
        "mtia",
        torch._C._get_privateuse1_backend_name(),
    ]


TensorListList: TypeAlias = list[list[Tensor | None]]
Indices: TypeAlias = list[int]
_foreach_supported_types = [torch.Tensor]


# This util function splits tensors into groups by device and dtype, which is useful before sending
# tensors off to a foreach implementation, which requires tensors to be on one device and dtype.
# If tensorlistlist contains more than one tensorlist, the following assumptions are made BUT NOT verified:
#   - tensorlists CAN be None
#   - all tensors in the first specified list cannot be None
#   - given an index i, all specified tensorlist[i]s match in dtype and device
# with_indices (bool, optional): whether to track previous indices as the last list per dictionary entry.
#   It comes in handy if there are Nones or literals in the tensorlists that are getting scattered out.
#   Whereas mutating a tensor in the resulting split-up tensorlists WILL propagate changes back to the
#   original input tensorlists, changing up Nones/literals WILL NOT propagate, and manual propagation
#   may be necessary. Check out torch/optim/sgd.py for an example.
@no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList,
    with_indices: bool = False,
) -> dict[tuple[torch.device, torch.dtype], tuple[TensorListList, Indices]]:
    return torch._C._group_tensors_by_device_and_dtype(tensorlistlist, with_indices)


def _device_has_foreach_support(device: torch.device) -> bool:
    return (
        device.type in (_get_foreach_kernels_supported_devices() + ["cpu"])
        and not torch.jit.is_scripting()
    )


def _has_foreach_support(tensors: list[Tensor], device: torch.device) -> bool:
    return _device_has_foreach_support(device) and all(
        t is None or type(t) in _foreach_supported_types for t in tensors
    )
<|endoftext|>import functools


@functools.cache
def has_helion_package() -> bool:
    try:
        import helion  # type: ignore[import-untyped, import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def has_helion() -> bool:
    return has_helion_package()
<|endoftext|>"""Miscellaneous utilities to aid with typing."""

from collections.abc import Callable
from typing import Any, cast, Concatenate, TypeVar
from typing_extensions import ParamSpec


# Helper to turn Optional[T] into T when we know None either isn't
# possible or should trigger an exception.
T = TypeVar("T")


def not_none(obj: T | None) -> T:
    if obj is None:
        raise TypeError("Invariant encountered: value was None when it should not be")
    return obj


_P = ParamSpec("_P")
_R = TypeVar("_R")
_A1 = TypeVar("_A1")


def copy_func_params(
    source_func: Callable[_P, Any],
) -> Callable[[Callable[..., _R]], Callable[_P, _R]]:
    """Cast the decorated function's call signature to the source_func's.

    Usage:
        def upstream_func(a: int, b: float, *, double: bool = False) -> float: ...
        @copy_func_params(upstream_func)
        def enhanced(a: int, b: float, *args: Any, double: bool = False, **kwargs: Any) -> str: ...
    """

    def return_func(func: Callable[..., _R]) -> Callable[_P, _R]:
        return cast(Callable[_P, _R], func)

    return return_func


def copy_method_params(
    source_method: Callable[Concatenate[Any, _P], Any],
) -> Callable[[Callable[..., _R]], Callable[Concatenate[_A1, _P], _R]]:
    """Cast the decorated *method*'s call signature to the source_method's.
    Keeps the first argument type (e.g., self/cls).
    """

    def return_func(func: Callable[..., _R]) -> Callable[Concatenate[_A1, _P], _R]:
        return cast(Callable[Concatenate[_A1, _P], _R], func)

    return return_func


# stricter variants to preserve the origin callers Return Type too.
# TODO: consider folding both these into the above variants with an optional
# parameter to control whether to copy the return type or not.
def copy_func_sig(
    source_func: Callable[_P, _R],
) -> Callable[[Callable[..., _R]], Callable[_P, _R]]:
    """Cast the decorated function's call signature and return type to the source_func's."""

    def _return(func: Callable[..., _R]) -> Callable[_P, _R]:
        return cast(Callable[_P, _R], func)

    return _return


def copy_method_sig(
    source_method: Callable[Concatenate[_A1, _P], _R],
) -> Callable[[Callable[..., _R]], Callable[Concatenate[_A1, _P], _R]]:
    """Cast the decorated *method*'s call signature to the source_method and return type."""

    def _return(func: Callable[..., _R]) -> Callable[Concatenate[_A1, _P], _R]:
        return cast(Callable[Concatenate[_A1, _P], _R], func)

    return _return
<|endoftext|># mypy: allow-untyped-defs
# This module provides a FAST (on GPU) content addressable store for storages
# (and tensors on top of them) with VERY WEAK portability guarantees (e.g.,
# don't expect CPU/CUDA to address to the same hash, don't expect it to be
# portable across devices) that is NOT cryptographically secure.  In return,
# we are able to hash 40G of tensor data on GPU in less than a second,
# compared to running SHA-1 in CPU which would take a minute or so.  The primary
# use case is for efficiently snapshotting intermediate tensor data for
# offline debugging, but it's been put in this module in case you think of
# another use case for it.  The hash function could be replaced with a
# straight reimplementation of SHA-1, which would give us much stronger
# portability guarantees.
#
# WARNING: THERE IS NO BC/FC GUARANTEE FOR THIS FORMAT!  If you need to format
# shift the result, consider packing it into a single torch.save object
# with traditional view sharing.
#
# Because of the weak portability guarantees, you can only write to the
# content store from a single process; we don't provide any capability
# of "reopening" a content store to add more things to it.  But we don't
# assume that you can keep all of the tensors you want to add to the store
# in memory at once, because you probably can't!  Nor do we assume that
# you know a priori whether or not two storages can be deduplicated or not.
#
# Note: only storages are content-addressed; tensors are name addressed
#
# Note: our padding strategy means that [1, 0] and [1] int16 tensors would
# map to the same (padded) storage.  We think this will be immaterial for most
# users.

import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict

import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch.multiprocessing.reductions import StorageWeakRef


def lazy_compile(**compile_kwargs):
    """Lazily wrap a function with torch.compile on the first call

    This avoids eagerly importing dynamo.
    """

    def decorate_fn(fn):
        @functools.wraps(fn)
        def compile_hook(*args, **kwargs):
            compiled_fn = torch.compile(fn, **compile_kwargs)
            globals()[fn.__name__] = functools.wraps(fn)(compiled_fn)
            return compiled_fn(*args, **kwargs)

        return compile_hook

    return decorate_fn


# Use of torch.compile is mandatory for (1) good memory usage
# and (2) xor_sum implementation.  This is our first instance of
# using PT2 to implement a kernel in PyTorch; if we get AOT capabilities
# it would be good to apply it here.
@lazy_compile(dynamic=True)
def hash_storage_kernel(x):
    # The randint calls are carefully written to hit things we
    # have lowerings for in inductor.  Lack of unsigned 32-bit integer
    # is a pain.
    a = torch.randint(
        -(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32
    ).abs()
    a = ((a % (2**31 - 1)) + 1).long()
    b = (
        torch.randint(-(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32)
        .abs()
        .long()
    )
    # This is a standard shift-multiply universal hash family
    # plus xor sum hash, using Philox to generate random numbers.
    # Our Philox RNG is not deterministic across devices so
    # don't use this for stable hashing.
    #
    # This assumes fixed length so you're also obligated to bucket
    # by the length of tensor as well
    return prims.xor_sum((a * x + b).int(), [0])


# Returns a hex digest of the data in the storage.  Guaranteed to be
# SHA-1 if stable_hash=True, otherwise it will consistent for a single
# process run but not necessarily across processes.
def hash_storage(storage: torch.UntypedStorage, *, stable_hash: bool = False) -> str:
    import torch._dynamo
    from torch._dynamo.utils import is_compile_supported

    device_type = storage.device.type
    if stable_hash or not is_compile_supported(device_type):
        cpu_storage = storage.cpu()
        # TODO: make storage support buffer protocol so this isn't
        # necessary
        buf = (ctypes.c_byte * cpu_storage.nbytes()).from_address(
            cpu_storage.data_ptr()
        )
        sha1 = hashlib.sha1(usedforsecurity=False)
        sha1.update(buf)
        return sha1.hexdigest()

    # TODO: factor this into a random utility
    if device_type == "cpu":
        generator = torch._C.default_generator
    elif device_type == "cuda":
        generator = torch.cuda.default_generators[storage.device.index]
    elif device_type == "mps":
        generator = torch.mps._get_default_mps_generator()
    elif device_type == "xpu":
        generator = torch.xpu.default_generators[storage.device.index]
    else:
        raise AssertionError(f"unhandled device type {device_type}")
    state = generator.get_state()
    try:
        generator.manual_seed(0)
        x = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)  # type: ignore[call-overload]
        # The dtype-casting view cannot be compiled, and so the
        # padding/reshaping also needs to be done externally even
        # though it could be profitably fused
        pad = -x.numel() % 4
        if pad > 0:
            x = F.pad(x, (0, pad), "constant", 0)
        x = x.view(torch.int32)
        # We run the 32-bit hash five times with differing parameters to
        # reduce chance of collision
        ITER = 5
        cs = [hash_storage_kernel(x).item() for _ in range(ITER)]
        return struct.pack(">" + "i" * ITER, *cs).hex()
    finally:
        generator.set_state(state)


class ContentStoreWriter:
    # Structure:
    #   storages/
    #     00/
    #       0000..00
    #   tensors/
    #     name
    def __init__(self, loc: str, stable_hash: bool = False) -> None:
        self.loc: str = loc
        self.seen_storage_hashes: set[str] = set()
        self.stable_hash = stable_hash

    # TODO: offer some sort of non-blocking API to speed things up
    def write_storage(self, storage: torch.UntypedStorage) -> str:
        h = hash_storage(storage, stable_hash=self.stable_hash)
        if h in self.seen_storage_hashes:
            return h
        # TODO: consider not using torch.save for this; we don't actually
        # need any metadata for the storage
        subfolder = os.path.join(self.loc, "storages")
        os.makedirs(subfolder, exist_ok=True)
        target = os.path.join(subfolder, h)
        if os.path.exists(target):
            return h
        torch.save(storage, target)
        self.seen_storage_hashes.add(h)
        return h

    def compute_tensor_metadata(self, t: torch.Tensor, h=None):
        if h is None:
            h = hash_storage(t.untyped_storage(), stable_hash=self.stable_hash)
        return (
            t.dtype,
            h,
            t.storage_offset(),
            tuple(t.shape),
            t.stride(),
            torch._utils.get_tensor_metadata(t),
        )

    def write_tensor(self, name: str, t: torch.Tensor) -> None:
        storage = t.untyped_storage()
        h = self.write_storage(storage)
        # TODO: Support more advanced snapshotting of requires_grad/grad/etc
        d, f = os.path.split(name)
        payload = self.compute_tensor_metadata(t, h=h)
        subfolder = os.path.join(self.loc, "tensors", d)
        os.makedirs(subfolder, exist_ok=True)
        torch.save(payload, os.path.join(subfolder, f))


class ContentStoreReader:
    def __init__(self, loc: str, *, cache=True) -> None:
        self.loc = loc
        self.storage_cache: (
            dict[torch.device | None, dict[str, StorageWeakRef]] | None
        ) = None
        if cache:
            self.storage_cache = defaultdict(dict)

    def read_storage(self, h: str, *, device=None) -> torch.UntypedStorage:
        if device is not None:
            device = torch.device(device)
        ws = (
            self.storage_cache[device].get(h)
            if self.storage_cache is not None
            else None
        )
        s: torch.UntypedStorage | None
        if ws is not None:
            s = torch.UntypedStorage._new_with_weak_ptr(ws.cdata)
            if s is not None:
                return s
        s = torch.load(
            os.path.join(self.loc, "storages", h),
            weights_only=True,
            map_location=device,
        )._untyped_storage
        if s is None:
            raise AssertionError(
                f"expected storage for hash {h} in {os.path.join(self.loc, 'storages')}, got None"
            )
        if self.storage_cache is not None:
            self.storage_cache[device][h] = StorageWeakRef(s)
        return s

    def read_tensor_metadata(self, name: str):
        fn = os.path.join(self.loc, "tensors", name)
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        return torch.load(fn, weights_only=True)

    def read_tensor(self, name: str, *, device=None) -> torch.Tensor:
        dtype, h, storage_offset, size, stride, metadata = self.read_tensor_metadata(
            name
        )
        storage = self.read_storage(h, device=device)
        t = torch.tensor([], dtype=dtype, device=storage.device)
        t.set_(storage, storage_offset, size, stride)
        torch._utils.set_tensor_metadata(t, metadata)
        return t
<|endoftext|># Owner(s): ["module: pytree"]

"""
Contains utility functions for working with nested python data structures.

A *pytree* is a Python nested data structure. It is a tree in the sense that
nodes are Python collections (e.g., list, tuple, dict) and the leaves are
Python values. Furthermore, a pytree should not contain reference cycles.

pytrees are useful for working with nested collections of Tensors. For example,
one can use `tree_map` to map a function over all Tensors inside some nested
collection of Tensors and `tree_leaves` to get a flat list of all Tensors
inside some nested collection. pytrees are helpful for implementing nested
collection support for PyTorch APIs.

This pytree implementation is not very performant due to Python overhead
To improve the performance we can move parts of the implementation to C++.
"""

import dataclasses
import functools
import importlib
import importlib.metadata
import json
import logging
import sys
import threading
import types
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from enum import Enum
from typing import (
    Any,
    cast,
    ClassVar,
    Final,
    Generic,
    NoReturn,
    overload,
    Protocol,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
)
from typing_extensions import deprecated, NamedTuple, Self, TypeIs

from torch.torch_version import TorchVersion as _TorchVersion


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    import torch.utils._cxx_pytree as cxx_pytree


__all__ = [
    "PyTree",
    "Context",
    "FlattenFn",
    "UnflattenFn",
    "DumpableContext",
    "ToDumpableContextFn",
    "FromDumpableContextFn",
    "PyTreeSpec",
    "TreeSpec",
    "LeafSpec",
    "keystr",
    "key_get",
    "register_pytree_node",
    "tree_is_leaf",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_unflatten",
    "tree_iter",
    "tree_leaves",
    "tree_leaves_with_path",
    "tree_structure",
    "tree_map",
    "tree_map_with_path",
    "tree_map_",
    "tree_map_only",
    "tree_map_only_",
    "tree_all",
    "tree_any",
    "tree_all_only",
    "tree_any_only",
    "treespec_dumps",
    "treespec_loads",
    "treespec_pprint",
    "is_namedtuple",
    "is_namedtuple_class",
    "is_namedtuple_instance",
    "is_structseq",
    "is_structseq_class",
    "is_structseq_instance",
]


T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U")
R = TypeVar("R")


DEFAULT_TREESPEC_SERIALIZATION_PROTOCOL = 1
NO_SERIALIZED_TYPE_NAME_FOUND = "NO_SERIALIZED_TYPE_NAME_FOUND"


class KeyEntry(Protocol):
    def __hash__(self) -> int: ...

    def __eq__(self, other: object) -> bool: ...

    def __str__(self) -> str: ...

    def get(self, parent: Any) -> Any: ...


class EnumEncoder(json.JSONEncoder):
    def default(self, obj: object) -> str | dict[str, Any]:
        if isinstance(obj, Enum):
            return {
                "__enum__": True,
                "fqn": f"{obj.__class__.__module__}:{obj.__class__.__qualname__}",
                "name": obj.name,
            }
        return cast(str, super().default(obj))


Context = Any
PyTree = Any
FlattenFn = Callable[[PyTree], tuple[list[Any], Context]]
UnflattenFn = Callable[[Iterable[Any], Context], PyTree]
DumpableContext = Any  # Any json dumpable text
ToDumpableContextFn = Callable[[Context], DumpableContext]
FromDumpableContextFn = Callable[[DumpableContext], Context]
ToStrFunc = Callable[["TreeSpec", list[str]], str]  # deprecated
MaybeFromStrFunc = Callable[[str], tuple[Any, Context, str] | None]  # deprecated
KeyPath = tuple[KeyEntry, ...]
FlattenWithKeysFn = Callable[[PyTree], tuple[list[tuple[KeyEntry, Any]], Any]]

# Keep deprecated alias for backward compatibility
FlattenFunc = FlattenFn  # deprecated
UnflattenFunc = UnflattenFn  # deprecated
FlattenWithKeysFunc = FlattenWithKeysFn  # deprecated


# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
# - flatten_with_keys_fn, which is a callable that takes a
#   pytree and returns a list of (keypath, value) pairs and a context.
class NodeDef(NamedTuple):
    type: type[Any]
    flatten_fn: FlattenFn
    unflatten_fn: UnflattenFn
    flatten_with_keys_fn: FlattenWithKeysFn | None


_NODE_REGISTRY_LOCK = threading.RLock()
SUPPORTED_NODES: dict[type[Any], NodeDef] = {}


# _SerializeNodeDef holds the following:
# - typ: the type of the node (e.g., "Dict", "List", etc)
# - serialized_type_name: the fully qualified name of the type, e.g. "collections.OrderedDict"
# - to_dumpable_context takes a TreeSpec, and returns a serialized string format of the
#   context, and the version number
# - from_dumpable_context takes in a string representation of the context, and the
#   version, and returns the deserialized context
class _SerializeNodeDef(NamedTuple):
    typ: type[Any]
    serialized_type_name: str
    to_dumpable_context: ToDumpableContextFn | None
    from_dumpable_context: FromDumpableContextFn | None


SUPPORTED_SERIALIZED_TYPES: dict[type[Any], _SerializeNodeDef] = {}
SERIALIZED_TYPE_TO_PYTHON_TYPE: dict[str, type[Any]] = {}

# NB: we try really hard to not import _cxx_pytree (which depends on optree)
# as much as possible. This is for isolation: a user who is not using C++ pytree
# shouldn't pay for it, and it helps makes things like cpython upgrades easier.
_optree_minimum_version = _TorchVersion("0.13.0")
try:
    _optree_version = importlib.metadata.version("optree")
except importlib.metadata.PackageNotFoundError:
    # No optree package found
    _cxx_pytree_dynamo_traceable = _cxx_pytree_exists = False
    _optree_version = _TorchVersion("0.0.0a0")
else:
    _optree_version = _TorchVersion(_optree_version)
    if _optree_version < _optree_minimum_version:
        # optree package less than our required minimum version.
        # Pretend the optree package doesn't exist.
        # NB: We will raise ImportError if the user directly tries to
        # `import torch.utils._cxx_pytree` (look in that file for the check).
        _cxx_pytree_dynamo_traceable = _cxx_pytree_exists = False
    else:
        _cxx_pytree_dynamo_traceable = _cxx_pytree_exists = True

_cxx_pytree_imported = False
_cxx_pytree_pending_imports: list[Any] = []


def register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    flatten_with_keys_fn: FlattenWithKeysFn | None = None,
) -> None:
    """Register a container-like type as pytree node.

    Note:
        :func:`register_dataclass` is a simpler way of registering a container-like
        type as a pytree node.

    Args:
        cls: the type to register
        flatten_fn: A callable that takes a pytree and returns a flattened
            representation of the pytree and additional context to represent the
            flattened pytree.
        unflatten_fn: A callable that takes a flattened version of the pytree,
            additional context, and returns an unflattened pytree.
        serialized_type_name: A keyword argument used to specify the fully qualified
            name used when serializing the tree spec.
        to_dumpable_context: An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable
            representation. This is used for json serialization, which is being
            used in torch.export right now.
        from_dumpable_context: An optional keyword argument to custom specify how
            to convert the custom json dumpable representation of the context
            back to the original context. This is used for json deserialization,
            which is being used in torch.export right now.
        flatten_with_keys_fn: An optional keyword argument to specify how to
            access each pytree leaf's keypath when flattening and tree-mapping.
            Like ``flatten_fn``, but in place of a List[leaf], it should return
            a List[(keypath, leaf)].
    """
    with _NODE_REGISTRY_LOCK:
        if cls in SUPPORTED_NODES:
            raise ValueError(f"{cls} is already registered as pytree node.")

    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        flatten_with_keys_fn=flatten_with_keys_fn,
    )

    if not _cxx_pytree_exists:
        return

    if _cxx_pytree_imported:
        import torch.utils._cxx_pytree as cxx_pytree

        cxx_pytree._private_register_pytree_node(
            cls,
            flatten_fn,
            unflatten_fn,
            serialized_type_name=serialized_type_name,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )
    else:
        args = (cls, flatten_fn, unflatten_fn)
        kwargs = {
            "serialized_type_name": serialized_type_name,
            "to_dumpable_context": to_dumpable_context,
            "from_dumpable_context": from_dumpable_context,
        }
        _cxx_pytree_pending_imports.append((args, kwargs))


def register_dataclass(
    cls: type[Any],
    *,
    field_names: list[str] | None = None,
    drop_field_names: list[str] | None = None,
    serialized_type_name: str | None = None,
) -> None:
    """
    Registers a type that has the semantics of a ``dataclasses.dataclass`` type
    as a pytree node.

    This is a simpler API than :func:`register_pytree_node` for registering
    a dataclass or a custom class with the semantics of a dataclass.

    Args:
        cls: The python type to register. The class must have the semantics of a
        dataclass; in particular, it must be constructed by passing the fields
        in.
        field_names (Optional[List[str]]): A list of field names that correspond
            to the **non-constant data** in this class. This list must contain
            all the fields that are used to initialize the class. This argument
            is optional if ``cls`` is a dataclass, in which case the fields will
            be taken from ``dataclasses.fields()``.
        drop_field_names (Optional[List[str]]): A list of field names that
            should not be included in the pytree.
        serialized_type_name: A keyword argument used to specify the fully
            qualified name used when serializing the tree spec. This is only
            needed for serializing the treespec in torch.export.

    Example:

        >>> from torch import Tensor
        >>> from dataclasses import dataclass
        >>> import torch.utils._pytree as pytree
        >>>
        >>> @dataclass
        >>> class Point:
        >>>     x: Tensor
        >>>     y: Tensor
        >>>
        >>> pytree.register_dataclass(Point)
        >>>
        >>> point = Point(torch.tensor(0), torch.tensor(1))
        >>> point = pytree.tree_map(lambda x: x + 1, point)
        >>> assert torch.allclose(point.x, torch.tensor(1))
        >>> assert torch.allclose(point.y, torch.tensor(2))

    """
    drop_field_names = drop_field_names or []

    if not dataclasses.is_dataclass(cls):
        if field_names is None:
            raise ValueError(
                "field_names must be specified with a list of all fields used to "
                f"initialize {cls}, as it is not a dataclass."
            )
    elif field_names is None:
        field_names = [f.name for f in dataclasses.fields(cls) if f.init]
    else:
        dataclass_init_fields = {f.name for f in dataclasses.fields(cls) if f.init}
        dataclass_init_fields.difference_update(drop_field_names)

        if dataclass_init_fields != set(field_names):
            error_msg = "field_names does not include all dataclass fields.\n"

            if missing := dataclass_init_fields - set(field_names):
                error_msg += (
                    f"Missing fields in `field_names`: {missing}. If you want "
                    "to include these fields in the pytree, please add them "
                    "to `field_names`, otherwise please add them to "
                    "`drop_field_names`.\n"
                )

            if unexpected := set(field_names) - dataclass_init_fields:
                error_msg += (
                    f"Unexpected fields in `field_names`: {unexpected}. "
                    "Please remove these fields, or add them to `drop_field_names`.\n"
                )

            raise ValueError(error_msg)

    def _flatten_fn(obj: Any) -> tuple[list[Any], Context]:
        flattened = []
        flat_names = []
        none_names = []
        for name in field_names:
            val = getattr(obj, name)
            if val is not None:
                flattened.append(val)
                flat_names.append(name)
            else:
                none_names.append(name)
        return flattened, [flat_names, none_names]

    def _unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, none_names = context
        return cls(
            **dict(zip(flat_names, values, strict=True)), **dict.fromkeys(none_names)
        )

    def _flatten_fn_with_keys(obj: Any) -> tuple[list[Any], Context]:
        flattened, (flat_names, _none_names) = _flatten_fn(obj)  # type: ignore[misc]
        return [
            (GetAttrKey(k), v) for k, v in zip(flat_names, flattened, strict=True)
        ], flat_names

    _private_register_pytree_node(
        cls,
        _flatten_fn,
        _unflatten_fn,
        serialized_type_name=serialized_type_name,
        flatten_with_keys_fn=_flatten_fn_with_keys,
    )


CONSTANT_NODES: set[type] = set()


def register_constant(cls: type[Any]) -> None:
    """Registers a type as a pytree node with no leaves.

    In a :func:`torch.compile` region, if instances of these types get passed to
    :func:`torch._dynamo.nonstrict_trace`-ed function, they treated as a
    constant (sometimes referred to as "static"):

    1. if the instance object existed before the :func:`torch.compile` region,
    we _assume_ no mutation will happen to it inside the :func:`torch.compile`
    region, require that it has non-default `__eq__` and `__hash__` methods, and
    we guard on the instance based on its `__eq__` method, i.e., if a new
    instance fails to match any instances from the previous compilations,
    :func:`torch.compile` will recompile the function using the new instance.

    2. else if the instance object is created inside the :func:`torch.compile`
    region, we currently don't support using it in a
    :func:`torch._dynamo.nonstrict_trace`-ed function.

    In general, if your class holds Tensors or dynamic int/float/bool (values that
    may change from run-to-run of a function being compiled), then you probably
    do not want to register it as a constant.

    Otherwise if you want to pass instance of a class to a
    :func:`torch._dynamo.nonstrict_trace`-ed function, but you either can't use
    :func:`register_pytree_node` on the class, or the class is "constant" enough
    that you don't want to bother using :func:`register_pytree_node`, you should
    consider using this function.

    Args:
        cls: the type to register as a constant. This type must be hashable.

    Example:

        >>> from dataclasses import dataclass
        >>> import torch.utils._pytree as pytree
        >>>
        >>> @dataclass(frozen=True)
        >>> class Config:
        >>>     norm: str
        >>>
        >>> pytree.register_constant(Config)
        >>>
        >>> config = Config("l2")
        >>> values, spec = pytree.tree_flatten(config)
        >>> assert len(values) == 0

    """
    if cls.__eq__ is object.__eq__:  # type: ignore[comparison-overlap]
        raise TypeError(
            "register_constant(cls) expects `cls` to have a non-default `__eq__` implementation."
        )

    # Class with a custom `__eq__` without `__hash__` won't inherit the default
    # `__hash__` from object; see https://stackoverflow.com/a/1608907.
    if cls.__hash__ is None:  # type: ignore[comparison-overlap]
        raise TypeError(
            "register_constant(cls) expects `cls` to have a non-default `__hash__` implementation."
        )

    def _flatten(x):  # type: ignore[no-untyped-def]
        return [], ConstantNode(x)

    def _unflatten(_, context):  # type: ignore[no-untyped-def]
        return context.value

    def _flatten_with_keys(x):  # type: ignore[no-untyped-def]
        return [], ConstantNode(x)

    with _NODE_REGISTRY_LOCK:
        _private_register_pytree_node(
            cls,
            _flatten,
            _unflatten,
            flatten_with_keys_fn=_flatten_with_keys,
        )
        CONSTANT_NODES.add(cls)


def is_constant_class(cls: type[Any]) -> bool:
    return isinstance(cls, type) and cls in CONSTANT_NODES


@dataclasses.dataclass(frozen=True, slots=True)
class ConstantNode(Generic[T]):
    value: T


def _is_constant_holder(spec: "TreeSpec") -> bool:
    """Checks if the spec is from a pytree registered with register_constant"""
    return isinstance(spec._context, ConstantNode)


def _retrieve_constant(spec: "TreeSpec") -> Any:
    """Given a spec from a pytree registered with register_constant, retrieves the constant"""
    if not _is_constant_holder(spec):
        raise AssertionError("spec does not correspond to a registered constant pytree")
    return tree_unflatten([], spec)


def _register_namedtuple(
    cls: type[Any],
    *,
    serialized_type_name: str,
) -> None:
    """
    Registers a namedtuple as a valid pytree node. By default namedtuples are
    valid pytree nodes, but they are not serializable. This API provides the
    argument `serialized_type_name` which allows these namedtuples to be
    serialized.

    Args:
        cls: the dataclass type to register
        serialized_type_name: The serialized name for the dataclass. This is
        required if you want to serialize the pytree TreeSpec containing this
        namedtuple.
    """
    _private_register_pytree_node(
        cls,
        _namedtuple_flatten,
        _namedtuple_unflatten,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=_namedtuple_serialize,
        from_dumpable_context=_namedtuple_deserialize,
        flatten_with_keys_fn=_namedtuple_flatten_with_keys,
    )


@deprecated(
    "`torch.utils._pytree._register_pytree_node` is deprecated. "
    "Please use `torch.utils._pytree.register_pytree_node` instead.",
    category=FutureWarning,
)
def _register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
    to_str_fn: ToStrFunc | None = None,  # deprecated
    maybe_from_str_fn: MaybeFromStrFunc | None = None,  # deprecated
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    flatten_with_keys_fn: FlattenWithKeysFn | None = None,
) -> None:
    """Register a container-like type as pytree node for the Python pytree only.

    Args:
        cls: the type to register
        flatten_fn: A callable that takes a pytree and returns a flattened
            representation of the pytree and additional context to represent the
            flattened pytree.
        unflatten_fn: A callable that takes a flattened version of the pytree,
            additional context, and returns an unflattened pytree.
        serialized_type_name: A keyword argument used to specify the fully qualified
            name used when serializing the tree spec.
        to_dumpable_context: An optional keyword argument to custom specify how
            to convert the context of the pytree to a custom json dumpable
            representation. This is used for json serialization, which is being
            used in torch.export right now.
        from_dumpable_context: An optional keyword argument to custom specify how
            to convert the custom json dumpable representation of the context
            back to the original context. This is used for json deserialization,
            which is being used in torch.export right now.
        flatten_with_keys_fn: An optional keyword argument to specify how to
            access each pytree leaf's keypath when flattening and tree-mapping.
            Like ``flatten_fn``, but in place of a List[leaf], it should return
            a List[(keypath, leaf)].
    """
    if to_str_fn is not None or maybe_from_str_fn is not None:
        warnings.warn(
            "`to_str_fn` and `maybe_from_str_fn` is deprecated. "
            "Please use `to_dumpable_context` and `from_dumpable_context` instead.",
            FutureWarning,
            stacklevel=2,
        )

    _private_register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
        flatten_with_keys_fn=flatten_with_keys_fn,
    )


def _deregister_pytree_node(
    cls: type[Any],
) -> None:
    """This is an internal function that is used to deregister a pytree node type
    for the Python pytree only. This should be only used inside PyTorch.
    """
    with _NODE_REGISTRY_LOCK:
        del SUPPORTED_NODES[cls]
        node_def = SUPPORTED_SERIALIZED_TYPES[cls]
        del SERIALIZED_TYPE_TO_PYTHON_TYPE[node_def.serialized_type_name]
        del SUPPORTED_SERIALIZED_TYPES[cls]
        CONSTANT_NODES.discard(cls)


def _private_register_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    flatten_with_keys_fn: FlattenWithKeysFn | None = None,
) -> None:
    """This is an internal function that is used to register a pytree node type
    for the Python pytree only. End-users should use :func:`register_pytree_node`
    instead.
    """
    from torch._library.opaque_object import is_custom_class

    if isinstance(cls, type) and is_custom_class(cls):
        # TODO: remove this allowance once downstream callers stop calling
        # register_constant on Enum subclasses. Enums are now natively
        # supported as opaque value types and don't need pytree registration.
        import enum

        if issubclass(cls, enum.Enum):
            log.warning(
                "%s is an Enum subclass and is now natively supported by "
                "torch.compile as an opaque value type. Calling "
                "register_constant() on Enum subclasses is deprecated and "
                "will be an error in a future release.",
                cls,
            )
        else:
            raise ValueError(
                f"{cls} cannot be registered as a pytree as it has been "
                "registered as an opaque object. Opaque objects must be pytree leaves."
            )

    with _NODE_REGISTRY_LOCK:
        if cls in SUPPORTED_NODES:
            # TODO: change this warning to an error after OSS/internal stabilize
            warnings.warn(
                f"{cls} is already registered as pytree node. "
                "Overwriting the previous registration.",
                stacklevel=2,
            )

        node_def = NodeDef(cls, flatten_fn, unflatten_fn, flatten_with_keys_fn)
        SUPPORTED_NODES[cls] = node_def

        if (to_dumpable_context is None) ^ (from_dumpable_context is None):
            raise ValueError(
                f"Both to_dumpable_context and from_dumpable_context for {cls} must "
                "be None or registered."
            )

        if serialized_type_name is None:
            serialized_type_name = NO_SERIALIZED_TYPE_NAME_FOUND

        serialize_node_def = _SerializeNodeDef(
            cls,
            serialized_type_name,
            to_dumpable_context,
            from_dumpable_context,
        )
        SUPPORTED_SERIALIZED_TYPES[cls] = serialize_node_def
        SERIALIZED_TYPE_TO_PYTHON_TYPE[serialized_type_name] = cls


@dataclasses.dataclass(frozen=True, slots=True)
class SequenceKey(Generic[T]):
    idx: int

    def __str__(self) -> str:
        return f"[{self.idx!r}]"

    def get(self, sequence: Sequence[T]) -> T:
        return sequence[self.idx]


K = TypeVar("K", bound=Hashable)


@dataclasses.dataclass(frozen=True, slots=True)
class MappingKey(Generic[K, T]):
    key: K

    def __str__(self) -> str:
        return f"[{self.key!r}]"

    def get(self, mapping: Mapping[K, T]) -> T:
        return mapping[self.key]


@dataclasses.dataclass(frozen=True, slots=True)
class GetAttrKey:
    name: str

    def __str__(self) -> str:
        return f".{self.name}"

    def get(self, obj: Any) -> Any:
        return getattr(obj, self.name)


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_namedtuple(obj: object | type) -> bool:
    """Return whether the object is an instance of namedtuple or a subclass of namedtuple."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_namedtuple_class(cls)


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_namedtuple_class(cls: type) -> bool:
    """Return whether the class is a subclass of namedtuple."""
    return (
        isinstance(cls, type)
        and issubclass(cls, tuple)
        and isinstance(getattr(cls, "_fields", None), tuple)
        and all(type(field) is str for field in cls._fields)  # type: ignore[attr-defined]
        and callable(getattr(cls, "_make", None))
        and callable(getattr(cls, "_asdict", None))
    )


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_namedtuple_instance(obj: object) -> bool:
    """Return whether the object is an instance of namedtuple."""
    return is_namedtuple_class(type(obj))


_T_co = TypeVar("_T_co", covariant=True)


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
class structseq(tuple[_T_co, ...]):
    """A generic type stub for CPython's ``PyStructSequence`` type."""

    __slots__: ClassVar[tuple[()]] = ()

    n_fields: Final[int]  # type: ignore[misc]
    n_sequence_fields: Final[int]  # type: ignore[misc]
    n_unnamed_fields: Final[int]  # type: ignore[misc]

    def __init_subclass__(cls) -> NoReturn:
        """Prohibit subclassing."""
        raise TypeError("type 'structseq' is not an acceptable base type")

    def __new__(
        cls: type[Self],
        sequence: Iterable[_T_co],
        # pyrefly: ignore [bad-function-definition]
        dict: dict[str, Any] = ...,
    ) -> Self:
        raise NotImplementedError


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_structseq(obj: object | type) -> bool:
    """Return whether the object is an instance of PyStructSequence or a class of PyStructSequence."""
    cls = obj if isinstance(obj, type) else type(obj)
    return is_structseq_class(cls)


# Set if the type allows subclassing (see CPython's Include/object.h)
Py_TPFLAGS_BASETYPE: int = 1 << 10


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_structseq_class(cls: type) -> bool:
    """Return whether the class is a class of PyStructSequence."""
    return (
        isinstance(cls, type)
        # Check direct inheritance from `tuple` rather than `issubclass(cls, tuple)`
        and cls.__bases__ == (tuple,)
        # Check PyStructSequence members
        and isinstance(getattr(cls, "n_fields", None), int)
        and isinstance(getattr(cls, "n_sequence_fields", None), int)
        and isinstance(getattr(cls, "n_unnamed_fields", None), int)
        # Check the type does not allow subclassing
        and not bool(cls.__flags__ & Py_TPFLAGS_BASETYPE)  # only works for CPython
    )


# Reference: https://github.com/metaopt/optree/blob/main/optree/typing.py
def is_structseq_instance(obj: object) -> bool:
    """Return whether the object is an instance of PyStructSequence."""
    return is_structseq_class(type(obj))


def _tuple_flatten(d: tuple[T, ...]) -> tuple[list[T], Context]:
    return list(d), None


def _tuple_flatten_with_keys(
    d: tuple[T, ...],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _tuple_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _tuple_unflatten(values: Iterable[T], context: Context) -> tuple[T, ...]:
    return tuple(values)


def _list_flatten(d: list[T]) -> tuple[list[T], Context]:
    return d, None


def _list_flatten_with_keys(d: list[T]) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _list_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _list_unflatten(values: Iterable[T], context: Context) -> list[T]:
    return list(values)


def _dict_flatten(d: dict[Any, T]) -> tuple[list[T], Context]:
    return list(d.values()), list(d.keys())


def _dict_flatten_with_keys(
    d: dict[Any, T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _dict_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(MappingKey(k), v) for k, v in zip(context, values, strict=True)], context


def _dict_unflatten(values: Iterable[T], context: Context) -> dict[Any, T]:
    return dict(zip(context, values, strict=True))


def _namedtuple_flatten(d: NamedTuple) -> tuple[list[Any], Context]:
    return list(d), type(d)


def _namedtuple_flatten_with_keys(
    d: NamedTuple,
) -> tuple[list[tuple[KeyEntry, Any]], Context]:
    values, context = _namedtuple_flatten(d)
    # pyrefly: ignore [bad-return]
    return (
        [
            (GetAttrKey(field), v)
            for field, v in zip(context._fields, values, strict=True)
        ],
        context,
    )


def _namedtuple_unflatten(values: Iterable[T], context: Context) -> NamedTuple:
    return cast(NamedTuple, context(*values))


def _namedtuple_serialize(context: Context) -> DumpableContext:
    if context not in SUPPORTED_SERIALIZED_TYPES:
        raise NotImplementedError(
            f"Can't serialize TreeSpec of namedtuple class {context} because we "
            "didn't register a serialized_type_name. Please register using "
            "`_register_namedtuple`."
        )

    serialize_node_def = SUPPORTED_SERIALIZED_TYPES[context]
    serialized_type_name = serialize_node_def.serialized_type_name

    if serialized_type_name == NO_SERIALIZED_TYPE_NAME_FOUND:
        raise NotImplementedError(
            f"Can't serialize TreeSpec of namedtuple class {context} because we "
            "couldn't find a serialized_type_name. Please register using "
            "`_register_namedtuple`."
        )
    return serialized_type_name


def _namedtuple_deserialize(dumpable_context: DumpableContext) -> Context:
    if dumpable_context not in SERIALIZED_TYPE_TO_PYTHON_TYPE:
        raise NotImplementedError(
            f"Can't deserialize TreeSpec of namedtuple class {dumpable_context} "
            "because we couldn't find a serialized name."
        )

    typ = SERIALIZED_TYPE_TO_PYTHON_TYPE[dumpable_context]
    return typ


def _ordereddict_flatten(d: OrderedDict[Any, T]) -> tuple[list[T], Context]:
    return list(d.values()), list(d.keys())


def _ordereddict_flatten_with_keys(
    d: OrderedDict[Any, T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _ordereddict_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(MappingKey(k), v) for k, v in zip(context, values, strict=True)], context


def _ordereddict_unflatten(
    values: Iterable[T],
    context: Context,
) -> OrderedDict[Any, T]:
    return OrderedDict((key, value) for key, value in zip(context, values, strict=True))


_odict_flatten = _ordereddict_flatten
_odict_unflatten = _ordereddict_unflatten


def _defaultdict_flatten(d: defaultdict[Any, T]) -> tuple[list[T], Context]:
    values, dict_context = _dict_flatten(d)
    return values, [d.default_factory, dict_context]


def _defaultdict_flatten_with_keys(
    d: defaultdict[Any, T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _defaultdict_flatten(d)
    _, dict_context = context
    # pyrefly: ignore [bad-return]
    return [
        (MappingKey(k), v) for k, v in zip(dict_context, values, strict=True)
    ], context


def _defaultdict_unflatten(
    values: Iterable[T],
    context: Context,
) -> defaultdict[Any, T]:
    default_factory, dict_context = context
    return defaultdict(default_factory, _dict_unflatten(values, dict_context))


def _defaultdict_serialize(context: Context) -> DumpableContext:
    default_factory, dict_context = context
    json_defaultdict = {
        "default_factory_module": default_factory.__module__,
        "default_factory_name": default_factory.__qualname__,
        "dict_context": dict_context,
    }
    return json_defaultdict


def _defaultdict_deserialize(dumpable_context: DumpableContext) -> Context:
    if not isinstance(dumpable_context, dict):
        raise AssertionError("dumpable_context must be a dict")

    expected_keys = {
        "default_factory_module",
        "default_factory_name",
        "dict_context",
    }
    if set(dumpable_context) != expected_keys:
        raise AssertionError(
            f"dumpable_context keys must be {expected_keys}, got {set(dumpable_context)}"
        )

    default_factory_module = dumpable_context["default_factory_module"]
    default_factory_name = dumpable_context["default_factory_name"]
    if not isinstance(default_factory_module, str):
        raise AssertionError("default_factory_module must be a string")
    if not isinstance(default_factory_name, str):
        raise AssertionError("default_factory_name must be a string")
    module = importlib.import_module(default_factory_module)
    default_factory = getattr(module, default_factory_name)

    dict_context = dumpable_context["dict_context"]
    return [default_factory, dict_context]


def _deque_flatten(d: deque[T]) -> tuple[list[T], Context]:
    return list(d), d.maxlen


def _deque_flatten_with_keys(
    d: deque[T],
) -> tuple[list[tuple[KeyEntry, T]], Context]:
    values, context = _deque_flatten(d)
    # pyrefly: ignore [bad-return]
    return [(SequenceKey(i), v) for i, v in enumerate(values)], context


def _deque_unflatten(values: Iterable[T], context: Context) -> deque[T]:
    return deque(values, maxlen=context)


_private_register_pytree_node(
    tuple,
    _tuple_flatten,
    _tuple_unflatten,
    serialized_type_name="builtins.tuple",
    flatten_with_keys_fn=_tuple_flatten_with_keys,
)
_private_register_pytree_node(
    list,
    _list_flatten,
    _list_unflatten,
    serialized_type_name="builtins.list",
    flatten_with_keys_fn=_list_flatten_with_keys,
)
_private_register_pytree_node(
    dict,
    _dict_flatten,
    _dict_unflatten,
    serialized_type_name="builtins.dict",
    flatten_with_keys_fn=_dict_flatten_with_keys,
)
_private_register_pytree_node(
    namedtuple,  # type: ignore[arg-type]
    _namedtuple_flatten,
    _namedtuple_unflatten,
    serialized_type_name="collections.namedtuple",
    to_dumpable_context=_namedtuple_serialize,
    from_dumpable_context=_namedtuple_deserialize,
    flatten_with_keys_fn=_namedtuple_flatten_with_keys,
)
_private_register_pytree_node(
    OrderedDict,
    _ordereddict_flatten,
    _ordereddict_unflatten,
    serialized_type_name="collections.OrderedDict",
    flatten_with_keys_fn=_ordereddict_flatten_with_keys,
)
_private_register_pytree_node(
    defaultdict,
    _defaultdict_flatten,
    _defaultdict_unflatten,
    serialized_type_name="collections.defaultdict",
    to_dumpable_context=_defaultdict_serialize,
    from_dumpable_context=_defaultdict_deserialize,
    flatten_with_keys_fn=_defaultdict_flatten_with_keys,
)
_private_register_pytree_node(
    deque,
    _deque_flatten,
    _deque_unflatten,
    serialized_type_name="collections.deque",
    flatten_with_keys_fn=_deque_flatten_with_keys,
)


STANDARD_DICT_TYPES: frozenset[type] = frozenset({dict, OrderedDict, defaultdict})
# pyrefly: ignore [no-matching-overload]
BUILTIN_TYPES: frozenset[type] = frozenset(
    # pyrefly: ignore [bad-argument-type]
    {
        tuple,
        list,
        dict,
        namedtuple,  # type: ignore[arg-type]
        OrderedDict,
        defaultdict,
        deque,
    },
)


@deprecated(
