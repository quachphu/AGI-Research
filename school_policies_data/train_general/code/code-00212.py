        ] = collections.defaultdict(int)

        out_storage_to_metadata_key_to_tensors: collections.defaultdict[
            StorageWeakRef | None,
            collections.defaultdict[MetadataKey, set[torch.Tensor]],
        ] = collections.defaultdict(lambda: collections.defaultdict(set))

        curr_storage = None
        for o in flat_f_outs:
            if isinstance(o, torch.Tensor):
                curr_storage = StorageWeakRef(o.untyped_storage())
                out_tensor_alias_counts[curr_storage] += 1
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                # This is an optimization on top of the "alias of intermediates" logic,
                # which you can read more about under Note [AOT Autograd: outputs aliasing inputs or intermediates!]
                #
                # Before describing the optimization: this is important for AOTAutograd to have good
                # perf around, multi-output views. HOWEVER:
                # - There is a more generic change to AOTAutograd that we'd like to make, that subsumes this case,
                #   around using pre-dispatch tracing to partition out a graph so we can faithfully replay all
                #   views without having to regenerate them at runtime.
                # - It's loosely described in this doc (more details will be added soon):
                #   https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit
                # - Once that change lands, we should just rip out this "optimization", since:
                #   (1) It will be fully unnecessary
                #   (2) Although it is only a few lines of code, it is a bit difficult to reason about
                #       its correctness with the autograd engine in all cases.
                #
                #
                # What is this optimization? Consider the below case:
                # def f(x):
                #     intermediate = x.mul(2)
                #     # x and intermediate here require grad
                #     o1, o2, ... o10 = intermediate.unbind(-1)
                #     return intermediate, o1, o2, ... o10
                # Now, the "intermediate base" handling in AOTAutograd implies that we must do the following:
                #   (1) return "intermediate as an extra output of the compiled graph
                #   (2) regenerate each aliased output off of "intermediate", **outside** of the autograd.Function.
                # The reason AOTAutograd ordinarily does this is for safety: the autograd engine needs to know
                # that o1 through o10 are all aliased, and if we blindly return o1 through o10 from the autograd.Function,
                # this information will be hidden.
                # In particular, mutating one alias might require autograd to update autograd metadata on the other aliases
                # (like their grad_fn, for example, when the autograd engine needs to do view-replay).
                #
                # However, intermediate_base logic can be bad for backward performance (we sometimes generate
                # as_strided calls during the intermediate base logic, which can have a slow backward formula).
                # Is it possible to find a set of conditions where it is **safe** to hide the output aliasing from autograd?
                #
                # For a set of outputs of the graph that alias each other, o_1...o_k, consider:
                # (1) They came from the same multi-output view op, e.g. o_1, ..., o_k = intermediate.unbind(0)
                # (2) If there are any other aliases of o_1 through o_k (in the example above, intermediate),
                #     **at most** 1 can escape from the graph (e.g. there is not some other graph input/output
                #     o_other, that aliases these outputs)
                # (3) o_1...o_k all require_grad, they all share the same ._base, and their ._base requires grad.
                #     This condition is important because it's what causes slowness in the intermediate_base
                #     codepath of aot_autograd. Ordinarily, o_1...o_k would all get a grad_fn, and
                #     aot_autograd's view-replay might give each output an AsStridedBackward as its grad_fn.
                #     "K" AsStridedBackward calls will be *much* slower than a single UnbindBackward.
                # In this setup, is it possible to mutate one of the outputs o_i in a way that would affect the autograd meta
                # of the other aliases?
                #
                # Claim: No! Consider a few example (which I'm pretty sure cover all cases of mutation w.r.t. autograd):
                # (a) What happens if we mutate any of o_1 through o_k directly?
                #     Autograd raises an error:
                #     "RuntimeError: Output 0 of UnbindBackward0 is a view and is being modified inplace. This view is
                #      the output of a function that returns multiple views. Such functions do not allow the output
                #      views to be modified inplace. You should replace the inplace operation by an out-of-place one."
                # (b) What if we take a view of o_k and mutate it, o_k.view(o_k.shape).mul_(2)?
                #     Autograd raises the same error- the "multi-output-view"ness of an alias propagates to future views.
                # (c) What if we mutate o_k under no_grad?
                #     Autograd raises the same error
                # (d) What if we detach and mutate, e.g. o_k.detach().mul_(2)?
                #     Autograd allows this, *but* autograd updates all alias's grad_fn's to be error functions when accessed.
                #     Autograd raises the same error
                # (e) What if we try to mutate another alias of o_1...o_k, that was **not** created from a multi-output view?
                #     We promised that there is at most **one** such alias, e.g. intermediate in the example above.
                #     You can mutate intermediate, but in eager mode this will change the grad_fn of o_1...o_k
                #     to be error fn's.
                #     Since intermediate was the *only* non-multi-output-alias, there are no other aliases
                #     of `intermediate` around that were produced by the compiled fn and have a valid grad_fn.
                #
                # Coming back to this optimization:
                # Given that it is not possible for mutating one of these aliases to affect the autograd metadata of another alias
                # without causing an error in eager mode, we will simple hide the aliasing from autograd during torch.compile
                # if all of the above conditions are met.
                # This has the slight downside that it's possible to write some "bad" code that autograd will raise an error on
                # in eager but fail to during torch.compile, but it has the benefit that this code has much better performance.
                # NOTE: if and when we eventually update AOTAutograd to do the "view graph slicing" defined here:
                # https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit,
                # then this optimization will probably matter less and might be ok to remove.
                is_cur_tensor_multi_out_view = isinstance(
                    o, FunctionalTensor
                ) and torch._functionalize_is_multi_output_view(  # type: ignore[attr-defined]
                    o.elem
                )
                if is_cur_tensor_multi_out_view:
                    num_aliased_tensors_that_are_multi_output_views[curr_storage] += 1
                if o.requires_grad:
                    out_storage_to_metadata_key_to_tensors[curr_storage][
                        MetadataKey.make(o)
                    ].add(o)

        # maps the id of an intermediate base to its index in the output of the compiled forward
        intermediate_base_tensor_id_to_output_idx: dict[int, int] = {}
        intermediate_bases: list[torch.Tensor] = []
        intermediate_bases_descs: list[AOTInput] = []
        # Why Do We Care If Storage Changed?
        # It's important to understand the implications of storage changes in complex scenarios. Take this example:
        #
        # def f(x):
        #     x_storage = x.untyped_storage()
        #     non_leaf_tensor = torch.ones(4, requires_grad=True).clone()
        #
        #     # Using no_grad() and _unsafe_preserve_version_counter to simulate the .data = operation
        #     with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(x):
        #         x.set_(non_leaf_tensor.untyped_storage())
        #
        #     out = x.view(-1)
        #
        #     # Restoring x to its original storage, again simulating .data = operation
        #     with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(x):
        #         x.set_(x_storage)
        #
        #     return out
        #
        # In this scenario, 'x' and 'out' have different shapes and are stored at different memory addresses, aka no aliasing.
        # However, due to how set_() and more specifically, set is functionalized, is defined to preserve eager semantics,
        # the autograd engine mistakenly assumes that 'x' and 'out' are aliased, treating 'x' as 'out._base'.
        # This misinterpretation leads to an 'alias_of_input' flag, causing an unnecessary as_strided() call to be generated,
        # which could lead to issues later in the code.
        for o, desc in zip(flat_f_outs, flat_f_outs_descs):
            functional_tensor_storage_changed = isinstance(
                o, FunctionalTensor
            ) and torch._functionalize_was_storage_changed(  # type: ignore[attr-defined]
                o.elem
            )
            curr_storage = (
                None
                if not isinstance(o, torch.Tensor)
                else StorageWeakRef(o.untyped_storage())
            )
            outs_with_identical_metadata_that_require_grad: list[torch.Tensor] = (
                []
                if not isinstance(o, Tensor)
                else [
                    curr
                    for curr in out_storage_to_metadata_key_to_tensors[curr_storage][
                        MetadataKey.make(o)
                    ]
                    if o is not curr
                ]
            )

            # See Note [Accessing .grad_fn on FunctionalTensor]
            # In-place operations on views will trigger a lazy rebase of the autograd graph;
            # this runs during access to the .grad_fn. The rebase logic will invoke view ops
            # on FunctionalTensors, so we must enable a FunctionalTensorMode here to ensure
            # these op calls succeed.
            grad_fn = None
            if isinstance(o, Tensor):
                with FunctionalTensorMode():
                    grad_fn = o.grad_fn

            is_result_of_custom_autograd_fn = False
            # Need to check for both custom cpp (CppFunction) and python (BackwardCFunction)
            # autograd fns
            if type(grad_fn).__name__ == "CppFunction":
                is_result_of_custom_autograd_fn = True
            if isinstance(grad_fn, torch.autograd.function.BackwardCFunction):
                is_result_of_custom_autograd_fn = True

            if not isinstance(o, Tensor):
                output_type = OutputType.non_alias
                base_idx = None
            elif (
                curr_storage in inp_storage_refs
                and grad_fn is not None
                and is_result_of_custom_autograd_fn
            ):
                output_type = OutputType.custom_function_view
                base_idx = None
            elif (
                curr_storage in inp_storage_refs
                and not functional_tensor_storage_changed
            ):
                # pyrefly: ignore [bad-index, index-error]
                base_idx = inp_storage_refs[curr_storage]
                is_input_tensor = id(o) in inp_tensor_ids
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = (
                    num_aliased_tensors_that_are_multi_output_views[curr_storage]
                )
                num_aliased_outs_that_are_not_multi_output_views = (
                    num_aliased_outs - num_multi_output_view_outs
                )
                if (
                    grad_fn is not None
                    and num_aliased_outs_that_are_not_multi_output_views == 0
                ):
                    # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                    # In particular, given:
                    # def f(x):
                    #     return list(x.unbind(0))
                    # The main reason we ordinarily try to regenerate these output aliases outside of the
                    # compiled autograd.Function is because if any of the outputs are later mutated,
                    # autograd needs to perform view-replay to regenerate them.
                    # However, autograd does not allow users to mutate multi-output views
                    # in any way that can change the autograd metadata of other aliases.
                    # So we hide this aliasing from autograd here.
                    log.debug(
                        "Encountered AOTAutograd case: differentiable outputs that \
alias each other from a multi-output view call"
                    )
                    output_type = OutputType.non_alias
                elif is_input_tensor:
                    output_type = OutputType.is_input
                else:
                    output_type = OutputType.alias_of_input
            elif functional_tensor_storage_changed and id(o) in inp_tensor_ids:
                # When there is a set_() on an input, we cannot rely on checking storages
                # to detect if we are returning an input (since the inputs storage is different)
                if curr_storage is None:
                    raise AssertionError("curr_storage must not be None")
                base_idx = inp_storage_refs[curr_storage]
                output_type = OutputType.is_input

            # We only need to handle the intermediate base case when both
            # the intermediate base and the output require gradients.
            # See Note [AOT Autograd: outputs aliasing inputs or intermediates!]
            elif o._base is not None and o.requires_grad and o._base.requires_grad:
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = (
                    num_aliased_tensors_that_are_multi_output_views[curr_storage]
                )
                num_aliased_outs_that_are_not_multi_output_views = (
                    num_aliased_outs - num_multi_output_view_outs
                )
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                if (
                    out_tensor_alias_counts[curr_storage] == 1
                    or num_aliased_outs_that_are_not_multi_output_views <= 1
                ):
                    # Note [Intermediate Bases Optimization]
                    # Normally if we have an output that aliases an intermediate,
                    # we need to add the extra "intermediate base" logic further down
                    # to prevent autograd from yelling at us if the user later tries to
                    # mutate that output.
                    # However, the common case here is if we have an output that aliases an intermediate,
                    # but doesn't alias any other outputs.
                    # In that case, autograd shouldn't have to worry about the aliasing at all
                    # (if that output is mutated, there are no other live aliases for autograd to worry about).
                    # The "intermediate bases" can hurt inductor perf by forcing more variables to become outputs.
                    # So as an optimization, we won't do intermediate base handling in this case.
                    # Instead, we'll hide the aliasing from autograd using aten._unsafe_view().
                    if (
                        out_tensor_alias_counts[curr_storage] != 1
                        and num_aliased_outs_that_are_not_multi_output_views <= 1
                    ):
                        log.debug(
                            "Encountered AOTAutograd case: differentiable outputs that alias each other \
from a multi-output view call"
                        )
                    output_type = OutputType.unsafe_view_alias
                    base_idx = None
                else:
                    # First, check if o's ._base is an existing output
                    maybe_existing_out_idx = out_tensor_ids.get(id(o._base))
                    if maybe_existing_out_idx is not None:
                        # Special case where the output is an alias of a graph intermediate, but that intermediate
                        # is itself also a user output.
                        output_type = (
                            OutputType.alias_of_intermediate_base_is_user_output
                        )
                        base_idx = maybe_existing_out_idx
                    else:
                        # Next, check if o's ._base is an intermediate base that we already returned
                        maybe_existing_base_output_idx = (
                            intermediate_base_tensor_id_to_output_idx.get(id(o._base))
                        )
                        if maybe_existing_base_output_idx is not None:
                            output_type = OutputType.alias_of_intermediate
                            base_idx = maybe_existing_base_output_idx
                        else:
                            # Otherwise, take o._base and explicitly return it as an output in the compiled graph
                            new_out_idx = len(intermediate_bases)
                            base_idx = new_out_idx
                            # Indicate to the logic later on (when we trace the joint)
                            # that this particular output should get it's ._base appended to the forward graph outputs
                            output_type = (
                                OutputType.alias_of_intermediate_save_as_output
                            )
                            intermediate_base_tensor_id_to_output_idx[id(o._base)] = (
                                new_out_idx
                            )
                            intermediate_bases.append(o._base)
                            # NB: The desc we picked here is guaranteed to be
                            # synchronized with the one in
                            # graph_capture_wrappers.py because we
                            # SPECIFICALLY notated this output as
                            # alias_of_intermediate_save_as_output
                            intermediate_bases_descs.append(
                                TangentAOTInput(IntermediateBaseAOTOutput(desc))
                            )
            elif (
                # See https://github.com/pytorch/pytorch/issues/100348 for this case.
                # This protects against the specific case where a user fn returns (output, output.detach())
                out_tensor_alias_counts[curr_storage] > 1
                and len(outs_with_identical_metadata_that_require_grad) > 0
                and not o.requires_grad
            ):
                # In theory we could use any of these tensors to regenerate the aliased outputs from,
                # since they all alias each other and have identical metadata
                out_alias = outs_with_identical_metadata_that_require_grad[0]
                existing_out_idx = out_tensor_ids[id(out_alias)]
                output_type = OutputType.alias_of_intermediate_base_is_user_output
                base_idx = existing_out_idx
            else:
                output_type = OutputType.non_alias
                base_idx = None

            if isinstance(o, torch.Tensor):
                dynamic_dims = {
                    i for i, s in enumerate(o.shape) if not is_concrete_int(s)
                }
            else:
                dynamic_dims = None

            # Save the current FunctionalTensor output.
            #
            # This will be used at runtime for reconstructing output views from
            # their respective base tensors.
            #
            # The FunctionalTensor will be saved if one of the 2 conditions below
            # is true:
            view_meta_sequence = None
            if (
                # 1. If the output_type is either of:
                #    (i) alias_of_intermediate;
                #    (ii) alias_of_intermediate_save_as_output; or
                #    (iii) alias_of_intermediate_base_is_user_output.
                #
                # No need to worry about in-place view operations here, since
                # this functionalization step eliminates mutations.
                #
                # i.e. we have access to the actual base tensor, before the
                # in-place operation was applied.
                output_type
                in (
                    OutputType.alias_of_intermediate,
                    OutputType.alias_of_intermediate_save_as_output,
                    OutputType.alias_of_intermediate_base_is_user_output,
                )
            ) or (
                # 2. If the output_type is alias_of_input, and no in-place view
                #    operation was run on the input (base tensor).
                #
                # In this case, we need to check for metadata mutation because
                # the runtime explicitly reconstructs the inputs, before actually
                # reconstructing the outputs. Due to in-place view operations, the
                # fully reconstructed input may not be this output base tensor
                # anymore.
                output_type == OutputType.alias_of_input
                and base_idx is not None
                and not input_info[base_idx].mutates_metadata
            ):
                if isinstance(o, FunctionalTensor):
                    view_meta_sequence = ViewMetaSequence(o)

            requires_grad = isinstance(o, torch.Tensor) and o.requires_grad
            out_info = OutputAliasInfo(
                output_type=output_type,
                raw_type=type(o),
                base_idx=base_idx,
                dynamic_dims=dynamic_dims,
                requires_grad=requires_grad,
                # A view created under no_grad() inherits requires_grad from
                # its base but has no grad_fn and does not participate in
                # differentiation.
                requires_grad_for_backward=requires_grad
                and (o._base is None or grad_fn is not None),
                view_meta_sequence=view_meta_sequence,
            )
            output_info.append(out_info)

        # See Note [AOT Autograd: Views to avoid tangents aliasing inputs]
        def view_avoid_dupes_with_primals(t: object) -> object:
            if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
                return transform_subclass(
                    t, lambda _, inner_t: view_avoid_dupes_with_primals(inner_t)
                )
            if isinstance(t, Tensor):
                return t.view(t.shape)
            return t

        # This analysis function returns *only* the outputs that are meant to be tangents to the backwards.
        # Anything that aliases (inputs returned in the fw due to metadata mutations, or outputs that alias inputs/intermediates)
        # are *regenerated* later, and not used directly in the autograd graph
        def _plain_fake_tensor_like_subclass(x: Any) -> torch.Tensor:
            # pyrefly: ignore [bad-context-manager]
            with detect_fake_mode():
                return torch.empty(
                    x.shape, dtype=x.dtype, device=x.device, layout=x.layout
                )

        def _is_subclass_mutated_input_tangent_always_subclass(inp: object) -> bool:
            return (
                isinstance(inp, torch.nested._internal.nested_tensor.NestedTensor)
                or torch._functorch.config.disable_guess_zero_tangent_for_mutated_input_subclass
            )

        f_input_tangents_pairs = [
            # Note: [AOTAutograd Tangent Subclassness for mutated inputs]
            # Generally when creating tangents to trace with, we assume that tangents will have
            # the same subclass-ness as their forward outs
            # however: for tangents that correspond to input mutations, in practice it is more likely
            # that these tangents will be plain tensors of zeros at runtime, so we tweak our guess
            # to assume that these tangents should always be plaint tensors.
            # Example:
            #  def f(x):
            #      x.mul_(2)
            #      return x + 1
            #  out = f(x)
            #  out.sum().backward()
            # In the above code, we will have a tangent "x_updated_tangent",
            # which will be a plain tensor of zeros, *unless* x is used in some compute after executing f
            #
            # However, there are exceptions to this logic. If a view is created from mutated input and is used in backward,
            # The tangent for this subclass input will be a subclass tensor.
            # Example:
            #  def f(a, b):
            #      a.mul_(2)
            #      b.mul_(3)
            #      return b.view(b.shape), a + b
            # a_out, b_out = f(..., Subclass)
            # (a * b).sum().backward()
            #
            # We can not deduce it easily now, so introducing a debug config to be able to turn off this for specific cases.
            # NJT guarantees to have its tangent as NJT, because it has dedicated integration in Autograd
            # See torch/csrc/autograd/python_function.cpp, use_zeros_like.
            (
                (
                    _plain_fake_tensor_like_subclass(inp)
                    if is_traceable_wrapper_subclass(inp)
                    and not _is_subclass_mutated_input_tangent_always_subclass(inp)
                    else inp
                ),
                TangentAOTInput(InputMutationAOTOutput(inp_desc)),
            )
            for inp, inp_desc, info in zip(flat_f_args, flat_f_args_descs, input_info)
            if info.mutation_type == MutationType.MUTATED_OUT_GRAPH
            and info.mutates_data
            and info.requires_grad
        ]
        f_input_tangents, f_input_tangents_descs = (
            [x[0] for x in f_input_tangents_pairs],
            [x[1] for x in f_input_tangents_pairs],
        )

        f_output_tangents_pairs = [
            (o, TangentAOTInput(desc))
            for o, info, desc in zip(flat_f_outs, output_info, flat_f_outs_descs)
            if info.output_type
            in [
                OutputType.non_alias,
                OutputType.unsafe_view_alias,
                OutputType.custom_function_view,
            ]
            and issubclass(info.raw_type, torch.Tensor)
            and info.requires_grad_for_backward
        ]
        f_output_tangents, f_output_tangents_descs = (
            [x[0] for x in f_output_tangents_pairs],
            [x[1] for x in f_output_tangents_pairs],
        )

        # intermediate bases are also included in the backward graph
        f_tangents = f_input_tangents + f_output_tangents + intermediate_bases
        f_tangents_descs = (
            f_input_tangents_descs + f_output_tangents_descs + intermediate_bases_descs
        )

        # TODO: I'm pretty sure you don't need a tree_map here
        traced_tangents = pytree.tree_map(from_fun, f_tangents)
        traced_tangents = pytree.tree_map(
            view_avoid_dupes_with_primals, traced_tangents
        )
        traced_tangents = [
            coerce_tangent_and_suggest_memory_format(tt)[0]
            for i, tt in enumerate(traced_tangents)
        ]
        # NB: update this if the maps above ever change structure.
        # Also, it might be helpful to add coercion information to the tangent desc!
        traced_tangents_descs = f_tangents_descs

        nonlocal static_input_indices
        static_input_indices = static_input_indices or []
        if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
            passed_indices = set(static_input_indices)
            static_input_indices = [
                i
                for i, arg in enumerate(flat_args)
                if (isinstance(arg, torch.nn.Parameter) or i in passed_indices)
            ]

        static_input_logger.debug(
            "static input indices metadata analysis: %s", static_input_indices
        )

        f_mutated_inputs = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutation_type == MutationType.MUTATED_OUT_GRAPH
        ]
        # Build the full list of forward graph outputs so the subclass wrapping
        # code knows exactly which graph outputs to wrap back into subclasses.
        # Including intermediate_bases unconditionally is safe: they are only
        # populated when outputs require grad (line ~539), so they are naturally
        # empty during pure inference.  In the "downgrade from training to
        # inference" path, num_intermediate_bases > 0 is already gated behind
        # `assert not req_subclass_dispatch` (aot_autograd.py), so the subclass
        # wrapping code that consumes subclass_fw_graph_out_meta never sees them.
        f_fw_graph_outs = [*f_mutated_inputs, *flat_f_outs, *intermediate_bases]
        fw_graph_outs = pytree.tree_map(from_fun, f_fw_graph_outs)

        grad_enabled_mutation = None
        if torch.is_grad_enabled() != prior_grad_enabled:
            grad_enabled_mutation = torch.is_grad_enabled()
            torch.set_grad_enabled(
                prior_grad_enabled
            )  # Restore the prior state after tracing it
            log.debug(
                (
                    "grad_mode mutation encountered in graph. "
                    "Will emit mutation epilogue, to set grad_mode=%s"
                ),
                grad_enabled_mutation,
            )

        subclass_inp_meta = create_subclass_meta(flat_args)
        subclass_fw_graph_out_meta = create_subclass_meta(fw_graph_outs)
        subclass_tangent_meta = create_subclass_meta(
            traced_tangents, count_symints=False, with_memory_format=True
        )

        metadata = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=len(intermediate_bases),
            keep_input_mutations=keep_input_mutations,
            traced_tangents=traced_tangents,
            traced_tangents_descs=traced_tangents_descs,
            subclass_inp_meta=subclass_inp_meta,
            subclass_fw_graph_out_meta=subclass_fw_graph_out_meta,
            subclass_tangent_meta=subclass_tangent_meta,
            grad_enabled_mutation=grad_enabled_mutation,
            static_input_indices=static_input_indices,
            tokens=mode._tokens,
        )
        return metadata

    return inner
<|endoftext|>"""
This file contains utilities related to functionalization in AOTAutograd:
1. converting to/from functional tensors
2. detecting Tensor mutations - both metadata and Tensor value
3. regenerating/replaying views from their base
4. checking if a graph is functional i.e. whether it contains any mutation ops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeGuard

import torch
from torch import Tensor
from torch._C import _functionalization
from torch._custom_class_base import CustomClassBase
from torch._logging import getArtifactLogger
from torch._subclasses.fake_tensor import is_fake_tensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq, SymIntEqByExpr
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    transform_subclass,
)


aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")


def to_fun(t: object) -> Any:
    if isinstance(t, Tensor):
        if is_traceable_wrapper_subclass(t):
            # See Note [Functionalization always runs last]
            # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
            # goes at the bottom.
            # recurse here, so we can support nested wrapper subclasses
            out = transform_subclass(t, lambda _, inner_t: to_fun(inner_t))
            torch._mirror_autograd_meta_to(t, out)  # type: ignore[attr-defined]
            return out
        else:
            return FunctionalTensor.to_functional(t)
    else:
        return t


def sync_functional_tensor(t: torch.Tensor) -> None:
    if is_traceable_wrapper_subclass(t):
        attrs, _ctx = t.__tensor_flatten__()  # type: ignore[attr-defined]
        for attr in attrs:
            match getattr(t, attr):
                case Tensor() as inner:
                    sync_functional_tensor(inner)
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
    else:
        torch._sync(t)


# When subclasses are involved, t here will usually look something like:
# SubclassA(SubclassB(FunctionalTensor(_to_fun_tensor(FakeTensor))))
def from_fun(t: object) -> object:
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # See Note [Functionalization always runs last]
        # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
        # goes at the bottom.
        # recurse here, so we can support nested wrapper subclasses
        out = transform_subclass(t, lambda _, inner_t: from_fun(inner_t))
        torch._mirror_autograd_meta_to(t, out)  # type: ignore[attr-defined]
        return out

    if not isinstance(t, FunctionalTensor):
        # quick sanity assert
        if isinstance(t, torch.Tensor):
            if torch._is_functional_tensor(t):  # type: ignore[attr-defined]
                raise AssertionError("expected non-functional tensor")
        return t
    sync_functional_tensor(t)
    return torch._from_functional_tensor(t.elem)


def is_fun(t: object) -> TypeGuard[FunctionalTensor | Tensor]:
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # See Note [Functionalization always runs last]
        # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
        # goes at the bottom.
        # recurse here, so we can support nested wrapper subclasses
        t_attrs, _ = t.__tensor_flatten__()  # type: ignore[attr-defined]
        got_fun: bool | None = None
        for attr in t_attrs:
            match getattr(t, attr):
                case Tensor() as v:
                    fun = is_fun(v)
                    if got_fun is None:
                        got_fun = fun
                    elif got_fun != fun:
                        raise AssertionError(
                            "mixed functional/non-functional inner tensors"
                        )
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return got_fun or False

    return isinstance(t, FunctionalTensor)


# t here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
# (3) Not a tensor
def has_data_mutation(t: object) -> bool:
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        # A tensor subclass was updated if any of its inner elements were updated
        for attr in attrs:
            match getattr(t, attr):
                case Tensor() as v:
                    if has_data_mutation(v):
                        return True
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return False
    else:
        if isinstance(t, torch.Tensor):
            if not isinstance(t, FunctionalTensor):
                raise AssertionError(f"expected FunctionalTensor, got {type(t)}")
            return torch._functionalize_has_data_mutation(t.elem)  # type: ignore[attr-defined]
        return False


def are_all_mutations_hidden_from_autograd(t: object) -> bool:
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        # If all inner elements are mutations hidden from autograd, then it is a mutation hidden from autograd.
        for attr in attrs:
            match getattr(t, attr):
                case Tensor() as v:
                    if not are_all_mutations_hidden_from_autograd(v):
                        return False
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return True
    elif isinstance(t, torch.Tensor):
        if not isinstance(t, FunctionalTensor):
            raise AssertionError(f"expected FunctionalTensor, got {type(t)}")
        return torch._functionalize_are_all_mutations_hidden_from_autograd(t.elem)
    else:
        return False


def are_all_mutations_under_no_grad_or_inference_mode(t: torch.Tensor) -> bool:
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        for attr in attrs:
            match getattr(t, attr):
                case Tensor() as v:
                    if not are_all_mutations_under_no_grad_or_inference_mode(v):
                        return False
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return True
    else:
        if not isinstance(t, FunctionalTensor):
            raise AssertionError(f"expected FunctionalTensor, got {type(t)}")
        return torch._functionalize_are_all_mutations_under_no_grad_or_inference_mode(
            t.elem
        )


def was_inductor_storage_resized(t: object) -> bool:
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        for attr in attrs:
            match getattr(t, attr):
                case Tensor() as v:
                    if was_inductor_storage_resized(v):
                        raise RuntimeError(
                            f"storage resizing is not supported on tensor subclass: {type(t)}"
                        )
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return False
    elif not isinstance(t, torch.Tensor):
        return False
    else:
        if not isinstance(t, FunctionalTensor):
            raise AssertionError(f"expected FunctionalTensor, got {type(t)}")
        return torch._functionalize_was_inductor_storage_resized(t.elem)


def was_shallow_copy_data(t: object) -> bool:
    if is_traceable_wrapper_subclass(t):
        return False
    if not isinstance(t, torch.Tensor):
        return False
    if not isinstance(t, FunctionalTensor):
        raise AssertionError(f"expected FunctionalTensor, got {type(t)}")
    return torch._functionalize_was_shallow_copy_data(t.elem)  # type: ignore[attr-defined]


# f_arg here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
# (3) Not a tensor
# Assumption: arg promises to be the "original" tensor wrapped by f_arg
# Note: "storage mutations" coming from set_() are a type of metadata mutation. So:
# - check_only_storage_mutation=True: only return true if there was a storage mutation
# - check_only_storage_mutation=False: return true if there was any metadata mutation (including a storage mutation)
def has_metadata_mutation(
    f_arg: object, arg: object, *, check_only_storage_mutation: bool
) -> bool:
    if is_traceable_wrapper_subclass(f_arg):
        attrs, _ = f_arg.__tensor_flatten__()
        # A tensor subclass was updated if any of its inner elements were updated
        for attr in attrs:
            match getattr(f_arg, attr):
                case Tensor():
                    f_inner_t = getattr(f_arg, attr)
                    inner_t = getattr(arg, attr)
                    if has_metadata_mutation(
                        f_inner_t,
                        inner_t,
                        check_only_storage_mutation=check_only_storage_mutation,
                    ):
                        return True
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return False
    else:
        if not isinstance(f_arg, torch.Tensor):
            if isinstance(arg, torch.Tensor):
                raise AssertionError(
                    f"f_arg is not a Tensor but arg is: {type(f_arg)} vs {type(arg)}"
                )
            return False
        if not isinstance(f_arg, FunctionalTensor):
            raise AssertionError(
                f"expected FunctionalTensor for f_arg, got {type(f_arg)}"
            )
        if not is_fake_tensor(arg):
            raise AssertionError(f"expected FakeTensor for arg, got {type(arg)}")

        arg_after = torch._from_functional_tensor(f_arg.elem)
        # This is true if the current tensor experienced at least one set_() call
        maybe_storage_changed = torch._functionalize_was_storage_changed(f_arg.elem)  # type: ignore[attr-defined]
        # However, multiple set_() calls can cancel out. So we also check whether the
        # storage of the tensor has changed.
        # Note: if an input experienced two set_() calls that cancel out, **and**
        # it experiences a data mutation, we pessimistically think that the set_()
        # call is necessary here. We could in theory fix this, but this will
        # hopefully never happen in user code, and is not needed for fsdp.
        if is_sparse_any(arg):
            # TODO:add sparse tensors support to functionalization
            same_storages = False
        else:
            same_storages = StorageWeakRef(arg.untyped_storage()) == StorageWeakRef(
                arg_after.untyped_storage()
            )
        has_storage_metadata_mutation = maybe_storage_changed and not same_storages
        if check_only_storage_mutation:
            return has_storage_metadata_mutation

        # storage metadata mutation is a type of metadata mutation, so return true if we saw one
        if has_storage_metadata_mutation:
            return True

        maybe_metadata_mutated = torch._functionalize_has_metadata_mutation(f_arg.elem)  # type: ignore[attr-defined]
        # This is true if the current tensor experienced at least one metadata mutation.
        # So if false, we know there was no metadata mutation
        if not maybe_metadata_mutated:
            return False

        # However, multi metadata mutations can cancel out.
        # So we also check if the concrete sizes/strides on the tensor have changed.
        same_sizes = arg.shape == arg_after.shape
        same_strides = arg.stride() == arg_after.stride()
        same_offsets = arg.storage_offset() == arg_after.storage_offset()
        has_metadata_mutation_ = maybe_metadata_mutated and not (
            same_sizes and same_strides and same_offsets
        )
        # We consider a tensor to have been metadata mutated if its storage was mutated through a set_() call.
        return has_metadata_mutation_


def gen_alias_from_base(
    aliased_base_tensor: Tensor,
    target_meta_tensor: Tensor,
    target_requires_grad: bool,
    target_view_meta_sequence: ViewMetaSequence | None = None,
    *,
    replay_views: bool,
) -> Tensor:
    # Patch the correct requires_grad field of the output tensor, depending on whether:
    # (i) the reconstructed output (out) was came from a tensor that requires grad or not;
    # and (ii) the concrete returned output does require grad or not.
    def patch_requires_grad(out: Tensor) -> Tensor:
        if aliased_base_tensor.requires_grad and not target_requires_grad:
            out = out.detach()
        elif not aliased_base_tensor.requires_grad and target_requires_grad:
            out.requires_grad_(True)
        return out

    # If provided, use the target functional tensor for replaying the views.
    #
    # In summary, we use the fact that FunctionalTensorWrapper saves the view
    # functions applied to itself (collected during functionalization) so as
    # to replay them (view functions) on the aliased_base_tensor.
    if (
        replay_views
        and target_view_meta_sequence is not None
        and not any(vm.has_symbolic_inputs for vm in target_view_meta_sequence.sequence)
    ):
        out = _functionalization.apply_view_meta_sequence(
            aliased_base_tensor, target_view_meta_sequence.sequence
        )
        # If re-applying the ViewMeta sequence succeeded, there should be no more
        # problems going forward. We just check we got to the target shape and
        # patch requires_grad flag.
        if out.shape != target_meta_tensor.shape:
            raise AssertionError(
                "incorrect out shape after application of ViewMeta sequence: "
                f"{tuple(out.shape)} (actual) vs {tuple(target_meta_tensor.shape)} (expected)"
            )
        return patch_requires_grad(out)

    # Try to do view-replay if possible.
    # fall back to .as_strided() if we can't.
    if target_meta_tensor._base is not None:
        # The base that we want to replay our view off of might have a different shape than the view's original base.
        b = target_meta_tensor._base
        abt = aliased_base_tensor
        # Don't unnecessarily call as_strided if nothing changed; as_strided's
        # backward is poorly implemented and slow
        if abt is not b and (
            abt.size() != b.size()
            or abt.stride() != b.stride()
            or abt.storage_offset() != b.storage_offset()
        ):
            reshaped_base_tensor = aliased_base_tensor.as_strided(
                b.size(), b.stride(), b.storage_offset()
            )
        else:
            reshaped_base_tensor = aliased_base_tensor
        out = target_meta_tensor._view_func(reshaped_base_tensor)  # type: ignore[attr-defined]
        # This shape mismatch can happen due to a bug in inplace/view handling in autograd.
        # Try putting a breakpoint here and running
        # `test/functorch/test_aotdispatch TestAOTAutograd.test_output_all_alias_types`
        # Also, https://github.com/pytorch/pytorch/issues/49825
        #
        # As a stopgap, we'll fall back to as_strided.
        if out is not None and out.shape == target_meta_tensor.shape:
            return patch_requires_grad(out)

    size = target_meta_tensor.size()
    stride = target_meta_tensor.stride()
    storage_offset = target_meta_tensor.storage_offset()
    # If the target lives on a different storage than the aliased base
    # (e.g. because inductor's copy_misaligned_inputs cloned the input to
    # obtain an aligned buffer), ``target.storage_offset()`` is expressed in
    # the cloned storage and would pick the wrong slice when applied via
    # ``as_strided()`` on the original aliased base tensor. Translate the
    # offset: the traced FakeTensor's storage_offset equals the trace-time
    # RELATIVE offset from the input, so add back the runtime input's
    # ``storage_offset`` to keep the alias anchored to the correct slice.
    # Compare storages via ``_cdata`` (raw c10::Storage handle) rather than
    # ``.data_ptr()`` so this is safe on fake/meta storages that would raise
    # from ``.data_ptr()`` during AOT tracing.
    if (
        aliased_base_tensor.untyped_storage()._cdata
        != target_meta_tensor.untyped_storage()._cdata
    ):
        storage_offset = aliased_base_tensor.storage_offset() + storage_offset
    if aliased_base_tensor.is_complex() and not target_meta_tensor.is_complex():
        aliased_out = torch.view_as_real(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    elif not aliased_base_tensor.is_complex() and target_meta_tensor.is_complex():
        aliased_out = torch.view_as_complex(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    else:
        aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
    # For outputs aliasing inputs, we need to check if the requires-gradness has changed.
    aliased_out = patch_requires_grad(aliased_out)
    # For outputs aliasing inputs, we need to check if the dtype has changed.
    # as_strided() is the "most generic" view, but it does not cover cross-dtype views
    if aliased_out.dtype != target_meta_tensor.dtype:
        aliased_out = aliased_out.view(target_meta_tensor.dtype)
    return aliased_out


def has_same_metadata(t1: Tensor, t2: Tensor) -> bool:
    return (
        guard_or_false(sym_eq(t1.size(), t2.size()))
        and guard_or_false(t1.layout == t2.layout)
        and (
            is_sparse_any(t1)
            or (
                guard_or_false(sym_eq(t1.stride(), t2.stride()))
                and guard_or_false(t1.storage_offset() == t2.storage_offset())
            )
        )
        and t1.is_conj() == t2.is_conj()
        and t1.is_neg() == t2.is_neg()
    )


@dataclass(frozen=True)
class MetadataKey:
    """
    This should be equal whenever has_same_metadata would return True
    """

    size: tuple[SymIntEqByExpr, ...]
    layout: torch.layout
    is_sparse: bool
    # these are empty when is_sparse
    stride: tuple[SymIntEqByExpr, ...] | None
    storage_offset: SymIntEqByExpr | None
    is_conj: bool
    is_neg: bool

    @staticmethod
    def make(t: Tensor) -> MetadataKey:
        is_sparse = is_sparse_any(t)
        return MetadataKey(
            size=tuple(SymIntEqByExpr(s) for s in t.size()),
            layout=t.layout,
            is_sparse=is_sparse,
            stride=None if is_sparse else tuple(SymIntEqByExpr(s) for s in t.stride()),
            storage_offset=None if is_sparse else SymIntEqByExpr(t.storage_offset()),
            is_conj=t.is_conj(),
            is_neg=t.is_neg(),
        )


# ViewMeta sequence wrapper for equality comparisons.
#
# Even though we can compare each ViewMeta instance, we compare the resulting
# tensor metadata, instead. That's because the creation of synthetic bases + the
# re-generation of input views might end-up creating a different sequence of
# ViewMeta that is semantically equivalent. i.e. gets to a tensor with the same
# metadata.
#
# Therefore, we store what the end result should look like as serializable
# metadata.
#
# When logging, this class should look like:
#
#     ViewMetaSequence(view, select_int, slice_Tensor)
#
# i.e. a parenthesized list of view operations within that ViewMeta sequence.
class ViewMetaSequence:
    def __init__(self, tensor: FunctionalTensor) -> None:
        if not torch._is_functional_tensor(tensor.elem):
            raise AssertionError("expected tensor.elem to be a functional tensor")
        self.sequence = _functionalization.get_view_meta_sequence(tensor.elem)
        self.metadata = MetadataKey.make(tensor)

    def __repr__(self) -> str:
        suffix = len("_ViewMeta")
        types = ", ".join(type(vm).__name__[:-suffix] for vm in self.sequence)
        return f"ViewMetaSequence({types})"

    def __eq__(self, other: object) -> bool:
        # If other is None, then it probably means that we weren't able to recreate
        # the ViewMeta sequence. One example is when we update the view metadata by
        # calling: create_synthetic_base_metadata.
        if other is None:
            return True

        # Comparison against any other type is not implemented.
        if not isinstance(other, ViewMetaSequence):
            return NotImplemented

        return self.metadata == other.metadata

    @classmethod
    def _from_parts(
        cls, sequence: list[_functionalization.ViewMeta], metadata: MetadataKey
    ) -> ViewMetaSequence:
        # Rebuild a ViewMetaSequence directly from its parts, bypassing the
        # FunctionalTensor-based __init__. This lets the recipe be reconstructed from
        # plain values rather than from a live FunctionalTensor or an embedded pickle.
        # Sole caller: torch._functorch._aot_autograd.source_emit, when baking a
        # ViewMetaSequence into standalone source; keep the attributes set here in sync
        # with __init__ (sequence, metadata) or the reconstructed object diverges.
        self = cls.__new__(cls)
        self.sequence = sequence
        self.metadata = metadata
        return self


# new_arg and arg here are either:
# (1) both a FakeTensor
# (2) both a traceable tensor subclass that holds a FakeTensor
# Pre-condition: the two args are the "old" and "new" inputs from running functionalization.
# When we run functionalization and wrap our inputs into FunctionalTensors,
# we can detect whether or not an input was mutated by checking to see if the inner tensor has changed
#
# Normally it would be enough just to check if arg is new_arg, which is normally enough for functionalization
# to confirm that inputs were not mutated when running the user's model with functionalization on.
# But when we have subclass inputs, we can't rely on that:
# `from_fun(to_fun(x)) is x` will return False, because the call to `from_fun` constructs
# a brand new subclass instance: we are calling __tensor_unflatten__, and going
# from Subclass(FakeTensor) to Subclass(FunctionalTensor(FakeTensor))
def was_tensor_updated(arg: torch.Tensor, new_arg: torch.Tensor) -> bool:
    if is_traceable_wrapper_subclass(arg):
        if not is_traceable_wrapper_subclass(new_arg):
            raise AssertionError(
                f"expected new_arg to be traceable wrapper subclass, got {type(new_arg)}"
            )
        attrs, _ = arg.__tensor_flatten__()
        new_attrs, _ = new_arg.__tensor_flatten__()
        if attrs != new_attrs:
            raise AssertionError(f"attrs mismatch: {attrs} != {new_attrs}")
        # A tensor subclass was updated if any of its inner elements were updated
        for attr in attrs:
            match getattr(arg, attr):
                case Tensor() as v:
                    if was_tensor_updated(v, getattr(new_arg, attr)):
                        return True
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return False
    else:
        return arg is not new_arg


# new_arg and arg here are either:
# (1) both a FakeTensor
# (2) both a traceable tensor subclass that holds a FakeTensor
# Pre-condition: the two args are the "old" and "new" inputs from running functionalization.
# When we run functionalization and wrap our inputs into FunctionalTensors,
# we can detect whether or not an input was mutated by checking to see if the inner tensor has changed,
# but shares storage with the old input
def was_tensor_metadata_updated(arg: Any, new_arg: Any) -> bool:
    if is_traceable_wrapper_subclass(arg):
        if not is_traceable_wrapper_subclass(new_arg):
            raise AssertionError(
                f"expected new_arg to be traceable wrapper subclass, got {type(new_arg)}"
            )
        attrs, _ = arg.__tensor_flatten__()
        new_attrs, _ = new_arg.__tensor_flatten__()
        if attrs != new_attrs:
            raise AssertionError(f"attrs mismatch: {attrs} != {new_attrs}")
        # A tensor subclass was updated if any of its inner elements were updated
        for attr in attrs:
            match getattr(arg, attr):
                case Tensor() as v:
                    if was_tensor_metadata_updated(v, getattr(new_arg, attr)):
                        return True
                case CustomClassBase():
                    pass
                case unexpected:
                    raise AssertionError(
                        f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                    )
        return False
    else:
        return arg is not new_arg and StorageWeakRef(
            arg.untyped_storage()
        ) == StorageWeakRef(new_arg.untyped_storage())


# Returns the number of detected copy_
def _is_functional_graph(fx_g: torch.fx.Graph) -> tuple[str | None, int]:
    allowed_mutation_ops = [
        torch.ops.aten.copy_.default,
        torch.ops.aten.set_.source_Tensor,
        torch.ops.aten.shallow_copy_data_.default,
    ]
    if hasattr(torch.ops.fsdp, "copy_"):
        allowed_mutation_ops.append(torch.ops.fsdp.copy_.default)

    placeholders = set()
    mutation_count = 0
    # NB: It would also be nice to verify that the mutations all happen at the
    # end, but we also do some administrative views after mutations so this
    # isn't actually true.  (TODO: Could this cause problems for Inductor?)
    error = None
    for n in fx_g.nodes:
        if n.op == "placeholder":
            placeholders.add(n)
        if isinstance(n.target, torch._ops.OpOverload):
            if n.target in allowed_mutation_ops:
                # Can only copy_/set_ into an input
                # this is mostly a hack to avoid failing XLA tests.
                # See https://github.com/pytorch/pytorch/pull/122434#issuecomment-2101012113
                if "set_buffer_donor_" not in str(n.args[0]):
                    if n.args[0] not in placeholders:
                        error = f"n={str(n)}, n.args[0]={str(n.args[0])}, placeholders={str(placeholders)}, graph={str(fx_g)}"
                mutation_count += 1
            else:
                if n.target._schema.is_mutable:
                    error = f"aot_autograd expected to have an entirely functional graph, but found {n.format_node()}"
    return error, mutation_count


def assert_functional_graph(fx_g: torch.fx.Graph) -> int:
    error, mutation_count = _is_functional_graph(fx_g)
    if error is not None:
        raise AssertionError(error)
    return mutation_count


def propagate_input_mutation_stacktraces(fx_g: torch.fx.Graph) -> None:
    placeholders = set()
    for n in fx_g.nodes:
        if n.op == "placeholder":
            placeholders.add(n)
        if isinstance(n.target, torch._ops.OpOverload):
            if n.target is torch.ops.aten.copy_.default:
                # Can only copy_ into an input, and can only do so once
                if "set_buffer_donor_" not in str(n.args[0]):
                    if n.args[0] not in placeholders:
                        raise AssertionError(
                            f"n={str(n)}, n.args[0]={str(n.args[0])}, placeholders={str(placeholders)}, graph={str(fx_g)}"
                        )
                    placeholders.remove(n.args[0])
                copy_from_node = n.args[1]
                # Pre-condition: every node has a "stack_trace" field in its meta,
                # but copy_() nodes do not (since we manually added them during functionalization).
                # Instead, we manually propagate here.
                if "stack_trace" in copy_from_node.meta:
                    n.meta["stack_trace"] = copy_from_node.meta["stack_trace"]


def _check_if_mutation_can_be_in_graph(
    keep_input_mutations: bool,
    mutates_data: bool,
    mutates_metadata: bool,
    mutations_hidden_from_autograd: bool,
    mutations_under_no_grad_or_inference_mode: bool,
    mutates_storage_metadata: bool,
    mutation_inductor_storage_resize: bool,
    requires_grad: bool,
) -> bool:
    if keep_input_mutations:
        in_graph = (
            mutates_data or mutates_storage_metadata or mutation_inductor_storage_resize
        ) and (
            (not mutates_metadata and not requires_grad)
            or mutations_hidden_from_autograd
            or mutations_under_no_grad_or_inference_mode
        )
    else:
        in_graph = False
    # See Note [set_() Input Mutations in AOTAutograd]
    # If there was a `set_()`, we require that all mutations were under no_grad,
    # so we can (safely) emit the set_() in the graph at runtime
    # resize_() gets the same treatment
    if mutation_inductor_storage_resize or mutates_storage_metadata:
        op_name = "resize_" if mutation_inductor_storage_resize else "set_"
        if not in_graph:
            raise AssertionError(f"""\
Encountered a {op_name} on a graph input, but the input has other mutations that we cannot
keep in the graph. This is not supported today. Current state:
  keep_input_mutations={keep_input_mutations}
  mutates_data={mutates_data}
  mutates_metadata={mutates_metadata}
  mutations_hidden_from_autograd={mutations_hidden_from_autograd}
  mutations_under_no_grad_or_inference_mode={mutations_under_no_grad_or_inference_mode}
  mutation_inductor_storage_resize={mutation_inductor_storage_resize}
  requires_grad={requires_grad}""")
    return in_graph
<|endoftext|>"""
AOTAutograd descriptors are a path-like data structure (similar to pytree
paths and sources) that describe the semantic meaning of an input/output to FX
graphs.  Although you may know the input/output meaning at the top level of
the original function you traced, because we have many graph capture wrappers
that change the calling convention, it can be difficult to tell how these
correspond to the actual FX graph you get back, to say nothing about the extra
arguments/outputs for tangents, gradients, etc.  Descriptors describe the meaning
of arguments.

Examples
--------

Before we talk about the precise semantics, it's helpful to look at some
examples to get some intuition for the meaning of descriptors.  Here are some
input descriptors you might find on the joint FX graph:

* PlainAOTInput(idx=0) - the first input from the original callable, as is

* ParamAOTInput(target="mod.weight") - the parameter with FQN mod.weight

* TangentAOTInput(output=PlainAOTOutput(idx=1)) - the input tangent
  corresponding to the gradients for the second output in the forward graph

* ViewBaseAOTInput(base_of=PlainAOTInput(idx=0)) - it turned out the first
  input was actually a (differentiable) view of a tensor which aliased with
  another input tensor.  We replaced this input with a single input for the
  base of all of these inputs, replacing the original inputs (one of which is
  mentioned in base_of).  We would generate a GradAOTOutput for *this* input
  (and not the original PlainAOTInputs!)  If you have a joint graph where a
  view base like this is undesirable, you can eliminate this by cloning
  the views outside of the compiled region (assuming you aren't mutating this
  tensor).

* SubclassGetAttrAOTInput(base=AOTInput(idx=0), attr="inner") - this tensor
  corresponds to the "inner" tensor from the tensor subclass that is at the
  first index.  In general, joint graphs from AOTAutograd never take tensor
  subclasses as inputs; they are always unpacked into their constituent plain
  tensor pieces; use the descriptors to identify the parts of the tensor that
  are related.  Note that this can be nested (if you have nested tensor
  subclasses!)

Here are some output descriptors you might find on the Joint FX graph:

* PlainAOTOutput(idx=0) - the first output from the original forward function,
  as is

* GradAOTOutput(grad_of=PlainAOTInput(idx=1)) - the computed gradient for the
  second input to the graph, an output of the backward graph

* InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=0)) - when the first
  input is mutated, the new value to be copied into the first input of the
  graph.  Sometimes, these outputs can be elided and the ``copy_`` is done directly
  in the graph (controlled by keep_input_mutations), but if the input
  mutation must be differentiated through we always generate an output like this

* IntermediateBaseAOTOutput(base_of=PlainAOTOutput(idx=0)) - if we return
  multiple outputs which alias each other, we instead replace them with a single
  output tensor representing the base of all the aliases.  This output indicates
  it is the base for /one/ of those original outputs.  If this is undesirable in
  the joint graph, clone all outputs before returning from the graph.

* SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), idx="inner") - this
  tensor corresponds to the inner tensor of the first original output which
  is a tensor subclass.  This and other subclass components of that output will
  get repacked into a tensor subclass.

High level semantics
--------------------

OK, let's formally define a descriptor.  Intuitively, suppose we have::

    def wrapped_graph(*args):
        ret = graph(*in_transform(args))
        return out_transform(ret)

Then the descriptor for input[i] to graph describes a function fin_i such that::

    fin_i(args) == in_transform(args)[i]

and the descriptor for output[j] from graph describes a function fout_j such that::

    fout_j(out_transform(ret)) == ret[j]

AKA input descriptors tell you how to get from outer inputs to inner inputs,
while output descriptors tell you how to get from outer outputs to inner
outputs (inverse data flow!)

We haven't said anything about what these transformations actually do.  There
are three major transformations AOTAutograd does (performed in this order):

* View/mutation handling
* Autograd
* Subclasses

So intuitively, descriptors are built like this:

1. **PlainAOTInput, PlainAOTOutput.**

   We start off descriptors describing the exact inputs/outputs of the
   original flattened user function.  This user function is assumed to already
   be flattened; you would chain on pytree KeyPaths to further describe where
   in the pytree each input/output lived if you needed to deal with
   unflattened functions: this can be done from userland on top of
   descriptors, so the main descriptors mechanism doesn't handle it.

2. **SyntheticBaseAOTInput, ViewBaseAOTInput, MetadataMutationAOTOutput,
   InputMutationAOTOutput, IntermediateBaseAOTOutput**

   We deal with mutations and aliasing by removing duplicate PlainAOTInputs
   and introduce some new artificial inputs/outputs.  These inputs do not
   have a straightforward correspondence to the original user inputs, but if
   you are implementing a pass that doesn't care about the exact semantics of
   inputs, you should handle all of these uniformly in the same way as regular
   inputs.

3. **TangentAOTInput, GradAOTOutput**

   We deal with autograd by introducing a tangent input for every
   differentiable AOTOutput (including the new ones introduced above), and a
   gradient output for every differentiable AOTInput (also including new ones
   introduced above.) The arguments to these AOTInput/AOTOutput can ONLY be
   the ones we already have above (from steps 1-2).  As AOTAutograd does not
   currently support double backwards, you never have tangents of grads or
   vice versa (but in the future we could!)

4. **SubclassGetAttrAOTInput, SubclassGetAttrAOTOutput, et al.**

   We deal with subclasses by introducing flattened inputs/outputs (including
   potentially symbolic sizes/strides) for every AOTInput/AOTOutput that was a
   subclass.  As above, the arguments to these AOTInput/AOTOutput can ONLY be
   the ones we have above (from steps 1-3).  Recursive subclasses are
   supported, so these descriptors can nest with each other (so descriptors
   from step 4 are fair game as well.)

5. **ForwardTokenAOTInput, ForwardTokenAOTOutput, BackwardTokenAOTInput, BackwardTokenAOTOutput.**

   Some extra token inputs/outputs get added, these are synthetic and are just here to
   prevent DCE/reordering.

The important thing about the pipeline is that descriptors can ONLY be
created from top-to-bottom.  So for example, you can have::

    SubclassGetAttrAOTInput(TangentAOTInput(PlainAOTOutput(...)))  # OK

As you can see that PlainAOTOutput -> TangentAOTInput ->
SubclassGetAttrAOTInput is consistent with the pipeline ordering), but you can
NEVER have::

    TangentAOTInput(SubclassGetAttrAOTOutput(PlainAOTOutput(...))  # BAD

This is inconsistent; we always do autograd BEFORE we process subclasses!

Similarly, for example, this is illegal::

    GradAOTOutput(SubclassGetAttrAOTInput(PlainAOTInput(...)))  # BAD

It is illegal because subclasses are handled *after* create joint during
wrapper construction.  Instead, you would have::

    SubclassGetAttrAOTOutput(GradAOTOutput(PlainAOTInput(...)))  # OK

This intuitively captures the fact that we always to autograd directly on the
subclass, rather than after desugaring the subclass into its inner tensors.

Descriptor index
----------------

Here is a list of all AOTInput/AOTOutput, organized by how likely you need to
handle them:

* AOTInput

  * Important:

    * PlainAOTInput (the primals!)
    * ParamAOTInput
    * TangentAOTInput
    * SubclassGetAttrAOTInput et al. (if you use subclasses)

  * View related (can be eliminated by cloning inputs to graph; if you don't
    eliminate them, make sure to handle pairing them with GradAOTOutput):

    * ViewBaseAOTInput
    * SyntheticBaseAOTInput

  * Non-tensor, mostly just ignore them:

    * DummyAOTInput
    * PhiloxForwardSeedAOTInput
    * PhiloxForwardBaseOffsetAOTInput
    * PhiloxBackwardSeedAOTInput
    * PhiloxBackwardBaseOffsetAOTInput
    * ForwardTokenAOTInput
    * BackwardTokenAOTInput

* AOTOutput

  * Important:

    * PlainAOTOutput
    * GradAOTOutput
    * SubclassGetAttrAOTOutput et al. (if you use subclasses)

  * More obscure (if not eliminated, make sure you handle pairing them with
    TangentAOTInput):

    * InputMutationAOTOutput (can be eliminated if mutations are non-differentiable)
    * IntermediateBaseAOTOutput (can be eliminated by cloning outputs of graph)
    * MetadataMutationAOTOutput (uhh, just don't mutate metadata?)

  * Non-tensor, mostly just ignore them:

    * PhiloxUpdatedForwardOffsetAOTOutput
    * PhiloxUpdatedBackwardOffsetAOTOutput
    * ForwardTokenAOTOutput
    * BackwardTokenAOTOutput
    * DummyAOTOutput

For convenience, we also have DifferentiableAOTInput and
DifferentiableAOTOutput to help you classify which inputs/outputs can be
wrapped by GradAOTOutput/TangentAOTInput (respectively), which are essentially
all tensor AOTInput/AOTOutput excluding the subclass descriptors.

Implementation details
----------------------

The stylized view above is good for understanding how to interpret
descriptors, but the way that descriptors are generated in code is a bit more
complicated.  Specifically, AOTAutograd is structured as a series of wrappers
on the original user function, which are composed together to form the final
function to trace.  As a result of this, AOTAutograd ends up first building
the full AOTInputs for a function to be traced (as it builds the wrappers and
modifies the flat arguments to be compatible with the new input signature of
the wrapper), and then in reverse builds up the AOTOutput as it is tracing.

There is one major exception to this general idea of "build AOTInput first",
and then "build AOTOutput second": when we create TangentAOTInput, we need to
reference AOTOutputs (which output we are the tangents of) which we generally
haven't created yet.  There's two ways we deal with this:

- After the precompile steps (dedup and synthetic base handling), we do an
  initial pass to collect forward metadata that produces the initial set of
  PlainAOTOutputs which we use to create the tangent inputs.

- We also sometimes just violate causality and predict that an AOTOutput will
  be created in a particular way at some later point in time when we build an
  AOTInput.

As of July 2025, here is an exhaustive description of how inputs/outputs
traverse the wrappers from AOTAutograd, and what descriptors can be introduced
at these phases.

::

                                Build wrappers (FLOWS DOWN)         Run trace (FLOWS UP)
    -------------------------------------------------------------------------------------------------
    Begin                       PlainAOTInput                       (n/a)
                                ParamAOTInput

    Precompile dedupe           (remove dupes)                      (nothing)

    Precompile synthetic base   SyntheticBaseAOTInput               MetadataMutationAOTOutput
                                ViewBaseAOTInput

    Forward metadata trace      PlainAOTOutput                      (n/a)
                                MetadataMutationAOTOutput

    Prepare for autograd        (nothing)                           InputMutationAOTOutput
                                                                    IntermediateBaseAOTOutput

    Create joint                TangentAOTInput                     GradAOTOutput
                                w/ InputMutationAOTOutput
                                w/ IntermediateBaseAOTOutput

    Precompile subclass         SubclassGetAttrAOTInput et al.      SubclassGetAttrAOTOutput et al.

    Effect tokens               ForwardTokenAOTInput                ForwardTokenAOTOutput
                                BackwardTokenAOTInput               BackwardTokenAOTOutput

    End                         (n/a)                               PlainAOTOutput

It can be helpful to separately write down the input flow and the output flow
for ease of understanding the data flow:

* Input desc propagation (happens as we build wrappers)

  * [IN] Begin with original calling convention (PlainAOTInput, ParamAOTInput)
  * [IN] Precompile dedupe: (removes duplicate AOTInputs)
  * [IN] Precompile synthetic base: SyntheticBaseAOTInput, ViewBaseAOTInput
  * Forward metadata trace (mini output desc propagation)

    * [OUT] Original output convention: PlainAOTOutput
    * [OUT] Precompile synthetic base: MetadataMutationAOTOutput

  * [IN] Prepare for autograd: (nothing)
  * [IN] Create joint: TangentAOTInput (potentially w/
    IntermediateBaseAOTOutput, InputMutationAOTOutput)
  * [IN] Precompile subclass: SubclassGetAttrAOTInput et al.
  * [IN] Effect tokens: ForwardTokenAOTInput, BackwardTokenAOTInput
    (Note: BackwardTokenAOTInput is technically generated not by a wrapper but
    actually done by token_discovery which implicitly adds extra arguments
    to the FX trace on-the-fly.)

* Trigger a trace with the modified inputs on the wrapper
* Output desc propagation (happens as we unwind from the user function call in trace)

  * [OUT] Begin with original calling convention: PlainAOTOutput
  * [OUT] Effect tokens: ForwardTokenAOTOutput, BackwardTokenAOTOutput
  * [OUT] Precompile subclass: SubclassGetAttrAOTOutput et al.
  * [OUT] Create joint: GradAOTOutput
  * [OUT] Prepare for autograd: InputMutationAOTOutput, IntermediateBaseAOTOutput
  * [OUT] Precompile synthetic base: MetadataMutationAOTOutput
  * [OUT] Precompile dedupe: (nothing)
"""

import dataclasses


# TODO: the is_* predicates are a little suspicious because (1) they're not
# used by anything and (2) they always report False even when a parameter got
# swizzled into a view base or deduped with a non-parameter.  It is pretty
# difficult to exercise these cases but it's not clear if you will write code
# that works correctly in those cases.


@dataclasses.dataclass(frozen=True)
class AOTInput:
    """Describes where an input from an AOTAutograd produced FX graph comes from"""

    def expr(self) -> str:
        raise NotImplementedError("Subclasses must implement expr()")

    def is_param(self) -> bool:
        """True if this input is a parameter or derived from a parameter (e.g., subclass attr)"""
        return False

    def is_buffer(self) -> bool:
        """True if this input is a buffer or derived from a buffer (e.g., subclass attr)"""
        return False

    def is_tangent(self) -> bool:
        """True if this input is a tangent or derived from a tangent (e.g., subclass attr)"""
        return False


# Note: Currently, our typing discipline for differentiable versus not is not
# very good, so feel free to rely on runtime tests instead.


@dataclasses.dataclass(frozen=True)
class DifferentiableAOTInput(AOTInput):
    """A subclass that classifies AOTInput that can be wrapped by GradAOTOutput"""


@dataclasses.dataclass(frozen=True)
class AOTOutput:
    """Describes where an output from an AOTAutograd produced FX graph will
    eventually be bundled into the final output"""

    def expr(self) -> str:
        raise NotImplementedError("Subclasses must implement expr()")

    def is_grad(self) -> bool:
        """True if this output is a grad or derived from a grad (e.g., subclass attr)"""
        return False


@dataclasses.dataclass(frozen=True)
class DifferentiableAOTOutput(AOTOutput):
    """A subclass that classifies AOTOutput that can be wrapped by TangentAOTInput"""


# ------------

# AOTInput

# ------------


@dataclasses.dataclass(frozen=True)
class ParamAOTInput(DifferentiableAOTInput):
    """The input is a parameter, whose FQN is target"""

    target: str

    def expr(self) -> str:
        return f"self.get_parameter({self.target!r})"

    def is_param(self) -> bool:
        return True

    def is_buffer(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class BufferAOTInput(DifferentiableAOTInput):
    """The input is a buffer, whose FQN is target"""

    target: str

    def expr(self) -> str:
        return f"self.get_buffer({self.target!r})"

    def is_param(self) -> bool:
        return False

    def is_buffer(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class DummyAOTInput(AOTInput):
    """In some circumstances, we want to call into a function that expects AOTInput, but
    we don't actually care about that logic (most typically, because some code is being used
    for both compile-time and run-time; AOTInput processing is not needed in this situation.
    Pass a dummy in this situation; but it is better to just have a version of the function
    that doesn't have this at all."""

    idx: int

    def expr(self) -> str:
        return f"__dummy{self.idx}"


@dataclasses.dataclass(frozen=True)
class PlainAOTInput(DifferentiableAOTInput):
    """The input is a plain input, corresponding to a particular positional index.

    Note that AOTInput is always relative to a function with a *flat* calling convention,
    e.g., as accepted by `aot_module_simplified`.  There are some AOTAutograd APIs that
    flatten pytrees, and we don't record PyTree key paths from the flattening (but we
    could and should!)
    """

    idx: int

    def expr(self) -> str:
        return f"args[{self.idx}]"


@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTInput(AOTInput):
    """Subclass inputs get unpacked into their constituent pieces before going into an FX
    graph.  This tells you which particular attribute of the subclass this particular
    input corresponds to (of the 'base' originally subclass argument.)
    """

    base: AOTInput
    attr: str

    def expr(self) -> str:
        return f"{self.base.expr()}.{self.attr}"

    def is_param(self) -> bool:
        return self.base.is_param()

    def is_buffer(self) -> bool:
        return self.base.is_buffer()

    def is_tangent(self) -> bool:
        return self.base.is_tangent()


@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTInput(AOTInput):
    """Which subclass this particular outer size SymInt input (at dim idx) came from."""

    base: AOTInput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.size({self.idx})"


@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTInput(AOTInput):
    """Which subclass this particular outer stride SymInt input (at dim idx) came from."""

    base: AOTInput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.stride({self.idx})"


@dataclasses.dataclass(frozen=True)
class ViewBaseAOTInput(DifferentiableAOTInput):
    """
    When multiple differentiable inputs are views of the same input, AOTAutograd will replace all of these
    views with a single input representing the base.  If this is undesirable, you can clone the views
    example inputs before passing them into AOTAutograd.

    TODO: In principle we could report ALL of the inputs who this is a base of.
    """

    base_of: AOTInput

    def expr(self) -> str:
        return f"{self.base_of.expr()}._base"


@dataclasses.dataclass(frozen=True)
class SyntheticBaseAOTInput(DifferentiableAOTInput):
    """This is similar to ViewBaseAOTInput, but this happens when none of the views were differentiable, so
    we weren't able to get our hands on the true original view and constructed a synthetic one instead
    for the sake of autograd.
    """

    base_of: AOTInput

    def expr(self) -> str:
        return f"__make_synthetic_base({self.base_of.expr()})"


@dataclasses.dataclass(frozen=True)
class PhiloxForwardSeedAOTInput(AOTInput):
    """The seed for functionalized Philox RNG calls, specifically for forward graph."""

    def expr(self) -> str:
        return "__philox_forward_seed"


@dataclasses.dataclass(frozen=True)
class PhiloxForwardBaseOffsetAOTInput(AOTInput):
    """The offset for functionalized Philox RNG calls, specifically for forward graph."""

    def expr(self) -> str:
        return "__philox_forward_base_offset"


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardSeedAOTInput(AOTInput):
    """The seed for functionalized Philox RNG calls, specifically for backward graph."""

    def expr(self) -> str:
        return "__philox_backward_seed"


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardBaseOffsetAOTInput(AOTInput):
    """The offset for functionalized Philox RNG calls, specifically for backward graph."""

    def expr(self) -> str:
        return "__philox_backward_base_offset"


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTInput(AOTInput):
    """The world token which is threaded through side-effectful operations"""

    idx: int

    def expr(self) -> str:
        return f"__forward_token{self.idx}"


@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTInput(AOTInput):
    """The world token which is threaded through side-effectful operations, for backwards"""

    idx: int

    def expr(self) -> str:
        return f"__backward_token{self.idx}"


# Technically the "output" here is redundant, tangents always correspond to
# outputs
# NB: this is marked differentiable as it /would/ be differentiable if we
# support double backwards, but we never generate this today because we
# don't support double backwards.
@dataclasses.dataclass(frozen=True)
class TangentAOTInput(DifferentiableAOTInput):
    """An input to the joint graph representing the tangent of an output."""

    output: DifferentiableAOTOutput

    def __post_init__(self) -> None:
        if not isinstance(self.output, DifferentiableAOTOutput):
            raise AssertionError(
                f"expected output to be DifferentiableAOTOutput, got {type(self.output)}"
            )

    def expr(self) -> str:
        return f"__output_tangent({self.output.expr()})"

    def is_tangent(self) -> bool:
        return True


# ------------

# AOTOutput

# ------------


@dataclasses.dataclass(frozen=True)
class PlainAOTOutput(DifferentiableAOTOutput):
    """A plain tensor output at position idx of the output tuple"""

    idx: int

    def expr(self) -> str:
        return f"output[{self.idx}]"


@dataclasses.dataclass(frozen=True)
class InputMutationAOTOutput(DifferentiableAOTOutput):
    """The mutated value of an input tensor, returned so we can appropriately propagate autograd."""

    mutated_input: AOTInput

    def expr(self) -> str:
        return f"__input_mutation({self.mutated_input.expr()})"


@dataclasses.dataclass(frozen=True)
class IntermediateBaseAOTOutput(DifferentiableAOTOutput):
    """An intermediate base of multiple outputs which alias each other.  We only report ONE of
    the outputs that contributed to this base"""

    base_of: "AOTOutput"

    def expr(self) -> str:
        return f"__intermediate_base({self.base_of.expr()})"


# TODO: it's a little dodgy this is differentiable lol, but we do generate
# these BEFORE autograd is handled
@dataclasses.dataclass(frozen=True)
class MetadataMutationAOTOutput(DifferentiableAOTOutput):
    idx: int

    def expr(self) -> str:
        return f"__aliased_arg_with_metadata_mutation{self.idx}"


# NB: this is marked differentiable as it /would/ be differentiable if we
# support double backwards, but we never generate this today because we
# don't support double backwards.
@dataclasses.dataclass(frozen=True)
class GradAOTOutput(DifferentiableAOTOutput):
    """An output representing the computed gradient for a differentiable input, in the joint graph"""

    grad_of: DifferentiableAOTInput

    def __post_init__(self) -> None:
        if not isinstance(self.grad_of, DifferentiableAOTInput):
            raise AssertionError(
                f"expected grad_of to be DifferentiableAOTInput, got {type(self.grad_of)}"
            )

    def expr(self) -> str:
        return f"__grad({self.grad_of.expr()})"

    def is_grad(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedForwardOffsetAOTOutput(AOTOutput):
    """The final offset from the functionalized RNG calls, forward only"""

    def expr(self) -> str:
        return "__philox_updated_forward_offset"


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedBackwardOffsetAOTOutput(AOTOutput):
    """The final offset from the functionalized RNG calls, backward only"""

    def expr(self) -> str:
        return "__philox_updated_backward_offset"


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTOutput(AOTOutput):
    """The world token output for side-effectful calls, returned so we cannot DCE it, forward only"""

    idx: int

    def expr(self) -> str:
        return f"__forward_token{self.idx}"


@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTOutput(AOTOutput):
    """The world token output for side-effectful calls, returned so we cannot DCE it, backward only"""

    idx: int

    def expr(self) -> str:
        return f"__backward_token{self.idx}"


# These are seemingly symmetric with their AOTInput counterparts.  The way to
# think about it is that a subclass could be an input or an output, and they
# get exploded into plain tensors on the way in and out.  So we need
# descriptors for both.
@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTOutput(AOTOutput):
    """This output will be bundled into a subclass at this location"""

    base: AOTOutput
    attr: str

    def expr(self) -> str:
        return f"{self.base.expr()}.{self.attr}"

    def is_grad(self) -> bool:
        return self.base.is_grad()


@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTOutput(AOTOutput):
    """This output size will be bundled into a subclass at this location"""

    base: AOTOutput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.size({self.idx})"


@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTOutput(AOTOutput):
    """This output stride will be bundled into a subclass at this location"""

    base: AOTOutput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.stride({self.idx})"


@dataclasses.dataclass(frozen=True)
class DummyAOTOutput(AOTOutput):
    """For cases when you don't actually care about descriptor propagation, do not use under normal
    circumstances."""

    idx: int

    def expr(self) -> str:
        return f"__dummy{self.idx}"


@dataclasses.dataclass(frozen=True)
class SavedForBackwardsAOTOutput(AOTOutput):
    idx: int

    def expr(self) -> str:
        return f"__saved_for_backwards_{self.idx}"


# Note [Activations with no version counter checks in eager]
# In eager, when a tensor is saved for backward, the autograd engine
# generally performs version counter checks when grabbing the activation
# at backward time.
# One exception is in a custom autograd.Function: if the user stashes a tensor
# for use in the backward using `ctx.foo = foo`, rather than
# `ctx.save_for_backward(foo)`, then the autograd engine will not know
# to perform any checks.
# In torch.compile, we handle autograd by tracing through it ahead of time,
# and wrapping the fw + bw graphs into a custom autograd.Function.
# In order to provide parity with eager around VC checks, though, we need
# to know which activations had gotten this "no VC checks" treatment in eager,
# so we can plumb this info into the AOTAutograd runtime,
# and avoid calling ctx.save_for_backward on these tensors.
#
# This dataclass tells us that a given AOTOutput corresponds to:
# - An activation output from the forward graph
# - that should *not* have its version counter checked at runtime in the backward.
#   This is done by stashing this activation on the ctx object directly
@dataclasses.dataclass(frozen=True)
class SavedForBackwardsNoVcCheckAOTOutput(AOTOutput):
    idx: int

    def expr(self) -> str:
        return f"__saved_for_backwards_no_vc_check_{self.idx}"
<|endoftext|>from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._custom_class_base import CustomClassBase
from torch._guards import detect_fake_mode
from torch._library.opaque_object import is_custom_class
from torch._subclasses import FakeTensor, FakeTensorMode
from torch._subclasses.fake_tensor import is_fake_tensor, maybe_get_fake_mode
from torch.fx.experimental.proxy_tensor import _pytree_subclasses_that_lose_info
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import config
from .descriptors import BufferAOTInput, DifferentiableAOTInput, ParamAOTInput
from .schemas import ActInputPath, AOTConfig, FakifiedFlatArgs
from .utils import get_loaded_async_collective_tensor_type


if TYPE_CHECKING:
    from collections.abc import Generator, KeysView

    from torch.distributed._functional_collectives import AsyncCollectiveTensor


static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "cudagraph_static_inputs"
)


def process_inputs(
    flat_args: list[Any],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: ShapeEnv | None,
    ignore_shape_env: bool = False,
) -> tuple[FakifiedFlatArgs, list[ActInputPath]]:
    """Convert real tensor inputs into fake tensors for AOT autograd tracing.

    Called at compile time (not runtime) to produce the fake inputs that AOT
    autograd traces through. Each real tensor is converted to a FakeTensor
    via ``fake_mode.from_tensor``, preserving shape, dtype, device, and
    symbolic shape information from the ShapeEnv. Non-tensor inputs (ints,
    SymInts, ScriptObjects) are converted or passed through as appropriate.

    Tensor subclass inputs (DTensor, etc.) are fakified recursively by walking
    their ``__tensor_flatten__`` attrs. AsyncCollectiveTensors are resolved via
    ``trigger_wait()`` before fakification so they don't appear in the traced
    metadata (see below).

    Called from ``aot_function``, ``aot_module_simplified``, and
    ``aot_export_module`` — anywhere AOT autograd needs fake inputs before
    graph capture.

    Returns:
        A tuple of (fakified_args, act_input_paths) where act_input_paths records
        which input paths held AsyncCollectiveTensors. The paths are stored on
        ViewAndMutationMeta so the runtime wrapper can re-wait those positions.
    """
    # Resolve AsyncCollectiveTensors before tracing. ACTs are transient
    # eager-mode wrappers for async collective overlap; if they leak into the
    # traced graph as input types, AOT autograd records them in
    # SubclassCreationMeta for output tangent metadata. At runtime, autograd
    # produces plain tensor tangents, causing a type mismatch. Unwrapping
    # here prevents ACT from appearing in the traced metadata.
    AsyncCollectiveTensor = get_loaded_async_collective_tensor_type()
    act_input_paths: list[ActInputPath] = []
    if AsyncCollectiveTensor is not None:
        tracing_context = torch._guards.TracingContext.try_get()
        for i, a in enumerate(flat_args):
            resolved_arg = _resolve_input_async_collectives(
                a, i, (), act_input_paths, AsyncCollectiveTensor
            )
            # Rebuilt wrapper inputs must keep Dynamo's symbolic context so
            # fakeification does not allocate source-less shape symbols.
            if resolved_arg is not a and isinstance(resolved_arg, torch.Tensor):
                if tracing_context is not None:
                    if a in tracing_context.tensor_to_context:
                        tracing_context.tensor_to_context[resolved_arg] = (
                            tracing_context.tensor_to_context[a]
                        )
            flat_args[i] = resolved_arg

    with fake_mode:

        def convert(idx: int, x: Any) -> Any:
            nonlocal ignore_shape_env
            if shape_env is not None and not ignore_shape_env:
                from torch._dynamo.source import ConstantSource

                if isinstance(x, int):
                    # We always specialize on scalar values in export.
                    if aot_config.is_export:
                        return x
                    source = ConstantSource(f"sym_{idx}")
                    return shape_env.create_symintnode(
                        shape_env.create_symbol(x, source, positive=x >= 0),
                        hint=x,
                        source=source,
                    )
            if isinstance(x, torch.ScriptObject) or is_custom_class(type(x)):
                return torch._library.fake_class_registry.maybe_to_fake_obj(
                    fake_mode, x
                )
            if not isinstance(x, torch.Tensor):
                return x
            if is_fake_tensor(x):
                # In the case of cross compilation we will have example inputs
                # with a different fake mode than our tracing fake mode.
                # In these cases we want to clone the fake tensor into our
                # inner fake mode.
                if maybe_get_fake_mode(x) is not fake_mode:
                    return fake_mode.from_tensor(x)
                return x
            if is_traceable_wrapper_subclass(x):
                attrs, _ = x.__tensor_flatten__()
                # See if all inner tensors are FakeTensors from this mode
                all_this_fake = True
                for a in attrs:
                    match getattr(x, a):
                        case FakeTensor() as v:
                            if v.fake_mode is not fake_mode:
                                # FakeTensor subclass from a different mode.
                                # Fall through to refakify.
                                all_this_fake = False
                                break
                        case torch.Tensor():
                            all_this_fake = False
                            break
                        case CustomClassBase():
                            pass
                        case unexpected:
                            raise AssertionError(
                                f"expected Tensor or CustomClassBase, got {type(unexpected)}"
                            )

                if all_this_fake:
                    return x

            # see note [Tensor Fakification and Symbol Caching]
            symbolic_context = None
            source = None
            trace = True
            if tracing_context := torch._guards.TracingContext.try_get():
                if x in tracing_context.tensor_to_context:
                    symbolic_context = tracing_context.tensor_to_context[x]
                    source = symbolic_context.tensor_source
                    # We already fakeified this tensor in Dynamo, don't
                    # dump the trace for it again
                    trace = False
            if (
                idx < aot_config.num_params_buffers
                and config.static_weight_shapes
                and not symbolic_context
            ):
                # TODO: Ensure that this codepath is never exercised from
                # Dynamo
                return fake_mode.from_tensor(x, static_shapes=True)

            result = fake_mode.from_tensor(
                x,
                static_shapes=ignore_shape_env,
                symbolic_context=symbolic_context,
                source=source,
                trace=trace,
            )
            return result

        return FakifiedFlatArgs(
            [convert(idx, x) for idx, x in enumerate(flat_args)]
        ), act_input_paths


def _resolve_input_async_collectives(
    x: object,
    idx: int,
    attr_path: tuple[str, ...],
    act_input_paths: list[ActInputPath],
    AsyncCollectiveTensor: type[AsyncCollectiveTensor],
) -> object:
    """Wait on ACT inputs, recording each top-level or wrapper-subclass path."""
    if isinstance(x, AsyncCollectiveTensor):
        act_input_paths.append((idx, attr_path))
        return x.trigger_wait()

    if not is_traceable_wrapper_subclass(x):
        return x

    attrs, meta = x.__tensor_flatten__()
    inner_tensors: dict[str, Any] | None = None
    for attr in attrs:
        inner = getattr(x, attr)
        match inner:
            case CustomClassBase():
                continue
            case torch.Tensor():
                resolved_inner = _resolve_input_async_collectives(
                    inner,
                    idx,
                    (*attr_path, attr),
                    act_input_paths,
                    AsyncCollectiveTensor,
                )
            case _:
                raise AssertionError(
                    f"expected Tensor or CustomClassBase, got {type(inner)}"
                )

        if resolved_inner is not inner:
            if inner_tensors is None:
                inner_tensors = {a: getattr(x, a) for a in attrs}
            inner_tensors[attr] = resolved_inner

    if inner_tensors is None:
        return x

    return type(x).__tensor_unflatten__(  # type: ignore[attr-defined]
        inner_tensors, meta, x.size(), x.stride()
    )


def construct_fake_mode(
    flat_args: list[Any], aot_config: AOTConfig
) -> tuple[FakeTensorMode, ShapeEnv | None]:
    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env
    return (fake_mode, shape_env)


def _try_get_metadata_from_dynamo(
    mod: torch.nn.Module,
    param_keys: KeysView[str],
    full_args_num: int,
    full_args_descs: list[DifferentiableAOTInput],
) -> tuple[list[torch._guards.Source | None] | None, list[int]]:
    """
    Metadata is forwarded from Dynamo to AOTDispatch via special fields on GraphModule.
    We first verify that `mod` does come from Dynamo, then we handle cases where
    metadata might be missing.

    Returns:
        aot_autograd_arg_pos_to_source: used to dedup params and their guards
        static_input_indices: used to identify static inputs for cudagraphs
    """
    # Note [Assumption on Dynamo Metadata]
    # This function assumes a graph module from dynamo provides `dynamo_compiled_id`,
    # _param_name_to_source, and every placeholder node has `_dynamo_source` attributes.
    # When gm is modified (e.g., DDPOptimizer via split_module), metadata needs to
    # be propagated in order to be recognized as a dynamo graph

    if not (isinstance(mod, torch.fx.GraphModule) and "dynamo_compile_id" in mod.meta):
        # graph was not captured by dynamo
        return None, []

    if not hasattr(mod, "_param_name_to_source"):
        # is from export
        static_input_indices = [
            i
            for i, node in enumerate(full_args_descs)
            if isinstance(node, (ParamAOTInput, BufferAOTInput))
        ]
        return None, static_input_indices

    # We now know this came from dynamo, and (1) we care about guards,
    # so setting up aot_autograd_arg_pos_to_source for downstream dedup guards
    # can now be done safely. (2) Dynamo logic protects the 1:1 sizing below.
    # Additionally, we mark static indices for cudagraphs.
    param_name_to_source = cast(
        dict[str, torch._guards.Source], mod._param_name_to_source
    )
    seen_sources = set()

    aot_autograd_arg_pos_to_source: list[torch._guards.Source | None] = []
    static_input_indices = []
    # Collect the new inputs lifted by aotdispatch
    for i, name in enumerate(param_keys):
        if name not in param_name_to_source:
            raise AssertionError(f"{name} not found in param_name_to_source")
        source = param_name_to_source[name]
        if source in seen_sources:
            raise AssertionError(f"source {source} already in seen_sources")
        if source is None:
            raise AssertionError(f"source must not be None for {name}")
        seen_sources.add(source)
        aot_autograd_arg_pos_to_source.append(source)

        static_input_indices.append(i)

    # Collect the dynamo graph inputs
    # TODO(mlazos): Revisit if this is still needed. With Dynamo install ID
    # matched tensors back into the Fx graph, this might not be necessary.
    for pos, node in enumerate(mod.graph.find_nodes(op="placeholder")):
        if not hasattr(node, "_dynamo_source"):
            raise AssertionError(f"node {node} must have _dynamo_source attribute")
        source = node._dynamo_source
        # `source`` specifies the source from user code. ddp optimizer may have
        # intermediate values becoming submodule placeholders which does not
        # have a source
        if source is not None and source in seen_sources:
            raise AssertionError(f"source {source} already in seen_sources")
        seen_sources.add(source)
        aot_autograd_arg_pos_to_source.append(source)
        source_name = source.name if source else str(source)

        # input[i] in dynamo is now:
        # input[i + len(extra_params)] in AOT,
        # where extra_params are the params/buffers that dynamo baked into the
        # OutputGraph
        actual_pos = pos + len(param_keys)

        if "tensor_dict" in node.meta and node.meta["tensor_dict"].get(
            "_dynamo_static_input_type", None
        ):
            static_inputs_log.debug(
                "Adding static input pos %s for source %s", actual_pos, source_name
            )
            static_input_indices.append(actual_pos)
        else:
            static_inputs_log.debug(
                "Non-static input pos %s for source %s", actual_pos, source_name
            )

    if full_args_num != len(aot_autograd_arg_pos_to_source):
        raise AssertionError(
            f"full_args_num={full_args_num} != len(aot_autograd_arg_pos_to_source)={len(aot_autograd_arg_pos_to_source)}"
        )
    return aot_autograd_arg_pos_to_source, static_input_indices


@contextmanager
def _detect_attribute_assignment(mod: torch.nn.Module) -> Generator[None, None, None]:
    # Do not allow assignment of tensor attributes during export unless
    # the attribute is registered as a buffer.

    NN_MODULE_STD_ATTRS = [
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_is_full_backward_hook",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_modules",
        "_non_persistent_buffers_set",
        "_parameters",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "training",
    ]
    NN_MODULE_LAZY_STD_ATTRS = [
        "_initialize_hook",
        "_load_hook",
    ]
    STD_ATTRS = {
        *NN_MODULE_STD_ATTRS,
        *NN_MODULE_LAZY_STD_ATTRS,
    }

    def _get_attributes(mod: torch.nn.Module) -> dict[str, Any]:
        # return any attributes of a module that are not standard attributes
        return {k: v for k, v in mod.__dict__.items() if k not in STD_ATTRS}

    def _get_all_module_attributes(mod: torch.nn.Module) -> dict[str, dict[str, Any]]:
        # return attributes from all modules and submodules
        result = {}
        for name, submodule in mod.named_modules():
            result[name] = _get_attributes(submodule)
        return result

    def _restore_all_module_attributes(
        mod: torch.nn.Module, snapshot: dict[str, dict[str, Any]]
    ) -> None:
        # restore attributes to all modules and submodules
        for name, submodule in mod.named_modules():
            if name in snapshot:
                submodule.__dict__.update(snapshot[name])

    # save state of attributes before enter
    snapshot = pytree.tree_map(
        lambda x: x,
        _get_all_module_attributes(mod),
        is_leaf=lambda x: type(x) in _pytree_subclasses_that_lose_info,
    )
    try:
        yield
    finally:
        # after exit, compare state of attributes with snapshot
        # to detect which tensor attributes were assigned

        def _collect_assigned_tensor_attributes(
            snapshot: dict[str, dict[str, Any]], new_attrs: dict[str, dict[str, Any]]
        ) -> list[str]:
            assigned_tensor_attributes = []

            def _compare_values(path: str, old_val: Any, new_val: Any) -> None:
                """Recursively compare values, handling containers."""
                # Same object, no change
                if old_val is new_val:
                    return

                if old_val is None or new_val is None:
                    if isinstance(new_val, torch.Tensor):
                        assigned_tensor_attributes.append(path)
                    return

                # Check if it's a tensor that was reassigned
                if isinstance(new_val, torch.Tensor):
                    assigned_tensor_attributes.append(path)
                    return

                # Handle dict containers
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    all_keys = set(old_val.keys()) | set(new_val.keys())
                    for key in all_keys:
                        old_item = old_val.get(key)
                        new_item = new_val.get(key)
                        _compare_values(f"{path}[{key!r}]", old_item, new_item)
                    return

                # Handle list/tuple containers
                if isinstance(old_val, (list, tuple)) and isinstance(
                    new_val, (list, tuple)
                ):
                    # Different lengths = mutation happened
                    max_len = max(len(old_val), len(new_val))
                    for i in range(max_len):
                        old_item = old_val[i] if i < len(old_val) else None
                        new_item = new_val[i] if i < len(new_val) else None
                        _compare_values(f"{path}[{i}]", old_item, new_item)
                    return

                # For other types, just check if they're different objects
                # (we don't care about non-tensor mutations)

            for module_name in snapshot.keys() | new_attrs.keys():
                old_module_attrs = snapshot.get(module_name, {})
                new_module_attrs = new_attrs.get(module_name, {})

                for attr_name in old_module_attrs.keys() | new_module_attrs.keys():
                    module_prefix = f"self.{module_name}." if module_name else "self."
                    full_path = f"{module_prefix}{attr_name}"

                    old_val = old_module_attrs.get(attr_name)
                    new_val = new_module_attrs.get(attr_name)
                    _compare_values(full_path, old_val, new_val)

            return assigned_tensor_attributes

        new_attrs = _get_all_module_attributes(mod)
        assigned_tensor_attributes = _collect_assigned_tensor_attributes(
            snapshot, new_attrs
        )
        # restore state of all attributes (including, e.g., of primitive types)
        _restore_all_module_attributes(mod, snapshot)

        if assigned_tensor_attributes:
            if len(assigned_tensor_attributes) > 1:
                noun, verb = "attributes", "were"
            else:
                noun, verb = "attribute", "was"
            warnings.warn(
                f"The tensor {noun} {', '.join(assigned_tensor_attributes)} {verb} assigned during export. "
                "Such attributes must be registered as buffers using the `register_buffer` API "
                "(https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer).",
                stacklevel=2,
            )
<|endoftext|>"""
This module provides result classes for AOT Autograd compilation.

Similar to how torch._inductor.output_code provides OutputCode classes for inductor
compilation results, this module provides AOTAutogradResult classes that represent
the compiled artifacts produced by AOT Autograd.

These results are:
- Serializable: can be saved/loaded from disk without recompilation
- Addressable: can be stored in caches with keys for later retrieval
- Reusable: can be used for both caching and ahead-of-time compilation (precompile)

The main result types are:
- GenericAOTAutogradResult: Abstract base for all AOT Autograd results
- AOTAutogradResult: Regular result that references FxGraphCache entries
- BundledAOTAutogradResult: Result that bundles the entire compiled code directly
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar

import torch
from torch._dynamo.precompile_context import BackendCacheArtifact
from torch._inductor.codecache import FxGraphCache
from torch._inductor.output_code import (
    CompiledFxGraph,
    CompiledFxGraphConstants,
    OutputCode,
)
from torch._inductor.utils import should_use_remote_fx_graph_cache
from torch._logging import getArtifactLogger

from .runtime_wrappers import (
    AOTDispatchAutograd,
    AOTDispatchAutogradCompileSpec,
    AOTDispatchSubclassWrapper,
    CachedAutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    FunctionalizedRngRuntimeWrapper,
    post_compile,
    RuntimeWrapper,
    SerializableCompiledFunction,
    SubclassMeta,
)
from .schemas import AOTAutogradCacheInfo, AOTConfig  # noqa: F401
from .utils import simple_wraps


if TYPE_CHECKING:
    from torch._inductor.compile_fx import _CompileFxKwargs

    from .schemas import CacheableAOTConfig, ViewAndMutationMeta

log = logging.getLogger(__name__)
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


TOut = TypeVar("TOut", bound=OutputCode)


class InductorOutput(ABC, Generic[TOut]):
    """
    Class representing a single inductor output
    """

    @abstractmethod
    def pre_save(self) -> None: ...

    @abstractmethod
    def load(self, example_inputs: Sequence[Any]) -> TOut: ...

    @abstractmethod
    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut: ...


TOutputCode = TypeVar("TOutputCode", bound=OutputCode)


@dataclass
class BundledOutputCodeLoadable(InductorOutput[TOutputCode], Generic[TOutputCode]):
    """
    A generic wrapper for OutputCode objects that are bundled directly in the cache
    (rather than looked up via FxGraphCache).

    This works for any OutputCode subclass (CompiledFxGraph, RegionalOutputCode, etc.)
    """

    result: TOutputCode

    def pre_save(self) -> None:
        disk_result = copy(self.result)
        disk_result.prepare_for_serialization()
        self.result = disk_result
        return

    def load(self, example_inputs: Sequence[Any]) -> TOutputCode:
        self.example_inputs = example_inputs
        return self.result

    def post_compile(
        self, result: TOutputCode, fx_config: _CompileFxKwargs
    ) -> TOutputCode:
        constants = CompiledFxGraphConstants()

        # Special handling for CompiledFxGraph - needs FxGraphCache.cache_hit_post_compile
        if isinstance(result, CompiledFxGraph):
            graph, cache_info = FxGraphCache.cache_hit_post_compile(
                result, {}, constants
            )
            if graph is None:
                raise RuntimeError("Failed to reload cache entry from disk")
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_bundled_cache_hit",  # always a hit
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )
            result = graph  # type: ignore[assignment]
            result.compile_region_name = (  # pyrefly: ignore[missing-attribute]
                fx_config.get("compile_region_name")
            )

        # Run normal post compile
        result.post_compile(self.example_inputs, constants, fx_config)

        # Let the CUDAGraph policy do outer-level wrapping (e.g. wrapping
        # an entire RegionalOutputCode as a single CUDA graph instead of
        # per-inner-region).
        import torch._inductor.config as _inductor_config

        policy = _inductor_config.cudagraph_policy
        if policy is not None:
            result = policy.wrap_output(result)

        return result


# Backwards compatibility alias
CompiledFxGraphLoadable: type[BundledOutputCodeLoadable[CompiledFxGraph]] = (
    BundledOutputCodeLoadable[CompiledFxGraph]
)


@dataclass
class FxGraphCacheLoadable(InductorOutput[CompiledFxGraph]):
    fx_graph_cache_info: tuple[str, list[str]]
    fx_graph_guard_expr: str | None

    def pre_save(self) -> None:
        return

    def _is_backward(self) -> bool:
        return False

    def load(self, example_inputs: Sequence[Any]) -> CompiledFxGraph:
        from .autograd_cache import FXGraphCacheMiss

        # [Note: AOTAutogradCache and FXGraphCache Guard interactions]
        # As mentioned, AOTAutograd takes in the symint inputs from dynamo's list of arguments.
        # FXGraphCache serializes guards that are needed in the shape_env based on these symint inputs to the graph.
        # The invariant that AOTAutograd uses here is that the sources for symints given to it by dynamo are exactly
        # the same as the ones it passes to inductor, for both the forward and backward passes.
        # (This does not mean that the tensor values passed in are the same: only that their symints are).
        # That is, AOTAutograd and Inductor never create new guards based on symints with different sources
        # than those passed to it by inductor.
        # We pass the post compile function, which sets various fx_config boxed values,
        # so we can call it only after we're sure both forward and backward have
        # Clear CompiledTritonKernels before loading from FXGraphCache
        torch._inductor.async_compile.CompiledTritonKernels.cache_clear()
        remote_cache = None
        constants = CompiledFxGraphConstants()
        if should_use_remote_fx_graph_cache():
            remote_cache = FxGraphCache.get_remote_cache()
        (cache_key, debug_lines) = self.fx_graph_cache_info

        def check_exact_guard_match(guard_expr: str, _hints: Any) -> bool:
            """
            AOTAutogradCache tracks its own guards, so we just need to treat these guard expressions as a second
            cache key of sorts: we just check for equality, i.e. the FXGraphCache entry with
            the exact same guards as we originally saved into the cache.
            """
            return guard_expr == self.fx_graph_guard_expr

        result, cache_info = FxGraphCache.load_with_key(
            cache_key,
            debug_lines,
            example_inputs,
            local=True,
            remote_cache=remote_cache,
            is_backward=self._is_backward(),
            constants=constants,
            evaluate_guards=check_exact_guard_match,
        )
        if result is None:
            log.info("FXGraphCache cache miss for key %s", self.fx_graph_cache_info)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_cache_miss",  # always a hit
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )

            raise FXGraphCacheMiss

        # No need to log chromium event because AOTAutograd will log that immediately for us
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_cache_hit",  # always a hit
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(cache_info),
        )
        self.example_inputs = example_inputs
        self.constants = constants
        return result

    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph:
        """
        Called after FXGraphCacheLoadable.load, mutates fx_config
        """
        result.compile_region_name = fx_config.get(  # pyrefly: ignore[bad-assignment]
            "compile_region_name"
        )
        result.post_compile(self.example_inputs, self.constants, fx_config)

        import torch._inductor.config as _inductor_config

        policy = _inductor_config.cudagraph_policy
        if policy is not None:
            result = policy.wrap_output(result)

        return result


@dataclass
class CompiledForward(FxGraphCacheLoadable):
    """
    Cacheable entry for a forward function
    """

    def _is_backward(self) -> bool:
        return False


@dataclass
class GenericCompiledBackward(InductorOutput[TOut]):
    # Used by AOTDispatchAutograd.post_compile
    backward_state_indices: list[int]
    num_symints_saved_for_bw_: int

    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut:
        # The concrete post_compile comes from the loadable mixin in each subclass MRO.
        compiled_bw = super().post_compile(  # pyrefly: ignore[missing-attribute]
            result, fx_config
        )
        # See note [Wrapping bw_compiler in disable]
        # This is done by _wrapped_bw_compiler in torch/_dynamo/backends/common.py
        # But since on cache hit we do not call the bw_compiler, we need to reapply the disable
        return torch._dynamo.disable(  # type: ignore[return-value]
            compiled_bw, reason="do not trace generated backwards pass"
        )


@dataclass
class CompiledBackward(GenericCompiledBackward[CompiledFxGraph], FxGraphCacheLoadable):
    """
    Cacheable entry for a backward function
    """

    def _is_backward(self) -> bool:
        return True


# Generic bundled forward/backward classes that work with any OutputCode type
@dataclass
class BundledCompiledForward(
    BundledOutputCodeLoadable[TOutputCode], Generic[TOutputCode]
):
    """
    Generic forward function for bundled compilation.
    Works with any OutputCode type (CompiledFxGraph, RegionalOutputCode, etc.)
    """


@dataclass
class BundledCompiledBackward(
    GenericCompiledBackward[TOutputCode],
    BundledOutputCodeLoadable[TOutputCode],
    Generic[TOutputCode],
):
    """
    Generic backward function for bundled compilation.
    Works with any OutputCode type (CompiledFxGraph, RegionalOutputCode, etc.)
    """


@dataclass
class SerializedGraphModule:
    fn: Callable[[dict[Any, Any], str], torch.nn.Module]
    args: tuple[Any, ...]

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.fn, self.args = gm.__reduce__()

    def deserialize(self) -> torch.fx.GraphModule:
        gm = self.fn(*self.args)
        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"expected fx.GraphModule, got {type(gm)}")
        return gm


def serialize_graph_module(gm: torch.fx.GraphModule) -> SerializedGraphModule:
    # NOTE: mutates the graph module
    gm.meta = {}
    for node in gm.graph.nodes:
        # pyrefly: ignore [implicit-any]
        node.meta = {}
    return SerializedGraphModule(gm)


TForward = TypeVar("TForward", bound="InductorOutput[Any]")
TBackward = TypeVar("TBackward", bound="GenericCompiledBackward[Any]")


@dataclass
class GenericAOTAutogradResult(Generic[TForward, TBackward]):
    """A single result from AOT Autograd compilation, genericized by Forward and Backward types.

    A TForward is always an InductorOutput of some sort, which represents the
    forward graph of the compile.
    A TBackward is an InductorOutput + metadata about the backward, useful for specific
    backward-only wrappers. This type is encapsulated by GenericCompiledBackward.

    Each AOTAutogradResult is essentially parameterized by 1. the method of loading
    from the cache (either Bundled or UnBundled), and 2. The type of the output. For now,
    the only type of output we support is Python Wrapper output, i.e. OutputCode.CompiledFxGraph,
    but the same technique works for C++ wrapper code; we'd just add an extra InductorOutput type.
    """

    # Forward and Backward info
    compiled_fw: TForward
    compiled_bw: TBackward | None

    # Code of the joint graph using print_readable()
    # Used for logging purposes
    aot_joint_graph_str: str | None
    aot_forward_graph_str: str | None
    aot_backward_graph_str: str | None
    min_cut_info_str: str | None

    # Runtime_metadata saved right before compilation
    runtime_metadata: ViewAndMutationMeta

    # Wrappers that run after each aot_dispatch_* function
    dispatch_wrappers: list[CompilerWrapper]

    # Used by AOTSubclassWrapper
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None

    # Used by RuntimeWrapper
    indices_of_inps_to_detach: list[int]

    # Time taken to trace/compile the forward
    # forward_time_taken includes AOTAutograd tracing time + inductor compilation time
    # backward_time_taken is essentially just the time inductor took to compile
    forward_time_taken_ns: int
    backward_time_taken_ns: int

    # Used by standalone_compile
    sanitized_aot_config: CacheableAOTConfig

    guards_expr: str | None

    # Used by Compiled Autograd
    serialized_bw_module: SerializedGraphModule | None

    def pre_save(self) -> None:
        """
        Perform any preparations to make the result ready for serialization.
        """
        self.compiled_fw.pre_save()
        if self.compiled_bw is not None:
            self.compiled_bw.pre_save()

    def _log_cached_graphs(self, aot_config: AOTConfig | CacheableAOTConfig) -> None:
        if not aot_config.enable_log:
            return

        if self.aot_joint_graph_str is not None:
            torch._logging.trace_structured(
                "aot_joint_graph", payload_fn=lambda: self.aot_joint_graph_str
            )
            aot_graphs_log.info(
                "Joint graph (from cache)\n\n%s", self.aot_joint_graph_str
            )

        if self.aot_forward_graph_str is not None:
            from torchgen.utils import dataclass_repr

            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_forward_graph_fw_metadata",
                    "encoding": "string",
                },
                payload_fn=lambda: dataclass_repr(self.runtime_metadata),
            )
            if self.maybe_subclass_meta is not None:
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "aot_forward_graph_fw_subclass_metadata",
                        "encoding": "string",
                    },
                    payload_fn=lambda: dataclass_repr(self.maybe_subclass_meta),
                )

            # It's called an inference graph if not running with autograd
            has_backward = self.aot_backward_graph_str is not None
            torch._logging.trace_structured(
                "aot_forward_graph" if has_backward else "aot_inference_graph",
                payload_fn=lambda: self.aot_forward_graph_str,
            )
            aot_graphs_log.info(
                "Forward graph (from cache)\n\n%s",
                self.aot_forward_graph_str,
            )

        if self.aot_backward_graph_str is not None:
            torch._logging.trace_structured(
                "aot_backward_graph", payload_fn=lambda: self.aot_backward_graph_str
            )
            aot_graphs_log.info(
                "Backward graph (from cache)\n\n%s",
                self.aot_backward_graph_str,
            )

        if self.min_cut_info_str is not None:
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "min_cut_information",
                    "encoding": "json",
                },
                payload_fn=lambda: self.min_cut_info_str,
            )

    def _load_and_post_compile(
        self,
        args: list[torch.Tensor],
        fx_config: _CompileFxKwargs,
    ) -> tuple[Callable[..., Any], Callable[..., Any] | None, bool]:
        from torch._dynamo.utils import CompileEventLogger

        compiled_fw_func = self.compiled_fw.load(args)
        if self.compiled_bw is not None:
            compiled_bw_func = self.compiled_bw.load(args)
            needs_autograd = True
            CompileEventLogger.try_add_pt2_compile(
                "backend_compile", dispatch_mode="autograd"
            )
            # Now that we've loaded forward and backward, call post compile on both
            # This avoids setting things like BoxedBools in fx_config until
            # after both forward and backward cache hit
            fw_fx_config: _CompileFxKwargs = {
                **fx_config,
                "is_backward": False,
            }
            bw_fx_config: _CompileFxKwargs = {
                **fx_config,
                "is_backward": True,
            }
            compiled_fw_func = self.compiled_fw.post_compile(
                compiled_fw_func, fw_fx_config
            )
            compiled_bw_func = self.compiled_bw.post_compile(
                compiled_bw_func, bw_fx_config
            )
            return compiled_fw_func, compiled_bw_func, needs_autograd

        inference_fx_config: _CompileFxKwargs = {
            **fx_config,
            "is_backward": False,
        }

        needs_autograd = False
        CompileEventLogger.try_add_pt2_compile(
            "backend_compile", dispatch_mode="inference"
        )
        compiled_fw_func = self.compiled_fw.post_compile(
            compiled_fw_func, inference_fx_config
        )
        return compiled_fw_func, None, needs_autograd

    def _apply_runtime_wrappers(
        self,
        compiled_fw_func: Callable[..., Any],
        compiled_bw_func: Callable[..., Any] | None,
        needs_autograd: bool,
        aot_config: AOTConfig,
    ) -> Callable[..., Any]:
        from torch._dynamo.utils import CompileEventLogger

        compiled_fw_func = AOTDispatchSubclassWrapper(
            trace_joint=needs_autograd,
            fw_only=None,
            maybe_subclass_meta=self.maybe_subclass_meta,
            num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
        ).post_compile(
            compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
        )

        req_subclass_dispatch = self.maybe_subclass_meta is not None
        CompileEventLogger.try_add_pt2_compile(
            "backend_compile", requires_subclass_dispatch=req_subclass_dispatch
        )

        # In autograd case, functionalizedRngWrapper should not modify outs
        return_new_outs = not needs_autograd
        compiled_fw_func = FunctionalizedRngRuntimeWrapper(
            return_new_outs=return_new_outs
        ).post_compile(
            compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
        )
        # pyrefly: ignore [missing-attribute]
        compiled_fw_func._boxed_call = True
        disable_amp = torch._C._is_any_autocast_enabled()

        if needs_autograd:
            if self.compiled_bw is None:
                raise AssertionError("compiled_bw must not be None when needs_autograd")

            cached_lazy_backward = None
            if self.serialized_bw_module is not None:
                cached_lazy_backward = CachedAutogradLazyBackwardCompileInfo(
                    self.serialized_bw_module.deserialize
                )
            # This function is run on both cache miss and cache hit, either here
            # or in aot_dispatch_autograd. On a cache hit,
            # 1. the bw is already compiled
            # 2. we don't need to save to the cache again
            # so those corresponding arguments are set to None.
            compile_spec = AOTDispatchAutogradCompileSpec(
                compiled_fw_func=compiled_fw_func,
                compiled_bw_func=compiled_bw_func,
                maybe_subclass_meta=self.maybe_subclass_meta,
                num_symints_saved_for_bw=self.compiled_bw.num_symints_saved_for_bw_,
                backward_state_indices=self.compiled_bw.backward_state_indices,
                disable_amp=disable_amp,
                indices_of_inps_to_detach=self.indices_of_inps_to_detach,
                lazy_backward_info=cached_lazy_backward,
                bw_compiler=None,
                aot_config=aot_config,
                fw_metadata=self.runtime_metadata,
                try_save_cache_entry=None,
            )
            compiled_function = AOTDispatchAutograd.post_compile(compile_spec)

        else:
            compiled_function = RuntimeWrapper(
                indices_of_inps_to_detach=self.indices_of_inps_to_detach,
                trace_joint=False,
                disable_amp=disable_amp,
            ).post_compile(
                compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
            )

        # Add serialization function back onto object
        compiled_function, _ = post_compile(
            self.dispatch_wrappers,
            compiled_function,
            aot_config,
            runtime_metadata=self.runtime_metadata,
        )
        return compiled_function

    def _check_guards(self, args: list[torch.Tensor]) -> None:
        if self.guards_expr:
            from .autograd_cache import AOTAutogradCache

            symints = AOTAutogradCache._filter_backed_symints(args)
            check = bool(AOTAutogradCache.evaluate_guards(self.guards_expr, symints))
            if check is not True:
                raise AssertionError(f"guards check failed: {check}")

    # Turn result into the original callable
    def wrap_post_compile(
        self,
        args: list[torch.Tensor],
        aot_config: AOTConfig | CacheableAOTConfig,
        fx_config: _CompileFxKwargs,
    ) -> Callable[..., Any]:
        """
        This function takes a result and carefully reconstructs the original callable
        that AOTAutograd returned the first time it was run. It does this by running the various
        post compile steps that AOTAutograd runs on its compiled artifact after running the fw/bw compilers.

        In the inference path, this consists of the Subclass, FunctionalzedRngRuntime, and RuntimeWrappers.
        In the autograd path, this consists of AOTAutogradDispatch.post_compile.

        The steps here should match exactly the steps that are run in aot_dispatch_base and aot_dispatch_autograd.

        Notably absent from the cached path are:
        - DebugAssertWrapper
        - FakifiedOutWrapper

        Which we'll handle separately later on, if necessary.
        """
        from torch._dynamo.utils import dynamo_timed

        self._log_cached_graphs(aot_config)
        # Cache hits only retain the cacheable subset of AOTConfig. Narrow once
        # here so the existing post-compile wrapper stack can keep its compile-time
        # AOTConfig annotations.
        runtime_aot_config = cast(AOTConfig, aot_config)
        with dynamo_timed("AOTAutogradCache.inductor_load"):
            compiled_fw_func, compiled_bw_func, needs_autograd = (
                self._load_and_post_compile(args, fx_config)
            )

        compiled_function = self._apply_runtime_wrappers(
            compiled_fw_func,
            compiled_bw_func,
            needs_autograd,
            runtime_aot_config,
        )
        # Now that we're pretty sure it's a successful load, add guards
        # to the existing shape environment from the cache.
        self._check_guards(args)
        return compiled_function


class AOTAutogradResult(GenericAOTAutogradResult[CompiledForward, CompiledBackward]):
    """
    Regular AOTAutogradResult: saves the forward/backward FxGraphCache keys
    and looks them up in FxGraphCache on load
    """


class BundledAOTAutogradResult(
    GenericAOTAutogradResult[
        BundledCompiledForward[TOutputCode], BundledCompiledBackward[TOutputCode]
    ],
    Generic[TOutputCode],
):
    """
    Generic AOTAutogradResult where we bundle the entire OutputCode directly
    (rather than looking it up via FxGraphCache).

    This works with any OutputCode type:
    - CompiledFxGraph: Traditional inductor compilation
    - RegionalOutputCode: Regional inductor compilation with GraphPickler serialization
    - Any future OutputCode subclasses

    Type parameter:
        TOutputCode: The OutputCode subclass (e.g., CompiledFxGraph, RegionalOutputCode)

    Usage with CompiledFxGraph:
        entry = BundledAOTAutogradResult[CompiledFxGraph](
            compiled_fw=BundledCompiledForward(result=CompiledFxGraph(...)),
            compiled_bw=BundledCompiledBackward(
                result=CompiledFxGraph(...),
                backward_state_indices=[...],
                num_symints_saved_for_bw_=...,
            ),
            ...
        )

    Usage with RegionalOutputCode:
        entry = BundledAOTAutogradResult[RegionalOutputCode](
            compiled_fw=BundledCompiledForward(result=RegionalOutputCode(gm)),
            compiled_bw=BundledCompiledBackward(
                result=RegionalOutputCode(gm),
                backward_state_indices=[...],
                num_symints_saved_for_bw_=...,
            ),
            ...
        )
    """


def deserialize_bundled_cache_entry(
    entry: BundledAOTAutogradResult[Any],
) -> Callable[..., Any]:
    from copy import deepcopy

    from torch._inductor.cudagraph_utils import BoxedDeviceIndex
    from torch._inductor.utils import BoxedBool

    # In the precompile use case, guards are already serialized
    # by dynamo, so we don't need to add them to the environment
    entry.guards_expr = None
    # TODO: this isn't exactly right, because cudagraphs needs to be a shared config
    # which is set by compile_fx. But in precompile, we never actually call compile_fx
    # so we don't have a place to track cudagraphs here.
    cudagraphs = BoxedBool(torch._inductor.config.triton.cudagraphs)
    boxed_forward_device_index = BoxedDeviceIndex(None)
    # We need to make a clean copy of the cache entry
    # in case it needs to be serialized again
    serializable_copy = deepcopy(entry)

    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    context = torch._guards.TracingContext.try_get()
    if context is None:
        # Create a clean environment when running fx graph post compile
        # if one is not available
        context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
    with torch._guards.tracing(context):
        compiled_fn = entry.wrap_post_compile(
            [],
            entry.sanitized_aot_config,
            {
                "cudagraphs": cudagraphs,
                "boxed_forward_device_index": boxed_forward_device_index,
            },
        )
    # Ensure the deserialized cache entry is still serializable

    compiled_fn = SerializableCompiledFunction(compiled_fn, lambda: serializable_copy)

    # TODO: this ignores flat_params, which can exist
    # if inline_builtin_nn_modules=False
    @simple_wraps(compiled_fn)
    def forward(*runtime_args: Any) -> Any:
        return compiled_fn(list(runtime_args))

    if not hasattr(compiled_fn, "serialize"):
        raise AssertionError("compiled_fn must have serialize attribute")
    forward.serialize = compiled_fn.serialize  # type: ignore[attr-defined]

    return forward


@dataclass
# pyrefly: ignore [implicit-any]
class BundledAOTAutogradCacheArtifact(BackendCacheArtifact[Callable]):
    # pyrefly: ignore [implicit-any]
    def after_deserialization(self) -> Callable:
        return deserialize_bundled_cache_entry(self.content)
<|endoftext|>"""
Functions in this module do most of the "work" of AOTAutograd.
An aot_dispatch_* function:
- Takes in the input flat_fn, flat_args, and some metadata
- Runs a set of pre compile wrappers (e.g. argument deduping)
- Runs the actual compiler
- Wraps the returned callable in a set of post compile wrappers
- Returns the wrapped callable and metadata.
"""

import copy
import dataclasses
import functools
import itertools
import logging
import operator
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._custom_class_base import CustomClassBase
from torch._dynamo.utils import (
    CompileEventLogger,
    detect_fake_mode,
    dynamo_timed,
    lazy_format_graph_code,
)
from torch._guards import CompileContext, TracingContext
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class_obj
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake_tensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import fx_placeholder_vals, guard_or_true
from torch.fx.graph_module import GraphModule
from torch.fx.passes._tensorify_python_scalars import tensorify_python_scalars
from torch.multiprocessing.reductions import StorageWeakRef
from torch.types import py_sym_types
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torchgen.utils import dataclass_repr

from .. import config
from .aot_autograd_result import GenericAOTAutogradResult, serialize_graph_module
from .autograd_cache import (
    AOTAutogradCache,
    should_bundle_autograd_cache,
    should_use_remote_autograd_cache,
)
from .descriptors import AOTOutput, PlainAOTOutput
from .graph_capture import aot_dispatch_autograd_graph, aot_dispatch_base_graph
from .logging_utils import track_graph_compiling
from .runtime_wrappers import (
    AOTDedupeWrapper,
    AOTDispatchAutograd,
    AOTDispatchAutogradCompileSpec,
    AOTDispatchSubclassWrapper,
    AOTSyntheticBaseWrapper,
    AutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    DebugAssertWrapper,
    EffectTokensWrapper,
    FakifiedOutWrapper,
    FunctionalizedRngRuntimeWrapper,
    make_runtime_safe,
    post_compile,
    pre_compile,
    RuntimeWrapper,
    SerializableCompiledFunction,
)
from .schemas import (
    AOTConfig,
    AOTGraphCapture,
    AOTState,
    FlatFn,
    FxValue,
    MutationType,
    SubclassMeta,
    ViewAndMutationMeta,
)
from .subclass_utils import compute_inner_mutated_inp_indices_from_subclass_meta
from .utils import (
    contain_metadata_mutation_ops,
    get_default_generator,
    make_boxed_func,
    simple_wraps,
    strict_zip,
    unlift_tokens,
)


def is_opaque_node(node: Any) -> bool:
    """Check if a node contains an opaque or non-tensor value (e.g., ProcessGroup)."""
    from torch._library.fake_class_registry import FakeScriptObject

    if not isinstance(node, torch.fx.Node):
        return False
    if "val" not in getattr(node, "meta", {}):
        return False
    val = node.meta["val"]
    if is_custom_class_obj(val):
        return True
    if isinstance(val, (torch.ScriptObject, FakeScriptObject)):
        return True
    return False


_thread_local = threading.local()


def _should_save_cache(*compiled_fns: Callable[..., Any]) -> bool:
    if should_bundle_autograd_cache():
        return True
    return all(
        getattr(fn, "_fx_graph_cache_key", None) is not None for fn in compiled_fns
    )


@contextmanager
def maybe_skip_decompose(aot_config: AOTConfig) -> Generator[AOTConfig, None, None]:
    if config.selective_decompose:
        yield dataclasses.replace(aot_config, decompositions={})
    else:
        yield aot_config


# Saved tensor hooks context
# Compiled saved tensor hooks are convenient way to inline some logic in the graphs
# for saved nodes from forward to backward. (E.g. activations quantization)
# In base implementation user does not have any additional information about saved value
# in the hook, except FakeTensor shape, dtype, device etc.
# _get_saved_tensor_hook_context gives additional graph information about that saved value,
# that can be used to make a decisions which pack/unpack to apply for particular saved value.
# This allows user to reuse saved tensors hooks api to apply selective pack/unpack in
# graph aware way.
# Alternative to this will be making user to write a custom pass that mucks with forward outputs,
# backward input metadata, which requires significantly more effort.
#
# As for now in context we expose forward graph, backward graph and current saved node,
# which contains node.meta with additional information about that fx.Node.
# Warning: This API may change without backward compatibility.
@contextmanager
def _saved_tensor_hook_context(state: dict[str, Any]) -> Generator[None, None, None]:
    previous_state = getattr(_thread_local, "state", None)
    try:
        _thread_local.state = state
        yield
    finally:
        # Clean up: restore previous state or remove attribute
        if previous_state is not None:
            _thread_local.state = previous_state
        else:
            if hasattr(_thread_local, "state"):
                delattr(_thread_local, "state")


def _get_saved_tensor_hook_context() -> dict[str, Any] | None:
    return getattr(_thread_local, "state", None)


zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

aten = torch.ops.aten

# Returns a Callable and a ViewAndMutationMeta.
# Currently, only export needs the ViewAndMutationMeta after this function.
# TODO: Refactor this
DispatchReturn = tuple[Callable[..., Any], ViewAndMutationMeta]


def _create_wrappers_for_dispatch(needs_autograd: bool) -> list[CompilerWrapper]:
    """
    Wrappers that run on every dispatch function
    """
    return [AOTDedupeWrapper(), AOTSyntheticBaseWrapper(trace_joint=needs_autograd)]


def aot_stage1_graph_capture(
    aot_state: AOTState,
    orig_flat_fn: FlatFn,
) -> AOTGraphCapture:
    # NB: flat_fn at this point coincides with the initial info from forward
    # metadata collection returning a list[Tensor].  We are now going to
    # augment the output to return a tuple[list[Tensor], list[AOTOutput]] and
    # then preserve this convention through the rest of the passes.

    # TODO: We could test for consistency with fw_metadata, but this is not a
    # big deal
    @simple_wraps(orig_flat_fn)
    def orig_flat_fn2(*args: FxValue) -> tuple[list[FxValue], list[AOTOutput]]:
        out = orig_flat_fn(*args)
        out_descs: list[AOTOutput] = type(out)(  # type: ignore[assignment]
            PlainAOTOutput(i)  # type: ignore[misc]
            for i in range(len(out))  # type: ignore[misc]
        )
        return out, out_descs

    aot_config = aot_state.aot_config

    wrappers = _create_wrappers_for_dispatch(aot_state.needs_autograd)
    flat_fn, aot_state.flat_args, aot_state.flat_args_descs, aot_state.fw_metadata = (
        pre_compile(
            wrappers,
            orig_flat_fn2,
            aot_state.flat_args,
            aot_state.flat_args_descs,
            aot_config,
            fw_metadata=aot_state.fw_metadata,
        )
    )
    if aot_config.disable_functionalization:
        # Effect tokens are introduced by FunctionalTensorMode.  The
        # disable_functionalization path intentionally traces without the
        # effect-token wrapper, so metadata-discovered tokens must not affect
        # graph signatures or forward/backward output partitioning.
        aot_state.fw_metadata.tokens = {}
        aot_state.fw_metadata.num_backward_tokens = 0

    # NB: This is currently only used for backwards, where fwd/bwd
    # deterministic TLS can be different
    aot_state.fw_metadata.deterministic = torch.are_deterministic_algorithms_enabled()
    updated_flat_args: list[Any] | tuple[list[Any], list[Any]]

    with maybe_skip_decompose(aot_config) as graph_capture_aot_config:
        # if config.selective_decompose, skip decomposition and apply selective_decompose
        # after we get the joint graph. See [Note: Selective Decomposition] for details.
        if aot_state.needs_autograd and not aot_config.pre_dispatch:
            # FYI: this being moved to trigger in export is new, seems fine!
            with dynamo_timed("aot_trace_joint_graph", log_pt2_compile_event=True):
                (
                    graph,
                    updated_flat_args,
                    updated_flat_args_descs,
                    maybe_subclass_meta,
                ) = aot_dispatch_autograd_graph(
                    flat_fn,
                    aot_state.flat_args,
                    aot_state.flat_args_descs,
                    graph_capture_aot_config,
                    fw_metadata=aot_state.fw_metadata,
                )
        else:
            graph, updated_flat_args, updated_flat_args_descs, maybe_subclass_meta = (
                aot_dispatch_base_graph(
                    flat_fn,
                    aot_state.flat_args,
                    aot_state.flat_args_descs,
                    graph_capture_aot_config,
                    fw_metadata=aot_state.fw_metadata,
                )
            )
    if config.selective_decompose:
        from torch.fx.experimental.proxy_tensor import selective_decompose
        from torch.fx.passes.regional_inductor import _needs_inductor_compile

        graph = selective_decompose(
            graph,
            *updated_flat_args,
            decomposition=aot_config.decompositions,
            should_decompose=_needs_inductor_compile,
            trace_joint_graph=aot_state.needs_autograd and not aot_config.pre_dispatch,
        )

    return AOTGraphCapture(
        wrappers=wrappers,
        graph_module=graph,
        updated_flat_args=updated_flat_args,
        updated_flat_args_descs=updated_flat_args_descs,
        maybe_subclass_meta=maybe_subclass_meta,
    )


def aot_stage2_export(
    aot_state: AOTState, aot_graph_capture: AOTGraphCapture
) -> DispatchReturn:
    graph = aot_graph_capture.graph_module
    aot_config = aot_state.aot_config
    wrappers = aot_graph_capture.wrappers

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="export")

    # NB: the wrappers that run in pre_compile for export are
    # either a no-op, because they're not needed, or will raise a runtime error,
    # since they don't support export.
    # We still run these wrappers to make sure that they're not needed pre compile,
    # but we technically don't need to run them post compile at all here.
    compiled_fn, aot_state.fw_metadata = post_compile(
        wrappers,
        graph,  # pyrefly: ignore [bad-argument-type]
        aot_config,
        runtime_metadata=aot_state.fw_metadata,
    )

    # Therefore, since no wrapperes run, we don't get back a callable - we get back the raw fx graph
    # (either a joint or an inference-only graph)
    if not isinstance(compiled_fn, torch.fx.GraphModule):
        raise AssertionError(
            f"expected compiled_fn to be GraphModule, got {type(compiled_fn)}"
        )
    return compiled_fn, aot_state.fw_metadata


def _get_inner_meta(
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
) -> ViewAndMutationMeta:
    """
    Util to get view and mutation metadata.
    """
    return (
        fw_metadata if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
    )


def _apply_tensorify_python_scalars(module: torch.fx.GraphModule) -> None:
    """
    Util to apply tensorify_python_scalars.
    """
    # TODO(anijain2305) - Add tensorify_python_scalars to the HOP graph passes.
    fake_mode = detect_fake_mode()
    if fake_mode is not None and fake_mode.shape_env is not None:
        tensorify_python_scalars(module, fake_mode.shape_env, fake_mode)


def aot_stage2_compile(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
    # pyrefly: ignore [implicit-any]
    fw_compiler: Callable,
    # pyrefly: ignore [implicit-any]
    bw_compiler: Callable | None = None,
    # pyrefly: ignore [implicit-any]
    inference_compiler: Callable | None = None,
) -> DispatchReturn:
    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler

    if aot_state.needs_autograd and not aot_state.aot_config.pre_dispatch:
        return aot_stage2_autograd(
            aot_state,
            aot_graph_capture,
            partition_fn,
            fw_compiler,
            bw_compiler,
        )
    else:
        return aot_stage2_inference(
            aot_state,
            aot_graph_capture,
            partition_fn,
            inference_compiler,
        )


def _log_inference_graph(
    fw_module: torch.fx.GraphModule,
    aot_config: AOTConfig,
) -> str | None:
    """
    Log the inference graph to the structured logger.
    Return a str representation of the graph.
    """
    if aot_config.enable_log:
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "torch._functorch.config",
                "encoding": "string",
            },
            payload_fn=lambda: torch._functorch.config.get_serializable_config_copy(),
        )

    # Save the forward_graph_str right after aot_dispatch_base_graph,
    # to save in the cache
    aot_forward_graph_str = None
    if aot_config.cache_info is not None:
        aot_forward_graph_str = fw_module.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            fast_sympy_print=True,
            expanded_def=True,
        )

    return aot_forward_graph_str


def _aot_stage2b_inference_compile(
    fw_module: torch.fx.GraphModule,
    updated_flat_args: list[Any],
    maybe_subclass_meta: SubclassMeta | None,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    # pyrefly: ignore [implicit-any]
    inference_compiler: Callable,
    # pyrefly: ignore [implicit-any]
) -> Callable:
    return _aot_stage2b_compile_forward_or_inference(
        fw_module,
        updated_flat_args,  # type: ignore[arg-type]
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        inference_compiler,
        is_inference=True,
    )[1]


def aot_stage2_inference(
    aot_state: AOTState,
    aot_graph_capture: AOTGraphCapture,
    # pyrefly: ignore [implicit-any]
    partition_fn: Callable,
    # pyrefly: ignore [implicit-any]
    inference_compiler: Callable,
) -> DispatchReturn:
    """
    Handles functions that don't need autograd. Runs wrappers and compiles with
    the stage-2 inference compiler.
    """

    aot_config = aot_state.aot_config
    fw_metadata = aot_state.fw_metadata
    fw_module = aot_graph_capture.graph_module
    wrappers = aot_graph_capture.wrappers
    updated_flat_args = aot_graph_capture.updated_flat_args
    maybe_subclass_meta = aot_graph_capture.maybe_subclass_meta

    CompileEventLogger.try_add_pt2_compile("backend_compile", dispatch_mode="inference")
    aot_forward_graph_str = _log_inference_graph(fw_module, aot_config)

    if not isinstance(fw_module, GraphModule):
        raise AssertionError(
            f"expected fw_module to be GraphModule, got {type(fw_module)}"
        )
    _apply_tensorify_python_scalars(fw_module)

    # When trace_autograd_ops=True, the inference graph may contain fw/bw
    # invoke_subgraph pairs from traced torch.autograd.grad/backward calls.
    # Partition them before the remat pass so that remat duplicates the
    # already-partitioned fw subgraphs (which produce saved tensors for bw).
    fw_module = run_joint_graph_passes_on_hops(
        fw_module, None, aot_config, default_partition_fn=partition_fn
    )

    # Apply AC rematerialization after HOP partitioning. This must happen
    # after partitioning so remat duplicates the partitioned fw subgraphs
    # (not the original unpartitioned ones).
    if torch._functorch.config.remat_using_tags_for_fwd_loss_bwd_graph:
        from torch._functorch._activation_checkpointing.remat_using_tags_for_fwd_loss_bwd_graph_pass import (
            remat_using_tags_for_fwd_loss_bwd_graph,
        )

        fw_module = remat_using_tags_for_fwd_loss_bwd_graph(fw_module)

    if _has_invoke_subgraph_node(fw_module):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "aot_inference_graph_after_hop_passes",
                "encoding": "string",
            },
            payload_fn=lambda: fw_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
                expanded_def=True,
            ),
        )

    compiled_fw = _aot_stage2b_inference_compile(
        fw_module,
        updated_flat_args,  # type: ignore[arg-type]
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
        inference_compiler,
    )

    entry = _cache_inference_info(
        aot_config,
        fw_metadata,
        maybe_subclass_meta,
        compiled_fw,
        aot_forward_graph_str,
        wrappers,
    )

    return _aot_stage2c_make_inference_function(
        aot_config,
        fw_metadata,
        compiled_fw,
        wrappers,
        entry,
    )


def _cache_inference_info(
    aot_config: AOTConfig,
    fw_metadata: ViewAndMutationMeta,
    maybe_subclass_meta: SubclassMeta | None,
    compiled_fw: Callable[..., Any],
    aot_forward_graph_str: str | None,
    wrappers: list[CompilerWrapper],
) -> GenericAOTAutogradResult[Any, Any] | None:
    make_runtime_safe(fw_metadata, maybe_subclass_meta)

    cache_info = aot_config.cache_info

    entry: GenericAOTAutogradResult[Any, Any] | None = None
    if cache_info is not None and _should_save_cache(compiled_fw):
        time_taken_ns = time.time_ns() - cache_info.start_time_ns
        guards_expr = AOTAutogradCache.generate_guards_expression(cache_info)
        entry = AOTAutogradCache.make_entry(
            compiled_fw_func=compiled_fw,  # type: ignore[arg-type]
            compiled_bw_func=None,
            aot_joint_graph_str=None,
            aot_forward_graph_str=aot_forward_graph_str,
            aot_backward_graph_str=None,
            runtime_metadata=fw_metadata,
            dispatch_wrappers=wrappers,
            maybe_subclass_meta=maybe_subclass_meta,
            num_fw_outs_saved_for_bw=None,
            indices_of_inps_to_detach=[],
            forward_time_taken_ns=time_taken_ns,
            backward_time_taken_ns=0,
            sanitized_aot_config=aot_config.to_cacheable(),
            guards_expr=guards_expr,
            backward_state_indices=None,
            num_symints_saved_for_bw=None,
            serialized_bw_module=None,
            min_cut_info_str=None,
        )
        AOTAutogradCache.save(
            cache_info.cache_key,
            entry,
            remote=should_use_remote_autograd_cache(),
        )

    return entry


def _aot_stage2c_make_inference_function(
    aot_config: AOTConfig,
    fw_metadata: ViewAndMutationMeta,
    compiled_fw: Callable[..., Any],
    wrappers: list[CompilerWrapper],
    entry: GenericAOTAutogradResult[Any, Any] | None,
) -> DispatchReturn:
    if entry is not None:
        compiled_fw = SerializableCompiledFunction(compiled_fw, lambda: entry)

    disable_amp = torch._C._is_any_autocast_enabled()
    compiled_fn = RuntimeWrapper(
        indices_of_inps_to_detach=[],
        trace_joint=False,
        disable_amp=disable_amp,
    ).post_compile(
        compiled_fw,
        aot_config,
        runtime_metadata=fw_metadata,
    )

    compiled_fn = post_compile(
        wrappers, compiled_fn, aot_config, runtime_metadata=fw_metadata
    )
    return compiled_fn


def collect_fw_donated_buffer_idxs(
    fw_ins: list[FakeTensor | None],
    user_fw_outs: list[FakeTensor | None],
    bw_outs: list[FakeTensor | None],
    saved_tensors: list[FakeTensor | None],
) -> list[int]:
    """
    Checks if the saved tensors are donated buffers, which means a saved tensor is not
    an alias of any tensors in fw_ins, user_fw_outs, and bw_outs.
    """

    storage_refs = set()

    for t in itertools.chain(fw_ins, user_fw_outs, bw_outs):
        # Only access storage if a tensor has storage (not sparse)
        if t is not None and is_fake_tensor(t) and not is_sparse_any(t):
            storage_refs.add(StorageWeakRef(t.untyped_storage()))

    num_saved_tensor = len(saved_tensors)
    donated_buffer_idxs = []
    for i in range(num_saved_tensor):
        t = saved_tensors[i]
        if (
            t is not None
            and is_fake_tensor(t)
            and not is_sparse_any(t)
            and StorageWeakRef(t.untyped_storage()) not in storage_refs
        ):
            donated_buffer_idxs.append(i)

    return donated_buffer_idxs


def collect_bw_donated_buffer_idxs(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
) -> list[int]:
    """
    Collects backward donated buffer indexes from fw_module and bw_module.
    """

    # [Note: Metadata mutation in proxy tracing]
    # node.meta["val"] is a snapshot of the tensor value when tracing a graph,
    # instead of the final state after the graph has run. node.meta["val"] is
    # not updated even if later there is a metadata mutation op.
    # See: https://github.com/pytorch/pytorch/pull/141308#issuecomment-2495798947
    #
    # Currently, metadata mutation op happens only for sacrificial parameter
    # specifically the `set_` op. This motivates banning metadata mutation from
    # proxy tracing.
    #
    # Since node.meta["val"] is used to detect donated buffer, we return an empty
    # list if there exists metadata mutation op.
    if contain_metadata_mutation_ops(fw_module) or contain_metadata_mutation_ops(
        bw_module
    ):
        return []

    fw_ins = fw_module.graph.find_nodes(op="placeholder")
    bw_outs = next(reversed(bw_module.graph.find_nodes(op="output"))).args[0]
    fw_outs = next(reversed(fw_module.graph.find_nodes(op="output"))).args[0]

    fw_ins = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in fw_ins
    ]
    fw_outs = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in fw_outs
    ]
    bw_outs = [
        n.meta["val"] if (hasattr(n, "meta") and "val" in n.meta) else None
        for n in bw_outs
    ]

    user_fw_outs = fw_outs[: fw_metadata.num_forward]
    saved_tensors = fw_outs[fw_metadata.tensors_saved_for_backwards_slice]

    fw_donated_buffer = collect_fw_donated_buffer_idxs(
        fw_ins,
        user_fw_outs,
        bw_outs,
        saved_tensors,
    )

    if fw_metadata.num_symints_saved_for_bw is None:
        raise AssertionError("fw_metadata.num_symints_saved_for_bw must not be None")
    return [fw_metadata.num_symints_saved_for_bw + i for i in fw_donated_buffer]


@dataclasses.dataclass
class InvokeSubgraphHopGraphs:
    """
    A data structure to hold all the information needed to partition the
    `joint_hop_gm` and joint graph and the restitch the `new_fw_hop_gm` and
    `new_bw_hop_gm` into the bigger `joint_gm`.
    """

    # To avoid re-partitioning subgraphs
    partitioning_done: bool = False
    old_num_fw_outputs: int | None = None
    old_num_fw_inputs: int | None = None

    new_fw_hop_gm: torch.fx.GraphModule | None = None
    new_bw_hop_gm: torch.fx.GraphModule | None = None
    new_num_sym_nodes: int | None = None
    new_num_saved_nodes: int | None = None


def prepare_for_partitioner(
    mod: torch.fx.GraphModule, num_primals: int, num_fw_outputs: int
) -> torch.fx.GraphModule:
    # min-cut partitioner requires the placeholders to have primals and
    # tangents string in the node.name. The signature of the joint graph is
    # (*primals, *tangents)

    # We also have to update the output signature which is right now
    # (*grads, *fw_outs) and we have to change to (*fw_outs, *grads) for the
    # partitioner to work.
    new_graph = torch.fx.Graph()
    env = {}

    primals_counter = itertools.count(0)
    tangents_counter = itertools.count(0)

    for idx, node in enumerate(mod.graph.nodes):
        if node.op == "placeholder":
            if idx < num_primals:
                env[node] = new_graph.placeholder(f"primals_{next(primals_counter)}")
            else:
                env[node] = new_graph.placeholder(f"tangents_{next(tangents_counter)}")
            env[node].meta = copy.copy(node.meta)
        elif node.op == "output":
            # Reverse the (*grads, *fw_outs) to (*fw_outs, *grads)
            # The reason for having the reversed signature in the first
            # place is to simplify step 3.
            old_outputs = node.args[0]
            new_outputs = (
                *old_outputs[-num_fw_outputs:],
                *old_outputs[:-num_fw_outputs],
            )
            new_outputs = [env[n] if n else None for n in new_outputs]
            new_graph.output(tuple(new_outputs))
        else:
            env[node] = new_graph.node_copy(node, lambda n: env[n])
            env[node].meta = copy.copy(node.meta)

    new_graph.lint()

    out = torch.fx.GraphModule(mod, new_graph)
    return out


def _get_partition_fn(
    fw_hop_node: torch.fx.Node,
    aot_config: AOTConfig,
    default_partition_fn: Callable[
        ..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]
    ]
    | None,
) -> tuple[bool, Callable[..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]]]:
    """
    Return either `default_partition_fn` or a HOP specific partition function.

    If a HOP specific partition function is returned, used_hop_custom_partition is True.

    See Note [InvokeSubgraphHOP Partitioner]
    """
    used_hop_custom_partition = False

    # Check for HOP-specific partition function first. This is needed because
    # run_joint_graph_passes_on_hops can be called without an outer stage-2
    # partition function (e.g., in aot_stage1_graph_capture before the remat
    # pass).
    if (
        fw_hop_node.target == torch._higher_order_ops.invoke_subgraph
        and "custom" in fw_hop_node.meta
        and "nested_region_config" in fw_hop_node.meta["custom"]
    ):
        hop_partition_fn = fw_hop_node.meta["custom"][
            "nested_region_config"
        ].partitioner
        if hop_partition_fn is not None:
            if callable(hop_partition_fn):
                raw_partitioner = hop_partition_fn
            elif not isinstance(hop_partition_fn, str):
                raise AssertionError(
                    f"expected hop_partition_fn to be str, got {type(hop_partition_fn)}"
                )
            else:
                match hop_partition_fn:
                    case "default_partition":
                        raw_partitioner = (
                            torch._functorch.partitioners.default_partition
                        )
                    case "min_cut_rematerialization_partition":
                        raw_partitioner = torch._functorch.partitioners.min_cut_rematerialization_partition
                    case _:
                        raise ValueError(
                            f"Unknown HOP partitioner config: {hop_partition_fn}"
                        )

            # Route through Inductor's `partition_fn` so joint-graph passes
            # (e.g. scatter_upon_const_tensor) run on the HOP subgraph before
            # the user-selected raw partitioner.
            from torch._inductor.compile_fx import (
                partition_fn as _inductor_partition_fn,
            )

            return True, functools.partial(
                _inductor_partition_fn, partitioner_fn_override=raw_partitioner
            )

    # Fall back to the parent partitioner from aot_config. When the outer
    # compile is Inductor this is already `compile_fx.partition_fn` so
    # joint-graph passes run there too.
    if default_partition_fn is None:
        raise AssertionError("default_partition_fn must not be None")
    return used_hop_custom_partition, default_partition_fn


def _has_invoke_subgraph_node(gm: torch.fx.GraphModule):
    from torch._higher_order_ops import invoke_subgraph

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is invoke_subgraph:
            return True
    return False


def run_joint_graph_passes_on_hops(
    joint_gm: torch.fx.GraphModule,
    joint_inputs: Any,
    aot_config: AOTConfig,
    *,
    default_partition_fn: Callable[
        ..., tuple[torch.fx.GraphModule, torch.fx.GraphModule]
    ]
    | None = None,
) -> torch.fx.GraphModule:
    """
    This pass runs the joint graph passes on the HOP graph. In torch.compile, we
    typically have many passes which work on the joint graph and then end with a
    partitioner.


    The partitioner part is quite mechanical to handle. HOP have their own
    forward and backward graph. The process can be broken into following steps

    1) Get a `joint_hop_gm` from the `fw_hop_gm` and `bw_hop_gm`
    2) Run joint graph passes on the `joint_hop_gm` to get `new_fw_hop_gm` and `new_bw_hop_gm`
    3) Stitch the `new_fw_hop_gm` and `new_bw_hop_gm` back into the `joint_gm`.

    The terminology used in the code is
    `joint_graph/joint_gm` : Refers to the main graph. This may contain many HOPs which have their own `hop_graph`
    `fw_hop_graph/fw_hop_gm` : Refers to the forward graph associated with a HOP.
    `bw_hop_graph/bw_hop_gm` : Refers to the backward graph associated with a HOP.
    `joint_hop_graph/joint_hop_gm` : Refers to the subgraph associated with the HOP like invoke_subgraph.
    `new_fw_hop_graph/new_fw_hop_gm` : Refers to the forward graph after partitioning is applied to `joint_hop_gm`.
    `new_bw_hop_graph/new_bw_hop_gm` : Refers to the backward graph after partitioning is applied to `joint_hop_gm`.

    NB: This pass works for invoke_subgraph today because we took extra care in
    the Autograd.Dispatch key of invoke_subgraph to vastly simplify Step 1.

    NB: This pass only matches **top-level** invoke_subgraph HOP nodes in
    `joint_gm`. It does not recurse into subgraph modules, so when one
    `nested_compile_region` is invoked from inside another, the inner
    invoke_subgraph HOPs live inside the outer's subgraph module and are not
    paired here. Top-level HOPs are still paired correctly via `call_id`;
    recursive partitioning of nested regions is left to downstream passes.
    """
    from torch._higher_order_ops import invoke_subgraph
    from torch._higher_order_ops.invoke_subgraph import (
        get_backward_nested_region_config,
    )

    def num_outputs(mod: torch.fx.GraphModule) -> int:
        return len(mod.graph.find_nodes(op="output")[0].args[0])

    def num_inputs(mod: torch.fx.GraphModule) -> int:
        return len(mod.graph.find_nodes(op="placeholder"))

    new_hop_graphs: dict[str, InvokeSubgraphHopGraphs] = defaultdict(
        lambda: InvokeSubgraphHopGraphs()
    )

    # Step 1 - Get a `joint_hop_gm` from the `fw_hop_gm` and `bw_hop_gm` This is
    # easy to do for `invoke_subgraph` HOP. During the Autograd dispatch key
    # tracing, we have put the joint_hop_graph in the backward hop graph itself.
    # So to recover the joint_hop_gm, we just have to look at the backward
    # HOP graphs.
    # So we will merge step 1 and step 2 in this next section

    # Save the fw and bwd hop nodes. We will later in-place modify the graph
    # using these nodes.
    # pyrefly: ignore [implicit-any]
    fw_hop_nodes = []
    # pyrefly: ignore [implicit-any]
    bw_hop_nodes = []
    for node in joint_gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is invoke_subgraph
            and isinstance(node.args[1], str)
        ):
            if node.args[1].startswith("fw"):
                fw_hop_nodes.append(node)
            elif node.args[1].startswith("bw"):
                bw_hop_nodes.append(node)

    if not bw_hop_nodes:
        return joint_gm

    # The fw and bw HOP counts are not necessarily equal. A fw HOP can have no
    # corresponding bw HOP when autograd never runs backward through that call
    # — e.g. the user does `x_d = x.detach().requires_grad_()` between two
    # regions, or some outputs are not used in the loss. Likewise, multiple fw
    # calls in the same compile region can share a single bw if autograd dedups
    # or DCEs intermediate bws.
    #
    # Pair fw and bw HOPs by `call_id` — a per-call counter stamped by
    # InvokeSubgraphAutogradOp's fw/bw on each FX node. This is unambiguous
    # regardless of how autograd dispatched the backward (single outer
    # `.backward()`, per-iter `.backward()` in a loop, interleaved regions).
    fws_by_call_id: dict[int, torch.fx.Node] = {}
    for fw in fw_hop_nodes:
        cid = fw.meta.get("custom", {}).get("call_id")
        if cid is None:
            continue
        if cid in fws_by_call_id:
            raise AssertionError(
                f"duplicate call_id={cid} on fw HOPs "
                f"{fws_by_call_id[cid].name!r} and {fw.name!r}"
            )
        fws_by_call_id[cid] = fw

    bw_to_fw_hop_node: dict[torch.fx.Node, torch.fx.Node] = {}
    paired_fws: set[torch.fx.Node] = set()
    for bw in bw_hop_nodes:
        cid = bw.meta.get("custom", {}).get("call_id")
        if cid is None:
            raise AssertionError(
                f"bw HOP {bw.args[1]!r} has no call_id in meta['custom']"
            )
        fw = fws_by_call_id.get(cid)
        if fw is None:
            raise AssertionError(
                f"could not find matching fw HOP for bw {bw.args[1]!r} "
                f"with call_id={cid} (fw call_ids: {sorted(fws_by_call_id)})"
            )
        bw_to_fw_hop_node[bw] = fw
        paired_fws.add(fw)

    # Extra fws share a compile region with a paired bw but have no bw of
    # their own (autograd may dedup or DCE the bw). They still must be
    # rewritten to call the new partitioned fw subgraph, otherwise the output
    # signatures diverge from the rewritten paired fws. Key by the bw
    # identifier so the key matches `new_hop_graphs`.
    fw_args_to_bw_identifier: dict[str, str] = {}
    for bw, fw in bw_to_fw_hop_node.items():
        fw_arg = fw.args[1]
        bw_arg = bw.args[1]
        if not (isinstance(fw_arg, str) and isinstance(bw_arg, str)):
            raise AssertionError(
                f"expected fw/bw invoke_subgraph HOP args[1] to be str identifiers, "
                f"got fw={type(fw_arg)}, bw={type(bw_arg)}"
            )
        fw_args_to_bw_identifier.setdefault(fw_arg, bw_arg.removeprefix("bw"))
    extra_fws_by_id: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for fw in fw_hop_nodes:
        if fw in paired_fws:
            continue
        bw_ident_key = fw_args_to_bw_identifier.get(fw.args[1])
        if bw_ident_key is None:
            continue
        extra_fws_by_id[bw_ident_key].append(fw)

    for node in bw_hop_nodes:
        identifier = node.args[1].removeprefix("bw")

        # If partitioning already done for this identifier, skip. This saves
        # redundant joint graph passes for same subgraphs.
        if new_hop_graphs[identifier].partitioning_done:
            continue

        # Collect some information from the forward hop graph
        fw_hop_node = bw_to_fw_hop_node[node]
        fw_subgraph_attr = fw_hop_node.args[0]
        if not isinstance(fw_subgraph_attr, torch.fx.Node):
            raise AssertionError(
                f"expected fw invoke_subgraph HOP args[0] to be torch.fx.Node, "
                f"got {type(fw_subgraph_attr)}"
            )
        fw_subgraph_attr_target = fw_subgraph_attr.target
        if not isinstance(fw_subgraph_attr_target, str):
            raise AssertionError(
                f"expected fw invoke_subgraph HOP args[0].target to be str, "
                f"got {type(fw_subgraph_attr_target)}"
            )
        fw_hop_gm = getattr(joint_gm, fw_subgraph_attr_target)
        if not isinstance(fw_hop_gm, torch.fx.GraphModule):
            raise AssertionError(
                f"expected fw_hop_gm to be GraphModule, got {type(fw_hop_gm)}"
            )
        num_fw_inputs = num_inputs(fw_hop_gm)
        num_fw_outputs = num_outputs(fw_hop_gm)
        new_hop_graphs[identifier].old_num_fw_inputs = num_fw_inputs
        new_hop_graphs[identifier].old_num_fw_outputs = num_fw_outputs

        # Step 1) - Get the `joint_hop_gm`. As mentioned earlier, the
        # backward graph is the joint graph.
        joint_hop_gm = getattr(joint_gm, node.args[0].target)
        if not isinstance(joint_hop_gm, torch.fx.GraphModule):
            raise AssertionError(
                f"expected joint_hop_gm to be GraphModule, got {type(joint_hop_gm)}"
            )

        # Prepare the graph for the partitioner
        joint_hop_gm = prepare_for_partitioner(
            joint_hop_gm, num_fw_inputs, num_fw_outputs
        )

        # TODO: invoke_subgraph should track which of its inputs static indices
        # so it can propagate them to the partitioner (and use in cudagraphs)
        static_lifetime_input_indices: list[int] = []

        used_hop_custom_partition, partition_fn = _get_partition_fn(
            fw_hop_node, aot_config, default_partition_fn
        )

        # Step 2) and 3) - Run joint graph passes and partitioner
        try:
            new_fw_hop_gm, new_bw_hop_gm = partition_fn(
                joint_hop_gm,
                [],
                num_fwd_outputs=num_fw_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices,
            )
        except Exception as e:
            if used_hop_custom_partition:
                raise RuntimeError(
                    f"Error in custom partition function for invoke_subgraph node {fw_hop_node.name}: {e}"
                ) from e
            else:
                raise

        # Save the new forward and backward graph modules
        new_hop_graphs[identifier].new_fw_hop_gm = new_fw_hop_gm
        new_hop_graphs[identifier].new_bw_hop_gm = new_bw_hop_gm

        # Save the number of symints and saved tensors
        new_fw_out_nodes = new_fw_hop_gm.graph.find_nodes(op="output")[0].args[0]
        extra_outputs = new_fw_out_nodes[num_fw_outputs:]
        symint_outputs = [n for n in extra_outputs if is_sym_node(n)]

        new_hop_graphs[identifier].new_num_sym_nodes = len(symint_outputs)
        new_hop_graphs[identifier].new_num_saved_nodes = len(extra_outputs) - len(
            symint_outputs
        )

        new_hop_graphs[identifier].partitioning_done = True

    # Step 3) Restitch the new fw and bw graphs back into the main graph.
    #
    # This is a very mechanical process. There are a quite a few pieces that we
    # need to connect together to make it work. Lets try to understand the
    # problem statement first.
    #
    # For the forward graph, the signature of the old_fw_hop_gm is
    #   inputs - (*primals)
    #   outputs - (*fw_outs)
    # Now the signature of the new_fw_hop_gm is
    #   inputs - (*primals)     -- This is same
    #   outputs - (*fw_outs, *saved_tensors)    - This is different
    # At a high level, this is an easy transformation, in the new graph we just
    # have to replace the old_fw_hop_gm with the new_fw_hop_gm. Everything else
    # falls into place, because the input signature (i.e. args) is same. And
    # even though output signature is different, fw_outs are still at the same
    # indexes as before. So the forward of the `joint_gm` works nicely.
    #
    # Now, lets look at the backward hop graph. Old signature
    #   inputs - (*primals, *tangents)
    #   outputs - (*grad_outs, *fw_outs)
    # New signature
    #   inputs - (*saved_tensors, *tangents) -- Different
    #   outputs - (*grad_outs)  -- Different
    # Here both input and output signature change. The output signature handling
    # is quite easy because the grads_out are sitting at the right place, so we
    # don't have to do anything.
    #
    # For the input signature, we have to collect the saved tensors from the
    # corresponding forward graph output. We collect all saved_tensors when we
    # see the forward graph, and save it into a map and then later use it during
    # the backward.

    # The stack of fw_nodes for invoke_subgraph HOP. There is an implicit
    # assumption about the graph structure, i.e., if we have hop1, hop2, hop3,
    # ... in the forward part of the joint graph, we will have .., hop3, hop2,
    # hop1 order for the backward. This structure allows us to just use a stack
    # to collect all the information that we need to pass from the forward hop
    # node to the corresponding backward node.

    already_added_new_hop_mods = set()

    def add_new_hop_gm(new_subgraph_mod: torch.fx.GraphModule, name: str) -> str:
        new_subgraph_attr_name = f"partitioned_{name}"
        if new_subgraph_attr_name in already_added_new_hop_mods:
            return new_subgraph_attr_name

        joint_gm.register_module(new_subgraph_attr_name, new_subgraph_mod)
        already_added_new_hop_mods.add(new_subgraph_attr_name)
        return new_subgraph_attr_name

    def propagate_meta_info(
        new_hop_gm: torch.fx.GraphModule,
        new_call_function_node: torch.fx.Node,
        old_call_function_node: torch.fx.Node,
    ) -> None:
        # Copy all the fields from the old call_function node. And then override
        # the `val` meta field with the outputs of new_hop_gm.
        new_call_function_node.meta = copy.copy(old_call_function_node.meta)

        output = new_hop_gm.graph.find_nodes(op="output")[0]
        out_example_vals = [n.meta["val"] if n else None for n in output.args[0]]
        new_call_function_node.meta["val"] = tuple(out_example_vals)

    for bw_node in reversed(bw_hop_nodes):
        identifier = bw_node.args[1].removeprefix("bw")

        # Make changes to the corresponding fw and bw node pair simultaneously.
        # The removes the need of any bookkeeping.

        # Fw node changes
        # Insert the new_fw_hop_gm. This is straightforward. Get the
        # new_fw_hop_gm, insert the hop_gm as a get_attr fw_node, and then
        # add a call_function fw_node. Additionally, also use getitem
        # call_functions to collect the saved_tensor nodes

        fw_node = bw_to_fw_hop_node[bw_node]
        new_fw_hop_gm = new_hop_graphs[identifier].new_fw_hop_gm
        if new_fw_hop_gm is None:
            raise AssertionError(
                f"new_fw_hop_gm for identifier {identifier} must not be None"
            )

        old_num_fw_outputs = new_hop_graphs[identifier].old_num_fw_outputs
        new_num_sym_nodes = new_hop_graphs[identifier].new_num_sym_nodes
        new_num_saved_nodes = new_hop_graphs[identifier].new_num_saved_nodes
        if old_num_fw_outputs is None:
            raise AssertionError(
                f"old_num_fw_outputs for identifier {identifier} must not be None"
            )
        if new_num_sym_nodes is None:
            raise AssertionError(
                f"new_num_sym_nodes for identifier {identifier} must not be None"
            )
        if new_num_saved_nodes is None:
            raise AssertionError(
                f"new_num_saved_nodes for identifier {identifier} must not be None"
            )
        total_outputs = old_num_fw_outputs + new_num_saved_nodes + new_num_sym_nodes

        extra_fw_outputs = []

        # Insert the new_fw_hop_gm into the joint_gm
        fw_subgraph_attr = fw_node.args[0]
        if not isinstance(fw_subgraph_attr, torch.fx.Node):
            raise AssertionError(
                f"expected fw invoke_subgraph HOP args[0] to be torch.fx.Node, "
                f"got {type(fw_subgraph_attr)}"
            )
        with joint_gm.graph.inserting_after(fw_node):
            new_fw_mod_attr_name = add_new_hop_gm(new_fw_hop_gm, f"fw{identifier}")
            new_fw_mod_attr = joint_gm.graph.get_attr(new_fw_mod_attr_name)
            new_fw_mod_attr.meta = copy.copy(fw_subgraph_attr.meta)

        # new_hop_fw_gm output signature is (*fw_outs, *saved_tensors)
        with joint_gm.graph.inserting_after(new_fw_mod_attr):
            new_fw_node = joint_gm.graph.call_function(
                the_function=invoke_subgraph,
                args=(
                    new_fw_mod_attr,
                    new_fw_mod_attr_name,
                    *fw_node.args[2:],
                ),
            )
            propagate_meta_info(new_fw_hop_gm, new_fw_node, fw_node)

        # old_num_fw_outputs = (*fw_outs)
        # new_num_fw_outputs = (*fw_outs, *saved_tensors, *sym_nodes)
        with joint_gm.graph.inserting_after(new_fw_node):
            for fw_out_idx in range(old_num_fw_outputs, total_outputs):
                saved_tensor_node = joint_gm.graph.call_function(
                    the_function=operator.getitem, args=(new_fw_node, fw_out_idx)
                )
                saved_tensor_node.meta = copy.copy(new_fw_node.meta)
                saved_tensor_node.meta["val"] = new_fw_node.meta["val"][fw_out_idx]
                extra_fw_outputs.append(saved_tensor_node)

        fw_node.replace_all_uses_with(new_fw_node)
        joint_gm.graph.erase_node(fw_node)

        # Bw node changes
        # Prepare the operands for the bwd graph
        # Old bw graph signature : (*primals, *tangents)
        # New signature will be : (*sym_nodes, *saved_tensors, *tangents)
        # We have already collected the saved_tensors in the forward hop processing.

        # extra_fw_outputs are in the order (*saved_nodes, *sym_nodes).
        # Partitioner has this quirk where the backward wants sym_nodes
        # first. So extract the sym and saved nodes.

        new_bw_hop_gm = new_hop_graphs[identifier].new_bw_hop_gm
        if new_bw_hop_gm is None:
            raise AssertionError(
                f"new_bw_hop_gm for identifier {identifier} must not be None"
            )

        saved_tensor_nodes = extra_fw_outputs[:new_num_saved_nodes]
        sym_nodes = extra_fw_outputs[new_num_saved_nodes:]

        num_primals = new_hop_graphs[identifier].old_num_fw_inputs
        if num_primals is None:
            raise AssertionError(
                f"num_primals for identifier {identifier} must not be None"
            )
        tangents = list(bw_node.args[2 + num_primals :])
        operands = sym_nodes + saved_tensor_nodes + tangents

        # Insert the new_bw_hop_gm into the joint_gm
        with joint_gm.graph.inserting_after(bw_node):
            new_bw_mod_attr_name = add_new_hop_gm(new_bw_hop_gm, bw_node.args[1])
            new_bw_mod_attr = joint_gm.graph.get_attr(new_bw_mod_attr_name)
            new_bw_mod_attr.meta = copy.copy(bw_node.args[0].meta)

        with joint_gm.graph.inserting_after(new_bw_mod_attr):
            new_bw_node = joint_gm.graph.call_function(
                the_function=invoke_subgraph,
                args=(
                    new_bw_mod_attr,
                    new_bw_mod_attr_name,
                    *operands,
                ),
            )
            propagate_meta_info(new_bw_hop_gm, new_bw_node, bw_node)
            # Since the partitioner is run after the graph passes, we have lost
            # the eager information and cannot faithfully extract the eager
            # inputs for the new partitioned backward graph. For the forward
            # graph, it was fine because the input signature remains same.
            new_bw_node.meta.pop("eager_input_vals", None)

            # When the region sets backward-specific inductor config, compile the
            # partitioned backward under it; the forward keeps its own config.
            fw_region_config = fw_node.meta.get("custom", {}).get(
                "nested_region_config"
            )
            # get_backward_nested_region_config returns fw_config unchanged when
            # the region has no distinct backward config, so identity tells us
            # whether to stamp.
            bw_region_config = get_backward_nested_region_config(fw_region_config)
            if bw_region_config is not fw_region_config:
                # Re-stamp on the fresh backward node's meta["custom"] (the source
                # of truth: unlike a GraphModule's meta it survives FX transforms).
                # Lowering picks it up via the subgraph-module mirror (_propagate_*)
                # or the ir.py node fallback.
                new_bw_node.meta["custom"] = {
                    **new_bw_node.meta.get("custom", {}),
                    "nested_region_config": bw_region_config,
                }

        bw_node.replace_all_uses_with(new_bw_node)
        joint_gm.graph.erase_node(bw_node)

    # Rewrite extra (unpaired) fws to call the new partitioned fw subgraph.
    # Their additional saved-tensor outputs become dead and are pruned by
    # eliminate_dead_code.
    for identifier, extras in extra_fws_by_id.items():
        if not new_hop_graphs[identifier].partitioning_done:
            continue
        new_fw_hop_gm = new_hop_graphs[identifier].new_fw_hop_gm
        if new_fw_hop_gm is None:
            continue
        for fw_node in extras:
            fw_subgraph_attr = fw_node.args[0]
            if not isinstance(fw_subgraph_attr, torch.fx.Node):
                raise AssertionError(
                    f"expected fw invoke_subgraph HOP args[0] to be torch.fx.Node, "
                    f"got {type(fw_subgraph_attr)}"
                )
            with joint_gm.graph.inserting_after(fw_node):
                new_attr_name = add_new_hop_gm(new_fw_hop_gm, f"fw{identifier}")
                new_attr = joint_gm.graph.get_attr(new_attr_name)
                new_attr.meta = copy.copy(fw_subgraph_attr.meta)
            with joint_gm.graph.inserting_after(new_attr):
                new_fw_node = joint_gm.graph.call_function(
                    the_function=invoke_subgraph,
                    args=(new_attr, new_attr_name, *fw_node.args[2:]),
                )
                propagate_meta_info(new_fw_hop_gm, new_fw_node, fw_node)
            fw_node.replace_all_uses_with(new_fw_node)
            joint_gm.graph.erase_node(fw_node)

    joint_gm.graph.eliminate_dead_code()
    joint_gm.graph.lint()
    joint_gm.recompile()
    return joint_gm


def maybe_log_graph(
    gm: torch.fx.GraphModule,
    graph_name: str,
    aot_config: AOTConfig,
    structured_log_prefix_fn: Callable[[], str],
    out_structured_logs: list[str] | None = None,
) -> None:
    if not aot_config.enable_log:
        return
    aot_graphs_log.debug(
        "%s",
        lazy_format_graph_code(
            f"{graph_name}",
            gm,
            aot_config.aot_id,
            include_stride=True,
            include_device=True,
            colored=True,
        ),
    )

    def gm_str_fn() -> str:
        return gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )

    if out_structured_logs is not None:
        out_structured_logs.append(f"{structured_log_prefix_fn()}:{gm_str_fn()}")
    else:
        trace_structured(
            f"{structured_log_prefix_fn()}",
            payload_fn=lambda: gm_str_fn(),
        )


def create_wrap_fn(
    fn: Callable[..., Any], args: tuple[Any, ...]
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    from torch.fx.experimental.proxy_tensor import maybe_enable_thunkify

    from .functional_utils import from_fun, has_data_mutation, to_fun

    def assert_no_mutation(t: Any) -> None:
        if has_data_mutation(t):
            raise AssertionError(
                "Saved tensors hooks with inputs mutations are not allowed"
            )

    @simple_wraps(fn)
    def _wrapper(*args: Any) -> Any:
        with maybe_enable_thunkify():
            disable_above = torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            )

            with disable_above:
                f_args = pytree.tree_map(to_fun, args)
                f_outs = fn(*f_args)
                pytree.tree_map(assert_no_mutation, f_args)
                return pytree.tree_map(from_fun, f_outs)

    return _wrapper, args


def prepare_hook_gm(
    aot_config: AOTConfig, fn: Callable[..., Any], args: tuple[Any, ...]
) -> torch.fx.GraphModule:
    from torch._functorch._aot_autograd.graph_capture import _create_graph

    fn, args = create_wrap_fn(fn, args)
    gm = _create_graph(fn, args, aot_config=aot_config)  # type: ignore[arg-type]
    return gm


# Inline Autograd saved_tensors_hooks into epilogue of forward graph
# and prologue of backward graph.
# This changes forward graph outputs and inputs.
# Pack hook can return tensors, sym scalars, constants.
# All tensors to save for backward will be grouped together at front.
# Sym scalars grouped on another end. Constants are inlined in the graph.
def maybe_inline_graph_saved_tensors_hooks(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    num_inner_fwd_outputs: int,
    inner_meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
    static_input_indices: list[int],
) -> None:
    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        return

    get_hooks = torch._functorch._aot_autograd.utils.top_saved_tensors_hooks
    are_inline_hooks = (
        torch._functorch._aot_autograd.utils.saved_tensors_hooks_are_inlineable
    )

    hooks = get_hooks()
    if not are_inline_hooks(hooks):
        return

    pack_hook_gm, unpack_hook_gm = hooks

    structured_logs: list[str] = []
    maybe_log_graph(
        fw_module,
        "Forward graph pre saved_tensors_hooks inlining",
        aot_config,
        lambda: "aot_forward_graph_pre_saved_tensors_hooks",
        structured_logs,
    )
    maybe_log_graph(
        bw_module,
        "Backward graph pre saved_tensors_hooks inlining",
        aot_config,
        lambda: "aot_backward_graph_pre_saved_tensors_hooks",
        structured_logs,
    )
    fw_g = fw_module.graph
    bw_g = bw_module.graph

    fw_g_names = {node.name for node in fw_g.nodes}
    bw_g_names = {node.name for node in bw_g.nodes}

    def _gen_unused_name(candidate: str) -> str:
        c = candidate
        i = 0
        while c in fw_g_names or c in bw_g_names:
            c = f"{candidate}_{i}"
            i = i + 1
        return c

    bw_g_inputs = bw_g.find_nodes(op="placeholder")

    fw_out_n = fw_g.output_node()
    fw_outs = fw_out_n.args[0]  # type: ignore[var-annotated]
    fw_outs_inner_set = set(fw_outs[:num_inner_fwd_outputs])  # type: ignore[index]
    fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]  # type: ignore[index]
    fw_outs_packed_tensors = []  # type: ignore[var-annotated]
    fw_outs_packed_syms = []  # type: ignore[var-annotated]

    # The main use case for saved_tensors_hooks is activation quantization,
    # for memory usage optimization.
    # Desired behavior is to quantize saved activations to free the original saved tensor.
    # Saved nodes may include forward inputs, outputs, parameters.
    # They may be held by something else and will not be deallocated after quantization.
    # Donated buffers are intermediates in the graph invisible for the user,
    # this guarantees that they can be deallocated.
    # Using this as a default behavior to select saved nodes to apply hooks.
    # There is also a config to apply hooks for all saved nodes without any filtering.
    # The plan is to propagate meta about the source of the saved node to the user hook function.
    mode = torch._functorch.config.saved_tensors_hooks_filtering_mode
    allow_set = None
    exclude_set = None

    if mode == "donated":
        # collect_bw_donated_buffer_idxs requires inner_meta to have num_symints_saved_for_bw
        inner_meta.num_symints_saved_for_bw = len(
            [n for n in fw_outs_saved_for_bw if is_sym_node(n)]  # type: ignore[arg-type]
        )
        # Count tensors with no version counter check (used in tensors_saved_for_backwards_slice)
        inner_meta.num_tensors_saved_with_no_vc_check = len(
            [
                n
                # pyrefly: ignore [not-iterable]
                for n in fw_outs_saved_for_bw
                if isinstance(n, torch.fx.Node)
                and n.meta.get("saved_tensor_with_no_vc_check", False)
            ]
        )
        bw_donated_idxs = collect_bw_donated_buffer_idxs(
            fw_module,
            bw_module,
            inner_meta,
        )
        fw_donated_idxs = [
            i - inner_meta.num_symints_saved_for_bw for i in bw_donated_idxs
        ]
        allow_set = {fw_outs_saved_for_bw[i].name for i in fw_donated_idxs}  # type: ignore[union-attr]
    elif mode == "no_static":
        fw_g_inputs = fw_g.find_nodes(op="placeholder")
        exclude_set = {fw_g_inputs[i].name for i in static_input_indices}

    if (allow_set is not None) and (not allow_set):
        # This means we have empty whitelist,
        # No donated (intermediate) saved.
        # Do not do anything in this case
        return

    if aot_config.enable_log:
        structured_logs.append(f"fw_outs_saved_for_bw:{fw_outs_saved_for_bw}")
        structured_logs.append(f"mode:{mode}")
        structured_logs.append(f"allow_set:{allow_set}")
        structured_logs.append(f"exclude_set:{exclude_set}")

    # pyrefly: ignore [not-iterable]
    for saved in fw_outs_saved_for_bw:
        if ((allow_set is not None) and (saved.name not in allow_set)) or (  # type: ignore[union-attr]
            (exclude_set is not None) and (saved.name in exclude_set)  # type: ignore[union-attr]
        ):
            if isinstance(saved.meta["val"], torch.Tensor):  # type: ignore[union-attr]
                fw_outs_packed_tensors.append(saved)
            continue

        val = saved.meta["val"]  # type: ignore[union-attr]
        if not isinstance(val, torch.Tensor):
            continue

        def _get_extra_info() -> dict[str, Any]:
            return {"_fw_graph": fw_g, "_bw_graph": bw_g, "_node": saved}

        with _saved_tensor_hook_context(_get_extra_info()):
            pack_out_val = pack_hook_gm(val)

        requires_sc_handling = any(
            is_traceable_wrapper_subclass(x) for x in pytree.tree_leaves(pack_out_val)
        )
        if requires_sc_handling:
            raise NotImplementedError(
                "Tensor subclasses in GraphModule saved tensors hooks are not supported"
                "You can workaround it by manually returning subclass's inner tensors"
                " in the pack hook, and reconstructing the subclass in the unpack hook"
            )

        with _saved_tensor_hook_context(_get_extra_info()):
            pack_gm = prepare_hook_gm(aot_config, pack_hook_gm, (val,))
            pack_g = pack_gm.graph
            maybe_log_graph(
                pack_gm,
                f"saved_tensors_pack_hook {saved.name}",  # type: ignore[union-attr]
                aot_config,
                lambda: f"aot_saved_tensors_hooks_pack {saved.name}",  # type: ignore[union-attr]
                structured_logs,
            )
            pack_out_val = pack_gm(val)

        # Install pack hook graph as eiplogue of fw_module.
        # Saved tensor output becomes input of pack hook graph.
        # Replace saved tensor output with pack hook graph output.
        # Outputs symbolic scalars, tensors  are accumulated separately.
        # Then in forward outputs and backward inputs installed in order
        # sym_scalars, packed_saved_tensors.
        # Keeping all tensors together allows to preserve
        # the same identification at runtime,
        # updating only number of saved sym_scalars and tensors.
        pack_g_inputs = pack_g.find_nodes(op="placeholder")
        if len(pack_g_inputs) != 1:
            raise AssertionError(
                f"expected exactly 1 pack_g_input, got {len(pack_g_inputs)}"
            )
        env = {pack_g_inputs[0]: saved}
        fw_pack_out_args = None
        with fw_g.inserting_before(fw_out_n):
            for node in pack_g.nodes:
